"""Q6_K dequantization — pure PyTorch port of llama.cpp's `dequantize_row_q6_K`.

Gemma 4 E4B's `token_embd.weight` is stored as Q6_K in the GGUF file.
The standard tq4 loader can't handle it (Q6_K ≠ tq4), so this module
provides the conversion. Once dequantized to FP32, the embedding slots
directly into `substrate.tok.weight` at the Gemma channel range.

Q6_K block format (from `ggml-quants.c`):
  struct block_q6_K {
      uint8_t  ql[128];      // quants, low 4 bits × 256 values
      uint8_t  qh[64];       // quants, high 2 bits × 256 values
      int8_t   scales[16];   // 16 scales, signed 8-bit
      ggml_half d;           // super-block scale (fp16)
  };   // 210 bytes total, 256 elements per block

Dequant per element i ∈ [0, 256):
  half = i // 128, within = i % 128, quarter = within // 32, l = within % 32
  ql_byte_idx = half*64 + l + (32 if quarter in {1,3} else 0)
  ql_shift    = 4 if quarter >= 2 else 0
  qh_byte_idx = half*32 + l
  qh_shift    = 2 * quarter
  scale_idx   = half*8 + (l // 16) + 2*quarter
  q = ((ql[ql_byte_idx] >> ql_shift) & 0xF) | (((qh[qh_byte_idx] >> qh_shift) & 3) << 4)
  q_signed = q - 32        # Q6_K uses symmetric signed quantization
  value = d * scales[scale_idx] * q_signed

The vectorized implementation below precomputes the 256-position pointer
table once, then batches over blocks for the actual dequant.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch


BLOCK_ELEMENTS_Q6K = 256
BLOCK_BYTES_Q6K = 210   # 128 ql + 64 qh + 16 scales + 2 d


def _q6k_position_tables(device: torch.device = torch.device("cpu")):
    """Precompute the position → (ql_idx, ql_shift, qh_idx, qh_shift,
    scale_idx) mapping. 256 positions, all constant per-block."""
    p = torch.arange(BLOCK_ELEMENTS_Q6K, device=device)
    half = p // 128
    within = p % 128
    quarter = within // 32
    l = within % 32
    is_scale = l // 16

    ql_idx = half * 64 + l + torch.where(
        (quarter == 1) | (quarter == 3),
        torch.full_like(quarter, 32),
        torch.zeros_like(quarter),
    )
    ql_shift = torch.where(
        quarter >= 2,
        torch.full_like(quarter, 4),
        torch.zeros_like(quarter),
    )
    qh_idx = half * 32 + l
    qh_shift = 2 * quarter
    scale_idx = half * 8 + is_scale + 2 * quarter
    return ql_idx, ql_shift, qh_idx, qh_shift, scale_idx


def dequantize_q6_k_blocks(
    ql: torch.Tensor,          # (N_blocks, 128) uint8
    qh: torch.Tensor,          # (N_blocks, 64) uint8
    scales: torch.Tensor,      # (N_blocks, 16) int8
    d: torch.Tensor,           # (N_blocks,) fp32
) -> torch.Tensor:
    """Dequantize a batch of Q6_K blocks to FP32.

    Returns: (N_blocks, 256) fp32.
    """
    N = ql.shape[0]
    assert ql.shape == (N, 128), f"ql shape {ql.shape}"
    assert qh.shape == (N, 64), f"qh shape {qh.shape}"
    assert scales.shape == (N, 16), f"scales shape {scales.shape}"
    assert d.shape == (N,), f"d shape {d.shape}"

    ql_idx, ql_shift, qh_idx, qh_shift, scale_idx = _q6k_position_tables(
        device=ql.device,
    )

    # Gather per-position bytes — (N, 256)
    ql_vals = ql[:, ql_idx]                    # uint8
    qh_vals = qh[:, qh_idx]                    # uint8
    scale_vals = scales[:, scale_idx].to(torch.int32)  # (N, 256) signed

    # Extract 6-bit quant: low 4 + high 2
    ql_low = (ql_vals.to(torch.int32) >> ql_shift) & 0xF
    qh_high = (qh_vals.to(torch.int32) >> qh_shift) & 0x3
    q = ql_low | (qh_high << 4)                # 0..63
    q_signed = q - 32                           # -32..31

    # y = d * scale * q_signed
    y = d.unsqueeze(1) * scale_vals.to(torch.float32) * q_signed.to(torch.float32)
    return y


def dequantize_q6_k_tensor(raw_bytes: bytes, n_elements: int) -> torch.Tensor:
    """Dequantize a contiguous Q6_K tensor stored as raw bytes.

    Args:
        raw_bytes: serialized Q6_K blocks (n_blocks × 210 bytes).
        n_elements: total elements to expect (must be divisible by 256).

    Returns:
        (n_elements,) fp32 tensor.
    """
    assert n_elements % BLOCK_ELEMENTS_Q6K == 0, (
        f"n_elements {n_elements} not divisible by {BLOCK_ELEMENTS_Q6K}"
    )
    n_blocks = n_elements // BLOCK_ELEMENTS_Q6K
    expected_bytes = n_blocks * BLOCK_BYTES_Q6K
    if len(raw_bytes) != expected_bytes:
        raise ValueError(
            f"Q6_K: got {len(raw_bytes)} bytes, expected {expected_bytes} "
            f"for {n_blocks} blocks"
        )

    data = np.frombuffer(raw_bytes, dtype=np.uint8)
    blocks = data.reshape(n_blocks, BLOCK_BYTES_Q6K)
    ql_np = blocks[:, 0:128]                       # (N, 128)
    qh_np = blocks[:, 128:192]                     # (N, 64)
    scales_np = blocks[:, 192:208].view(np.int8)   # (N, 16) signed
    d_np = blocks[:, 208:210].copy().view(np.float16).astype(np.float32)
    d_np = d_np.reshape(n_blocks)

    ql = torch.from_numpy(np.ascontiguousarray(ql_np))
    qh = torch.from_numpy(np.ascontiguousarray(qh_np))
    scales = torch.from_numpy(np.ascontiguousarray(scales_np))
    d = torch.from_numpy(np.ascontiguousarray(d_np))

    y = dequantize_q6_k_blocks(ql, qh, scales, d)
    return y.reshape(-1)[:n_elements]


def extract_q6_k_tensor(reader, tensor_name: str) -> torch.Tensor:
    """Read a Q6_K tensor from a GGUF reader and dequantize to FP32.

    Returns: fp32 tensor with shape matching the GGUF metadata.
    """
    tensor = None
    for t in reader.tensors:
        if t.name == tensor_name:
            tensor = t
            break
    if tensor is None:
        raise KeyError(f"tensor {tensor_name!r} not in GGUF")

    # Q6_K has GGUF type id 14.
    from calm.llm_computer.tq4_gguf_loader import _get_ggml_type_id
    type_id = _get_ggml_type_id(tensor)
    if type_id != 14:
        raise ValueError(
            f"{tensor_name!r} is type id {type_id}, not Q6_K (14)"
        )

    shape = tuple(int(d) for d in tensor.shape)
    n_elements = int(np.prod(shape))
    raw = tensor.data.tobytes()
    y = dequantize_q6_k_tensor(raw, n_elements)
    # GGUF shape is stored reversed (inner dim first); reshape accordingly.
    # For Gemma's token_embd the GGUF shape [2560, 262144] means 262144
    # rows × 2560 cols in Python (row-major) — matching our
    # substrate.tok.weight (vocab, d_model) layout.
    return y.reshape(shape[-1], shape[0] if len(shape) > 1 else 1).squeeze(-1)
