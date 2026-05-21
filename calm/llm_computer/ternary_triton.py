"""Triton fused matmul kernel for native ternary weights (TRM-1.58 / Slice 13 Gate B).

Pattern from `tq4_triton.py`: stream packed weights into the dot product
inline, never materialize the dequantized weight matrix. This is the
key memory-bandwidth win — quantized weight is ~16× smaller than BF16
(2 bits per weight + 1 scale per tensor), so we want to keep it that
way through the matmul.

Ternary packing (2 bits per weight):
    0b00 -> -1
    0b01 ->  0
    0b10 -> +1
    0b11 -> reserved (unused; treated as 0 if it ever appears)
4 weights packed per byte. For an (out_features, in_features) weight:
    packed: (out_features, in_features / 4) uint8

Single FP32 scale per tensor (matches BitNet b1.58 absmean convention).

Forward contract:
    y = x @ (centroids[unpack(packed_w)] * scale).T
where centroids = [-1, 0, +1, 0] (4th index defensive). Activation x stays
BF16/FP32 in Gate B.1 — int8 per-token absmax can land as B.2 if the
kernel proves viable.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


# =========================================================================
# Packing / unpacking utilities (host-side, CPU/GPU agnostic)
# =========================================================================


@torch.no_grad()
def quantize_to_ternary_indices(weight: torch.Tensor, eps: float = 1e-8):
    """Absmean quantize FP weight to ternary INDICES + scale.

    Returns:
        indices: (out_features, in_features) int8 in {-1, 0, +1}
        scale: float scalar (FP32)
    """
    scale = weight.abs().mean().clamp_min(eps)
    w_norm = weight / scale
    indices = torch.clamp(torch.round(w_norm), -1.0, 1.0).to(torch.int8)
    return indices, float(scale.item())


@torch.no_grad()
def pack_ternary_2bit(indices: torch.Tensor) -> torch.Tensor:
    """Pack ternary indices {-1, 0, +1} as 2 bits per weight, 4 weights
    per byte.

    Encoding: -1 -> 0b00 (0), 0 -> 0b01 (1), +1 -> 0b10 (2).
    Reserved: 0b11 (3) — never written; treated as 0 if encountered.

    Args:
        indices: (out_features, in_features) int8 in {-1, 0, +1}.
            in_features MUST be divisible by 4.
    Returns:
        packed: (out_features, in_features // 4) uint8.
    """
    assert indices.dtype == torch.int8
    out_f, in_f = indices.shape
    assert in_f % 4 == 0, f"in_features must be divisible by 4 (got {in_f})"
    # Map {-1, 0, +1} -> {0, 1, 2}
    codes = (indices + 1).to(torch.uint8)
    # Reshape into (out_f, in_f//4, 4) and pack each group of 4 into 1 byte
    codes = codes.reshape(out_f, in_f // 4, 4)
    packed = (
        (codes[:, :, 0])
        | (codes[:, :, 1] << 2)
        | (codes[:, :, 2] << 4)
        | (codes[:, :, 3] << 6)
    )
    return packed.contiguous()


@torch.no_grad()
def unpack_ternary_2bit(packed: torch.Tensor, in_features: int) -> torch.Tensor:
    """Inverse of pack_ternary_2bit. Returns int8 in {-1, 0, +1}."""
    assert packed.dtype == torch.uint8
    out_f = packed.shape[0]
    assert in_features % 4 == 0
    assert packed.shape[1] == in_features // 4
    # Unpack each byte into 4 codes
    p = packed.to(torch.int32)
    c0 = (p) & 0x3
    c1 = (p >> 2) & 0x3
    c2 = (p >> 4) & 0x3
    c3 = (p >> 6) & 0x3
    codes = torch.stack([c0, c1, c2, c3], dim=-1).reshape(out_f, in_features)
    # Map {0, 1, 2, 3} -> {-1, 0, +1, 0 (defensive on reserved)}
    indices = torch.where(codes == 3, torch.tensor(1, device=codes.device, dtype=codes.dtype), codes) - 1
    return indices.to(torch.int8)


# =========================================================================
# Triton matmul kernel — streams packed ternary weights
# =========================================================================


@triton.jit
def _ternary_matmul_kernel(
    x_ptr,           # (M, K) FP32, contiguous
    w_packed_ptr,    # (N, K/4) uint8, contiguous (row-major over N)
    scale_ptr,       # (1,) FP32, per-tensor scale
    y_ptr,           # (M, N) FP32, output
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """One program tile: BLOCK_M rows × BLOCK_N cols of output.

    Streams packed weights inline: each byte unpacks to 4 ternary codes
    (-1, 0, +1) which then multiply the corresponding 4 x values.

    BLOCK_K is the inner-dim tile size; MUST be divisible by 4 (one byte
    of packed weight per 4 K elements). At each inner step we load
    BLOCK_K/4 packed bytes per output row, unpack to BLOCK_K signed
    weights, and accumulate into the BLOCK_M × BLOCK_N tile.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = rm < M
    mask_n = rn < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    scale = tl.load(scale_ptr)

    # Iterate over K dimension in BLOCK_K chunks
    for k0 in range(0, K, BLOCK_K):
        k_offs = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offs < K

        # Load x tile: (BLOCK_M, BLOCK_K)
        x_offs = rm[:, None] * stride_xm + k_offs[None, :] * stride_xk
        x = tl.load(
            x_ptr + x_offs,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )

        # Load packed weight tile: (BLOCK_N, BLOCK_K/4) uint8
        # Then unpack to (BLOCK_N, BLOCK_K) signed.
        # Each byte at offset (n, k_byte) decodes to 4 ternary codes
        # for K positions [k_byte*4 : k_byte*4 + 4].
        # Compute byte offsets: k_byte = k_offs // 4, k_inbyte = k_offs % 4
        k_byte = k_offs // 4                        # (BLOCK_K,)
        k_inbyte = k_offs % 4                        # (BLOCK_K,)
        w_offs = rn[:, None] * stride_wn + k_byte[None, :] * stride_wk
        # mask for valid bytes (k_byte must be in [0, K/4))
        mask_k_byte = k_byte < (K // 4)
        w_byte = tl.load(
            w_packed_ptr + w_offs,
            mask=mask_n[:, None] & mask_k_byte[None, :],
            other=0,
        ).to(tl.int32)
        # Extract 2-bit code at position k_inbyte: shift right by 2*k_inbyte,
        # then mask 0x3. k_inbyte is per-K so this is column-wise.
        shift = 2 * k_inbyte[None, :]               # (1, BLOCK_K)
        code = (w_byte >> shift) & 0x3              # (BLOCK_N, BLOCK_K)
        # Map {0, 1, 2, 3} -> {-1, 0, +1, 0 (defensive)}.
        # The math: ternary = code - 1, then clamp at 0 if code == 3.
        # Simpler branchless: ternary = (code == 0) * -1 + (code == 2) * 1.
        # Use that to avoid extra branch — Triton's compiler folds well.
        is_neg = (code == 0).to(tl.float32)
        is_pos = (code == 2).to(tl.float32)
        ternary = is_pos - is_neg                   # (BLOCK_N, BLOCK_K)

        # Accumulate: acc += x @ ternary.T
        # x: (BLOCK_M, BLOCK_K), ternary: (BLOCK_N, BLOCK_K)
        # Result tile: (BLOCK_M, BLOCK_N) -- tl.dot expects (M, K) @ (K, N)
        # so transpose ternary view.
        # input_precision="ieee" forces strict FP32 matmul; default tf32 on
        # Ampere+ gives ~1e-3 relative error which exceeds Gate B parity gate.
        acc += tl.dot(x, tl.trans(ternary), input_precision="ieee")

    # Scale and store
    acc = acc * scale
    y_offs = rm[:, None] * stride_ym + rn[None, :] * stride_yn
    tl.store(y_ptr + y_offs, acc, mask=mask_m[:, None] & mask_n[None, :])


def ternary_matmul_triton(
    x: torch.Tensor, w_packed: torch.Tensor, scale: float,
    in_features: int, out_features: int,
    BLOCK_M: int = 16, BLOCK_N: int = 32, BLOCK_K: int = 32,
) -> torch.Tensor:
    """Host wrapper for the ternary matmul kernel.

    Args:
        x: (M, in_features) FP32, contiguous.
        w_packed: (out_features, in_features // 4) uint8, contiguous.
        scale: FP32 per-tensor scale.
        in_features, out_features: as named.
        BLOCK_M, BLOCK_N, BLOCK_K: kernel tile sizes (BLOCK_K must be %4==0).

    Returns:
        y: (M, out_features) FP32.
    """
    assert x.is_contiguous() and x.dtype == torch.float32
    assert w_packed.is_contiguous() and w_packed.dtype == torch.uint8
    assert w_packed.shape == (out_features, in_features // 4)
    assert in_features % 4 == 0
    assert BLOCK_K % 4 == 0
    M = x.shape[0]
    assert x.shape[1] == in_features
    y = torch.empty((M, out_features), device=x.device, dtype=torch.float32)
    scale_tensor = torch.tensor([scale], device=x.device, dtype=torch.float32)
    grid = (
        (M + BLOCK_M - 1) // BLOCK_M,
        (out_features + BLOCK_N - 1) // BLOCK_N,
    )
    _ternary_matmul_kernel[grid](
        x, w_packed, scale_tensor, y,
        M, out_features, in_features,
        x.stride(0), x.stride(1),
        w_packed.stride(0), w_packed.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4,
    )
    return y


# =========================================================================
# v2: W1.58A8 INT8 tensor-core path (Slice 14a)
# =========================================================================
#
# v1 above does FP32 matmul (input_precision="ieee") so → CUDA cores, not
# tensor cores. v2 quantizes activations to int8 per-token absmax, casts
# the unpacked ternary codes to int8 in {-1, 0, +1}, and runs
# tl.dot(int8, int8, out_dtype=int32) which lowers to INT8 tensor cores
# on Ada (compute capability 8.9). Dequant epilogue scales by
# x_scale_per_row × w_scale_per_tensor at the end.
#
# Gate B v1 preserved as ablation baseline per `tq4_triton.py` v1→v2→v4→v5
# precedent. v2 is the speed-claim kernel; v1 is the parity baseline.


@torch.no_grad()
def quantize_activation_int8_pertoken(x: torch.Tensor, eps: float = 1e-8):
    """Per-token (per-row) absmax int8 quantization.

    Args:
        x: (M, K) FP32, contiguous.
    Returns:
        x_int8: (M, K) int8
        x_scale: (M,) FP32 — per-row scale s.t. x ≈ x_int8.float() * x_scale[:, None]
    """
    assert x.dim() == 2 and x.dtype == torch.float32
    # Per-row absmax
    row_max = x.abs().amax(dim=1).clamp_min(eps)              # (M,)
    scale = row_max / 127.0                                    # (M,)
    x_q = torch.round(x / scale[:, None]).clamp(-127, 127).to(torch.int8)
    return x_q, scale


@triton.jit
def _ternary_matmul_int8_kernel(
    x_ptr,           # (M, K) int8
    w_packed_ptr,    # (N, K/4) uint8
    x_scale_ptr,     # (M,) FP32 per-row
    w_scale_ptr,     # (1,) FP32 per-tensor
    y_ptr,           # (M, N) FP32
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """INT8 tensor-core ternary matmul.

    Math: y = x_int8 @ ternary_int8.T  (accumulated as int32),
          then y_fp32 = y_int32 * x_scale[:, None] * w_scale.

    Ternary codes {-1, 0, +1} fit cleanly in int8 so the INT8 tensor-core
    matmul has no precision loss from the weight side. The only quant
    loss is from int8 activation absmax (~1/127 ≈ 0.78% per element).
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = rm < M
    mask_n = rn < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    for k0 in range(0, K, BLOCK_K):
        k_offs = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offs < K

        # Load x tile as int8: (BLOCK_M, BLOCK_K)
        x_offs = rm[:, None] * stride_xm + k_offs[None, :] * stride_xk
        x_i8 = tl.load(
            x_ptr + x_offs,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0,
        )

        # Load packed weight tile and unpack as int8 in {-1, 0, +1}.
        k_byte = k_offs // 4
        k_inbyte = k_offs % 4
        w_offs = rn[:, None] * stride_wn + k_byte[None, :] * stride_wk
        mask_k_byte = k_byte < (K // 4)
        w_byte = tl.load(
            w_packed_ptr + w_offs,
            mask=mask_n[:, None] & mask_k_byte[None, :],
            other=0,
        ).to(tl.int32)
        shift = 2 * k_inbyte[None, :]
        code = (w_byte >> shift) & 0x3                        # (BLOCK_N, BLOCK_K)
        # Map {0, 1, 2, 3} -> int8 {-1, 0, +1, 0}
        is_neg = (code == 0).to(tl.int8)
        is_pos = (code == 2).to(tl.int8)
        w_i8 = is_pos - is_neg                                # (BLOCK_N, BLOCK_K) int8

        # INT8 tensor-core matmul: int8 @ int8 -> int32
        acc += tl.dot(x_i8, tl.trans(w_i8), out_dtype=tl.int32)

    # Dequant epilogue: int32 * x_scale[:, None] * w_scale -> fp32
    acc_fp = acc.to(tl.float32)
    x_scale = tl.load(x_scale_ptr + rm, mask=mask_m, other=1.0)   # (BLOCK_M,)
    w_scale = tl.load(w_scale_ptr)                                # scalar
    y_fp = acc_fp * x_scale[:, None] * w_scale

    y_offs = rm[:, None] * stride_ym + rn[None, :] * stride_yn
    tl.store(y_ptr + y_offs, y_fp, mask=mask_m[:, None] & mask_n[None, :])


def ternary_matmul_triton_v2_prequant(
    x_i8: torch.Tensor, x_scale: torch.Tensor,
    w_packed: torch.Tensor, w_scale: float,
    in_features: int, out_features: int,
    BLOCK_M: int = 32, BLOCK_N: int = 32, BLOCK_K: int = 32,
) -> torch.Tensor:
    """v2 host wrapper with PRE-QUANTIZED activations.

    Use this when bench/runtime amortizes the activation quant cost across
    multiple matmuls (real forward pass: x quantized once per layer-input,
    reused by W_qkv/W_out/ff_in/ff_out projections).

    Args:
        x_i8: (M, in_features) int8.
        x_scale: (M,) FP32 per-row scale.
        w_packed: (out_features, in_features // 4) uint8.
        w_scale: FP32 per-tensor weight scale.
    Returns:
        y: (M, out_features) FP32.
    """
    assert x_i8.is_contiguous() and x_i8.dtype == torch.int8
    assert x_scale.is_contiguous() and x_scale.dtype == torch.float32
    assert w_packed.is_contiguous() and w_packed.dtype == torch.uint8
    assert w_packed.shape == (out_features, in_features // 4)
    assert in_features % 4 == 0 and BLOCK_K % 4 == 0
    assert BLOCK_K >= 16, "INT8 tensor cores want BLOCK_K >= 16"

    M = x_i8.shape[0]
    assert x_i8.shape[1] == in_features
    assert x_scale.shape == (M,)

    y = torch.empty((M, out_features), device=x_i8.device, dtype=torch.float32)
    w_scale_tensor = torch.tensor([w_scale], device=x_i8.device, dtype=torch.float32)
    grid = (
        (M + BLOCK_M - 1) // BLOCK_M,
        (out_features + BLOCK_N - 1) // BLOCK_N,
    )
    _ternary_matmul_int8_kernel[grid](
        x_i8, w_packed, x_scale, w_scale_tensor, y,
        M, out_features, in_features,
        x_i8.stride(0), x_i8.stride(1),
        w_packed.stride(0), w_packed.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4,
    )
    return y


def ternary_matmul_triton_v2(
    x: torch.Tensor, w_packed: torch.Tensor, w_scale: float,
    in_features: int, out_features: int,
    BLOCK_M: int = 32, BLOCK_N: int = 32, BLOCK_K: int = 32,
) -> torch.Tensor:
    """v2 host wrapper: W1.58A8 INT8 tensor-core path.

    Convenience wrapper that quantizes activations inline. For perf-
    sensitive bench/runtime use `ternary_matmul_triton_v2_prequant` to
    amortize activation quant across multiple matmuls.

    Args:
        x: (M, in_features) FP32 — host-side quantized to int8 here.
        w_packed: (out_features, in_features // 4) uint8.
        w_scale: FP32 per-tensor weight scale.
    Returns:
        y: (M, out_features) FP32.
    """
    assert x.is_contiguous() and x.dtype == torch.float32
    M = x.shape[0]
    assert x.shape[1] == in_features
    x_i8, x_scale = quantize_activation_int8_pertoken(x)
    x_i8 = x_i8.contiguous()
    x_scale = x_scale.contiguous()
    return ternary_matmul_triton_v2_prequant(
        x_i8, x_scale, w_packed, w_scale,
        in_features, out_features,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )


def ternary_linear_triton(x: torch.Tensor, weight: torch.Tensor,
                           bias: torch.Tensor | None = None) -> torch.Tensor:
    """End-to-end: take a FP `weight` (treated as the master weight),
    absmean-quantize → pack → run Triton matmul → return y = x @ W_q^T + bias.

    Drop-in replacement for TernaryLinear.forward in the Triton path.
    Activations stay in BF16/FP32 (B.1 scope; int8 path is B.2 if viable).

    For correctness parity testing — the standard production path will
    pre-pack the weight once and stash `w_packed` + `scale` on the module.
    """
    out_features, in_features = weight.shape
    assert in_features % 4 == 0, f"in_features must be %4 for ternary packing (got {in_features})"
    indices, scale = quantize_to_ternary_indices(weight)
    w_packed = pack_ternary_2bit(indices).to(weight.device)
    # x reshape to (M, K) for matmul, then unflatten back
    orig_shape = x.shape
    x_flat = x.reshape(-1, in_features).contiguous().to(torch.float32)
    y_flat = ternary_matmul_triton(x_flat, w_packed, scale,
                                    in_features, out_features)
    if bias is not None:
        y_flat = y_flat + bias.to(torch.float32)
    return y_flat.reshape(*orig_shape[:-1], out_features).to(x.dtype)
