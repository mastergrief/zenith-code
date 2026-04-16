"""Triton fused dequant + matmul kernels for tq4 (and Q6_K).

The hot path. PyTorch dequant of tq4 weights materializes a full FP32
W tensor per call (~26M elements for ffn_up) — bandwidth-bound on a
4070M at ~6.8 ms per linear, or ~3 sec per token across 378 calls.

These kernels stream tq4 bytes directly into the dot product, never
materializing W. Memory bandwidth drops 8× (13.5 MB vs 100 MB) and
the launch overhead drops because the whole linear is one kernel.

Math equivalence with the existing path:
- Standard: y = x @ W where W = (centroids[codes] @ Pi) * d
- Equivalent: y = (x @ Pi.T) @ (centroids[codes] * d)  -- since Pi orthogonal
- The kernel takes pre-rotated x_rot and the un-rotated centroid weights.
- x is rotated outside the kernel by a single small matmul (cheap).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _tq4_matvec_kernel(
    x_rot_ptr,      # (in_features,) FP32, pre-rotated by Pi.T
    qs_ptr,         # (n_blocks * 128,) uint8, packed nybbles
    d_ptr,          # (n_blocks,) FP32, per-block scale
    centroids_ptr,  # (16,) FP32, Lloyd-Max levels
    y_ptr,          # (out_features,) FP32, output
    in_features,
    out_features,
    BPR: tl.constexpr,           # blocks per row of W (= in_features / 256)
    BLOCK_HALF: tl.constexpr,    # 128 (half block, after nybble unpack)
):
    """One program = one output row. Loops over BPR blocks of W and
    accumulates dot products with the corresponding x_rot segments."""
    pid = tl.program_id(0)
    if pid >= out_features:
        return

    half_idx = tl.arange(0, BLOCK_HALF)  # (128,)
    acc = tl.zeros((), dtype=tl.float32)

    for b in range(BPR):
        block_idx = pid * BPR + b
        d_block = tl.load(d_ptr + block_idx)

        # Load 128 packed bytes
        qs = tl.load(qs_ptr + block_idx * BLOCK_HALF + half_idx).to(tl.int32)
        low = qs & 0xF
        high = (qs >> 4) & 0xF
        c_low = tl.load(centroids_ptr + low)
        c_high = tl.load(centroids_ptr + high)

        # x_rot positions for this block: low at evens, high at odds
        x_base = b * 256
        x_low = tl.load(x_rot_ptr + x_base + 2 * half_idx)
        x_high = tl.load(x_rot_ptr + x_base + 2 * half_idx + 1)

        block_dot = tl.sum(c_low * x_low) + tl.sum(c_high * x_high)
        acc += block_dot * d_block

    tl.store(y_ptr + pid, acc)


def tq4_matvec_triton(
    x_rot: torch.Tensor,         # (in_features,) FP32, pre-rotated
    qs: torch.Tensor,            # (n_blocks, 128) uint8
    d: torch.Tensor,             # (n_blocks,) FP32
    centroids: torch.Tensor,     # (16,) FP32
    out_features: int,
    in_features: int,
) -> torch.Tensor:
    """tq4 matrix-vector multiply, fused dequant + matmul. Returns
    y of shape (out_features,) FP32.

    x_rot must already be rotated by Pi.T (do this once before calling
    a stack of these — Pi is shared across all linears in the model)."""
    assert x_rot.is_contiguous() and x_rot.dtype == torch.float32
    assert qs.is_contiguous() and qs.dtype == torch.uint8
    assert d.is_contiguous() and d.dtype == torch.float32
    assert centroids.is_contiguous() and centroids.dtype == torch.float32
    assert in_features % 256 == 0
    bpr = in_features // 256
    n_blocks = qs.shape[0]
    assert n_blocks == out_features * bpr, (
        f"qs has {n_blocks} blocks, expected {out_features * bpr}")

    y = torch.empty(out_features, device=x_rot.device, dtype=torch.float32)
    grid = (out_features,)
    _tq4_matvec_kernel[grid](
        x_rot, qs.view(-1), d, centroids, y,
        in_features, out_features,
        BPR=bpr,
        BLOCK_HALF=128,
        num_warps=4,
    )
    return y


def tq4_linear_triton(
    x: torch.Tensor,
    qs: torch.Tensor,
    d: torch.Tensor,
    pi: torch.Tensor,           # (256, 256) FP32
    centroids: torch.Tensor,
    out_features: int,
    in_features: int,
) -> torch.Tensor:
    """Full tq4 linear: rotate x by Pi.T, then fused matvec.
    Handles arbitrary leading batch dims by flattening + looping.
    For the (B=1, S=1) decode case this is the hot path."""
    *batch, in_f = x.shape
    assert in_f == in_features and in_f % 256 == 0
    bpr = in_f // 256
    # Pi rotation: small matmul, cheap
    x_rot = (x.reshape(*batch, bpr, 256) @ pi.T).reshape(*batch, in_f)
    if not x_rot.is_contiguous():
        x_rot = x_rot.contiguous()

    if x_rot.dim() == 1 or (x_rot.dim() > 1 and x_rot.numel() == in_f):
        # Single vector (could be (in,) or (1, 1, in) etc.)
        y_flat = tq4_matvec_triton(
            x_rot.reshape(in_f), qs, d, centroids, out_features, in_features)
        return y_flat.reshape(*batch, out_features)

    # Multi-vector: loop. Slow but correct; replace with batched kernel later.
    flat = x_rot.reshape(-1, in_f)
    out = torch.empty(flat.shape[0], out_features,
                       device=x.device, dtype=torch.float32)
    for i in range(flat.shape[0]):
        out[i] = tq4_matvec_triton(
            flat[i].contiguous(), qs, d, centroids,
            out_features, in_features)
    return out.reshape(*batch, out_features)


# --- Q6_K fused dequant + matvec (output head / embeddings) ---

@triton.jit
def _q6k_matvec_kernel(
    h_ptr,         # (d_model,) FP32, the hidden state row
    ql_ptr,        # (n_blocks * 128,) uint8 — low 4 bits, packed
    qh_ptr,        # (n_blocks * 64,)  uint8 — high 2 bits, packed
    scales_ptr,    # (n_blocks * 16,)  int8  — per-sub-block scales
    d_ptr,         # (n_blocks,)       FP32  — per-block super-scale
    y_ptr,         # (vocab_size,)     FP32  — output
    vocab_size,
    BPR: tl.constexpr,        # blocks per vocab row (= d_model / 256)
    BLOCK_ELEMENTS: tl.constexpr,  # 256
):
    """One program = one vocab row's logit. Fused Q6_K dequant + dot product
    with h. Uses llama.cpp's Q6_K layout exactly (see q6k_dequant.py)."""
    pid = tl.program_id(0)
    if pid >= vocab_size:
        return

    pos = tl.arange(0, BLOCK_ELEMENTS)             # 0..255
    half = pos // 128
    within = pos % 128
    quarter = within // 32
    l = within % 32

    # Per-position byte indices and bit shifts (constant across blocks).
    ql_idx = half * 64 + l + tl.where(
        (quarter == 1) | (quarter == 3), 32, 0)
    ql_shift = tl.where(quarter >= 2, 4, 0)
    qh_idx = half * 32 + l
    qh_shift = 2 * quarter
    scale_idx = half * 8 + (l // 16) + 2 * quarter

    acc = tl.zeros((), dtype=tl.float32)
    for b in range(BPR):
        block_idx = pid * BPR + b
        d_block = tl.load(d_ptr + block_idx)

        ql_vals = tl.load(ql_ptr + block_idx * 128 + ql_idx).to(tl.int32)
        qh_vals = tl.load(qh_ptr + block_idx * 64 + qh_idx).to(tl.int32)
        scale_vals = tl.load(scales_ptr + block_idx * 16 + scale_idx).to(tl.int32)

        ql_low = (ql_vals >> ql_shift) & 0xF
        qh_high = (qh_vals >> qh_shift) & 0x3
        q = ql_low | (qh_high << 4)        # 0..63
        q_signed = q - 32                  # -32..31

        val = d_block * scale_vals.to(tl.float32) * q_signed.to(tl.float32)

        h_block = tl.load(h_ptr + b * BLOCK_ELEMENTS + pos)
        acc += tl.sum(val * h_block)

    tl.store(y_ptr + pid, acc)


def q6k_matvec_triton(
    h: torch.Tensor,           # (d_model,) FP32
    ql: torch.Tensor,          # (n_blocks, 128) uint8
    qh: torch.Tensor,          # (n_blocks, 64)  uint8
    scales: torch.Tensor,      # (n_blocks, 16)  int8
    d: torch.Tensor,           # (n_blocks,)     FP32
    vocab_size: int,
    d_model: int,
) -> torch.Tensor:
    """Fused Q6_K dequant + matvec for the output head / embedding lookup.
    Returns logits of shape (vocab_size,) FP32."""
    assert h.is_contiguous() and h.dtype == torch.float32
    assert ql.is_contiguous() and ql.dtype == torch.uint8
    assert qh.is_contiguous() and qh.dtype == torch.uint8
    assert scales.is_contiguous() and scales.dtype == torch.int8
    assert d.is_contiguous() and d.dtype == torch.float32
    assert d_model % 256 == 0
    bpr = d_model // 256
    assert ql.shape == (vocab_size * bpr, 128)
    assert qh.shape == (vocab_size * bpr, 64)
    assert scales.shape == (vocab_size * bpr, 16)
    assert d.shape == (vocab_size * bpr,)

    y = torch.empty(vocab_size, device=h.device, dtype=torch.float32)
    grid = (vocab_size,)
    _q6k_matvec_kernel[grid](
        h, ql.view(-1), qh.view(-1), scales.view(-1), d, y,
        vocab_size,
        BPR=bpr,
        BLOCK_ELEMENTS=256,
        num_warps=4,
    )
    return y
