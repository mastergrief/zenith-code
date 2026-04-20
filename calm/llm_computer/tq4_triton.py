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
    BLOCK_M: tl.constexpr,       # rows of W per program (tiling factor)
):
    """One program = BLOCK_M consecutive output rows. The x_rot segment
    for each input block is loaded ONCE per program and reused across
    BLOCK_M dot products — saves redundant x_rot reads vs the BLOCK_M=1
    version. Within the program, BLOCK_M dot products run in parallel."""
    pid = tl.program_id(0)
    row_base = pid * BLOCK_M
    if row_base >= out_features:
        return

    half_idx = tl.arange(0, BLOCK_HALF)            # (128,)
    m_idx = tl.arange(0, BLOCK_M)                  # (BLOCK_M,) row offsets
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for b in range(BPR):
        # x_rot for this input block: load once, broadcast across BLOCK_M rows
        x_base = b * 256
        x_low = tl.load(x_rot_ptr + x_base + 2 * half_idx)    # (128,)
        x_high = tl.load(x_rot_ptr + x_base + 2 * half_idx + 1)

        # Per-row block index: (row_base + m_idx) * BPR + b. Build (BLOCK_M,).
        block_idx_m = (row_base + m_idx) * BPR + b   # (BLOCK_M,)
        # Per-row d
        d_m = tl.load(d_ptr + block_idx_m)            # (BLOCK_M,)

        # Per-row qs: (BLOCK_M, 128) — broadcast block_idx_m over half_idx
        qs_offsets = (block_idx_m[:, None] * BLOCK_HALF
                      + half_idx[None, :])             # (BLOCK_M, 128)
        qs_m = tl.load(qs_ptr + qs_offsets).to(tl.int32)
        low_m = qs_m & 0xF
        high_m = (qs_m >> 4) & 0xF
        c_low_m = tl.load(centroids_ptr + low_m)      # (BLOCK_M, 128)
        c_high_m = tl.load(centroids_ptr + high_m)

        # Dot products: (BLOCK_M, 128) x (128,) → (BLOCK_M,)
        block_dot = (tl.sum(c_low_m * x_low[None, :], axis=1)
                     + tl.sum(c_high_m * x_high[None, :], axis=1))
        acc += block_dot * d_m

    # Store BLOCK_M outputs, masking out-of-range rows
    rows = row_base + m_idx
    tl.store(y_ptr + rows, acc, mask=rows < out_features)


def tq4_matvec_triton_v1(
    x_rot: torch.Tensor, qs: torch.Tensor, d: torch.Tensor,
    centroids: torch.Tensor, out_features: int, in_features: int,
) -> torch.Tensor:
    """V1 dispatch: original global-memory gather (pre-R53.29 baseline).
    Kept for bench comparison; production uses tq4_matvec_triton → v2."""
    assert x_rot.is_contiguous() and x_rot.dtype == torch.float32
    assert qs.is_contiguous() and qs.dtype == torch.uint8
    assert d.is_contiguous() and d.dtype == torch.float32
    assert centroids.is_contiguous() and centroids.dtype == torch.float32
    assert in_features % 256 == 0
    bpr = in_features // 256
    y = torch.empty(out_features, device=x_rot.device, dtype=torch.float32)
    BLOCK_M = _pick_block_m(out_features)
    grid = ((out_features + BLOCK_M - 1) // BLOCK_M,)
    _tq4_matvec_kernel[grid](
        x_rot, qs.view(-1), d, centroids, y,
        in_features, out_features,
        BPR=bpr, BLOCK_HALF=128, BLOCK_M=BLOCK_M,
        num_warps=4,
    )
    return y


@triton.jit
def _tq4_matvec_kernel_v2(
    x_rot_ptr, qs_ptr, d_ptr, centroids_ptr, y_ptr,
    in_features, out_features,
    BPR: tl.constexpr,
    BLOCK_HALF: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """V2 — ports TurboQuant commit 51481c3 shared-mem LUT.
    Loads centroids into a program-local (16,) tile once, uses tl.gather."""
    pid = tl.program_id(0)
    row_base = pid * BLOCK_M
    if row_base >= out_features:
        return

    centroid_tile = tl.load(centroids_ptr + tl.arange(0, 16))

    half_idx = tl.arange(0, BLOCK_HALF)
    m_idx = tl.arange(0, BLOCK_M)
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for b in range(BPR):
        x_base = b * 256
        x_low = tl.load(x_rot_ptr + x_base + 2 * half_idx)
        x_high = tl.load(x_rot_ptr + x_base + 2 * half_idx + 1)

        block_idx_m = (row_base + m_idx) * BPR + b
        d_m = tl.load(d_ptr + block_idx_m)

        qs_offsets = (block_idx_m[:, None] * BLOCK_HALF + half_idx[None, :])
        qs_m = tl.load(qs_ptr + qs_offsets).to(tl.int32)
        low_m = qs_m & 0xF
        high_m = (qs_m >> 4) & 0xF

        low_flat = tl.reshape(low_m, (BLOCK_M * BLOCK_HALF,))
        high_flat = tl.reshape(high_m, (BLOCK_M * BLOCK_HALF,))
        c_low_m = tl.reshape(tl.gather(centroid_tile, low_flat, axis=0),
                             (BLOCK_M, BLOCK_HALF))
        c_high_m = tl.reshape(tl.gather(centroid_tile, high_flat, axis=0),
                              (BLOCK_M, BLOCK_HALF))

        block_dot = (tl.sum(c_low_m * x_low[None, :], axis=1)
                     + tl.sum(c_high_m * x_high[None, :], axis=1))
        acc += block_dot * d_m

    rows = row_base + m_idx
    tl.store(y_ptr + rows, acc, mask=rows < out_features)


@triton.jit
def _tq4_matvec_kernel_v4(
    x_rot_ptr, qs_u32_ptr, d_ptr, centroids_ptr, y_ptr,
    in_features, out_features,
    BPR: tl.constexpr,
    BLOCK_QUARTER: tl.constexpr,   # 32 = 128 bytes / 4 bytes per uint32
    BLOCK_M: tl.constexpr,
):
    """V4 — v2 + vectorized uint32 weight loads.

    qs is loaded as uint32 (4 bytes per load) instead of uint8 (1 byte).
    Each uint32 packs 8 nybbles (4 low + 4 high). Trades more arithmetic
    (bit-shift unpacks) for fewer load instructions and better coalescing.

    TurboQuant commit 51481c3 reports +45% from this technique in CUDA."""
    pid = tl.program_id(0)
    row_base = pid * BLOCK_M
    if row_base >= out_features:
        return

    centroid_tile = tl.load(centroids_ptr + tl.arange(0, 16))

    # 32 uint32s per 128-byte half-block. Each uint32 = 4 bytes = 8 nybbles.
    quarter_idx = tl.arange(0, BLOCK_QUARTER)   # (32,)
    m_idx = tl.arange(0, BLOCK_M)
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    BLOCK_HALF: tl.constexpr = 128

    for b in range(BPR):
        x_base = b * 256
        # x_rot — 128 low + 128 high fp32 values
        x_low = tl.load(x_rot_ptr + x_base + 2 * tl.arange(0, BLOCK_HALF))
        x_high = tl.load(x_rot_ptr + x_base + 2 * tl.arange(0, BLOCK_HALF) + 1)

        block_idx_m = (row_base + m_idx) * BPR + b
        d_m = tl.load(d_ptr + block_idx_m)

        # qs loaded as (BLOCK_M, 32) uint32 — 32 loads × BLOCK_M rows instead of 128
        u32_offsets = (block_idx_m[:, None] * BLOCK_QUARTER
                       + quarter_idx[None, :])
        qs_u32 = tl.load(qs_u32_ptr + u32_offsets).to(tl.uint32)

        # Unpack 4 bytes per uint32 → 4 low-nybbles + 4 high-nybbles
        # per byte. Byte 0 is at bits 0..7, byte 1 at 8..15, etc.
        # Each byte has low nybble in bits 0..3, high in bits 4..7.
        b0 = (qs_u32) & 0xFF
        b1 = (qs_u32 >> 8) & 0xFF
        b2 = (qs_u32 >> 16) & 0xFF
        b3 = (qs_u32 >> 24) & 0xFF

        # Shape (BLOCK_M, 32) each. Reconstruct (BLOCK_M, 128) by interleaving:
        # byte-index in the 128-byte half = 4*quarter_idx + byte_offset_in_u32
        # Stack into (BLOCK_M, 32, 4) then reshape (BLOCK_M, 128).
        bytes_stack = tl.join(tl.join(b0, b1), tl.join(b2, b3))  # (BLOCK_M, 32, 4)
        qs_m = tl.reshape(bytes_stack.to(tl.int32), (BLOCK_M, BLOCK_HALF))

        low_m = qs_m & 0xF
        high_m = (qs_m >> 4) & 0xF

        low_flat = tl.reshape(low_m, (BLOCK_M * BLOCK_HALF,))
        high_flat = tl.reshape(high_m, (BLOCK_M * BLOCK_HALF,))
        c_low_m = tl.reshape(tl.gather(centroid_tile, low_flat, axis=0),
                             (BLOCK_M, BLOCK_HALF))
        c_high_m = tl.reshape(tl.gather(centroid_tile, high_flat, axis=0),
                              (BLOCK_M, BLOCK_HALF))

        block_dot = (tl.sum(c_low_m * x_low[None, :], axis=1)
                     + tl.sum(c_high_m * x_high[None, :], axis=1))
        acc += block_dot * d_m

    rows = row_base + m_idx
    tl.store(y_ptr + rows, acc, mask=rows < out_features)


def tq4_matvec_triton_v4(
    x_rot: torch.Tensor, qs: torch.Tensor, d: torch.Tensor,
    centroids: torch.Tensor, out_features: int, in_features: int,
) -> torch.Tensor:
    """V4 dispatch: v2 + uint32 vectorized qs loads."""
    assert x_rot.is_contiguous() and x_rot.dtype == torch.float32
    assert qs.is_contiguous() and qs.dtype == torch.uint8
    assert d.is_contiguous() and d.dtype == torch.float32
    assert centroids.is_contiguous() and centroids.dtype == torch.float32
    assert in_features % 256 == 0
    bpr = in_features // 256
    # Reinterpret qs (uint8 of shape (n_blocks, 128)) as uint32 of shape
    # (n_blocks, 32) — view only, zero copy.
    qs_u32 = qs.view(torch.int32)  # Triton treats int32 ptr same as uint32
    y = torch.empty(out_features, device=x_rot.device, dtype=torch.float32)
    BLOCK_M = _pick_block_m(out_features)
    grid = ((out_features + BLOCK_M - 1) // BLOCK_M,)
    _tq4_matvec_kernel_v4[grid](
        x_rot, qs_u32.view(-1), d, centroids, y,
        in_features, out_features,
        BPR=bpr, BLOCK_QUARTER=32, BLOCK_M=BLOCK_M,
        num_warps=4,
    )
    return y


# NOTE: v3 (fp16 x_rot activation) tested in R53.30 — null result.
# Upcast-inside-dot overhead offsets the BW savings on Ada; kept below
# for future reference but not dispatched from production.
@triton.jit
def _tq4_matvec_kernel_v3(
    x_rot_ptr, qs_ptr, d_ptr, centroids_ptr, y_ptr,
    in_features, out_features,
    BPR: tl.constexpr,
    BLOCK_HALF: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """V3 — v2 + fp16 activation buffer (TurboQuant fp16-activation win).
    Expects x_rot_ptr pointing to fp16 tensor. Halves x_rot bandwidth.
    Upcasts to fp32 inside dot product accumulation for numerical stability."""
    pid = tl.program_id(0)
    row_base = pid * BLOCK_M
    if row_base >= out_features:
        return

    centroid_tile = tl.load(centroids_ptr + tl.arange(0, 16))  # fp32

    half_idx = tl.arange(0, BLOCK_HALF)
    m_idx = tl.arange(0, BLOCK_M)
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for b in range(BPR):
        x_base = b * 256
        # fp16 loads — half the bytes of fp32
        x_low = tl.load(x_rot_ptr + x_base + 2 * half_idx).to(tl.float32)
        x_high = tl.load(x_rot_ptr + x_base + 2 * half_idx + 1).to(tl.float32)

        block_idx_m = (row_base + m_idx) * BPR + b
        d_m = tl.load(d_ptr + block_idx_m)

        qs_offsets = (block_idx_m[:, None] * BLOCK_HALF + half_idx[None, :])
        qs_m = tl.load(qs_ptr + qs_offsets).to(tl.int32)
        low_m = qs_m & 0xF
        high_m = (qs_m >> 4) & 0xF

        low_flat = tl.reshape(low_m, (BLOCK_M * BLOCK_HALF,))
        high_flat = tl.reshape(high_m, (BLOCK_M * BLOCK_HALF,))
        c_low_m = tl.reshape(tl.gather(centroid_tile, low_flat, axis=0),
                             (BLOCK_M, BLOCK_HALF))
        c_high_m = tl.reshape(tl.gather(centroid_tile, high_flat, axis=0),
                              (BLOCK_M, BLOCK_HALF))

        block_dot = (tl.sum(c_low_m * x_low[None, :], axis=1)
                     + tl.sum(c_high_m * x_high[None, :], axis=1))
        acc += block_dot * d_m

    rows = row_base + m_idx
    tl.store(y_ptr + rows, acc, mask=rows < out_features)


def tq4_matvec_triton_v3(
    x_rot: torch.Tensor, qs: torch.Tensor, d: torch.Tensor,
    centroids: torch.Tensor, out_features: int, in_features: int,
) -> torch.Tensor:
    """V3 dispatch: v2 + fp16 x_rot activation."""
    assert qs.is_contiguous() and qs.dtype == torch.uint8
    assert d.is_contiguous() and d.dtype == torch.float32
    assert centroids.is_contiguous() and centroids.dtype == torch.float32
    assert in_features % 256 == 0
    # Ensure fp16 contiguous x_rot
    if x_rot.dtype != torch.float16:
        x_rot = x_rot.to(torch.float16).contiguous()
    elif not x_rot.is_contiguous():
        x_rot = x_rot.contiguous()
    bpr = in_features // 256
    y = torch.empty(out_features, device=x_rot.device, dtype=torch.float32)
    BLOCK_M = _pick_block_m(out_features)
    grid = ((out_features + BLOCK_M - 1) // BLOCK_M,)
    _tq4_matvec_kernel_v3[grid](
        x_rot, qs.view(-1), d, centroids, y,
        in_features, out_features,
        BPR=bpr, BLOCK_HALF=128, BLOCK_M=BLOCK_M,
        num_warps=4,
    )
    return y


# ----------------------------------------------------------------------
# V5 — int8 activation + int8 centroid LUT + int32 accumulation.
# NULL RESULT (iteration 1). Attempted port of TurboQuant CUDA commit
# 51481c3's dp4a path to Triton. Correctness OK (cosine 0.99996, max
# rel err ≤1.01%) but perf +9.4% SLOWER on aggregate vs v2:
#   (2560, 2048): -0.3%   (2560, 512): +42.6%   (2048, 2560): -5.8%
#   (2560, 10240): +0.7%  (10240, 2560): +12.0%
# Diagnosis: v2 already hits 284-364 GB/s on large shapes (L2-resident,
# above HBM peak). Compute-bound on L2, not BW-bound on HBM, so v5's
# 4× activation BW reduction doesn't help. Meanwhile int8→int32 cast +
# int32 mul emits more Triton instructions than fp32 FMA (same IPC on
# Ada), so compute increases. CUDA's 3.5× came from __dp4a intrinsic
# (4 int8 MACs/cycle); Triton's `a.to(int32) * b.to(int32)` pattern
# does NOT emit IDP4A — compiler falls back to scalar int32.
# Next lever if revisiting: tl.dot with int8 inputs + tensor cores,
# or explicit packed-uint32 dp4a emulation.
# Kept for reference; not dispatched from production.
# ----------------------------------------------------------------------


@triton.jit
def _tq4_matvec_kernel_v5(
    x_rot_q8_ptr,       # (in_features,) int8, pre-quantized per 256-block
    x_scale_ptr,        # (bpr,) fp32, per-block activation scale (max_abs/127)
    qs_ptr,             # (n_blocks * 128,) uint8
    d_ptr,              # (n_blocks,) fp32 — fused d * centroid_rescale
    centroids_i8_ptr,   # (16,) int8, centroids rounded to int8
    y_ptr,
    in_features, out_features,
    BPR: tl.constexpr,
    BLOCK_HALF: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """V5 — int8 path. `d_ptr` must be pre-fused: d_fused = d_tq4 *
    centroid_rescale where centroid_rescale = centroid_max_abs / 127.
    `x_scale_ptr` holds per-block activation max_abs/127. Final output:
    y = sum(int8_centroid * int8_activation) * d_fused * x_scale (fp32)."""
    pid = tl.program_id(0)
    row_base = pid * BLOCK_M
    if row_base >= out_features:
        return

    centroid_tile = tl.load(centroids_i8_ptr + tl.arange(0, 16))  # int8

    half_idx = tl.arange(0, BLOCK_HALF)
    m_idx = tl.arange(0, BLOCK_M)
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for b in range(BPR):
        x_base = b * 256
        # int8 activation loads — 1 byte each vs 4 for fp32
        x_low = tl.load(x_rot_q8_ptr + x_base + 2 * half_idx)    # int8
        x_high = tl.load(x_rot_q8_ptr + x_base + 2 * half_idx + 1)
        x_sc = tl.load(x_scale_ptr + b)  # fp32

        block_idx_m = (row_base + m_idx) * BPR + b
        d_m = tl.load(d_ptr + block_idx_m)  # fp32, pre-fused with centroid_rescale

        qs_offsets = (block_idx_m[:, None] * BLOCK_HALF + half_idx[None, :])
        qs_m = tl.load(qs_ptr + qs_offsets).to(tl.int32)
        low_m = qs_m & 0xF
        high_m = (qs_m >> 4) & 0xF

        # Gather int8 centroids
        low_flat = tl.reshape(low_m, (BLOCK_M * BLOCK_HALF,))
        high_flat = tl.reshape(high_m, (BLOCK_M * BLOCK_HALF,))
        c_low_m = tl.reshape(tl.gather(centroid_tile, low_flat, axis=0),
                             (BLOCK_M, BLOCK_HALF))  # int8
        c_high_m = tl.reshape(tl.gather(centroid_tile, high_flat, axis=0),
                              (BLOCK_M, BLOCK_HALF))

        # int8 × int8 → int32 product, sum along contraction dim.
        # Max block sum magnitude: 128 * 127 * 127 = 2.06M — fits in int32.
        prod_low = c_low_m.to(tl.int32) * x_low[None, :].to(tl.int32)
        prod_high = c_high_m.to(tl.int32) * x_high[None, :].to(tl.int32)
        block_sum = (tl.sum(prod_low, axis=1)
                     + tl.sum(prod_high, axis=1))  # (BLOCK_M,) int32

        acc += block_sum.to(tl.float32) * d_m * x_sc

    rows = row_base + m_idx
    tl.store(y_ptr + rows, acc, mask=rows < out_features)


def _quantize_activation_q8(x_rot: torch.Tensor, bpr: int
                             ) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-256-block max-abs int8 quantization. Returns (int8_vals, fp32_scale).
    scale[b] = max_abs[b] / 127. Recover fp32 as int8_val * scale[b]."""
    x_blocks = x_rot.reshape(bpr, 256)
    max_abs = x_blocks.abs().amax(dim=1)                        # (bpr,)
    scale = max_abs / 127.0
    # Avoid div-by-zero — zero blocks produce zero output anyway
    inv_scale = torch.where(max_abs > 0,
                            127.0 / max_abs.clamp(min=1e-30),
                            torch.zeros_like(max_abs))
    q8 = (x_blocks * inv_scale.unsqueeze(1)).round().clamp(-127, 127).to(torch.int8)
    return q8.reshape(-1).contiguous(), scale.contiguous()


def _prep_centroids_i8(centroids_fp32: torch.Tensor
                        ) -> tuple[torch.Tensor, float]:
    """Quantize fp32 centroids to int8. Returns (int8_centroids, rescale).
    rescale = max_abs / 127 so fp32_centroid ≈ int8_val * rescale."""
    max_abs = float(centroids_fp32.abs().max().item())
    if max_abs == 0:
        return torch.zeros(16, dtype=torch.int8, device=centroids_fp32.device), 1.0
    rescale = max_abs / 127.0
    c_i8 = (centroids_fp32 / rescale).round().clamp(-127, 127).to(torch.int8)
    return c_i8.contiguous(), rescale


def tq4_matvec_triton_v5(
    x_rot: torch.Tensor, qs: torch.Tensor, d: torch.Tensor,
    centroids: torch.Tensor, out_features: int, in_features: int,
) -> torch.Tensor:
    """V5 dispatch: int8 activation + int8 centroid + int32 accumulation.
    Activation is quantized per-block inside this call. In production
    the caller should cache x_rot_q8/x_scale across Q/K/V/output or
    gate/up to amortize quant cost."""
    assert x_rot.is_contiguous() and x_rot.dtype == torch.float32
    assert qs.is_contiguous() and qs.dtype == torch.uint8
    assert d.is_contiguous() and d.dtype == torch.float32
    assert centroids.is_contiguous() and centroids.dtype == torch.float32
    assert in_features % 256 == 0
    bpr = in_features // 256

    # Per-block int8 activation + scale
    x_q8, x_scale = _quantize_activation_q8(x_rot, bpr)
    # int8 centroids + rescale folded into d
    c_i8, c_rescale = _prep_centroids_i8(centroids)
    d_fused = (d * c_rescale).contiguous()

    y = torch.empty(out_features, device=x_rot.device, dtype=torch.float32)
    BLOCK_M = _pick_block_m(out_features)
    grid = ((out_features + BLOCK_M - 1) // BLOCK_M,)
    _tq4_matvec_kernel_v5[grid](
        x_q8, x_scale, qs.view(-1), d_fused, c_i8, y,
        in_features, out_features,
        BPR=bpr, BLOCK_HALF=128, BLOCK_M=BLOCK_M,
        num_warps=4,
    )
    return y


def tq4_matvec_triton_v5_prequant(
    x_q8: torch.Tensor, x_scale: torch.Tensor,
    qs: torch.Tensor, d_fused: torch.Tensor, centroids_i8: torch.Tensor,
    out_features: int, in_features: int,
) -> torch.Tensor:
    """V5 fast path: activation already quantized, d already fused with
    centroid rescale. For apples-to-apples perf bench against v2."""
    assert x_q8.is_contiguous() and x_q8.dtype == torch.int8
    assert x_scale.is_contiguous() and x_scale.dtype == torch.float32
    assert qs.is_contiguous() and qs.dtype == torch.uint8
    assert d_fused.is_contiguous() and d_fused.dtype == torch.float32
    assert centroids_i8.is_contiguous() and centroids_i8.dtype == torch.int8
    assert in_features % 256 == 0
    bpr = in_features // 256
    y = torch.empty(out_features, device=x_q8.device, dtype=torch.float32)
    BLOCK_M = _pick_block_m(out_features)
    grid = ((out_features + BLOCK_M - 1) // BLOCK_M,)
    _tq4_matvec_kernel_v5[grid](
        x_q8, x_scale, qs.view(-1), d_fused, centroids_i8, y,
        in_features, out_features,
        BPR=bpr, BLOCK_HALF=128, BLOCK_M=BLOCK_M,
        num_warps=4,
    )
    return y


# ----------------------------------------------------------------------
# V6 — int8 tl.dot via Ada int8 tensor cores (IMMA).
# NULL RESULT (iteration 2 after v5 null). Ports CUDA dp4a concept to
# Triton via tl.dot(a_i8, b_i8). Correctness 5/5 cosine 0.99996.
# Aggregate +4.7% SLOWER vs v2, with revealing BPR-dependent pattern:
#   BPR=8  (2048,2560 attn_out):  -8.9% (WINS)
#   BPR=10 (2560,2048 / 2560,10240): +3.5% to +11.4%
#   BPR=40 (10240,2560 ffn_down):  +22.5% (LOSES hard)
# Diagnosis: per-block tl.dot has fixed setup cost; short BPR
# amortizes well, long BPR eats the savings. Plus N=1 padded to N=16
# wastes 15/16 of tensor-core FLOPs.
# Next lever if revisiting (iteration 3, not attempted):
# widen K per tl.dot to combine multiple blocks (K=512 or K=1024),
# cutting call count proportionally. Per-block scales (d_m, x_scale)
# are the blocker — need pre-folding into weight preprocessing.
# Kept for reference; not dispatched from production.
# ----------------------------------------------------------------------


@triton.jit
def _tq4_matvec_kernel_v6(
    x_rot_q8_ptr,       # (in_features,) int8
    x_scale_ptr,        # (bpr,) fp32
    qs_ptr,             # (n_blocks * 128,) uint8
    d_ptr,              # (n_blocks,) fp32, pre-fused with centroid_rescale
    centroids_i8_ptr,   # (16,) int8
    y_ptr,
    in_features, out_features,
    BPR: tl.constexpr,
    BLOCK_HALF: tl.constexpr,   # 128
    BLOCK_FULL: tl.constexpr,   # 256 (K dim for tl.dot)
    BLOCK_M: tl.constexpr,      # ≥ 16 required for tensor cores
    N_PAD: tl.constexpr,        # 16 (N dim padding for tensor core min)
):
    """V6 — one tl.dot per block, N=16-padded, int8 tensor cores."""
    pid = tl.program_id(0)
    row_base = pid * BLOCK_M
    if row_base >= out_features:
        return

    centroid_tile = tl.load(centroids_i8_ptr + tl.arange(0, 16))  # int8

    half_idx = tl.arange(0, BLOCK_HALF)
    full_idx = tl.arange(0, BLOCK_FULL)
    m_idx = tl.arange(0, BLOCK_M)
    n_idx = tl.arange(0, N_PAD)
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for b in range(BPR):
        x_base = b * 256
        # Load full 256 int8 x_rot (low+high interleaved naturally by position)
        x_full = tl.load(x_rot_q8_ptr + x_base + full_idx)  # (256,) int8
        x_sc = tl.load(x_scale_ptr + b)  # fp32

        # Per-row block index and d
        block_idx_m = (row_base + m_idx) * BPR + b
        d_m = tl.load(d_ptr + block_idx_m)  # (BLOCK_M,) fp32

        # Load qs for all rows this block: (BLOCK_M, 128)
        qs_offsets = (block_idx_m[:, None] * BLOCK_HALF + half_idx[None, :])
        qs_m = tl.load(qs_ptr + qs_offsets).to(tl.int32)
        low_m = qs_m & 0xF       # (BLOCK_M, 128)
        high_m = (qs_m >> 4) & 0xF

        # Gather int8 centroids
        low_flat = tl.reshape(low_m, (BLOCK_M * BLOCK_HALF,))
        high_flat = tl.reshape(high_m, (BLOCK_M * BLOCK_HALF,))
        c_low_m = tl.reshape(tl.gather(centroid_tile, low_flat, axis=0),
                             (BLOCK_M, BLOCK_HALF))   # int8
        c_high_m = tl.reshape(tl.gather(centroid_tile, high_flat, axis=0),
                              (BLOCK_M, BLOCK_HALF))

        # Build full K=256 weight tile: [c_low | c_high] along K.
        # tq4 layout: first 128 elements = low nybbles, next 128 = high.
        # x_rot layout: interleaved (x_rot[0]=low[0], x_rot[1]=high[0], ...)
        # So we need c_m[r, k] where k = 2*i gives low[i], k=2*i+1 gives high[i].
        # Equivalent: construct c_m by interleaving along dim 1.
        # tl.join doesn't stack along existing dim — instead use reshape:
        #   stack(c_low, c_high) along new axis: (BLOCK_M, 128, 2) then
        #   reshape to (BLOCK_M, 256). The last axis becomes innermost.
        c_stack = tl.join(c_low_m, c_high_m)  # (BLOCK_M, 128, 2) int8
        c_m = tl.reshape(c_stack, (BLOCK_M, BLOCK_FULL))  # (BLOCK_M, 256) int8

        # Broadcast x to (256, N_PAD=16): first column = x, rest = 0
        # Cheaper: tile x across N, multiply result by one-hot at end.
        # But Triton's tl.zeros + assignment is awkward. Instead tile x
        # and take [:, 0] post-dot.
        x_tile = tl.broadcast_to(x_full[:, None], (BLOCK_FULL, N_PAD))  # (256, 16) int8

        # tl.dot: (BLOCK_M, 256) × (256, 16) = (BLOCK_M, 16) int32
        block_prod = tl.dot(c_m, x_tile, out_dtype=tl.int32)  # (BLOCK_M, 16)

        # Take [:, 0] — all 16 columns are identical (x was broadcast)
        block_sum = tl.sum(block_prod * (n_idx[None, :] == 0).to(tl.int32),
                           axis=1)  # (BLOCK_M,) int32

        acc += block_sum.to(tl.float32) * d_m * x_sc

    rows = row_base + m_idx
    tl.store(y_ptr + rows, acc, mask=rows < out_features)


def tq4_matvec_triton_v6_prequant(
    x_q8: torch.Tensor, x_scale: torch.Tensor,
    qs: torch.Tensor, d_fused: torch.Tensor, centroids_i8: torch.Tensor,
    out_features: int, in_features: int,
) -> torch.Tensor:
    """V6 fast path: activation pre-quantized. Requires BLOCK_M ≥ 16
    (tensor core constraint). Caller responsible for only dispatching
    when out_features supports BLOCK_M=16+."""
    assert x_q8.is_contiguous() and x_q8.dtype == torch.int8
    assert x_scale.is_contiguous() and x_scale.dtype == torch.float32
    assert qs.is_contiguous() and qs.dtype == torch.uint8
    assert d_fused.is_contiguous() and d_fused.dtype == torch.float32
    assert centroids_i8.is_contiguous() and centroids_i8.dtype == torch.int8
    assert in_features % 256 == 0
    bpr = in_features // 256
    BLOCK_M = _pick_block_m(out_features)
    assert BLOCK_M >= 16, (
        f"v6 needs BLOCK_M >= 16 for int8 tensor cores, got {BLOCK_M} "
        f"for out_features={out_features}. Use v5 or v2 fallback.")

    y = torch.empty(out_features, device=x_q8.device, dtype=torch.float32)
    grid = ((out_features + BLOCK_M - 1) // BLOCK_M,)
    _tq4_matvec_kernel_v6[grid](
        x_q8, x_scale, qs.view(-1), d_fused, centroids_i8, y,
        in_features, out_features,
        BPR=bpr, BLOCK_HALF=128, BLOCK_FULL=256, BLOCK_M=BLOCK_M, N_PAD=16,
        num_warps=4,
    )
    return y


def tq4_matvec_triton_v6(
    x_rot: torch.Tensor, qs: torch.Tensor, d: torch.Tensor,
    centroids: torch.Tensor, out_features: int, in_features: int,
) -> torch.Tensor:
    """V6 dispatch: quantize activation inside call. Falls back to v2
    when BLOCK_M < 16 (tensor cores unavailable)."""
    assert x_rot.is_contiguous() and x_rot.dtype == torch.float32
    BLOCK_M = _pick_block_m(out_features)
    if BLOCK_M < 16:
        return tq4_matvec_triton_v2(x_rot, qs, d, centroids,
                                     out_features, in_features)
    bpr = in_features // 256
    x_q8, x_scale = _quantize_activation_q8(x_rot, bpr)
    c_i8, c_rescale = _prep_centroids_i8(centroids)
    d_fused = (d * c_rescale).contiguous()
    return tq4_matvec_triton_v6_prequant(
        x_q8, x_scale, qs, d_fused, c_i8, out_features, in_features)


def tq4_matvec_triton_v2(
    x_rot: torch.Tensor, qs: torch.Tensor, d: torch.Tensor,
    centroids: torch.Tensor, out_features: int, in_features: int,
) -> torch.Tensor:
    """V2 dispatch: shared-mem LUT variant."""
    assert x_rot.is_contiguous() and x_rot.dtype == torch.float32
    assert qs.is_contiguous() and qs.dtype == torch.uint8
    assert d.is_contiguous() and d.dtype == torch.float32
    assert centroids.is_contiguous() and centroids.dtype == torch.float32
    assert in_features % 256 == 0
    bpr = in_features // 256
    y = torch.empty(out_features, device=x_rot.device, dtype=torch.float32)
    BLOCK_M = _pick_block_m(out_features)
    grid = ((out_features + BLOCK_M - 1) // BLOCK_M,)
    _tq4_matvec_kernel_v2[grid](
        x_rot, qs.view(-1), d, centroids, y,
        in_features, out_features,
        BPR=bpr, BLOCK_HALF=128, BLOCK_M=BLOCK_M,
        num_warps=4,
    )
    return y


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
    BLOCK_M = _pick_block_m(out_features)
    grid = ((out_features + BLOCK_M - 1) // BLOCK_M,)
    # R53.29: use v2 kernel (shared-mem LUT via tl.gather) — -5% to -10%
    # faster on aggregate across Gemma 4 E4B shapes vs the baseline
    # global-memory gather. Verified correct via test_tq4_matvec_v2_correctness.
    _tq4_matvec_kernel_v2[grid](
        x_rot, qs.view(-1), d, centroids, y,
        in_features, out_features,
        BPR=bpr,
        BLOCK_HALF=128,
        BLOCK_M=BLOCK_M,
        num_warps=4,
    )
    return y


def _pick_block_m(out_features: int) -> int:
    """Pick BLOCK_M based on out_features. Tuned on RTX 4070M for the shapes
    used by gemma-4-E4B-it. The bigger the matmul the more rows we want to
    bundle (amortizes x_rot loads); the smaller, the fewer (need enough
    programs to fill the SMs)."""
    # Heuristic from the per-shape sweep: target ~32-64 programs total when
    # out_features is large, BLOCK_M=1 when out_features is small.
    if out_features >= 4096:
        return 64
    if out_features >= 2048:
        return 32
    if out_features >= 1024:
        return 16
    if out_features >= 512:
        return 4
    return 1


@triton.jit
def _tq4_matmul_kernel(
    x_ptr,          # (n_seq * in_features,) FP32 — flat 2D as 1D
    qs_ptr, d_ptr, centroids_ptr,
    y_ptr,          # (n_seq * out_features,) FP32 — flat 2D as 1D
    in_features, out_features, n_seq,
    BPR: tl.constexpr,
    BLOCK_HALF: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """One program = BLOCK_M output rows for one sequence position.
    Grid is (out_features/BLOCK_M, n_seq) — eliminates the Python loop
    over sequence positions during prefill (S>1)."""
    pid_m = tl.program_id(0)
    pid_s = tl.program_id(1)
    row_base = pid_m * BLOCK_M
    if row_base >= out_features:
        return
    if pid_s >= n_seq:
        return

    half_idx = tl.arange(0, BLOCK_HALF)
    m_idx = tl.arange(0, BLOCK_M)
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    x_seq_off = pid_s * in_features
    for b in range(BPR):
        x_low = tl.load(x_ptr + x_seq_off + b * 256 + 2 * half_idx)
        x_high = tl.load(x_ptr + x_seq_off + b * 256 + 2 * half_idx + 1)

        block_idx_m = (row_base + m_idx) * BPR + b
        d_m = tl.load(d_ptr + block_idx_m)
        qs_off = block_idx_m[:, None] * BLOCK_HALF + half_idx[None, :]
        qs_m = tl.load(qs_ptr + qs_off).to(tl.int32)
        c_low = tl.load(centroids_ptr + (qs_m & 0xF))
        c_high = tl.load(centroids_ptr + ((qs_m >> 4) & 0xF))
        acc += d_m * (tl.sum(c_low * x_low[None, :], axis=1)
                      + tl.sum(c_high * x_high[None, :], axis=1))

    rows = row_base + m_idx
    y_off = pid_s * out_features + rows
    tl.store(y_ptr + y_off, acc, mask=rows < out_features)


def tq4_linear_triton(
    x: torch.Tensor,
    qs: torch.Tensor,
    d: torch.Tensor,
    pi: torch.Tensor,           # (256, 256) FP32
    centroids: torch.Tensor,
    out_features: int,
    in_features: int,
) -> torch.Tensor:
    """Full tq4 linear: rotate x by Pi.T, then fused matvec/matmul.
    Single batch position uses the matvec kernel; multi-position prefill
    uses the batched matmul kernel (one launch instead of S launches)."""
    *batch, in_f = x.shape
    assert in_f == in_features and in_f % 256 == 0
    bpr = in_f // 256
    x_rot = (x.reshape(*batch, bpr, 256) @ pi.T).reshape(*batch, in_f)
    if not x_rot.is_contiguous():
        x_rot = x_rot.contiguous()

    n_seq = max(1, x_rot.numel() // in_f)
    if n_seq == 1:
        y_flat = tq4_matvec_triton(
            x_rot.reshape(in_f), qs, d, centroids, out_features, in_features)
        return y_flat.reshape(*batch, out_features)

    # Batched: launch (out_tiles, n_seq) grid in one shot.
    flat = x_rot.reshape(n_seq, in_f)
    y = torch.empty(n_seq, out_features, device=x.device, dtype=torch.float32)
    BLOCK_M = _pick_block_m(out_features)
    grid = ((out_features + BLOCK_M - 1) // BLOCK_M, n_seq)
    _tq4_matmul_kernel[grid](
        flat.view(-1), qs.view(-1), d, centroids, y.view(-1),
        in_features, out_features, n_seq,
        BPR=bpr, BLOCK_HALF=128, BLOCK_M=BLOCK_M,
        num_warps=4,
    )
    return y.reshape(*batch, out_features)


# --- Dual tq4 matvec: shared input, two parallel weight matrices ---
#
# Used for ffn_gate + ffn_up (same x → two outputs) and optionally for
# attn_k + attn_v in own-KV layers (same shape, same x). Saves one full
# Python/launch round-trip and one Pi-rotation per call.

@triton.jit
def _tq4_matvec_dual_kernel(
    x_rot_ptr,
    qs_a_ptr, d_a_ptr,
    qs_b_ptr, d_b_ptr,
    centroids_ptr,
    y_a_ptr, y_b_ptr,
    in_features,
    out_features,
    BPR: tl.constexpr,
    BLOCK_HALF: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    row_base = pid * BLOCK_M
    if row_base >= out_features:
        return

    half_idx = tl.arange(0, BLOCK_HALF)
    m_idx = tl.arange(0, BLOCK_M)
    acc_a = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc_b = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for b in range(BPR):
        x_base = b * 256
        x_low = tl.load(x_rot_ptr + x_base + 2 * half_idx)
        x_high = tl.load(x_rot_ptr + x_base + 2 * half_idx + 1)

        block_idx_m = (row_base + m_idx) * BPR + b

        # Matrix A
        d_a_m = tl.load(d_a_ptr + block_idx_m)
        qs_a_off = block_idx_m[:, None] * BLOCK_HALF + half_idx[None, :]
        qs_a_m = tl.load(qs_a_ptr + qs_a_off).to(tl.int32)
        c_a_low = tl.load(centroids_ptr + (qs_a_m & 0xF))
        c_a_high = tl.load(centroids_ptr + ((qs_a_m >> 4) & 0xF))
        acc_a += d_a_m * (tl.sum(c_a_low * x_low[None, :], axis=1)
                           + tl.sum(c_a_high * x_high[None, :], axis=1))

        # Matrix B
        d_b_m = tl.load(d_b_ptr + block_idx_m)
        qs_b_off = block_idx_m[:, None] * BLOCK_HALF + half_idx[None, :]
        qs_b_m = tl.load(qs_b_ptr + qs_b_off).to(tl.int32)
        c_b_low = tl.load(centroids_ptr + (qs_b_m & 0xF))
        c_b_high = tl.load(centroids_ptr + ((qs_b_m >> 4) & 0xF))
        acc_b += d_b_m * (tl.sum(c_b_low * x_low[None, :], axis=1)
                           + tl.sum(c_b_high * x_high[None, :], axis=1))

    rows = row_base + m_idx
    mask = rows < out_features
    tl.store(y_a_ptr + rows, acc_a, mask=mask)
    tl.store(y_b_ptr + rows, acc_b, mask=mask)


def tq4_linear_dual_triton(
    x: torch.Tensor,
    qs_a: torch.Tensor, d_a: torch.Tensor,
    qs_b: torch.Tensor, d_b: torch.Tensor,
    pi: torch.Tensor,
    centroids: torch.Tensor,
    out_features: int,
    in_features: int,
):
    """Two tq4 linears sharing the same input. Returns (y_a, y_b)."""
    *batch, in_f = x.shape
    assert in_f == in_features and in_f % 256 == 0
    bpr = in_f // 256
    x_rot = (x.reshape(*batch, bpr, 256) @ pi.T).reshape(*batch, in_f)
    if not x_rot.is_contiguous():
        x_rot = x_rot.contiguous()
    flat = x_rot.reshape(-1, in_f)
    n = flat.shape[0]
    y_a = torch.empty(n, out_features, device=x.device, dtype=torch.float32)
    y_b = torch.empty(n, out_features, device=x.device, dtype=torch.float32)
    # Dual kernel has 2x register pressure vs single — cap BLOCK_M lower.
    BLOCK_M = min(32, _pick_block_m(out_features))
    grid = ((out_features + BLOCK_M - 1) // BLOCK_M,)
    for i in range(n):
        _tq4_matvec_dual_kernel[grid](
            flat[i].contiguous(),
            qs_a.view(-1), d_a, qs_b.view(-1), d_b,
            centroids,
            y_a[i], y_b[i],
            in_features, out_features,
            BPR=bpr, BLOCK_HALF=128, BLOCK_M=BLOCK_M,
            num_warps=4,
        )
    return (y_a.reshape(*batch, out_features),
            y_b.reshape(*batch, out_features))


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


@triton.jit
def _q6k_lookup_kernel(
    token_ids_ptr,
    ql_ptr, qh_ptr, scales_ptr, d_ptr,
    out_ptr,
    n_tokens, d_model,
    BPR: tl.constexpr,
    BLOCK_ELEMENTS: tl.constexpr,
):
    """One program = one (token, block) pair. Writes 256 dequantized
    elements to out[token, block*256:(block+1)*256]. Used for the
    main token embedding and the per-layer token embedding lookups."""
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)
    if pid_n >= n_tokens or pid_b >= BPR:
        return

    token_id = tl.load(token_ids_ptr + pid_n)
    block_idx = token_id * BPR + pid_b

    pos = tl.arange(0, BLOCK_ELEMENTS)
    half = pos // 128
    within = pos % 128
    quarter = within // 32
    l = within % 32
    ql_idx = half * 64 + l + tl.where(
        (quarter == 1) | (quarter == 3), 32, 0)
    ql_shift = tl.where(quarter >= 2, 4, 0)
    qh_idx = half * 32 + l
    qh_shift = 2 * quarter
    scale_idx = half * 8 + (l // 16) + 2 * quarter

    ql_vals = tl.load(ql_ptr + block_idx * 128 + ql_idx).to(tl.int32)
    qh_vals = tl.load(qh_ptr + block_idx * 64 + qh_idx).to(tl.int32)
    scale_vals = tl.load(scales_ptr + block_idx * 16 + scale_idx).to(tl.int32)
    d_block = tl.load(d_ptr + block_idx)

    ql_low = (ql_vals >> ql_shift) & 0xF
    qh_high = (qh_vals >> qh_shift) & 0x3
    q = ql_low | (qh_high << 4)
    q_signed = q - 32

    val = d_block * scale_vals.to(tl.float32) * q_signed.to(tl.float32)

    out_offsets = pid_n * d_model + pid_b * BLOCK_ELEMENTS + pos
    tl.store(out_ptr + out_offsets, val)


def q6k_lookup_triton(
    token_ids: torch.Tensor,
    ql: torch.Tensor, qh: torch.Tensor,
    scales: torch.Tensor, d: torch.Tensor,
    vocab_size: int, d_model: int,
) -> torch.Tensor:
    """Q6_K embedding lookup. Returns (N, d_model) FP32."""
    flat = token_ids.flatten().to(torch.int64).contiguous()
    n_tokens = flat.numel()
    bpr = d_model // 256
    out = torch.empty(n_tokens, d_model,
                       device=token_ids.device, dtype=torch.float32)
    grid = (n_tokens, bpr)
    _q6k_lookup_kernel[grid](
        flat, ql.view(-1), qh.view(-1), scales.view(-1), d,
        out.view(-1),
        n_tokens, d_model,
        BPR=bpr, BLOCK_ELEMENTS=256,
        num_warps=2,
    )
    return out


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
