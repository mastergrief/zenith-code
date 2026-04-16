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
    _tq4_matvec_kernel[grid](
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
