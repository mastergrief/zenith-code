"""Fused flash-attention decode kernel with tq4 K/V.

MVP shape contract (decode-only, S_q=1):

  q_rot: (n_heads_q, d_head) fp32, pre-rotated by Pi (caller responsibility).
  k_qs:  (n_heads_kv, N * bpr, 128) uint8  — head-major, contiguous per head
  k_d:   (n_heads_kv, N * bpr) fp32
  v_qs:  (n_heads_kv, N * bpr, 128) uint8
  v_d:   (n_heads_kv, N * bpr) fp32

where bpr = d_head // 256 (= 1 for Gemma E4B). For d_head=256 each
position is exactly one tq4 block per head.

Math (Pi orthogonal so rotations cancel in inner products):

  scores[h, n] = (Pi @ Q[h]) · (Pi @ K[kv_h, n])           = Q · K
              = q_rot[h] @ k_dequant_rotated[kv_h, n]
  out[h]       = sum_n softmax(scores[h])[n] * V[kv_h, n]
              = Pi.T @ (sum_n p[n] * v_dequant_rotated[kv_h, n])

So we score directly against the rotated K codes (skip per-block Pi.T
inside the kernel — saves 1 matmul per tq4 block) and post-rotate the
weighted V sum once per head outside the kernel.

Two kernels:
  - K side: existing `tq4_matvec_triton` already does
    `scores = K_rotated @ q_rot` per head — reuse it.
  - V side: new `_tq4_weighted_v_kernel` — properly parallel
    (grid = (n_heads_q,), each program owns one Q head, streams N tq4
    blocks of V, accumulates fp32 d_head output in registers).

Replaces the 257-line scaffold from commit `571c3ad` whose
`_tq4_flash_decode_kernel` was a `pass`-body placeholder and whose
`_tq4_weighted_sum_kernel` only did work in pid==0.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# ============================================================================
# K-side: fused multi-head score kernel. Single launch computes scores for
# all Q heads × all N positions, replacing the prior per-head Python loop
# that invoked `tq4_matvec_triton` n_heads_q times per layer (8 × 42 = 336
# launches per decode step on Gemma E4B; the launch overhead dominated the
# streaming-byte-load savings at N≤1024).
# ============================================================================

@triton.jit
def _tq4_k_scores_multihead_kernel(
    q_rot_ptr,          # (n_heads_q, d_head) fp32, pre-rotated
    k_qs_ptr,           # (n_heads_kv, N * BPR, 128) uint8 — head-major
    k_d_ptr,            # (n_heads_kv, N * BPR) fp32
    centroids_ptr,      # (16,) fp32
    scores_ptr,         # (n_heads_q, N) fp32 — output
    N,
    GQA_REPEAT: tl.constexpr,
    BPR: tl.constexpr,
    BLOCK_HALF: tl.constexpr,   # 128
    BLOCK_M: tl.constexpr,      # N-rows per program
):
    """One program = (q_head, M-tile of N positions). Loads this head's
    Q once, streams BLOCK_M tq4 K blocks, computes BLOCK_M scores."""
    pid_h = tl.program_id(0)
    pid_m = tl.program_id(1)
    row_base = pid_m * BLOCK_M
    if row_base >= N:
        return

    kv_h = pid_h // GQA_REPEAT
    half_idx = tl.arange(0, BLOCK_HALF)
    m_idx = tl.arange(0, BLOCK_M)
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    centroid_tile = tl.load(centroids_ptr + tl.arange(0, 16))

    q_head_base = pid_h * (BPR * 256)
    kv_qs_head_base = kv_h * (N * BPR * BLOCK_HALF)
    kv_d_head_base = kv_h * (N * BPR)

    for b in range(BPR):
        x_base = b * 256
        x_low = tl.load(q_rot_ptr + q_head_base + x_base + 2 * half_idx)
        x_high = tl.load(q_rot_ptr + q_head_base + x_base + 2 * half_idx + 1)

        block_idx_m = (row_base + m_idx) * BPR + b   # (BLOCK_M,)
        d_m = tl.load(k_d_ptr + kv_d_head_base + block_idx_m)

        qs_offsets = (kv_qs_head_base
                      + block_idx_m[:, None] * BLOCK_HALF
                      + half_idx[None, :])             # (BLOCK_M, 128)
        qs_m = tl.load(k_qs_ptr + qs_offsets).to(tl.int32)
        low_m = qs_m & 0xF
        high_m = (qs_m >> 4) & 0xF

        low_flat = tl.reshape(low_m, (BLOCK_M * BLOCK_HALF,))
        high_flat = tl.reshape(high_m, (BLOCK_M * BLOCK_HALF,))
        c_low = tl.reshape(tl.gather(centroid_tile, low_flat, axis=0),
                           (BLOCK_M, BLOCK_HALF))
        c_high = tl.reshape(tl.gather(centroid_tile, high_flat, axis=0),
                            (BLOCK_M, BLOCK_HALF))

        block_dot = (tl.sum(c_low * x_low[None, :], axis=1)
                   + tl.sum(c_high * x_high[None, :], axis=1))
        acc += d_m * block_dot

    scores_base = pid_h * N
    rows = row_base + m_idx
    tl.store(scores_ptr + scores_base + rows, acc, mask=rows < N)


def tq4_k_scores_multihead(
    q_rot: torch.Tensor,          # (n_heads_q, d_head) fp32, pre-rotated
    k_qs: torch.Tensor,           # (n_heads_kv, N*BPR, 128) uint8
    k_d: torch.Tensor,            # (n_heads_kv, N*BPR) fp32
    centroids: torch.Tensor,      # (16,) fp32
) -> torch.Tensor:
    """Returns (n_heads_q, N) fp32 attention scores. Replaces the prior
    per-head Python loop with a single 2D-grid Triton launch."""
    n_heads_q, d_head = q_rot.shape
    n_heads_kv, n_blocks_per_head, _ = k_qs.shape
    bpr = d_head // 256
    assert bpr * 256 == d_head
    assert n_blocks_per_head % bpr == 0
    N = n_blocks_per_head // bpr

    # BLOCK_M picks itself by out size; match the heuristic used by
    # `_pick_block_m` in `tq4_triton.py` (tuned for RTX 4070M / Ada).
    if N >= 4096:
        BLOCK_M = 64
    elif N >= 2048:
        BLOCK_M = 32
    elif N >= 1024:
        BLOCK_M = 16
    elif N >= 512:
        BLOCK_M = 4
    else:
        BLOCK_M = 1

    scores = torch.empty(n_heads_q, N, device=q_rot.device, dtype=torch.float32)
    grid = (n_heads_q, (N + BLOCK_M - 1) // BLOCK_M)
    _tq4_k_scores_multihead_kernel[grid](
        q_rot.contiguous(), k_qs.view(-1), k_d.view(-1),
        centroids, scores.view(-1),
        N,
        GQA_REPEAT=n_heads_q // n_heads_kv,
        BPR=bpr, BLOCK_HALF=128, BLOCK_M=BLOCK_M,
        num_warps=4,
    )
    return scores


# ============================================================================
# V-side: parallel-over-d_head weighted-sum kernel. Grid (n_heads_q, d_tiles)
# gives more SM occupancy than the old grid (n_heads_q,) which had only 8
# programs on Gemma E4B (88% of SMs idle on a 36-SM Ada part).
# ============================================================================

@triton.jit
def _tq4_weighted_v_kernel(
    weights_ptr,        # (n_heads_q, N) fp32 — softmax weights per Q head
    v_qs_ptr,           # (n_heads_kv, N*BPR, 128) uint8 — head-major
    v_d_ptr,            # (n_heads_kv, N*BPR) fp32
    centroids_ptr,      # (16,) fp32
    out_ptr,            # (n_heads_q, D_HEAD) fp32 — in rotated domain
    N,                  # current sequence length
    GQA_REPEAT: tl.constexpr,
    BLOCK_HALF: tl.constexpr,   # 128 — half-block (nybble unpacked)
    D_HEAD: tl.constexpr,       # 256 per block; BPR=D_HEAD/256
    BPR: tl.constexpr,
    D_TILE: tl.constexpr,       # d_head split per program: BLOCK_HALF / N_D_TILES
    N_D_TILES: tl.constexpr,    # BLOCK_HALF // D_TILE
):
    """Grid: (n_heads_q, N_D_TILES × BPR). Each program handles one Q head
    and one (D_TILE-wide) slice of the d_head output. Serial N loop within
    the program; parallelism comes from the grid over (head, d_tile)."""
    pid_h = tl.program_id(0)
    pid_d = tl.program_id(1)
    # pid_d encodes (bpr_idx, tile_idx_within_half).
    bpr_idx = pid_d // N_D_TILES
    tile_idx = pid_d % N_D_TILES

    kv_h = pid_h // GQA_REPEAT
    tile_off = tile_idx * D_TILE
    tile_idx_range = tl.arange(0, D_TILE)
    half_abs = tile_off + tile_idx_range          # indices within BLOCK_HALF

    centroid_tile = tl.load(centroids_ptr + tl.arange(0, 16))

    acc_low = tl.zeros((D_TILE,), dtype=tl.float32)
    acc_high = tl.zeros((D_TILE,), dtype=tl.float32)

    weights_base = pid_h * N
    kv_qs_base = kv_h * (N * BPR * BLOCK_HALF)
    kv_d_base = kv_h * (N * BPR)

    for n in range(N):
        w = tl.load(weights_ptr + weights_base + n)

        blk_idx = n * BPR + bpr_idx
        d_blk = tl.load(v_d_ptr + kv_d_base + blk_idx)

        qs_offsets = kv_qs_base + blk_idx * BLOCK_HALF + half_abs
        qs_blk = tl.load(v_qs_ptr + qs_offsets).to(tl.int32)
        low = qs_blk & 0xF
        high = (qs_blk >> 4) & 0xF
        c_low = tl.gather(centroid_tile, low, axis=0)
        c_high = tl.gather(centroid_tile, high, axis=0)

        scale = w * d_blk
        acc_low += scale * c_low
        acc_high += scale * c_high

    # Write back: low/high interleaved. For this D_TILE slice, positions
    # 2*(tile_off + i) and 2*(tile_off + i) + 1 within the BPR-th block
    # of D_HEAD's output.
    out_block_off = pid_h * D_HEAD + bpr_idx * 256
    low_offsets = out_block_off + 2 * half_abs
    high_offsets = out_block_off + 2 * half_abs + 1
    tl.store(out_ptr + low_offsets, acc_low)
    tl.store(out_ptr + high_offsets, acc_high)


def tq4_weighted_v(
    weights: torch.Tensor,        # (n_heads_q, N) fp32
    v_qs: torch.Tensor,           # (n_heads_kv, N*BPR, 128) uint8
    v_d: torch.Tensor,            # (n_heads_kv, N*BPR) fp32
    centroids: torch.Tensor,      # (16,) fp32
    n_heads_q: int,
    n_heads_kv: int,
    d_head: int = 256,
) -> torch.Tensor:
    """Returns (n_heads_q, d_head) fp32 in the Pi-rotated domain. Apply
    Pi.T outside per head to recover the unrotated output.

    Grid parallelization: (n_heads_q, N_D_TILES × BPR). For Gemma E4B
    d_head=256 (BPR=1) with D_TILE=32, that's 8 × 4 = 32 programs —
    ~88% SM occupancy on the 36-SM 4070M (up from 22% with the old
    grid=(n_heads_q,) kernel)."""
    assert weights.is_contiguous() and weights.dtype == torch.float32
    assert v_qs.is_contiguous() and v_qs.dtype == torch.uint8
    assert v_d.is_contiguous() and v_d.dtype == torch.float32
    assert centroids.is_contiguous() and centroids.dtype == torch.float32
    assert n_heads_q % n_heads_kv == 0
    bpr = d_head // 256
    assert bpr * 256 == d_head, f"d_head={d_head} not a multiple of 256"

    N = weights.shape[1]
    # D_TILE=32 gives N_D_TILES=4 (128/32). Multiply by BPR for total
    # d-tile programs per head. For d_head=256 (BPR=1) that's 4/head; for
    # d_head=512 (BPR=2) that's 8/head.
    D_TILE = 32
    N_D_TILES = 128 // D_TILE   # 4

    out = torch.empty(n_heads_q, d_head,
                      device=weights.device, dtype=torch.float32)
    grid = (n_heads_q, N_D_TILES * bpr)
    _tq4_weighted_v_kernel[grid](
        weights, v_qs.view(-1), v_d.view(-1),
        centroids, out.view(-1),
        N,
        GQA_REPEAT=n_heads_q // n_heads_kv,
        BLOCK_HALF=128, D_HEAD=d_head, BPR=bpr,
        D_TILE=D_TILE, N_D_TILES=N_D_TILES,
        num_warps=2,
    )
    return out


# ============================================================================
# Top-level wrapper: K-side via tq4_matvec_triton, V-side via the kernel above.
# ============================================================================

def fused_tq4_qjl_flash_attn_decode(
    q: torch.Tensor,              # (n_heads_q, d_head) fp32 — UNROTATED query
    k_qjl: "list",                # per-head Tq4QjlTensor for K (n_heads_kv items)
    v_qs: torch.Tensor,           # (n_heads_kv, N, 128) uint8 — head-major
    v_d: torch.Tensor,            # (n_heads_kv, N) fp32
    centroids_3bit: torch.Tensor, # (8,) fp32 — Lloyd-Max for 3-bit Q_mse
    centroids_tq4: torch.Tensor,  # (16,) fp32 — Lloyd-Max for 4-bit V tq4
    pi: torch.Tensor,             # (d_head, d_head) fp32
    jl: torch.Tensor,             # (d_head, d_head) fp32 — Gaussian JL
    attn_mask: torch.Tensor,      # (N,) fp32 — additive (0 or -inf)
    softcap: float = 0.0,
) -> torch.Tensor:
    """Inner-product-optimal variant — Phase 3.

    K is tq4_qjl (Algorithm 2: 3-bit Q_mse + 1-bit QJL on residual). The
    score `<K[t], Q>` uses `qjl_inner_product` which is unbiased over
    JL realizations — eliminates the bias the Q_mse-only path has.

    V stays regular tq4 (4-bit Q_mse). V is consumed by the linear
    weighted sum `sum_t softmax[t] * V[t]`, not an inner product, so
    Q_mse-optimal is the right objective there. Reuses the existing
    `tq4_weighted_v` kernel.

    Input Q is the standard UNROTATED Q from `attn_q`; the rotation is
    folded into the qjl estimator (which rotates y internally to y_rot).
    """
    from calm.llm_computer.tq4_qjl_torch import qjl_inner_product

    n_heads_q, d_head = q.shape
    n_heads_kv = len(k_qjl)
    assert n_heads_q % n_heads_kv == 0
    gqa_repeat = n_heads_q // n_heads_kv
    N = k_qjl[0].n_blocks

    # K-side scoring per Q head via the unbiased estimator. One call per
    # Q head; qjl_inner_product handles the rotation + estimator math.
    scores = torch.empty(n_heads_q, N, device=q.device, dtype=torch.float32)
    for h in range(n_heads_q):
        kv_h = h // gqa_repeat
        # qjl_inner_product accepts y of shape (HEAD_DIM,) → returns (n_blocks,)
        scores[h] = qjl_inner_product(
            k_qjl[kv_h], q[h].contiguous(),
            pi=pi, centroids_3bit=centroids_3bit, jl=jl,
        )

    if softcap > 0:
        scores = softcap * torch.tanh(scores / softcap)
    scores = scores + attn_mask[None, :]
    weights = torch.softmax(scores, dim=-1)

    # V-side weighted sum unchanged from the Q_mse path. V dequant outputs
    # values in the rotated domain (consistent with the Q_mse encoding); the
    # final Pi.T outside un-rotates per head.
    out_rotated = tq4_weighted_v(
        weights.contiguous(), v_qs.contiguous(), v_d.contiguous(),
        centroids_tq4, n_heads_q, n_heads_kv, d_head,
    )
    out = out_rotated @ pi
    return out


def fused_tq4_flash_attn_decode(
    q_rot: torch.Tensor,          # (n_heads_q, d_head) fp32, pre-rotated
    k_qs: torch.Tensor,           # (n_heads_kv, N, 128) uint8 — head-major
    k_d: torch.Tensor,            # (n_heads_kv, N) fp32
    v_qs: torch.Tensor,           # (n_heads_kv, N, 128) uint8 — head-major
    v_d: torch.Tensor,            # (n_heads_kv, N) fp32
    centroids: torch.Tensor,      # (16,) fp32
    pi: torch.Tensor,             # (d_head, d_head) fp32 — for output unrotate
    attn_mask: torch.Tensor,      # (N,) fp32 — additive (0 or -inf)
    softcap: float = 0.0,
) -> torch.Tensor:
    """Decode (S_q=1) attention with tq4 K/V. Returns (n_heads_q, d_head)
    fp32 in the standard unrotated domain (ready to feed into attn_output).
    """
    n_heads_q, d_head = q_rot.shape
    n_heads_kv = k_qs.shape[0]
    N = k_qs.shape[1] // (d_head // 256)
    assert n_heads_q % n_heads_kv == 0

    # Single multi-head K-scoring launch: grid (n_heads_q, N_tiles) replaces
    # the previous per-head Python loop that issued n_heads_q separate
    # tq4_matvec_triton launches (8 × 42 layers = 336 launches per decode
    # step on Gemma E4B — launch overhead dominated at N≤1024).
    scores = tq4_k_scores_multihead(q_rot, k_qs, k_d, centroids)

    if softcap > 0:
        scores = softcap * torch.tanh(scores / softcap)
    scores = scores + attn_mask[None, :]
    weights = torch.softmax(scores, dim=-1)

    # V-side weighted sum, one program per Q head.
    out_rotated = tq4_weighted_v(
        weights.contiguous(), v_qs.contiguous(), v_d.contiguous(),
        centroids, n_heads_q, n_heads_kv, d_head,
    )

    # Unrotate per head. Pi is (256, 256) and rotates per-256-element block.
    # For d_head=256 (bpr=1) this is a direct (H, 256) @ (256, 256) matmul.
    # For d_head=512 (bpr=2, Gemma global layers) reshape to (H, 2, 256),
    # apply Pi per-block, flatten back. Pi orthogonal ⇒ Pi.T = Pi^-1
    # applied as `out_rotated @ Pi` matches dequantize_tq4's convention.
    bpr = d_head // 256
    if bpr == 1:
        out = out_rotated @ pi
    else:
        out = (out_rotated.reshape(n_heads_q, bpr, 256) @ pi).reshape(
            n_heads_q, d_head)
    return out
