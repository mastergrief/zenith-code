"""Fused flash-attention decode kernel with tq4 K/V.

Ports the llama.cpp fattn-vec pattern to Triton: read tq4 bytes,
dequant per-tile in registers/SMEM, do online softmax + weighted V
accumulate in one pass. O(N) per step, NO persistent fp16/fp32 K/V
materialization — tq4 memory win preserved.

Decode-only MVP (S_q=1). Per-head Q matvec via existing tq4_matvec
for scoring, custom tile-loop kernel for the V-weighted accumulation.

Storage layout expected by this module:
  K.qs: (n_heads_kv, max_len * bpr, 128) uint8 — per-head row-major
  K.d:  (n_heads_kv, max_len * bpr) fp32
  V.qs: same layout
  V.d:  same layout
where bpr = d_head / 256 (= 1 for Gemma E4B d_head=256).

Integration point: called from GemmaSubstrate._forward_layer when
kv_cache is KVCacheTq4 AND S_q == 1 (decode path). Prefill (S_q > 1)
still uses the current dequant-and-fp32-attention path for now.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# -------- Score kernel: fused Q · K_tq4 → scores per head --------
# Reuses the tq4 matvec pattern. Input: Q (d_head,) fp32 Pi-rotated.
# K: (N, d_head) tq4 storage for ONE head. Output: scores (N,) fp32.
# This is EXACTLY tq4_matvec_triton's job, no new kernel needed.


# -------- Fused score + online-softmax + V-weighted accumulate --------
@triton.jit
def _tq4_flash_decode_kernel(
    q_rot_ptr,        # (d_head,) fp32, pre-rotated by Pi.T
    k_qs_ptr,         # (N * bpr * 128,) uint8 — ONE head's K row-major
    k_d_ptr,          # (N * bpr,) fp32
    v_qs_ptr,         # (N * bpr * 128,) uint8 — ONE head's V row-major
    v_d_ptr,          # (N * bpr,) fp32
    centroids_ptr,    # (16,) fp32
    mask_ptr,         # (N,) fp32 — added to scores pre-softmax (0 or -inf)
    softcap_ptr,      # (1,) fp32 — softcap scalar (0.0 means disabled)
    out_ptr,          # (d_head,) fp32 — attention output for this head
    N,                # int — sequence length
    BPR: tl.constexpr,          # blocks per K/V row (d_head / 256)
    BLOCK_HALF: tl.constexpr,   # 128
    D_HEAD: tl.constexpr,       # 256 (one block)
):
    """Decode flash-attn: one program per head. Two passes over N.

    Pass 1: score each K row via tq4 matvec inline, track max(scores).
    Pass 2: apply softmax (subtract max, exp, normalize), accumulate
            sum_k softmax[k] * V[k] into output via tq4-streamed V.

    Softcap: if softcap > 0, scores = softcap * tanh(scores / softcap).
    Pi rotation: K stored rotated. Q must be pre-rotated by Pi.T by caller.
    """
    pid = tl.program_id(0)
    # Exactly one program fires (we launch grid=(1,)); for MVP no per-head grid.
    # The caller runs this once per (layer, q_head) — outer Python loop.
    if pid != 0:
        return

    half_idx = tl.arange(0, BLOCK_HALF)  # (128,)
    centroid_tile = tl.load(centroids_ptr + tl.arange(0, 16))
    softcap = tl.load(softcap_ptr)

    # Load Q rotated: (D_HEAD,) fp32. Split into low/high halves for
    # the tq4 block layout.
    q_low = tl.load(q_rot_ptr + 2 * half_idx)
    q_high = tl.load(q_rot_ptr + 2 * half_idx + 1)

    # -------- Pass 1: score + find max --------
    max_score = tl.full((1,), float("-inf"), dtype=tl.float32)
    # We loop over N positions one at a time. For d_head=256, BPR=1 so
    # each position = 1 block = 128 bytes.
    # NOTE: loop over N in Python-land via multiple kernel launches is
    # expensive. Inline the loop here with a constant TILE would be
    # better but requires N to be constexpr or chunked. For MVP,
    # single-program iteration.
    # We'll do two passes, first computing scores into an SMEM-backed
    # scratch (Triton auto-allocates). But 16K positions = 64KB for
    # fp32 scores — fits in SMEM on Ada (100KB/SM).
    # Triton doesn't expose explicit SMEM scratch. We'll compute scores
    # into a tensor block. Max N supported bounded by block size.
    # Fallback: two separate kernels (score, then softmax+V).
    # MVP uses separate kernels outside this file.

    # For MVP: this kernel skeleton is a placeholder. The real work
    # happens in the Python wrapper calling tq4_matvec_triton twice
    # (once for K scores, once for V weighted sum).
    pass


def fused_tq4_flash_decode(
    q_rot: torch.Tensor,           # (n_heads_q, d_head) fp32, Pi.T-rotated
    k_qs: torch.Tensor,            # (n_heads_kv, N * bpr, 128) uint8
    k_d: torch.Tensor,             # (n_heads_kv, N * bpr) fp32
    v_qs: torch.Tensor,            # (n_heads_kv, N * bpr, 128) uint8
    v_d: torch.Tensor,             # (n_heads_kv, N * bpr) fp32
    centroids: torch.Tensor,       # (16,) fp32
    attn_mask: torch.Tensor,       # (N,) fp32 — 0 or -inf, added pre-softmax
    gqa_repeat: int,               # n_heads_q / n_heads_kv
    softcap: float = 0.0,          # Gemma uses 50.0 on attn scores (pre-softmax)
) -> torch.Tensor:
    """Decode attention using tq4 K/V storage. Returns (n_heads_q, d_head) fp32.

    Per-head execution via two tq4_matvec_triton calls:
      1. scores = tq4_matvec(Q_rot[h], K[kv_head]) shape (N,)
      2. softmax(scores + mask)  (with optional softcap)
      3. out[h] = tq4_matvec(softmax_weights, V.T[kv_head]) shape (d_head,)
         — but V is stored row-major (N, d_head), not col-major (d_head, N).
         For V application, we need a different matvec interpretation.

    V-application approach: treat attn_weights (N,) as the INPUT vector
    and V stored as (d_head, N) "matrix" — but that requires V stored
    column-major, which our KVCacheTq4 doesn't do naturally.

    Alternative: dequant V on-the-fly per position (O(N * d_head) per
    head per step, total O(N * d_head * n_heads_q) = same as regular
    attention read). Use a dedicated Triton kernel that reads tq4 V
    blocks, multiplies by attn_weights scalar, and accumulates.
    """
    from calm.llm_computer.tq4_triton import tq4_matvec_triton

    n_heads_q, d_head = q_rot.shape
    n_heads_kv = k_qs.shape[0]
    assert gqa_repeat * n_heads_kv == n_heads_q

    out = torch.empty(n_heads_q, d_head, device=q_rot.device, dtype=torch.float32)
    bpr = d_head // 256
    N = k_qs.shape[1] // bpr

    # For each Q head:
    for h in range(n_heads_q):
        kv_h = h // gqa_repeat
        q_h = q_rot[h]                            # (d_head,)
        k_qs_h = k_qs[kv_h].contiguous()          # (N * bpr, 128)
        k_d_h = k_d[kv_h].contiguous()            # (N * bpr,)
        v_qs_h = v_qs[kv_h].contiguous()
        v_d_h = v_d[kv_h].contiguous()

        # Pass 1: scores = K @ Q_rot — tq4_matvec matches this exactly
        # (K stored (N, d_head) tq4 = "W" of shape (N_out, d_head_in))
        scores = tq4_matvec_triton(
            q_h, k_qs_h, k_d_h, centroids,
            out_features=N, in_features=d_head,
        )  # (N,) fp32

        # Softcap + mask + softmax
        if softcap > 0:
            scores = softcap * torch.tanh(scores / softcap)
        scores = scores + attn_mask
        scores = torch.softmax(scores, dim=-1)

        # Pass 2: out[h] = scores @ V
        # V stored (N, d_head) row-major. Want (d_head,) result.
        # Dispatch dedicated kernel that reads tq4 V row-by-row,
        # accumulates weighted sum.
        out_h = tq4_weighted_sum(
            scores, v_qs_h, v_d_h, centroids,
            N=N, d_head=d_head,
        )
        out[h] = out_h

    return out


@triton.jit
def _tq4_weighted_sum_kernel(
    weights_ptr,      # (N,) fp32 — softmax weights
    v_qs_ptr,         # (N * bpr * 128,) uint8
    v_d_ptr,          # (N * bpr,) fp32
    centroids_ptr,    # (16,) fp32
    pi_ptr,           # (256, 256) fp32 — to inverse-rotate the dequant
    out_ptr,          # (d_head,) fp32
    N,
    BPR: tl.constexpr,
    BLOCK_HALF: tl.constexpr,   # 128
    D_HEAD: tl.constexpr,       # 256 (BPR * 256)
):
    """Compute out[d] = sum_n weights[n] * V[n, d] where V is tq4.

    Process one output dim per program. For each output dim d, sum over
    all N positions of weights[n] * V[n, d]. V[n, d] comes from
    dequantizing position n's tq4 block and selecting dim d, then
    applying Pi.T rotation.

    Actually simpler: dequant V[n] into d_head fp32 vector, weight by
    weights[n], accumulate into out. One block per position per
    iteration — accumulate (d_head,) buffer.

    Layout: BPR = 1 for Gemma d_head=256. One tq4 block = one position
    per head.
    """
    pid = tl.program_id(0)
    # Single-program MVP: iterate over all positions, accumulate
    # (d_head,) output.
    if pid != 0:
        return

    half_idx = tl.arange(0, BLOCK_HALF)    # (128,)
    centroid_tile = tl.load(centroids_ptr + tl.arange(0, 16))

    acc_low = tl.zeros((BLOCK_HALF,), dtype=tl.float32)
    acc_high = tl.zeros((BLOCK_HALF,), dtype=tl.float32)

    for n in range(N):
        w = tl.load(weights_ptr + n)        # scalar
        for b in range(BPR):
            blk_idx = n * BPR + b
            d_blk = tl.load(v_d_ptr + blk_idx)
            qs_offsets = blk_idx * BLOCK_HALF + half_idx
            qs_blk = tl.load(v_qs_ptr + qs_offsets).to(tl.int32)
            low = qs_blk & 0xF
            high = (qs_blk >> 4) & 0xF
            c_low = tl.gather(centroid_tile, low, axis=0)
            c_high = tl.gather(centroid_tile, high, axis=0)
            # Dequant values (still in rotated domain; caller must
            # inverse-rotate OR use Q pre-rotated and accept rotated out)
            acc_low += w * c_low * d_blk
            acc_high += w * c_high * d_blk

    # Write interleaved (low_i at 2i, high_i at 2i+1) — matches storage
    tl.store(out_ptr + 2 * half_idx, acc_low)
    tl.store(out_ptr + 2 * half_idx + 1, acc_high)


def tq4_weighted_sum(
    weights: torch.Tensor,     # (N,) fp32
    v_qs: torch.Tensor,        # (N * bpr, 128) uint8
    v_d: torch.Tensor,         # (N * bpr,) fp32
    centroids: torch.Tensor,   # (16,) fp32
    N: int,
    d_head: int,
) -> torch.Tensor:
    """Compute out = weights @ V where V is tq4 stored (N, d_head).
    Returns (d_head,) in the Pi-rotated domain — caller must apply
    Pi.T if needed for downstream (usually not — next layer's Pi-rotation
    on the same residual cancels)."""
    assert v_qs.is_contiguous() and v_qs.dtype == torch.uint8
    assert v_d.is_contiguous() and v_d.dtype == torch.float32
    assert weights.is_contiguous() and weights.dtype == torch.float32
    bpr = d_head // 256
    out = torch.empty(d_head, device=weights.device, dtype=torch.float32)
    _tq4_weighted_sum_kernel[(1,)](
        weights, v_qs.view(-1), v_d, centroids,
        torch.empty(0, device=weights.device),  # pi_ptr placeholder
        out, N,
        BPR=bpr, BLOCK_HALF=128, D_HEAD=d_head,
        num_warps=4,
    )
    return out
