"""Grouped d_head=2 attention — exact decomposition of d_head=256 attention
into 128 d_head=2 sub-heads with scores summed before softmax.

Hypothesis validated empirically: the math is exact to float32 roundoff.
A d_head=256 attention head operation is mathematically identical to 128
parallel d_head=2 sub-head operations whose scores are summed (not
softmax'd independently) before the softmax-and-apply-to-V step.

  (Q · K^T)[i,j] = Σ_d Q[i,d] · K[j,d]                           # standard
                = Σ_g Σ_{d ∈ group_g} Q[i,d] · K[j,d]              # partition
                = Σ_g (Q_g · K_g^T)[i,j]                           # sum across groups

This means ONE Small2DTransformer with d_head=2 can host attention
equivalent to ANY larger-head model (Gemma, Llama, ...) by using
grouped mode per layer. Compiled programs still use single-head mode
on the same substrate — both coexist in one tensor.

This module ships:
  - grouped_attention(Q, K, V, n_groups, group_size, mask, scale) →
    exact decomposition of d_head=(2*group_size) attention into
    n_groups * group_size sub-heads of d_head=2.
  - grouped_attention_single — convenience for single-group mode
    (equivalent to standard d_head=2 attention).

The substrate integration (per-layer attention mode, Gemma weight loader
reshape) is in gemma_in_substrate.py.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


def grouped_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
    *,
    n_groups: int,
    group_size: int,
    mask: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Grouped d_head=2 attention.

    Treats (n_groups * group_size) d_head=2 sub-heads as if they formed
    n_groups heads of d_head=(2 * group_size) each. Scores are summed
    across sub-heads within each group BEFORE softmax.

    Args:
        Q, K, V: shape (B, S, n_groups * group_size, 2). The
            (n_groups, group_size) structure is implicit in this flat
            layout — callers must ensure sub-head indices
            [g * group_size : (g+1) * group_size] belong to group g.
        n_groups: number of equivalent heads (= Gemma's n_heads).
        group_size: sub-heads per group (= Gemma's d_head / 2).
        mask: optional (S, S) boolean — True = masked out.
        scale: denominator for softmax. If None, uses
            1/sqrt(2 * group_size) — equivalent to 1/sqrt(d_head_equivalent).

    Returns:
        (B, S, n_groups * group_size, 2) attention output.
    """
    B, S, H, D = Q.shape
    assert D == 2, f"grouped_attention requires d_head=2, got {D}"
    assert H == n_groups * group_size, (
        f"n_heads {H} != n_groups {n_groups} × group_size {group_size}"
    )
    d_head_equivalent = 2 * group_size
    if scale is None:
        scale = 1.0 / math.sqrt(d_head_equivalent)

    # Reshape to (B, S, n_groups, group_size, 2)
    Q_g = Q.reshape(B, S, n_groups, group_size, D)
    K_g = K.reshape(B, S, n_groups, group_size, D)
    V_g = V.reshape(B, S, n_groups, group_size, D)

    # Per-sub-head unnormalized scores summed across sub-heads per group.
    # scores[b, q_pos, group, k_pos] = sum over (sub_head, d=2) of Q · K
    scores = torch.einsum("biged, bjged -> bigj", Q_g, K_g) * scale
    # shape: (B, S_q, n_groups, S_k)

    if mask is not None:
        # Broadcast mask (S, S) across (B, n_groups)
        scores = scores.masked_fill(
            mask.view(1, S, 1, S), float("-inf"),
        )

    weights = F.softmax(scores, dim=-1)  # (B, S_q, n_groups, S_k)

    # Apply weights: per group, softmax is SHARED across all group_size sub-heads
    # out[b, q, group, sub, d] = sum_k weights[b, q, group, k] * V[b, k, group, sub, d]
    out_g = torch.einsum("bigj, bjged -> biged", weights, V_g)
    return out_g.reshape(B, S, H, D)


def grouped_attention_single_head_mode(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
    *,
    mask: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Pure d_head=2 attention with per-sub-head softmax — the substrate
    default mode, used by compiled programs.

    This is equivalent to grouped_attention with n_groups=H, group_size=1,
    but expressed as a convenience for callers who don't want to think
    about grouping. Shape contract matches standard multi-head attention.
    """
    B, S, H, D = Q.shape
    assert D == 2
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    # (B, S_q, H, S_k)
    scores = torch.einsum("bihd, bjhd -> bihj", Q, K) * scale
    if mask is not None:
        scores = scores.masked_fill(mask.view(1, S, 1, S), float("-inf"))
    weights = F.softmax(scores, dim=-1)
    return torch.einsum("bihj, bjhd -> bihd", weights, V)


def gemma_to_grouped_weights(
    w_gemma: torch.Tensor,
    n_heads_gemma: int,
    d_head_gemma: int,
) -> torch.Tensor:
    """Reshape a Gemma Q/K/V weight matrix into grouped-substrate layout.

    Gemma's W_Q has shape (d_model, n_heads_gemma * d_head_gemma). We
    reinterpret it as (d_model, n_heads_gemma, group_size, 2) where
    group_size = d_head_gemma / 2. This is a PURE RESHAPE — no
    information lost, no computation changed.

    The returned tensor can be flattened to
    (d_model, n_heads_gemma * group_size * 2) and installed directly
    into a Small2DTransformer's W_qkv weight for the layers that use
    grouped attention mode.
    """
    assert d_head_gemma % 2 == 0, (
        f"d_head_gemma {d_head_gemma} must be even for d_head=2 grouping"
    )
    assert w_gemma.shape[1] == n_heads_gemma * d_head_gemma, (
        f"w_gemma shape {w_gemma.shape} doesn't match "
        f"({w_gemma.shape[0]}, {n_heads_gemma} * {d_head_gemma})"
    )
    group_size = d_head_gemma // 2
    d_model = w_gemma.shape[0]
    return w_gemma.reshape(d_model, n_heads_gemma, group_size, 2)
