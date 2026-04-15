"""Tests for grouped attention decomposition."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from calm.llm_computer.grouped_attention import (
    gemma_to_grouped_weights,
    grouped_attention,
    grouped_attention_single_head_mode,
)


def _standard_attention(Q, K, V, mask=None, scale=None):
    """Reference: standard d_head attention (no grouping)."""
    d_head = Q.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(d_head)
    scores = (Q @ K.transpose(-1, -2)) * scale
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    return weights @ V


# ----- Core hypothesis: grouped == standard d_head=big -----

def test_grouped_matches_standard_d_head_256():
    """128 d_head=2 sub-heads summed pre-softmax == d_head=256 standard."""
    torch.manual_seed(42)
    B, S = 2, 6
    d_head_big = 256
    group_size = 128
    Q = torch.randn(B, S, d_head_big)
    K = torch.randn(B, S, d_head_big)
    V = torch.randn(B, S, d_head_big)

    # Standard path: one head of d_head=256
    mask_std = torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1)
    out_std = _standard_attention(Q, K, V, mask=mask_std)

    # Grouped path: 128 sub-heads of d_head=2, scores summed pre-softmax
    Q_sub = Q.reshape(B, S, 1, group_size, 2).reshape(B, S, group_size, 2)
    K_sub = K.reshape(B, S, group_size, 2)
    V_sub = V.reshape(B, S, group_size, 2)
    out_grp = grouped_attention(
        Q_sub, K_sub, V_sub,
        n_groups=1, group_size=group_size, mask=mask_std,
    ).reshape(B, S, d_head_big)

    assert torch.allclose(out_std, out_grp, atol=1e-5), (
        f"max diff: {(out_std - out_grp).abs().max()}"
    )


def test_grouped_matches_multihead_d_head_64():
    """4 heads of d_head=64 == 4 groups of 32 sub-heads of d_head=2."""
    torch.manual_seed(0)
    B, S = 3, 5
    n_heads_gemma = 4
    d_head_gemma = 64
    group_size = 32  # d_head_gemma / 2
    total_d = n_heads_gemma * d_head_gemma

    # Standard multi-head: split per head, attention per head, concat
    Q = torch.randn(B, S, n_heads_gemma, d_head_gemma)
    K = torch.randn(B, S, n_heads_gemma, d_head_gemma)
    V = torch.randn(B, S, n_heads_gemma, d_head_gemma)

    # Per-head reference
    mask = torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1)
    out_std = torch.zeros(B, S, n_heads_gemma, d_head_gemma)
    for h in range(n_heads_gemma):
        out_std[:, :, h, :] = _standard_attention(
            Q[:, :, h, :], K[:, :, h, :], V[:, :, h, :], mask=mask,
        )

    # Grouped: (B, S, n_groups=4 * group_size=32, 2) = (B, S, 128, 2)
    Q_sub = Q.reshape(B, S, n_heads_gemma * group_size, 2)
    K_sub = K.reshape(B, S, n_heads_gemma * group_size, 2)
    V_sub = V.reshape(B, S, n_heads_gemma * group_size, 2)
    out_grp = grouped_attention(
        Q_sub, K_sub, V_sub,
        n_groups=n_heads_gemma, group_size=group_size, mask=mask,
    )
    # Reshape back to (B, S, n_heads_gemma, d_head_gemma)
    out_grp = out_grp.reshape(B, S, n_heads_gemma, d_head_gemma)

    assert torch.allclose(out_std, out_grp, atol=1e-5), (
        f"multi-head decomposition fails: max diff "
        f"{(out_std - out_grp).abs().max()}"
    )


def test_grouped_without_mask():
    """Non-causal case also matches."""
    torch.manual_seed(1)
    B, S = 1, 4
    d_head_big = 32
    group_size = 16
    Q = torch.randn(B, S, d_head_big)
    K = torch.randn(B, S, d_head_big)
    V = torch.randn(B, S, d_head_big)
    out_std = _standard_attention(Q, K, V)
    out_grp = grouped_attention(
        Q.reshape(B, S, group_size, 2),
        K.reshape(B, S, group_size, 2),
        V.reshape(B, S, group_size, 2),
        n_groups=1, group_size=group_size,
    ).reshape(B, S, d_head_big)
    assert torch.allclose(out_std, out_grp, atol=1e-5)


# ----- Single-head mode (compiled programs) -----

def test_single_head_mode_is_pure_d_head_2():
    """grouped_attention_single_head_mode is standard per-head softmax."""
    torch.manual_seed(2)
    B, S, H = 2, 4, 5
    Q = torch.randn(B, S, H, 2)
    K = torch.randn(B, S, H, 2)
    V = torch.randn(B, S, H, 2)
    mask = torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1)
    out = grouped_attention_single_head_mode(Q, K, V, mask=mask)
    # Compare against per-head reference
    out_ref = torch.zeros_like(Q)
    for h in range(H):
        out_ref[:, :, h, :] = _standard_attention(
            Q[:, :, h, :], K[:, :, h, :], V[:, :, h, :], mask=mask,
        )
    assert torch.allclose(out, out_ref, atol=1e-5)


def test_single_head_differs_from_grouped_same_shape():
    """Sanity: per-head softmax ≠ summed-pre-softmax."""
    torch.manual_seed(3)
    B, S, H = 1, 3, 4
    Q = torch.randn(B, S, H, 2)
    K = torch.randn(B, S, H, 2)
    V = torch.randn(B, S, H, 2)
    out_single = grouped_attention_single_head_mode(Q, K, V)
    out_grouped = grouped_attention(
        Q, K, V, n_groups=1, group_size=H,
    )
    # They should NOT be equal (different attention patterns)
    assert not torch.allclose(out_single, out_grouped, atol=1e-3)


# ----- Weight reshape -----

def test_gemma_to_grouped_weights_preserves_elements():
    torch.manual_seed(4)
    d_model, n_heads, d_head = 128, 4, 32
    w = torch.randn(d_model, n_heads * d_head)
    reshaped = gemma_to_grouped_weights(w, n_heads, d_head)
    # Shape check
    assert reshaped.shape == (d_model, n_heads, d_head // 2, 2)
    # Element count check
    assert reshaped.numel() == w.numel()
    # Flatten the grouped form; it should match the original
    w_flat = reshaped.reshape(d_model, n_heads * d_head)
    assert torch.equal(w_flat, w)


def test_gemma_to_grouped_odd_d_head_rejected():
    w = torch.zeros(64, 4 * 33)  # d_head=33 (odd)
    with pytest.raises(AssertionError, match="even"):
        gemma_to_grouped_weights(w, n_heads_gemma=4, d_head_gemma=33)


def test_wrong_shape_rejected():
    w = torch.zeros(64, 128)
    with pytest.raises(AssertionError, match="doesn't match"):
        gemma_to_grouped_weights(w, n_heads_gemma=4, d_head_gemma=64)


# ----- Integration test: full loop with weight reshape -----

def test_end_to_end_gemma_equivalent():
    """Build Gemma-style attention with W_Q, W_K, W_V, reshape, run
    grouped, compare to standard."""
    torch.manual_seed(5)
    B, S = 2, 4
    d_model = 128
    n_heads_gemma = 4
    d_head_gemma = 32
    group_size = d_head_gemma // 2

    W_Q = torch.randn(d_model, n_heads_gemma * d_head_gemma) * 0.1
    W_K = torch.randn(d_model, n_heads_gemma * d_head_gemma) * 0.1
    W_V = torch.randn(d_model, n_heads_gemma * d_head_gemma) * 0.1

    x = torch.randn(B, S, d_model)
    Q = x @ W_Q  # (B, S, n_heads*d_head)
    K = x @ W_K
    V = x @ W_V

    Q_h = Q.reshape(B, S, n_heads_gemma, d_head_gemma)
    K_h = K.reshape(B, S, n_heads_gemma, d_head_gemma)
    V_h = V.reshape(B, S, n_heads_gemma, d_head_gemma)

    mask = torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1)
    out_std = torch.zeros_like(Q_h)
    for h in range(n_heads_gemma):
        out_std[:, :, h, :] = _standard_attention(
            Q_h[:, :, h, :], K_h[:, :, h, :], V_h[:, :, h, :], mask=mask,
        )

    # Now reshape everything to sub-head layout and use grouped_attention
    Q_sub = Q.reshape(B, S, n_heads_gemma * group_size, 2)
    K_sub = K.reshape(B, S, n_heads_gemma * group_size, 2)
    V_sub = V.reshape(B, S, n_heads_gemma * group_size, 2)
    out_grp = grouped_attention(
        Q_sub, K_sub, V_sub,
        n_groups=n_heads_gemma, group_size=group_size, mask=mask,
    ).reshape(B, S, n_heads_gemma, d_head_gemma)

    assert torch.allclose(out_std, out_grp, atol=1e-5), (
        f"end-to-end mismatch: {(out_std - out_grp).abs().max()}"
    )


if __name__ == "__main__":
    test_grouped_matches_standard_d_head_256()
    print("[ok] 128 sub-heads == d_head=256 (Gemma-scale)")
    test_grouped_matches_multihead_d_head_64()
    print("[ok] 4 heads of d_head=64 == grouped equivalent")
    test_grouped_without_mask()
    print("[ok] non-causal case")
    test_single_head_mode_is_pure_d_head_2()
    print("[ok] single-head mode = standard d_head=2")
    test_single_head_differs_from_grouped_same_shape()
    print("[ok] single-head != grouped on same shape")
    test_gemma_to_grouped_weights_preserves_elements()
    print("[ok] weight reshape preserves elements")
    test_gemma_to_grouped_odd_d_head_rejected()
    print("[ok] odd d_head rejected")
    test_wrong_shape_rejected()
    print("[ok] wrong shape rejected")
    test_end_to_end_gemma_equivalent()
    print("[ok] end-to-end Gemma-equivalent attention")
