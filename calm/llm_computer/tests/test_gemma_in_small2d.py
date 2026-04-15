"""Tests for Gemma-in-substrate integration.

Verifies the specific dimensional constraint and provides a positive
test for UNIFORM-attention models (which DO fit cleanly)."""

from __future__ import annotations

import pytest
import torch

from calm.llm_computer.gemma4_config import Gemma4Config
from calm.llm_computer.gemma_in_small2d import (
    substrate_config_for_gemma_swa_only,
)
from calm.llm_computer.grouped_attention import grouped_attention
from calm.llm_computer.grouped_small2d import (
    GroupedSmall2DConfig, GroupedSmall2DTransformer,
)


def test_swa_only_config_dimensions_cleanly():
    """SWA-only substrate fits at d_model=2048 (not Gemma's 2560)."""
    gemma_cfg = Gemma4Config()
    substrate_cfg = substrate_config_for_gemma_swa_only(gemma_cfg)

    # Expected: 8 groups × 128 sub-heads = 1024 sub-heads, d_model=2048
    expected_n_sub_heads = gemma_cfg.n_heads * (gemma_cfg.swa_head_dim // 2)
    assert substrate_cfg.n_heads == expected_n_sub_heads
    assert substrate_cfg.d_model == expected_n_sub_heads * 2
    assert substrate_cfg.n_layers == gemma_cfg.n_layers
    # All layers use grouped mode
    assert all(m == "grouped" for m in substrate_cfg.layer_modes)
    # n_groups × group_size must equal n_heads
    for ng, gs in zip(substrate_cfg.layer_n_groups, substrate_cfg.layer_group_sizes):
        assert ng * gs == substrate_cfg.n_heads
    # group_size × 2 = Gemma's SWA d_head
    assert substrate_cfg.layer_group_sizes[0] * 2 == gemma_cfg.swa_head_dim


def test_uniform_attention_model_fits_cleanly():
    """A UNIFORM-attention model (one d_head across all layers) fits
    cleanly in one Small2DTransformer via grouped decomposition.
    This is the positive case our grouped attention supports."""
    # Hypothetical: 8-head model, d_head=64 throughout, d_model=512
    n_heads_gemma = 8
    d_head_gemma = 64
    n_layers = 4
    d_model = n_heads_gemma * d_head_gemma  # 512
    group_size = d_head_gemma // 2  # 32
    n_sub_heads = n_heads_gemma * group_size  # 256

    # Substrate config: all layers grouped, same n_groups, same group_size
    substrate_cfg = GroupedSmall2DConfig(
        vocab_size=100,
        d_model=d_model,  # 512
        n_heads=n_sub_heads,  # 256
        n_layers=n_layers,
        d_ffn=1024,
        max_len=16,
        use_hard_max=False,
        layer_modes=tuple(["grouped"] * n_layers),
        layer_n_groups=tuple([n_heads_gemma] * n_layers),
        layer_group_sizes=tuple([group_size] * n_layers),
    )
    # Must build without error
    model = GroupedSmall2DTransformer(substrate_cfg)
    assert model._grouped_config.d_model == d_model
    # Forward pass works
    x = torch.randint(0, 100, (1, 4), dtype=torch.long)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 4, 100)
    assert torch.isfinite(out).all()


def test_gemma4_heterogeneous_attention_constraint_documented():
    """Document the constraint: Gemma 4's mixed SWA (d_head=256) +
    full (d_head=512) doesn't fit one Small2DTransformer at d_model=2560.

    This test codifies the reason: for both layer types to decompose
    cleanly into d_head=2 sub-heads with the SAME substrate n_heads,
    we'd need n_heads=8*128 for SWA AND n_heads=8*256 for full —
    contradiction.

    Workarounds documented in gemma_in_small2d.py:
      - Upscale substrate to d_model=4096 (SWA gets padding)
      - Split into two tensors (one for SWA, one for full)
      - Use UnifiedCHRLMCard (multi-stream, not single-tensor)
    """
    gemma_cfg = Gemma4Config()
    # What SWA would need
    swa_n_sub_heads = gemma_cfg.n_heads * (gemma_cfg.swa_head_dim // 2)
    # What full would need
    full_n_sub_heads = gemma_cfg.n_heads * (gemma_cfg.full_head_dim // 2)
    # They're different
    assert swa_n_sub_heads != full_n_sub_heads
    # A single uniform n_heads substrate can host ONE of them cleanly.


def test_grouped_attention_math_works_at_gemma_scale():
    """The math of grouped decomposition works at Gemma-size d_model.
    Even if the full Gemma 4 doesn't fit in one tensor, the underlying
    attention decomposition scales correctly."""
    # Simulate one SWA layer's attention at proper Gemma E4B scale
    # d_head_gemma=256, n_heads=8, so 128 sub-heads of d_head=2 per group
    B, S = 1, 3  # small to keep test fast
    n_heads_gemma = 8
    d_head_gemma = 256
    group_size = d_head_gemma // 2  # 128
    n_sub_heads = n_heads_gemma * group_size  # 1024
    d_model = n_sub_heads * 2  # 2048

    torch.manual_seed(0)
    # Random Q/K/V in (B, S, n_sub_heads, 2) form
    Q = torch.randn(B, S, n_sub_heads, 2) * 0.05
    K = torch.randn(B, S, n_sub_heads, 2) * 0.05
    V = torch.randn(B, S, n_sub_heads, 2) * 0.05
    mask = torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1)

    out = grouped_attention(
        Q, K, V,
        n_groups=n_heads_gemma, group_size=group_size, mask=mask,
    )
    assert out.shape == (B, S, n_sub_heads, 2)
    assert torch.isfinite(out).all()

    # Verify it equals standard 8-head d_head=256 attention
    import math
    import torch.nn.functional as F
    Q_h = Q.reshape(B, S, n_heads_gemma, d_head_gemma)
    K_h = K.reshape(B, S, n_heads_gemma, d_head_gemma)
    V_h = V.reshape(B, S, n_heads_gemma, d_head_gemma)
    out_std = torch.zeros_like(Q_h)
    for h in range(n_heads_gemma):
        scores = (Q_h[:, :, h] @ K_h[:, :, h].transpose(-1, -2)) / math.sqrt(d_head_gemma)
        scores = scores.masked_fill(mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        out_std[:, :, h] = weights @ V_h[:, :, h]

    out_flat = out.reshape(B, S, d_model)
    out_std_flat = out_std.reshape(B, S, d_model)
    assert torch.allclose(out_flat, out_std_flat, atol=1e-5)


if __name__ == "__main__":
    test_swa_only_config_dimensions_cleanly()
    print("[ok] SWA-only substrate config clean")
    test_uniform_attention_model_fits_cleanly()
    print("[ok] uniform-attention model fits cleanly")
    test_gemma4_heterogeneous_attention_constraint_documented()
    print("[ok] Gemma 4 heterogeneous attention constraint documented")
    test_grouped_attention_math_works_at_gemma_scale()
    print("[ok] grouped attention math at Gemma scale (n_sub_heads=1024)")
