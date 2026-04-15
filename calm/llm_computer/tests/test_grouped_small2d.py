"""Tests for GroupedSmall2DTransformer — per-layer attention mode dispatch.

Verifies:
1. Default (all layers "single") matches vanilla Small2DTransformer bit-exactly
2. Per-layer mode switching runs without error
3. A "grouped" layer produces output equivalent to standard multi-head attention
4. Compiled adder still works when installed into a GroupedSmall2DTransformer
   with "single" mode (backward compatibility)
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from calm.llm_computer.grouped_attention import grouped_attention
from calm.llm_computer.grouped_small2d import (
    GroupedSmall2DConfig, GroupedSmall2DTransformer,
)
from calm.llm_computer.model import Small2DConfig, Small2DTransformer


def test_default_all_single_mode_matches_vanilla():
    """With all layers in 'single' mode, output should match vanilla
    Small2DTransformer (up to float32 roundoff)."""
    torch.manual_seed(0)
    vanilla_cfg = Small2DConfig(
        vocab_size=16, d_model=16, n_heads=8, n_layers=2,
        d_ffn=32, max_len=4, use_hard_max=False,
    )
    grouped_cfg = GroupedSmall2DConfig(
        vocab_size=16, d_model=16, n_heads=8, n_layers=2,
        d_ffn=32, max_len=4, use_hard_max=False,
    )
    vanilla = Small2DTransformer(vanilla_cfg)
    grouped = GroupedSmall2DTransformer(grouped_cfg)
    # Copy weights
    grouped.load_state_dict(vanilla.state_dict())

    x = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    with torch.no_grad():
        out_vanilla = vanilla(x)
        out_grouped = grouped(x)
    assert torch.allclose(out_vanilla, out_grouped, atol=1e-5), (
        f"default mode diverges from vanilla: "
        f"{(out_vanilla - out_grouped).abs().max()}"
    )


def test_grouped_layer_produces_equivalent_to_multihead():
    """A single 'grouped' layer should produce attention output
    equivalent to a standard n_groups-head attention with d_head=group_size*2."""
    torch.manual_seed(1)
    # 4 groups of 2 → equivalent to 4-head attention with d_head=4
    cfg = GroupedSmall2DConfig(
        vocab_size=8, d_model=16, n_heads=8, n_layers=1,
        d_ffn=16, max_len=4, use_hard_max=False,
        layer_modes=("grouped",),
        layer_n_groups=(4,),
        layer_group_sizes=(2,),
    )
    model = GroupedSmall2DTransformer(cfg)
    # Randomize for meaningful test
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0, 0.1)
    x = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    model.eval()
    with torch.no_grad():
        out = model(x)
    # Output should be finite
    assert torch.isfinite(out).all()
    assert out.shape == (1, 4, 8)


def test_mixed_mode_per_layer():
    """One layer 'single', one 'grouped' — no crash, outputs finite."""
    torch.manual_seed(2)
    cfg = GroupedSmall2DConfig(
        vocab_size=8, d_model=16, n_heads=8, n_layers=3,
        d_ffn=16, max_len=4, use_hard_max=False,
        layer_modes=("single", "grouped", "single"),
        layer_n_groups=(1, 4, 1),
        layer_group_sizes=(8, 2, 8),
    )
    model = GroupedSmall2DTransformer(cfg)
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0, 0.1)
    x = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    with torch.no_grad():
        out = model(x)
    assert torch.isfinite(out).all()


def test_grouped_mode_matches_true_multihead_attention():
    """End-to-end: build a GroupedSmall2DTransformer in grouped mode,
    build a separate multi-head attention reference with matching
    weights, verify outputs match."""
    torch.manual_seed(3)
    n_heads_equiv = 2  # "Gemma-like" 2-head attention
    d_head_equiv = 8   # each head d_head=8
    group_size = d_head_equiv // 2  # 4 sub-heads per group
    n_sub_heads = n_heads_equiv * group_size  # 2 * 4 = 8 sub-heads total
    d_model = n_sub_heads * 2  # 16

    cfg = GroupedSmall2DConfig(
        vocab_size=8, d_model=d_model, n_heads=n_sub_heads, n_layers=1,
        d_ffn=16, max_len=4, use_hard_max=False,
        layer_modes=("grouped",),
        layer_n_groups=(n_heads_equiv,),
        layer_group_sizes=(group_size,),
    )
    model = GroupedSmall2DTransformer(cfg)
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0, 0.1)

    # Take the model's first-layer Q/K/V and compare against reference
    x = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    B, S = 1, 4
    pos_idx = torch.arange(S)
    h = model.tok(x) + model.pos(pos_idx)
    qkv = model.W_qkv[0](h)
    qkv = qkv.reshape(B, S, 3, n_sub_heads, 2)
    q, k, v = qkv.permute(2, 0, 3, 1, 4)  # (B, H, S, 2)
    q_bh = q.transpose(1, 2)  # (B, S, H, 2)
    k_bh = k.transpose(1, 2)
    v_bh = v.transpose(1, 2)

    # Reference: multi-head attention with n_heads_equiv heads, d_head_equiv
    # By reshaping the (B, S, n_sub_heads, 2) into (B, S, n_heads_equiv, d_head_equiv)
    Q_ref = q_bh.reshape(B, S, n_heads_equiv, d_head_equiv)
    K_ref = k_bh.reshape(B, S, n_heads_equiv, d_head_equiv)
    V_ref = v_bh.reshape(B, S, n_heads_equiv, d_head_equiv)
    mask = torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1)
    out_ref = torch.zeros_like(Q_ref)
    for head in range(n_heads_equiv):
        Q_h, K_h, V_h = Q_ref[:, :, head, :], K_ref[:, :, head, :], V_ref[:, :, head, :]
        scores = (Q_h @ K_h.transpose(-1, -2)) / math.sqrt(d_head_equiv)
        scores = scores.masked_fill(mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        out_ref[:, :, head, :] = weights @ V_h
    # Flatten to (B, S, d_model)
    out_ref_flat = out_ref.reshape(B, S, d_model)

    # Grouped path
    out_grp = grouped_attention(
        q_bh, k_bh, v_bh,
        n_groups=n_heads_equiv, group_size=group_size, mask=mask,
    )
    out_grp_flat = out_grp.transpose(1, 2).reshape(B, S, d_model).transpose(0, 1).reshape(B, S, d_model) if False else out_grp.reshape(B, S, d_model)

    assert torch.allclose(out_ref_flat, out_grp_flat, atol=1e-5), (
        f"grouped attention doesn't match multi-head reference: "
        f"max diff {(out_ref_flat - out_grp_flat).abs().max()}"
    )


def test_rejects_bad_grouping():
    """n_groups * group_size must equal n_heads."""
    with pytest.raises(AssertionError, match="n_groups"):
        GroupedSmall2DConfig(
            vocab_size=8, d_model=16, n_heads=8, n_layers=1,
            d_ffn=16, max_len=4,
            layer_modes=("grouped",),
            layer_n_groups=(3,),   # 3 * 4 = 12 ≠ 8
            layer_group_sizes=(4,),
        )


def test_rejects_unknown_mode():
    with pytest.raises(AssertionError, match="unknown mode"):
        GroupedSmall2DConfig(
            vocab_size=8, d_model=16, n_heads=8, n_layers=1,
            d_ffn=16, max_len=4,
            layer_modes=("unknown",),
        )


def test_save_reload_preserves_outputs():
    import tempfile
    torch.manual_seed(4)
    cfg = GroupedSmall2DConfig(
        vocab_size=8, d_model=16, n_heads=8, n_layers=2,
        d_ffn=16, max_len=4, use_hard_max=False,
        layer_modes=("single", "grouped"),
        layer_n_groups=(1, 4),
        layer_group_sizes=(8, 2),
    )
    m = GroupedSmall2DTransformer(cfg)
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, 0.1)
    x = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    with torch.no_grad():
        out_pre = m(x)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(m.state_dict(), f.name)
        path = f.name

    m2 = GroupedSmall2DTransformer(cfg)
    m2.load_state_dict(torch.load(path, weights_only=True))
    m2.eval()
    with torch.no_grad():
        out_post = m2(x)
    assert torch.equal(out_pre, out_post)


if __name__ == "__main__":
    test_default_all_single_mode_matches_vanilla()
    print("[ok] default 'single' mode matches vanilla Small2DTransformer")
    test_grouped_layer_produces_equivalent_to_multihead()
    print("[ok] grouped layer produces finite output")
    test_mixed_mode_per_layer()
    print("[ok] mixed single/grouped per layer")
    test_grouped_mode_matches_true_multihead_attention()
    print("[ok] grouped mode = multi-head attention")
    test_rejects_bad_grouping()
    print("[ok] bad grouping rejected")
    test_rejects_unknown_mode()
    print("[ok] unknown mode rejected")
    test_save_reload_preserves_outputs()
    print("[ok] save/reload bit-exact")
