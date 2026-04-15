"""Tests for unified tensor partitioning."""

from __future__ import annotations

import pytest
import torch

from calm.llm_computer.unified_tensor import (
    UnifiedTensorConfig, build_unified_substrate, install_padded_weight,
)


def _tiny_cfg():
    """Tiny test config — keep memory small but exercise all the
    partitioning paths."""
    return UnifiedTensorConfig(
        gemma_d_model=256,
        gemma_n_heads=4,
        gemma_n_kv_heads=2,
        gemma_n_layers=3,
        gemma_d_ffn=512,
        gemma_swa_head_dim=32,
        gemma_full_head_dim=64,
        gemma_vocab_size=100,
        gemma_max_position=32,
        gemma_full_layer_indices=(1,),  # just 1 full layer in this tiny model
        hrm_specialists=("math", "nl", "router"),
        hrm_d_model=8,
        hrm_n_heads=4,
        n_compiled_sub_heads=16,
        keyed_memory_channels=32,
        call_stack_channels=16,
        card_scratchpad_channels=16,
    )


def test_config_sizes_substrate_correctly():
    """Substrate must be large enough for the largest layer type."""
    cfg = _tiny_cfg()
    # Full attention: 4 heads × 64 d_head = 256, so sub-heads = 128
    # SWA: 4 heads × 32 d_head = 128, sub-heads = 64
    # Plus HRMs: 3 × 4 = 12 sub-heads
    # Plus compiled: 16 sub-heads
    # SWA total extras: 64 + 12 + 16 = 92
    # Full requirement: 128
    # max(128, 92) = 128 → substrate_n_heads >= 128
    assert cfg.substrate_n_heads >= 128
    assert cfg.substrate_d_model == cfg.substrate_n_heads * 2


def test_channel_map_doesnt_overflow():
    """Every allocated region fits within substrate_d_model."""
    cfg = _tiny_cfg()
    D = cfg.substrate_d_model
    # Each allocation is within bounds
    assert cfg.gemma_channels[1] <= D
    for name, (lo, hi) in cfg.hrm_channels.items():
        assert hi <= D, f"HRM {name} overflows"
    assert cfg.keyed_memory_range[1] <= D
    assert cfg.call_stack_range[1] <= D
    assert cfg.card_scratchpad_range[1] <= D
    assert cfg.free_channels[1] == D


def test_channel_regions_dont_overlap():
    """All channel allocations are disjoint."""
    cfg = _tiny_cfg()
    regions = [
        ("gemma", cfg.gemma_channels),
        *[(f"hrm_{n}", r) for n, r in cfg.hrm_channels.items()],
        ("keyed", cfg.keyed_memory_range),
        ("stack", cfg.call_stack_range),
        ("scratchpad", cfg.card_scratchpad_range),
    ]
    for i, (name_i, (lo_i, hi_i)) in enumerate(regions):
        for name_j, (lo_j, hi_j) in regions[i+1:]:
            assert hi_i <= lo_j or hi_j <= lo_i, (
                f"regions {name_i} and {name_j} overlap"
            )


def test_sub_heads_dont_overflow():
    cfg = _tiny_cfg()
    N = cfg.substrate_n_heads
    assert cfg.gemma_swa_sub_heads_range[1] <= N
    for lo, hi in cfg.hrm_sub_heads.values():
        assert hi <= N
    assert cfg.compiled_sub_heads_range[1] <= N
    assert cfg.free_sub_heads[1] == N
    assert cfg.gemma_full_sub_heads_range[1] <= N


def test_substrate_builds_and_zeros():
    cfg = _tiny_cfg()
    model = build_unified_substrate(cfg)
    for p in model.parameters():
        assert (p == 0).all(), "substrate should start fully zeroed"


def test_substrate_forward_zero_weights_produces_zero_logits():
    """With all weights zero, forward pass produces zero logits."""
    cfg = _tiny_cfg()
    model = build_unified_substrate(cfg)
    model.eval()
    x = torch.randint(0, cfg.gemma_vocab_size, (1, 4), dtype=torch.long)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 4, cfg.gemma_vocab_size)
    # Zero weights → zero intermediate activations → zero logits
    assert torch.allclose(out, torch.zeros_like(out))


def test_install_padded_weight_corner():
    """install_padded_weight copies a smaller tensor into a corner.

    nn.Linear(in, out).weight shape is (out, in)."""
    import torch.nn as nn
    target = nn.Linear(20, 32, bias=False)  # weight shape (32, 20)
    with torch.no_grad():
        target.weight.zero_()
    source = torch.randn(10, 20) * 0.1  # (out=10, in=20)
    install_padded_weight(target, source, row_offset=0, col_offset=0)
    # Top 10×20 block matches source
    assert torch.equal(target.weight[:10, :20], source)
    # Rest is still zero
    assert torch.allclose(target.weight[10:, :], torch.zeros(22, 20))


def test_install_padded_weight_offset():
    import torch.nn as nn
    target = nn.Linear(16, 32, bias=False)
    with torch.no_grad():
        target.weight.zero_()
    source = torch.randn(5, 4) * 0.1
    install_padded_weight(target, source, row_offset=5, col_offset=3)
    # Placed at [5:10, 3:7]
    assert torch.equal(target.weight[5:10, 3:7], source)
    # Outside is still zero
    assert torch.allclose(target.weight[:5, :], torch.zeros(5, 16))
    assert torch.allclose(target.weight[10:, :], torch.zeros(22, 16))
    assert torch.allclose(target.weight[5:10, :3], torch.zeros(5, 3))
    assert torch.allclose(target.weight[5:10, 7:], torch.zeros(5, 9))


def test_install_padded_weight_rejects_overflow():
    import torch.nn as nn
    target = nn.Linear(8, 8, bias=False)
    source = torch.randn(10, 10)
    with pytest.raises(AssertionError, match="row overflow"):
        install_padded_weight(target, source, row_offset=0, col_offset=0)


def test_describe_includes_all_regions():
    cfg = _tiny_cfg()
    desc = cfg.describe()
    assert "Gemma" in desc
    assert "math" in desc
    assert "router" in desc
    assert "keyed_mem" in desc
    assert "call_stack" in desc
    assert "card_scratch" in desc


def test_gemma_default_config_dimensions_are_sane():
    """Default config with Gemma 4 E4B dims — verify the substrate
    doesn't need to be absurd."""
    cfg = UnifiedTensorConfig()  # default = Gemma 4 E4B
    # Gemma full: 8 × 512/2 = 2048 sub-heads → d_model=4096
    # Gemma SWA: 8 × 256/2 = 1024 sub-heads
    # HRM extras (6 specialists × 16 sub-heads) = 96
    # Compiled: 128
    # SWA total: 1024 + 96 + 128 = 1248
    # Full requirement: 2048
    # max(2048, 1248) = 2048 → d_model=4096
    assert cfg.substrate_d_model == 4096
    assert cfg.substrate_n_heads == 2048


if __name__ == "__main__":
    test_config_sizes_substrate_correctly()
    print("[ok] substrate sized correctly")
    test_channel_map_doesnt_overflow()
    print("[ok] channel map in bounds")
    test_channel_regions_dont_overlap()
    print("[ok] channel regions disjoint")
    test_sub_heads_dont_overflow()
    print("[ok] sub-head map in bounds")
    test_substrate_builds_and_zeros()
    print("[ok] substrate zero-inits")
    test_substrate_forward_zero_weights_produces_zero_logits()
    print("[ok] zero weights → zero logits")
    test_install_padded_weight_corner()
    print("[ok] install weight at corner")
    test_install_padded_weight_offset()
    print("[ok] install weight with offset")
    test_install_padded_weight_rejects_overflow()
    print("[ok] overflow rejected")
    test_describe_includes_all_regions()
    print("[ok] describe readable")
    test_gemma_default_config_dimensions_are_sane()
    print("[ok] Gemma 4 E4B default: substrate d_model=4096, n_heads=2048")
