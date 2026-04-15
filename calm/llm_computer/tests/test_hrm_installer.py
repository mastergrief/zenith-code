"""Tests for HRM → substrate installation."""

from __future__ import annotations

import pytest
import torch

from calm.llm_computer.hrm_installer import (
    build_tiny_hrm_for_testing, install_hrm_full, install_hrm_into_substrate,
)
from calm.llm_computer.unified_tensor import (
    UnifiedTensorConfig, build_unified_substrate,
)


def _tiny_cfg():
    return UnifiedTensorConfig(
        gemma_d_model=256, gemma_n_heads=4, gemma_n_kv_heads=2,
        gemma_n_layers=2, gemma_d_ffn=512,
        gemma_swa_head_dim=32, gemma_full_head_dim=64,
        gemma_vocab_size=100, gemma_max_position=32,
        gemma_full_layer_indices=(1,),
        hrm_specialists=("math", "nl"),
        hrm_d_model=8, hrm_n_heads=4,
        n_compiled_sub_heads=16,
        keyed_memory_channels=16, call_stack_channels=8,
        card_scratchpad_channels=8,
    )


def test_install_hrm_places_weights_in_reserved_range():
    cfg = _tiny_cfg()
    substrate = build_unified_substrate(cfg)
    hrm = build_tiny_hrm_for_testing(
        d_model=cfg.hrm_d_model, n_heads=cfg.hrm_n_heads, n_layers=2,
    )

    # Before: substrate is all zeros
    assert (substrate.W_qkv[0].weight == 0).all()

    # Install math HRM
    install_hrm_full(substrate, cfg, hrm, "math")

    # After: substrate weights are non-zero in math's reserved region
    ch_lo, ch_hi = cfg.hrm_channels["math"]
    sh_lo, sh_hi = cfg.hrm_sub_heads["math"]
    # Q portion: rows [2*sh_lo, 2*sh_hi], cols [ch_lo, ch_hi]
    q_slice = substrate.W_qkv[0].weight[2*sh_lo:2*sh_hi, ch_lo:ch_hi]
    assert (q_slice != 0).any(), "HRM Q weights should be installed"
    # Verify they match HRM's weights
    assert torch.equal(
        q_slice,
        hrm.W_qkv[0].weight[0:cfg.hrm_d_model, :],
    ), "HRM Q weights should be byte-for-byte installed"


def test_install_doesnt_touch_other_regions():
    cfg = _tiny_cfg()
    substrate = build_unified_substrate(cfg)
    hrm = build_tiny_hrm_for_testing(
        d_model=cfg.hrm_d_model, n_heads=cfg.hrm_n_heads,
    )
    install_hrm_full(substrate, cfg, hrm, "math")

    ch_lo, ch_hi = cfg.hrm_channels["math"]
    sh_lo, sh_hi = cfg.hrm_sub_heads["math"]
    # Gemma's region (columns 0..gemma_d_model) should still be zero
    # in the math sub-head row range
    assert (substrate.W_qkv[0].weight[2*sh_lo:2*sh_hi, :ch_lo] == 0).all(), (
        "Gemma's input channels should be untouched"
    )
    # NL HRM's region should still be zero
    nl_ch_lo, nl_ch_hi = cfg.hrm_channels["nl"]
    assert (substrate.W_qkv[0].weight[2*sh_lo:2*sh_hi, nl_ch_lo:nl_ch_hi] == 0).all()


def test_two_hrms_coexist():
    cfg = _tiny_cfg()
    substrate = build_unified_substrate(cfg)
    hrm_math = build_tiny_hrm_for_testing(cfg.hrm_d_model, cfg.hrm_n_heads)
    hrm_nl = build_tiny_hrm_for_testing(cfg.hrm_d_model, cfg.hrm_n_heads)

    install_hrm_full(substrate, cfg, hrm_math, "math")
    install_hrm_full(substrate, cfg, hrm_nl, "nl")

    # Both HRMs occupy different regions; both should have their weights
    sh_math_lo, sh_math_hi = cfg.hrm_sub_heads["math"]
    sh_nl_lo, sh_nl_hi = cfg.hrm_sub_heads["nl"]
    ch_math_lo, ch_math_hi = cfg.hrm_channels["math"]
    ch_nl_lo, ch_nl_hi = cfg.hrm_channels["nl"]

    q_math = substrate.W_qkv[0].weight[2*sh_math_lo:2*sh_math_hi, ch_math_lo:ch_math_hi]
    q_nl = substrate.W_qkv[0].weight[2*sh_nl_lo:2*sh_nl_hi, ch_nl_lo:ch_nl_hi]
    assert torch.equal(q_math, hrm_math.W_qkv[0].weight[0:cfg.hrm_d_model, :])
    assert torch.equal(q_nl, hrm_nl.W_qkv[0].weight[0:cfg.hrm_d_model, :])
    # They're in different regions, so they don't conflict
    assert not torch.equal(q_math, q_nl)


def test_unknown_hrm_raises():
    cfg = _tiny_cfg()
    substrate = build_unified_substrate(cfg)
    hrm = build_tiny_hrm_for_testing(cfg.hrm_d_model, cfg.hrm_n_heads)
    with pytest.raises(KeyError, match="not in config"):
        install_hrm_into_substrate(substrate, cfg, hrm, "ghost_hrm", layer_idx=0)


def test_hrm_dimension_mismatch_rejected():
    cfg = _tiny_cfg()
    substrate = build_unified_substrate(cfg)
    # Build HRM with wrong d_model
    hrm_wrong = build_tiny_hrm_for_testing(
        d_model=16,  # cfg expects 8
        n_heads=cfg.hrm_n_heads,
    )
    with pytest.raises(AssertionError, match="d_model"):
        install_hrm_into_substrate(substrate, cfg, hrm_wrong, "math", 0)


def test_hrm_too_many_layers_rejected():
    cfg = _tiny_cfg()
    substrate = build_unified_substrate(cfg)
    # HRM with more layers than substrate
    hrm_deep = build_tiny_hrm_for_testing(
        d_model=cfg.hrm_d_model, n_heads=cfg.hrm_n_heads,
        n_layers=cfg.gemma_n_layers + 5,
    )
    with pytest.raises(ValueError, match="more layers"):
        install_hrm_full(substrate, cfg, hrm_deep, "math")


if __name__ == "__main__":
    test_install_hrm_places_weights_in_reserved_range()
    print("[ok] HRM weights installed into reserved range")
    test_install_doesnt_touch_other_regions()
    print("[ok] HRM install preserves other regions")
    test_two_hrms_coexist()
    print("[ok] two HRMs coexist without interference")
    test_unknown_hrm_raises()
    print("[ok] unknown HRM name rejected")
    test_hrm_dimension_mismatch_rejected()
    print("[ok] dimension mismatch rejected")
    test_hrm_too_many_layers_rejected()
    print("[ok] too-many-layers HRM rejected")
