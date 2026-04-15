"""Tests for Gemma-style architectural upgrades."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from calm.llm_computer.gemma_style import (
    GemmaStyleConfig, LayerwiseRMSNorm, RMSNorm, geglu, swiglu,
    sliding_window_mask,
)


# ----- RMSNorm -----

def test_rmsnorm_basic_shape():
    norm = RMSNorm(d_model=8)
    x = torch.randn(2, 4, 8)
    out = norm(x)
    assert out.shape == x.shape


def test_rmsnorm_normalizes_to_unit_rms():
    """After RMSNorm with weight=1, RMS of each vector should be 1 (modulo eps)."""
    norm = RMSNorm(d_model=8)
    x = torch.randn(1, 2, 8) * 100  # large input
    out = norm(x)
    rms = out.pow(2).mean(dim=-1).sqrt()
    # Should all be ~1.0
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)


def test_rmsnorm_weight_scales():
    """Multiplying the weight scales the output by that factor."""
    norm = RMSNorm(d_model=4)
    with torch.no_grad():
        norm.weight.fill_(2.0)
    x = torch.tensor([[[1.0, 1.0, 1.0, 1.0]]])
    out = norm(x)
    # RMS of [1,1,1,1] = 1. So x/rms = [1,1,1,1], then * 2 = [2,2,2,2]
    assert torch.allclose(out, torch.full_like(x, 2.0), atol=1e-3)


# ----- Sliding window mask -----

def test_sliding_window_causal_for_window_equals_seq():
    """With window >= seq_len, mask should match standard causal."""
    mask = sliding_window_mask(seq_len=4, window=100, device=torch.device("cpu"))
    causal = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)
    assert torch.equal(mask, causal)


def test_sliding_window_restricts_long_range():
    """With window=2, query 3 can see only keys {2, 3}."""
    mask = sliding_window_mask(seq_len=5, window=2, device=torch.device("cpu"))
    # q=3 row: should be masked at 0, 1 (too far back) AND at 4 (causal)
    row = mask[3]
    assert row[0].item() is True
    assert row[1].item() is True
    assert row[2].item() is False  # visible
    assert row[3].item() is False  # self
    assert row[4].item() is True   # future


def test_sliding_window_first_positions():
    """Positions below window size behave like causal."""
    mask = sliding_window_mask(seq_len=5, window=3, device=torch.device("cpu"))
    # q=0 can only see self
    assert not mask[0, 0]
    assert mask[0, 1]  # causal mask


# ----- GeGLU -----

def test_geglu_shape():
    gate = torch.randn(2, 4, 8)
    val = torch.randn(2, 4, 8)
    out = geglu(gate, val)
    assert out.shape == (2, 4, 8)


def test_geglu_zero_gate_gives_zero():
    """GELU(0) ≈ 0, so zero gate → zero output (approximately)."""
    gate = torch.zeros(1, 2, 4)
    val = torch.ones(1, 2, 4)
    out = geglu(gate, val)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


def test_swiglu_shape_and_zero():
    gate = torch.zeros(1, 2, 4)
    val = torch.ones(1, 2, 4)
    out = swiglu(gate, val)
    # SiLU(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


def test_swiglu_differs_from_geglu():
    gate = torch.full((1, 1, 4), -1.0)
    val = torch.ones(1, 1, 4)
    sw = swiglu(gate, val)
    ge = geglu(gate, val)
    # SiLU and GELU differ at negative inputs
    assert not torch.allclose(sw, ge)


def test_geglu_differs_from_reglu():
    """With negative gate, GeGLU ≠ ReGLU because GELU ≠ ReLU in
    the negative region."""
    gate = torch.full((1, 1, 4), -1.0)
    val = torch.ones(1, 1, 4)
    geglu_out = geglu(gate, val)
    reglu_out = F.relu(gate) * val
    # ReGLU is 0 (ReLU clips negative); GeGLU has a small non-zero
    # contribution from GELU's smooth decay
    assert torch.allclose(reglu_out, torch.zeros_like(reglu_out))
    assert not torch.allclose(geglu_out, reglu_out)


# ----- GemmaStyleConfig -----

def test_config_defaults_all_off():
    cfg = GemmaStyleConfig()
    for layer in range(4):
        assert not cfg.geglu_at(layer)
        assert cfg.window_at(layer) is None
        assert not cfg.rmsnorm_at(layer)


def test_config_per_layer_geglu():
    cfg = GemmaStyleConfig(use_geglu_per_layer=(True, False, True))
    cfg.validate(n_layers=3)
    assert cfg.geglu_at(0)
    assert not cfg.geglu_at(1)
    assert cfg.geglu_at(2)


def test_config_per_layer_windows():
    cfg = GemmaStyleConfig(attention_windows=(None, 4, None, 8))
    cfg.validate(n_layers=4)
    assert cfg.window_at(0) is None
    assert cfg.window_at(1) == 4
    assert cfg.window_at(3) == 8


def test_config_validates_length():
    cfg = GemmaStyleConfig(use_geglu_per_layer=(True, False))
    try:
        cfg.validate(n_layers=3)
    except AssertionError:
        return
    raise AssertionError("should have raised on length mismatch")


# ----- LayerwiseRMSNorm -----

def test_layerwise_rmsnorm_applies_only_where_enabled():
    """Enabled layers normalize; disabled layers are identity."""
    norm_bank = LayerwiseRMSNorm(
        n_layers=3, d_model=4, enabled_per_layer=[True, False, True],
    )
    x = torch.randn(1, 2, 4) * 100
    # Layer 0: normalized (rms ≈ 1)
    out0 = norm_bank(x, layer_idx=0)
    rms0 = out0.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms0, torch.ones_like(rms0), atol=1e-3)
    # Layer 1: identity
    out1 = norm_bank(x, layer_idx=1)
    assert torch.equal(out1, x)
    # Layer 2: normalized
    out2 = norm_bank(x, layer_idx=2)
    rms2 = out2.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms2, torch.ones_like(rms2), atol=1e-3)


def test_layerwise_rmsnorm_params_only_for_enabled():
    """Parameter count = d_model × count_of_enabled (one weight vector each)."""
    nb = LayerwiseRMSNorm(n_layers=4, d_model=8,
                           enabled_per_layer=[True, False, True, False])
    total = sum(p.numel() for p in nb.parameters())
    # 2 enabled layers × 8 params each = 16
    assert total == 16


if __name__ == "__main__":
    test_rmsnorm_basic_shape()
    print("[ok] RMSNorm shape")
    test_rmsnorm_normalizes_to_unit_rms()
    print("[ok] RMSNorm normalizes to RMS=1")
    test_rmsnorm_weight_scales()
    print("[ok] RMSNorm weight scales output")
    test_sliding_window_causal_for_window_equals_seq()
    print("[ok] window >= seq_len = pure causal")
    test_sliding_window_restricts_long_range()
    print("[ok] window restricts long-range attention")
    test_sliding_window_first_positions()
    print("[ok] first positions behave causally")
    test_geglu_shape()
    print("[ok] GeGLU shape")
    test_geglu_zero_gate_gives_zero()
    print("[ok] GeGLU with zero gate = zero")
    test_geglu_differs_from_reglu()
    print("[ok] GeGLU ≠ ReGLU in negative gate region")
    test_config_defaults_all_off()
    print("[ok] config defaults all off")
    test_config_per_layer_geglu()
    print("[ok] per-layer GeGLU config")
    test_config_per_layer_windows()
    print("[ok] per-layer window config")
    test_config_validates_length()
    print("[ok] config validates length")
    test_layerwise_rmsnorm_applies_only_where_enabled()
    print("[ok] LayerwiseRMSNorm enabled/disabled")
    test_layerwise_rmsnorm_params_only_for_enabled()
    print("[ok] LayerwiseRMSNorm params only for enabled")
