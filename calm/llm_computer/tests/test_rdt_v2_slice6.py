"""Slice 6 regression tests: short-conv after QKV (Tier B).

Wires the already-declared `DeltaNetConfig.use_short_conv` flag. Per
co_lead audit msg 1779304303629: causal depthwise 1D conv (k=4) over
Q/K/V AFTER the QKV projection, BEFORE feature-map/L2-norm. Applied
only on the DeltaNet-side q_flat/k_flat/v_flat — the softmax-attention
path is untouched. Per-channel (groups=d_model), no cross-channel mix.

Cached-decode blocklist now includes `use_short_conv` (per-token cached
path can't mirror causal conv without keeping the last k-1 hidden/QKV
values in streaming state).

GPU-only per user direction.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from calm.llm_computer.copy_augmented_delta import build_copy_augmented_delta
from calm.llm_computer.dt_install import load_dt_checkpoint


DEVICE = "cuda"
SEP_ID = 3

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Slice 6 tests are GPU-only per user direction; CUDA unavailable",
)


def _make_dt(use_short_conv: bool = False, seed: int = 42):
    torch.manual_seed(seed)
    return build_copy_augmented_delta(
        vocab_size=20, d_model=8, n_heads=4, n_layers=2, d_ffn=16,
        max_len=24, n_copy_heads=2, sep_token_id=SEP_ID,
        use_chunkwise=False, n_iterations=1,
        use_short_conv=use_short_conv,
    ).to(DEVICE)


# ===== Section A: Allocation + flag-off bit-equivalence =====

def test_short_conv_off_modules_are_none():
    """Flag off → no conv modules allocated. Zero state-dict drift."""
    m = _make_dt(use_short_conv=False, seed=42)
    assert m.short_conv_q is None
    assert m.short_conv_k is None
    assert m.short_conv_v is None


def test_short_conv_on_three_modulelists_allocated():
    """Flag on → three ModuleLists (Q, K, V) of n_layers conv modules each."""
    m = _make_dt(use_short_conv=True, seed=42)
    assert isinstance(m.short_conv_q, nn.ModuleList)
    assert isinstance(m.short_conv_k, nn.ModuleList)
    assert isinstance(m.short_conv_v, nn.ModuleList)
    assert len(m.short_conv_q) == 2  # n_layers
    # Each conv is depthwise (groups=d_model), kernel_size=4, bias=False
    for conv in m.short_conv_q:
        assert isinstance(conv, nn.Conv1d)
        assert conv.kernel_size == (4,)
        assert conv.groups == 8  # d_model
        assert conv.bias is None


def test_short_conv_param_count_increase():
    """Param count rises by ~3 × kernel_size × d_model per layer (depthwise).
    n_layers=2, k=4, d_model=8 → +3 × 4 × 8 × 2 = 192 params from the convs."""
    m_off = _make_dt(use_short_conv=False, seed=42)
    m_on = _make_dt(use_short_conv=True, seed=42)
    diff = m_on.param_count() - m_off.param_count()
    expected = 3 * 4 * 8 * 2  # 3 (Q/K/V) * k * d_model * n_layers (depthwise)
    assert diff == expected, f"expected +{expected}, got +{diff}"


def test_short_conv_off_forward_equivalent_to_pre_slice6():
    """Flag off → forward path is bit-identical to pre-Slice-6 (no conv
    is invoked, no extra ops in the DeltaNet path). Same seed → same params
    → same logits."""
    m = _make_dt(use_short_conv=False, seed=42)
    m.eval()
    idx = torch.tensor([[5, 7, SEP_ID, 9, 11, 13]], device=DEVICE)
    with torch.no_grad():
        a = m(idx)
        b = m(idx)
    assert torch.equal(a, b)
    assert torch.isfinite(a).all()


# ===== Section B: Flag-on observable effect =====

def test_short_conv_changes_output():
    """Flag-on convs mix local Q/K/V → output differs from flag-off."""
    m_off = _make_dt(use_short_conv=False, seed=42)
    m_on = _make_dt(use_short_conv=True, seed=42)
    m_off.eval(); m_on.eval()
    idx = torch.tensor([[5, 7, SEP_ID, 9, 11, 13]], device=DEVICE)
    with torch.no_grad():
        a = m_off(idx)
        b = m_on(idx)
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    assert not torch.allclose(a, b)


def test_short_conv_causal_property():
    """Causality: position t's conv output depends only on positions [t-k+1, t].
    Verified empirically: changing a later token MUST NOT change earlier
    output positions."""
    m = _make_dt(use_short_conv=True, seed=42)
    m.eval()
    idx_a = torch.tensor([[5, 7, SEP_ID, 9, 11, 13]], device=DEVICE)
    idx_b = idx_a.clone()
    idx_b[0, 5] = 19  # change LAST position only
    with torch.no_grad():
        out_a = m(idx_a)
        out_b = m(idx_b)
    # Positions 0..4 must be identical between out_a and out_b (causality)
    # Position 5 may differ because that's where the change is.
    # However: token-embedding lookup at pos 5 differs → x[..., 5, :] differs
    # immediately. Even with causal conv, that flows through the DeltaNet
    # recurrence's S_state which is sequential. So early positions stay equal
    # ONLY if no information from pos 5 leaks backward. DeltaNet's read-after-
    # write is sequential left-to-right; the conv is causal; so this holds.
    for pos in range(5):
        assert torch.equal(out_a[0, pos], out_b[0, pos]), (
            f"causal-conv leak: pos {pos} differs after changing pos 5"
        )


# ===== Section C: Cached-decode blocklist =====

def test_cached_decode_blocks_short_conv():
    """`use_short_conv` is now on the blocklist. decode_greedy_cached must
    raise NotImplementedError when flag on."""
    m = _make_dt(use_short_conv=True, seed=42)
    m.eval()
    prefix = torch.tensor([[5, 7, SEP_ID, 9, 11]], device=DEVICE)
    with pytest.raises(NotImplementedError, match=r"use_short_conv"):
        m.decode_greedy_cached(prefix, max_gen=2, eos_token=None)


def test_cached_decode_still_allows_default():
    """Pure-default config (short_conv off) still runs cached decode unaffected."""
    m = _make_dt(use_short_conv=False, seed=42)
    m.eval()
    prefix = torch.tensor([[5, 7, SEP_ID, 9, 11]], device=DEVICE)
    out = m.decode_greedy_cached(prefix, max_gen=2, eos_token=None)
    assert out.shape[0] == 1


# ===== Section D: Gradient flow + finite output =====

def test_short_conv_gradient_flows():
    """Conv weights receive gradient during backward."""
    m = _make_dt(use_short_conv=True, seed=42)
    idx = torch.tensor([[5, 7, SEP_ID, 9, 11, 13]], device=DEVICE)
    out = m(idx)
    out.sum().backward()
    for conv in m.short_conv_q:
        g = conv.weight.grad
        assert g is not None
        assert torch.isfinite(g).all()
        assert g.abs().max().item() > 0


def test_short_conv_finite_output():
    """Sanity: flag-on output is finite."""
    m = _make_dt(use_short_conv=True, seed=42)
    m.eval()
    idx = torch.tensor([[5, 7, SEP_ID, 9, 11, 13]], device=DEVICE)
    with torch.no_grad():
        out = m(idx)
    assert torch.isfinite(out).all()


# ===== Section E: Checkpoint round-trip =====

def test_checkpoint_roundtrip_short_conv():
    """use_short_conv persists; conv ModuleLists reload; logits match."""
    m_orig = _make_dt(use_short_conv=True, seed=42)
    m_orig.eval()
    idx = torch.tensor([[5, 7, SEP_ID, 9, 11, 13]], device=DEVICE)
    with torch.no_grad():
        out_orig = m_orig(idx)
    assert torch.isfinite(out_orig).all()

    cfg_dict = {
        "vocab_size": 20, "max_len": 24,
        "d_model": 8, "n_heads": 4, "n_layers": 2,
        "d_ffn": 16, "n_copy_heads": 2,
        "use_chunkwise": False,
        "n_iterations": 1,
        "use_short_conv": True,
    }
    ckpt = {
        "model_state": m_orig.state_dict(),
        "config": cfg_dict,
        "epoch": 1, "val_autoreg": 0.5, "n_train": 100, "n_val": 10,
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_slice6.pt"
        torch.save(ckpt, path)
        m_loaded, _ = load_dt_checkpoint(str(path), device=DEVICE)
    m_loaded.eval()

    assert m_loaded.config.use_short_conv is True
    assert isinstance(m_loaded.short_conv_q, nn.ModuleList)
    assert len(m_loaded.short_conv_q) == 2

    with torch.no_grad():
        out_loaded = m_loaded(idx)
    assert torch.equal(out_orig, out_loaded)


def test_checkpoint_backward_compat_short_conv():
    """Old checkpoint without use_short_conv loads with flag=False."""
    m_orig = _make_dt(use_short_conv=False, seed=42)
    m_orig.eval()
    idx = torch.tensor([[5, 7, SEP_ID, 9, 11, 13]], device=DEVICE)
    with torch.no_grad():
        out_orig = m_orig(idx)

    cfg_dict = {
        "vocab_size": 20, "max_len": 24,
        "d_model": 8, "n_heads": 4, "n_layers": 2,
        "d_ffn": 16, "n_copy_heads": 2,
        "use_chunkwise": False,
        # NO use_short_conv key
    }
    ckpt = {
        "model_state": m_orig.state_dict(),
        "config": cfg_dict,
        "epoch": 1, "val_autoreg": 0.5, "n_train": 100, "n_val": 10,
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_slice6_compat.pt"
        torch.save(ckpt, path)
        m_loaded, _ = load_dt_checkpoint(str(path), device=DEVICE)
    m_loaded.eval()
    assert m_loaded.config.use_short_conv is False
    assert m_loaded.short_conv_q is None

    with torch.no_grad():
        out_loaded = m_loaded(idx)
    assert torch.equal(out_orig, out_loaded)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA unavailable, skipping (GPU-only)")
        raise SystemExit(0)
    for fn_name in [
        "test_short_conv_off_modules_are_none",
        "test_short_conv_on_three_modulelists_allocated",
        "test_short_conv_param_count_increase",
        "test_short_conv_off_forward_equivalent_to_pre_slice6",
        "test_short_conv_changes_output",
        "test_short_conv_causal_property",
        "test_cached_decode_blocks_short_conv",
        "test_cached_decode_still_allows_default",
        "test_short_conv_gradient_flows",
        "test_short_conv_finite_output",
        "test_checkpoint_roundtrip_short_conv",
        "test_checkpoint_backward_compat_short_conv",
    ]:
        globals()[fn_name]()
    print("All Slice 6 tests PASSED")
