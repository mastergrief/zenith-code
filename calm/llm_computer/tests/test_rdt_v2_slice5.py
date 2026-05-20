"""Slice 5 regression tests: H-boundary RMSNorm + cached-decode guard.

Closes the LayerNorm gap that Slice 4 explicitly punted. RMSNorm at the
H-cycle hand-off (`z_H = h_norm(z_L)`) stabilizes magnitude in the
`h_cycles × n_iters × use_input_injection` regime that previously
NaN'd on toy d_model=8.

Co-lead audit msg 1779303378368 added the cached-decode guard: a hard
`NotImplementedError` at `decode_greedy_cached` entry for any non-flat
control-flow flag, because the cached path is flat-layer-pass and would
silently ignore Slice 1-5 architecture at facade-install inference.

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
    reason="Slice 5 tests are GPU-only per user direction; CUDA unavailable",
)


def _make_dt(use_h_rmsnorm: bool = False, h_cycles: int = 1, n_iter: int = 1,
             use_inject: bool = False, use_z: bool = False,
             use_softmax: bool = False, use_prefix: bool = False,
             use_loop_idx: bool = False, seed: int = 42):
    torch.manual_seed(seed)
    return build_copy_augmented_delta(
        vocab_size=20, d_model=8, n_heads=4, n_layers=2, d_ffn=16,
        max_len=24, n_copy_heads=2, sep_token_id=SEP_ID,
        use_chunkwise=False,
        n_iterations=n_iter,
        h_cycles=h_cycles,
        use_input_injection=use_inject,
        use_z_init=use_z,
        use_softmax_attn=use_softmax,
        use_prefix_lm=use_prefix,
        use_loop_index=use_loop_idx,
        use_h_rmsnorm=use_h_rmsnorm,
    ).to(DEVICE)


# ===== Section A: Flag-off bit-equivalence (load-bearing baseline guard) =====

def test_h_rmsnorm_off_h_norm_is_none():
    """Flag off → no h_norm module allocated. Confirms zero state-dict
    drift for Slice 1-4 checkpoints."""
    m = _make_dt(use_h_rmsnorm=False, seed=42)
    assert m.h_norm is None


def test_h_rmsnorm_on_h_norm_is_rmsnorm():
    """Flag on → h_norm is a torch.nn.RMSNorm with the right shape."""
    m = _make_dt(use_h_rmsnorm=True, seed=42)
    assert isinstance(m.h_norm, nn.RMSNorm)
    # RMSNorm.weight is the per-channel learnable scale; shape == (d_model,)
    assert m.h_norm.weight.shape == (8,)


def test_h_rmsnorm_h_cycles_1_inert():
    """At h_cycles=1, the H-boundary normalization NEVER fires (early
    return branch in _forward_backbone). Flag on vs off must produce
    identical output."""
    m_off = _make_dt(use_h_rmsnorm=False, h_cycles=1, n_iter=2, seed=42)
    m_on = _make_dt(use_h_rmsnorm=True, h_cycles=1, n_iter=2, seed=42)
    m_off.eval(); m_on.eval()
    idx = torch.tensor([[5, 7, SEP_ID, 9, 11, 13]], device=DEVICE)
    with torch.no_grad():
        a = m_off(idx)
        b = m_on(idx)
    # h_cycles=1 takes the early-return branch in _forward_backbone, which
    # never touches self.h_norm. Flag-on/off output is bit-identical at h=1.
    assert torch.equal(a, b)


# ===== Section B: Flag-on enables the previously-NaN regime =====

def test_h_rmsnorm_unlocks_previously_nan_regime():
    """The Slice 4 finding: at h=2, n=3, inject=True, z=True the
    h_rmsnorm-off path produces NaN. With Slice 5 RMSNorm on, the same
    config produces finite output. This is the target win."""
    # Confirm Slice 4 behavior: flag off → NaN somewhere
    m_off = _make_dt(use_h_rmsnorm=False, h_cycles=2, n_iter=3,
                     use_inject=True, use_z=True, seed=42)
    m_off.eval()
    idx = torch.tensor([[5, 7, SEP_ID, 9, 11, 13]], device=DEVICE)
    with torch.no_grad():
        out_off = m_off(idx)
    # We expect at least one NaN in the unstabilized regime
    assert not torch.isfinite(out_off).all(), (
        "Slice 5 baseline assumption broken: h_cycles>1 + inject + z_init "
        "is supposed to produce NaN without RMSNorm. If this passes, the "
        "test is no longer guarding the right invariant."
    )

    # With RMSNorm on, output must be finite at the same config
    m_on = _make_dt(use_h_rmsnorm=True, h_cycles=2, n_iter=3,
                    use_inject=True, use_z=True, seed=42)
    m_on.eval()
    with torch.no_grad():
        out_on = m_on(idx)
    assert torch.isfinite(out_on).all(), (
        "Slice 5 RMSNorm failed to stabilize the previously-NaN regime."
    )


def test_h_rmsnorm_changes_output_at_h_gt_1():
    """At h_cycles > 1, flag on produces different output than flag off
    (RMSNorm rescales the H-cycle carry — output meaningfully different)."""
    m_off = _make_dt(use_h_rmsnorm=False, h_cycles=2, n_iter=2, seed=42)
    m_on = _make_dt(use_h_rmsnorm=True, h_cycles=2, n_iter=2, seed=42)
    m_off.eval(); m_on.eval()
    idx = torch.tensor([[5, 7, SEP_ID, 9, 11, 13]], device=DEVICE)
    with torch.no_grad():
        a = m_off(idx)
        b = m_on(idx)
    assert not torch.allclose(a, b)


# ===== Section C: Cached-decode guard (codex's explicit ask) =====

def test_cached_decode_guard_fires_on_h_rmsnorm():
    """Codex msg 1779303378368: cached decode must raise NotImplementedError
    when `h_cycles>1 AND use_h_rmsnorm=True`. The full-forward path
    honors RMSNorm via _forward_backbone; the cached path does NOT."""
    m = _make_dt(use_h_rmsnorm=True, h_cycles=2, n_iter=2, seed=42)
    m.eval()
    prefix = torch.tensor([[5, 7, SEP_ID, 9, 11]], device=DEVICE)
    with pytest.raises(NotImplementedError, match=r"(h_cycles|use_h_rmsnorm)"):
        m.decode_greedy_cached(prefix, max_gen=2, eos_token=None)


def test_cached_decode_guard_fires_on_h_cycles_alone():
    """h_cycles>1 alone (even without h_rmsnorm) blocks cached decode
    because the flat path doesn't run the outer H loop."""
    m = _make_dt(use_h_rmsnorm=False, h_cycles=2, n_iter=1, seed=42)
    m.eval()
    prefix = torch.tensor([[5, 7, SEP_ID, 9, 11]], device=DEVICE)
    with pytest.raises(NotImplementedError, match=r"h_cycles"):
        m.decode_greedy_cached(prefix, max_gen=2, eos_token=None)


def test_cached_decode_guard_fires_on_n_iter_gt_1():
    """n_iterations > 1 blocks cached decode (flat path doesn't run L loop)."""
    m = _make_dt(use_h_rmsnorm=False, h_cycles=1, n_iter=3, seed=42)
    m.eval()
    prefix = torch.tensor([[5, 7, SEP_ID, 9, 11]], device=DEVICE)
    with pytest.raises(NotImplementedError, match=r"n_iterations"):
        m.decode_greedy_cached(prefix, max_gen=2, eos_token=None)


def test_cached_decode_guard_blocks_z_init_and_inject():
    """All Slice 1-5 control-flow flags are blocked except use_gated_attention."""
    # use_z_init alone (with h=1, n=1) — gate is inert at runtime but guard
    # blocks anyway since cached path can't honor the swap if ever exercised.
    m = _make_dt(use_h_rmsnorm=False, h_cycles=1, n_iter=1,
                 use_z=True, seed=42)
    m.eval()
    prefix = torch.tensor([[5, 7, SEP_ID, 9, 11]], device=DEVICE)
    with pytest.raises(NotImplementedError, match=r"use_z_init"):
        m.decode_greedy_cached(prefix, max_gen=2, eos_token=None)


def test_cached_decode_guard_blocks_prefix_lm():
    m = _make_dt(use_h_rmsnorm=False, h_cycles=1, n_iter=1,
                 use_prefix=True, seed=42)
    m.eval()
    prefix = torch.tensor([[5, 7, SEP_ID, 9, 11]], device=DEVICE)
    with pytest.raises(NotImplementedError, match=r"use_prefix_lm"):
        m.decode_greedy_cached(prefix, max_gen=2, eos_token=None)


def test_cached_decode_guard_blocks_softmax_attn():
    m = _make_dt(use_h_rmsnorm=False, h_cycles=1, n_iter=1,
                 use_softmax=True, seed=42)
    m.eval()
    prefix = torch.tensor([[5, 7, SEP_ID, 9, 11]], device=DEVICE)
    with pytest.raises(NotImplementedError, match=r"use_softmax_attn"):
        m.decode_greedy_cached(prefix, max_gen=2, eos_token=None)


def test_cached_decode_guard_blocks_loop_index():
    m = _make_dt(use_h_rmsnorm=False, h_cycles=1, n_iter=1,
                 use_loop_idx=True, seed=42)
    m.eval()
    prefix = torch.tensor([[5, 7, SEP_ID, 9, 11]], device=DEVICE)
    with pytest.raises(NotImplementedError, match=r"use_loop_index"):
        m.decode_greedy_cached(prefix, max_gen=2, eos_token=None)


def test_cached_decode_allows_gated_attention():
    """use_gated_attention is mirrored in the cached path (Slice 1) → ALLOWED.
    Cached decode runs without raising for a gated-only config."""
    # Build a config with only use_gated_attention=True (no other flags)
    torch.manual_seed(42)
    m = build_copy_augmented_delta(
        vocab_size=20, d_model=8, n_heads=4, n_layers=2, d_ffn=16,
        max_len=24, n_copy_heads=2, sep_token_id=SEP_ID,
        use_chunkwise=False,
        use_gated_attention=True,  # The one allowlist case
    ).to(DEVICE)
    m.eval()
    prefix = torch.tensor([[5, 7, SEP_ID, 9, 11]], device=DEVICE)
    # Must NOT raise
    out = m.decode_greedy_cached(prefix, max_gen=2, eos_token=None)
    assert out.shape[0] == 1
    assert out.shape[1] <= 2


def test_cached_decode_allows_flat_default():
    """Pure-default config (all flags off) runs cached decode unaffected."""
    m = _make_dt(seed=42)
    m.eval()
    prefix = torch.tensor([[5, 7, SEP_ID, 9, 11]], device=DEVICE)
    out = m.decode_greedy_cached(prefix, max_gen=2, eos_token=None)
    assert out.shape[0] == 1


# ===== Section D: Gradient flow through RMSNorm scale =====

def test_h_rmsnorm_scale_gradient_flows():
    """RMSNorm.weight (per-channel scale) must receive gradient when the
    norm fires (h_cycles > 1, flag on)."""
    m = _make_dt(use_h_rmsnorm=True, h_cycles=2, n_iter=2, seed=42)
    idx = torch.tensor([[5, 7, SEP_ID, 9, 11, 13]], device=DEVICE)
    out = m(idx)
    loss = out.sum()
    loss.backward()
    g = m.h_norm.weight.grad
    assert g is not None
    assert torch.isfinite(g).all()
    assert g.abs().max().item() > 0


# ===== Section E: Checkpoint round-trip =====

def test_checkpoint_roundtrip_h_rmsnorm():
    """use_h_rmsnorm persists through save+load, h_norm module re-allocated,
    logits match."""
    m_orig = _make_dt(use_h_rmsnorm=True, h_cycles=2, n_iter=2, seed=42)
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
        "n_iterations": 2,
        "h_cycles": 2,
        "use_h_rmsnorm": True,
    }
    ckpt = {
        "model_state": m_orig.state_dict(),
        "config": cfg_dict,
        "epoch": 1, "val_autoreg": 0.5, "n_train": 100, "n_val": 10,
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_slice5.pt"
        torch.save(ckpt, path)
        m_loaded, _ = load_dt_checkpoint(str(path), device=DEVICE)
    m_loaded.eval()

    assert m_loaded.config.use_h_rmsnorm is True
    assert isinstance(m_loaded.h_norm, nn.RMSNorm)

    with torch.no_grad():
        out_loaded = m_loaded(idx)
    assert torch.equal(out_orig, out_loaded)


def test_checkpoint_backward_compat_h_rmsnorm():
    """Old checkpoint without use_h_rmsnorm key loads with flag=False."""
    m_orig = _make_dt(seed=42)
    m_orig.eval()
    idx = torch.tensor([[5, 7, SEP_ID, 9, 11, 13]], device=DEVICE)
    with torch.no_grad():
        out_orig = m_orig(idx)

    cfg_dict = {
        "vocab_size": 20, "max_len": 24,
        "d_model": 8, "n_heads": 4, "n_layers": 2,
        "d_ffn": 16, "n_copy_heads": 2,
        "use_chunkwise": False,
        # NO use_h_rmsnorm key
    }
    ckpt = {
        "model_state": m_orig.state_dict(),
        "config": cfg_dict,
        "epoch": 1, "val_autoreg": 0.5, "n_train": 100, "n_val": 10,
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_slice5_compat.pt"
        torch.save(ckpt, path)
        m_loaded, _ = load_dt_checkpoint(str(path), device=DEVICE)
    m_loaded.eval()
    assert m_loaded.config.use_h_rmsnorm is False
    assert m_loaded.h_norm is None

    with torch.no_grad():
        out_loaded = m_loaded(idx)
    assert torch.equal(out_orig, out_loaded)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA unavailable, skipping (GPU-only)")
        raise SystemExit(0)
    test_h_rmsnorm_off_h_norm_is_none()
    test_h_rmsnorm_on_h_norm_is_rmsnorm()
    test_h_rmsnorm_h_cycles_1_inert()
    test_h_rmsnorm_unlocks_previously_nan_regime()
    test_h_rmsnorm_changes_output_at_h_gt_1()
    test_cached_decode_guard_fires_on_h_rmsnorm()
    test_cached_decode_guard_fires_on_h_cycles_alone()
    test_cached_decode_guard_fires_on_n_iter_gt_1()
    test_cached_decode_guard_blocks_z_init_and_inject()
    test_cached_decode_guard_blocks_prefix_lm()
    test_cached_decode_guard_blocks_softmax_attn()
    test_cached_decode_guard_blocks_loop_index()
    test_cached_decode_allows_gated_attention()
    test_cached_decode_allows_flat_default()
    test_h_rmsnorm_scale_gradient_flows()
    test_checkpoint_roundtrip_h_rmsnorm()
    test_checkpoint_backward_compat_h_rmsnorm()
    print("All Slice 5 tests PASSED")
