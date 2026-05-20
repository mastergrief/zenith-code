"""Slice 10a regression tests: ACT halt-head telemetry (Tier B).

Per co_lead audit msg 1779304303629: telemetry ONLY, NO behavior change.
Forward output is bit-identical to flag-off; the halt head reads z_H
and produces per-H-cycle probabilities into `self.last_halt_probs`
without participating in any decision. Untrained head can't safely
drive inference yet — that's deferred to S10c.

GPU-only.
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
    reason="Slice 10a tests are GPU-only per user direction; CUDA unavailable",
)


def _make_dt(use_halt_head: bool = False, h_cycles: int = 2, n_iter: int = 2,
             seed: int = 42):
    torch.manual_seed(seed)
    return build_copy_augmented_delta(
        vocab_size=20, d_model=8, n_heads=4, n_layers=2, d_ffn=16,
        max_len=24, n_copy_heads=2, sep_token_id=SEP_ID,
        use_chunkwise=False,
        n_iterations=n_iter, h_cycles=h_cycles,
        use_halt_head=use_halt_head,
    ).to(DEVICE)


# ===== Section A: Allocation =====

def test_halt_head_off_module_is_none():
    """Flag off → halt_head is None, last_halt_probs is empty list."""
    m = _make_dt(use_halt_head=False, seed=42)
    assert m.halt_head is None
    assert m.last_halt_probs == []


def test_halt_head_on_module_is_linear():
    """Flag on → halt_head is nn.Linear(d_model, 1, bias=True)."""
    m = _make_dt(use_halt_head=True, seed=42)
    assert isinstance(m.halt_head, nn.Linear)
    assert m.halt_head.in_features == 8
    assert m.halt_head.out_features == 1
    # Bias initialized to 0 (so sigmoid(0)=0.5 at init — neutral halt prob)
    assert m.halt_head.bias.item() == pytest.approx(0.0)


def test_halt_head_param_count_increase():
    """+d_model weight + 1 bias = 9 params at d_model=8."""
    m_off = _make_dt(use_halt_head=False, seed=42)
    m_on = _make_dt(use_halt_head=True, seed=42)
    diff = m_on.param_count() - m_off.param_count()
    assert diff == 9  # d_model (8) + bias (1)


# ===== Section B: TELEMETRY ONLY — forward output bit-identical =====

def test_halt_head_forward_output_unchanged_at_h_cycles_1():
    """At h_cycles=1, flag-on vs flag-off MUST produce bit-identical
    forward output. The halt head reads x but doesn't write back.
    RNG isolation around halt_head allocation preserves downstream
    param consistency."""
    m_off = _make_dt(use_halt_head=False, h_cycles=1, n_iter=2, seed=42)
    m_on = _make_dt(use_halt_head=True, h_cycles=1, n_iter=2, seed=42)
    m_off.eval(); m_on.eval()
    idx = torch.tensor([[5, 7, SEP_ID, 9, 11, 13]], device=DEVICE)
    with torch.no_grad():
        a = m_off(idx)
        b = m_on(idx)
    assert torch.equal(a, b), (
        "halt_head telemetry must not perturb forward output — RNG "
        "isolation regressed or telemetry leaked into computation."
    )


def test_halt_head_forward_output_unchanged_at_h_cycles_2():
    """Same invariant at h_cycles=2 (the path that actually invokes the
    H/L loop)."""
    m_off = _make_dt(use_halt_head=False, h_cycles=2, n_iter=2, seed=42)
    m_on = _make_dt(use_halt_head=True, h_cycles=2, n_iter=2, seed=42)
    m_off.eval(); m_on.eval()
    idx = torch.tensor([[5, 7, SEP_ID, 9, 11, 13]], device=DEVICE)
    with torch.no_grad():
        a = m_off(idx)
        b = m_on(idx)
    assert torch.equal(a, b)


# ===== Section C: Halt-prob collection contract =====

def test_halt_probs_length_matches_h_cycles():
    """last_halt_probs has exactly h_cycles entries (both h=1 and h>1
    code paths)."""
    idx = torch.tensor([[5, 7, SEP_ID, 9, 11, 13]], device=DEVICE)
    for h in (1, 2, 3):
        m = _make_dt(use_halt_head=True, h_cycles=h, n_iter=2, seed=42)
        m.eval()
        with torch.no_grad():
            m(idx)
        assert len(m.last_halt_probs) == h, (
            f"expected {h} halt probs at h_cycles={h}, got {len(m.last_halt_probs)}"
        )


def test_halt_probs_in_unit_interval():
    """Each halt prob = sigmoid(halt_logit) → must be in [0, 1]. At toy
    d_model=8 with deep H cycles, fp32 underflow can push sigmoid to
    EXACTLY 0 or 1 — that's still mathematically valid, so use closed
    interval not open."""
    m = _make_dt(use_halt_head=True, h_cycles=3, n_iter=2, seed=42)
    m.eval()
    idx = torch.tensor([[5, 7, SEP_ID, 9, 11, 13]], device=DEVICE)
    with torch.no_grad():
        m(idx)
    for p in m.last_halt_probs:
        assert (p >= 0).all() and (p <= 1).all()
        assert torch.isfinite(p).all()


def test_halt_probs_neutral_at_init_h1():
    """With bias=0 init and small-random halt-head weights AT h_cycles=1
    (where z_H is just the freshly-built residual, magnitudes haven't
    compounded across H cycles yet), halt prob is close to 0.5 (neutral).
    Catches a future bias-init regression."""
    m = _make_dt(use_halt_head=True, h_cycles=1, n_iter=1, seed=42)
    m.eval()
    idx = torch.tensor([[5, 7, SEP_ID, 9, 11, 13]], device=DEVICE)
    with torch.no_grad():
        m(idx)
    # halt_logit = w^T z + 0, w small → halt_prob ≈ sigmoid(small) ≈ 0.5
    p = m.last_halt_probs[0]
    assert 0.2 < p.mean().item() < 0.8


def test_halt_probs_reset_each_forward():
    """The list is reset at start of each forward — calling forward twice
    leaves length == h_cycles, not 2 * h_cycles."""
    m = _make_dt(use_halt_head=True, h_cycles=2, n_iter=2, seed=42)
    m.eval()
    idx = torch.tensor([[5, 7, SEP_ID, 9, 11, 13]], device=DEVICE)
    with torch.no_grad():
        m(idx)
        m(idx)
    assert len(m.last_halt_probs) == 2  # not 4


# ===== Section D: Gradient flow through halt head =====

def test_halt_head_weight_gradient_flows():
    """halt_head.weight receives gradient when something differentiable
    references the halt probs (foreshadowing S10b training loss). This
    proves the gradient path is wired correctly."""
    m = _make_dt(use_halt_head=True, h_cycles=2, n_iter=2, seed=42)
    idx = torch.tensor([[5, 7, SEP_ID, 9, 11, 13]], device=DEVICE)
    _ = m(idx)
    # Simulate a halt-loss term (sum of halt probs)
    halt_loss = sum(p.sum() for p in m.last_halt_probs)
    halt_loss.backward()
    g = m.halt_head.weight.grad
    assert g is not None
    assert torch.isfinite(g).all()
    assert g.abs().max().item() > 0


# ===== Section E: Cached-decode blocklist =====

def test_cached_decode_blocks_halt_head():
    """`use_halt_head` is added to the blocklist defensively per
    co_lead — at S10a it doesn't fork behavior, but S10c will."""
    m = _make_dt(use_halt_head=True, h_cycles=1, n_iter=1, seed=42)
    m.eval()
    prefix = torch.tensor([[5, 7, SEP_ID, 9, 11]], device=DEVICE)
    with pytest.raises(NotImplementedError, match=r"use_halt_head"):
        m.decode_greedy_cached(prefix, max_gen=2, eos_token=None)


# ===== Section F: Checkpoint round-trip =====

def test_checkpoint_roundtrip_halt_head():
    """use_halt_head persists; halt_head module reloads; logits match."""
    m_orig = _make_dt(use_halt_head=True, h_cycles=2, n_iter=2, seed=42)
    m_orig.eval()
    idx = torch.tensor([[5, 7, SEP_ID, 9, 11, 13]], device=DEVICE)
    with torch.no_grad():
        out_orig = m_orig(idx)

    cfg_dict = {
        "vocab_size": 20, "max_len": 24,
        "d_model": 8, "n_heads": 4, "n_layers": 2,
        "d_ffn": 16, "n_copy_heads": 2,
        "use_chunkwise": False,
        "n_iterations": 2,
        "h_cycles": 2,
        "use_halt_head": True,
    }
    ckpt = {
        "model_state": m_orig.state_dict(),
        "config": cfg_dict,
        "epoch": 1, "val_autoreg": 0.5, "n_train": 100, "n_val": 10,
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_slice10a.pt"
        torch.save(ckpt, path)
        m_loaded, _ = load_dt_checkpoint(str(path), device=DEVICE)
    m_loaded.eval()
    assert m_loaded.config.use_halt_head is True
    assert isinstance(m_loaded.halt_head, nn.Linear)

    with torch.no_grad():
        out_loaded = m_loaded(idx)
    assert torch.equal(out_orig, out_loaded)


def test_checkpoint_backward_compat_halt_head():
    """Old checkpoint without use_halt_head key loads with flag=False.
    Uses h_cycles=1, n_iter=1 so the cfg_dict's omission of those keys
    matches the original model's defaults (otherwise the loaded model
    would build with h_cycles=1 while the original has h_cycles=2,
    diverging the forward computation independently of halt_head)."""
    m_orig = _make_dt(use_halt_head=False, h_cycles=1, n_iter=1, seed=42)
    m_orig.eval()
    idx = torch.tensor([[5, 7, SEP_ID, 9, 11, 13]], device=DEVICE)
    with torch.no_grad():
        out_orig = m_orig(idx)

    cfg_dict = {
        "vocab_size": 20, "max_len": 24,
        "d_model": 8, "n_heads": 4, "n_layers": 2,
        "d_ffn": 16, "n_copy_heads": 2,
        "use_chunkwise": False,
        # NO use_halt_head key, NO h_cycles, NO n_iterations
        # (all default to 1 in DeltaNetConfig)
    }
    ckpt = {
        "model_state": m_orig.state_dict(),
        "config": cfg_dict,
        "epoch": 1, "val_autoreg": 0.5, "n_train": 100, "n_val": 10,
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_slice10a_compat.pt"
        torch.save(ckpt, path)
        m_loaded, _ = load_dt_checkpoint(str(path), device=DEVICE)
    m_loaded.eval()
    assert m_loaded.config.use_halt_head is False
    assert m_loaded.halt_head is None

    with torch.no_grad():
        out_loaded = m_loaded(idx)
    assert torch.equal(out_orig, out_loaded)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA unavailable, skipping (GPU-only)")
        raise SystemExit(0)
    for fn_name in [
        "test_halt_head_off_module_is_none",
        "test_halt_head_on_module_is_linear",
        "test_halt_head_param_count_increase",
        "test_halt_head_forward_output_unchanged_at_h_cycles_1",
        "test_halt_head_forward_output_unchanged_at_h_cycles_2",
        "test_halt_probs_length_matches_h_cycles",
        "test_halt_probs_in_unit_interval",
        "test_halt_probs_neutral_at_init_h1",
        "test_halt_probs_reset_each_forward",
        "test_halt_head_weight_gradient_flows",
        "test_cached_decode_blocks_halt_head",
        "test_checkpoint_roundtrip_halt_head",
        "test_checkpoint_backward_compat_halt_head",
    ]:
        globals()[fn_name]()
    print("All Slice 10a tests PASSED")
