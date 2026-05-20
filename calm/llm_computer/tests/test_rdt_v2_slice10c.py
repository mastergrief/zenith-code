"""Slice 10c regression tests: ACT greedy inference halt (Tier B).

Per co_lead audit msg 1779304303629: requires trained halt head from
S10b. Opt-in via `forward(idx, act_inference=True)` kwarg. Stop H/L
iteration when cumulative halt prob > `self.act_threshold` AND
executed at least `self.act_min_iters` cycles. `self.last_act_halt_step`
records the actual count.

GPU-only.
"""
from __future__ import annotations

import pytest
import torch

from calm.llm_computer.copy_augmented_delta import build_copy_augmented_delta


DEVICE = "cuda"
SEP_ID = 3

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Slice 10c tests are GPU-only per user direction; CUDA unavailable",
)


def _make_dt(use_halt_head: bool = True, h_cycles: int = 3, n_iter: int = 2,
             seed: int = 42):
    torch.manual_seed(seed)
    return build_copy_augmented_delta(
        vocab_size=20, d_model=8, n_heads=4, n_layers=2, d_ffn=16,
        max_len=24, n_copy_heads=2, sep_token_id=SEP_ID,
        use_chunkwise=False,
        n_iterations=n_iter, h_cycles=h_cycles,
        use_halt_head=use_halt_head,
    ).to(DEVICE)


# ===== Section A: act_inference=False default unchanged =====

def test_act_inference_default_false_bit_identical():
    """forward(idx) with no kwarg == forward(idx, act_inference=False).
    Both produce bit-identical output."""
    m = _make_dt(h_cycles=2, n_iter=2, seed=42)
    m.eval()
    idx = torch.tensor([[5, 7, SEP_ID, 9, 11, 13]], device=DEVICE)
    with torch.no_grad():
        a = m(idx)
        b = m(idx, act_inference=False)
    assert torch.equal(a, b)


def test_act_inference_false_runs_all_h_cycles():
    """At act_inference=False, all `h_cycles` H cycles execute regardless
    of halt prob. `last_act_halt_step` stays None (not set)."""
    m = _make_dt(h_cycles=3, n_iter=2, seed=42)
    m.eval()
    idx = torch.tensor([[5, 7, SEP_ID, 9, 11, 13]], device=DEVICE)
    with torch.no_grad():
        m(idx, act_inference=False)
    # last_halt_probs has all h_cycles entries (S10a telemetry)
    assert len(m.last_halt_probs) == 3
    # last_act_halt_step was never set (act_inference was off)
    assert m.last_act_halt_step is None


# ===== Section B: act_inference=True requires halt_head =====

def test_act_inference_true_without_halt_head_raises():
    """forward(act_inference=True) on a model built with
    use_halt_head=False raises ValueError citing the missing flag."""
    m = _make_dt(use_halt_head=False, h_cycles=2, n_iter=2, seed=42)
    m.eval()
    idx = torch.tensor([[5, 7, SEP_ID, 9, 11, 13]], device=DEVICE)
    with pytest.raises(ValueError, match=r"use_halt_head"):
        with torch.no_grad():
            m(idx, act_inference=True)


# ===== Section C: Early halt fires =====

def test_act_inference_halt_threshold_zero_breaks_at_first_eligible_step():
    """With act_threshold=0.0 and act_min_iters=1, the loop breaks at
    H cycle 1 (cumulative halt > 0 immediately on any positive halt
    prob). `last_act_halt_step == 1`."""
    m = _make_dt(h_cycles=3, n_iter=2, seed=42)
    m.eval()
    m.act_threshold = 0.0
    m.act_min_iters = 1
    idx = torch.tensor([[5, 7, SEP_ID, 9, 11, 13]], device=DEVICE)
    with torch.no_grad():
        m(idx, act_inference=True)
    assert m.last_act_halt_step == 1
    # Halt probs collected up through the halt cycle (not all 3)
    assert len(m.last_halt_probs) == 1


def test_act_inference_min_iters_respected():
    """Even with threshold=0.0, act_min_iters=2 forces at least 2 cycles
    to execute before halt can fire."""
    m = _make_dt(h_cycles=5, n_iter=2, seed=42)
    m.eval()
    m.act_threshold = 0.0
    m.act_min_iters = 2
    idx = torch.tensor([[5, 7, SEP_ID, 9, 11, 13]], device=DEVICE)
    with torch.no_grad():
        m(idx, act_inference=True)
    assert m.last_act_halt_step == 2  # min_iters floor
    assert len(m.last_halt_probs) == 2


def test_act_inference_high_threshold_runs_all_cycles():
    """When threshold is unreachable (> h_cycles since each halt prob
    is in [0, 1]), the loop runs all h_cycles. `last_act_halt_step == h_cycles`."""
    m = _make_dt(h_cycles=3, n_iter=2, seed=42)
    m.eval()
    m.act_threshold = 100.0  # never crossable
    m.act_min_iters = 1
    idx = torch.tensor([[5, 7, SEP_ID, 9, 11, 13]], device=DEVICE)
    with torch.no_grad():
        m(idx, act_inference=True)
    assert m.last_act_halt_step == 3
    assert len(m.last_halt_probs) == 3


# ===== Section D: h_cycles=1 degenerate =====

def test_act_inference_h_cycles_1_runs_single_cycle():
    """At h_cycles=1, only one cycle exists. last_act_halt_step==1
    regardless of threshold/min_iters."""
    m = _make_dt(h_cycles=1, n_iter=2, seed=42)
    m.eval()
    idx = torch.tensor([[5, 7, SEP_ID, 9, 11, 13]], device=DEVICE)
    with torch.no_grad():
        m(idx, act_inference=True)
    assert m.last_act_halt_step == 1


# ===== Section E: act_inference + return_per_iter compose =====

def test_act_inference_with_return_per_iter():
    """When BOTH flags on, per_iter list length matches actually-
    executed cycles (NOT the full configured h_cycles)."""
    m = _make_dt(h_cycles=5, n_iter=2, seed=42)
    m.eval()
    m.act_threshold = 0.0
    m.act_min_iters = 1
    idx = torch.tensor([[5, 7, SEP_ID, 9, 11, 13]], device=DEVICE)
    with torch.no_grad():
        final, per_iter = m(idx, return_per_iter=True, act_inference=True)
    assert m.last_act_halt_step == 1
    assert len(per_iter) == 1  # only 1 cycle actually ran


# ===== Section F: act_inference doesn't affect training (no kwarg) =====

def test_act_inference_off_during_training():
    """During training (no act_inference kwarg → default False), the
    full h_cycles run regardless of halt-head behavior. Confirms ACT
    inference doesn't accidentally fire during gradient-flowing
    forwards used in training loss."""
    m = _make_dt(h_cycles=3, n_iter=2, seed=42)
    # NOT eval mode — simulating training forward
    idx = torch.tensor([[5, 7, SEP_ID, 9, 11, 13]], device=DEVICE)
    out = m(idx)
    # No early halt
    assert m.last_act_halt_step is None
    assert len(m.last_halt_probs) == 3
    # Forward still produces grad-flowing output
    assert out.requires_grad


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA unavailable, skipping (GPU-only)")
        raise SystemExit(0)
    for fn_name in [
        "test_act_inference_default_false_bit_identical",
        "test_act_inference_false_runs_all_h_cycles",
        "test_act_inference_true_without_halt_head_raises",
        "test_act_inference_halt_threshold_zero_breaks_at_first_eligible_step",
        "test_act_inference_min_iters_respected",
        "test_act_inference_high_threshold_runs_all_cycles",
        "test_act_inference_h_cycles_1_runs_single_cycle",
        "test_act_inference_with_return_per_iter",
        "test_act_inference_off_during_training",
    ]:
        globals()[fn_name]()
    print("All Slice 10c tests PASSED")
