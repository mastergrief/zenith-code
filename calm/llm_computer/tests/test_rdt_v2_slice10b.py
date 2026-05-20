"""Slice 10b regression tests: ACT training contract (Tier B).

Adds `compute_act_loss(per_iter_log_probs, halt_probs, target_ids,
loss_mask, ponder_weight)` helper. Combines per-iter accuracy loss
weighted by halt distribution + ponder cost.

Per co_lead audit msg 1779304303629: TRAINING-SIDE ONLY. No inference
behavior change. Greedy inference halt (using the now-trained head)
comes in S10c.

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
    reason="Slice 10b tests are GPU-only per user direction; CUDA unavailable",
)


def _make_dt(h_cycles: int = 2, n_iter: int = 2, seed: int = 42):
    torch.manual_seed(seed)
    return build_copy_augmented_delta(
        vocab_size=20, d_model=8, n_heads=4, n_layers=2, d_ffn=16,
        max_len=24, n_copy_heads=2, sep_token_id=SEP_ID,
        use_chunkwise=False,
        n_iterations=n_iter, h_cycles=h_cycles,
        use_halt_head=True,  # required for halt_probs
    ).to(DEVICE)


# ===== Section A: API contract =====

def test_compute_act_loss_returns_scalar():
    """`compute_act_loss` returns a 0-d tensor (scalar) suitable for
    `.backward()`."""
    m = _make_dt(h_cycles=2, n_iter=2, seed=42)
    idx = torch.tensor([[5, 7, SEP_ID, 9, 11, 13]], device=DEVICE)
    target = torch.tensor([[7, 3, 9, 11, 13, 0]], device=DEVICE)
    _, per_iter_lp = m(idx, return_per_iter=True)
    loss = m.compute_act_loss(per_iter_lp, m.last_halt_probs, target)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_compute_act_loss_with_mask():
    """`loss_mask` restricts the per-iter NLL to masked positions."""
    m = _make_dt(h_cycles=2, n_iter=2, seed=42)
    idx = torch.tensor([[5, 7, SEP_ID, 9, 11, 13]], device=DEVICE)
    target = torch.tensor([[7, 3, 9, 11, 13, 0]], device=DEVICE)
    mask = torch.tensor([[False, False, False, True, True, True]], device=DEVICE)
    _, per_iter_lp = m(idx, return_per_iter=True)
    loss = m.compute_act_loss(per_iter_lp, m.last_halt_probs, target, loss_mask=mask)
    assert torch.isfinite(loss)


def test_compute_act_loss_rejects_length_mismatch():
    """per_iter_log_probs and halt_probs must align in length."""
    m = _make_dt(h_cycles=2, n_iter=2, seed=42)
    idx = torch.tensor([[5, 7, SEP_ID, 9, 11, 13]], device=DEVICE)
    target = torch.tensor([[7, 3, 9, 11, 13, 0]], device=DEVICE)
    _, per_iter_lp = m(idx, return_per_iter=True)
    # Trim halt_probs to a different length
    bad_halt = m.last_halt_probs[:1]
    with pytest.raises(ValueError, match=r"must align"):
        m.compute_act_loss(per_iter_lp, bad_halt, target)


# ===== Section B: Halt-distribution properties =====

def test_act_loss_halt_distribution_sums_to_one_property():
    """The halt distribution is normalized internally → as ponder_weight=0
    and all per-iter NLL are equal, the weighted loss equals that NLL
    (sanity: weighted average over a probability distribution)."""
    m = _make_dt(h_cycles=3, n_iter=2, seed=42)
    m.eval()
    idx = torch.tensor([[5, 7, SEP_ID, 9, 11, 13]], device=DEVICE)
    target = torch.tensor([[7, 3, 9, 11, 13, 0]], device=DEVICE)
    with torch.no_grad():
        _, per_iter_lp = m(idx, return_per_iter=True)

        # Force all per-iter NLL to be equal by replacing them with copies
        # of the first one — then weighted loss must equal NLL_0.
        first_lp = per_iter_lp[0]
        per_iter_lp_uniform = [first_lp] * len(per_iter_lp)

        loss = m.compute_act_loss(
            per_iter_lp_uniform, m.last_halt_probs, target,
            ponder_weight=0.0,
        )
        # Direct NLL of first_lp (no halt weighting)
        tok_lp = first_lp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        direct_nll = -tok_lp.mean()
    assert torch.allclose(loss, direct_nll, atol=1e-5, rtol=1e-5)


def test_act_loss_ponder_term_positive():
    """Ponder cost is `Σ_i (i+1) * p_halt_i` — always positive when
    halt_dist sums to 1. With ponder_weight > 0, loss STRICTLY exceeds
    the pure-weighted-NLL version."""
    m = _make_dt(h_cycles=3, n_iter=2, seed=42)
    m.eval()
    idx = torch.tensor([[5, 7, SEP_ID, 9, 11, 13]], device=DEVICE)
    target = torch.tensor([[7, 3, 9, 11, 13, 0]], device=DEVICE)
    with torch.no_grad():
        _, per_iter_lp = m(idx, return_per_iter=True)
        loss_no_ponder = m.compute_act_loss(
            per_iter_lp, m.last_halt_probs, target, ponder_weight=0.0,
        )
        loss_with_ponder = m.compute_act_loss(
            per_iter_lp, m.last_halt_probs, target, ponder_weight=0.1,
        )
    assert loss_with_ponder.item() > loss_no_ponder.item()


# ===== Section C: Gradient flow through halt head AND backbone =====

def test_act_loss_gradient_flows_to_halt_head():
    """ACT loss backward populates halt_head.weight.grad — confirms the
    training contract actually trains the halt head."""
    m = _make_dt(h_cycles=2, n_iter=2, seed=42)
    idx = torch.tensor([[5, 7, SEP_ID, 9, 11, 13]], device=DEVICE)
    target = torch.tensor([[7, 3, 9, 11, 13, 0]], device=DEVICE)
    _, per_iter_lp = m(idx, return_per_iter=True)
    loss = m.compute_act_loss(per_iter_lp, m.last_halt_probs, target)
    loss.backward()
    g = m.halt_head.weight.grad
    assert g is not None
    assert torch.isfinite(g).all()
    assert g.abs().max().item() > 0


def test_act_loss_gradient_flows_to_backbone():
    """ACT loss also propagates gradient to W_qkv (backbone weights)
    via the per-iter log_probs path. Confirms it's not just a halt-head
    training signal — accuracy still matters."""
    m = _make_dt(h_cycles=2, n_iter=2, seed=42)
    idx = torch.tensor([[5, 7, SEP_ID, 9, 11, 13]], device=DEVICE)
    target = torch.tensor([[7, 3, 9, 11, 13, 0]], device=DEVICE)
    _, per_iter_lp = m(idx, return_per_iter=True)
    loss = m.compute_act_loss(per_iter_lp, m.last_halt_probs, target)
    loss.backward()
    g = m.W_qkv[0].weight.grad
    assert g is not None
    assert torch.isfinite(g).all()
    assert g.abs().max().item() > 0


# ===== Section D: Inference behavior UNCHANGED =====

def test_compute_act_loss_does_not_change_forward():
    """Calling compute_act_loss after a forward must not perturb subsequent
    forward outputs. ACT is training-side only."""
    m = _make_dt(h_cycles=2, n_iter=2, seed=42)
    m.eval()
    idx = torch.tensor([[5, 7, SEP_ID, 9, 11, 13]], device=DEVICE)
    target = torch.tensor([[7, 3, 9, 11, 13, 0]], device=DEVICE)

    with torch.no_grad():
        out_before = m(idx)
    # Now compute ACT loss in a non-grad context — just exercises the helper
    with torch.no_grad():
        _, per_iter_lp = m(idx, return_per_iter=True)
        _ = m.compute_act_loss(per_iter_lp, m.last_halt_probs, target)
    with torch.no_grad():
        out_after = m(idx)
    assert torch.equal(out_before, out_after)


# ===== Section E: Edge cases =====

def test_act_loss_h_cycles_1_degenerate():
    """At h_cycles=1, per_iter list has length 1 and halt distribution
    collapses to [1.0] (remainder folded into last iter). Weighted loss
    equals NLL at the single iter."""
    m = _make_dt(h_cycles=1, n_iter=2, seed=42)
    m.eval()
    idx = torch.tensor([[5, 7, SEP_ID, 9, 11, 13]], device=DEVICE)
    target = torch.tensor([[7, 3, 9, 11, 13, 0]], device=DEVICE)
    with torch.no_grad():
        final, per_iter_lp = m(idx, return_per_iter=True)
        loss = m.compute_act_loss(
            per_iter_lp, m.last_halt_probs, target, ponder_weight=0.0,
        )
        # Direct NLL of the only per-iter entry (== final)
        tok_lp = final.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        direct_nll = -tok_lp.mean()
    assert torch.allclose(loss, direct_nll, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA unavailable, skipping (GPU-only)")
        raise SystemExit(0)
    for fn_name in [
        "test_compute_act_loss_returns_scalar",
        "test_compute_act_loss_with_mask",
        "test_compute_act_loss_rejects_length_mismatch",
        "test_act_loss_halt_distribution_sums_to_one_property",
        "test_act_loss_ponder_term_positive",
        "test_act_loss_gradient_flows_to_halt_head",
        "test_act_loss_gradient_flows_to_backbone",
        "test_compute_act_loss_does_not_change_forward",
        "test_act_loss_h_cycles_1_degenerate",
    ]:
        globals()[fn_name]()
    print("All Slice 10b tests PASSED")
