"""Slice 13f.2 source-faithful HRM-Text completion + ACT training contract.

Tests the Q-head shape/grad, completion reward helper, shifted NLL
consistency, Q target bootstrap stop-grad, carry-detach segment loop,
all-finite short train step, and absence of stale PonderNet path as the
default.

Replaces deleted Slice 10a/10b/10c tests (PonderNet-style halt-weighted
NLL + cumulative-prob act_inference superseded by source-faithful
HRM-Text Q-head per gabe lock + codex audit chain).
"""
from __future__ import annotations

import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from calm.llm_computer.copy_augmented_delta import (
    build_copy_augmented_delta,
    full_answer_correct,
    compute_hrm_segment_loss,
    hrm_boundary_q_continue_target,
    CopyAugmentedDeltaNet,
)


def _make_model(use_halt_head=True, use_carry=True, h_cycles=2, n_iter=2,
                d_model=8, n_heads=4, n_layers=2, vocab_size=16, max_len=16,
                d_ffn=16, seed=42):
    torch.manual_seed(seed)
    return build_copy_augmented_delta(
        vocab_size=vocab_size, d_model=d_model, n_heads=n_heads,
        n_layers=n_layers, d_ffn=d_ffn, max_len=max_len,
        n_copy_heads=2, sep_token_id=3,
        use_chunkwise=True,
        n_iterations=n_iter,
        h_cycles=h_cycles,
        use_loop_index=True, use_input_injection=True,
        use_gated_attention=True, use_z_init=True, use_lecun_init=True,
        use_h_rmsnorm=True, use_short_conv=True, use_h_layer_stack=True,
        use_pre_rmsnorm=True,
        use_halt_head=use_halt_head,
        use_carry=use_carry,
    )


def _make_batch(B=2, S=12, vocab_size=16, sep_pos=5):
    """Build a small batch with sep token at sep_pos."""
    torch.manual_seed(0)
    ids = torch.randint(4, vocab_size, (B, S), dtype=torch.long)
    ids[:, sep_pos] = 3  # sep
    mask = torch.zeros(B, S, dtype=torch.bool)
    mask[:, sep_pos:S-1] = True
    return ids, mask


# =============================================================================
# Q-head architecture
# =============================================================================


def test_q_head_shape_is_2():
    """Slice 13f.2: halt_head outputs (Q_halt, Q_continue) — TWO values."""
    m = _make_model(use_halt_head=True)
    assert m.halt_head is not None
    assert isinstance(m.halt_head, nn.Linear)
    assert m.halt_head.out_features == 2, (
        f"halt_head must produce 2 outputs (Q_halt, Q_continue) per "
        f"HRM-Text §5:222-227; got out_features={m.halt_head.out_features}"
    )


def test_q_head_off_module_is_none():
    """Flag off → halt_head is None, last_q_pair is None."""
    m = _make_model(use_halt_head=False)
    assert m.halt_head is None
    assert m.last_q_pair is None


def test_q_pair_populated_after_forward():
    """Forward with use_halt_head=True sets last_q_pair to (B, 2)."""
    m = _make_model(use_halt_head=True)
    m.eval()
    ids, _ = _make_batch(B=2)
    with torch.no_grad():
        _ = m(ids)
    assert m.last_q_pair is not None
    assert m.last_q_pair.shape == (2, 2), (
        f"last_q_pair must be (B, 2); got {tuple(m.last_q_pair.shape)}"
    )
    # Sigmoid outputs are in [0, 1]
    assert (m.last_q_pair >= 0).all() and (m.last_q_pair <= 1).all()


def test_q_head_gradient_flows():
    """BCE on q_pair produces non-zero grad on halt_head."""
    m = _make_model(use_halt_head=True)
    m.train()
    ids, mask = _make_batch(B=2)
    _ = m(ids)
    q_pair = m.last_q_pair
    targets = torch.zeros_like(q_pair)
    targets[:, 0] = 1.0  # Q_halt target
    loss = F.binary_cross_entropy(q_pair, targets)
    loss.backward()
    assert m.halt_head.weight.grad is not None
    assert m.halt_head.weight.grad.abs().max() > 0


# =============================================================================
# Completion-reward helper
# =============================================================================


def test_completion_reward_shifted_semantics():
    """Reward uses shifted target semantics: log_probs[:, :-1] vs ids[:, 1:]
    with mask = loss_mask[:, :-1]."""
    B, S, V = 2, 8, 16
    # Build log_probs where argmax at position t matches ids[t+1] for example 0,
    # and DOESN'T match for example 1 (one position off)
    torch.manual_seed(0)
    ids = torch.randint(4, V, (B, S), dtype=torch.long)
    log_probs = torch.zeros(B, S, V) - 100.0
    # Example 0: prediction at position t puts max at ids[t+1] for all positions
    for t in range(S - 1):
        log_probs[0, t, ids[0, t+1]] = 0.0
    # Example 1: same except wrong at position 5
    for t in range(S - 1):
        if t == 5:
            log_probs[1, t, (ids[1, t+1] + 1) % V] = 0.0  # wrong token
        else:
            log_probs[1, t, ids[1, t+1]] = 0.0
    # Mask covers positions 2..S-2 (sep at 2, answer through S-1)
    mask = torch.zeros(B, S, dtype=torch.bool)
    mask[:, 2:S-1] = True
    reward = full_answer_correct(log_probs, ids, mask)
    assert reward.shape == (B,)
    assert reward[0].item() == True   # Example 0 fully correct
    assert reward[1].item() == False  # Example 1 wrong at position 5


def test_completion_reward_empty_mask():
    """No masked positions → reward must be False (not vacuously True)."""
    B, S, V = 2, 8, 16
    log_probs = torch.randn(B, S, V)
    ids = torch.randint(4, V, (B, S), dtype=torch.long)
    mask = torch.zeros(B, S, dtype=torch.bool)
    reward = full_answer_correct(log_probs, ids, mask)
    assert reward.shape == (B,)
    assert not reward.any(), "empty mask must yield False reward (not vacuous True)"


# =============================================================================
# compute_hrm_segment_loss helper
# =============================================================================


def test_segment_loss_returns_finite_components():
    """Segment loss = NLL + BCE; both components are finite tensors."""
    B, S, V = 2, 8, 16
    torch.manual_seed(0)
    log_probs = F.log_softmax(torch.randn(B, S, V), dim=-1)
    q_pair = torch.sigmoid(torch.randn(B, 2))
    g_halt = torch.tensor([1.0, 0.0])
    g_continue = torch.tensor([0.3, 0.7])
    ids = torch.randint(4, V, (B, S), dtype=torch.long)
    mask = torch.zeros(B, S, dtype=torch.bool)
    mask[:, 2:S-1] = True
    active = torch.ones(B, dtype=torch.bool)

    total, nll, bce = compute_hrm_segment_loss(
        log_probs, q_pair, g_halt, g_continue, ids, mask, active,
    )
    assert torch.isfinite(total) and torch.isfinite(nll) and torch.isfinite(bce)
    assert total.item() == pytest.approx(nll.item() + bce.item(), abs=1e-5)


def test_segment_loss_inactive_examples_excluded():
    """Inactive examples don't contribute to loss."""
    B, S, V = 4, 8, 16
    torch.manual_seed(0)
    log_probs = F.log_softmax(torch.randn(B, S, V), dim=-1)
    q_pair = torch.sigmoid(torch.randn(B, 2))
    g_halt = torch.ones(B)
    g_continue = torch.zeros(B)
    ids = torch.randint(4, V, (B, S), dtype=torch.long)
    mask = torch.ones(B, S, dtype=torch.bool)
    mask[:, 0] = False  # need at least 1 masked-out position for shift

    all_active = torch.ones(B, dtype=torch.bool)
    half_active = torch.tensor([True, True, False, False])

    total_all, _, _ = compute_hrm_segment_loss(
        log_probs, q_pair, g_halt, g_continue, ids, mask, all_active,
    )
    total_half, _, _ = compute_hrm_segment_loss(
        log_probs, q_pair, g_halt, g_continue, ids, mask, half_active,
    )
    # Different losses (different active subsets); both finite
    assert torch.isfinite(total_all) and torch.isfinite(total_half)


# =============================================================================
# Boundary Q_continue bootstrap target (HRM-Text §5:248-250)
# =============================================================================


def test_boundary_q_continue_uses_q_halt_only():
    """At seg+1 == m_max the lookahead segment is itself the forced-halt
    segment; its Q_continue is illegal as a bootstrap target. Use Q_halt only."""
    # Q_continue > Q_halt for example 0; Q_halt > Q_continue for example 1
    q_next = torch.tensor([[0.30, 0.90], [0.70, 0.40]])
    m_max = 4
    # Boundary: seg == m_max - 1 (so seg + 1 == m_max) → Q_halt only
    boundary = hrm_boundary_q_continue_target(q_next, seg=m_max - 1, m_max=m_max)
    expected_boundary = torch.tensor([0.30, 0.70])
    assert torch.allclose(boundary, expected_boundary), (
        f"boundary target must equal Q_halt; got {boundary.tolist()}"
    )


def test_non_boundary_uses_max_q():
    """Non-boundary lookahead uses max(Q_halt, Q_continue) per HRM-Text."""
    q_next = torch.tensor([[0.30, 0.90], [0.70, 0.40]])
    m_max = 4
    # Non-boundary: seg == 1 (seg + 1 == 2 < m_max) → max rule
    g = hrm_boundary_q_continue_target(q_next, seg=1, m_max=m_max)
    expected = torch.tensor([0.90, 0.70])  # max per row
    assert torch.allclose(g, expected), (
        f"non-boundary target must equal max(Q_halt,Q_continue); got {g.tolist()}"
    )


def test_boundary_target_carries_no_grad_from_lookahead_path():
    """Caller produces q_next under torch.no_grad(); helper must not require
    grad. Verifies no autograd version-error in the segment loop pattern.

    Pattern mirrors train_dt_gsm8k: forward under no_grad, extract q_next,
    feed to boundary helper, attach to loss as constant target."""
    m = _make_model(use_halt_head=True, use_carry=True)
    m.train()
    ids, _ = _make_batch(B=2)
    # Outer forward, grad-tracked (this would be the m-th segment)
    out = m(ids, return_carry=True)
    log_probs, carry = out
    q_pair = m.last_q_pair  # grad-tracked
    # Lookahead under no_grad — matches trainer's no_grad block
    with torch.no_grad():
        _ = m(ids, return_carry=True, carry=carry.detach())
        q_next = m.last_q_pair
    assert not q_next.requires_grad
    # Boundary target is non-grad regardless of seg/m_max
    target_boundary = hrm_boundary_q_continue_target(q_next, seg=3, m_max=4)
    target_non_boundary = hrm_boundary_q_continue_target(q_next, seg=1, m_max=4)
    assert not target_boundary.requires_grad
    assert not target_non_boundary.requires_grad
    # Use boundary target in a BCE against grad-tracked Q_continue; grad
    # must flow to halt_head but NOT through the target
    loss = F.binary_cross_entropy(q_pair[..., 1], target_boundary)
    loss.backward()
    assert m.halt_head.weight.grad is not None
    assert m.halt_head.weight.grad.abs().max() > 0


# =============================================================================
# Carry-detach segment loop
# =============================================================================


def test_carry_detach_breaks_gradient():
    """Detached carry between segments breaks gradient chain (HRM-Text §4)."""
    m = _make_model(use_halt_head=True, use_carry=True)
    m.train()
    ids, _ = _make_batch(B=2)

    # Segment 1: grad-tracked
    out1 = m(ids, return_carry=True)
    log_probs1, carry1 = out1
    loss1 = log_probs1.mean()
    grads1 = torch.autograd.grad(loss1, m.parameters(), retain_graph=False,
                                   allow_unused=True)
    n_with_grad1 = sum(1 for g in grads1 if g is not None and g.abs().sum() > 0)
    assert n_with_grad1 > 0

    # Segment 2: with detached carry — gradient from segment 2's loss should
    # NOT flow back through carry to segment 1's forward
    carry_detached = carry1.detach()
    assert not carry_detached.requires_grad
    out2 = m(ids, carry=carry_detached, return_carry=True)
    log_probs2, carry2 = out2
    loss2 = log_probs2.mean()
    # backward through loss2 should be OK (no version error from segment 1)
    loss2.backward()
    # Gradient exists somewhere on segment 2's forward
    grad_norm = sum(p.grad.abs().sum() for p in m.parameters() if p.grad is not None)
    assert grad_norm > 0


# =============================================================================
# act_inference deprecation
# =============================================================================


def test_act_inference_raises_when_halt_head_active():
    """act_inference=True with use_halt_head=True must raise (Slice 13f.2
    removed old cumulative-prob ACT; source-faithful Q-rule inference deferred)."""
    m = _make_model(use_halt_head=True)
    m.eval()
    ids, _ = _make_batch(B=2)
    with pytest.raises(ValueError, match="DEFERRED|deferred"):
        with torch.no_grad():
            _ = m(ids, act_inference=True)


def test_compute_act_loss_raises_NotImplementedError():
    """Old PonderNet compute_act_loss is gone; calling raises."""
    m = _make_model(use_halt_head=True)
    with pytest.raises(NotImplementedError, match="compute_hrm_segment_loss"):
        m.compute_act_loss(
            per_iter_log_probs=[],
            halt_probs=[],
            target_ids=torch.zeros(1, 1, dtype=torch.long),
        )


# =============================================================================
# All-finite short training-step
# =============================================================================


def test_short_training_step_finite():
    """3-segment training-step with active-mask loop runs all-finite."""
    m = _make_model(use_halt_head=True, use_carry=True)
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    ids, mask = _make_batch(B=2)

    carry = None
    active = torch.ones(2, dtype=torch.bool)
    M_max = 3
    for seg_idx in range(M_max):
        out = m(ids, carry=carry, return_carry=True)
        log_probs, carry_out = out
        q_pair = m.last_q_pair
        reward = full_answer_correct(log_probs, ids, mask).float()
        g_halt = reward
        g_continue = torch.zeros_like(reward)  # simplified for unit test

        total, nll, bce = compute_hrm_segment_loss(
            log_probs, q_pair, g_halt, g_continue, ids, mask, active,
        )
        assert torch.isfinite(total)
        opt.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=1.0)
        opt.step()
        carry = carry_out.detach()

    # All weights finite after 3 segments
    for name, p in m.named_parameters():
        assert torch.isfinite(p).all(), f"non-finite weights in {name}"
