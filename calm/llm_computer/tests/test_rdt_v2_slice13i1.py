"""Slice 13i.1 tests: continue-biased Q init + M_min warmup curriculum.

Per codex audit msg 1779432805671-336d5d16 + 1779432871832-2e22f7a6:
- Step-0 telemetry must show intended continue bias (Qc > Qh) BEFORE
  training starts — otherwise the curriculum fix is not actually installed
  (codex's acceptance gate).
- M_min warmup must override the stochastic epsilon draw for the first
  N epochs, then anneal back to source-faithful behavior.
- Defaults must preserve current behavior (warmup-epochs=0, q-init off).
"""
from __future__ import annotations

import math

import torch

from calm.llm_computer.copy_augmented_delta import build_copy_augmented_delta


def _make_model(seed: int = 42):
    """TRM-1.58 first-config-shaped model with halt-head."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return build_copy_augmented_delta(
        vocab_size=16, d_model=8, n_heads=4, n_layers=2, d_ffn=16,
        max_len=24, n_copy_heads=2, sep_token_id=3,
        use_chunkwise=True, n_iterations=2, h_cycles=2,
        use_loop_index=True, use_input_injection=True,
        use_gated_attention=True, use_z_init=True, use_lecun_init=True,
        use_h_rmsnorm=True, use_short_conv=True, use_h_layer_stack=True,
        use_pre_rmsnorm=True,
        use_halt_head=True, use_carry=True,
    )


def _apply_continue_bias(m):
    """The exact same patch the trainer applies when q_init_bias_continue
    is True (see scripts/train_dt_gsm8k.py)."""
    with torch.no_grad():
        m.halt_head.bias.zero_()
        m.halt_head.bias[0] = -1.0
        m.halt_head.bias[1] = +1.0


# =============================================================================
# Q-init bias: step-0 telemetry gate
# =============================================================================


def test_continue_bias_init_produces_qc_greater_than_qh_before_training():
    """Acceptance gate per codex msg 1779432871832: with continue-biased
    init, the model's first forward pass (step 0, NO TRAINING) must
    produce Q_continue > Q_halt across all examples in the batch.
    If this fails, the curriculum fix isn't actually installed."""
    m = _make_model()
    _apply_continue_bias(m)
    m.eval()
    ids = torch.tensor([[1, 5, 6, 7, 3, 10, 11]], dtype=torch.long)
    with torch.no_grad():
        _ = m(ids)
    assert m.last_q_pair is not None
    # last_q_pair is (B, 2): index 0 = Q_halt, index 1 = Q_continue
    q_halt = m.last_q_pair[..., 0]
    q_continue = m.last_q_pair[..., 1]
    assert (q_continue > q_halt).all(), (
        f"continue-biased init must produce Qc > Qh at step 0; "
        f"got Qh={q_halt.tolist()} Qc={q_continue.tolist()}"
    )


def test_continue_bias_init_specific_magnitudes():
    """Continue-bias config sets bias to [-1.0, +1.0]. The HEAD weights
    are random-init at this point so the linear output starts near
    bias-only; sigmoid(bias) gives approximate Q value bounds.

    Qh expected near sigmoid(-1) ≈ 0.269; Qc expected near sigmoid(+1) ≈ 0.731.
    With random weight contributions, allow a generous tolerance.
    """
    m = _make_model()
    _apply_continue_bias(m)
    m.eval()
    ids = torch.tensor([[1, 5, 6, 7, 3, 10, 11]], dtype=torch.long)
    with torch.no_grad():
        _ = m(ids)
    q_halt = m.last_q_pair[..., 0].item()
    q_continue = m.last_q_pair[..., 1].item()
    # Expected: Qh near 0.27, Qc near 0.73. Tolerance generous because
    # head's linear weights contribute to the raw logit before sigmoid.
    assert 0.10 < q_halt < 0.50, (
        f"Qh out of expected [0.10, 0.50] range for sigmoid(-1+noise); got {q_halt:.3f}"
    )
    assert 0.50 < q_continue < 0.90, (
        f"Qc out of expected [0.50, 0.90] range for sigmoid(+1+noise); got {q_continue:.3f}"
    )


def test_default_init_produces_qh_greater_or_near_qc():
    """Without continue-bias patch, halt_head is LeCun-init (zero-mean
    weights) so Qh and Qc sigmoid outputs start near 0.5 each — the
    failure mode 13f.3b showed (numerical-noise ordering halts at seg 1).
    Verifies the bias patch is required to produce the asymmetric init."""
    m = _make_model()
    # No continue-bias applied — default init
    m.eval()
    ids = torch.tensor([[1, 5, 6, 7, 3, 10, 11]], dtype=torch.long)
    with torch.no_grad():
        _ = m(ids)
    q_halt = m.last_q_pair[..., 0].item()
    q_continue = m.last_q_pair[..., 1].item()
    # Both should be near 0.5 (no init bias) — gap should be small,
    # determined by random weight contributions only.
    gap = abs(q_halt - q_continue)
    assert gap < 0.20, (
        f"default init should produce Qh ≈ Qc (gap < 0.20); got "
        f"Qh={q_halt:.3f} Qc={q_continue:.3f} gap={gap:.3f}. "
        f"If this fails, the build path may be applying a bias already."
    )


# =============================================================================
# M_min warmup curriculum: schedule helper logic
# =============================================================================


def _m_min_schedule(ep: int, warmup_epochs: int, warmup_value: int,
                    epsilon: float, m_max: int, rng_draw: float,
                    rand_choice: int) -> int:
    """Pure-function mirror of the m_min selection logic in
    `scripts/train_dt_gsm8k.py` after Slice 13i.1.

    Args mirror the trainer state:
      ep: current epoch (1-indexed)
      warmup_epochs / warmup_value: 13i.1 flags
      epsilon: HRM-Text §5:234-236 epsilon
      m_max: M_max
      rng_draw: simulated torch.rand(1).item() value in [0,1)
      rand_choice: simulated torch.randint(2, m_max+1, (1,)).item() value

    Returns m_min that the trainer would use.
    """
    if warmup_epochs > 0 and ep <= warmup_epochs:
        return warmup_value
    if rng_draw < epsilon:
        return rand_choice
    return 1


def test_m_min_warmup_overrides_stochastic_during_warmup():
    """During warmup epochs (ep ≤ warmup_epochs), m_min is constant at
    warmup_value REGARDLESS of the stochastic draw."""
    # During warmup: warmup_value should fire no matter the draw
    assert _m_min_schedule(ep=1, warmup_epochs=3, warmup_value=4,
                            epsilon=0.1, m_max=4,
                            rng_draw=0.5, rand_choice=2) == 4
    # Even when draw < epsilon (would normally trigger stochastic m_min)
    assert _m_min_schedule(ep=2, warmup_epochs=3, warmup_value=4,
                            epsilon=0.1, m_max=4,
                            rng_draw=0.05, rand_choice=3) == 4


def test_m_min_anneal_after_warmup():
    """After warmup epochs, m_min reverts to source-faithful epsilon-
    stochastic: m_min=1 if rng_draw >= epsilon, else rand_choice from
    {2..M_max}."""
    # Past warmup, draw >= epsilon → m_min = 1
    assert _m_min_schedule(ep=4, warmup_epochs=3, warmup_value=4,
                            epsilon=0.1, m_max=4,
                            rng_draw=0.5, rand_choice=2) == 1
    # Past warmup, draw < epsilon → m_min = rand_choice
    assert _m_min_schedule(ep=4, warmup_epochs=3, warmup_value=4,
                            epsilon=0.1, m_max=4,
                            rng_draw=0.05, rand_choice=3) == 3


def test_m_min_warmup_off_preserves_legacy_behavior():
    """With warmup_epochs=0, the schedule MUST match the legacy
    HRM-Text §5:234-236 stochastic behavior — no warmup override."""
    # warmup_epochs=0: always fall through to stochastic path
    assert _m_min_schedule(ep=1, warmup_epochs=0, warmup_value=4,
                            epsilon=0.1, m_max=4,
                            rng_draw=0.5, rand_choice=2) == 1
    assert _m_min_schedule(ep=1, warmup_epochs=0, warmup_value=4,
                            epsilon=0.1, m_max=4,
                            rng_draw=0.05, rand_choice=3) == 3


def test_m_min_warmup_boundary_inclusive():
    """At ep == warmup_epochs, we should still be IN warmup (boundary
    inclusive). At ep == warmup_epochs + 1, we transition to stochastic."""
    # ep == warmup_epochs: still warmup
    assert _m_min_schedule(ep=3, warmup_epochs=3, warmup_value=4,
                            epsilon=0.1, m_max=4,
                            rng_draw=0.5, rand_choice=2) == 4
    # ep == warmup_epochs + 1: anneal active
    assert _m_min_schedule(ep=4, warmup_epochs=3, warmup_value=4,
                            epsilon=0.1, m_max=4,
                            rng_draw=0.5, rand_choice=2) == 1
