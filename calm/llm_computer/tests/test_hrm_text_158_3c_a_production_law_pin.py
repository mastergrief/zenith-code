"""Tests for 3C-A production law pin (einsum_q15q16_rescale_q24_v1)."""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from calm.hrm_text_158.native_full_stack.bounded_delta_learner import default_dry_run_rank_vote_spec
from calm.hrm_text_158.native_full_stack.integer_marginal_attribution import (
    INTEGER_MARGINAL_ATTRIBUTION_LAW_ID_V0,
    INTEGER_MARGINAL_ATTRIBUTION_PRODUCTION_LAW_ID,
    _rescale_accumulator_to_attribution_q,
    integer_marginal_attribution_from_captures,
)
from calm.hrm_text_158.native_full_stack.integer_sparse_rank_votes import (
    CREDIT_LAW_NEG_ATTRIBUTION_Q31_V1,
    INTEGER_SPARSE_RANK_PRODUCTION_CREDIT_LAW_ID,
    credit_q31_from_attribution,
    sparse_rank_votes_from_captures_reference,
)
from calm.hrm_text_158.native_full_stack.realistic_gradient_parity_probe import (
    VERDICT_BROAD_HOLDS,
    VERDICT_FRACTIONAL_COLLAPSE,
    build_per_candidate_parity_records,
    build_trainer_16x16_capture_fixture,
    classify_tier_parity_verdict,
    run_realistic_gradient_parity_probe,
    run_tier1_trainer_16x16_capture,
)
from calm.hrm_text_158.native_full_stack.trainer_sub2_authority import (
    derive_trainer_sub2_authority_states,
    select_trainer_eligible_bitlinears,
    trainer_authoritative_forward_context,
)


def _t1_capture_tensors():
    fixture = build_trainer_16x16_capture_fixture()
    model = fixture.model
    eligible = select_trainer_eligible_bitlinears(model, use_ternary_bulk=True)
    states = derive_trainer_sub2_authority_states(eligible)
    model.train(True)
    model.zero_grad(set_to_none=True)
    with trainer_authoritative_forward_context(
        eligible,
        states,
        device="cpu",
        requires_grad=True,
    ) as handle:
        out = model(fixture.batch["x"])
        loss = F.mse_loss(out, fixture.batch["target"])
        loss.backward()
        capture = handle.captures["proj"]
        state = states["proj"]
    return capture, tuple(int(dim) for dim in state.q_levels.shape), state.q_levels.reshape(-1)


def test_v0_law_reproduces_fractional_collapse_on_t1():
    capture, weight_shape, q_flat = _t1_capture_tensors()
    _records, summary = build_per_candidate_parity_records(
        inputs=capture["inputs"],
        grad_outputs=capture["grad_outputs"],
        weight_shape=weight_shape,
        q_levels_flat=q_flat,
        spec=default_dry_run_rank_vote_spec(),
        attribution_law_id=INTEGER_MARGINAL_ATTRIBUTION_LAW_ID_V0,
        max_records=10_000,
    )
    events_v0 = integer_marginal_attribution_from_captures(
        capture["inputs"],
        capture["grad_outputs"],
        weight_shape=weight_shape,
        law_id=INTEGER_MARGINAL_ATTRIBUTION_LAW_ID_V0,
    )
    assert events_v0.law_id == INTEGER_MARGINAL_ATTRIBUTION_LAW_ID_V0
    assert summary["rank_positions_match_rate"] == pytest.approx(0.1875)
    verdict = classify_tier_parity_verdict(
        tier_id="T1",
        measurement_valid=True,
        rank_positions_match_rate=summary["rank_positions_match_rate"],
        events_match_rate=summary["events_match_rate"],
        fractional_collision_share_of_mismatches=summary[
            "fractional_collision_share_of_mismatches"
        ],
    )
    assert verdict.parity_verdict == VERDICT_FRACTIONAL_COLLAPSE


def test_production_v1_law_broad_holds_on_full_t1():
    result = run_tier1_trainer_16x16_capture()
    assert result.measurement_valid is True
    verdict = classify_tier_parity_verdict(
        tier_id="T1",
        measurement_valid=result.measurement_valid,
        rank_positions_match_rate=result.aggregate_metrics["rank_positions_match_rate"],
        events_match_rate=result.aggregate_metrics["events_match_rate"],
        fractional_collision_share_of_mismatches=result.aggregate_metrics[
            "fractional_collision_share_of_mismatches"
        ],
    )
    assert verdict.parity_verdict == VERDICT_BROAD_HOLDS
    assert result.aggregate_metrics["rank_positions_match_rate"] == pytest.approx(1.0)
    assert result.aggregate_metrics["events_match_rate"] == pytest.approx(1.0)


def test_production_defaults_on_sparse_rank_path():
    capture, weight_shape, q_flat = _t1_capture_tensors()
    result = sparse_rank_votes_from_captures_reference(
        capture["inputs"],
        capture["grad_outputs"],
        weight_shape=weight_shape,
        q_levels_flat=q_flat,
    )
    assert result.credit_law_id == INTEGER_SPARSE_RANK_PRODUCTION_CREDIT_LAW_ID
    assert result.credit_law_id == CREDIT_LAW_NEG_ATTRIBUTION_Q31_V1
    events = integer_marginal_attribution_from_captures(
        capture["inputs"],
        capture["grad_outputs"],
        weight_shape=weight_shape,
    )
    assert events.law_id == INTEGER_MARGINAL_ATTRIBUTION_PRODUCTION_LAW_ID


def test_rescale_shift_constants_match_design():
    from calm.hrm_text_158.native_full_stack.integer_marginal_attribution import (
        ATTRIBUTION_RESCALE_SHIFT_V0,
        ATTRIBUTION_RESCALE_SHIFT_V1,
    )

    assert ATTRIBUTION_RESCALE_SHIFT_V0 == 31
    assert ATTRIBUTION_RESCALE_SHIFT_V1 == 24


def test_saturation_fail_closed_on_int32_overflow():
    huge = torch.tensor([2**60], dtype=torch.int64)
    from calm.hrm_text_158.native_full_stack.integer_marginal_attribution import (
        ATTRIBUTION_RESCALE_SHIFT_V1,
    )

    with pytest.raises(ValueError, match="overflowed int32"):
        _rescale_accumulator_to_attribution_q(huge, shift=ATTRIBUTION_RESCALE_SHIFT_V1)


def test_credit_v1_rejects_int32_min_attribution():
    with pytest.raises(ValueError, match="INT32_MIN"):
        credit_q31_from_attribution(
            torch.tensor([-(2**31)], dtype=torch.int32),
            credit_law_id=CREDIT_LAW_NEG_ATTRIBUTION_Q31_V1,
        )


def test_full_probe_t1_only_uses_production_law(tmp_path, monkeypatch):
    missing = tmp_path / "missing.pt"
    monkeypatch.setenv("HRM_TEXT_158_PROBE_CHECKPOINT", str(missing))
    receipt = run_realistic_gradient_parity_probe(run_t2=False)
    assert receipt.tier_verdicts["T1"].parity_verdict == VERDICT_BROAD_HOLDS
    assert receipt.gpu_gate_recommendation != "reopen_3c_a_before_gpu"
