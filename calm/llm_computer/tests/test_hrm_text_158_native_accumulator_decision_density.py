"""C1.1b accumulator decision-density feasibility/classification tests."""
from __future__ import annotations

from typing import Any

import pytest
import torch

from calm.hrm_text_158.native_full_stack.accumulator_compression import (
    CandidateClassification,
    required_decision_dimension_names,
)
from calm.hrm_text_158.native_full_stack.accumulator_decision_density import (
    DECISION_EXACT_INFEASIBLE,
    VALUE_ENTROPY_IS_NOT_DECISION_EXACT,
    AccumulatorDecisionDensityInput,
    assess_default_accumulator_candidates,
    c1_1a_prior_large_accumulator_budget_bits_per_weight,
    classify_accumulator_candidate,
    dense_fixed_width_bits_per_weight,
    index_bits_for_numel,
    measure_accumulator_decision_density,
    project_sparse_accumulator_bpw,
    validate_accumulator_decision_density_report,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap import GlobalRateCapSpec
from calm.hrm_text_158.native_full_stack.vote_update import (
    VoteUpdateInputs,
    VoteUpdateSpec,
    VoteUpdateState,
)


def _spec(**kwargs) -> VoteUpdateSpec:
    base = dict(
        threshold_abs=10,
        accumulator_clip_min=-127,
        accumulator_clip_max=127,
        decay_numerator=1,
        decay_denominator=1,
        max_abs_per_tensor=128,
        fraction_per_tensor=1.0,
    )
    base.update(kwargs)
    return VoteUpdateSpec(**base)


def _state(q: list[int], acc: list[int]) -> VoteUpdateState:
    return VoteUpdateState(
        q_levels=torch.tensor(q, dtype=torch.int8),
        accumulators=torch.tensor(acc, dtype=torch.int16),
    )


def _inputs(votes: list[int], **kwargs) -> VoteUpdateInputs:
    converted = {}
    for name, value in kwargs.items():
        if value is None:
            converted[name] = None
        elif name.endswith("moves"):
            converted[name] = torch.tensor(value, dtype=torch.int8)
        else:
            converted[name] = torch.tensor(value, dtype=torch.int16)
    return VoteUpdateInputs(votes=torch.tensor(votes, dtype=torch.int16), **converted)


def _density_input(
    state_key: str,
    q: list[int],
    acc: list[int],
    votes: list[int],
    *,
    spec: VoteUpdateSpec | None = None,
    **vote_kwargs,
) -> AccumulatorDecisionDensityInput:
    return AccumulatorDecisionDensityInput(
        state_key=state_key,
        state=_state(q, acc),
        vote_inputs=_inputs(votes, **vote_kwargs),
        spec=spec or _spec(),
    )


def _assert_no_tensors(value: Any) -> None:
    if isinstance(value, torch.Tensor):
        raise AssertionError("compact report must not include raw tensors")
    if isinstance(value, dict):
        for child in value.values():
            _assert_no_tensors(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _assert_no_tensors(child)


def test_sparse_projection_charges_overhead_and_dense_fixed_width_is_out():
    target = c1_1a_prior_large_accumulator_budget_bits_per_weight()

    assert dense_fixed_width_bits_per_weight(1) > target
    assert index_bits_for_numel(16_384) == 14
    assert index_bits_for_numel(4096 * 4096) == 24

    lower_bound = project_sparse_accumulator_bpw(
        eligible_weight_count=16_384,
        stored_row_count=208,
        target_bits_per_weight=target,
        flag_bits_per_row=0,
        tensor_metadata_bits=0,
    )
    assert lower_bound.projected_bits_per_weight == pytest.approx(
        (208 * (14 + 16)) / 16_384,
    )
    assert lower_bound.fits_target is True

    overhead_charged = project_sparse_accumulator_bpw(
        eligible_weight_count=16_384,
        stored_row_count=208,
        target_bits_per_weight=target,
        flag_bits_per_row=2,
        tensor_metadata_bits=64,
    )
    assert overhead_charged.payload_only_bits_per_weight == pytest.approx((208 * 16) / 16_384)
    assert overhead_charged.payload_only_would_fit is True
    assert overhead_charged.projected_bits_per_weight > target
    assert overhead_charged.fits_target is False


def test_next_step_active_threshold_uses_fixture_votes_not_current_magnitude():
    report = measure_accumulator_decision_density(
        [
            _density_input(
                "post_vote.crossing",
                q=[0] * 8,
                acc=[0] * 8,
                votes=[11, -11, 0, 0, 0, 0, 0, 0],
            ),
        ],
        target_bits_per_weight=0.4,
        tensor_metadata_bits=0,
    )

    validate_accumulator_decision_density_report(report)
    assert "fixture_vote" in report.active_definition
    assert report.fixture_vote_record == "actual_fixture_votes"
    assert report.fixture_vote_abs_max == 11
    assert report.current_magnitude_threshold_count == 0
    assert report.active_next_step_count == 2
    assert report.decision_relevant_exact_count == 2
    assert report.far_row_count == 6


def test_ranking_sensitive_adversary_catches_naive_near_threshold_sparse_projection():
    q = [0] * 64
    acc = [0] * 64
    votes = [40] * 48 + [0] * 16
    report = measure_accumulator_decision_density(
        [_density_input("dense.cap", q=q, acc=acc, votes=votes)],
        global_cap_spec=GlobalRateCapSpec(cap=2, step=1),
        target_bits_per_weight=0.4,
        tensor_metadata_bits=0,
    )

    validate_accumulator_decision_density_report(report)
    assert report.current_magnitude_threshold_count == 0
    assert report.active_next_step_count == 48
    assert report.ranking_sensitive_exact_count == 48
    assert report.global_cap_row_count == 48
    assert report.global_cap_accepted_count == 2
    assert report.global_cap_deferred_count == 46
    assert report.cap_frontier_diagnostic_count == 2
    assert report.ranking_sensitive_exact_count > report.cap_frontier_diagnostic_count

    naive_current_only = project_sparse_accumulator_bpw(
        eligible_weight_count=64,
        stored_row_count=report.current_magnitude_threshold_count,
        target_bits_per_weight=0.4,
        tensor_metadata_bits=0,
    )
    assert naive_current_only.fits_target is True
    assert report.sparse_exact_projection.fits_target is False
    assert report.sparse_exact_projection.backlog_entry_count == 46
    assert report.sparse_exact_projection.backlog_storage_bits > 0


def test_cap_saturation_backlog_growth_makes_queue_metadata_visible():
    backlog = {
        "backlog.cap": {
            i: {"first_step": 1, "last_deferred_step": 1, "defer_count": 1}
            for i in range(1, 8)
        },
    }
    report = measure_accumulator_decision_density(
        [
            _density_input(
                "backlog.cap",
                q=[0] * 16,
                acc=[0] * 16,
                votes=[30] * 16,
            ),
        ],
        global_cap_spec=GlobalRateCapSpec(cap=1, step=2),
        deferred_backlog=backlog,
        target_bits_per_weight=4.0,
        tensor_metadata_bits=0,
    )

    no_backlog_projection = project_sparse_accumulator_bpw(
        eligible_weight_count=report.eligible_weight_count,
        stored_row_count=report.decision_relevant_exact_count,
        target_bits_per_weight=4.0,
        tensor_metadata_bits=0,
        backlog_entry_count=0,
    )
    assert report.global_cap_saturated is True
    assert report.backlog_state_carry_count == 15
    assert report.backlog_max_age_steps == 1
    assert report.backlog_max_defer_count == 2
    assert report.sparse_exact_projection.backlog_storage_bits > 0
    assert report.sparse_exact_projection.projected_bits_per_weight > (
        no_backlog_projection.projected_bits_per_weight
    )


def test_far_value_entropy_is_separate_and_compact_not_decision_exact_evidence():
    report = measure_accumulator_decision_density(
        [
            _density_input(
                "low.entropy",
                q=[0] * 12,
                acc=[0] * 12,
                votes=[11] + [0] * 11,
            ),
        ],
        target_bits_per_weight=0.4,
        tensor_metadata_bits=0,
    )
    payload = report.to_dict()

    validate_accumulator_decision_density_report(report)
    assert report.raw_arrays_included is False
    assert report.far_value_entropy.density_note == VALUE_ENTROPY_IS_NOT_DECISION_EXACT
    assert report.far_value_entropy.unique_value_count == 1
    assert report.far_value_entropy.shannon_entropy_bits_per_value == pytest.approx(0.0)
    _assert_no_tensors(payload)

    sparse_attempt = classify_accumulator_candidate(
        candidate_name="low_entropy_sparse_exact_attempt",
        classification=CandidateClassification.DECISION_EXACT,
        projection=report.sparse_exact_projection,
        covered_decision_dimensions=required_decision_dimension_names(),
    )
    assert sparse_attempt.classification == DECISION_EXACT_INFEASIBLE
    assert sparse_attempt.decision_exact_feasible is False


def test_candidate_classification_requires_budget_dimensions_and_bounded_delta_guardrail():
    fit_projection = project_sparse_accumulator_bpw(
        eligible_weight_count=16_384,
        stored_row_count=1,
        target_bits_per_weight=c1_1a_prior_large_accumulator_budget_bits_per_weight(),
        tensor_metadata_bits=0,
    )
    decision_exact = classify_accumulator_candidate(
        candidate_name="tiny_sparse_exact_probe",
        classification=CandidateClassification.DECISION_EXACT,
        projection=fit_projection,
        covered_decision_dimensions=required_decision_dimension_names(),
    )
    assert decision_exact.classification == CandidateClassification.DECISION_EXACT.value
    assert decision_exact.decision_exact_feasible is True
    assert decision_exact.c2_eligible_by_default is True

    over_budget = project_sparse_accumulator_bpw(
        eligible_weight_count=64,
        stored_row_count=48,
        target_bits_per_weight=0.4,
        tensor_metadata_bits=0,
    )
    infeasible = classify_accumulator_candidate(
        candidate_name="over_budget_sparse_exact_probe",
        classification=CandidateClassification.DECISION_EXACT,
        projection=over_budget,
        covered_decision_dimensions=required_decision_dimension_names(),
    )
    assert infeasible.classification == DECISION_EXACT_INFEASIBLE
    assert "overhead-inclusive" in infeasible.infeasibility_reason
    assert infeasible.candidate_assessment is None

    with pytest.raises(ValueError, match="hypothesis and guardrail"):
        classify_accumulator_candidate(
            candidate_name="bounded_without_guardrail",
            classification=CandidateClassification.BOUNDED_DELTA_WITH_REPORT,
            projection=over_budget,
            covered_decision_dimensions=required_decision_dimension_names(),
        )


def test_default_candidate_assessments_include_four_families_without_encoders_or_overclaims():
    report = measure_accumulator_decision_density(
        [
            _density_input(
                "candidate.family",
                q=[0] * 32,
                acc=[0] * 32,
                votes=[25] * 20 + [0] * 12,
            ),
        ],
        global_cap_spec=GlobalRateCapSpec(cap=2, step=1),
        target_bits_per_weight=0.4,
        tensor_metadata_bits=0,
    )
    candidates = assess_default_accumulator_candidates(report)
    by_name = {candidate.candidate_name: candidate for candidate in candidates}

    assert set(by_name) == {
        "sparse_exact_decision_set",
        "event_coded_crossing_residual_log",
        "bucketed_residual_with_exact_guard_band",
        "hybrid_hot_exact_cold_bucket",
    }
    assert by_name["sparse_exact_decision_set"].classification == DECISION_EXACT_INFEASIBLE
    for name in (
        "event_coded_crossing_residual_log",
        "bucketed_residual_with_exact_guard_band",
        "hybrid_hot_exact_cold_bucket",
    ):
        candidate = by_name[name]
        assert candidate.classification == CandidateClassification.BOUNDED_DELTA_WITH_REPORT.value
        assert candidate.candidate_assessment is not None
        assert candidate.candidate_assessment.bounded_delta_hypothesis
        assert candidate.candidate_assessment.guardrail
        assert candidate.c2_eligible_by_default is False

    with pytest.raises(ValueError, match="physical sub-2 decision_exact claim"):
        validate_accumulator_decision_density_report(
            report,
            claimed_decision_exact_physical_sub2=True,
        )
