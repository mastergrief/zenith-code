"""C1.1c bounded-delta accumulator ledger/oracle tests."""
from __future__ import annotations

import hashlib
from typing import Any

import pytest
import torch

from calm.hrm_text_158.native_full_stack.bounded_delta_accumulator import (
    ALGORITHMIC_LOCAL_VOTE_UPDATE_EXECUTABLE_NOT_PHYSICAL_SUB2,
    BOUNDED_DELTA_ADMISSION_FAILED,
    BOUNDED_DELTA_GUARDRAIL_FAILED,
    BOUNDED_DELTA_LEDGER_FAILED,
    BOUNDED_DELTA_WITH_REPORT,
    COARSE_SIGNED_CHARGE_SPARSE_FRONTIER_CANDIDATE,
    EVENT_CODED_CROSSING_RESIDUAL_LOG_CANDIDATE,
    BoundedDeltaAccumulatorState,
    BoundedDeltaGuardSpec,
    BoundedDeltaOracleInput,
    bounded_delta_inclusive_ledger,
    compare_bounded_delta_step_to_int16_oracle,
    encode_budget_capped_hybrid_reference,
    execute_direct_bounded_local_vote_update_candidate,
    INTRINSIC_BOUNDED_UPDATE_DOMAIN_GAP,
    project_bounded_delta_accumulator_bpw,
    validate_bounded_delta_inclusive_ledger,
    decode_bounded_accumulator_to_i16,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap import GlobalRateCapSpec
from calm.hrm_text_158.native_full_stack.q_entropy_packing import (
    base3_q_entropy_ledger_for_shapes,
    default_base3_q_entropy_ledger_table,
)
from calm.hrm_text_158.native_full_stack.vote_update import (
    VoteUpdateInputs,
    VoteUpdateSpec,
    VoteUpdateState,
    apply_integer_vote_update_reference,
)


def _prior_large_q_ledger():
    for row in default_base3_q_entropy_ledger_table():
        if row.regime_name == "prior_large_fixture_base3_q":
            return row
    raise AssertionError("prior_large_fixture_base3_q missing")


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


def _state(
    numel: int,
    *,
    acc_overrides: dict[int, int] | None = None,
    q_overrides: dict[int, int] | None = None,
) -> VoteUpdateState:
    q = torch.zeros(numel, dtype=torch.int8)
    acc = torch.zeros(numel, dtype=torch.int16)
    for index, value in (q_overrides or {}).items():
        q[int(index)] = int(value)
    for index, value in (acc_overrides or {}).items():
        acc[int(index)] = int(value)
    return VoteUpdateState(q_levels=q, accumulators=acc)


def _inputs(numel: int, votes: dict[int, int]) -> VoteUpdateInputs:
    out = torch.zeros(numel, dtype=torch.int16)
    for index, value in votes.items():
        out[int(index)] = int(value)
    return VoteUpdateInputs(votes=out)


def _assert_no_tensors(value: Any) -> None:
    if isinstance(value, torch.Tensor):
        raise AssertionError("compact report must not include raw tensors")
    if isinstance(value, dict):
        for child in value.values():
            _assert_no_tensors(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _assert_no_tensors(child)


def _identity_sha(state_key: str, indices: tuple[int, ...]) -> str:
    h = hashlib.sha256()
    for index in sorted(indices):
        h.update(state_key.encode("utf-8"))
        h.update(b":")
        h.update(str(int(index)).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _ordered_identity_sha(state_key: str, indices: tuple[int, ...]) -> str:
    h = hashlib.sha256()
    for index in indices:
        h.update(state_key.encode("utf-8"))
        h.update(b":")
        h.update(str(int(index)).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _ordered_value_sha(state_key: str, label: str, values: dict[int, int]) -> str:
    h = hashlib.sha256()
    for index, value in values.items():
        h.update(state_key.encode("utf-8"))
        h.update(b":")
        h.update(label.encode("utf-8"))
        h.update(b":")
        h.update(str(int(index)).encode("utf-8"))
        h.update(b"=")
        h.update(str(int(value)).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def test_inclusive_ledger_rejects_dense_cold_under_prior_large_budget():
    q_ledger = _prior_large_q_ledger()
    assert q_ledger.remaining_accumulator_budget_bits_per_weight == pytest.approx(0.38232421875)

    one_bit_dense = project_bounded_delta_accumulator_bpw(
        eligible_weight_count=q_ledger.eligible_weight_count,
        hot_exact_row_count=0,
        dense_cold_bits_per_weight=1.0,
    )
    one_bit_ledger = bounded_delta_inclusive_ledger(q_ledger, one_bit_dense)

    assert one_bit_dense.dense_cold_bits_per_weight == pytest.approx(1.0)
    assert one_bit_ledger.bounded_delta_acc_bits_per_weight == pytest.approx(1.0)
    assert one_bit_ledger.accumulator_fits_remaining_budget is False
    assert one_bit_ledger.inclusive_target_achieved is False
    assert one_bit_ledger.claimable_physical_sub2 is False
    assert one_bit_ledger.ledger_status == "bounded_delta_inclusive_ledger_failed"
    with pytest.raises(ValueError, match="physical sub-2 claim"):
        validate_bounded_delta_inclusive_ledger(
            one_bit_ledger,
            claimed_physical_sub2_achieved=True,
        )

    int4_dense = project_bounded_delta_accumulator_bpw(
        eligible_weight_count=q_ledger.eligible_weight_count,
        hot_exact_row_count=0,
        dense_cold_bits_per_weight=4.0,
    )
    int4_ledger = bounded_delta_inclusive_ledger(q_ledger, int4_dense)
    assert int4_ledger.claimable_physical_sub2 is False
    assert int4_ledger.packed_inclusive_physical_bits_per_weight > one_bit_ledger.packed_inclusive_physical_bits_per_weight


def test_inclusive_ledger_recomputes_budget_from_selected_q_scale_regime():
    prior_one_scale = base3_q_entropy_ledger_for_shapes(
        regime_name="prior_large_one_scale_for_c1p1c",
        logical_shapes=((128, 128),),
        scale_count=1,
        accumulator_bits_per_weight=0.0,
    )
    prior_many_scales = base3_q_entropy_ledger_for_shapes(
        regime_name="prior_large_many_scales_for_c1p1c",
        logical_shapes=((128, 128),),
        scale_count=128,
        accumulator_bits_per_weight=0.0,
    )
    projection = project_bounded_delta_accumulator_bpw(
        eligible_weight_count=16_384,
        hot_exact_row_count=80,
        tensor_metadata_bits=0,
        bucket_metadata_bits=0,
        guardrail_metadata_bits=0,
    )

    one_scale_ledger = bounded_delta_inclusive_ledger(prior_one_scale, projection)
    many_scale_ledger = bounded_delta_inclusive_ledger(prior_many_scales, projection)

    assert one_scale_ledger.remaining_accumulator_budget_bits_per_weight == pytest.approx(
        2.0
        - prior_one_scale.q_packed_total_bits_per_weight
        - prior_one_scale.frozen_scale_fp32_bits_per_weight,
    )
    assert many_scale_ledger.remaining_accumulator_budget_bits_per_weight == pytest.approx(
        2.0
        - prior_many_scales.q_packed_total_bits_per_weight
        - prior_many_scales.frozen_scale_fp32_bits_per_weight,
    )
    assert one_scale_ledger.remaining_accumulator_budget_bits_per_weight != pytest.approx(
        many_scale_ledger.remaining_accumulator_budget_bits_per_weight,
    )
    assert one_scale_ledger.claimable_physical_sub2 is True
    assert many_scale_ledger.claimable_physical_sub2 is False
    assert many_scale_ledger.bounded_delta_acc_bits_per_weight <= 0.38232421875
    assert many_scale_ledger.accumulator_fits_remaining_budget is False


def test_zero_drift_when_hot_exact_covers_all_decision_risk_rows():
    q_ledger = _prior_large_q_ledger()
    numel = q_ledger.eligible_weight_count
    state = _state(numel, acc_overrides={5: 9, 7: -9})
    votes = _inputs(numel, {5: 2, 7: -2})
    report = compare_bounded_delta_step_to_int16_oracle(
        [
            BoundedDeltaOracleInput(
                state_key="hot.covered",
                state=state,
                vote_inputs=votes,
                vote_spec=_spec(max_abs_per_tensor=4),
                hot_exact_indices=(5, 7),
            ),
        ],
        q_ledger_row=q_ledger,
        guard_spec=BoundedDeltaGuardSpec(),
        global_cap_spec=GlobalRateCapSpec(cap=1, step=1),
        tensor_offsets={"hot.covered": 0},
        tensor_metadata_bits=0,
        bucket_metadata_bits=0,
        guardrail_metadata_bits=0,
    )

    assert report.classification == BOUNDED_DELTA_WITH_REPORT
    assert report.guard_passed is True
    assert report.ledger.claimable_physical_sub2 is True
    assert report.measured_report.candidate_changed_count == 0
    assert report.measured_report.accepted_changed_count == 0
    assert report.measured_report.deferred_changed_count == 0
    assert report.measured_report.q_changed_count == 0
    assert report.measured_report.hot_risk_changed_count == 0
    assert report.measured_report.accumulator_residual_hash_match is True
    assert report.candidate_assessment.classification == BOUNDED_DELTA_WITH_REPORT
    assert report.candidate_assessment.c2_eligible_by_default is False
    assert report.admission_passed is True
    assert report.rejection_telemetry.summary == "admission_pass"
    assert report.rejection_telemetry.failed_surfaces == ()
    assert report.candidate_assessment.preserved_information
    assert report.candidate_assessment.sub2_persistent_strategy is not None
    _assert_no_tensors(report.to_dict())


def test_direct_bounded_local_vote_update_executes_sparse_event_domain_without_dense_decode():
    state = _state(4, acc_overrides={0: 9, 2: -9})
    votes = _inputs(4, {0: 2, 2: -2})
    bounded = encode_budget_capped_hybrid_reference(
        state,
        hot_exact_indices=(0, 2),
        cold_default_value=0,
    )
    spec = _spec(max_abs_per_tensor=4)

    result = execute_direct_bounded_local_vote_update_candidate(
        state_key="toy.local",
        q_levels=state.q_levels,
        bounded_accumulator=bounded,
        sparse_vote_events={0: 2, 2: -2},
        vote_spec=spec,
    )
    oracle = apply_integer_vote_update_reference(state, votes, spec)
    decoded = decode_bounded_accumulator_to_i16(result.next_bounded_accumulator)
    applied = tuple(
        int(index)
        for index in oracle.plan.applied_indices.detach().cpu().to(torch.int64).tolist()
    )
    directions = {
        int(index): int(direction)
        for index, direction in zip(
            applied,
            oracle.plan.applied_directions.detach().cpu().to(torch.int16).tolist(),
        )
    }
    thresholds = {
        int(index): int(threshold)
        for index, threshold in zip(
            applied,
            oracle.plan.applied_thresholds.detach().cpu().to(torch.int32).tolist(),
        )
    }

    assert result.proof["pass"] is True
    assert result.proof["candidate_dense_decode_used"] is False
    assert result.proof["candidate_accumulator_transient_over2_used"] is False
    assert result.proof["candidate_vote_transient_over2_used"] is False
    assert result.proof["candidate_dense_vote_authority_used"] is False
    assert result.proof["coverage_domain"]["no_global_cap"] is True
    assert result.proof["coverage_domain"]["sparse_vote_events_only"] is True
    assert result.proof["coverage_domain"]["supports_default_mass_crossing"] is False
    assert result.proof["candidate_count"] == int(oracle.plan.candidate_indices.numel())
    assert result.proof["max_flips"] == spec.max_flips(4)
    assert result.proof["pre_veto_selected_flip_count"] == int(oracle.plan.applied_indices.numel())
    assert result.proof["q_changed_count"] == 2
    assert result.proof["applied_row_count"] == 2
    assert result.proof["applied_row_identities_sha256"] == _identity_sha("toy.local", applied)
    assert (
        result.proof["ordered_applied_row_identities_sha256"]
        == _ordered_identity_sha("toy.local", applied)
    )
    assert (
        result.proof["applied_directions_sha256"]
        == _ordered_value_sha("toy.local", "direction", directions)
    )
    assert (
        result.proof["applied_thresholds_sha256"]
        == _ordered_value_sha("toy.local", "threshold", thresholds)
    )
    assert result.next_q_levels.tolist() == [1, 0, -1, 0]
    assert decoded.tolist() == [1, 0, -1, 0]
    assert oracle.q_levels.tolist() == [1, 0, -1, 0]
    assert oracle.accumulators.tolist() == [1, 0, -1, 0]
    assert result.proof["hot_exact_row_count_after"] == 2
    assert result.proof["cold_exception_row_count_after"] == 0
    assert result.proof["storage_projection"]["bounded_delta_acc_bits_per_weight"] >= 2.0
    assert result.proof["accumulator_physical_sub2_pass"] is False
    assert (
        result.proof["scoped_label"]
        == ALGORITHMIC_LOCAL_VOTE_UPDATE_EXECUTABLE_NOT_PHYSICAL_SUB2
    )
    assert (
        result.proof["terminal_classification"]
        == ALGORITHMIC_LOCAL_VOTE_UPDATE_EXECUTABLE_NOT_PHYSICAL_SUB2
    )
    assert result.proof["scoped_physical_budget_claim"] == "algorithmic_only_not_physical_sub2"
    assert result.proof["q_storage_physical_budget_covered_by_scoped_proof"] is False
    assert result.proof["frozen_scale_physical_budget_covered_by_scoped_proof"] is False
    _assert_no_tensors(result.proof)


def test_direct_bounded_local_vote_update_respects_q_boundary_noop_without_threshold_consumption():
    state = _state(
        4,
        acc_overrides={0: 9, 1: -9},
        q_overrides={0: 1, 1: -1},
    )
    votes = _inputs(4, {0: 2, 1: -2})
    bounded = encode_budget_capped_hybrid_reference(
        state,
        hot_exact_indices=(0, 1),
        cold_default_value=0,
    )

    result = execute_direct_bounded_local_vote_update_candidate(
        state_key="toy.boundary",
        q_levels=state.q_levels,
        bounded_accumulator=bounded,
        sparse_vote_events={0: 2, 1: -2},
        vote_spec=_spec(max_abs_per_tensor=4),
    )
    decoded = decode_bounded_accumulator_to_i16(result.next_bounded_accumulator)

    assert result.proof["pass"] is True
    assert result.proof["candidate_count"] == 0
    assert result.proof["applied_row_count"] == 0
    assert result.proof["q_changed_count"] == 0
    assert result.next_q_levels.tolist() == [1, -1, 0, 0]
    assert decoded.tolist() == [11, -11, 0, 0]
    assert result.proof["residual_after_threshold_sha256"] == _ordered_value_sha(
        "toy.boundary",
        "",
        {},
    )
    _assert_no_tensors(result.proof)


def test_direct_bounded_local_vote_update_flags_default_mass_crossing_domain_gap():
    bounded = BoundedDeltaAccumulatorState(
        logical_shape=(4,),
        cold_default_value=10,
        hot_exact_indices=(),
        hot_exact_values=(),
        cold_exception_indices=(),
        cold_exception_values=(),
    )
    q_levels = torch.zeros(4, dtype=torch.int8)

    result = execute_direct_bounded_local_vote_update_candidate(
        state_key="toy.default.crossing",
        q_levels=q_levels,
        bounded_accumulator=bounded,
        sparse_vote_events={},
        vote_spec=_spec(max_abs_per_tensor=4),
    )

    assert result.proof["pass"] is False
    assert result.proof["scoped_label"] is None
    assert result.proof["terminal_classification"] == INTRINSIC_BOUNDED_UPDATE_DOMAIN_GAP
    assert result.proof["domain_gap_dimension"] == "implicit_default_mass_crossing"
    assert result.proof["default_mass_crossing_count"] == 4
    assert result.proof["candidate_dense_decode_used"] is False
    assert result.proof["candidate_accumulator_transient_over2_used"] is False
    assert result.proof["candidate_vote_transient_over2_used"] is False
    assert result.next_q_levels.tolist() == [0, 0, 0, 0]
    _assert_no_tensors(result.proof)


def test_guardrail_failure_when_measured_decision_drift_exceeds_predeclared_spec():
    q_ledger = _prior_large_q_ledger()
    numel = q_ledger.eligible_weight_count
    state = _state(numel, acc_overrides={11: 9})
    votes = _inputs(numel, {11: 2})
    guard = BoundedDeltaGuardSpec(max_candidate_changed_fraction=0.0)

    report = compare_bounded_delta_step_to_int16_oracle(
        [
            BoundedDeltaOracleInput(
                state_key="cold.drift",
                state=state,
                vote_inputs=votes,
                vote_spec=_spec(max_abs_per_tensor=4),
                hot_exact_indices=(),
                cold_default_value=0,
            ),
        ],
        q_ledger_row=q_ledger,
        guard_spec=guard,
        global_cap_spec=GlobalRateCapSpec(cap=1, step=1),
        tensor_offsets={"cold.drift": 0},
        tensor_metadata_bits=0,
        bucket_metadata_bits=0,
        guardrail_metadata_bits=0,
    )

    assert report.ledger.claimable_physical_sub2 is True
    assert report.classification == BOUNDED_DELTA_GUARDRAIL_FAILED
    assert report.classification != BOUNDED_DELTA_WITH_REPORT
    assert report.guard_spec.max_candidate_changed_fraction == 0.0
    assert report.measured_report.candidate_changed_fraction == pytest.approx(1.0)
    assert "candidate_changed_fraction" in report.failed_metrics
    assert report.measured_report.max_abs_acc_error > 0
    payload = report.to_dict()
    assert "guard_spec" in payload
    assert "measured_report" in payload
    _assert_no_tensors(payload)


def test_oracle_comparison_uses_identical_votes_cap_backlog_and_offsets():
    q_ledger = _prior_large_q_ledger()
    numel = q_ledger.eligible_weight_count
    state = _state(numel, acc_overrides={3: 9, 4: 9})
    votes = _inputs(numel, {3: 2, 4: 2})
    backlog = {"oracle.parity": {4: {"first_step": 1, "last_deferred_step": 1, "defer_count": 1}}}

    report = compare_bounded_delta_step_to_int16_oracle(
        [
            BoundedDeltaOracleInput(
                state_key="oracle.parity",
                state=state,
                vote_inputs=votes,
                vote_spec=_spec(max_abs_per_tensor=4),
                hot_exact_indices=(3, 4),
            ),
        ],
        q_ledger_row=q_ledger,
        guard_spec=BoundedDeltaGuardSpec(),
        global_cap_spec=GlobalRateCapSpec(cap=1, step=2),
        deferred_backlog=backlog,
        tensor_offsets={"oracle.parity": 1024},
        tensor_metadata_bits=0,
        bucket_metadata_bits=0,
        guardrail_metadata_bits=0,
    )
    parity = report.measured_report.oracle_parity

    assert report.classification == BOUNDED_DELTA_WITH_REPORT
    assert parity["same_initial_q"] is True
    assert parity["same_votes_sha256"] is True
    assert parity["same_cap_spec"] is True
    assert parity["same_deferred_backlog"] is True
    assert parity["same_tensor_offsets"] is True
    assert parity["path_difference"] == "bounded path differs only by encode_decode_accumulator_loss"
    assert report.measured_report.accepted_changed_count == 0
    assert report.measured_report.deferred_changed_count == 0


def test_ledger_failure_is_not_reported_as_physical_sub2_success():
    q_ledger = _prior_large_q_ledger()
    numel = q_ledger.eligible_weight_count
    state = _state(numel)
    votes = _inputs(numel, {})

    report = compare_bounded_delta_step_to_int16_oracle(
        [
            BoundedDeltaOracleInput(
                state_key="ledger.fail",
                state=state,
                vote_inputs=votes,
                vote_spec=_spec(),
                hot_exact_indices=(),
            ),
        ],
        q_ledger_row=q_ledger,
        guard_spec=BoundedDeltaGuardSpec(),
        storage_projection=project_bounded_delta_accumulator_bpw(
            eligible_weight_count=numel,
            hot_exact_row_count=0,
            dense_cold_bits_per_weight=1.0,
        ),
    )

    assert report.guard_passed is True
    assert report.ledger.claimable_physical_sub2 is False
    assert report.classification == BOUNDED_DELTA_LEDGER_FAILED
    assert report.claimable_physical_sub2_with_guardrail is False


def test_event_coded_candidate_allows_non_decisive_candidate_mask_and_residual_drift():
    q_ledger = _prior_large_q_ledger()
    numel = q_ledger.eligible_weight_count
    state = _state(numel, acc_overrides={5: 9, 7: 9})
    votes = _inputs(numel, {5: 2, 7: 2})

    report = compare_bounded_delta_step_to_int16_oracle(
        [
            BoundedDeltaOracleInput(
                state_key="event.candidate",
                state=state,
                vote_inputs=votes,
                vote_spec=_spec(max_abs_per_tensor=1),
                hot_exact_indices=(5,),
                cold_default_value=0,
            ),
        ],
        q_ledger_row=q_ledger,
        candidate_name=EVENT_CODED_CROSSING_RESIDUAL_LOG_CANDIDATE,
        guard_spec=BoundedDeltaGuardSpec(max_candidate_changed_fraction=1.0),
        tensor_metadata_bits=0,
        bucket_metadata_bits=0,
        guardrail_metadata_bits=0,
    )

    assert report.classification == BOUNDED_DELTA_WITH_REPORT
    assert report.admission_passed is True
    assert report.measured_report.candidate_changed_fraction > 0.0
    assert report.measured_report.accepted_changed_fraction == pytest.approx(0.0)
    assert report.measured_report.q_changed_fraction == pytest.approx(0.0)
    assert report.measured_report.fired_or_accepted_residual_changed_count == 0
    assert report.measured_report.hot_residual_changed_count == 0
    surfaces = {
        item.surface: item for item in report.rejection_telemetry.surfaces
    }
    assert (
        surfaces["candidate_mask"].status == "allowed_non_decisive_divergence"
    )
    assert (
        surfaces["accumulator_residuals"].status
        == "allowed_non_fired_cold_residual_divergence"
    )
    assert report.rejection_telemetry.summary == "admission_pass"


def test_event_coded_candidate_rejects_when_accepted_row_residual_drifts():
    q_ledger = _prior_large_q_ledger()
    numel = q_ledger.eligible_weight_count
    state = _state(numel, acc_overrides={5: 11})
    votes = _inputs(numel, {})

    report = compare_bounded_delta_step_to_int16_oracle(
        [
            BoundedDeltaOracleInput(
                state_key="event.residual.drift",
                state=state,
                vote_inputs=votes,
                vote_spec=_spec(max_abs_per_tensor=4),
                hot_exact_indices=(),
                cold_default_value=0,
                cold_exception_indices=(5,),
                cold_exception_values=(13,),
            ),
        ],
        q_ledger_row=q_ledger,
        candidate_name=EVENT_CODED_CROSSING_RESIDUAL_LOG_CANDIDATE,
        guard_spec=BoundedDeltaGuardSpec(max_candidate_changed_fraction=1.0),
        tensor_metadata_bits=0,
        bucket_metadata_bits=0,
        guardrail_metadata_bits=0,
    )

    assert report.classification == BOUNDED_DELTA_ADMISSION_FAILED
    assert report.guard_passed is True
    assert report.admission_passed is False
    assert report.measured_report.accepted_changed_fraction == pytest.approx(0.0)
    assert report.measured_report.q_changed_fraction == pytest.approx(0.0)
    assert report.measured_report.fired_or_accepted_residual_changed_count == 1
    assert report.measured_report.hot_residual_changed_count == 0
    assert report.rejection_telemetry.summary == "fired_or_accepted_residual_drift"
    surfaces = {
        item.surface: item for item in report.rejection_telemetry.surfaces
    }
    assert surfaces["accumulator_residuals"].status == "fired_or_accepted_residual_drift"
    assert "fired/accepted rows" in surfaces["accumulator_residuals"].detail


def test_coarse_candidate_rejection_telemetry_flags_revisit_contract():
    q_ledger = _prior_large_q_ledger()
    numel = q_ledger.eligible_weight_count
    state = _state(numel, acc_overrides={5: 20, 6: 21, 7: 22})
    votes = _inputs(numel, {})

    report = compare_bounded_delta_step_to_int16_oracle(
        [
            BoundedDeltaOracleInput(
                state_key="coarse.contract",
                state=state,
                vote_inputs=votes,
                vote_spec=_spec(max_abs_per_tensor=3),
                hot_exact_indices=(),
                cold_default_value=0,
                cold_exception_indices=(5, 6, 7),
                cold_exception_values=(22, 21, 20),
            ),
        ],
        q_ledger_row=q_ledger,
        candidate_name=COARSE_SIGNED_CHARGE_SPARSE_FRONTIER_CANDIDATE,
        guard_spec=BoundedDeltaGuardSpec(max_cap_frontier_rank_delta=10),
        global_cap_spec=GlobalRateCapSpec(cap=3, step=1),
        tensor_offsets={"coarse.contract": 0},
        tensor_metadata_bits=0,
        bucket_metadata_bits=0,
        guardrail_metadata_bits=0,
        dense_cold_bits_per_weight=0.25,
    )

    assert report.classification == BOUNDED_DELTA_ADMISSION_FAILED
    assert report.guard_passed is True
    assert report.admission_passed is False
    assert report.rejection_telemetry.summary == "revisit_divergence_contract"
    assert report.admission_failed_surfaces == ("cap_frontier_rank_delta",)
    surfaces = {
        item.surface: item for item in report.rejection_telemetry.surfaces
    }
    assert surfaces["cap_frontier_rank_delta"].status == "revisit_divergence_contract"
    assert surfaces["accumulator_residuals"].status == "pass"
