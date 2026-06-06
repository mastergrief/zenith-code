"""C1.1c representative bounded-delta drift-vs-budget verdict tests."""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import pytest
import torch

from calm.hrm_text_158.native_full_stack.bounded_delta_accumulator import (
    BOUNDED_DELTA_GUARDRAIL_FAILED,
    BOUNDED_DELTA_LEDGER_FAILED,
    BOUNDED_DELTA_WITH_REPORT,
    BoundedDeltaGuardSpec,
    BoundedDeltaOracleInput,
    COARSE_SIGNED_CHARGE_SPARSE_FRONTIER_CANDIDATE,
    EVENT_CODED_CROSSING_RESIDUAL_LOG_CANDIDATE,
    HYBRID_HOT_EXACT_COLD_DEFAULT_CANDIDATE,
    compare_bounded_delta_step_to_int16_oracle,
    project_bounded_delta_accumulator_bpw,
)
from calm.hrm_text_158.native_full_stack.bounded_delta_representative_verdict import (
    ABSOLUTE_COUNT_LOWER_BOUND_DIAGNOSTIC,
    ACCUMULATOR_FREE_NULL_BASELINE,
    A_COLD_EXCEPTION_BUDGET_LEVER_LABEL,
    A_FUNDAMENTALLY_OVER_LABEL,
    BACKLOG_K_POLICIES,
    CAPACITY_LOCALIZATION_DIAGNOSTIC_LABEL,
    CANDIDATE_ADMISSION_DIAGNOSTIC_LABEL,
    CUMULATIVE_SCHEDULE_MODE,
    DECISION_STATISTIC_UPPER_BOUND_LABEL,
    HOT_BUDGET_POINT_LABELS,
    K_SWEEP_JOINT_INFEASIBLE,
    K_SWEEP_MINIMAL_VIABLE_PASS,
    K_SWEEP_REPRESENTATION_WALL,
    OBSERVABLE_RANK_FEATURES_INSUFFICIENT,
    ONE_STEP_LOCAL_DIAGNOSTIC_MODE,
    ORACLE_UPPER_BOUND_ADMISSION_DIAGNOSTIC,
    PER_ROW_COMPRESSION_CLOSED_TINY_FIXTURE_LOWER_BOUND_ONLY,
    RATE_HELD_B_STILL_OVER_SCALE_HEADROOM_CANDIDATE_BRANCH_A,
    RATE_HELD_B_STORAGE_DIAGNOSTIC,
    RATE_HELD_COUNT_ROUNDING_POLICY,
    REAL_BACKLOG_LOWER_BOUND_LABEL,
    REPRESENTATIVE_TRACE_UNDERPOWERED_FOR_CLOSURE,
    SCALE_APPROPRIATE_B_STORAGE_LABEL,
    SCALE_APPROPRIATE_COMPARISON_AMBIGUOUS_NEEDS_BACKLOG_DENSITY_TRACE,
    SPARSE_AMORTIZED_CANDIDATE_RESURRECTED_FOR_HARDER_TRACE,
    TINY_FIXTURE_HEADROOM_SOURCE,
    VIRTUAL_DECISION_STATISTIC_CANDIDATE,
    run_candidate_admission_diagnostic,
    run_candidate_capacity_localization_diagnostic,
    run_decision_statistic_upper_bound_diagnostic,
    run_real_backlog_lower_bound_diagnostic,
    run_scale_appropriate_b_storage_comparison,
    run_representative_bounded_delta_drift_verdict,
    validate_candidate_admission_diagnostic_report,
    validate_candidate_capacity_localization_report,
    validate_decision_statistic_upper_bound_report,
    validate_real_backlog_lower_bound_diagnostic_report,
    validate_scale_appropriate_b_storage_comparison_report,
    validate_representative_bounded_delta_drift_verdict_report,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap import GlobalRateCapSpec
from calm.hrm_text_158.native_full_stack.q_entropy_packing import (
    default_base3_q_entropy_ledger_table,
)
from calm.hrm_text_158.native_full_stack.vote_update import (
    VoteUpdateInputs,
    VoteUpdateSpec,
    VoteUpdateState,
)


def _prior_large_q_ledger():
    for row in default_base3_q_entropy_ledger_table():
        if row.regime_name == "prior_large_fixture_base3_q":
            return row
    raise AssertionError("prior_large_fixture_base3_q missing")


def _state(numel: int, *, acc_overrides: dict[int, int] | None = None) -> VoteUpdateState:
    q = torch.zeros(numel, dtype=torch.int8)
    acc = torch.zeros(numel, dtype=torch.int16)
    for index, value in (acc_overrides or {}).items():
        acc[int(index)] = int(value)
    return VoteUpdateState(q_levels=q, accumulators=acc)


def _inputs(numel: int, votes: dict[int, int]) -> VoteUpdateInputs:
    out = torch.zeros(numel, dtype=torch.int16)
    for index, value in votes.items():
        out[int(index)] = int(value)
    return VoteUpdateInputs(votes=out)


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


def _assert_no_tensors(value: Any) -> None:
    if isinstance(value, torch.Tensor):
        raise AssertionError("compact report must not include raw tensors")
    if isinstance(value, dict):
        for child in value.values():
            _assert_no_tensors(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _assert_no_tensors(child)


@lru_cache(maxsize=1)
def _representative_report():
    return run_representative_bounded_delta_drift_verdict()


@lru_cache(maxsize=1)
def _candidate_admission_report():
    return run_candidate_admission_diagnostic()


@lru_cache(maxsize=1)
def _capacity_localization_report():
    return run_candidate_capacity_localization_diagnostic()


@lru_cache(maxsize=1)
def _real_backlog_lower_bound_report():
    return run_real_backlog_lower_bound_diagnostic()


@lru_cache(maxsize=1)
def _scale_appropriate_b_storage_report():
    return run_scale_appropriate_b_storage_comparison()


@lru_cache(maxsize=1)
def _decision_statistic_upper_bound_report():
    return run_decision_statistic_upper_bound_diagnostic()


def test_bounded_backlog_policy_is_opt_in_and_charged_as_actual_stored_backlog():
    q_ledger = _prior_large_q_ledger()
    numel = q_ledger.eligible_weight_count
    state = _state(numel, acc_overrides={3: 9, 4: 9})
    votes = _inputs(numel, {3: 2, 4: 2})
    exact_backlog = {
        "backlog.policy": {4: {"first_step": 1, "last_deferred_step": 1, "defer_count": 1}}
    }
    bounded_stored = {
        "backlog.policy": {4: {"first_step": 2, "last_deferred_step": 2, "defer_count": 1}}
    }

    report = compare_bounded_delta_step_to_int16_oracle(
        [
            BoundedDeltaOracleInput(
                state_key="backlog.policy",
                state=state,
                vote_inputs=votes,
                vote_spec=_spec(max_abs_per_tensor=4),
                hot_exact_indices=(3, 4),
            )
        ],
        q_ledger_row=q_ledger,
        guard_spec=BoundedDeltaGuardSpec(max_backlog_key_changed_fraction=1.0),
        global_cap_spec=GlobalRateCapSpec(cap=1, step=2),
        deferred_backlog=exact_backlog,
        bounded_deferred_backlog={},
        bounded_stored_deferred_backlog=bounded_stored,
        tensor_offsets={"backlog.policy": 0},
        storage_projection=project_bounded_delta_accumulator_bpw(
            eligible_weight_count=numel,
            hot_exact_row_count=2,
            backlog_entry_count=1,
            tensor_metadata_bits=0,
            bucket_metadata_bits=0,
            guardrail_metadata_bits=0,
        ),
    )
    parity = report.measured_report.oracle_parity

    assert parity["bounded_backlog_policy_active"] is True
    assert parity["same_deferred_backlog"] is False
    assert parity["exact_input_deferred_backlog_count"] == 1
    assert parity["bounded_input_deferred_backlog_count"] == 0
    assert parity["bounded_stored_deferred_backlog_count"] == 1
    assert "bounded_backlog_encode_drop" in parity["path_difference"]
    assert report.measured_report.backlog_key_union_count == 1
    assert report.storage_projection.backlog_entry_count == 1
    assert report.ledger.claimable_physical_sub2 is True

    with pytest.raises(ValueError, match="actual bounded stored backlog"):
        compare_bounded_delta_step_to_int16_oracle(
            [
                BoundedDeltaOracleInput(
                    state_key="backlog.policy",
                    state=state,
                    vote_inputs=votes,
                    vote_spec=_spec(max_abs_per_tensor=4),
                    hot_exact_indices=(3, 4),
                )
            ],
            q_ledger_row=q_ledger,
            guard_spec=BoundedDeltaGuardSpec(max_backlog_key_changed_fraction=1.0),
            global_cap_spec=GlobalRateCapSpec(cap=1, step=2),
            deferred_backlog=exact_backlog,
            bounded_deferred_backlog={},
            bounded_stored_deferred_backlog=bounded_stored,
            tensor_offsets={"backlog.policy": 0},
            storage_projection=project_bounded_delta_accumulator_bpw(
                eligible_weight_count=numel,
                hot_exact_row_count=2,
                backlog_entry_count=0,
                tensor_metadata_bits=0,
                bucket_metadata_bits=0,
                guardrail_metadata_bits=0,
            ),
        )


def test_bounded_stored_backlog_must_be_subset_of_actual_bounded_output():
    q_ledger = _prior_large_q_ledger()
    numel = q_ledger.eligible_weight_count
    state = _state(numel, acc_overrides={3: 9, 4: 9})
    votes = _inputs(numel, {3: 2, 4: 2})

    with pytest.raises(ValueError, match="invented_or_stale_ids"):
        compare_bounded_delta_step_to_int16_oracle(
            [
                BoundedDeltaOracleInput(
                    state_key="backlog.policy",
                    state=state,
                    vote_inputs=votes,
                    vote_spec=_spec(max_abs_per_tensor=4),
                    hot_exact_indices=(3, 4),
                )
            ],
            q_ledger_row=q_ledger,
            guard_spec=BoundedDeltaGuardSpec(max_backlog_key_changed_fraction=1.0),
            global_cap_spec=GlobalRateCapSpec(cap=1, step=2),
            bounded_deferred_backlog={},
            bounded_stored_deferred_backlog={
                "backlog.policy": {
                    9: {"first_step": 2, "last_deferred_step": 2, "defer_count": 1}
                }
            },
            tensor_offsets={"backlog.policy": 0},
            storage_projection=project_bounded_delta_accumulator_bpw(
                eligible_weight_count=numel,
                hot_exact_row_count=2,
                backlog_entry_count=1,
                tensor_metadata_bits=0,
                bucket_metadata_bits=0,
                guardrail_metadata_bits=0,
            ),
        )


def test_bounded_backlog_union_count_uses_charged_stored_identities():
    q_ledger = _prior_large_q_ledger()
    numel = q_ledger.eligible_weight_count
    state = _state(numel, acc_overrides={11: -9})
    votes = _inputs(numel, {11: 18})

    report = compare_bounded_delta_step_to_int16_oracle(
        [
            BoundedDeltaOracleInput(
                state_key="backlog.policy",
                state=state,
                vote_inputs=votes,
                vote_spec=_spec(max_abs_per_tensor=4),
                hot_exact_indices=(),
            )
        ],
        q_ledger_row=q_ledger,
        guard_spec=BoundedDeltaGuardSpec(
            max_candidate_changed_fraction=1.0,
            max_deferred_changed_fraction=1.0,
            max_backlog_key_changed_fraction=1.0,
            max_cap_frontier_rank_delta=999,
            hot_risk_rows_require_zero_drift=False,
        ),
        global_cap_spec=GlobalRateCapSpec(cap=0, step=2),
        bounded_deferred_backlog={},
        bounded_stored_deferred_backlog={},
        tensor_offsets={"backlog.policy": 0},
        storage_projection=project_bounded_delta_accumulator_bpw(
            eligible_weight_count=numel,
            hot_exact_row_count=0,
            backlog_entry_count=0,
            tensor_metadata_bits=0,
            bucket_metadata_bits=0,
            guardrail_metadata_bits=0,
        ),
    )
    parity = report.measured_report.oracle_parity

    assert parity["bounded_backlog_policy_active"] is True
    assert parity["bounded_stored_deferred_backlog_count"] == 0
    assert report.measured_report.backlog_key_changed_count == 0
    assert report.measured_report.backlog_key_union_count == 0


def test_representative_verdict_is_cumulative_and_compact():
    report = _representative_report()
    payload = report.to_dict()

    validate_representative_bounded_delta_drift_verdict_report(report)
    assert report.terminal_mode == CUMULATIVE_SCHEDULE_MODE
    assert report.terminal_science_question_closed is True
    assert report.raw_arrays_included is False
    assert report.source_bindingness.primary_bindingness == "binding_for_in_tree_native_loop_distribution"
    assert report.source_bindingness.s1_bindingness == "partial_for_s1_real_dynamics"
    assert "residual accumulator-value diversity" in report.residual_diversity_caveat
    assert "C2" in report.guard_bound_adequacy_statement
    _assert_no_tensors(payload)


def test_strict_same_backlog_control_reports_hard_regime_ledger_failure():
    report = _representative_report()
    by_name = {step.schedule_name: step for step in report.one_step_local_diagnostic_reports}

    assert by_name["cap_saturated"].mode == ONE_STEP_LOCAL_DIAGNOSTIC_MODE
    assert by_name["cap_saturated"].bounded_reinitialized_from_exact is True
    assert by_name["cap_saturated"].classification == BOUNDED_DELTA_LEDGER_FAILED
    assert by_name["cap_saturated"].guard_passed is True
    assert by_name["cap_saturated"].exact_backlog_entry_count > 0
    assert by_name["backlog_growth"].classification == BOUNDED_DELTA_LEDGER_FAILED
    assert by_name["backlog_growth"].exact_backlog_entry_count > (
        by_name["cap_saturated"].exact_backlog_entry_count
    )


def test_cumulative_curve_pre_registers_hot_budgets_and_k_policies():
    report = _representative_report()
    labels = {
        (run.hot_budget_label, run.backlog_policy_k)
        for run in report.cumulative_curve_reports
    }

    assert report.hot_budget_points == HOT_BUDGET_POINT_LABELS
    assert report.backlog_k_policies == BACKLOG_K_POLICIES
    assert labels == {
        (hot_label, backlog_k)
        for backlog_k in BACKLOG_K_POLICIES
        for hot_label in HOT_BUDGET_POINT_LABELS
    }
    for run in report.cumulative_curve_reports:
        assert run.mode == CUMULATIVE_SCHEDULE_MODE
        assert len(run.per_step_reports) == 4
        for step in run.per_step_reports:
            assert step.bounded_reinitialized_from_exact is False
            assert step.bounded_delta_report.measured_report.oracle_parity[
                "cumulative_carry_forward"
            ] is True


def test_primary_cumulative_curve_charges_bounded_backlog_and_reports_raw_drift():
    report = _representative_report()
    primary = next(run for run in report.cumulative_curve_reports if run.curve_label == report.primary_curve_label)
    terminal = primary.per_step_reports[-1]
    bounded_report = terminal.bounded_delta_report
    parity = bounded_report.measured_report.oracle_parity

    assert primary.hot_budget_label == "hotmax"
    assert primary.backlog_policy_k == 32
    assert report.terminal_classification == primary.terminal_classification
    assert report.terminal_classification in {
        BOUNDED_DELTA_WITH_REPORT,
        BOUNDED_DELTA_GUARDRAIL_FAILED,
        BOUNDED_DELTA_LEDGER_FAILED,
    }
    assert terminal.exact_backlog_entry_count > terminal.bounded_stored_backlog_entry_count
    assert terminal.bounded_stored_backlog_entry_count <= 32
    assert bounded_report.storage_projection.backlog_entry_count == (
        terminal.bounded_stored_backlog_entry_count
    )
    assert bounded_report.ledger.claimable_physical_sub2 is True
    assert parity["bounded_backlog_policy_active"] is True
    assert parity["same_deferred_backlog"] is False
    assert bounded_report.measured_report.candidate_union_count >= 0
    assert bounded_report.measured_report.accepted_union_count >= 0
    assert bounded_report.measured_report.backlog_key_union_count > 0


def test_candidate_admission_diagnostic_is_null_anchored_and_oracle_upper_bound():
    report = _candidate_admission_report()
    payload = report.to_dict()

    validate_candidate_admission_diagnostic_report(report)
    assert report.label == CANDIDATE_ADMISSION_DIAGNOSTIC_LABEL
    assert report.null_baseline_label == ACCUMULATOR_FREE_NULL_BASELINE
    assert "upper bound" in " ".join(report.non_claims)
    by_name = {run.candidate_name: run for run in report.candidate_runs}

    assert set(by_name) == {
        HYBRID_HOT_EXACT_COLD_DEFAULT_CANDIDATE,
        EVENT_CODED_CROSSING_RESIDUAL_LOG_CANDIDATE,
        COARSE_SIGNED_CHARGE_SPARSE_FRONTIER_CANDIDATE,
    }
    for run in report.candidate_runs:
        assert run.builder_label == ORACLE_UPPER_BOUND_ADMISSION_DIAGNOSTIC
        assert len(run.per_step_reports) == 4
        assert run.terminal_decision.oracle_upper_bound_only is True
        for step in run.per_step_reports:
            assert step.builder_label == ORACLE_UPPER_BOUND_ADMISSION_DIAGNOSTIC
            assert step.null_baseline_comparison.compared_surfaces == (
                "accepted_rows",
                "deferred_rows",
                "final_q_changes",
                "backlog_carry",
            )
            assert step.bounded_delta_report.measured_report.oracle_parity["builder_label"] == (
                ORACLE_UPPER_BOUND_ADMISSION_DIAGNOSTIC
            )
    assert (
        by_name[HYBRID_HOT_EXACT_COLD_DEFAULT_CANDIDATE]
        .per_step_reports[-1]
        .backlog_truncation_attribution
        .bounded_stored_truncation_count
        == 0
    )
    assert (
        by_name[EVENT_CODED_CROSSING_RESIDUAL_LOG_CANDIDATE]
        .per_step_reports[-1]
        .backlog_truncation_attribution
        .bounded_stored_truncation_count
        > 0
    )
    assert (
        by_name[COARSE_SIGNED_CHARGE_SPARSE_FRONTIER_CANDIDATE]
        .per_step_reports[-1]
        .backlog_truncation_attribution
        .bounded_stored_truncation_count
        > 0
    )
    _assert_no_tensors(payload)


def test_candidate_capacity_localization_reports_a_budget_direction_and_bc_k_sweeps():
    report = _capacity_localization_report()
    payload = report.to_dict()

    validate_candidate_capacity_localization_report(report)
    assert report.label == CAPACITY_LOCALIZATION_DIAGNOSTIC_LABEL
    assert report.backlog_k_schedule[-1] == "unbounded"
    assert report.candidate_a_budget_report.terminal_budget_direction_label == (
        A_FUNDAMENTALLY_OVER_LABEL
    )
    assert (
        report.candidate_a_budget_report.original_terminal_rejection_summary
        == "exact_surface_miss"
    )
    assert "surface-faithful tighter-cold encoding" in report.candidate_a_budget_report.non_claim
    assert len(report.candidate_a_budget_report.per_step_readouts) == 4
    terminal_a = report.candidate_a_budget_report.per_step_readouts[-1]
    assert terminal_a.packed_inclusive_physical_bits_per_weight > 2.0
    assert terminal_a.cold_zero_counterfactual_bits_per_weight > 2.0
    assert terminal_a.cold_zero_counterfactual_bits_per_weight == pytest.approx(
        terminal_a.packed_inclusive_physical_bits_per_weight,
    )
    for step in report.candidate_a_budget_report.per_step_readouts:
        assert step.cold_zero_counterfactual_bits_per_weight <= step.packed_inclusive_physical_bits_per_weight

    by_name = {run.candidate_name: run for run in report.sweep_runs}
    assert by_name[EVENT_CODED_CROSSING_RESIDUAL_LOG_CANDIDATE].terminal_decision.status == (
        K_SWEEP_JOINT_INFEASIBLE
    )
    assert by_name[EVENT_CODED_CROSSING_RESIDUAL_LOG_CANDIDATE].terminal_decision.decisive_k_label == (
        "4096"
    )
    assert by_name[EVENT_CODED_CROSSING_RESIDUAL_LOG_CANDIDATE].terminal_decision.decisive_k_value == 4096
    assert by_name[COARSE_SIGNED_CHARGE_SPARSE_FRONTIER_CANDIDATE].terminal_decision.status == (
        K_SWEEP_REPRESENTATION_WALL
    )
    assert by_name[COARSE_SIGNED_CHARGE_SPARSE_FRONTIER_CANDIDATE].terminal_decision.decisive_k_label == (
        "unbounded"
    )
    assert by_name[COARSE_SIGNED_CHARGE_SPARSE_FRONTIER_CANDIDATE].terminal_decision.decisive_k_value is None

    for run in report.sweep_runs:
        assert run.terminal_decision.status in {
            K_SWEEP_MINIMAL_VIABLE_PASS,
            K_SWEEP_JOINT_INFEASIBLE,
            K_SWEEP_REPRESENTATION_WALL,
        }
        assert run.sweep_entries[0].k_label == "32"
        assert run.sweep_entries[-1].k_label == "unbounded"
        for entry in run.sweep_entries:
            assert len(entry.per_step_reports) == 4
        for step in run.sweep_entries[-1].per_step_reports:
            assert step.backlog_truncation_attribution.bounded_input_truncation_count == 0
            assert step.backlog_truncation_attribution.bounded_stored_truncation_count == 0
            assert step.bounded_delta_report.measured_report.oracle_parity["builder_label"] == (
                ORACLE_UPPER_BOUND_ADMISSION_DIAGNOSTIC
            )
    _assert_no_tensors(payload)


def test_real_backlog_lower_bound_reports_trace_strength_and_headroom_comparison():
    report = _real_backlog_lower_bound_report()
    payload = report.to_dict()

    validate_real_backlog_lower_bound_diagnostic_report(report)
    assert report.label == REAL_BACKLOG_LOWER_BOUND_LABEL
    assert report.candidate_name == EVENT_CODED_CROSSING_RESIDUAL_LOG_CANDIDATE
    assert report.terminal_decision.terminal_label == (
        PER_ROW_COMPRESSION_CLOSED_TINY_FIXTURE_LOWER_BOUND_ONLY
    )
    assert report.terminal_decision.headroom_source == TINY_FIXTURE_HEADROOM_SOURCE
    assert report.backlog_k_schedule[-1] == "unbounded"
    assert len(report.exact_trace_summary.per_step_reports) == 1
    assert report.exact_trace_summary.stop_reason == "nontrivial_backlog_reached"
    assert report.exact_trace_summary.nontrivial_backlog_reached is True
    assert report.exact_trace_summary.plateau_detected is False
    assert report.terminal_decision.eligible_weight_count == 160
    assert report.terminal_decision.q_packed_data_bits_per_weight == pytest.approx(2.0)
    assert report.terminal_decision.q_packed_metadata_bits_per_weight == pytest.approx(3.2)
    assert report.terminal_decision.q_packed_total_bits_per_weight == pytest.approx(5.2)
    assert report.terminal_decision.frozen_scale_fp32_bits_per_weight == pytest.approx(0.4)
    assert (
        report.terminal_decision.actual_remaining_accumulator_headroom_bits_per_weight
        == pytest.approx(-3.6)
    )
    assert report.terminal_decision.minimal_surface_faithful_k_label == "5"
    assert report.terminal_decision.minimal_surface_faithful_k_value == 5
    assert (
        report.terminal_decision.minimal_surface_faithful_peak_bounded_delta_acc_bits_per_weight
        == pytest.approx(3.5)
    )
    assert (
        report.terminal_decision.headroom_minus_minimal_surface_faithful_peak_bits_per_weight
        == pytest.approx(-7.1)
    )
    assert report.terminal_decision.minimal_surface_faithful_k_fits_headroom is False
    assert report.terminal_decision.global_per_row_compression_closed is False
    assert report.terminal_decision.branch_a_trigger is False
    assert report.sweep_entries[0].k_label == report.backlog_k_schedule[0]
    assert report.sweep_entries[-1].k_label == "unbounded"
    for entry in report.sweep_entries:
        assert len(entry.per_step_reports) == len(report.exact_trace_summary.per_step_reports)
        for step in entry.per_step_reports:
            assert step.measured_report.oracle_parity["builder_label"] == (
                REAL_BACKLOG_LOWER_BOUND_LABEL
            )
    _assert_no_tensors(payload)


def test_scale_appropriate_b_storage_uses_rate_held_rows_for_the_decision():
    report = _scale_appropriate_b_storage_report()
    payload = report.to_dict()

    validate_scale_appropriate_b_storage_comparison_report(report)
    assert report.label == SCALE_APPROPRIATE_B_STORAGE_LABEL
    assert report.candidate_name == EVENT_CODED_CROSSING_RESIDUAL_LOG_CANDIDATE
    assert report.source_terminal_label == PER_ROW_COMPRESSION_CLOSED_TINY_FIXTURE_LOWER_BOUND_ONLY
    assert report.source_minimal_surface_faithful_k_label == "5"
    assert report.source_minimal_surface_faithful_k_value == 5
    assert report.source_tiny_eligible_weight_count == 160
    assert report.density_rounding_policy == RATE_HELD_COUNT_ROUNDING_POLICY
    assert report.terminal_decision.terminal_label == (
        RATE_HELD_B_STILL_OVER_SCALE_HEADROOM_CANDIDATE_BRANCH_A
    )
    assert report.terminal_decision.candidate_branch_a_trigger_earned is True
    assert report.terminal_decision.branch_a_trigger is False
    assert report.terminal_decision.global_per_row_compression_closed is False
    assert report.terminal_decision.required_rows_all_rate_held_exceed_scale_headroom is True
    assert report.terminal_decision.rate_held_density_assumption_explicit is True

    by_name = {row.q_regime_name: row for row in report.row_comparisons}
    assert set(report.required_q_ledger_rows) == {
        "prior_large_fixture_base3_q",
        "illustrative_4096x4096_one_tensor_one_scale_base3_q",
    }
    assert set(report.sensitivity_q_ledger_rows) == {
        "illustrative_4096x4096_one_tensor_per_row_scale_base3_q",
    }
    for name in report.required_q_ledger_rows:
        row = by_name[name]
        assert row.row_role == "required_gate"
        assert row.rate_held_b_storage_exceeds_scale_headroom is True
        assert row.rate_held_b_storage_peak_bounded_delta_acc_bits_per_weight > (
            row.scale_appropriate_headroom_bits_per_weight
        )
        assert row.absolute_count_lower_bound_peak_bounded_delta_acc_bits_per_weight < (
            row.rate_held_b_storage_peak_bounded_delta_acc_bits_per_weight
        )
        assert row.absolute_count_lower_bound_step_reports[0].projection_label == (
            ABSOLUTE_COUNT_LOWER_BOUND_DIAGNOSTIC
        )
        assert row.absolute_count_lower_bound_step_reports[0].decisive_for_branch is False
        assert row.rate_held_b_storage_step_reports[0].projection_label == (
            RATE_HELD_B_STORAGE_DIAGNOSTIC
        )
        assert row.rate_held_b_storage_step_reports[0].decisive_for_branch is True
        assert row.rate_held_b_storage_step_reports[0].rounding_policy == (
            RATE_HELD_COUNT_ROUNDING_POLICY
        )
        assert "hot=2/160" in row.density_assumption
        assert "event=2/160" in row.density_assumption
        assert "backlog=5/160" in row.density_assumption
    prior_large = by_name["prior_large_fixture_base3_q"]
    assert prior_large.scale_appropriate_headroom_bits_per_weight == pytest.approx(
        0.38232421875
    )
    assert prior_large.absolute_count_lower_bound_peak_bounded_delta_acc_bits_per_weight == pytest.approx(
        0.0335693359375
    )
    assert prior_large.rate_held_b_storage_peak_bounded_delta_acc_bits_per_weight == pytest.approx(
        2.25
    )
    assert prior_large.rate_held_b_storage_step_reports[0].target_hot_exact_row_count == 205
    assert prior_large.rate_held_b_storage_step_reports[0].target_event_delta_count == 205
    assert prior_large.rate_held_b_storage_step_reports[0].target_backlog_entry_count == 512
    one_scale_4096 = by_name["illustrative_4096x4096_one_tensor_one_scale_base3_q"]
    assert one_scale_4096.scale_appropriate_headroom_bits_per_weight == pytest.approx(
        0.3999824523925781
    )
    assert one_scale_4096.absolute_count_lower_bound_peak_bounded_delta_acc_bits_per_weight == pytest.approx(
        3.814697265625e-05
    )
    assert one_scale_4096.rate_held_b_storage_peak_bounded_delta_acc_bits_per_weight == pytest.approx(
        2.800015449523926
    )
    assert one_scale_4096.rate_held_b_storage_step_reports[0].target_hot_exact_row_count == 209716
    assert one_scale_4096.rate_held_b_storage_step_reports[0].target_event_delta_count == 209716
    assert one_scale_4096.rate_held_b_storage_step_reports[0].target_backlog_entry_count == 524288
    sensitivity = by_name["illustrative_4096x4096_one_tensor_per_row_scale_base3_q"]
    assert sensitivity.row_role == "sensitivity_only"
    assert sensitivity.scale_appropriate_headroom_bits_per_weight == pytest.approx(
        0.39217185974121094
    )
    assert sensitivity.rate_held_b_storage_peak_bounded_delta_acc_bits_per_weight == pytest.approx(
        2.800015449523926
    )
    _assert_no_tensors(payload)


def test_decision_statistic_upper_bound_fits_headroom_but_fails_shuffle_falsifier():
    report = _decision_statistic_upper_bound_report()
    payload = report.to_dict()

    validate_decision_statistic_upper_bound_report(report)
    assert report.label == DECISION_STATISTIC_UPPER_BOUND_LABEL
    assert report.candidate_name == VIRTUAL_DECISION_STATISTIC_CANDIDATE
    assert report.source_scale_comparison_label == SCALE_APPROPRIATE_B_STORAGE_LABEL
    assert report.source_scale_terminal_label == (
        RATE_HELD_B_STILL_OVER_SCALE_HEADROOM_CANDIDATE_BRANCH_A
    )
    assert report.strictest_required_q_regime_name == "prior_large_fixture_base3_q"
    assert report.strictest_required_eligible_weight_count == 16384
    assert report.strictest_required_headroom_bits_per_weight == pytest.approx(
        0.38232421875
    )
    assert report.terminal_decision.terminal_label == OBSERVABLE_RANK_FEATURES_INSUFFICIENT
    assert report.terminal_decision.budget_fits_strictest_required_headroom is True
    assert report.terminal_decision.inclusive_sub2_if_installed is True
    assert report.terminal_decision.first_budget_failure_step is None
    assert report.terminal_decision.first_insufficient_step == "cap_saturated"
    assert report.terminal_decision.peak_statistic_step == "backlog_growth"
    assert report.terminal_decision.peak_statistic_bits_per_weight == pytest.approx(
        0.010009765625
    )
    assert report.terminal_decision.any_step_frontier_tie_crosses_boundary is True
    assert report.terminal_decision.all_steps_shuffle_preserve_outcome is False

    by_name = {step.schedule_name: step for step in report.step_reports}
    assert by_name["sparse_unsaturated"].observable_rank_features_sufficient is True
    assert by_name["moderate_unsaturated"].observable_rank_features_sufficient is True

    cap_saturated = by_name["cap_saturated"]
    assert cap_saturated.candidate_row_count == 1536
    assert cap_saturated.accepted_row_count == 256
    assert cap_saturated.deferred_row_count == 1280
    assert cap_saturated.statistic_schema.total_bits == 160
    assert cap_saturated.statistic_schema.total_bits_per_weight_strictest_required_row == pytest.approx(
        0.009765625
    )
    assert cap_saturated.statistic_schema.fits_strictest_required_headroom is True
    assert cap_saturated.frontier_tie_bucket_count == 2
    assert cap_saturated.canonical_matches_exact is True
    assert cap_saturated.shuffled_matches_exact is False
    assert cap_saturated.shuffle_preserves_outcome is False
    assert "row-identity order" in (cap_saturated.insufficiency_reason or "")
    cap_buckets = {
        (bucket.state_key, bucket.current_q_level, bucket.move_direction): bucket
        for bucket in cap_saturated.bucket_summaries
    }
    assert cap_buckets[("proj_in", 0, -1)].frontier_tie_crosses_boundary is True
    assert cap_buckets[("proj_in", 0, 1)].frontier_tie_crosses_boundary is True
    assert cap_buckets[("proj_out", 0, -1)].accepted_count == 0
    assert cap_buckets[("proj_out", 0, 1)].accepted_count == 0

    backlog_growth = by_name["backlog_growth"]
    assert backlog_growth.candidate_row_count == 2816
    assert backlog_growth.statistic_schema.total_bits == 164
    assert backlog_growth.statistic_schema.total_bits_per_weight_strictest_required_row == pytest.approx(
        0.010009765625
    )
    assert backlog_growth.observable_rank_features_sufficient is False
    _assert_no_tensors(payload)
