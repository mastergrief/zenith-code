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
    ACCUMULATOR_FREE_NULL_BASELINE,
    BACKLOG_K_POLICIES,
    CANDIDATE_ADMISSION_DIAGNOSTIC_LABEL,
    CUMULATIVE_SCHEDULE_MODE,
    HOT_BUDGET_POINT_LABELS,
    ONE_STEP_LOCAL_DIAGNOSTIC_MODE,
    ORACLE_UPPER_BOUND_ADMISSION_DIAGNOSTIC,
    run_representative_bounded_delta_drift_verdict,
    run_candidate_admission_diagnostic,
    validate_candidate_admission_diagnostic_report,
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
