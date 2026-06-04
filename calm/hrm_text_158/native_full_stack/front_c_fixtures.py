"""Synthetic CPU fixtures for the Front-C projection scaffold."""
from __future__ import annotations

from calm.hrm_text_158.native_full_stack.front_c_projection import (
    FrontCDecisionPath,
    FrontCDecisionSurfaceStep,
)
from calm.hrm_text_158.native_full_stack.q_entropy_packing import (
    Base3QEntropyLedgerRow,
    default_base3_q_entropy_ledger_table,
)


def front_c_prior_large_q_ledger() -> Base3QEntropyLedgerRow:
    for row in default_base3_q_entropy_ledger_table():
        if row.regime_name == "prior_large_fixture_base3_q":
            return row
    raise RuntimeError("prior_large_fixture_base3_q ledger row is missing")


def front_c_timeline_churn_fixture(
    *,
    eligible_weight_count: int | None = None,
) -> tuple[FrontCDecisionSurfaceStep, ...]:
    eligible = (
        int(eligible_weight_count)
        if eligible_weight_count is not None
        else front_c_prior_large_q_ledger().eligible_weight_count
    )
    return (
        FrontCDecisionSurfaceStep(
            step=0,
            eligible_weight_count=eligible,
            current_magnitude_threshold_keys=(("fixture", 5),),
            active_next_step_keys=(("fixture", 5), ("fixture", 7)),
            ranking_sensitive_exact_keys=(("fixture", 5), ("fixture", 7)),
            global_cap_frontier_keys=(("fixture", 5),),
            backlog_carry_keys=(),
            replay_veto_residual_keys=(("fixture", 9),),
        ),
        FrontCDecisionSurfaceStep(
            step=1,
            eligible_weight_count=eligible,
            current_magnitude_threshold_keys=(("fixture", 11),),
            active_next_step_keys=(("fixture", 7), ("fixture", 11)),
            ranking_sensitive_exact_keys=(("fixture", 7), ("fixture", 11)),
            global_cap_frontier_keys=(("fixture", 11),),
            backlog_carry_keys=(("fixture", 13),),
            replay_veto_residual_keys=(),
        ),
    )


def front_c_overhead_failure_timeline_fixture(
    *,
    eligible_weight_count: int | None = None,
    stored_row_count: int = 390,
) -> tuple[FrontCDecisionSurfaceStep, ...]:
    eligible = (
        int(eligible_weight_count)
        if eligible_weight_count is not None
        else front_c_prior_large_q_ledger().eligible_weight_count
    )
    return (
        FrontCDecisionSurfaceStep(
            step=0,
            eligible_weight_count=eligible,
            active_next_step_keys=(("overhead", index) for index in range(int(stored_row_count))),
            ranking_sensitive_exact_keys=(("overhead", index) for index in range(int(stored_row_count))),
        ),
    )


def front_c_zero_drift_decision_paths() -> tuple[FrontCDecisionPath, FrontCDecisionPath]:
    dense = FrontCDecisionPath(
        label="dense_int16_fixture",
        q_flip_directions=(("fixture", 5, 1), ("fixture", 7, -1)),
        accepted_under_global_cap_keys=(("fixture", 5),),
        deferred_under_global_cap_keys=(("fixture", 7),),
        backlog_keys=(("fixture", 13),),
        replay_veto_decision_keys=(("fixture", 9),),
    )
    sparse = FrontCDecisionPath(
        label="sparse_acc_fixture",
        q_flip_directions=(("fixture", 5, 1), ("fixture", 7, -1)),
        accepted_under_global_cap_keys=(("fixture", 5),),
        deferred_under_global_cap_keys=(("fixture", 7),),
        backlog_keys=(("fixture", 13),),
        replay_veto_decision_keys=(("fixture", 9),),
    )
    return dense, sparse


def front_c_count_only_timeline_artifact(
    *,
    eligible_weight_count: int | None = None,
) -> dict[str, int]:
    eligible = (
        int(eligible_weight_count)
        if eligible_weight_count is not None
        else front_c_prior_large_q_ledger().eligible_weight_count
    )
    return {
        "step": 0,
        "eligible_weight_count": eligible,
        "decision_relevant_exact_count": 3,
    }
