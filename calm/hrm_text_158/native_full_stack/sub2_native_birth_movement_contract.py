"""Pure Step-2 movement contract/report helpers.

This module is intentionally contract-only. It owns the movement-first report
and packet-facing contract semantics for the hybrid persistent-sub2 lane1 path.
It must stay free of harness/runtime/file I/O dependencies.
"""
from __future__ import annotations

from typing import Any, Mapping

from calm.hrm_text_158.native_full_stack.sub2_native_birth_scaffold import (
    DENSE_TRANSIENT_CREDIT_ROLE_TRAINING_COMPUTE_CONTROL_ONLY,
    HYBRID_MOVEMENT_CONTRACT_SCOPE,
    HYBRID_MOVEMENT_METRIC_NAME,
    HYBRID_PERSISTENT_MODE_APPLIED_CROSSING_DIRECTION_PLUS_4BIT_RESIDUAL,
    RUNTIME_STATE_AUTHORITY_SUB2_PERSISTENT_HYBRID_DENSE_TRANSIENT_CREDIT,
)


TIERB_LANE1_HYBRID_MOVEMENT_CONTRACT_SCHEMA_VERSION = (
    "hrm158_tierb_lane1_hybrid_movement_contract/v0"
)
TIERB_LANE1_HYBRID_MOVEMENT_TERMINAL_SEMANTICS_SCHEMA_VERSION = (
    "hrm158_tierb_lane1_hybrid_movement_terminal_semantics/v0"
)
TIERB_LANE1_HYBRID_MOVEMENT_REPORT_SCHEMA_VERSION = (
    "hrm158_tierb_lane1_hybrid_movement_report/v0"
)


def tierb_lane1_hybrid_movement_success_contract(
    *,
    mode: str,
    steps: int,
    support_name: str,
    support_row_count: int,
    support_audit_every_steps: int,
    support_audit_batch_size: int,
    support_plateau_warmup_step: int,
    support_plateau_patience_audits: int,
) -> dict[str, Any]:
    return {
        "schema": TIERB_LANE1_HYBRID_MOVEMENT_CONTRACT_SCHEMA_VERSION,
        "contract_scope": HYBRID_MOVEMENT_CONTRACT_SCOPE,
        "mode": mode,
        "steps": int(steps),
        "selected_lanes": ["lane1"],
        "dynamics_claim": True,
        "learning_claim": True,
        "support_name": support_name,
        "support_row_count": int(support_row_count),
        "support_audit_every_steps": int(support_audit_every_steps),
        "support_audit_batch_size": int(support_audit_batch_size),
        "support_plateau_warmup_step": int(support_plateau_warmup_step),
        "support_plateau_patience_audits": int(support_plateau_patience_audits),
        "required_terminal_report": "lane1_hybrid_movement_report",
        "terminal_taxonomy": [
            "support_exact_moved",
            "flat_exact_plateau",
            "hard_fail",
        ],
        "movement_bar": {
            "best_support_wide_strict_exact_delta_min": 1,
            "q_changed_total_must_be_positive": True,
            "hard_fail_required_false": True,
        },
        "persistent_authority_contract": {
            "runtime_state_authority": RUNTIME_STATE_AUTHORITY_SUB2_PERSISTENT_HYBRID_DENSE_TRANSIENT_CREDIT,
            "persistent_mode": HYBRID_PERSISTENT_MODE_APPLIED_CROSSING_DIRECTION_PLUS_4BIT_RESIDUAL,
            "dense_transient_selection_role": DENSE_TRANSIENT_CREDIT_ROLE_TRAINING_COMPUTE_CONTROL_ONLY,
        },
        "explicitly_not_claimed": [
            "120/120 acquisition",
            "qacc-only attribution",
            "full-runtime sub-2 success",
            "retention",
        ],
        "watcher": {
            "per_step_compact_ndjson": True,
            "no_black_box_terminal_only": True,
            "support_audit_every_steps": int(support_audit_every_steps),
            "support_plateau_warmup_step": int(support_plateau_warmup_step),
            "support_plateau_patience_audits": int(support_plateau_patience_audits),
        },
        "stop_conditions": [
            "nonfinite",
            "coverage-fail",
            "budget-fail",
            "qacc-hot-loop-budget-fail",
            "persistent_sidecar_budget_fail",
            "persistent_sidecar_state_authority_fail",
            "SHA-drift",
            "FP-shell-proof-fail",
            "parity-fail",
            "forbidden-.pt",
            "flat_exact_plateau",
        ],
    }


def tierb_lane1_hybrid_movement_terminal_semantics_contract(
    *,
    steps: int,
    support_target_strict_exact: int,
    support_audit_every_steps: int,
    support_plateau_warmup_step: int,
    support_plateau_patience_audits: int,
) -> dict[str, Any]:
    return {
        "schema": TIERB_LANE1_HYBRID_MOVEMENT_TERMINAL_SEMANTICS_SCHEMA_VERSION,
        "applies": True,
        "steps": int(steps),
        "support_target_strict_exact": int(support_target_strict_exact),
        "support_audit_every_steps": int(support_audit_every_steps),
        "support_plateau_warmup_step": int(support_plateau_warmup_step),
        "support_plateau_patience_audits": int(support_plateau_patience_audits),
        "success_exit_code": 0,
        "failure_exit_code": 2,
        "success_when": (
            "best support-wide strict exact improves by >=1 after a real update, "
            "q_changed_total stays > 0, and no hard fail fires"
        ),
        "flat_exact_plateau_when": (
            "after warmup, best support-wide strict exact fails to improve across "
            "the preregistered patience window or the bounded ceiling is exhausted "
            "without the +1 strict-exact movement bar"
        ),
        "hard_fail_when": [
            "nonfinite",
            "coverage-fail",
            "budget-fail",
            "qacc-hot-loop-budget-fail",
            "persistent_sidecar_budget_fail",
            "persistent_sidecar_state_authority_fail",
            "SHA-drift",
            "FP-shell-proof-fail",
            "parity-fail",
            "forbidden-.pt",
        ],
    }


def tierb_lane1_hybrid_movement_report(
    lane_summary: Mapping[str, Any],
    *,
    validation_tiny: bool,
    tiny_non_citable_label: str,
    support_name: str,
    support_target_strict_exact: int,
) -> dict[str, Any]:
    history = list(lane_summary.get("support_exact_audit_history") or [])
    last = history[-1] if history else {}
    baseline_event = next(
        (event for event in history if bool(event.get("baseline_native_init"))),
        (history[0] if history else {}),
    )
    baseline_count = int(baseline_event.get("strict_exact_count", 0)) if history else 0
    best_strict_exact_count = max(
        (int(event.get("strict_exact_count", 0)) for event in history),
        default=0,
    )
    best_delta = int(best_strict_exact_count - baseline_count)
    q_changed_total = int(lane_summary.get("q_changed_total", 0))
    stop_reasons = [
        str(event.get("reason"))
        for event in lane_summary.get("stop_conditions_triggered") or []
    ]
    hard_fail_reasons = [reason for reason in stop_reasons if reason != "flat_exact_plateau"]
    measurement_pass = bool(
        lane_summary.get("lane") == "lane1"
        and lane_summary.get("role") == "full_hybrid_qacc_activation_codec"
        and history
        and all(bool(event.get("pass")) for event in history)
        and int(last.get("support_row_count", 0)) == int(support_target_strict_exact)
    )
    if validation_tiny:
        classification = "support_audit_only_validation_tiny"
        terminal_pass = bool(measurement_pass)
    elif hard_fail_reasons:
        classification = "hard_fail"
        terminal_pass = False
    elif q_changed_total > 0 and best_delta >= 1:
        classification = "support_exact_moved"
        terminal_pass = True
    else:
        classification = "flat_exact_plateau"
        terminal_pass = False
    return {
        "schema": TIERB_LANE1_HYBRID_MOVEMENT_REPORT_SCHEMA_VERSION,
        "label": (
            tiny_non_citable_label
            if validation_tiny
            else "native_hybrid_persistent_sub2_movement_smoke"
        ),
        "validation_tiny": bool(validation_tiny),
        "selected_lanes": ["lane1"],
        "movement_contract_scope": HYBRID_MOVEMENT_CONTRACT_SCOPE,
        "support_name": support_name,
        "audit_count": len(history),
        "baseline_step": baseline_event.get("step"),
        "strict_exact": last.get("strict_exact"),
        "strict_exact_count": int(last.get("strict_exact_count", 0)),
        "strict_exact_total": int(last.get("strict_exact_total", 0)),
        "best_strict_exact_count": int(best_strict_exact_count),
        "baseline_strict_exact_count": int(baseline_count),
        "best_strict_exact_delta": int(best_delta),
        "q_changed_total": int(q_changed_total),
        "nonimproving_audit_streak": int(last.get("nonimproving_audit_streak", 0)),
        "last_improving_step": last.get("last_improving_step"),
        "hard_fail_reasons": hard_fail_reasons,
        "measurement_pass": bool(measurement_pass),
        "classification": classification,
        "expected_runtime_exit_code": 0 if terminal_pass else 2,
        "movement_bar": {
            HYBRID_MOVEMENT_METRIC_NAME: int(best_delta),
            "q_changed_total": int(q_changed_total),
            "best_support_wide_strict_exact_delta_min": 1,
            "q_changed_total_must_be_positive": True,
            "hard_fail_required_false": True,
        },
        "claim_boundary": [
            "persistent-sub2 hybrid movement smoke only",
            "algorithmic local-update law reused; this label is law-proof only and is not itself a physical-sub2 storage claim",
            "persistent sidecar storage is physically sub-2 only for this represented event set and must be re-ledgered every runtime step",
            "dense transient selection is training-control only",
            "not full accumulator substitute",
            "not 120/120 acquisition",
            "not qacc-only attribution",
            "not full-runtime sub-2 success",
            "not retention",
        ],
        "pass": bool(terminal_pass),
    }


__all__ = [
    "TIERB_LANE1_HYBRID_MOVEMENT_CONTRACT_SCHEMA_VERSION",
    "TIERB_LANE1_HYBRID_MOVEMENT_REPORT_SCHEMA_VERSION",
    "TIERB_LANE1_HYBRID_MOVEMENT_TERMINAL_SEMANTICS_SCHEMA_VERSION",
    "tierb_lane1_hybrid_movement_report",
    "tierb_lane1_hybrid_movement_success_contract",
    "tierb_lane1_hybrid_movement_terminal_semantics_contract",
]
