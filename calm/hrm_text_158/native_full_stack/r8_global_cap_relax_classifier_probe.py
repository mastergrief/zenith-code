"""Read-only R8 global-cap-relax classifier over relax-run sidecar vs frozen R7 baseline."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from calm.hrm_text_158.native_full_stack.r7_cap_defer_pressure_instrumentation import (
    R7_SIDECAR_FILENAME,
    R7_STEP_CHUNK_SCHEMA_VERSION,
    iter_sidecar_chunks,
    validate_accounting_invariant,
)
from calm.hrm_text_158.native_full_stack.r7_mechanism_classifier_probe import (
    MIN_MEASURED_STEPS,
    compute_run_metrics,
)

R8_PROBE_SCHEMA_VERSION = "hrm_text_158_r8_global_cap_relax_classifier_probe/v1"

R7_BASELINE_RUN_ID = "r7_from_clean_cap_defer_seed44_43_20260623T111514Z_d85208d8"
R7_BASELINE_TERMINAL_RECEIPT_SHA256_PREFIX = "8eca45ca"

R7_BASELINE: dict[str, Any] = {
    "run_id": R7_BASELINE_RUN_ID,
    "terminal_receipt_sha256_prefix": R7_BASELINE_TERMINAL_RECEIPT_SHA256_PREFIX,
    "run_mean_deferred_saturation": 0.7984375,
    "run_max_deferred_backlog_max_age_steps": 7,
    "pressure_growth_ratio": 13196964.0,
    "accepted_from_prior_deferred_total": 695,
    "accepted_fresh_total": 1353,
    "q_transition_mass_ratio": 4.881264783373974e-05,
    "steps_observed": 10,
}

BRANCH_HARNESS_FAIL = "R8_HARNESS_FAIL"
BRANCH_SCHEMA_FAIL = "R8_SCHEMA_FAIL"
BRANCH_ARTIFACT_INSUFFICIENT = "R8_ARTIFACT_INSUFFICIENT"
BRANCH_CARRIER_CAPACITY_FAIL = "R8_CARRIER_CAPACITY_FAIL"
BRANCH_CAP_RELAX_DESTABILIZES = "R8_CAP_RELAX_DESTABILIZES"
BRANCH_CAP_WAS_BINDING = "R8_CAP_WAS_BINDING"
BRANCH_NOT_CAP_BOUND = "R8_NOT_CAP_BOUND"
BRANCH_RELAXATION_INSUFFICIENT = "R8_RELAXATION_INSUFFICIENT"
BRANCH_UNCLASSIFIED = "R8_UNCLASSIFIED"

SATURATION_BINDING_MAX = 0.48
SATURATION_NOT_BOUND_MIN = 0.6787
SATURATION_PARTIAL_DRAIN_REL_DROP = 0.10
SATURATION_STILL_SATURATED_MIN = 0.50
MAX_AGE_BINDING_MAX = 5
MAX_AGE_NOT_BOUND_MIN = 5
MAX_AGE_STILL_AGING_MIN = 4
MAX_AGE_PARTIAL_DRAIN_ABS_DROP = 1
PRESSURE_GROWTH_STALL_MAX = 1.5
PRESSURE_GROWTH_NOT_BOUND_MIN = 1.5

EXPLICIT_NON_CLAIMS: tuple[str, ...] = (
    "cap_relax_not_mechanism_proof",
    "no_bank_gate",
    "no_sub2_win",
    "no_readiness_flip",
    "audit_report_only_not_veto",
    "single_variable_cap_only",
    "10_step_diagnostic_not_long_run",
    "proxy_not_proof",
    "no_stability_verdict",
    "no_decision_surface_claim",
    "diagnostic_10_step_not_mechanism_proof",
    "parsed_regression_telemetry_only",
    "loss_mean_telemetry_only",
)

NEXT_ACTION_BY_BRANCH: dict[str, str] = {
    BRANCH_HARNESS_FAIL: "stop_and_fix_inputs",
    BRANCH_SCHEMA_FAIL: "stop_and_fix_inputs",
    BRANCH_ARTIFACT_INSUFFICIENT: "instrumentation_not_interpretation",
    BRANCH_CARRIER_CAPACITY_FAIL: "stop_and_fix_carrier_capacity",
    BRANCH_CAP_RELAX_DESTABILIZES: "stop_cap_relax_destabilized",
    BRANCH_CAP_WAS_BINDING: "cap_binding_confirmed_relax_helped",
    BRANCH_NOT_CAP_BOUND: "cap_not_primary_binding_mechanism",
    BRANCH_RELAXATION_INSUFFICIENT: "increase_relaxation_or_redesign",
    BRANCH_UNCLASSIFIED: "manual_review_required",
}

PRIOR_AUDIT_SUPPORTS = ("L0b", "math_a0", "L0c1")


def detect_carrier_capacity_fail(diagnostic_receipt: Mapping[str, Any] | None) -> bool:
    if not diagnostic_receipt:
        return False
    stop_reason = str(diagnostic_receipt.get("stop_reason") or "").lower()
    if "headroom_breach" in stop_reason or "narrow_carrier" in stop_reason:
        return True
    for step_report in (diagnostic_receipt.get("step_reports") or {}).values():
        if not isinstance(step_report, Mapping):
            continue
        step_result = step_report.get("step_result") or {}
        if isinstance(step_result, Mapping) and step_result.get("headroom_breach"):
            return True
        headroom = step_report.get("headroom_telemetry") or {}
        if isinstance(headroom, Mapping) and (
            headroom.get("would_strict_raise_step")
            or headroom.get("boundary_value_error_caught")
        ):
            return True
    return False


def detect_baseline_correct_strict_regressions(
    prior_audit: Mapping[str, Any] | None,
) -> list[dict[str, str]]:
    if not prior_audit or not prior_audit.get("enabled"):
        return []
    deltas = prior_audit.get("deltas") or {}
    start_reports = prior_audit.get("start_reports") or {}
    final_reports = prior_audit.get("final_reports") or {}
    regressions: list[dict[str, str]] = []
    for support in PRIOR_AUDIT_SUPPORTS:
        delta = deltas.get(support)
        if not isinstance(delta, Mapping):
            continue
        start = start_reports.get(support) or {}
        final = final_reports.get(support) or {}
        if start.get("audit_mismatch") or final.get("audit_mismatch"):
            continue
        for row_id in delta.get("new_strict_failure_row_ids") or []:
            regressions.append({"support": str(support), "row_id": str(row_id)})
    return regressions


def build_audit_summary(
    prior_audit: Mapping[str, Any] | None,
    *,
    diagnostic_receipt: Mapping[str, Any] | None,
) -> dict[str, Any]:
    regressions = detect_baseline_correct_strict_regressions(prior_audit)
    loss_mean = None
    if diagnostic_receipt:
        timing = diagnostic_receipt.get("timing_summary") or {}
        if isinstance(timing, Mapping):
            loss_mean = timing.get("loss_mean")
    parsed_new_counts: dict[str, int] = {}
    if prior_audit and prior_audit.get("enabled"):
        for support, delta in (prior_audit.get("deltas") or {}).items():
            if isinstance(delta, Mapping):
                parsed_new_counts[str(support)] = len(
                    delta.get("new_parsed_failure_row_ids") or []
                )
    return {
        "enabled": bool(prior_audit and prior_audit.get("enabled")),
        "requested_supports": list((prior_audit or {}).get("requested_supports") or []),
        "baseline_correct_strict_regressions": regressions,
        "baseline_correct_strict_regression_count": len(regressions),
        "parsed_new_failure_counts_telemetry_only": parsed_new_counts,
        "loss_mean_telemetry_only": loss_mean,
    }


def _has_drain_signal(metrics: Mapping[str, Any]) -> bool:
    saturation = float(metrics["run_mean_deferred_saturation"])
    max_age = int(metrics["run_max_deferred_backlog_max_age_steps"])
    return saturation < SATURATION_BINDING_MAX or max_age <= MAX_AGE_BINDING_MAX


def _partial_drain(metrics: Mapping[str, Any]) -> bool:
    baseline_sat = float(R7_BASELINE["run_mean_deferred_saturation"])
    baseline_age = int(R7_BASELINE["run_max_deferred_backlog_max_age_steps"])
    saturation = float(metrics["run_mean_deferred_saturation"])
    max_age = int(metrics["run_max_deferred_backlog_max_age_steps"])
    rel_sat_drop = (baseline_sat - saturation) / baseline_sat if baseline_sat > 0 else 0.0
    abs_age_drop = baseline_age - max_age
    return rel_sat_drop > SATURATION_PARTIAL_DRAIN_REL_DROP or abs_age_drop > MAX_AGE_PARTIAL_DRAIN_ABS_DROP


def _still_saturated_or_aging(metrics: Mapping[str, Any]) -> bool:
    saturation = float(metrics["run_mean_deferred_saturation"])
    max_age = int(metrics["run_max_deferred_backlog_max_age_steps"])
    return saturation >= SATURATION_STILL_SATURATED_MIN or max_age >= MAX_AGE_STILL_AGING_MIN


def _baseline_comparison(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "baseline_run_id": R7_BASELINE_RUN_ID,
        "baseline_terminal_receipt_sha256_prefix": R7_BASELINE_TERMINAL_RECEIPT_SHA256_PREFIX,
        "relax_vs_baseline": {
            "run_mean_deferred_saturation": {
                "relax": float(metrics["run_mean_deferred_saturation"]),
                "baseline": float(R7_BASELINE["run_mean_deferred_saturation"]),
            },
            "run_max_deferred_backlog_max_age_steps": {
                "relax": int(metrics["run_max_deferred_backlog_max_age_steps"]),
                "baseline": int(R7_BASELINE["run_max_deferred_backlog_max_age_steps"]),
            },
            "pressure_growth_ratio": {
                "relax": float(metrics["pressure_growth_ratio"]),
                "baseline": float(R7_BASELINE["pressure_growth_ratio"]),
            },
            "accepted_from_prior_deferred_total": {
                "relax": int(metrics["accepted_from_prior_deferred_total"]),
                "baseline": int(R7_BASELINE["accepted_from_prior_deferred_total"]),
            },
            "accepted_fresh_total": {
                "relax": int(metrics["accepted_fresh_total"]),
                "baseline": int(R7_BASELINE["accepted_fresh_total"]),
            },
            "q_transition_mass_ratio": {
                "relax": float(metrics["q_transition_mass_ratio"]),
                "baseline": float(R7_BASELINE["q_transition_mass_ratio"]),
            },
            "steps_observed": {
                "relax": int(metrics["steps_observed"]),
                "baseline": int(R7_BASELINE["steps_observed"]),
            },
        },
        "thresholds": {
            "SATURATION_BINDING_MAX": SATURATION_BINDING_MAX,
            "SATURATION_NOT_BOUND_MIN": SATURATION_NOT_BOUND_MIN,
            "MAX_AGE_BINDING_MAX": MAX_AGE_BINDING_MAX,
            "MAX_AGE_NOT_BOUND_MIN": MAX_AGE_NOT_BOUND_MIN,
            "PRESSURE_GROWTH_STALL_MAX": PRESSURE_GROWTH_STALL_MAX,
            "PRESSURE_GROWTH_NOT_BOUND_MIN": PRESSURE_GROWTH_NOT_BOUND_MIN,
        },
    }


def select_branch(
    *,
    harness_fail: bool,
    schema_fail: bool,
    metrics: Mapping[str, Any],
    diagnostic_receipt: Mapping[str, Any] | None,
    prior_audit: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if harness_fail:
        return {"branch": BRANCH_HARNESS_FAIL, "next_action": NEXT_ACTION_BY_BRANCH[BRANCH_HARNESS_FAIL]}
    if schema_fail:
        return {"branch": BRANCH_SCHEMA_FAIL, "next_action": NEXT_ACTION_BY_BRANCH[BRANCH_SCHEMA_FAIL]}
    if int(metrics["steps_observed"]) < MIN_MEASURED_STEPS:
        return {
            "branch": BRANCH_ARTIFACT_INSUFFICIENT,
            "next_action": NEXT_ACTION_BY_BRANCH[BRANCH_ARTIFACT_INSUFFICIENT],
            "reason": "fewer_than_eight_measured_steps",
        }
    if detect_carrier_capacity_fail(diagnostic_receipt):
        return {
            "branch": BRANCH_CARRIER_CAPACITY_FAIL,
            "next_action": NEXT_ACTION_BY_BRANCH[BRANCH_CARRIER_CAPACITY_FAIL],
        }

    audit_summary = build_audit_summary(prior_audit, diagnostic_receipt=diagnostic_receipt)
    regressions = audit_summary["baseline_correct_strict_regressions"]
    has_regression = len(regressions) > 0
    saturation = float(metrics["run_mean_deferred_saturation"])
    max_age = int(metrics["run_max_deferred_backlog_max_age_steps"])
    pressure_growth = float(metrics["pressure_growth_ratio"])
    accepted_prior = int(metrics["accepted_from_prior_deferred_total"])

    if has_regression and _has_drain_signal(metrics):
        return {
            "branch": BRANCH_CAP_RELAX_DESTABILIZES,
            "next_action": NEXT_ACTION_BY_BRANCH[BRANCH_CAP_RELAX_DESTABILIZES],
            "baseline_correct_strict_regression_count": len(regressions),
            "regressions": regressions,
        }

    was_binding_checks = {
        "saturation_lte_binding_max": saturation <= SATURATION_BINDING_MAX,
        "max_age_lte_binding_max": max_age <= MAX_AGE_BINDING_MAX,
        "accepted_prior_gt_baseline": accepted_prior > int(R7_BASELINE["accepted_from_prior_deferred_total"]),
        "pressure_growth_stalled": pressure_growth < PRESSURE_GROWTH_STALL_MAX,
        "no_strict_regression": not has_regression,
    }
    if all(was_binding_checks.values()):
        return {
            "branch": BRANCH_CAP_WAS_BINDING,
            "next_action": NEXT_ACTION_BY_BRANCH[BRANCH_CAP_WAS_BINDING],
            "checks_passed": was_binding_checks,
        }

    not_bound_checks = {
        "saturation_gte_not_bound_min": saturation >= SATURATION_NOT_BOUND_MIN,
        "max_age_gte_not_bound_min": max_age >= MAX_AGE_NOT_BOUND_MIN,
        "pressure_growth_gte_not_bound_min": pressure_growth >= PRESSURE_GROWTH_NOT_BOUND_MIN,
    }
    if all(not_bound_checks.values()):
        return {
            "branch": BRANCH_NOT_CAP_BOUND,
            "next_action": NEXT_ACTION_BY_BRANCH[BRANCH_NOT_CAP_BOUND],
            "checks_passed": not_bound_checks,
        }

    insufficient_checks = {
        "partial_drain": _partial_drain(metrics),
        "still_saturated_or_aging": _still_saturated_or_aging(metrics),
    }
    if all(insufficient_checks.values()):
        return {
            "branch": BRANCH_RELAXATION_INSUFFICIENT,
            "next_action": NEXT_ACTION_BY_BRANCH[BRANCH_RELAXATION_INSUFFICIENT],
            "checks_passed": insufficient_checks,
        }

    failed_thresholds: list[str] = []
    for name, passed in was_binding_checks.items():
        if not passed:
            failed_thresholds.append(f"cap_was_binding:{name}")
    for name, passed in not_bound_checks.items():
        if not passed:
            failed_thresholds.append(f"not_cap_bound:{name}")
    for name, passed in insufficient_checks.items():
        if not passed:
            failed_thresholds.append(f"relaxation_insufficient:{name}")
    if has_regression and not _has_drain_signal(metrics):
        failed_thresholds.append("destabilizes:regression_without_drain")
    if not has_regression and _has_drain_signal(metrics):
        failed_thresholds.append("destabilizes:drain_without_regression")

    return {
        "branch": BRANCH_UNCLASSIFIED,
        "next_action": NEXT_ACTION_BY_BRANCH[BRANCH_UNCLASSIFIED],
        "failed_thresholds": failed_thresholds,
        "metric_snapshot": dict(metrics),
        "audit_summary": audit_summary,
    }


def build_classifier_from_chunks(
    *,
    chunks: Sequence[Mapping[str, Any]],
    harness_fail: bool = False,
    run_root: str | None = None,
    head_sha256: str | None = None,
    sidecar_path: str | None = None,
    diagnostic_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    schema_failures: list[str] = []
    for chunk in chunks:
        if chunk.get("schema_version") != R7_STEP_CHUNK_SCHEMA_VERSION:
            schema_failures.append(f"step{chunk.get('step')}:schema_version")
            continue
        failures = list(chunk.get("accounting_invariant_failures") or [])
        if failures:
            schema_failures.extend(f"step{chunk.get('step')}:{item}" for item in failures)
    metrics = compute_run_metrics(chunks) if chunks else compute_run_metrics([])
    prior_audit = (diagnostic_receipt or {}).get("prior_audit")
    branch = select_branch(
        harness_fail=harness_fail,
        schema_fail=bool(schema_failures),
        metrics=metrics,
        diagnostic_receipt=diagnostic_receipt,
        prior_audit=prior_audit if isinstance(prior_audit, Mapping) else None,
    )
    audit_summary = build_audit_summary(
        prior_audit if isinstance(prior_audit, Mapping) else None,
        diagnostic_receipt=diagnostic_receipt,
    )
    result: dict[str, Any] = {
        "schema_version": R8_PROBE_SCHEMA_VERSION,
        "raw_arrays_included": False,
        "run_root": run_root,
        "head_sha256": head_sha256,
        "sidecar_path": sidecar_path,
        "sidecar_source": "relax_run_own_diagnostic_sidecar",
        "r7_baseline_provenance": dict(R7_BASELINE),
        "run_metrics": metrics,
        "baseline_comparison": _baseline_comparison(metrics),
        "audit_summary": audit_summary,
        "branch_selection": branch,
        "explicit_non_claims": list(EXPLICIT_NON_CLAIMS),
    }
    if schema_failures:
        result["schema_failures"] = schema_failures
    return result


def load_diagnostic_receipt(run_root: Path) -> dict[str, Any] | None:
    receipt_path = run_root / "diagnostic" / "receipt.json"
    if not receipt_path.is_file():
        return None
    return json.loads(receipt_path.read_text(encoding="utf-8"))


def build_classifier_probe_receipt(
    *,
    run_root: Path,
    head_sha256: str,
    sidecar_relative: str = f"diagnostic/{R7_SIDECAR_FILENAME}",
) -> dict[str, Any]:
    sidecar_path = run_root / sidecar_relative
    diagnostic_receipt = load_diagnostic_receipt(run_root)
    if not sidecar_path.is_file():
        return build_classifier_from_chunks(
            chunks=[],
            harness_fail=True,
            run_root=str(run_root),
            head_sha256=head_sha256,
            sidecar_path=str(sidecar_path),
            diagnostic_receipt=diagnostic_receipt,
        )
    chunks = iter_sidecar_chunks(sidecar_path)
    return build_classifier_from_chunks(
        chunks=chunks,
        harness_fail=False,
        run_root=str(run_root),
        head_sha256=head_sha256,
        sidecar_path=str(sidecar_path),
        diagnostic_receipt=diagnostic_receipt,
    )


def validate_step_summary_for_field_presence(step_summary: Mapping[str, Any]) -> list[str]:
    return validate_accounting_invariant(step_summary)
