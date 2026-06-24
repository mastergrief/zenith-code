"""S3bb branch-C decision-parity dual-arm classifier (W5 treatment vs W6 oracle)."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from calm.hrm_text_158.native_full_stack.narrow_accumulator_codec import (
    W5_SIGNED_MAX,
    W5_SIGNED_MIN,
)
from calm.hrm_text_158.native_full_stack.s3bb_headroom_telemetry import (
    CLASSIFIER_HARNESS_OR_LIVENESS_FAIL,
    DEFAULT_CROSSING_THRESHOLD_ABS,
    MEASURED_STEPS_REQUIRED,
    WARMUP_STEPS,
    _shared_measured_step_ids,
    _treatment_headroom_breach,
    compare_arm_wiring_guards,
    crossing_bool_w6,
)

CLASSIFIER_DOMAIN_OR_HEADROOM_FAIL = "DOMAIN_OR_HEADROOM_FAIL"
CLASSIFIER_DECISION_MISMATCH = "DECISION_MISMATCH"
CLASSIFIER_FLIP_EQUIVALENT_DYNAMICS_DRIFT = "FLIP_EQUIVALENT_DYNAMICS_DRIFT"
CLASSIFIER_DECISION_PARITY_OK = "DECISION_PARITY_OK"

DECISION_PARITY_CLASSIFIER_PRECEDENCE: tuple[str, ...] = (
    CLASSIFIER_HARNESS_OR_LIVENESS_FAIL,
    CLASSIFIER_DOMAIN_OR_HEADROOM_FAIL,
    CLASSIFIER_DECISION_MISMATCH,
    CLASSIFIER_FLIP_EQUIVALENT_DYNAMICS_DRIFT,
    CLASSIFIER_DECISION_PARITY_OK,
)

DEFAULT_FINAL_METRIC_EPSILON = 1e-3
CROSSING_Q_POLICY = "each_arm_own_q_state"
FINAL_METRIC_KEYS_REQUIRED: tuple[str, ...] = ("loss",)
FINAL_METRIC_KEYS_AT_LEAST_ONE: tuple[str, ...] = ("accuracy", "exact_accuracy")


def accumulator_out_of_domain_mask_w5(acc) -> Any:
    import torch

    if acc.dtype != torch.int16:
        raise ValueError(f"accumulator_out_of_domain_mask_w5 requires torch.int16, got {acc.dtype}")
    values = acc.to(dtype=torch.int32)
    return (values < W5_SIGNED_MIN) | (values > W5_SIGNED_MAX)


def _treatment_w5_domain_breach(treatment_receipt: Mapping[str, Any]) -> bool:
    if str(treatment_receipt.get("stop_reason") or "") == "headroom_breach":
        return True
    from calm.hrm_text_158.native_full_stack.s3bb_headroom_telemetry import (
        _iter_sidecar_records,
        resolve_headroom_wiring_sidecar_path,
    )

    sidecar_path = resolve_headroom_wiring_sidecar_path(treatment_receipt)
    if sidecar_path is not None:
        for record in _iter_sidecar_records(sidecar_path):
            for lane in record.get("accumulator_lanes") or []:
                value = int(lane)
                if value < W5_SIGNED_MIN or value > W5_SIGNED_MAX:
                    return True
    step_reports = treatment_receipt.get("step_reports") or {}
    for report in step_reports.values():
        telemetry = report.get("headroom_telemetry") or {}
        if bool(telemetry.get("boundary_value_error_caught")):
            return True
    return False


def _compare_crossing_with_own_q(
    oracle_receipt: Mapping[str, Any],
    treatment_receipt: Mapping[str, Any],
    *,
    threshold_abs: int = DEFAULT_CROSSING_THRESHOLD_ABS,
) -> dict[str, Any]:
    from calm.hrm_text_158.native_full_stack.s3bb_headroom_telemetry import (
        _iter_sidecar_records,
        _sidecar_record_key,
        resolve_headroom_wiring_sidecar_path,
    )

    oracle_path = resolve_headroom_wiring_sidecar_path(oracle_receipt)
    treatment_path = resolve_headroom_wiring_sidecar_path(treatment_receipt)
    if oracle_path is None or treatment_path is None:
        raise ValueError("decision-parity crossing compare requires both wiring sidecars")

    measured = _shared_measured_step_ids(oracle_receipt, treatment_receipt)
    crossing_disagreements = 0
    total_lanes = 0

    oracle_iter = _iter_sidecar_records(oracle_path)
    treatment_iter = _iter_sidecar_records(treatment_path)
    oracle_record = next(oracle_iter, None)
    treatment_record = next(treatment_iter, None)
    while oracle_record is not None or treatment_record is not None:
        if oracle_record is None or treatment_record is None:
            raise ValueError("oracle and treatment sidecar record counts differ")
        step_id, _ = _sidecar_record_key(oracle_record)
        if int(step_id) > WARMUP_STEPS:
            o_vals = [int(v) for v in oracle_record["accumulator_lanes"]]
            t_vals = [int(v) for v in treatment_record["accumulator_lanes"]]
            o_q = [int(v) for v in oracle_record["q_lanes"]]
            t_q = [int(v) for v in treatment_record["q_lanes"]]
            for o_val, t_val, o_qv, t_qv in zip(o_vals, t_vals, o_q, t_q, strict=True):
                total_lanes += 1
                o_cross = crossing_bool_w6(int(o_val), int(o_qv), threshold_abs=int(threshold_abs))
                t_cross = crossing_bool_w6(int(t_val), int(t_qv), threshold_abs=int(threshold_abs))
                if o_cross != t_cross:
                    crossing_disagreements += 1
        oracle_record = next(oracle_iter, None)
        treatment_record = next(treatment_iter, None)

    return {
        "per_step_crossing_bool_disagreement_count": int(crossing_disagreements),
        "total_lane_count": int(total_lanes),
        "crossing_q_policy": CROSSING_Q_POLICY,
        "measured_step_count": int(len(measured)),
    }


def _tensor_stats_for_step(
    step_reports: Mapping[str, Any],
    step_id: str,
) -> dict[str, Any] | None:
    report = step_reports.get(step_id)
    if not isinstance(report, Mapping):
        return None
    step_result = report.get("step_result")
    if not isinstance(step_result, Mapping):
        return None
    tensor_stats = step_result.get("tensor_stats")
    if not isinstance(tensor_stats, Mapping):
        return None
    return dict(tensor_stats)


def audit_observable_coverage(
    oracle_receipt: Mapping[str, Any],
    treatment_receipt: Mapping[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    """Fail-closed coverage audit before decision-parity OK is reachable."""
    failures: list[str] = []
    oracle_steps = oracle_receipt.get("step_reports") or {}
    treatment_steps = treatment_receipt.get("step_reports") or {}
    measured_step_ids = _shared_measured_step_ids(oracle_receipt, treatment_receipt)
    compared_module_steps = 0
    applied_indices_present_module_steps = 0
    q_sha256_after_present_module_steps = 0
    state_key_parity_failures: list[dict[str, Any]] = []

    if len(measured_step_ids) == 0:
        failures.append("shared_measured_step_count_zero")

    for step_id in measured_step_ids:
        o_stats = _tensor_stats_for_step(oracle_steps, step_id)
        t_stats = _tensor_stats_for_step(treatment_steps, step_id)
        if o_stats is None or t_stats is None:
            failures.append(f"tensor_stats_missing_step_{step_id}")
            continue

        o_keys = set(o_stats)
        t_keys = set(t_stats)
        if o_keys != t_keys or len(o_keys) == 0:
            failures.append(f"state_key_parity_fail_step_{step_id}")
            state_key_parity_failures.append(
                {
                    "step": int(step_id),
                    "oracle_only": sorted(o_keys - t_keys),
                    "treatment_only": sorted(t_keys - o_keys),
                    "oracle_count": int(len(o_keys)),
                    "treatment_count": int(len(t_keys)),
                }
            )
            continue

        for state_key in sorted(o_keys):
            compared_module_steps += 1
            o_entry = o_stats[state_key]
            t_entry = t_stats[state_key]
            if not isinstance(o_entry, Mapping) or not isinstance(t_entry, Mapping):
                failures.append(f"tensor_stats_entry_invalid_step_{step_id}_{state_key}")
                continue

            if "applied_indices" not in o_entry or "applied_indices" not in t_entry:
                failures.append(f"applied_indices_absent_step_{step_id}_{state_key}")
            else:
                applied_indices_present_module_steps += 1

            o_sha = o_entry.get("q_sha256_after")
            t_sha = t_entry.get("q_sha256_after")
            if not isinstance(o_sha, str) or not o_sha or not isinstance(t_sha, str) or not t_sha:
                failures.append(f"q_sha256_after_absent_step_{step_id}_{state_key}")
            else:
                q_sha256_after_present_module_steps += 1

    if compared_module_steps == 0:
        failures.append("compared_module_steps_zero")

    final_metric_coverage = _audit_final_metric_coverage(oracle_steps, treatment_steps, failures)

    coverage_stats = {
        "shared_measured_step_count": int(len(measured_step_ids)),
        "compared_module_steps": int(compared_module_steps),
        "applied_indices_present_module_steps": int(applied_indices_present_module_steps),
        "q_sha256_after_present_module_steps": int(q_sha256_after_present_module_steps),
        "state_key_parity_failures": state_key_parity_failures[:16],
        "final_metric_coverage": final_metric_coverage,
        "observable_coverage_pass": len(failures) == 0,
    }
    return list(dict.fromkeys(failures)), coverage_stats


def _audit_final_metric_coverage(
    oracle_steps: Mapping[str, Any],
    treatment_steps: Mapping[str, Any],
    failures: list[str],
) -> dict[str, Any]:
    coverage: dict[str, Any] = {
        "final_step": None,
        "required_keys": list(FINAL_METRIC_KEYS_REQUIRED),
        "accuracy_family_keys": list(FINAL_METRIC_KEYS_AT_LEAST_ONE),
        "oracle_present": [],
        "treatment_present": [],
        "exemptions": [],
    }
    if not oracle_steps or not treatment_steps:
        failures.append("step_reports_empty_for_final_metrics")
        return coverage

    final_step = str(max(int(step_id) for step_id in oracle_steps))
    coverage["final_step"] = final_step
    if final_step not in treatment_steps:
        failures.append("final_step_not_shared")
        return coverage

    o_metrics = oracle_steps[final_step].get("metrics")
    t_metrics = treatment_steps[final_step].get("metrics")
    if not isinstance(o_metrics, Mapping) or not isinstance(t_metrics, Mapping):
        failures.append("final_metrics_missing")
        return coverage

    o_keys = set(o_metrics)
    t_keys = set(t_metrics)
    coverage["oracle_present"] = sorted(o_keys)
    coverage["treatment_present"] = sorted(t_keys)

    for key in FINAL_METRIC_KEYS_REQUIRED:
        if key not in o_metrics or key not in t_metrics:
            failures.append(f"final_metric_missing_{key}")

    if not any(key in o_metrics and key in t_metrics for key in FINAL_METRIC_KEYS_AT_LEAST_ONE):
        failures.append("final_metric_missing_accuracy_family")

    return coverage


def compare_arm_applied_mask_parity(
    oracle_receipt: Mapping[str, Any],
    treatment_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    oracle_steps = oracle_receipt.get("step_reports") or {}
    treatment_steps = treatment_receipt.get("step_reports") or {}
    mismatches: list[dict[str, Any]] = []
    compared_modules = 0
    for step_id in _shared_measured_step_ids(oracle_receipt, treatment_receipt):
        o_stats = _tensor_stats_for_step(oracle_steps, step_id)
        t_stats = _tensor_stats_for_step(treatment_steps, step_id)
        if o_stats is None or t_stats is None:
            continue
        if set(o_stats) != set(t_stats) or len(o_stats) == 0:
            continue
        for state_key in sorted(o_stats):
            o_entry = o_stats[state_key]
            t_entry = t_stats[state_key]
            if "applied_indices" not in o_entry or "applied_indices" not in t_entry:
                continue
            compared_modules += 1
            o_applied = list(o_entry["applied_indices"])
            t_applied = list(t_entry["applied_indices"])
            if o_applied != t_applied:
                mismatches.append(
                    {
                        "step": int(step_id),
                        "state_key": str(state_key),
                        "oracle_len": len(o_applied),
                        "treatment_len": len(t_applied),
                    }
                )
    return {
        "applied_mask_parity_pass": len(mismatches) == 0 and compared_modules > 0,
        "applied_mask_mismatch_count": int(len(mismatches)),
        "applied_mask_compared_module_steps": int(compared_modules),
        "applied_mask_mismatches": mismatches[:16],
    }


def compare_arm_q_trajectory_parity(
    oracle_receipt: Mapping[str, Any],
    treatment_receipt: Mapping[str, Any],
    *,
    final_metric_epsilon: float = DEFAULT_FINAL_METRIC_EPSILON,
) -> dict[str, Any]:
    oracle_steps = oracle_receipt.get("step_reports") or {}
    treatment_steps = treatment_receipt.get("step_reports") or {}
    q_changed_mismatches: list[int] = []
    q_sha_mismatches: list[dict[str, Any]] = []
    for step_id in _shared_measured_step_ids(oracle_receipt, treatment_receipt):
        o_report = oracle_steps[step_id]
        t_report = treatment_steps[step_id]
        if int(o_report.get("q_changed_count") or 0) != int(t_report.get("q_changed_count") or 0):
            q_changed_mismatches.append(int(step_id))
        o_stats = _tensor_stats_for_step(oracle_steps, step_id)
        t_stats = _tensor_stats_for_step(treatment_steps, step_id)
        if o_stats is None or t_stats is None or set(o_stats) != set(t_stats):
            continue
        for state_key in sorted(o_stats):
            o_entry = o_stats[state_key]
            t_entry = t_stats[state_key]
            if "q_sha256_after" not in o_entry or "q_sha256_after" not in t_entry:
                continue
            o_sha = str(o_entry["q_sha256_after"])
            t_sha = str(t_entry["q_sha256_after"])
            if not o_sha or not t_sha:
                continue
            if o_sha != t_sha:
                q_sha_mismatches.append(
                    {"step": int(step_id), "state_key": str(state_key), "oracle": o_sha, "treatment": t_sha}
                )

    steps_mismatch = int(oracle_receipt.get("steps_completed") or 0) != int(
        treatment_receipt.get("steps_completed") or 0
    )
    stop_mismatch = str(oracle_receipt.get("stop_reason") or "") != str(
        treatment_receipt.get("stop_reason") or ""
    )
    final_metrics_mismatch = False
    final_metric_keys_compared: list[str] = []
    if oracle_steps and treatment_steps:
        final_step = str(max(int(s) for s in oracle_steps))
        o_metrics = oracle_steps[final_step].get("metrics")
        t_metrics = treatment_steps[final_step].get("metrics")
        if isinstance(o_metrics, Mapping) and isinstance(t_metrics, Mapping):
            comparable_keys = sorted(
                set(FINAL_METRIC_KEYS_REQUIRED)
                | {key for key in FINAL_METRIC_KEYS_AT_LEAST_ONE if key in o_metrics and key in t_metrics}
            )
            for key in comparable_keys:
                o_val = o_metrics.get(key)
                t_val = t_metrics.get(key)
                if o_val is None or t_val is None:
                    continue
                final_metric_keys_compared.append(key)
                if isinstance(o_val, list) and isinstance(t_val, list) and o_val and t_val:
                    if abs(float(o_val[-1]) - float(t_val[-1])) > float(final_metric_epsilon):
                        final_metrics_mismatch = True
                elif abs(float(o_val) - float(t_val)) > float(final_metric_epsilon):
                    final_metrics_mismatch = True

    drift = (
        bool(q_changed_mismatches)
        or bool(q_sha_mismatches)
        or steps_mismatch
        or stop_mismatch
        or final_metrics_mismatch
    )
    return {
        "q_trajectory_parity_pass": not drift,
        "q_changed_count_mismatch_steps": q_changed_mismatches,
        "q_sha256_after_mismatch_count": int(len(q_sha_mismatches)),
        "q_sha256_after_mismatches": q_sha_mismatches[:16],
        "steps_completed_mismatch": bool(steps_mismatch),
        "stop_reason_mismatch": bool(stop_mismatch),
        "final_metrics_mismatch": bool(final_metrics_mismatch),
        "final_metric_keys_compared": final_metric_keys_compared,
        "final_metric_epsilon": float(final_metric_epsilon),
    }


def classify_s3bb_decision_parity_run(
    oracle_receipt: Mapping[str, Any],
    treatment_receipt: Mapping[str, Any],
    *,
    harness_failures: Sequence[str] | None = None,
    require_w5_ledger: bool = False,
    final_metric_epsilon: float = DEFAULT_FINAL_METRIC_EPSILON,
) -> tuple[str, dict[str, Any]]:
    failures = list(dict.fromkeys(harness_failures or ()))
    coverage_failures, coverage_stats = audit_observable_coverage(oracle_receipt, treatment_receipt)
    failures.extend(coverage_failures)
    failures = list(dict.fromkeys(failures))
    bit_equality_diagnostics = compare_arm_wiring_guards(oracle_receipt, treatment_receipt)
    crossing_stats = _compare_crossing_with_own_q(oracle_receipt, treatment_receipt)
    applied_mask_stats = compare_arm_applied_mask_parity(oracle_receipt, treatment_receipt)
    q_trajectory_stats = compare_arm_q_trajectory_parity(
        oracle_receipt,
        treatment_receipt,
        final_metric_epsilon=float(final_metric_epsilon),
    )
    decision_parity_stats = {
        "observable_coverage": coverage_stats,
        "bit_equality_diagnostics": bit_equality_diagnostics,
        "crossing_parity": crossing_stats,
        "applied_mask_parity": applied_mask_stats,
        "q_trajectory_parity": q_trajectory_stats,
        "crossing_q_policy": CROSSING_Q_POLICY,
    }

    if failures:
        return CLASSIFIER_HARNESS_OR_LIVENESS_FAIL, decision_parity_stats

    oracle_steps = int(oracle_receipt.get("steps_completed") or 0)
    treatment_steps = int(treatment_receipt.get("steps_completed") or 0)
    if oracle_steps < MEASURED_STEPS_REQUIRED or treatment_steps < MEASURED_STEPS_REQUIRED:
        return CLASSIFIER_HARNESS_OR_LIVENESS_FAIL, decision_parity_stats

    if _treatment_headroom_breach(treatment_receipt) or _treatment_w5_domain_breach(treatment_receipt):
        return CLASSIFIER_DOMAIN_OR_HEADROOM_FAIL, decision_parity_stats

    if require_w5_ledger:
        ledger = treatment_receipt.get("r5_persistent_ledger") or {}
        if not bool(ledger.get("enabled")) or not bool(ledger.get("r5_ledger_pass")):
            return CLASSIFIER_HARNESS_OR_LIVENESS_FAIL, decision_parity_stats

    if (
        int(crossing_stats["per_step_crossing_bool_disagreement_count"]) > 0
        or not bool(applied_mask_stats["applied_mask_parity_pass"])
    ):
        return CLASSIFIER_DECISION_MISMATCH, decision_parity_stats

    if not bool(q_trajectory_stats["q_trajectory_parity_pass"]):
        return CLASSIFIER_FLIP_EQUIVALENT_DYNAMICS_DRIFT, decision_parity_stats

    return CLASSIFIER_DECISION_PARITY_OK, decision_parity_stats


def emit_s3bb_decision_parity_receipt(
    oracle_receipt: Mapping[str, Any],
    treatment_receipt: Mapping[str, Any],
    *,
    harness_failures: Sequence[str] | None = None,
    require_w5_ledger: bool = False,
    final_metric_epsilon: float = DEFAULT_FINAL_METRIC_EPSILON,
) -> dict[str, Any]:
    primary, decision_parity_stats = classify_s3bb_decision_parity_run(
        oracle_receipt,
        treatment_receipt,
        harness_failures=harness_failures,
        require_w5_ledger=bool(require_w5_ledger),
        final_metric_epsilon=float(final_metric_epsilon),
    )
    coverage_failures, coverage_stats = audit_observable_coverage(oracle_receipt, treatment_receipt)
    merged_harness_failures = list(
        dict.fromkeys(list(harness_failures or ()) + list(coverage_failures))
    )
    return {
        "slice_id": "w5_decision_parity_run_s3bb_v0",
        "primary_classifier": primary,
        "classifier_precedence": list(DECISION_PARITY_CLASSIFIER_PRECEDENCE),
        "harness_failures": merged_harness_failures,
        "observable_coverage": coverage_stats,
        "oracle_steps_completed": int(oracle_receipt.get("steps_completed") or 0),
        "treatment_steps_completed": int(treatment_receipt.get("steps_completed") or 0),
        "treatment_stop_reason": str(treatment_receipt.get("stop_reason") or ""),
        "decision_parity_stats": decision_parity_stats,
        "bit_equality_not_required_for_ok": True,
        "explicit_non_claims": [
            "decision_parity_dynamics_only",
            "not_lossless",
            "not_sub2_inclusive",
            "not_sub2_win",
            "not_readiness",
        ],
    }
