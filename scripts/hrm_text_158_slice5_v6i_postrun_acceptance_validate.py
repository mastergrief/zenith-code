#!/usr/bin/env python3
"""Post-run acceptance validator for v6i extended-step CLASSIFIER_ONLY packet."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from scripts.hrm_text_158_slice5_v6h_postrun_acceptance_validate import (
    EXPECTED_PATH_ROUTE,
    MATERIAL_SPEEDUP_RATIO_THRESHOLD,
    V6G_CPU_BASELINE_SPARSE_CAP_SECONDS,
    _read_json,
    _sparse_cap_cost_shift_failures,
    _sparse_cap_elapsed_by_step,
    _subphase_elapsed_by_step,
    _subphase_nesting_failures,
    validate_v6h_acceptance,
)

V6H_STEPS_GATE = 3
EXTENDED_STEP_RATIO_DENOMINATOR = V6G_CPU_BASELINE_SPARSE_CAP_SECONDS["3"]
REQUIRED_CAP_STEPS_GE_3 = 256
CAP_SCHEDULE_FAILURE_CLASS = "cap_schedule_drift_detected"


def _extended_step_ratio_denominator(step: int) -> float:
    key = str(step)
    if key in V6G_CPU_BASELINE_SPARSE_CAP_SECONDS:
        return float(V6G_CPU_BASELINE_SPARSE_CAP_SECONDS[key])
    return float(EXTENDED_STEP_RATIO_DENOMINATOR)


def _arm_receipt_path(run_root: Path, arm_subdir: str) -> Path:
    return run_root / arm_subdir / "receipt.json"


def _step_reports(receipt_path: Path) -> dict[str, Any]:
    if not receipt_path.is_file():
        return {}
    payload = _read_json(receipt_path)
    step_reports = payload.get("step_reports")
    return step_reports if isinstance(step_reports, dict) else {}


def _cap_schedule_precondition_failures(
    run_root: Path,
    *,
    max_steps_hard: int,
) -> tuple[list[str], dict[str, Any]]:
    failures: list[str] = []
    report: dict[str, Any] = {}
    for arm_name, arm_subdir in (
        ("baseline", "baseline_snapshot_off"),
        ("instrumented", "instrumented_snapshot_on"),
    ):
        step_reports = _step_reports(_arm_receipt_path(run_root, arm_subdir))
        arm_report: dict[str, Any] = {}
        for step in range(3, max_steps_hard + 1):
            key = str(step)
            step_report = step_reports.get(key)
            if not isinstance(step_report, dict):
                failures.append(f"cap_schedule_missing_step_report_{arm_name}_step_{step}")
                continue
            global_summary = (step_report.get("step_result") or {}).get("global_summary") or {}
            cap = global_summary.get("global_rate_cap_cap")
            arm_report[key] = {
                "global_rate_cap_cap": cap,
                "q_changed_count": step_report.get("q_changed_count"),
                "global_rate_cap_accepted_count": global_summary.get("global_rate_cap_accepted_count"),
            }
            if cap is None:
                failures.append(f"cap_schedule_missing_global_rate_cap_cap_{arm_name}_step_{step}")
            elif int(cap) != REQUIRED_CAP_STEPS_GE_3:
                failures.append(
                    f"{CAP_SCHEDULE_FAILURE_CLASS}_{arm_name}_step_{step}:"
                    f"cap_{cap}_expected_{REQUIRED_CAP_STEPS_GE_3}"
                )
        report[arm_name] = arm_report
    return failures, report


def _ratio_for_step(elapsed: float, step: int) -> float:
    return float(elapsed) / _extended_step_ratio_denominator(step)


def _late_window_ratios_present(ratios_by_step: dict[str, float], late_steps: list[int]) -> bool:
    return all(str(step) in ratios_by_step for step in late_steps)


def _trending_down(ratios_by_step: dict[str, float], late_steps: list[int]) -> bool:
    if len(late_steps) < 3 or not _late_window_ratios_present(ratios_by_step, late_steps):
        return False
    values = [ratios_by_step[str(step)] for step in late_steps]
    return values[0] > values[1] > values[2]


def _late_window_material_clear(ratios_by_step: dict[str, float], late_steps: list[int]) -> bool:
    if not _late_window_ratios_present(ratios_by_step, late_steps):
        return False
    return all(
        ratios_by_step[str(step)] < MATERIAL_SPEEDUP_RATIO_THRESHOLD for step in late_steps
    )


def _classify_warmup_verdict(
    *,
    trending_baseline: bool,
    trending_instrumented: bool,
    clear_baseline: bool,
    clear_instrumented: bool,
) -> str:
    trending = trending_baseline or trending_instrumented
    if not trending and clear_baseline and clear_instrumented:
        return "WARMUP_AMORTIZES_CLEARS"
    if not trending and not clear_baseline and not clear_instrumented:
        return "STRUCTURAL_PLATEAU"
    return "NOISY_AMBIGUOUS"


def validate_v6i_acceptance(run_root: Path, *, max_steps_hard: int = 8) -> dict[str, Any]:
    failures: list[str] = []
    checks: dict[str, Any] = {
        "classifier_only": True,
        "max_steps_hard": max_steps_hard,
        "late_window_steps": [max_steps_hard - 2, max_steps_hard - 1, max_steps_hard],
        "extended_step_ratio_denominator_seconds": EXTENDED_STEP_RATIO_DENOMINATOR,
        "extended_step_ratio_denominator_rationale": "workload_matched_v6g_step3_cap256",
    }

    v6h_receipt = validate_v6h_acceptance(run_root, max_steps_hard=V6H_STEPS_GATE)
    checks["v6h_gate_receipt"] = {
        "schema": v6h_receipt.get("schema"),
        "pass": v6h_receipt.get("pass"),
        "failure_count": len(v6h_receipt.get("failures", [])),
    }
    checks["v6h_gate_checks"] = v6h_receipt.get("checks", {})
    for failure in v6h_receipt.get("failures", []):
        failures.append(f"v6h_gate:{failure}")

    cap_failures, cap_report = _cap_schedule_precondition_failures(
        run_root,
        max_steps_hard=max_steps_hard,
    )
    failures.extend(cap_failures)
    checks["cap_schedule_precondition_report"] = cap_report

    baseline_sparse = _sparse_cap_elapsed_by_step(run_root / "baseline_snapshot_off")
    instrumented_sparse = _sparse_cap_elapsed_by_step(run_root / "instrumented_snapshot_on")
    checks["baseline_sparse_cap_apply_elapsed_by_step"] = baseline_sparse
    checks["instrumented_sparse_cap_apply_elapsed_by_step"] = instrumented_sparse

    workload_report: dict[str, Any] = {}
    for arm_name, arm_subdir in (
        ("baseline", "baseline_snapshot_off"),
        ("instrumented", "instrumented_snapshot_on"),
    ):
        workload_report[arm_name] = _step_reports(_arm_receipt_path(run_root, arm_subdir))
    checks["cap_workload_step_reports"] = workload_report

    extended_speedup_report: dict[str, dict[str, Any]] = {}
    for arm_name, sparse_by_step in (
        ("baseline", baseline_sparse),
        ("instrumented", instrumented_sparse),
    ):
        arm_report: dict[str, Any] = {}
        for step in range(1, max_steps_hard + 1):
            key = str(step)
            elapsed = sparse_by_step.get(key)
            if elapsed is None:
                failures.append(f"missing_{arm_name}_sparse_cap_elapsed_step_{step}")
                continue
            ratio = _ratio_for_step(float(elapsed), step)
            arm_report[key] = {
                "elapsed_seconds": elapsed,
                "ratio_denominator_seconds": _extended_step_ratio_denominator(step),
                "ratio_to_reference": ratio,
                "material_win": ratio < MATERIAL_SPEEDUP_RATIO_THRESHOLD,
            }
        extended_speedup_report[arm_name] = arm_report
    checks["sparse_cap_speedup_all_steps"] = extended_speedup_report

    late_steps = checks["late_window_steps"]
    late_window_report: dict[str, Any] = {}
    classifier_inputs: dict[str, Any] = {}
    for arm_name, arm_report in extended_speedup_report.items():
        ratios = {
            key: float(value["ratio_to_reference"])
            for key, value in arm_report.items()
            if key in {str(step) for step in late_steps}
        }
        late_window_report[arm_name] = {
            "ratios_by_step": ratios,
            "late_window_material_clear": _late_window_material_clear(ratios, late_steps),
            "still_trending_down": _trending_down(ratios, late_steps),
        }
        classifier_inputs[arm_name] = late_window_report[arm_name]
    checks["late_window_report"] = late_window_report

    late_window_complete = all(
        _late_window_ratios_present(late_window_report[arm]["ratios_by_step"], late_steps)
        for arm in ("baseline", "instrumented")
    )
    checks["late_window_complete"] = late_window_complete

    warmup_classifier_verdict = None
    if (
        not cap_failures
        and not v6h_receipt.get("failures")
        and late_window_complete
    ):
        warmup_classifier_verdict = _classify_warmup_verdict(
            trending_baseline=bool(classifier_inputs["baseline"]["still_trending_down"]),
            trending_instrumented=bool(classifier_inputs["instrumented"]["still_trending_down"]),
            clear_baseline=bool(classifier_inputs["baseline"]["late_window_material_clear"]),
            clear_instrumented=bool(classifier_inputs["instrumented"]["late_window_material_clear"]),
        )
    checks["warmup_classifier_verdict"] = warmup_classifier_verdict
    checks["arm_execution_order"] = ["baseline_snapshot_off", "instrumented_snapshot_on"]
    checks["confounds_acknowledged"] = [
        "arm_order_warm_cache_baseline_before_instrumented",
        "workload_shrink_closed_by_cap256_assertion_steps_ge_3",
        "timer_boundary_blind_spot_inside_sparse_cap_apply_parent_window",
    ]

    for arm_name, arm_dir in (
        ("baseline", run_root / "baseline_snapshot_off"),
        ("instrumented", run_root / "instrumented_snapshot_on"),
    ):
        sparse_by_step = baseline_sparse if arm_name == "baseline" else instrumented_sparse
        subphase_by_step = _subphase_elapsed_by_step(arm_dir)
        failures.extend(_subphase_nesting_failures(arm_dir, arm=arm_name))
        failures.extend(
            _sparse_cap_cost_shift_failures(
                arm=arm_name,
                sparse_by_step=sparse_by_step,
                subphase_by_step=subphase_by_step,
                max_steps_hard=max_steps_hard,
            )
        )

    return {
        "schema": "hrm_text_158_slice5_v6i_postrun_acceptance_validate/v1",
        "run_root": str(run_root),
        "pass": not failures,
        "failures": failures,
        "checks": checks,
        "expected_path_route": EXPECTED_PATH_ROUTE,
        "warmup_classifier_verdict": warmup_classifier_verdict,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-steps-hard", type=int, default=8)
    args = parser.parse_args(argv)

    run_root = Path(args.run_root)
    out = Path(args.out)
    receipt = validate_v6i_acceptance(run_root, max_steps_hard=args.max_steps_hard)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
