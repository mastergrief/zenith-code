#!/usr/bin/env python3
"""Post-run acceptance validator for v6h A-prime GPU seam runtime-proof packet."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PARENT_PT = Path(
    "calm/hrm/checkpoints/"
    "hrm_text_158_phase3_L0c1_seed0017_replay83_n12k_lr7p5e5_pc1p0_rsL0b1math1r1b2_1_anchorsv1r3_from_L0b_final_step01500.pt"
)
PARENT_SHA = "9b4e311a22787e7d4808bde7bc2953568d767a2ee8ac648942a3f5dbb7b4d5ec"
EXPECTED_PATH_ROUTE = "PATH_GPU_SEAM_EXERCISED"
AGGREGATE_RECEIPT = "prelaunch/live_carrier_scale_smoke_receipt.json"
V6G_CPU_BASELINE_SPARSE_CAP_SECONDS = {"1": 99.0, "2": 102.6, "3": 128.4}
MATERIAL_SPEEDUP_RATIO_THRESHOLD = 0.5
SPARSE_CAP_APPLY_PHASE_ID = "sparse_cap_apply"
SUBPHASE_ARTIFACTS = {
    "cap_selection_cpu_copy": "sparse_cap_apply_cap_selection_cpu_copy.jsonl",
    "post_cap_apply_sync": "sparse_cap_apply_post_cap_apply_sync.jsonl",
    "boundary_normalize": "sparse_cap_apply_boundary_normalize.jsonl",
}
PEAK_EMIT_DELTA_SECONDS_MAX = (
    MATERIAL_SPEEDUP_RATIO_THRESHOLD * min(V6G_CPU_BASELINE_SPARSE_CAP_SECONDS.values())
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _launch_rc(run_root: Path, arm: str) -> int | None:
    rc_path = run_root / "prelaunch" / f"{arm}_launch_rc.txt"
    if not rc_path.is_file():
        return None
    text = rc_path.read_text(encoding="utf-8").strip()
    return int(text) if text.isdigit() else None


def _sparse_cap_elapsed_by_step(arm_dir: Path) -> dict[str, float]:
    path = arm_dir / "liveness_milestones" / "sparse_cap_apply.jsonl"
    if not path.is_file():
        return {}
    out: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("milestone_kind") != "phase_complete":
            continue
        step = row.get("optimizer_step_index")
        elapsed = row.get("elapsed_since_phase_enter_seconds")
        if step is None or elapsed is None:
            continue
        out[str(int(step))] = float(elapsed)
    return out


def _subphase_elapsed_totals(arm_dir: Path) -> dict[str, float]:
    totals: dict[str, float] = {}
    for sub_phase, rel in SUBPHASE_ARTIFACTS.items():
        path = arm_dir / "liveness_milestones" / rel
        if not path.is_file():
            continue
        total = 0.0
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            elapsed = row.get("elapsed_since_phase_enter_seconds")
            if elapsed is not None:
                total += float(elapsed)
        totals[sub_phase] = total
    return totals


def _subphase_elapsed_by_step(arm_dir: Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for sub_phase, rel in SUBPHASE_ARTIFACTS.items():
        path = arm_dir / "liveness_milestones" / rel
        if not path.is_file():
            continue
        by_step: dict[str, float] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            step = row.get("optimizer_step_index")
            elapsed = row.get("elapsed_since_phase_enter_seconds")
            if step is None or elapsed is None:
                continue
            by_step[str(int(step))] = float(elapsed)
        out[sub_phase] = by_step
    return out


def _subphase_nesting_failures(arm_dir: Path, *, arm: str) -> list[str]:
    failures: list[str] = []
    for sub_phase, rel in SUBPHASE_ARTIFACTS.items():
        path = arm_dir / "liveness_milestones" / rel
        if not path.is_file():
            continue
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            parent_phase_id = row.get("parent_phase_id")
            if parent_phase_id != SPARSE_CAP_APPLY_PHASE_ID:
                failures.append(
                    f"subphase_not_nested_in_sparse_cap_apply_{arm}_{sub_phase}_line_{line_no}:"
                    f"{parent_phase_id!r}"
                )
    return failures


def _sparse_cap_cost_shift_failures(
    *,
    arm: str,
    sparse_by_step: dict[str, float],
    subphase_by_step: dict[str, dict[str, float]],
    max_steps_hard: int,
) -> list[str]:
    failures: list[str] = []
    for step in range(1, max_steps_hard + 1):
        key = str(step)
        parent_elapsed = sparse_by_step.get(key)
        if parent_elapsed is None:
            continue
        for sub_phase, by_step in subphase_by_step.items():
            sub_elapsed = by_step.get(key)
            if sub_elapsed is None:
                continue
            if float(sub_elapsed) > float(parent_elapsed):
                failures.append(
                    f"sparse_cap_cost_shift_subphase_exceeds_parent_{arm}_{sub_phase}_step_{step}:"
                    f"{sub_elapsed:.3f}s_gt_parent_{parent_elapsed:.3f}s"
                )
    return failures


def validate_v6h_acceptance(run_root: Path, *, max_steps_hard: int = 3) -> dict[str, Any]:
    failures: list[str] = []
    checks: dict[str, Any] = {}

    baseline_receipt_path = run_root / "baseline_snapshot_off" / "receipt.json"
    instrumented_receipt_path = run_root / "instrumented_snapshot_on" / "receipt.json"
    if not baseline_receipt_path.is_file():
        failures.append("missing_baseline_receipt_json")
    if not instrumented_receipt_path.is_file():
        failures.append("missing_instrumented_receipt_json")

    baseline_rc = _launch_rc(run_root, "baseline")
    instrumented_rc = _launch_rc(run_root, "instrumented")
    checks["baseline_launch_rc"] = baseline_rc
    checks["instrumented_launch_rc"] = instrumented_rc
    if baseline_rc != 0:
        failures.append(f"baseline_launch_rc_not_zero:{baseline_rc}")
    if instrumented_rc != 0:
        failures.append(f"instrumented_launch_rc_not_zero:{instrumented_rc}")

    aggregate_path = run_root / AGGREGATE_RECEIPT
    if aggregate_path.is_file():
        aggregate = _read_json(aggregate_path)
        checks["aggregate_receipt_schema"] = aggregate.get("schema")
        if aggregate.get("schema") != "hrm_text_158_slice5_live_carrier_gpu_scale_smoke_receipt/v1":
            failures.append("aggregate_smoke_receipt_schema_missing")

        for key in (
            "persistent_accumulator_event_coded_live_baseline",
            "persistent_accumulator_event_coded_live_instrumented",
        ):
            value = aggregate.get(key)
            checks[key] = value
            if value is None:
                failures.append(f"{key}_null")

        deltas = aggregate.get("per_step_duration_delta_seconds")
        checks["per_step_duration_delta_seconds"] = deltas
        if not isinstance(deltas, dict):
            failures.append("aggregate_per_step_duration_delta_seconds_missing")
        else:
            for step in range(1, max_steps_hard + 1):
                if str(step) not in deltas:
                    failures.append(f"aggregate_per_step_duration_delta_missing_step_{step}")

        peak_emit = aggregate.get("peak_per_step_emit_delta_seconds")
        checks["peak_per_step_emit_delta_seconds"] = peak_emit
        checks["peak_per_step_emit_delta_seconds_max_allowed"] = PEAK_EMIT_DELTA_SECONDS_MAX
        if peak_emit is None:
            failures.append("aggregate_peak_per_step_emit_delta_seconds_null")
        elif float(peak_emit) >= PEAK_EMIT_DELTA_SECONDS_MAX:
            failures.append(
                f"aggregate_peak_per_step_emit_delta_seconds_too_high:"
                f"{float(peak_emit):.3f}s_gte_{PEAK_EMIT_DELTA_SECONDS_MAX:.3f}s"
            )
    else:
        failures.append("missing_aggregate_live_carrier_scale_smoke_receipt")

    triage_path = run_root / "prelaunch" / "bounded_steps_triage_receipt.json"
    if triage_path.is_file():
        triage = _read_json(triage_path)
        triage_class = triage.get("bounded_steps_triage_class")
        checks["bounded_steps_triage_class"] = triage_class
        if triage_class == "WRAPPER_BUDGET_TOO_TIGHT":
            failures.append("bounded_steps_triage_still_wrapper_budget_too_tight")
    else:
        failures.append("missing_bounded_steps_triage_receipt")

    if PARENT_PT.is_file():
        actual_parent_sha = hashlib.sha256(PARENT_PT.read_bytes()).hexdigest()
        checks["parent_pt_sha256"] = actual_parent_sha
        if actual_parent_sha != PARENT_SHA:
            failures.append("parent_pt_sha_mismatch")
    else:
        failures.append("parent_pt_missing")

    path_evidence_path = run_root / "prelaunch" / "cap_selection_path_evidence_receipt.json"
    if path_evidence_path.is_file():
        path_evidence = _read_json(path_evidence_path)
        overall_route = path_evidence.get("overall_terminal_route")
        checks["cap_selection_path_evidence_overall_route"] = overall_route
        if overall_route != EXPECTED_PATH_ROUTE:
            failures.append(f"unexpected_cap_selection_path_route:{overall_route}")
        per_arm = path_evidence.get("per_arm", [])
        checks["cap_selection_path_evidence_per_arm"] = per_arm
        for arm_row in per_arm:
            if arm_row.get("terminal_route") != EXPECTED_PATH_ROUTE:
                failures.append(
                    f"arm_not_gpu_seam:{arm_row.get('arm')}:{arm_row.get('terminal_route')}"
                )
        if path_evidence.get("schema") != "hrm_text_158_slice5_cap_selection_path_evidence/v1":
            failures.append("cap_selection_path_evidence_schema_missing")
    else:
        failures.append("missing_cap_selection_path_evidence_receipt")

    baseline_sparse = _sparse_cap_elapsed_by_step(run_root / "baseline_snapshot_off")
    instrumented_sparse = _sparse_cap_elapsed_by_step(run_root / "instrumented_snapshot_on")
    checks["baseline_sparse_cap_apply_elapsed_by_step"] = baseline_sparse
    checks["instrumented_sparse_cap_apply_elapsed_by_step"] = instrumented_sparse
    checks["v6g_cpu_sparse_cap_baseline_seconds_by_step"] = dict(
        V6G_CPU_BASELINE_SPARSE_CAP_SECONDS
    )
    checks["material_speedup_ratio_threshold"] = MATERIAL_SPEEDUP_RATIO_THRESHOLD

    speedup_report: dict[str, dict[str, Any]] = {}
    for arm_name, sparse_by_step in (
        ("baseline", baseline_sparse),
        ("instrumented", instrumented_sparse),
    ):
        arm_report: dict[str, Any] = {}
        for step in range(1, max_steps_hard + 1):
            key = str(step)
            elapsed = sparse_by_step.get(key)
            v6g_cpu = V6G_CPU_BASELINE_SPARSE_CAP_SECONDS.get(key)
            if elapsed is None:
                failures.append(f"missing_{arm_name}_sparse_cap_elapsed_step_{step}")
                continue
            if v6g_cpu is None:
                failures.append(f"missing_v6g_cpu_baseline_step_{step}")
                continue
            ratio = float(elapsed) / float(v6g_cpu)
            arm_report[key] = {
                "elapsed_seconds": elapsed,
                "v6g_cpu_baseline_seconds": v6g_cpu,
                "ratio_to_v6g_cpu": ratio,
                "material_win": ratio < MATERIAL_SPEEDUP_RATIO_THRESHOLD,
            }
            if ratio >= MATERIAL_SPEEDUP_RATIO_THRESHOLD:
                failures.append(
                    f"sparse_cap_not_materially_faster_{arm_name}_step_{step}:"
                    f"{elapsed:.3f}s_vs_v6g_{v6g_cpu:.3f}s_ratio_{ratio:.3f}"
                )
        speedup_report[arm_name] = arm_report
    checks["sparse_cap_speedup_vs_v6g_cpu_baseline"] = speedup_report

    cost_shift_report: dict[str, Any] = {}
    for arm_name, arm_dir in (
        ("baseline", run_root / "baseline_snapshot_off"),
        ("instrumented", run_root / "instrumented_snapshot_on"),
    ):
        sparse_by_step = (
            baseline_sparse if arm_name == "baseline" else instrumented_sparse
        )
        subphase_by_step = _subphase_elapsed_by_step(arm_dir)
        nesting_failures = _subphase_nesting_failures(arm_dir, arm=arm_name)
        failures.extend(nesting_failures)
        shift_failures = _sparse_cap_cost_shift_failures(
            arm=arm_name,
            sparse_by_step=sparse_by_step,
            subphase_by_step=subphase_by_step,
            max_steps_hard=max_steps_hard,
        )
        failures.extend(shift_failures)
        cost_shift_report[arm_name] = {
            "subphase_elapsed_by_step": subphase_by_step,
            "nesting_failures": nesting_failures,
            "cost_shift_failures": shift_failures,
        }
    checks["sparse_cap_cost_shift_report"] = cost_shift_report

    checks["baseline_sparse_cap_subphase_elapsed_totals"] = _subphase_elapsed_totals(
        run_root / "baseline_snapshot_off"
    )
    checks["instrumented_sparse_cap_subphase_elapsed_totals"] = _subphase_elapsed_totals(
        run_root / "instrumented_snapshot_on"
    )

    return {
        "schema": "hrm_text_158_slice5_v6h_postrun_acceptance_validate/v1",
        "run_root": str(run_root),
        "aggregate_receipt_path": AGGREGATE_RECEIPT,
        "pass": not failures,
        "failures": failures,
        "checks": checks,
        "expected_path_route": EXPECTED_PATH_ROUTE,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-steps-hard", type=int, default=3)
    args = parser.parse_args(argv)

    run_root = Path(args.run_root)
    out = Path(args.out)
    receipt = validate_v6h_acceptance(run_root, max_steps_hard=args.max_steps_hard)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
