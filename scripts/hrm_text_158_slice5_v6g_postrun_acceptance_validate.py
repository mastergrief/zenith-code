#!/usr/bin/env python3
"""Post-run acceptance validator for v6g outer-budget smoke packet."""

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
EXPECTED_PATH_ROUTE = "PATH_CPU_RESIDENT_CAP_REFERENCE"
AGGREGATE_RECEIPT = "prelaunch/live_carrier_scale_smoke_receipt.json"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _launch_rc(run_root: Path, arm: str) -> int | None:
    rc_path = run_root / "prelaunch" / f"{arm}_launch_rc.txt"
    if not rc_path.is_file():
        return None
    text = rc_path.read_text(encoding="utf-8").strip()
    return int(text) if text.isdigit() else None


def validate_v6g_acceptance(run_root: Path, *, max_steps_hard: int = 3) -> dict[str, Any]:
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
        if peak_emit is None:
            failures.append("aggregate_peak_per_step_emit_delta_seconds_null")
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
        if overall_route not in {
            EXPECTED_PATH_ROUTE,
            "PATH_GPU_SEAM_EXERCISED",
            "SUBMILESTONE_INSTRUMENTATION_INVALID",
        }:
            failures.append(f"cap_selection_path_evidence_route_missing_or_invalid:{overall_route}")
        if overall_route != EXPECTED_PATH_ROUTE:
            failures.append(f"unexpected_cap_selection_path_route:{overall_route}")
        if path_evidence.get("schema") != "hrm_text_158_slice5_cap_selection_path_evidence/v1":
            failures.append("cap_selection_path_evidence_schema_missing")
    else:
        failures.append("missing_cap_selection_path_evidence_receipt")

    return {
        "schema": "hrm_text_158_slice5_v6g_postrun_acceptance_validate/v1",
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
    receipt = validate_v6g_acceptance(run_root, max_steps_hard=args.max_steps_hard)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
