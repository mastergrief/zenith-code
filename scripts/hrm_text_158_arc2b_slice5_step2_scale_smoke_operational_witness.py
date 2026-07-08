#!/usr/bin/env python3
"""Mechanical operational witness for Arc #2b Slice-5 Step-2 scale_smoke."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

RECEIPT_SCHEMA = "hrm_text_158_arc2b_slice5_step2_scale_smoke_operational_witness/v1"
DIAGNOSTIC_SUBDIR = "d_recompute_window_diagnostic"
LIVENESS_FAILURE_CLASS = "LIVENESS_FAILURE"
WARMUP_EXHAUSTED_REASON = "liveness_failure_exhausted_retries"
DEFAULT_SMOKE_STEPS = 5
DECAY_NUMERATOR = 1
DECAY_DENOMINATOR = 2


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _check_last_active_phase(
    path: Path,
    failures: list[str],
    *,
    label: str,
    require_when_smoke_complete: bool,
    smoke_complete: bool,
) -> None:
    if not path.is_file():
        if require_when_smoke_complete and smoke_complete:
            failures.append(f"{label}_last_active_phase_missing")
        return

    phase = _load_json(path)
    if "failure_class" not in phase and "guard_event" not in phase:
        if "liveness_failure" in phase:
            failures.append(f"{label}_stale_boolean_only_liveness_failure")
        else:
            failures.append(f"{label}_last_active_phase_missing_schema_fields")
        return

    if str(phase.get("failure_class")) == LIVENESS_FAILURE_CLASS:
        failures.append(f"{label}_failure_class_liveness_failure")

    if "liveness_failure" in phase and bool(phase.get("liveness_failure")):
        failures.append(f"{label}_liveness_failure_true")


def _check_warmup_retry_witness(path: Path, failures: list[str]) -> None:
    if not path.is_file():
        return
    witness = _load_json(path)
    if "final_rc" not in witness:
        failures.append("warmup_retry_final_rc_missing")
    elif int(witness["final_rc"]) != 0:
        failures.append("warmup_retry_final_rc_nonzero")
    if "final_reason" not in witness:
        failures.append("warmup_retry_final_reason_missing")
    elif str(witness["final_reason"]) == WARMUP_EXHAUSTED_REASON:
        failures.append("warmup_retry_liveness_failure_exhausted_retries")


def build_operational_witness(
    run_root: Path,
    *,
    smoke_steps: int = DEFAULT_SMOKE_STEPS,
) -> dict[str, Any]:
    failures: list[str] = []
    run_root = Path(run_root)
    scratch = run_root / DIAGNOSTIC_SUBDIR
    probe_receipt_path = scratch / "receipt.json"
    probe_receipt: dict[str, Any] = {}
    steps_completed = 0

    if not probe_receipt_path.is_file():
        failures.append("missing_probe_receipt")
    else:
        probe_receipt = _load_json(probe_receipt_path)
        steps_completed = int(probe_receipt.get("steps_completed") or 0)
        if steps_completed != int(smoke_steps):
            failures.append(f"steps_completed_{steps_completed}_expected_{int(smoke_steps)}")

    smoke_complete = steps_completed == int(smoke_steps)

    _check_last_active_phase(
        scratch / "last_active_phase.json",
        failures,
        label="smoke",
        require_when_smoke_complete=True,
        smoke_complete=smoke_complete,
    )

    warmup_witness_path = run_root / "prelaunch" / "calibration_warmup_retry_witness.json"
    _check_warmup_retry_witness(warmup_witness_path, failures)

    _check_last_active_phase(
        run_root / "calibration_warmup" / "last_active_phase.json",
        failures,
        label="calibration_warmup",
        require_when_smoke_complete=False,
        smoke_complete=smoke_complete,
    )

    global_summary = dict(probe_receipt.get("bounded_delta_global_summary") or {})
    parallel_mode = str(global_summary.get("sparse_cap_apply_parallel_mode") or "")
    if parallel_mode != "serial_cpu":
        failures.append(f"sparse_cap_apply_parallel_mode_{parallel_mode or 'missing'}_expected_serial_cpu")

    canonical_live = scratch / "live_carrier_snapshot.jsonl"
    doubled_live = scratch / DIAGNOSTIC_SUBDIR / "live_carrier_snapshot.jsonl"
    if doubled_live.is_file():
        failures.append("doubled_live_carrier_snapshot_path_present")
    if smoke_complete and not canonical_live.is_file():
        failures.append("missing_canonical_live_carrier_snapshot")

    log_path = scratch / "recompute_window_log.jsonl"
    log_rows = _load_jsonl_rows(log_path)
    if len(log_rows) != int(smoke_steps):
        failures.append(f"recompute_window_log_rows_{len(log_rows)}_expected_{int(smoke_steps)}")
    else:
        for index, row in enumerate(log_rows, start=1):
            replay_constants = dict(row.get("replay_constants") or {})
            num = int(replay_constants.get("decay_numerator", -1))
            den = int(replay_constants.get("decay_denominator", -1))
            if num != DECAY_NUMERATOR or den != DECAY_DENOMINATOR:
                failures.append(
                    f"recompute_window_log_row_{index}_decay_{num}_{den}_expected_{DECAY_NUMERATOR}_{DECAY_DENOMINATOR}"
                )

    return {
        "schema": RECEIPT_SCHEMA,
        "run_root": str(run_root),
        "smoke_steps": int(smoke_steps),
        "steps_completed": int(steps_completed),
        "failures": failures,
        "pass": not failures,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--smoke-steps", type=int, default=DEFAULT_SMOKE_STEPS)
    args = parser.parse_args(argv)

    receipt = build_operational_witness(
        args.run_root,
        smoke_steps=int(args.smoke_steps),
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if receipt.get("pass") is not True:
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 1
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
