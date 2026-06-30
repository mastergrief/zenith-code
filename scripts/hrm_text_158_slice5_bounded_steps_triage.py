#!/usr/bin/env python3
"""Bounded-steps triage for Slice-5 smoke runs (classification/harness only)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

TRIAGE_SCHEMA = "hrm_text_158_slice5_bounded_steps_triage_receipt/v1"
ARMS = ("baseline_snapshot_off", "instrumented_snapshot_on")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _steps_completed(scratch: Path) -> int:
    receipt = _read_json(scratch / "receipt.json")
    if receipt.get("steps_completed") is not None:
        return int(receipt["steps_completed"])
    sparse_cap = _read_jsonl(scratch / "liveness_milestones" / "sparse_cap_apply.jsonl")
    return len(
        [
            row
            for row in sparse_cap
            if row.get("milestone_kind") == "phase_complete"
        ]
    )


def _phase_guard(scratch: Path) -> dict[str, Any]:
    return _read_json(scratch / "last_active_phase.json")


def triage_bounded_steps(
    *,
    run_root: Path,
    max_steps_hard: int,
) -> dict[str, Any]:
    baseline = run_root / "baseline_snapshot_off"
    instrumented = run_root / "instrumented_snapshot_on"
    steps_base = _steps_completed(baseline)
    steps_instr = _steps_completed(instrumented)
    guard_base = _phase_guard(baseline)
    guard_instr = _phase_guard(instrumented)

    evidence: list[dict[str, Any]] = []
    per_arm: dict[str, Any] = {}

    for arm_name, scratch, steps_done in (
        ("baseline", baseline, steps_base),
        ("instrumented", instrumented, steps_instr),
    ):
        guard = _phase_guard(scratch)
        per_arm[arm_name] = {
            "steps_completed": int(steps_done),
            "phase_guard_phase": guard.get("phase"),
            "phase_guard_failure": guard.get("failure_class"),
            "elapsed_since_start_seconds": guard.get("elapsed_since_start_seconds"),
        }

    instr_outer_timeout = (
        steps_instr >= int(max_steps_hard)
        and guard_instr.get("failure_class") == "LIVENESS_FAILURE"
        and guard_instr.get("phase") == "bounded_steps"
    )
    base_step_stall = (
        guard_base.get("failure_class") == "LIVENESS_FAILURE"
        and guard_base.get("phase") == "sparse_cap_apply"
        and steps_base < int(max_steps_hard)
    )

    if instr_outer_timeout:
        bounded_steps_triage_class = "WRAPPER_BUDGET_TOO_TIGHT"
        evidence.append(
            {
                "arm": "instrumented",
                "reason": "max_steps_completed_then_outer_bounded_steps_timeout",
                "steps_completed": steps_instr,
                "max_steps_hard": int(max_steps_hard),
                "elapsed": guard_instr.get("elapsed_since_start_seconds"),
            }
        )
    elif base_step_stall:
        bounded_steps_triage_class = "PER_STEP_STALL_CONFIRMED"
        evidence.append(
            {
                "arm": "baseline",
                "reason": "sparse_cap_apply_step_stall",
                "steps_completed": steps_base,
                "phase": guard_base.get("phase"),
            }
        )
    else:
        bounded_steps_triage_class = "CLASSIFIER_ORDERING_MASK"
        evidence.append({"reason": "no_primary_wrapper_or_step_stall_signature"})

    return {
        "schema": TRIAGE_SCHEMA,
        "run_root": str(run_root),
        "max_steps_hard": int(max_steps_hard),
        "bounded_steps_triage_class": bounded_steps_triage_class,
        "per_arm": per_arm,
        "evidence": evidence,
        "instrumented_outer_timeout_after_max_steps": bool(instr_outer_timeout),
        "baseline_sparse_cap_step_stall": bool(base_step_stall),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--max-steps-hard", type=int, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    receipt = triage_bounded_steps(
        run_root=args.run_root,
        max_steps_hard=int(args.max_steps_hard),
    )
    text = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
