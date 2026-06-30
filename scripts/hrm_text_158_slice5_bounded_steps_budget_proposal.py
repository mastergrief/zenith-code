#!/usr/bin/env python3
"""Bounded-steps budget proposal from measured milestone costs (triage-gated)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from scripts.hrm_text_158_slice5_bounded_steps_triage import triage_bounded_steps

PROPOSAL_SCHEMA = "hrm_text_158_slice5_bounded_steps_budget_proposal/v1"
PHASE_BUDGETS = {
    "sparse_cap_apply": 180,
    "step_forward_backward": 90,
    "sparse_vote_construction": 120,
    "live_carrier_snapshot_emit": 60,
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def propose_bounded_steps_budget(
    *,
    run_root: Path,
    max_steps_hard: int,
) -> dict[str, Any]:
    triage = triage_bounded_steps(run_root=run_root, max_steps_hard=int(max_steps_hard))
    if triage.get("bounded_steps_triage_class") != "WRAPPER_BUDGET_TOO_TIGHT":
        return {
            "schema": PROPOSAL_SCHEMA,
            "pass": False,
            "reason": "triage_not_wrapper_budget_too_tight",
            "triage_class": triage.get("bounded_steps_triage_class"),
            "budget_proposal_seconds": None,
        }

    instr = run_root / "instrumented_snapshot_on"
    per_step_sparse_cap: list[float] = []
    for row in _read_jsonl(instr / "liveness_milestones" / "sparse_cap_apply.jsonl"):
        if row.get("milestone_kind") == "phase_complete":
            per_step_sparse_cap.append(float(row.get("elapsed_since_phase_enter_seconds", 0.0)))
    per_step_overhead = 0.0
    for phase_id in ("step_forward_backward", "sparse_vote_construction", "live_carrier_snapshot_emit"):
        rows = _read_jsonl(instr / "liveness_milestones" / f"{phase_id}.jsonl")
        elapsed = [
            float(row.get("elapsed_since_phase_enter_seconds", 0.0))
            for row in rows
            if row.get("milestone_kind") == "phase_complete"
        ]
        if elapsed:
            per_step_overhead += sum(elapsed) / len(elapsed)

    measured_per_step = [
        float(sparse) + float(per_step_overhead) for sparse in per_step_sparse_cap
    ]
    observed_total = float(
        (triage.get("per_arm") or {}).get("instrumented", {}).get(
            "elapsed_since_start_seconds"
        )
        or 0.0
    )
    symbolic_per_step = sum(PHASE_BUDGETS.values())
    proposal = float(max_steps_hard) * max(measured_per_step or [symbolic_per_step])
    return {
        "schema": PROPOSAL_SCHEMA,
        "pass": True,
        "triage_class": triage.get("bounded_steps_triage_class"),
        "max_steps_hard": int(max_steps_hard),
        "measured_per_step_sparse_cap_seconds": per_step_sparse_cap,
        "measured_per_step_total_seconds": measured_per_step,
        "observed_total_seconds": observed_total,
        "symbolic_per_step_budget_sum": symbolic_per_step,
        "budget_proposal_seconds": round(proposal, 3),
        "formula": "max_steps_hard * max(measured_per_step_total_seconds)",
        "note": "proposal only; no launch-packet mutation in Slice B-DIAG",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--max-steps-hard", type=int, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    receipt = propose_bounded_steps_budget(
        run_root=args.run_root,
        max_steps_hard=int(args.max_steps_hard),
    )
    text = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0 if receipt.get("pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
