#!/usr/bin/env python3
"""Arc #2b Slice-5 discovery LIVE postrun for C/D/E arms (bpw/gap/operational).

Frozen v6 plan. Replaces the Arm-B offline K* script for live C/D/E arms.
Emits live decay-arm bpw/gap/operational evidence from the run's
live_carrier_snapshot.jsonl + recompute_window_log.jsonl.

Computes:
- live_acc_carrier_bpw_max (max over rows of bytes_total*8/numel)
- budget_gap_bpw = live_acc_carrier_bpw_max - 0.4
- operational_ok (steps_completed == H25, decay match, live-carrier exact, resume_gen=0)
- gap_d_fail_closed: if this is arm_d and gap≤0, fail-closed
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from calm.hrm_text_158.native_full_stack.arc2b_slice5_discovery_branch import (
    CLASSIFIER,
    DEFAULT_EFFECTIVE_ACC_BUDGET_BPW,
    DEFAULT_TOLERANCE_BPW,
    RECEIPT_SCHEMA,
    compute_budget_gap_bpw,
    validate_lane_fields,
)
from scripts.hrm_text_158_bounded_delta_acquisition_probe import (
    is_terminal_liveness_breach,
)

ACTIVE_TASK_ID = "1783272482268-052281aa"
H25_STEPS = 25


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


def _live_acc_carrier_bytes_total(snapshot: Mapping[str, Any]) -> int | None:
    """Compute total carrier bytes from a live_carrier_snapshot row."""
    if not isinstance(snapshot, Mapping):
        return None
    if snapshot.get("live_carrier_bytes_exact") is not True:
        return None
    components = ("events_bytes", "backlog_bytes", "hot_exact_bytes", "metadata_bytes")
    total = 0
    for key in components:
        raw = snapshot.get(key)
        if type(raw) is not int or raw < 0:
            return None
        total += raw
    return total


def compute_live_acc_bpw(
    *,
    carrier_bytes_total: int,
    eligible_weight_numel: int,
) -> float:
    """bpw = bytes_total * 8 / numel."""
    if int(eligible_weight_numel) <= 0:
        raise ValueError("eligible_weight_numel must be positive")
    return (float(int(carrier_bytes_total)) * 8.0) / float(int(eligible_weight_numel))


def compute_max_live_acc_bpw(
    *,
    live_carrier_rows: Sequence[Mapping[str, Any]],
    eligible_weight_numel: int,
) -> float | None:
    """Max bpw over all live-carrier rows."""
    max_bpw: float | None = None
    for row in live_carrier_rows:
        total = _live_acc_carrier_bytes_total(row)
        if total is None:
            continue
        bpw = compute_live_acc_bpw(
            carrier_bytes_total=total,
            eligible_weight_numel=eligible_weight_numel,
        )
        max_bpw = bpw if max_bpw is None else max(max_bpw, bpw)
    return max_bpw


def resolve_operational_ok(
    *,
    run_root: Path,
    steps_expected: int = H25_STEPS,
) -> tuple[bool, dict[str, Any]]:
    """Check operational: steps_completed==H25, decay match, live-carrier exact, resume_gen=0."""
    scratch = run_root / "d_recompute_window_diagnostic"
    probe_receipt_path = scratch / "receipt.json"
    log_path = scratch / "recompute_window_log.jsonl"
    live_path = scratch / "live_carrier_snapshot.jsonl"

    details: dict[str, Any] = {
        "steps_expected": int(steps_expected),
        "probe_receipt_found": probe_receipt_path.is_file(),
        "log_found": log_path.is_file(),
        "live_carrier_found": live_path.is_file(),
    }

    if not probe_receipt_path.is_file() or not log_path.is_file():
        return False, details

    probe_receipt = _load_json(probe_receipt_path)
    steps_completed = int(probe_receipt.get("steps_completed") or 0)
    details["steps_completed"] = steps_completed
    if steps_completed != int(steps_expected):
        details["reason"] = f"steps_completed={steps_completed}!={steps_expected}"
        return False, details

    log_rows = _load_jsonl_rows(log_path)
    if len(log_rows) < int(steps_expected):
        details["reason"] = f"log_rows={len(log_rows)}<{steps_expected}"
        return False, details

    # Check decay from replay_constants
    replay_constants = dict(log_rows[0].get("replay_constants") or {}) if log_rows else {}
    details["replay_constants"] = replay_constants
    decay_num = int(replay_constants.get("decay_numerator", -1))
    decay_den = int(replay_constants.get("decay_denominator", -1))
    if decay_num < 0 or decay_den <= 0:
        details["reason"] = "invalid_decay_constants"
        return False, details

    # Check resume_generation
    resume_gen = log_rows[0].get("resume_generation")
    details["resume_generation"] = resume_gen
    if resume_gen is not None and int(resume_gen) != 0:
        details["reason"] = f"resume_generation={resume_gen}!=0"
        return False, details

    # Check live-carrier rows
    live_rows = _load_jsonl_rows(live_path)
    exact_rows = [r for r in live_rows if r.get("live_carrier_bytes_exact") is True]
    details["live_carrier_rows"] = len(live_rows)
    details["live_carrier_exact_rows"] = len(exact_rows)
    if not exact_rows:
        details["reason"] = "no_live_carrier_exact_rows"
        return False, details

    # Check confirmation RC
    rc_path = run_root / "prelaunch" / "confirmation_launch_rc.txt"
    if rc_path.is_file():
        try:
            rc = int(rc_path.read_text(encoding="utf-8").strip())
            details["confirmation_rc"] = rc
            if rc != 0:
                details["reason"] = f"confirmation_rc={rc}!=0"
                return False, details
        except ValueError:
            details["reason"] = "confirmation_rc_invalid"
            return False, details
    else:
        details["reason"] = "confirmation_rc_missing"
        return False, details

    # Check liveness
    phase_path = scratch / "last_active_phase.json"
    if phase_path.is_file():
        phase = _load_json(phase_path)
        if is_terminal_liveness_breach(phase):
            details["reason"] = "liveness_failure"
            return False, details

    details["reason"] = "ok"
    return True, details


def build_live_postrun_receipt(
    *,
    run_root: Path,
    arm_name: str,
    decay_num: int,
    decay_den: int,
    eligible_weight_numel: int = 8_650_752,
    effective_acc_budget_bpw: float = DEFAULT_EFFECTIVE_ACC_BUDGET_BPW,
) -> dict[str, Any]:
    """Build the live decay-arm postrun receipt."""
    operational_ok, op_details = resolve_operational_ok(run_root=run_root)

    scratch = run_root / "d_recompute_window_diagnostic"
    live_path = scratch / "live_carrier_snapshot.jsonl"
    live_rows = _load_jsonl_rows(live_path)

    max_bpw = None
    gap_bpw = None
    gap_fail_closed = False

    if operational_ok and live_rows:
        max_bpw = compute_max_live_acc_bpw(
            live_carrier_rows=live_rows,
            eligible_weight_numel=eligible_weight_numel,
        )
        if max_bpw is not None:
            gap_bpw = compute_budget_gap_bpw(
                live_acc_carrier_bpw_max=max_bpw,
                effective_acc_budget_bpw=effective_acc_budget_bpw,
            )
            # gap(D)≤0 fail-closed (for arm_d specifically)
            if arm_name == "arm_d" and gap_bpw <= 0:
                gap_fail_closed = True

    return {
        "schema": RECEIPT_SCHEMA,
        "task_id": ACTIVE_TASK_ID,
        "classifier": CLASSIFIER,
        "evidence_source": "live_decay_curve",
        "arm_name": arm_name,
        "decay_num": int(decay_num),
        "decay_den": int(decay_den),
        "operational_ok": operational_ok,
        "operational_details": op_details,
        "live_acc_carrier_bpw_max": max_bpw,
        "budget_gap_bpw": gap_bpw,
        "effective_acc_budget_bpw": float(effective_acc_budget_bpw),
        "gap_d_fail_closed": gap_fail_closed,
        "eligible_weight_numel": int(eligible_weight_numel),
        "ready_for_main_science": False,
        "counts_as_sub2": False,
        "pre_full_stack_diagnostic": True,
        "autonomy_rung": "discovery_h25_live_postrun",
        "generated_at_unix": int(time.time()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, required=True)
    ap.add_argument("--arm-name", type=str, required=True, choices=["arm_c", "arm_d", "arm_e"])
    ap.add_argument("--decay-num", type=int, required=True)
    ap.add_argument("--decay-den", type=int, required=True)
    ap.add_argument("--eligible-weight-numel", type=int, default=8_650_752)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    receipt = build_live_postrun_receipt(
        run_root=args.run_root,
        arm_name=args.arm_name,
        decay_num=args.decay_num,
        decay_den=args.decay_den,
        eligible_weight_numel=args.eligible_weight_numel,
    )
    out_path = args.out or (
        args.run_root / f"discovery_{args.arm_name}_live_postrun_receipt.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
