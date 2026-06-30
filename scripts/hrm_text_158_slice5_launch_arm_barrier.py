#!/usr/bin/env python3
"""Launch arm barrier: wait for both probe arms to exit before postrun receipts."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

BARRIER_SCHEMA = "hrm_text_158_slice5_launch_arm_barrier_receipt/v1"
ARMS = ("baseline_snapshot_off", "instrumented_snapshot_on")
BASELINE_RC = "baseline_launch_rc.txt"
INSTRUMENTED_RC = "instrumented_launch_rc.txt"


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _arm_pid(scratch: Path) -> int | None:
    pid_path = scratch / "probe.pid"
    if not pid_path.is_file():
        return None
    text = pid_path.read_text(encoding="utf-8").strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _milestone_quiescence_met(run_root: Path) -> bool:
    for arm in ARMS:
        if not (run_root / arm / "last_active_phase.json").is_file():
            return False
    return True


def _launch_rcs_present(prelaunch: Path) -> bool:
    return (
        (prelaunch / BASELINE_RC).is_file()
        and (prelaunch / INSTRUMENTED_RC).is_file()
    )


def _any_arm_pid_alive(run_root: Path) -> bool:
    for arm in ARMS:
        pid = _arm_pid(run_root / arm)
        if pid is not None and _pid_alive(pid):
            return True
    return False


def wait_launch_arm_barrier(
    *,
    run_root: Path,
    timeout_seconds: float = 7200.0,
    poll_interval_seconds: float = 5.0,
) -> dict[str, Any]:
    prelaunch = run_root / "prelaunch"
    deadline = time.monotonic() + float(timeout_seconds)
    failures: list[str] = []
    while time.monotonic() < deadline:
        if _launch_rcs_present(prelaunch) and not _any_arm_pid_alive(run_root):
            if _milestone_quiescence_met(run_root):
                return {
                    "schema": BARRIER_SCHEMA,
                    "run_root": str(run_root),
                    "pass": True,
                    "launch_rcs_present": True,
                    "any_arm_pid_alive": False,
                    "milestone_quiescence_met": True,
                    "failures": failures,
                }
        time.sleep(float(poll_interval_seconds))
    if not _launch_rcs_present(prelaunch):
        failures.append("launch_rc_files_missing")
    if _any_arm_pid_alive(run_root):
        failures.append("arm_pid_still_alive")
    if not _milestone_quiescence_met(run_root):
        failures.append("milestone_quiescence_missing")
    return {
        "schema": BARRIER_SCHEMA,
        "run_root": str(run_root),
        "pass": False,
        "launch_rcs_present": _launch_rcs_present(prelaunch),
        "any_arm_pid_alive": _any_arm_pid_alive(run_root),
        "milestone_quiescence_met": _milestone_quiescence_met(run_root),
        "failures": failures,
    }


def assert_postrun_barrier_ready(*, run_root: Path) -> dict[str, Any]:
    """Fail-closed check used before postrun receipt commands."""
    prelaunch = run_root / "prelaunch"
    failures: list[str] = []
    if not _launch_rcs_present(prelaunch):
        failures.append("launch_rc_files_missing")
    if _any_arm_pid_alive(run_root):
        failures.append("arm_pid_still_alive")
    if not _milestone_quiescence_met(run_root):
        failures.append("milestone_quiescence_missing")
    return {
        "schema": BARRIER_SCHEMA,
        "run_root": str(run_root),
        "pass": not failures,
        "launch_rcs_present": _launch_rcs_present(prelaunch),
        "any_arm_pid_alive": _any_arm_pid_alive(run_root),
        "milestone_quiescence_met": _milestone_quiescence_met(run_root),
        "failures": failures,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=7200.0)
    parser.add_argument("--poll-interval-seconds", type=float, default=5.0)
    parser.add_argument(
        "--assert-only",
        action="store_true",
        help="Fail-closed barrier check without waiting (for postrun commands).",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.assert_only:
        receipt = assert_postrun_barrier_ready(run_root=args.run_root)
    else:
        receipt = wait_launch_arm_barrier(
            run_root=args.run_root,
            timeout_seconds=float(args.timeout_seconds),
            poll_interval_seconds=float(args.poll_interval_seconds),
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
