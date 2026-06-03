#!/usr/bin/env python3
"""Fixture tests for ai_room_heartbeat_watchdog.py (hardened v1).

CPU-static / no-loop: builds a temp channel-log JSONL + invokes the watchdog via
subprocess with --dry-run + --now override + injected movement signals
(AIWD_TEST_FILE_FRESH / AIWD_TEST_PROC_CORRELATED), asserting the decided
ACTION/WAKE without touching the real room or real machine state.

Covers co_lead's acceptance + the three hardening fixtures that must FAIL on the
pre-hardening code (1780474157599): prior-EXTEND + no fresh movement => STALL/
RECYCLE; unrelated terminal from worker B must not clean worker A; process-only
(no fresh artifact) in a code phase must NOT extend.

Run: python3 .claude/hooks/test_ai_room_heartbeat_watchdog.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

WD = Path(__file__).with_name("ai_room_heartbeat_watchdog.py")

T_HB = "2026-06-03T00:00:00Z"
NOW_OVERDUE = "2026-06-03T01:00:00Z"     # +1h ⇒ past explicit due(00:10)+grace(10m)
NOW_SOON = "2026-06-03T00:05:00Z"
DUE_EXP = "2026-06-03T00:10:00Z"
T_WD = "2026-06-03T00:30:00Z"            # a prior watchdog post time (after hb)
TASK = "1780347615017-1538f834"


def rec(frm, body, ts=T_HB, kind="status_update", mid=None):
    return json.dumps({"ts": ts, "id": mid or f"{ts}-{frm}", "from": frm,
                       "kind": kind, "body": body})


def hb_body(phase="cpu-proof", due=DUE_EXP, with_due=True, task=TASK):
    due_part = f"next_heartbeat_due={due} " if with_due else ""
    return (f"IMPLEMENTING active-cap. task {task} {due_part}phase={phase} "
            f"expected_next_artifact=/home/gabe/x/run.json")


def run(log_lines, now, file_fresh=None, proc_corr=None):
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as fh:
        fh.write("\n".join(log_lines) + "\n")
        path = fh.name
    env = {"PATH": "/usr/bin:/bin"}
    if file_fresh is not None:
        env["AIWD_TEST_FILE_FRESH"] = file_fresh
    if proc_corr is not None:
        env["AIWD_TEST_PROC_CORRELATED"] = proc_corr
    proc = subprocess.run(
        [sys.executable, str(WD), "--dry-run", "--channel-log", path, "--now", now],
        capture_output=True, text=True, env=env)
    Path(path).unlink(missing_ok=True)
    return proc.stdout


CASES = []


def case(name, log_lines, now, ff, pc, expect):
    CASES.append((name, log_lines, now, ff, pc, expect))


# 1. stale + no movement (code phase) -> STALL (wake), 1st miss => re-drive
case("stale_no_movement", [rec("codex", hb_body(), mid="hb1")],
     NOW_OVERDUE, "0", "0", ["ACTION=stall", "WAKE=True", "RECOMMEND re-drive"])

# 2. fresh file movement -> EXTEND
case("live_movement", [rec("codex", hb_body(), mid="hb2")],
     NOW_OVERDUE, "1", "0", ["ACTION=extend", "WAKE=False", "WATCHDOG_HEARTBEAT_EXTEND"])

# 3. file moved (1st pass, no prior watchdog) -> EXTEND
case("file_moved_first_pass", [rec("codex", hb_body(phase="gpu-proof"), mid="hb3")],
     NOW_OVERDUE, "1", "0", ["ACTION=extend", "WAKE=False"])

# 4. HARDENING: prior EXTEND + NO fresh movement -> STALL + RECYCLE (escalation)
case("prior_extend_no_fresh_movement",
     [rec("codex", hb_body(), mid="hb4"),
      rec("watchdog", f"WATCHDOG_HEARTBEAT_EXTEND worker codex task {TASK} live",
          ts=T_WD, mid="wd_ext")],
     NOW_OVERDUE, "0", "0", ["ACTION=stall", "WAKE=True", "RECOMMEND RECYCLE"])

# 5. missing metadata on gated IMPLEMENTING -> due-soon default -> STALL (not invisible)
case("missing_metadata_due_soon",
     [rec("codex", hb_body(with_due=False), mid="hb5")],
     NOW_OVERDUE, "0", "0", ["ACTION=stall", "WAKE=True", "metadata_present=False"])

# 6. HARDENING: process-only (no fresh file) in a CODE phase -> must NOT extend -> STALL
case("process_only_code_phase_no_extend",
     [rec("codex", hb_body(phase="cpu-proof"), mid="hb6")],
     NOW_OVERDUE, "0", "1", ["ACTION=stall", "WAKE=True"])

# 7. GPU phase + correlated process (no fresh file) -> EXTEND (correlated proc counts)
case("gpu_phase_proc_correlated",
     [rec("codex", hb_body(phase="gpu-proof"), mid="hb7")],
     NOW_OVERDUE, "0", "1", ["ACTION=extend", "WAKE=False"])

# 8. HARDENING: unrelated terminal from worker B must NOT clean worker A's overdue hb
case("worker_B_terminal_doesnt_clean_A",
     [rec("codex", hb_body(phase="cpu-proof"), ts=T_HB, mid="hbA"),
      rec("codex_other", f"VALIDATION RECEIPT — unrelated slice _acquires=false task 1780000000000-deadbeef",
          ts="2026-06-03T00:20:00Z", kind="validation_receipt", mid="vrB")],
     NOW_OVERDUE, "0", "0", ["ACTION=stall", "WAKE=True", "worker codex"])

# 9. not overdue -> CLEAN
case("not_overdue_clean",
     [rec("codex", hb_body(due="2026-06-03T02:00:00Z"), mid="hb9")],
     NOW_SOON, "0", "0", ["CLEAN"])

# 10. own latest post is terminal -> CLEAN (worker done)
case("own_terminal_clean",
     [rec("codex", hb_body(), ts=T_HB, mid="hb10"),
      rec("codex", f"VALIDATION RECEIPT — slice task {TASK} _acquires=false",
          ts="2026-06-03T00:20:00Z", kind="validation_receipt", mid="vr10")],
     NOW_OVERDUE, "0", "0", ["CLEAN"])

# 11. empty log -> CLEAN (fail-quiet)
case("empty_log_clean", [], NOW_OVERDUE, "0", "0", ["CLEAN"])


def main():
    failures = 0
    for name, log_lines, now, ff, pc, expect in CASES:
        out = run(log_lines, now, ff, pc)
        ok = all(s in out for s in expect)
        print(f"{'PASS' if ok else 'FAIL'} {name}: expect {expect}")
        if not ok:
            print(f"   --- got ---\n{out}")
            failures += 1
    print(f"\n{len(CASES) - failures}/{len(CASES)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
