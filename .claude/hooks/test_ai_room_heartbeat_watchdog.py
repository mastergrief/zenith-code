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

import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

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
EXTRA_TESTS = []


def case(name, log_lines, now, ff, pc, expect):
    CASES.append((name, log_lines, now, ff, pc, expect))


def extra_test(name):
    def deco(fn):
        EXTRA_TESTS.append((name, fn))
        return fn
    return deco


def load_watchdog_module():
    spec = importlib.util.spec_from_file_location("ai_room_heartbeat_watchdog_test", WD)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def emit_decision():
    return {"action": "stall", "wake": True, "kind": "status_update", "body": "WATCHDOG_TEST_BODY"}


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


# --- retire-after-K + exact-worker-match hardening (co_lead-converged) ----------
T_S1, T_S2, T_S3 = ("2026-06-03T00:15:00Z", "2026-06-03T00:20:00Z",
                    "2026-06-03T00:25:00Z")
T_RET = "2026-06-03T00:30:00Z"
NOW_LATE = "2026-06-03T02:00:00Z"


def wd_stall(worker, ts, mid, task="?"):
    return rec("watchdog",
               f"WATCHDOG_STALL — heartbeat_overdue. task {task} worker {worker}; "
               f"last hb x phase y. RECOMMEND RECYCLE.", ts=ts, mid=mid)


def wd_retired(worker, ts, mid):
    return rec("watchdog",
               f"WATCHDOG_RETIRED — worker {worker} heartbeat x retired after "
               f"3 no-movement stalls.", ts=ts, mid=mid)


# 12. HARDENING: RETIRE after K exact-worker stalls + no movement -> non-wake RETIRED
case("retire_after_k",
     [rec("codex", hb_body(), mid="hbR"),
      wd_stall("codex", T_S1, "s1"), wd_stall("codex", T_S2, "s2"),
      wd_stall("codex", T_S3, "s3")],
     NOW_OVERDUE, "0", "0", ["ACTION=retire", "WAKE=False", "WATCHDOG_RETIRED"])

# 13. HARDENING: EXACT worker match — codex_1 stalls do NOT count against codex
case("exact_worker_codex1_no_count",
     [rec("codex", hb_body(), mid="hbE"),
      wd_stall("codex_1", T_S1, "c1"), wd_stall("codex_1", T_S2, "c2"),
      wd_stall("codex_1", T_S3, "c3")],
     NOW_OVERDUE, "0", "0", ["ACTION=stall", "RECOMMEND re-drive"])

# 14. HARDENING: already RETIRED for same worker -> CLEAN (no re-emit)
case("already_retired_clean",
     [rec("codex", hb_body(), mid="hbC"),
      wd_stall("codex", T_S1, "k1"), wd_stall("codex", T_S2, "k2"),
      wd_stall("codex", T_S3, "k3"), wd_retired("codex", T_RET, "ret1")],
     NOW_OVERDUE, "0", "0", ["CLEAN"])

# 15. HARDENING: NEWER heartbeat after prior stalls/retired -> re-monitors (n_stall=0)
case("newer_hb_remonitors",
     [wd_stall("codex", T_S1, "n1"), wd_stall("codex", T_S2, "n2"),
      wd_stall("codex", T_S3, "n3"), wd_retired("codex", T_RET, "nret"),
      rec("codex", hb_body(due="2026-06-03T00:50:00Z"),
          ts="2026-06-03T00:40:00Z", mid="hbN")],
     NOW_LATE, "0", "0", ["ACTION=stall", "RECOMMEND re-drive"])

# 16. HARDENING: movement WINS over retire — prior K stalls but fresh movement -> EXTEND
case("moved_wins_over_retire",
     [rec("codex", hb_body(), mid="hbM"),
      wd_stall("codex", T_S1, "m1"), wd_stall("codex", T_S2, "m2"),
      wd_stall("codex", T_S3, "m3")],
     NOW_OVERDUE, "1", "0", ["ACTION=extend", "WAKE=False"])

# 17. HARDENING: SAME-TASK other-worker — codex_1 stalls citing TASK do NOT count
#     against codex (worker attribution is authoritative over the task fallback).
case("same_task_other_worker_no_count",
     [rec("codex", hb_body(), mid="hbST"),
      wd_stall("codex_1", T_S1, "st1", task=TASK),
      wd_stall("codex_1", T_S2, "st2", task=TASK),
      wd_stall("codex_1", T_S3, "st3", task=TASK)],
     NOW_OVERDUE, "0", "0", ["ACTION=stall", "RECOMMEND re-drive"])


# --- unanswered-gate tracking (2026-06-10 ack-idle hardening) -------------------
NOW_ANCIENT = "2026-06-03T02:30:00Z"     # >GATE_MAX_AGE (2h) after T_HB


def grec(target, ts=T_HB, mid="gate1", deadline=300, body=None):
    return json.dumps({"ts": ts, "id": mid, "from": "claude", "to": target,
                       "kind": "task_dispatch", "requires_response_from": target,
                       "response_deadline_secs": deadline,
                       "body": body or f"+1 RUN — execute packet now. task {TASK}"})


def wreply(frm, ts, mid, reply_to=None, body="start signal: running"):
    d = {"ts": ts, "id": mid, "from": frm, "kind": "msg", "body": body}
    if reply_to:
        d["reply_to"] = reply_to
    return json.dumps(d)


def wd_gate(marker, gate_id, ts, mid):
    return rec("watchdog", f"{marker} — worker codex_2: gate {gate_id} ...",
               ts=ts, mid=mid)


# G1. unanswered lapsed gate -> GATE_REWAKE (wake, to worker)
case("gate_unanswered_rewake", [grec("codex_2", mid="g1")],
     NOW_OVERDUE, "0", "0", ["ACTION=gate_rewake", "WAKE=True", "GATE_REWAKE", "g1"])

# G2. answered via threaded reply -> CLEAN
case("gate_answered_reply_clean",
     [grec("codex_2", mid="g2"),
      wreply("codex_2", "2026-06-03T00:02:00Z", "r2", reply_to="g2")],
     NOW_OVERDUE, "0", "0", ["CLEAN"])

# G3. answered via body citation (unthreaded operator report) -> CLEAN
case("gate_answered_citation_clean",
     [grec("codex_2", mid="g3"),
      wreply("codex_2", "2026-06-03T00:30:00Z", "r3",
             body="receipt for gate g3 posted; exit 0")],
     NOW_OVERDUE, "0", "0", ["CLEAN"])

# G4. superseded by a NEWER claude engagement with the same worker -> CLEAN
case("gate_superseded_clean",
     [grec("codex_2", mid="g4"),
      grec("codex_2", ts="2026-06-03T00:40:00Z", mid="g4b",
           body="fresh posture brief; reply with ack"),
      wreply("codex_2", "2026-06-03T00:41:00Z", "r4", reply_to="g4b")],
     NOW_OVERDUE, "0", "0", ["CLEAN"])

# G5. ancient gate (>GATE_MAX_AGE) -> CLEAN (never resurrected)
case("gate_ancient_clean", [grec("codex_2", mid="g5")],
     NOW_ANCIENT, "0", "0", ["CLEAN"])

# G6. two prior GATE_REWAKEs -> escalate to claude (wake)
case("gate_escalates_after_max_rewakes",
     [grec("codex_2", mid="g6"),
      wd_gate("GATE_REWAKE", "g6", T_S1, "gw1"),
      wd_gate("GATE_REWAKE", "g6", T_S2, "gw2")],
     NOW_OVERDUE, "0", "0", ["ACTION=gate_escalate", "WAKE=True", "RECOMMEND claude"])

# G7. already escalated -> CLEAN (single escalation, no cry-wolf)
case("gate_after_escalate_clean",
     [grec("codex_2", mid="g7"),
      wd_gate("GATE_REWAKE", "g7", T_S1, "ge1"),
      wd_gate("GATE_REWAKE", "g7", T_S2, "ge2"),
      wd_gate("GATE_ESCALATE", "g7", T_S3, "ge3")],
     NOW_OVERDUE, "0", "0", ["CLEAN"])

# G8. deadline not yet lapsed -> CLEAN
case("gate_not_due_clean", [grec("codex_2", mid="g8", deadline=86400)],
     NOW_OVERDUE, "0", "0", ["CLEAN"])

# G9. requires_response_from co_lead (non-worker) -> CLEAN (exempt)
case("gate_colead_exempt_clean", [grec("codex_co_lead", mid="g9")],
     NOW_OVERDUE, "0", "0", ["CLEAN"])


@extra_test("emit_nonzero_logs_failure_not_posted")
def test_emit_nonzero_logs_failure_not_posted():
    wd = load_watchdog_module()
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return SimpleNamespace(returncode=7, stdout="partial out\n", stderr="cron env boom\n")

    orig_run = wd.subprocess.run
    wd.subprocess.run = fake_run
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            wd.emit(emit_decision(), dry_run=False, channel="claw-code")
    finally:
        wd.subprocess.run = orig_run

    out = buf.getvalue()
    assert calls, "subprocess.run was not called"
    assert "POST_FAILED" in out, out
    assert "rc=7" in out, out
    assert "cron env boom" in out, out
    assert "POSTED action=" not in out, out


@extra_test("emit_success_uses_channel_and_logs_msg_id")
def test_emit_success_uses_channel_and_logs_msg_id():
    wd = load_watchdog_module()
    calls = []
    old_channel = os.environ.get("AI_ROOM_CHANNEL")

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return SimpleNamespace(returncode=0, stdout="posted id=1780494000000-deadbeef from=watchdog\n",
                               stderr="")

    orig_run = wd.subprocess.run
    wd.subprocess.run = fake_run
    os.environ["AI_ROOM_CHANNEL"] = "outer-channel"
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            wd.emit(emit_decision(), dry_run=False, channel="claw-code")
        assert os.environ.get("AI_ROOM_CHANNEL") == "outer-channel", "emit mutated global env"
    finally:
        wd.subprocess.run = orig_run
        if old_channel is None:
            os.environ.pop("AI_ROOM_CHANNEL", None)
        else:
            os.environ["AI_ROOM_CHANNEL"] = old_channel

    out = buf.getvalue()
    assert calls, "subprocess.run was not called"
    cmd, kwargs = calls[0]
    assert cmd[1:4] == ["--channel", "claw-code", "post"], cmd
    assert kwargs["env"]["AI_ROOM_CHANNEL"] == "claw-code", kwargs["env"].get("AI_ROOM_CHANNEL")
    assert wd._resolve_emit_channel(wd.DEFAULT_CHANNEL_LOG, None) == "claw-code"
    assert "POSTED action=stall wake=True channel=claw-code msg_id=1780494000000-deadbeef" in out, out


def main():
    failures = 0
    for name, log_lines, now, ff, pc, expect in CASES:
        out = run(log_lines, now, ff, pc)
        ok = all(s in out for s in expect)
        print(f"{'PASS' if ok else 'FAIL'} {name}: expect {expect}")
        if not ok:
            print(f"   --- got ---\n{out}")
            failures += 1
    for name, fn in EXTRA_TESTS:
        try:
            fn()
            ok = True
        except AssertionError as e:
            ok = False
            print(f"   --- assertion ---\n{e}")
        except Exception as e:
            ok = False
            print(f"   --- exception ---\n{type(e).__name__}: {e}")
        print(f"{'PASS' if ok else 'FAIL'} {name}")
        if not ok:
            failures += 1
    total = len(CASES) + len(EXTRA_TESTS)
    print(f"\n{total - failures}/{total} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
