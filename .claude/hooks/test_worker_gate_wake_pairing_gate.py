#!/usr/bin/env python3
"""Fixture tests for worker_gate_wake_pairing_gate.py.

CPU-static / no-loop: constructs PreToolUse payloads + a fixture channel
log, invokes the hook via subprocess, asserts exit codes (0 allow / 2
block). Run: python3 .claude/hooks/test_worker_gate_wake_pairing_gate.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

HOOK = Path(__file__).with_name("worker_gate_wake_pairing_gate.py")

# Reference timestamps (the hook measures recency vs the newest record).
T_OLD = "2026-06-02T19:00:00Z"
T_NEW = "2026-06-02T20:00:00Z"
T_NEWER = "2026-06-02T20:05:00Z"

TASK = "1780347615017-1538f834"
OTHER_TASK = "1780000000000-deadbeef"


def tu(ts, frm="claude", to="codex", owner="codex", status="in_progress",
       notify=True, reply_to=TASK, mid=None):
    """Build a task_update log record."""
    return json.dumps({
        "ts": ts, "id": mid or f"{ts}-x", "from": frm, "to": to,
        "owner": owner, "kind": "task_update", "status": status,
        "notify": notify, "reply_to": reply_to, "body": "n", "note": "n",
    })


# A task_update MSG id (shares the <unix_ms>-<hex> shape with task ids).
WAKE_MSGID = "1780433063191-dcaa337c"


def other(ts, kind="msg"):
    """A non-task_update record to advance 'newest' ts."""
    return json.dumps({"ts": ts, "id": f"{ts}-o", "from": "codex",
                       "to": "claude", "kind": kind, "body": "x"})


def run(payload: dict, log_lines: list[str]) -> int:
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as fh:
        fh.write("\n".join(log_lines) + ("\n" if log_lines else ""))
        log_path = fh.name
    proc = subprocess.run(
        [sys.executable, str(HOOK)],
        input=json.dumps(payload),
        capture_output=True, text=True,
        env={"AI_ROOM_CHANNEL_LOG": log_path, "PATH": "/usr/bin:/bin"},
    )
    Path(log_path).unlink(missing_ok=True)
    return proc.returncode


def post(kind="msg", to="codex", body="+1 IMPLEMENT — proceed."):
    return {"tool_name": "mcp__ai-room__ai_room_post",
            "tool_input": {"kind": kind, "to": to, "body": body}}


CASES = []


def case(name, expected, payload, log_lines):
    CASES.append((name, expected, payload, log_lines))


# 1. gate to parked worker, NO wake-pairing in log -> BLOCK
case("gate_no_pairing", 2, post(), [other(T_NEW)])
# 2. gate, recent claude->codex notify in_progress same-task wake -> ALLOW
case("gate_valid_pairing", 0, post(body=f"+1 IMPLEMENT task {TASK}"),
     [tu(T_NEW)])
# 3. gate with valid WAKE_VERIFIED bypass -> ALLOW
case("gate_wake_verified", 0,
     post(body="+1 LAUNCH now.\nWAKE_VERIFIED: codex mid-turn, posted 20s ago this round"),
     [other(T_NEW)])
# 4. gate citing TASK, but wake is for a DIFFERENT task -> BLOCK (same-task bind)
case("gate_wrong_task_pairing", 2, post(body=f"+1 LAUNCH task {TASK}"),
     [tu(T_NEW, reply_to=OTHER_TASK)])
# 5. gate to co_lead -> ALLOW (exempt)
case("gate_to_co_lead", 0, post(to="codex_co_lead"), [other(T_NEW)])
# 6. broadcast (to not a single string) -> ALLOW
case("broadcast", 0,
     {"tool_name": "mcp__ai-room__ai_room_post",
      "tool_input": {"kind": "msg", "to": ["codex", "codex_co_lead"],
                     "body": "+1 IMPLEMENT"}},
     [other(T_NEW)])
# 7. non-gate msg (no directive) -> ALLOW
case("non_gate_msg", 0, post(body="status: looks good, concur."),
     [other(T_NEW)])
# 8. ack kind carrying gate text -> ALLOW (kind filter)
case("ack_kind", 0, post(kind="ack", body="+1 IMPLEMENT ack"), [other(T_NEW)])
# 9. malformed payload -> ALLOW (fail-open)
case("malformed", 0, {"not": "a real payload"}, [other(T_NEW)])
# 10. wake-pairing exists but STALE (older than window) -> BLOCK
case("gate_stale_pairing", 2, post(body=f"+1 IMPLEMENT task {TASK}"),
     [tu(T_OLD), other(T_NEWER)])
# 11. WAKE_VERIFIED inside a blockquote (quoted) -> BLOCK (quoted-strip)
case("wake_verified_quoted", 2,
     post(body="+1 IMPLEMENT now.\n> WAKE_VERIFIED: codex was active earlier round"),
     [other(T_NEW)])
# 12. WAKE_VERIFIED trivial reason -> BLOCK
case("wake_verified_trivial", 2,
     post(body="+1 IMPLEMENT now.\nWAKE_VERIFIED: ok"), [other(T_NEW)])
# 13. wake pairing present but from codex (not claude) -> BLOCK (must be claude-issued)
case("gate_pairing_from_worker", 2, post(body=f"+1 IMPLEMENT task {TASK}"),
     [tu(T_NEW, frm="codex")])
# 14. wake pairing but notify=false -> BLOCK
case("gate_pairing_no_notify", 2, post(body=f"+1 IMPLEMENT task {TASK}"),
     [tu(T_NEW, notify=False)])
# 15. wake pairing target via owner only (to=list incl claude) same task -> ALLOW
case("gate_pairing_owner_match", 0, post(body=f"+1 IMPLEMENT task {TASK}"),
     [tu(T_NEW, to=["claude", "codex_co_lead"], owner="codex")])
# 16. WAKE_PAIRED cites a task_update MSG id (not a task id) resolving to TASK -> ALLOW
#     (the canonical paired pattern; a msg id must NOT be misread as a task id)
case("wake_paired_cites_msgid_same_task", 0,
     post(body=f"+1 IMPLEMENT now.\nWAKE_PAIRED: task_update {WAKE_MSGID}"),
     [tu(T_NEW, mid=WAKE_MSGID, reply_to=TASK)])
# 17. labelled task TASK but the only wake (cited msg id) is for a DIFFERENT task -> BLOCK
case("labelled_task_wake_other_task", 2,
     post(body=f"+1 IMPLEMENT task {TASK}\nWAKE_PAIRED: task_update {WAKE_MSGID}"),
     [tu(T_NEW, mid=WAKE_MSGID, reply_to=OTHER_TASK)])


def main() -> int:
    failures = 0
    for name, expected, payload, log_lines in CASES:
        got = run(payload, log_lines)
        ok = got == expected
        print(f"{'PASS' if ok else 'FAIL'} {name}: expected exit {expected}, got {got}")
        if not ok:
            failures += 1
    print(f"\n{len(CASES) - failures}/{len(CASES)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
