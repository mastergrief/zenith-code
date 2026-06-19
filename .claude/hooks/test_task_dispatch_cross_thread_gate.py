#!/usr/bin/env python3
"""Fixture tests for task_dispatch_cross_thread_gate.py (serialized routing)."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

HOOK = Path(__file__).with_name("task_dispatch_cross_thread_gate.py")


def run(payload: dict) -> int:
    proc = subprocess.run(
        [sys.executable, str(HOOK)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin"},
    )
    return proc.returncode


def dispatch(to: str, body: str) -> dict:
    return {
        "tool_name": "mcp__ai-room__ai_room_post",
        "tool_input": {"kind": "task_dispatch", "to": to, "body": body},
    }


CASES = []


def case(name: str, expected: int, payload: dict):
    CASES.append((name, expected, payload))


BODY_OK = (
    "REPORT_TO: [claude]\n"
    "CROSS_THREAD_REQUIRED: yes\n"
    "+1 implement — proceed."
)
BODY_PARALLEL = (
    "REPORT_TO: [claude, codex_co_lead]\n"
    "CROSS_THREAD_REQUIRED: yes\n"
    "+1 implement — proceed."
)
BODY_MISSING = "+1 implement — proceed."


case("worker_claude_only_allow", 0, dispatch("codex", BODY_OK))
case("worker_parallel_block", 2, dispatch("codex", BODY_PARALLEL))
case("worker_missing_markers_block", 2, dispatch("codex", BODY_MISSING))
case("co_lead_handoff_allow", 0, dispatch("codex_co_lead", BODY_PARALLEL))
case("non_dispatch_kind_allow", 0, {
    "tool_name": "mcp__ai-room__ai_room_post",
    "tool_input": {"kind": "msg", "to": "codex", "body": BODY_MISSING},
})
case("waiver_allow", 0, dispatch(
    "codex",
    BODY_MISSING + "\nCROSS_THREAD_WAIVER: live training abort, co_lead offline this round",
))
case("waiver_trivial_block", 2, dispatch("codex", BODY_MISSING + "\nCROSS_THREAD_WAIVER: ok"))
case("malformed_fail_open", 0, {"tool_name": "mcp__ai-room__ai_room_post", "tool_input": {}})


def main() -> int:
    failed = 0
    for name, expected, payload in CASES:
        rc = run(payload)
        if rc != expected:
            print(f"FAIL {name}: expected exit {expected}, got {rc}")
            failed += 1
        else:
            print(f"PASS {name}")
    if failed:
        print(f"{failed}/{len(CASES)} failed")
        return 1
    print(f"ALL {len(CASES)} PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
