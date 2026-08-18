#!/usr/bin/env python3
"""
PreToolUse hook on `mcp__ai-room__ai_room_post` / `ai_room_reply`
(both hyphenated and underscored name shapes).

Serialized review routing (routing redesign LANE 1): worker material
dispatches/receipts address **claude gate-1 ONLY**. co_lead reviews frozen
handoffs after claude cross-threads — NOT in parallel on the raw worker receipt.

This hook blocks worker dispatches that still instruct parallel routing to
both co-leads via REPORT_TO. THINKING stays parallel; artifact review gates
are sequential.

Trigger conditions (block only when ALL hold):
  1. kind == "task_dispatch"
  2. single-string `to` targeting a named worker (not broadcast / co_lead)
  3. body carries REPORT_TO listing `codex_co_lead` as a worker routing sink
     OR lacks `claude` in REPORT_TO
  4. no valid CROSS_THREAD_WAIVER (reason >= MIN_WAIVER_REASON_CHARS)
  5. missing CROSS_THREAD_REQUIRED: yes (audit marker that cross-thread handoff
     will occur via claude)

co_lead handoff dispatches (to=codex_co_lead) are exempt — claude routes gate-2.

Failure modes (fail-open): empty stdin, JSON parse failures, unexpected schema.
"""

from __future__ import annotations

import json
import os
import re
import sys

CO_LEAD_HANDLES = {"codex_co_lead"}

TARGET_TOOLS = {
    "mcp__ai-room__ai_room_post",
    "mcp__ai-room__ai_room_reply",
    "mcp__ai_room__ai_room_post",
    "mcp__ai_room__ai_room_reply",
}

REPORT_TO_RE = re.compile(r"(?im)^\s*REPORT_TO\s*:\s*\[([^\]]*)\]")

CROSS_THREAD_REQUIRED_RE = re.compile(
    r"(?im)^\s*CROSS_THREAD_REQUIRED\s*:\s*yes\b"
)

CROSS_THREAD_WAIVER_RE = re.compile(
    r"(?im)^\s*CROSS_THREAD_WAIVER\s*:\s*(.+?)\s*$"
)

MIN_WAIVER_REASON_CHARS = 10


def fail_open(reason: str) -> int:
    if os.environ.get("CROSS_THREAD_GATE_DEBUG"):
        print(
            f"[task_dispatch_cross_thread_gate] fail-open: {reason}",
            file=sys.stderr,
        )
    return 0


def report_to_members(body: str) -> set[str] | None:
    m = REPORT_TO_RE.search(body)
    if not m:
        return None
    return {
        tok.strip().lower()
        for tok in m.group(1).split(",")
        if tok.strip()
    }


def main() -> int:
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            return fail_open("empty stdin")
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        return fail_open(f"json decode failed: {exc}")
    except Exception as exc:  # noqa: BLE001
        return fail_open(f"stdin read failed: {exc}")

    if payload.get("tool_name", "") not in TARGET_TOOLS:
        return 0

    tool_input = payload.get("tool_input") or {}
    if not isinstance(tool_input, dict):
        return fail_open("tool_input not a dict")

    if tool_input.get("kind") != "task_dispatch":
        return 0

    to = tool_input.get("to")
    if not isinstance(to, str):
        return 0
    target_handle = to.strip()
    if not target_handle:
        return 0

    if target_handle in CO_LEAD_HANDLES:
        return 0

    body = tool_input.get("body", "")
    if not isinstance(body, str) or not body:
        return 0

    waiver = CROSS_THREAD_WAIVER_RE.search(body)
    if waiver:
        reason = waiver.group(1).strip()
        if len(reason) >= MIN_WAIVER_REASON_CHARS:
            return 0
        msg_lines = [
            "BLOCKED [task_dispatch_cross_thread_gate] CROSS_THREAD_WAIVER reason is too trivial:",
            f"  Found: {reason!r} ({len(reason)} chars)",
            f"  Minimum: {MIN_WAIVER_REASON_CHARS} chars of specific justification",
            "",
            "Replace the waiver with a concrete reason, OR use serialized routing:",
            "  REPORT_TO: [claude]",
            "  CROSS_THREAD_REQUIRED: yes",
        ]
        print("\n".join(msg_lines), file=sys.stderr)
        return 2

    members = report_to_members(body)
    has_required = bool(CROSS_THREAD_REQUIRED_RE.search(body))

    parallel_routing = bool(members and "codex_co_lead" in members)
    has_claude = bool(members and "claude" in members)

    if has_claude and not parallel_routing and has_required:
        return 0

    missing = []
    if members is None:
        missing.append("REPORT_TO: [claude]  (worker material sink is claude gate-1 only)")
    elif not has_claude:
        missing.append("REPORT_TO must include claude (gate-1 sink)")
    if parallel_routing:
        missing.append(
            "REPORT_TO must NOT list codex_co_lead — co_lead gate-2 follows "
            "claude's frozen handoff, not parallel worker routing"
        )
    if not has_required:
        missing.append("CROSS_THREAD_REQUIRED: yes")

    msg_lines = [
        f"BLOCKED [task_dispatch_cross_thread_gate] worker dispatch to '{target_handle}' "
        "violates serialized review routing:",
        "",
    ]
    for item in missing:
        msg_lines.append(f"  - {item}")
    msg_lines += [
        "",
        "Serialized protocol: worker material receipts → claude gate-1 ONLY.",
        "Claude verifies/freezes, then cross-threads co_lead gate-2 on the frozen artifact.",
        "THINKING stays parallel; artifact review gates are sequential.",
        "",
        "Add BOTH lines to the dispatch body:",
        "",
        "  REPORT_TO: [claude]",
        "  CROSS_THREAD_REQUIRED: yes",
        "",
        "Do NOT route worker receipts to codex_co_lead in parallel.",
        "co_lead handoff dispatches (to=codex_co_lead) remain allowed separately.",
        "",
        "Emergency bypass (auditable; reason >= "
        f"{MIN_WAIVER_REASON_CHARS} chars):",
        "  CROSS_THREAD_WAIVER: <specific reason>",
    ]
    print("\n".join(msg_lines), file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
