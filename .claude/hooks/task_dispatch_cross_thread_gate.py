#!/usr/bin/env python3
"""
PreToolUse hook on `mcp__ai-room__ai_room_post` / `ai_room_reply`
(both hyphenated and underscored name shapes).

Deterministic enforcement of the R&D-team cross-thread protocol
(`.claude/rules/AI_ROOM_COLLAB.md` §"Cross-thread is mandatory at
thinking boundaries" + `.claude/rules/CLAUDEX_ORCHESTRATION.md`
§"Worker task shape"):

> Non-trivial worker dispatches carry `REPORT_TO: [claude,
> codex_co_lead]` + `CROSS_THREAD_REQUIRED: yes` so the
> research/strategy co-lead stays in the loop on every material
> worker output. Waiver only via explicit `CROSS_THREAD_WAIVER:
> <reason>`.

This is the v1 fail-CLOSED version of what `cross_thread_audit.py`
(Stop hook) only LOGS. The Stop audit stays as the post-hoc record;
this PreToolUse gate makes worker dispatches deterministic at the
point of the tool call.

This is a BLOCK-AND-EXPLAIN guardrail (NOT auto-anything). On a
malformed worker dispatch it blocks with a message naming the missing
marker(s) and the two resolution paths (add the markers, or add a
waiver). The fix is one line in the dispatch body, so the block is
cheap and self-correcting.

Scope (do not overclaim):
  - Fires only on Claude Code outbound calls to the locally registered
    `mcp__ai-room__ai_room_post` / `ai_room_reply` tools, both
    hyphenated (`ai-room`) and underscored (`ai_room`) shapes.
  - Does NOT enforce Codex-originated posts or other clients.

Trigger conditions (block only when ALL hold):
  1. The tool call is `kind == "task_dispatch"` (other kinds skip —
     acks / status / design / msg traffic is untouched).
  2. The dispatch targets a single named handle (broadcast / multi-
     target / null `to` skip).
  3. Target handle is NOT in CO_LEAD_HANDLES (co_lead is the audit
     recipient, not a dispatched worker — exempt by design).
  4. The body does NOT carry BOTH:
       - a `REPORT_TO: [...]` list containing `claude` AND
         `codex_co_lead`, AND
       - a `CROSS_THREAD_REQUIRED: yes` line.
  5. AND the body does NOT carry a valid `CROSS_THREAD_WAIVER:
     <reason>` line (reason >= MIN_WAIVER_REASON_CHARS after strip).

Failure modes (fail-open):
  - Empty stdin, JSON parse failures, unexpected schema → exit 0
    (allow). Never wedge a turn on a hook bug; the protocol still
    applies operationally, this gate is a guardrail not a substitute
    for claude-side discipline.
"""

from __future__ import annotations

import json
import os
import re
import sys

# Co_lead is the cross-thread RECIPIENT, not a dispatched worker.
# A task_dispatch to co_lead is not the pattern this gate governs.
CO_LEAD_HANDLES = {"codex_co_lead"}

# Tool name matchers (hyphenated + underscored shapes).
TARGET_TOOLS = {
    "mcp__ai-room__ai_room_post",
    "mcp__ai-room__ai_room_reply",
    "mcp__ai_room__ai_room_post",
    "mcp__ai_room__ai_room_reply",
}

# REPORT_TO line: capture the bracketed list; both required tokens are
# checked for membership in main() (order-independent, extras allowed).
REPORT_TO_RE = re.compile(r"(?im)^\s*REPORT_TO\s*:\s*\[([^\]]*)\]")

# CROSS_THREAD_REQUIRED: yes (case-insensitive, line-anchored).
CROSS_THREAD_REQUIRED_RE = re.compile(
    r"(?im)^\s*CROSS_THREAD_REQUIRED\s*:\s*yes\b"
)

# Waiver line; reason captured for length check.
CROSS_THREAD_WAIVER_RE = re.compile(
    r"(?im)^\s*CROSS_THREAD_WAIVER\s*:\s*(.+?)\s*$"
)

# Required REPORT_TO members.
REQUIRED_REPORT_TO = ("claude", "codex_co_lead")

# Minimum waiver reason length (after strip). Trivial reasons like
# "ok" / "busy" / "." defeat the audit point and are rejected.
MIN_WAIVER_REASON_CHARS = 10


def fail_open(reason: str) -> int:
    """Exit 0 (allow) on any unexpected condition."""
    if os.environ.get("CROSS_THREAD_GATE_DEBUG"):
        print(
            f"[task_dispatch_cross_thread_gate] fail-open: {reason}",
            file=sys.stderr,
        )
    return 0


def report_to_has_required(body: str) -> bool:
    """True if a REPORT_TO list is present AND contains both required
    members (case-insensitive, order-independent, extras allowed)."""
    m = REPORT_TO_RE.search(body)
    if not m:
        return False
    members = {
        tok.strip().lower()
        for tok in m.group(1).split(",")
        if tok.strip()
    }
    return all(req in members for req in REQUIRED_REPORT_TO)


def main() -> int:
    # Read hook event JSON from stdin.
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            return fail_open("empty stdin")
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        return fail_open(f"json decode failed: {exc}")
    except Exception as exc:  # noqa: BLE001 — fail-open
        return fail_open(f"stdin read failed: {exc}")

    tool_name = payload.get("tool_name", "")
    if tool_name not in TARGET_TOOLS:
        return 0  # not our matcher (defensive)

    tool_input = payload.get("tool_input") or {}
    if not isinstance(tool_input, dict):
        return fail_open("tool_input not a dict")

    # Filter: only task_dispatch kind.
    if tool_input.get("kind") != "task_dispatch":
        return 0

    # Target handle: only single-string `to` is enforced. Broadcast (None),
    # multi-target (list), or missing → allow (cross-thread shape is for
    # single-worker dispatches; a multi-target post is not a worker dispatch).
    to = tool_input.get("to")
    if not isinstance(to, str):
        return 0
    target_handle = to.strip()
    if not target_handle:
        return 0

    # Co_lead exempt by design (cross-thread recipient, not a worker).
    if target_handle in CO_LEAD_HANDLES:
        return 0

    body = tool_input.get("body", "")
    if not isinstance(body, str) or not body:
        return 0  # empty dispatch — let other validators handle

    # Waiver short-circuit: explicit, non-trivial reason → allow.
    waiver = CROSS_THREAD_WAIVER_RE.search(body)
    if waiver:
        reason = waiver.group(1).strip()
        if len(reason) >= MIN_WAIVER_REASON_CHARS:
            return 0
        # Waiver present but reason too trivial — block with a distinct message.
        msg_lines = [
            "BLOCKED [task_dispatch_cross_thread_gate] CROSS_THREAD_WAIVER reason is too trivial:",
            f"  Found: {reason!r} ({len(reason)} chars)",
            f"  Minimum: {MIN_WAIVER_REASON_CHARS} chars of specific justification",
            "",
            "A waiver bypasses the deterministic cross-thread gate, so co_lead must be able to",
            'audit WHY (e.g. "live training abort, codex_co_lead context-pressured this round").',
            "Replace the waiver line with a concrete reason, OR add the standard markers:",
            "  REPORT_TO: [claude, codex_co_lead]",
            "  CROSS_THREAD_REQUIRED: yes",
        ]
        print("\n".join(msg_lines), file=sys.stderr)
        return 2

    # Main gate: require BOTH markers.
    has_report_to = report_to_has_required(body)
    has_required = bool(CROSS_THREAD_REQUIRED_RE.search(body))
    if has_report_to and has_required:
        return 0  # well-formed worker dispatch

    missing = []
    if not has_report_to:
        missing.append("REPORT_TO: [claude, codex_co_lead]  (list must contain BOTH claude and codex_co_lead)")
    if not has_required:
        missing.append("CROSS_THREAD_REQUIRED: yes")

    msg_lines = [
        f"BLOCKED [task_dispatch_cross_thread_gate] worker dispatch to '{target_handle}' is missing cross-thread marker(s):",
        "",
    ]
    for item in missing:
        msg_lines.append(f"  - {item}")
    msg_lines += [
        "",
        "Cross-thread protocol (deterministic v1): non-trivial worker dispatches must keep",
        "codex_co_lead in the loop. Add BOTH lines to the dispatch body:",
        "",
        "  REPORT_TO: [claude, codex_co_lead]",
        "  CROSS_THREAD_REQUIRED: yes",
        "",
        "This routes worker design/audit/run receipts to codex_co_lead natively (reply",
        "to=[claude, codex_co_lead]) so the co-lead can concur or flag one load-bearing",
        "hole before claude synthesizes / banks / commits / dispatches the next slice.",
        "",
        "Emergency / live-ops bypass (auditable; reason >= "
        f"{MIN_WAIVER_REASON_CHARS} chars):",
        "  CROSS_THREAD_WAIVER: <specific reason, e.g. 'live training abort, co_lead offline'>",
        "",
        "co_lead is exempt (cross-thread recipient, not a dispatched worker); acks / status /",
        "design / msg kinds are untouched — this gate fires only on kind=task_dispatch.",
    ]
    print("\n".join(msg_lines), file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
