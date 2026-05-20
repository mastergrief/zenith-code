#!/usr/bin/env python3
"""
PreToolUse hook on `mcp__ai-room__ai_room_post` and `mcp__ai-room__ai_room_reply`
(both hyphenated and underscored name shapes).

Enforces child-task boundary discipline on ai-room task dispatches:
dispatching a child task to a worker handle that is currently bound to a
*different* active task is a lane-discipline violation. Stale worker
context bleeds into the new child task, and per-worker frame-too-big
risk grows non-linearly with retained context.

This is a BLOCK-AND-EXPLAIN guardrail (NOT auto-recycle). Auto-recycle
from a PreToolUse hook can destroy useful unsent state or collide with
a worker that is about to post a receipt; the block-and-explain version
gives most of the safety with less weirdness.

Ported from zenith-fitness with claw-code channel + charter adaptations.

Scope (do not overclaim):
  - This hook only fires on Claude Code outbound calls to the locally
    registered `mcp__ai-room__ai_room_post` / `ai_room_reply` tools, in
    both hyphenated (`ai-room`) and underscored (`ai_room`) shapes.
  - It does NOT enforce Codex-originated ai-room posts or other clients.
    Universal enforcement requires an ai-room MCP/server-side gate or a
    Codex-side rule/hook follow-up.

Trigger conditions (block only when ALL hold):
  1. The tool call is `kind == "task_dispatch"` (other kinds skip).
  2. The dispatch targets a single named handle (broadcast / multi-target /
     null `to` skip).
  3. Target handle is NOT in CO_LEAD_HANDLES (exempt by design — co_lead
     audits across child tasks within a cycle).
  4. The dispatch body cites a task_id (regex match for
     `(?:[Yy]our task|[Tt]ask)[:\\s]*[`"']?(\\d{13,}-[a-f0-9]+)`).
  5. The most recent `task_update` from the target handle in the active
     channel log has `status=in_progress` AND its `reply_to` (task_id)
     differs from the new dispatch's task_id.
  6. The dispatch body does NOT contain a `RETAIN OVERRIDE: <reason>`
     line (case-insensitive, anywhere in body, with non-empty reason
     >= MIN_OVERRIDE_REASON_CHARS).

Block message:
  Names predicate + two resolution paths:
    (a) Recycle: kill_claudex(handle) + spawn_claudex + redispatch.
    (b) Retain override: add `RETAIN OVERRIDE: <reason>` line with
        specific justification (>= 10 chars).

Failure modes (fail-open):
  - Empty stdin, JSON parse failures, missing channel log, log read errors,
    unexpected schema → exit 0 (allow). Goal: never wedge a turn on a hook
    bug. The rule still applies operationally; this hook is a guardrail,
    not a substitute for claude-side discipline.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

# Default channel log location (claw-code).
# Override via AI_ROOM_CHANNEL_LOG env var when needed.
DEFAULT_CHANNEL_LOG = Path("/home/gabe/.ai-room/channels/claw-code/messages.jsonl")

# Co_lead exempt by design (cross-cycle audit lane).
CO_LEAD_HANDLES = {"codex_co_lead"}

# Task ID format: <unix_ms>-<8 hex chars>
TASK_ID_RE = re.compile(r"(\d{13,}-[0-9a-f]{6,12})")
# Detect "Your task: `<id>`" / "Task: `<id>`" / "task_id `<id>`" patterns.
NEW_TASK_RE = re.compile(
    r"(?:[Yy]our\s+task|[Tt]ask\s*(?:id)?)\s*[:\s]+[`\"']?(\d{13,}-[0-9a-f]{6,12})",
    re.MULTILINE,
)

# Detect retain override line (case-insensitive). Captures the reason text
# after the colon; non-trivial-reason length check happens in main().
RETAIN_OVERRIDE_RE = re.compile(
    r"(?im)^\s*RETAIN\s+OVERRIDE\s*:\s*(.+?)\s*$"
)

# Minimum reason length (after stripping whitespace) for a retain override
# to be considered specific enough for audit. Short reasons like "ok" or
# "continue" or "." are rejected — co_lead audit needs concrete justification
# to flag drift.
MIN_OVERRIDE_REASON_CHARS = 10

# Tool name matchers (hyphenated + underscored shapes).
TARGET_TOOLS = {
    "mcp__ai-room__ai_room_post",
    "mcp__ai-room__ai_room_reply",
    "mcp__ai_room__ai_room_post",
    "mcp__ai_room__ai_room_reply",
}


def fail_open(reason: str) -> int:
    """Exit 0 (allow) on any unexpected condition. Optionally log to stderr
    only at debug level (kept silent to avoid noisy hook output)."""
    if os.environ.get("TASK_DISPATCH_GATE_DEBUG"):
        print(f"[task_dispatch_child_boundary_gate] fail-open: {reason}", file=sys.stderr)
    return 0


def find_active_task_for_handle(channel_log: Path, handle: str) -> str | None:
    """Find the target handle's currently-bound task_id, if any.

    Algorithm:
      1. Scan the channel log (bounded tail).
      2. Track the LATEST `task_update` per task_id (by ts) regardless of
         who posted it — task closures posted by claude must clear the
         binding even though `from != handle`.
      3. Track which task_ids the target handle has ever posted on.
      4. Return the most recent task_id where (handle has touched it) AND
         (latest status across all parties is `in_progress`).

    Returns None if no active binding (handle never claimed, or all
    handle-touched tasks are closed/transitional).
    """
    if not channel_log.exists():
        return None

    try:
        with channel_log.open("r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
    except OSError:
        return None

    tail = lines[-5000:] if len(lines) > 5000 else lines

    # task_id -> {"status": str, "ts": str}
    task_latest: dict[str, dict[str, str]] = {}
    handle_touched: set[str] = set()

    for raw in tail:
        raw = raw.strip()
        if not raw:
            continue
        try:
            rec = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if rec.get("kind") != "task_update":
            continue
        task_id = rec.get("reply_to")
        status = rec.get("status")
        sender = rec.get("from")
        ts = rec.get("ts", "")
        if not task_id or not status:
            continue
        if sender == handle:
            handle_touched.add(task_id)
        existing = task_latest.get(task_id)
        if existing is None or ts > existing["ts"]:
            task_latest[task_id] = {"status": status, "ts": ts}

    candidates: list[tuple[str, str]] = []
    for task_id in handle_touched:
        latest = task_latest.get(task_id)
        if latest is None:
            continue
        if latest["status"] == "in_progress":
            candidates.append((latest["ts"], task_id))

    if not candidates:
        return None

    candidates.sort(reverse=True)
    return candidates[0][1]


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
        return 0  # not our matcher (defensive; matcher should already filter)

    tool_input = payload.get("tool_input") or {}
    if not isinstance(tool_input, dict):
        return fail_open("tool_input not a dict")

    # Filter: only task_dispatch kind.
    if tool_input.get("kind") != "task_dispatch":
        return 0

    # Target handle: only single-string `to` is enforced. Broadcast (None),
    # multi-target (list), or missing → allow.
    to = tool_input.get("to")
    if not isinstance(to, str):
        return 0
    target_handle = to.strip()
    if not target_handle:
        return 0

    # Co_lead exempt by design.
    if target_handle in CO_LEAD_HANDLES:
        return 0

    body = tool_input.get("body", "")
    if not isinstance(body, str) or not body:
        return 0  # empty dispatch — let other validators handle

    # Retain override: short-circuits all other checks IF the reason is
    # non-trivial (>= MIN_OVERRIDE_REASON_CHARS after stripping). Trivial
    # reasons like "ok" / "continue" / "." defeat the audit point.
    override = RETAIN_OVERRIDE_RE.search(body)
    if override:
        reason = override.group(1).strip()
        if len(reason) >= MIN_OVERRIDE_REASON_CHARS:
            return 0
        # Reason too trivial — block with a distinct message that names the
        # found reason and the minimum length.
        msg_lines = [
            f"BLOCKED [task_dispatch_child_boundary_gate] RETAIN OVERRIDE reason is too trivial for audit:",
            f"  Found: {reason!r} ({len(reason)} chars)",
            f"  Minimum: {MIN_OVERRIDE_REASON_CHARS} chars of specific justification",
            "",
            "Worker lifecycle / Retain override syntax: reason must be specific enough for",
            'co_lead audit (e.g. "defect-cycle scope subset of files just edited",',
            '"tiny-adjacent slice in same module, no boundary crossed"). Vague overrides',
            'like "ok" or "continue" defeat the audit point and are flagable drift patterns.',
            "",
            "Resolve by replacing the override line with a concrete justification, OR removing the",
            "override and recycling the worker per the standard child-task boundary rule.",
        ]
        print("\n".join(msg_lines), file=sys.stderr)
        return 2

    # Extract intended new task_id from dispatch body.
    new_match = NEW_TASK_RE.search(body)
    if not new_match:
        # No task_id detected in body. Fail-open: if the dispatch doesn't cite
        # a task_id, this hook can't enforce. Other validators (e.g.,
        # task contract lint) handle that case.
        return fail_open("no task_id detected in dispatch body")
    new_task_id = new_match.group(1)

    # Look up channel log path.
    log_env = os.environ.get("AI_ROOM_CHANNEL_LOG")
    channel_log = Path(log_env) if log_env else DEFAULT_CHANNEL_LOG

    # Find target handle's currently-bound task.
    active_task = find_active_task_for_handle(channel_log, target_handle)

    if active_task is None:
        return 0  # no active binding; fresh dispatch is fine

    if active_task == new_task_id:
        return 0  # same-task continuation; allowed

    # MISMATCH + no override: BLOCK.
    msg_lines = [
        f"BLOCKED [task_dispatch_child_boundary_gate] child-task boundary violation for handle '{target_handle}'.",
        "",
        f"  Target handle: {target_handle}",
        f"  Active binding (task_id): {active_task}",
        f"  Blocked dispatch (task_id): {new_task_id}",
        "",
        "Worker lifecycle / Child-task boundary: spawn fresh handle per child task.",
        "Stale worker context bleeds into the new child task and accumulates",
        "frame-too-big risk.",
        "",
        "Resolve via ONE of:",
        f"  (a) Close the active binding if work is done:",
        f"      ai_room_task_complete(task_id='{active_task}')  # or task_update with status=completed",
        f"  (b) Recycle the worker:",
        f"      ai_room_kill_claudex(handle='{target_handle}')",
        f"      ai_room_spawn_claudex(handle='<new>', role=<role>)",
        f"      redispatch task `{new_task_id}` to the fresh handle",
        f"  (c) Retain override (auditable; non-trivial reason >= {MIN_OVERRIDE_REASON_CHARS} chars):",
        f"      add 'RETAIN OVERRIDE: <specific justification>' line in the dispatch body",
        f"      e.g. 'RETAIN OVERRIDE: defect-cycle scope subset of files just edited'",
        "",
        "Co_lead is exempt from this gate (cross-cycle audit lane by design); verify target handle is correct.",
    ]
    print("\n".join(msg_lines), file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
