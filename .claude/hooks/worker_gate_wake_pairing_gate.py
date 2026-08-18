#!/usr/bin/env python3
"""
PreToolUse hook on `mcp__ai-room__ai_room_post` / `ai_room_reply`
(both hyphenated and underscored name shapes).

Deterministic prevention of the ACK-IDLE worker hang
(`.claude/rules/CLAUDEX_ORCHESTRATION.md` §"Completed-task ack-idle" +
§"Wake semantics"; `.claude/rules/AI_ROOM_COLLAB.md` §"Task sharing"):

> A worker that finished its turn "holding for Claude's gate" has no
> self-driving work. A `+1` / drive directive that arrives as a plain
> `kind=msg` does NOT reliably re-wake a worker whose turn already
> ended (authority != wake: a +1 msg is authority, not a wake event —
> same as `task_update` being durable state but not a wake). Its own
> `resume_check` returns "idle ok", so it sits idle for minutes/hours.
> The re-drive fix is to PAIR the gate with a wake:
> `task_update(notify=true, to=<worker>, status=in_progress)` THIS turn,
> then the direct gate post.

This gate makes that pairing deterministic at the point of the gate
post, instead of relying on orchestrator discipline (which failed
twice in one session — once with a 12-second near-miss where the +1
landed just after the worker's turn ended).

Design (STATEFUL, per co_lead review): rather than trust an honor-system
marker, the gate VERIFIES that a real, recent, target-bound wake-pairing
`task_update` exists in the channel log:
  from == "claude", notify == true, status == "in_progress",
  target-bound (worker in `to` OR `owner` == worker),
  same-task (reply_to == task_context) when a task context is resolvable,
  within a recency window relative to the newest log record.
`WAKE_VERIFIED: <reason>` remains the bypass for a worker confirmed
mid-turn / active (no re-wake needed).

Task-context resolution (co_lead fold — task ids and MSG ids share the
`<unix_ms>-<hex>` shape, so a body that cites the paired `task_update`
MSG id must NOT be misread as a task id):
  1. an explicit labelled task id in the body (`task <id>` /
     `task_id=<id>` / `Task: <id>`, but never `task_update <id>`),
  2. else the post's own `reply_to`, resolved as a task id directly or
     via the cited record's `reply_to`,
  3. else a cited `task_update <id>` / `WAKE_PAIRED: ... <id>` MSG id,
     resolved against the log to that record's `reply_to` (its task),
  4. else no task binding — verify by target + recency only (safe: a
     real target-bound wake still clears; we just don't over-constrain).

This is a BLOCK-AND-EXPLAIN guardrail (NOT auto-anything). The fix is
one paired `task_update(notify=true)` call (or one bypass line).

Scope (do not overclaim):
  - Fires only on Claude Code outbound calls to the locally registered
    `mcp__ai-room__ai_room_post` / `ai_room_reply` tools (both `ai-room`
    and `ai_room` shapes). Does NOT enforce Codex-originated posts.
  - Governs FOLLOW-UP gate/drive posts (kind=msg / question_answered).
    Initial `kind=task_dispatch` has its own wake lifecycle + is covered
    by the child-boundary and cross-thread gates — skipped here.

Failure modes (fail-open):
  - Empty stdin, JSON parse failures, missing/unreadable channel log,
    unexpected schema → exit 0 (allow). Never wedge a turn on a hook
    bug; the ack-idle invariant still applies operationally.
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

# Default channel log location (claw-code). Override via env.
DEFAULT_CHANNEL_LOG = Path(
    "/home/gabe/.ai-room/channels/claw-code/messages.jsonl"
)

# Co_lead is multi-task self-driving (co-lead/audit lane), not a parked
# single-task worker. Exempt.
CO_LEAD_HANDLES = {"codex_co_lead"}

# Tool name matchers (hyphenated + underscored shapes).
TARGET_TOOLS = {
    "mcp__ai-room__ai_room_post",
    "mcp__ai-room__ai_room_reply",
    "mcp__ai_room__ai_room_post",
    "mcp__ai_room__ai_room_reply",
}

# Kinds that carry follow-up gates/drives.
GATE_KINDS = {"msg", "question_answered"}

# Gate / drive directive signals.
GATE_DIRECTIVE_RE = re.compile(
    r"(?im)"
    r"(\+\s*1\s+(implement|commit|launch|push)\b"
    r"|\bEXECUTION\s+WAKE\b"
    r"|\b(implement|proceed|launch|run|execute)\s+now\b)"
)

# id shape shared by task ids and msg ids: <unix_ms>-<hex>.
ID_RE = r"\d{13,}-[0-9a-f]{6,12}"

# Explicit LABELLED task id in the body. `task`/`task id`/`task_id`
# followed by `:`/`=`/space then the id. `task_update <id>` does NOT
# match (the "_update " between "task" and the id breaks the pattern),
# so a cited task_update msg id is never mistaken for a task id here.
LABELLED_TASK_ID_RE = re.compile(
    r"(?i)\btask(?:[ _-]?id)?\s*[:=]?\s*[`\"']?(" + ID_RE + r")"
)

# A cited task_update MSG id (the wake-pairing receipt pattern). Captures
# the id following a `task_update` / `WAKE_PAIRED:` mention; resolved
# against the log to its task (`reply_to`).
CITED_TASK_UPDATE_MSG_RE = re.compile(
    r"(?im)(?:task[_\s]?update|WAKE_PAIRED\s*:)[^\n]*?(" + ID_RE + r")"
)

# WAKE_VERIFIED bypass (line-anchored; reason captured for length check).
WAKE_VERIFIED_RE = re.compile(r"(?im)^\s*WAKE_VERIFIED\s*:\s*(.+?)\s*$")
MIN_VERIFIED_REASON_CHARS = 10

# Recency window: a wake-pairing older than this (relative to the newest
# log record) is stale and does not count as paired-with-this-gate.
WAKE_WINDOW_SECONDS = 1800

# Bounded tail scan of the channel log.
TAIL_LINES = 4000


def fail_open(reason: str) -> int:
    if os.environ.get("WORKER_GATE_WAKE_GATE_DEBUG"):
        print(
            f"[worker_gate_wake_pairing_gate] fail-open: {reason}",
            file=sys.stderr,
        )
    return 0


def strip_quoted(body: str) -> str:
    """Drop blockquote lines so a quoted bypass marker inside cited text
    cannot accidentally exempt the post."""
    return "\n".join(
        ln for ln in body.splitlines() if not ln.lstrip().startswith(">")
    )


def _parse_ts(ts: str) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def _target_bound(rec: dict, handle: str) -> bool:
    if rec.get("owner") == handle:
        return True
    to = rec.get("to")
    if isinstance(to, str):
        return to.strip() == handle
    if isinstance(to, list):
        return handle in {str(t).strip() for t in to}
    return False


def scan_log(channel_log: Path, handle: str):
    """Single bounded scan of the channel log. Returns:
      tu_replyto:  {task_update msg id -> reply_to}  (for resolving cited ids)
      task_ids:    set of reply_to values on task_update records (task ids)
      wakes:       [(ts_dt, reply_to)] for qualifying claude->handle wakes
      newest_ts:   newest record ts (recency reference)
    Returns (None, None, None, None) if the log is unreadable.
    """
    if not channel_log.exists():
        return None, None, None, None
    try:
        with channel_log.open("r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
    except OSError:
        return None, None, None, None

    tail = lines[-TAIL_LINES:] if len(lines) > TAIL_LINES else lines
    tu_replyto: dict[str, str] = {}
    task_ids: set[str] = set()
    wakes: list[tuple[datetime, str | None]] = []
    newest_ts: datetime | None = None

    for raw in tail:
        raw = raw.strip()
        if not raw:
            continue
        try:
            rec = json.loads(raw)
        except json.JSONDecodeError:
            continue
        ts = _parse_ts(rec.get("ts", ""))
        if ts and (newest_ts is None or ts > newest_ts):
            newest_ts = ts
        if rec.get("kind") != "task_update":
            continue
        rid = rec.get("id")
        reply_to = rec.get("reply_to")
        if rid and reply_to:
            tu_replyto[rid] = reply_to
            task_ids.add(reply_to)
        if (
            rec.get("from") == "claude"
            and rec.get("notify")
            and rec.get("status") == "in_progress"
            and _target_bound(rec, handle)
        ):
            if ts:
                wakes.append((ts, reply_to))

    return tu_replyto, task_ids, wakes, newest_ts


def resolve_task_context(
    tool_input: dict, body: str, tu_replyto: dict, task_ids: set
) -> str | None:
    """Resolve the gate's task context (see module docstring). Returns a
    task id or None (None => verify by target+recency only)."""
    # 1. explicit labelled task id in the body (authoritative).
    m = LABELLED_TASK_ID_RE.search(body)
    if m:
        return m.group(1)
    # 2. the post's own reply_to: a task id directly, or a cited record's task.
    rt = tool_input.get("reply_to")
    if isinstance(rt, str) and rt:
        if rt in task_ids:
            return rt
        if rt in tu_replyto:
            return tu_replyto[rt]
    # 3. fallback: a cited task_update / WAKE_PAIRED MSG id -> resolve to its task.
    m = CITED_TASK_UPDATE_MSG_RE.search(body)
    if m and m.group(1) in tu_replyto:
        return tu_replyto[m.group(1)]
    # 4. no resolvable task context.
    return None


def main() -> int:
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
        return 0

    tool_input = payload.get("tool_input") or {}
    if not isinstance(tool_input, dict):
        return fail_open("tool_input not a dict")

    # Filter 1: gate-bearing kinds only (default kind is "msg").
    if tool_input.get("kind", "msg") not in GATE_KINDS:
        return 0

    # Filter 2: single-string `to` only.
    to = tool_input.get("to")
    if not isinstance(to, str):
        return 0
    target_handle = to.strip()
    if not target_handle:
        return 0

    # Filter 3: co_lead exempt.
    if target_handle in CO_LEAD_HANDLES:
        return 0

    body = tool_input.get("body", "")
    if not isinstance(body, str) or not body:
        return 0

    # Filter 4: only fire on an actual gate/drive directive.
    unquoted = strip_quoted(body)
    if not GATE_DIRECTIVE_RE.search(unquoted):
        return 0

    # Filter 5: WAKE_VERIFIED bypass (active worker), non-trivial, unquoted.
    verified = WAKE_VERIFIED_RE.search(unquoted)
    if verified and len(verified.group(1).strip()) >= MIN_VERIFIED_REASON_CHARS:
        return 0

    # Filter 6: stateful wake-pairing verification.
    log_env = os.environ.get("AI_ROOM_CHANNEL_LOG")
    channel_log = Path(log_env) if log_env else DEFAULT_CHANNEL_LOG
    tu_replyto, task_ids, wakes, newest_ts = scan_log(channel_log, target_handle)
    if tu_replyto is None:
        return fail_open("channel log unreadable")

    task_context = resolve_task_context(tool_input, body, tu_replyto, task_ids)

    paired = False
    for ts, reply_to in wakes:
        if task_context is not None and reply_to != task_context:
            continue  # wake is for a different task
        if newest_ts is None or (newest_ts - ts).total_seconds() <= WAKE_WINDOW_SECONDS:
            paired = True
            break
    if paired:
        return 0

    task_note = f" for task {task_context}" if task_context else ""
    msg_lines = [
        f"BLOCKED [worker_gate_wake_pairing_gate] gate/drive to '{target_handle}'{task_note} has no verified wake-pairing.",
        "",
        "A +1 / drive sent as a plain msg does NOT reliably re-wake a worker whose turn already",
        "ended (authority != wake). A worker parked 'holding for the gate' then sits idle for",
        "minutes/hours — its resume_check returns 'idle ok'. This is the ack-idle hang.",
        "",
        "No recent claude->worker, notify=true, in_progress wake-pairing task_update was found",
        f"for '{target_handle}'{task_note}. Resolve via ONE of:",
        "",
        "  (a) PAIR an explicit wake (preferred — the worker is parked):",
        f"      ai_room_task_update(task_id=..., status=\"in_progress\", owner=\"{target_handle}\",",
        f"                          notify=true, to=\"{target_handle}\")   # THIS turn, before this post",
        "      then re-send this gate post.",
        "",
        "  (b) ASSERT the worker is active (no re-wake needed):",
        f"      WAKE_VERIFIED: <reason >= {MIN_VERIFIED_REASON_CHARS} chars, e.g. 'codex mid-turn, posted 20s ago this round'>",
        "",
        "co_lead is exempt (multi-task self-driving); broadcast / multi-target and",
        "ack/status/design/task_dispatch kinds are untouched. The wake-pairing is bound to the",
        "same target (to/owner) and, when a task context is resolvable, the same task (reply_to).",
        "Note: a cited `task_update <msg-id>` / `WAKE_PAIRED: <msg-id>` is resolved to its task —",
        "it is NOT treated as a task id (task ids and msg ids share the <unix_ms>-<hex> shape).",
    ]
    print("\n".join(msg_lines), file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
