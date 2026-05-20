#!/usr/bin/env python3
"""
Stop hook: end-of-turn audit for R&D-team cross-thread discipline.

Enforces (by post-hoc audit, NOT block) the rule in
`.claude/rules/AI_ROOM_COLLAB.md` §"Cross-thread is mandatory at
thinking boundaries":

> Every thinking-class step in the R&D loop cross-threads to codex.
> Implementation-class steps stay solo with claude. ... Cross-thread
> is the default rate of the channel, not occasional.

The Stop hook fires at end of every assistant turn. We scan the
transcript for the last completed turn and ask:

  1. Did claude post substantively to gabe in the ai-room channel?
     (outbound `mcp__ai-room__ai_room_post`/`_reply` with the
     4-shape @gabe predicate from `at_gabe_askuserquestion_gate.py`;
     kind != "ack"; body >= MIN_SUBSTANTIVE_BODY_CHARS)

  2. Did claude cross-thread to codex_co_lead this turn?
     (outbound `mcp__ai-room__ai_room_post`/`_reply` with
     `to` containing "codex_co_lead", `requires_response_from`
     containing "codex_co_lead", reply_to whose sender resolves
     to codex_co_lead, or `@codex_co_lead` body mention outside
     blockquotes/quoted prior text)

If (1) AND NOT (2) → log a violation to /tmp/cross_thread_audit.jsonl
for periodic review by gabe.

This is an AUDIT hook, not a gate:
  - Always exits 0 (does NOT block).
  - Fail-open on all error conditions (transcript missing, parse
    error, log write failure) — goal: never wedge a turn on hook
    behavior.

Audit log location:
  /tmp/cross_thread_audit.jsonl (default; override via
  CROSS_THREAD_AUDIT_LOG env var). Append-only JSONL. Survives within
  the WSL session; cleared on reboot. Use a persistent path if you
  want longer history.

Log schema (one JSON object per line):
  {
    "ts": "ISO8601 UTC",
    "session_id": "<claude-code session id>",
    "violation": "no_cross_thread_before_gabe_room_response",
    "turn_summary": {
      "gabe_room_posts": <int>,
      "codex_cross_threads": <int>,
      "gabe_post_kinds": [<str>...],
      "gabe_post_body_lens": [<int>...]
    },
    "transcript_path": "<absolute path>"
  }

Scope (do not overclaim):
  - Only audits outbound `mcp__ai-room__ai_room_post`/`_reply` tool
    calls — does NOT audit chat-side assistant text responses.
    Chat-side substantive responses are harder to classify
    mechanically; v1 ships the room-post surface only.
  - Skips ack-kind messages.
  - Skips trivial outbound (body < MIN_SUBSTANTIVE_BODY_CHARS) — most
    one-liner status/concur posts pass without audit.
  - Codex-side cross-thread is detected by outbound posts FROM claude
    TO codex_co_lead; cross-threads initiated by codex would not
    register here (codex's posts are not claude's tool calls).
"""

import datetime
import json
import os
import re
import sys
from pathlib import Path

WATCHED_TOOLS = {
    "mcp__ai-room__ai_room_post",
    "mcp__ai-room__ai_room_reply",
    "mcp__ai_room__ai_room_post",
    "mcp__ai_room__ai_room_reply",
}

# Body length threshold below which a gabe-addressed room post is
# treated as trivial (not substantive). Most one-liner relays / status
# posts fall under this.
MIN_SUBSTANTIVE_BODY_CHARS = 200

# Default channel log location (claw-code). Override with
# AI_ROOM_CHANNEL_LOG env var.
DEFAULT_CHANNEL_LOG = Path("/home/gabe/.ai-room/channels/claw-code/messages.jsonl")

# Default audit log location. Override with CROSS_THREAD_AUDIT_LOG env var.
DEFAULT_AUDIT_LOG = Path("/tmp/cross_thread_audit.jsonl")

# Match a literal @gabe or @codex_co_lead mention as a word.
AT_GABE_RE = re.compile(r"(?<![A-Za-z0-9_])@gabe\b", re.IGNORECASE)
AT_CODEX_RE = re.compile(r"(?<![A-Za-z0-9_])@codex_co_lead\b", re.IGNORECASE)

CHANNEL_FROM_RE = re.compile(r'from="([^"]+)"')


def strip_quoted_segments(body: str) -> str:
    """Remove markdown blockquote lines, fenced code blocks, and inline
    backtick spans so a mention sitting inside quoted prior text or a
    documented code example does not trip the textual matcher."""
    lines_out: list[str] = []
    in_fence = False
    for ln in body.splitlines():
        stripped_lead = ln.lstrip()
        if stripped_lead.startswith("```") or stripped_lead.startswith("~~~"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if stripped_lead.startswith(">"):
            continue
        ln_clean = re.sub(r"`[^`]*`", "", ln)
        lines_out.append(ln_clean)
    return "\n".join(lines_out)


def to_targets_handle(to_field, handle: str) -> bool:
    """True if `to_field` (string or list) targets `handle` (lowercase compare)."""
    if isinstance(to_field, str):
        return to_field.strip().lower() == handle
    if isinstance(to_field, list):
        return any(
            isinstance(t, str) and t.strip().lower() == handle for t in to_field
        )
    return False


def to_is_empty(to_field) -> bool:
    if to_field is None:
        return True
    if isinstance(to_field, str):
        return not to_field.strip()
    if isinstance(to_field, list):
        return not any(isinstance(t, str) and t.strip() for t in to_field)
    return False


def resolve_reply_to_sender(channel_log: Path, msg_id: str) -> str | None:
    """Look up `msg_id` in the channel log JSONL and return its `from` field."""
    if not channel_log.exists() or not msg_id:
        return None
    try:
        with channel_log.open("r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()[-2000:]
    except OSError:
        return None
    for raw in reversed(lines):
        raw = raw.strip()
        if not raw:
            continue
        try:
            rec = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if rec.get("id") == msg_id:
            sender = rec.get("from")
            return sender if isinstance(sender, str) else None
    return None


def addresses_handle(tool_input: dict, handle: str, channel_log: Path) -> bool:
    """Full 4-shape predicate: structural `to`, `requires_response_from`,
    reply_to auto-target, or @mention in body outside quoted text.
    `handle` is lowercase (e.g., "gabe", "codex_co_lead").
    """
    if not isinstance(tool_input, dict):
        return False

    to_field = tool_input.get("to")
    rrf = tool_input.get("requires_response_from")
    reply_to = tool_input.get("reply_to")
    body = tool_input.get("body", "") or ""
    if not isinstance(body, str):
        body = str(body)

    if to_targets_handle(to_field, handle) or to_targets_handle(rrf, handle):
        return True

    if to_is_empty(to_field) and isinstance(reply_to, str) and reply_to.strip():
        sender = resolve_reply_to_sender(channel_log, reply_to.strip())
        if isinstance(sender, str) and sender.strip().lower() == handle:
            return True

    cleaned_body = strip_quoted_segments(body)
    if handle == "gabe" and AT_GABE_RE.search(cleaned_body):
        return True
    if handle == "codex_co_lead" and AT_CODEX_RE.search(cleaned_body):
        return True

    return False


def _is_human_prompt_text(text: str) -> bool:
    """Heuristic: True for fresh human prompts (turn boundary). False for
    intervening peer-channel injections and pure system-reminders."""
    has_channel = "<channel source=" in text
    has_sysreminder = "<system-reminder>" in text
    if has_sysreminder and not has_channel:
        return False
    if has_channel:
        m = CHANNEL_FROM_RE.search(text)
        if m and m.group(1).strip().lower() != "gabe":
            return False
    return True


def is_real_user_prompt(entry: dict) -> bool:
    """True when entry is a user-role message representing a NEW human prompt."""
    msg = entry.get("message") or {}
    role = msg.get("role") or entry.get("type")
    if role != "user":
        return False
    content = msg.get("content")
    if isinstance(content, list):
        non_tool_result_blocks = [
            b for b in content
            if isinstance(b, dict) and b.get("type") != "tool_result"
        ]
        if not non_tool_result_blocks:
            return False
        for block in non_tool_result_blocks:
            text = block.get("text", "") if isinstance(block, dict) else ""
            if not _is_human_prompt_text(text):
                return False
        return True
    if isinstance(content, str):
        return _is_human_prompt_text(content)
    return True


def scan_turn_tool_uses(transcript_path: str) -> list[dict]:
    """Return all assistant tool_use blocks since the most recent real
    user-prompt boundary."""
    p = Path(transcript_path)
    if not p.exists():
        return []
    try:
        with p.open("r", encoding="utf-8", errors="replace") as f:
            entries = []
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    entries.append(json.loads(ln))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []

    last_user_idx = -1
    for i, entry in enumerate(entries):
        if is_real_user_prompt(entry):
            last_user_idx = i

    scan = entries[last_user_idx + 1:] if last_user_idx >= 0 else entries

    tool_uses: list[dict] = []
    for entry in scan:
        msg = entry.get("message") or {}
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use":
                tool_uses.append(block)
    return tool_uses


def main() -> None:
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            sys.exit(0)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            sys.exit(0)

        transcript_path = data.get("transcript_path") or ""
        if not transcript_path:
            sys.exit(0)

        log_env = os.environ.get("AI_ROOM_CHANNEL_LOG")
        channel_log = Path(log_env) if log_env else DEFAULT_CHANNEL_LOG

        tool_uses = scan_turn_tool_uses(transcript_path)
        if not tool_uses:
            sys.exit(0)

        gabe_substantive_posts: list[dict] = []
        codex_cross_threads: list[dict] = []

        for tu in tool_uses:
            tool_name = tu.get("name", "")
            if tool_name not in WATCHED_TOOLS:
                continue
            tool_input = tu.get("input", {}) or {}
            if not isinstance(tool_input, dict):
                continue

            kind = (tool_input.get("kind") or "").strip().lower()
            if kind == "ack":
                continue

            body = tool_input.get("body", "") or ""
            if not isinstance(body, str):
                body = str(body)
            body_len = len(body)

            if addresses_handle(tool_input, "codex_co_lead", channel_log):
                codex_cross_threads.append({"kind": kind, "body_len": body_len})

            if addresses_handle(tool_input, "gabe", channel_log):
                if body_len >= MIN_SUBSTANTIVE_BODY_CHARS:
                    gabe_substantive_posts.append({"kind": kind, "body_len": body_len})

        # Violation: substantive gabe room post(s) without any cross-thread to codex.
        if gabe_substantive_posts and not codex_cross_threads:
            audit_log_env = os.environ.get("CROSS_THREAD_AUDIT_LOG")
            audit_log = Path(audit_log_env) if audit_log_env else DEFAULT_AUDIT_LOG
            record = {
                "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "session_id": data.get("session_id", ""),
                "violation": "no_cross_thread_before_gabe_room_response",
                "turn_summary": {
                    "gabe_room_posts": len(gabe_substantive_posts),
                    "codex_cross_threads": 0,
                    "gabe_post_kinds": [p["kind"] for p in gabe_substantive_posts],
                    "gabe_post_body_lens": [p["body_len"] for p in gabe_substantive_posts],
                },
                "transcript_path": transcript_path,
            }
            try:
                audit_log.parent.mkdir(parents=True, exist_ok=True)
                with audit_log.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record) + "\n")
            except OSError:
                # Audit logging is best-effort; never wedge a turn on log write failure.
                pass

        sys.exit(0)
    except Exception as e:  # pragma: no cover — fail-open, never wedge
        print(f"cross_thread_audit hook error: {e}", file=sys.stderr)
        sys.exit(0)


if __name__ == "__main__":
    main()
