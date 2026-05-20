#!/usr/bin/env python3
"""
PreToolUse hook on `mcp__ai-room__ai_room_post` and `mcp__ai-room__ai_room_reply`.

Enforces a User-input Capture clause for claw-code's ai-room collaboration:
outbound ai-room posts that ask the user a decision must be preceded by an
`AskUserQuestion` capture in the same assistant turn. Block when the post
addresses gabe but no `AskUserQuestion` was captured.

Ported from zenith-fitness with claw-code channel + charter adaptations.

Scope (do not overclaim):
  - This hook only fires on Claude Code outbound calls to the locally
    registered `mcp__ai-room__ai_room_post` / `ai_room_reply` tools, in
    both hyphenated (`ai-room`) and underscored (`ai_room`) name shapes.
  - It does NOT enforce Codex-originated ai-room posts or any other
    client. Truly universal enforcement requires an ai-room MCP/server-
    side gate or Codex-side rule/hook follow-up.

Trigger conditions (any → potential block):
  1. Structural ASK: `to == "gabe"` (string or list containing "gabe") OR
     `requires_response_from == "gabe"` (string or list containing "gabe").
  2. Reply-to auto-target ASK: `to` is unset/empty AND `reply_to` is set
     AND `reply_to`'s sender (resolved via channel log lookup) is gabe.
     `ai_room_reply` auto-targets the original sender of `reply_to` when
     `to` is omitted, so a reply to a gabe-authored message effectively
     addresses gabe even without an explicit `to: "gabe"` field.
  3. Textual ASK: body contains a `@gabe` mention OUTSIDE markdown
     blockquote lines (`> ...`), fenced code blocks (``` ``` ``` and
     `~~~`), and inline code spans (`` `...` ``).

Block decision:
  - Skip ack-kind messages entirely (`kind == "ack"`).
  - Structural ASK + no AskUserQuestion this turn → exit 2 (block).
  - Textual ASK + no AskUserQuestion this turn + no relay-source signature
    in the body → exit 2 (block). Body containing one of
    {AskUserQuestion, captured via, locked answer, user-input capture,
     chat-side capture} is treated as a relay of an already-captured
    answer and is allowed.
  - Anything else → exit 0.

Failure modes:
  - Empty stdin, JSON parse failures of the hook event, transcript read /
    parse errors, unexpected schema in the transcript → fail-open
    (exit 0). Goal: never wedge a turn on a hook bug.
  - MISSING `transcript_path` in the hook event when a structural ask
    is in flight → conservative block (exit 2). Without a transcript we
    cannot prove an `AskUserQuestion` capture exists, so a direct
    `to: "gabe"` / `requires_response_from: "gabe"` without that proof
    fails closed. This is intentional: structural asks to gabe without
    transcript evidence are safer to block than to allow.
"""
import json
import os
import re
import sys
from pathlib import Path

# Default channel log location (claw-code). Override with AI_ROOM_CHANNEL_LOG.
DEFAULT_CHANNEL_LOG = Path("/home/gabe/.ai-room/channels/claw-code/messages.jsonl")

WATCHED_TOOLS = {
    # Hyphenated server-name shape (Claude Code default for `ai-room` MCP).
    "mcp__ai-room__ai_room_post",
    "mcp__ai-room__ai_room_reply",
    # Underscored server-name shape (some client variants).
    "mcp__ai_room__ai_room_post",
    "mcp__ai_room__ai_room_reply",
}

# Capture-source signatures that indicate the body is a relay of an
# already-captured locked answer rather than a fresh question to gabe.
RELAY_SOURCE_PATTERNS = [
    r"AskUserQuestion",
    r"captured via",
    r"locked answer",
    r"user-input capture",
    r"chat-side capture",
]
RELAY_SOURCE_RE = re.compile("|".join(RELAY_SOURCE_PATTERNS), re.IGNORECASE)

# Match a literal @gabe mention as a word (not @gabe-something).
AT_GABE_RE = re.compile(r"(?<![A-Za-z0-9_])@gabe\b", re.IGNORECASE)


def strip_quoted_segments(body: str) -> str:
    """Remove markdown blockquote lines, fenced code blocks, and inline
    backtick spans so a @gabe sitting inside quoted prior text or a
    documented code example does not trip the textual matcher.

    Fence handling is line-by-line state-tracked: ```...``` and ~~~...~~~
    fences are excised, including the fence delimiters themselves. An
    unclosed fence keeps the rest of the body excluded — conservative
    direction, since unmatched fences make the matcher less aggressive
    rather than more.
    """
    lines_out: list[str] = []
    in_fence = False
    for ln in body.splitlines():
        stripped_lead = ln.lstrip()
        # Toggle fence state on a line starting with ``` or ~~~ (allows
        # `` ```python `` etc.).
        if stripped_lead.startswith("```") or stripped_lead.startswith("~~~"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        # Skip blockquote lines (allow leading whitespace before `>`).
        if stripped_lead.startswith(">"):
            continue
        # Strip inline backtick code spans: `...`
        ln_clean = re.sub(r"`[^`]*`", "", ln)
        lines_out.append(ln_clean)
    return "\n".join(lines_out)


def to_targets_user(to_field, user: str) -> bool:
    if isinstance(to_field, str):
        return to_field.strip().lower() == user
    if isinstance(to_field, list):
        return any(
            isinstance(t, str) and t.strip().lower() == user for t in to_field
        )
    return False


def to_is_empty(to_field) -> bool:
    """True when `to` is unset, None, empty string, or empty list. Used to
    decide whether to fall back to reply_to auto-target inference."""
    if to_field is None:
        return True
    if isinstance(to_field, str):
        return not to_field.strip()
    if isinstance(to_field, list):
        return not any(isinstance(t, str) and t.strip() for t in to_field)
    return False


def resolve_reply_to_sender(channel_log: Path, msg_id: str) -> str | None:
    """Look up `msg_id` in the channel log JSONL and return its `from` field.
    Bounded tail scan. Fail-open (None) on missing log, parse error, or msg
    not found."""
    if not channel_log.exists() or not msg_id:
        return None
    try:
        with channel_log.open("r", encoding="utf-8", errors="replace") as fh:
            # Bounded tail scan: read last ~2000 lines. Most reply_to msgs
            # are recent; older msgs scrolling out of the window fail-open
            # to None which means we can't infer the sender — safer than
            # blocking on transient lookup miss.
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


CHANNEL_FROM_RE = re.compile(r'from="([^"]+)"')


def _is_human_prompt_text(text: str) -> bool:
    """Heuristic classifier for user-message content text. True for new
    human/gabe prompts; False for intervening event injections (peer channel
    events, system-reminders) that should NOT serve as turn boundaries.

    Decision tree:
      - Pure system-reminder injection (no channel-source) → False (runtime event)
      - Channel event with `from="<peer>"` where peer != gabe → False (peer event)
      - Channel event with `from="gabe"` → True (gabe REPL prompt)
      - Anything else → True (direct typed prompt, default open)
    """
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
    """True when entry is a user-role message representing a NEW human prompt
    that should serve as a turn boundary. False for tool results, peer
    channel-event injections, and pure system-reminder injections.

    Why each filter:
    - Tool results (role=user with tool_result content blocks) are the
      assistant's own tool roundtrips, not new prompts.
    - Peer channel events (`<channel source="ai-room" from="codex_*" ...>`)
      arrive as user-role injections but they're peer messages interrupting
      the current assistant turn, not new gabe prompts.
    - Pure system-reminders (`<system-reminder>` without `<channel source=`)
      are runtime injections (date context, task-tracker pings, hook output)
      that shouldn't reset turn boundaries.

    Without these filters, AskUserQuestion captures earlier in a multi-tool
    assistant turn get missed because every tool_result + every peer channel
    event resets the scan window. Observed live in zenith-fitness:
    relaying a gabe-locked answer after a cross-thread roundtrip to a
    peer got blocked because the peer's reply system-reminder bumped the
    turn boundary past the AskUserQuestion call.
    """
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
        # Check text blocks for peer-event / system-reminder markers
        for block in non_tool_result_blocks:
            text = block.get("text", "") if isinstance(block, dict) else ""
            if not _is_human_prompt_text(text):
                return False
        return True
    if isinstance(content, str):
        return _is_human_prompt_text(content)
    return True


def find_askuserquestion_in_turn(transcript_path: str) -> bool:
    """Scan the transcript JSONL for an AskUserQuestion tool_use in the
    current assistant turn (since the most recent REAL user prompt). Returns
    True if found, False otherwise. Fail-open on parse errors.
    """
    p = Path(transcript_path)
    if not p.exists():
        return False
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
        return False

    # Find the index of the last REAL user prompt (turn boundary), skipping
    # tool_result-only entries.
    last_user_idx = -1
    for i, entry in enumerate(entries):
        if is_real_user_prompt(entry):
            last_user_idx = i

    scan = entries[last_user_idx + 1 :] if last_user_idx >= 0 else entries

    for entry in scan:
        msg = entry.get("message") or {}
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use" and block.get("name") == "AskUserQuestion":
                return True
    return False


def block(reason_lines: list[str]) -> None:
    print(
        "BLOCKED [at_gabe_askuserquestion_gate] outbound ai-room post addresses gabe but no AskUserQuestion captured this turn.",
        file=sys.stderr,
    )
    for ln in reason_lines:
        print(f"  - {ln}", file=sys.stderr)
    print(
        "Per User-input Capture discipline (.claude/rules/AI_ROOM_COLLAB.md §\"Autonomy\" + §\"Disagreement\"):",
        file=sys.stderr,
    )
    print(
        "  Run AskUserQuestion first to capture structured intent.",
        file=sys.stderr,
    )
    print(
        "  Then post the relay with options/locked-answer/source/effect/rejected-alternatives.",
        file=sys.stderr,
    )
    print(
        "  The chat-side answer is provenance context; the room post IS the durable gate.",
        file=sys.stderr,
    )
    sys.exit(2)


def main() -> None:
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            sys.exit(0)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            sys.exit(0)

        tool_name = data.get("tool_name", "")
        if tool_name not in WATCHED_TOOLS:
            sys.exit(0)

        tool_input = data.get("tool_input", {}) or {}
        if not isinstance(tool_input, dict):
            sys.exit(0)

        # Skip ack-kind messages — they are not decision asks.
        if (tool_input.get("kind") or "").strip().lower() == "ack":
            sys.exit(0)

        to_field = tool_input.get("to")
        rrf = tool_input.get("requires_response_from")
        reply_to = tool_input.get("reply_to")
        body = tool_input.get("body", "") or ""
        if not isinstance(body, str):
            body = str(body)

        # Trigger 1: explicit `to` or `requires_response_from` mentions gabe.
        structural_ask = to_targets_user(to_field, "gabe") or to_targets_user(
            rrf, "gabe"
        )

        # Trigger 2: reply_to auto-target. When `to` is unset/empty AND
        # `reply_to` is set, ai_room_reply auto-targets the original sender of
        # `reply_to`. If that sender is gabe, the post effectively addresses
        # gabe even without an explicit `to: "gabe"` field.
        replyto_addresses_gabe = False
        replyto_sender: str | None = None
        if not structural_ask and to_is_empty(to_field) and isinstance(
            reply_to, str
        ) and reply_to.strip():
            log_env = os.environ.get("AI_ROOM_CHANNEL_LOG")
            channel_log = Path(log_env) if log_env else DEFAULT_CHANNEL_LOG
            replyto_sender = resolve_reply_to_sender(channel_log, reply_to.strip())
            if isinstance(replyto_sender, str) and replyto_sender.strip().lower() == "gabe":
                replyto_addresses_gabe = True

        cleaned_body = strip_quoted_segments(body)
        textual_ask = bool(AT_GABE_RE.search(cleaned_body))

        if not (structural_ask or replyto_addresses_gabe or textual_ask):
            sys.exit(0)

        transcript_path = data.get("transcript_path") or ""
        captured = (
            find_askuserquestion_in_turn(transcript_path)
            if transcript_path
            else False
        )

        if captured:
            sys.exit(0)

        # No AskUserQuestion found in this assistant turn.
        if structural_ask or replyto_addresses_gabe:
            reasons = []
            if to_targets_user(to_field, "gabe"):
                reasons.append(f"to includes 'gabe' (value: {to_field!r})")
            if to_targets_user(rrf, "gabe"):
                reasons.append(f"requires_response_from includes 'gabe' (value: {rrf!r})")
            if replyto_addresses_gabe:
                reasons.append(
                    f"reply_to auto-targets gabe (reply_to={reply_to!r}, sender={replyto_sender!r})"
                )
            block(reasons)

        # Textual @gabe outside quoted text. Allow if body carries a
        # relay-source signature (i.e., it is a relay of an answer already
        # captured upstream this run, not a fresh ask). Otherwise block.
        if textual_ask:
            if RELAY_SOURCE_RE.search(body):
                sys.exit(0)
            block([
                "body contains @gabe mention outside blockquotes",
                "no relay-source signature found in body (e.g., 'AskUserQuestion', 'captured via', 'locked answer', 'user-input capture', 'chat-side capture')",
            ])

        sys.exit(0)
    except Exception as e:  # pragma: no cover — fail-open, do not wedge
        print(f"at_gabe_askuserquestion_gate hook error: {e}", file=sys.stderr)
        sys.exit(0)


if __name__ == "__main__":
    main()
