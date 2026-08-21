#!/usr/bin/env python3
"""PreToolUse guard for the `advisor` peer's two outbound tools.

Wired ONLY in `.claude/agents/fable-advisor.md` frontmatter. It is deliberately
agent-local: hooks in `.claude/settings.json` are project-global and would apply
this to every session. Because the wiring already guarantees the caller, the
guard resolves no caller identity at all -- re-deriving a fact the wiring
guarantees is what made an earlier revision's predicate unsatisfiable.

The advisor is a pre-artifact advisory peer with two permitted outbound shapes.

REPLY (`ai_room_reply`): answering a solicitation from Claude. A tools allowlist
does not achieve that on its own -- `ai_room_reply` accepts `kind="task_dispatch"`
-- so the surface is closed here rather than by tool selection.

INITIATION (`ai_room_post`): An initiated post may carry: (i) a correction to
my own record; (ii) verbatim-marked Gabe-directed content — route licenses
included, because the authority is Gabe's and I am its transport; (iii)
standing-obligation output. An initiated post may NEVER carry self-originated
route judgment — a route created, changed, or killed absent either a Claude
solicitation or a captured Gabe directive.

The artifact bar survives both transports: it is a rule about what the advisor
may be shown and may judge, not about which tool carries the message, and no
predicate here relaxes it.

REPLY allows only when every predicate holds:

  P9 the payload's tool_name is one of the two acting tools (evaluated FIRST)
  P1 the call's key set is a subset of {body, reply_to, kind}
  P2 body and reply_to are both present and non-empty strings
  P3 kind, when present, is exactly one of {msg, design_proposal}
  P4 the channel journal resolves under C1-C5 and is readable
  P5 reply_to names a record present in that resolved journal
  P6 the parent record's from is exactly "claude"
  P7 the parent record's to is a scalar string exactly equal to "advisor"
  P8 the parent record's kind is exactly one of {msg, design_proposal}

INITIATION allows only when every predicate holds:

  P9  as above
  P10 the call's key set is a subset of {body, to, kind}
  P11 body and to are both present, non-empty strings -- `to` scalar, never a
      list and never absent, because both spellings mean broadcast
  P12 to is exactly "claude"
  P13 kind, when present, is exactly one of {msg, design_proposal}

P1 and P10 are whitelists, not blacklists of known-bad fields, so anything added
to the schema later fails by default instead of needing enumeration. That is what
closes dispatch, board mutation, and `requires_response_from` on the initiation
path without enumerating them: they are simply not in the allowlist. P8 exists
because a solicitation is a msg or a design_proposal -- a dispatch, review
request, or gate record is not something the advisor may answer at all.

P12 is why workers cannot be addressed: `codex`, `codex_co_lead`, `gabe` and any
broadcast fail it. Initiation is a channel to Claude, so an advisor ruling still
reaches a worker only by Claude relaying it, and `@gabe` routing is untouched.
P13 is deliberately a separate constant from the reply path's SOLICITATION_KINDS
even though the two currently hold the same members: they answer different
questions, and one set would silently move both.

P9 is numbered last but runs first; P1-P8 keep their numbers because frozen gate
records already cite them. It exists because an earlier revision wrote
`if payload.get("tool_name") not in (None, ACTING_TOOL): return 0`, which ALLOWS
every tool the guard does not recognise and evaluates a payload with no
tool_name as though it were a reply. Measured: `tool_name="Bash"` carrying a
conforming body/reply_to returned rc=0, as did a payload with tool_name omitted.
That is a fail-OPEN seam in an authorization hook whose own docstring claimed
fail-closed. "Not my tool, don't interfere" is a reasonable default for a
project-global hook; this one is agent-local and matcher-bound to a single tool,
so anything else arriving is anomalous and is refused.

Fail-closed: an unparseable payload, an unresolvable or unreadable channel, an
unexpected tool_name, a reply_to matching no record, a parent whose fields do not
satisfy P6-P8, or any unexpected exception is a REJECT.

One deliberate exception, stated because the prose previously claimed otherwise:
`find_parent` SKIPS individual journal lines that do not parse as JSON objects
and keeps scanning. It does not reject on them. The journal is an append-only
NDJSON log under concurrent writers, so a torn or partial line is an ordinary
transient; rejecting the whole evaluation on one would silence the advisor for
as long as that line existed -- the same total-guard failure this file has
already produced twice. Skipping is safe here because admission requires an
exact `reply_to` id match against a well-formed record, and a line that does not
parse cannot supply one. A malformed line can therefore withhold admission,
never grant it.
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import sys

REPLY_TOOL = "mcp__ai-room__ai_room_reply"
POST_TOOL = "mcp__ai-room__ai_room_post"
ACTING_TOOLS = frozenset({REPLY_TOOL, POST_TOOL})

# The server's channel pattern (config.py `^[A-Za-z0-9_-]{1,64}$`), anchored with
# \Z rather than $ -- here the name composes a filesystem path, and $ also matches
# before a trailing newline, so "claw-code\n" would satisfy a $-anchored check.
CHANNEL_NAME_RE = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")

ALLOWED_KEYS = frozenset({"body", "reply_to", "kind"})
REQUIRED_KEYS = ("body", "reply_to")
SOLICITATION_KINDS = frozenset({"msg", "design_proposal"})
ADVISOR_HANDLE = "advisor"
SOLICITOR_HANDLE = "claude"

# Initiation path. Held separately from the reply path's sets on purpose: they
# answer different questions, so a later change to one must not move the other.
INITIATION_KEYS = frozenset({"body", "to", "kind"})
INITIATION_REQUIRED_KEYS = ("body", "to")
INITIATION_KINDS = frozenset({"msg", "design_proposal"})
INITIATION_RECIPIENT = "claude"


class Reject(Exception):
    """Carries the predicate id so a receipt can name what fired."""

    def __init__(self, predicate: str, detail: str) -> None:
        super().__init__(f"{predicate}: {detail}")
        self.predicate = predicate
        self.detail = detail


def resolve_journal() -> pathlib.Path:
    """C1-C5. Mirrors the room's own precedence; no constant default.

    C3/C4 track `_resolve_room` in the server's paths module: AI_ROOM_DIR IS the
    room directory (it is not joined with channels/<name>), and AI_ROOM_CHANNEL
    alone resolves to ~/.ai-room/channels/<channel>.

    An earlier revision required AI_ROOM_DIR *and* AI_ROOM_CHANNEL together, and
    joined them wrongly. Measured against the live peer, whose environment sets
    AI_ROOM_CHANNEL but not AI_ROOM_DIR: every legitimate reply was rejected with
    "P4 failed -- no channel could be resolved from the environment". Failing
    closed is correct behaviour for an unresolvable channel; failing closed on
    the only environment the advisor ever runs in made the guard total.

    There is still no fallback to the unchannelled ~/.ai-room (`_resolve_room`
    step 4) and none to a hardcoded channel. This guard decides whether an
    outbound message is permitted, and one that silently consults a different
    channel than the one it guards would admit a reply whose parent it never saw.
    """
    direct = os.environ.get("AI_ROOM_CHANNEL_LOG")
    if direct:
        return pathlib.Path(direct)
    alt = os.environ.get("AI_ROOM_CHANNEL_LOG_PATH")
    if alt:
        return pathlib.Path(alt)
    room = os.environ.get("AI_ROOM_DIR")
    if room:
        return pathlib.Path(room) / "messages.jsonl"
    channel = os.environ.get("AI_ROOM_CHANNEL")
    if channel:
        # The channel name now composes a filesystem path, so it is validated
        # against the server's own pattern rather than trusted.
        if not CHANNEL_NAME_RE.match(channel):
            raise Reject("P4", f"channel name {channel!r} is not [A-Za-z0-9_-]{{1,64}}")
        return pathlib.Path.home() / ".ai-room" / "channels" / channel / "messages.jsonl"
    raise Reject("P4", "no channel could be resolved from the environment")


def find_parent(journal: pathlib.Path, parent_id: str) -> dict:
    try:
        text = journal.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise Reject("P4", f"resolved journal is unreadable: {exc}") from exc
    found = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict) and record.get("id") == parent_id:
            found = record  # last write wins
    if found is None:
        raise Reject("P5", f"reply_to {parent_id!r} names no record in this channel")
    return found


def check(tool_input: dict) -> None:
    keys = set(tool_input)
    extra = sorted(keys - ALLOWED_KEYS)
    if extra:
        raise Reject("P1", f"non-allowlisted key(s) present: {', '.join(extra)}")

    for key in REQUIRED_KEYS:
        value = tool_input.get(key)
        if not isinstance(value, str) or not value.strip():
            raise Reject("P2", f"{key} must be a present, non-empty string")

    kind = tool_input.get("kind")
    if kind is not None and kind not in SOLICITATION_KINDS:
        raise Reject("P3", f"kind {kind!r} is not a solicitation kind")

    parent = find_parent(resolve_journal(), tool_input["reply_to"])

    if parent.get("from") != SOLICITOR_HANDLE:
        raise Reject("P6", f"parent is from {parent.get('from')!r}, not {SOLICITOR_HANDLE!r}")

    to = parent.get("to")
    if not isinstance(to, str):
        raise Reject("P7", f"parent `to` must be a scalar string, got {type(to).__name__}")
    if to != ADVISOR_HANDLE:
        raise Reject("P7", f"parent was addressed to {to!r}, not {ADVISOR_HANDLE!r}")

    parent_kind = parent.get("kind")
    if parent_kind not in SOLICITATION_KINDS:
        raise Reject("P8", f"parent kind {parent_kind!r} is not a solicitation")


def check_initiation(tool_input: dict) -> None:
    """P10-P13. An advisor-initiated post, addressed to Claude and no one else.

    No journal lookup: an initiation has no parent by definition, so there is
    nothing to resolve. The reply path's P4-P8 do not apply and are not faked.
    """
    extra = sorted(set(tool_input) - INITIATION_KEYS)
    if extra:
        raise Reject("P10", f"non-allowlisted key(s) present: {', '.join(extra)}")

    for key in INITIATION_REQUIRED_KEYS:
        value = tool_input.get(key)
        if not isinstance(value, str) or not value.strip():
            raise Reject(
                "P11",
                f"{key} must be a present, non-empty scalar string "
                "(a list or an absent `to` is a broadcast)",
            )

    to = tool_input["to"].strip()
    if to != INITIATION_RECIPIENT:
        raise Reject(
            "P12",
            f"initiation addressed to {to!r}; the advisor may initiate only to "
            f"{INITIATION_RECIPIENT!r}",
        )

    kind = tool_input.get("kind")
    if kind is not None and kind not in INITIATION_KINDS:
        raise Reject("P13", f"kind {kind!r} is not permitted for an initiation")


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read() or "{}")
        tool_name = payload.get("tool_name")
        if tool_name not in ACTING_TOOLS:
            raise Reject(
                "P9", f"tool_name {tool_name!r} is not one of {sorted(ACTING_TOOLS)}")
        tool_input = payload.get("tool_input")
        if not isinstance(tool_input, dict):
            raise Reject("P2", "tool_input missing or not an object")
        if tool_name == POST_TOOL:
            check_initiation(tool_input)
        else:
            check(tool_input)
    except Reject as rej:
        print(
            f"BLOCKED [advisor_outbound_gate] {rej.predicate} failed -- {rej.detail}.\n"
            "The advisor may reply to a Claude solicitation addressed to it "
            "(kind msg or design_proposal), using only body/reply_to/kind; or "
            "initiate a post to `claude` alone (kind msg or design_proposal), "
            "using only body/to/kind.",
            file=sys.stderr,
        )
        return 2
    except Exception as exc:  # noqa: BLE001 - fail closed on anything unexpected
        print(
            f"BLOCKED [advisor_outbound_gate] guard could not evaluate the call "
            f"({type(exc).__name__}: {exc}); failing closed.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
