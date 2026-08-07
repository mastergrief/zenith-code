"""Negative battery for .claude/hooks/advisor_outbound_gate.py.

Scope by CONSTRUCTION, never by a fixed total and never by a closed coverage
list: every predicate the guard declares (P1-P8 plus P9, the outer tool_name
check) carries at least one negative arm asserting THAT predicate's rejection,
every RESOLVING channel-resolution branch carries a standalone arm that SELECTS
AND EXECUTES that branch, and at least one positive control is observed passing.

Naming exact identifiers is REQUIRED -- they are what makes the contract
auditable. What is banned is narrower: a total, or a closed list, standing in
for the coverage contract, so that adding a predicate or a branch silently makes
the stated scope false. An earlier revision of this header overcorrected to
"never by enumeration", which is both untrue of this file and actively harmful:
followed literally it would strip the very identifiers above.

"Executes" is load-bearing and was learned the expensive way. C2 was once
covered only as the losing side of a precedence contest, which proves ordering
while never running C2's own resolution -- measured, the suite stayed green with
the C2 branch deleted outright. A closed coverage list is the same hazard as a
fixed total: it goes stale the moment a branch is added, and it did.

The count is deliberately absent. An earlier header read "Sixteen arms: N1-N15
reject, P0 allows" and "scores 15/15"; adding N16-N18, a second positive control
and the channel-positive arms made all three numbers false while every one of
them still looked authoritative. This lineage already moved the canonical task
contract off fixed counts for exactly that reason, and the header did not
follow. A number here describes the battery on the day it was typed; the
construction rule describes it permanently.

Each negative asserts the PREDICATE that fired, not merely that something
failed -- an arm checking only "rc==2" is satisfied by any rejection, including
one for the wrong reason, which is how a battery goes green over nothing. The
positive controls are load-bearing in the other direction: without an observed
silent pass, a guard that rejects everything scores perfectly and is worthless.

Every arm drives the real hook as a SUBPROCESS with a real PreToolUse payload on
stdin. Importing the module and calling `check()` would test a reimplementation
of the invocation path rather than the path that runs in production.

Run: PYTHONPATH=. python3 -m pytest tests/test_advisor_outbound_gate_v1.py -q
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import subprocess
import sys
import tempfile

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
GUARD = REPO_ROOT / ".claude" / "hooks" / "advisor_outbound_gate.py"
TOOL = "mcp__ai-room__ai_room_reply"

PARENT_ID = "1700000000000-aaaaaaaa"
GOOD_PARENT = {"id": PARENT_ID, "from": "claude", "to": "advisor", "kind": "msg",
               "body": "what decompositions would you consider here?"}
GOOD_CALL = {"body": "three alternatives, with predicted failure modes", "reply_to": PARENT_ID}

# The live journal must never be touched by this suite; [data] assertion below.
LIVE_JOURNAL = pathlib.Path.home() / ".ai-room" / "channels" / "claw-code" / "messages.jsonl"


def write_journal(path: pathlib.Path, records) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


def run_guard(tool_input, env_overrides, *, tool_name=TOOL, omit_tool_name=False):
    env = dict(os.environ)
    # Clear every resolution input so an arm can only see what it sets.
    for key in ("AI_ROOM_CHANNEL_LOG", "AI_ROOM_CHANNEL_LOG_PATH", "AI_ROOM_DIR", "AI_ROOM_CHANNEL"):
        env.pop(key, None)
    env["HOME"] = str(pathlib.Path(tempfile.mkdtemp(prefix="advisor-guard-home-")))
    env.update({k: str(v) for k, v in env_overrides.items()})
    payload_obj = {"tool_input": tool_input}
    if not omit_tool_name:
        payload_obj["tool_name"] = tool_name
    payload = json.dumps(payload_obj)
    proc = subprocess.run(
        [sys.executable, str(GUARD)],
        input=payload, capture_output=True, text=True, env=env, timeout=30,
    )
    return proc


def assert_rejected(proc, predicate):
    assert proc.returncode == 2, (
        f"expected REJECT (rc=2), got rc={proc.returncode}; stderr={proc.stderr!r}"
    )
    assert f"{predicate} failed" in proc.stderr, (
        f"expected {predicate} to fire; stderr={proc.stderr!r}"
    )


@pytest.fixture
def channel(tmp_path):
    """A resolvable channel whose journal holds one conforming parent."""
    journal = write_journal(tmp_path / "chan-a" / "messages.jsonl", [GOOD_PARENT])
    return {"AI_ROOM_CHANNEL_LOG": journal}, journal


# --- P0: the positive control ------------------------------------------------

def test_P0_conforming_call_is_allowed(channel):
    env, _ = channel
    proc = run_guard(dict(GOOD_CALL), env)
    assert proc.returncode == 0, f"positive control must ALLOW; stderr={proc.stderr!r}"


def test_P0_with_explicit_allowed_kind(channel):
    env, _ = channel
    proc = run_guard({**GOOD_CALL, "kind": "design_proposal"}, env)
    assert proc.returncode == 0, f"design_proposal must ALLOW; stderr={proc.stderr!r}"


# --- P9: the outer payload names the acting tool ------------------------------

def test_N17_missing_tool_name_rejected(channel):
    """MEASURED fail-open, not a hypothetical: the earlier
    `not in (None, ACTING_TOOL)` form evaluated a tool_name-less payload as
    though it were a reply and returned rc=0."""
    env, _ = channel
    proc = run_guard(dict(GOOD_CALL), env, omit_tool_name=True)
    assert_rejected(proc, "P9")


def test_N18_wrong_tool_name_rejected(channel):
    """The worse half of the same defect: `tool_name="Bash"` carrying a
    conforming body/reply_to returned rc=0 -- the guard ALLOWED every tool it did
    not recognise. Agent-local matcher binding makes anything but the acting tool
    anomalous, so it is refused rather than waved through."""
    env, _ = channel
    proc = run_guard(dict(GOOD_CALL), env, tool_name="Bash")
    assert_rejected(proc, "P9")


# --- P1: key allowlist -------------------------------------------------------

def test_N1_explicit_recipient_override_rejected(channel):
    env, _ = channel
    assert_rejected(run_guard({**GOOD_CALL, "to": "codex"}, env), "P1")


def test_N2_response_obligation_field_rejected(channel):
    env, _ = channel
    assert_rejected(run_guard({**GOOD_CALL, "requires_response_from": "claude"}, env), "P1")


# --- P2: required fields -----------------------------------------------------

def test_N3_missing_reply_to_rejected(channel):
    env, _ = channel
    assert_rejected(run_guard({"body": "unsolicited"}, env), "P2")


# --- P3: own kind ------------------------------------------------------------

def test_N4_receipt_kind_rejected(channel):
    env, _ = channel
    assert_rejected(run_guard({**GOOD_CALL, "kind": "validation_receipt"}, env), "P3")


# --- P4: channel resolution --------------------------------------------------

def test_N5_no_channel_env_rejected():
    """C4: unresolvable is a REJECT. There is no constant default to fall to."""
    assert_rejected(run_guard(dict(GOOD_CALL), {}), "P4")


def test_N6_unreadable_journal_rejected(tmp_path):
    missing = tmp_path / "nope" / "messages.jsonl"
    assert_rejected(run_guard(dict(GOOD_CALL), {"AI_ROOM_CHANNEL_LOG": missing}), "P4")


def test_C2_ai_room_channel_log_path_alone_resolves(tmp_path):
    """C2 must SELECT and EXECUTE, not merely lose a precedence contest.

    The precedence arm (N7) sets C2 only as the branch that must NOT win, so it
    proves ordering while never running C2's own resolution. Measured: with C2
    deleted from the guard, every C2 arm that existed before this one still
    passed. Deleting the branch, returning the wrong path, or making it
    unreadable-by-construction would all have left the suite green.

    C2 is an advertised authorization-journal source, so an unexecuted fallback
    can rot silently while the precedence negatives keep passing -- and the guard
    then becomes total in exactly the environments that use it, which is the
    production failure already observed for C3 and C4. This arm is the smallest
    check that fires on that."""
    journal = write_journal(tmp_path / "alt" / "messages.jsonl", [GOOD_PARENT])
    proc = run_guard(dict(GOOD_CALL), {"AI_ROOM_CHANNEL_LOG_PATH": journal})
    assert proc.returncode == 0, (
        f"AI_ROOM_CHANNEL_LOG_PATH alone must resolve and ALLOW; stderr={proc.stderr!r}"
    )


def test_C3_ai_room_dir_alone_resolves(tmp_path):
    """AI_ROOM_DIR IS the room dir -- it is NOT joined with channels/<name>.
    Measured against paths.py _resolve_room, whose first branch returns it
    directly. An earlier revision joined it wrongly and required a channel too."""
    write_journal(tmp_path / "room" / "messages.jsonl", [GOOD_PARENT])
    proc = run_guard(dict(GOOD_CALL), {"AI_ROOM_DIR": tmp_path / "room"})
    assert proc.returncode == 0, f"AI_ROOM_DIR alone must resolve; stderr={proc.stderr!r}"


def test_C4_ai_room_channel_alone_resolves(tmp_path):
    """The environment the advisor ACTUALLY runs in: AI_ROOM_CHANNEL set,
    AI_ROOM_DIR unset. Measured on the live peer -- every legitimate reply was
    rejected with "P4 failed -- no channel could be resolved from the
    environment" until this branch existed. Fails if C4 regresses to requiring
    AI_ROOM_DIR, which would make the guard reject the only path it must allow."""
    write_journal(tmp_path / ".ai-room" / "channels" / "chan-x" / "messages.jsonl",
                  [GOOD_PARENT])
    proc = run_guard(dict(GOOD_CALL), {"AI_ROOM_CHANNEL": "chan-x", "HOME": tmp_path})
    assert proc.returncode == 0, f"AI_ROOM_CHANNEL alone must resolve; stderr={proc.stderr!r}"


@pytest.mark.parametrize("bad", ["../../etc", "chan/../other", "chan-x\n", "a" * 65])
def test_N16_non_conforming_channel_name_rejected(tmp_path, bad):
    """The channel name composes a filesystem path, so it is validated rather
    than trusted. `chan-x\\n` is separate on purpose: a $-anchored pattern -- the
    server's own spelling -- accepts a trailing newline.

    Asserting the NAME-VALIDATION message, not merely "P4 fired": measured, an
    unvalidated `../../etc` also rejects as P4-unreadable, so a P4-only assertion
    passes with validation removed. `chan/../other` is the case that shows why it
    matters -- it resolves to a real, readable, DIFFERENT channel, which is the
    one outcome this guard must never have."""
    proc = run_guard(dict(GOOD_CALL), {"AI_ROOM_CHANNEL": bad, "HOME": tmp_path})
    assert_rejected(proc, "P4")
    assert "is not [A-Za-z0-9_-]" in proc.stderr, (
        f"rejected, but not by name validation; stderr={proc.stderr!r}"
    )


def test_C1_wins_over_ai_room_channel(tmp_path):
    """Precedence: an explicit journal must not be overridden by a channel that
    happens to hold the parent. Fails if the branches are reordered."""
    write_journal(tmp_path / ".ai-room" / "channels" / "chan-x" / "messages.jsonl",
                  [GOOD_PARENT])
    empty = write_journal(tmp_path / "explicit" / "messages.jsonl", [])
    assert_rejected(run_guard(dict(GOOD_CALL),
                              {"AI_ROOM_CHANNEL_LOG": empty,
                               "AI_ROOM_CHANNEL": "chan-x", "HOME": tmp_path}), "P5")


# --- P5: parent resolution in THIS channel -----------------------------------

def test_N7_env_precedence_conflict_C1_wins(tmp_path):
    """C1 must win over C2. The parent lives ONLY in the C2 channel, so if the
    guard consulted C2 -- or merged the two -- it would find the parent and
    ALLOW. This arm fails exactly when precedence is wrong."""
    c1 = write_journal(tmp_path / "c1" / "messages.jsonl", [])
    c2 = write_journal(tmp_path / "c2" / "messages.jsonl", [GOOD_PARENT])
    proc = run_guard(dict(GOOD_CALL),
                     {"AI_ROOM_CHANNEL_LOG": c1, "AI_ROOM_CHANNEL_LOG_PATH": c2})
    assert_rejected(proc, "P5")


def test_N8_fabricated_parent_id_rejected(channel):
    env, _ = channel
    assert_rejected(run_guard({**GOOD_CALL, "reply_to": "1700000000000-ffffffff"}, env), "P5")


def test_N9_parent_only_in_a_different_channel_rejected(tmp_path):
    """Same id, wrong channel. The guard must be bound to the channel it guards."""
    write_journal(tmp_path / "other" / "messages.jsonl", [GOOD_PARENT])
    here = write_journal(tmp_path / "here" / "messages.jsonl", [])
    assert_rejected(run_guard(dict(GOOD_CALL), {"AI_ROOM_CHANNEL_LOG": here}), "P5")


# --- P6: parent author -------------------------------------------------------

def test_N10_non_claude_solicitation_rejected(tmp_path):
    journal = write_journal(tmp_path / "c" / "messages.jsonl",
                            [{**GOOD_PARENT, "from": "codex"}])
    assert_rejected(run_guard(dict(GOOD_CALL), {"AI_ROOM_CHANNEL_LOG": journal}), "P6")


# --- P7: parent addressed to advisor, as a scalar ----------------------------

def test_N11_broadcast_parent_rejected(tmp_path):
    journal = write_journal(tmp_path / "c" / "messages.jsonl",
                            [{**GOOD_PARENT, "to": None}])
    assert_rejected(run_guard(dict(GOOD_CALL), {"AI_ROOM_CHANNEL_LOG": journal}), "P7")


def test_N12_multi_target_parent_rejected(tmp_path):
    journal = write_journal(tmp_path / "c" / "messages.jsonl",
                            [{**GOOD_PARENT, "to": ["claude", "codex"]}])
    assert_rejected(run_guard(dict(GOOD_CALL), {"AI_ROOM_CHANNEL_LOG": journal}), "P7")


def test_N13_single_element_array_parent_rejected(tmp_path):
    """Separate from N12 on purpose: ["advisor"] is what a scalar check written
    as a truthiness test waves through."""
    journal = write_journal(tmp_path / "c" / "messages.jsonl",
                            [{**GOOD_PARENT, "to": ["advisor"]}])
    assert_rejected(run_guard(dict(GOOD_CALL), {"AI_ROOM_CHANNEL_LOG": journal}), "P7")


# --- P8: parent kind is a solicitation ---------------------------------------

def test_N14_task_dispatch_parent_rejected(tmp_path):
    """The arm an earlier revision declared but never constrained: a Claude-
    authored, scalar, correctly-addressed parent that is nonetheless a dispatch."""
    journal = write_journal(tmp_path / "c" / "messages.jsonl",
                            [{**GOOD_PARENT, "kind": "task_dispatch"}])
    assert_rejected(run_guard(dict(GOOD_CALL), {"AI_ROOM_CHANNEL_LOG": journal}), "P8")


def test_N15_review_request_parent_rejected(tmp_path):
    journal = write_journal(tmp_path / "c" / "messages.jsonl",
                            [{**GOOD_PARENT, "kind": "review_request"}])
    assert_rejected(run_guard(dict(GOOD_CALL), {"AI_ROOM_CHANNEL_LOG": journal}), "P8")


# --- [data]: the live journal is never written to ----------------------------

def test_live_journal_untouched_by_this_suite():
    """Every fixture journal lives under tmp_path. If any arm resolved to the
    real channel, this digest would move."""
    if not LIVE_JOURNAL.exists():
        pytest.skip("no live journal on this machine")
    before = hashlib.sha256(LIVE_JOURNAL.read_bytes()).hexdigest()
    run_guard(dict(GOOD_CALL), {})  # C4 reject; must not create or write anything
    after = hashlib.sha256(LIVE_JOURNAL.read_bytes()).hexdigest()
    assert before == after, "the suite must never write to the live journal"
