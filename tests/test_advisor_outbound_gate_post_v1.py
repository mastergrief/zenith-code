"""Initiation-path battery for .claude/hooks/advisor_outbound_gate.py.

This file covers the initiation path's P10-P13 plus one reply positive
control. The reply-path battery remains tests/test_advisor_outbound_gate_v1.py.

Every arm drives the real hook as a SUBPROCESS with a real PreToolUse
payload on stdin. Importing the module and calling `check_initiation`
would test a reimplementation of the invocation path rather than the
path that runs in production.

Each negative asserts the PREDICATE that fired, not merely rc==2.
WAKE_VERIFIED is not a hook predicate and is not asserted here.

Run: PYTHONPATH=. python3 -m pytest tests/test_advisor_outbound_gate_post_v1.py -q
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import tempfile

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
GUARD = REPO_ROOT / ".claude" / "hooks" / "advisor_outbound_gate.py"
POST_TOOL = "mcp__ai-room__ai_room_post"
REPLY_TOOL = "mcp__ai-room__ai_room_reply"

PARENT_ID = "1700000000000-aaaaaaaa"
GOOD_PARENT = {"id": PARENT_ID, "from": "claude", "to": "advisor", "kind": "msg",
               "body": "what decompositions would you consider here?"}
GOOD_REPLY = {"body": "three alternatives, with predicted failure modes", "reply_to": PARENT_ID}
GOOD_POST = {"body": "ruling: renew the licensed route unchanged", "to": "claude"}


def write_journal(path: pathlib.Path, records) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


def run_guard(tool_input, env_overrides, *, tool_name=POST_TOOL, omit_tool_name=False):
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
        [str(GUARD)],
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


# --- ADMITTED ------------------------------------------------------------------

def test_admitted_conforming_reply(channel):
    env, _ = channel
    proc = run_guard(dict(GOOD_REPLY), env, tool_name=REPLY_TOOL)
    assert proc.returncode == 0, f"conforming reply must ALLOW; stderr={proc.stderr!r}"


def test_admitted_conforming_initiated_post_to_claude():
    """Initiation needs no journal — P4-P8 do not apply."""
    proc = run_guard(dict(GOOD_POST), {})
    assert proc.returncode == 0, f"conforming post to claude must ALLOW; stderr={proc.stderr!r}"


# --- P10: initiation key allowlist ---------------------------------------------

def test_rejected_non_allowlisted_key():
    assert_rejected(run_guard({**GOOD_POST, "reply_to": PARENT_ID}, {}), "P10")


# --- P11: scalar non-empty required keys body and to ---------------------------

def test_rejected_broadcast_absent_to():
    assert_rejected(run_guard({"body": GOOD_POST["body"]}, {}), "P11")


def test_rejected_broadcast_to_as_list():
    assert_rejected(run_guard({"body": GOOD_POST["body"], "to": ["claude"]}, {}), "P11")


def test_rejected_missing_body():
    assert_rejected(run_guard({"to": "claude"}, {}), "P11")


def test_rejected_empty_body():
    assert_rejected(run_guard({"body": "", "to": "claude"}, {}), "P11")


# --- P12: addressee exactly claude ---------------------------------------------

def test_rejected_another_handle():
    assert_rejected(run_guard({**GOOD_POST, "to": "codex"}, {}), "P12")


# --- P13: initiation kind ------------------------------------------------------

def test_admitted_kind_msg():
    proc = run_guard({**GOOD_POST, "kind": "msg"}, {})
    assert proc.returncode == 0, f"kind=msg must ALLOW; stderr={proc.stderr!r}"


def test_admitted_kind_design_proposal():
    proc = run_guard({**GOOD_POST, "kind": "design_proposal"}, {})
    assert proc.returncode == 0, f"kind=design_proposal must ALLOW; stderr={proc.stderr!r}"


def test_rejected_disallowed_kind():
    assert_rejected(run_guard({**GOOD_POST, "kind": "task_dispatch"}, {}), "P13")
