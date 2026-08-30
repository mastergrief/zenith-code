#!/usr/bin/env python3
"""Tests for enforce-monitor-on-bg-shell.sh — the Bash background/poll gate.

This hook is a BLOCKING PreToolUse gate, so it is graded on both directions:
an under-fire hides long-running work from Monitor, and an over-fire wedges
ordinary commands. The allow-side cases below are not padding — they are the
reason the deny-side regexes are written narrowly.

The hook exits 0 whether it allows or denies (a silent allow is "exit 0, no
output"), so every assertion reads `permissionDecision` out of the emitted
JSON rather than the exit code. Asserting on exit status alone would pass
against a hook that never ran.
"""
from __future__ import annotations

import json
import pathlib
import subprocess

import pytest

HOOK = pathlib.Path(__file__).with_name("enforce-monitor-on-bg-shell.sh")


def decide(command: str, run_in_background: bool = False) -> str | None:
    """Return the hook's permissionDecision, or None for a silent allow."""
    payload = json.dumps(
        {
            "tool_name": "Bash",
            "tool_input": {
                "command": command,
                "run_in_background": run_in_background,
            },
        }
    )
    proc = subprocess.run(
        ["bash", str(HOOK)],
        input=payload,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, f"hook exited {proc.returncode}: {proc.stderr}"
    out = proc.stdout.strip()
    if not out:
        return None
    return json.loads(out)["hookSpecificOutput"]["permissionDecision"]


# --- deny side: every pattern workflow.md:150 forbids -------------------------

DENY_CASES = [
    ("setsid ./run.sh", "setsid"),
    ("nohup ./run.sh", "nohup"),
    ("until [ -f /tmp/x ]; do sleep 5; done", "until-poll"),
    ("./run.sh & disown", "disown"),
    ("./long-train.sh > /tmp/t.log 2>&1 &", "trailing &"),
    ("while true; do echo hi; sleep 30; done", "while-poll"),
    ("while :; do check; sleep 10; done", "while-poll shorthand"),
    ("make build &", "trailing & after simple command"),
    ("cd /tmp; setsid ./x", "setsid after separator"),
]


@pytest.mark.parametrize("command,label", DENY_CASES)
def test_blocked(command: str, label: str) -> None:
    assert decide(command) == "deny", f"{label!r} should be blocked: {command!r}"


def test_run_in_background_flag_blocked() -> None:
    assert decide("echo hi", run_in_background=True) == "deny"


def test_denial_reason_names_the_hook() -> None:
    payload = json.dumps(
        {"tool_name": "Bash", "tool_input": {"command": "setsid ./x"}}
    )
    proc = subprocess.run(
        ["bash", str(HOOK)], input=payload, capture_output=True, text=True, timeout=30
    )
    reason = json.loads(proc.stdout)["hookSpecificOutput"]["permissionDecisionReason"]
    assert "enforce-monitor-on-bg-shell.sh" in reason


# --- allow side: the over-fire controls --------------------------------------
#
# Each of these contains a substring the deny regexes look for, and each must
# still pass. A regex change that trips one of these is a regression even if
# every deny case still fires.

ALLOW_CASES = [
    ("a && b", "&& is not backgrounding"),
    ("cmd > log 2>&1", "2>&1 is fd redirection"),
    ("cmd 2>&1 | tee log", "redirection through a pipe"),
    ("./run.sh &> combined.log", "&> is bash redirection"),
    ("grep 'while true' file", "loop keyword inside quotes"),
    ("rg 'nohup' docs/", "nohup as a search argument"),
    ("rg 'disown' docs/", "disown as a search argument"),
    ("echo \"run in background\"", "prose in a quoted string"),
    ("git log --oneline", "ordinary git"),
    ("ls -la /tmp", "ordinary listing"),
    ("sleep 5", "a bare sleep is not a poll loop"),
    ("for f in *.py; do echo $f; done", "for-loop without sleep"),
    ("python3 -m pytest test_x.py -q", "ordinary test run"),
]


@pytest.mark.parametrize("command,label", ALLOW_CASES)
def test_allowed(command: str, label: str) -> None:
    assert decide(command) is None, f"{label!r} must not be blocked: {command!r}"


# --- fail-open shape guards ---------------------------------------------------

def test_empty_input_allows() -> None:
    proc = subprocess.run(
        ["bash", str(HOOK)], input="", capture_output=True, text=True, timeout=30
    )
    assert proc.returncode == 0
    assert proc.stdout.strip() == ""


def test_malformed_json_allows() -> None:
    proc = subprocess.run(
        ["bash", str(HOOK)], input="{not json", capture_output=True, text=True, timeout=30
    )
    assert proc.returncode == 0
    assert proc.stdout.strip() == ""
