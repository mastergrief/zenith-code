#!/usr/bin/env python3
"""Fixture tests for at_gabe_askuserquestion_gate.py.

CPU-static / no-loop: unit-checks the negated-mention guard and invokes the
hook via subprocess asserting exit codes (0 allow / 2 block). The focus is the
negation/definitional guard added so a RULE statement about not addressing gabe
("workers never @gabe directly") does not trip the textual matcher, while an
affirmative ask ("@gabe decide X") still gates.
Run: python3 .claude/hooks/test_at_gabe_askuserquestion_gate.py
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HOOK = Path(__file__).with_name("at_gabe_askuserquestion_gate.py")

_spec = importlib.util.spec_from_file_location("at_gabe_gate", HOOK)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def run(payload: dict) -> int:
    """Invoke the hook with an empty channel-log + no transcript (so no
    AskUserQuestion is provable) and return the exit code."""
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as fh:
        log_path = fh.name
    proc = subprocess.run(
        [sys.executable, str(HOOK)],
        input=json.dumps(payload),
        capture_output=True, text=True,
        env={"AI_ROOM_CHANNEL_LOG": log_path, "PATH": "/usr/bin:/bin"},
    )
    Path(log_path).unlink(missing_ok=True)
    return proc.returncode


def post(body: str, to="codex", kind="msg") -> dict:
    # No transcript_path -> captured=False; to != gabe -> structural_ask False.
    return {"tool_name": "mcp__ai-room__ai_room_post",
            "tool_input": {"kind": kind, "to": to, "body": body}}


# --- unit: strip_negated_at_gabe blanks definitional mentions, keeps asks ---
UNIT = [
    ("workers never @gabe directly", False),
    ("codex must not @gabe for decisions", False),
    ("don't @gabe directly", False),
    ("@gabe please pick option A or B", True),
    ("relaying to @gabe: locked answer is A", True),
    # Mixed-clause adversarial cases (co_lead gate-2 r1): a negation in an
    # UNRELATED clause must NOT suppress a genuine ask after the boundary.
    ("Do not delay; @gabe should we choose A or B?", True),
    ("I cannot decide — @gabe please pick A.", True),
    ("no rush, @gabe pick one", True),
    ("never mind that. @gabe which option?", True),
    ("avoid drift! @gabe decide now", True),
    # Same-clause uncertainty adversarial cases (co_lead gate-2 r2): generic
    # negation words in the SAME clause must not swallow a genuine ask —
    # only a prohibition binding directly to @gabe strips.
    ("I'm not sure @gabe which option should we use?", True),
    ("I cannot decide @gabe please pick A", True),
    ("No idea @gabe can you choose?", True),
    # Negated modal QUESTIONS with @gabe as subject (co_lead gate-2 r3):
    # genuine asks — must NOT strip.
    ("Shouldn't @gabe choose the safer option?", True),
    ("Won't @gabe decide between A and B?", True),
    ("Why should not @gabe pick A?", True),
    ("Why cannot @gabe decide?", True),
    # Direct prohibition of the addressing act still stripped:
    # imperatives (no actor needed) + actor+modal rule statements.
    ("do not @gabe for routine receipts", False),
    ("workers cannot @gabe for durable decisions", False),
    ("refrain from @gabe on routine receipts", False),
]

# --- e2e: exit code via subprocess ---
CASES = [
    # Definitional/negated mentions with no capture -> ALLOW (0).
    ("neg_never", 0, post("workers never @gabe directly")),
    ("neg_do_not", 0, post("do not @gabe; bubble to claude")),
    ("neg_must_not", 0, post("codex must not @gabe for durable decisions")),
    # Affirmative textual ask, no relay signature, no capture -> BLOCK (2).
    ("affirmative_ask", 2, post("@gabe should we pick A or B?")),
    # Mixed-clause adversarial: unrelated negation must NOT bypass -> BLOCK (2).
    ("mixed_semicolon", 2, post("Do not delay; @gabe should we choose A or B?")),
    ("mixed_emdash", 2, post("I cannot decide — @gabe please pick A.")),
    # Same-clause uncertainty adversarial (r2): must NOT bypass -> BLOCK (2).
    ("sameclause_notsure", 2, post("I'm not sure @gabe which option should we use?")),
    ("sameclause_cannot", 2, post("I cannot decide @gabe please pick A")),
    ("sameclause_noidea", 2, post("No idea @gabe can you choose?")),
    # Negated modal questions, @gabe as subject (r3): must NOT bypass -> BLOCK (2).
    ("modalq_shouldnt", 2, post("Shouldn't @gabe choose the safer option?")),
    ("modalq_wont", 2, post("Won't @gabe decide between A and B?")),
    ("modalq_why_shouldnot", 2, post("Why should not @gabe pick A?")),
    ("modalq_why_cannot", 2, post("Why cannot @gabe decide?")),
    # Affirmative but carries a relay-source signature -> ALLOW (0).
    ("affirmative_relay", 0,
     post("@gabe decision relayed — locked answer captured via AskUserQuestion")),
    # ack kind is always skipped -> ALLOW (0).
    ("ack_skip", 0, post("@gabe?", kind="ack")),
]


def main() -> int:
    failures = 0
    for body, expect_ask in UNIT:
        stripped = _mod.strip_negated_at_gabe(_mod.strip_quoted_segments(body))
        got = bool(_mod.AT_GABE_RE.search(stripped))
        ok = got == expect_ask
        print(f"[{'PASS' if ok else 'FAIL'}] unit  {body!r} -> ask={got}")
        failures += not ok
    for name, expected, payload in CASES:
        rc = run(payload)
        ok = rc == expected
        print(f"[{'PASS' if ok else 'FAIL'}] e2e   {name}: rc={rc} (want {expected})")
        failures += not ok
    print(f"\n{'ALL PASS' if not failures else f'{failures} FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
