"""Regression tests: watchdog marker matching must be structural, not substring.

Bug (task 1781182645882-4d1dfc23): markers QUOTED inside ack/review prose
("ACK — 'IMPLEMENTING — Stage 3' received") substring-matched
HEARTBEAT_MARKERS, classifying plain acks as heartbeats. The worker's latest
gated post became a phantom heartbeat with no due field that later acks could
never clear, producing 3 false RECYCLE alarms.
"""
import importlib.util
import pathlib

import pytest

_HOOK = pathlib.Path(__file__).resolve().parents[1] / ".claude" / "hooks" / \
    "ai_room_heartbeat_watchdog.py"
_spec = importlib.util.spec_from_file_location("ai_room_heartbeat_watchdog", _HOOK)
wd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wd)


def _rec(frm, body, ts="2026-06-12T12:00:00Z"):
    return {"from": frm, "body": body, "ts": ts}


# --- heartbeat classification -------------------------------------------------

def test_real_heartbeat_line_start_bold():
    assert wd._is_heartbeat_body("**IMPLEMENTING — Stage 1** | gate `123`\nphase=edit")


def test_real_milestone_heartbeat_line_start():
    assert wd._is_heartbeat_body("MILESTONE HEARTBEAT — step 500 reached")


def test_real_due_field_assignment():
    assert wd._is_heartbeat_body("phase=edit next_heartbeat_due=2026-06-12T15:00:00Z")


def test_quoted_implementing_inline_is_not_heartbeat():
    body = ('**ACK** — your "IMPLEMENTING — Stage 3" status received; '
            "parked until the receipt lands.")
    assert not wd._is_heartbeat_body(body)


def test_blockquoted_heartbeat_is_not_heartbeat():
    body = "Reviewing this:\n> IMPLEMENTING — Stage 1 | gate `123`\nlooks fine."
    assert not wd._is_heartbeat_body(body)


def test_due_field_mentioned_without_assignment_is_not_heartbeat():
    body = "Reminder: include next_heartbeat_due in every milestone post."
    assert not wd._is_heartbeat_body(body)


def test_blockquoted_due_field_is_not_heartbeat():
    body = "> phase=edit next_heartbeat_due=2026-06-12T15:00:00Z\nquoted above."
    assert not wd._is_heartbeat_body(body)


# --- terminal classification --------------------------------------------------

def test_real_terminal_receipt_line_start():
    assert wd._is_terminal_body("**VALIDATION RECEIPT — Stage 1** | gate `123`")


def test_quoted_terminal_is_not_terminal():
    body = '**ACK** — the "VALIDATION RECEIPT — Stage 1" was reviewed upstream.'
    assert not wd._is_terminal_body(body)


def test_blockquoted_terminal_is_not_terminal():
    assert not wd._is_terminal_body("> VALIDATION RECEIPT — Stage 1\nquoted.")


def test_task_complete_token_still_matches_unquoted():
    assert wd._is_terminal_body("done — recorded via ai_room_task_complete just now")


# --- end-to-end: the false-RECYCLE scenario ------------------------------------

def test_ack_quoting_implementing_does_not_create_phantom_heartbeat():
    """Worker posts real terminal, then an ack QUOTING 'IMPLEMENTING'. Under
    substring matching the ack became the latest 'heartbeat' (phantom,
    unclearable). Structural matching must yield no active heartbeat."""
    records = [
        _rec("codex", "**IMPLEMENTING — Stage 1** | gate `1`\nphase=edit",
             "2026-06-12T10:00:00Z"),
        _rec("codex", "**VALIDATION RECEIPT — Stage 1** | gate `1`",
             "2026-06-12T10:10:00Z"),
        _rec("codex", '**ACK** — "IMPLEMENTING — Stage 2" plan noted; parked.',
             "2026-06-12T10:20:00Z"),
    ]
    assert wd.find_active_heartbeats(records) == []


def test_real_heartbeat_still_detected_end_to_end():
    records = [
        _rec("codex", "**IMPLEMENTING — Stage 2** | gate `2`\nphase=edit "
             "next_heartbeat_due=2026-06-12T15:00:00Z", "2026-06-12T11:00:00Z"),
    ]
    hbs = wd.find_active_heartbeats(records)
    assert len(hbs) == 1 and hbs[0]["worker"] == "codex"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
