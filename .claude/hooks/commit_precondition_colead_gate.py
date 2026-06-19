#!/usr/bin/env python3
"""
PreToolUse hook on Bash — commit-precondition co_lead gate (routing redesign LANE 1).

Once a Bash command is recognized as `git commit`, fail-CLOSED unless the
ai-room channel log shows a fresh codex_co_lead validation/diff PASS that echoes
the staged DIFF_DIGEST matching `git diff --cached` at commit time.

`git push` is NOT co_lead-gated; only force-push patterns are blocked here
(Claude/Gabe retain push authority after reviewed commit).

Fail-OPEN only: malformed stdin before command recognition, or non-commit/push
commands. Missing/unreadable channel log AFTER commit recognition → BLOCK.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from typing import Any

DEFAULT_CHANNEL_LOG = "/home/gabe/.ai-room/channels/claw-code/messages.jsonl"

GIT_COMMIT_RE = re.compile(r"(?<![\w/])git\b[^;\n|&]*\bcommit\b", re.IGNORECASE)
GIT_PUSH_RE = re.compile(r"(?<![\w/])git\b[^;\n|&]*\bpush\b", re.IGNORECASE)
FORCE_PUSH_RE = re.compile(r"(?:--force(?:-with-lease)?|\s-f\b)")
PLUS_REFSPEC_RE = re.compile(r"(?<![\w])\+[\w./:-]+")

DIFF_DIGEST_RE = re.compile(r"(?im)^\s*DIFF_DIGEST\s*:\s*([0-9a-f]{64})\s*$")
COLEAD_GATE_OVERRIDE_RE = re.compile(
    r"(?im)^\s*CO_LEAD_GATE_OVERRIDE\s*:\s*(.+?)\s*$"
)
TASK_ID_RE = re.compile(r"\b(\d{13}-[0-9a-f]{6,8})\b")

COLEAD_PASS_MARKERS = (
    re.compile(r"(?im)co_lead\s+gate-2\s+PASS"),
    re.compile(r"(?im)validation/diff\s+(?:review\s*:\s*)?PASS"),
    re.compile(r"(?im)\bgate-2\s+PASS\b"),
)
COLEAD_DEFERRAL_MARKERS = (
    re.compile(r"(?im)\bno\s+(?:co_lead\s+)?approval\b"),
    re.compile(r"(?im)\bno\s+dual-accept\b"),
    re.compile(r"(?im)\bdeferred?\s+until\b"),
    re.compile(r"(?im)\bholding\s+(?:for|until)\b"),
    re.compile(r"(?im)\bvisibility\s+only\b"),
)
COLEAD_BLOCK_MARKERS = (
    re.compile(r"(?im)co_lead\s+gate-2\s+(?:BLOCK|REVISE)"),
    re.compile(r"(?im)\bgate-2\s+(?:BLOCK|REVISE)\b"),
    re.compile(r"(?im)validation/diff\s+.*\b(?:BLOCK|REVISE)\b"),
)

MIN_COLEAD_GATE_OVERRIDE_REASON_CHARS = 10


def fail_open(reason: str) -> int:
    if os.environ.get("COMMIT_PRECONDITION_GATE_DEBUG"):
        print(
            f"[commit_precondition_colead_gate] fail-open: {reason}",
            file=sys.stderr,
        )
    return 0


def _parse_ts(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        v = float(value)
        return v / (1000.0 if v > 1e12 else 1.0)
    if isinstance(value, str):
        try:
            s = value.strip()
            if s.endswith("Z"):
                s = s[:-1] + "+00:00"
            from datetime import datetime, timezone

            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except Exception:
            try:
                v = float(value)
                return v / (1000.0 if v > 1e12 else 1.0)
            except Exception:
                return None
    return None


def _read_records(path: str) -> list[dict[str, Any]] | None:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
    except FileNotFoundError:
        return None
    except OSError:
        return None
    records: list[dict[str, Any]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            return None
        if isinstance(rec, dict):
            records.append(rec)
    return records


def _body(rec: dict[str, Any]) -> str:
    body = rec.get("body")
    return body if isinstance(body, str) else ""


def _staged_digest() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "diff", "--cached"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return hashlib.sha256(proc.stdout.encode("utf-8")).hexdigest()


def _extract_digest(body: str) -> str | None:
    m = DIFF_DIGEST_RE.search(body)
    return m.group(1).lower() if m else None


def _is_worker_receipt(rec: dict[str, Any]) -> bool:
    frm = str(rec.get("from", ""))
    if frm in {"claude", "codex_co_lead", "gabe", "watchdog"}:
        return False
    kind = str(rec.get("kind", ""))
    body = _body(rec)
    if kind == "validation_receipt":
        return True
    return any(
        marker in body
        for marker in (
            "VALIDATION RECEIPT",
            "VALIDATION_RECEIPT",
            "IMPLEMENTATION RECEIPT",
            "TERMINAL RECEIPT",
        )
    )


def _is_claude_freeze(rec: dict[str, Any]) -> bool:
    if rec.get("from") != "claude":
        return False
    body = _body(rec)
    if _extract_digest(body) is None:
        return False
    return any(
        tok in body
        for tok in (
            "gate-1 freeze",
            "FREEZE LOCKED",
            "frozen handoff",
            "validation/diff handoff",
            "review_request",
        )
    ) or str(rec.get("kind", "")) in {"review_request", "msg"}


def _colead_verdict(body: str) -> str:
    for pat in COLEAD_BLOCK_MARKERS:
        if pat.search(body):
            return "block"
    for pat in COLEAD_DEFERRAL_MARKERS:
        if pat.search(body):
            return "unknown"
    for pat in COLEAD_PASS_MARKERS:
        if pat.search(body):
            return "pass"
    return "unknown"


def _has_force_plus_refspec(command: str) -> bool:
    if not GIT_PUSH_RE.search(command):
        return False
    return bool(PLUS_REFSPEC_RE.search(command))


def _same_thread(rec: dict[str, Any], anchor_ids: set[str]) -> bool:
    if not anchor_ids:
        return True
    rid = str(rec.get("id", ""))
    reply_to = str(rec.get("reply_to", ""))
    body = _body(rec)
    if rid in anchor_ids or reply_to in anchor_ids:
        return True
    return any(aid in body for aid in anchor_ids)


def _find_fresh_colead_pass(
    records: list[dict[str, Any]],
    staged_digest: str,
) -> tuple[bool, str]:
    freeze_ts: float | None = None
    freeze_ids: set[str] = set()
    task_ids: set[str] = set()

    for rec in records:
        if not _is_claude_freeze(rec):
            continue
        digest = _extract_digest(_body(rec))
        if digest != staged_digest:
            continue
        ts = _parse_ts(rec.get("ts"))
        if freeze_ts is None or (ts is not None and ts >= freeze_ts):
            freeze_ts = ts
            freeze_ids = set()
            rid = str(rec.get("id", ""))
            if rid:
                freeze_ids.add(rid)
            reply_to = str(rec.get("reply_to", ""))
            if reply_to:
                freeze_ids.add(reply_to)
            task_ids = set()
            m = TASK_ID_RE.search(_body(rec))
            if m:
                task_ids.add(m.group(1))

    if freeze_ts is None:
        return False, "no claude freeze/handoff carrying matching DIFF_DIGEST"

    anchor_ids = set(freeze_ids)
    worker_ts: float | None = None
    for rec in records:
        if not _is_worker_receipt(rec):
            continue
        body = _body(rec)
        if not (
            _same_thread(rec, anchor_ids)
            or (task_ids and any(tid in body for tid in task_ids))
        ):
            continue
        ts = _parse_ts(rec.get("ts"))
        if ts is not None and (worker_ts is None or ts > worker_ts):
            worker_ts = ts
            rid = str(rec.get("id", ""))
            if rid:
                anchor_ids.add(rid)

    if worker_ts is not None and freeze_ts <= worker_ts:
        return False, "claude freeze must be after scoped worker receipt on-thread"

    best_pass_ts: float | None = None
    for rec in records:
        if rec.get("from") != "codex_co_lead":
            continue
        body = _body(rec)
        digest = _extract_digest(body)
        if digest != staged_digest:
            continue
        verdict = _colead_verdict(body)
        if verdict != "pass":
            continue
        ts = _parse_ts(rec.get("ts"))
        if ts is None or ts <= freeze_ts:
            continue
        if worker_ts is not None and ts <= worker_ts:
            continue
        if not (
            _same_thread(rec, anchor_ids)
            or (task_ids and any(tid in body for tid in task_ids))
        ):
            continue
        if best_pass_ts is None or ts >= best_pass_ts:
            best_pass_ts = ts

    if best_pass_ts is None:
        return False, "no codex_co_lead validation/diff PASS echoing staged DIFF_DIGEST on-thread after freeze"
    return True, "fresh co_lead PASS matches staged DIFF_DIGEST"


def _bash_command(payload: dict[str, Any]) -> str:
    tool_input = payload.get("tool_input") or {}
    if isinstance(tool_input, dict):
        cmd = tool_input.get("command")
        if isinstance(cmd, str):
            return cmd
    cmd = payload.get("command")
    return cmd if isinstance(cmd, str) else ""


def main() -> int:
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            return fail_open("empty stdin")
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        return fail_open(f"json decode failed: {exc}")
    except Exception as exc:  # noqa: BLE001
        return fail_open(f"stdin read failed: {exc}")

    command = _bash_command(payload)
    if not command.strip():
        return fail_open("empty command")

    if COLEAD_GATE_OVERRIDE_RE.search(command):
        m = COLEAD_GATE_OVERRIDE_RE.search(command)
        reason = (m.group(1).strip() if m else "")
        if len(reason) < MIN_COLEAD_GATE_OVERRIDE_REASON_CHARS:
            msg = (
                "BLOCKED [commit_precondition_colead_gate] CO_LEAD_GATE_OVERRIDE "
                f"reason too trivial ({len(reason)} chars; need "
                f">={MIN_COLEAD_GATE_OVERRIDE_REASON_CHARS}) — explicit persisted "
                "narrow override required."
            )
            print(msg, file=sys.stderr)
            return 2
        return 0

    is_commit = bool(GIT_COMMIT_RE.search(command))
    is_push = bool(GIT_PUSH_RE.search(command))
    if not is_commit and not is_push:
        return 0

    if is_push and (
        FORCE_PUSH_RE.search(command) or _has_force_plus_refspec(command)
    ):
        msg = (
            "BLOCKED [commit_precondition_colead_gate] force push is forbidden "
            "via this hook (use non-force fast-forward after reviewed commit)."
        )
        print(msg, file=sys.stderr)
        return 2

    if is_push and not is_commit:
        return 0

    log_path = (
        os.environ.get("AI_ROOM_CHANNEL_LOG")
        or os.environ.get("AI_ROOM_CHANNEL_LOG_PATH")
        or DEFAULT_CHANNEL_LOG
    )
    records = _read_records(log_path)
    if records is None:
        msg = (
            "BLOCKED [commit_precondition_colead_gate] git commit recognized but "
            f"channel log missing/unreadable at {log_path!r} — no proven co_lead "
            "validation/diff gate."
        )
        print(msg, file=sys.stderr)
        return 2

    staged_digest = _staged_digest()
    if staged_digest is None:
        msg = (
            "BLOCKED [commit_precondition_colead_gate] git commit recognized but "
            "could not compute staged diff digest (git diff --cached failed)."
        )
        print(msg, file=sys.stderr)
        return 2

    ok, reason = _find_fresh_colead_pass(records, staged_digest)
    if ok:
        return 0

    msg_lines = [
        "BLOCKED [commit_precondition_colead_gate] git commit without fresh co_lead validation/diff gate:",
        f"  staged DIFF_DIGEST: {staged_digest}",
        f"  reason: {reason}",
        "",
        "Required chain: worker receipt → claude gate-1 freeze/handoff with DIFF_DIGEST",
        "→ codex_co_lead gate-2 PASS echoing the same DIFF_DIGEST on-thread.",
        "git push is not co_lead-gated; only reviewed commits pass this hook.",
    ]
    print("\n".join(msg_lines), file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
