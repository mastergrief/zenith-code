#!/usr/bin/env python3
"""Fixture tests for commit_precondition_colead_gate.py."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HOOK = Path(__file__).with_name("commit_precondition_colead_gate.py")

DIGEST = "a" * 64
OTHER_DIGEST = "b" * 64
TASK = "1781862264540-5c174b3f"
WORKER_RCPT = "1781864000000-worker01"
FREEZE_ID = "1781864100000-freeze01"
PASS_ID = "1781864200000-pass01"


def rec(frm: str, body: str, ts: str, mid: str, reply_to: str = "") -> str:
    obj = {"ts": ts, "id": mid, "from": frm, "kind": "validation_receipt", "body": body}
    if reply_to:
        obj["reply_to"] = reply_to
    return json.dumps(obj)


def worker_body() -> str:
    return f"VALIDATION RECEIPT — slice task {TASK}"


def freeze_body(digest: str = DIGEST) -> str:
    return (
        f"claude gate-1 freeze handoff task {TASK}\n"
        f"DIFF_DIGEST: {digest}\n"
        f"reply_to worker receipt {WORKER_RCPT}"
    )


def pass_body(digest: str = DIGEST) -> str:
    return (
        f"co_lead gate-2 PASS validation/diff\n"
        f"DIFF_DIGEST: {digest}\n"
        f"task {TASK} threaded to {FREEZE_ID}"
    )


def fresh_chain(digest: str = DIGEST) -> list[str]:
    return [
        rec("codex", worker_body(), "2026-06-19T10:00:00Z", WORKER_RCPT),
        rec("claude", freeze_body(digest), "2026-06-19T10:05:00Z", FREEZE_ID, WORKER_RCPT),
        rec("codex_co_lead", pass_body(digest), "2026-06-19T10:10:00Z", PASS_ID, FREEZE_ID),
    ]


def run(command: str, log_lines: list[str] | None, cwd: Path | None = None) -> int:
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as fh:
        path = fh.name
        if log_lines is not None:
            fh.write("\n".join(log_lines) + ("\n" if log_lines else ""))
    env = {
        "PATH": "/usr/bin:/bin",
        "AI_ROOM_CHANNEL_LOG": path,
    }
    payload = {"tool_input": {"command": command}}
    proc = subprocess.run(
        [sys.executable, str(HOOK)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        env=env,
        cwd=str(cwd) if cwd else None,
    )
    Path(path).unlink(missing_ok=True)
    return proc.returncode


def setup_git_repo_with_staged(content: str = "hello\n") -> Path:
    tmp = Path(tempfile.mkdtemp())
    subprocess.run(["git", "init"], cwd=tmp, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=tmp,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "test"],
        cwd=tmp,
        check=True,
        capture_output=True,
    )
    f = tmp / "file.txt"
    f.write_text(content, encoding="utf-8")
    subprocess.run(["git", "add", "file.txt"], cwd=tmp, check=True, capture_output=True)
    return tmp


def staged_digest(repo: Path) -> str:
    proc = subprocess.run(
        ["git", "diff", "--cached"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    return hashlib.sha256(proc.stdout.encode("utf-8")).hexdigest()


CASES = []


def case(name: str, expected: int, command: str, log_lines: list[str] | None, repo: Path | None):
    CASES.append((name, expected, command, log_lines, repo))


repo = setup_git_repo_with_staged()
digest = staged_digest(repo)

case("non_git_allow", 0, "echo hello", None, None)
case("commit_no_log_block", 2, "git commit -m test", None, repo)
case("commit_unreadable_log_block", 2, "git commit -m test", ["{not json}"], repo)
case("commit_fresh_pass_allow", 0, "git commit -m test", fresh_chain(digest), repo)
case("commit_stale_digest_block", 2, "git commit -m test", fresh_chain(OTHER_DIGEST), repo)
case("commit_block_revise_block", 2, "git commit -m test", [
    rec("codex", worker_body(), "2026-06-19T10:00:00Z", WORKER_RCPT),
    rec("claude", freeze_body(digest), "2026-06-19T10:05:00Z", FREEZE_ID, WORKER_RCPT),
    rec(
        "codex_co_lead",
        f"co_lead gate-2 REVISE\nDIFF_DIGEST: {digest}",
        "2026-06-19T10:10:00Z",
        PASS_ID,
        FREEZE_ID,
    ),
], repo)
case("commit_pre_receipt_pass_block", 2, "git commit -m test", [
    rec("codex_co_lead", pass_body(digest), "2026-06-19T09:00:00Z", PASS_ID),
    rec("codex", worker_body(), "2026-06-19T10:00:00Z", WORKER_RCPT),
    rec("claude", freeze_body(digest), "2026-06-19T10:05:00Z", FREEZE_ID, WORKER_RCPT),
], repo)
case("push_allow", 0, "git push origin feature/test", fresh_chain(digest), repo)
case("push_force_block", 2, "git push --force origin feature/test", fresh_chain(digest), repo)
case("push_plus_refspec_block", 2, "git push origin +feature/test", fresh_chain(digest), repo)
case("commit_deferral_echo_block", 2, "git commit -m test", [
    rec("codex", worker_body(), "2026-06-19T10:00:00Z", WORKER_RCPT),
    rec("claude", freeze_body(digest), "2026-06-19T10:05:00Z", FREEZE_ID, WORKER_RCPT),
    rec(
        "codex_co_lead",
        f"Receipt noted. DIFF_DIGEST: {digest}\n"
        f"dual-accept pending — no co_lead approval yet; holding until Claude gate-1",
        "2026-06-19T10:10:00Z",
        PASS_ID,
        FREEZE_ID,
    ),
], repo)
case("commit_unrelated_later_worker_allow", 0, "git commit -m test", fresh_chain(digest) + [
    rec(
        "codex",
        "VALIDATION RECEIPT — unrelated task 1781999999999-deadbeef",
        "2026-06-19T11:00:00Z",
        "1781864999999-unrel01",
    ),
], repo)
case("override_trivial_block", 2, "git commit -m test\nCO_LEAD_GATE_OVERRIDE: x", None, repo)
case("override_allow", 0, "git commit -m test\nCO_LEAD_GATE_OVERRIDE: docs-only emergency", None, repo)


def main() -> int:
    failed = 0
    for name, expected, command, log_lines, repo_path in CASES:
        rc = run(command, log_lines, repo_path)
        if rc != expected:
            print(f"FAIL {name}: expected exit {expected}, got {rc}")
            failed += 1
        else:
            print(f"PASS {name}")
    if failed:
        print(f"{failed}/{len(CASES)} failed")
        return 1
    print(f"ALL {len(CASES)} PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
