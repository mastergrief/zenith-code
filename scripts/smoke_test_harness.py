#!/usr/bin/env python3
"""Zenith harness end-to-end smoke test.

Drives the live `zenith` binary via stdin to verify command handling, config
loading, sessions, tool calls, and the streaming display fix from c11232a.

Run from anywhere:
    python3 scripts/smoke_test_harness.py            # all tests
    python3 scripts/smoke_test_harness.py --quick    # skip model invocations

Each test uses a fresh temp cwd so .zenith_sessions stays out of the repo.
Model tests require llama-server on :8080 (auto-detected, skipped if absent).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
ZENITH = REPO_ROOT / "bin" / "zenith"
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def llama_running() -> bool:
    try:
        with urllib.request.urlopen("http://localhost:8080/health", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def run_harness(
    prompts: list[str],
    cwd: str,
    env: dict[str, str] | None = None,
    timeout: int = 30,
) -> tuple[int, str, str]:
    """Drive the harness via stdin. Returns (rc, stdout_clean, stderr_clean).

    Always appends /exit. Output has ANSI codes stripped for assertions.
    """
    stdin_data = "\n".join(prompts) + "\n/exit\n"
    proc_env = dict(os.environ)
    if env:
        proc_env.update(env)
    result = subprocess.run(
        [str(ZENITH)],
        input=stdin_data,
        capture_output=True,
        text=True,
        cwd=cwd,
        env=proc_env,
        timeout=timeout,
    )
    return result.returncode, strip_ansi(result.stdout), strip_ansi(result.stderr)


# ── Test definitions ───────────────────────────────────────────────


def test_command_battery(cwd: str) -> None:
    """Run many slash commands in one shot to amortize startup overhead."""
    cmds = [
        "/help",
        "/agents",
        "/backend",
        "/effort",
        "/distill status",
        "/sessions",
        "/cd",
        "/spawn tester quality assurance specialist",
        "/agents",
        "/switch tester",
        "/agents",
        "/effort low",
        "/effort",
        "/reset",
        "/nonsense",
    ]
    rc, out, _ = run_harness(cmds, cwd=cwd)
    assert rc == 0, f"non-zero exit: {rc}"

    assert "ZENITH CODE" in out and "Multi-Agent Harness" in out, "missing /help banner"
    assert "coder" in out and "reviewer" in out and "planner" in out, "missing default agents"
    assert "llamacpp" in out or "ollama" in out, "missing /backend output"
    assert "max_tokens" in out, "missing /effort output"
    assert "specialist-orchestrator" in out, "missing /distill status"
    assert "Saved sessions" in out or "No saved sessions" in out, "missing /sessions output"
    assert "Spawned agent" in out and "tester" in out, "missing /spawn output"
    assert "Switched to" in out, "missing /switch output"
    assert "Effort set to" in out and "low" in out, "missing /effort low confirmation"
    assert "All agent histories cleared" in out, "missing /reset confirmation"
    assert "Unknown command" in out, "missing /nonsense rejection"


def test_config_env_override(cwd: str) -> None:
    """ZENITH_EFFORT env var should set startup effort."""
    rc, out, _ = run_harness(
        ["/effort"],
        cwd=cwd,
        env={"ZENITH_EFFORT": "max"},
    )
    assert rc == 0
    # Banner shows "Effort: max" because effort != medium
    assert "Effort: max" in out, "ZENITH_EFFORT env var was ignored"


def test_config_zenithrc(cwd: str) -> None:
    """A .zenithrc file in cwd should be picked up at startup."""
    (Path(cwd) / ".zenithrc").write_text('{"effort": "low", "ctx_size": 32768}\n')
    rc, out, _ = run_harness(["/effort"], cwd=cwd)
    assert rc == 0
    assert "Effort: low" in out, ".zenithrc effort=low was ignored"


def test_config_cli_overrides_env(cwd: str) -> None:
    """CLI --effort should override the env var."""
    # Pass --effort via ZENITH arguments by appending to bin/zenith call.
    # bin/zenith forwards "$@" to harness, so we add --effort here.
    proc_env = dict(os.environ)
    proc_env["ZENITH_EFFORT"] = "low"
    result = subprocess.run(
        [str(ZENITH), "--effort", "max"],
        input="/effort\n/exit\n",
        capture_output=True,
        text=True,
        cwd=cwd,
        env=proc_env,
        timeout=30,
    )
    out = strip_ansi(result.stdout)
    assert result.returncode == 0
    assert "Effort: max" in out, "--effort CLI flag did not override ZENITH_EFFORT env"


def test_session_save_and_list(cwd: str) -> None:
    """Save creates a file; sessions lists it."""
    rc, out, _ = run_harness(["/save", "/sessions"], cwd=cwd)
    assert rc == 0
    assert "Session saved to" in out, "missing /save confirmation"
    assert "Saved sessions" in out, "missing /sessions listing after save"
    sessions_dir = Path(cwd) / ".zenith_sessions"
    assert sessions_dir.exists(), ".zenith_sessions dir not created"
    files = list(sessions_dir.glob("*.json"))
    assert files, "no session files written"


def test_chat_no_double_print(cwd: str) -> None:
    """Verify the c11232a streaming-display fix is still working.

    Sends a chat prompt, then compares the canonical response (from the
    saved session JSON) against the displayed stdout. The response must
    appear exactly once in the displayed output, not twice.
    """
    rc, out, _ = run_harness(
        ["what is 2+2 and why? answer in one short sentence."],
        cwd=cwd,
        env={"ZENITH_EFFORT": "low"},
        timeout=90,
    )
    assert rc == 0, f"non-zero exit: {rc}"
    assert "auto-saved" in out, "no auto-save line — chat may not have completed"

    sessions_dir = Path(cwd) / ".zenith_sessions"
    files = sorted(sessions_dir.glob("*.json"))
    assert files, "no session file written after chat"
    data = json.loads(files[-1].read_text())
    responses = [
        m.get("content", "")
        for m in data["history"]
        if m.get("role") == "assistant" and m.get("content")
    ]
    assert responses, "no assistant response in saved session"
    response = responses[-1]
    assert response, "assistant response is empty"

    # Use a fingerprint long enough to disambiguate from prompt echo.
    # 40 chars covers most short answers without overlapping prompt text.
    fingerprint = response.strip()[:40]
    if len(fingerprint) >= 20:
        count = out.count(fingerprint)
        assert count == 1, (
            f"response appears {count}x in displayed output (expected 1) — "
            f"the c11232a streaming display fix may have regressed. "
            f"Fingerprint: {fingerprint!r}"
        )


def test_chat_tool_call(cwd: str) -> None:
    """Ask a question that should trigger a tool call to read a known file."""
    (Path(cwd) / "marker.txt").write_text("THE_SECRET_PHRASE_IS_PURPLE_OWL_42\n")
    rc, out, _ = run_harness(
        ["read marker.txt and tell me what the secret phrase is"],
        cwd=cwd,
        env={"ZENITH_EFFORT": "low"},
        timeout=120,
    )
    assert rc == 0
    assert "[tool]" in out, "no tool call observed in output"
    assert "PURPLE_OWL" in out, "model did not retrieve marker file content"


def test_swap_no_arg_shows_current_and_available(cwd: str) -> None:
    """`/swap` with no arg should print the loaded model and list available GGUFs."""
    rc, out, _ = run_harness(["/swap"], cwd=cwd)
    assert rc == 0, f"non-zero exit: {rc}"
    assert "Current:" in out, "missing 'Current:' line for loaded model"
    assert "Available in ~/models/:" in out, "missing 'Available in ~/models/' listing"
    assert ".gguf" in out or "loaded" in out, "no GGUF entries shown in available list"


# ── Runner ─────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Zenith harness smoke tests")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip model-invocation tests (run command/config tests only)",
    )
    args = parser.parse_args()

    if not ZENITH.exists():
        print(f"FAIL: launcher not found at {ZENITH}")
        sys.exit(2)

    has_llama = llama_running()
    if not has_llama:
        print("WARN: llama-server not running on :8080 — model tests will be skipped\n")

    fast_tests: list[tuple[str, Callable[[str], None]]] = [
        ("command battery (15 slash commands)", test_command_battery),
        ("config: ZENITH_EFFORT env var", test_config_env_override),
        ("config: .zenithrc file pickup", test_config_zenithrc),
        ("config: CLI flag overrides env", test_config_cli_overrides_env),
        ("session: /save + /sessions", test_session_save_and_list),
    ]
    slow_tests: list[tuple[str, Callable[[str], None]]] = [
        ("chat: streaming display single-print", test_chat_no_double_print),
        ("chat: tool call (read_file)", test_chat_tool_call),
        ("/swap: no-arg shows current + available", test_swap_no_arg_shows_current_and_available),
    ]

    tests = list(fast_tests)
    if not args.quick and has_llama:
        tests.extend(slow_tests)

    passed = 0
    failed = 0

    for name, fn in tests:
        tmpdir = tempfile.mkdtemp(prefix="zenith_smoke_")
        try:
            fn(tmpdir)
            print(f"  PASS  {name}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {name}")
            print(f"        {e}")
            failed += 1
        except subprocess.TimeoutExpired:
            print(f"  FAIL  {name}: timeout")
            failed += 1
        except Exception as e:
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
            failed += 1
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    print(f"\n{passed} passed, {failed} failed, {len(tests)} total")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
