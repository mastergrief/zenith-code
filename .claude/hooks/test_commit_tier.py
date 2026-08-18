#!/usr/bin/env python3
"""Tests for the tiered commit gate — CLAUDEX §"Commit and push gates".

LOW (docs / tests, nothing control-plane) commits under claude gate-1 alone;
everything else keeps the co_lead DIFF_DIGEST PASS. This WEAKENS an
authorization gate, so the tests are weighted toward proving it fails closed:
the HIGH cases are the ones that matter, and the integration pair below differs
only in which path is staged.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import subprocess

import pytest

HOOKS = pathlib.Path(__file__).parent
GATE = HOOKS / "commit_precondition_colead_gate.py"

_spec = importlib.util.spec_from_file_location(
    "_tier_classifier", HOOKS / "colead_commit_gate_classifier.py"
)
assert _spec and _spec.loader
classifier = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(classifier)


# --- pure allowlist ----------------------------------------------------------

LOW_SETS = [
    (["docs/guide.md"], "docs markdown"),
    (["README.md"], "root readme"),
    (["rust/README.md"], "nested readme"),
    (["tests/test_agent.py"], "tests dir"),
    (["agents/test_harness.py"], "test_ prefixed module outside tests/"),
    (["docs/a.md", "tests/test_b.py"], "all-low mixed set"),
    (["NOTES.txt"], "plain text"),
]

HIGH_SETS = [
    ([], "empty staged set is not proof of low risk"),
    ([".claude/rules/workflow.md"], "rules are control plane"),
    ([".claude/hooks/x.py"], "hooks are authorization"),
    ([".claude/agents/co-lead.md"], "role definitions"),
    ([".claude/MEMORY/atlas/x_arc.md"], "durable receipts"),
    (["CLAUDE.md"], "root manifest"),
    (["AGENTS.md"], "root agent manifest"),
    ([".mcp.json"], "mcp wiring"),
    ([".github/workflows/ci.yml"], "CI is control plane"),
    ([".codex/agents/developer.toml"], "codex role home"),
    (["rust/src/main.rs"], "source is not LOW"),
    (["calm/hrm/checkpoints/model.pt"], "banked artifact"),
    (["docs/a.md", ".claude/rules/x.md"], "MIXED set is HIGH"),
    (["docs/a.md", "rust/src/main.rs"], "MIXED docs+source is HIGH"),
    (["../outside.md"], "parent traversal"),
    (["/etc/passwd.md"], "absolute path"),
    (["~/notes.md"], "home-relative path"),
    (["nested/.claude/rules/x.md"], "control-plane dir at any depth"),
    ([".CLAUDE/rules/x.md"], "control-plane dir case variant"),
    (["claude.md"], "CLAUDE.md case variant"),
    (["agents.md"], "AGENTS.md case variant"),
    ([".GITHUB/workflows/ci.yml"], "github dir case variant"),
    ([".CODEX/agents/x.toml"], "codex dir case variant"),
    ([".MCP.json"], ".mcp.json case variant"),
    ([".GITIGNORE"], ".gitignore case variant"),
    (["docs/a.MD"], "doc suffix case variant stays HIGH"),
    (["TEST_X.PY"], "test_ prefix case variant stays HIGH"),
    (["Tests/a.py"], "tests dir case variant stays HIGH"),
    (["notes.TXT"], "txt suffix case variant stays HIGH"),
]


@pytest.mark.parametrize("paths,label", LOW_SETS)
def test_low(paths: list[str], label: str) -> None:
    assert classifier.commit_tier(paths) == "LOW", label


@pytest.mark.parametrize("paths,label", HIGH_SETS)
def test_high(paths: list[str], label: str) -> None:
    assert classifier.commit_tier(paths) == "HIGH", label


def test_deny_list_case_variants_are_high() -> None:
    """Deny-list comparisons must treat case variants as the same object."""
    for paths, label in (
        ([".CLAUDE/rules/x.md"], "dir .CLAUDE"),
        (["claude.md"], "basename claude.md"),
        (["agents.md"], "basename agents.md"),
        ([".GITHUB/workflows/ci.yml"], "dir .GITHUB"),
        ([".CODEX/agents/x.toml"], "dir .CODEX"),
        ([".MCP.json"], "basename .MCP.json"),
        ([".GITIGNORE"], "basename .GITIGNORE"),
    ):
        assert classifier.commit_tier(paths) == "HIGH", label


def test_allow_list_case_variants_stay_high() -> None:
    """Allow-list comparisons stay case-sensitive — a fold here widens toward LOW."""
    for paths, label in (
        (["docs/a.MD"], "docs/a.MD"),
        (["TEST_X.PY"], "TEST_X.PY"),
        (["Tests/a.py"], "Tests/"),
        (["notes.TXT"], "notes.TXT"),
    ):
        assert classifier.commit_tier(paths) == "HIGH", label


# --- integration: same repo, same command, only the staged class differs ------


def _repo(tmp_path: pathlib.Path, rel: str) -> pathlib.Path:
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    target = repo / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("content\n")
    subprocess.run(["git", "add", rel], cwd=repo, check=True)
    return repo


def _run_gate(repo: pathlib.Path, log: pathlib.Path) -> subprocess.CompletedProcess:
    payload = json.dumps(
        {
            "tool_name": "Bash",
            "tool_input": {"command": f"git -C {repo} commit -m msg"},
        }
    )
    return subprocess.run(
        ["python3", str(GATE)],
        input=payload,
        capture_output=True,
        text=True,
        timeout=60,
        env={
            "PATH": "/usr/bin:/bin",
            "AI_ROOM_CHANNEL_LOG": str(log),
            # Makes the allow REASON observable. Without it an exit 0 cannot be
            # told apart from a fail-open on some earlier branch.
            "COMMIT_PRECONDITION_GATE_DEBUG": "1",
        },
    )


def test_low_staged_set_is_allowed(tmp_path: pathlib.Path) -> None:
    log = tmp_path / "messages.jsonl"
    log.write_text("")  # readable, no co_lead PASS anywhere in it
    repo = _repo(tmp_path, "docs/guide.md")
    proc = _run_gate(repo, log)
    assert proc.returncode == 0, f"LOW commit should pass: {proc.stderr}"
    # Exit 0 alone would also be produced by a fail-open on any earlier branch
    # (unrecognized commit shape, empty command, …), so assert WHICH branch
    # allowed it. Without this the test passes against a gate that never ran.
    assert "LOW-tier staged set" in proc.stderr, (
        f"allowed, but not by the tier branch: {proc.stderr!r}"
    )


def test_high_staged_set_is_blocked(tmp_path: pathlib.Path) -> None:
    """Identical setup to the LOW case except the staged path is control-plane.

    This is the negative-path control: the empty channel log contains no
    co_lead PASS in either test, so a block here proves the gate still fires
    for its OWN reason and that the LOW pass above came from the tier decision
    rather than from the gate having been disabled.
    """
    log = tmp_path / "messages.jsonl"
    log.write_text("")
    repo = _repo(tmp_path, ".claude/rules/workflow.md")
    proc = _run_gate(repo, log)
    assert proc.returncode == 2, "control-plane commit must still be gated"
    assert "without fresh co_lead validation/diff gate" in proc.stderr


def test_missing_channel_log_still_blocks_before_tiering(
    tmp_path: pathlib.Path,
) -> None:
    """A LOW staged set must not rescue an unreadable channel log."""
    repo = _repo(tmp_path, "docs/guide.md")
    proc = _run_gate(repo, tmp_path / "does-not-exist.jsonl")
    assert proc.returncode == 2
    assert "channel log missing/unreadable" in proc.stderr


_gspec = importlib.util.spec_from_file_location("_tier_gate", GATE)
assert _gspec and _gspec.loader
_gate = importlib.util.module_from_spec(_gspec)
_gspec.loader.exec_module(_gate)


def test_unreadable_staged_listing_is_high(tmp_path: pathlib.Path) -> None:
    """git listing failure must tier HIGH — same fail-closed as empty."""
    missing = tmp_path / "not-a-repo"
    missing.mkdir()
    tier, reason = _gate._staged_commit_tier(f"git -C {missing} commit -m msg")
    assert tier == "HIGH", reason
    assert reason in {"staged path listing failed", "target repo unresolved", "staged path listing raised"}


def test_empty_index_listing_is_high(tmp_path: pathlib.Path) -> None:
    """A real git repo with nothing staged is an empty listing, hence HIGH."""
    repo = tmp_path / "empty"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    tier, reason = _gate._staged_commit_tier(f"git -C {repo} commit -m msg")
    assert tier == "HIGH", reason
    assert classifier.commit_tier([]) == "HIGH"
