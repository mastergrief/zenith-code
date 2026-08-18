#!/usr/bin/env python3
"""Tests for god_file_growth_gate.py — decision core + fail-open main().

Run: python3 -m pytest .claude/hooks/test_god_file_growth_gate.py -q
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from god_file_growth_gate import CAP_LINES, evaluate  # noqa: E402

HOOK = Path(__file__).parent / "god_file_growth_gate.py"


def _lines(n: int) -> str:
    return "\n".join(f"line {i}" for i in range(n))


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    (tmp_path / "scripts").mkdir()
    (tmp_path / "docs").mkdir()
    return tmp_path


def _write(root: Path, rel: str, n_lines: int) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(_lines(n_lines) + "\n")
    return p


# ---- governed-surface routing ----

def test_non_governed_path_allowed(root: Path):
    p = root / "docs" / "big.py"
    d, _ = evaluate("Write", {"file_path": str(p), "content": _lines(9000)}, root, {})
    assert d == "allow"


def test_non_python_governed_dir_allowed(root: Path):
    p = root / "scripts" / "big.md"
    d, _ = evaluate("Write", {"file_path": str(p), "content": _lines(9000)}, root, {})
    assert d == "allow"


def test_path_outside_project_allowed(root: Path, tmp_path_factory):
    other = tmp_path_factory.mktemp("elsewhere") / "scripts" / "big.py"
    other.parent.mkdir(parents=True, exist_ok=True)
    d, _ = evaluate("Write", {"file_path": str(other), "content": _lines(9000)}, root, {})
    assert d == "allow"


# ---- Write semantics ----

def test_new_file_over_cap_denied(root: Path):
    p = root / "scripts" / "new_tool.py"
    d, reason = evaluate("Write", {"file_path": str(p), "content": _lines(CAP_LINES + 1)}, root, {})
    assert d == "deny"
    assert "scripts/new_tool.py" in reason and str(CAP_LINES + 1) in reason


def test_new_file_at_cap_allowed(root: Path):
    p = root / "scripts" / "new_tool.py"
    d, _ = evaluate("Write", {"file_path": str(p), "content": _lines(CAP_LINES)}, root, {})
    assert d == "allow"


def test_overwrite_shrinking_over_cap_file_allowed(root: Path):
    p = _write(root, "scripts/legacy.py", 800)
    d, _ = evaluate("Write", {"file_path": str(p), "content": _lines(700)}, root, {})
    assert d == "allow"  # 700 > cap but 700 < current 800: shrink never blocked


def test_overwrite_growing_over_cap_file_denied(root: Path):
    p = _write(root, "scripts/legacy.py", 800)
    d, _ = evaluate("Write", {"file_path": str(p), "content": _lines(801)}, root, {})
    assert d == "deny"


# ---- Edit semantics ----

def test_edit_growth_past_cap_denied(root: Path):
    p = _write(root, "scripts/mid.py", 499)
    d, _ = evaluate(
        "Edit",
        {"file_path": str(p), "old_string": "line 10", "new_string": "line 10\nextra\nextra2"},
        root, {},
    )
    assert d == "deny"  # 499 + 2 = 501 > 500


def test_edit_growth_within_cap_allowed(root: Path):
    p = _write(root, "scripts/mid.py", 400)
    d, _ = evaluate(
        "Edit",
        {"file_path": str(p), "old_string": "line 10", "new_string": "line 10\nextra"},
        root, {},
    )
    assert d == "allow"


def test_edit_shrink_on_over_cap_file_allowed(root: Path):
    p = _write(root, "scripts/legacy.py", 800)
    d, _ = evaluate(
        "Edit",
        {"file_path": str(p), "old_string": "line 10\nline 11\nline 12", "new_string": "line 10"},
        root, {},
    )
    assert d == "allow"


def test_edit_replace_all_multiplies_delta(root: Path):
    p = root / "scripts" / "rep.py"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(("marker()\n" + _lines(3) + "\n") * 100)  # 400 lines, 100 markers
    d, _ = evaluate(
        "Edit",
        {"file_path": str(p), "old_string": "marker()", "new_string": "marker()\npadding\npadding2",
         "replace_all": True},
        root, {},
    )
    assert d == "deny"  # 400 + 2*100 = 600 > 500


def test_edit_missing_old_string_allowed(root: Path):
    p = _write(root, "scripts/mid.py", 499)
    d, _ = evaluate(
        "Edit",
        {"file_path": str(p), "old_string": "NOT PRESENT", "new_string": "x\n" * 50},
        root, {},
    )
    assert d == "allow"  # tool itself will error; hook stays out of the way


# ---- Edit EOF/trailing-newline boundary (exact-result counting, not delta) ----

def test_edit_trailing_newline_at_cap_is_neutral_allowed(root: Path):
    # File at exactly cap, no trailing newline. Appending a trailing
    # newline adds a "\n" but NOT a line (splitlines unchanged) — a
    # delta estimator overcounts this and false-denies at the boundary.
    p = root / "scripts" / "atcap.py"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(_lines(CAP_LINES))  # no trailing newline
    d, _ = evaluate(
        "Edit",
        {"file_path": str(p), "old_string": f"line {CAP_LINES - 1}",
         "new_string": f"line {CAP_LINES - 1}\n"},
        root, {},
    )
    assert d == "allow"


def test_edit_trailing_newline_at_grandfathered_cap_allowed(root: Path):
    p = root / "scripts" / "legacy.py"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(_lines(800))  # no trailing newline
    baseline = {"scripts/legacy.py": {"max_lines": 800, "reason": "grandfathered"}}
    d, _ = evaluate(
        "Edit",
        {"file_path": str(p), "old_string": "line 799", "new_string": "line 799\n"},
        root, baseline,
    )
    assert d == "allow"


def test_edit_real_extra_line_at_cap_denied(root: Path):
    p = root / "scripts" / "atcap.py"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(_lines(CAP_LINES))
    d, _ = evaluate(
        "Edit",
        {"file_path": str(p), "old_string": f"line {CAP_LINES - 1}",
         "new_string": f"line {CAP_LINES - 1}\nreal new line"},
        root, {},
    )
    assert d == "deny"


# ---- baseline / grandfathering ----

def test_grandfathered_file_may_grow_to_baseline(root: Path):
    p = _write(root, "scripts/legacy.py", 800)
    baseline = {"scripts/legacy.py": {"max_lines": 900, "reason": "grandfathered"}}
    d, _ = evaluate("Write", {"file_path": str(p), "content": _lines(900)}, root, baseline)
    assert d == "allow"


def test_grandfathered_file_growth_past_baseline_denied(root: Path):
    p = _write(root, "scripts/legacy.py", 800)
    baseline = {"scripts/legacy.py": {"max_lines": 900, "reason": "grandfathered"}}
    d, reason = evaluate("Write", {"file_path": str(p), "content": _lines(901)}, root, baseline)
    assert d == "deny"
    assert "900" in reason


@pytest.mark.parametrize(
    "entry",
    [
        {"max_lines": "not an int"},                      # non-int
        {"max_lines": "900", "reason": "coercible str"},  # numeric string
        {"max_lines": 900.0, "reason": "float"},          # float
        {"max_lines": True, "reason": "bool"},            # bool is not int here
        {"max_lines": 900},                               # missing reason
        {"max_lines": 900, "reason": ""},                 # blank reason
        {"max_lines": 900, "reason": "   "},              # whitespace reason
        {"max_lines": 400, "reason": "below cap"},        # not above base cap
        {"max_lines": -5, "reason": "nonpositive"},       # nonpositive
        "just a number",                                   # non-dict entry
    ],
)
def test_invalid_baseline_entry_falls_back_to_base_cap(root: Path, entry):
    p = _write(root, "scripts/legacy.py", 400)
    baseline = {"scripts/legacy.py": entry}
    d, _ = evaluate("Write", {"file_path": str(p), "content": _lines(600)}, root, baseline)
    assert d == "deny"  # entry must NOT raise the effective cap


def test_valid_baseline_entry_requires_int_and_reason(root: Path):
    p = _write(root, "scripts/legacy.py", 400)
    baseline = {"scripts/legacy.py": {"max_lines": 900, "reason": "reviewed exception"}}
    d, _ = evaluate("Write", {"file_path": str(p), "content": _lines(600)}, root, baseline)
    assert d == "allow"


# ---- main() end-to-end via subprocess (stdin contract + fail-open) ----

def _run_main(payload, env_root: Path):
    import os
    env = dict(os.environ, CLAUDE_PROJECT_DIR=str(env_root))
    proc = subprocess.run(
        [sys.executable, str(HOOK)],
        input=payload if isinstance(payload, str) else json.dumps(payload),
        capture_output=True, text=True, env=env, timeout=30,
    )
    return proc


def test_main_denies_over_cap_write(root: Path):
    p = root / "scripts" / "new_tool.py"
    proc = _run_main(
        {"tool_name": "Write", "tool_input": {"file_path": str(p), "content": _lines(600)}},
        root,
    )
    assert proc.returncode == 0
    out = json.loads(proc.stdout)
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_main_allows_within_cap(root: Path):
    p = root / "scripts" / "new_tool.py"
    proc = _run_main(
        {"tool_name": "Write", "tool_input": {"file_path": str(p), "content": _lines(100)}},
        root,
    )
    assert proc.returncode == 0
    assert proc.stdout.strip() == ""


def test_main_fail_open_on_garbage_stdin(root: Path):
    proc = _run_main("not json {", root)
    assert proc.returncode == 0
    assert proc.stdout.strip() == ""


def test_main_fail_open_on_empty_stdin(root: Path):
    proc = _run_main("", root)
    assert proc.returncode == 0
    assert proc.stdout.strip() == ""


def test_main_ignores_other_tools(root: Path):
    proc = _run_main({"tool_name": "Bash", "tool_input": {"command": "ls"}}, root)
    assert proc.returncode == 0
    assert proc.stdout.strip() == ""


# ---- live-repo baseline sanity (tracked corpus, index bytes) ----

def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    ).stdout


def test_live_baseline_keys_are_tracked_and_match_index_bytes():
    """Every baseline key must be a git-TRACKED governed file, and its
    max_lines must equal the INDEX-byte line count at introduction. An
    entry for an absent/untracked path would pre-authorize a future new
    file at that path to skip the cap; a stale count means the baseline
    was not regenerated with the corpus it exempts. (Ratchet property:
    if a grandfathered file later shrinks in a commit, this test fails
    until the baseline entry is ratcheted down with it.)"""
    repo = Path(__file__).resolve().parents[2]
    data = json.loads((repo / ".claude" / "god_file_baseline.json").read_text())
    files = data["files"]
    assert isinstance(files, dict) and len(files) > 0
    tracked = set(_git(repo, "ls-files").splitlines())
    for rel, entry in files.items():
        assert rel.startswith(("scripts/", "calm/", "agents/")) and rel.endswith(".py")
        assert type(entry["max_lines"]) is int and entry["max_lines"] > CAP_LINES
        assert isinstance(entry["reason"], str) and entry["reason"].strip()
        assert rel in tracked, f"baseline key not git-tracked: {rel}"
        index_lines = len(_git(repo, "show", f":{rel}").splitlines())
        assert entry["max_lines"] == index_lines, (
            f"baseline count for {rel} ({entry['max_lines']}) != index bytes "
            f"({index_lines}); regenerate or ratchet the entry"
        )


def test_live_baseline_covers_all_tracked_over_cap_files():
    """Every TRACKED over-cap governed file must be grandfathered (no
    day-1 blocks on maintenance). Scans the tracked set via index bytes
    — untracked worktree files are deliberately out of scope: they earn
    no exemption."""
    repo = Path(__file__).resolve().parents[2]
    data = json.loads((repo / ".claude" / "god_file_baseline.json").read_text())
    files = data["files"]
    uncovered = []
    for rel in _git(repo, "ls-files").splitlines():
        if not rel.startswith(("scripts/", "calm/", "agents/")) or not rel.endswith(".py"):
            continue
        if "__pycache__" in rel:
            continue
        n = len(_git(repo, "show", f":{rel}").splitlines())
        if n > CAP_LINES and rel not in files:
            uncovered.append((rel, n))
    assert not uncovered, f"tracked over-cap files missing from baseline: {uncovered}"
