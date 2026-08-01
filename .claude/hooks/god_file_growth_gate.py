#!/usr/bin/env python3
"""
PreToolUse hook on `Edit` / `Write` — god-file growth gate.

Mechanical enforcement of the `.claude/rules/architecture_discipline.md`
stop condition: governed source files must not grow past the 500-line
cap. Rule text: "A file exceeds 500 lines while mixing CLI, IO, training
loop, telemetry, validation, or artifact contracts" turns the next
mutating gate into a refactor gate. The semantic "mixing" judgment stays
human/review-side; this gate enforces the measurable half — line-count
growth — at edit time, block-and-explain style.

Semantics:
  - Governed surface: `*.py` under `scripts/`, `calm/`, `agents/`
    relative to the project root (CLAUDE_PROJECT_DIR). Everything else
    is out of scope (allow).
  - Cap: max(500, grandfathered baseline). Existing over-cap files are
    grandfathered at their line count as of hook introduction via
    `.claude/god_file_baseline.json` — they may be edited and may
    SHRINK, but may not grow past their recorded baseline.
  - Blocks only GROWTH past the cap: resulting_lines > cap AND
    resulting_lines > current_lines. Shrinking or neutral edits to an
    over-cap file always pass (never punish a refactor).
  - The escape hatch is a reviewed edit to the baseline file itself
    (raise `max_lines` with a written reason), not an inline override
    marker — the exception then lives in a diffable, gated artifact.

Failure modes (fail-open): empty stdin, JSON parse failure, unreadable
target/baseline, unexpected schema → allow. Never wedge a turn on a
hook bug; the architecture invariant still applies operationally.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

CAP_LINES = 500
GOVERNED_PREFIXES = ("scripts/", "calm/", "agents/")
BASELINE_RELPATH = ".claude/god_file_baseline.json"
RULE_POINTER = ".claude/rules/architecture_discipline.md"


def _count_lines(text: str) -> int:
    if not text:
        return 0
    return len(text.splitlines())


def _load_baseline(project_root: Path) -> dict:
    try:
        raw = json.loads((project_root / BASELINE_RELPATH).read_text())
        files = raw.get("files", {})
        return files if isinstance(files, dict) else {}
    except Exception:
        return {}


def evaluate(
    tool_name: str,
    tool_input: dict,
    project_root: Path,
    baseline: dict,
    cap: int = CAP_LINES,
):
    """Pure decision core. Returns (decision, reason) where decision is
    "allow" or "deny". All uncertainty resolves to allow (fail-open)."""
    file_path = tool_input.get("file_path")
    if not isinstance(file_path, str) or not file_path:
        return "allow", "no file_path"

    try:
        rel = os.path.relpath(Path(file_path).resolve(), project_root.resolve())
    except Exception:
        return "allow", "unresolvable path"
    rel = rel.replace(os.sep, "/")

    if rel.startswith("../") or not rel.endswith(".py"):
        return "allow", "outside governed surface"
    if not rel.startswith(GOVERNED_PREFIXES):
        return "allow", "outside governed surface"

    target = Path(file_path)
    try:
        current_text = target.read_text(errors="replace") if target.exists() else ""
    except Exception:
        return "allow", "unreadable target (fail-open)"
    current_lines = _count_lines(current_text)

    if tool_name == "Write":
        content = tool_input.get("content")
        if not isinstance(content, str):
            return "allow", "no content"
        resulting_lines = _count_lines(content)
    elif tool_name == "Edit":
        old = tool_input.get("old_string")
        new = tool_input.get("new_string")
        if not isinstance(old, str) or not isinstance(new, str) or not old:
            return "allow", "malformed edit (tool will error)"
        if old not in current_text:
            return "allow", "old_string absent (tool will error)"
        # Exact resulting text, not a newline-delta estimate: a delta
        # miscounts at EOF (e.g. old='a', new='a\n' on a file ending
        # without a newline estimates +1 line while splitlines() says
        # the count is unchanged), which would false-deny a neutral
        # edit at a cap boundary.
        if tool_input.get("replace_all"):
            resulting_text = current_text.replace(old, new)
        else:
            resulting_text = current_text.replace(old, new, 1)
        resulting_lines = _count_lines(resulting_text)
    else:
        return "allow", "untracked tool"

    entry = baseline.get(rel)
    grandfathered = 0
    if isinstance(entry, dict):
        ml = entry.get("max_lines")
        reason_field = entry.get("reason")
        # Strict escape-hatch schema: an entry only raises the cap when
        # max_lines is a true int (not bool/str/float) above the base
        # cap AND a non-empty written reason is present. Anything else
        # is an invalid exemption and falls back to the base cap.
        if (
            type(ml) is int
            and ml > cap
            and isinstance(reason_field, str)
            and reason_field.strip()
        ):
            grandfathered = ml
    effective_cap = max(cap, grandfathered)

    if resulting_lines > effective_cap and resulting_lines > current_lines:
        reason = (
            f"god-file growth gate: {rel} would grow to {resulting_lines} lines "
            f"(current {current_lines}, cap {effective_cap}"
            f"{', grandfathered baseline ' + str(grandfathered) if grandfathered else ''}). "
            f"{RULE_POINTER} stop condition: do not extend an over-cap file — "
            f"extract the new logic to a separate module/facade/test module "
            f"instead (promote on second use; new semantic families get new "
            f"files). Shrinking/neutral edits to this file are always allowed. "
            f"If growth here is a deliberate, reviewed exception, raise this "
            f"file's max_lines entry (with a written reason) in "
            f"{BASELINE_RELPATH} first — that change is itself diffable and "
            f"review-gated. Do not bypass by chunking the addition into "
            f"multiple small edits."
        )
        return "deny", reason

    return "allow", "within cap"


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read() or "{}")
        tool_name = payload.get("tool_name", "")
        tool_input = payload.get("tool_input") or {}
        if tool_name not in ("Edit", "Write") or not isinstance(tool_input, dict):
            return 0
        project_root = Path(
            os.environ.get("CLAUDE_PROJECT_DIR") or payload.get("cwd") or os.getcwd()
        )
        baseline = _load_baseline(project_root)
        decision, reason = evaluate(tool_name, tool_input, project_root, baseline)
    except Exception:
        return 0  # fail-open

    if decision == "deny":
        print(
            json.dumps(
                {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": reason,
                    }
                }
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
