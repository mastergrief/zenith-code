#!/usr/bin/env python3
"""Replayable reference-fidelity check for r4b bit-equivalence anchor."""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ACCUMULATOR = REPO_ROOT / "calm/hrm_text_158/native_full_stack/bounded_delta_accumulator.py"
BASE_SHA = "3936d74"


def _extract_function(text: str, name: str, end_marker: str) -> str:
    start = text.index(f"def {name}")
    end = text.index(end_marker, start)
    return text[start:end]


def main() -> int:
    git_show = subprocess.run(
        ["git", "show", f"{BASE_SHA}:calm/hrm_text_158/native_full_stack/bounded_delta_accumulator.py"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    original = _extract_function(
        git_show.stdout,
        "execute_direct_bounded_local_vote_update_candidate",
        "def bounded_delta_candidate_assessment",
    )
    current_text = ACCUMULATOR.read_text(encoding="utf-8")
    reference = _extract_function(
        current_text,
        "_execute_direct_bounded_local_vote_update_reference_3936d74",
        "def bounded_delta_candidate_assessment",
    ).replace(
        "def _execute_direct_bounded_local_vote_update_reference_3936d74",
        "def execute_direct_bounded_local_vote_update_candidate",
        1,
    )
    original_sha = hashlib.sha256(original.encode("utf-8")).hexdigest()
    reference_sha = hashlib.sha256(reference.encode("utf-8")).hexdigest()
    match = original_sha == reference_sha
    print(
        {
            "command": (
                "git show 3936d74:.../bounded_delta_accumulator.py | "
                "extract execute_direct... vs _execute_direct_reference_3936d74"
            ),
            "original_body_sha256": original_sha,
            "reference_body_sha256": reference_sha,
            "byte_faithful": match,
            "artifact": str(ACCUMULATOR),
        }
    )
    return 0 if match else 1


if __name__ == "__main__":
    raise SystemExit(main())
