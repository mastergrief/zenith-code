from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from hrm_text_158_d_recompute_window_run_packet import (  # noqa: E402
    verify_required_ancestor,
)

SLICE1_ANCESTOR = "5cc8fb95fc093d1897ddc128d850e3df9a1ff5d3"


def test_required_ancestor_present_when_head_contains_science_fix() -> None:
    ok, required, head = verify_required_ancestor(SLICE1_ANCESTOR, cwd=REPO_ROOT)
    assert ok is True
    assert required == SLICE1_ANCESTOR
    assert head
    proc = subprocess.run(
        ["git", "merge-base", "--is-ancestor", SLICE1_ANCESTOR, "HEAD"],
        cwd=REPO_ROOT,
        check=False,
    )
    assert proc.returncode == 0


def test_required_ancestor_missing_when_not_in_history() -> None:
    missing = "ffffffffffffffffffffffffffffffffffffffff"

    def fake_run(argv, **kwargs):  # type: ignore[no-untyped-def]
        if argv[:3] == ["git", "rev-parse", "HEAD"]:
            return subprocess.CompletedProcess(argv, 0, "deadbeef" * 5, "")
        if argv[:2] == ["git", "merge-base"]:
            return subprocess.CompletedProcess(argv, 1, "", "")
        raise AssertionError(f"unexpected argv: {argv}")

    with patch(
        "hrm_text_158_d_recompute_window_run_packet.subprocess.run",
        side_effect=fake_run,
    ):
        ok, required, head = verify_required_ancestor(missing, cwd=REPO_ROOT)
    assert ok is False
    assert required == missing
    assert head == "deadbeef" * 5
