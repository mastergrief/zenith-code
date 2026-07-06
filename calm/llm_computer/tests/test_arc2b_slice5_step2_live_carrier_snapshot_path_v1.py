"""CPU regression: live_carrier_snapshot path must not double-join d_recompute_window_diagnostic."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from calm.hrm_text_158.native_full_stack.d_recompute_window_emit import (
    D_RECOMPUTE_WINDOW_LOG_FILENAME,
)
from calm.hrm_text_158.native_full_stack.d_recompute_window_live_carrier_snapshot import (
    initialize_live_carrier_snapshot_log,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
PROBE_PATH = REPO_ROOT / "scripts" / "hrm_text_158_bounded_delta_acquisition_probe.py"
LIVE_CARRIER_SNAPSHOT_FILENAME = "live_carrier_snapshot.jsonl"
DOUBLED_JOIN_FRAGMENT = (
    '"d_recompute_window_diagnostic" / "live_carrier_snapshot.jsonl"'
)


def resolve_live_carrier_snapshot_path(scratch_root: Path) -> Path:
    """Mirror probe.py post-fix contract: snapshot lives directly under scratch_root."""
    return Path(scratch_root) / LIVE_CARRIER_SNAPSHOT_FILENAME


def test_live_carrier_snapshot_non_doubled_when_scratch_is_diagnostic_subdir(
    tmp_path: Path,
) -> None:
    scratch = tmp_path / "d_recompute_window_diagnostic"
    scratch.mkdir(parents=True)

    recompute_log_path = scratch / D_RECOMPUTE_WINDOW_LOG_FILENAME
    recompute_log_path.write_text("", encoding="utf-8")
    receipt_path = scratch / "receipt.json"
    receipt_path.write_text("{}", encoding="utf-8")
    run_log_path = scratch / "run.log"
    run_log_path.write_text("", encoding="utf-8")

    snapshot_path = resolve_live_carrier_snapshot_path(scratch)
    initialize_live_carrier_snapshot_log(snapshot_path)

    doubled_path = (
        scratch / "d_recompute_window_diagnostic" / LIVE_CARRIER_SNAPSHOT_FILENAME
    )

    assert snapshot_path.is_file()
    assert snapshot_path == scratch / LIVE_CARRIER_SNAPSHOT_FILENAME
    assert not doubled_path.exists()
    assert recompute_log_path.is_file()
    assert receipt_path.is_file()
    assert run_log_path.is_file()


def test_probe_source_no_doubled_live_carrier_path_construction() -> None:
    probe_src = PROBE_PATH.read_text(encoding="utf-8")
    assert DOUBLED_JOIN_FRAGMENT not in probe_src
    assert 'Path(scratch_root) / "live_carrier_snapshot.jsonl"' in probe_src


def test_scope_guard_no_other_doubled_live_carrier_path_callers() -> None:
    proc = subprocess.run(
        [
            "rg",
            "-n",
            r'd_recompute_window_diagnostic.*live_carrier_snapshot|live_carrier_snapshot\.jsonl',
            "--glob",
            "*.py",
            str(REPO_ROOT),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    doubled_construction = [
        line
        for line in lines
        if re.search(
            r'd_recompute_window_diagnostic.*live_carrier_snapshot\.jsonl',
            line,
        )
        and "test_arc2b_slice5_step2_live_carrier_snapshot_path_v1.py" not in line
        and "DOUBLED_JOIN_FRAGMENT" not in line
        and "doubled_path" not in line
        and "doubled_construction" not in line
    ]
    assert doubled_construction == [], (
        "active doubled live_carrier_snapshot path construction remains: "
        + "; ".join(doubled_construction)
    )
