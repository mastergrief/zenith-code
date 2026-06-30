"""Fixture tests for v6g postrun acceptance validator."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
FIXTURE_POS = (
    REPO
    / "calm/llm_computer/tests/fixtures/slice5_v6g_acceptance_validate_positive"
)
FIXTURE_NEG = (
    REPO
    / "calm/llm_computer/tests/fixtures/slice5_v6g_acceptance_validate_negative"
)
SCRIPT = REPO / "scripts/hrm_text_158_slice5_v6g_postrun_acceptance_validate.py"


def _run_validator(run_root: Path) -> tuple[int, dict]:
    out = run_root / "prelaunch" / "v6g_postrun_acceptance_validate_receipt.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--run-root",
            str(run_root),
            "--out",
            str(out),
            "--max-steps-hard",
            "3",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONPATH": str(REPO)},
    )
    receipt = json.loads(proc.stdout) if proc.stdout.strip() else {}
    return proc.returncode, receipt


def test_v6g_acceptance_validator_positive_fixture_passes() -> None:
    exit_code, receipt = _run_validator(FIXTURE_POS)
    assert exit_code == 0, receipt
    assert receipt["pass"] is True
    assert receipt["checks"]["per_step_duration_delta_seconds"] == {"1": 1.0, "2": 2.0, "3": 3.0}
    assert receipt["checks"]["peak_per_step_emit_delta_seconds"] == 3.0
    assert receipt["checks"]["persistent_accumulator_event_coded_live_baseline"] is True
    assert receipt["checks"]["persistent_accumulator_event_coded_live_instrumented"] is True


def test_v6g_acceptance_validator_negative_fixture_fails_missing_aggregate_timing() -> None:
    exit_code, receipt = _run_validator(FIXTURE_NEG)
    assert exit_code == 1
    assert receipt["pass"] is False
    assert "aggregate_peak_per_step_emit_delta_seconds_null" in receipt["failures"]
