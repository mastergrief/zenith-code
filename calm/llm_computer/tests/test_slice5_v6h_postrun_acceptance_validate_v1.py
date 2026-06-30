"""Fixture tests for v6h postrun acceptance validator."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
FIXTURE_POS = REPO / "calm/llm_computer/tests/fixtures/slice5_v6h_acceptance_validate_positive"
FIXTURE_NEG = REPO / "calm/llm_computer/tests/fixtures/slice5_v6h_acceptance_validate_negative"
FIXTURE_NEG_BASELINE_FAST_INSTRUMENTED_SLOW = (
    REPO
    / "calm/llm_computer/tests/fixtures/slice5_v6h_acceptance_validate_negative_baseline_fast_instrumented_slow"
)
FIXTURE_NEG_COST_SHIFT = (
    REPO
    / "calm/llm_computer/tests/fixtures/slice5_v6h_acceptance_validate_negative_cost_shift_subphase_or_emit"
)
SCRIPT = REPO / "scripts/hrm_text_158_slice5_v6h_postrun_acceptance_validate.py"


def _run_validator(run_root: Path) -> tuple[int, dict]:
    out = run_root / "prelaunch" / "v6h_postrun_acceptance_validate_receipt.json"
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


def test_v6h_acceptance_validator_positive_fixture_passes() -> None:
    exit_code, receipt = _run_validator(FIXTURE_POS)
    assert exit_code == 0, receipt
    assert receipt["pass"] is True
    assert receipt["checks"]["cap_selection_path_evidence_overall_route"] == "PATH_GPU_SEAM_EXERCISED"
    speedup = receipt["checks"]["sparse_cap_speedup_vs_v6g_cpu_baseline"]
    assert speedup["baseline"]["1"]["material_win"] is True
    assert speedup["instrumented"]["3"]["elapsed_seconds"] == 16.0
    assert receipt["checks"]["sparse_cap_cost_shift_report"]["baseline"]["cost_shift_failures"] == []


def test_v6h_acceptance_validator_negative_fixture_fails_wrong_route_or_slow_cap() -> None:
    exit_code, receipt = _run_validator(FIXTURE_NEG)
    assert exit_code == 1
    assert receipt["pass"] is False
    assert any(
        f.startswith("unexpected_cap_selection_path_route:")
        or f.startswith("sparse_cap_not_materially_faster_baseline_step_")
        or f.startswith("sparse_cap_not_materially_faster_instrumented_step_")
        or "aggregate_peak_per_step_emit_delta_seconds_null" in f
        for f in receipt["failures"]
    )


def test_v6h_acceptance_validator_fails_baseline_fast_instrumented_slow() -> None:
    exit_code, receipt = _run_validator(FIXTURE_NEG_BASELINE_FAST_INSTRUMENTED_SLOW)
    assert exit_code == 1
    assert receipt["pass"] is False
    assert any(
        f.startswith("sparse_cap_not_materially_faster_instrumented_step_") for f in receipt["failures"]
    )
    assert not any(
        f.startswith("sparse_cap_not_materially_faster_baseline_step_") for f in receipt["failures"]
    )


def test_v6h_acceptance_validator_fails_cost_shift_subphase_exceeds_parent() -> None:
    exit_code, receipt = _run_validator(FIXTURE_NEG_COST_SHIFT)
    assert exit_code == 1
    assert receipt["pass"] is False
    assert any(
        f.startswith("sparse_cap_cost_shift_subphase_exceeds_parent_") for f in receipt["failures"]
    )
