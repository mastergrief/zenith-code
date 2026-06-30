"""Fixture tests for v6i postrun acceptance validator."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts/hrm_text_158_slice5_v6i_postrun_acceptance_validate.py"
FIXTURES = REPO / "calm/llm_computer/tests/fixtures"
V6H_POSITIVE_FIXTURE = REPO / "calm/llm_computer/tests/fixtures/slice5_v6h_acceptance_validate_positive"


def _run_validator(run_root: Path) -> tuple[int, dict]:
    out = run_root / "prelaunch" / "v6i_postrun_acceptance_validate_receipt.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--run-root",
            str(run_root),
            "--out",
            str(out),
            "--max-steps-hard",
            "8",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONPATH": str(REPO)},
    )
    receipt = json.loads(proc.stdout) if proc.stdout.strip() else {}
    return proc.returncode, receipt


def _meta(run_root: Path) -> dict:
    return json.loads((run_root / "fixture_meta.json").read_text(encoding="utf-8"))


def test_v6i_acceptance_validator_warmup_clears_fixture() -> None:
    root = FIXTURES / "slice5_v6i_acceptance_validate_warmup_clears"
    exit_code, receipt = _run_validator(root)
    meta = _meta(root)
    assert exit_code == 0, receipt
    assert receipt["pass"] is True
    assert receipt["warmup_classifier_verdict"] == meta["expected_verdict"] == "WARMUP_AMORTIZES_CLEARS"
    assert receipt["checks"]["late_window_report"]["baseline"]["still_trending_down"] is False
    assert receipt["checks"]["late_window_report"]["instrumented"]["still_trending_down"] is False


def test_v6i_acceptance_validator_structural_plateau_fixture() -> None:
    root = FIXTURES / "slice5_v6i_acceptance_validate_structural_plateau"
    exit_code, receipt = _run_validator(root)
    meta = _meta(root)
    assert exit_code == 0, receipt
    assert receipt["pass"] is True
    assert receipt["warmup_classifier_verdict"] == meta["expected_verdict"] == "STRUCTURAL_PLATEAU"


def test_v6i_acceptance_validator_noisy_ambiguous_fixture() -> None:
    root = FIXTURES / "slice5_v6i_acceptance_validate_noisy_ambiguous"
    exit_code, receipt = _run_validator(root)
    meta = _meta(root)
    assert exit_code == 0, receipt
    assert receipt["pass"] is True
    assert receipt["warmup_classifier_verdict"] == meta["expected_verdict"] == "NOISY_AMBIGUOUS"


def test_v6i_acceptance_validator_arms_disagree_no_trend_is_noisy_not_plateau() -> None:
    root = FIXTURES / "slice5_v6i_acceptance_validate_arms_disagree_no_trend"
    exit_code, receipt = _run_validator(root)
    meta = _meta(root)
    assert exit_code == 0, receipt
    assert receipt["pass"] is True
    assert receipt["warmup_classifier_verdict"] == "NOISY_AMBIGUOUS"
    assert receipt["warmup_classifier_verdict"] != meta["not_verdict"]
    late = receipt["checks"]["late_window_report"]
    assert late["baseline"]["late_window_material_clear"] is True
    assert late["instrumented"]["late_window_material_clear"] is False
    assert late["baseline"]["still_trending_down"] is False
    assert late["instrumented"]["still_trending_down"] is False


def test_v6i_acceptance_validator_cap_schedule_drift_fixture_fails() -> None:
    root = FIXTURES / "slice5_v6i_acceptance_validate_cap_schedule_drift"
    exit_code, receipt = _run_validator(root)
    meta = _meta(root)
    assert exit_code == 1
    assert receipt["pass"] is False
    assert receipt["warmup_classifier_verdict"] is None
    assert any(f.startswith(meta["failure_prefix"]) for f in receipt["failures"])


def test_v6i_acceptance_validator_short_run_fails_closed_without_crash() -> None:
    from scripts.hrm_text_158_slice5_v6i_postrun_acceptance_validate import validate_v6i_acceptance

    receipt = validate_v6i_acceptance(V6H_POSITIVE_FIXTURE, max_steps_hard=8)
    assert receipt["pass"] is False
    assert receipt["warmup_classifier_verdict"] is None
    assert receipt["checks"]["late_window_complete"] is False
    assert any(
        failure.startswith("missing_baseline_sparse_cap_elapsed_step_")
        or failure.startswith("missing_instrumented_sparse_cap_elapsed_step_")
        for failure in receipt["failures"]
    )
    assert any(failure.startswith("missing_baseline_sparse_cap_elapsed_step_6") for failure in receipt["failures"])


def test_v6i_acceptance_validator_reports_cap_workload_from_step_reports() -> None:
    root = FIXTURES / "slice5_v6i_acceptance_validate_warmup_clears"
    _, receipt = _run_validator(root)
    workload = receipt["checks"]["cap_workload_step_reports"]["baseline"]["6"]
    assert workload["q_changed_count"] == 256
    assert workload["step_result"]["global_summary"]["global_rate_cap_accepted_count"] == 256
