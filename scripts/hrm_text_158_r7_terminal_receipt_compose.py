#!/usr/bin/env python3
"""Compose R7 from-clean terminal receipt from classifier + diagnostic receipts."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

BRANCH_HARNESS_FAIL = "R7_HARNESS_FAIL"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compose_terminal_receipt(run_root: Path) -> dict:
    classifier_path = run_root / "r7_mechanism_classifier_receipt.json"
    diag_receipt_path = run_root / "diagnostic" / "receipt.json"
    if not classifier_path.is_file():
        raise SystemExit("R7_TERMINAL_COMPOSE_FAIL:missing_classifier_receipt")
    classifier = json.loads(classifier_path.read_text(encoding="utf-8"))
    if classifier.get("metrics"):
        raise SystemExit("R7_TERMINAL_COMPOSE_FAIL:forbidden_metrics_key")
    run_metrics = classifier.get("run_metrics")
    if not isinstance(run_metrics, dict):
        raise SystemExit("R7_TERMINAL_COMPOSE_FAIL:missing_run_metrics")

    branch_sel = dict(classifier.get("branch_selection") or {})
    branch_name = str(branch_sel.get("branch") or "")
    harness_fail_branch = branch_name == BRANCH_HARNESS_FAIL

    diag_receipt: dict | None = None
    if diag_receipt_path.is_file():
        diag_receipt = json.loads(diag_receipt_path.read_text(encoding="utf-8"))
    elif not harness_fail_branch:
        raise SystemExit("R7_TERMINAL_COMPOSE_FAIL:missing_diagnostic_receipt")

    required = [
        "steps_observed",
        "pressure_mass_first",
        "pressure_mass_last",
        "pressure_growth_ratio",
        "run_max_deferred_backlog_max_age_steps",
        "run_mean_deferred_saturation",
        "q_transition_mass_ratio",
    ]
    missing = [k for k in required if k not in run_metrics]
    if missing and not harness_fail_branch:
        raise SystemExit("R7_TERMINAL_COMPOSE_FAIL:missing_run_metrics_keys:" + ",".join(missing))

    steps_obs = int(run_metrics.get("steps_observed", 0))
    if steps_obs < 8 and not harness_fail_branch:
        branch_sel = {
            "branch": "R7_ARTIFACT_INSUFFICIENT",
            "next_action": "instrumentation_not_interpretation",
            "reason": "fewer_than_eight_measured_steps",
            "terminal_override": True,
        }

    sidecar_path = run_root / "diagnostic" / "r7_cap_defer_pressure_sidecar.jsonl"
    terminal = {
        "schema_version": "hrm_text_158_r7_from_clean_terminal_receipt/v1",
        "run_root": str(run_root),
        "diagnostic_scratch_root": str(run_root / "diagnostic"),
        "head_sha256": classifier.get("head_sha256"),
        "branch_selection": branch_sel,
        "primary_branch": branch_sel.get("branch"),
        "next_action": branch_sel.get("next_action"),
        "steps_completed": diag_receipt.get("steps_completed") if diag_receipt else 0,
        "run_metrics": run_metrics,
        "steps_observed": steps_obs,
        "pressure_growth_ratio": run_metrics.get("pressure_growth_ratio"),
        "run_max_deferred_backlog_max_age_steps": run_metrics.get(
            "run_max_deferred_backlog_max_age_steps"
        ),
        "run_mean_deferred_saturation": run_metrics.get("run_mean_deferred_saturation"),
        "q_transition_mass_ratio": run_metrics.get("q_transition_mass_ratio"),
        "classifier_receipt_path": str(classifier_path),
        "classifier_receipt_sha256": sha256_file(classifier_path),
        "diagnostic_receipt_path": str(diag_receipt_path) if diag_receipt else None,
        "diagnostic_receipt_sha256": sha256_file(diag_receipt_path) if diag_receipt else None,
        "sidecar_path": str(sidecar_path) if sidecar_path.is_file() else None,
        "sidecar_sha256": sha256_file(sidecar_path) if sidecar_path.is_file() else None,
        "explicit_non_claims": classifier.get("explicit_non_claims", []),
        "run_mode": "from_clean_single_arm_diagnostic",
        "reused_prior_run": False,
    }
    if harness_fail_branch and diag_receipt is None:
        terminal["harness_fail_without_diagnostic_receipt"] = True
        terminal["diagnostic_never_launched"] = True
    out = run_root / "terminal_receipt.json"
    out.write_text(json.dumps(terminal, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return terminal


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="R7 terminal receipt compose.")
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args(argv)
    terminal = compose_terminal_receipt(args.run_root)
    print(json.dumps(terminal))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
