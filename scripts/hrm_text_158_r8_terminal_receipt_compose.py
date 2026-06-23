#!/usr/bin/env python3
"""Compose R8 global_cap_relax_512 terminal receipt from classifier + diagnostic."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

BRANCH_HARNESS_FAIL = "R8_HARNESS_FAIL"
BRANCH_ARTIFACT_INSUFFICIENT = "R8_ARTIFACT_INSUFFICIENT"
CLASSIFIER_RECEIPT_NAME = "r8_global_cap_relax_classifier_receipt.json"
TERMINAL_SCHEMA_VERSION = "hrm_text_158_r8_global_cap_relax_terminal_receipt/v1"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compose_terminal_receipt(run_root: Path) -> dict:
    classifier_path = run_root / CLASSIFIER_RECEIPT_NAME
    diag_receipt_path = run_root / "diagnostic" / "receipt.json"
    if not classifier_path.is_file():
        raise SystemExit("R8_TERMINAL_COMPOSE_FAIL:missing_classifier_receipt")
    classifier = json.loads(classifier_path.read_text(encoding="utf-8"))
    if classifier.get("metrics"):
        raise SystemExit("R8_TERMINAL_COMPOSE_FAIL:forbidden_metrics_key")
    run_metrics = classifier.get("run_metrics")
    if not isinstance(run_metrics, dict):
        raise SystemExit("R8_TERMINAL_COMPOSE_FAIL:missing_run_metrics")

    branch_sel = dict(classifier.get("branch_selection") or {})
    branch_name = str(branch_sel.get("branch") or "")
    harness_fail_branch = branch_name == BRANCH_HARNESS_FAIL

    diag_receipt: dict | None = None
    diagnostic_complete = diag_receipt_path.is_file()
    if diagnostic_complete:
        diag_receipt = json.loads(diag_receipt_path.read_text(encoding="utf-8"))

    run_incomplete = not diagnostic_complete and not harness_fail_branch
    diagnostic_crashed = run_incomplete

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
    if missing and not harness_fail_branch and diagnostic_complete:
        raise SystemExit("R8_TERMINAL_COMPOSE_FAIL:missing_run_metrics_keys:" + ",".join(missing))

    steps_obs = int(run_metrics.get("steps_observed", 0))
    if steps_obs < 8 and not harness_fail_branch and diagnostic_complete:
        branch_sel = {
            "branch": BRANCH_ARTIFACT_INSUFFICIENT,
            "next_action": "instrumentation_not_interpretation",
            "reason": "fewer_than_eight_measured_steps",
            "terminal_override": True,
        }

    sidecar_path = run_root / "diagnostic" / "r7_cap_defer_pressure_sidecar.jsonl"
    terminal = {
        "schema_version": TERMINAL_SCHEMA_VERSION,
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
        "audit_summary": classifier.get("audit_summary"),
        "baseline_comparison": classifier.get("baseline_comparison"),
        "r7_baseline_provenance": classifier.get("r7_baseline_provenance"),
        "classifier_receipt_path": str(classifier_path),
        "classifier_receipt_sha256": sha256_file(classifier_path),
        "diagnostic_receipt_path": str(diag_receipt_path) if diag_receipt else None,
        "diagnostic_receipt_sha256": sha256_file(diag_receipt_path) if diag_receipt else None,
        "sidecar_path": str(sidecar_path) if sidecar_path.is_file() else None,
        "sidecar_sha256": sha256_file(sidecar_path) if sidecar_path.is_file() else None,
        "explicit_non_claims": classifier.get("explicit_non_claims", []),
        "run_mode": "r8_global_cap_relax_512_from_clean_single_arm_diagnostic",
        "reused_prior_run": False,
    }
    if harness_fail_branch and diag_receipt is None:
        terminal["harness_fail_without_diagnostic_receipt"] = True
        terminal["diagnostic_never_launched"] = True
    if run_incomplete:
        terminal["run_incomplete"] = True
        terminal["diagnostic_crashed"] = diagnostic_crashed
        terminal["diagnostic_receipt_missing"] = True
    out = run_root / "terminal_receipt.json"
    out.write_text(json.dumps(terminal, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return terminal


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="R8 terminal receipt compose.")
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args(argv)
    terminal = compose_terminal_receipt(args.run_root)
    print(json.dumps(terminal))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
