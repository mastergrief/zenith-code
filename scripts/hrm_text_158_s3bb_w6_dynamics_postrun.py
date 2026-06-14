#!/usr/bin/env python3
"""Dual-arm S3bb postrun comparator for headroom telemetry receipts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from calm.hrm_text_158.native_full_stack.s3bb_headroom_telemetry import (
    CLASSIFIER_HEADROOM_BREACH,
    RECEIPT_EMIT_PROFILE_SLIM,
    compare_arm_wiring_guards,
    emit_s3bb_classifier_receipt,
    resolve_headroom_wiring_sidecar_path,
    validate_headroom_telemetry_block,
)

ORACLE_ARM_DIR = "int16_oracle_flag_off"
TREATMENT_ARM_DIR = "w6_carrier_flag_on"
SUMMARY_FILENAME = "s3bb_headroom_summary.json"


def _load_arm_receipt(run_root: Path, arm_dir: str) -> dict[str, Any]:
    receipt_path = run_root / arm_dir / "receipt.json"
    if not receipt_path.is_file():
        raise FileNotFoundError(f"missing arm receipt: {receipt_path}")
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"arm receipt must be a JSON object: {receipt_path}")
    return payload


def _collect_harness_failures(
    oracle_receipt: dict[str, Any],
    treatment_receipt: dict[str, Any],
) -> list[str]:
    failures: list[str] = []
    for label, receipt in (
        ("oracle", oracle_receipt),
        ("treatment", treatment_receipt),
    ):
        slim_emit = str(receipt.get("receipt_emit_profile") or "") == RECEIPT_EMIT_PROFILE_SLIM
        if slim_emit:
            sidecar_path = resolve_headroom_wiring_sidecar_path(receipt)
            if sidecar_path is None or not sidecar_path.is_file():
                failures.append(f"{label}_missing_headroom_wiring_sidecar")
        step_reports = receipt.get("step_reports") or {}
        if not isinstance(step_reports, dict):
            failures.append(f"{label}_step_reports_not_object")
            continue
        for step_id, report in sorted(step_reports.items(), key=lambda item: int(item[0])):
            if not isinstance(report, dict):
                failures.append(f"{label}_step_{step_id}_report_not_object")
                continue
            telemetry = report.get("headroom_telemetry")
            if telemetry is None:
                failures.append(f"{label}_step_{step_id}_missing_headroom_telemetry")
                continue
            try:
                validate_headroom_telemetry_block(telemetry)
            except ValueError as exc:
                failures.append(f"{label}_step_{step_id}_{exc}")
            if slim_emit and (telemetry.get("accumulator_snapshots_by_state_key") or {}):
                failures.append(f"{label}_step_{step_id}_slim_receipt_has_inline_snapshots")
    return failures


def build_s3bb_headroom_summary(
    *,
    run_root: Path,
    oracle_receipt: dict[str, Any],
    treatment_receipt: dict[str, Any],
    harness_failures: list[str],
) -> dict[str, Any]:
    guards = compare_arm_wiring_guards(oracle_receipt, treatment_receipt)
    classifier_receipt = emit_s3bb_classifier_receipt(
        oracle_receipt,
        treatment_receipt,
        harness_failures=harness_failures,
    )
    return {
        "slice_id": "w6_gpu_dynamics_parity_run_s3bb_v0",
        "run_root": str(run_root),
        "oracle_arm": ORACLE_ARM_DIR,
        "treatment_arm": TREATMENT_ARM_DIR,
        "oracle_steps_completed": int(oracle_receipt.get("steps_completed") or 0),
        "treatment_steps_completed": int(treatment_receipt.get("steps_completed") or 0),
        "treatment_stop_reason": str(treatment_receipt.get("stop_reason") or ""),
        "wiring_guards": guards,
        "harness_failures": list(harness_failures),
        "primary_classifier": classifier_receipt["primary_classifier"],
    }


def run_postrun(*, run_root: Path, json_out: Path) -> dict[str, Any]:
    oracle_receipt = _load_arm_receipt(run_root, ORACLE_ARM_DIR)
    treatment_receipt = _load_arm_receipt(run_root, TREATMENT_ARM_DIR)
    harness_failures = _collect_harness_failures(oracle_receipt, treatment_receipt)
    summary = build_s3bb_headroom_summary(
        run_root=run_root,
        oracle_receipt=oracle_receipt,
        treatment_receipt=treatment_receipt,
        harness_failures=harness_failures,
    )
    classifier_receipt = emit_s3bb_classifier_receipt(
        oracle_receipt,
        treatment_receipt,
        harness_failures=harness_failures,
    )

    summary_path = run_root / SUMMARY_FILENAME
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(
        json.dumps(classifier_receipt, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return classifier_receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Emit S3bb dual-arm headroom classifier receipts from probe outputs.",
    )
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)

    receipt = run_postrun(run_root=args.run_root, json_out=args.json_out)
    if receipt.get("primary_classifier") == CLASSIFIER_HEADROOM_BREACH:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
