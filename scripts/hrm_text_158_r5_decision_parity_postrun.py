#!/usr/bin/env python3
"""Dual-arm S3bb postrun comparator for W5 decision-parity receipts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from calm.hrm_text_158.native_full_stack.s3bb_decision_parity import (
    CLASSIFIER_DECISION_PARITY_OK,
    CLASSIFIER_DOMAIN_OR_HEADROOM_FAIL,
    emit_s3bb_decision_parity_receipt,
)
from calm.hrm_text_158.native_full_stack.s3bb_headroom_telemetry import (
    RECEIPT_EMIT_PROFILE_SLIM,
    resolve_headroom_wiring_sidecar_path,
    validate_headroom_telemetry_block,
)

ORACLE_ARM_DIR = "w6_on_q_on_oracle"
TREATMENT_ARM_DIR = "w5_on_q_on_treatment"
SUMMARY_FILENAME = "r5_decision_parity_summary.json"


def preflight_arm_receipt_dirs(
    run_root: Path,
    *,
    oracle_arm_dir: str,
    treatment_arm_dir: str,
) -> None:
    missing: list[str] = []
    for label, arm_dir in (("oracle", oracle_arm_dir), ("treatment", treatment_arm_dir)):
        receipt_path = run_root / arm_dir / "receipt.json"
        if not receipt_path.is_file():
            missing.append(f"{label}={receipt_path}")
    if missing:
        raise FileNotFoundError(
            "decision-parity postrun arm receipt preflight failed; missing: "
            + "; ".join(missing)
            + f" (oracle_arm_dir={oracle_arm_dir!r}, treatment_arm_dir={treatment_arm_dir!r})"
        )


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
    if bool(treatment_receipt.get("persistent_accumulator_w5_byte_packed")):
        ledger = treatment_receipt.get("r5_persistent_ledger") or {}
        if not bool(ledger.get("enabled")):
            failures.append("treatment_missing_r5_persistent_ledger")
    return failures


def run_postrun(
    *,
    run_root: Path,
    json_out: Path,
    oracle_arm_dir: str = ORACLE_ARM_DIR,
    treatment_arm_dir: str = TREATMENT_ARM_DIR,
    require_w5_ledger: bool = True,
) -> dict[str, Any]:
    preflight_arm_receipt_dirs(
        run_root,
        oracle_arm_dir=str(oracle_arm_dir),
        treatment_arm_dir=str(treatment_arm_dir),
    )
    oracle_receipt = _load_arm_receipt(run_root, str(oracle_arm_dir))
    treatment_receipt = _load_arm_receipt(run_root, str(treatment_arm_dir))
    harness_failures = _collect_harness_failures(oracle_receipt, treatment_receipt)
    classifier_receipt = emit_s3bb_decision_parity_receipt(
        oracle_receipt,
        treatment_receipt,
        harness_failures=harness_failures,
        require_w5_ledger=bool(require_w5_ledger),
    )
    summary = {
        "slice_id": "w5_decision_parity_run_s3bb_v0",
        "run_root": str(run_root),
        "oracle_arm": str(oracle_arm_dir),
        "treatment_arm": str(treatment_arm_dir),
        "primary_classifier": classifier_receipt["primary_classifier"],
        "harness_failures": list(harness_failures),
        "applied_mask_parity": classifier_receipt["decision_parity_stats"]["applied_mask_parity"],
        "crossing_parity": classifier_receipt["decision_parity_stats"]["crossing_parity"],
    }
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
        description="Emit S3bb dual-arm W5 decision-parity classifier receipts.",
    )
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--oracle-arm-dir", default=ORACLE_ARM_DIR)
    parser.add_argument("--treatment-arm-dir", default=TREATMENT_ARM_DIR)
    parser.add_argument(
        "--no-require-w5-ledger",
        action="store_true",
        help="Skip fail-closed R5 ledger presence check on treatment arm.",
    )
    args = parser.parse_args(argv)
    run_postrun(
        run_root=args.run_root,
        json_out=args.json_out,
        oracle_arm_dir=str(args.oracle_arm_dir),
        treatment_arm_dir=str(args.treatment_arm_dir),
        require_w5_ledger=not bool(args.no_require_w5_ledger),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
