#!/usr/bin/env python3
"""Dual-arm W7 dense-acc in-vivo confirmation postrun classifier."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from calm.hrm_text_158.native_full_stack.s3bb_decision_parity import (
    classify_s3bb_decision_parity_run,
)
from calm.hrm_text_158.native_full_stack.w7_dense_acc_in_vivo_confirmation import (
    classify_w7_in_vivo_dual_arm,
    resolve_confirmation_envelope,
    verify_dual_arm_w7_configuration,
)

ORACLE_ARM_DIR = "int16_oracle_flag_off"
TREATMENT_ARM_DIR = "w7_dense_acc_treatment"


def _load_arm_receipt(run_root: Path, arm_dir: str) -> dict[str, Any]:
    receipt_path = run_root / arm_dir / "receipt.json"
    if not receipt_path.is_file():
        raise FileNotFoundError(f"missing arm receipt: {receipt_path}")
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"arm receipt must be a JSON object: {receipt_path}")
    return payload


def emit_w7_in_vivo_classifier_receipt(
    *,
    run_root: Path,
    oracle_arm_dir: str = ORACLE_ARM_DIR,
    treatment_arm_dir: str = TREATMENT_ARM_DIR,
    observer_too_expensive: bool = False,
) -> dict[str, Any]:
    oracle_receipt = _load_arm_receipt(run_root, oracle_arm_dir)
    treatment_receipt = _load_arm_receipt(run_root, treatment_arm_dir)
    envelope_id = str(oracle_receipt.get("envelope_id") or "")
    envelope = resolve_confirmation_envelope(envelope_id or None)
    parity_primary, parity = classify_s3bb_decision_parity_run(oracle_receipt, treatment_receipt)
    parity_break = str(parity_primary) not in {
        "W6_HEADROOM_SUFFICIENT_PARITY_OK",
        "W5_DECISION_PARITY_OK",
        "PARITY_OK",
    } and bool(
        parity.get("crossing_parity", {}).get("per_step_crossing_bool_disagreement_count", 0)
        or parity.get("applied_mask_parity", {}).get("mismatch_count", 0)
        or parity.get("q_trajectory_parity", {}).get("mismatch_count", 0)
    )
    floor_width, arm_failures = verify_dual_arm_w7_configuration(
        oracle_receipt=oracle_receipt,
        treatment_receipt=treatment_receipt,
    )
    classifier = classify_w7_in_vivo_dual_arm(
        oracle_receipt=oracle_receipt,
        treatment_receipt=treatment_receipt,
        envelope=envelope,
        observer_too_expensive=observer_too_expensive,
        harness_failures=arm_failures,
        parity_break=parity_break,
        confirmed_vote_acc_floor_width=floor_width,
    )
    classifier["run_root"] = str(run_root)
    classifier["oracle_arm_dir"] = oracle_arm_dir
    classifier["treatment_arm_dir"] = treatment_arm_dir
    classifier["s3bb_parity_primary_classifier"] = parity_primary
    classifier["s3bb_parity_receipt"] = parity
    return classifier


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="W7 in-vivo dual-arm postrun classifier")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--oracle-arm-dir", default=ORACLE_ARM_DIR)
    parser.add_argument("--treatment-arm-dir", default=TREATMENT_ARM_DIR)
    parser.add_argument("--observer-too-expensive", action="store_true")
    args = parser.parse_args(argv)
    receipt = emit_w7_in_vivo_classifier_receipt(
        run_root=args.run_root,
        oracle_arm_dir=str(args.oracle_arm_dir),
        treatment_arm_dir=str(args.treatment_arm_dir),
        observer_too_expensive=bool(args.observer_too_expensive),
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(args.json_out), "primary_classifier": receipt["primary_classifier"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
