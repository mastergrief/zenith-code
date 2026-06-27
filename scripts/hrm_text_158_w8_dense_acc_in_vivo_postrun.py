#!/usr/bin/env python3
"""Dual-arm W8 dense-acc in-vivo confirmation postrun classifier."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from calm.hrm_text_158.native_full_stack.s3bb_decision_parity import (
    classify_s3bb_decision_parity_run,
)
from calm.hrm_text_158.native_full_stack.w8_dense_acc_in_vivo_confirmation import (
    classify_w8_in_vivo_dual_arm,
    derive_w8_parity_inputs,
    resolve_confirmation_envelope,
    verify_dual_arm_w8_configuration,
)

ORACLE_ARM_DIR = "int16_oracle_flag_off"
TREATMENT_ARM_DIR = "w8_dense_acc_treatment"


def _load_arm_receipt(run_root: Path, arm_dir: str) -> dict[str, Any]:
    receipt_path = run_root / arm_dir / "receipt.json"
    if not receipt_path.is_file():
        raise FileNotFoundError(f"missing arm receipt: {receipt_path}")
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"arm receipt must be a JSON object: {receipt_path}")
    return payload


def emit_w8_in_vivo_classifier_receipt(
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
    parity_primary, parity_stats = classify_s3bb_decision_parity_run(
        oracle_receipt,
        treatment_receipt,
    )
    sidecar_coverage = parity_stats.get("sidecar_coverage_diagnostics") or {}
    parity_inputs = derive_w8_parity_inputs(
        parity_primary,
        parity_stats,
        sidecar_coverage,
        oracle_receipt=oracle_receipt,
        treatment_receipt=treatment_receipt,
    )
    floor_width, arm_failures = verify_dual_arm_w8_configuration(
        oracle_receipt=oracle_receipt,
        treatment_receipt=treatment_receipt,
    )
    classifier = classify_w8_in_vivo_dual_arm(
        oracle_receipt=oracle_receipt,
        treatment_receipt=treatment_receipt,
        envelope=envelope,
        observer_too_expensive=observer_too_expensive,
        harness_failures=arm_failures,
        parity_break=bool(parity_inputs["parity_break"]),
        structural_fail=bool(parity_inputs["structural_fail"]),
        structural_reason=parity_inputs.get("structural_reason"),
        confirmed_vote_acc_floor_width=floor_width,
        oracle_max_sidecar_abs=parity_inputs.get("oracle_max_sidecar_abs"),
        treatment_max_sidecar_abs=parity_inputs.get("treatment_max_sidecar_abs"),
    )
    classifier["run_root"] = str(run_root)
    classifier["oracle_arm_dir"] = oracle_arm_dir
    classifier["treatment_arm_dir"] = treatment_arm_dir
    classifier["s3bb_parity_primary_classifier"] = parity_primary
    classifier["s3bb_parity_receipt"] = parity_stats
    classifier["w8_parity_bridge"] = parity_inputs
    classifier["sidecar_coverage_diagnostics"] = parity_inputs.get(
        "sidecar_coverage_diagnostics"
    )
    classifier["w8_accumulator_clip_contract"] = parity_inputs.get("w8_accumulator_clip_contract")
    classifier["o1_lane_equality_vacuous"] = parity_inputs.get("o1_lane_equality_vacuous")
    classifier["o1_lane_equality_load_bearing"] = parity_inputs.get("o1_lane_equality_load_bearing")
    classifier["prereg_o1_o4_adjudicable"] = parity_inputs.get("prereg_o1_o4_adjudicable")
    classifier["prereg_w8_breaks_parity_citation"] = parity_inputs.get(
        "prereg_w8_breaks_parity_citation"
    )
    classifier["s3bb_w5w6_domain_primary_inapplicable"] = parity_inputs.get(
        "s3bb_w5w6_domain_primary_inapplicable"
    )
    classifier["s3bb_w5w6_domain_primary_recorded"] = parity_inputs.get(
        "s3bb_w5w6_domain_primary_recorded"
    )
    return classifier


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="W8 in-vivo dual-arm postrun classifier")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--oracle-arm-dir", default=ORACLE_ARM_DIR)
    parser.add_argument("--treatment-arm-dir", default=TREATMENT_ARM_DIR)
    parser.add_argument("--observer-too-expensive", action="store_true")
    args = parser.parse_args(argv)
    receipt = emit_w8_in_vivo_classifier_receipt(
        run_root=args.run_root,
        oracle_arm_dir=str(args.oracle_arm_dir),
        treatment_arm_dir=str(args.treatment_arm_dir),
        observer_too_expensive=bool(args.observer_too_expensive),
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.json_out),
                "primary_classifier": receipt["primary_classifier"],
                "banks_w8_transparency": receipt.get("banks_w8_transparency"),
                "structural_fail": receipt.get("structural_fail"),
                "parity_break": receipt.get("parity_break"),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
