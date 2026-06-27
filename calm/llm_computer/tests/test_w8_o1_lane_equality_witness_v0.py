from __future__ import annotations

import copy
from pathlib import Path

import pytest

from calm.hrm_text_158.native_full_stack.s3bb_headroom_telemetry import (
    MEASURED_STEPS_REQUIRED,
    WARMUP_STEPS,
    append_headroom_wiring_sidecar_chunk,
)
from calm.hrm_text_158.native_full_stack.w8_dense_acc_in_vivo_confirmation import (
    CLASSIFIER_W8_BREAKS_LIVE_PARITY,
    CLASSIFIER_W8_IN_VIVO_CONFIRMED,
    CONFIRMATION_ENVELOPE_CANONICAL_T10_PREREG_V24,
    derive_w8_parity_inputs,
    resolve_confirmation_envelope,
    verify_dual_arm_w8_configuration,
)
from calm.hrm_text_158.native_full_stack.w8_o1_lane_equality_witness import (
    O1_SKIP_POLICY,
    O1_WITNESS_DOMAIN,
    STRUCTURAL_REASON_W8_LANE_OUT_OF_DOMAIN,
    compare_w8_o1_lane_equality_streaming,
)
from scripts.hrm_text_158_w8_dense_acc_in_vivo_postrun import emit_w8_in_vivo_classifier_receipt


def _step_report(*, step_id: str, vote_abs_max: int = 24, would_strict_raise: bool = False) -> dict:
    return {
        "step_id": step_id,
        "q_changed_count": 2,
        "metrics": {"loss": 0.5, "accuracy": 0.25},
        "vote_pressure": {"tiny.proj": {"vote_abs_max": int(vote_abs_max)}},
        "headroom_telemetry": {
            "would_strict_raise_step": bool(would_strict_raise),
            "out_of_domain_lane_count": 1 if would_strict_raise else 0,
        },
        "step_result": {
            "tensor_stats": {
                "tiny.proj": {
                    "applied_indices": [0, 2],
                    "q_sha256_after": "same_sha",
                    "flip_count": 2,
                }
            }
        },
    }


def _dual_arm_receipts(
    tmp_path: Path,
    *,
    oracle_acc: list[int],
    treatment_acc: list[int],
    vote_abs_max: int = 24,
    would_strict_raise: bool = False,
) -> tuple[dict, dict]:
    oracle_path = tmp_path / "oracle_sidecar.jsonl"
    treatment_path = tmp_path / "treatment_sidecar.jsonl"
    step_reports = {
        str(step): _step_report(
            step_id=str(step),
            vote_abs_max=vote_abs_max,
            would_strict_raise=would_strict_raise,
        )
        for step in range(1, MEASURED_STEPS_REQUIRED + 1)
    }
    for step in range(1, MEASURED_STEPS_REQUIRED + 1):
        append_headroom_wiring_sidecar_chunk(
            oracle_path,
            step=step,
            state_key="tiny.proj",
            accumulator_lanes=oracle_acc,
            q_lanes=[0, 1, -1],
        )
        append_headroom_wiring_sidecar_chunk(
            treatment_path,
            step=step,
            state_key="tiny.proj",
            accumulator_lanes=treatment_acc,
            q_lanes=[0, 1, -1],
        )
    oracle = {
        "steps_completed": MEASURED_STEPS_REQUIRED,
        "stop_reason": "",
        "envelope_id": CONFIRMATION_ENVELOPE_CANONICAL_T10_PREREG_V24,
        "dense_accumulator_w8_clip": False,
        "step_reports": copy.deepcopy(step_reports),
        "headroom_wiring_sidecar_path": str(oracle_path),
    }
    treatment = {
        "steps_completed": MEASURED_STEPS_REQUIRED,
        "stop_reason": "",
        "envelope_id": CONFIRMATION_ENVELOPE_CANONICAL_T10_PREREG_V24,
        "dense_accumulator_w8_clip": True,
        "step_reports": copy.deepcopy(step_reports),
        "headroom_wiring_sidecar_path": str(treatment_path),
    }
    return oracle, treatment


def _clean_s3bb_stats(sidecar_coverage: dict) -> dict:
    return {
        "sidecar_coverage_diagnostics": sidecar_coverage,
        "bit_equality_diagnostics": {
            "total_lane_count": 0,
            "vote_update_state_accumulator_equality_rate": 1.0,
            "sidecar_coverage_diagnostics": sidecar_coverage,
        },
        "crossing_parity": {
            "per_step_crossing_bool_disagreement_count": 0,
            "total_lane_count": 1,
        },
        "applied_mask_parity": {
            "applied_mask_mismatch_count": 0,
            "applied_mask_parity_pass": True,
        },
        "q_trajectory_parity": {
            "q_sha256_after_mismatch_count": 0,
            "q_trajectory_parity_pass": True,
            "final_metrics_mismatch": False,
        },
    }


def test_w8_o1_non_vacuous_when_w6_strict_raise_but_lanes_within_w8_domain(
    tmp_path: Path,
) -> None:
    oracle, treatment = _dual_arm_receipts(
        tmp_path,
        oracle_acc=[24, 48, 72],
        treatment_acc=[24, 48, 72],
        would_strict_raise=True,
    )
    stats = compare_w8_o1_lane_equality_streaming(
        oracle,
        treatment,
        oracle_sidecar_path=oracle["headroom_wiring_sidecar_path"],
        treatment_sidecar_path=treatment["headroom_wiring_sidecar_path"],
    )
    assert stats["o1_witness_domain"] == O1_WITNESS_DOMAIN
    assert stats["o1_skip_policy"] == O1_SKIP_POLICY
    assert int(stats["total_lane_count"]) > 0
    assert float(stats["vote_update_state_accumulator_equality_rate"]) == 1.0
    assert "would_strict_raise" not in str(stats)


def test_w8_o1_structural_fail_when_lane_outside_w8_domain(tmp_path: Path) -> None:
    oracle, treatment = _dual_arm_receipts(
        tmp_path,
        oracle_acc=[24, 48, 128],
        treatment_acc=[24, 48, 72],
    )
    stats = compare_w8_o1_lane_equality_streaming(
        oracle,
        treatment,
        oracle_sidecar_path=oracle["headroom_wiring_sidecar_path"],
        treatment_sidecar_path=treatment["headroom_wiring_sidecar_path"],
    )
    assert stats["structural_fail"] is True
    assert stats["structural_reason"] == STRUCTURAL_REASON_W8_LANE_OUT_OF_DOMAIN


def test_w8_o1_inequality_flows_to_breaks_via_bridge(tmp_path: Path) -> None:
    oracle, treatment = _dual_arm_receipts(
        tmp_path,
        oracle_acc=[24, 48, 72],
        treatment_acc=[24, 48, 87],
    )
    primary = "DECISION_PARITY_OK"
    sidecar_coverage = {
        "structural_fail": False,
        "shared_key_count": 10,
        "oracle_row_count": 10,
        "treatment_row_count": 10,
    }
    stats = _clean_s3bb_stats(sidecar_coverage)
    bridge = derive_w8_parity_inputs(
        primary,
        stats,
        sidecar_coverage,
        oracle_receipt=oracle,
        treatment_receipt=treatment,
    )
    assert bridge["o1_lane_equality_load_bearing"] is True
    assert bridge["parity_break"] is True
    assert "o1_accumulator_lane_inequality" in bridge["parity_break_driving_keys"]


def test_w8_o1_equality_confirm_eligible_via_postrun_end_to_end(tmp_path: Path) -> None:
    oracle, treatment = _dual_arm_receipts(
        tmp_path,
        oracle_acc=[24, 48, 72],
        treatment_acc=[24, 48, 72],
        would_strict_raise=True,
    )
    run_root = tmp_path / "run"
    oracle_dir = run_root / "int16_oracle_flag_off"
    treatment_dir = run_root / "w8_dense_acc_treatment"
    oracle_dir.mkdir(parents=True)
    treatment_dir.mkdir(parents=True)
    import json

    (oracle_dir / "receipt.json").write_text(json.dumps(oracle), encoding="utf-8")
    (treatment_dir / "receipt.json").write_text(json.dumps(treatment), encoding="utf-8")
    receipt = emit_w8_in_vivo_classifier_receipt(run_root=run_root)
    assert receipt["o1_lane_equality_load_bearing"] is True
    assert receipt["o1_lane_equality_vacuous"] is False
    assert receipt["primary_classifier"] == CLASSIFIER_W8_IN_VIVO_CONFIRMED
    assert receipt["banks_w8_transparency"] is True


def test_w8_o1_inequality_breaks_via_postrun_end_to_end(tmp_path: Path) -> None:
    oracle, treatment = _dual_arm_receipts(
        tmp_path,
        oracle_acc=[24, 48, 72],
        treatment_acc=[24, 48, 87],
    )
    run_root = tmp_path / "run_break"
    oracle_dir = run_root / "int16_oracle_flag_off"
    treatment_dir = run_root / "w8_dense_acc_treatment"
    oracle_dir.mkdir(parents=True)
    treatment_dir.mkdir(parents=True)
    import json

    (oracle_dir / "receipt.json").write_text(json.dumps(oracle), encoding="utf-8")
    (treatment_dir / "receipt.json").write_text(json.dumps(treatment), encoding="utf-8")
    receipt = emit_w8_in_vivo_classifier_receipt(run_root=run_root)
    assert receipt["primary_classifier"] == CLASSIFIER_W8_BREAKS_LIVE_PARITY
    assert receipt["banks_w8_transparency"] is False
    assert "o1_accumulator_lane_inequality" in receipt["w8_parity_bridge"]["parity_break_driving_keys"]


def test_w8_o1_witness_does_not_reference_w6_strict_raise_skip() -> None:
    import inspect

    source = inspect.getsource(compare_w8_o1_lane_equality_streaming)
    assert "_wiring_guard_skip_steps" not in source
    assert "would_strict_raise" not in source
