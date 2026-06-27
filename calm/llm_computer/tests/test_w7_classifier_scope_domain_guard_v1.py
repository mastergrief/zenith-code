from __future__ import annotations

import copy
from pathlib import Path

import pytest

from calm.hrm_text_158.native_full_stack.s3bb_decision_parity import (
    CLASSIFIER_DOMAIN_OR_HEADROOM_FAIL,
    classify_s3bb_decision_parity_run,
)
from calm.hrm_text_158.native_full_stack.s3bb_headroom_telemetry import (
    MEASURED_STEPS_REQUIRED,
    append_headroom_wiring_sidecar_chunk,
)
from calm.hrm_text_158.native_full_stack.w7_dense_acc_in_vivo_confirmation import (
    CLASSIFIER_RUN_HEALTH_FAIL,
    CLASSIFIER_W7_BREAKS_LIVE_PARITY,
    CONFIRMATION_ENVELOPE_CANONICAL_T10_PREREG_V24,
    DIVERGENCE_CHARACTERIZATION_W7_READ_CLAMP_ASYMMETRY,
    DIVERGENCE_ONSET_STATE_KEY,
    PREREG_PACKET_W7_BREAKS_PARITY_CITATION,
    STRUCTURAL_REASON_O1_MISSING_EVIDENCE,
    W7_ACCUMULATOR_CLIP_CONTRACT_C,
    derive_w7_parity_inputs,
    extract_w7_parity_signals,
    prereg_o1_o4_adjudicable,
)
from scripts.hrm_text_158_w7_dense_acc_in_vivo_postrun import emit_w7_in_vivo_classifier_receipt


def _step_report(*, step_id: str, vote_abs_max: int = 24) -> dict:
    return {
        "step_id": step_id,
        "q_changed_count": 2,
        "metrics": {"loss": 0.5, "accuracy": 0.25},
        "vote_pressure": {"tiny.proj": {"vote_abs_max": int(vote_abs_max)}},
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
) -> tuple[dict, dict]:
    oracle_path = tmp_path / "oracle_sidecar.jsonl"
    treatment_path = tmp_path / "treatment_sidecar.jsonl"
    step_reports = {
        str(step): _step_report(step_id=str(step), vote_abs_max=24)
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
        "dense_accumulator_w7_clip": False,
        "step_reports": copy.deepcopy(step_reports),
        "headroom_wiring_sidecar_path": str(oracle_path),
    }
    treatment = {
        "steps_completed": MEASURED_STEPS_REQUIRED,
        "stop_reason": "",
        "envelope_id": CONFIRMATION_ENVELOPE_CANONICAL_T10_PREREG_V24,
        "dense_accumulator_w7_clip": True,
        "step_reports": copy.deepcopy(step_reports),
        "headroom_wiring_sidecar_path": str(treatment_path),
    }
    return oracle, treatment


def test_w7_domain_bypass_when_treatment_lane_exceeds_w5_not_w7(tmp_path: Path) -> None:
    oracle, treatment = _dual_arm_receipts(
        tmp_path,
        oracle_acc=[5, -9, 10],
        treatment_acc=[5, -9, 16],
    )
    primary, stats = classify_s3bb_decision_parity_run(oracle, treatment)
    assert primary == CLASSIFIER_DOMAIN_OR_HEADROOM_FAIL
    bridge = derive_w7_parity_inputs(
        primary,
        stats,
        stats["sidecar_coverage_diagnostics"],
        oracle_receipt=oracle,
        treatment_receipt=treatment,
    )
    assert bridge["s3bb_w5w6_domain_primary_inapplicable"] is True
    assert bridge["structural_fail"] is False
    assert bridge["s3bb_w5w6_domain_primary_recorded"] == CLASSIFIER_DOMAIN_OR_HEADROOM_FAIL


def test_vacuous_o1_without_o2_o4_signals_is_run_health_fail(tmp_path: Path) -> None:
    oracle, treatment = _dual_arm_receipts(
        tmp_path,
        oracle_acc=[5, -9, 10],
        treatment_acc=[5, -9, 10],
    )
    primary, stats = classify_s3bb_decision_parity_run(oracle, treatment)
    stats = dict(stats)
    stats["bit_equality_diagnostics"] = {
        "total_lane_count": 0,
        "vote_update_state_accumulator_equality_rate": 1.0,
        "sidecar_coverage_diagnostics": stats["sidecar_coverage_diagnostics"],
    }
    stats["crossing_parity"] = {
        "per_step_crossing_bool_disagreement_count": 0,
        "total_lane_count": 0,
    }
    stats["applied_mask_parity"] = {
        "applied_mask_mismatch_count": 0,
        "applied_mask_parity_pass": True,
    }
    stats["q_trajectory_parity"] = {
        "q_sha256_after_mismatch_count": 0,
        "q_trajectory_parity_pass": True,
        "final_metrics_mismatch": False,
    }
    bridge = derive_w7_parity_inputs(
        primary,
        stats,
        stats["sidecar_coverage_diagnostics"],
        oracle_receipt=oracle,
        treatment_receipt=treatment,
    )
    assert bridge["o1_lane_equality_vacuous"] is True
    assert bridge["structural_fail"] is True
    assert bridge["structural_reason"] == STRUCTURAL_REASON_O1_MISSING_EVIDENCE


def test_vacuous_o1_with_applied_mask_mismatch_allows_breaks(tmp_path: Path) -> None:
    oracle, treatment = _dual_arm_receipts(
        tmp_path,
        oracle_acc=[5, -9, 10],
        treatment_acc=[5, -9, 10],
    )
    treatment["step_reports"]["4"]["step_result"]["tensor_stats"]["tiny.proj"]["applied_indices"] = [
        1
    ]
    primary, stats = classify_s3bb_decision_parity_run(oracle, treatment)
    stats = dict(stats)
    stats["bit_equality_diagnostics"] = {
        "total_lane_count": 0,
        "vote_update_state_accumulator_equality_rate": 1.0,
        "sidecar_coverage_diagnostics": stats["sidecar_coverage_diagnostics"],
    }
    bridge = derive_w7_parity_inputs(
        primary,
        stats,
        stats["sidecar_coverage_diagnostics"],
        oracle_receipt=oracle,
        treatment_receipt=treatment,
    )
    assert bridge["o1_lane_equality_vacuous"] is True
    assert bridge["prereg_o1_o4_adjudicable"] is True
    assert bridge["structural_fail"] is False
    assert bridge["parity_break"] is True
    assert "applied_mask_mismatch_count" in bridge["parity_break_driving_keys"]


def test_contract_c_emitted_in_bridge(tmp_path: Path) -> None:
    oracle, treatment = _dual_arm_receipts(
        tmp_path,
        oracle_acc=[5, -9, 10],
        treatment_acc=[5, -9, 10],
    )
    primary, stats = classify_s3bb_decision_parity_run(oracle, treatment)
    bridge = derive_w7_parity_inputs(
        primary,
        stats,
        stats["sidecar_coverage_diagnostics"],
        oracle_receipt=oracle,
        treatment_receipt=treatment,
    )
    assert bridge["w7_accumulator_clip_contract"]["contract_id"] == (
        W7_ACCUMULATOR_CLIP_CONTRACT_C["contract_id"]
    )
    assert bridge["prereg_w7_breaks_parity_citation"] == PREREG_PACKET_W7_BREAKS_PARITY_CITATION


def test_prereg_o1_o4_adjudicable_helper() -> None:
    signals = extract_w7_parity_signals(
        {
            "crossing_parity": {"per_step_crossing_bool_disagreement_count": 0},
            "applied_mask_parity": {"applied_mask_mismatch_count": 1},
            "q_trajectory_parity": {"q_sha256_after_mismatch_count": 0},
        }
    )
    assert prereg_o1_o4_adjudicable(
        o1_lane_witness={"o1_lane_equality_load_bearing": False},
        parity_signals=signals,
    )


@pytest.mark.skipif(
    not Path(
        "/home/gabe/claw-code-creditdir/transient_fp_credit/"
        "w7_dense_acc_in_vivo_seed43_43_2189e72008/int16_oracle_flag_off/receipt.json"
    ).is_file(),
    reason="2189e72008 run artifacts not present on this host",
)
def test_reclassify_2189e72008_predicate_derived_breaks(tmp_path: Path) -> None:
    run_root = Path(
        "/home/gabe/claw-code-creditdir/transient_fp_credit/"
        "w7_dense_acc_in_vivo_seed43_43_2189e72008"
    )
    receipt = emit_w7_in_vivo_classifier_receipt(run_root=run_root)
    bridge = receipt["w7_parity_bridge"]
    assert bridge["s3bb_w5w6_domain_primary_inapplicable"] is True
    assert bridge["o1_lane_equality_vacuous"] is True
    assert bridge["prereg_o1_o4_adjudicable"] is True
    assert bridge["structural_fail"] is False
    assert receipt["parity_break"] is True
    assert receipt["primary_classifier"] == CLASSIFIER_W7_BREAKS_LIVE_PARITY
    divergence = receipt["divergence_characterization"]
    assert divergence is not None
    assert divergence["focus_state_key"] == DIVERGENCE_ONSET_STATE_KEY
    assert divergence["characterization"] == DIVERGENCE_CHARACTERIZATION_W7_READ_CLAMP_ASYMMETRY
    assert divergence["onset_step"] == 4
    onset = next(row for row in divergence["per_step_series"] if int(row["step"]) == 4)
    assert int(onset["diff_count"]) == 28914
    terminal = next(row for row in divergence["per_step_series"] if int(row["step"]) == 10)
    assert int(terminal["diff_count"]) == 968935
    assert int(terminal["oracle_max_abs"]) == 127
    assert int(terminal["treatment_max_abs"]) == 87
    assert receipt["primary_classifier"] != CLASSIFIER_RUN_HEALTH_FAIL
    out_path = tmp_path / "reclassify_2189e72008.json"
    out_path.write_text("{}", encoding="utf-8")
