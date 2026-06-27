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
from calm.hrm_text_158.native_full_stack.w8_dense_acc_in_vivo_confirmation import (
    CLASSIFIER_LIVE_FLOOR_MUCH_BELOW_W8,
    CLASSIFIER_RUN_HEALTH_FAIL,
    CLASSIFIER_W8_BREAKS_LIVE_PARITY,
    CLASSIFIER_W8_IN_VIVO_CONFIRMED,
    CLASSIFIER_W8_IN_VIVO_TRANSPARENT,
    CONFIRMATION_ENVELOPE_CANONICAL_T10_PREREG_V24,
    PREREG_PACKET_W8_BREAKS_PARITY_CITATION,
    STRUCTURAL_REASON_O1_MISSING_EVIDENCE,
    W8_ACCUMULATOR_CLIP_CONTRACT,
    classify_w8_in_vivo_dual_arm,
    derive_w8_parity_inputs,
    extract_w8_parity_signals,
    prereg_o1_o4_adjudicable,
    resolve_confirmation_envelope,
    verify_dual_arm_w8_configuration,
)


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
    vote_abs_max: int = 24,
) -> tuple[dict, dict]:
    oracle_path = tmp_path / "oracle_sidecar.jsonl"
    treatment_path = tmp_path / "treatment_sidecar.jsonl"
    step_reports = {
        str(step): _step_report(step_id=str(step), vote_abs_max=vote_abs_max)
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


def test_w8_domain_bypass_when_treatment_lane_exceeds_w5_not_w8(tmp_path: Path) -> None:
    oracle, treatment = _dual_arm_receipts(
        tmp_path,
        oracle_acc=[5, -9, 10],
        treatment_acc=[5, -9, 16],
    )
    primary, stats = classify_s3bb_decision_parity_run(oracle, treatment)
    assert primary == CLASSIFIER_DOMAIN_OR_HEADROOM_FAIL
    bridge = derive_w8_parity_inputs(
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
    bridge = derive_w8_parity_inputs(
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
    bridge = derive_w8_parity_inputs(
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


def test_contract_emitted_in_bridge(tmp_path: Path) -> None:
    oracle, treatment = _dual_arm_receipts(
        tmp_path,
        oracle_acc=[5, -9, 10],
        treatment_acc=[5, -9, 10],
    )
    primary, stats = classify_s3bb_decision_parity_run(oracle, treatment)
    bridge = derive_w8_parity_inputs(
        primary,
        stats,
        stats["sidecar_coverage_diagnostics"],
        oracle_receipt=oracle,
        treatment_receipt=treatment,
    )
    assert bridge["w8_accumulator_clip_contract"]["contract_id"] == (
        W8_ACCUMULATOR_CLIP_CONTRACT["contract_id"]
    )
    assert bridge["prereg_w8_breaks_parity_citation"] == PREREG_PACKET_W8_BREAKS_PARITY_CITATION


def test_prereg_o1_o4_adjudicable_helper() -> None:
    signals = extract_w8_parity_signals(
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


def test_verify_dual_arm_w8_configuration_flags() -> None:
    floor, failures = verify_dual_arm_w8_configuration(
        oracle_receipt={"dense_accumulator_w8_clip": False},
        treatment_receipt={"dense_accumulator_w8_clip": True},
    )
    assert floor == 8
    assert failures == []
    _, failures = verify_dual_arm_w8_configuration(
        oracle_receipt={"dense_accumulator_w8_clip": True},
        treatment_receipt={"dense_accumulator_w8_clip": True},
    )
    assert "oracle_dense_accumulator_w8_clip_must_be_false" in failures


def test_low_pressure_terminals_do_not_bank_transparency(tmp_path: Path) -> None:
    envelope = resolve_confirmation_envelope(CONFIRMATION_ENVELOPE_CANONICAL_T10_PREREG_V24)
    assert envelope is not None
    oracle, treatment = _dual_arm_receipts(
        tmp_path,
        oracle_acc=[5, -9, 10],
        treatment_acc=[5, -9, 10],
        vote_abs_max=20,
    )
    floor_width, _ = verify_dual_arm_w8_configuration(
        oracle_receipt=oracle,
        treatment_receipt=treatment,
    )
    transparent = classify_w8_in_vivo_dual_arm(
        oracle_receipt=oracle,
        treatment_receipt=treatment,
        envelope=envelope,
        parity_break=False,
        confirmed_vote_acc_floor_width=floor_width,
        oracle_max_sidecar_abs=10,
        treatment_max_sidecar_abs=10,
    )
    assert transparent["primary_classifier"] == CLASSIFIER_W8_IN_VIVO_TRANSPARENT
    assert transparent["banks_w8_transparency"] is False
    assert transparent["banks_w8_carrier_faithfulness"] is False
    assert transparent["informative_only"] is True
    # peak = 10-1+20 = 29 -> LIVE_FLOOR_MUCH_BELOW_W8 when sidecar abs unknown
    below = classify_w8_in_vivo_dual_arm(
        oracle_receipt=oracle,
        treatment_receipt=treatment,
        envelope=envelope,
        parity_break=False,
        confirmed_vote_acc_floor_width=floor_width,
        oracle_max_sidecar_abs=None,
        treatment_max_sidecar_abs=None,
    )
    assert below["primary_classifier"] == CLASSIFIER_LIVE_FLOOR_MUCH_BELOW_W8
    assert below["banks_w8_transparency"] is False
    assert below["banks_w8_carrier_faithfulness"] is False


def test_confirmed_banks_transparency_at_peak33(tmp_path: Path) -> None:
    envelope = resolve_confirmation_envelope(CONFIRMATION_ENVELOPE_CANONICAL_T10_PREREG_V24)
    assert envelope is not None
    oracle, treatment = _dual_arm_receipts(
        tmp_path,
        oracle_acc=[5, -9, 10],
        treatment_acc=[5, -9, 10],
        vote_abs_max=24,
    )
    floor_width, _ = verify_dual_arm_w8_configuration(
        oracle_receipt=oracle,
        treatment_receipt=treatment,
    )
    confirmed = classify_w8_in_vivo_dual_arm(
        oracle_receipt=oracle,
        treatment_receipt=treatment,
        envelope=envelope,
        parity_break=False,
        confirmed_vote_acc_floor_width=floor_width,
        oracle_max_sidecar_abs=100,
        treatment_max_sidecar_abs=100,
    )
    assert confirmed["primary_classifier"] == CLASSIFIER_W8_IN_VIVO_CONFIRMED
    assert confirmed["banks_w8_transparency"] is True
    assert confirmed["banks_w8_carrier_faithfulness"] is True
    assert confirmed["eager_tier_rules_unlock"] is False


def test_breaks_terminal_does_not_bank(tmp_path: Path) -> None:
    envelope = resolve_confirmation_envelope(CONFIRMATION_ENVELOPE_CANONICAL_T10_PREREG_V24)
    assert envelope is not None
    oracle, treatment = _dual_arm_receipts(
        tmp_path,
        oracle_acc=[5, -9, 10],
        treatment_acc=[5, -9, 10],
    )
    floor_width, _ = verify_dual_arm_w8_configuration(
        oracle_receipt=oracle,
        treatment_receipt=treatment,
    )
    breaks = classify_w8_in_vivo_dual_arm(
        oracle_receipt=oracle,
        treatment_receipt=treatment,
        envelope=envelope,
        parity_break=True,
        confirmed_vote_acc_floor_width=floor_width,
    )
    assert breaks["primary_classifier"] == CLASSIFIER_W8_BREAKS_LIVE_PARITY
    assert breaks["banks_w8_transparency"] is False
    assert breaks["banks_w8_carrier_faithfulness"] is False


def test_probe_w8_flag_mutual_exclusion() -> None:
    from scripts.hrm_text_158_bounded_delta_acquisition_probe import run_c2p1_probe

    with pytest.raises(ValueError, match="mutually exclusive"):
        run_c2p1_probe(
            parent=Path("calm/hrm/checkpoints/dummy.pt"),
            parent_sha256="0" * 64,
            scratch_root=Path("/tmp/w8_mutual_exclusion"),
            phase="w8-dense-acc-in-vivo-confirmation",
            device="cpu",
            eligible_scope="all-bitlinear",
            steps=1,
            batch_size=1,
            max_steps_hard=1,
            dense_accumulator_w7_clip=True,
            dense_accumulator_w8_clip=True,
            enabled=True,
        )
