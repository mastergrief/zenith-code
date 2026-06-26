"""CPU tests for W7 in-vivo confirmation envelope wiring."""
from __future__ import annotations

import pytest

from calm.hrm_text_158.native_full_stack.accumulator_real_dynamics_verdict import (
    default_vote_update_spec as canonical_acquisition_vote_update_spec,
)
from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    canonical_acquisition_peak_reachable,
    canonical_acquisition_rank_vote_spec,
    default_dry_run_rank_vote_spec,
    dry_run_rank_vote_peak_reachable,
    max_vote_abs_for_rank_spec,
)
from calm.hrm_text_158.native_full_stack.w7_dense_acc_in_vivo_confirmation import (
    CLASSIFIER_RUN_HEALTH_FAIL,
    CLASSIFIER_W7_IN_VIVO_CONFIRMED,
    CONFIRMATION_ENVELOPE_CANONICAL_T10_PREREG_V24,
    CPU_PHASE0_REACHABLE_PEAK,
    W7_IN_VIVO_CONFIRMED_MIN_PEAK,
    clip_table_for_peak,
    classify_w7_in_vivo_dual_arm,
    resolve_confirmation_envelope,
    verify_dual_arm_w7_configuration,
)


def _dual_arm_receipts(
    *,
    envelope,
    vote_abs_max: int,
    oracle_w7: bool = False,
    treatment_w7: bool = True,
) -> tuple[dict, dict]:
    step_reports = {
        "1": {
            "vote_pressure": {
                "mod.a": {"vote_abs_max": int(vote_abs_max)},
            }
        }
    }
    oracle = {
        **envelope.receipt_fields(),
        "dense_accumulator_w7_clip": bool(oracle_w7),
        "persistent_accumulator_w5_byte_packed": False,
        "persistent_accumulator_w6_byte_packed": False,
        "persistent_accumulator_event_coded_live": False,
        "step_reports": step_reports,
    }
    treatment = {
        **envelope.receipt_fields(),
        "dense_accumulator_w7_clip": bool(treatment_w7),
        "persistent_accumulator_w5_byte_packed": False,
        "persistent_accumulator_w6_byte_packed": False,
        "persistent_accumulator_event_coded_live": False,
        "step_reports": step_reports,
    }
    return oracle, treatment


def test_dry_run_envelope_peak_is_four_not_thirty_three() -> None:
    assert dry_run_rank_vote_peak_reachable(threshold_abs=1) == 4
    assert max_vote_abs_for_rank_spec(default_dry_run_rank_vote_spec()) == 4
    assert canonical_acquisition_peak_reachable(threshold_abs=10) == 33


def test_canonical_envelope_wires_t10_and_prereg_vote_bins() -> None:
    envelope = resolve_confirmation_envelope(CONFIRMATION_ENVELOPE_CANONICAL_T10_PREREG_V24)
    assert envelope is not None
    assert envelope.live_threshold_abs == 10
    assert envelope.wired_max_vote_abs == 24
    assert envelope.wired_reachable_peak_estimate == 33
    assert envelope.confirmation_vote_path == "rank_bucketed_acquisition_canonical"
    spec = envelope.vote_update_spec(max_abs_per_tensor=4096)
    canonical = canonical_acquisition_vote_update_spec()
    assert spec.threshold_abs == canonical.threshold_abs == 10
    rank_spec = canonical_acquisition_rank_vote_spec()
    assert max_vote_abs_for_rank_spec(rank_spec) == 24


def test_clip_table_against_peak_thirty_three() -> None:
    table = clip_table_for_peak(reachable_peak=CPU_PHASE0_REACHABLE_PEAK)
    assert table["W5"]["clip_abs"] == 15
    assert table["W6"]["clip_abs"] == 31
    assert table["W7"]["clip_abs"] == 63
    assert table["W5"]["clips_reachable_peak"] is True
    assert table["W6"]["clips_reachable_peak"] is True
    assert table["W7"]["clips_reachable_peak"] is False
    assert table["W4"]["clip_abs"] == 7
    assert table["W4"]["clips_reachable_peak"] is True


def test_w7_in_vivo_confirmed_requires_floor_seven_and_peak_thirty_three() -> None:
    envelope = resolve_confirmation_envelope(CONFIRMATION_ENVELOPE_CANONICAL_T10_PREREG_V24)
    assert envelope is not None
    oracle_receipt, treatment_receipt = _dual_arm_receipts(
        envelope=envelope,
        vote_abs_max=24,
    )
    floor_width, arm_failures = verify_dual_arm_w7_configuration(
        oracle_receipt=oracle_receipt,
        treatment_receipt=treatment_receipt,
    )
    assert floor_width == 7
    assert arm_failures == []
    result = classify_w7_in_vivo_dual_arm(
        oracle_receipt=oracle_receipt,
        treatment_receipt=treatment_receipt,
        envelope=envelope,
        parity_break=False,
        confirmed_vote_acc_floor_width=floor_width,
    )
    assert result["primary_classifier"] == CLASSIFIER_W7_IN_VIVO_CONFIRMED
    assert result["rules_promotion_unlock_predicate"] is True
    assert result["live_reachable_peak_estimate"] >= W7_IN_VIVO_CONFIRMED_MIN_PEAK
    assert result["confirmed_vote_acc_floor_width"] == 7
    assert result["live_max_vote_abs_observed"] == 24
    assert result["live_max_vote_abs_source"] == "step_reports.vote_pressure"
    assert result["observed_step_count"] == 1
    assert result["wired_max_vote_abs"] == 24
    assert result["native_loop_injection_confirmation"] == "pending"


def test_under_pressed_dry_run_peak_does_not_confirm_w7() -> None:
    envelope = resolve_confirmation_envelope(CONFIRMATION_ENVELOPE_CANONICAL_T10_PREREG_V24)
    assert envelope is not None
    oracle_receipt, treatment_receipt = _dual_arm_receipts(
        envelope=envelope,
        vote_abs_max=4,
    )
    floor_width, _ = verify_dual_arm_w7_configuration(
        oracle_receipt=oracle_receipt,
        treatment_receipt=treatment_receipt,
    )
    result = classify_w7_in_vivo_dual_arm(
        oracle_receipt=oracle_receipt,
        treatment_receipt=treatment_receipt,
        envelope=envelope,
        parity_break=False,
        confirmed_vote_acc_floor_width=floor_width,
    )
    assert result["primary_classifier"] != CLASSIFIER_W7_IN_VIVO_CONFIRMED
    assert result["rules_promotion_unlock_predicate"] is False


def test_configured_rank_bins_without_observed_pressure_does_not_confirm() -> None:
    envelope = resolve_confirmation_envelope(CONFIRMATION_ENVELOPE_CANONICAL_T10_PREREG_V24)
    assert envelope is not None
    oracle_receipt = {
        **envelope.receipt_fields(),
        "dense_accumulator_w7_clip": False,
        "persistent_accumulator_w5_byte_packed": False,
        "persistent_accumulator_w6_byte_packed": False,
        "persistent_accumulator_event_coded_live": False,
        "updater_config": {
            "rank_vote_spec": {
                "rank_bins": [
                    {"vote_abs": 24},
                    {"vote_abs": 16},
                ]
            }
        },
        "step_reports": {},
    }
    treatment_receipt = {
        **oracle_receipt,
        "dense_accumulator_w7_clip": True,
    }
    floor_width, arm_failures = verify_dual_arm_w7_configuration(
        oracle_receipt=oracle_receipt,
        treatment_receipt=treatment_receipt,
    )
    assert floor_width == 7
    assert arm_failures == []
    result = classify_w7_in_vivo_dual_arm(
        oracle_receipt=oracle_receipt,
        treatment_receipt=treatment_receipt,
        envelope=envelope,
        parity_break=False,
        confirmed_vote_acc_floor_width=floor_width,
    )
    assert result["primary_classifier"] == CLASSIFIER_RUN_HEALTH_FAIL
    assert "oracle_missing_observed_vote_pressure" in result["harness_failures"]
    assert "treatment_missing_observed_vote_pressure" in result["harness_failures"]
    assert result["live_max_vote_abs_observed"] is None
    assert result["live_max_vote_abs_source"] == "none"
    assert result["rules_promotion_unlock_predicate"] is False


def test_treatment_w7_flag_absent_does_not_confirm() -> None:
    envelope = resolve_confirmation_envelope(CONFIRMATION_ENVELOPE_CANONICAL_T10_PREREG_V24)
    assert envelope is not None
    oracle_receipt, treatment_receipt = _dual_arm_receipts(
        envelope=envelope,
        vote_abs_max=24,
        treatment_w7=False,
    )
    floor_width, arm_failures = verify_dual_arm_w7_configuration(
        oracle_receipt=oracle_receipt,
        treatment_receipt=treatment_receipt,
    )
    assert floor_width is None
    assert "treatment_dense_accumulator_w7_clip_must_be_true" in arm_failures
    result = classify_w7_in_vivo_dual_arm(
        oracle_receipt=oracle_receipt,
        treatment_receipt=treatment_receipt,
        envelope=envelope,
        parity_break=False,
        confirmed_vote_acc_floor_width=floor_width,
        harness_failures=arm_failures,
    )
    assert result["primary_classifier"] == CLASSIFIER_RUN_HEALTH_FAIL
    assert result["confirmed_vote_acc_floor_width"] is None
    assert result["rules_promotion_unlock_predicate"] is False


def test_oracle_w7_on_does_not_confirm() -> None:
    envelope = resolve_confirmation_envelope(CONFIRMATION_ENVELOPE_CANONICAL_T10_PREREG_V24)
    assert envelope is not None
    oracle_receipt, treatment_receipt = _dual_arm_receipts(
        envelope=envelope,
        vote_abs_max=24,
        oracle_w7=True,
        treatment_w7=True,
    )
    floor_width, arm_failures = verify_dual_arm_w7_configuration(
        oracle_receipt=oracle_receipt,
        treatment_receipt=treatment_receipt,
    )
    assert floor_width is None
    assert "oracle_dense_accumulator_w7_clip_must_be_false" in arm_failures
    result = classify_w7_in_vivo_dual_arm(
        oracle_receipt=oracle_receipt,
        treatment_receipt=treatment_receipt,
        envelope=envelope,
        parity_break=False,
        confirmed_vote_acc_floor_width=floor_width,
        harness_failures=arm_failures,
    )
    assert result["primary_classifier"] == CLASSIFIER_RUN_HEALTH_FAIL
    assert result["confirmed_vote_acc_floor_width"] is None
    assert result["rules_promotion_unlock_predicate"] is False
