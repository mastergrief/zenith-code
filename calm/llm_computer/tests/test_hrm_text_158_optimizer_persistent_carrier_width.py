"""Focused tests for optimizer persistent carrier width narrowability receipts."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from calm.hrm_text_158.native_full_stack.optimizer_persistent_carrier_width import (
    BRANCH_BLOCKED_BY_LAW_MISMATCH,
    BRANCH_BLOCKED_BY_LEARNING,
    BRANCH_INT16_REQUIRED,
    BRANCH_INT4_VIABLE,
    BRANCH_INT8_REQUIRED,
    LEARNING_DAMAGES,
    LEARNING_NOT_MEASURED,
    NARROWABILITY_NON_CLAIMS,
    OPTIMIZER_PERSISTENT_CARRIER_WIDTH_NARROWABILITY_SCHEMA_VERSION,
    OPTIMIZER_PERSISTENT_CARRIER_WIDTH_NARROWABILITY_TARGET_NAME,
    PARENT_SHA_HISTORICAL_V8C2_4DDEACC8,
    PARENT_SHA_LOCKED_ARM_A_9DB27EE4,
    PROOF_LAW_HISTORICAL_NO_DRAIN_CONTROL,
    PROOF_LAW_LOCKED_ARM_A,
    build_optimizer_persistent_carrier_width_narrowability_receipt,
    classify_from_parent_receipt_file,
    validate_optimizer_persistent_carrier_width_narrowability_receipt,
)

ARM_A_PARENT = Path(
    "/home/gabe/claw-code-creditdir/transient_fp_credit/"
    "racc_real_credit_drain_armA_20260616T191949ZZ/receipts/"
    "racc_real_credit_drain_run_receipt.json"
)
V8C2_PARENT = Path(
    "/home/gabe/claw-code-creditdir/transient_fp_credit/"
    "racc_deep_run_20260616T134548Z/receipts/racc_deep_run_receipt.json"
)


@pytest.mark.skipif(not ARM_A_PARENT.is_file(), reason="arm-A parent receipt missing")
def test_classify_locked_arm_a_parent_9db27ee4():
    receipt = classify_from_parent_receipt_file(
        ARM_A_PARENT,
        proof_law_id=PROOF_LAW_LOCKED_ARM_A,
    )
    validate_optimizer_persistent_carrier_width_narrowability_receipt(receipt)
    assert receipt.parent_receipt_sha256 == PARENT_SHA_LOCKED_ARM_A_9DB27EE4
    assert receipt.peak_decoded_abs_max_over_run == 7
    assert receipt.carrier_int4_viable is True
    assert receipt.encoding_round_trip_max_delta == 0
    assert receipt.width_encoding_viable is True
    assert receipt.branch_id == BRANCH_BLOCKED_BY_LEARNING
    assert receipt.learning_co_gate_verdict == LEARNING_DAMAGES
    assert receipt.ready_to_flip is False
    assert receipt.optimizer_credit_state_sub2_claim is False
    assert receipt.readiness_row_flip_authorized is False
    assert "width encoding viable" in receipt.non_claims[1]


@pytest.mark.skipif(not V8C2_PARENT.is_file(), reason="v8c2 parent receipt missing")
def test_classify_historical_control_parent_4ddeacc8():
    receipt = classify_from_parent_receipt_file(
        V8C2_PARENT,
        proof_law_id=PROOF_LAW_HISTORICAL_NO_DRAIN_CONTROL,
        control_only=True,
        learning_co_gate_verdict=LEARNING_NOT_MEASURED,
    )
    assert receipt.parent_receipt_sha256 == PARENT_SHA_HISTORICAL_V8C2_4DDEACC8
    assert receipt.peak_decoded_abs_max_over_run == 48
    assert receipt.carrier_int4_viable is False
    assert receipt.carrier_int8_viable is True
    assert receipt.width_encoding_viable is True
    assert receipt.branch_id == BRANCH_INT8_REQUIRED
    assert receipt.control_only is True


def test_parent_law_mismatch_rejected():
    with pytest.raises(ValueError, match="incompatible with proof_law_id"):
        build_optimizer_persistent_carrier_width_narrowability_receipt(
            proof_law_id=PROOF_LAW_LOCKED_ARM_A,
            parent_receipt_sha256=PARENT_SHA_HISTORICAL_V8C2_4DDEACC8,
            peak_decoded_abs_max_over_run=48,
            peak_frac_over_int4=0.5,
            peak_frac_over_int8=0.0,
        )


def test_historical_control_requires_control_only_flag():
    with pytest.raises(ValueError, match="control_only=true"):
        build_optimizer_persistent_carrier_width_narrowability_receipt(
            proof_law_id=PROOF_LAW_HISTORICAL_NO_DRAIN_CONTROL,
            parent_receipt_sha256=PARENT_SHA_HISTORICAL_V8C2_4DDEACC8,
            peak_decoded_abs_max_over_run=48,
            peak_frac_over_int4=0.5,
            peak_frac_over_int8=0.0,
            control_only=False,
        )


def test_int4_viable_width_only_slice():
    receipt = build_optimizer_persistent_carrier_width_narrowability_receipt(
        proof_law_id=PROOF_LAW_LOCKED_ARM_A,
        parent_receipt_sha256=PARENT_SHA_LOCKED_ARM_A_9DB27EE4,
        peak_decoded_abs_max_over_run=7,
        peak_frac_over_int4=0.0,
        peak_frac_over_int8=0.0,
        learning_co_gate_verdict=LEARNING_NOT_MEASURED,
    )
    assert receipt.branch_id == BRANCH_INT4_VIABLE
    assert receipt.width_encoding_viable is True


def test_int16_required_branch():
    receipt = build_optimizer_persistent_carrier_width_narrowability_receipt(
        proof_law_id=PROOF_LAW_HISTORICAL_NO_DRAIN_CONTROL,
        parent_receipt_sha256=PARENT_SHA_HISTORICAL_V8C2_4DDEACC8,
        peak_decoded_abs_max_over_run=200,
        peak_frac_over_int4=1.0,
        peak_frac_over_int8=1.0,
        control_only=True,
    )
    assert receipt.branch_id == BRANCH_INT16_REQUIRED


def test_validator_rejects_width_encoding_viable_inconsistency():
    receipt = build_optimizer_persistent_carrier_width_narrowability_receipt(
        proof_law_id=PROOF_LAW_LOCKED_ARM_A,
        parent_receipt_sha256=PARENT_SHA_LOCKED_ARM_A_9DB27EE4,
        peak_decoded_abs_max_over_run=7,
        peak_frac_over_int4=0.0,
        peak_frac_over_int8=0.0,
        learning_co_gate_verdict=LEARNING_NOT_MEASURED,
    )
    broken = receipt.__class__(
        **{
            **receipt.__dict__,
            "width_encoding_viable": False,
        }
    )
    with pytest.raises(ValueError, match="width_encoding_viable must equal"):
        validate_optimizer_persistent_carrier_width_narrowability_receipt(broken)


def test_validator_rejects_flip_claims():
    receipt = build_optimizer_persistent_carrier_width_narrowability_receipt(
        proof_law_id=PROOF_LAW_LOCKED_ARM_A,
        parent_receipt_sha256=PARENT_SHA_LOCKED_ARM_A_9DB27EE4,
        peak_decoded_abs_max_over_run=7,
        peak_frac_over_int4=0.0,
        peak_frac_over_int8=0.0,
    )
    broken = receipt.__class__(
        **{
            **receipt.__dict__,
            "ready_to_flip": True,
        }
    )
    with pytest.raises(ValueError, match="forbids flip"):
        validate_optimizer_persistent_carrier_width_narrowability_receipt(broken)


def test_schema_constants():
    receipt = build_optimizer_persistent_carrier_width_narrowability_receipt(
        proof_law_id=PROOF_LAW_LOCKED_ARM_A,
        parent_receipt_sha256=PARENT_SHA_LOCKED_ARM_A_9DB27EE4,
        peak_decoded_abs_max_over_run=7,
        peak_frac_over_int4=0.0,
        peak_frac_over_int8=0.0,
        learning_co_gate_verdict=LEARNING_NOT_MEASURED,
    )
    assert (
        receipt.schema_version
        == OPTIMIZER_PERSISTENT_CARRIER_WIDTH_NARROWABILITY_SCHEMA_VERSION
    )
    assert receipt.target_name == OPTIMIZER_PERSISTENT_CARRIER_WIDTH_NARROWABILITY_TARGET_NAME
    assert receipt.non_claims == NARROWABILITY_NON_CLAIMS
    assert json.loads(json.dumps(receipt.to_dict()))["branch_id"] == BRANCH_INT4_VIABLE
