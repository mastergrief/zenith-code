"""Tests for the Step-2A candidate-only persistent-core absence receipt."""
from __future__ import annotations

from dataclasses import replace

import pytest

import calm.hrm_text_158.native_full_stack as native_full_stack
from calm.hrm_text_158.native_full_stack.full_sub2_runtime_readiness import (
    FIXTURE_STEP2A_CANDIDATE_PERSISTENT_CORE_ABSENCE,
    RUNTIME_CLASS_MISSING,
    RUNTIME_CLASS_PRE_FULL_STACK_DIAGNOSTIC,
    RUNTIME_CLASS_SUB2,
    RUNTIME_CLASS_TRANSIENT_FP_DEBT,
    SURFACE_DENSE_INT16_PERSISTENT_ACCUMULATOR_ABSENCE,
    SURFACE_PERSISTENT_QACC_AUTHORITY,
    SURFACE_Q_SIDECAR_VOTE_CARRIER,
    fixture_full_sub2_runtime_ready_for_science,
)
from calm.hrm_text_158.native_full_stack.persistent_core_sub2_absence import (
    PERSISTENT_CORE_SUB2_ABSENCE_DENSE_CREDIT_CLASS,
    PERSISTENT_CORE_SUB2_ABSENCE_LIVE_BLOCKED_SURFACES,
    build_persistent_core_sub2_absence_candidate_receipt,
    validate_persistent_core_sub2_absence_candidate_receipt,
)


def _surface_classes(receipt):
    return {surface.surface_id: surface.classification for surface in receipt.surfaces}


def test_candidate_absence_receipt_passes_without_authorizing_live_rows():
    receipt = build_persistent_core_sub2_absence_candidate_receipt()

    validate_persistent_core_sub2_absence_candidate_receipt(receipt)
    assert receipt.pass_receipt is True
    assert receipt.candidate_persistent_core_absence_proven is True
    assert receipt.production_authority_claim_authorized is False
    assert receipt.live_runtime_authority_converted is False
    assert receipt.trainer_entrypoint_uses_candidate is False
    assert receipt.readiness_row_flip_authorized is False
    assert receipt.readiness_row_flip_authorized_surface_names == ()
    assert receipt.dense_credit_classification == PERSISTENT_CORE_SUB2_ABSENCE_DENSE_CREDIT_CLASS
    assert (
        receipt.live_rows_remain_debt_or_blocker
        == PERSISTENT_CORE_SUB2_ABSENCE_LIVE_BLOCKED_SURFACES
    )
    assert receipt.sidecar_report.persistent_dense_shadow_present is False
    assert receipt.sidecar_report.persistent_dense_shadow_bytes == 0
    assert receipt.physical_persistent_bits_per_weight < 2.0
    assert receipt.effective_persistent_bits_per_weight < 2.0
    assert receipt.optimizer_fp_master_excluded is True


def test_step2a_candidate_fixture_does_not_flip_live_rows():
    receipt = fixture_full_sub2_runtime_ready_for_science(
        FIXTURE_STEP2A_CANDIDATE_PERSISTENT_CORE_ABSENCE
    )
    classes = _surface_classes(receipt)

    assert receipt.ready_for_main_science is False
    assert receipt.main_science_launch_blocked is True
    assert SURFACE_PERSISTENT_QACC_AUTHORITY in receipt.blocker_surface_names
    assert SURFACE_DENSE_INT16_PERSISTENT_ACCUMULATOR_ABSENCE in receipt.blocker_surface_names
    assert SURFACE_Q_SIDECAR_VOTE_CARRIER in receipt.blocker_surface_names
    assert classes[SURFACE_PERSISTENT_QACC_AUTHORITY] == RUNTIME_CLASS_TRANSIENT_FP_DEBT
    assert (
        classes[SURFACE_DENSE_INT16_PERSISTENT_ACCUMULATOR_ABSENCE]
        == RUNTIME_CLASS_MISSING
    )
    assert classes[SURFACE_Q_SIDECAR_VOTE_CARRIER] == RUNTIME_CLASS_PRE_FULL_STACK_DIAGNOSTIC
    assert RUNTIME_CLASS_SUB2 not in {
        classes[SURFACE_PERSISTENT_QACC_AUTHORITY],
        classes[SURFACE_DENSE_INT16_PERSISTENT_ACCUMULATOR_ABSENCE],
        classes[SURFACE_Q_SIDECAR_VOTE_CARRIER],
    }


def test_candidate_receipt_cannot_satisfy_ready_for_main_science():
    candidate = build_persistent_core_sub2_absence_candidate_receipt()
    readiness = fixture_full_sub2_runtime_ready_for_science(
        FIXTURE_STEP2A_CANDIDATE_PERSISTENT_CORE_ABSENCE
    )

    assert candidate.pass_receipt is True
    assert readiness.ready_for_main_science is False
    assert readiness.blocker_surface_names


def test_persistent_dense_shadow_presence_fails_validation():
    receipt = build_persistent_core_sub2_absence_candidate_receipt()
    bad_report = replace(receipt.sidecar_report, persistent_dense_shadow_present=True)
    bad_receipt = replace(
        receipt,
        sidecar_report=bad_report,
        no_dense_int16_persistent_accumulator_authority_candidate=False,
    )

    with pytest.raises(ValueError, match="dense shadow"):
        validate_persistent_core_sub2_absence_candidate_receipt(bad_receipt)


def test_persistent_dense_shadow_bytes_fail_validation():
    receipt = build_persistent_core_sub2_absence_candidate_receipt()
    bad_report = replace(receipt.sidecar_report, persistent_dense_shadow_bytes=2)
    bad_receipt = replace(
        receipt,
        sidecar_report=bad_report,
        no_dense_int16_persistent_accumulator_authority_candidate=False,
    )

    with pytest.raises(ValueError, match="dense shadow bytes"):
        validate_persistent_core_sub2_absence_candidate_receipt(bad_receipt)


def test_inclusive_bpw_at_or_above_two_fails_validation():
    receipt = build_persistent_core_sub2_absence_candidate_receipt()
    ledger = dict(receipt.sidecar_report.movement_overlay.persistent_sidecar_ledger)
    ledger["inclusive_bits_per_weight"] = 2.0
    ledger["inclusive_lt2"] = False
    bad_overlay = replace(
        receipt.sidecar_report.movement_overlay,
        persistent_sidecar_ledger=ledger,
    )
    bad_report = replace(receipt.sidecar_report, movement_overlay=bad_overlay)
    bad_receipt = replace(
        receipt,
        sidecar_report=bad_report,
        physical_persistent_bits_per_weight=2.0,
        effective_persistent_bits_per_weight=2.0,
    )

    with pytest.raises(ValueError, match="bits/weight"):
        validate_persistent_core_sub2_absence_candidate_receipt(bad_receipt)


def test_optimizer_proof_with_eligible_state_fails_validation():
    proof = {
        "eligible_master_identity_pass": True,
        "optimizer_checks": {
            "eligible_params_in_optimizer": 1,
            "eligible_optimizer_state_entries": 1,
            "pass": False,
        },
        "pass": False,
    }

    with pytest.raises(ValueError, match="eligible FP masters"):
        build_persistent_core_sub2_absence_candidate_receipt(
            optimizer_identity_proof=proof,
        )


@pytest.mark.parametrize("classification", ("explicit_exception", "sub2"))
def test_dense_credit_cannot_be_laundered_as_exception_or_sub2(classification):
    with pytest.raises(ValueError, match="transient_fp_debt"):
        build_persistent_core_sub2_absence_candidate_receipt(
            dense_credit_classification=classification,
        )


def test_readiness_row_flip_authorization_fails_validation():
    with pytest.raises(ValueError, match="readiness row"):
        build_persistent_core_sub2_absence_candidate_receipt(
            readiness_row_flip_authorized=True,
            readiness_row_flip_authorized_surface_names=(
                SURFACE_PERSISTENT_QACC_AUTHORITY,
            ),
        )


def test_candidate_receipt_exports_from_native_full_stack_facade():
    receipt = native_full_stack.build_persistent_core_sub2_absence_candidate_receipt()

    assert receipt.pass_receipt is True
    assert (
        native_full_stack.PERSISTENT_CORE_SUB2_ABSENCE_DENSE_CREDIT_CLASS
        == PERSISTENT_CORE_SUB2_ABSENCE_DENSE_CREDIT_CLASS
    )
    assert (
        "build_persistent_core_sub2_absence_candidate_receipt"
        in native_full_stack.__all__
    )
