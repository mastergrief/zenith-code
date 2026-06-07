"""Tests for the 2B0 local-only live authority receipt."""
from __future__ import annotations

from dataclasses import replace

import pytest

import calm.hrm_text_158.native_full_stack as native_full_stack
from calm.hrm_text_158.native_full_stack.full_sub2_runtime_readiness import (
    FIXTURE_CURRENT_REPO,
    RUNTIME_CLASS_MISSING,
    RUNTIME_CLASS_PRE_FULL_STACK_DIAGNOSTIC,
    RUNTIME_CLASS_TRANSIENT_FP_DEBT,
    SURFACE_DENSE_INT16_PERSISTENT_ACCUMULATOR_ABSENCE,
    SURFACE_PERSISTENT_QACC_AUTHORITY,
    SURFACE_Q_SIDECAR_VOTE_CARRIER,
    fixture_full_sub2_runtime_ready_for_science,
)
from calm.hrm_text_158.native_full_stack.live_local_sub2_authority import (
    LIVE_LOCAL_SUB2_AUTHORITY_UNCOVERED_BLOCKERS,
    build_live_local_sub2_authority_receipt,
    validate_live_local_sub2_authority_receipt,
)


def _surface_classes(receipt):
    return {surface.surface_id: surface.classification for surface in receipt.surfaces}


def test_live_local_receipt_passes_only_declared_local_domain():
    receipt = build_live_local_sub2_authority_receipt()
    proof = receipt.proof_by_key["step2b0.local.proj"]

    validate_live_local_sub2_authority_receipt(receipt)
    assert receipt.pass_receipt is True
    assert receipt.local_authority_seam_executable is True
    assert receipt.exact_local_parity_pass is True
    assert receipt.local_persistent_core_sub2 is True
    assert receipt.no_dense_int16_counted_authority_local is True
    assert receipt.dense_oracle_control_used_for_comparison is True
    assert receipt.physical_persistent_bits_per_weight < 2.0
    assert receipt.effective_persistent_bits_per_weight < 2.0
    assert receipt.sidecar_report.persistent_dense_shadow_present is False
    assert receipt.sidecar_report.persistent_dense_shadow_bytes == 0
    assert receipt.sidecar_report.total_event_count == 2
    assert proof["parity_pass"] is True
    assert proof["candidate_q_sha256_after"] == proof["oracle_q_sha256_after"]
    assert proof["candidate_bounded_decode_sha256_after"] == proof["oracle_acc_sha256_after"]
    assert (
        proof["applied_row_identities_sha256"]
        == proof["oracle_applied_row_identities_sha256"]
    )
    assert (
        proof["residual_after_threshold_sha256"]
        == proof["oracle_residual_after_threshold_sha256"]
    )
    assert proof["q_changed_count"] == 2
    assert proof["applied_row_count"] == 2
    assert proof["candidate_dense_decode_used"] is False
    assert proof["candidate_dense_vote_authority_used"] is False


def test_live_local_receipt_keeps_trainer_and_readiness_claims_false():
    receipt = build_live_local_sub2_authority_receipt()

    assert receipt.production_authority_claim_authorized is False
    assert receipt.live_runtime_authority_converted is False
    assert receipt.trainer_entrypoint_uses_candidate is False
    assert receipt.readiness_row_flip_authorized is False
    assert receipt.readiness_row_flip_authorized_surface_names == ()
    assert receipt.current_repo_readiness_rows_may_flip is False
    assert receipt.uncovered_blockers == LIVE_LOCAL_SUB2_AUTHORITY_UNCOVERED_BLOCKERS
    for blocker in ("global_cap", "replay_ce_veto", "pc_aux", "trainer_integration"):
        assert blocker in receipt.uncovered_blockers


def test_current_repo_readiness_fixture_remains_unflipped_by_2b0_receipt():
    local_receipt = build_live_local_sub2_authority_receipt()
    readiness = fixture_full_sub2_runtime_ready_for_science(FIXTURE_CURRENT_REPO)
    classes = _surface_classes(readiness)

    assert local_receipt.pass_receipt is True
    assert readiness.ready_for_main_science is False
    assert readiness.main_science_launch_blocked is True
    assert classes[SURFACE_PERSISTENT_QACC_AUTHORITY] == RUNTIME_CLASS_TRANSIENT_FP_DEBT
    assert (
        classes[SURFACE_DENSE_INT16_PERSISTENT_ACCUMULATOR_ABSENCE]
        == RUNTIME_CLASS_MISSING
    )
    assert classes[SURFACE_Q_SIDECAR_VOTE_CARRIER] == RUNTIME_CLASS_PRE_FULL_STACK_DIAGNOSTIC


def test_exact_local_parity_is_required_for_receipt_validation():
    receipt = build_live_local_sub2_authority_receipt()
    proof_by_key = {
        key: dict(value)
        for key, value in receipt.proof_by_key.items()
    }
    proof_by_key["step2b0.local.proj"]["parity_pass"] = False
    bad = replace(
        receipt,
        pass_receipt=False,
        exact_local_parity_pass=False,
        proof_by_key=proof_by_key,
    )

    with pytest.raises(ValueError, match="exact local parity"):
        validate_live_local_sub2_authority_receipt(bad)


def test_forbidden_trainer_or_row_flip_claims_fail_validation():
    receipt = build_live_local_sub2_authority_receipt()

    with pytest.raises(ValueError, match="trainer entrypoint"):
        validate_live_local_sub2_authority_receipt(
            replace(receipt, trainer_entrypoint_uses_candidate=True)
        )
    with pytest.raises(ValueError, match="readiness row"):
        validate_live_local_sub2_authority_receipt(
            replace(receipt, readiness_row_flip_authorized=True)
        )
    with pytest.raises(ValueError, match="FIXTURE_CURRENT_REPO"):
        validate_live_local_sub2_authority_receipt(
            replace(receipt, current_repo_readiness_rows_may_flip=True)
        )


def test_live_local_receipt_exports_from_native_full_stack_facade():
    receipt = native_full_stack.build_live_local_sub2_authority_receipt()

    assert receipt.pass_receipt is True
    assert (
        native_full_stack.LIVE_LOCAL_SUB2_AUTHORITY_UNCOVERED_BLOCKERS
        == LIVE_LOCAL_SUB2_AUTHORITY_UNCOVERED_BLOCKERS
    )
    assert "build_live_local_sub2_authority_receipt" in native_full_stack.__all__
