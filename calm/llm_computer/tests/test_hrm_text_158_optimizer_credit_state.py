"""Focused tests for the optimizer-credit fail-closed blocker receipt."""
from __future__ import annotations

from dataclasses import replace

import pytest

import calm.hrm_text_158.native_full_stack as native_full_stack
from calm.hrm_text_158.native_full_stack.optimizer_credit_state import (
    OPTIMIZER_CREDIT_STATE_ALLOWED_DEBT_ANCHORS,
    OPTIMIZER_CREDIT_STATE_BLOCKED_REASON,
    OPTIMIZER_CREDIT_STATE_FAIL_CLOSED_NON_CLAIMS,
    OPTIMIZER_CREDIT_STATE_FAIL_CLOSED_RECEIPT_SCHEMA_VERSION,
    OPTIMIZER_CREDIT_STATE_FAIL_CLOSED_TARGET_NAME,
    OPTIMIZER_CREDIT_STATE_FP_EXCEPTION_CAVEAT,
    OPTIMIZER_CREDIT_STATE_REQUIRED_DEBT_ANCHORS,
    OptimizerCreditStateFailClosedReceipt,
    build_optimizer_credit_state_fail_closed_receipt,
    validate_optimizer_credit_state_fail_closed_receipt,
)


def _debt_anchor(name: str) -> dict[str, str]:
    return {
        "anchor_name": name,
        "source_anchor": f"calm/hrm_text_158/native_full_stack/{name}.py:1",
        "evidence": f"{name} remains observed as dense transient optimizer debt",
        "debt_kind": "dense_transient_over2_credit_debt",
    }


def _debt_anchors() -> tuple[dict[str, str], ...]:
    return tuple(_debt_anchor(name) for name in OPTIMIZER_CREDIT_STATE_REQUIRED_DEBT_ANCHORS)


def test_optimizer_credit_state_fail_closed_receipt_enumerates_dense_debt_without_flip():
    receipt = build_optimizer_credit_state_fail_closed_receipt()

    validate_optimizer_credit_state_fail_closed_receipt(receipt)
    assert receipt.schema_version == OPTIMIZER_CREDIT_STATE_FAIL_CLOSED_RECEIPT_SCHEMA_VERSION
    assert receipt.target_name == OPTIMIZER_CREDIT_STATE_FAIL_CLOSED_TARGET_NAME
    assert receipt.allowed_debt_anchors == OPTIMIZER_CREDIT_STATE_ALLOWED_DEBT_ANCHORS
    assert receipt.required_debt_anchors == OPTIMIZER_CREDIT_STATE_REQUIRED_DEBT_ANCHORS
    assert tuple(anchor.anchor_name for anchor in receipt.debt_anchors) == (
        "weighted_grad",
        "credit",
        "projected_moves",
        "dense_rank_votes_before_sparse_event_extraction",
        "optimizer_credit_state_resolved_false",
        "credit_ranking_update_law_pivot_deferred",
    )
    assert receipt.optimizer_credit_state_sub2_claim is False
    assert receipt.optimizer_credit_state_resolved is False
    assert receipt.readiness_row_flip_authorized is False
    assert receipt.fp_exception_laundering_claim is False
    assert receipt.real_native_integer_attribution_present is False
    assert receipt.real_native_integer_credit_ranking_present is False
    assert receipt.no_hidden_bf16_fp_optimizer_state_proven is False
    assert receipt.gpu_runtime_receipt_present is False
    assert receipt.ready_to_flip is False
    assert "fail-closed optimizer/credit-state harness" in receipt.blocked_reason
    assert "weighted_grad" in receipt.blocked_reason
    assert "projected_moves" in receipt.blocked_reason
    assert "dense_rank_votes" in receipt.blocked_reason
    assert "credit_capture_tensors" in receipt.fp_exception_caveat
    assert "attribution-only" in receipt.fp_exception_caveat
    assert any(".pt artifacts" in non_claim for non_claim in receipt.non_claims)
    assert any("attribution-only" in non_claim for non_claim in receipt.non_claims)
    assert receipt.non_claims == OPTIMIZER_CREDIT_STATE_FAIL_CLOSED_NON_CLAIMS
    assert receipt.to_dict()["ready_to_flip"] is False


def test_optimizer_credit_state_fail_closed_receipt_rejects_missing_unknown_and_laundering():
    with pytest.raises(ValueError, match="missing required debt anchors"):
        build_optimizer_credit_state_fail_closed_receipt(
            debt_anchors=[
                anchor
                for anchor in _debt_anchors()
                if anchor["anchor_name"] != "credit"
            ]
        )

    with pytest.raises(ValueError, match="Step 3C allowlist"):
        build_optimizer_credit_state_fail_closed_receipt(
            debt_anchors=(*_debt_anchors(), _debt_anchor("adam_moment"))
        )

    claim_flags = (
        "optimizer_credit_state_sub2_claim",
        "optimizer_credit_state_resolved",
        "readiness_row_flip_authorized",
        "fp_exception_laundering_claim",
    )
    for flag_name in claim_flags:
        with pytest.raises(ValueError, match=flag_name):
            build_optimizer_credit_state_fail_closed_receipt(**{flag_name: True})

    with pytest.raises(ValueError, match="ready_to_flip cannot be true"):
        build_optimizer_credit_state_fail_closed_receipt(ready_to_flip=True)


def test_optimizer_credit_state_receipt_rejects_drifted_contract_fields():
    receipt = build_optimizer_credit_state_fail_closed_receipt()

    drifted = replace(
        receipt,
        blocked_reason=receipt.blocked_reason + " drifted",
    )
    with pytest.raises(ValueError, match="blocked reason must be exact"):
        validate_optimizer_credit_state_fail_closed_receipt(drifted)

    drifted = replace(
        receipt,
        fp_exception_caveat=receipt.fp_exception_caveat + " drifted",
    )
    with pytest.raises(ValueError, match="credit_capture_tensors"):
        validate_optimizer_credit_state_fail_closed_receipt(drifted)

    drifted = replace(
        receipt,
        debt_anchors=(replace(receipt.debt_anchors[0], debt_kind="sparse_sub2"),)
        + receipt.debt_anchors[1:],
    )
    with pytest.raises(ValueError, match="dense transient debt"):
        validate_optimizer_credit_state_fail_closed_receipt(drifted)


def test_native_full_stack_exports_optimizer_credit_state_contract_surface():
    assert (
        native_full_stack.OPTIMIZER_CREDIT_STATE_FAIL_CLOSED_RECEIPT_SCHEMA_VERSION
        == OPTIMIZER_CREDIT_STATE_FAIL_CLOSED_RECEIPT_SCHEMA_VERSION
    )
    assert native_full_stack.OPTIMIZER_CREDIT_STATE_BLOCKED_REASON == (
        OPTIMIZER_CREDIT_STATE_BLOCKED_REASON
    )
    assert native_full_stack.OPTIMIZER_CREDIT_STATE_FP_EXCEPTION_CAVEAT == (
        OPTIMIZER_CREDIT_STATE_FP_EXCEPTION_CAVEAT
    )
    assert native_full_stack.build_optimizer_credit_state_fail_closed_receipt is (
        build_optimizer_credit_state_fail_closed_receipt
    )
    assert isinstance(
        native_full_stack.build_optimizer_credit_state_fail_closed_receipt(),
        OptimizerCreditStateFailClosedReceipt,
    )
    assert (
        "OPTIMIZER_CREDIT_STATE_FAIL_CLOSED_RECEIPT_SCHEMA_VERSION"
        in native_full_stack.__all__
    )
    assert "build_optimizer_credit_state_fail_closed_receipt" in native_full_stack.__all__
    assert "validate_optimizer_credit_state_fail_closed_receipt" in native_full_stack.__all__
