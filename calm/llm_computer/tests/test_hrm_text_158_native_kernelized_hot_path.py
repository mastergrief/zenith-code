"""Focused tests for the native kernelized hot-path fail-closed receipt."""
from __future__ import annotations

from dataclasses import replace

import pytest

import calm.hrm_text_158.native_full_stack as native_full_stack
from calm.hrm_text_158.native_full_stack.native_kernelized_hot_path import (
    NATIVE_KERNELIZED_HOT_PATH_ALLOWED_BLOCKER_ANCHORS,
    NATIVE_KERNELIZED_HOT_PATH_BLOCKED_REASON,
    NATIVE_KERNELIZED_HOT_PATH_DEVICE_LAUNDERING_CAVEAT,
    NATIVE_KERNELIZED_HOT_PATH_FAIL_CLOSED_NON_CLAIMS,
    NATIVE_KERNELIZED_HOT_PATH_FAIL_CLOSED_RECEIPT_SCHEMA_VERSION,
    NATIVE_KERNELIZED_HOT_PATH_FAIL_CLOSED_TARGET_NAME,
    NATIVE_KERNELIZED_HOT_PATH_REQUIRED_BLOCKER_ANCHORS,
    NativeKernelizedHotPathFailClosedReceipt,
    build_native_kernelized_hot_path_fail_closed_receipt,
    validate_native_kernelized_hot_path_fail_closed_receipt,
)


def _blocker_anchor(name: str) -> dict[str, str]:
    return {
        "anchor_name": name,
        "source_anchor": f"calm/hrm_text_158/native_full_stack/{name}.py:1",
        "evidence": f"{name} remains observed as native hot-loop blocker evidence",
        "blocker_kind": "pre_full_stack_hot_loop_residency_blocker",
    }


def _blocker_anchors() -> tuple[dict[str, str], ...]:
    return tuple(
        _blocker_anchor(name)
        for name in NATIVE_KERNELIZED_HOT_PATH_REQUIRED_BLOCKER_ANCHORS
    )


def test_native_kernelized_hot_path_receipt_enumerates_current_blockers_without_flip():
    receipt = build_native_kernelized_hot_path_fail_closed_receipt()

    validate_native_kernelized_hot_path_fail_closed_receipt(receipt)
    assert (
        receipt.schema_version
        == NATIVE_KERNELIZED_HOT_PATH_FAIL_CLOSED_RECEIPT_SCHEMA_VERSION
    )
    assert receipt.target_name == NATIVE_KERNELIZED_HOT_PATH_FAIL_CLOSED_TARGET_NAME
    assert (
        receipt.allowed_blocker_anchors
        == NATIVE_KERNELIZED_HOT_PATH_ALLOWED_BLOCKER_ANCHORS
    )
    assert (
        receipt.required_blocker_anchors
        == NATIVE_KERNELIZED_HOT_PATH_REQUIRED_BLOCKER_ANCHORS
    )
    assert tuple(anchor.anchor_name for anchor in receipt.blocker_anchors) == (
        "qacc_kernelized_false",
        "qacc_update_vote_selection_apply_cpu_reference",
        "triton_preplan_only",
        "q_acc_apply_final_row_torch_cuda_reference",
        "global_cap_margin_only_reference_default_off",
        "full_loop_reference_stitch_no_native_speed_claim",
        "device_cuda_not_hot_loop_residency",
    )
    assert receipt.native_kernelized_hot_path_claim is False
    assert receipt.hot_loop_residency_claim is False
    assert receipt.device_cuda_laundering_claim is False
    assert receipt.readiness_row_flip_authorized is False
    assert receipt.qacc_kernelized is False
    assert receipt.qacc_update_over_64_cpu_reference is True
    assert receipt.vote_selection_cpu_reference is True
    assert receipt.q_acc_apply_cpu_reference is True
    assert receipt.triton_preplan_only is True
    assert receipt.q_acc_apply_final_row_torch_cuda_reference is True
    assert receipt.global_cap_margin_only_reference is True
    assert receipt.full_loop_native_custom_kernel_speed_claim is False
    assert receipt.real_device_resident_kernelized_hot_loop_present is False
    assert receipt.exact_cpu_oracle_parity_present is False
    assert receipt.gpu_runtime_receipt_present is False
    assert receipt.no_cpu_row_materialization_before_apply is False
    assert receipt.ready_to_flip is False
    assert "fail-closed native kernelized hot-path harness" in receipt.blocked_reason
    assert "qacc_kernelized=false" in receipt.blocked_reason
    assert "CPU-reference" in receipt.blocked_reason
    assert "Triton preplan" in receipt.blocked_reason
    assert "final-row torch-CUDA reference" in receipt.blocked_reason
    assert "native custom kernel speed claim" in receipt.blocked_reason
    assert "device=cuda" in receipt.device_laundering_caveat
    assert "hot-loop residency" in receipt.device_laundering_caveat
    assert any("resource lane" in non_claim for non_claim in receipt.non_claims)
    assert any("device=cuda" in non_claim for non_claim in receipt.non_claims)
    assert receipt.non_claims == NATIVE_KERNELIZED_HOT_PATH_FAIL_CLOSED_NON_CLAIMS
    assert receipt.to_dict()["ready_to_flip"] is False


def test_native_kernelized_hot_path_rejects_missing_unknown_and_laundering_claims():
    with pytest.raises(ValueError, match="missing required blocker anchors"):
        build_native_kernelized_hot_path_fail_closed_receipt(
            blocker_anchors=[
                anchor
                for anchor in _blocker_anchors()
                if anchor["anchor_name"] != "triton_preplan_only"
            ]
        )

    with pytest.raises(ValueError, match="Step 4A allowlist"):
        build_native_kernelized_hot_path_fail_closed_receipt(
            blocker_anchors=(*_blocker_anchors(), _blocker_anchor("cuda_tensor_seen"))
        )

    claim_flags = (
        "native_kernelized_hot_path_claim",
        "hot_loop_residency_claim",
        "readiness_row_flip_authorized",
    )
    for flag_name in claim_flags:
        with pytest.raises(ValueError, match=flag_name):
            build_native_kernelized_hot_path_fail_closed_receipt(**{flag_name: True})

    with pytest.raises(ValueError, match="device_cuda_laundering_claim"):
        build_native_kernelized_hot_path_fail_closed_receipt(
            device_cuda_laundering_claim=True
        )

    with pytest.raises(ValueError, match="ready_to_flip cannot be true"):
        build_native_kernelized_hot_path_fail_closed_receipt(ready_to_flip=True)


def test_native_kernelized_hot_path_rejects_cuda_looking_partial_proof_without_hot_loop():
    with pytest.raises(ValueError, match="native_kernelized_hot_path_claim"):
        build_native_kernelized_hot_path_fail_closed_receipt(
            native_kernelized_hot_path_claim=True,
            gpu_runtime_receipt_present=True,
            exact_cpu_oracle_parity_present=True,
            qacc_kernelized=True,
        )

    with pytest.raises(ValueError, match="hot_loop_residency_claim"):
        build_native_kernelized_hot_path_fail_closed_receipt(
            hot_loop_residency_claim=True,
            real_device_resident_kernelized_hot_loop_present=True,
            gpu_runtime_receipt_present=True,
            exact_cpu_oracle_parity_present=True,
        )


def test_native_kernelized_hot_path_receipt_rejects_drifted_contract_fields():
    receipt = build_native_kernelized_hot_path_fail_closed_receipt()

    drifted = replace(receipt, blocked_reason=receipt.blocked_reason + " drifted")
    with pytest.raises(ValueError, match="blocked reason must be exact"):
        validate_native_kernelized_hot_path_fail_closed_receipt(drifted)

    drifted = replace(
        receipt,
        device_laundering_caveat=receipt.device_laundering_caveat + " drifted",
    )
    with pytest.raises(ValueError, match="device-vs-hot-loop caveat"):
        validate_native_kernelized_hot_path_fail_closed_receipt(drifted)

    drifted = replace(
        receipt,
        blocker_anchors=(
            replace(receipt.blocker_anchors[0], blocker_kind="device_only"),
        )
        + receipt.blocker_anchors[1:],
    )
    with pytest.raises(ValueError, match="hot-loop blocker evidence"):
        validate_native_kernelized_hot_path_fail_closed_receipt(drifted)


def test_native_full_stack_exports_native_kernelized_hot_path_contract_surface():
    assert (
        native_full_stack.NATIVE_KERNELIZED_HOT_PATH_FAIL_CLOSED_RECEIPT_SCHEMA_VERSION
        == NATIVE_KERNELIZED_HOT_PATH_FAIL_CLOSED_RECEIPT_SCHEMA_VERSION
    )
    assert native_full_stack.NATIVE_KERNELIZED_HOT_PATH_BLOCKED_REASON == (
        NATIVE_KERNELIZED_HOT_PATH_BLOCKED_REASON
    )
    assert native_full_stack.NATIVE_KERNELIZED_HOT_PATH_DEVICE_LAUNDERING_CAVEAT == (
        NATIVE_KERNELIZED_HOT_PATH_DEVICE_LAUNDERING_CAVEAT
    )
    assert (
        native_full_stack.build_native_kernelized_hot_path_fail_closed_receipt
        is build_native_kernelized_hot_path_fail_closed_receipt
    )
    assert isinstance(
        native_full_stack.build_native_kernelized_hot_path_fail_closed_receipt(),
        NativeKernelizedHotPathFailClosedReceipt,
    )
    assert (
        "NATIVE_KERNELIZED_HOT_PATH_FAIL_CLOSED_RECEIPT_SCHEMA_VERSION"
        in native_full_stack.__all__
    )
    assert (
        "build_native_kernelized_hot_path_fail_closed_receipt"
        in native_full_stack.__all__
    )
    assert (
        "validate_native_kernelized_hot_path_fail_closed_receipt"
        in native_full_stack.__all__
    )
