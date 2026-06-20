"""Focused tests for the native kernelized hot-path fail-closed receipt."""
from __future__ import annotations

from dataclasses import replace

import pytest

import calm.hrm_text_158.native_full_stack as native_full_stack
from calm.hrm_text_158.native_full_stack.native_kernelized_hot_path import (
    B2_3_STANDALONE_QACC_APPLY_COMMIT_SHA,
    B2_3_STANDALONE_QACC_APPLY_GPU_RECEIPT_MSG_ID,
    B2_4_COMPOSITION_GPU_RECEIPT_MSG_ID,
    B2_4_COMPOSITION_QACC_APPLY_COMMIT_SHA,
    NATIVE_KERNELIZED_HOT_PATH_ALLOWED_BLOCKER_ANCHORS,
    NATIVE_KERNELIZED_HOT_PATH_BLOCKED_REASON,
    NATIVE_KERNELIZED_HOT_PATH_DEVICE_LAUNDERING_CAVEAT,
    NATIVE_KERNELIZED_HOT_PATH_FAIL_CLOSED_NON_CLAIMS,
    NATIVE_KERNELIZED_HOT_PATH_FAIL_CLOSED_RECEIPT_SCHEMA_VERSION,
    NATIVE_KERNELIZED_HOT_PATH_FAIL_CLOSED_TARGET_NAME,
    NATIVE_KERNELIZED_HOT_PATH_REQUIRED_BLOCKER_ANCHORS,
    NativeKernelizedHotPathFailClosedReceipt,
    _future_proof_gate,
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


def test_required_blocker_anchor_count_is_six() -> None:
    assert len(NATIVE_KERNELIZED_HOT_PATH_REQUIRED_BLOCKER_ANCHORS) == 6
    assert len(NATIVE_KERNELIZED_HOT_PATH_ALLOWED_BLOCKER_ANCHORS) == 6


def test_composition_blocker_anchor_retired() -> None:
    assert "composition_paths_still_call_torch_cuda_apply" not in (
        NATIVE_KERNELIZED_HOT_PATH_REQUIRED_BLOCKER_ANCHORS
    )
    assert "composition_paths_still_call_torch_cuda_apply" not in (
        NATIVE_KERNELIZED_HOT_PATH_ALLOWED_BLOCKER_ANCHORS
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
        "global_cap_margin_only_reference_default_off",
        "full_loop_reference_stitch_no_native_speed_claim",
        "device_cuda_not_hot_loop_residency",
    )
    assert receipt.standalone_qacc_apply_native_proven is True
    assert receipt.standalone_qacc_apply_exact_parity_present is True
    assert receipt.standalone_qacc_apply_gpu_receipt_present is True
    assert receipt.composition_qacc_apply_native_proven is True
    assert receipt.composition_exact_parity_present is True
    assert receipt.composition_gpu_receipt_present is True
    assert receipt.native_kernelized_hot_path_claim is False
    assert receipt.hot_loop_residency_claim is False
    assert receipt.device_cuda_laundering_claim is False
    assert receipt.readiness_row_flip_authorized is False
    assert receipt.qacc_kernelized is False
    assert receipt.qacc_update_over_64_cpu_reference is True
    assert receipt.vote_selection_cpu_reference is True
    assert receipt.q_acc_apply_cpu_reference is True
    assert receipt.triton_preplan_only is True
    assert receipt.q_acc_apply_final_row_torch_cuda_reference is False
    assert receipt.global_cap_margin_only_reference is True
    assert receipt.full_loop_native_custom_kernel_speed_claim is False
    assert receipt.real_device_resident_kernelized_hot_loop_present is False
    assert receipt.exact_cpu_oracle_parity_present is False
    assert receipt.gpu_runtime_receipt_present is False
    assert receipt.no_cpu_row_materialization_before_apply is True
    assert receipt.ready_to_flip is False
    assert receipt.blocked_reason == NATIVE_KERNELIZED_HOT_PATH_BLOCKED_REASON
    assert "fail-closed native kernelized hot-path harness" in receipt.blocked_reason
    assert "standalone q_acc_apply apply-kernel proven (B2-3)" in receipt.blocked_reason
    assert "qacc_kernelized=false" in receipt.blocked_reason
    assert "CPU-reference" in receipt.blocked_reason
    assert "Triton preplan" in receipt.blocked_reason
    assert "composition paths still call torch-CUDA q_acc_apply" not in receipt.blocked_reason
    assert "native custom kernel speed claim" in receipt.blocked_reason
    assert "cap SELECTION" in receipt.smallest_missing_proof
    assert "full-loop native proof" in receipt.smallest_missing_proof
    assert "no CPU row materialization before apply" not in receipt.smallest_missing_proof
    assert "device=cuda" in receipt.device_laundering_caveat
    assert "hot-loop residency" in receipt.device_laundering_caveat
    assert any("resource lane" in non_claim for non_claim in receipt.non_claims)
    assert any("device=cuda" in non_claim for non_claim in receipt.non_claims)
    assert any(
        "composed-path q_acc_apply APPLY parity is proven (B2-4)" in non_claim
        for non_claim in receipt.non_claims
    )
    assert receipt.non_claims == NATIVE_KERNELIZED_HOT_PATH_FAIL_CLOSED_NON_CLAIMS
    assert receipt.to_dict()["ready_to_flip"] is False
    assert B2_3_STANDALONE_QACC_APPLY_COMMIT_SHA.startswith("5d90643")
    assert B2_3_STANDALONE_QACC_APPLY_GPU_RECEIPT_MSG_ID == "1781972683995"


def test_composition_satisfied_evidence_present_and_citation_bound() -> None:
    receipt = build_native_kernelized_hot_path_fail_closed_receipt()

    assert receipt.composition_qacc_apply_native_proven is True
    assert receipt.composition_exact_parity_present is True
    assert receipt.composition_gpu_receipt_present is True
    assert B2_4_COMPOSITION_QACC_APPLY_COMMIT_SHA == (
        "99727af5dc7b9eac989096b34bdfb46586fe6c12"
    )
    assert B2_4_COMPOSITION_GPU_RECEIPT_MSG_ID == "1781979673113"


def test_composition_conjuncts_flipped() -> None:
    receipt = build_native_kernelized_hot_path_fail_closed_receipt()

    assert receipt.q_acc_apply_final_row_torch_cuda_reference is False
    assert receipt.no_cpu_row_materialization_before_apply is True


def test_future_proof_gate_remains_false_with_standalone_proven():
    receipt = build_native_kernelized_hot_path_fail_closed_receipt()

    assert receipt.standalone_qacc_apply_native_proven is True
    assert _future_proof_gate(receipt) is False
    assert receipt.ready_to_flip is False
    assert receipt.native_kernelized_hot_path_claim is False
    assert receipt.hot_loop_residency_claim is False
    assert receipt.readiness_row_flip_authorized is False


def test_future_proof_gate_remains_false_with_composition_proven() -> None:
    receipt = build_native_kernelized_hot_path_fail_closed_receipt()

    assert receipt.composition_qacc_apply_native_proven is True
    assert receipt.q_acc_apply_final_row_torch_cuda_reference is False
    assert receipt.no_cpu_row_materialization_before_apply is True
    assert _future_proof_gate(receipt) is False
    assert receipt.ready_to_flip is False
    assert receipt.native_kernelized_hot_path_claim is False
    assert receipt.hot_loop_residency_claim is False
    assert receipt.readiness_row_flip_authorized is False


@pytest.mark.parametrize(
    "flag_name",
    (
        "ready_to_flip",
        "native_kernelized_hot_path_claim",
        "hot_loop_residency_claim",
        "readiness_row_flip_authorized",
    ),
)
def test_standalone_proven_rejects_laundering_claims(flag_name: str):
    with pytest.raises(ValueError, match="standalone_qacc_apply_native_proven cannot coexist"):
        build_native_kernelized_hot_path_fail_closed_receipt(**{flag_name: True})


@pytest.mark.parametrize(
    "flag_name",
    (
        "ready_to_flip",
        "native_kernelized_hot_path_claim",
        "hot_loop_residency_claim",
        "readiness_row_flip_authorized",
    ),
)
def test_composition_proven_rejects_laundering_claims(flag_name: str):
    with pytest.raises(ValueError, match="cannot coexist with ready_to_flip or hot-path laundering"):
        build_native_kernelized_hot_path_fail_closed_receipt(
            standalone_qacc_apply_native_proven=False,
            standalone_qacc_apply_exact_parity_present=False,
            standalone_qacc_apply_gpu_receipt_present=False,
            **{flag_name: True},
        )


def test_standalone_proven_rejects_partial_evidence_without_native_flag():
    with pytest.raises(ValueError, match="partial standalone q_acc_apply evidence"):
        build_native_kernelized_hot_path_fail_closed_receipt(
            standalone_qacc_apply_native_proven=False,
            standalone_qacc_apply_exact_parity_present=True,
            standalone_qacc_apply_gpu_receipt_present=False,
        )

    with pytest.raises(ValueError, match="partial standalone q_acc_apply evidence"):
        build_native_kernelized_hot_path_fail_closed_receipt(
            standalone_qacc_apply_native_proven=False,
            standalone_qacc_apply_exact_parity_present=False,
            standalone_qacc_apply_gpu_receipt_present=True,
        )


def test_standalone_proven_requires_coupled_evidence_fields():
    with pytest.raises(ValueError, match="requires coupled exact-parity"):
        build_native_kernelized_hot_path_fail_closed_receipt(
            standalone_qacc_apply_exact_parity_present=False,
        )

    with pytest.raises(ValueError, match="requires coupled exact-parity"):
        build_native_kernelized_hot_path_fail_closed_receipt(
            standalone_qacc_apply_gpu_receipt_present=False,
        )


def test_composition_evidence_requires_coupled_fields() -> None:
    with pytest.raises(ValueError, match="requires coupled evidence fields"):
        build_native_kernelized_hot_path_fail_closed_receipt(
            composition_exact_parity_present=False,
        )

    with pytest.raises(ValueError, match="requires coupled evidence fields"):
        build_native_kernelized_hot_path_fail_closed_receipt(
            composition_gpu_receipt_present=False,
        )


def test_composition_evidence_requires_both_flipped_conjuncts() -> None:
    with pytest.raises(ValueError, match="q_acc_apply_final_row_torch_cuda_reference is False"):
        build_native_kernelized_hot_path_fail_closed_receipt(
            q_acc_apply_final_row_torch_cuda_reference=True,
        )

    with pytest.raises(ValueError, match="no_cpu_row_materialization_before_apply is True"):
        build_native_kernelized_hot_path_fail_closed_receipt(
            no_cpu_row_materialization_before_apply=False,
        )

    with pytest.raises(ValueError, match="flipped composition conjuncts without"):
        build_native_kernelized_hot_path_fail_closed_receipt(
            composition_qacc_apply_native_proven=False,
            composition_exact_parity_present=False,
            composition_gpu_receipt_present=False,
            q_acc_apply_final_row_torch_cuda_reference=False,
            no_cpu_row_materialization_before_apply=True,
        )


def test_ledger_v3_default_builder_preserves_remaining_gate_conjuncts() -> None:
    receipt = build_native_kernelized_hot_path_fail_closed_receipt()

    assert receipt.composition_qacc_apply_native_proven is True
    assert receipt.composition_exact_parity_present is True
    assert receipt.composition_gpu_receipt_present is True
    assert receipt.q_acc_apply_final_row_torch_cuda_reference is False
    assert receipt.no_cpu_row_materialization_before_apply is True
    assert receipt.exact_cpu_oracle_parity_present is False
    assert receipt.gpu_runtime_receipt_present is False
    assert receipt.q_acc_apply_cpu_reference is True
    assert receipt.qacc_kernelized is False
    assert receipt.qacc_update_over_64_cpu_reference is True
    assert receipt.vote_selection_cpu_reference is True
    assert receipt.triton_preplan_only is True
    assert receipt.global_cap_margin_only_reference is True
    assert receipt.full_loop_native_custom_kernel_speed_claim is False
    assert receipt.real_device_resident_kernelized_hot_loop_present is False
    assert _future_proof_gate(receipt) is False


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
            build_native_kernelized_hot_path_fail_closed_receipt(
                standalone_qacc_apply_native_proven=False,
                standalone_qacc_apply_exact_parity_present=False,
                standalone_qacc_apply_gpu_receipt_present=False,
                composition_qacc_apply_native_proven=False,
                composition_exact_parity_present=False,
                composition_gpu_receipt_present=False,
                q_acc_apply_final_row_torch_cuda_reference=True,
                no_cpu_row_materialization_before_apply=False,
                **{flag_name: True},
            )

    with pytest.raises(ValueError, match="device_cuda_laundering_claim"):
        build_native_kernelized_hot_path_fail_closed_receipt(
            device_cuda_laundering_claim=True
        )

    with pytest.raises(ValueError, match="ready_to_flip cannot be true"):
        build_native_kernelized_hot_path_fail_closed_receipt(
            standalone_qacc_apply_native_proven=False,
            standalone_qacc_apply_exact_parity_present=False,
            standalone_qacc_apply_gpu_receipt_present=False,
            composition_qacc_apply_native_proven=False,
            composition_exact_parity_present=False,
            composition_gpu_receipt_present=False,
            q_acc_apply_final_row_torch_cuda_reference=True,
            no_cpu_row_materialization_before_apply=False,
            ready_to_flip=True,
        )


def test_native_kernelized_hot_path_rejects_cuda_looking_partial_proof_without_hot_loop():
    no_evidence = dict(
        standalone_qacc_apply_native_proven=False,
        standalone_qacc_apply_exact_parity_present=False,
        standalone_qacc_apply_gpu_receipt_present=False,
        composition_qacc_apply_native_proven=False,
        composition_exact_parity_present=False,
        composition_gpu_receipt_present=False,
        q_acc_apply_final_row_torch_cuda_reference=True,
        no_cpu_row_materialization_before_apply=False,
    )
    with pytest.raises(ValueError, match="native_kernelized_hot_path_claim"):
        build_native_kernelized_hot_path_fail_closed_receipt(
            native_kernelized_hot_path_claim=True,
            gpu_runtime_receipt_present=True,
            exact_cpu_oracle_parity_present=True,
            qacc_kernelized=True,
            **no_evidence,
        )

    with pytest.raises(ValueError, match="hot_loop_residency_claim"):
        build_native_kernelized_hot_path_fail_closed_receipt(
            hot_loop_residency_claim=True,
            real_device_resident_kernelized_hot_loop_present=True,
            gpu_runtime_receipt_present=True,
            exact_cpu_oracle_parity_present=True,
            **no_evidence,
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
