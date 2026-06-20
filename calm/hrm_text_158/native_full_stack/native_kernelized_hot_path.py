"""Fail-closed native kernelized hot-path blocker receipt for HRM-Text-1.58.

This is a CPU/static contract only. It separates device residency from true
hot-loop residency and rejects CUDA-looking reference stitches as readiness
proof for the native_kernelized_hot_path row.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


NATIVE_KERNELIZED_HOT_PATH_FAIL_CLOSED_RECEIPT_SCHEMA_VERSION = (
    "hrm_text_158_native_kernelized_hot_path_fail_closed/"
    "v3.composition_apply_proven_gate_partial"
)
NATIVE_KERNELIZED_HOT_PATH_FAIL_CLOSED_TARGET_NAME = (
    "step4a_native_kernelized_hot_path_fail_closed"
)

B2_3_STANDALONE_QACC_APPLY_COMMIT_SHA = (
    "5d9064322d86158801ac80a931082f75c69cfda4"
)
B2_3_STANDALONE_QACC_APPLY_GPU_RECEIPT_MSG_ID = "1781972683995"

B2_4_COMPOSITION_QACC_APPLY_COMMIT_SHA = (
    "99727af5dc7b9eac989096b34bdfb46586fe6c12"
)
B2_4_COMPOSITION_GPU_RECEIPT_MSG_ID = "1781979673113"

NATIVE_KERNELIZED_HOT_PATH_REQUIRED_BLOCKER_ANCHORS = (
    "qacc_kernelized_false",
    "qacc_update_vote_selection_apply_cpu_reference",
    "triton_preplan_only",
    "global_cap_margin_only_reference_default_off",
    "full_loop_reference_stitch_no_native_speed_claim",
    "device_cuda_not_hot_loop_residency",
)
NATIVE_KERNELIZED_HOT_PATH_ALLOWED_BLOCKER_ANCHORS = (
    NATIVE_KERNELIZED_HOT_PATH_REQUIRED_BLOCKER_ANCHORS
)

NATIVE_KERNELIZED_HOT_PATH_BLOCKED_REASON = (
    "fail-closed native kernelized hot-path harness only; standalone q_acc_apply "
    "apply-kernel proven (B2-3), but qacc_kernelized=false, qacc update/"
    "vote-selection remain CPU-reference in the strict scaffold, Triton preplan "
    "is elementwise-only, global cap is MARGIN-only/default-off reference, and "
    "the full-loop receipt is a reference stitch with no native custom kernel "
    "speed claim"
)
NATIVE_KERNELIZED_HOT_PATH_DEVICE_LAUNDERING_CAVEAT = (
    "device=cuda, VRAM residency, torch-CUDA reference tensors, or CPU row "
    "materialization before apply are not true hot-loop residency"
)
NATIVE_KERNELIZED_HOT_PATH_FAIL_CLOSED_NON_CLAIMS = (
    "native hot-path blocker characterization is not learning, acquisition, retention, or throughput",
    "device=cuda and VRAM residency are not hot-loop residency or kernelization proof",
    "torch-CUDA reference seams are blocker evidence, not native custom kernel readiness",
    "CPU row materialization before q/acc apply keeps the hot loop blocked",
    "composed-path q_acc_apply APPLY parity is proven (B2-4); cap SELECTION, full-loop, and hot-loop residency are not",
    "this receipt does not launch GPU, acquire a resource lane, write checkpoints, or mutate .pt artifacts",
)

_DEFAULT_BLOCKER_ANCHORS = (
    {
        "anchor_name": "qacc_kernelized_false",
        "source_anchor": "calm/hrm_text_158/native_full_stack/sub2_native_birth_scaffold.py:858",
        "evidence": "strict scaffold rejects reports unless qacc_kernelized=false",
    },
    {
        "anchor_name": "qacc_update_vote_selection_apply_cpu_reference",
        "source_anchor": "calm/hrm_text_158/native_full_stack/sub2_native_birth_scaffold.py:963",
        "evidence": "strict scaffold treats qacc update/vote-selection/apply as CPU-reference blocker fields",
    },
    {
        "anchor_name": "triton_preplan_only",
        "source_anchor": "calm/hrm_text_158/native_full_stack/vote_update.py:705",
        "evidence": "Triton preplan covers decayed+vote/candidate/direction only; ordering, veto residuals, and q mutation remain CPU reference",
    },
    {
        "anchor_name": "global_cap_margin_only_reference_default_off",
        "source_anchor": "calm/hrm_text_158/native_full_stack/global_rate_cap_gpu.py:1",
        "evidence": "global cap CUDA bridge is default-off MARGIN-only/reference and not trainer/full-loop migration",
    },
    {
        "anchor_name": "full_loop_reference_stitch_no_native_speed_claim",
        "source_anchor": "calm/hrm_text_158/native_full_stack/full_loop_receipt.py:332",
        "evidence": "full-loop receipt is an engineering reference stitch with native_custom_kernel_speed_claim=False",
    },
    {
        "anchor_name": "device_cuda_not_hot_loop_residency",
        "source_anchor": "calm/hrm_text_158/native_full_stack/full_loop_receipt.py:1",
        "evidence": "device placement and torch-CUDA references do not prove device-resident kernelized update-law hot-loop residency",
    },
)


@dataclass(frozen=True)
class NativeKernelizedHotPathBlockerObservation:
    anchor_name: str
    source_anchor: str
    evidence: str
    blocker_kind: str = "pre_full_stack_hot_loop_residency_blocker"

    def to_dict(self) -> dict[str, str]:
        return {
            "anchor_name": self.anchor_name,
            "source_anchor": self.source_anchor,
            "evidence": self.evidence,
            "blocker_kind": self.blocker_kind,
        }


@dataclass(frozen=True)
class NativeKernelizedHotPathFailClosedReceipt:
    schema_version: str
    target_name: str
    allowed_blocker_anchors: tuple[str, ...]
    required_blocker_anchors: tuple[str, ...]
    native_kernelized_hot_path_claim: bool
    hot_loop_residency_claim: bool
    device_cuda_laundering_claim: bool
    readiness_row_flip_authorized: bool
    qacc_kernelized: bool
    qacc_update_over_64_cpu_reference: bool
    vote_selection_cpu_reference: bool
    q_acc_apply_cpu_reference: bool
    triton_preplan_only: bool
    q_acc_apply_final_row_torch_cuda_reference: bool
    global_cap_margin_only_reference: bool
    full_loop_native_custom_kernel_speed_claim: bool
    real_device_resident_kernelized_hot_loop_present: bool
    exact_cpu_oracle_parity_present: bool
    gpu_runtime_receipt_present: bool
    no_cpu_row_materialization_before_apply: bool
    standalone_qacc_apply_native_proven: bool
    standalone_qacc_apply_exact_parity_present: bool
    standalone_qacc_apply_gpu_receipt_present: bool
    composition_qacc_apply_native_proven: bool
    composition_exact_parity_present: bool
    composition_gpu_receipt_present: bool
    ready_to_flip: bool
    blocked_reason: str
    blocker_anchors: tuple[NativeKernelizedHotPathBlockerObservation, ...]
    device_laundering_caveat: str
    smallest_missing_proof: str
    non_claims: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "target_name": self.target_name,
            "allowed_blocker_anchors": list(self.allowed_blocker_anchors),
            "required_blocker_anchors": list(self.required_blocker_anchors),
            "native_kernelized_hot_path_claim": self.native_kernelized_hot_path_claim,
            "hot_loop_residency_claim": self.hot_loop_residency_claim,
            "device_cuda_laundering_claim": self.device_cuda_laundering_claim,
            "readiness_row_flip_authorized": self.readiness_row_flip_authorized,
            "qacc_kernelized": self.qacc_kernelized,
            "qacc_update_over_64_cpu_reference": self.qacc_update_over_64_cpu_reference,
            "vote_selection_cpu_reference": self.vote_selection_cpu_reference,
            "q_acc_apply_cpu_reference": self.q_acc_apply_cpu_reference,
            "triton_preplan_only": self.triton_preplan_only,
            "q_acc_apply_final_row_torch_cuda_reference": self.q_acc_apply_final_row_torch_cuda_reference,
            "global_cap_margin_only_reference": self.global_cap_margin_only_reference,
            "full_loop_native_custom_kernel_speed_claim": self.full_loop_native_custom_kernel_speed_claim,
            "real_device_resident_kernelized_hot_loop_present": self.real_device_resident_kernelized_hot_loop_present,
            "exact_cpu_oracle_parity_present": self.exact_cpu_oracle_parity_present,
            "gpu_runtime_receipt_present": self.gpu_runtime_receipt_present,
            "no_cpu_row_materialization_before_apply": self.no_cpu_row_materialization_before_apply,
            "standalone_qacc_apply_native_proven": self.standalone_qacc_apply_native_proven,
            "standalone_qacc_apply_exact_parity_present": (
                self.standalone_qacc_apply_exact_parity_present
            ),
            "standalone_qacc_apply_gpu_receipt_present": (
                self.standalone_qacc_apply_gpu_receipt_present
            ),
            "composition_qacc_apply_native_proven": self.composition_qacc_apply_native_proven,
            "composition_exact_parity_present": self.composition_exact_parity_present,
            "composition_gpu_receipt_present": self.composition_gpu_receipt_present,
            "ready_to_flip": self.ready_to_flip,
            "blocked_reason": self.blocked_reason,
            "blocker_anchors": [anchor.to_dict() for anchor in self.blocker_anchors],
            "device_laundering_caveat": self.device_laundering_caveat,
            "smallest_missing_proof": self.smallest_missing_proof,
            "non_claims": list(self.non_claims),
        }


def _require_nonempty_string(value: object, *, field_name: str) -> str:
    text = str(value)
    if not text.strip():
        raise ValueError(f"{field_name} must be non-empty")
    return text


def _future_proof_gate(receipt: NativeKernelizedHotPathFailClosedReceipt) -> bool:
    """Evaluate the composed hot-loop gate (standalone evidence excluded)."""

    return (
        receipt.real_device_resident_kernelized_hot_loop_present
        and receipt.exact_cpu_oracle_parity_present
        and receipt.gpu_runtime_receipt_present
        and receipt.no_cpu_row_materialization_before_apply
        and receipt.qacc_kernelized
        and not receipt.qacc_update_over_64_cpu_reference
        and not receipt.vote_selection_cpu_reference
        and not receipt.q_acc_apply_cpu_reference
        and not receipt.triton_preplan_only
        and not receipt.q_acc_apply_final_row_torch_cuda_reference
        and not receipt.global_cap_margin_only_reference
        and receipt.full_loop_native_custom_kernel_speed_claim
    )


def _summarize_native_hot_path_blocker_anchors(
    blocker_anchors: Sequence[Mapping[str, object]],
) -> tuple[NativeKernelizedHotPathBlockerObservation, ...]:
    grouped: dict[str, Mapping[str, object] | None] = {
        anchor: None for anchor in NATIVE_KERNELIZED_HOT_PATH_ALLOWED_BLOCKER_ANCHORS
    }
    for item in blocker_anchors:
        anchor_name = item.get("anchor_name", item.get("name"))
        if anchor_name not in grouped:
            raise ValueError(
                "native_kernelized_hot_path receipt anchors must be exactly the "
                f"Step 4A allowlist {NATIVE_KERNELIZED_HOT_PATH_ALLOWED_BLOCKER_ANCHORS!r}; "
                f"got {anchor_name!r}"
            )
        if grouped[str(anchor_name)] is not None:
            raise ValueError(
                f"duplicate native_kernelized_hot_path blocker anchor: {anchor_name!r}"
            )
        grouped[str(anchor_name)] = item

    missing = [anchor for anchor, item in grouped.items() if item is None]
    if missing:
        raise ValueError(
            "native_kernelized_hot_path receipt missing required blocker anchors: "
            + ", ".join(missing)
        )

    observations: list[NativeKernelizedHotPathBlockerObservation] = []
    for anchor_name in NATIVE_KERNELIZED_HOT_PATH_REQUIRED_BLOCKER_ANCHORS:
        item = grouped[anchor_name]
        assert item is not None
        observations.append(
            NativeKernelizedHotPathBlockerObservation(
                anchor_name=anchor_name,
                source_anchor=_require_nonempty_string(
                    item.get("source_anchor", ""),
                    field_name=f"{anchor_name}.source_anchor",
                ),
                evidence=_require_nonempty_string(
                    item.get("evidence", ""),
                    field_name=f"{anchor_name}.evidence",
                ),
                blocker_kind=_require_nonempty_string(
                    item.get("blocker_kind", "pre_full_stack_hot_loop_residency_blocker"),
                    field_name=f"{anchor_name}.blocker_kind",
                ),
            )
        )
    return tuple(observations)


def build_native_kernelized_hot_path_fail_closed_receipt(
    *,
    blocker_anchors: Sequence[Mapping[str, object]] = _DEFAULT_BLOCKER_ANCHORS,
    native_kernelized_hot_path_claim: bool = False,
    hot_loop_residency_claim: bool = False,
    device_cuda_laundering_claim: bool = False,
    readiness_row_flip_authorized: bool = False,
    qacc_kernelized: bool = False,
    qacc_update_over_64_cpu_reference: bool = True,
    vote_selection_cpu_reference: bool = True,
    q_acc_apply_cpu_reference: bool = True,
    triton_preplan_only: bool = True,
    q_acc_apply_final_row_torch_cuda_reference: bool = False,
    global_cap_margin_only_reference: bool = True,
    full_loop_native_custom_kernel_speed_claim: bool = False,
    real_device_resident_kernelized_hot_loop_present: bool = False,
    exact_cpu_oracle_parity_present: bool = False,
    gpu_runtime_receipt_present: bool = False,
    no_cpu_row_materialization_before_apply: bool = True,
    standalone_qacc_apply_native_proven: bool = True,
    standalone_qacc_apply_exact_parity_present: bool = True,
    standalone_qacc_apply_gpu_receipt_present: bool = True,
    composition_qacc_apply_native_proven: bool = True,
    composition_exact_parity_present: bool = True,
    composition_gpu_receipt_present: bool = True,
    ready_to_flip: bool = False,
    smallest_missing_proof: str = (
        "composed-path cap SELECTION native proof, full-loop native proof + "
        "speed claim, vote-selection + triton-preplan + qacc kernelization, "
        "and reviewed GPU receipt for the full hot loop"
    ),
) -> NativeKernelizedHotPathFailClosedReceipt:
    """Build the Step 4A fail-closed native hot-path blocker receipt."""

    receipt = NativeKernelizedHotPathFailClosedReceipt(
        schema_version=NATIVE_KERNELIZED_HOT_PATH_FAIL_CLOSED_RECEIPT_SCHEMA_VERSION,
        target_name=NATIVE_KERNELIZED_HOT_PATH_FAIL_CLOSED_TARGET_NAME,
        allowed_blocker_anchors=NATIVE_KERNELIZED_HOT_PATH_ALLOWED_BLOCKER_ANCHORS,
        required_blocker_anchors=NATIVE_KERNELIZED_HOT_PATH_REQUIRED_BLOCKER_ANCHORS,
        native_kernelized_hot_path_claim=bool(native_kernelized_hot_path_claim),
        hot_loop_residency_claim=bool(hot_loop_residency_claim),
        device_cuda_laundering_claim=bool(device_cuda_laundering_claim),
        readiness_row_flip_authorized=bool(readiness_row_flip_authorized),
        qacc_kernelized=bool(qacc_kernelized),
        qacc_update_over_64_cpu_reference=bool(qacc_update_over_64_cpu_reference),
        vote_selection_cpu_reference=bool(vote_selection_cpu_reference),
        q_acc_apply_cpu_reference=bool(q_acc_apply_cpu_reference),
        triton_preplan_only=bool(triton_preplan_only),
        q_acc_apply_final_row_torch_cuda_reference=bool(
            q_acc_apply_final_row_torch_cuda_reference
        ),
        global_cap_margin_only_reference=bool(global_cap_margin_only_reference),
        full_loop_native_custom_kernel_speed_claim=bool(
            full_loop_native_custom_kernel_speed_claim
        ),
        real_device_resident_kernelized_hot_loop_present=bool(
            real_device_resident_kernelized_hot_loop_present
        ),
        exact_cpu_oracle_parity_present=bool(exact_cpu_oracle_parity_present),
        gpu_runtime_receipt_present=bool(gpu_runtime_receipt_present),
        no_cpu_row_materialization_before_apply=bool(
            no_cpu_row_materialization_before_apply
        ),
        standalone_qacc_apply_native_proven=bool(standalone_qacc_apply_native_proven),
        standalone_qacc_apply_exact_parity_present=bool(
            standalone_qacc_apply_exact_parity_present
        ),
        standalone_qacc_apply_gpu_receipt_present=bool(
            standalone_qacc_apply_gpu_receipt_present
        ),
        composition_qacc_apply_native_proven=bool(composition_qacc_apply_native_proven),
        composition_exact_parity_present=bool(composition_exact_parity_present),
        composition_gpu_receipt_present=bool(composition_gpu_receipt_present),
        ready_to_flip=bool(ready_to_flip),
        blocked_reason=NATIVE_KERNELIZED_HOT_PATH_BLOCKED_REASON,
        blocker_anchors=_summarize_native_hot_path_blocker_anchors(blocker_anchors),
        device_laundering_caveat=NATIVE_KERNELIZED_HOT_PATH_DEVICE_LAUNDERING_CAVEAT,
        smallest_missing_proof=_require_nonempty_string(
            smallest_missing_proof,
            field_name="smallest_missing_proof",
        ),
        non_claims=NATIVE_KERNELIZED_HOT_PATH_FAIL_CLOSED_NON_CLAIMS,
    )
    validate_native_kernelized_hot_path_fail_closed_receipt(receipt)
    return receipt


def validate_native_kernelized_hot_path_fail_closed_receipt(
    receipt: NativeKernelizedHotPathFailClosedReceipt,
) -> None:
    if (
        receipt.schema_version
        != NATIVE_KERNELIZED_HOT_PATH_FAIL_CLOSED_RECEIPT_SCHEMA_VERSION
    ):
        raise ValueError("native_kernelized_hot_path fail-closed receipt schema mismatch")
    if receipt.target_name != NATIVE_KERNELIZED_HOT_PATH_FAIL_CLOSED_TARGET_NAME:
        raise ValueError("native_kernelized_hot_path fail-closed receipt target mismatch")
    if (
        receipt.allowed_blocker_anchors
        != NATIVE_KERNELIZED_HOT_PATH_ALLOWED_BLOCKER_ANCHORS
    ):
        raise ValueError("native_kernelized_hot_path allowed blocker anchors must be exact")
    if (
        receipt.required_blocker_anchors
        != NATIVE_KERNELIZED_HOT_PATH_REQUIRED_BLOCKER_ANCHORS
    ):
        raise ValueError("native_kernelized_hot_path required blocker anchors must be exact")
    observed_names = tuple(anchor.anchor_name for anchor in receipt.blocker_anchors)
    if observed_names != NATIVE_KERNELIZED_HOT_PATH_REQUIRED_BLOCKER_ANCHORS:
        raise ValueError(
            "native_kernelized_hot_path blocker anchors must match the required Step 4A set"
        )
    for anchor in receipt.blocker_anchors:
        if not anchor.source_anchor or not anchor.evidence:
            raise ValueError(f"{anchor.anchor_name} is missing anchor evidence")
        if "hot_loop" not in anchor.blocker_kind:
            raise ValueError(
                f"{anchor.anchor_name} must remain classified as hot-loop blocker evidence"
            )

    standalone_evidence = (
        receipt.standalone_qacc_apply_exact_parity_present,
        receipt.standalone_qacc_apply_gpu_receipt_present,
    )
    if receipt.standalone_qacc_apply_native_proven:
        if not all(standalone_evidence):
            raise ValueError(
                "standalone_qacc_apply_native_proven requires coupled exact-parity "
                "and GPU-receipt evidence fields"
            )
        laundering_flags = (
            receipt.ready_to_flip,
            receipt.native_kernelized_hot_path_claim,
            receipt.hot_loop_residency_claim,
            receipt.readiness_row_flip_authorized,
        )
        if any(laundering_flags):
            raise ValueError(
                "standalone_qacc_apply_native_proven cannot coexist with "
                "ready_to_flip or hot-path laundering claims"
            )
    elif any(standalone_evidence):
        raise ValueError(
            "partial standalone q_acc_apply evidence without "
            "standalone_qacc_apply_native_proven is invalid"
        )

    composition_evidence = (
        receipt.composition_exact_parity_present,
        receipt.composition_gpu_receipt_present,
    )
    composition_conjuncts_flipped = (
        not receipt.q_acc_apply_final_row_torch_cuda_reference
        and receipt.no_cpu_row_materialization_before_apply
    )
    if receipt.composition_qacc_apply_native_proven:
        if not all(composition_evidence):
            raise ValueError(
                "composition_qacc_apply_native_proven requires coupled evidence fields"
            )
        if receipt.q_acc_apply_final_row_torch_cuda_reference:
            raise ValueError(
                "composition_qacc_apply_native_proven requires "
                "q_acc_apply_final_row_torch_cuda_reference is False"
            )
        if not receipt.no_cpu_row_materialization_before_apply:
            raise ValueError(
                "composition_qacc_apply_native_proven requires "
                "no_cpu_row_materialization_before_apply is True"
            )
        laundering_flags = (
            receipt.ready_to_flip,
            receipt.native_kernelized_hot_path_claim,
            receipt.hot_loop_residency_claim,
            receipt.readiness_row_flip_authorized,
        )
        if any(laundering_flags):
            raise ValueError(
                "composition_qacc_apply_native_proven cannot coexist with "
                "ready_to_flip or hot-path laundering claims"
            )
    elif any(composition_evidence):
        raise ValueError(
            "partial composition q_acc_apply evidence without "
            "composition_qacc_apply_native_proven is invalid"
        )
    elif composition_conjuncts_flipped:
        raise ValueError(
            "flipped composition conjuncts without "
            "composition_qacc_apply_native_proven evidence is invalid"
        )

    future_proof_gate = _future_proof_gate(receipt)
    laundering_claims = {
        "native_kernelized_hot_path_claim": (
            receipt.native_kernelized_hot_path_claim
        ),
        "hot_loop_residency_claim": receipt.hot_loop_residency_claim,
        "readiness_row_flip_authorized": receipt.readiness_row_flip_authorized,
    }
    for label, value in laundering_claims.items():
        if bool(value) and not (future_proof_gate and receipt.ready_to_flip):
            raise ValueError(
                f"{label} requires real device-resident kernelized hot loop, "
                "exact CPU-oracle parity, no CPU row materialization before apply, "
                "GPU receipt, and ready_to_flip=True"
            )
    if receipt.device_cuda_laundering_claim:
        raise ValueError(
            "device_cuda_laundering_claim is always invalid: device=cuda is not "
            "true hot-loop residency"
        )
    if receipt.ready_to_flip and not (
        future_proof_gate
        and receipt.native_kernelized_hot_path_claim
        and receipt.hot_loop_residency_claim
        and receipt.readiness_row_flip_authorized
    ):
        raise ValueError(
            "ready_to_flip cannot be true without native kernelized hot-loop "
            "residency proof and explicit non-laundered row authorization"
        )
    if receipt.blocked_reason != NATIVE_KERNELIZED_HOT_PATH_BLOCKED_REASON:
        raise ValueError("native_kernelized_hot_path blocked reason must be exact")
    if (
        receipt.device_laundering_caveat
        != NATIVE_KERNELIZED_HOT_PATH_DEVICE_LAUNDERING_CAVEAT
    ):
        raise ValueError(
            "native_kernelized_hot_path must keep device-vs-hot-loop caveat exact"
        )
    if receipt.non_claims != NATIVE_KERNELIZED_HOT_PATH_FAIL_CLOSED_NON_CLAIMS:
        raise ValueError("native_kernelized_hot_path receipt non-claims must be exact")
