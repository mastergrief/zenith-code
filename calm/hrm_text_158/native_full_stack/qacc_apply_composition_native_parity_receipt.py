"""B2-4 composition-scoped native q_acc_apply parity receipt (schema/validator only).

Pass mint requires real GPU composed-path exact parity (test-operator launch).
Builder is fail-closed; direct dataclass + validate() for GPU proof only.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

from calm.hrm_text_158.native_full_stack.qacc_apply_native_parity_receipt import (
    QaccApplyNativeToken,
    canonical_tensor_payload_sha256,
)

QACC_APPLY_COMPOSITION_NATIVE_PARITY_RECEIPT_SCHEMA_VERSION = (
    "hrm_text_158_qacc_apply_composition_native_parity/v0.b2_4_impl"
)
B2_3_PARENT_COMMIT = "5d9064322d86158801ac80a931082f75c69cfda4"

QACC_APPLY_B2_4_BLOCKED_REASON = (
    "b2_4_impl_fail_closed_no_composition_parity_until_gpu_proof"
)

QACC_APPLY_B2_4_NON_CLAIMS = (
    "B2-4-impl does not flip native_kernelized_hot_path gate conjuncts",
    "B2-4-impl does not flip optimizer_credit_state or readiness rows",
    "composition_qacc_apply_parity_pass=True requires non-skipped GPU composed-path proof",
    "standalone B2-3 proof does not imply composed-path proof",
    "gpu_command_satisfied=True is minted only by real GPU composition exact-parity",
)


@dataclass(frozen=True)
class QaccApplyCompositionNativeParityReceipt:
    schema_version: str = QACC_APPLY_COMPOSITION_NATIVE_PARITY_RECEIPT_SCHEMA_VERSION
    parent_b2_3_commit: str = B2_3_PARENT_COMMIT
    composition_qacc_apply_parity_pass: bool = False
    gpu_command_satisfied: bool = False
    no_cpu_row_materialization_before_apply: bool = False
    composition_native_routing: bool = False
    exact_q_output_hash: str = ""
    exact_acc_output_hash: str = ""
    cpu_oracle_q_hash: str = ""
    cpu_oracle_acc_hash: str = ""
    parity_atol: float = 0.0
    parity_rtol: float = 0.0
    wrapper_token: QaccApplyNativeToken | None = None
    blocked_reason: str = QACC_APPLY_B2_4_BLOCKED_REASON
    non_claims: tuple[str, ...] = QACC_APPLY_B2_4_NON_CLAIMS


def validate_qacc_apply_composition_native_parity_receipt(
    receipt: QaccApplyCompositionNativeParityReceipt,
) -> None:
    if (
        receipt.schema_version
        != QACC_APPLY_COMPOSITION_NATIVE_PARITY_RECEIPT_SCHEMA_VERSION
    ):
        raise ValueError("composition parity receipt schema mismatch")
    if receipt.blocked_reason != QACC_APPLY_B2_4_BLOCKED_REASON:
        raise ValueError("composition parity blocked_reason must be exact")
    if receipt.non_claims != QACC_APPLY_B2_4_NON_CLAIMS:
        raise ValueError("composition parity non_claims must be exact")

    pass_state = bool(receipt.composition_qacc_apply_parity_pass)
    gpu_state = bool(receipt.gpu_command_satisfied)
    if pass_state != gpu_state:
        raise ValueError("composition pass-state requires both pass-bits equal")
    if pass_state:
        if not receipt.composition_native_routing:
            raise ValueError("pass-state requires composition_native_routing=True")
        if not receipt.no_cpu_row_materialization_before_apply:
            raise ValueError(
                "pass-state requires no_cpu_row_materialization_before_apply=True"
            )
        if receipt.wrapper_token is None:
            raise ValueError("pass-state requires wrapper_token from composition launch")
        if not receipt.exact_q_output_hash or not receipt.exact_acc_output_hash:
            raise ValueError("pass-state requires exact output hashes")
        if (
            receipt.exact_q_output_hash != receipt.cpu_oracle_q_hash
            or receipt.exact_acc_output_hash != receipt.cpu_oracle_acc_hash
        ):
            raise ValueError("pass-state requires exact/oracle hash equality")
        if receipt.parity_atol != 0.0 or receipt.parity_rtol != 0.0:
            raise ValueError("pass-state requires atol=0 and rtol=0")
    else:
        if receipt.wrapper_token is not None:
            raise ValueError("fail-state must not carry wrapper_token")


def composition_receipt_tensor_hash(tensor_bytes: bytes) -> str:
    return canonical_tensor_payload_sha256(tensor_bytes)


def composition_receipt_to_dict(
    receipt: QaccApplyCompositionNativeParityReceipt,
) -> dict[str, Any]:
    return {
        "schema_version": receipt.schema_version,
        "parent_b2_3_commit": receipt.parent_b2_3_commit,
        "composition_qacc_apply_parity_pass": receipt.composition_qacc_apply_parity_pass,
        "gpu_command_satisfied": receipt.gpu_command_satisfied,
        "no_cpu_row_materialization_before_apply": (
            receipt.no_cpu_row_materialization_before_apply
        ),
        "composition_native_routing": receipt.composition_native_routing,
        "exact_q_output_hash": receipt.exact_q_output_hash,
        "exact_acc_output_hash": receipt.exact_acc_output_hash,
        "cpu_oracle_q_hash": receipt.cpu_oracle_q_hash,
        "cpu_oracle_acc_hash": receipt.cpu_oracle_acc_hash,
        "parity_atol": receipt.parity_atol,
        "parity_rtol": receipt.parity_rtol,
        "wrapper_token_present": receipt.wrapper_token is not None,
        "blocked_reason": receipt.blocked_reason,
        "non_claims": list(receipt.non_claims),
    }
