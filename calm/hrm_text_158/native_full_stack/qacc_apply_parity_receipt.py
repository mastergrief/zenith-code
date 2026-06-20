"""Fail-closed QaccApplyParityReceipt for B1 (schema/validator/builder only; no native kernel).

B1 scope: receipt dataclass + builder + validator.  All bools default False.
parity_pass remains False until B2 GPU parity.  Native marker is defined as
positive proof + negative exclusions, not a label.
"""
from __future__ import annotations

from dataclasses import dataclass

QACC_APPLY_PARITY_RECEIPT_SCHEMA_VERSION = (
    "hrm_text_158_qacc_apply_parity/v0.b1_fail_closed"
)
CPU_ORACLE_COMMIT_SHA_SHORT = "d4a846a"

QACC_APPLY_B1_BLOCKED_REASON = (
    "b1_fail_closed_no_native_qacc_parity_until_b2"
)

QACC_APPLY_B1_NON_CLAIMS = (
    "B1 does not implement or invoke a native q_acc_apply kernel",
    "B1 does not run GPU code, torch.cuda, or emit .pt artifacts",
    "B1 does not claim qacc_apply_parity_pass=True",
    "B1 does not flip optimizer_credit_state, readiness_row, or native_kernelized_hot_path",
    "global_cap_gpu_native=True is a laundering label, not native proof",
    "q_acc_apply_mutation_torch_cuda_reference_under_cap_rows is a reference path, not native proof",
)


@dataclass(frozen=True)
class QaccApplyParityReceipt:
    schema_version: str = QACC_APPLY_PARITY_RECEIPT_SCHEMA_VERSION
    cpu_oracle_commit_sha_short: str = CPU_ORACLE_COMMIT_SHA_SHORT
    qacc_apply_parity_pass: bool = False
    q_acc_apply_cpu_reference: bool = True
    q_acc_apply_final_row_torch_cuda_reference: bool = True
    q_acc_apply_native_hot_loop_kernel: bool = False
    native_call_path_marker_present: bool = False
    global_cap_gpu_native_marker_seen: bool = False
    torch_cuda_ref_under_cap_rows_invoked: bool = False
    liveness_fail: bool = False
    blocked_reason: str = QACC_APPLY_B1_BLOCKED_REASON
    non_claims: tuple[str, ...] = QACC_APPLY_B1_NON_CLAIMS


def build_qacc_apply_parity_receipt(
    *,
    qacc_apply_parity_pass: bool = False,
    q_acc_apply_cpu_reference: bool = True,
    q_acc_apply_final_row_torch_cuda_reference: bool = True,
    q_acc_apply_native_hot_loop_kernel: bool = False,
    native_call_path_marker_present: bool = False,
    global_cap_gpu_native_marker_seen: bool = False,
    torch_cuda_ref_under_cap_rows_invoked: bool = False,
    liveness_fail: bool = False,
    blocked_reason: str | None = None,
) -> QaccApplyParityReceipt:
    """Fail-closed builder for QaccApplyParityReceipt.

    B1 builder enforces anti-laundering invariants at construction time.
    Any prohibited input raises ValueError before the receipt is minted.
    The constructed receipt is then validated before return, so builder
    and validator remain synchronized as the single source of truth.
    """
    _blocked_reason = (
        blocked_reason
        if blocked_reason is not None
        else QACC_APPLY_B1_BLOCKED_REASON
    )

    # --- Input pre-checks (clearer early-error messages) ---

    if qacc_apply_parity_pass:
        raise ValueError(
            "B1 receipt cannot mint qacc_apply_parity_pass=True; "
            "native q_acc_apply kernel parity is deferred to B2"
        )

    if global_cap_gpu_native_marker_seen:
        raise ValueError(
            "global_cap_gpu_native_marker_seen=True is prohibited in B1; "
            "global_cap_gpu_native=True is a laundering label, not native proof"
        )

    if torch_cuda_ref_under_cap_rows_invoked:
        raise ValueError(
            "torch_cuda_ref_under_cap_rows_invoked=True is prohibited in B1; "
            "vote_update.py:1174 torch-CUDA reference path is not native proof"
        )

    # AMENDMENT 1: marker-present without actual native kernel = fail-closed trap
    if native_call_path_marker_present and not q_acc_apply_native_hot_loop_kernel:
        raise ValueError(
            "native_call_path_marker_present=True requires "
            "q_acc_apply_native_hot_loop_kernel=True (marker without kernel = laundering)"
        )

    if not _blocked_reason:
        raise ValueError(
            "blocked_reason must be a non-empty string (exact B1 constant)"
        )

    # AMENDMENT 2: exact blocked_reason for parity_pass=False state
    if not qacc_apply_parity_pass and _blocked_reason != QACC_APPLY_B1_BLOCKED_REASON:
        raise ValueError(
            f"blocked_reason must be exactly QACC_APPLY_B1_BLOCKED_REASON "
            f"in B1 fail-closed state; got {_blocked_reason!r}"
        )

    receipt = QaccApplyParityReceipt(
        schema_version=QACC_APPLY_PARITY_RECEIPT_SCHEMA_VERSION,
        cpu_oracle_commit_sha_short=CPU_ORACLE_COMMIT_SHA_SHORT,
        qacc_apply_parity_pass=False,
        q_acc_apply_cpu_reference=bool(q_acc_apply_cpu_reference),
        q_acc_apply_final_row_torch_cuda_reference=bool(
            q_acc_apply_final_row_torch_cuda_reference
        ),
        q_acc_apply_native_hot_loop_kernel=bool(q_acc_apply_native_hot_loop_kernel),
        native_call_path_marker_present=bool(native_call_path_marker_present),
        global_cap_gpu_native_marker_seen=False,
        torch_cuda_ref_under_cap_rows_invoked=False,
        liveness_fail=bool(liveness_fail),
        blocked_reason=_blocked_reason,
        non_claims=QACC_APPLY_B1_NON_CLAIMS,
    )

    # FIX 3: builder validates before return — single source of truth
    validate_qacc_apply_parity_receipt(receipt)
    return receipt


def validate_qacc_apply_parity_receipt(receipt: QaccApplyParityReceipt) -> None:
    """Fail-closed validator.  Accepts only the B1-legal state."""
    if receipt.schema_version != QACC_APPLY_PARITY_RECEIPT_SCHEMA_VERSION:
        raise ValueError(
            f"schema_version mismatch: expected {QACC_APPLY_PARITY_RECEIPT_SCHEMA_VERSION!r}, "
            f"got {receipt.schema_version!r}"
        )

    if receipt.cpu_oracle_commit_sha_short != CPU_ORACLE_COMMIT_SHA_SHORT:
        raise ValueError(
            f"cpu_oracle_commit_sha_short mismatch: expected {CPU_ORACLE_COMMIT_SHA_SHORT!r}, "
            f"got {receipt.cpu_oracle_commit_sha_short!r}"
        )

    # --- Field-level checks — every exposed B1-legal bool/field is validated ---

    if receipt.qacc_apply_parity_pass:
        raise ValueError(
            "qacc_apply_parity_pass must be False in B1 (no native parity proven)"
        )

    if not receipt.q_acc_apply_cpu_reference:
        raise ValueError(
            "q_acc_apply_cpu_reference must be True in B1 (reference path is the only active path)"
        )

    # FIX 1: final_row torch-CUDA reference must stay True in B1
    if not receipt.q_acc_apply_final_row_torch_cuda_reference:
        raise ValueError(
            "q_acc_apply_final_row_torch_cuda_reference must be True in B1 "
            "(final-row torch-CUDA reference is still active until B2)"
        )

    if receipt.q_acc_apply_native_hot_loop_kernel:
        raise ValueError(
            "q_acc_apply_native_hot_loop_kernel must be False in B1 (no native kernel yet)"
        )

    if receipt.native_call_path_marker_present:
        raise ValueError(
            "native_call_path_marker_present must be False in B1 "
            "(marker without kernel = laundering; deferred to B2)"
        )

    # FIX 2: liveness_fail must stay False in B1
    if receipt.liveness_fail:
        raise ValueError(
            "liveness_fail must be False in B1 (no liveness failure to report)"
        )

    # --- Negative exclusions ---
    if receipt.global_cap_gpu_native_marker_seen:
        raise ValueError(
            "global_cap_gpu_native_marker_seen must be False in B1; "
            "global_rate_cap_gpu.py:856 label is not native proof"
        )

    if receipt.torch_cuda_ref_under_cap_rows_invoked:
        raise ValueError(
            "torch_cuda_ref_under_cap_rows_invoked must be False in B1; "
            "vote_update.py:1174 reference path is not native proof"
        )

    # --- Exact constant assertions (AMENDMENT 2) ---
    if receipt.blocked_reason != QACC_APPLY_B1_BLOCKED_REASON:
        raise ValueError(
            f"blocked_reason must be exactly {QACC_APPLY_B1_BLOCKED_REASON!r}; "
            f"got {receipt.blocked_reason!r}"
        )

    if receipt.non_claims != QACC_APPLY_B1_NON_CLAIMS:
        raise ValueError(
            "non_claims must match the exact B1 tuple"
        )
