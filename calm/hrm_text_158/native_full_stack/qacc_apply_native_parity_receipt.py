"""Fail-closed QaccApplyNativeParityReceipt for B2 (schema/validator/builder; no kernel).

B2-2a scope: receipt dataclass + builder + validator + hash helpers.  All bools default False.
parity_pass remains False until B2-3 GPU parity.  gpu_command_satisfied forced False.
Native marker is defined as positive proof + negative exclusions, not a label.

Does NOT call torch.cuda, does NOT emit .pt artifacts.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

QACC_APPLY_NATIVE_PARITY_RECEIPT_SCHEMA_VERSION = (
    "hrm_text_158_qacc_apply_native_parity/v0.b2"
)
B1_PARENT_COMMIT = "ebfd6868ba8179226357cf93a3e06a9646d27751"

QACC_APPLY_B2_BLOCKED_REASON = (
    "b2a_fail_closed_no_native_qacc_parity_until_b2_3_gpu_proof"
)

QACC_APPLY_B2_NON_CLAIMS = (
    "B2-2a does not implement or invoke a native q_acc_apply kernel",
    "B2-2a does not run GPU code, torch.cuda, or emit .pt artifacts",
    "B2-2a builder CANNOT mint qacc_apply_parity_pass=True",
    "B2-2a builder CANNOT mint gpu_command_satisfied=True",
    "B2-2a does not accept caller-supplied mock tokens for pass-minting",
    "qacc_apply_parity_pass=True is the SOLE authority of B2-3 non-skipped GPU exact-parity",
    "B2-2a does not flip optimizer_credit_state, readiness_row, or native_kernelized_hot_path",
    "global_cap_gpu_native=True is a laundering label, not native proof",
    "q_acc_apply_mutation_torch_cuda_reference_under_cap_rows is a reference path, not native proof",
)

TOKEN_INPUT_PAYLOAD_KEYS = frozenset(
    {
        "q_levels",
        "new_accumulators",
        "accepted_indices",
        "accepted_directions",
        "accepted_thresholds",
        "replay_veto_indices",
        "replay_veto_directions",
        "replay_veto_thresholds",
        "original_accumulators",
        "mutate_outputs",
    }
)
TOKEN_OUTPUT_PAYLOAD_KEYS = frozenset({"q_levels", "accumulators"})


@dataclass(frozen=True)
class QaccApplyNativeToken:
    kernel_family: str = ""
    kernel_symbol: str = ""
    kernel_source_sha256: str = ""
    wrapper_launch_nonce: str = ""
    input_payload_hashes: dict[str, str] | None = None
    output_payload_hashes: dict[str, str] | None = None
    backend: str = ""
    launch_time_ns: int = 0


@dataclass(frozen=True)
class QaccApplyNativeParityReceipt:
    schema_version: str = QACC_APPLY_NATIVE_PARITY_RECEIPT_SCHEMA_VERSION
    parent_b1_commit: str = B1_PARENT_COMMIT
    native_hot_loop_kernel: bool = False
    native_call_path_marker_present: bool = False
    q_acc_apply_cpu_reference: bool = True
    q_acc_apply_final_row_torch_cuda_reference: bool = True
    global_cap_gpu_native_marker_seen: bool = False
    torch_cuda_ref_under_cap_rows_invoked: bool = False
    qacc_apply_parity_pass: bool = False
    accepted_indices_unique: bool = True
    replay_indices_unique: bool = True
    accepted_then_replay_order: bool = True
    mutate_outputs_path: str = "True"
    exact_q_output_hash: str = ""
    exact_acc_output_hash: str = ""
    cpu_oracle_q_hash: str = ""
    cpu_oracle_acc_hash: str = ""
    parity_atol: float = 0.0
    parity_rtol: float = 0.0
    wrapper_token: QaccApplyNativeToken | None = None
    gpu_command_satisfied: bool = False
    blocked_reason: str = QACC_APPLY_B2_BLOCKED_REASON
    non_claims: tuple[str, ...] = QACC_APPLY_B2_NON_CLAIMS


# --- hash helpers ---


def canonical_tensor_payload_sha256(tensor_bytes: bytes) -> str:
    return hashlib.sha256(tensor_bytes).hexdigest()


def _encode_bool(flag: bool) -> bytes:
    return b"1" if flag else b"0"


def hash_qacc_apply_input_payloads(
    *,
    q_levels_bytes: bytes,
    new_accumulators_bytes: bytes,
    accepted_indices_bytes: bytes,
    accepted_directions_bytes: bytes,
    accepted_thresholds_bytes: bytes,
    replay_veto_indices_bytes: bytes | None = None,
    replay_veto_directions_bytes: bytes | None = None,
    replay_veto_thresholds_bytes: bytes | None = None,
    original_accumulators_bytes: bytes | None = None,
    mutate_outputs: bool = True,
) -> dict[str, str]:
    """Always emit exactly TOKEN_INPUT_PAYLOAD_KEYS (10 keys).

    Missing optional inputs hash the canonical empty payload (b"")
    so key presence is deterministic regardless of runtime absence.
    """
    return {
        "q_levels": canonical_tensor_payload_sha256(q_levels_bytes),
        "new_accumulators": canonical_tensor_payload_sha256(new_accumulators_bytes),
        "accepted_indices": canonical_tensor_payload_sha256(accepted_indices_bytes),
        "accepted_directions": canonical_tensor_payload_sha256(accepted_directions_bytes),
        "accepted_thresholds": canonical_tensor_payload_sha256(accepted_thresholds_bytes),
        "replay_veto_indices": canonical_tensor_payload_sha256(
            replay_veto_indices_bytes if replay_veto_indices_bytes is not None else b""
        ),
        "replay_veto_directions": canonical_tensor_payload_sha256(
            replay_veto_directions_bytes if replay_veto_directions_bytes is not None else b""
        ),
        "replay_veto_thresholds": canonical_tensor_payload_sha256(
            replay_veto_thresholds_bytes if replay_veto_thresholds_bytes is not None else b""
        ),
        "original_accumulators": canonical_tensor_payload_sha256(
            original_accumulators_bytes if original_accumulators_bytes is not None else b""
        ),
        "mutate_outputs": canonical_tensor_payload_sha256(_encode_bool(mutate_outputs)),
    }


def hash_qacc_apply_output_payloads(
    *,
    q_levels_bytes: bytes,
    accumulators_bytes: bytes,
) -> dict[str, str]:
    return {
        "q_levels": canonical_tensor_payload_sha256(q_levels_bytes),
        "accumulators": canonical_tensor_payload_sha256(accumulators_bytes),
    }


# --- builder ---


def _pre_check_input_laundering(
    *,
    qacc_apply_parity_pass: bool,
    gpu_command_satisfied: bool,
    global_cap_gpu_native_marker_seen: bool,
    torch_cuda_ref_under_cap_rows_invoked: bool,
) -> None:
    if qacc_apply_parity_pass:
        raise ValueError(
            "B2-2a builder CANNOT mint qacc_apply_parity_pass=True; "
            "parity pass is the SOLE authority of B2-3 non-skipped GPU exact-parity"
        )

    if gpu_command_satisfied:
        raise ValueError(
            "B2-2a builder CANNOT mint gpu_command_satisfied=True; "
            "gpu_command_satisfied is the SOLE authority of B2-3 non-skipped GPU exact-parity"
        )

    if global_cap_gpu_native_marker_seen:
        raise ValueError(
            "global_cap_gpu_native_marker_seen=True is prohibited; "
            "global_cap_gpu_native=True is a laundering label"
        )

    if torch_cuda_ref_under_cap_rows_invoked:
        raise ValueError(
            "torch_cuda_ref_under_cap_rows_invoked=True is prohibited; "
            "vote_update.py:1174 torch-CUDA reference path is not native proof"
        )


def build_qacc_apply_native_parity_receipt(
    *,
    native_hot_loop_kernel: bool = False,
    native_call_path_marker_present: bool = False,
    q_acc_apply_cpu_reference: bool = True,
    q_acc_apply_final_row_torch_cuda_reference: bool = True,
    global_cap_gpu_native_marker_seen: bool = False,
    torch_cuda_ref_under_cap_rows_invoked: bool = False,
    qacc_apply_parity_pass: bool = False,
    accepted_indices_unique: bool = True,
    replay_indices_unique: bool = True,
    accepted_then_replay_order: bool = True,
    mutate_outputs_path: str = "True",
    exact_q_output_hash: str = "",
    exact_acc_output_hash: str = "",
    cpu_oracle_q_hash: str = "",
    cpu_oracle_acc_hash: str = "",
    parity_atol: float = 0.0,
    parity_rtol: float = 0.0,
    wrapper_token: QaccApplyNativeToken | None = None,
    gpu_command_satisfied: bool = False,
    blocked_reason: str | None = None,
) -> QaccApplyNativeParityReceipt:
    _blocked_reason = (
        blocked_reason if blocked_reason is not None else QACC_APPLY_B2_BLOCKED_REASON
    )

    _pre_check_input_laundering(
        qacc_apply_parity_pass=bool(qacc_apply_parity_pass),
        gpu_command_satisfied=bool(gpu_command_satisfied),
        global_cap_gpu_native_marker_seen=bool(global_cap_gpu_native_marker_seen),
        torch_cuda_ref_under_cap_rows_invoked=bool(
            torch_cuda_ref_under_cap_rows_invoked
        ),
    )

    if not _blocked_reason:
        raise ValueError("blocked_reason must be non-empty")
    if _blocked_reason != QACC_APPLY_B2_BLOCKED_REASON:
        raise ValueError(
            f"blocked_reason must be exactly {QACC_APPLY_B2_BLOCKED_REASON!r}; "
            f"got {_blocked_reason!r}"
        )

    # FIX 4: B2-2a builder forces token=None (blocked receipts carry no token)
    receipt = QaccApplyNativeParityReceipt(
        schema_version=QACC_APPLY_NATIVE_PARITY_RECEIPT_SCHEMA_VERSION,
        parent_b1_commit=B1_PARENT_COMMIT,
        native_hot_loop_kernel=bool(native_hot_loop_kernel),
        native_call_path_marker_present=bool(native_call_path_marker_present),
        q_acc_apply_cpu_reference=bool(q_acc_apply_cpu_reference),
        q_acc_apply_final_row_torch_cuda_reference=bool(
            q_acc_apply_final_row_torch_cuda_reference
        ),
        global_cap_gpu_native_marker_seen=False,
        torch_cuda_ref_under_cap_rows_invoked=False,
        qacc_apply_parity_pass=False,
        accepted_indices_unique=bool(accepted_indices_unique),
        replay_indices_unique=bool(replay_indices_unique),
        accepted_then_replay_order=bool(accepted_then_replay_order),
        mutate_outputs_path=str(mutate_outputs_path),
        exact_q_output_hash=str(exact_q_output_hash),
        exact_acc_output_hash=str(exact_acc_output_hash),
        cpu_oracle_q_hash=str(cpu_oracle_q_hash),
        cpu_oracle_acc_hash=str(cpu_oracle_acc_hash),
        parity_atol=float(parity_atol),
        parity_rtol=float(parity_rtol),
        wrapper_token=None,
        gpu_command_satisfied=False,
        blocked_reason=_blocked_reason,
        non_claims=QACC_APPLY_B2_NON_CLAIMS,
    )

    validate_qacc_apply_native_parity_receipt(receipt)
    return receipt


# --- validator ---


def _validate_token_pass(token: QaccApplyNativeToken) -> None:
    if token.kernel_family != "triton_qacc_apply":
        raise ValueError(
            f"wrapper_token.kernel_family must be 'triton_qacc_apply'; "
            f"got {token.kernel_family!r}"
        )
    if not token.kernel_symbol:
        raise ValueError("wrapper_token.kernel_symbol must be non-empty")
    if not token.kernel_source_sha256:
        raise ValueError("wrapper_token.kernel_source_sha256 must be non-empty")
    if not token.wrapper_launch_nonce:
        raise ValueError("wrapper_token.wrapper_launch_nonce must be non-empty")
    if not isinstance(token.input_payload_hashes, dict):
        raise ValueError("wrapper_token.input_payload_hashes must be a dict")
    if not isinstance(token.output_payload_hashes, dict):
        raise ValueError("wrapper_token.output_payload_hashes must be a dict")
    if token.backend != "cuda":
        raise ValueError(f"wrapper_token.backend must be 'cuda'; got {token.backend!r}")
    if token.launch_time_ns <= 0:
        raise ValueError("wrapper_token.launch_time_ns must be > 0")

    # FIX 3: exact key-set enforcement
    if set(token.input_payload_hashes.keys()) != TOKEN_INPUT_PAYLOAD_KEYS:
        raise ValueError(
            f"input_payload_hashes keys must be exactly {sorted(TOKEN_INPUT_PAYLOAD_KEYS)}"
        )
    if set(token.output_payload_hashes.keys()) != TOKEN_OUTPUT_PAYLOAD_KEYS:
        raise ValueError(
            f"output_payload_hashes keys must be exactly {sorted(TOKEN_OUTPUT_PAYLOAD_KEYS)}"
        )


def validate_qacc_apply_native_parity_receipt(
    receipt: QaccApplyNativeParityReceipt,
) -> None:
    # --- Absolute checks (both states) ---
    if receipt.schema_version != QACC_APPLY_NATIVE_PARITY_RECEIPT_SCHEMA_VERSION:
        raise ValueError(
            f"schema_version mismatch: expected {QACC_APPLY_NATIVE_PARITY_RECEIPT_SCHEMA_VERSION!r}, "
            f"got {receipt.schema_version!r}"
        )
    if receipt.parent_b1_commit != B1_PARENT_COMMIT:
        raise ValueError(
            f"parent_b1_commit mismatch: expected {B1_PARENT_COMMIT!r}, "
            f"got {receipt.parent_b1_commit!r}"
        )
    if receipt.global_cap_gpu_native_marker_seen:
        raise ValueError("global_cap_gpu_native_marker_seen must be False")
    if receipt.torch_cuda_ref_under_cap_rows_invoked:
        raise ValueError("torch_cuda_ref_under_cap_rows_invoked must be False")
    if not receipt.accepted_indices_unique:
        raise ValueError("accepted_indices_unique must be True")
    if not receipt.replay_indices_unique:
        raise ValueError("replay_indices_unique must be True")
    if not receipt.accepted_then_replay_order:
        raise ValueError("accepted_then_replay_order must be True")
    if receipt.mutate_outputs_path != "True":
        raise ValueError(
            f"mutate_outputs_path must be 'True'; got {receipt.mutate_outputs_path!r}"
        )
    if receipt.parity_atol != 0.0:
        raise ValueError(f"parity_atol must be 0.0; got {receipt.parity_atol!r}")
    if receipt.parity_rtol != 0.0:
        raise ValueError(f"parity_rtol must be 0.0; got {receipt.parity_rtol!r}")
    if receipt.blocked_reason != QACC_APPLY_B2_BLOCKED_REASON:
        raise ValueError(
            f"blocked_reason must be exactly {QACC_APPLY_B2_BLOCKED_REASON!r}; "
            f"got {receipt.blocked_reason!r}"
        )
    if receipt.non_claims != QACC_APPLY_B2_NON_CLAIMS:
        raise ValueError("non_claims must match the exact B2 tuple")

    # --- State branch ---
    if receipt.qacc_apply_parity_pass or receipt.gpu_command_satisfied:
        if not receipt.qacc_apply_parity_pass or not receipt.gpu_command_satisfied:
            raise ValueError(
                "pass-state requires BOTH qacc_apply_parity_pass=True "
                "and gpu_command_satisfied=True"
            )

        if receipt.q_acc_apply_cpu_reference:
            raise ValueError("q_acc_apply_cpu_reference must be False in pass-state")
        if receipt.q_acc_apply_final_row_torch_cuda_reference:
            raise ValueError("q_acc_apply_final_row_torch_cuda_reference must be False in pass-state")
        if not receipt.native_hot_loop_kernel:
            raise ValueError("native_hot_loop_kernel must be True in pass-state")
        if not receipt.native_call_path_marker_present:
            raise ValueError("native_call_path_marker_present must be True in pass-state")
        if receipt.exact_q_output_hash == "":
            raise ValueError("exact_q_output_hash must be non-empty in pass-state")
        if receipt.exact_acc_output_hash == "":
            raise ValueError("exact_acc_output_hash must be non-empty in pass-state")
        if receipt.exact_q_output_hash != receipt.cpu_oracle_q_hash:
            raise ValueError("exact_q_output_hash must equal cpu_oracle_q_hash")
        if receipt.exact_acc_output_hash != receipt.cpu_oracle_acc_hash:
            raise ValueError("exact_acc_output_hash must equal cpu_oracle_acc_hash")

        # TOKEN: required in pass-state (FIX 4 invariant: token present ⟺ pass-state)
        token = receipt.wrapper_token
        if token is None:
            raise ValueError("wrapper_token is required in pass-state")
        _validate_token_pass(token)
    else:
        # BLOCKED-STATE: token MUST be None (FIX 4)
        if receipt.wrapper_token is not None:
            raise ValueError(
                "wrapper_token must be None in blocked-state (token present ⟺ pass-state)"
            )
