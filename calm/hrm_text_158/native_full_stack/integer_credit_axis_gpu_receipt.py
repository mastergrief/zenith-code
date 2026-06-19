"""CPU-validatable scaffold for BR-3C-H GPU credit-axis kernel validation (H.1a).

Receipt builder/validator, two-tier branch classifier, and boundary-guard trap logic.
Does NOT implement Triton/CUDA kernel bodies (deferred to H.2).
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from typing import Any, Mapping, Sequence

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from calm.hrm_text_158.native_full_stack.integer_native_optimizer_credit_path_design import (
    BRANCH_D_INTEGER_VIABLE,
    IntegerCreditAxisIntegrationReceipt,
    canonical_tensor_payload_sha256,
    validate_integer_credit_axis_integration_receipt,
)
from calm.hrm_text_158.native_full_stack.integer_sparse_rank_votes import (
    CanonicalRankVoteBin,
    strict_integer_sparse_rank_bucketed_vote_events_from_credit,
)
from calm.hrm_text_158.native_full_stack.optimizer_credit_state import (
    OPTIMIZER_CREDIT_STATE_FP_EXCEPTION_CAVEAT,
    OPTIMIZER_CREDIT_STATE_FAIL_CLOSED_NON_CLAIMS,
)

CREDIT_AXIS_GPU_KERNEL_VALIDATION_SCHEMA_VERSION = (
    "hrm_text_158_credit_axis_gpu_kernel_validation/v0"
)
CPU_ORACLE_COMMIT_SHA = "d4a846a41cdf164e0718c711990518a305938650"
CPU_ORACLE_COMMIT_SHA_SHORT = "d4a846a"
AUTHORITY_GPU_EVIDENCE_ONLY = "gpu_evidence_only"

RUN_GPU_CREDIT_AXIS_KERNEL_ENV = "RUN_GPU_CREDIT_AXIS_KERNEL"

BR_H_GPU_KERNEL_MISSING = "BR-H-GPU-KERNEL-MISSING"
BR_H_GPU_KERNEL_SEAM_AMBIGUOUS = "BR-H-GPU-KERNEL-SEAM-AMBIGUOUS"
BR_H_GPU_DISPATCH_HELD = "BR-H-GPU-DISPATCH-HELD"
BR_H_LIVENESS_FAIL = "BR-H-LIVENESS-FAIL"
BR_H_NOT_KERNELIZED = "BR-H-NOT-KERNELIZED"
BR_H_HIDDEN_FP_BF16 = "BR-H-HIDDEN-FP-BF16"
BR_H_GPU_INTEGER_NONDETERMINISM_OR_OVERFLOW = (
    "BR-H-GPU-INTEGER-NONDETERMINISM-OR-OVERFLOW"
)
BR_H_PARITY_DRIFT = "BR-H-PARITY-DRIFT"
BR_H_NATIVE_INTEGER_PARITY_CLEAN = "BR-H-NATIVE-INTEGER-PARITY-CLEAN"

REGISTERED_GPU_VALIDATION_BRANCH_IDS = frozenset(
    {
        BR_H_GPU_KERNEL_MISSING,
        BR_H_GPU_KERNEL_SEAM_AMBIGUOUS,
        BR_H_GPU_DISPATCH_HELD,
        BR_H_LIVENESS_FAIL,
        BR_H_NOT_KERNELIZED,
        BR_H_HIDDEN_FP_BF16,
        BR_H_GPU_INTEGER_NONDETERMINISM_OR_OVERFLOW,
        BR_H_PARITY_DRIFT,
        BR_H_NATIVE_INTEGER_PARITY_CLEAN,
    }
)

GPU_PAYLOAD_HASH_KEYS = (
    "attribution_events_hash",
    "projected_move_indices_hash",
    "projected_moves_hash",
    "credit_q31_hash",
    "sparse_vote_events_hash",
)

CREDIT_AXIS_GPU_KERNEL_NON_CLAIMS = (
    *OPTIMIZER_CREDIT_STATE_FAIL_CLOSED_NON_CLAIMS,
    "GPU credit-axis kernel validation is gpu_evidence_only; does NOT flip optimizer_credit_state or native_kernelized_hot_path rows",
    "H.0/H.2 CLEAN proves credit-axis kernel parity only; qacc update/vote-selection/apply remain out of scope",
    "device=cuda residency is not hot-loop kernelization proof",
    "does NOT authorize full training, optimizer resume, or throughput claims",
    "parity CLEAN feeds readiness input rows only; flip deferred to separate GPU runtime receipt",
    "cpu_integration_branch_id (BR-D) and gpu_validation_branch_id (BR-H) are distinct namespaces; BR-D-INTEGER-VIABLE carry does NOT imply BR-H-NATIVE-INTEGER-PARITY-CLEAN",
)

FORBIDDEN_GPU_KERNEL_RECEIPT_FIELDS = (
    "ready_to_flip",
    "optimizer_credit_state_sub2_claim",
    "readiness_row_flip_authorized",
    "real_native_integer_attribution_present",
    "real_native_integer_credit_ranking_present",
    "native_kernelized_hot_path_sub2_claim",
    "qacc_kernelized",
    "gpu_runtime_receipt_present",
    "branch_d_integer_viable_claimed",
    "fp_exception_laundering_claim",
)

_FORBIDDEN_FP_DTYPES = frozenset({torch.float32, torch.bfloat16})
_FORBIDDEN_ATEN_OPS = frozenset(
    {
        "aten::to",
        "aten::_to_copy",
        "aten::log2",
        "aten::pow",
    }
)


def _is_forbidden_aten_op(op_name: str) -> bool:
    if op_name in _FORBIDDEN_ATEN_OPS:
        return True
    for forbidden in _FORBIDDEN_ATEN_OPS:
        base = forbidden.split(".", 1)[0]
        if op_name.startswith(f"{base}."):
            return True
    return False


class CreditAxisKernelBoundaryViolation(RuntimeError):
    """Raised when the credit-axis GPU boundary guard detects forbidden FP ops."""


@dataclass(frozen=True)
class CreditAxisStageNativeEvidence:
    s1_native: bool = False
    s2_native: bool = False
    s3_native: bool = False
    s4_native: bool = False

    @property
    def whole_pipeline_native(self) -> bool:
        return self.s1_native and self.s2_native and self.s3_native and self.s4_native


def torch_cuda_reference_only_from_stage_evidence(
    evidence: CreditAxisStageNativeEvidence,
) -> bool:
    return not evidence.whole_pipeline_native


def _recompute_torch_cuda_reference_only(
    *,
    torch_cuda_reference_only: bool,
    stage_native_evidence: CreditAxisStageNativeEvidence | None,
) -> bool:
    if stage_native_evidence is None:
        return torch_cuda_reference_only
    recomputed = torch_cuda_reference_only_from_stage_evidence(stage_native_evidence)
    if torch_cuda_reference_only != recomputed:
        raise ValueError(
            "torch_cuda_reference_only disagrees with stage_native_evidence recompute"
        )
    return recomputed


@dataclass(frozen=True)
class CreditAxisGpuKernelRuntimeEvidence:
    liveness_fail: bool = False
    hot_loop_kernel_invoked: bool = False
    torch_cuda_reference_only: bool = True
    hidden_fp_violation_count: int = 0
    boundary_or_manifest_dtype_violation: bool = False
    overflow_guard_tripped: bool = False
    gpu_output_repeat_stable: bool = False
    gpu_payload_hashes: Mapping[str, str] | None = None
    cpu_oracle_payload_hashes: Mapping[str, str] | None = None
    device_residency_cuda: bool = False
    hot_loop_integer_only: bool = False
    stage_native_evidence: CreditAxisStageNativeEvidence | None = None


@dataclass(frozen=True)
class CreditAxisGpuKernelValidationReceipt:
    schema_version: str
    cpu_oracle_commit_sha: str
    cpu_integration_receipt_digest: str
    cpu_integration_data_digest: str
    cpu_integration_branch_id: str
    gpu_validation_branch_id: str
    gpu_payload_hashes: dict[str, str]
    gpu_output_repeat_stable: bool
    overflow_guard_tripped: bool
    parity_pass: bool
    authority_level: str
    hot_loop_kernel_invoked: bool
    torch_cuda_reference_only: bool
    hidden_fp_violation_count: int
    device_residency_cuda: bool
    hot_loop_integer_only: bool
    fp_exception_caveat: str
    non_claims: tuple[str, ...]
    ready_to_flip: bool = False
    optimizer_credit_state_sub2_claim: bool = False
    readiness_row_flip_authorized: bool = False
    real_native_integer_attribution_present: bool = False
    real_native_integer_credit_ranking_present: bool = False
    native_kernelized_hot_path_sub2_claim: bool = False
    qacc_kernelized: bool = False
    gpu_runtime_receipt_present: bool = False
    branch_d_integer_viable_claimed: bool = False
    fp_exception_laundering_claim: bool = False


def credit_axis_gpu_kernel_hard_false_snapshot() -> dict[str, bool]:
    return {field: False for field in FORBIDDEN_GPU_KERNEL_RECEIPT_FIELDS}


def run_gpu_credit_axis_kernel_env_enabled() -> bool:
    return os.environ.get(RUN_GPU_CREDIT_AXIS_KERNEL_ENV, "").strip() == "1"


def _is_br_d_branch_id(branch_id: str) -> bool:
    return str(branch_id).startswith("BR-D-")


def _is_br_h_branch_id(branch_id: str) -> bool:
    return str(branch_id).startswith("BR-H-")


def cpu_oracle_payload_hashes_from_integration_receipt(
    receipt: IntegerCreditAxisIntegrationReceipt,
) -> dict[str, str]:
    return {
        "attribution_events_hash": receipt.attribution_events_hash,
        "projected_move_indices_hash": receipt.projected_move_indices_hash,
        "projected_moves_hash": receipt.projected_moves_hash,
        "credit_q31_hash": receipt.credit_q31_hash,
    }


@dataclass(frozen=True)
class CpuOraclePayloadHashesForGpuParity:
    """All 5 keys required for GPU parity / CLEAN gate."""

    attribution_events_hash: str
    projected_move_indices_hash: str
    projected_moves_hash: str
    credit_q31_hash: str
    sparse_vote_events_hash: str
    integration_data_digest_sha256: str
    integration_branch_id: str
    sparse_oracle_source: str


def attribution_events_payload_sha256(
    flat_indices: torch.Tensor,
    attribution_q31: torch.Tensor,
) -> str:
    flat_hash = canonical_tensor_payload_sha256(flat_indices)
    attr_hash = canonical_tensor_payload_sha256(attribution_q31)
    return hashlib.sha256((flat_hash + attr_hash).encode("utf-8")).hexdigest()


def sparse_vote_events_payload_sha256(
    indices: torch.Tensor,
    values: torch.Tensor,
) -> str:
    idx_hash = canonical_tensor_payload_sha256(indices)
    val_hash = canonical_tensor_payload_sha256(values)
    return hashlib.sha256((idx_hash + val_hash).encode("utf-8")).hexdigest()


def build_cpu_oracle_payload_hashes_for_gpu_parity(
    *,
    integration_receipt: IntegerCreditAxisIntegrationReceipt,
    credit_q31: torch.Tensor,
    projected_moves: torch.Tensor,
    projected_move_indices: torch.Tensor,
    rank_bin_spec_canonical: tuple[CanonicalRankVoteBin, ...],
    credit_law_id: str = "credit_neg_attribution_q31_v1",
) -> CpuOraclePayloadHashesForGpuParity:
    validate_integer_credit_axis_integration_receipt(integration_receipt)
    if integration_receipt.branch_id != BRANCH_D_INTEGER_VIABLE:
        raise ValueError("integration receipt branch_id must be BR-D-INTEGER-VIABLE")
    sparse = strict_integer_sparse_rank_bucketed_vote_events_from_credit(
        credit_q31=credit_q31,
        projected_moves=projected_moves,
        flat_indices=projected_move_indices,
        canonical_bins=rank_bin_spec_canonical,
        credit_law_id=credit_law_id,
    )
    sparse_hash = sparse_vote_events_payload_sha256(sparse.indices, sparse.values)
    return CpuOraclePayloadHashesForGpuParity(
        attribution_events_hash=integration_receipt.attribution_events_hash,
        projected_move_indices_hash=integration_receipt.projected_move_indices_hash,
        projected_moves_hash=integration_receipt.projected_moves_hash,
        credit_q31_hash=integration_receipt.credit_q31_hash,
        sparse_vote_events_hash=sparse_hash,
        integration_data_digest_sha256=integration_receipt.integration_data_digest_sha256,
        integration_branch_id=integration_receipt.branch_id,
        sparse_oracle_source="strict_integer_sparse_rank_bucketed_vote_events_from_credit",
    )


def cpu_oracle_payload_hashes_from_gpu_parity(
    oracle: CpuOraclePayloadHashesForGpuParity,
) -> dict[str, str]:
    return {
        "attribution_events_hash": oracle.attribution_events_hash,
        "projected_move_indices_hash": oracle.projected_move_indices_hash,
        "projected_moves_hash": oracle.projected_moves_hash,
        "credit_q31_hash": oracle.credit_q31_hash,
        "sparse_vote_events_hash": oracle.sparse_vote_events_hash,
    }


def pipeline_result_to_live_gpu_tensors(result: Any) -> dict[str, torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
    return {
        "attribution_events_hash": (result.flat_indices, result.attribution_q31),
        "projected_move_indices_hash": result.projected_move_indices,
        "projected_moves_hash": result.projected_moves,
        "credit_q31_hash": result.credit_q31,
        "sparse_vote_events_hash": (result.sparse_vote_indices, result.sparse_vote_values),
    }


def cpu_integration_receipt_digest_sha256(
    receipt: IntegerCreditAxisIntegrationReceipt,
) -> str:
    payload = {
        "branch_id": receipt.branch_id,
        "integration_data_digest_sha256": receipt.integration_data_digest_sha256,
        "attribution_events_hash": receipt.attribution_events_hash,
        "projected_move_indices_hash": receipt.projected_move_indices_hash,
        "projected_moves_hash": receipt.projected_moves_hash,
        "credit_q31_hash": receipt.credit_q31_hash,
        "candidate_run_id": receipt.candidate_run_id,
        "comparable_set_id": receipt.comparable_set_id,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _payload_hash_mismatch(
    *,
    gpu_payload_hashes: Mapping[str, str] | None,
    cpu_oracle_payload_hashes: Mapping[str, str] | None,
) -> bool:
    if gpu_payload_hashes is None or cpu_oracle_payload_hashes is None:
        return False
    for key in GPU_PAYLOAD_HASH_KEYS:
        if key not in gpu_payload_hashes or key not in cpu_oracle_payload_hashes:
            return True
        if gpu_payload_hashes[key] != cpu_oracle_payload_hashes[key]:
            return True
    return False


def _all_payload_hashes_match(
    *,
    gpu_payload_hashes: Mapping[str, str] | None,
    cpu_oracle_payload_hashes: Mapping[str, str] | None,
) -> bool:
    if gpu_payload_hashes is None or cpu_oracle_payload_hashes is None:
        return False
    for key in GPU_PAYLOAD_HASH_KEYS:
        if gpu_payload_hashes.get(key) != cpu_oracle_payload_hashes.get(key):
            return False
    return True


def classify_credit_axis_gpu_prelaunch_branch(
    *,
    triton_available: bool,
    cuda_available: bool,
    kernel_module_built: bool,
    seam_resolves_to_credit_axis_kernel: bool,
    dispatch_env_enabled: bool | None = None,
) -> str:
    if not triton_available or not cuda_available or not kernel_module_built:
        return BR_H_GPU_KERNEL_MISSING
    if not seam_resolves_to_credit_axis_kernel:
        return BR_H_GPU_KERNEL_SEAM_AMBIGUOUS
    if dispatch_env_enabled is None:
        dispatch_env_enabled = run_gpu_credit_axis_kernel_env_enabled()
    if not dispatch_env_enabled:
        return BR_H_GPU_DISPATCH_HELD
    raise ValueError("prelaunch checks passed; no tier-1 terminal branch")


def classify_credit_axis_gpu_runtime_branch(
    evidence: CreditAxisGpuKernelRuntimeEvidence,
) -> str:
    if evidence.liveness_fail:
        return BR_H_LIVENESS_FAIL
    if not evidence.hot_loop_kernel_invoked or evidence.torch_cuda_reference_only:
        return BR_H_NOT_KERNELIZED
    if (
        evidence.hidden_fp_violation_count > 0
        or evidence.boundary_or_manifest_dtype_violation
    ):
        return BR_H_HIDDEN_FP_BF16
    if evidence.overflow_guard_tripped or not evidence.gpu_output_repeat_stable:
        return BR_H_GPU_INTEGER_NONDETERMINISM_OR_OVERFLOW
    if _payload_hash_mismatch(
        gpu_payload_hashes=evidence.gpu_payload_hashes,
        cpu_oracle_payload_hashes=evidence.cpu_oracle_payload_hashes,
    ):
        return BR_H_PARITY_DRIFT
    if _all_payload_hashes_match(
        gpu_payload_hashes=evidence.gpu_payload_hashes,
        cpu_oracle_payload_hashes=evidence.cpu_oracle_payload_hashes,
    ):
        return BR_H_NATIVE_INTEGER_PARITY_CLEAN
    return BR_H_NOT_KERNELIZED


def _normalized_aten_op_name(func: Any) -> str:
    name_attr = getattr(func, "name", None)
    if callable(name_attr):
        try:
            return str(name_attr())
        except Exception:
            pass
    if isinstance(name_attr, str):
        return name_attr
    schema = getattr(func, "_schema", None)
    if schema is not None:
        schema_name = getattr(schema, "name", None)
        if schema_name:
            return str(schema_name)
    return str(func)


class CreditAxisKernelBoundaryGuard(TorchDispatchMode):
    """Trap-logic boundary guard: record forbidden FP ops during credit-axis hot path."""

    def __init__(self, *, fail_closed: bool = False) -> None:
        super().__init__()
        self.fail_closed = bool(fail_closed)
        self.hidden_fp_violation_count = 0
        self.violating_op_names: list[str] = []
        self.boundary_or_manifest_dtype_violation = False

    def _record_violation(self, op_name: str) -> None:
        self.hidden_fp_violation_count += 1
        self.violating_op_names.append(op_name)
        if self.fail_closed:
            raise CreditAxisKernelBoundaryViolation(
                f"credit-axis boundary guard rejected op {op_name}"
            )

    def _inspect_tensor(self, tensor: torch.Tensor) -> None:
        if tensor.dtype in _FORBIDDEN_FP_DTYPES:
            self.boundary_or_manifest_dtype_violation = True
            self._record_violation(f"dtype:{tensor.dtype}")

    def _inspect_nested(self, obj: Any) -> None:
        if isinstance(obj, torch.Tensor):
            self._inspect_tensor(obj)
        elif isinstance(obj, (tuple, list)):
            for item in obj:
                self._inspect_nested(item)
        elif isinstance(obj, dict):
            for item in obj.values():
                self._inspect_nested(item)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # type: ignore[no-untyped-def]
        op_name = _normalized_aten_op_name(func)
        if _is_forbidden_aten_op(op_name):
            self._record_violation(op_name)
        kwargs = {} if kwargs is None else kwargs
        for arg in args:
            self._inspect_nested(arg)
        for value in kwargs.values():
            self._inspect_nested(value)
        result = func(*args, **kwargs)
        self._inspect_nested(result)
        return result


def _payload_hash_for_live_tensor_key(
    key: str,
    value: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
) -> str:
    if key == "attribution_events_hash":
        if not isinstance(value, tuple) or len(value) != 2:
            raise ValueError("attribution_events_hash requires (flat_indices, attribution_q31)")
        return attribution_events_payload_sha256(value[0], value[1])
    if key == "sparse_vote_events_hash":
        if not isinstance(value, tuple) or len(value) != 2:
            raise ValueError("sparse_vote_events_hash requires (indices, values)")
        return sparse_vote_events_payload_sha256(value[0], value[1])
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"payload key {key} requires a tensor live carrier")
    return canonical_tensor_payload_sha256(value)


def _recompute_gpu_payload_hashes_from_live_tensors(
    live_gpu_tensors: Mapping[str, torch.Tensor | tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, str]:
    recomputed: dict[str, str] = {}
    for key in GPU_PAYLOAD_HASH_KEYS:
        if key not in live_gpu_tensors:
            raise ValueError(f"missing live GPU tensor for payload key {key}")
        recomputed[key] = _payload_hash_for_live_tensor_key(
            key, live_gpu_tensors[key]
        )
    return recomputed


def build_credit_axis_gpu_kernel_validation_receipt(
    *,
    cpu_integration_receipt: IntegerCreditAxisIntegrationReceipt,
    runtime_evidence: CreditAxisGpuKernelRuntimeEvidence,
    gpu_payload_hashes: Mapping[str, str] | None = None,
    live_gpu_tensors: Mapping[str, torch.Tensor | tuple[torch.Tensor, torch.Tensor]] | None = None,
    cpu_oracle_payload_hashes: Mapping[str, str] | None = None,
    rank_bin_spec_canonical: tuple[CanonicalRankVoteBin, ...] | None = None,
    credit_law_id: str = "credit_neg_attribution_q31_v1",
) -> CreditAxisGpuKernelValidationReceipt:
    validate_integer_credit_axis_integration_receipt(cpu_integration_receipt)
    if not _is_br_d_branch_id(cpu_integration_receipt.branch_id):
        raise ValueError("cpu integration receipt branch_id must be BR-D namespace")
    if gpu_payload_hashes is None and live_gpu_tensors is not None:
        gpu_payload_hashes = _recompute_gpu_payload_hashes_from_live_tensors(
            live_gpu_tensors
        )
    if cpu_oracle_payload_hashes is None:
        if rank_bin_spec_canonical is None:
            raise ValueError(
                "gate-grade receipt requires rank_bin_spec_canonical for 5-key oracle"
            )
        oracle_5 = build_cpu_oracle_payload_hashes_for_gpu_parity(
            integration_receipt=cpu_integration_receipt,
            credit_q31=cpu_integration_receipt.bound_credit_q31,
            projected_moves=cpu_integration_receipt.bound_projected_moves,
            projected_move_indices=cpu_integration_receipt.bound_projected_move_indices,
            rank_bin_spec_canonical=rank_bin_spec_canonical,
            credit_law_id=credit_law_id,
        )
        cpu_oracle_payload_hashes = cpu_oracle_payload_hashes_from_gpu_parity(oracle_5)
    torch_cuda_reference_only = _recompute_torch_cuda_reference_only(
        torch_cuda_reference_only=runtime_evidence.torch_cuda_reference_only,
        stage_native_evidence=runtime_evidence.stage_native_evidence,
    )
    evidence = CreditAxisGpuKernelRuntimeEvidence(
        liveness_fail=runtime_evidence.liveness_fail,
        hot_loop_kernel_invoked=runtime_evidence.hot_loop_kernel_invoked,
        torch_cuda_reference_only=torch_cuda_reference_only,
        hidden_fp_violation_count=runtime_evidence.hidden_fp_violation_count,
        boundary_or_manifest_dtype_violation=(
            runtime_evidence.boundary_or_manifest_dtype_violation
        ),
        overflow_guard_tripped=runtime_evidence.overflow_guard_tripped,
        gpu_output_repeat_stable=runtime_evidence.gpu_output_repeat_stable,
        gpu_payload_hashes=gpu_payload_hashes,
        cpu_oracle_payload_hashes=cpu_oracle_payload_hashes,
        device_residency_cuda=runtime_evidence.device_residency_cuda,
        hot_loop_integer_only=runtime_evidence.hot_loop_integer_only,
        stage_native_evidence=runtime_evidence.stage_native_evidence,
    )
    gpu_validation_branch_id = classify_credit_axis_gpu_runtime_branch(evidence)
    if gpu_validation_branch_id == BR_H_NATIVE_INTEGER_PARITY_CLEAN:
        raise ValueError(
            "BR-H-NATIVE-INTEGER-PARITY-CLEAN is unreachable from CPU scaffold builder"
        )
    if _is_br_d_branch_id(gpu_validation_branch_id):
        raise ValueError("gpu_validation_branch_id must be BR-H namespace")
    carried_hashes = dict(gpu_payload_hashes or {})
    parity_pass = _all_payload_hashes_match(
        gpu_payload_hashes=carried_hashes if carried_hashes else None,
        cpu_oracle_payload_hashes=cpu_oracle_payload_hashes,
    )
    receipt = CreditAxisGpuKernelValidationReceipt(
        schema_version=CREDIT_AXIS_GPU_KERNEL_VALIDATION_SCHEMA_VERSION,
        cpu_oracle_commit_sha=CPU_ORACLE_COMMIT_SHA,
        cpu_integration_receipt_digest=cpu_integration_receipt_digest_sha256(
            cpu_integration_receipt
        ),
        cpu_integration_data_digest=cpu_integration_receipt.integration_data_digest_sha256,
        cpu_integration_branch_id=cpu_integration_receipt.branch_id,
        gpu_validation_branch_id=gpu_validation_branch_id,
        gpu_payload_hashes=carried_hashes,
        gpu_output_repeat_stable=runtime_evidence.gpu_output_repeat_stable,
        overflow_guard_tripped=runtime_evidence.overflow_guard_tripped,
        parity_pass=parity_pass,
        authority_level=AUTHORITY_GPU_EVIDENCE_ONLY,
        hot_loop_kernel_invoked=runtime_evidence.hot_loop_kernel_invoked,
        torch_cuda_reference_only=torch_cuda_reference_only,
        hidden_fp_violation_count=runtime_evidence.hidden_fp_violation_count,
        device_residency_cuda=runtime_evidence.device_residency_cuda,
        hot_loop_integer_only=runtime_evidence.hot_loop_integer_only,
        fp_exception_caveat=OPTIMIZER_CREDIT_STATE_FP_EXCEPTION_CAVEAT,
        non_claims=CREDIT_AXIS_GPU_KERNEL_NON_CLAIMS,
        **credit_axis_gpu_kernel_hard_false_snapshot(),
    )
    validate_credit_axis_gpu_kernel_validation_receipt(
        receipt,
        cpu_integration_receipt=cpu_integration_receipt,
        live_gpu_tensors=live_gpu_tensors,
        rank_bin_spec_canonical=rank_bin_spec_canonical,
        credit_law_id=credit_law_id,
    )
    return receipt


def validate_credit_axis_gpu_kernel_validation_receipt_shape_only(
    receipt: CreditAxisGpuKernelValidationReceipt,
) -> None:
    """Non-authoritative structural checks only.

    Commit/runtime gates must NOT call this helper; use
    validate_credit_axis_gpu_kernel_validation_receipt instead.
    """
    if receipt.schema_version != CREDIT_AXIS_GPU_KERNEL_VALIDATION_SCHEMA_VERSION:
        raise ValueError("credit axis gpu kernel validation schema mismatch")
    if receipt.cpu_oracle_commit_sha != CPU_ORACLE_COMMIT_SHA:
        raise ValueError("cpu_oracle_commit_sha must bind to d4a846a")
    if receipt.authority_level != AUTHORITY_GPU_EVIDENCE_ONLY:
        raise ValueError("authority_level must be gpu_evidence_only")
    if receipt.fp_exception_caveat != OPTIMIZER_CREDIT_STATE_FP_EXCEPTION_CAVEAT:
        raise ValueError("gpu kernel receipt must keep exact FP-exception caveat")
    if receipt.non_claims != CREDIT_AXIS_GPU_KERNEL_NON_CLAIMS:
        raise ValueError("gpu kernel receipt non_claims must be exact")
    for field in FORBIDDEN_GPU_KERNEL_RECEIPT_FIELDS:
        if bool(getattr(receipt, field)):
            raise ValueError(f"{field} is forbidden on gpu kernel validation receipt")
    if not _is_br_d_branch_id(receipt.cpu_integration_branch_id):
        raise ValueError("cpu_integration_branch_id must be BR-D namespace")
    if not _is_br_h_branch_id(receipt.gpu_validation_branch_id):
        raise ValueError("gpu_validation_branch_id must be BR-H namespace")
    if _is_br_d_branch_id(receipt.gpu_validation_branch_id):
        raise ValueError("gpu_validation_branch_id must not carry BR-D value")
    if receipt.cpu_integration_branch_id == receipt.gpu_validation_branch_id:
        raise ValueError(
            "cpu_integration_branch_id and gpu_validation_branch_id must not be equal"
        )


def _validate_credit_axis_gpu_kernel_oracle_bind(
    receipt: CreditAxisGpuKernelValidationReceipt,
    cpu_integration_receipt: IntegerCreditAxisIntegrationReceipt,
) -> None:
    validate_integer_credit_axis_integration_receipt(cpu_integration_receipt)
    if (
        receipt.cpu_integration_data_digest
        != cpu_integration_receipt.integration_data_digest_sha256
    ):
        raise ValueError("cpu_integration_data_digest bind mismatch")
    if receipt.cpu_integration_branch_id != cpu_integration_receipt.branch_id:
        raise ValueError("cpu_integration_branch_id must equal oracle branch_id")
    expected_digest = cpu_integration_receipt_digest_sha256(cpu_integration_receipt)
    if receipt.cpu_integration_receipt_digest != expected_digest:
        raise ValueError("cpu_integration_receipt_digest bind mismatch")


def validate_credit_axis_gpu_kernel_validation_receipt(
    receipt: CreditAxisGpuKernelValidationReceipt,
    *,
    cpu_integration_receipt: IntegerCreditAxisIntegrationReceipt | None = None,
    live_gpu_tensors: Mapping[str, torch.Tensor | tuple[torch.Tensor, torch.Tensor]] | None = None,
    rank_bin_spec_canonical: tuple[CanonicalRankVoteBin, ...] | None = None,
    credit_law_id: str = "credit_neg_attribution_q31_v1",
    stage_native_evidence: CreditAxisStageNativeEvidence | None = None,
) -> None:
    if cpu_integration_receipt is None:
        raise ValueError("gate-grade validation requires cpu_integration_receipt bind")
    validate_credit_axis_gpu_kernel_validation_receipt_shape_only(receipt)
    _validate_credit_axis_gpu_kernel_oracle_bind(receipt, cpu_integration_receipt)
    if rank_bin_spec_canonical is None:
        raise ValueError("gate-grade validation requires rank_bin_spec_canonical for 5-key oracle")
    cpu_oracle_hashes = cpu_oracle_payload_hashes_from_gpu_parity(
        build_cpu_oracle_payload_hashes_for_gpu_parity(
            integration_receipt=cpu_integration_receipt,
            credit_q31=cpu_integration_receipt.bound_credit_q31,
            projected_moves=cpu_integration_receipt.bound_projected_moves,
            projected_move_indices=cpu_integration_receipt.bound_projected_move_indices,
            rank_bin_spec_canonical=rank_bin_spec_canonical,
            credit_law_id=credit_law_id,
        )
    )
    if live_gpu_tensors is not None:
        recomputed = _recompute_gpu_payload_hashes_from_live_tensors(live_gpu_tensors)
        for key, value in recomputed.items():
            if receipt.gpu_payload_hashes.get(key) != value:
                raise ValueError(f"gpu_payload_hashes[{key}] mismatch vs live tensor")
    recomputed_torch_ref = receipt.torch_cuda_reference_only
    if stage_native_evidence is not None:
        recomputed_torch_ref = torch_cuda_reference_only_from_stage_evidence(
            stage_native_evidence
        )
        if receipt.torch_cuda_reference_only != recomputed_torch_ref:
            raise ValueError(
                "torch_cuda_reference_only disagrees with stage_native_evidence recompute"
            )
    evidence = CreditAxisGpuKernelRuntimeEvidence(
        liveness_fail=False,
        hot_loop_kernel_invoked=receipt.hot_loop_kernel_invoked,
        torch_cuda_reference_only=recomputed_torch_ref,
        hidden_fp_violation_count=receipt.hidden_fp_violation_count,
        boundary_or_manifest_dtype_violation=False,
        overflow_guard_tripped=receipt.overflow_guard_tripped,
        gpu_output_repeat_stable=receipt.gpu_output_repeat_stable,
        gpu_payload_hashes=receipt.gpu_payload_hashes or None,
        cpu_oracle_payload_hashes=cpu_oracle_hashes,
        device_residency_cuda=receipt.device_residency_cuda,
        hot_loop_integer_only=receipt.hot_loop_integer_only,
        stage_native_evidence=stage_native_evidence,
    )
    recomputed_branch = classify_credit_axis_gpu_runtime_branch(evidence)
    if receipt.gpu_validation_branch_id != recomputed_branch:
        raise ValueError("gpu_validation_branch_id mismatch vs recomputed classifier")
    if receipt.gpu_validation_branch_id == BR_H_NATIVE_INTEGER_PARITY_CLEAN:
        if live_gpu_tensors is None:
            raise ValueError("CLEAN requires live_gpu_tensors for hash recompute")
        if not receipt.hot_loop_kernel_invoked:
            raise ValueError("CLEAN requires hot_loop_kernel_invoked=true")
        if receipt.torch_cuda_reference_only:
            raise ValueError("CLEAN requires torch_cuda_reference_only=false")
        if not receipt.gpu_output_repeat_stable:
            raise ValueError("CLEAN requires gpu_output_repeat_stable=true")
        if receipt.overflow_guard_tripped:
            raise ValueError("CLEAN requires overflow_guard_tripped=false")
        if receipt.hidden_fp_violation_count > 0:
            raise ValueError("CLEAN requires hidden_fp_violation_count=0")
        recomputed = _recompute_gpu_payload_hashes_from_live_tensors(live_gpu_tensors)
        for key in GPU_PAYLOAD_HASH_KEYS:
            if receipt.gpu_payload_hashes.get(key) != recomputed.get(key):
                raise ValueError(
                    f"CLEAN requires gpu_payload_hashes[{key}] match live recompute"
                )
        if not _all_payload_hashes_match(
            gpu_payload_hashes=recomputed,
            cpu_oracle_payload_hashes=cpu_oracle_hashes,
        ):
            raise ValueError("CLEAN requires every live hash == cpu oracle digest")
    if receipt.parity_pass:
        if not _all_payload_hashes_match(
            gpu_payload_hashes=receipt.gpu_payload_hashes or None,
            cpu_oracle_payload_hashes=cpu_oracle_hashes,
        ):
            raise ValueError("parity_pass mismatch vs cpu oracle hashes")
