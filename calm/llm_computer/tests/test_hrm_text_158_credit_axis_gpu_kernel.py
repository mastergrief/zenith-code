"""CPU scaffold tests for BR-3C-H.1a credit-axis GPU kernel receipt (no GPU run)."""
from __future__ import annotations

from dataclasses import replace
import importlib

import pytest
import torch
import torch.nn.functional as F

from calm.hrm_text_158.bit_linear import BitLinear
from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    authoritative_forward_context,
    default_dry_run_rank_vote_spec,
    make_bounded_tensor_state,
)
from calm.hrm_text_158.native_full_stack.integer_credit_axis_gpu_kernel import (
    CREDIT_AXIS_KERNEL_SEAM_NAME,
    CreditAxisKernelNotAvailable,
    credit_axis_kernelized_sparse_pipeline_cuda,
)
from calm.hrm_text_158.native_full_stack.integer_credit_axis_gpu_receipt import (
    AUTHORITY_GPU_EVIDENCE_ONLY,
    BR_H_GPU_DISPATCH_HELD,
    BR_H_GPU_INTEGER_NONDETERMINISM_OR_OVERFLOW,
    BR_H_GPU_KERNEL_MISSING,
    BR_H_NATIVE_INTEGER_PARITY_CLEAN,
    BR_H_NOT_KERNELIZED,
    BR_H_PARITY_DRIFT,
    BRANCH_D_INTEGER_VIABLE,
    CreditAxisGpuKernelRuntimeEvidence,
    CreditAxisGpuKernelValidationReceipt,
    CreditAxisKernelBoundaryGuard,
    CreditAxisKernelBoundaryViolation,
    CREDIT_AXIS_GPU_KERNEL_NON_CLAIMS,
    CREDIT_AXIS_GPU_KERNEL_VALIDATION_SCHEMA_VERSION,
    FORBIDDEN_GPU_KERNEL_RECEIPT_FIELDS,
    RUN_GPU_CREDIT_AXIS_KERNEL_ENV,
    cpu_integration_receipt_digest_sha256,
    build_credit_axis_gpu_kernel_validation_receipt,
    classify_credit_axis_gpu_prelaunch_branch,
    classify_credit_axis_gpu_runtime_branch,
    credit_axis_gpu_kernel_hard_false_snapshot,
    cpu_oracle_payload_hashes_from_integration_receipt,
    run_gpu_credit_axis_kernel_env_enabled,
    validate_credit_axis_gpu_kernel_validation_receipt,
    validate_credit_axis_gpu_kernel_validation_receipt_shape_only,
)
from calm.hrm_text_158.native_full_stack.integer_native_optimizer_credit_path_design import (
    prove_integer_credit_axis_integration,
)
from calm.hrm_text_158.native_full_stack.integer_sparse_rank_votes import (
    canonical_rank_vote_spec,
)


class _Tiny(torch.nn.Module):
    def __init__(self, in_features: int = 3, out_features: int = 2) -> None:
        super().__init__()
        self.proj = BitLinear(in_features, out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def _rank_bins():
    return canonical_rank_vote_spec(default_dry_run_rank_vote_spec())


def _green_integration_receipt():
    torch.manual_seed(158)
    model = _Tiny()
    with torch.no_grad():
        model.proj.weight.zero_()
    eligible = {"proj": model.proj}
    q = torch.zeros_like(model.proj.weight.detach(), dtype=torch.int8)
    state = make_bounded_tensor_state(
        "proj",
        q,
        torch.tensor(1.0, dtype=torch.float32),
        hot_exact_indices=tuple(range(int(q.numel()))),
    )
    x = torch.tensor([[1.0, -2.0, 3.0]], dtype=torch.float32)
    target = torch.tensor([[2.0, -1.0]], dtype=torch.float32)
    model.zero_grad(set_to_none=True)
    with authoritative_forward_context(
        eligible,
        {"proj": state},
        device="cpu",
        requires_grad=True,
    ) as handle:
        out = model(x)
        loss = F.mse_loss(out, target)
        loss.backward()
    capture = handle.captures["proj"]
    weight_shape = tuple(int(dim) for dim in state.q_levels.shape)
    q_levels_flat = state.q_levels.reshape(-1)
    return prove_integer_credit_axis_integration(
        capture["inputs"],
        capture["grad_outputs"],
        weight_shape=weight_shape,
        q_levels_flat=q_levels_flat,
        rank_spec=default_dry_run_rank_vote_spec(),
        comparable_set_id="br3c_h1a_scaffold",
        reference_oracle_run_id="oracle_br3c_h1a",
        candidate_run_id="candidate_br3c_h1a",
    )


def test_import_scaffold_modules() -> None:
    receipt_mod = importlib.import_module(
        "calm.hrm_text_158.native_full_stack.integer_credit_axis_gpu_receipt"
    )
    kernel_mod = importlib.import_module(
        "calm.hrm_text_158.native_full_stack.integer_credit_axis_gpu_kernel"
    )
    assert receipt_mod.CREDIT_AXIS_GPU_KERNEL_VALIDATION_SCHEMA_VERSION
    assert kernel_mod.CREDIT_AXIS_KERNEL_SEAM_NAME == CREDIT_AXIS_KERNEL_SEAM_NAME


def test_runtime_classifier_partition_orders_4_5_6() -> None:
    cpu_hashes = {
        "attribution_events_hash": "a" * 64,
        "projected_move_indices_hash": "b" * 64,
        "projected_moves_hash": "c" * 64,
        "credit_q31_hash": "d" * 64,
        "sparse_vote_events_hash": "e" * 64,
    }
    stable_gpu = dict(cpu_hashes)
    drift_gpu = dict(cpu_hashes)
    drift_gpu["credit_q31_hash"] = "f" * 64
    unstable_gpu_a = dict(cpu_hashes)
    unstable_gpu_b = dict(cpu_hashes)
    unstable_gpu_b["credit_q31_hash"] = "9" * 64

    base = dict(
        hot_loop_kernel_invoked=True,
        torch_cuda_reference_only=False,
        hidden_fp_violation_count=0,
        boundary_or_manifest_dtype_violation=False,
        cpu_oracle_payload_hashes=cpu_hashes,
    )
    assert (
        classify_credit_axis_gpu_runtime_branch(
            CreditAxisGpuKernelRuntimeEvidence(
                **base,
                overflow_guard_tripped=True,
                gpu_output_repeat_stable=True,
                gpu_payload_hashes=stable_gpu,
            )
        )
        == BR_H_GPU_INTEGER_NONDETERMINISM_OR_OVERFLOW
    )
    assert (
        classify_credit_axis_gpu_runtime_branch(
            CreditAxisGpuKernelRuntimeEvidence(
                **base,
                overflow_guard_tripped=False,
                gpu_output_repeat_stable=False,
                gpu_payload_hashes=unstable_gpu_a,
            )
        )
        == BR_H_GPU_INTEGER_NONDETERMINISM_OR_OVERFLOW
    )
    assert (
        classify_credit_axis_gpu_runtime_branch(
            CreditAxisGpuKernelRuntimeEvidence(
                **base,
                overflow_guard_tripped=False,
                gpu_output_repeat_stable=True,
                gpu_payload_hashes=drift_gpu,
            )
        )
        == BR_H_PARITY_DRIFT
    )
    assert (
        classify_credit_axis_gpu_runtime_branch(
            CreditAxisGpuKernelRuntimeEvidence(
                **base,
                overflow_guard_tripped=False,
                gpu_output_repeat_stable=True,
                gpu_payload_hashes=stable_gpu,
            )
        )
        == BR_H_NATIVE_INTEGER_PARITY_CLEAN
    )
    del unstable_gpu_a, unstable_gpu_b


def test_branch_namespace_cpu_d_and_gpu_h_stored_separately() -> None:
    integration = _green_integration_receipt()
    assert integration.branch_id == BRANCH_D_INTEGER_VIABLE
    receipt = build_credit_axis_gpu_kernel_validation_receipt(
        cpu_integration_receipt=integration,
        runtime_evidence=CreditAxisGpuKernelRuntimeEvidence(
            hot_loop_kernel_invoked=False,
            torch_cuda_reference_only=True,
        ),
    )
    assert receipt.cpu_integration_branch_id == BRANCH_D_INTEGER_VIABLE
    assert receipt.gpu_validation_branch_id == BR_H_NOT_KERNELIZED
    assert receipt.cpu_integration_branch_id != receipt.gpu_validation_branch_id
    validate_credit_axis_gpu_kernel_validation_receipt(
        receipt,
        cpu_integration_receipt=integration,
    )


def test_clean_unreachable_on_cpu_scaffold_builder() -> None:
    integration = _green_integration_receipt()
    cpu_hashes = dict(cpu_oracle_payload_hashes_from_integration_receipt(integration))
    gpu_hashes = dict(cpu_hashes)
    gpu_hashes["sparse_vote_events_hash"] = "e" * 64
    receipt = build_credit_axis_gpu_kernel_validation_receipt(
        cpu_integration_receipt=integration,
        runtime_evidence=CreditAxisGpuKernelRuntimeEvidence(
            hot_loop_kernel_invoked=True,
            torch_cuda_reference_only=False,
            gpu_output_repeat_stable=True,
            overflow_guard_tripped=False,
        ),
        gpu_payload_hashes=gpu_hashes,
    )
    assert receipt.gpu_validation_branch_id == BR_H_PARITY_DRIFT
    assert receipt.gpu_validation_branch_id != BR_H_NATIVE_INTEGER_PARITY_CLEAN


def test_forbidden_flags_rejected() -> None:
    integration = _green_integration_receipt()
    receipt = build_credit_axis_gpu_kernel_validation_receipt(
        cpu_integration_receipt=integration,
        runtime_evidence=CreditAxisGpuKernelRuntimeEvidence(),
    )
    for field in FORBIDDEN_GPU_KERNEL_RECEIPT_FIELDS:
        tampered = replace(receipt, **{field: True})
        with pytest.raises(ValueError, match=field):
            validate_credit_axis_gpu_kernel_validation_receipt(
                tampered,
                cpu_integration_receipt=integration,
            )


def test_tampered_cpu_integration_digest_rejected() -> None:
    integration = _green_integration_receipt()
    receipt = build_credit_axis_gpu_kernel_validation_receipt(
        cpu_integration_receipt=integration,
        runtime_evidence=CreditAxisGpuKernelRuntimeEvidence(),
    )
    tampered = replace(
        receipt,
        cpu_integration_data_digest="deadbeef" * 8,
    )
    with pytest.raises(ValueError, match="cpu_integration_data_digest"):
        validate_credit_axis_gpu_kernel_validation_receipt(
            tampered,
            cpu_integration_receipt=integration,
        )


def test_prelaunch_missing_kernel_when_triton_unavailable() -> None:
    assert (
        classify_credit_axis_gpu_prelaunch_branch(
            triton_available=False,
            cuda_available=True,
            kernel_module_built=False,
            seam_resolves_to_credit_axis_kernel=True,
            dispatch_env_enabled=True,
        )
        == BR_H_GPU_KERNEL_MISSING
    )


def test_prelaunch_dispatch_held_when_env_unset() -> None:
    assert (
        classify_credit_axis_gpu_prelaunch_branch(
            triton_available=True,
            cuda_available=True,
            kernel_module_built=True,
            seam_resolves_to_credit_axis_kernel=True,
            dispatch_env_enabled=False,
        )
        == BR_H_GPU_DISPATCH_HELD
    )


def test_missing_kernel_branch_no_silent_fallback(monkeypatch) -> None:
    monkeypatch.delenv(RUN_GPU_CREDIT_AXIS_KERNEL_ENV, raising=False)
    assert run_gpu_credit_axis_kernel_env_enabled() is False
    with pytest.raises(CreditAxisKernelNotAvailable, match=BR_H_GPU_DISPATCH_HELD):
        credit_axis_kernelized_sparse_pipeline_cuda(
            capture_inputs=(torch.zeros(1, 3),),
            capture_grad_outputs=(torch.zeros(1, 2),),
            weight_shape=(2, 3),
            q_levels_flat=torch.zeros(6, dtype=torch.int8),
            rank_bin_spec_canonical=_rank_bins(),
            credit_law_id="neg_attribution_q31_v0",
        )

    monkeypatch.setenv(RUN_GPU_CREDIT_AXIS_KERNEL_ENV, "1")
    with pytest.raises(CreditAxisKernelNotAvailable, match=BR_H_GPU_KERNEL_MISSING):
        credit_axis_kernelized_sparse_pipeline_cuda(
            capture_inputs=(torch.zeros(1, 3),),
            capture_grad_outputs=(torch.zeros(1, 2),),
            weight_shape=(2, 3),
            q_levels_flat=torch.zeros(6, dtype=torch.int8),
            rank_bin_spec_canonical=_rank_bins(),
            credit_law_id="neg_attribution_q31_v0",
        )


def test_seam_is_credit_axis_not_vote_preplan() -> None:
    vote_update = importlib.import_module(
        "calm.hrm_text_158.native_full_stack.vote_update"
    )
    assert hasattr(vote_update, "vote_update_preplan_triton")
    assert CREDIT_AXIS_KERNEL_SEAM_NAME != "vote_update_preplan_triton"
    assert CREDIT_AXIS_KERNEL_SEAM_NAME == "credit_axis_kernelized_sparse_pipeline_cuda"


def test_boundary_guard_trap_logic_records_violations() -> None:
    guard = CreditAxisKernelBoundaryGuard(fail_closed=False)
    with guard:
        _ = torch.tensor([1.0, 2.0], dtype=torch.float32)
    assert guard.hidden_fp_violation_count >= 1
    assert guard.boundary_or_manifest_dtype_violation is True


def test_boundary_guard_fail_closed_raises() -> None:
    guard = CreditAxisKernelBoundaryGuard(fail_closed=True)
    with pytest.raises(CreditAxisKernelBoundaryViolation):
        with guard:
            _ = torch.tensor([1.0], dtype=torch.float32)


def test_synthetic_classifier_parity_drift_without_minting_clean_receipt() -> None:
    integration = _green_integration_receipt()
    cpu_hashes = cpu_oracle_payload_hashes_from_integration_receipt(integration)
    gpu_hashes = dict(cpu_hashes)
    gpu_hashes["projected_moves_hash"] = "1" * 64
    branch = classify_credit_axis_gpu_runtime_branch(
        CreditAxisGpuKernelRuntimeEvidence(
            hot_loop_kernel_invoked=True,
            torch_cuda_reference_only=False,
            gpu_output_repeat_stable=True,
            overflow_guard_tripped=False,
            gpu_payload_hashes=gpu_hashes,
            cpu_oracle_payload_hashes=cpu_hashes,
        )
    )
    assert branch == BR_H_PARITY_DRIFT
    receipt = build_credit_axis_gpu_kernel_validation_receipt(
        cpu_integration_receipt=integration,
        runtime_evidence=CreditAxisGpuKernelRuntimeEvidence(
            hot_loop_kernel_invoked=True,
            torch_cuda_reference_only=False,
            gpu_output_repeat_stable=True,
            overflow_guard_tripped=False,
        ),
        gpu_payload_hashes=gpu_hashes,
    )
    assert receipt.gpu_validation_branch_id == BR_H_PARITY_DRIFT
    assert receipt.authority_level == AUTHORITY_GPU_EVIDENCE_ONLY
    assert receipt.non_claims == CREDIT_AXIS_GPU_KERNEL_NON_CLAIMS


def test_hard_false_snapshot_defaults() -> None:
    snapshot = credit_axis_gpu_kernel_hard_false_snapshot()
    assert set(snapshot) == set(FORBIDDEN_GPU_KERNEL_RECEIPT_FIELDS)
    assert all(value is False for value in snapshot.values())


def test_non_clean_receipt_rejected_without_cpu_oracle_bind() -> None:
    integration = _green_integration_receipt()
    receipt = build_credit_axis_gpu_kernel_validation_receipt(
        cpu_integration_receipt=integration,
        runtime_evidence=CreditAxisGpuKernelRuntimeEvidence(
            hot_loop_kernel_invoked=False,
            torch_cuda_reference_only=True,
        ),
    )
    assert receipt.gpu_validation_branch_id == BR_H_NOT_KERNELIZED
    with pytest.raises(ValueError, match="cpu_integration_receipt bind"):
        validate_credit_axis_gpu_kernel_validation_receipt(receipt)


def test_clean_rejected_without_live_gpu_tensors(monkeypatch) -> None:
    integration = _green_integration_receipt()
    sparse_hash = "e" * 64
    full_cpu_hashes = {
        **cpu_oracle_payload_hashes_from_integration_receipt(integration),
        "sparse_vote_events_hash": sparse_hash,
    }
    monkeypatch.setattr(
        "calm.hrm_text_158.native_full_stack.integer_credit_axis_gpu_receipt.cpu_oracle_payload_hashes_from_integration_receipt",
        lambda _receipt: dict(full_cpu_hashes),
    )
    forged = CreditAxisGpuKernelValidationReceipt(
        schema_version=CREDIT_AXIS_GPU_KERNEL_VALIDATION_SCHEMA_VERSION,
        cpu_oracle_commit_sha="d4a846a41cdf164e0718c711990518a305938650",
        cpu_integration_receipt_digest=cpu_integration_receipt_digest_sha256(integration),
        cpu_integration_data_digest=integration.integration_data_digest_sha256,
        cpu_integration_branch_id=integration.branch_id,
        gpu_validation_branch_id=BR_H_NATIVE_INTEGER_PARITY_CLEAN,
        gpu_payload_hashes=dict(full_cpu_hashes),
        gpu_output_repeat_stable=True,
        overflow_guard_tripped=False,
        parity_pass=True,
        authority_level=AUTHORITY_GPU_EVIDENCE_ONLY,
        hot_loop_kernel_invoked=True,
        torch_cuda_reference_only=False,
        hidden_fp_violation_count=0,
        device_residency_cuda=True,
        hot_loop_integer_only=True,
        fp_exception_caveat=integration.fp_exception_caveat,
        non_claims=CREDIT_AXIS_GPU_KERNEL_NON_CLAIMS,
        **credit_axis_gpu_kernel_hard_false_snapshot(),
    )
    with pytest.raises(ValueError, match="live_gpu_tensors"):
        validate_credit_axis_gpu_kernel_validation_receipt(
            forged,
            cpu_integration_receipt=integration,
        )


def test_clean_rejected_when_live_recompute_mismatches_carried(monkeypatch) -> None:
    integration = _green_integration_receipt()
    sparse_hash = "e" * 64
    full_cpu_hashes = {
        **cpu_oracle_payload_hashes_from_integration_receipt(integration),
        "sparse_vote_events_hash": sparse_hash,
    }
    monkeypatch.setattr(
        "calm.hrm_text_158.native_full_stack.integer_credit_axis_gpu_receipt.cpu_oracle_payload_hashes_from_integration_receipt",
        lambda _receipt: dict(full_cpu_hashes),
    )
    live_tensors = {
        "attribution_events_hash": integration.bound_candidate_attribution_events.flat_indices,
        "projected_move_indices_hash": integration.bound_projected_move_indices,
        "projected_moves_hash": integration.bound_projected_moves,
        "credit_q31_hash": integration.bound_credit_q31,
        "sparse_vote_events_hash": torch.tensor([1, 2, 3], dtype=torch.int16),
    }
    forged = CreditAxisGpuKernelValidationReceipt(
        schema_version=CREDIT_AXIS_GPU_KERNEL_VALIDATION_SCHEMA_VERSION,
        cpu_oracle_commit_sha="d4a846a41cdf164e0718c711990518a305938650",
        cpu_integration_receipt_digest=cpu_integration_receipt_digest_sha256(integration),
        cpu_integration_data_digest=integration.integration_data_digest_sha256,
        cpu_integration_branch_id=integration.branch_id,
        gpu_validation_branch_id=BR_H_NATIVE_INTEGER_PARITY_CLEAN,
        gpu_payload_hashes=dict(full_cpu_hashes),
        gpu_output_repeat_stable=True,
        overflow_guard_tripped=False,
        parity_pass=True,
        authority_level=AUTHORITY_GPU_EVIDENCE_ONLY,
        hot_loop_kernel_invoked=True,
        torch_cuda_reference_only=False,
        hidden_fp_violation_count=0,
        device_residency_cuda=True,
        hot_loop_integer_only=True,
        fp_exception_caveat=integration.fp_exception_caveat,
        non_claims=CREDIT_AXIS_GPU_KERNEL_NON_CLAIMS,
        **credit_axis_gpu_kernel_hard_false_snapshot(),
    )
    with pytest.raises(ValueError, match="mismatch vs live tensor"):
        validate_credit_axis_gpu_kernel_validation_receipt(
            forged,
            cpu_integration_receipt=integration,
            live_gpu_tensors=live_tensors,
        )


def test_shape_only_validator_skips_oracle_bind() -> None:
    integration = _green_integration_receipt()
    receipt = build_credit_axis_gpu_kernel_validation_receipt(
        cpu_integration_receipt=integration,
        runtime_evidence=CreditAxisGpuKernelRuntimeEvidence(),
    )
    tampered = replace(receipt, cpu_integration_data_digest="deadbeef" * 8)
    validate_credit_axis_gpu_kernel_validation_receipt_shape_only(tampered)


def test_boundary_guard_traps_int_to_float32_to_op() -> None:
    x_int = torch.tensor([1, 2, 3], dtype=torch.int32)
    guard = CreditAxisKernelBoundaryGuard(fail_closed=True)
    with pytest.raises(CreditAxisKernelBoundaryViolation, match="aten::"):
        with guard:
            x_int.to(torch.float32)


def test_boundary_guard_traps_log2_on_int_input() -> None:
    x_int = torch.tensor([4, 8, 16], dtype=torch.int32)
    guard = CreditAxisKernelBoundaryGuard(fail_closed=True)
    with pytest.raises(CreditAxisKernelBoundaryViolation, match="aten::log2"):
        with guard:
            torch.log2(x_int)


def test_boundary_guard_traps_pow_on_int_input() -> None:
    base = torch.tensor([2, 3], dtype=torch.int32)
    exponent = torch.tensor([3, 2], dtype=torch.int32)
    guard = CreditAxisKernelBoundaryGuard(fail_closed=True)
    with pytest.raises(CreditAxisKernelBoundaryViolation, match="aten::pow"):
        with guard:
            torch.pow(base, exponent)


def test_manual_clean_receipt_rejected_without_live_gpu_bind() -> None:
    integration = _green_integration_receipt()
    cpu_hashes = cpu_oracle_payload_hashes_from_integration_receipt(integration)
    forged = CreditAxisGpuKernelValidationReceipt(
        schema_version=CREDIT_AXIS_GPU_KERNEL_VALIDATION_SCHEMA_VERSION,
        cpu_oracle_commit_sha="d4a846a41cdf164e0718c711990518a305938650",
        cpu_integration_receipt_digest=cpu_integration_receipt_digest_sha256(integration),
        cpu_integration_data_digest=integration.integration_data_digest_sha256,
        cpu_integration_branch_id=integration.branch_id,
        gpu_validation_branch_id=BR_H_NATIVE_INTEGER_PARITY_CLEAN,
        gpu_payload_hashes=dict(cpu_hashes),
        gpu_output_repeat_stable=True,
        overflow_guard_tripped=False,
        parity_pass=True,
        authority_level=AUTHORITY_GPU_EVIDENCE_ONLY,
        hot_loop_kernel_invoked=False,
        torch_cuda_reference_only=True,
        hidden_fp_violation_count=0,
        device_residency_cuda=False,
        hot_loop_integer_only=False,
        fp_exception_caveat=integration.fp_exception_caveat,
        non_claims=CREDIT_AXIS_GPU_KERNEL_NON_CLAIMS,
        **credit_axis_gpu_kernel_hard_false_snapshot(),
    )
    with pytest.raises(ValueError):
        validate_credit_axis_gpu_kernel_validation_receipt(
            forged,
            cpu_integration_receipt=integration,
        )
