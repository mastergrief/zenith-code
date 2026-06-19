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
    CREDIT_AXIS_KERNEL_MANIFEST_DIR,
    CREDIT_AXIS_KERNEL_SEAM_NAME,
    CreditAxisKernelNotAvailable,
    check_s1_shape_bounds_or_raise,
    credit_axis_kernel_module_built,
    credit_axis_kernelized_sparse_pipeline_cuda,
    default_pipeline_source_forbid_check,
    dense_compact_prefix_scan_reference,
    s1_row_major_compact_reference,
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
    CreditAxisStageNativeEvidence,
    CREDIT_AXIS_GPU_KERNEL_NON_CLAIMS,
    CREDIT_AXIS_GPU_KERNEL_VALIDATION_SCHEMA_VERSION,
    FORBIDDEN_GPU_KERNEL_RECEIPT_FIELDS,
    RUN_GPU_CREDIT_AXIS_KERNEL_ENV,
    build_cpu_oracle_payload_hashes_for_gpu_parity,
    cpu_integration_receipt_digest_sha256,
    build_credit_axis_gpu_kernel_validation_receipt,
    classify_credit_axis_gpu_prelaunch_branch,
    classify_credit_axis_gpu_runtime_branch,
    credit_axis_gpu_kernel_hard_false_snapshot,
    cpu_oracle_payload_hashes_from_integration_receipt,
    cpu_oracle_payload_hashes_from_gpu_parity,
    run_gpu_credit_axis_kernel_env_enabled,
    torch_cuda_reference_only_from_stage_evidence,
    validate_credit_axis_gpu_kernel_validation_receipt,
    validate_credit_axis_gpu_kernel_validation_receipt_shape_only,
)
from calm.hrm_text_158.native_full_stack.integer_native_optimizer_credit_path_design import (
    prove_integer_credit_axis_integration,
    validate_integer_credit_axis_integration_receipt,
)
from calm.hrm_text_158.native_full_stack.integer_marginal_attribution import (
    IntegerMarginalAttributionEvents,
    projected_moves_from_integer_attribution,
    streaming_sparse_attribution_from_captures,
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


def _receipt_kwargs(integration):
    return {
        "cpu_integration_receipt": integration,
        "rank_bin_spec_canonical": _rank_bins(),
    }


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
        **_receipt_kwargs(integration),
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
        rank_bin_spec_canonical=_rank_bins(),
    )


def test_clean_unreachable_on_cpu_scaffold_builder() -> None:
    integration = _green_integration_receipt()
    oracle_5 = build_cpu_oracle_payload_hashes_for_gpu_parity(
        integration_receipt=integration,
        credit_q31=integration.bound_credit_q31,
        projected_moves=integration.bound_projected_moves,
        projected_move_indices=integration.bound_projected_move_indices,
        rank_bin_spec_canonical=_rank_bins(),
    )
    gpu_hashes = dict(cpu_oracle_payload_hashes_from_gpu_parity(oracle_5))
    gpu_hashes["sparse_vote_events_hash"] = "e" * 64
    receipt = build_credit_axis_gpu_kernel_validation_receipt(
        **_receipt_kwargs(integration),
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
        **_receipt_kwargs(integration),
        runtime_evidence=CreditAxisGpuKernelRuntimeEvidence(),
    )
    for field in FORBIDDEN_GPU_KERNEL_RECEIPT_FIELDS:
        tampered = replace(receipt, **{field: True})
        with pytest.raises(ValueError, match=field):
            validate_credit_axis_gpu_kernel_validation_receipt(
                tampered,
                cpu_integration_receipt=integration,
                rank_bin_spec_canonical=_rank_bins(),
            )


def test_tampered_cpu_integration_digest_rejected() -> None:
    integration = _green_integration_receipt()
    receipt = build_credit_axis_gpu_kernel_validation_receipt(
        **_receipt_kwargs(integration),
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
            rank_bin_spec_canonical=_rank_bins(),
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
    monkeypatch.setattr(
        "calm.hrm_text_158.native_full_stack.integer_credit_axis_gpu_kernel.credit_axis_kernel_module_built",
        lambda: False,
    )
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
    oracle_5 = build_cpu_oracle_payload_hashes_for_gpu_parity(
        integration_receipt=integration,
        credit_q31=integration.bound_credit_q31,
        projected_moves=integration.bound_projected_moves,
        projected_move_indices=integration.bound_projected_move_indices,
        rank_bin_spec_canonical=_rank_bins(),
    )
    gpu_hashes = dict(cpu_oracle_payload_hashes_from_gpu_parity(oracle_5))
    gpu_hashes["projected_moves_hash"] = "1" * 64
    branch = classify_credit_axis_gpu_runtime_branch(
        CreditAxisGpuKernelRuntimeEvidence(
            hot_loop_kernel_invoked=True,
            torch_cuda_reference_only=False,
            gpu_output_repeat_stable=True,
            overflow_guard_tripped=False,
            gpu_payload_hashes=gpu_hashes,
            cpu_oracle_payload_hashes=cpu_oracle_payload_hashes_from_gpu_parity(oracle_5),
        )
    )
    assert branch == BR_H_PARITY_DRIFT
    receipt = build_credit_axis_gpu_kernel_validation_receipt(
        **_receipt_kwargs(integration),
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
        **_receipt_kwargs(integration),
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
    oracle_5 = build_cpu_oracle_payload_hashes_for_gpu_parity(
        integration_receipt=integration,
        credit_q31=integration.bound_credit_q31,
        projected_moves=integration.bound_projected_moves,
        projected_move_indices=integration.bound_projected_move_indices,
        rank_bin_spec_canonical=_rank_bins(),
    )
    full_cpu_hashes = cpu_oracle_payload_hashes_from_gpu_parity(oracle_5)
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
            rank_bin_spec_canonical=_rank_bins(),
        )


def test_clean_rejected_when_live_recompute_mismatches_carried(monkeypatch) -> None:
    integration = _green_integration_receipt()
    oracle_5 = build_cpu_oracle_payload_hashes_for_gpu_parity(
        integration_receipt=integration,
        credit_q31=integration.bound_credit_q31,
        projected_moves=integration.bound_projected_moves,
        projected_move_indices=integration.bound_projected_move_indices,
        rank_bin_spec_canonical=_rank_bins(),
    )
    full_cpu_hashes = cpu_oracle_payload_hashes_from_gpu_parity(oracle_5)
    live_tensors = {
        "attribution_events_hash": (
            integration.bound_candidate_attribution_events.flat_indices,
            integration.bound_candidate_attribution_events.attribution_q31,
        ),
        "projected_move_indices_hash": integration.bound_projected_move_indices,
        "projected_moves_hash": integration.bound_projected_moves,
        "credit_q31_hash": integration.bound_credit_q31,
        "sparse_vote_events_hash": (
            integration.bound_projected_move_indices,
            torch.tensor([1, 2, 3], dtype=torch.int16),
        ),
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
            rank_bin_spec_canonical=_rank_bins(),
        )


def test_shape_only_validator_skips_oracle_bind() -> None:
    integration = _green_integration_receipt()
    receipt = build_credit_axis_gpu_kernel_validation_receipt(
        **_receipt_kwargs(integration),
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
            rank_bin_spec_canonical=_rank_bins(),
        )


def test_five_key_oracle_bind_passes_when_sparse_matches_strict_cpu_path() -> None:
    integration = _green_integration_receipt()
    oracle_5 = build_cpu_oracle_payload_hashes_for_gpu_parity(
        integration_receipt=integration,
        credit_q31=integration.bound_credit_q31,
        projected_moves=integration.bound_projected_moves,
        projected_move_indices=integration.bound_projected_move_indices,
        rank_bin_spec_canonical=_rank_bins(),
    )
    assert oracle_5.projected_moves_hash == integration.projected_moves_hash
    assert oracle_5.sparse_oracle_source.startswith("strict_integer_sparse")


def test_five_key_oracle_rejects_forged_sparse_hash() -> None:
    integration = _green_integration_receipt()
    oracle_5 = build_cpu_oracle_payload_hashes_for_gpu_parity(
        integration_receipt=integration,
        credit_q31=integration.bound_credit_q31,
        projected_moves=integration.bound_projected_moves,
        projected_move_indices=integration.bound_projected_move_indices,
        rank_bin_spec_canonical=_rank_bins(),
    )
    gpu_hashes = dict(cpu_oracle_payload_hashes_from_gpu_parity(oracle_5))
    gpu_hashes["sparse_vote_events_hash"] = "f" * 64
    branch = classify_credit_axis_gpu_runtime_branch(
        CreditAxisGpuKernelRuntimeEvidence(
            hot_loop_kernel_invoked=True,
            torch_cuda_reference_only=False,
            gpu_output_repeat_stable=True,
            overflow_guard_tripped=False,
            gpu_payload_hashes=gpu_hashes,
            cpu_oracle_payload_hashes=cpu_oracle_payload_hashes_from_gpu_parity(oracle_5),
        )
    )
    assert branch == BR_H_PARITY_DRIFT


def test_five_key_oracle_rejects_missing_sparse_key() -> None:
    integration = _green_integration_receipt()
    four_only = cpu_oracle_payload_hashes_from_integration_receipt(integration)
    branch = classify_credit_axis_gpu_runtime_branch(
        CreditAxisGpuKernelRuntimeEvidence(
            hot_loop_kernel_invoked=True,
            torch_cuda_reference_only=False,
            gpu_output_repeat_stable=True,
            overflow_guard_tripped=False,
            gpu_payload_hashes=four_only,
            cpu_oracle_payload_hashes={
                **four_only,
                "sparse_vote_events_hash": "a" * 64,
            },
        )
    )
    assert branch == BR_H_PARITY_DRIFT


def test_integration_four_keys_still_bound_to_d4a846a() -> None:
    integration = _green_integration_receipt()
    oracle_5 = build_cpu_oracle_payload_hashes_for_gpu_parity(
        integration_receipt=integration,
        credit_q31=integration.bound_credit_q31,
        projected_moves=integration.bound_projected_moves,
        projected_move_indices=integration.bound_projected_move_indices,
        rank_bin_spec_canonical=_rank_bins(),
    )
    with pytest.raises(ValueError):
        validate_integer_credit_axis_integration_receipt(
            replace(integration, attribution_events_hash="0" * 64)
        )
    assert oracle_5.attribution_events_hash == integration.attribution_events_hash


def _synthetic_captures(*, out_features: int, in_features: int, batch: int = 2, seq: int = 1):
    torch.manual_seed(1580)
    inputs = (torch.randn(batch, seq, in_features),)
    grad_outputs = (torch.randn(batch, seq, out_features),)
    return inputs, grad_outputs, (out_features, in_features)


def test_default_seam_source_forbids_cpu_reference_surfaces() -> None:
    violations = default_pipeline_source_forbid_check()
    assert violations == []


def test_s1_compact_prefix_scan_contract_row_major_order() -> None:
    inputs, grad_outputs, shape = _synthetic_captures(out_features=2, in_features=5)
    cpu_events, _ = streaming_sparse_attribution_from_captures(
        inputs,
        grad_outputs,
        weight_shape=shape,
    )
    row_attrs = []
    out_features, in_features = shape
    for row_index in range(out_features):
        mask = (cpu_events.flat_indices // in_features) == row_index
        row_attr = torch.zeros(in_features, dtype=torch.int32)
        if bool(mask.any().item()):
            cols = (cpu_events.flat_indices[mask] % in_features).tolist()
            vals = cpu_events.attribution_q31[mask].tolist()
            for col, val in zip(cols, vals):
                row_attr[int(col)] = int(val)
        row_attrs.append(row_attr)
    sim_flat, sim_attr = s1_row_major_compact_reference(row_attrs, in_features=in_features)
    assert torch.equal(sim_flat, cpu_events.flat_indices)
    assert torch.equal(sim_attr, cpu_events.attribution_q31)


def test_s2_compact_prefix_scan_contract_event_order() -> None:
    integration = _green_integration_receipt()
    events = integration.bound_candidate_attribution_events.as_integer_marginal_attribution_events()
    q_flat = integration.bound_q_levels_flat
    cpu_idx, cpu_moves = projected_moves_from_integer_attribution(events, q_flat)
    move_dense = torch.zeros(events.event_count(), dtype=torch.int8)
    for pos in range(int(events.event_count())):
        q_level = int(q_flat[int(events.flat_indices[pos])].item())
        grad_value = int(events.attribution_q31[pos].item())
        from calm.hrm_text_158.native_full_stack.integer_marginal_attribution import (
            _scalar_projected_move,
        )

        move_dense[pos] = _scalar_projected_move(q_level=q_level, grad_value=grad_value)
    sim_idx, sim_moves, _ = dense_compact_prefix_scan_reference(events.flat_indices, move_dense)
    assert torch.equal(sim_idx, cpu_idx)
    assert torch.equal(sim_moves, cpu_moves)


def test_s1_ttir_static_contains_sequence_stride() -> None:
    ttir_path = CREDIT_AXIS_KERNEL_MANIFEST_DIR / "s1_attribution.ttir"
    assert ttir_path.is_file(), "committed s1_attribution.ttir required for H.1b gate"
    ttir = ttir_path.read_text(encoding="utf-8")
    lowered = ttir.lower()
    assert "grad_stride_s" in lowered or (
        "for" in lowered and "range" in lowered
    )


def test_stage_native_evidence_recompute_rejects_lying_flag() -> None:
    integration = _green_integration_receipt()
    stage = CreditAxisStageNativeEvidence(
        s1_native=True,
        s2_native=True,
        s3_native=True,
        s4_native=False,
    )
    with pytest.raises(ValueError, match="torch_cuda_reference_only disagrees"):
        build_credit_axis_gpu_kernel_validation_receipt(
            **_receipt_kwargs(integration),
            runtime_evidence=CreditAxisGpuKernelRuntimeEvidence(
                hot_loop_kernel_invoked=True,
                torch_cuda_reference_only=False,
                stage_native_evidence=stage,
            ),
        )


def test_torch_cuda_reference_only_from_stage_evidence() -> None:
    native = CreditAxisStageNativeEvidence(True, True, True, True)
    ref = CreditAxisStageNativeEvidence(True, True, True, False)
    assert torch_cuda_reference_only_from_stage_evidence(native) is False
    assert torch_cuda_reference_only_from_stage_evidence(ref) is True


def test_s1_exceeds_supported_max_fail_closed() -> None:
    reason = check_s1_shape_bounds_or_raise(
        out_features=2,
        in_features=5000,
        n_capture_pairs=1,
        batch=1,
        sequence=1,
    )
    assert reason is not None
    assert "shape_exceeds_s1_supported_max" in reason


def test_s1_sequence_exceeds_supported_max_fail_closed() -> None:
    reason = check_s1_shape_bounds_or_raise(
        out_features=2,
        in_features=3,
        n_capture_pairs=1,
        batch=1,
        sequence=65,
    )
    assert reason is not None
    assert "shape_exceeds_s1_supported_max:sequence" in reason


def test_torch_sort_s4_classifies_not_kernelized_not_clean() -> None:
    evidence = CreditAxisGpuKernelRuntimeEvidence(
        hot_loop_kernel_invoked=True,
        torch_cuda_reference_only=True,
        gpu_output_repeat_stable=True,
        overflow_guard_tripped=False,
        gpu_payload_hashes={
            "attribution_events_hash": "a" * 64,
            "projected_move_indices_hash": "b" * 64,
            "projected_moves_hash": "c" * 64,
            "credit_q31_hash": "d" * 64,
            "sparse_vote_events_hash": "e" * 64,
        },
        cpu_oracle_payload_hashes={
            "attribution_events_hash": "a" * 64,
            "projected_move_indices_hash": "b" * 64,
            "projected_moves_hash": "c" * 64,
            "credit_q31_hash": "d" * 64,
            "sparse_vote_events_hash": "e" * 64,
        },
    )
    assert classify_credit_axis_gpu_runtime_branch(evidence) == BR_H_NOT_KERNELIZED


def test_kernel_module_built_when_triton_present() -> None:
    assert credit_axis_kernel_module_built() is True
