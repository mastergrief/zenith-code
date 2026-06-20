"""B2-3 GPU exact-parity proof for native Triton q_acc_apply.

SOLE legitimate minter of qacc_apply_parity_pass=True + gpu_command_satisfied=True.
Requires real CUDA launch via apply_qacc_mutation_triton_native; mints receipt via
direct dataclass construction + validate() (NOT the B2-2a builder).

Formal GPU proof is executed by test-operator with HRM_TEXT_158_RUN_GPU_Q_ACC_APPLY=1
and gpu:0 lane. Missing preconditions → pytest.fail (never skip).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import pytest
import torch

from calm.hrm_text_158.native_full_stack.qacc_apply_native_parity_receipt import (
    QACC_APPLY_B2_BLOCKED_REASON,
    QACC_APPLY_B2_NON_CLAIMS,
    QaccApplyNativeParityReceipt,
    QaccApplyNativeToken,
    canonical_tensor_payload_sha256,
    hash_qacc_apply_input_payloads,
    validate_qacc_apply_native_parity_receipt,
)
from calm.hrm_text_158.native_full_stack.qacc_apply_triton_kernel import (
    _qacc_apply_accepted_pass_kernel,
    apply_qacc_mutation_triton_native,
    compare_qacc_outputs,
)
from calm.hrm_text_158.native_full_stack.vote_update import (
    RUN_GPU_Q_ACC_APPLY_ENV,
    _apply_threshold_residual_in_place,
)

# Binding 5: committed env constant (RUN_GPU_Q_ACC_APPLY_NATIVE is prose alias only).
RUN_GPU_Q_ACC_APPLY_ENV_MODULE = "HRM_TEXT_158_RUN_GPU_Q_ACC_APPLY"


def _canonical_receipt_tensor_hash(t: torch.Tensor) -> str:
    """ONE helper for native + oracle receipt hashes (binding 1)."""
    return canonical_tensor_payload_sha256(
        t.detach().cpu().contiguous().view(-1).numpy().tobytes()
    )


def _cpu_oracle_qacc_apply_mutation(
    *,
    q_levels: torch.Tensor,
    new_accumulators: torch.Tensor,
    accepted_indices: torch.Tensor,
    accepted_directions: torch.Tensor,
    accepted_thresholds: torch.Tensor,
    replay_veto_indices: torch.Tensor | None = None,
    replay_veto_directions: torch.Tensor | None = None,
    replay_veto_thresholds: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU oracle mirroring vote_update.py:1280-1297 (binding 6)."""
    q_i16 = q_levels.flatten().to(torch.int16).clone()
    acc_i32 = new_accumulators.flatten().to(torch.int32).clone()

    accepted = accepted_indices.flatten().to(dtype=torch.int64)
    accepted_dirs = accepted_directions.flatten().to(dtype=torch.int16)
    accepted_thresh = accepted_thresholds.flatten().to(dtype=torch.int32)

    if accepted.numel() > 0:
        q_i16[accepted] = (q_i16[accepted] + accepted_dirs).clamp(-1, 1)
        _apply_threshold_residual_in_place(
            acc_i32,
            indices=accepted,
            directions=accepted_dirs,
            thresholds=accepted_thresh,
        )

    if replay_veto_indices is not None:
        replay = replay_veto_indices.flatten().to(dtype=torch.int64)
        replay_dirs = replay_veto_directions.flatten().to(dtype=torch.int16)
        replay_thresh = replay_veto_thresholds.flatten().to(dtype=torch.int32)
        _apply_threshold_residual_in_place(
            acc_i32,
            indices=replay,
            directions=replay_dirs,
            thresholds=replay_thresh,
        )

    q_out = q_i16.view_as(q_levels).to(torch.int8).contiguous()
    acc_out = acc_i32.view_as(new_accumulators).to(torch.int16).contiguous()
    return q_out, acc_out


def _assert_token_coupled_to_invocation(
    *,
    token: QaccApplyNativeToken,
    q_levels: torch.Tensor,
    new_accumulators: torch.Tensor,
    accepted_indices: torch.Tensor,
    accepted_directions: torch.Tensor,
    accepted_thresholds: torch.Tensor,
    q_native: torch.Tensor,
    acc_native: torch.Tensor,
    replay_veto_indices: torch.Tensor | None = None,
    replay_veto_directions: torch.Tensor | None = None,
    replay_veto_thresholds: torch.Tensor | None = None,
) -> None:
    """Binding 2: token hashes must match THIS launch's inputs and outputs."""
    exact_q_hash = _canonical_receipt_tensor_hash(q_native)
    exact_acc_hash = _canonical_receipt_tensor_hash(acc_native)

    assert token.output_payload_hashes is not None
    assert token.input_payload_hashes is not None
    assert token.output_payload_hashes["q_levels"] == exact_q_hash
    assert token.output_payload_hashes["accumulators"] == exact_acc_hash

    expected_input = hash_qacc_apply_input_payloads(
        q_levels_bytes=q_levels.detach().cpu().contiguous().view(-1).numpy().tobytes(),
        new_accumulators_bytes=new_accumulators.detach().cpu().contiguous().view(-1).numpy().tobytes(),
        accepted_indices_bytes=accepted_indices.detach().cpu().contiguous().view(-1).numpy().tobytes(),
        accepted_directions_bytes=accepted_directions.detach().cpu().contiguous().view(-1).numpy().tobytes(),
        accepted_thresholds_bytes=accepted_thresholds.detach().cpu().contiguous().view(-1).numpy().tobytes(),
        replay_veto_indices_bytes=(
            replay_veto_indices.detach().cpu().contiguous().view(-1).numpy().tobytes()
            if replay_veto_indices is not None
            else None
        ),
        replay_veto_directions_bytes=(
            replay_veto_directions.detach().cpu().contiguous().view(-1).numpy().tobytes()
            if replay_veto_directions is not None
            else None
        ),
        replay_veto_thresholds_bytes=(
            replay_veto_thresholds.detach().cpu().contiguous().view(-1).numpy().tobytes()
            if replay_veto_thresholds is not None
            else None
        ),
        original_accumulators_bytes=None,
        mutate_outputs=True,
    )
    assert token.input_payload_hashes == expected_input


def _mint_b2_3_pass_receipt_from_parity(
    *,
    q_native: torch.Tensor,
    acc_native: torch.Tensor,
    q_oracle: torch.Tensor,
    acc_oracle: torch.Tensor,
    token: QaccApplyNativeToken,
) -> QaccApplyNativeParityReceipt:
    """Binding 8: direct dataclass + validate(); NOT build_*."""
    exact_q_hash = _canonical_receipt_tensor_hash(q_native)
    exact_acc_hash = _canonical_receipt_tensor_hash(acc_native)
    oracle_q_hash = _canonical_receipt_tensor_hash(q_oracle)
    oracle_acc_hash = _canonical_receipt_tensor_hash(acc_oracle)

    receipt = QaccApplyNativeParityReceipt(
        qacc_apply_parity_pass=True,
        gpu_command_satisfied=True,
        native_hot_loop_kernel=True,
        native_call_path_marker_present=True,
        q_acc_apply_cpu_reference=False,
        q_acc_apply_final_row_torch_cuda_reference=False,
        global_cap_gpu_native_marker_seen=False,
        torch_cuda_ref_under_cap_rows_invoked=False,
        accepted_indices_unique=True,
        replay_indices_unique=True,
        accepted_then_replay_order=True,
        mutate_outputs_path="True",
        exact_q_output_hash=exact_q_hash,
        exact_acc_output_hash=exact_acc_hash,
        cpu_oracle_q_hash=oracle_q_hash,
        cpu_oracle_acc_hash=oracle_acc_hash,
        parity_atol=0.0,
        parity_rtol=0.0,
        wrapper_token=token,
        blocked_reason=QACC_APPLY_B2_BLOCKED_REASON,
        non_claims=QACC_APPLY_B2_NON_CLAIMS,
    )
    validate_qacc_apply_native_parity_receipt(receipt)
    return receipt


def _require_b2_3_gpu_preconditions() -> None:
    """Binding 4: fail cleanly when GPU proof preconditions are unmet."""
    if not torch.cuda.is_available():
        pytest.fail("B2-3 GPU exact-parity requires CUDA device (not skipped)")
    if os.environ.get(RUN_GPU_Q_ACC_APPLY_ENV) != "1":
        pytest.fail(
            f"{RUN_GPU_Q_ACC_APPLY_ENV}=1 is required for B2-3 GPU exact-parity "
            "(not skipped)"
        )
    if _qacc_apply_accepted_pass_kernel is None:
        pytest.fail("Triton q_acc_apply kernels are unavailable (not skipped)")


@pytest.fixture(autouse=True)
def _b2_3_gpu_preconditions(request: pytest.FixtureRequest) -> None:
    """Autouse fail gate for GPU parity tests; CPU-only helper tests exempt."""
    if request.node.name in (
        "test_pass_receipt_rejects_hash_mismatch",
        "test_env_constant_matches_committed_stack",
    ):
        return
    _require_b2_3_gpu_preconditions()


@dataclass(frozen=True)
class _RowFixture:
    case_id: str
    q: list[int]
    acc: list[int]
    accepted_idx: list[int]
    accepted_dir: list[int]
    accepted_thresh: list[int]
    replay_idx: list[int] | None = None
    replay_dir: list[int] | None = None
    replay_thresh: list[int] | None = None


def _fixture_tensors(
    fx: _RowFixture,
    *,
    device: torch.device,
) -> dict[str, torch.Tensor | None]:
    q = torch.tensor(fx.q, dtype=torch.int8, device=device).view(1, -1)
    acc = torch.tensor(fx.acc, dtype=torch.int32, device=device).view(1, -1)
    accepted_indices = torch.tensor(fx.accepted_idx, dtype=torch.int64, device=device)
    accepted_directions = torch.tensor(fx.accepted_dir, dtype=torch.int16, device=device)
    accepted_thresholds = torch.tensor(fx.accepted_thresh, dtype=torch.int32, device=device)
    replay_indices = replay_directions = replay_thresholds = None
    if fx.replay_idx is not None:
        replay_indices = torch.tensor(fx.replay_idx, dtype=torch.int64, device=device)
        replay_directions = torch.tensor(fx.replay_dir, dtype=torch.int16, device=device)
        replay_thresholds = torch.tensor(fx.replay_thresh, dtype=torch.int32, device=device)
    return {
        "q_levels": q,
        "new_accumulators": acc,
        "accepted_indices": accepted_indices,
        "accepted_directions": accepted_directions,
        "accepted_thresholds": accepted_thresholds,
        "replay_veto_indices": replay_indices,
        "replay_veto_directions": replay_directions,
        "replay_veto_thresholds": replay_thresholds,
    }


def _run_parity_case(fx: _RowFixture) -> None:
    device = torch.device("cuda")
    tensors = _fixture_tensors(fx, device=device)
    cpu_q = tensors["q_levels"].detach().cpu()
    cpu_acc = tensors["new_accumulators"].detach().cpu()
    cpu_accepted_idx = tensors["accepted_indices"].detach().cpu()
    cpu_accepted_dir = tensors["accepted_directions"].detach().cpu()
    cpu_accepted_thresh = tensors["accepted_thresholds"].detach().cpu()
    cpu_replay_idx = (
        tensors["replay_veto_indices"].detach().cpu()
        if tensors["replay_veto_indices"] is not None
        else None
    )
    cpu_replay_dir = (
        tensors["replay_veto_directions"].detach().cpu()
        if tensors["replay_veto_directions"] is not None
        else None
    )
    cpu_replay_thresh = (
        tensors["replay_veto_thresholds"].detach().cpu()
        if tensors["replay_veto_thresholds"] is not None
        else None
    )

    q_oracle, acc_oracle = _cpu_oracle_qacc_apply_mutation(
        q_levels=cpu_q,
        new_accumulators=cpu_acc,
        accepted_indices=cpu_accepted_idx,
        accepted_directions=cpu_accepted_dir,
        accepted_thresholds=cpu_accepted_thresh,
        replay_veto_indices=cpu_replay_idx,
        replay_veto_directions=cpu_replay_dir,
        replay_veto_thresholds=cpu_replay_thresh,
    )

    q_native, acc_native, token = apply_qacc_mutation_triton_native(
        q_levels=tensors["q_levels"],
        new_accumulators=tensors["new_accumulators"],
        accepted_indices=tensors["accepted_indices"],
        accepted_directions=tensors["accepted_directions"],
        accepted_thresholds=tensors["accepted_thresholds"],
        replay_veto_indices=tensors["replay_veto_indices"],
        replay_veto_directions=tensors["replay_veto_directions"],
        replay_veto_thresholds=tensors["replay_veto_thresholds"],
    )

    compare = compare_qacc_outputs(
        native_q=q_native.detach().cpu(),
        native_acc=acc_native.detach().cpu(),
        oracle_q=q_oracle,
        oracle_acc=acc_oracle,
    )
    assert compare["pass_all"] is True, f"{fx.case_id}: {compare}"

    assert _canonical_receipt_tensor_hash(q_native) == _canonical_receipt_tensor_hash(q_oracle)
    assert _canonical_receipt_tensor_hash(acc_native) == _canonical_receipt_tensor_hash(
        acc_oracle
    )

    _assert_token_coupled_to_invocation(
        token=token,
        q_levels=tensors["q_levels"],
        new_accumulators=tensors["new_accumulators"],
        accepted_indices=tensors["accepted_indices"],
        accepted_directions=tensors["accepted_directions"],
        accepted_thresholds=tensors["accepted_thresholds"],
        q_native=q_native,
        acc_native=acc_native,
        replay_veto_indices=tensors["replay_veto_indices"],
        replay_veto_directions=tensors["replay_veto_directions"],
        replay_veto_thresholds=tensors["replay_veto_thresholds"],
    )


_ROW_FIXTURES: tuple[_RowFixture, ...] = (
    _RowFixture("accepted_only", [0, 0, 1, 0], [0, 0, 0, 0], [2], [1], [3]),
    _RowFixture("replay_only", [0, -1, 0, 0], [0, 0, 0, 0], [], [], [], [1], [-1], [2]),
    _RowFixture("overlap_small", [0, 0, 0, 0], [5, 0, 0, 0], [0], [1], [3], [0], [-1], [2]),
    _RowFixture(
        "overlap_high_threshold",
        [0, 0, 0, 0],
        [90_000, 0, 0, 0],
        [0],
        [1],
        [40_000],
        [0],
        [-1],
        [40_000],
    ),
    _RowFixture("q_saturate_pos", [1, 0, 0, 0], [0, 0, 0, 0], [0], [1], [3]),
    _RowFixture("q_saturate_neg", [-1, 0, 0, 0], [0, 0, 0, 0], [0], [-1], [3]),
    _RowFixture("acc_clamp_hi", [0, 0, 0, 0], [10, 0, 0, 0], [0], [1], [5]),
    _RowFixture("acc_clamp_lo", [0, 0, 0, 0], [-10, 0, 0, 0], [0], [-1], [5]),
    _RowFixture("empty_both", [0, 1, -1, 0], [1, 2, 3, 4], [], [], []),
    _RowFixture(
        "multi_disjoint",
        [0, 0, 1, -1],
        [0, 0, 0, 0],
        [1, 2],
        [1, -1],
        [3, 3],
        [3],
        [-1],
        [2],
    ),
)

_OVERLAP_HIGH_THRESHOLD = next(
    fx for fx in _ROW_FIXTURES if fx.case_id == "overlap_high_threshold"
)


@pytest.mark.parametrize("fx", _ROW_FIXTURES, ids=[fx.case_id for fx in _ROW_FIXTURES])
def test_gpu_row_case_exact_parity(fx: _RowFixture) -> None:
    """Per-row GPU exact parity (all cases must pass on GPU)."""
    _run_parity_case(fx)


def test_gpu_exact_parity_mints_pass_receipt() -> None:
    """Binding 3: pass receipt minted from overlap_high_threshold launch on GPU."""
    fx = _OVERLAP_HIGH_THRESHOLD
    device = torch.device("cuda")
    tensors = _fixture_tensors(fx, device=device)

    cpu_q = tensors["q_levels"].detach().cpu()
    cpu_acc = tensors["new_accumulators"].detach().cpu()
    q_oracle, acc_oracle = _cpu_oracle_qacc_apply_mutation(
        q_levels=cpu_q,
        new_accumulators=cpu_acc,
        accepted_indices=tensors["accepted_indices"].detach().cpu(),
        accepted_directions=tensors["accepted_directions"].detach().cpu(),
        accepted_thresholds=tensors["accepted_thresholds"].detach().cpu(),
        replay_veto_indices=tensors["replay_veto_indices"].detach().cpu(),
        replay_veto_directions=tensors["replay_veto_directions"].detach().cpu(),
        replay_veto_thresholds=tensors["replay_veto_thresholds"].detach().cpu(),
    )

    q_native, acc_native, token = apply_qacc_mutation_triton_native(
        q_levels=tensors["q_levels"],
        new_accumulators=tensors["new_accumulators"],
        accepted_indices=tensors["accepted_indices"],
        accepted_directions=tensors["accepted_directions"],
        accepted_thresholds=tensors["accepted_thresholds"],
        replay_veto_indices=tensors["replay_veto_indices"],
        replay_veto_directions=tensors["replay_veto_directions"],
        replay_veto_thresholds=tensors["replay_veto_thresholds"],
    )

    compare = compare_qacc_outputs(
        native_q=q_native.detach().cpu(),
        native_acc=acc_native.detach().cpu(),
        oracle_q=q_oracle,
        oracle_acc=acc_oracle,
    )
    assert compare["pass_all"] is True

    _assert_token_coupled_to_invocation(
        token=token,
        q_levels=tensors["q_levels"],
        new_accumulators=tensors["new_accumulators"],
        accepted_indices=tensors["accepted_indices"],
        accepted_directions=tensors["accepted_directions"],
        accepted_thresholds=tensors["accepted_thresholds"],
        q_native=q_native,
        acc_native=acc_native,
        replay_veto_indices=tensors["replay_veto_indices"],
        replay_veto_directions=tensors["replay_veto_directions"],
        replay_veto_thresholds=tensors["replay_veto_thresholds"],
    )

    receipt = _mint_b2_3_pass_receipt_from_parity(
        q_native=q_native,
        acc_native=acc_native,
        q_oracle=q_oracle,
        acc_oracle=acc_oracle,
        token=token,
    )
    assert receipt.qacc_apply_parity_pass is True
    assert receipt.gpu_command_satisfied is True
    assert receipt.wrapper_token is not None
    assert receipt.wrapper_token.kernel_family == "triton_qacc_apply"


def test_pass_receipt_rejects_hash_mismatch() -> None:
    """Binding 7: mismatched exact/oracle hashes must not validate as pass-state."""
    token = QaccApplyNativeToken(
        kernel_family="triton_qacc_apply",
        kernel_symbol="_qacc_apply_accepted_pass_kernel+_qacc_apply_replay_pass_kernel",
        kernel_source_sha256="a" * 64,
        wrapper_launch_nonce="negative-test-nonce",
        input_payload_hashes={
            "q_levels": "q_in",
            "new_accumulators": "acc_in",
            "accepted_indices": "ai",
            "accepted_directions": "ad",
            "accepted_thresholds": "at",
            "replay_veto_indices": "ri",
            "replay_veto_directions": "rd",
            "replay_veto_thresholds": "rt",
            "original_accumulators": "oa",
            "mutate_outputs": "mo",
        },
        output_payload_hashes={"q_levels": "q_out", "accumulators": "acc_out"},
        backend="cuda",
        launch_time_ns=1,
    )
    bad = QaccApplyNativeParityReceipt(
        qacc_apply_parity_pass=True,
        gpu_command_satisfied=True,
        native_hot_loop_kernel=True,
        native_call_path_marker_present=True,
        q_acc_apply_cpu_reference=False,
        q_acc_apply_final_row_torch_cuda_reference=False,
        global_cap_gpu_native_marker_seen=False,
        torch_cuda_ref_under_cap_rows_invoked=False,
        accepted_indices_unique=True,
        replay_indices_unique=True,
        accepted_then_replay_order=True,
        mutate_outputs_path="True",
        exact_q_output_hash="deadbeef" * 8,
        exact_acc_output_hash="cafebabe" * 8,
        cpu_oracle_q_hash="baadf00d" * 8,
        cpu_oracle_acc_hash="feedface" * 8,
        parity_atol=0.0,
        parity_rtol=0.0,
        wrapper_token=token,
        blocked_reason=QACC_APPLY_B2_BLOCKED_REASON,
        non_claims=QACC_APPLY_B2_NON_CLAIMS,
    )
    with pytest.raises(ValueError, match="exact_q_output_hash must equal cpu_oracle_q_hash"):
        validate_qacc_apply_native_parity_receipt(bad)


def test_env_constant_matches_committed_stack() -> None:
    """Binding 5: module env binding matches vote_update committed constant."""
    assert RUN_GPU_Q_ACC_APPLY_ENV_MODULE == RUN_GPU_Q_ACC_APPLY_ENV
    assert RUN_GPU_Q_ACC_APPLY_ENV == "HRM_TEXT_158_RUN_GPU_Q_ACC_APPLY"
