"""B2-4 GPU composition exact-parity proof for native dispatcher apply-under-cap-rows.

Routes through apply_cap_row_mutation_with_device_rows (not the Triton wrapper directly).
SOLE legitimate minter of composition_qacc_apply_parity_pass=True on GPU proof.

Formal GPU proof is executed by test-operator with both env gates + gpu:0 lane.
Missing preconditions → pytest.fail (never skip).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import pytest
import torch

from calm.hrm_text_158.native_full_stack.global_rate_cap_gpu import (
    DeviceGlobalRateCapStateRows,
)
from calm.hrm_text_158.native_full_stack.qacc_apply_composition_dispatch import (
    RUN_GPU_Q_ACC_APPLY_NATIVE_ENV,
    apply_cap_row_mutation_with_device_rows,
)
from calm.hrm_text_158.native_full_stack.qacc_apply_composition_native_parity_receipt import (
    QACC_APPLY_B2_4_BLOCKED_REASON,
    QACC_APPLY_B2_4_NON_CLAIMS,
    QaccApplyCompositionNativeParityReceipt,
    validate_qacc_apply_composition_native_parity_receipt,
)
from calm.hrm_text_158.native_full_stack.qacc_apply_native_parity_receipt import (
    QaccApplyNativeToken,
    canonical_tensor_payload_sha256,
    hash_qacc_apply_input_payloads,
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

RUN_GPU_Q_ACC_APPLY_ENV_MODULE = "HRM_TEXT_158_RUN_GPU_Q_ACC_APPLY"

_CPU_ONLY_TESTS = frozenset(
    {
        "test_pass_receipt_rejects_hash_mismatch",
        "test_env_constants_match_committed_stack",
        "test_module_contains_runnable_parity_tests",
    }
)


def _canonical_receipt_tensor_hash(t: torch.Tensor) -> str:
    return canonical_tensor_payload_sha256(
        t.detach().cpu().contiguous().view(-1).numpy().tobytes()
    )


def _empty_device_tensors(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.empty(0, dtype=torch.int64, device=device),
        torch.empty(0, dtype=torch.int16, device=device),
        torch.empty(0, dtype=torch.int32, device=device),
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
    """CPU oracle mirroring vote_update.py:1280-1297."""
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

    if replay_veto_indices is not None and replay_veto_indices.numel() > 0:
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


def _install_token_spy(monkeypatch: pytest.MonkeyPatch) -> list[QaccApplyNativeToken]:
    """Call-through spy: delegates to frozen wrapper, captures genuine launch token."""
    captured: list[QaccApplyNativeToken] = []
    real = apply_qacc_mutation_triton_native

    def _spy(**kwargs: Any) -> tuple[torch.Tensor, torch.Tensor, QaccApplyNativeToken]:
        q_out, acc_out, token = real(**kwargs)
        captured.append(token)
        return q_out, acc_out, token

    monkeypatch.setattr(
        "calm.hrm_text_158.native_full_stack.qacc_apply_composition_dispatch.apply_qacc_mutation_triton_native",
        _spy,
    )
    return captured


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
    replay_veto_indices: torch.Tensor,
    replay_veto_directions: torch.Tensor,
    replay_veto_thresholds: torch.Tensor,
) -> None:
    exact_q_hash = _canonical_receipt_tensor_hash(q_native)
    exact_acc_hash = _canonical_receipt_tensor_hash(acc_native)

    assert token.output_payload_hashes is not None
    assert token.input_payload_hashes is not None
    assert token.output_payload_hashes["q_levels"] == exact_q_hash
    assert token.output_payload_hashes["accumulators"] == exact_acc_hash

    replay_idx_bytes = (
        replay_veto_indices.detach().cpu().contiguous().view(-1).numpy().tobytes()
        if replay_veto_indices.numel() > 0
        else None
    )
    replay_dir_bytes = (
        replay_veto_directions.detach().cpu().contiguous().view(-1).numpy().tobytes()
        if replay_veto_directions.numel() > 0
        else None
    )
    replay_thresh_bytes = (
        replay_veto_thresholds.detach().cpu().contiguous().view(-1).numpy().tobytes()
        if replay_veto_thresholds.numel() > 0
        else None
    )

    expected_input = hash_qacc_apply_input_payloads(
        q_levels_bytes=q_levels.detach().cpu().contiguous().view(-1).numpy().tobytes(),
        new_accumulators_bytes=new_accumulators.detach().cpu().contiguous().view(-1).numpy().tobytes(),
        accepted_indices_bytes=accepted_indices.detach().cpu().contiguous().view(-1).numpy().tobytes(),
        accepted_directions_bytes=accepted_directions.detach().cpu().contiguous().view(-1).numpy().tobytes(),
        accepted_thresholds_bytes=accepted_thresholds.detach().cpu().contiguous().view(-1).numpy().tobytes(),
        replay_veto_indices_bytes=replay_idx_bytes,
        replay_veto_directions_bytes=replay_dir_bytes,
        replay_veto_thresholds_bytes=replay_thresh_bytes,
        original_accumulators_bytes=None,
        mutate_outputs=True,
    )
    assert token.input_payload_hashes == expected_input


def _mint_b2_4_pass_receipt_from_parity(
    *,
    q_native: torch.Tensor,
    acc_native: torch.Tensor,
    q_oracle: torch.Tensor,
    acc_oracle: torch.Tensor,
    token: QaccApplyNativeToken,
    no_cpu_materialization: bool,
) -> QaccApplyCompositionNativeParityReceipt:
    receipt = QaccApplyCompositionNativeParityReceipt(
        composition_qacc_apply_parity_pass=True,
        gpu_command_satisfied=True,
        no_cpu_row_materialization_before_apply=no_cpu_materialization,
        composition_native_routing=True,
        exact_q_output_hash=_canonical_receipt_tensor_hash(q_native),
        exact_acc_output_hash=_canonical_receipt_tensor_hash(acc_native),
        cpu_oracle_q_hash=_canonical_receipt_tensor_hash(q_oracle),
        cpu_oracle_acc_hash=_canonical_receipt_tensor_hash(acc_oracle),
        parity_atol=0.0,
        parity_rtol=0.0,
        wrapper_token=token,
        blocked_reason=QACC_APPLY_B2_4_BLOCKED_REASON,
        non_claims=QACC_APPLY_B2_4_NON_CLAIMS,
    )
    validate_qacc_apply_composition_native_parity_receipt(receipt)
    return receipt


def _require_b2_4_gpu_preconditions() -> None:
    if not torch.cuda.is_available():
        pytest.fail("B2-4 composition GPU exact-parity requires CUDA device (not skipped)")
    if os.environ.get(RUN_GPU_Q_ACC_APPLY_ENV) != "1":
        pytest.fail(
            f"{RUN_GPU_Q_ACC_APPLY_ENV}=1 is required for B2-4 composition GPU proof "
            "(not skipped)"
        )
    if os.environ.get(RUN_GPU_Q_ACC_APPLY_NATIVE_ENV) != "1":
        pytest.fail(
            f"{RUN_GPU_Q_ACC_APPLY_NATIVE_ENV}=1 is required for B2-4 composition GPU proof "
            "(not skipped)"
        )
    if _qacc_apply_accepted_pass_kernel is None:
        pytest.fail("Triton q_acc_apply kernels are unavailable (not skipped)")


@pytest.fixture(autouse=True)
def _b2_4_gpu_preconditions(request: pytest.FixtureRequest) -> None:
    if request.node.name in _CPU_ONLY_TESTS:
        return
    _require_b2_4_gpu_preconditions()


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
) -> dict[str, torch.Tensor]:
    q = torch.tensor(fx.q, dtype=torch.int8, device=device).view(1, -1)
    acc = torch.tensor(fx.acc, dtype=torch.int32, device=device).view(1, -1)
    accepted_indices = torch.tensor(fx.accepted_idx, dtype=torch.int64, device=device)
    accepted_directions = torch.tensor(fx.accepted_dir, dtype=torch.int16, device=device)
    accepted_thresholds = torch.tensor(fx.accepted_thresh, dtype=torch.int32, device=device)
    if fx.replay_idx is not None:
        replay_indices = torch.tensor(fx.replay_idx, dtype=torch.int64, device=device)
        replay_directions = torch.tensor(fx.replay_dir, dtype=torch.int16, device=device)
        replay_thresholds = torch.tensor(fx.replay_thresh, dtype=torch.int32, device=device)
    else:
        replay_indices, replay_directions, replay_thresholds = _empty_device_tensors(device)
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


def _fixture_to_state_rows(
    fx: _RowFixture,
    *,
    device: torch.device,
) -> DeviceGlobalRateCapStateRows:
    tensors = _fixture_tensors(fx, device=device)
    empty_i64, empty_i16, empty_i32 = _empty_device_tensors(device)
    accepted_indices = tensors["accepted_indices"]
    return DeviceGlobalRateCapStateRows(
        state_key="proj_in",
        accepted_indices=accepted_indices,
        accepted_directions=tensors["accepted_directions"],
        accepted_thresholds=tensors["accepted_thresholds"],
        accepted_global_flat_indices=accepted_indices.clone(),
        deferred_indices=empty_i64,
        deferred_directions=empty_i16,
        deferred_thresholds=empty_i32,
        deferred_global_flat_indices=empty_i64,
    )


def _run_composition_parity_case(
    fx: _RowFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> QaccApplyNativeToken:
    device = torch.device("cuda")
    tensors = _fixture_tensors(fx, device=device)
    state_rows = _fixture_to_state_rows(fx, device=device)
    captured = _install_token_spy(monkeypatch)

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

    result = apply_cap_row_mutation_with_device_rows(
        q_levels=tensors["q_levels"],
        new_accumulators=tensors["new_accumulators"],
        state_rows=state_rows,
        replay_veto_indices=tensors["replay_veto_indices"],
        replay_veto_directions=tensors["replay_veto_directions"],
        replay_veto_thresholds=tensors["replay_veto_thresholds"],
        mutate_outputs=True,
        original_accumulators=None,
        scope="b2_4_composition_gpu_parity",
    )

    assert result.backend == "cuda_native_triton", fx.case_id
    assert result.stats["composition_native_routing"] is True
    assert result.stats["cpu_selected_rows_materialized_before_q_acc_apply"] is False
    assert result.stats["python_row_lists_materialized_before_q_acc_apply"] is False

    q_native = result.q_levels
    acc_native = result.accumulators

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

    assert len(captured) == 1, f"{fx.case_id}: expected one captured launch token"
    token = captured[0]
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
    return token


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


def test_module_contains_runnable_parity_tests() -> None:
    """Structural: runnable parity harness wired (placeholder removed at impl)."""
    assert callable(_run_composition_parity_case)
    assert _OVERLAP_HIGH_THRESHOLD.case_id == "overlap_high_threshold"
    assert len(_ROW_FIXTURES) == 10


@pytest.mark.parametrize("fx", _ROW_FIXTURES, ids=[fx.case_id for fx in _ROW_FIXTURES])
def test_gpu_composition_row_case_exact_parity(
    fx: _RowFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Per-row GPU composition exact parity via dispatcher native path."""
    _run_composition_parity_case(fx, monkeypatch)


def test_gpu_composition_exact_parity_mints_pass_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Receipt-bearing invocation: overlap_high_threshold mints composition pass receipt."""
    fx = _OVERLAP_HIGH_THRESHOLD
    device = torch.device("cuda")
    tensors = _fixture_tensors(fx, device=device)
    state_rows = _fixture_to_state_rows(fx, device=device)
    captured = _install_token_spy(monkeypatch)

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

    result = apply_cap_row_mutation_with_device_rows(
        q_levels=tensors["q_levels"],
        new_accumulators=tensors["new_accumulators"],
        state_rows=state_rows,
        replay_veto_indices=tensors["replay_veto_indices"],
        replay_veto_directions=tensors["replay_veto_directions"],
        replay_veto_thresholds=tensors["replay_veto_thresholds"],
        mutate_outputs=True,
        original_accumulators=None,
        scope="b2_4_composition_gpu_parity_receipt",
    )

    q_native = result.q_levels
    acc_native = result.accumulators
    compare = compare_qacc_outputs(
        native_q=q_native.detach().cpu(),
        native_acc=acc_native.detach().cpu(),
        oracle_q=q_oracle,
        oracle_acc=acc_oracle,
    )
    assert compare["pass_all"] is True
    assert result.stats["cpu_selected_rows_materialized_before_q_acc_apply"] is False
    assert result.stats["python_row_lists_materialized_before_q_acc_apply"] is False

    assert len(captured) == 1
    token = captured[0]
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

    receipt = _mint_b2_4_pass_receipt_from_parity(
        q_native=q_native,
        acc_native=acc_native,
        q_oracle=q_oracle,
        acc_oracle=acc_oracle,
        token=token,
        no_cpu_materialization=True,
    )
    assert receipt.composition_qacc_apply_parity_pass is True
    assert receipt.gpu_command_satisfied is True
    assert receipt.wrapper_token is not None
    assert receipt.wrapper_token.kernel_family == "triton_qacc_apply"


def test_pass_receipt_rejects_hash_mismatch() -> None:
    """Local negative: mismatched exact/oracle hashes must not validate as pass-state."""
    token = QaccApplyNativeToken(
        kernel_family="triton_qacc_apply",
        kernel_symbol="_qacc_apply_accepted_pass_kernel+_qacc_apply_replay_pass_kernel",
        kernel_source_sha256="a" * 64,
        wrapper_launch_nonce="composition-negative-test",
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
    bad = QaccApplyCompositionNativeParityReceipt(
        composition_qacc_apply_parity_pass=True,
        gpu_command_satisfied=True,
        no_cpu_row_materialization_before_apply=True,
        composition_native_routing=True,
        exact_q_output_hash="deadbeef" * 8,
        exact_acc_output_hash="cafebabe" * 8,
        cpu_oracle_q_hash="baadf00d" * 8,
        cpu_oracle_acc_hash="feedface" * 8,
        parity_atol=0.0,
        parity_rtol=0.0,
        wrapper_token=token,
        blocked_reason=QACC_APPLY_B2_4_BLOCKED_REASON,
        non_claims=QACC_APPLY_B2_4_NON_CLAIMS,
    )
    with pytest.raises(ValueError, match="pass-state requires exact/oracle hash equality"):
        validate_qacc_apply_composition_native_parity_receipt(bad)


def test_env_constants_match_committed_stack() -> None:
    assert RUN_GPU_Q_ACC_APPLY_ENV_MODULE == RUN_GPU_Q_ACC_APPLY_ENV
    assert RUN_GPU_Q_ACC_APPLY_ENV == "HRM_TEXT_158_RUN_GPU_Q_ACC_APPLY"
    assert RUN_GPU_Q_ACC_APPLY_NATIVE_ENV == "HRM_TEXT_158_RUN_GPU_Q_ACC_APPLY_NATIVE"
