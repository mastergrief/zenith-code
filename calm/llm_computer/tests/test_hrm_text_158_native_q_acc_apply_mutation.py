"""Phase-1 q/acc apply-mutation torch-CUDA reference receipt tests."""
from __future__ import annotations

import os
import time
from typing import Any

import pytest
import torch

from calm.hrm_text_158.native_full_stack.global_rate_cap import (
    GlobalRateCapSpec,
    GlobalRateCapTensorInput,
    apply_global_rate_cap_reference,
)
from calm.hrm_text_158.native_full_stack.vote_update import (
    Q_ACC_APPLY_MUTATION_TORCH_CUDA_REFERENCE_SCOPE,
    RUN_GPU_Q_ACC_APPLY_ENV,
    VoteUpdateInputs,
    VoteUpdateSpec,
    VoteUpdateState,
    plan_integer_vote_update_reference,
    q_acc_apply_mutation_torch_cuda_reference_under_cap_rows,
)


GPU_Q_ACC_APPLY = pytest.mark.skipif(
    os.environ.get(RUN_GPU_Q_ACC_APPLY_ENV) != "1" or not torch.cuda.is_available(),
    reason=(
        "q/acc apply-mutation CUDA receipt deferred; set "
        "HRM_TEXT_158_RUN_GPU_Q_ACC_APPLY=1 only inside a granted gpu:0 lane"
    ),
)


def _spec(**kwargs) -> VoteUpdateSpec:
    base = dict(
        threshold_abs=10,
        accumulator_clip_min=-127,
        accumulator_clip_max=127,
        decay_numerator=1,
        decay_denominator=1,
        max_abs_per_tensor=64,
        fraction_per_tensor=1.0,
    )
    base.update(kwargs)
    return VoteUpdateSpec(**base)


def _state(q: list[int] | torch.Tensor, acc: list[int] | torch.Tensor) -> VoteUpdateState:
    return VoteUpdateState(
        q_levels=torch.as_tensor(q, dtype=torch.int8),
        accumulators=torch.as_tensor(acc, dtype=torch.int16),
    )


def _inputs(votes: list[int] | torch.Tensor, **kwargs) -> VoteUpdateInputs:
    converted = {}
    for name, value in kwargs.items():
        if value is None:
            converted[name] = None
        elif name.endswith("moves"):
            converted[name] = torch.as_tensor(value, dtype=torch.int8)
        else:
            converted[name] = torch.as_tensor(value, dtype=torch.int16)
    return VoteUpdateInputs(votes=torch.as_tensor(votes, dtype=torch.int16), **converted)


def _accepted_row_tensors(plan, accepted_indices: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    direction_by_index = {
        int(idx): int(plan.applied_directions[pos].item())
        for pos, idx in enumerate(plan.applied_indices.tolist())
    }
    threshold_by_index = {
        int(idx): int(plan.applied_thresholds[pos].item())
        for pos, idx in enumerate(plan.applied_indices.tolist())
    }
    return (
        torch.tensor(accepted_indices, dtype=torch.int64),
        torch.tensor([direction_by_index[int(idx)] for idx in accepted_indices], dtype=torch.int16),
        torch.tensor([threshold_by_index[int(idx)] for idx in accepted_indices], dtype=torch.int32),
    )


def _cap_fixture(
    *,
    q: list[int] | torch.Tensor,
    acc: list[int] | torch.Tensor,
    votes: list[int] | torch.Tensor,
    cap: int,
    step: int = 1,
    mutate_outputs: bool = True,
    spec: VoteUpdateSpec | None = None,
    **vote_kwargs,
) -> dict[str, Any]:
    state = _state(q, acc)
    inputs = _inputs(votes, **vote_kwargs)
    plan = plan_integer_vote_update_reference(state, inputs, spec or _spec())
    item = GlobalRateCapTensorInput(state_key="fixture.tensor", state=state, plan=plan)
    cpu_result = apply_global_rate_cap_reference(
        [item],
        GlobalRateCapSpec(cap=cap, step=step, mutate_outputs=mutate_outputs),
        tensor_offsets={"fixture.tensor": 0},
    )
    tensor_result = cpu_result.tensor_results[0]
    accepted_indices = [row.flat_index for row in cpu_result.accepted_rows]
    accepted, accepted_dirs, accepted_thresholds = _accepted_row_tensors(plan, accepted_indices)
    return {
        "state": state,
        "plan": plan,
        "cpu_result": cpu_result,
        "tensor_result": tensor_result,
        "accepted_indices": accepted,
        "accepted_directions": accepted_dirs,
        "accepted_thresholds": accepted_thresholds,
        "replay_indices": plan.replay_ce_veto_indices,
        "replay_directions": plan.replay_veto_directions,
        "replay_thresholds": plan.replay_veto_thresholds,
    }


def _cuda_apply_from_fixture(fixture: dict[str, Any], *, mutate_outputs: bool = True):
    state = fixture["state"]
    plan = fixture["plan"]
    return q_acc_apply_mutation_torch_cuda_reference_under_cap_rows(
        q_levels=state.q_levels.cuda(),
        new_accumulators=plan.new_acc_i32.cuda(),
        accepted_indices=fixture["accepted_indices"].cuda(),
        accepted_directions=fixture["accepted_directions"].cuda(),
        accepted_thresholds=fixture["accepted_thresholds"].cuda(),
        replay_veto_indices=fixture["replay_indices"].cuda(),
        replay_veto_directions=fixture["replay_directions"].cuda(),
        replay_veto_thresholds=fixture["replay_thresholds"].cuda(),
        mutate_outputs=mutate_outputs,
        original_accumulators=state.accumulators.cuda(),
    )


def test_cap_fixture_derives_final_rows_not_raw_local_plan_rows():
    fixture = _cap_fixture(
        q=[0, 0, 0, 0],
        acc=[0, 0, 0, 0],
        votes=[30, 30, 30, 0],
        cap=1,
        replay_ce_veto_votes=[0, -1, 0, 0],
        replay_ce_veto_moves=[0, 0, 0, 0],
    )
    plan = fixture["plan"]
    tensor = fixture["tensor_result"]

    assert plan.applied_indices.tolist() == [0, 2]
    assert plan.replay_ce_veto_indices.tolist() == [1]
    assert fixture["accepted_indices"].tolist() == [0]
    assert tensor.stats["global_rate_cap_deferred_indices"] == [2]
    assert tensor.q_levels.tolist() == [1, 0, 0, 0]
    assert tensor.accumulators.tolist() == [9, 9, 30, 0]


def test_q_acc_apply_default_off_before_gpu_lane(monkeypatch):
    monkeypatch.delenv(RUN_GPU_Q_ACC_APPLY_ENV, raising=False)
    with pytest.raises(RuntimeError, match=RUN_GPU_Q_ACC_APPLY_ENV):
        q_acc_apply_mutation_torch_cuda_reference_under_cap_rows(
            q_levels=torch.zeros(1, dtype=torch.int8),
            new_accumulators=torch.zeros(1, dtype=torch.int16),
            accepted_indices=torch.empty(0, dtype=torch.int64),
            accepted_directions=torch.empty(0, dtype=torch.int16),
            accepted_thresholds=torch.empty(0, dtype=torch.int32),
        )


@GPU_Q_ACC_APPLY
def test_q_acc_apply_cuda_matches_cap_fixture_for_accepted_deferred_and_replay_rows():
    fixture = _cap_fixture(
        q=[0, 0, 0, 0],
        acc=[0, 0, 0, 0],
        votes=[30, 30, 30, 0],
        cap=1,
        replay_ce_veto_votes=[0, -1, 0, 0],
        replay_ce_veto_moves=[0, 0, 0, 0],
    )

    result = _cuda_apply_from_fixture(fixture)
    tensor = fixture["tensor_result"]

    assert result.scope == Q_ACC_APPLY_MUTATION_TORCH_CUDA_REFERENCE_SCOPE
    assert result.backend == "cuda"
    assert result.q_levels.detach().cpu().tolist() == tensor.q_levels.tolist()
    assert result.accumulators.detach().cpu().tolist() == tensor.accumulators.tolist()
    assert result.stats["accepted_count"] == 1
    assert result.stats["replay_veto_count"] == 1
    assert result.stats["q_changed_count"] == 1


@GPU_Q_ACC_APPLY
def test_q_acc_apply_cuda_matches_positive_negative_and_empty_accepted_cap_cases():
    signed_fixture = _cap_fixture(
        q=[0, 0],
        acc=[-30, 30],
        votes=[0, 0],
        cap=2,
    )
    signed = _cuda_apply_from_fixture(signed_fixture)
    assert signed.q_levels.detach().cpu().tolist() == [-1, 1]
    assert signed.accumulators.detach().cpu().tolist() == [-9, 9]

    empty_fixture = _cap_fixture(
        q=[0, 0],
        acc=[0, 0],
        votes=[30, -30],
        cap=0,
    )
    empty = _cuda_apply_from_fixture(empty_fixture)
    assert empty.q_levels.detach().cpu().tolist() == [0, 0]
    assert empty.accumulators.detach().cpu().tolist() == [30, -30]
    assert empty.stats["accepted_count"] == 0


@GPU_Q_ACC_APPLY
def test_q_acc_apply_cuda_freeze_matches_mutate_outputs_false_reference():
    fixture = _cap_fixture(
        q=[0, 0, 0],
        acc=[0, 0, 0],
        votes=[30, 30, 30],
        cap=1,
        mutate_outputs=False,
    )

    result = _cuda_apply_from_fixture(fixture, mutate_outputs=False)
    tensor = fixture["tensor_result"]

    assert tensor.stats["ternary_mutation_frozen"] is True
    assert result.q_levels.detach().cpu().tolist() == tensor.q_levels.tolist()
    assert result.accumulators.detach().cpu().tolist() == tensor.accumulators.tolist()
    assert result.stats["mutate_outputs"] is False
    assert result.stats["q_changed_count"] == 0


@GPU_Q_ACC_APPLY
def test_q_acc_apply_cuda_clamps_synthetic_boundary_rows():
    result = q_acc_apply_mutation_torch_cuda_reference_under_cap_rows(
        q_levels=torch.tensor([1, -1], dtype=torch.int8, device="cuda"),
        new_accumulators=torch.tensor([15, -15], dtype=torch.int16, device="cuda"),
        accepted_indices=torch.tensor([0, 1], dtype=torch.int64, device="cuda"),
        accepted_directions=torch.tensor([1, -1], dtype=torch.int16, device="cuda"),
        accepted_thresholds=torch.tensor([10, 10], dtype=torch.int32, device="cuda"),
        original_accumulators=torch.tensor([0, 0], dtype=torch.int16, device="cuda"),
    )

    assert result.q_levels.detach().cpu().tolist() == [1, -1]
    assert result.accumulators.detach().cpu().tolist() == [5, -5]


@GPU_Q_ACC_APPLY
def test_q_acc_apply_cuda_perf_and_memory_receipt_scaffold():
    numel = 16_384
    idx = torch.arange(numel)
    q = torch.zeros(numel, dtype=torch.int8)
    acc = ((idx % 257) - 128).to(torch.int16)
    votes = torch.where(
        idx % 7 == 0,
        torch.full_like(idx, 37),
        torch.where(idx % 7 == 1, torch.full_like(idx, -37), torch.zeros_like(idx)),
    ).to(torch.int16)
    fixture = _cap_fixture(q=q, acc=acc, votes=votes, cap=512, spec=_spec(max_abs_per_tensor=2048))
    tensor = fixture["tensor_result"]

    torch.cuda.reset_peak_memory_stats()
    warm = _cuda_apply_from_fixture(fixture)
    torch.cuda.synchronize()
    assert warm.q_levels.detach().cpu().tolist() == tensor.q_levels.tolist()
    assert warm.accumulators.detach().cpu().tolist() == tensor.accumulators.tolist()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(20):
        result = _cuda_apply_from_fixture(fixture)
    end.record()
    torch.cuda.synchronize()
    cuda_ms = start.elapsed_time(end) / 20.0
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()

    assert result.q_levels.detach().cpu().tolist() == tensor.q_levels.tolist()
    assert result.accumulators.detach().cpu().tolist() == tensor.accumulators.tolist()
    assert cuda_ms > 0.0
    assert peak_allocated > 0
    assert peak_reserved >= peak_allocated
    print(
        "q_acc_apply_mutation_torch_cuda_reference_receipt_scaffold "
        f"numel={numel} accepted={fixture['accepted_indices'].numel()} "
        f"replay={fixture['replay_indices'].numel()} cuda_ms={cuda_ms:.4f} "
        f"peak_allocated={peak_allocated} peak_reserved={peak_reserved} "
        f"scope={Q_ACC_APPLY_MUTATION_TORCH_CUDA_REFERENCE_SCOPE}"
    )
