"""Phase-1 selection/top-k/tie-break CUDA-reference receipt tests."""
from __future__ import annotations

import os
import time

import pytest
import torch

from calm.hrm_text_158.native_full_stack.selection_topk import (
    RUN_GPU_SELECTION_TOPK_ENV,
    SELECTION_TOPK_CONTROL_FLOW_NOTE,
    SELECTION_TOPK_TIEBREAK_CUDA_REFERENCE_SCOPE,
    select_pre_veto_candidates_from_plan,
    selection_topk_tiebreak_reference_on_device,
)
from calm.hrm_text_158.native_full_stack.vote_update import (
    VoteUpdateInputs,
    VoteUpdateSpec,
    VoteUpdateState,
    plan_integer_vote_update_reference,
)


GPU_SELECTION_TOPK = pytest.mark.skipif(
    os.environ.get(RUN_GPU_SELECTION_TOPK_ENV) != "1" or not torch.cuda.is_available(),
    reason=(
        "selection/top-k/tie-break CUDA receipt deferred; set "
        "HRM_TEXT_158_RUN_GPU_SELECTION_TOPK=1 only inside a granted gpu:0 lane"
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


def _plan(
    q: list[int] | torch.Tensor,
    acc: list[int] | torch.Tensor,
    votes: list[int] | torch.Tensor,
    spec: VoteUpdateSpec,
    *,
    device: str = "cpu",
):
    q_t = torch.as_tensor(q, dtype=torch.int8, device=device)
    acc_t = torch.as_tensor(acc, dtype=torch.int16, device=device)
    vote_t = torch.as_tensor(votes, dtype=torch.int16, device=device)
    return plan_integer_vote_update_reference(
        VoteUpdateState(q_levels=q_t, accumulators=acc_t),
        VoteUpdateInputs(votes=vote_t),
        spec,
    )


def _expected_pre_veto_directions(plan, threshold_abs: int) -> torch.Tensor:
    selected_acc = plan.new_acc_i32.flatten()[plan.pre_veto_selected_indices]
    return torch.where(
        selected_acc >= int(threshold_abs),
        torch.ones_like(selected_acc, dtype=torch.int16),
        -torch.ones_like(selected_acc, dtype=torch.int16),
    )


def _expected_pre_veto_thresholds(plan, threshold_abs: int) -> torch.Tensor:
    return torch.full(
        plan.pre_veto_selected_indices.shape,
        int(threshold_abs),
        dtype=torch.int32,
        device=plan.pre_veto_selected_indices.device,
    )


def _assert_selection_matches_plan(plan, result, threshold_abs: int) -> None:
    assert result.scope == SELECTION_TOPK_TIEBREAK_CUDA_REFERENCE_SCOPE
    assert torch.equal(result.pre_veto_selected_indices, plan.pre_veto_selected_indices)
    assert torch.equal(result.selected_directions, _expected_pre_veto_directions(plan, threshold_abs))
    assert torch.equal(result.selected_thresholds, _expected_pre_veto_thresholds(plan, threshold_abs))


def test_selection_reference_matches_cpu_tie_break_and_directions():
    spec = _spec(max_abs_per_tensor=2)
    plan = _plan([0, 0, 0, 0], [12, 12, -13, 13], [0, 0, 0, 0], spec)

    result = select_pre_veto_candidates_from_plan(plan, threshold_abs=spec.threshold_abs)

    _assert_selection_matches_plan(plan, result, spec.threshold_abs)
    assert result.pre_veto_selected_indices.tolist() == [2, 3]
    assert result.selected_directions.tolist() == [-1, 1]
    assert "control-flow sort seam" in SELECTION_TOPK_CONTROL_FLOW_NOTE


def test_selection_reference_stays_pre_veto_when_replay_veto_removes_apply():
    spec = _spec(max_abs_per_tensor=2)
    plan = plan_integer_vote_update_reference(
        VoteUpdateState(
            q_levels=torch.tensor([0, 0], dtype=torch.int8),
            accumulators=torch.tensor([12, -12], dtype=torch.int16),
        ),
        VoteUpdateInputs(
            votes=torch.tensor([0, 0], dtype=torch.int16),
            replay_ce_veto_votes=torch.tensor([-1, 1], dtype=torch.int16),
            replay_ce_veto_moves=torch.tensor([0, 0], dtype=torch.int8),
        ),
        spec,
    )

    result = select_pre_veto_candidates_from_plan(plan, threshold_abs=spec.threshold_abs)

    assert plan.pre_veto_selected_indices.tolist() == [0, 1]
    assert plan.applied_indices.tolist() == []
    assert plan.applied_directions.tolist() == []
    _assert_selection_matches_plan(plan, result, spec.threshold_abs)
    assert result.selected_directions.tolist() == [1, -1]


def test_selection_reference_handles_budget_limits_empty_candidates_and_zero_budget():
    fraction_spec = _spec(max_abs_per_tensor=10, fraction_per_tensor=0.25)
    fraction_plan = _plan([0, 0, 0, 0], [12, 11, 10, -12], [0, 0, 0, 0], fraction_spec)
    fraction_result = select_pre_veto_candidates_from_plan(
        fraction_plan,
        threshold_abs=fraction_spec.threshold_abs,
    )
    assert fraction_result.pre_veto_selected_indices.tolist() == [0]

    empty_plan = _plan([1, -1], [20, -20], [0, 0], _spec())
    empty_result = select_pre_veto_candidates_from_plan(empty_plan, threshold_abs=10)
    assert empty_plan.candidate_indices.tolist() == []
    assert empty_result.pre_veto_selected_indices.numel() == 0

    zero_plan = _plan([0, 0], [20, -20], [0, 0], _spec(max_abs_per_tensor=0))
    zero_result = select_pre_veto_candidates_from_plan(zero_plan, threshold_abs=10)
    assert zero_plan.candidate_indices.tolist() == [0, 1]
    assert zero_result.pre_veto_selected_indices.numel() == 0


def test_selection_reference_rejects_invalid_candidates_and_budget():
    with pytest.raises(ValueError, match="max_flips"):
        selection_topk_tiebreak_reference_on_device(
            new_acc_i32=torch.tensor([10], dtype=torch.int32),
            candidate_indices=torch.tensor([0], dtype=torch.int64),
            threshold_abs=10,
            max_flips=-1,
        )
    with pytest.raises(ValueError, match="out-of-range"):
        selection_topk_tiebreak_reference_on_device(
            new_acc_i32=torch.tensor([10], dtype=torch.int32),
            candidate_indices=torch.tensor([1], dtype=torch.int64),
            threshold_abs=10,
            max_flips=1,
        )


@GPU_SELECTION_TOPK
def test_selection_cuda_matches_cpu_reference_representative_cases():
    spec = _spec(max_abs_per_tensor=3)
    q = [0, 0, 1, -1, 0, 0, 0]
    acc = [12, -12, 30, -30, 19, -19, 10]
    votes = [0, 0, 0, 0, 1, -1, 0]
    cpu_plan = _plan(q, acc, votes, spec, device="cpu")
    cuda_plan = _plan(q, acc, votes, spec, device="cuda")

    result = select_pre_veto_candidates_from_plan(cuda_plan, threshold_abs=spec.threshold_abs)

    assert result.backend == "cuda"
    assert result.pre_veto_selected_indices.detach().cpu().tolist() == cpu_plan.pre_veto_selected_indices.tolist()
    assert result.selected_directions.detach().cpu().tolist() == _expected_pre_veto_directions(
        cpu_plan,
        spec.threshold_abs,
    ).tolist()
    assert result.selected_thresholds.detach().cpu().tolist() == _expected_pre_veto_thresholds(
        cpu_plan,
        spec.threshold_abs,
    ).tolist()


@GPU_SELECTION_TOPK
def test_selection_cuda_matches_cpu_reference_large_multishape_tensor():
    spec = _spec(max_abs_per_tensor=128, fraction_per_tensor=0.5)
    idx = torch.arange(64 * 65, device="cuda").reshape(64, 65)
    q = torch.zeros_like(idx, dtype=torch.int8)
    q[idx % 23 == 0] = 1
    q[idx % 29 == 0] = -1
    acc = ((idx % 257) - 128).to(torch.int16)
    votes = torch.where(
        idx % 7 == 0,
        torch.full_like(idx, 17),
        torch.where(idx % 7 == 1, torch.full_like(idx, -17), torch.zeros_like(idx)),
    ).to(torch.int16)
    cuda_plan = _plan(q, acc, votes, spec, device="cuda")
    cpu_plan = _plan(q.cpu(), acc.cpu(), votes.cpu(), spec, device="cpu")

    result = select_pre_veto_candidates_from_plan(cuda_plan, threshold_abs=spec.threshold_abs)

    assert result.pre_veto_selected_indices.detach().cpu().tolist() == cpu_plan.pre_veto_selected_indices.tolist()
    assert result.selected_directions.detach().cpu().tolist() == _expected_pre_veto_directions(
        cpu_plan,
        spec.threshold_abs,
    ).tolist()
    assert result.selected_thresholds.detach().cpu().tolist() == _expected_pre_veto_thresholds(
        cpu_plan,
        spec.threshold_abs,
    ).tolist()


@GPU_SELECTION_TOPK
def test_selection_cuda_perf_and_memory_receipt_scaffold():
    spec = _spec(max_abs_per_tensor=512, fraction_per_tensor=0.25)
    idx = torch.arange(262_144, device="cuda")
    q = torch.zeros_like(idx, dtype=torch.int8)
    q[idx % 31 == 0] = 1
    q[idx % 37 == 0] = -1
    acc = ((idx % 511) - 255).to(torch.int16)
    votes = torch.where(
        idx % 11 == 0,
        torch.full_like(idx, 23),
        torch.where(idx % 11 == 1, torch.full_like(idx, -23), torch.zeros_like(idx)),
    ).to(torch.int16)
    cuda_plan = _plan(q, acc, votes, spec, device="cuda")
    cpu_plan = _plan(q.cpu(), acc.cpu(), votes.cpu(), spec, device="cpu")

    torch.cuda.reset_peak_memory_stats()
    select_pre_veto_candidates_from_plan(cuda_plan, threshold_abs=spec.threshold_abs)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(20):
        cuda_result = select_pre_veto_candidates_from_plan(cuda_plan, threshold_abs=spec.threshold_abs)
    end.record()
    torch.cuda.synchronize()
    cuda_ms = start.elapsed_time(end) / 20.0

    start_cpu = time.perf_counter()
    for _ in range(20):
        cpu_result = select_pre_veto_candidates_from_plan(cpu_plan, threshold_abs=spec.threshold_abs)
    cpu_ms = (time.perf_counter() - start_cpu) * 1000.0 / 20.0

    assert cuda_result.pre_veto_selected_indices.detach().cpu().tolist() == cpu_result.pre_veto_selected_indices.tolist()
    assert cuda_result.selected_directions.detach().cpu().tolist() == cpu_result.selected_directions.tolist()
    assert cuda_result.selected_thresholds.detach().cpu().tolist() == cpu_result.selected_thresholds.tolist()
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    assert cuda_ms > 0.0
    assert cpu_ms > 0.0
    assert peak_allocated > 0
    assert peak_reserved >= peak_allocated
    print(
        "selection_topk_tiebreak_cuda_reference_receipt_scaffold "
        f"numel={idx.numel()} selected={cuda_result.pre_veto_selected_indices.numel()} "
        f"cuda_ms={cuda_ms:.4f} cpu_ms={cpu_ms:.4f} "
        f"peak_allocated={peak_allocated} peak_reserved={peak_reserved} "
        f"scope={SELECTION_TOPK_TIEBREAK_CUDA_REFERENCE_SCOPE}"
    )
