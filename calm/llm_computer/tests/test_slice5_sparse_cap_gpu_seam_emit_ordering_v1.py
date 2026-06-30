from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any
from unittest import mock

import pytest
import torch

from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    apply_bounded_delta_vote_step,
    credit_from_weighted_grad,
    default_dry_run_rank_vote_spec,
    make_event_coded_live_tensor_state,
    project_s1_gradient_to_moves,
    sparse_rank_bucketed_int16_vote_events,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap import GlobalRateCapSpec
from calm.hrm_text_158.native_full_stack.global_rate_cap_gpu import RUN_GPU_GLOBAL_RATE_CAP_ENV
from calm.hrm_text_158.native_full_stack.sparse_cap_gpu_seam_adapter import (
    SparseCapGpuSeamApplyResult,
)
from calm.hrm_text_158.native_full_stack.vote_update import (
    RUN_GPU_Q_ACC_APPLY_ENV,
    VoteUpdateSpec,
)

GPU_EMIT_ORDERING = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="sparse cap emit ordering requires CUDA device states",
)


@dataclass
class _RecordingSparseCapEmitter:
    records: list[tuple[str, str]]
    order: list[str]

    def record_sparse_cap_sub_phase(
        self,
        sub_phase_id: str,
        *,
        optimizer_step_index: int,
        milestone_kind: str,
    ) -> None:
        self.records.append((str(sub_phase_id), str(milestone_kind)))
        if milestone_kind == "cap_gpu_seam_done":
            self.order.append("emit")


def _vote_spec(*, threshold_abs: int = 8) -> VoteUpdateSpec:
    return VoteUpdateSpec(
        threshold_abs=int(threshold_abs),
        accumulator_clip_min=-127,
        accumulator_clip_max=127,
        max_abs_per_tensor=16,
    )


def _cuda_sparse_cap_fixture():
    rank_spec = default_dry_run_rank_vote_spec()
    q_a = torch.tensor([[0, 1, -1, 0]], dtype=torch.int8)
    q_b = torch.tensor([[1, 0, 0, -1]], dtype=torch.int8)
    weighted_grad = torch.tensor([[-1.0, -2.0, 3.0, 4.0]])
    moves = project_s1_gradient_to_moves(weighted_grad, q_a)
    credit = credit_from_weighted_grad(weighted_grad)
    sparse_a = sparse_rank_bucketed_int16_vote_events(credit, moves, rank_spec)
    sparse_b = sparse_rank_bucketed_int16_vote_events(credit, moves, rank_spec)
    spec = _vote_spec(threshold_abs=8)
    cap = GlobalRateCapSpec(cap=4, step=1, mutate_outputs=True)
    device = "cuda:0"
    states_cpu = {
        "mod.a": make_event_coded_live_tensor_state("mod.a", q_a, 0.25, demotion_band=1),
        "mod.b": make_event_coded_live_tensor_state("mod.b", q_b, 0.25, demotion_band=1),
    }
    states = {
        key: _event_coded_state_on_device(state, device) for key, state in states_cpu.items()
    }
    return states, {"mod.a": sparse_a, "mod.b": sparse_b}, {"mod.a": spec, "mod.b": spec}, cap


def _event_coded_state_on_device(state, device: str):
    from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
        BoundedDeltaTensorState,
    )

    return BoundedDeltaTensorState(
        state_key=state.state_key,
        q_levels=state.q_levels.to(device),
        frozen_scale=state.frozen_scale,
        bounded_accumulator=state.bounded_accumulator,
        exact_accumulator_shadow=state.exact_accumulator_shadow,
        bounded_accumulator_fresh_for_exact_shadow=state.bounded_accumulator_fresh_for_exact_shadow,
        bounded_accumulator_rebuild_hot_exact_indices=state.bounded_accumulator_rebuild_hot_exact_indices,
        bounded_accumulator_rebuild_cold_default_value=state.bounded_accumulator_rebuild_cold_default_value,
        event_coded_live_carrier=state.event_coded_live_carrier,
    )


def _apply_with_gpu_lane(
    monkeypatch: pytest.MonkeyPatch,
    *,
    emitter: _RecordingSparseCapEmitter,
) -> Any:
    monkeypatch.setenv(RUN_GPU_GLOBAL_RATE_CAP_ENV, "1")
    monkeypatch.setenv(RUN_GPU_Q_ACC_APPLY_ENV, "1")
    states, sparse_by_key, vote_specs, cap = _cuda_sparse_cap_fixture()
    return apply_bounded_delta_vote_step(
        states,
        None,
        vote_specs,
        candidate_sparse_vote_events_by_key=sparse_by_key,
        global_cap_spec=cap,
        event_coded_sparse_vote_authority=True,
        sparse_cap_submilestone_emit=emitter,
    )


@GPU_EMIT_ORDERING
def test_cap_gpu_seam_done_not_emitted_when_seam_raises(monkeypatch) -> None:
    emitter = _RecordingSparseCapEmitter(records=[], order=[])
    monkeypatch.setattr(
        "calm.hrm_text_158.native_full_stack.sparse_cap_gpu_seam_adapter."
        "apply_sparse_event_coded_cap_via_gpu_seam",
        mock.Mock(side_effect=RuntimeError("simulated cap seam stall")),
    )
    with pytest.raises(RuntimeError, match="simulated cap seam stall"):
        _apply_with_gpu_lane(monkeypatch, emitter=emitter)
    assert ("cap_selection_cpu_copy", "cap_gpu_seam_done") not in emitter.records
    assert emitter.order == []


@GPU_EMIT_ORDERING
def test_cap_gpu_seam_done_emitted_only_after_seam_returns(monkeypatch) -> None:
    emitter = _RecordingSparseCapEmitter(records=[], order=[])
    seam_order: list[str] = []
    import calm.hrm_text_158.native_full_stack.sparse_cap_gpu_seam_adapter as adapter_mod

    original_seam = adapter_mod.apply_sparse_event_coded_cap_via_gpu_seam

    def _tracking_seam(*args: Any, **kwargs: Any) -> SparseCapGpuSeamApplyResult:
        seam_order.append("seam_enter")
        result = original_seam(*args, **kwargs)
        seam_order.append("seam_return")
        return result

    monkeypatch.setattr(adapter_mod, "apply_sparse_event_coded_cap_via_gpu_seam", _tracking_seam)
    _apply_with_gpu_lane(monkeypatch, emitter=emitter)
    assert ("cap_selection_cpu_copy", "cap_gpu_seam_done") in emitter.records
    assert seam_order == ["seam_enter", "seam_return"]
    assert emitter.order == ["emit"]
    combined = seam_order + emitter.order
    assert combined.index("seam_return") < combined.index("emit")


@GPU_EMIT_ORDERING
def test_cpu_shim_done_emitted_only_after_reference_returns(monkeypatch) -> None:
    cpu_emitter_records: list[tuple[str, str]] = []
    reference_order: list[str] = []
    import calm.hrm_text_158.native_full_stack.bounded_delta_learner as learner_mod

    class _CpuShimEmitter:
        def record_sparse_cap_sub_phase(
            self,
            sub_phase_id: str,
            *,
            optimizer_step_index: int,
            milestone_kind: str,
        ) -> None:
            cpu_emitter_records.append((str(sub_phase_id), str(milestone_kind)))

    original_reference = learner_mod.apply_global_rate_cap_reference

    def _tracking_reference(*args: Any, **kwargs: Any):
        reference_order.append("reference_enter")
        result = original_reference(*args, **kwargs)
        reference_order.append("reference_return")
        return result

    monkeypatch.setattr(learner_mod, "apply_global_rate_cap_reference", _tracking_reference)
    monkeypatch.setattr(
        "calm.hrm_text_158.native_full_stack.sparse_cap_gpu_seam_adapter."
        "sparse_cap_gpu_lane_enabled",
        lambda: False,
    )
    states, sparse_by_key, vote_specs, cap = _cuda_sparse_cap_fixture()
    apply_bounded_delta_vote_step(
        states,
        None,
        vote_specs,
        candidate_sparse_vote_events_by_key=sparse_by_key,
        global_cap_spec=cap,
        event_coded_sparse_vote_authority=True,
        sparse_cap_submilestone_emit=_CpuShimEmitter(),
    )
    assert ("cap_selection_cpu_copy", "cap_reference_cpu_resident_done") in cpu_emitter_records
    assert reference_order == ["reference_enter", "reference_return"]
    assert reference_order.index("reference_return") < len(cpu_emitter_records)
