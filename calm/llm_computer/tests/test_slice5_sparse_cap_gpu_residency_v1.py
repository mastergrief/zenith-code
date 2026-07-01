from __future__ import annotations

import copy
import os
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
    apply_sparse_event_coded_cap_via_gpu_seam,
    prepare_sparse_cap_selection_inputs,
)
from calm.hrm_text_158.native_full_stack.vote_update import (
    RUN_GPU_Q_ACC_APPLY_ENV,
    VoteUpdateSpec,
)
from calm.llm_computer.tests.test_slice5_sparse_cap_gpu_seam_parity_v1 import (
    _multi_state_sparse_inputs,
    _spec,
)

GPU_SPARSE_CAP_RESIDENCY = pytest.mark.skipif(
    os.environ.get(RUN_GPU_GLOBAL_RATE_CAP_ENV) != "1"
    or os.environ.get(RUN_GPU_Q_ACC_APPLY_ENV) != "1"
    or not torch.cuda.is_available(),
    reason="sparse cap GPU residency requires CUDA lane env gates",
)


def _forbidden_cuda_q_numels(cpu_inputs) -> set[int]:
    per_state_numels = [int(item.state.q_levels.numel()) for item in cpu_inputs]
    forbidden = set(per_state_numels)
    forbidden.add(sum(per_state_numels))
    return forbidden


def _install_cuda_q_transfer_guards(
    monkeypatch: pytest.MonkeyPatch,
    forbidden_numels: set[int],
    *,
    observed: list[tuple[str, int]],
) -> None:
    original_cpu = torch.Tensor.cpu
    original_clone = torch.Tensor.clone
    original_detach = torch.Tensor.detach

    def _check(tensor: torch.Tensor, op: str) -> None:
        if tensor.dtype == torch.int8 and tensor.device.type == "cuda":
            numel = int(tensor.numel())
            if numel in forbidden_numels:
                observed.append((op, numel))
                raise AssertionError(
                    f"full q_levels {op} forbidden on sparse CUDA GPU cap path (numel={numel})"
                )

    def guarded_cpu(self: torch.Tensor, *args, **kwargs):
        _check(self, "cpu")
        return original_cpu(self, *args, **kwargs)

    def guarded_clone(self: torch.Tensor, *args, **kwargs):
        _check(self, "clone")
        return original_clone(self, *args, **kwargs)

    def guarded_detach(self: torch.Tensor, *args, **kwargs):
        tensor = original_detach(self, *args, **kwargs)
        if self.dtype == torch.int8 and self.device.type == "cuda":
            numel = int(self.numel())
            if numel in forbidden_numels:

                def guarded_detach_cpu(*cpu_args, **cpu_kwargs):
                    observed.append(("detach_cpu", numel))
                    raise AssertionError(
                        f"full q_levels detach().cpu() forbidden on sparse CUDA GPU cap path "
                        f"(numel={numel})"
                    )

                tensor.cpu = guarded_detach_cpu  # type: ignore[method-assign]
        return tensor

    monkeypatch.setattr(torch.Tensor, "cpu", guarded_cpu)
    monkeypatch.setattr(torch.Tensor, "clone", guarded_clone)
    monkeypatch.setattr(torch.Tensor, "detach", guarded_detach)


def _learner_writeback_fixture():
    rank_spec = default_dry_run_rank_vote_spec()
    q_a = torch.tensor([[0, 1, -1, 0]], dtype=torch.int8)
    q_b = torch.tensor([[1, 0, 0, -1]], dtype=torch.int8)
    weighted_grad = torch.tensor([[-1.0, -2.0, 3.0, 4.0]])
    moves = project_s1_gradient_to_moves(weighted_grad, q_a)
    credit = credit_from_weighted_grad(weighted_grad)
    sparse_a = sparse_rank_bucketed_int16_vote_events(credit, moves, rank_spec)
    sparse_b = sparse_rank_bucketed_int16_vote_events(credit, moves, rank_spec)
    spec = VoteUpdateSpec(
        threshold_abs=8,
        accumulator_clip_min=-127,
        accumulator_clip_max=127,
        max_abs_per_tensor=16,
    )
    cap = GlobalRateCapSpec(cap=4, step=1, mutate_outputs=True)
    states = {
        "mod.a": make_event_coded_live_tensor_state("mod.a", q_a, 0.25, demotion_band=1),
        "mod.b": make_event_coded_live_tensor_state("mod.b", q_b, 0.25, demotion_band=1),
    }
    return states, {"mod.a": sparse_a, "mod.b": sparse_b}, {"mod.a": spec, "mod.b": spec}, cap


@GPU_SPARSE_CAP_RESIDENCY
def test_sparse_cap_gpu_residency_no_full_q_levels_d2h(monkeypatch) -> None:
    monkeypatch.setenv(RUN_GPU_GLOBAL_RATE_CAP_ENV, "1")
    monkeypatch.setenv(RUN_GPU_Q_ACC_APPLY_ENV, "1")

    cpu_inputs = _multi_state_sparse_inputs("cpu")
    forbidden = _forbidden_cuda_q_numels(cpu_inputs)
    assert forbidden == {4, 8}
    observed: list[tuple[str, int]] = []
    _install_cuda_q_transfer_guards(monkeypatch, forbidden, observed=observed)

    prepared = prepare_sparse_cap_selection_inputs(cpu_inputs)
    assert all(item.state.q_levels.device.type == "cpu" for item in prepared)
    assert all(item.plan.applied_indices.device.type == "cuda" for item in prepared)
    assert all(item.state.q_levels.device.type == "cpu" for item in cpu_inputs)

    with mock.patch(
        "calm.hrm_text_158.native_full_stack.sparse_cap_gpu_seam_adapter."
        "apply_global_rate_cap_reference",
        side_effect=AssertionError("CPU apply_global_rate_cap_reference re-entered"),
    ):
        result = apply_sparse_event_coded_cap_via_gpu_seam(
            cap_inputs=cpu_inputs,
            spec=_spec(),
        )

    assert result.tensor_results
    assert observed == []
    assert all(item.state.q_levels.device.type == "cpu" for item in cpu_inputs)


@GPU_SPARSE_CAP_RESIDENCY
def test_sparse_cap_gpu_writeback_no_per_state_full_q_d2h(monkeypatch) -> None:
    monkeypatch.setenv(RUN_GPU_GLOBAL_RATE_CAP_ENV, "1")
    monkeypatch.setenv(RUN_GPU_Q_ACC_APPLY_ENV, "1")

    states, sparse_by_key, vote_specs, cap = _learner_writeback_fixture()
    forbidden = {4, 8}
    observed: list[tuple[str, int]] = []
    _install_cuda_q_transfer_guards(monkeypatch, forbidden, observed=observed)

    result = apply_bounded_delta_vote_step(
        copy.deepcopy(states),
        None,
        vote_specs,
        candidate_sparse_vote_events_by_key=sparse_by_key,
        global_cap_spec=cap,
        event_coded_sparse_vote_authority=True,
        local_selection_ordering_step=1,
    )

    assert result.global_summary.get("sparse_cap_submilestone_cap_selection_path") == "gpu_seam"
    assert observed == []
    for state in result.tensor_states.values():
        assert state.q_levels.device.type == "cpu"
        assert int(state.q_levels.numel()) == 4


@GPU_SPARSE_CAP_RESIDENCY
def test_sparse_cap_gpu_residency_apply_global_rate_cap_reference_not_called(monkeypatch) -> None:
    monkeypatch.setenv(RUN_GPU_GLOBAL_RATE_CAP_ENV, "1")
    monkeypatch.setenv(RUN_GPU_Q_ACC_APPLY_ENV, "1")
    cpu_inputs = _multi_state_sparse_inputs("cpu")

    with mock.patch(
        "calm.hrm_text_158.native_full_stack.sparse_cap_gpu_seam_adapter."
        "apply_global_rate_cap_reference",
        side_effect=AssertionError("CPU cap reference must not run on GPU lane"),
    ) as sentinel:
        apply_sparse_event_coded_cap_via_gpu_seam(cap_inputs=cpu_inputs, spec=_spec())
        sentinel.assert_not_called()
