from __future__ import annotations

import os
from unittest import mock

import pytest
import torch

from calm.hrm_text_158.native_full_stack.global_rate_cap_gpu import RUN_GPU_GLOBAL_RATE_CAP_ENV
from calm.hrm_text_158.native_full_stack.sparse_cap_gpu_seam_adapter import (
    apply_sparse_event_coded_cap_via_gpu_seam,
    prepare_gpu_sparse_cap_inputs,
)
from calm.hrm_text_158.native_full_stack.vote_update import RUN_GPU_Q_ACC_APPLY_ENV
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


@GPU_SPARSE_CAP_RESIDENCY
def test_sparse_cap_gpu_residency_no_full_q_levels_d2h(monkeypatch) -> None:
    monkeypatch.setenv(RUN_GPU_GLOBAL_RATE_CAP_ENV, "1")
    monkeypatch.setenv(RUN_GPU_Q_ACC_APPLY_ENV, "1")

    cuda_inputs = _multi_state_sparse_inputs("cuda")
    full_q_numel = sum(int(item.state.q_levels.numel()) for item in cuda_inputs)
    observed: list[int] = []

    original_detach = torch.Tensor.detach

    def guarded_detach(self: torch.Tensor, *args, **kwargs):
        tensor = original_detach(self, *args, **kwargs)
        if (
            self.dtype == torch.int8
            and int(self.numel()) == full_q_numel
            and self.device.type == "cuda"
        ):
            observed.append(int(self.numel()))

            def guarded_cpu(*cpu_args, **cpu_kwargs):
                raise AssertionError(
                    "full q_levels D2H forbidden on sparse CUDA GPU cap path"
                )

            tensor.cpu = guarded_cpu  # type: ignore[method-assign]
        return tensor

    monkeypatch.setattr(torch.Tensor, "detach", guarded_detach)

    prepared = prepare_gpu_sparse_cap_inputs(cuda_inputs)
    assert all(item.state.q_levels.device.type == "cuda" for item in prepared)
    assert all(item.plan.new_acc_i32.device.type == "cuda" for item in prepared)

    with mock.patch(
        "calm.hrm_text_158.native_full_stack.sparse_cap_gpu_seam_adapter."
        "apply_global_rate_cap_reference",
        side_effect=AssertionError("CPU apply_global_rate_cap_reference re-entered"),
    ):
        result = apply_sparse_event_coded_cap_via_gpu_seam(
            cap_inputs=cuda_inputs,
            spec=_spec(),
        )

    assert result.tensor_results
    assert observed == []


@GPU_SPARSE_CAP_RESIDENCY
def test_sparse_cap_gpu_residency_apply_global_rate_cap_reference_not_called(monkeypatch) -> None:
    monkeypatch.setenv(RUN_GPU_GLOBAL_RATE_CAP_ENV, "1")
    monkeypatch.setenv(RUN_GPU_Q_ACC_APPLY_ENV, "1")
    cuda_inputs = _multi_state_sparse_inputs("cuda")

    with mock.patch(
        "calm.hrm_text_158.native_full_stack.sparse_cap_gpu_seam_adapter."
        "apply_global_rate_cap_reference",
        side_effect=AssertionError("CPU cap reference must not run on GPU lane"),
    ) as sentinel:
        apply_sparse_event_coded_cap_via_gpu_seam(cap_inputs=cuda_inputs, spec=_spec())
        sentinel.assert_not_called()
