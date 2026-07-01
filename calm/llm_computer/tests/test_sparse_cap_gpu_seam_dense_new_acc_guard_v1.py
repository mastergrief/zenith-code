from __future__ import annotations

import os

import pytest
import torch

from calm.hrm_text_158.native_full_stack.global_rate_cap_gpu import RUN_GPU_GLOBAL_RATE_CAP_ENV
from calm.hrm_text_158.native_full_stack.sparse_cap_gpu_seam_adapter import (
    apply_sparse_event_coded_cap_via_gpu_seam,
)
from calm.hrm_text_158.native_full_stack.vote_update import RUN_GPU_Q_ACC_APPLY_ENV
from calm.llm_computer.tests.test_slice5_sparse_cap_gpu_seam_parity_v1 import (
    _multi_state_sparse_inputs,
    _spec,
)

GPU_R3 = pytest.mark.skipif(
    os.environ.get(RUN_GPU_GLOBAL_RATE_CAP_ENV) != "1"
    or os.environ.get(RUN_GPU_Q_ACC_APPLY_ENV) != "1"
    or not torch.cuda.is_available(),
    reason="sparse cap dense-new_acc guard requires CUDA lane env gates",
)


def _forbidden_dense_new_acc_numels(cpu_inputs) -> set[int]:
    return {int(item.state.q_levels.numel()) for item in cpu_inputs}


def _install_dense_new_acc_guards(
    monkeypatch: pytest.MonkeyPatch,
    forbidden_numels: set[int],
    *,
    observed: list[tuple[str, int]],
) -> None:
    original_zeros = torch.zeros
    original_clone = torch.Tensor.clone

    def guarded_zeros(*args, **kwargs):
        tensor = original_zeros(*args, **kwargs)
        if tensor.dtype == torch.int32 and int(tensor.numel()) in forbidden_numels:
            observed.append(("zeros", int(tensor.numel())))
            raise AssertionError(
                f"q.numel-scale int32 new_acc zeros forbidden on sparse GPU seam "
                f"(numel={int(tensor.numel())})"
            )
        return tensor

    def guarded_clone(self: torch.Tensor, *args, **kwargs):
        if self.dtype == torch.int32 and int(self.numel()) in forbidden_numels:
            observed.append(("clone", int(self.numel())))
            raise AssertionError(
                f"q.numel-scale int32 new_acc clone forbidden on sparse GPU seam "
                f"(numel={int(self.numel())})"
            )
        return original_clone(self, *args, **kwargs)

    monkeypatch.setattr(torch, "zeros", guarded_zeros)
    monkeypatch.setattr(torch.Tensor, "clone", guarded_clone)


@GPU_R3
def test_sparse_cap_gpu_seam_forbids_q_numel_scale_dense_new_acc(monkeypatch) -> None:
    cpu_inputs = _multi_state_sparse_inputs("cpu")
    forbidden = _forbidden_dense_new_acc_numels(cpu_inputs)
    observed: list[tuple[str, int]] = []
    _install_dense_new_acc_guards(monkeypatch, forbidden, observed=observed)
    result = apply_sparse_event_coded_cap_via_gpu_seam(
        cap_inputs=cpu_inputs,
        spec=_spec(),
    )
    assert result.tensor_results
    assert observed == []
