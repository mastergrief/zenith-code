from __future__ import annotations

import hashlib
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

GPU_ALIAS = pytest.mark.skipif(
    os.environ.get(RUN_GPU_GLOBAL_RATE_CAP_ENV) != "1"
    or os.environ.get(RUN_GPU_Q_ACC_APPLY_ENV) != "1"
    or not torch.cuda.is_available(),
    reason="sparse cap GPU alias immutability requires CUDA lane env gates",
)


def _tensor_bytes_sha(tensor: torch.Tensor) -> str:
    payload = tensor.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


@GPU_ALIAS
def test_sparse_cap_gpu_seam_input_q_and_carrier_byte_immutable() -> None:
    cpu_inputs = _multi_state_sparse_inputs("cpu")
    before_q = {
        item.state_key: _tensor_bytes_sha(item.state.q_levels) for item in cpu_inputs
    }
    result = apply_sparse_event_coded_cap_via_gpu_seam(
        cap_inputs=cpu_inputs,
        spec=_spec(),
    )
    assert result.tensor_results
    after_q = {
        item.state_key: _tensor_bytes_sha(item.state.q_levels) for item in cpu_inputs
    }
    assert before_q == after_q
