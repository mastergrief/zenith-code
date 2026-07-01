from __future__ import annotations

import os

import pytest
import torch

from calm.hrm_text_158.native_full_stack.global_rate_cap import (
    GlobalRateCapSpec,
    GlobalRateCapTensorInput,
    tensor_offsets_for_vote_update_states,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap_gpu import (
    RUN_GPU_GLOBAL_RATE_CAP_ENV,
    select_global_rate_cap_rows_torch_cuda_reference,
)
from calm.hrm_text_158.native_full_stack.vote_update import (
    RUN_GPU_Q_ACC_APPLY_ENV,
    VoteUpdatePlan,
)
from calm.llm_computer.tests.test_slice5_sparse_cap_gpu_seam_parity_v1 import (
    _multi_state_sparse_inputs,
    _sparse_plan,
    _spec,
)

GPU_FAIL_CLOSED = pytest.mark.skipif(
    os.environ.get(RUN_GPU_GLOBAL_RATE_CAP_ENV) != "1"
    or os.environ.get(RUN_GPU_Q_ACC_APPLY_ENV) != "1"
    or not torch.cuda.is_available(),
    reason="sparse cap GPU fail-closed lookup requires CUDA lane env gates",
)


@GPU_FAIL_CLOSED
def test_sparse_cap_gpu_selection_raises_on_missing_active_index() -> None:
    cpu_inputs = _multi_state_sparse_inputs("cpu")
    item = cpu_inputs[0]
    bad_plan = _sparse_plan(
        item.state.q_levels,
        applied_indices=[1, 2, 99],
        applied_directions=[1, 1, 1],
        applied_thresholds=[10, 10, 10],
        active_idx=[1, 2, 3],
        post_active=[30, 25, 20],
        device=torch.device("cpu"),
    )
    bad_inputs = [
        GlobalRateCapTensorInput(
            state_key=item.state_key,
            state=item.state,
            plan=bad_plan,
            vote_inputs=item.vote_inputs,
        ),
        *cpu_inputs[1:],
    ]
    with pytest.raises(ValueError, match="event-coded sparse abs_new_acc lookup miss"):
        select_global_rate_cap_rows_torch_cuda_reference(
            bad_inputs,
            _spec(),
            tensor_offsets=tensor_offsets_for_vote_update_states(bad_inputs),
            materialize_cpu_telemetry=False,
            event_coded_sparse_cap_enabled=True,
        )
