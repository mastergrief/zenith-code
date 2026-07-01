from __future__ import annotations

import os

import pytest
import torch

from calm.hrm_text_158.native_full_stack.event_coded_vote_update_adapter import (
    shape_only_accumulator_stub,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap import (
    GlobalRateCapOrderingMode,
    GlobalRateCapSpec,
    GlobalRateCapTensorInput,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap_gpu import RUN_GPU_GLOBAL_RATE_CAP_ENV
from calm.hrm_text_158.native_full_stack.sparse_cap_gpu_seam_adapter import (
    apply_sparse_event_coded_cap_via_gpu_seam,
    cpu_sparse_cap_oracle,
    parity_witness_tensors,
)
from calm.hrm_text_158.native_full_stack.vote_update import (
    RUN_GPU_Q_ACC_APPLY_ENV,
    VoteUpdateAccumulatorFormat,
    VoteUpdatePlan,
    VoteUpdateState,
)
from calm.llm_computer.tests.test_slice5_sparse_cap_gpu_seam_parity_v1 import (
    _shape_stub_new_acc_i32,
    _shape_stub_q_i16,
)

GPU_REPLAY = pytest.mark.skipif(
    os.environ.get(RUN_GPU_GLOBAL_RATE_CAP_ENV) != "1"
    or os.environ.get(RUN_GPU_Q_ACC_APPLY_ENV) != "1"
    or not torch.cuda.is_available(),
    reason="sparse cap replay-veto parity requires CUDA lane env gates",
)


def _replay_veto_sparse_inputs() -> list[GlobalRateCapTensorInput]:
    dev = torch.device("cpu")
    q = torch.zeros(6, dtype=torch.int8, device=dev)
    state = VoteUpdateState(
        q_levels=q,
        accumulators=shape_only_accumulator_stub(q),
        accumulator_format=VoteUpdateAccumulatorFormat.EVENT_CODED_LIVE_CARRIER,
    )
    applied = torch.tensor([1, 2, 3, 4], dtype=torch.int64, device=dev)
    replay = torch.tensor([5], dtype=torch.int64, device=dev)
    plan = VoteUpdatePlan(
        q_i16=_shape_stub_q_i16(q),
        new_acc_i32=_shape_stub_new_acc_i32(q),
        candidate_indices=applied,
        pre_veto_selected_indices=applied,
        applied_indices=applied,
        applied_directions=torch.tensor([1, 1, 1, 1], dtype=torch.int16, device=dev),
        applied_thresholds=torch.tensor([10, 10, 10, 10], dtype=torch.int32, device=dev),
        replay_ce_veto_indices=replay,
        replay_veto_directions=torch.tensor([-1], dtype=torch.int16, device=dev),
        replay_veto_thresholds=torch.tensor([8], dtype=torch.int32, device=dev),
        pc_aux_negative_indices=torch.empty(0, dtype=torch.int64, device=dev),
        pc_aux_veto_indices=torch.empty(0, dtype=torch.int64, device=dev),
        stats={"event_coded_live_carrier_plan": True},
        event_coded_sparse_active_idx=torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64, device=dev),
        event_coded_sparse_post_active_i32=torch.tensor(
            [30, 25, 20, 18, 12],
            dtype=torch.int32,
            device=dev,
        ),
    )
    return [
        GlobalRateCapTensorInput(
            state_key="state_replay",
            state=state,
            plan=plan,
        )
    ]


@GPU_REPLAY
def test_sparse_cap_gpu_seam_parity_replay_veto_active() -> None:
    spec = GlobalRateCapSpec(cap=2, step=7, ordering_mode=GlobalRateCapOrderingMode.MARGIN)
    cpu_inputs = _replay_veto_sparse_inputs()
    cpu_result = cpu_sparse_cap_oracle(cpu_inputs, spec)
    gpu_result = apply_sparse_event_coded_cap_via_gpu_seam(
        cap_inputs=cpu_inputs,
        spec=spec,
    )
    witnesses = parity_witness_tensors(cpu_result, gpu_result)
    state_key = "state_replay"
    witness = witnesses["per_state"][state_key]
    assert witness["accepted_local_gpu"] == witness["accepted_local_cpu"]
    assert witness["deferred_local_gpu"] == witness["deferred_local_cpu"]
    assert witness["q_sha_gpu"] == witness["q_sha_cpu"]
    assert gpu_result.deferred_backlog == cpu_result.deferred_backlog
    assert int(gpu_result.step_summary.get("q_changed_count", -1)) == int(
        cpu_result.step_summary.get("q_changed_count", -2)
    )
