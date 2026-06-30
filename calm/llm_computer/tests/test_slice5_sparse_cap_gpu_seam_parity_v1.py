from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from calm.hrm_text_158.native_full_stack.event_coded_vote_update_adapter import (
    _shape_stub_new_acc_i32,
    _shape_stub_q_i16,
    shape_only_accumulator_stub,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap import (
    GlobalRateCapOrderingMode,
    GlobalRateCapSpec,
    GlobalRateCapTensorInput,
    tensor_offsets_for_vote_update_states,
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

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures/slice5_sparse_cap_gpu_seam_parity_multi_state"

GPU_SPARSE_CAP_PARITY = pytest.mark.skipif(
    os.environ.get(RUN_GPU_GLOBAL_RATE_CAP_ENV) != "1"
    or os.environ.get(RUN_GPU_Q_ACC_APPLY_ENV) != "1"
    or not torch.cuda.is_available(),
    reason="sparse cap GPU seam parity requires CUDA lane env gates",
)


def _spec() -> GlobalRateCapSpec:
    return GlobalRateCapSpec(cap=3, step=7, ordering_mode=GlobalRateCapOrderingMode.MARGIN)


def _sparse_plan(
    q_levels: torch.Tensor,
    *,
    applied_indices: list[int],
    applied_directions: list[int],
    applied_thresholds: list[int],
    active_idx: list[int],
    post_active: list[int],
    device: torch.device,
) -> VoteUpdatePlan:
    applied = torch.tensor(applied_indices, dtype=torch.int64, device=device)
    return VoteUpdatePlan(
        q_i16=_shape_stub_q_i16(q_levels),
        new_acc_i32=_shape_stub_new_acc_i32(q_levels),
        candidate_indices=applied,
        pre_veto_selected_indices=applied,
        applied_indices=applied,
        applied_directions=torch.tensor(applied_directions, dtype=torch.int16, device=device),
        applied_thresholds=torch.tensor(applied_thresholds, dtype=torch.int32, device=device),
        replay_ce_veto_indices=torch.empty(0, dtype=torch.int64, device=device),
        replay_veto_directions=torch.empty(0, dtype=torch.int16, device=device),
        replay_veto_thresholds=torch.empty(0, dtype=torch.int32, device=device),
        pc_aux_negative_indices=torch.empty(0, dtype=torch.int64, device=device),
        pc_aux_veto_indices=torch.empty(0, dtype=torch.int64, device=device),
        stats={"event_coded_live_carrier_plan": True},
        event_coded_sparse_active_idx=torch.tensor(active_idx, dtype=torch.int64, device=device),
        event_coded_sparse_post_active_i32=torch.tensor(post_active, dtype=torch.int32, device=device),
    )


def _multi_state_sparse_inputs(device: str) -> list[GlobalRateCapTensorInput]:
    dev = torch.device(device)
    state_a_q = torch.zeros(4, dtype=torch.int8, device=dev)
    state_b_q = torch.zeros(4, dtype=torch.int8, device=dev)
    state_a = VoteUpdateState(
        q_levels=state_a_q,
        accumulators=shape_only_accumulator_stub(state_a_q),
        accumulator_format=VoteUpdateAccumulatorFormat.EVENT_CODED_LIVE_CARRIER,
    )
    state_b = VoteUpdateState(
        q_levels=state_b_q,
        accumulators=shape_only_accumulator_stub(state_b_q),
        accumulator_format=VoteUpdateAccumulatorFormat.EVENT_CODED_LIVE_CARRIER,
    )
    plan_a = _sparse_plan(
        state_a_q,
        applied_indices=[1, 2, 3],
        applied_directions=[1, 1, 1],
        applied_thresholds=[10, 10, 10],
        active_idx=[1, 2, 3],
        post_active=[30, 25, 20],
        device=dev,
    )
    plan_b = _sparse_plan(
        state_b_q,
        applied_indices=[0, 1],
        applied_directions=[1, -1],
        applied_thresholds=[10, 10],
        active_idx=[0, 1],
        post_active=[28, 22],
        device=dev,
    )
    return [
        GlobalRateCapTensorInput(state_key="state_a", state=state_a, plan=plan_a),
        GlobalRateCapTensorInput(state_key="state_b", state=state_b, plan=plan_b),
    ]


def test_fixture_metadata_present() -> None:
    readme = FIXTURE_DIR / "README.md"
    assert readme.is_file()
    assert "nonzero offset" in readme.read_text(encoding="utf-8")


@GPU_SPARSE_CAP_PARITY
def test_sparse_cap_gpu_seam_parity_multi_state_local_global_tensor() -> None:
    spec = _spec()
    cuda_inputs = _multi_state_sparse_inputs("cuda")
    offsets = tensor_offsets_for_vote_update_states(cuda_inputs)
    assert offsets["state_a"] == 0
    assert offsets["state_b"] == 4

    cpu_inputs = _multi_state_sparse_inputs("cpu")
    cpu_result = cpu_sparse_cap_oracle(cpu_inputs, spec)
    gpu_result = apply_sparse_event_coded_cap_via_gpu_seam(
        cap_inputs=cuda_inputs,
        spec=spec,
    )
    witnesses = parity_witness_tensors(cpu_result, gpu_result)

    for state_key, witness in witnesses["per_state"].items():
        assert witness["accepted_local_gpu"] == witness["accepted_local_cpu"]
        assert witness["deferred_local_gpu"] == witness["deferred_local_cpu"]
        assert witness["accepted_global_sha_gpu"] == witness["accepted_global_sha_cpu"]
        assert witness["q_sha_gpu"] == witness["q_sha_cpu"]

    assert gpu_result.accepted_flat_by_key["state_b"] == tuple(
        int(row.flat_index) for row in cpu_result.accepted_rows if row.state_key == "state_b"
    )
    state_b_rows = gpu_result.gpu_apply.selection.rows_by_state["state_b"]
    assert state_b_rows.accepted_global_flat_indices.tolist() != (
        state_b_rows.accepted_indices.tolist()
    )


@GPU_SPARSE_CAP_PARITY
def test_sparse_cap_gpu_seam_uses_local_not_global_for_accepted_flat_by_key() -> None:
    spec = _spec()
    cuda_inputs = _multi_state_sparse_inputs("cuda")
    gpu_result = apply_sparse_event_coded_cap_via_gpu_seam(cap_inputs=cuda_inputs, spec=spec)
    for state_key, local_accepted in gpu_result.accepted_flat_by_key.items():
        state_rows = gpu_result.gpu_apply.selection.rows_by_state[state_key]
        assert local_accepted == tuple(int(x) for x in state_rows.accepted_indices.tolist())
        global_accepted = state_rows.accepted_global_flat_indices.tolist()
        if state_key == "state_b":
            assert global_accepted != list(local_accepted)


def test_sparse_cap_gpu_lane_skips_on_cpu_only_env(monkeypatch) -> None:
    monkeypatch.delenv(RUN_GPU_GLOBAL_RATE_CAP_ENV, raising=False)
    monkeypatch.delenv(RUN_GPU_Q_ACC_APPLY_ENV, raising=False)
    with pytest.raises(RuntimeError, match=RUN_GPU_GLOBAL_RATE_CAP_ENV):
        apply_sparse_event_coded_cap_via_gpu_seam(
            cap_inputs=_multi_state_sparse_inputs("cpu"),
            spec=_spec(),
        )
