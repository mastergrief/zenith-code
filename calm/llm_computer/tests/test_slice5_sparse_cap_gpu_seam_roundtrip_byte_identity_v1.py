"""GPU seam vs CPU fallback serialize/resume byte-identity (Slice A-prime A3)."""
from __future__ import annotations

import copy
import json
import os

import pytest
import torch

from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    apply_bounded_delta_vote_step,
    build_authoritative_checkpoint_payload,
    credit_from_weighted_grad,
    default_dry_run_rank_vote_spec,
    make_event_coded_live_tensor_state,
    project_s1_gradient_to_moves,
    sparse_rank_bucketed_int16_vote_events,
)
from calm.hrm_text_158.native_full_stack.event_coded_vote_update_adapter import (
    C8_TRANSIENT_DENSE_COMPUTE_NUMEL_KEY,
    carrier_content_sha256,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap import GlobalRateCapSpec
from calm.hrm_text_158.native_full_stack.global_rate_cap_gpu import RUN_GPU_GLOBAL_RATE_CAP_ENV
from calm.hrm_text_158.native_full_stack.vote_update import (
    RUN_GPU_Q_ACC_APPLY_ENV,
    VoteUpdateSpec,
)

GPU_ROUNDTRIP = pytest.mark.skipif(
    os.environ.get(RUN_GPU_GLOBAL_RATE_CAP_ENV) != "1"
    or os.environ.get(RUN_GPU_Q_ACC_APPLY_ENV) != "1"
    or not torch.cuda.is_available(),
    reason="sparse cap GPU seam round-trip requires CUDA lane env gates",
)


def _canonical_checkpoint_bytes(tensor_states: dict) -> bytes:
    payload = build_authoritative_checkpoint_payload(
        tensor_states,
        step=1,
        updater_config={"test": True},
    )
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _fixture():
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
    assert all(state.q_levels.device.type == "cpu" for state in states.values())
    return states, {"mod.a": sparse_a, "mod.b": sparse_b}, {"mod.a": spec, "mod.b": spec}, cap


@GPU_ROUNDTRIP
def test_sparse_cap_gpu_seam_vs_cpu_fallback_checkpoint_byte_identity(monkeypatch) -> None:
    states, sparse_by_key, vote_specs, cap = _fixture()
    common = dict(
        votes_by_key=None,
        vote_specs_by_key=vote_specs,
        candidate_sparse_vote_events_by_key=sparse_by_key,
        global_cap_spec=cap,
        event_coded_sparse_vote_authority=True,
        local_selection_ordering_step=1,
    )

    monkeypatch.delenv(RUN_GPU_GLOBAL_RATE_CAP_ENV, raising=False)
    monkeypatch.delenv(RUN_GPU_Q_ACC_APPLY_ENV, raising=False)
    cpu_result = apply_bounded_delta_vote_step(copy.deepcopy(states), **common)

    monkeypatch.setenv(RUN_GPU_GLOBAL_RATE_CAP_ENV, "1")
    monkeypatch.setenv(RUN_GPU_Q_ACC_APPLY_ENV, "1")
    gpu_result = apply_bounded_delta_vote_step(copy.deepcopy(states), **common)

    assert gpu_result.global_summary.get("sparse_cap_submilestone_cap_selection_path") == "gpu_seam"
    assert cpu_result.global_summary.get("sparse_cap_submilestone_cap_selection_path") == "cpu_reference"
    assert int(gpu_result.global_summary.get(C8_TRANSIENT_DENSE_COMPUTE_NUMEL_KEY, -1)) == 0
    assert int(cpu_result.global_summary.get(C8_TRANSIENT_DENSE_COMPUTE_NUMEL_KEY, -1)) == 0
    assert gpu_result.global_summary.get("transient_q_mirror_for_gpu_cap") is True
    assert gpu_result.global_summary.get("persistent_q_authority_device") == "cpu"
    assert gpu_result.global_summary.get("cuda_q_not_saved_state") is True

    for key, state in gpu_result.tensor_states.items():
        assert state.q_levels.device.type == "cpu"
        carrier = state.event_coded_live_carrier
        assert carrier is not None
        cpu_carrier = cpu_result.tensor_states[key].event_coded_live_carrier
        assert carrier_content_sha256(carrier) == carrier_content_sha256(cpu_carrier)

    gpu_bytes = _canonical_checkpoint_bytes(gpu_result.tensor_states)
    cpu_bytes = _canonical_checkpoint_bytes(cpu_result.tensor_states)
    assert gpu_bytes == cpu_bytes
