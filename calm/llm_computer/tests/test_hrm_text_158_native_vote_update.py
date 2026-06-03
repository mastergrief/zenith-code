"""Phase-1 Slice 2A integer q/vote/update bridge tests.

These tests are CPU/static by default. CUDA preplan tests are default-off and
must only run with HRM_TEXT_158_RUN_GPU_VOTE_UPDATE=1 inside a granted gpu lane.
"""
from __future__ import annotations

import os

import pytest
import torch

from calm.hrm_text_158.native_full_stack.vote_update import (
    DEFERRED_GLOBAL_CAP,
    INT16_ACC_TRANSITIONAL_NOTE,
    INT8_Q_TRANSITIONAL_NOTE,
    RUN_GPU_VOTE_UPDATE_ENV,
    VoteUpdateAccumulatorFormat,
    VoteUpdateInputs,
    VoteUpdateQFormat,
    VoteUpdateSpec,
    VoteUpdateState,
    VoteUpdateVoteFormat,
    apply_integer_vote_update_reference,
    plan_integer_vote_update_reference,
    validate_vote_update_contract,
    vote_update_preplan_triton,
)


GPU_VOTE_UPDATE = pytest.mark.skipif(
    os.environ.get(RUN_GPU_VOTE_UPDATE_ENV) != "1" or not torch.cuda.is_available(),
    reason=(
        "vote-update GPU receipt deferred; set HRM_TEXT_158_RUN_GPU_VOTE_UPDATE=1 "
        "only inside a granted gpu:0 lane"
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


def _state(q: list[int], acc: list[int]) -> VoteUpdateState:
    return VoteUpdateState(
        q_levels=torch.tensor(q, dtype=torch.int8),
        accumulators=torch.tensor(acc, dtype=torch.int16),
    )


def _inputs(votes: list[int], **kwargs) -> VoteUpdateInputs:
    converted = {}
    for name, value in kwargs.items():
        if value is None:
            converted[name] = None
        elif name.endswith("moves"):
            converted[name] = torch.tensor(value, dtype=torch.int8)
        else:
            converted[name] = torch.tensor(value, dtype=torch.int16)
    return VoteUpdateInputs(votes=torch.tensor(votes, dtype=torch.int16), **converted)


def test_basic_reference_exact_int16_update_and_residuals():
    state = _state([0, 1, -1, 0], [8, 20, -20, -8])
    inputs = _inputs([3, 5, -5, -3])

    plan = plan_integer_vote_update_reference(state, inputs, _spec())
    result = apply_integer_vote_update_reference(state, inputs, _spec())

    assert plan.candidate_indices.tolist() == [0, 3]
    assert plan.pre_veto_selected_indices.tolist() == [0, 3]
    assert plan.applied_indices.tolist() == [0, 3]
    assert result.q_levels.tolist() == [1, 1, -1, -1]
    assert result.accumulators.tolist() == [1, 25, -25, -1]
    assert result.stats["flip_count"] == 2
    assert result.stats["global_cap_policy"] == DEFERRED_GLOBAL_CAP


def test_tie_break_is_largest_abs_acc_then_lower_flat_index():
    state = _state([0, 0, 0, 0], [12, 12, -13, 13])
    inputs = _inputs([0, 0, 0, 0])

    plan = plan_integer_vote_update_reference(
        state,
        inputs,
        _spec(max_abs_per_tensor=2, fraction_per_tensor=1.0),
    )

    assert plan.candidate_indices.tolist() == [0, 1, 2, 3]
    assert plan.pre_veto_selected_indices.tolist() == [2, 3]
    assert plan.applied_indices.tolist() == [2, 3]
    assert plan.applied_directions.tolist() == [-1, 1]


def test_fractional_flip_budget_limits_local_tensor_selection():
    state = _state([0, 0, 0, 0], [12, 11, 10, -12])
    inputs = _inputs([0, 0, 0, 0])

    plan = plan_integer_vote_update_reference(
        state,
        inputs,
        _spec(max_abs_per_tensor=10, fraction_per_tensor=0.25),
    )

    assert plan.stats["max_flips"] == 1
    assert plan.pre_veto_selected_indices.tolist() == [0]
    assert plan.applied_indices.tolist() == [0]


def test_replay_veto_subtracts_threshold_and_does_not_mutate_q():
    state = _state([0], [0])
    inputs = _inputs(
        [12],
        replay_ce_veto_votes=[-1],
        replay_ce_veto_moves=[0],
    )

    plan = plan_integer_vote_update_reference(state, inputs, _spec())
    result = apply_integer_vote_update_reference(state, inputs, _spec())

    assert plan.pre_veto_selected_indices.tolist() == [0]
    assert plan.replay_ce_veto_indices.tolist() == [0]
    assert plan.applied_indices.tolist() == []
    assert result.q_levels.tolist() == [0]
    assert result.accumulators.tolist() == [2]
    assert result.stats["replay_ce_veto_count"] == 1
    assert result.stats["vetoed_accumulator_residual_policy"] == (
        "subtract_threshold_then_clamp_without_q_mutation"
    )


def test_pc_aux_negative_is_metadata_not_apply_veto():
    state = _state([0], [0])
    inputs = _inputs(
        [12],
        pc_aux_votes=[-1],
        pc_aux_moves=[0],
    )

    plan = plan_integer_vote_update_reference(state, inputs, _spec())
    result = apply_integer_vote_update_reference(state, inputs, _spec())

    assert plan.pc_aux_negative_indices.tolist() == [0]
    assert plan.applied_indices.tolist() == [0]
    assert result.q_levels.tolist() == [1]
    assert result.accumulators.tolist() == [2]


def test_truncating_decay_and_clip_edges_are_exact():
    state = _state([0, 0, 0], [-9, 9, 0])
    inputs = _inputs([0, 0, 200])
    spec = _spec(
        threshold_abs=20,
        decay_numerator=1,
        decay_denominator=2,
        accumulator_clip_min=-5,
        accumulator_clip_max=5,
    )

    plan = plan_integer_vote_update_reference(state, inputs, spec)

    # torch.div(..., rounding_mode="trunc"): -9/2 -> -4, +9/2 -> +4.
    assert plan.new_acc_i32.tolist() == [-4, 4, 5]
    assert plan.candidate_indices.tolist() == []


def test_threshold_jitter_is_explicitly_rejected_not_silently_omitted():
    with pytest.raises(NotImplementedError, match="threshold_jitter"):
        plan_integer_vote_update_reference(
            _state([0], [0]),
            _inputs([12]),
            _spec(threshold_jitter_enabled=True),
        )


def test_global_cap_is_explicitly_deferred_not_silently_implemented():
    with pytest.raises(NotImplementedError, match="global multi-tensor cap"):
        plan_integer_vote_update_reference(
            _state([0], [0]),
            _inputs([12]),
            _spec(global_cap_policy="implemented_global_cap"),
        )


def test_rejects_fp_master_and_moment_like_tensors():
    fp_state = VoteUpdateState(
        q_levels=torch.randn(2, dtype=torch.float32),
        accumulators=torch.zeros(2, dtype=torch.int16),
    )
    with pytest.raises(ValueError, match="FP master"):
        validate_vote_update_contract(fp_state, _inputs([0, 0]), _spec())

    fp_acc_state = VoteUpdateState(
        q_levels=torch.zeros(2, dtype=torch.int8),
        accumulators=torch.zeros(2, dtype=torch.float32),
    )
    with pytest.raises(ValueError, match="optimizer/moment"):
        validate_vote_update_contract(fp_acc_state, _inputs([0, 0]), _spec())


def test_pack_ready_future_formats_are_named_not_implemented():
    assert "transitional" in INT8_Q_TRANSITIONAL_NOTE
    assert "transitional" in INT16_ACC_TRANSITIONAL_NOTE

    with pytest.raises(NotImplementedError, match="future packed q"):
        validate_vote_update_contract(
            VoteUpdateState(
                q_levels=torch.zeros(2, dtype=torch.int8),
                accumulators=torch.zeros(2, dtype=torch.int16),
                q_format=VoteUpdateQFormat.PACKED_TERNARY,
            ),
            _inputs([0, 0]),
            _spec(),
        )
    with pytest.raises(NotImplementedError, match="compressed accumulator"):
        validate_vote_update_contract(
            VoteUpdateState(
                q_levels=torch.zeros(2, dtype=torch.int8),
                accumulators=torch.zeros(2, dtype=torch.int16),
                accumulator_format=VoteUpdateAccumulatorFormat.COMPRESSED_ACCUMULATORS,
            ),
            _inputs([0, 0]),
            _spec(),
        )
    with pytest.raises(NotImplementedError, match="compressed vote"):
        validate_vote_update_contract(
            _state([0, 0], [0, 0]),
            VoteUpdateInputs(
                votes=torch.zeros(2, dtype=torch.int16),
                vote_format=VoteUpdateVoteFormat.COMPRESSED_VOTES,
            ),
            _spec(),
        )


def test_triton_preplan_default_off_before_any_gpu_path(monkeypatch):
    monkeypatch.delenv(RUN_GPU_VOTE_UPDATE_ENV, raising=False)

    with pytest.raises(RuntimeError, match=RUN_GPU_VOTE_UPDATE_ENV):
        vote_update_preplan_triton(_state([0], [0]), _inputs([12]), _spec())


@GPU_VOTE_UPDATE
def test_triton_preplan_deferred_gpu_smoke():
    state = VoteUpdateState(
        q_levels=torch.tensor([0, 1, -1], dtype=torch.int8, device="cuda"),
        accumulators=torch.tensor([8, 20, -20], dtype=torch.int16, device="cuda"),
    )
    inputs = VoteUpdateInputs(votes=torch.tensor([3, 5, -5], dtype=torch.int16, device="cuda"))

    out = vote_update_preplan_triton(state, inputs, _spec())

    assert out["scope"] == "elementwise_preplan_only_deferred_global_cap"
    assert out["new_accumulators"].detach().cpu().tolist() == [11, 25, -25]
    assert out["candidate_mask_int8"].detach().cpu().tolist() == [1, 0, 0]

