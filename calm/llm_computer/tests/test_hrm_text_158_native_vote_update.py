"""Phase-1 Slice 2A integer q/vote/update bridge tests.

These tests are CPU/static by default. CUDA preplan tests are default-off and
must only run with HRM_TEXT_158_RUN_GPU_VOTE_UPDATE=1 inside a granted gpu lane.
"""
from __future__ import annotations

from dataclasses import replace
import inspect
from unittest.mock import patch

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
    apply_integer_vote_update_from_frozen_plan,
    apply_integer_vote_update_reference,
    plan_integer_vote_update_reference,
    validate_vote_update_contract,
    vote_update_preplan_triton,
)
import calm.hrm_text_158.native_full_stack.vote_update as vote_update_mod


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
        elif name == "pc_aux_mode":
            converted[name] = value
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
    assert result.stats["pc_aux_mode"] == "telemetry"
    assert result.stats["pc_aux_veto_count"] == 0


def test_pc_aux_veto_mode_masks_selected_flip_without_replay_threshold_consumption():
    state = _state([0], [0])
    inputs = _inputs(
        [12],
        pc_aux_votes=[-1],
        pc_aux_moves=[0],
        pc_aux_mode="veto",
    )

    plan = plan_integer_vote_update_reference(state, inputs, _spec())
    result = apply_integer_vote_update_reference(state, inputs, _spec())

    assert plan.pc_aux_negative_indices.tolist() == [0]
    assert plan.pc_aux_veto_indices.tolist() == [0]
    assert plan.applied_indices.tolist() == []
    assert result.q_levels.tolist() == [0]
    assert result.accumulators.tolist() == [12]
    assert result.stats["pc_aux_mode"] == "veto"
    assert result.stats["pc_aux_veto_count"] == 1
    assert result.stats["pc_aux_veto_accumulator_residual_policy"] == (
        "q_mutation_veto_only_accumulator_retained"
    )


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


def _dense_candidate_mask_from_plan(plan, shape: torch.Size, *, device: torch.device) -> torch.Tensor:
    mask = torch.zeros(int(torch.tensor(shape).prod().item()), dtype=torch.int8, device=device)
    if plan.candidate_indices.numel() > 0:
        mask[plan.candidate_indices.to(device)] = 1
    return mask.view(shape)


def _torch_gpu_preplan_baseline(
    state: VoteUpdateState,
    inputs: VoteUpdateInputs,
    spec: VoteUpdateSpec,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    spec.validate()
    q = state.q_levels
    acc_i32 = state.accumulators.to(torch.int32)
    vote_i32 = inputs.votes.to(torch.int32)
    prod = acc_i32 * int(spec.decay_numerator)
    decayed = torch.div(prod, int(spec.decay_denominator), rounding_mode="trunc")
    new_acc_i32 = (decayed + vote_i32).clamp(
        int(spec.accumulator_clip_min),
        int(spec.accumulator_clip_max),
    )
    pos_candidate = (new_acc_i32 >= int(spec.threshold_abs)) & (q.to(torch.int16) < 1)
    neg_candidate = (new_acc_i32 <= -int(spec.threshold_abs)) & (q.to(torch.int16) > -1)
    candidate = (pos_candidate | neg_candidate).to(torch.int8)
    direction = torch.where(
        new_acc_i32 >= int(spec.threshold_abs),
        torch.ones_like(new_acc_i32, dtype=torch.int8),
        -torch.ones_like(new_acc_i32, dtype=torch.int8),
    )
    return new_acc_i32.to(torch.int16), candidate, direction


def _assert_triton_preplan_matches_reference(
    state: VoteUpdateState,
    inputs: VoteUpdateInputs,
    spec: VoteUpdateSpec,
    *,
    block: int = 1024,
) -> dict[str, torch.Tensor | str]:
    assert -(2**15) <= spec.accumulator_clip_min <= spec.accumulator_clip_max < 2**15
    out = vote_update_preplan_triton(state, inputs, spec, block=block)
    plan = plan_integer_vote_update_reference(state, inputs, spec)
    expected_candidate = _dense_candidate_mask_from_plan(
        plan,
        state.q_levels.shape,
        device=state.q_levels.device,
    )
    expected_direction = torch.where(
        plan.new_acc_i32 >= int(spec.threshold_abs),
        torch.ones_like(plan.new_acc_i32, dtype=torch.int8),
        -torch.ones_like(plan.new_acc_i32, dtype=torch.int8),
    )

    assert out["scope"] == "elementwise_preplan_only_deferred_global_cap"
    assert torch.equal(out["new_accumulators"], plan.new_acc_i32.to(torch.int16))
    assert torch.equal(out["candidate_mask_int8"], expected_candidate)

    candidate_positions = expected_candidate.to(torch.bool)
    if bool(candidate_positions.any().item()):
        assert torch.equal(
            out["direction_int8"][candidate_positions],
            expected_direction[candidate_positions],
        )
    return out


@GPU_VOTE_UPDATE
def test_triton_preplan_gpu_matches_reference_for_threshold_saturation_and_clip_cases():
    state = VoteUpdateState(
        q_levels=torch.tensor([0, 0, 1, -1, 1, -1, 0, 0], dtype=torch.int8, device="cuda"),
        accumulators=torch.tensor([8, -8, 8, -8, 20, -20, 5, -5], dtype=torch.int16, device="cuda"),
    )
    inputs = VoteUpdateInputs(
        votes=torch.tensor([3, -3, 5, -5, 5, -5, 200, -200], dtype=torch.int16, device="cuda")
    )

    out = _assert_triton_preplan_matches_reference(state, inputs, _spec())

    assert out["new_accumulators"].detach().cpu().tolist() == [11, -11, 13, -13, 25, -25, 127, -127]
    assert out["candidate_mask_int8"].detach().cpu().tolist() == [1, 1, 0, 0, 0, 0, 1, 1]
    assert out["direction_int8"][out["candidate_mask_int8"].to(torch.bool)].detach().cpu().tolist() == [1, -1, 1, -1]


@GPU_VOTE_UPDATE
def test_triton_preplan_gpu_matches_reference_for_decay_truncation_case():
    state = VoteUpdateState(
        q_levels=torch.tensor([0, 0, 0], dtype=torch.int8, device="cuda"),
        accumulators=torch.tensor([-9, 9, 0], dtype=torch.int16, device="cuda"),
    )
    inputs = VoteUpdateInputs(votes=torch.tensor([0, 0, 200], dtype=torch.int16, device="cuda"))
    spec = _spec(
        threshold_abs=4,
        decay_numerator=1,
        decay_denominator=2,
        accumulator_clip_min=-5,
        accumulator_clip_max=5,
    )

    out = _assert_triton_preplan_matches_reference(state, inputs, spec)

    assert out["new_accumulators"].detach().cpu().tolist() == [-4, 4, 5]
    assert out["candidate_mask_int8"].detach().cpu().tolist() == [1, 1, 1]


@GPU_VOTE_UPDATE
def test_triton_preplan_gpu_matches_reference_across_multiple_blocks():
    numel = 2053
    idx = torch.arange(numel, device="cuda")
    q = torch.zeros(numel, dtype=torch.int8, device="cuda")
    q[idx % 17 == 0] = 1
    q[idx % 19 == 0] = -1
    acc = ((idx % 31) - 15).to(torch.int16)
    votes = torch.where(
        idx % 5 == 0,
        torch.full_like(idx, 20),
        torch.where(idx % 5 == 1, torch.full_like(idx, -20), torch.zeros_like(idx)),
    ).to(torch.int16)
    state = VoteUpdateState(q_levels=q, accumulators=acc)
    inputs = VoteUpdateInputs(votes=votes)

    _assert_triton_preplan_matches_reference(state, inputs, _spec(), block=128)


@GPU_VOTE_UPDATE
def test_triton_preplan_perf_and_memory_receipt_scaffold():
    numel = 262_144
    idx = torch.arange(numel, device="cuda")
    q = torch.zeros(numel, dtype=torch.int8, device="cuda")
    q[idx % 23 == 0] = 1
    q[idx % 29 == 0] = -1
    acc = ((idx % 255) - 127).to(torch.int16)
    votes = torch.where(
        idx % 7 == 0,
        torch.full_like(idx, 13),
        torch.where(idx % 7 == 1, torch.full_like(idx, -13), torch.zeros_like(idx)),
    ).to(torch.int16)
    state = VoteUpdateState(q_levels=q, accumulators=acc)
    inputs = VoteUpdateInputs(votes=votes)
    spec = _spec(threshold_abs=10, accumulator_clip_min=-127, accumulator_clip_max=127)

    torch.cuda.reset_peak_memory_stats()
    vote_update_preplan_triton(state, inputs, spec)
    _torch_gpu_preplan_baseline(state, inputs, spec)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(20):
        triton_out = vote_update_preplan_triton(state, inputs, spec)
    end.record()
    torch.cuda.synchronize()
    triton_ms = start.elapsed_time(end) / 20.0

    start.record()
    for _ in range(20):
        baseline_new_acc, baseline_candidate, baseline_direction = _torch_gpu_preplan_baseline(
            state,
            inputs,
            spec,
        )
    end.record()
    torch.cuda.synchronize()
    baseline_ms = start.elapsed_time(end) / 20.0

    assert torch.equal(triton_out["new_accumulators"], baseline_new_acc)
    assert torch.equal(triton_out["candidate_mask_int8"], baseline_candidate)
    candidate_positions = baseline_candidate.to(torch.bool)
    if bool(candidate_positions.any().item()):
        assert torch.equal(
            triton_out["direction_int8"][candidate_positions],
            baseline_direction[candidate_positions],
        )

    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    assert triton_ms > 0.0
    assert baseline_ms > 0.0
    assert peak_allocated > 0
    assert peak_reserved >= peak_allocated
    print(
        "vote_update_preplan_gpu_receipt_scaffold "
        f"numel={numel} triton_ms={triton_ms:.4f} torch_baseline_ms={baseline_ms:.4f} "
        f"peak_allocated={peak_allocated} peak_reserved={peak_reserved} "
        "scope=elementwise_preplan_only_deferred_global_cap"
    )


def test_from_frozen_plan_bitexact_matches_apply_reference_basic():
    state, inputs = _state([0, 1, -1, 0], [8, 20, -20, -8]), _inputs([3, 5, -5, -3])
    plan = plan_integer_vote_update_reference(state, inputs, _spec())
    ref = apply_integer_vote_update_reference(state, inputs, _spec())
    frozen = apply_integer_vote_update_from_frozen_plan(state, plan)
    assert torch.equal(frozen.q_levels, ref.q_levels) and torch.equal(frozen.accumulators, ref.accumulators)
    assert frozen.stats["flip_count"] == ref.stats["flip_count"]


def test_from_frozen_plan_bitexact_empty_applied():
    state, inputs = _state([0, 0, 0], [-9, 9, 0]), _inputs([0, 0, 200])
    spec = _spec(threshold_abs=20, decay_numerator=1, decay_denominator=2,
                 accumulator_clip_min=-5, accumulator_clip_max=5)
    plan = plan_integer_vote_update_reference(state, inputs, spec)
    assert plan.applied_indices.numel() == 0
    ref = apply_integer_vote_update_reference(state, inputs, spec)
    frozen = apply_integer_vote_update_from_frozen_plan(state, plan)
    assert torch.equal(frozen.q_levels, ref.q_levels) and torch.equal(frozen.accumulators, ref.accumulators)


def test_from_frozen_plan_bitexact_with_replay_veto():
    state, inputs = _state([0], [0]), _inputs([12], replay_ce_veto_votes=[-1], replay_ce_veto_moves=[0])
    plan = plan_integer_vote_update_reference(state, inputs, _spec())
    assert plan.replay_ce_veto_indices.tolist() == [0] and plan.applied_indices.numel() == 0
    ref = apply_integer_vote_update_reference(state, inputs, _spec())
    frozen = apply_integer_vote_update_from_frozen_plan(state, plan)
    assert torch.equal(frozen.q_levels, ref.q_levels) and torch.equal(frozen.accumulators, ref.accumulators)


def test_from_frozen_plan_bitexact_boundary_and_ordering_cases():
    state, inputs = _state([0, 0, 0, 0], [12, 12, -13, 13]), _inputs([0, 0, 0, 0])
    spec = _spec(max_abs_per_tensor=2, fraction_per_tensor=1.0)
    plan = plan_integer_vote_update_reference(state, inputs, spec)
    ref = apply_integer_vote_update_reference(state, inputs, spec)
    frozen = apply_integer_vote_update_from_frozen_plan(state, plan)
    assert torch.equal(frozen.q_levels, ref.q_levels) and torch.equal(frozen.accumulators, ref.accumulators)
    assert plan.applied_indices.tolist() == [2, 3]


def test_from_frozen_plan_does_not_call_planner():
    state = _state([0, 1, -1, 0], [8, 20, -20, -8])
    plan = plan_integer_vote_update_reference(state, _inputs([3, 5, -5, -3]), _spec())
    with patch.object(vote_update_mod, "plan_integer_vote_update_reference",
                      side_effect=AssertionError("planner must not be called")):
        apply_integer_vote_update_from_frozen_plan(state, plan)


def test_modified_directions_do_not_reselect_candidates():
    state = _state([0, 1, -1, 0], [8, 20, -20, -8])
    plan = plan_integer_vote_update_reference(state, _inputs([3, 5, -5, -3]), _spec())
    assert plan.applied_indices.numel() > 0
    negated = replace(plan, applied_directions=(-plan.applied_directions).to(plan.applied_directions.dtype))
    with patch.object(vote_update_mod, "plan_integer_vote_update_reference",
                      side_effect=AssertionError("no replan")):
        out = apply_integer_vote_update_from_frozen_plan(state, negated)
    assert out.plan.applied_indices.tolist() == plan.applied_indices.tolist()
    assert out.plan.applied_directions.tolist() == negated.applied_directions.tolist()


def test_public_api_rejects_validation_bypass_kwargs():
    sig = inspect.signature(apply_integer_vote_update_from_frozen_plan)
    assert "validate_q_levels" not in sig.parameters and "validate_plan_binding" not in sig.parameters
    state, plan = _state([0], [0]), plan_integer_vote_update_reference(_state([0], [0]), _inputs([0]), _spec())
    with pytest.raises(TypeError):
        apply_integer_vote_update_from_frozen_plan(state, plan, validate_q_levels=False)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        apply_integer_vote_update_from_frozen_plan(state, plan, validate_plan_binding=False)  # type: ignore[call-arg]


def test_apply_reference_plans_once_then_private_core_only():
    state = _state([0, 1, -1, 0], [8, 20, -20, -8])
    counts = {"planner": 0, "private": 0, "public": 0, "validator": 0}
    real = {
        "planner": vote_update_mod.plan_integer_vote_update_reference,
        "private": vote_update_mod._apply_integer_vote_update_from_frozen_plan_trusted,
        "public": vote_update_mod.apply_integer_vote_update_from_frozen_plan,
        "validator": vote_update_mod._validate_frozen_plan_binding,
    }
    def _wrap(key):
        def _fn(*a, **k):
            counts[key] += 1
            return real[key](*a, **k)
        return _fn
    with (
        patch.object(vote_update_mod, "plan_integer_vote_update_reference", side_effect=_wrap("planner")),
        patch.object(vote_update_mod, "_apply_integer_vote_update_from_frozen_plan_trusted", side_effect=_wrap("private")),
        patch.object(vote_update_mod, "apply_integer_vote_update_from_frozen_plan", side_effect=_wrap("public")),
        patch.object(vote_update_mod, "_validate_frozen_plan_binding", side_effect=_wrap("validator")),
    ):
        apply_integer_vote_update_reference(state, _inputs([3, 5, -5, -3]), _spec())
    assert counts == {"planner": 1, "private": 1, "public": 0, "validator": 0}


def test_malformed_plan_q_binding_mismatch_rejected():
    state = _state([0, 0], [0, 0])
    plan = plan_integer_vote_update_reference(state, _inputs([0, 0]), _spec())
    bad_q = plan.q_i16.clone(); bad_q.view(-1)[0] = 1
    with pytest.raises(ValueError, match="q base binding"):
        apply_integer_vote_update_from_frozen_plan(state, replace(plan, q_i16=bad_q))


def test_malformed_plan_rank_not_1d_rejected():
    state = _state([0, 0], [0, 0])
    plan = plan_integer_vote_update_reference(state, _inputs([12, 0]), _spec())
    assert plan.applied_indices.numel() >= 1
    with pytest.raises(ValueError, match="ndim==1"):
        apply_integer_vote_update_from_frozen_plan(
            state, replace(plan, applied_indices=plan.applied_indices.view(1, -1)))


def test_malformed_plan_duplicate_or_oor_indices_rejected():
    state = _state([0, 0, 0], [0, 0, 0])
    plan = plan_integer_vote_update_reference(state, _inputs([12, 12, 0]), _spec())
    if plan.applied_indices.numel() < 1:
        plan = replace(plan, applied_indices=torch.tensor([0], dtype=torch.int64),
                       applied_directions=torch.tensor([1], dtype=torch.int16),
                       applied_thresholds=torch.tensor([10], dtype=torch.int32))
    with pytest.raises(ValueError, match="duplicate"):
        apply_integer_vote_update_from_frozen_plan(state, replace(
            plan, applied_indices=torch.tensor([0, 0], dtype=torch.int64),
            applied_directions=torch.tensor([1, 1], dtype=torch.int16),
            applied_thresholds=torch.tensor([10, 10], dtype=torch.int32)))
    with pytest.raises(ValueError, match="out-of-range"):
        apply_integer_vote_update_from_frozen_plan(state, replace(
            plan, applied_indices=torch.tensor([99], dtype=torch.int64),
            applied_directions=torch.tensor([1], dtype=torch.int16),
            applied_thresholds=torch.tensor([10], dtype=torch.int32)))


def test_malformed_plan_non_pm1_directions_or_nonpositive_thresholds_rejected():
    state = _state([0, 0], [0, 0])
    plan = plan_integer_vote_update_reference(state, _inputs([12, 0]), _spec())
    assert plan.applied_indices.numel() >= 1
    with pytest.raises(ValueError, match="±1|exactly"):
        apply_integer_vote_update_from_frozen_plan(
            state, replace(plan, applied_directions=torch.zeros_like(plan.applied_directions)))
    with pytest.raises(ValueError, match="positive"):
        apply_integer_vote_update_from_frozen_plan(
            state, replace(plan, applied_thresholds=torch.zeros_like(plan.applied_thresholds)))


def test_malformed_plan_replay_length_or_shape_mismatch_rejected():
    state = _state([0], [0])
    plan = plan_integer_vote_update_reference(
        state, _inputs([12], replay_ce_veto_votes=[-1], replay_ce_veto_moves=[0]), _spec())
    with pytest.raises(ValueError, match="replay_veto_directions length"):
        apply_integer_vote_update_from_frozen_plan(
            state, replace(plan, replay_veto_directions=torch.tensor([], dtype=torch.int16)))


@pytest.mark.parametrize("field,dtype", [
    ("applied_directions", torch.float32), ("applied_directions", torch.bool),
    ("applied_thresholds", torch.float32), ("applied_thresholds", torch.bool),
    ("replay_ce_veto_indices", torch.float32),
    ("replay_veto_directions", torch.float32), ("replay_veto_directions", torch.bool),
    ("replay_veto_thresholds", torch.float32), ("replay_veto_thresholds", torch.bool),
])
def test_empty_wrong_dtype_plan_arrays_rejected(field, dtype):
    state = _state([0, 0], [0, 0])
    plan = plan_integer_vote_update_reference(state, _inputs([0, 0]), _spec())
    empty = dict(applied_indices=torch.tensor([], dtype=torch.int64),
                 applied_directions=torch.tensor([], dtype=torch.int16),
                 applied_thresholds=torch.tensor([], dtype=torch.int32),
                 replay_ce_veto_indices=torch.tensor([], dtype=torch.int64),
                 replay_veto_directions=torch.tensor([], dtype=torch.int16),
                 replay_veto_thresholds=torch.tensor([], dtype=torch.int32))
    base = replace(plan, **empty)
    with pytest.raises(ValueError, match="int32/int64|explicit integer dtype"):
        apply_integer_vote_update_from_frozen_plan(state, replace(base, **{field: torch.tensor([], dtype=dtype)}))


def test_valid_typed_empty_applied_and_replay_parity():
    state, inputs = _state([0, 0, 0], [-9, 9, 0]), _inputs([0, 0, 200])
    spec = _spec(threshold_abs=20, decay_numerator=1, decay_denominator=2,
                 accumulator_clip_min=-5, accumulator_clip_max=5)
    plan = plan_integer_vote_update_reference(state, inputs, spec)
    assert plan.applied_indices.numel() == 0 and plan.applied_directions.dtype == torch.int16
    assert plan.applied_thresholds.dtype == torch.int32
    assert plan.replay_ce_veto_indices.dtype in (torch.int32, torch.int64)
    ref = apply_integer_vote_update_reference(state, inputs, spec)
    frozen = apply_integer_vote_update_from_frozen_plan(state, plan)
    assert torch.equal(frozen.q_levels, ref.q_levels) and torch.equal(frozen.accumulators, ref.accumulators)
