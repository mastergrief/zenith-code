"""Sparse event-coded planner v1 parity + regression tests."""
from __future__ import annotations

import json

import pytest
import torch

from calm.hrm_text_158.native_full_stack.bounded_delta_learner import apply_bounded_delta_vote_step
from calm.hrm_text_158.native_full_stack.event_coded_acc_live_carrier import EventCodedAccLiveState
from calm.hrm_text_158.native_full_stack.event_coded_vote_update_adapter import (
    C8_TRANSIENT_DENSE_COMPUTE_NUMEL_KEY,
    EVENT_CODED_CAP_BOUNDARY_DENSIFIED_KEY,
    EVENT_CODED_PLANNER_TRANSIENT_DENSE_NUMEL_KEY,
    EventCodedVoteUpdateState,
    apply_event_coded_integer_vote_update_from_plan,
    apply_event_coded_integer_vote_update_reference,
    carrier_content_sha256,
    densify_new_acc_i32_at_cap_boundary,
    plan_event_coded_integer_vote_update,
    plan_event_coded_integer_vote_update_dense_oracle,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap import GlobalRateCapSpec
from calm.hrm_text_158.native_full_stack.vote_update import VoteUpdateInputs, VoteUpdateSpec


def _vote_spec(*, threshold_abs: int = 8) -> VoteUpdateSpec:
    return VoteUpdateSpec(
        threshold_abs=int(threshold_abs),
        accumulator_clip_min=-127,
        accumulator_clip_max=127,
        max_abs_per_tensor=16,
    )


def _make_state(*, numel: int = 16, demotion_band: int = 1) -> EventCodedVoteUpdateState:
    q = torch.zeros(numel, dtype=torch.int8)
    carrier = EventCodedAccLiveState(logical_numel=int(numel), demotion_band=int(demotion_band))
    return EventCodedVoteUpdateState(q_levels=q, carrier=carrier)


def _votes_for_indices(indices: dict[int, int], *, numel: int = 16) -> torch.Tensor:
    votes = torch.zeros(numel, dtype=torch.int16)
    for flat_index, magnitude in indices.items():
        votes[int(flat_index)] = int(magnitude)
    return votes


def _plan_indices(plan) -> dict[str, tuple[int, ...]]:
    return {
        "candidate": tuple(int(x) for x in plan.candidate_indices.detach().cpu().tolist()),
        "applied": tuple(int(x) for x in plan.applied_indices.detach().cpu().tolist()),
    }


def test_sparse_planner_matches_dense_oracle_flip_sets() -> None:
    state = _make_state(numel=64)
    votes = _votes_for_indices({0: 12, 7: -9, 31: 6}, numel=64)
    inputs = VoteUpdateInputs(votes=votes)
    spec = _vote_spec()
    sparse = plan_event_coded_integer_vote_update(state, inputs, spec)
    dense = plan_event_coded_integer_vote_update_dense_oracle(state, inputs, spec)
    assert _plan_indices(sparse) == _plan_indices(dense)


def test_cap_on_sparse_path_matches_dense_oracle() -> None:
    state = _make_state(numel=16)
    votes = _votes_for_indices({0: 12, 1: 12, 2: 12})
    inputs = VoteUpdateInputs(votes=votes)
    spec = _vote_spec(threshold_abs=10)
    sparse = plan_event_coded_integer_vote_update(state, inputs, spec)
    dense = plan_event_coded_integer_vote_update_dense_oracle(state, inputs, spec)
    sparse_dense = densify_new_acc_i32_at_cap_boundary(sparse, state.q_levels)
    dense_dense = densify_new_acc_i32_at_cap_boundary(dense, state.q_levels)
    assert _plan_indices(sparse_dense) == _plan_indices(dense_dense)
    assert bool(sparse_dense.stats[EVENT_CODED_CAP_BOUNDARY_DENSIFIED_KEY]) is True
    assert int(sparse_dense.stats[C8_TRANSIENT_DENSE_COMPUTE_NUMEL_KEY]) == 16


def test_sparse_planner_transient_dense_compute_numel_zero() -> None:
    state = _make_state()
    plan = plan_event_coded_integer_vote_update(
        state,
        VoteUpdateInputs(votes=_votes_for_indices({0: 8})),
        _vote_spec(),
    )
    assert int(plan.stats[EVENT_CODED_PLANNER_TRANSIENT_DENSE_NUMEL_KEY]) == 0
    result = apply_event_coded_integer_vote_update_from_plan(
        state,
        VoteUpdateInputs(votes=_votes_for_indices({0: 8})),
        _vote_spec(),
        plan,
    )
    assert int(result.stats[EVENT_CODED_PLANNER_TRANSIENT_DENSE_NUMEL_KEY]) == 0
    assert int(result.stats[C8_TRANSIENT_DENSE_COMPUTE_NUMEL_KEY]) == 0


def test_cap_boundary_is_only_dense_alloc_site(monkeypatch: pytest.MonkeyPatch) -> None:
    zeros_calls: list[int] = []
    original_zeros = torch.zeros

    def _track_zeros(size, *args, **kwargs):
        if isinstance(size, int):
            zeros_calls.append(int(size))
        elif hasattr(size, "numel"):
            zeros_calls.append(int(size.numel()))
        return original_zeros(size, *args, **kwargs)

    monkeypatch.setattr(torch, "zeros", _track_zeros)
    state = _make_state(numel=32)
    votes = VoteUpdateInputs(votes=_votes_for_indices({1: 10}, numel=32))
    spec = _vote_spec()
    zeros_calls.clear()
    sparse = plan_event_coded_integer_vote_update(state, votes, spec)
    assert zeros_calls == [1, 1]
    densify_new_acc_i32_at_cap_boundary(sparse, state.q_levels)
    assert zeros_calls == [1, 1, 32]


def test_apply_from_plan_no_replan(monkeypatch: pytest.MonkeyPatch) -> None:
    from calm.hrm_text_158.native_full_stack import event_coded_vote_update_adapter as adapter_mod
    from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
        make_event_coded_live_tensor_state,
    )

    plan_calls = 0
    original_plan = adapter_mod.plan_event_coded_integer_vote_update

    def _track_plan(*args, **kwargs):
        nonlocal plan_calls
        plan_calls += 1
        return original_plan(*args, **kwargs)

    def _forbid_reference(*args, **kwargs):
        raise AssertionError("cap-ON replan path must remain unreachable")

    monkeypatch.setattr(adapter_mod, "plan_event_coded_integer_vote_update", _track_plan)
    monkeypatch.setattr(
        adapter_mod,
        "apply_event_coded_integer_vote_update_reference",
        _forbid_reference,
    )

    q = torch.zeros((4, 4), dtype=torch.int8)
    states = {
        "toy.proj": make_event_coded_live_tensor_state("toy.proj", q, 0.25, demotion_band=1),
    }
    votes = torch.zeros((4, 4), dtype=torch.int16)
    votes.view(-1)[0] = 12
    apply_bounded_delta_vote_step(
        states,
        {"toy.proj": votes},
        {"toy.proj": _vote_spec(threshold_abs=10)},
        global_cap_spec=GlobalRateCapSpec(cap=1, step=1, mutate_outputs=True),
    )
    assert plan_calls == 1


def test_sparse_backing_tensors_not_in_stats_or_serialized_surfaces() -> None:
    state = _make_state()
    plan = plan_event_coded_integer_vote_update(
        state,
        VoteUpdateInputs(votes=_votes_for_indices({0: 8})),
        _vote_spec(),
    )
    assert plan.event_coded_sparse_active_idx is not None
    for value in plan.stats.values():
        assert not isinstance(value, torch.Tensor)
    assert "event_coded_sparse_active_idx" not in plan.stats
    result = apply_event_coded_integer_vote_update_from_plan(
        state,
        VoteUpdateInputs(votes=_votes_for_indices({0: 8})),
        _vote_spec(),
        plan,
    )
    json.dumps(result.stats)
    for value in result.stats.values():
        assert not isinstance(value, torch.Tensor)


def test_dense_reference_oracle_still_available() -> None:
    state = _make_state()
    result = apply_event_coded_integer_vote_update_reference(
        state,
        VoteUpdateInputs(votes=_votes_for_indices({0: 8})),
        _vote_spec(),
    )
    assert int(result.stats[C8_TRANSIENT_DENSE_COMPUTE_NUMEL_KEY]) == 16
    before = carrier_content_sha256(state.carrier)
    after = carrier_content_sha256(result.carrier)
    assert before != after
