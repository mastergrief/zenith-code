"""CPU regression: votes-emit collector routes event-coded live carrier correctly."""
from __future__ import annotations

import math

import pytest
import torch

from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    LOCAL_SELECTION_ORDER_CURRENT_MARGIN_INDEX,
    BoundedDeltaAccumulatorState,
    BoundedDeltaTensorState,
    make_event_coded_live_tensor_state,
)
from calm.hrm_text_158.native_full_stack.vote_update import VoteUpdateSpec
from calm.hrm_text_158.native_full_stack.votes_emit_collector import (
    _collect_vote_plans_by_key,
    _preview_warmup_tags,
)


def _vote_spec() -> VoteUpdateSpec:
    return VoteUpdateSpec(
        threshold_abs=1,
        accumulator_clip_min=-127,
        accumulator_clip_max=127,
        max_abs_per_tensor=4096,
    )


def _make_dense_state(*, numel: int = 64) -> BoundedDeltaTensorState:
    side = int(math.sqrt(numel))
    if side * side != numel:
        shape = (numel,)
    else:
        shape = (side, side)
    q = torch.tensor([-1, 0, 1], dtype=torch.int8)
    idx = torch.arange(numel, dtype=torch.long) % 3
    q_levels = q[idx].view(shape).contiguous()
    acc = torch.zeros(numel, dtype=torch.int16)
    bounded = BoundedDeltaAccumulatorState(
        logical_shape=tuple(int(dim) for dim in q_levels.shape),
        cold_default_value=0,
        hot_exact_indices=(),
        hot_exact_values=(),
        cold_exception_indices=(),
        cold_exception_values=(),
        candidate_name="cold_default",
        raw_arrays_included=False,
    )
    return BoundedDeltaTensorState(
        state_key="proj",
        q_levels=q_levels,
        frozen_scale=torch.tensor(1.0, dtype=torch.float32),
        bounded_accumulator=bounded,
        exact_accumulator_shadow=acc.view_as(q_levels),
        bounded_accumulator_fresh_for_exact_shadow=False,
    )


def _event_coded_fixture() -> tuple[BoundedDeltaTensorState, torch.Tensor]:
    numel = 64
    side = 8
    q = torch.zeros((side, side), dtype=torch.int8)
    state = make_event_coded_live_tensor_state(
        "proj",
        q,
        1.0,
        demotion_band=1,
    )
    votes = torch.zeros((side, side), dtype=torch.int16)
    votes[0, 0] = 2
    return state, votes


def test_event_coded_collect_vote_plans_routes_to_event_coded_planner() -> None:
    state, votes = _event_coded_fixture()
    spec = _vote_spec()
    plans = _collect_vote_plans_by_key(
        tensor_states={"proj": state},
        votes_by_key={"proj": votes},
        vote_specs_by_key={"proj": spec},
        two_tier_carry_w6_enabled=False,
        local_loss_delta_by_key=None,
        local_selection_ordering_seed=0,
        optimizer_step_index=0,
        local_selection_ordering_mode=LOCAL_SELECTION_ORDER_CURRENT_MARGIN_INDEX,
    )
    plan = plans["proj"]
    assert plan.stats.get("event_coded_live_carrier_plan") is True
    assert hasattr(plan, "applied_indices")
    assert hasattr(plan, "pre_veto_selected_indices")
    assert hasattr(plan, "new_acc_i32")


def test_event_coded_preview_warmup_tags_routes_to_event_coded_planner() -> None:
    state, votes = _event_coded_fixture()
    spec = _vote_spec()
    tags = _preview_warmup_tags(
        tensor_states={"proj": state},
        votes_by_key={"proj": votes},
        vote_specs_by_key={"proj": spec},
        two_tier_carry_w6_enabled=False,
        local_loss_delta_by_key=None,
        local_selection_ordering_seed=0,
        optimizer_step_index=0,
        local_selection_ordering_mode=LOCAL_SELECTION_ORDER_CURRENT_MARGIN_INDEX,
    )
    assert "warmup_apply_class" in tags


def test_event_coded_votes_emit_rejects_two_tier_w6_enabled() -> None:
    state, votes = _event_coded_fixture()
    spec = _vote_spec()
    with pytest.raises(
        ValueError,
        match="two_tier_carry_w6_enabled forbidden on event-coded live carrier votes-emit path",
    ):
        _collect_vote_plans_by_key(
            tensor_states={"proj": state},
            votes_by_key={"proj": votes},
            vote_specs_by_key={"proj": spec},
            two_tier_carry_w6_enabled=True,
            local_loss_delta_by_key=None,
            local_selection_ordering_seed=0,
            optimizer_step_index=0,
            local_selection_ordering_mode=LOCAL_SELECTION_ORDER_CURRENT_MARGIN_INDEX,
        )
    with pytest.raises(
        ValueError,
        match="two_tier_carry_w6_enabled forbidden on event-coded live carrier votes-emit path",
    ):
        _preview_warmup_tags(
            tensor_states={"proj": state},
            votes_by_key={"proj": votes},
            vote_specs_by_key={"proj": spec},
            two_tier_carry_w6_enabled=True,
            local_loss_delta_by_key=None,
            local_selection_ordering_seed=0,
            optimizer_step_index=0,
            local_selection_ordering_mode=LOCAL_SELECTION_ORDER_CURRENT_MARGIN_INDEX,
        )


def test_dense_collect_vote_plans_uses_dense_planner() -> None:
    state = _make_dense_state()
    votes = torch.randint(-3, 4, state.q_levels.shape, dtype=torch.int16)
    spec = _vote_spec()
    plans = _collect_vote_plans_by_key(
        tensor_states={"proj": state},
        votes_by_key={"proj": votes},
        vote_specs_by_key={"proj": spec},
        two_tier_carry_w6_enabled=False,
        local_loss_delta_by_key=None,
        local_selection_ordering_seed=0,
        optimizer_step_index=0,
        local_selection_ordering_mode=LOCAL_SELECTION_ORDER_CURRENT_MARGIN_INDEX,
    )
    plan = plans["proj"]
    assert plan.stats.get("event_coded_live_carrier_plan") is not True
