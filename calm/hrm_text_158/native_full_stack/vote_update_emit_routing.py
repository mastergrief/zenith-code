"""Shared vote-update planner routing for votes-emit and oracle-screen paths."""
from __future__ import annotations

from typing import Any

from calm.hrm_text_158.native_full_stack.event_coded_vote_update_adapter import (
    EventCodedVoteUpdateState,
    plan_event_coded_integer_vote_update_reference,
)
from calm.hrm_text_158.native_full_stack.vote_update import (
    VoteUpdateInputs,
    VoteUpdatePlan,
    VoteUpdateSpec,
    plan_integer_vote_update_reference,
)


def plan_vote_update_for_emit(
    vote_state: Any,
    inputs: VoteUpdateInputs,
    spec: VoteUpdateSpec,
    *,
    local_selection_ordering_mode: str,
    local_selection_ordering_seed: int,
    local_selection_ordering_step: int,
    two_tier_carry_w6_enabled: bool,
) -> VoteUpdatePlan:
    if isinstance(vote_state, EventCodedVoteUpdateState):
        if two_tier_carry_w6_enabled:
            raise ValueError(
                "two_tier_carry_w6_enabled forbidden on event-coded live carrier votes-emit path"
            )
        return plan_event_coded_integer_vote_update_reference(
            vote_state,
            inputs,
            spec,
            local_selection_ordering_mode=str(local_selection_ordering_mode),
            local_selection_ordering_seed=int(local_selection_ordering_seed),
            local_selection_ordering_step=int(local_selection_ordering_step),
            observation=None,
        )
    return plan_integer_vote_update_reference(
        vote_state,
        inputs,
        spec,
        local_selection_ordering_mode=str(local_selection_ordering_mode),
        local_selection_ordering_seed=int(local_selection_ordering_seed),
        local_selection_ordering_step=int(local_selection_ordering_step),
        two_tier_carry_w6_enabled=bool(two_tier_carry_w6_enabled),
    )
