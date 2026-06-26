"""Regression tests for assert-path carrier_content_sha256 dedup (harness-only)."""
from __future__ import annotations

from unittest import mock

import pytest
import torch

from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    make_event_coded_live_tensor_state,
)
from calm.hrm_text_158.native_full_stack.event_coded_acc_live_carrier import (
    EventCodedAccEvent,
    EventCodedAccLiveState,
)
from calm.hrm_text_158.native_full_stack.event_coded_vote_update_adapter import (
    C8StepObservation,
    EventCodedVoteUpdateState,
    apply_event_coded_integer_vote_update_reference,
    assert_c8_runtime_guards,
    carrier_content_sha256,
    measure_persistent_dense_accumulator_materialized_numel,
)
from calm.hrm_text_158.native_full_stack.oracle_screen_runner import (
    hash_bounded_delta_tensor_states_pre_update,
)
from calm.hrm_text_158.native_full_stack.vote_update import VoteUpdateInputs, VoteUpdateSpec


def _vote_spec() -> VoteUpdateSpec:
    return VoteUpdateSpec(
        threshold_abs=10,
        accumulator_clip_min=-127,
        accumulator_clip_max=127,
        max_abs_per_tensor=16,
    )


def _tiny_state() -> dict[str, object]:
    q = torch.zeros((4, 4), dtype=torch.int8)
    return {
        "toy.proj": make_event_coded_live_tensor_state("toy.proj", q, 0.25, demotion_band=1),
    }


def _votes_for_index(flat_index: int, magnitude: int = 12) -> torch.Tensor:
    votes = torch.zeros(16, dtype=torch.int16)
    votes[int(flat_index)] = int(magnitude)
    return votes.view(4, 4)


def _fixture_carrier() -> EventCodedAccLiveState:
    return EventCodedAccLiveState.with_hot_exact(
        logical_numel=64,
        demotion_band=1,
        hot_exact={3: 5, 17: -4},
        events=[
            EventCodedAccEvent(flat_index=3, direction=1, residual_mag=2, event_type=1),
            EventCodedAccEvent(flat_index=17, direction=0, residual_mag=1, event_type=1),
        ],
    )


def test_assert_guard_semantics_dense_authority_still_raises() -> None:
    carrier = _fixture_carrier()
    observation = C8StepObservation()
    shadow = torch.zeros(16, dtype=torch.int16)
    with pytest.raises(ValueError, match="dense persistent accumulator authority"):
        assert_c8_runtime_guards(
            carrier,
            observation=observation,
            persistent_dense_accumulator_materialized_numel=measure_persistent_dense_accumulator_materialized_numel(
                exact_accumulator_shadow=shadow,
                event_coded_live_carrier=carrier,
                eligible_numel=16,
            ),
        )


def test_assert_guard_semantics_triton_preplan_still_raises() -> None:
    carrier = _fixture_carrier()
    observation = C8StepObservation(vote_update_preplan_triton_invoked=True)
    with pytest.raises(ValueError, match="Triton preplan forbidden"):
        assert_c8_runtime_guards(
            carrier,
            observation=observation,
            persistent_dense_accumulator_materialized_numel=0,
        )


def test_checkpoint_boundary_hash_stable_unchanged_by_assert_dedup() -> None:
    carrier = _fixture_carrier()
    first = carrier_content_sha256(carrier)
    second = carrier_content_sha256(carrier)
    assert first == second
    assert len(first) == 64


def test_apply_returned_stats_still_reports_content_sha_after() -> None:
    states = _tiny_state()
    prior = states["toy.proj"]
    vu = prior.vote_update_state()
    assert isinstance(vu, EventCodedVoteUpdateState)
    result = apply_event_coded_integer_vote_update_reference(
        vu,
        VoteUpdateInputs(votes=_votes_for_index(0)),
        _vote_spec(),
    )
    sha = result.stats.get("event_coded_live_carrier_content_sha256_after")
    assert isinstance(sha, str) and len(sha) == 64
    assert sha == carrier_content_sha256(result.carrier)


def test_assert_path_does_not_call_carrier_content_sha256() -> None:
    carrier = _fixture_carrier()
    observation = C8StepObservation()
    with mock.patch(
        "calm.hrm_text_158.native_full_stack.event_coded_vote_update_adapter.carrier_content_sha256",
        wraps=carrier_content_sha256,
    ) as sha_mock:
        assert_c8_runtime_guards(
            carrier,
            observation=observation,
            persistent_dense_accumulator_materialized_numel=0,
        )
        assert sha_mock.call_count == 0


def test_retained_hash_sites_still_call_carrier_content_sha256() -> None:
    states = _tiny_state()
    tensor_states = states
    votes_by_key = {"toy.proj": _votes_for_index(1)}
    prior = states["toy.proj"]
    vu = prior.vote_update_state()
    assert isinstance(vu, EventCodedVoteUpdateState)

    with mock.patch(
        "calm.hrm_text_158.native_full_stack.event_coded_vote_update_adapter.carrier_content_sha256",
        wraps=carrier_content_sha256,
    ) as adapter_sha_mock, mock.patch(
        "calm.hrm_text_158.native_full_stack.oracle_screen_runner.carrier_content_sha256",
        wraps=carrier_content_sha256,
    ) as oracle_sha_mock:
        hash_bounded_delta_tensor_states_pre_update(tensor_states)
        apply_event_coded_integer_vote_update_reference(
            vu,
            VoteUpdateInputs(votes=votes_by_key["toy.proj"]),
            _vote_spec(),
        )
    assert oracle_sha_mock.call_count >= 1
    assert adapter_sha_mock.call_count >= 1
