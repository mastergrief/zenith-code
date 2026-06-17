"""r4b CPU bit-equivalence — medium fixtures (T2)."""
from __future__ import annotations

import random

import torch

from calm.hrm_text_158.native_full_stack.bounded_delta_accumulator import (
    BoundedDeltaAccumulatorState,
    _execute_direct_bounded_local_vote_update_reference_3936d74,
    encode_budget_capped_hybrid_reference,
    execute_direct_bounded_local_vote_update_candidate,
)
from calm.hrm_text_158.native_full_stack.sparse_vote_events import SparseVoteEvents
from calm.hrm_text_158.native_full_stack.vote_update import VoteUpdateSpec, VoteUpdateState


def _spec(**kwargs) -> VoteUpdateSpec:
    base = dict(
        threshold_abs=10,
        accumulator_clip_min=-127,
        accumulator_clip_max=127,
        decay_numerator=1,
        decay_denominator=1,
        max_abs_per_tensor=4,
        fraction_per_tensor=1.0,
    )
    base.update(kwargs)
    return VoteUpdateSpec(**base)


def _assert_bit_equivalent(reference, candidate) -> None:
    assert reference.next_bounded_accumulator == candidate.next_bounded_accumulator
    assert torch.equal(reference.next_q_levels, candidate.next_q_levels)
    assert reference.proof == candidate.proof


def _run_pair(
    *,
    numel: int,
    hot_indices: tuple[int, ...],
    cold_pairs: tuple[tuple[int, int], ...],
    q_overrides: dict[int, int],
    sparse_dict: dict[int, int],
    vote_spec: VoteUpdateSpec,
    cold_default: int = 0,
) -> None:
    q = torch.zeros(numel, dtype=torch.int8)
    acc = torch.zeros(numel, dtype=torch.int16)
    for index, value in q_overrides.items():
        q[int(index)] = int(value)
    for index, value in cold_pairs:
        acc[int(index)] = int(value)
    for index in hot_indices:
        acc[int(index)] = int(acc[int(index)].item()) or 9

    state = VoteUpdateState(q_levels=q, accumulators=acc)
    bounded = encode_budget_capped_hybrid_reference(
        state,
        hot_exact_indices=hot_indices,
        cold_default_value=cold_default,
        cold_exception_indices=tuple(index for index, _value in cold_pairs),
        cold_exception_values=tuple(value for _index, value in cold_pairs) if cold_pairs else None,
    )
    kwargs = dict(
        state_key="r4b.medium",
        q_levels=q,
        bounded_accumulator=bounded,
        vote_spec=vote_spec,
    )
    reference = _execute_direct_bounded_local_vote_update_reference_3936d74(
        sparse_vote_events=sparse_dict,
        **kwargs,
    )
    candidate_dict = execute_direct_bounded_local_vote_update_candidate(
        sparse_vote_events=sparse_dict,
        **kwargs,
    )
    candidate_carrier = execute_direct_bounded_local_vote_update_candidate(
        sparse_vote_events=SparseVoteEvents.from_dict(sparse_dict),
        **kwargs,
    )
    _assert_bit_equivalent(reference, candidate_dict)
    _assert_bit_equivalent(reference, candidate_carrier)


def test_r4b_medium_random_sparse_events() -> None:
    rng = random.Random(17)
    numel = 10_000
    for _case in range(8):
        hot = tuple(sorted(rng.sample(range(numel), k=rng.randint(1, 12))))
        cold = tuple(
            (index, rng.randint(-40, 40))
            for index in sorted(rng.sample([i for i in range(numel) if i not in hot], k=rng.randint(0, 8)))
        )
        sparse = {
            index: rng.randint(-8, 8)
            for index in rng.sample(range(numel), k=rng.randint(0, 200))
            if rng.randint(-8, 8) != 0
        }
        q_overrides = {
            index: rng.choice([-1, 0, 1])
            for index in rng.sample(range(numel), k=rng.randint(0, 40))
        }
        _run_pair(
            numel=numel,
            hot_indices=hot,
            cold_pairs=cold,
            q_overrides=q_overrides,
            sparse_dict=sparse,
            vote_spec=_spec(max_abs_per_tensor=rng.randint(1, 8)),
        )


def test_r4b_medium_decay_and_max_flips() -> None:
    numel = 10_000
    hot = tuple(range(0, 16))
    sparse = {index: 3 for index in range(16, 36)}
    _run_pair(
        numel=numel,
        hot_indices=hot,
        cold_pairs=((40, 25), (41, -25)),
        q_overrides={},
        sparse_dict=sparse,
        vote_spec=_spec(
            decay_numerator=2,
            decay_denominator=3,
            max_abs_per_tensor=2,
            threshold_abs=5,
        ),
    )


def test_r4b_medium_default_mass_crossing_domain_gap() -> None:
    bounded = BoundedDeltaAccumulatorState(
        logical_shape=(10_000,),
        cold_default_value=10,
        hot_exact_indices=(),
        hot_exact_values=(),
        cold_exception_indices=(),
        cold_exception_values=(),
    )
    q_levels = torch.zeros(10_000, dtype=torch.int8)
    sparse: dict[int, int] = {}
    kwargs = dict(
        state_key="r4b.medium.gap",
        q_levels=q_levels,
        bounded_accumulator=bounded,
        vote_spec=_spec(max_abs_per_tensor=4),
    )
    reference = _execute_direct_bounded_local_vote_update_reference_3936d74(
        sparse_vote_events=sparse,
        **kwargs,
    )
    candidate = execute_direct_bounded_local_vote_update_candidate(
        sparse_vote_events=SparseVoteEvents.from_dict(sparse),
        **kwargs,
    )
    _assert_bit_equivalent(reference, candidate)
    assert reference.proof["pass"] is False
    assert reference.proof["domain_gap_dimension"] == "implicit_default_mass_crossing"
