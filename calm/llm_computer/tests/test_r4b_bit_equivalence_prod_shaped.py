"""r4b CPU bit-equivalence — prod-shaped fixtures (T3)."""
from __future__ import annotations

import random

import torch

from calm.hrm_text_158.native_full_stack.bounded_delta_accumulator import (
    _execute_direct_bounded_local_vote_update_reference_3936d74,
    _tensor_sha256,
    encode_budget_capped_hybrid_reference,
    execute_direct_bounded_local_vote_update_candidate,
    bounded_accumulator_decoded_sha256,
)
from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    apply_bounded_delta_vote_step,
    make_bounded_tensor_state,
)
from calm.hrm_text_158.native_full_stack.sparse_vote_events import SparseVoteEvents
from calm.hrm_text_158.native_full_stack.vote_update import VoteUpdateSpec, VoteUpdateState
from calm.hrm_text_158.native_full_stack.bounded_delta_accumulator import (
    ACCUMULATOR_SUBSTITUTE_LOCAL_VOTE_UPDATE_EXECUTABLE,
)


PROD_NUMEL = 29_600_000


def _spec() -> VoteUpdateSpec:
    return VoteUpdateSpec(
        threshold_abs=10,
        accumulator_clip_min=-127,
        accumulator_clip_max=127,
        max_abs_per_tensor=4,
        fraction_per_tensor=1.0,
    )


def _bundle_sha(result) -> dict[str, str]:
    return {
        "q_sha256": _tensor_sha256(result.next_q_levels),
        "bounded_sha256": bounded_accumulator_decoded_sha256(result.next_bounded_accumulator),
        "proof_sha256": __import__("hashlib").sha256(
            repr(sorted(result.proof.items())).encode("utf-8")
        ).hexdigest(),
    }


def test_r4b_prod_shaped_single_key_execute_direct() -> None:
    rng = random.Random(170)
    numel = PROD_NUMEL
    hot = tuple(rng.sample(range(numel), 32))
    sparse_indices = rng.sample(range(numel), 400)
    sparse_dict = {index: rng.randint(-6, 6) for index in sparse_indices}
    sparse_dict = {k: v for k, v in sparse_dict.items() if v != 0}

    q = torch.zeros(numel, dtype=torch.int8)
    acc = torch.zeros(numel, dtype=torch.int16)
    for index in hot:
        acc[index] = 12

    state = VoteUpdateState(q_levels=q, accumulators=acc)
    bounded = encode_budget_capped_hybrid_reference(
        state,
        hot_exact_indices=hot,
        cold_default_value=0,
    )
    kwargs = dict(
        state_key="r4b.prod",
        q_levels=q,
        bounded_accumulator=bounded,
        vote_spec=_spec(),
    )
    reference = _execute_direct_bounded_local_vote_update_reference_3936d74(
        sparse_vote_events=sparse_dict,
        **kwargs,
    )
    candidate = execute_direct_bounded_local_vote_update_candidate(
        sparse_vote_events=SparseVoteEvents.from_dict(sparse_dict),
        **kwargs,
    )
    assert _bundle_sha(reference) == _bundle_sha(candidate)
    assert reference.proof == candidate.proof
    assert reference.next_bounded_accumulator == candidate.next_bounded_accumulator


def test_r4b_prod_shaped_multi_key_apply_step() -> None:
    rng = random.Random(32)
    numel_per_key = PROD_NUMEL // 32
    states = {}
    votes_by_key = {}
    specs_by_key = {}
    sparse_by_key = {}
    for key_index in range(32):
        key = f"layer.{key_index}.weight"
        q = torch.zeros(numel_per_key, dtype=torch.int8)
        acc = torch.zeros(numel_per_key, dtype=torch.int16)
        hot = (int(rng.random() * numel_per_key), int(rng.random() * numel_per_key))
        acc[list(hot)[0]] = 11
        acc[list(hot)[1]] = -11
        vote_tensor = torch.zeros(numel_per_key, dtype=torch.int16)
        sparse = {}
        for pick in rng.sample(range(numel_per_key), 20):
            vote = rng.randint(-4, 4)
            if vote != 0:
                vote_tensor[pick] = vote
                sparse[pick] = vote
        states[key] = make_bounded_tensor_state(
            key,
            q,
            1.0,
            acc,
            hot_exact_indices=hot,
            cold_default_value=0,
        )
        votes_by_key[key] = vote_tensor
        specs_by_key[key] = _spec()
        sparse_by_key[key] = SparseVoteEvents.from_dict(sparse)

    ref_sparse = {key: events.to_dict() for key, events in sparse_by_key.items()}
    ref_step = apply_bounded_delta_vote_step(
        states,
        votes_by_key,
        specs_by_key,
        candidate_mode=ACCUMULATOR_SUBSTITUTE_LOCAL_VOTE_UPDATE_EXECUTABLE,
        candidate_sparse_vote_events_by_key=ref_sparse,
        candidate_oracle_control_enabled=False,
    )
    cand_step = apply_bounded_delta_vote_step(
        states,
        votes_by_key,
        specs_by_key,
        candidate_mode=ACCUMULATOR_SUBSTITUTE_LOCAL_VOTE_UPDATE_EXECUTABLE,
        candidate_sparse_vote_events_by_key=sparse_by_key,
        candidate_oracle_control_enabled=False,
    )
    assert ref_step.global_summary["q_changed_count"] == cand_step.global_summary["q_changed_count"]
    for key in sorted(states):
        ref_proof = ref_step.global_summary["candidate_local_update_proof_by_key"][key]
        cand_proof = cand_step.global_summary["candidate_local_update_proof_by_key"][key]
        assert ref_proof == cand_proof
        ref_state = ref_step.tensor_states[key]
        cand_state = cand_step.tensor_states[key]
        assert torch.equal(ref_state.q_levels, cand_state.q_levels)
        assert ref_state.bounded_accumulator == cand_state.bounded_accumulator
