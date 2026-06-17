"""R5 CPU bit-equivalence — high-density apply vs frozen reference (ordering/truncation)."""
from __future__ import annotations

import hashlib
import random

import pytest
import torch

from calm.hrm_text_158.native_full_stack.bounded_delta_accumulator import (
    BoundedDeltaAccumulatorState,
    _execute_direct_bounded_local_vote_update_reference_3936d74,
    _tensor_sha256,
    bounded_accumulator_decoded_sha256,
    execute_direct_bounded_local_vote_update_candidate,
)
from calm.hrm_text_158.native_full_stack.sparse_vote_events import SparseVoteEvents
from calm.hrm_text_158.native_full_stack.vote_update import VoteUpdateSpec

# Claude gate-1 template: numel=100K, density {0.65, 0.30}, votes ±15, hot=32 acc=12.
BIT_EQUIV_NUMEL = 100_000
PROD_NUMEL = 925_000
PROD_EVENTS = 601_250
DENSITIES = (0.65, 0.30)
MAX_FLIPS_CASES = (4, 64, 4096)
SEEDS = (11, 23, 37)


def _bundle_sha(result) -> dict[str, str]:
    return {
        "q_sha256": _tensor_sha256(result.next_q_levels),
        "bounded_sha256": bounded_accumulator_decoded_sha256(result.next_bounded_accumulator),
        "proof_sha256": hashlib.sha256(
            repr(sorted(result.proof.items())).encode("utf-8")
        ).hexdigest(),
    }


def _make_state(numel: int) -> BoundedDeltaAccumulatorState:
    rng = random.Random(17)
    hot = tuple(sorted(rng.sample(range(numel), 32)))
    cold = tuple(sorted(rng.sample([i for i in range(numel) if i not in hot], 48)))
    return BoundedDeltaAccumulatorState(
        logical_shape=(numel,),
        cold_default_value=0,
        hot_exact_indices=hot,
        hot_exact_values=tuple(12 for _ in hot),
        cold_exception_indices=cold,
        cold_exception_values=tuple(-9 for _ in cold),
        candidate_name="r5.high_density",
        raw_arrays_included=False,
    )


def _make_sparse_dict(numel: int, density: float, *, seed: int) -> dict[int, int]:
    rng = random.Random(seed)
    events = int(numel * density)
    events = min(events, numel)
    sparse: dict[int, int] = {}
    for index in rng.sample(range(numel), events):
        vote = rng.randint(-15, 15)
        if vote != 0:
            sparse[index] = vote
    return sparse


def _spec(max_flips: int) -> VoteUpdateSpec:
    return VoteUpdateSpec(
        threshold_abs=10,
        accumulator_clip_min=-127,
        accumulator_clip_max=127,
        max_abs_per_tensor=max_flips,
        fraction_per_tensor=1.0,
    )


@pytest.mark.parametrize("density", DENSITIES)
@pytest.mark.parametrize("max_flips", MAX_FLIPS_CASES)
@pytest.mark.parametrize("seed", SEEDS)
def test_r5_high_density_bit_equivalence_vs_reference(
    density: float, max_flips: int, seed: int
) -> None:
    """Dense carrier vs frozen reference at high density; stresses max_flips truncation."""
    numel = BIT_EQUIV_NUMEL
    sparse_dict = _make_sparse_dict(numel, density, seed=seed)
    q = torch.zeros(numel, dtype=torch.int8)
    state = _make_state(numel)
    kwargs = dict(
        state_key=f"r5.hd.{density}.{max_flips}.{seed}",
        q_levels=q,
        bounded_accumulator=state,
        vote_spec=_spec(max_flips),
    )
    reference = _execute_direct_bounded_local_vote_update_reference_3936d74(
        sparse_vote_events=sparse_dict,
        **kwargs,
    )
    candidate_count = int(reference.proof.get("candidate_count", 0))
    # Broad ±15 votes over high-density support must exercise real truncation, not vacuous top-4.
    assert candidate_count > 1000, f"candidate_count={candidate_count} too small to stress truncation"
    assert candidate_count > max_flips, (
        f"candidate_count={candidate_count} must exceed max_flips={max_flips}"
    )
    candidate = execute_direct_bounded_local_vote_update_candidate(
        sparse_vote_events=SparseVoteEvents.from_dict(sparse_dict),
        **kwargs,
    )
    assert reference.proof.get("max_flips") == candidate.proof.get("max_flips")
    assert _bundle_sha(reference) == _bundle_sha(candidate)
    assert reference.proof == candidate.proof
    assert reference.next_bounded_accumulator == candidate.next_bounded_accumulator


def test_r5_high_density_prod_scale_bit_equivalence_vs_reference() -> None:
    """Real arm-A scale (925K numel / 601K events ≈65%) vs frozen reference."""
    numel = PROD_NUMEL
    sparse_dict = _make_sparse_dict(numel, PROD_EVENTS / numel, seed=170)
    q = torch.zeros(numel, dtype=torch.int8)
    state = _make_state(numel)
    max_flips = 4
    kwargs = dict(
        state_key="r5.hd.prod_scale",
        q_levels=q,
        bounded_accumulator=state,
        vote_spec=_spec(max_flips),
    )
    reference = _execute_direct_bounded_local_vote_update_reference_3936d74(
        sparse_vote_events=sparse_dict,
        **kwargs,
    )
    candidate_count = int(reference.proof.get("candidate_count", 0))
    assert candidate_count > 1000
    assert candidate_count > max_flips
    candidate = execute_direct_bounded_local_vote_update_candidate(
        sparse_vote_events=SparseVoteEvents.from_dict(sparse_dict),
        **kwargs,
    )
    assert _bundle_sha(reference) == _bundle_sha(candidate)
    assert reference.proof == candidate.proof
    assert reference.next_bounded_accumulator == candidate.next_bounded_accumulator
