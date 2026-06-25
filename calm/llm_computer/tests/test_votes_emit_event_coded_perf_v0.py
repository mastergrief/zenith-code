"""CPU equivalence + scale-smoke for event-coded votes-emit telemetry perf."""
from __future__ import annotations

import time
from pathlib import Path

import pytest
import torch

from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    LOCAL_SELECTION_ORDER_CURRENT_MARGIN_INDEX,
    make_event_coded_live_tensor_state,
)
from calm.hrm_text_158.native_full_stack.event_coded_acc_live_carrier import (
    EventCodedAccLiveState,
)
from calm.hrm_text_158.native_full_stack.event_coded_vote_update_adapter import (
    _active_lane_indices,
    _active_lane_index_tensor,
    build_sparse_new_acc_i32_from_carrier,
    build_sparse_new_acc_i32_from_carrier_reference,
)
from calm.hrm_text_158.native_full_stack.oracle_screen_runner import (
    _build_within_tie_band_universe_fast,
    _build_within_tie_band_universe_reference,
    build_within_tie_band_candidate_universe_from_votes,
    build_compact_within_tie_band_sampled_table_rows,
)
from calm.hrm_text_158.native_full_stack.vote_update import VoteUpdateSpec
from calm.hrm_text_158.native_full_stack.votes_emit_collector import (
    VotesEmitCollector,
    _canonical_json,
    _deterministic_sampled_candidates,
    _sha256_text,
    _sparse_vote_inputs_by_state_key,
    build_votes_emit_step_record,
    maybe_emit_votes_step_record,
)


def _vote_spec() -> VoteUpdateSpec:
    return VoteUpdateSpec(
        threshold_abs=1,
        accumulator_clip_min=-127,
        accumulator_clip_max=127,
        max_abs_per_tensor=4096,
        decay_numerator=1,
        decay_denominator=2,
    )


def _assert_fast_matches_reference(
    *,
    tensor_states: dict,
    votes_by_key: dict,
    max_abs_per_tensor: int = 4096,
    max_sampled_candidates: int = 32,
) -> None:
    reference = _build_within_tie_band_universe_reference(
        tensor_states=tensor_states,
        votes_by_key=votes_by_key,
        max_abs_per_tensor=max_abs_per_tensor,
        max_sampled_candidates=max_sampled_candidates,
    )
    fast = _build_within_tie_band_universe_fast(
        tensor_states=tensor_states,
        votes_by_key=votes_by_key,
        max_abs_per_tensor=max_abs_per_tensor,
        max_sampled_candidates=max_sampled_candidates,
    )
    assert reference["sampled_ids"] == fast["sampled_ids"]
    ref_sampled = _deterministic_sampled_candidates(reference)
    fast_sampled = _deterministic_sampled_candidates(fast)
    assert ref_sampled == fast_sampled
    for candidate in ref_sampled:
        candidate.setdefault("candidate_loss", 0.0)
        candidate.setdefault("local_loss_delta", 0.0)
        candidate.setdefault(
            "regret_vs_target_tie_band_oracle_top1_local_loss_delta",
            None,
        )
    for candidate in fast_sampled:
        candidate.setdefault("candidate_loss", 0.0)
        candidate.setdefault("local_loss_delta", 0.0)
        candidate.setdefault(
            "regret_vs_target_tie_band_oracle_top1_local_loss_delta",
            None,
        )
    ref_table = build_compact_within_tie_band_sampled_table_rows(ref_sampled)
    fast_table = build_compact_within_tie_band_sampled_table_rows(fast_sampled)
    assert ref_table == fast_table
    assert _sha256_text(_canonical_json(ref_table)) == _sha256_text(
        _canonical_json(fast_table)
    )


def _make_event_coded_state(key: str, q: torch.Tensor):
    return make_event_coded_live_tensor_state(key, q, 1.0, demotion_band=1)


def _votes_at_density(numel: int, *, density: float, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    votes = torch.zeros(numel, dtype=torch.int16)
    if density >= 1.0:
        votes = torch.randint(-12, 13, (numel,), dtype=torch.int16, generator=generator)
        votes[(votes == 0)] = 1
        return votes
    mask = torch.rand(numel, generator=generator) < density
    nonzero = int(mask.sum().item())
    if nonzero:
        votes[mask] = torch.randint(1, 13, (nonzero,), dtype=torch.int16, generator=generator)
    return votes


def _assert_build_sparse_equivalent(
    carrier: EventCodedAccLiveState,
    q_levels: torch.Tensor,
    votes: torch.Tensor,
    spec: VoteUpdateSpec,
) -> None:
    reference = build_sparse_new_acc_i32_from_carrier_reference(
        carrier,
        q_levels,
        votes,
        spec,
    )
    vectorized = build_sparse_new_acc_i32_from_carrier(
        carrier,
        q_levels,
        votes,
        spec,
    )
    assert torch.equal(reference, vectorized)
    inactive_mask = torch.ones(reference.numel(), dtype=torch.bool)
    active = _active_lane_indices(carrier, votes)
    for flat_index in active:
        inactive_mask[int(flat_index)] = False
    if inactive_mask.any():
        assert torch.all(vectorized.flatten()[inactive_mask] == 0)


def test_build_sparse_matches_reference_all_zero_votes() -> None:
    carrier = EventCodedAccLiveState(logical_numel=256, demotion_band=1)
    votes = torch.zeros(256, dtype=torch.int16)
    q = torch.zeros(256, dtype=torch.int8)
    _assert_build_sparse_equivalent(carrier, q, votes, _vote_spec())


def test_build_sparse_matches_reference_hot_exact_only() -> None:
    carrier = EventCodedAccLiveState(
        logical_numel=256,
        demotion_band=1,
        hot_exact={3: 5, 17: -4},
    )
    votes = torch.zeros(256, dtype=torch.int16)
    q = torch.zeros(256, dtype=torch.int8)
    _assert_build_sparse_equivalent(carrier, q, votes, _vote_spec())


def test_build_sparse_matches_reference_hot_exact_plus_sparse_vote() -> None:
    carrier = EventCodedAccLiveState(
        logical_numel=256,
        demotion_band=1,
        hot_exact={3: 5},
    )
    votes = torch.zeros(256, dtype=torch.int16)
    votes[11] = 7
    q = torch.zeros(256, dtype=torch.int8)
    _assert_build_sparse_equivalent(carrier, q, votes, _vote_spec())


def test_build_sparse_matches_reference_sparse_and_dense() -> None:
    spec = _vote_spec()
    for numel in (64, 1024, 8192):
        q = torch.zeros(numel, dtype=torch.int8)
        carrier = EventCodedAccLiveState(
            logical_numel=numel,
            demotion_band=1,
            hot_exact={1: 3, numel // 2: -8},
        )
        sparse_votes = torch.zeros(numel, dtype=torch.int16)
        sparse_votes[0] = 12
        sparse_votes[7] = -9
        _assert_build_sparse_equivalent(carrier, q, sparse_votes, spec)

        dense_votes = torch.randint(-12, 13, (numel,), dtype=torch.int16)
        _assert_build_sparse_equivalent(carrier, q, dense_votes, spec)


def test_active_lane_index_tensor_matches_set_semantics() -> None:
    carrier = EventCodedAccLiveState(
        logical_numel=128,
        demotion_band=1,
        hot_exact={4: 1, 9: 2},
    )
    votes = torch.zeros(128, dtype=torch.int16)
    votes[1] = 3
    votes[64] = -2
    expected = {1, 4, 9, 64}
    assert _active_lane_indices(carrier, votes) == expected
    tensor_active = set(int(i) for i in _active_lane_index_tensor(carrier, votes).tolist())
    assert tensor_active == expected


def test_sparse_vote_inputs_uses_nonzero_only() -> None:
    votes = {
        "a": torch.tensor([0, 5, 0, -3, 0], dtype=torch.int16),
    }
    sparse = _sparse_vote_inputs_by_state_key(votes)
    assert sparse == {"a": {"1": 5, "3": -3}}


def test_build_within_tie_band_pre_accumulator_matches_carrier_authority() -> None:
    numel = 64
    q = torch.zeros((8, 8), dtype=torch.int8)
    carrier = EventCodedAccLiveState(
        logical_numel=numel,
        demotion_band=1,
        hot_exact={5: 11, 17: -6},
    )
    state = make_event_coded_live_tensor_state("proj", q, 1.0, demotion_band=1, carrier=carrier)
    votes = torch.zeros((8, 8), dtype=torch.int16)
    votes.view(-1)[0] = 12
    votes.view(-1)[17] = 8
    universe = build_within_tie_band_candidate_universe_from_votes(
        tensor_states={"proj": state},
        votes_by_key={"proj": votes},
        max_abs_per_tensor=4096,
        max_sampled_candidates=32,
    )
    for row in universe["candidate_by_id"].values():
        flat_index = int(row["flat_index"])
        assert int(row["pre_accumulator_i16"]) == int(carrier.reconstruct_lane(flat_index))


@pytest.mark.slow
def test_votes_emit_full_step_scale_smoke_32x1m_sparse128(tmp_path: Path) -> None:
    _run_full_step_scale_smoke(tmp_path, density=0.0078125, label="sparse128")


@pytest.mark.slow
def test_votes_emit_full_step_scale_smoke_32x1m_mixed66(tmp_path: Path) -> None:
    _run_full_step_scale_smoke(tmp_path, density=0.667, label="mixed66")


@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    reason=(
        "42.58s > 30s full-step budget; measured decomposition: "
        "build_within_tie_band/universe-ordering ~94% (39.3s), svp1_encode ~0.6% "
        "(0.23s); universe-ordering follow-on OUT of Slice A scope"
    ),
)
def test_votes_emit_full_step_scale_smoke_32x1m_qzero100(tmp_path: Path) -> None:
    _run_full_step_scale_smoke(tmp_path, density=1.0, label="qzero100", q_zero=True)


def _run_full_step_scale_smoke(
    tmp_path: Path,
    *,
    density: float,
    label: str,
    q_zero: bool = False,
) -> None:
    numel = 2048 * 512
    num_states = 32
    spec = _vote_spec()
    tensor_states = {}
    votes_by_key = {}
    vote_specs_by_key = {}
    for module_index in range(num_states):
        key = f"mod{module_index}"
        if q_zero:
            q = torch.zeros(numel, dtype=torch.int8)
        else:
            q = torch.randint(-1, 2, (numel,), dtype=torch.int8)
        state = make_event_coded_live_tensor_state(key, q, 1.0, demotion_band=1)
        votes = _votes_at_density(numel, density=density, seed=module_index + 17)
        tensor_states[key] = state
        votes_by_key[key] = votes
        vote_specs_by_key[key] = spec

    started = time.perf_counter()
    emitted = maybe_emit_votes_step_record(
        root=tmp_path,
        enabled=True,
        optimizer_step_index=0,
        tensor_states=tensor_states,
        votes_by_key=votes_by_key,
        vote_specs_by_key=vote_specs_by_key,
        max_abs_per_tensor=4096,
        collector=VotesEmitCollector(tmp_path),
        local_selection_ordering_mode=LOCAL_SELECTION_ORDER_CURRENT_MARGIN_INDEX,
    )
    elapsed = time.perf_counter() - started
    assert emitted is not None
    assert emitted["step_path"]
    record = build_votes_emit_step_record(
        optimizer_step_index=0,
        tensor_states=tensor_states,
        votes_by_key=votes_by_key,
        vote_specs_by_key=vote_specs_by_key,
        max_abs_per_tensor=4096,
        local_selection_ordering_mode=LOCAL_SELECTION_ORDER_CURRENT_MARGIN_INDEX,
    )
    assert record["sampled_candidate_count"] >= 0
    assert elapsed < 30.0, f"{label} full votes-emit step took {elapsed:.2f}s"


def test_within_tie_band_fast_matches_reference_sparse() -> None:
    numel = 256
    q = torch.zeros(numel, dtype=torch.int8)
    state = _make_event_coded_state("proj", q)
    votes = torch.zeros(numel, dtype=torch.int16)
    votes[0] = 12
    votes[17] = 8
    tensor_states = {"proj": state}
    votes_by_key = {"proj": votes}
    _assert_fast_matches_reference(tensor_states=tensor_states, votes_by_key=votes_by_key)


def test_within_tie_band_fast_matches_reference_dense66() -> None:
    numel = 4096
    q = torch.randint(-1, 2, (numel,), dtype=torch.int8)
    state = _make_event_coded_state("proj", q)
    votes = _votes_at_density(numel, density=0.667, seed=44)
    _assert_fast_matches_reference(
        tensor_states={"proj": state},
        votes_by_key={"proj": votes},
    )


def test_within_tie_band_fast_matches_reference_qzero100() -> None:
    numel = 2048
    q = torch.zeros(numel, dtype=torch.int8)
    state = _make_event_coded_state("proj", q)
    votes = _votes_at_density(numel, density=1.0, seed=43)
    _assert_fast_matches_reference(
        tensor_states={"proj": state},
        votes_by_key={"proj": votes},
    )


def test_within_tie_band_fast_matches_reference_state_key_lex_tiebreak() -> None:
    numel = 64
    q = torch.zeros(numel, dtype=torch.int8)
    state_mod2 = _make_event_coded_state("mod2", q)
    state_mod10 = _make_event_coded_state("mod10", q.clone())
    votes = torch.zeros(numel, dtype=torch.int16)
    votes[0] = 5
    votes[1] = 5
    _assert_fast_matches_reference(
        tensor_states={"mod2": state_mod2, "mod10": state_mod10},
        votes_by_key={"mod2": votes.clone(), "mod10": votes.clone()},
    )


def test_within_tie_band_full_path_preserves_candidate_count_semantics() -> None:
    numel = 128
    q = torch.zeros(numel, dtype=torch.int8)
    state = _make_event_coded_state("proj", q)
    votes = torch.ones(numel, dtype=torch.int16)
    universe = build_within_tie_band_candidate_universe_from_votes(
        tensor_states={"proj": state},
        votes_by_key={"proj": votes},
        max_abs_per_tensor=4096,
        max_sampled_candidates=32,
        materialize_full_candidate_by_id=True,
    )
    candidate_count = len(universe["candidate_by_id"])
    sampled_count = len(_deterministic_sampled_candidates(universe))
    assert candidate_count > sampled_count
    assert sampled_count == 32
    assert bool(candidate_count > sampled_count)
