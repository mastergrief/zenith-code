"""B-vec: vectorized array-backed event-coded carrier apply (pure equivalence + scale-smoke)."""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
import torch

from calm.hrm_text_158.native_full_stack.event_coded_acc_live_carrier import (
    EventCodedAccLiveState,
    StepSurfaceRecord,
    _PackedHotTable,
    _apply_step_dict_impl,
    apply_step_dict_reference,
)
from calm.hrm_text_158.native_full_stack.event_coded_vote_update_adapter import (
    _votes_dict_from_tensor,
    apply_event_coded_cap_mutations,
    carrier_content_sha256,
)
from calm.hrm_text_158.native_full_stack.vote_update import VoteUpdatePlan


def _minimal_plan(
    *,
    applied_indices: list[int],
    numel: int,
) -> VoteUpdatePlan:
    applied = torch.tensor(applied_indices, dtype=torch.int64)
    empty_i64 = torch.tensor([], dtype=torch.int64)
    empty_i16 = torch.tensor([], dtype=torch.int16)
    empty_i8 = torch.tensor([], dtype=torch.int8)
    return VoteUpdatePlan(
        q_i16=torch.zeros(numel, dtype=torch.int16),
        new_acc_i32=torch.zeros(numel, dtype=torch.int32),
        candidate_indices=applied.clone(),
        pre_veto_selected_indices=applied.clone(),
        applied_indices=applied,
        applied_directions=torch.ones(len(applied_indices), dtype=torch.int8),
        applied_thresholds=torch.full((len(applied_indices),), 10, dtype=torch.int16),
        replay_ce_veto_indices=empty_i64,
        replay_veto_directions=empty_i8,
        replay_veto_thresholds=empty_i16,
        pc_aux_negative_indices=empty_i64,
        pc_aux_veto_indices=empty_i64,
        stats={},
    )


def _record_tuple(record: StepSurfaceRecord) -> tuple[Any, ...]:
    return (
        record.crossing_indices,
        record.applied_indices,
        record.backlog_indices,
        dict(record.q_levels),
        record.hot_exact_row_count,
        record.promotion_count,
        record.demotion_on_decay_count,
        record.demotion_on_crossing_count,
    )


def _assert_apply_equivalent(
    carrier_seed: EventCodedAccLiveState,
    *,
    votes: dict[int, int],
    step_index: int = 0,
) -> None:
    fast = carrier_seed.cow_copy()
    oracle_carrier = carrier_seed.cow_copy()
    oracle_record = apply_step_dict_reference(
        oracle_carrier,
        int(step_index),
        votes=dict(votes),
    )
    fast_record = fast.apply_step(int(step_index), votes=dict(votes))
    assert _record_tuple(fast_record) == _record_tuple(oracle_record)


def test_e1_sparse_votes_equivalence() -> None:
    carrier = EventCodedAccLiveState(logical_numel=64, demotion_band=1)
    _assert_apply_equivalent(carrier, votes={0: 6, 3: -2})


def test_e2_hot_exact_only_equivalence() -> None:
    carrier = EventCodedAccLiveState.with_hot_exact(
        logical_numel=128,
        demotion_band=1,
        hot_exact={2: 4, 17: -3, 63: 8},
    )
    _assert_apply_equivalent(carrier, votes={})


def test_e3_dense_vote_pattern_equivalence() -> None:
    numel = 256
    carrier = EventCodedAccLiveState.with_hot_exact(
        logical_numel=numel,
        demotion_band=1,
        hot_exact={1: 2, numel // 2: -5},
    )
    votes = {index: ((-1) ** index) * (index % 7 + 1) for index in range(0, numel, 3)}
    _assert_apply_equivalent(carrier, votes=votes)


def test_e4_delayed_crossing_equivalence() -> None:
    carrier = EventCodedAccLiveState.with_hot_exact(
        logical_numel=32,
        demotion_band=3,
        hot_exact={4: 9},
    )
    carrier.q_levels[4] = 0
    _assert_apply_equivalent(carrier, votes={4: 2})


def test_e5_decay_without_selection_equivalence() -> None:
    carrier = EventCodedAccLiveState.with_hot_exact(
        logical_numel=32,
        demotion_band=3,
        hot_exact={7: 1},
    )
    _assert_apply_equivalent(carrier, votes={})


def test_e6_cap_residual_then_empty_apply_equivalence() -> None:
    carrier = EventCodedAccLiveState.with_hot_exact(
        logical_numel=64,
        demotion_band=1,
        hot_exact={5: 11},
    )
    _assert_apply_equivalent(carrier, votes={0: 8})
    carrier2 = carrier.cow_copy()
    _assert_apply_equivalent(carrier2, votes={})


def test_e8_promotion_demotion_counts_equivalence() -> None:
    carrier = EventCodedAccLiveState(logical_numel=48, demotion_band=2)
    for step, vote in enumerate(({0: 12}, {1: -12}, {}, {0: 1})):
        _assert_apply_equivalent(carrier, votes=vote, step_index=step)
        carrier = carrier.cow_copy()
        carrier.apply_step(step, votes=vote)


def test_e9_backlog_fields_preserved() -> None:
    carrier = EventCodedAccLiveState(logical_numel=16, demotion_band=1)
    carrier.backlog.add(3)
    _assert_apply_equivalent(carrier, votes={2: 4})


def test_e10_q_level_effects_equivalence() -> None:
    carrier = EventCodedAccLiveState.with_hot_exact(
        logical_numel=32,
        demotion_band=1,
        hot_exact={8: 9},
    )
    carrier.q_levels[8] = 1
    _assert_apply_equivalent(carrier, votes={8: 3})


def test_e11_step_surface_record_shape_equivalence() -> None:
    carrier = EventCodedAccLiveState(logical_numel=24, demotion_band=1)
    _assert_apply_equivalent(carrier, votes={0: 15, 11: -4})


def test_e12_checkpoint_sha_equivalence_after_step() -> None:
    import copy

    from calm.hrm_text_158.native_full_stack.event_coded_acc_live_carrier import (
        _apply_step_dict_impl,
        _carrier_as_dict_state,
    )

    carrier = EventCodedAccLiveState.with_hot_exact(
        logical_numel=64,
        demotion_band=1,
        hot_exact={3: 6, 9: -2},
    )
    fast = carrier.cow_copy()
    fast.apply_step(0, votes={0: 5})
    oracle = copy.deepcopy(_carrier_as_dict_state(carrier))
    _apply_step_dict_impl(oracle, 0, votes={0: 5})
    assert carrier_content_sha256(fast) == carrier_content_sha256(oracle)


def test_e13_double_decay_characterization() -> None:
    """Cap residual in demotion band persists until the 2nd empty-vote apply demotes it."""

    carrier = EventCodedAccLiveState(logical_numel=32, demotion_band=3)
    q = torch.zeros(32, dtype=torch.int8)
    plan = _minimal_plan(applied_indices=[5], numel=32)
    local = carrier.cow_copy()
    local.apply_step(0, votes={5: 12})
    # Cap residual write alone (|residual|=2 < demotion_band=3) keeps lane 5 hot.
    residual_only = local.cow_copy()
    residual_only.hot_exact[5] = 2
    assert 5 in residual_only.hot_exact
    # Full cap path runs the 2nd apply_step(votes={}) which demotes sub-band lanes.
    _q_out, after_cap = apply_event_coded_cap_mutations(
        local,
        q,
        plan,
        [5],
        step_index=1,
    )
    assert 5 not in after_cap.hot_exact


def test_b6_hot_exact_public_mutation_invalidates_packed_hash() -> None:
    carrier = EventCodedAccLiveState.with_hot_exact(
        logical_numel=32,
        demotion_band=1,
        hot_exact={1: 2},
    )
    sha_before = carrier_content_sha256(carrier)
    packed_before = carrier.hot_packed_bytes()
    carrier.hot_exact[3] = 5
    sha_after = carrier_content_sha256(carrier)
    packed_after = carrier.hot_packed_bytes()
    assert sha_before != sha_after
    assert packed_before != packed_after


def test_b2_duplicate_hot_risk_override_matches_oracle() -> None:
    carrier = EventCodedAccLiveState.with_hot_exact(
        logical_numel=64,
        demotion_band=3,
        hot_exact={4: 9},
    )
    carrier.q_levels[4] = 0
    votes = {4: 2}
    duplicate_override = [4, 4, 4]
    fast = carrier.cow_copy()
    fast_record = fast.apply_step(
        0,
        votes=votes,
        hot_risk_override=duplicate_override,
    )
    oracle_carrier = carrier.cow_copy()
    oracle_record = apply_step_dict_reference(
        oracle_carrier,
        0,
        votes=votes,
        hot_risk_override=duplicate_override,
    )
    assert _record_tuple(fast_record) == _record_tuple(oracle_record)


def test_b2_cold_non_voted_override_matches_oracle() -> None:
    """Override lane not hot, not voted, not near-threshold — proxy replacement must promote it."""
    carrier = EventCodedAccLiveState.with_hot_exact(
        logical_numel=64,
        demotion_band=3,
        hot_exact={},
    )
    cold_override = [10, 10]
    fast = carrier.cow_copy()
    fast_record = fast.apply_step(
        0,
        votes={},
        hot_risk_override=cold_override,
    )
    oracle_carrier = carrier.cow_copy()
    oracle_record = _apply_step_dict_impl(
        oracle_carrier,
        0,
        votes={},
        hot_risk_override=cold_override,
    )
    assert _record_tuple(fast_record) == _record_tuple(oracle_record)
    assert dict(fast.hot_exact) == dict(oracle_carrier.hot_exact)
    assert carrier_content_sha256(fast) == carrier_content_sha256(oracle_carrier)


def test_cow_fork_isolates_mutations() -> None:
    parent = EventCodedAccLiveState.with_hot_exact(
        logical_numel=32,
        demotion_band=1,
        hot_exact={1: 3, 2: 4},
    )
    parent_sha = carrier_content_sha256(parent)
    fork_a = parent.cow_copy()
    fork_b = parent.cow_copy()
    fork_a.apply_step(0, votes={0: 8})
    fork_b.apply_step(0, votes={3: -6})
    assert carrier_content_sha256(parent) == parent_sha
    assert carrier_content_sha256(fork_a) != carrier_content_sha256(fork_b)


def test_votes_dict_from_tensor_matches_tolist_reference() -> None:
    votes = torch.zeros(1024, dtype=torch.int16)
    votes[0] = 12
    votes[17] = -3
    votes[1023] = 5
    reference = {
        int(index): int(value)
        for index, value in enumerate(votes.flatten().tolist())
        if int(value) != 0
    }
    assert _votes_dict_from_tensor(votes) == reference


def _make_large_hot_carrier(numel: int, hot_count: int) -> EventCodedAccLiveState:
    """Stress proxy: 600k hot rows; carries in [-3, 4] (within demotion band, realistic density)."""
    rng = np.random.default_rng(17)
    indices = np.sort(rng.choice(numel, size=hot_count, replace=False))
    values = rng.integers(-3, 4, size=hot_count, dtype=np.int16)
    return EventCodedAccLiveState(
        logical_numel=int(numel),
        demotion_band=1,
        _hot=_PackedHotTable.from_arrays(indices, values),
    )


@dataclass(frozen=True)
class _CapPathDoubleApplyResult:
    carrier: EventCodedAccLiveState
    first_apply_record: StepSurfaceRecord
    second_apply_record: StepSurfaceRecord
    first_apply_seconds: float
    cap_path_seconds: float


def _simulate_cap_path_double_apply(
    carrier: EventCodedAccLiveState,
    *,
    accepted_indices: list[int],
) -> _CapPathDoubleApplyResult:
    q = torch.zeros(carrier.logical_numel, dtype=torch.int8)
    if accepted_indices:
        plan = _minimal_plan(applied_indices=accepted_indices, numel=carrier.logical_numel)
    else:
        plan = _minimal_plan(applied_indices=[0], numel=carrier.logical_numel)
    local = carrier.cow_copy()
    vote_keys = accepted_indices[: min(8, len(accepted_indices))]
    t_first = time.perf_counter()
    first_record = local.apply_step(0, votes={int(i): 1 for i in vote_keys})
    first_apply_seconds = time.perf_counter() - t_first
    t_cap = time.perf_counter()
    _q_out, updated = apply_event_coded_cap_mutations(
        local,
        q,
        plan,
        accepted_indices,
        step_index=1,
    )
    cap_path_seconds = time.perf_counter() - t_cap
    second_record = updated.step_records[-1]
    return _CapPathDoubleApplyResult(
        carrier=updated,
        first_apply_record=first_record,
        second_apply_record=second_record,
        first_apply_seconds=first_apply_seconds,
        cap_path_seconds=cap_path_seconds,
    )


@pytest.mark.slow
def test_full_cap_path_scale_smoke_600k_x32_under_30s() -> None:
    numel = 1_048_576
    hot_count = 600_000
    carriers = [_make_large_hot_carrier(numel, hot_count) for _ in range(32)]
    accepted = list(range(0, min(128, hot_count), max(1, hot_count // 128)))
    apply_total = 0.0
    cap_total = 0.0
    last_result: _CapPathDoubleApplyResult | None = None
    t0 = time.perf_counter()
    for carrier in carriers:
        result = _simulate_cap_path_double_apply(carrier, accepted_indices=accepted)
        last_result = result
        apply_total += result.first_apply_seconds
        cap_total += result.cap_path_seconds
    elapsed = time.perf_counter() - t0
    assert last_result is not None
    first_rec = last_result.first_apply_record
    second_rec = last_result.second_apply_record
    print(
        f"scale_smoke elapsed={elapsed:.2f}s hot={hot_count} states=32 "
        f"apply_total={apply_total:.2f}s cap_total={cap_total:.2f}s "
        f"apply_avg={apply_total/32:.2f}s cap_path_avg={cap_total/32:.2f}s "
        f"hot_rows={last_result.carrier._hot.indices_array().size} "
        f"first_crossings={len(first_rec.crossing_indices)} "
        f"second_crossings={len(second_rec.crossing_indices)} "
        f"first_demotion_decay={first_rec.demotion_on_decay_count} "
        f"first_demotion_cross={first_rec.demotion_on_crossing_count} "
        f"second_demotion_decay={second_rec.demotion_on_decay_count} "
        f"second_demotion_cross={second_rec.demotion_on_crossing_count}"
    )
    assert elapsed < 30.0, f"full cap-path scale-smoke exceeded 30s budget: {elapsed:.2f}s"
