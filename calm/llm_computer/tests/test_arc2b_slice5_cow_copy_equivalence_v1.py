"""Bit-exact carrier equivalence: legacy vote+cow_copy-cap vs in-place merged path."""

from __future__ import annotations

import copy
import hashlib
from typing import Any, Sequence

import numpy as np
import torch

from calm.hrm_text_158.native_full_stack.event_coded_acc_live_carrier import (
    EventCodedAccLiveState,
    StepSurfaceRecord,
    _PackedHotTable,
    merge_hot_table_arrays,
)
from calm.hrm_text_158.native_full_stack.event_coded_vote_update_adapter import (
    EventCodedVoteUpdateState,
    apply_event_coded_carrier_step,
    apply_event_coded_integer_vote_update_from_plan,
    apply_event_coded_vote_and_cap_from_plan,
    carrier_content_sha256,
    _sync_q_levels_tensor,
)
from calm.hrm_text_158.native_full_stack.vote_update import VoteUpdateInputs, VoteUpdatePlan, VoteUpdateSpec


def _minimal_plan(*, applied_indices: list[int], numel: int) -> VoteUpdatePlan:
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
        int(record.step_index),
        record.crossing_indices,
        record.applied_indices,
        record.backlog_indices,
        dict(record.q_levels),
        record.hot_exact_row_count,
        record.promotion_count,
        record.demotion_on_decay_count,
        record.demotion_on_crossing_count,
    )


def serialize_carrier_equivalence_snapshot(
    carrier: EventCodedAccLiveState,
    q_levels: torch.Tensor,
) -> dict[str, Any]:
    return {
        "carrier_sha256": carrier_content_sha256(carrier),
        "q_sha256": hashlib.sha256(q_levels.detach().cpu().numpy().tobytes()).hexdigest(),
        "step_records": [_record_tuple(record) for record in carrier.step_records],
        "events_len": len(carrier.events),
        "backlog": tuple(sorted(int(i) for i in carrier.backlog)),
        "q_levels_dict": dict(carrier.q_levels),
        "hot_exact": dict(carrier.hot_exact),
    }


def _legacy_apply_event_coded_cap_mutations_with_cow_copy(
    carrier: EventCodedAccLiveState,
    q_levels: torch.Tensor,
    plan: VoteUpdatePlan,
    accepted_indices: Sequence[int],
    *,
    step_index: int,
) -> tuple[torch.Tensor, EventCodedAccLiveState]:
    """Pre-slice-2 reference: redundant cow_copy before cap mutations."""
    if not accepted_indices:
        return q_levels, carrier
    updated = carrier.cow_copy()
    q_out = q_levels.detach().cpu().clone().to(torch.int8)
    q_flat = q_out.flatten()
    threshold_flat = plan.applied_thresholds.detach().cpu().flatten()
    direction_flat = plan.applied_directions.detach().cpu().flatten()
    applied_flat = plan.applied_indices.detach().cpu().flatten()
    index_by_pos = {int(idx): pos for pos, idx in enumerate(applied_flat.tolist())}
    cap_indices: list[int] = []
    cap_values: list[int] = []
    for flat_index in accepted_indices:
        idx = int(flat_index)
        pos = index_by_pos.get(idx)
        if pos is None:
            continue
        direction = int(direction_flat[pos].item())
        q_flat[idx] = int(max(-1, min(1, int(q_flat[idx].item()) + direction)))
        updated.q_levels[idx] = int(q_flat[idx].item())
        carry = int(updated.reconstruct_lane(idx))
        residual = carry - direction * int(threshold_flat[pos].item())
        cap_indices.append(idx)
        cap_values.append(int(residual))
    if cap_indices:
        cap_idx = np.array(cap_indices, dtype=np.int32)
        cap_val = np.array(cap_values, dtype=np.int16)
        hot_idx, hot_val = merge_hot_table_arrays(
            updated._hot.indices_array(),
            updated._hot.values_array(),
            np.empty(0, dtype=np.int32),
            cap_idx,
            cap_val,
        )
        updated._hot.replace_arrays(hot_idx, hot_val)
        updated._invalidate_packed_caches()
    apply_event_coded_carrier_step(updated, votes={}, step_index=int(step_index))
    q_synced = _sync_q_levels_tensor(updated, q_out)
    return q_synced, updated


def _legacy_vote_then_cap(
    state: EventCodedVoteUpdateState,
    inputs: VoteUpdateInputs,
    spec: VoteUpdateSpec,
    plan: VoteUpdatePlan,
    accepted_indices: Sequence[int],
    *,
    step_index: int,
) -> tuple[torch.Tensor, EventCodedAccLiveState]:
    vote_result = apply_event_coded_integer_vote_update_from_plan(
        state,
        inputs,
        spec,
        plan,
        step_index=int(step_index),
        lightweight_runtime_stats=True,
    )
    return _legacy_apply_event_coded_cap_mutations_with_cow_copy(
        vote_result.carrier,
        vote_result.q_levels,
        plan,
        accepted_indices,
        step_index=int(step_index),
    )


def _make_vote_state(
    *,
    numel: int,
    seed: int,
    hot_count: int,
) -> tuple[EventCodedVoteUpdateState, VoteUpdateInputs, VoteUpdateSpec]:
    rng = np.random.default_rng(int(seed))
    indices = np.sort(rng.choice(numel, size=hot_count, replace=False))
    values = rng.integers(-4, 5, size=hot_count, dtype=np.int16)
    carrier = EventCodedAccLiveState(
        logical_numel=int(numel),
        demotion_band=3,
        _hot=_PackedHotTable.from_arrays(indices, values),
    )
    q_levels = torch.zeros(numel, dtype=torch.int8)
    for flat_index in indices[: min(8, hot_count)]:
        carrier.q_levels[int(flat_index)] = 1 if int(values[list(indices).index(flat_index)]) >= 0 else -1
    votes = torch.zeros(numel, dtype=torch.int16)
    vote_active = indices[: min(16, hot_count)]
    for flat_index in vote_active:
        votes[int(flat_index)] = int(rng.integers(1, 12))
    spec = VoteUpdateSpec(
        threshold_abs=10,
        accumulator_clip_min=-127,
        accumulator_clip_max=127,
        decay_numerator=1,
        decay_denominator=2,
    )
    state = EventCodedVoteUpdateState(q_levels=q_levels, carrier=carrier)
    inputs = VoteUpdateInputs(votes=votes)
    return state, inputs, spec


def test_in_place_cap_mutations_bit_exact_vs_legacy_cow_copy() -> None:
    numel = 256
    applied = [5, 17, 42, 99]
    plan = _minimal_plan(applied_indices=applied, numel=numel)
    carrier_legacy_seed = EventCodedAccLiveState.with_hot_exact(
        logical_numel=numel,
        demotion_band=3,
        hot_exact={5: 12, 17: -11, 42: 3},
    )
    carrier_new_seed = carrier_legacy_seed.cow_copy()
    q = torch.zeros(numel, dtype=torch.int8)
    vote_carrier = carrier_legacy_seed.cow_copy()
    vote_carrier.apply_step(0, votes={5: 12, 17: -8})
    legacy_q, legacy_carrier = _legacy_apply_event_coded_cap_mutations_with_cow_copy(
        vote_carrier.cow_copy(),
        q,
        plan,
        applied,
        step_index=1,
    )
    new_vote_carrier = carrier_new_seed.cow_copy()
    new_vote_carrier.apply_step(0, votes={5: 12, 17: -8})
    new_q, new_carrier = _legacy_apply_event_coded_cap_mutations_with_cow_copy(
        new_vote_carrier,
        q.clone(),
        plan,
        applied,
        step_index=1,
    )
    # new path uses production in-place cap (no cow_copy between vote fork and cap)
    from calm.hrm_text_158.native_full_stack.event_coded_vote_update_adapter import (
        apply_event_coded_cap_mutations,
    )

    inplace_vote = carrier_new_seed.cow_copy()
    inplace_vote.apply_step(0, votes={5: 12, 17: -8})
    inplace_q, inplace_carrier = apply_event_coded_cap_mutations(
        inplace_vote,
        q.clone(),
        plan,
        applied,
        step_index=1,
    )
    legacy_snap = serialize_carrier_equivalence_snapshot(legacy_carrier, legacy_q)
    inplace_snap = serialize_carrier_equivalence_snapshot(inplace_carrier, inplace_q)
    assert inplace_snap == legacy_snap
    assert legacy_snap == serialize_carrier_equivalence_snapshot(new_carrier, new_q)


def test_merged_vote_and_cap_bit_exact_vs_legacy_multi_step() -> None:
    numel = 512
    state, inputs, spec = _make_vote_state(numel=numel, seed=43, hot_count=48)
    legacy_state = EventCodedVoteUpdateState(
        q_levels=state.q_levels.clone(),
        carrier=state.carrier.cow_copy(),
    )
    merged_state = EventCodedVoteUpdateState(
        q_levels=state.q_levels.clone(),
        carrier=state.carrier.cow_copy(),
    )
    rng = np.random.default_rng(43)
    applied_sets = [
        [int(x) for x in rng.choice(32, size=4, replace=False)],
        [int(x) for x in rng.choice(32, size=6, replace=False) + 64],
        [int(x) for x in rng.choice(32, size=3, replace=False) + 128],
        [],
        [int(x) for x in rng.choice(24, size=5, replace=False) + 200],
    ]
    for step_index, applied in enumerate(applied_sets):
        if not applied:
            plan = _minimal_plan(applied_indices=[step_index % 16], numel=numel)
            accepted: list[int] = []
        else:
            plan = _minimal_plan(applied_indices=applied, numel=numel)
            accepted = list(applied)
        legacy_q, legacy_carrier = _legacy_vote_then_cap(
            legacy_state,
            inputs,
            spec,
            plan,
            accepted,
            step_index=step_index,
        )
        legacy_state = EventCodedVoteUpdateState(
            q_levels=legacy_q,
            carrier=legacy_carrier,
        )
        merged_result = apply_event_coded_vote_and_cap_from_plan(
            merged_state,
            inputs,
            spec,
            plan,
            accepted,
            step_index=step_index,
            lightweight_runtime_stats=True,
        )
        merged_state = EventCodedVoteUpdateState(
            q_levels=merged_result.q_levels,
            carrier=merged_result.carrier,
        )
        legacy_snap = serialize_carrier_equivalence_snapshot(legacy_carrier, legacy_q)
        merged_snap = serialize_carrier_equivalence_snapshot(
            merged_result.carrier,
            merged_result.q_levels,
        )
        assert merged_snap == legacy_snap, f"step {step_index} diverged"


def test_merged_path_eliminates_redundant_cap_cow_copy() -> None:
    numel = 128
    applied = [3, 7, 11]
    plan = _minimal_plan(applied_indices=applied, numel=numel)
    carrier = EventCodedAccLiveState.with_hot_exact(
        logical_numel=numel,
        demotion_band=3,
        hot_exact={i: 12 for i in applied},
    )
    state = EventCodedVoteUpdateState(q_levels=torch.zeros(numel, dtype=torch.int8), carrier=carrier)
    inputs = VoteUpdateInputs(votes=torch.zeros(numel, dtype=torch.int16))
    spec = VoteUpdateSpec(
        threshold_abs=10,
        accumulator_clip_min=-127,
        accumulator_clip_max=127,
        decay_numerator=1,
        decay_denominator=2,
    )
    cow_copy_calls: list[int] = []

    original_cow_copy = EventCodedAccLiveState.cow_copy

    def counting_cow_copy(self: EventCodedAccLiveState) -> EventCodedAccLiveState:
        cow_copy_calls.append(id(self))
        return original_cow_copy(self)

    EventCodedAccLiveState.cow_copy = counting_cow_copy  # type: ignore[method-assign]
    try:
        apply_event_coded_vote_and_cap_from_plan(
            state,
            inputs,
            spec,
            plan,
            applied,
            step_index=0,
            lightweight_runtime_stats=True,
        )
    finally:
        EventCodedAccLiveState.cow_copy = original_cow_copy  # type: ignore[method-assign]
    assert len(cow_copy_calls) == 1
