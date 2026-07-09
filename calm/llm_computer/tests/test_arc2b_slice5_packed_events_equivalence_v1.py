"""Slice-5 packed event-store equivalence (codec-byte live facade).

Frozen +1: 1783547797755 / 1783547802486. Folds A+B pinned.
Fail-closed: sha/bytes/record mismatch or materialize-on-export → block.
"""

from __future__ import annotations

import hashlib
import sys
from typing import Any

import numpy as np
import torch

from calm.hrm_text_158.native_full_stack.event_coded_acc_checkpoint_codec import (
    EventCodedAccEvent,
    encode_event_coded_acc_events,
    pack_event_coded_acc_checkpoint_v1,
    pack_event_coded_acc_checkpoint_v1_from_packed_events,
    unpack_event_coded_acc_checkpoint_reference,
)
from calm.hrm_text_158.native_full_stack.event_coded_acc_event_store import (
    EventCodedAccEventStore,
)
from calm.hrm_text_158.native_full_stack.event_coded_acc_live_carrier import (
    EventCodedAccLiveState,
    StepSurfaceRecord,
    _PackedHotTable,
)
from calm.hrm_text_158.native_full_stack.event_coded_vote_update_adapter import (
    EventCodedVoteUpdateState,
    apply_event_coded_vote_and_cap_from_plan,
    carrier_content_sha256,
    hydrate_event_coded_live_carrier_from_packed,
)
from calm.hrm_text_158.native_full_stack.vote_update import (
    VoteUpdateInputs,
    VoteUpdatePlan,
    VoteUpdateSpec,
)


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
    # Decisive-record contract: record.q_levels is applied∪crossing only.
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


def _assert_decisive_record_contract(
    record: StepSurfaceRecord,
    live_q_levels: dict[int, int],
) -> None:
    decisive = {int(i) for i in record.applied_indices} | {
        int(i) for i in record.crossing_indices
    }
    assert set(record.q_levels.keys()) == decisive
    for index, value in record.q_levels.items():
        assert int(value) == int(live_q_levels.get(int(index), 0))


def _carrier_snap(carrier: EventCodedAccLiveState, q_levels: torch.Tensor) -> dict[str, Any]:
    decoded = tuple(carrier.events)
    return {
        "carrier_sha256": carrier_content_sha256(carrier),
        "q_sha256": hashlib.sha256(q_levels.detach().cpu().numpy().tobytes()).hexdigest(),
        "step_records": [_record_tuple(r) for r in carrier.step_records],
        # DIRECT live-q coverage (full dense map) — not via record.q_levels.
        "live_q_levels_dict": dict(carrier.q_levels),
        "events": decoded,
        "events_bytes": carrier._event_store.encode_bytes(),
        "events_len": len(carrier.events),
        "backlog": tuple(sorted(int(i) for i in carrier.backlog)),
        "q_levels_dict": dict(carrier.q_levels),
        "hot_exact": dict(carrier.hot_exact),
        "byte_snapshot": {
            k: carrier.live_carrier_byte_snapshot()[k]
            for k in (
                "events_bytes",
                "backlog_bytes",
                "hot_exact_bytes",
                "live_acc_carrier_bytes_total",
            )
        },
    }


def _make_events(n: int, *, seed: int = 0) -> tuple[EventCodedAccEvent, ...]:
    rng = np.random.default_rng(seed)
    out: list[EventCodedAccEvent] = []
    for i in range(n):
        out.append(
            EventCodedAccEvent(
                flat_index=int(rng.integers(0, 1_000_000)),
                direction=int(rng.integers(0, 2)),
                residual_mag=int(rng.integers(0, 15)),
                event_type=int(rng.integers(0, 3)),
            )
        )
    return tuple(out)


def test_codec_concat_associativity() -> None:
    events = _make_events(64, seed=7)
    batch = encode_event_coded_acc_events(events)
    joined = b"".join(encode_event_coded_acc_events((e,)) for e in events)
    assert batch == joined
    store = EventCodedAccEventStore.empty()
    for e in events:
        store.append(e)
    assert store.encode_bytes() == batch
    assert len(store) == len(events)


def test_fold_a_no_materialize_export_byte_equal_legacy() -> None:
    events = _make_events(48, seed=11)
    carrier = EventCodedAccLiveState(
        logical_numel=1_048_576,
        events=events,
        backlog={1, 9, 42},
        _hot=_PackedHotTable.from_dict({3: 7, 9: -2}),
    )
    carrier._event_store.reset_materialize_count()
    before = carrier._event_store.materialize_count

    class _ForbidIter(EventCodedAccEventStore):
        def __iter__(self):  # type: ignore[override]
            raise AssertionError("export must not iterate/materialize events")

        def as_tuple(self):  # type: ignore[override]
            raise AssertionError("export must not as_tuple/materialize events")

    # Swap store methods on the live instance to catch accidental materialize.
    store = carrier._event_store
    orig_iter = store.__class__.__iter__
    orig_as_tuple = store.__class__.as_tuple

    def boom_iter(self):  # noqa: ANN001
        raise AssertionError("__iter__ forbidden during export")

    def boom_as_tuple(self):  # noqa: ANN001
        raise AssertionError("as_tuple forbidden during export")

    store.__class__.__iter__ = boom_iter  # type: ignore[method-assign]
    store.__class__.as_tuple = boom_as_tuple  # type: ignore[method-assign]
    try:
        packed = carrier.to_checkpoint_payload()
    finally:
        store.__class__.__iter__ = orig_iter  # type: ignore[method-assign]
        store.__class__.as_tuple = orig_as_tuple  # type: ignore[method-assign]

    assert carrier._event_store.materialize_count == before
    legacy = pack_event_coded_acc_checkpoint_v1(
        logical_numel=int(carrier.logical_numel),
        events=events,
        backlog_indices=tuple(sorted(carrier.backlog)),
        hot_exact_indices=carrier._hot.indices_array(),
        hot_exact_values=carrier._hot.values_array(),
    )
    assert bytes(packed.events_packed.tolist()) == bytes(legacy.events_packed.tolist())
    assert int(packed.event_count) == len(events) == int(legacy.event_count)
    assert bytes(packed.backlog_packed.tolist()) == bytes(legacy.backlog_packed.tolist())
    assert bytes(packed.hot_exact_packed.tolist()) == bytes(legacy.hot_exact_packed.tolist())

    # Factory path itself never needs Event shells.
    factory = pack_event_coded_acc_checkpoint_v1_from_packed_events(
        logical_numel=int(carrier.logical_numel),
        events_bytes=carrier._event_store.encode_bytes(),
        event_count=len(carrier._event_store),
        backlog_indices=tuple(sorted(carrier.backlog)),
        hot_exact_indices=carrier._hot.indices_array(),
        hot_exact_values=carrier._hot.values_array(),
    )
    assert bytes(factory.events_packed.tolist()) == bytes(legacy.events_packed.tolist())


def test_fold_b_events_append_coherence_and_cow_independence() -> None:
    carrier = EventCodedAccLiveState(logical_numel=64)
    e0 = EventCodedAccEvent(flat_index=2, direction=1, residual_mag=3, event_type=1)
    carrier.events.append(e0)
    assert len(carrier.events) == 1
    assert len(carrier._event_store) == 1
    assert carrier._live_carrier_events_bytes == len(carrier._event_store.encode_bytes())
    assert carrier._event_store.encode_bytes() == encode_event_coded_acc_events((e0,))

    child = carrier.cow_copy()
    e1 = EventCodedAccEvent(flat_index=5, direction=0, residual_mag=1, event_type=0)
    child.events.append(e1)
    assert len(carrier.events) == 1
    assert len(child.events) == 2
    assert carrier._event_store.encode_bytes() == encode_event_coded_acc_events((e0,))
    assert child._event_store.encode_bytes() == encode_event_coded_acc_events((e0, e1))
    assert carrier._live_carrier_events_bytes == len(encode_event_coded_acc_events((e0,)))
    assert child._live_carrier_events_bytes == len(encode_event_coded_acc_events((e0, e1)))


def test_dual_path_bit_exact_vote_cap_and_checkpoint_rt() -> None:
    # List-oracle vs incremental packed store (pre-facade HEAD semantics).
    list_events: list[EventCodedAccEvent] = []
    store = EventCodedAccEventStore.empty()
    for e in _make_events(80, seed=21):
        list_events.append(e)
        store.append(e)
    assert store.encode_bytes() == encode_event_coded_acc_events(tuple(list_events))
    assert store.as_tuple() == tuple(list_events)

    numel = 256
    seed_hot = {5: 12, 17: -11, 42: 3, 99: 8}
    applied_sets = [
        [5, 17, 42],
        [],
        [99, 5],
        [17],
        [3, 7, 11],  # sparse / not all hot
    ]
    packed_carrier = EventCodedAccLiveState.with_hot_exact(
        logical_numel=numel,
        demotion_band=3,
        hot_exact=dict(seed_hot),
    )
    twin_carrier = EventCodedAccLiveState.with_hot_exact(
        logical_numel=numel,
        demotion_band=3,
        hot_exact=dict(seed_hot),
    )

    q_packed = torch.zeros(numel, dtype=torch.int8)
    q_twin = torch.zeros(numel, dtype=torch.int8)
    state_p = EventCodedVoteUpdateState(q_levels=q_packed, carrier=packed_carrier)
    state_t = EventCodedVoteUpdateState(q_levels=q_twin, carrier=twin_carrier)
    inputs = VoteUpdateInputs(votes=torch.zeros(numel, dtype=torch.int16))
    spec = VoteUpdateSpec(
        threshold_abs=10,
        accumulator_clip_min=-127,
        accumulator_clip_max=127,
        decay_numerator=1,
        decay_denominator=2,
    )

    for step_index, applied in enumerate(applied_sets):
        plan = _minimal_plan(
            applied_indices=applied if applied else [step_index % 16],
            numel=numel,
        )
        accepted = list(applied)
        vote_map = {int(i): 12 for i in (applied[:2] if applied else [])}
        if vote_map:
            state_p.carrier.apply_step(step_index, votes=vote_map)
            state_t.carrier.apply_step(step_index, votes=vote_map)
        res_p = apply_event_coded_vote_and_cap_from_plan(
            state_p,
            inputs,
            spec,
            plan,
            accepted,
            step_index=step_index,
            lightweight_runtime_stats=True,
        )
        res_t = apply_event_coded_vote_and_cap_from_plan(
            state_t,
            inputs,
            spec,
            plan,
            accepted,
            step_index=step_index,
            lightweight_runtime_stats=True,
        )
        state_p = EventCodedVoteUpdateState(
            q_levels=res_p.q_levels, carrier=res_p.carrier
        )
        state_t = EventCodedVoteUpdateState(
            q_levels=res_t.q_levels, carrier=res_t.carrier
        )
        snap_p = _carrier_snap(res_p.carrier, res_p.q_levels)
        snap_t = _carrier_snap(res_t.carrier, res_t.q_levels)
        assert snap_p == snap_t, f"step {step_index} diverged"
        # DIRECT live-q + decisive-record contract (Item2 Class-A fold).
        # Historical records keep decisive keys only; value==live only for latest.
        assert snap_p["live_q_levels_dict"] == dict(res_p.carrier.q_levels)
        for record in res_p.carrier.step_records:
            decisive = {int(i) for i in record.applied_indices} | {
                int(i) for i in record.crossing_indices
            }
            assert set(record.q_levels.keys()) == decisive
        if res_p.carrier.step_records:
            _assert_decisive_record_contract(
                res_p.carrier.step_records[-1],
                res_p.carrier.q_levels,
            )
        # decode==legacy encode (list semantics)
        decoded = tuple(res_p.carrier.events)
        assert res_p.carrier._event_store.encode_bytes() == encode_event_coded_acc_events(
            decoded
        )

    # Checkpoint round-trip (packed path).
    payload = state_p.carrier.to_checkpoint_payload()
    hydrated = hydrate_event_coded_live_carrier_from_packed(payload)
    assert carrier_content_sha256(hydrated) == carrier_content_sha256(state_p.carrier)
    assert len(hydrated.events) == len(state_p.carrier.events)
    assert tuple(hydrated.events) == tuple(state_p.carrier.events)
    decoded_ref, _backlog = unpack_event_coded_acc_checkpoint_reference(payload)
    assert decoded_ref == tuple(state_p.carrier.events)

    # cow_copy then apply_step(votes={}) divergence-free vs sibling.
    a = state_p.carrier.cow_copy()
    b = state_p.carrier.cow_copy()
    a.apply_step(99, votes={})
    assert carrier_content_sha256(b) == carrier_content_sha256(state_p.carrier)
    b.apply_step(99, votes={})
    assert carrier_content_sha256(a) == carrier_content_sha256(b)


def test_adversarial_empty_sparse_multi_carrier_cow_fork() -> None:
    carriers = [
        EventCodedAccLiveState(logical_numel=32),
        EventCodedAccLiveState.with_hot_exact(
            logical_numel=64, hot_exact={1: 5, 2: -3}, demotion_band=2
        ),
        EventCodedAccLiveState(
            logical_numel=128,
            events=_make_events(5, seed=3),
            backlog=set(),
        ),
    ]
    # empty votes
    carriers[0].apply_step(0, votes={})
    assert len(carriers[0].events) == 0
    # sparse crossing-ish votes
    carriers[1].apply_step(1, votes={1: 20, 2: -20, 50: 3})
    assert len(carriers[1].events) >= 1
    # multi-carrier independence after cow
    parent = carriers[2]
    child = parent.cow_copy()
    child.apply_step(2, votes={10: 15})
    assert len(parent.events) == 5
    assert len(child.events) >= 5
    assert parent._event_store.encode_bytes() != child._event_store.encode_bytes() or len(
        child.events
    ) == len(parent.events)


def _deep_sizeof(obj: Any, *, _seen: set[int] | None = None) -> int:
    """Shallow+attrs deep size (same spirit as amp attribution deep_sizeof)."""
    seen = _seen if _seen is not None else set()
    oid = id(obj)
    if oid in seen:
        return 0
    seen.add(oid)
    size = sys.getsizeof(obj)
    if isinstance(obj, dict):
        for k, v in obj.items():
            size += _deep_sizeof(k, _seen=seen) + _deep_sizeof(v, _seen=seen)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for item in obj:
            size += _deep_sizeof(item, _seen=seen)
    elif hasattr(obj, "__dict__"):
        size += _deep_sizeof(vars(obj), _seen=seen)
    elif hasattr(obj, "__slots__"):
        for name in obj.__slots__:
            if hasattr(obj, name):
                size += _deep_sizeof(getattr(obj, name), _seen=seen)
    return int(size)


def test_footprint_reduction_at_100k_events() -> None:
    n = 100_000
    events = _make_events(n, seed=99)
    legacy_list = list(events)
    legacy_bytes = _deep_sizeof(legacy_list)
    store = EventCodedAccEventStore.from_events(events)
    packed_payload = len(store.encode_bytes())
    # Live retained cost ≈ packed codec bytes (+ tiny store shell); do not count
    # decoded Event shells (lazy).
    packed_bytes = packed_payload + sys.getsizeof(store)
    ratio = float(legacy_bytes) / float(max(1, packed_bytes))
    assert n >= 100_000
    assert ratio >= 20.0, (
        f"footprint ratio {ratio:.2f} < 20 (legacy={legacy_bytes} packed={packed_bytes})"
    )
    carrier = EventCodedAccLiveState(logical_numel=max(n + 1, 1_048_576), events=store)
    carrier._event_store.reset_materialize_count()
    assert len(carrier._event_store.encode_bytes()) == packed_payload
    assert carrier._event_store.materialize_count == 0
    assert len(carrier.events) == n


def test_hidden_materialization_absent_on_sha_and_cow() -> None:
    carrier = EventCodedAccLiveState(
        logical_numel=128,
        events=_make_events(200, seed=5),
    )
    carrier._event_store.reset_materialize_count()
    _ = carrier_content_sha256(carrier)
    assert carrier._event_store.materialize_count == 0
    child = carrier.cow_copy()
    assert carrier._event_store.materialize_count == 0
    assert child._event_store.materialize_count == 0
    _ = child.to_checkpoint_payload()
    assert child._event_store.materialize_count == 0
    snap = child.live_carrier_byte_snapshot()
    assert int(snap["events_bytes"]) == len(child._event_store.encode_bytes())
    assert child._event_store.materialize_count == 0
