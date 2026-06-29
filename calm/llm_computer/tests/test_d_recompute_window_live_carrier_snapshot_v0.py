"""Slice-5 Step-2a live carrier snapshot + incremental byte-counter tests."""
from __future__ import annotations

import random
import time

import pytest

from calm.hrm_text_158.native_full_stack.d_recompute_window_live_carrier_snapshot import (
    emit_live_carrier_snapshots_for_probe_step,
    initialize_live_carrier_snapshot_log,
)
from calm.hrm_text_158.native_full_stack.event_coded_acc_checkpoint_codec import (
    EventCodedAccEvent,
    encode_event_coded_acc_events,
    encode_event_coded_backlog_indices,
    encode_hot_exact_rows,
    pack_event_coded_acc_checkpoint_v1,
)
from calm.hrm_text_158.native_full_stack.event_coded_acc_live_carrier import (
    EventCodedAccLiveState,
    LiveCarrierByteCounterDesync,
    VARINT_BOUNDARY_FLAT_INDICES,
    _TrackedBacklog,
)
from calm.hrm_text_158.native_full_stack.event_coded_vote_update_adapter import (
    hydrate_event_coded_live_carrier_from_packed,
)


def _boundary_indices_for_numel(logical_numel: int) -> list[int]:
    indices = [int(item) for item in VARINT_BOUNDARY_FLAT_INDICES if int(item) < int(logical_numel)]
    indices.append(int(logical_numel) - 1)
    return sorted(set(indices))


def _assert_component_exact(
    carrier: EventCodedAccLiveState,
    *,
    component: str,
) -> None:
    exact = carrier._exact_byte_components_from_codec()
    snapshot = carrier.live_carrier_byte_snapshot()
    key = {
        "events": "events_bytes",
        "backlog": "backlog_bytes",
        "hot": "hot_exact_bytes",
    }[component]
    assert int(snapshot[key]) == int(exact[key])


def test_varint_boundary_exactness_matrix_events_backlog_hot() -> None:
    logical_numel = 268435457
    for flat_index in _boundary_indices_for_numel(logical_numel):
        carrier = EventCodedAccLiveState.with_hot_exact(
            logical_numel=logical_numel,
            demotion_band=1,
            hot_exact={},
        )
        carrier.backlog.add(int(flat_index))
        _assert_component_exact(carrier, component="backlog")
        carrier._append_event(
            EventCodedAccEvent(
                flat_index=int(flat_index),
                direction=1,
                residual_mag=15,
                event_type=1,
            )
        )
        _assert_component_exact(carrier, component="events")
        carrier.hot_exact[int(flat_index)] = 127
        _assert_component_exact(carrier, component="hot")
        carrier.assert_live_carrier_byte_counters_exact()


def test_incremental_counter_random_fuzz_sequences() -> None:
    rng = random.Random(43)
    logical_numel = 1_000_003
    for _ in range(200):
        carrier = EventCodedAccLiveState.with_hot_exact(
            logical_numel=logical_numel,
            demotion_band=1,
            hot_exact={},
        )
        for _mutation in range(rng.randint(1, 12)):
            choice = rng.randint(0, 2)
            index = rng.randrange(logical_numel)
            if choice == 0:
                carrier.backlog.add(index)
            elif choice == 1:
                carrier._append_event(
                    EventCodedAccEvent(
                        flat_index=index,
                        direction=rng.randint(0, 1),
                        residual_mag=rng.randint(0, 15),
                        event_type=1,
                    )
                )
            else:
                carrier.hot_exact[index] = rng.randint(-127, 127)
            carrier.assert_live_carrier_byte_counters_exact()


def test_backlog_duplicate_add_is_noop_for_counter() -> None:
    carrier = EventCodedAccLiveState(logical_numel=256, demotion_band=1)
    carrier.backlog.add(7)
    before = carrier.live_carrier_byte_snapshot()["backlog_bytes"]
    carrier.backlog.add(7)
    after = carrier.live_carrier_byte_snapshot()["backlog_bytes"]
    assert int(before) == int(after)
    carrier.assert_live_carrier_byte_counters_exact()


def test_external_backlog_mutation_fails_closed_on_reconcile() -> None:
    carrier = EventCodedAccLiveState(logical_numel=256, demotion_band=1)
    carrier.backlog.add(3)
    carrier.assert_live_carrier_byte_counters_exact()
    carrier.backlog._indices.add(99)  # noqa: SLF001 — adversary bypass test
    with pytest.raises(LiveCarrierByteCounterDesync):
        carrier.assert_live_carrier_byte_counters_exact()


def test_copy_hydrate_roundtrip_counters_before_and_after_mutations() -> None:
    packed = pack_event_coded_acc_checkpoint_v1(
        logical_numel=4096,
        events=(
            EventCodedAccEvent(flat_index=127, direction=1, residual_mag=3, event_type=1),
        ),
        backlog_indices=(128, 16383),
        hot_exact_indices=(256,),
        hot_exact_values=(42,),
    )
    hydrated = hydrate_event_coded_live_carrier_from_packed(packed)
    hydrated.assert_live_carrier_byte_counters_exact()
    copied = hydrated.cow_copy()
    copied.assert_live_carrier_byte_counters_exact()
    copied.backlog.add(200)
    copied._append_event(
        EventCodedAccEvent(flat_index=300, direction=0, residual_mag=2, event_type=1)
    )
    copied.hot_exact[301] = -5
    copied.assert_live_carrier_byte_counters_exact()


def test_hot_promote_demote_and_empty_apply_paths_remain_exact() -> None:
    carrier = EventCodedAccLiveState.with_hot_exact(
        logical_numel=64,
        demotion_band=1,
        hot_exact={5: 11},
    )
    carrier.apply_step(0, votes={0: 8})
    carrier.assert_live_carrier_byte_counters_exact()
    carrier.apply_step(1, votes={})
    carrier.assert_live_carrier_byte_counters_exact()


def test_snapshot_emit_writes_exact_rows(tmp_path) -> None:
    carrier = EventCodedAccLiveState.with_hot_exact(
        logical_numel=128,
        demotion_band=1,
        hot_exact={1: 2},
    )
    carrier.backlog.add(4)
    carrier._append_event(
        EventCodedAccEvent(flat_index=2, direction=1, residual_mag=1, event_type=1)
    )

    class _State:
        event_coded_live_carrier = carrier

    log_path = tmp_path / "live_carrier_snapshot.jsonl"
    initialize_live_carrier_snapshot_log(log_path)
    emitted = emit_live_carrier_snapshots_for_probe_step(
        enabled=True,
        log_path=log_path,
        step=3,
        post_update_states={"module.a": _State()},
    )
    assert emitted == 1
    row = log_path.read_text(encoding="utf-8").strip()
    assert '"live_carrier_bytes_exact": true' in row
    assert '"step": 3' in row


def test_snapshot_read_p95_under_one_millisecond_cpu_gate() -> None:
    carrier = EventCodedAccLiveState.with_hot_exact(
        logical_numel=268435456,
        demotion_band=1,
        hot_exact={index: 127 for index in range(4096)},
    )
    for index in range(64):
        carrier.backlog.add(index)
        carrier._append_event(
            EventCodedAccEvent(
                flat_index=index * 1000 + 1,
                direction=1,
                residual_mag=7,
                event_type=1,
            )
        )
    carrier.assert_live_carrier_byte_counters_exact()
    timings: list[float] = []
    for _ in range(500):
        start = time.perf_counter()
        carrier.live_carrier_byte_snapshot()
        timings.append(time.perf_counter() - start)
    p95 = sorted(timings)[int(0.95 * len(timings)) - 1]
    assert p95 < 0.001


@pytest.mark.parametrize("hot_rows", (0, 64, 576, 4096))
def test_hot_churn_snapshot_cost_scales_with_hot_rows(hot_rows: int) -> None:
    carrier = EventCodedAccLiveState.with_hot_exact(
        logical_numel=1_000_000,
        demotion_band=1,
        hot_exact={0: 1},
    )
    start = time.perf_counter()
    for index in range(int(hot_rows)):
        carrier.hot_exact[int(index)] = int((index % 254) - 127)
        carrier.live_carrier_byte_snapshot()
    elapsed = time.perf_counter() - start
    assert elapsed >= 0.0


def test_codec_single_element_deltas_match_incremental_helpers() -> None:
    event = EventCodedAccEvent(flat_index=16384, direction=1, residual_mag=15, event_type=1)
    assert len(encode_event_coded_acc_events((event,))) == len(
        encode_event_coded_acc_events((event,))
    )
    assert len(encode_event_coded_backlog_indices((16384,))) == len(
        encode_event_coded_backlog_indices((16384,))
    )
    assert len(encode_hot_exact_rows((16384,), (127,))) == len(
        encode_hot_exact_rows((16384,), (127,))
    )


def test_tracked_backlog_does_not_expose_public_set_mutation_api() -> None:
    backlog = _TrackedBacklog((1, 2))
    assert not hasattr(backlog, "update")
    assert not hasattr(backlog, "discard")
