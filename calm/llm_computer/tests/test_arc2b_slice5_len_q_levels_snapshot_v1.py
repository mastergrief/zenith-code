"""Stage A LEN-LOG: passive global len(q_levels) snapshot fields + unit witness."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from calm.hrm_text_158.native_full_stack.d_recompute_window_live_carrier_snapshot import (
    DuplicateLiveCarrierObjectIdError,
    GLOBAL_Q_LEVELS_DOMINANCE_THRESHOLD,
    PerModuleQLevelsThresholdCompareError,
    assert_global_q_levels_threshold_unit,
    collect_q_levels_length_pass,
    emit_live_carrier_snapshots_for_probe_step,
    initialize_live_carrier_snapshot_log,
)
from calm.hrm_text_158.native_full_stack.event_coded_acc_live_carrier import (
    EventCodedAccLiveState,
)

EMITTER_PATH = (
    Path(__file__).resolve().parents[2]
    / "hrm_text_158"
    / "native_full_stack"
    / "d_recompute_window_live_carrier_snapshot.py"
)


class _State:
    def __init__(self, carrier: EventCodedAccLiveState) -> None:
        self.event_coded_live_carrier = carrier


def _carrier_with_q(n_keys: int, *, logical_numel: int = 10_000) -> EventCodedAccLiveState:
    carrier = EventCodedAccLiveState(logical_numel=logical_numel, demotion_band=1)
    for i in range(int(n_keys)):
        carrier.q_levels[i] = 1 if (i % 2) == 0 else -1
    return carrier


def test_multi_carrier_global_equals_sum_and_identical_on_every_row(tmp_path: Path) -> None:
    a = _carrier_with_q(10)
    b = _carrier_with_q(25)
    c = _carrier_with_q(7)
    states = {
        "mod.a": _State(a),
        "mod.b": _State(b),
        "mod.c": _State(c),
    }
    log_path = tmp_path / "live_carrier_snapshot.jsonl"
    initialize_live_carrier_snapshot_log(log_path)
    emitted = emit_live_carrier_snapshots_for_probe_step(
        enabled=True,
        log_path=log_path,
        step=5,
        post_update_states=states,
    )
    assert emitted == 3
    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 3
    expected_global = 10 + 25 + 7
    globals_seen = {int(r["global_len_q_levels"]) for r in rows}
    assert globals_seen == {expected_global}
    by_key = {r["state_key"]: int(r["len_q_levels"]) for r in rows}
    assert by_key == {"mod.a": 10, "mod.b": 25, "mod.c": 7}
    assert sum(by_key.values()) == expected_global


def test_duplicate_carrier_object_id_fail_closed(tmp_path: Path) -> None:
    shared = _carrier_with_q(3)
    states = {
        "mod.a": _State(shared),
        "mod.b": _State(shared),  # same object id
    }
    log_path = tmp_path / "live_carrier_snapshot.jsonl"
    initialize_live_carrier_snapshot_log(log_path)
    with pytest.raises(DuplicateLiveCarrierObjectIdError):
        emit_live_carrier_snapshots_for_probe_step(
            enabled=True,
            log_path=log_path,
            step=1,
            post_update_states=states,
        )
    # Fail-closed before write: log stays empty.
    assert log_path.read_text(encoding="utf-8").strip() == ""


def test_emit_does_not_mutate_q_levels_identity_or_len(tmp_path: Path) -> None:
    carrier = _carrier_with_q(12)
    q_id_before = id(carrier.q_levels)
    len_before = len(carrier.q_levels)
    keys_before = set(carrier.q_levels.keys())
    log_path = tmp_path / "live_carrier_snapshot.jsonl"
    initialize_live_carrier_snapshot_log(log_path)
    emit_live_carrier_snapshots_for_probe_step(
        enabled=True,
        log_path=log_path,
        step=2,
        post_update_states={"mod.a": _State(carrier)},
    )
    assert id(carrier.q_levels) == q_id_before
    assert len(carrier.q_levels) == len_before
    assert set(carrier.q_levels.keys()) == keys_before


def test_collect_pass_matches_emit_global() -> None:
    a = _carrier_with_q(4)
    b = _carrier_with_q(6)
    states = {"x": _State(a), "y": _State(b)}
    per_key, global_len = collect_q_levels_length_pass(states)
    assert per_key == {"x": 4, "y": 6}
    assert global_len == 10


def test_unit_witness_rejects_per_module_threshold_compare() -> None:
    with pytest.raises(PerModuleQLevelsThresholdCompareError):
        assert_global_q_levels_threshold_unit(
            value=100,
            field_name="len_q_levels",
            threshold=GLOBAL_Q_LEVELS_DOMINANCE_THRESHOLD,
        )
    # Allowed global field names pass.
    assert_global_q_levels_threshold_unit(
        value=5_000_000,
        field_name="global_len_q_levels",
    )
    assert_global_q_levels_threshold_unit(
        value=5_000_000,
        field_name="explicit_sum_of_per_module_len_q_levels",
    )


def test_source_banned_calls_on_new_q_levels_observer() -> None:
    """NEW q_levels observer must not materialize/copy q_levels."""

    src = EMITTER_PATH.read_text(encoding="utf-8")
    # Narrow to the Stage A helper + emit body (file is small; still be explicit).
    banned = [
        r"dict\(\s*carrier\.q_levels\s*\)",
        r"list\(\s*carrier\.q_levels",
        r"carrier\.q_levels\.copy\s*\(",
        r"carrier\.q_levels\.keys\s*\(",
        r"carrier\.q_levels\.values\s*\(",
        r"carrier\.q_levels\.items\s*\(",
        r"deep_sizeof\s*\(\s*carrier\.q_levels",
    ]
    for pattern in banned:
        assert re.search(pattern, src) is None, f"banned pattern present: {pattern}"
    # Positive: len(carrier.q_levels) is the allowed observer.
    assert "len(carrier.q_levels)" in src
