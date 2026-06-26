"""Terminal-cardinality 32-module carrier growth sidecar screen (Phase 1)."""
from __future__ import annotations

import time
from typing import Any
from unittest import mock

import pytest

from calm.hrm_text_158.native_full_stack.carrier_growth_summary import (
    build_carrier_growth_module_row,
    build_carrier_growth_step_record_compact,
    sidecar_sha256,
)
from calm.hrm_text_158.native_full_stack.event_coded_acc_live_carrier import (
    EventCodedAccLiveState,
)
from calm.llm_computer.tests.carrier_growth_terminal_fixture_v0 import (
    build_terminal_history_carrier_fixture,
    terminal_fixture_receipt_dict,
    terminal_fixture_spec,
)

SIDECAR_BYTES_CAP = 10_240
SIDECAR_WALL_SECONDS_CAP = 180.0
MODULE_COUNT = 32


def _assert_no_raw_index_arrays(payload: Any, *, path: str = "") -> None:
    if isinstance(payload, list):
        if payload and all(isinstance(item, int) for item in payload):
            raise AssertionError(f"raw index array at {path or '<root>'}")
        for index, item in enumerate(payload):
            child = f"{path}[{index}]" if path else f"[{index}]"
            _assert_no_raw_index_arrays(item, path=child)
    elif isinstance(payload, dict):
        for key, value in payload.items():
            child = f"{path}.{key}" if path else str(key)
            _assert_no_raw_index_arrays(value, path=child)


@pytest.mark.slow
def test_carrier_growth_sidecar_32module_terminal_screen_v0() -> None:
    spec = terminal_fixture_spec()
    encode_patch = (
        "calm.hrm_text_158.native_full_stack.event_coded_acc_checkpoint_codec"
        ".encode_event_coded_acc_events"
    )
    per_module_walls: list[float] = []
    module_rows: list[dict[str, Any]] = []

    carrier, step_record, votes, receipt = build_terminal_history_carrier_fixture(seed=44)
    index_receipt = terminal_fixture_receipt_dict(receipt)

    started_total = time.perf_counter()
    for module_index in range(MODULE_COUNT):
        with mock.patch(encode_patch) as encode_mock, mock.patch.object(
            EventCodedAccLiveState,
            "hot_packed_bytes",
            autospec=True,
        ) as hot_pack_mock:
            module_started = time.perf_counter()
            module_rows.append(
                build_carrier_growth_module_row(
                    state_key=f"mod{module_index:02d}",
                    carrier=carrier,
                    step_record=step_record,
                    votes=votes,
                    cap_accepted_rows=0,
                    q_changed_rows=0,
                )
            )
            per_module_walls.append(time.perf_counter() - module_started)
            encode_mock.assert_not_called()
            hot_pack_mock.assert_not_called()

    hook_started = time.perf_counter()
    sidecar = build_carrier_growth_step_record_compact(
        optimizer_step_index=20,
        module_rows=module_rows,
    )
    compact_wall = time.perf_counter() - hook_started
    total_wall = time.perf_counter() - started_total

    _assert_no_raw_index_arrays(sidecar)
    assert "modules" not in sidecar
    assert sidecar["compact"] is True
    sidecar_sha = sidecar_sha256(sidecar)
    second = build_carrier_growth_step_record_compact(
        optimizer_step_index=20,
        module_rows=module_rows,
    )
    assert sidecar_sha256(second) == sidecar_sha

    assert index_receipt["all_indices_lt_logical_numel"] is True
    assert index_receipt["event_count_target"] == spec.event_count_target
    assert index_receipt["hot_row_count_target"] == spec.hot_row_count_target
    assert index_receipt["logical_numel"] == spec.logical_numel

    sidecar_bytes = int(sidecar["rollup"]["sidecar_bytes"])
    assert sidecar_bytes <= SIDECAR_BYTES_CAP, (
        f"terminal screen sidecar_bytes={sidecar_bytes} exceeds cap {SIDECAR_BYTES_CAP}"
    )
    assert total_wall <= SIDECAR_WALL_SECONDS_CAP, (
        f"terminal screen total_wall={total_wall:.3f}s exceeds {SIDECAR_WALL_SECONDS_CAP}s "
        f"(per_module_max={max(per_module_walls):.3f}s compact={compact_wall:.3f}s "
        f"index_receipt={index_receipt})"
    )
