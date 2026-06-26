"""Manifest-calibrated terminal-history carrier fixture for Phase 1 cost screens."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from calm.hrm_text_158.native_full_stack.event_coded_acc_checkpoint_codec import (
    EventCodedAccEvent,
)
from calm.hrm_text_158.native_full_stack.event_coded_acc_live_carrier import (
    EventCodedAccLiveState,
    StepSurfaceRecord,
    _PackedHotTable,
)

MANIFEST_PATH = (
    Path(__file__).resolve().parents[3]
    / "artifacts"
    / "consensus_prep"
    / "v4_live_phase_a_diagnostic_tier1_run_2189e72004_evidence_manifest.json"
)

# Calibrated from manifest L78-86 (worst-module payload bytes @ 4B/event, 5B/hot-row).
EVENT_PAYLOAD_BYTES_MAX = 10_743_901
HOT_PAYLOAD_BYTES_MAX = 6_265_242
EVENT_COUNT_TARGET = EVENT_PAYLOAD_BYTES_MAX // 4
HOT_ROW_COUNT_TARGET = HOT_PAYLOAD_BYTES_MAX // 5
LOGICAL_NUMEL_TARGET = max(EVENT_COUNT_TARGET, HOT_ROW_COUNT_TARGET) + 10_000


@dataclass(frozen=True)
class TerminalFixtureSpec:
    event_count_target: int
    hot_row_count_target: int
    logical_numel: int
    manifest_path: str


@dataclass(frozen=True)
class TerminalFixtureReceipt:
    event_count_target: int
    hot_row_count_target: int
    logical_numel: int
    max_hot_index: int
    max_event_index: int
    all_indices_lt_logical_numel: bool


def terminal_fixture_spec() -> TerminalFixtureSpec:
    return TerminalFixtureSpec(
        event_count_target=int(EVENT_COUNT_TARGET),
        hot_row_count_target=int(HOT_ROW_COUNT_TARGET),
        logical_numel=int(LOGICAL_NUMEL_TARGET),
        manifest_path=str(MANIFEST_PATH),
    )


def _build_events(*, event_count: int, unique_lanes: int) -> list[EventCodedAccEvent]:
    events: list[EventCodedAccEvent] = []
    chunk = 250_000
    for start in range(0, int(event_count), chunk):
        end = min(int(event_count), start + chunk)
        events.extend(
            EventCodedAccEvent(
                flat_index=int(i % unique_lanes),
                direction=1,
                residual_mag=1,
                event_type=0,
            )
            for i in range(start, end)
        )
    return events


def build_terminal_history_carrier_fixture(
    *,
    event_count: int | None = None,
    hot_row_count: int | None = None,
    logical_numel: int | None = None,
    seed: int = 44,
) -> tuple[EventCodedAccLiveState, StepSurfaceRecord, torch.Tensor, TerminalFixtureReceipt]:
    spec = terminal_fixture_spec()
    event_count_target = int(event_count if event_count is not None else spec.event_count_target)
    hot_row_count_target = int(
        hot_row_count if hot_row_count is not None else spec.hot_row_count_target
    )
    logical_numel_value = int(logical_numel if logical_numel is not None else spec.logical_numel)
    if logical_numel_value < hot_row_count_target:
        raise ValueError(
            "logical_numel must be >= hot_row_count_target "
            f"({logical_numel_value} < {hot_row_count_target})"
        )

    unique_lanes = max(1, int(round(event_count_target * 0.7)))
    if unique_lanes >= logical_numel_value:
        unique_lanes = logical_numel_value - 1

    hot_indices = np.arange(hot_row_count_target, dtype=np.int32)
    magnitudes = ((hot_indices + int(seed)) % 11).astype(np.int16)
    magnitudes[magnitudes > 5] -= 11
    hot_table = _PackedHotTable.from_arrays(hot_indices, magnitudes)

    events = _build_events(event_count=event_count_target, unique_lanes=unique_lanes)
    q_locked_through = int(unique_lanes * 0.6)
    q_levels = {int(i): int((i + seed) % 3) for i in range(q_locked_through)}

    carrier = EventCodedAccLiveState(
        logical_numel=logical_numel_value,
        demotion_band=1,
        events=events,
        q_levels=q_levels,
    )
    carrier._hot = hot_table  # noqa: SLF001 — test fixture uses packed hot table directly.

    max_event_index = max((int(event.flat_index) for event in carrier.events), default=-1)
    max_hot_index = int(hot_indices[-1]) if hot_indices.size else -1
    max_vote_index = logical_numel_value - 1
    all_indices_lt = (
        max_event_index < logical_numel_value
        and max_hot_index < logical_numel_value
        and max_vote_index < logical_numel_value
    )
    receipt = TerminalFixtureReceipt(
        event_count_target=event_count_target,
        hot_row_count_target=hot_row_count_target,
        logical_numel=logical_numel_value,
        max_hot_index=max_hot_index,
        max_event_index=max_event_index,
        all_indices_lt_logical_numel=bool(all_indices_lt),
    )
    if not all_indices_lt:
        raise ValueError(f"terminal fixture index realism failed: {receipt}")

    generator = torch.Generator().manual_seed(int(seed))
    votes = torch.zeros(logical_numel_value, dtype=torch.int16)
    mask = torch.rand(logical_numel_value, generator=generator) < 0.667
    nonzero = int(mask.sum().item())
    if nonzero:
        votes[mask] = torch.randint(1, 13, (nonzero,), dtype=torch.int16, generator=generator)

    step_record = StepSurfaceRecord(
        step_index=19,
        crossing_indices=tuple(range(min(128, unique_lanes))),
        applied_indices=tuple(range(min(64, unique_lanes))),
        backlog_indices=(),
        q_levels=dict(q_levels),
        hot_exact_row_count=int(hot_row_count_target),
        promotion_count=0,
        demotion_on_decay_count=0,
        demotion_on_crossing_count=0,
    )
    carrier.step_records.append(step_record)
    return carrier, step_record, votes, receipt


def terminal_fixture_receipt_dict(receipt: TerminalFixtureReceipt) -> dict[str, Any]:
    return {
        "event_count_target": int(receipt.event_count_target),
        "hot_row_count_target": int(receipt.hot_row_count_target),
        "logical_numel": int(receipt.logical_numel),
        "max_hot_index": int(receipt.max_hot_index),
        "max_event_index": int(receipt.max_event_index),
        "all_indices_lt_logical_numel": bool(receipt.all_indices_lt_logical_numel),
    }
