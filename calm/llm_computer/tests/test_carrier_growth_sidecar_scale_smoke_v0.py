"""CPU scale-smoke for carrier growth summary sidecar (Phase 0 stub)."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
import torch

from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    make_event_coded_live_tensor_state,
    make_live_event_coded_tensor_state,
)
from calm.hrm_text_158.native_full_stack.carrier_growth_summary import (
    EST_BYTES_PER_EVENT,
    EST_BYTES_PER_HOT_ROW,
    build_carrier_growth_module_row,
    build_carrier_growth_step_record,
    estimate_events_payload_bytes,
    estimate_hot_exact_payload_bytes,
    sidecar_sha256,
)
from calm.hrm_text_158.native_full_stack.event_coded_acc_live_carrier import (
    EventCodedAccLiveState,
)
from calm.hrm_text_158.native_full_stack.event_coded_vote_update_adapter import (
    EventCodedVoteUpdateState,
    apply_event_coded_integer_vote_update_reference,
)
from calm.hrm_text_158.native_full_stack.vote_update import VoteUpdateInputs, VoteUpdateSpec

SIDECAR_BYTES_CAP = 10_240
SIDECAR_WALL_SECONDS_CAP = 2.0
NUMEL = 2048 * 512  # 1,048,576 (~1.05M)
NUM_STEPS = 20


def _vote_spec() -> VoteUpdateSpec:
    return VoteUpdateSpec(
        threshold_abs=10,
        accumulator_clip_min=-127,
        accumulator_clip_max=127,
        max_abs_per_tensor=16,
    )


def _votes_at_density(numel: int, *, density: float, seed: int) -> torch.Tensor:
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


def _q_changed_count(before_q: torch.Tensor, after_q: torch.Tensor) -> int:
    return int((after_q != before_q).sum().item())


def _apply_steps_and_emit_sidecars(
    *,
    density: float,
    seed: int,
) -> tuple[list[str], list[int], list[float]]:
    key = "mod0"
    q = torch.randint(-1, 2, (NUMEL,), dtype=torch.int8, generator=torch.Generator().manual_seed(seed))
    state = make_event_coded_live_tensor_state(key, q, 1.0, demotion_band=1)
    spec = _vote_spec()
    shas: list[str] = []
    byte_sizes: list[int] = []
    wall_seconds: list[float] = []

    for step in range(NUM_STEPS):
        votes = _votes_at_density(NUMEL, density=density, seed=seed + step + 17)
        vu = state.vote_update_state()
        assert isinstance(vu, EventCodedVoteUpdateState)
        before_q = vu.q_levels.clone()
        result = apply_event_coded_integer_vote_update_reference(
            vu,
            VoteUpdateInputs(votes=votes),
            spec,
            step_index=step,
        )
        state = make_live_event_coded_tensor_state(
            state,
            result.q_levels.reshape(state.q_levels.shape),
            result.carrier,
        )
        carrier = result.carrier
        step_record = carrier.step_records[-1]
        q_changed = _q_changed_count(before_q, result.q_levels)

        encode_patch = (
            "calm.hrm_text_158.native_full_stack.event_coded_acc_checkpoint_codec"
            ".encode_event_coded_acc_events"
        )
        with mock.patch(encode_patch) as encode_mock, mock.patch.object(
            EventCodedAccLiveState,
            "hot_packed_bytes",
            autospec=True,
        ) as hot_pack_mock:
            started = time.perf_counter()
            module_row = build_carrier_growth_module_row(
                state_key=key,
                carrier=carrier,
                step_record=step_record,
                votes=votes,
                cap_accepted_rows=0,
                q_changed_rows=q_changed,
            )
            sidecar = build_carrier_growth_step_record(
                optimizer_step_index=step + 1,
                module_rows=[module_row],
            )
            elapsed = time.perf_counter() - started
            encode_mock.assert_not_called()
            hot_pack_mock.assert_not_called()

        _assert_no_raw_index_arrays(sidecar)
        shas.append(sidecar_sha256(sidecar))
        byte_sizes.append(int(sidecar["rollup"]["sidecar_bytes"]))
        wall_seconds.append(elapsed)

    return shas, byte_sizes, wall_seconds


@pytest.mark.slow
@pytest.mark.parametrize(
    ("density", "label"),
    [
        (0.0078125, "sparse128"),
        (0.667, "mixed66"),
    ],
)
def test_carrier_growth_sidecar_scale_smoke_v0(
    tmp_path: Path,
    density: float,
    label: str,
) -> None:
    del tmp_path  # Phase 0 is in-memory only; tmp_path reserved for Phase 1 file emission.

    first_shas, first_bytes, first_wall = _apply_steps_and_emit_sidecars(density=density, seed=44)
    second_shas, second_bytes, second_wall = _apply_steps_and_emit_sidecars(density=density, seed=44)

    assert first_shas == second_shas, f"{label}: sidecar sha256 mismatch across deterministic replays"
    assert first_bytes == second_bytes

    max_bytes = max(first_bytes)
    max_wall = max(first_wall)
    p95_index = max(0, int(round(0.95 * (len(first_wall) - 1))))
    p95_wall = sorted(first_wall)[p95_index]

    assert max_bytes <= SIDECAR_BYTES_CAP, (
        f"{label}: sidecar_bytes max={max_bytes} exceeds cap {SIDECAR_BYTES_CAP}"
    )
    assert max_wall <= SIDECAR_WALL_SECONDS_CAP, (
        f"{label}: isolated sidecar wall max={max_wall:.3f}s exceeds {SIDECAR_WALL_SECONDS_CAP}s "
        f"(p95={p95_wall:.3f}s)"
    )


def test_byte_estimators_are_rolling_not_full_encode() -> None:
    assert estimate_events_payload_bytes(event_count=100) == 100 * EST_BYTES_PER_EVENT
    assert estimate_hot_exact_payload_bytes(hot_row_count=50) == 50 * EST_BYTES_PER_HOT_ROW

    carrier = EventCodedAccLiveState(logical_numel=64, demotion_band=1)
    votes = torch.zeros(64, dtype=torch.int16)
    step_record = carrier.apply_step(0, votes={1: 12})
    row = build_carrier_growth_module_row(
        state_key="toy",
        carrier=carrier,
        step_record=step_record,
        votes=votes,
    )
    assert row["est_events_payload_bytes"] == len(carrier.events) * EST_BYTES_PER_EVENT
    assert row["est_hot_exact_payload_bytes"] == len(carrier._hot) * EST_BYTES_PER_HOT_ROW

    encode_patch = (
        "calm.hrm_text_158.native_full_stack.event_coded_acc_checkpoint_codec"
        ".encode_event_coded_acc_events"
    )
    with mock.patch(encode_patch) as encode_mock, mock.patch.object(
        EventCodedAccLiveState,
        "hot_packed_bytes",
        autospec=True,
    ) as hot_pack_mock:
        build_carrier_growth_step_record(
            optimizer_step_index=1,
            module_rows=[row],
        )
        encode_mock.assert_not_called()
        hot_pack_mock.assert_not_called()
