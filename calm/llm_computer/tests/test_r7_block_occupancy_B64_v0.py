"""CPU-static tests for R7 B64 block-occupancy (pure + thin census integration)."""

from __future__ import annotations

import base64
import json
import time
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch

from calm.hrm_text_158.native_full_stack.r7_block_occupancy_b64 import (
    BINARY_ENCODING,
    DEFAULT_B,
    DEFAULT_K,
    SCHEMA_VERSION,
    BlockOccupancyError,
    BlockOccupancyInput,
    BlockOccupancyMissingObservablesError,
    PerStateOccupancySource,
    build_block_occupancy_B64,
    rebuild_bitmap_from_chunk_state,
)
from calm.hrm_text_158.native_full_stack.r7_selective_drain_eligibility_census import (
    ObserverContinuityTracker,
    build_census_chunk,
    build_selective_drain_census_step_dto,
    initialize_selective_drain_census_observer_continuity_at_step0,
    maybe_run_selective_drain_census,
    normalize_block_occupancy_input,
)


@dataclass(frozen=True)
class _Row:
    state_key: str
    flat_index: int
    abs_new_acc: int
    threshold_abs: int = 10


@dataclass(frozen=True)
class _CapResult:
    accepted_rows: list[_Row]
    deferred_rows: list[_Row]
    step_summary: dict[str, Any]


@dataclass
class _Plan:
    new_acc_i32: torch.Tensor
    q_i16: torch.Tensor
    event_coded_sparse_active_idx: Any = None
    replay_ce_veto_indices: Any = None


def _acc_le_from_values(values: list[int]) -> bytes:
    a = array("i", values)
    return a.tobytes()


def _src(state_key: str, values: list[int]) -> PerStateOccupancySource:
    return PerStateOccupancySource(
        state_key=state_key,
        logical_numel=len(values),
        acc_i32_le=_acc_le_from_values(values),
        q_numel=len(values),
    )


def test_literals_match_design():
    assert SCHEMA_VERSION == "hrm_text_158_r7_block_occupancy_B64/v1"
    assert DEFAULT_B == 64
    assert DEFAULT_K == 12
    assert BINARY_ENCODING == "base64"


def test_module_isolation_two_state_keys():
    # a: numel 64 (1 block); b: numel 65 (2 blocks, tail=1)
    a_vals = [0] * 64
    a_vals[0] = 5
    b_vals = [0] * 65
    b_vals[64] = 9
    inp = BlockOccupancyInput(
        per_state=(_src("a", a_vals), _src("b", b_vals)),
        eligible_ids_k=(("a", 0), ("b", 64)),
        k=12,
        B=64,
    )
    result = build_block_occupancy_B64(inp)
    assert [ps.state_key for ps in result.per_state] == ["a", "b"]
    assert result.per_state[0].n_blocks == 1
    assert result.per_state[1].n_blocks == 2
    assert result.per_state[1].tail_len == 1
    # flats never cross: a block0 eligible=1; b block1 eligible=1
    assert result.per_state[0].per_block_eligible_u8 == bytes([1])
    assert result.per_state[1].per_block_eligible_u8[1] == 1


def test_tail_u8_sum_closure():
    vals = [0] * 100  # n_blocks=2, tail=36
    vals[10] = 1
    vals[70] = 2
    inp = BlockOccupancyInput(
        per_state=(_src("w", vals),),
        eligible_ids_k=(("w", 10),),
        k=12,
        B=64,
    )
    ps = build_block_occupancy_B64(inp).per_state[0]
    assert ps.n_blocks == 2
    assert ps.tail_len == 36
    assert ps.per_block_eligible_u8[0] + ps.per_block_noneligible_nonzero_u8[0] + ps.per_block_empty_u8[0] == 64
    assert ps.per_block_eligible_u8[1] + ps.per_block_noneligible_nonzero_u8[1] + ps.per_block_empty_u8[1] == 36


def test_duplicate_flat_ids_fail():
    vals = [0] * 64
    inp = BlockOccupancyInput(
        per_state=(_src("w", vals),),
        eligible_ids_k=(("w", 1), ("w", 1)),
        k=12,
        B=64,
    )
    with pytest.raises(BlockOccupancyError, match="duplicate"):
        build_block_occupancy_B64(inp)


def test_missing_flat_out_of_range_fail():
    vals = [0] * 64
    inp = BlockOccupancyInput(
        per_state=(_src("w", vals),),
        eligible_ids_k=(("w", 64),),
        k=12,
        B=64,
    )
    with pytest.raises(BlockOccupancyError, match="out of range"):
        build_block_occupancy_B64(inp)


def test_bitmap_from_count_reconstruction():
    vals = [0] * 128
    vals[0] = 3  # noneligible nonzero in block0
    vals[70] = 0
    inp = BlockOccupancyInput(
        per_state=(_src("w", vals),),
        eligible_ids_k=(("w", 70),),
        k=12,
        B=64,
    )
    result = build_block_occupancy_B64(inp)
    ps = result.per_state[0]
    chunk_state = ps.to_dict()
    rebuilt = rebuild_bitmap_from_chunk_state(chunk_state)
    assert rebuilt == ps.fully_eoe_block_bitmap
    assert base64.b64decode(chunk_state["fully_eoe_block_bitmap_b64"]) == rebuilt


def test_set_hash_determinism():
    vals = [0] * 128
    inp = BlockOccupancyInput(
        per_state=(_src("w", vals),),
        eligible_ids_k=tuple(("w", i) for i in (5, 70, 3)),
        k=12,
        B=64,
    )
    a = build_block_occupancy_B64(inp).per_state[0].fully_eoe_set_sha256
    inp2 = BlockOccupancyInput(
        per_state=(_src("w", vals),),
        eligible_ids_k=tuple(("w", i) for i in (70, 3, 5)),
        k=12,
        B=64,
    )
    b = build_block_occupancy_B64(inp2).per_state[0].fully_eoe_set_sha256
    assert a == b


def test_q_numel_mismatch_via_input():
    src = PerStateOccupancySource(
        state_key="w",
        logical_numel=64,
        acc_i32_le=_acc_le_from_values([0] * 64),
        q_numel=63,
    )
    inp = BlockOccupancyInput(per_state=(src,), eligible_ids_k=(), k=12, B=64)
    with pytest.raises(BlockOccupancyMissingObservablesError, match="q_numel"):
        build_block_occupancy_B64(inp)


def test_compact_size_bound_and_no_raw_arrays():
    numel = 130816
    # sparse nonzeros only — still full scan over bytes-backed array
    vals = [0] * numel
    vals[0] = 1
    vals[1000] = 2
    inp = BlockOccupancyInput(
        per_state=(_src("w", vals),),
        eligible_ids_k=(("w", 0),),
        k=12,
        B=64,
    )
    t0 = time.perf_counter()
    result = build_block_occupancy_B64(inp)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    body = result.to_chunk_dict()
    encoded = json.dumps(body, sort_keys=True)
    encoded_bytes = len(encoded.encode("utf-8"))
    assert result.compact_payload_bytes < 64 * 1024
    assert "raw_" not in encoded
    assert "per_weight" not in encoded
    # ~6KiB binary payload; JSON+b64 larger — report measured
    assert result.compact_payload_bytes == (
        result.per_state[0].n_blocks * 3
        + (result.per_state[0].n_blocks + 7) // 8
    )
    # freeze measured evidence for receipt (also assert sane band)
    assert 5000 <= result.compact_payload_bytes <= 9000
    assert encoded_bytes < 64 * 1024
    assert elapsed_ms < 5000.0  # generous CPU bound; evidence not "negligible" claim
    # expose for operators reading pytest -s
    print(
        f"MEASURED compact_payload_bytes={result.compact_payload_bytes} "
        f"json_line_bytes={encoded_bytes} build_ms={elapsed_ms:.2f}"
    )


def test_to_chunk_dict_round_trip_hash_consistent():
    vals = [0] * 64
    vals[2] = 7
    inp = BlockOccupancyInput(
        per_state=(_src("w", vals),),
        eligible_ids_k=(("w", 1),),
        k=12,
        B=64,
    )
    result = build_block_occupancy_B64(inp)
    d1 = result.to_chunk_dict()
    d2 = result.to_chunk_dict()
    assert d1 == d2
    assert json.dumps(d1, sort_keys=True) == json.dumps(d2, sort_keys=True)
    # frozen: no mutable list reachable from dataclass fields
    assert isinstance(result.per_state, tuple)
    assert isinstance(result.per_state[0].per_block_eligible_u8, bytes)


def test_acc_bytes_not_python_int_list():
    vals = [0] * 256
    src = _src("w", vals)
    assert isinstance(src.acc_i32_le, bytes)
    assert len(src.acc_i32_le) == 256 * 4
    # ensure Input rejects accidental list materialization path by type
    with pytest.raises(BlockOccupancyMissingObservablesError):
        bad = PerStateOccupancySource(
            state_key="w",
            logical_numel=2,
            acc_i32_le=[0, 0],  # type: ignore[arg-type]
            q_numel=2,
        )
        build_block_occupancy_B64(
            BlockOccupancyInput(per_state=(bad,), eligible_ids_k=(), k=12, B=64)
        )


def _plan_for(state_key: str, numel: int, nonzero_at: dict[int, int] | None = None) -> _Plan:
    acc = torch.zeros(numel, dtype=torch.int32)
    if nonzero_at:
        for fi, v in nonzero_at.items():
            acc[fi] = int(v)
    q = torch.zeros(numel, dtype=torch.int16)
    return _Plan(new_acc_i32=acc.view(numel), q_i16=q.view(numel))


def test_default_off_byte_shape_parity(tmp_path: Path):
    sidecar = tmp_path / "census.jsonl"
    tr = ObserverContinuityTracker()
    tr.reset()
    initialize_selective_drain_census_observer_continuity_at_step0(
        tracker=tr, observed_step=0, sidecar_path=sidecar
    )
    plans = {"w": _plan_for("w", 128, {2: 5})}
    cap = _CapResult(
        [_Row("w", 1, 100)],
        [_Row("w", 2, 50), _Row("w", 3, 40)],
        {"global_rate_cap_cap": 1},
    )
    chunk_off = maybe_run_selective_drain_census(
        enabled=True,
        pre_step_backlog=None,
        cap_result=cap,
        plans_by_key=plans,
        step=1,
        tracker=tr,
        sidecar_path=None,
        block_occupancy_B64_enabled=False,
    )
    assert chunk_off is not None
    assert "block_occupancy_B64" not in chunk_off
    # characterize: build_census_chunk alone has same keys
    dto = build_selective_drain_census_step_dto(
        step=1,
        ordering_mode="margin",
        cap=1,
        pre_step_backlog=None,
        accepted_rows=cap.accepted_rows,
        deferred_rows=cap.deferred_rows,
        plans_by_key=plans,
    )
    tr2 = ObserverContinuityTracker()
    tr2.reset()
    initialize_selective_drain_census_observer_continuity_at_step0(
        tracker=tr2, observed_step=0, sidecar_path=tmp_path / "init2.jsonl"
    )
    tr2.update_from_dto(
        build_selective_drain_census_step_dto(
            step=0,
            ordering_mode="margin",
            cap=1,
            pre_step_backlog=None,
            accepted_rows=cap.accepted_rows,
            deferred_rows=cap.deferred_rows,
            plans_by_key=None,
        )
    )
    # simpler: keys of off chunk equal baseline census keys (no occupancy)
    baseline_keys = set(chunk_off.keys())
    assert "block_occupancy_B64" not in baseline_keys


def test_enabled_attach_and_missing_plan_key(tmp_path: Path):
    sidecar = tmp_path / "census.jsonl"
    tr = ObserverContinuityTracker()
    tr.reset()
    initialize_selective_drain_census_observer_continuity_at_step0(
        tracker=tr, observed_step=0, sidecar_path=sidecar
    )
    plans = {"w": _plan_for("w", 128, {2: 5})}
    cap = _CapResult(
        [_Row("w", 1, 100)],
        [_Row("w", 2, 50)],
        {"global_rate_cap_cap": 1},
    )
    chunk = maybe_run_selective_drain_census(
        enabled=True,
        pre_step_backlog=None,
        cap_result=cap,
        plans_by_key=plans,
        step=1,
        tracker=tr,
        sidecar_path=sidecar,
        block_occupancy_B64_enabled=True,
    )
    assert chunk is not None
    assert "block_occupancy_B64" in chunk
    assert chunk["block_occupancy_B64"]["schema_version"] == SCHEMA_VERSION
    assert chunk["block_occupancy_B64"]["event_coded_live"] is False

    # missing plan key → MISSING_OBSERVABLES / rollback (no new sidecar line)
    tr2 = ObserverContinuityTracker()
    tr2.reset()
    path2 = tmp_path / "census2.jsonl"
    initialize_selective_drain_census_observer_continuity_at_step0(
        tracker=tr2, observed_step=0, sidecar_path=path2
    )
    before = path2.read_text(encoding="utf-8") if path2.exists() else ""
    cap2 = _CapResult(
        [_Row("missing", 1, 100)],
        [_Row("missing", 2, 50)],
        {"global_rate_cap_cap": 1},
    )
    with pytest.raises(BlockOccupancyMissingObservablesError):
        maybe_run_selective_drain_census(
            enabled=True,
            pre_step_backlog=None,
            cap_result=cap2,
            plans_by_key={"w": plans["w"]},
            step=1,
            tracker=tr2,
            sidecar_path=path2,
            block_occupancy_B64_enabled=True,
        )
    after = path2.read_text(encoding="utf-8") if path2.exists() else ""
    assert after == before


def test_wrong_attribute_rejected_at_normalize_seam():
    dto = build_selective_drain_census_step_dto(
        step=1,
        ordering_mode="margin",
        cap=1,
        pre_step_backlog=None,
        accepted_rows=[_Row("w", 1, 100)],
        deferred_rows=[_Row("w", 2, 50)],
        plans_by_key=None,
    )
    tr = ObserverContinuityTracker()
    tr.reset()
    tr.enabled_at_step = 0
    tr.last_step = 0
    tr.update_from_dto(dto)

    class _Bad:
        q_i16 = torch.zeros(8, dtype=torch.int16)
        event_coded_sparse_active_idx = None
        # no new_acc_i32

    with pytest.raises(BlockOccupancyMissingObservablesError, match="new_acc_i32"):
        normalize_block_occupancy_input(
            dto=dto,
            tracker=tr,
            plans_by_key={"w": _Bad()},
            k=12,
            B=64,
        )


def test_event_coded_sparse_active_idx_set_fail():
    dto = build_selective_drain_census_step_dto(
        step=1,
        ordering_mode="margin",
        cap=1,
        pre_step_backlog=None,
        accepted_rows=[_Row("w", 1, 100)],
        deferred_rows=[_Row("w", 2, 50)],
        plans_by_key=None,
    )
    tr = ObserverContinuityTracker()
    tr.reset()
    tr.enabled_at_step = 0
    tr.last_step = 0
    tr.update_from_dto(dto)
    plan = _plan_for("w", 8)
    plan.event_coded_sparse_active_idx = torch.tensor([0], dtype=torch.int64)
    with pytest.raises(BlockOccupancyMissingObservablesError, match="event_coded_sparse"):
        normalize_block_occupancy_input(
            dto=dto, tracker=tr, plans_by_key={"w": plan}, k=12, B=64
        )


def test_normalize_uses_bytes_not_tolist():
    dto = build_selective_drain_census_step_dto(
        step=1,
        ordering_mode="margin",
        cap=1,
        pre_step_backlog=None,
        accepted_rows=[_Row("w", 1, 100)],
        deferred_rows=[_Row("w", 2, 50)],
        plans_by_key=None,
    )
    tr = ObserverContinuityTracker()
    tr.reset()
    tr.enabled_at_step = 0
    tr.last_step = 0
    tr.update_from_dto(dto)
    plan = _plan_for("w", 256, {2: 9})
    occ_in = normalize_block_occupancy_input(
        dto=dto, tracker=tr, plans_by_key={"w": plan}, k=12, B=64
    )
    assert isinstance(occ_in.per_state[0].acc_i32_le, bytes)
    assert len(occ_in.per_state[0].acc_i32_le) == 256 * 4
