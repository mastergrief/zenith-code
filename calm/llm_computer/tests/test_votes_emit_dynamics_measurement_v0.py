"""CPU tests for votes-emit dynamics measurement infrastructure."""
from __future__ import annotations

import copy
import inspect
import json
import math
from pathlib import Path

import pytest
import torch

from calm.hrm_text_158.bit_linear import BitLinear
from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    BoundedDeltaAccumulatorState,
    BoundedDeltaTensorState,
)
from calm.hrm_text_158.native_full_stack.event_coded_acc_checkpoint_codec import (
    EventCodedAccEvent,
    PackedEventCodedAccState,
    pack_event_coded_acc_checkpoint_reference,
    unpack_event_coded_acc_checkpoint_reference,
)
from calm.hrm_text_158.native_full_stack.persistent_state_budget import (
    PACKED_EVENT_CODED_ACC_FORMAT,
    measure_r4_persistent_state_budget,
    measure_r4b_persistent_state_budget,
    measure_r4v_event_coded_acc_budget,
)
from calm.hrm_text_158.native_full_stack.qscale_linear import QScaleWeightState
from calm.hrm_text_158.native_full_stack.two_tier_threshold_semantics import (
    frozen_threshold_semantics_block,
)
from calm.hrm_text_158.native_full_stack.vote_update import VoteUpdateSpec
from calm.hrm_text_158.native_full_stack.votes_emit_collector import (
    VotesEmitCollector,
    build_votes_emit_step_record,
    maybe_emit_votes_step_record,
)
from calm.llm_computer.tests.test_hrm_text_158_native_bounded_delta_acquisition_probe import (
    _tiny_parent_blob,
)
from scripts.hrm_text_158_votes_emit_scale_smoke import (
    build_votes_emit_scale_smoke_receipt,
    run_votes_emit_scale_smoke,
)


class _TinyTernary(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = BitLinear(16, 16, bias=False)
        self.tail = torch.nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tail(self.proj(x))


def _make_state(*, numel: int = 64) -> BoundedDeltaTensorState:
    side = int(math.sqrt(numel))
    if side * side != numel:
        side = numel
        shape = (numel,)
    else:
        shape = (side, side)
    q = torch.tensor([-1, 0, 1], dtype=torch.int8)
    idx = torch.arange(numel, dtype=torch.long) % 3
    q_levels = q[idx].view(shape).contiguous()
    acc = torch.zeros(numel, dtype=torch.int16)
    bounded = BoundedDeltaAccumulatorState(
        logical_shape=tuple(int(dim) for dim in q_levels.shape),
        cold_default_value=0,
        hot_exact_indices=(),
        hot_exact_values=(),
        cold_exception_indices=(),
        cold_exception_values=(),
        candidate_name="cold_default",
        raw_arrays_included=False,
    )
    return BoundedDeltaTensorState(
        state_key="proj",
        q_levels=q_levels,
        frozen_scale=torch.tensor(1.0, dtype=torch.float32),
        bounded_accumulator=bounded,
        exact_accumulator_shadow=acc.view_as(q_levels),
        bounded_accumulator_fresh_for_exact_shadow=False,
    )


def _vote_spec() -> VoteUpdateSpec:
    return VoteUpdateSpec(
        threshold_abs=1,
        accumulator_clip_min=-127,
        accumulator_clip_max=127,
        max_abs_per_tensor=4096,
    )


def _votes_for_state(state: BoundedDeltaTensorState) -> torch.Tensor:
    return torch.randint(-3, 4, state.q_levels.shape, dtype=torch.int16)


def test_event_coded_acc_checkpoint_v0_roundtrip_preserves_events_incl_tail_padding() -> None:
    events = (
        EventCodedAccEvent(flat_index=0, direction=0, residual_mag=3, event_type=1),
        EventCodedAccEvent(flat_index=127, direction=1, residual_mag=9, event_type=2),
        EventCodedAccEvent(flat_index=4095, direction=1, residual_mag=15, event_type=3),
    )
    packed = pack_event_coded_acc_checkpoint_reference(
        logical_numel=4096,
        events=events,
        backlog_indices=[7, 129],
    )
    roundtripped_events, roundtripped_backlog = unpack_event_coded_acc_checkpoint_reference(
        packed
    )
    assert roundtripped_events == events
    assert roundtripped_backlog == (7, 129)
    assert int(packed.events_packed.numel()) > len(events)
    assert packed.schema == "event_coded_acc_checkpoint/v0"


def test_event_coded_acc_checkpoint_v0_schema_literal_is_exact() -> None:
    packed = pack_event_coded_acc_checkpoint_reference(
        logical_numel=16,
        events=(
            EventCodedAccEvent(flat_index=1, direction=0, residual_mag=2, event_type=1),
        ),
    )
    assert packed.schema == "event_coded_acc_checkpoint/v0"
    bad = PackedEventCodedAccState(
        events_packed=packed.events_packed,
        backlog_packed=packed.backlog_packed,
        logical_numel=packed.logical_numel,
        event_count=packed.event_count,
        backlog_entry_count=packed.backlog_entry_count,
        schema="event_coded_acc_checkpoint/v1",
        format=packed.format,
    )
    with pytest.raises(ValueError, match="event_coded_acc_checkpoint/v0"):
        unpack_event_coded_acc_checkpoint_reference(bad)


def test_measure_r4v_rejects_wrong_payload_schema() -> None:
    packed = pack_event_coded_acc_checkpoint_reference(
        logical_numel=3,
        events=(),
    )
    bad = PackedEventCodedAccState(
        events_packed=packed.events_packed,
        backlog_packed=packed.backlog_packed,
        logical_numel=packed.logical_numel,
        event_count=packed.event_count,
        backlog_entry_count=packed.backlog_entry_count,
        schema="event_coded_acc_checkpoint/v1",
        format=packed.format,
    )
    qstate = QScaleWeightState(
        q_levels=torch.tensor([[0, 1, -1]], dtype=torch.int8),
        scale=torch.tensor(1.0, dtype=torch.float32),
    )
    with pytest.raises(ValueError, match="event_coded_acc_checkpoint/v0"):
        measure_r4v_event_coded_acc_budget([qstate], [bad], state_keys=["proj"])


def test_event_coded_acc_checkpoint_v0_pack_rejects_event_count_mismatch() -> None:
    packed = pack_event_coded_acc_checkpoint_reference(
        logical_numel=16,
        events=(
            EventCodedAccEvent(flat_index=1, direction=0, residual_mag=2, event_type=1),
        ),
    )
    bad = PackedEventCodedAccState(
        events_packed=packed.events_packed,
        backlog_packed=packed.backlog_packed,
        logical_numel=packed.logical_numel,
        event_count=2,
        backlog_entry_count=packed.backlog_entry_count,
        schema=packed.schema,
        format=packed.format,
    )
    with pytest.raises(ValueError, match="event_count mismatch"):
        unpack_event_coded_acc_checkpoint_reference(bad)


def test_event_coded_acc_checkpoint_v0_unpack_rejects_wrong_byte_length() -> None:
    packed = pack_event_coded_acc_checkpoint_reference(
        logical_numel=16,
        events=(
            EventCodedAccEvent(flat_index=1, direction=0, residual_mag=2, event_type=1),
        ),
    )
    bad = PackedEventCodedAccState(
        events_packed=torch.cat((packed.events_packed, packed.events_packed[:1])),
        backlog_packed=packed.backlog_packed,
        logical_numel=packed.logical_numel,
        event_count=packed.event_count,
        backlog_entry_count=packed.backlog_entry_count,
        schema=packed.schema,
        format=packed.format,
    )
    with pytest.raises(ValueError, match="packed byte length must match format-specific ceiling"):
        unpack_event_coded_acc_checkpoint_reference(bad)


def test_measure_r4v_event_coded_acc_budget_from_actual_saved_bytes_metadata_inclusive() -> None:
    q = _make_state(numel=256).q_levels
    packed = pack_event_coded_acc_checkpoint_reference(
        logical_numel=int(q.numel()),
        events=(
            EventCodedAccEvent(flat_index=3, direction=1, residual_mag=4, event_type=1),
            EventCodedAccEvent(flat_index=9, direction=0, residual_mag=2, event_type=0),
        ),
        backlog_indices=[11],
    )
    qstate = QScaleWeightState(q_levels=q, scale=torch.tensor(1.0, dtype=torch.float32))
    report = measure_r4v_event_coded_acc_budget(
        [qstate],
        [packed],
        state_keys=["proj"],
    )
    payload_bytes = int(
        packed.events_packed.numel() + packed.backlog_packed.numel()
    )
    metadata_bytes = int(packed.metadata_bytes)
    expected_bpw = (payload_bytes + metadata_bytes) * 8 / int(q.numel())
    assert report.r4v_actual_events_payload_bytes == packed.events_packed.numel()
    assert report.r4v_actual_backlog_payload_bytes == packed.backlog_packed.numel()
    assert report.r4v_actual_acc_metadata_bytes == metadata_bytes
    assert report.r4v_acc_inclusive_physical_bits_per_weight == pytest.approx(
        expected_bpw,
        rel=1e-6,
    )


def test_measure_r4v_failclosed_when_payload_missing_events_packed() -> None:
    qstate = QScaleWeightState(
        q_levels=torch.tensor([[0, 1, -1]], dtype=torch.int8),
        scale=torch.tensor(1.0, dtype=torch.float32),
    )
    with pytest.raises(ValueError, match="missing events_packed"):
        measure_r4v_event_coded_acc_budget(
            [qstate],
            [
                type(
                    "BadPayload",
                    (),
                    {
                        "schema": "event_coded_acc_checkpoint/v0",
                        "format": PACKED_EVENT_CODED_ACC_FORMAT,
                        "backlog_packed": torch.zeros(0, dtype=torch.uint8),
                        "logical_numel": 3,
                        "event_count": 0,
                        "backlog_entry_count": 0,
                        "metadata_bytes": 24,
                    },
                )()
            ],
            state_keys=["proj"],
        )


def test_votes_emit_collector_writes_full_per_step_field_set_cpu_tiny(tmp_path: Path) -> None:
    state = _make_state(numel=64)
    votes = _votes_for_state(state)
    record = build_votes_emit_step_record(
        optimizer_step_index=1,
        tensor_states={"proj": state},
        votes_by_key={"proj": votes},
        vote_specs_by_key={"proj": _vote_spec()},
        max_abs_per_tensor=4096,
    )
    required = {
        "optimizer_step_index",
        "warmup_apply_class",
        "applied_flip_count",
        "threshold_semantics",
        "sampled_candidate_table",
    }
    assert required.issubset(record)
    assert record["threshold_semantics"] == frozen_threshold_semantics_block()
    assert len(record["sampled_candidate_table"]) <= 32
    first_row = record["sampled_candidate_table"][0]
    assert {
        "pre_accumulator_i16",
        "new_acc_i32_signed",
        "vote_value",
        "current_q_level",
        "proposal_direction",
    }.issubset(first_row)
    collector = VotesEmitCollector(tmp_path)
    emit_receipt = collector.emit_step(record, optimizer_step_index=1)
    step_path = Path(emit_receipt["step_path"])
    assert step_path.is_file()
    payload = json.loads(step_path.read_text(encoding="utf-8"))
    assert payload["sampled_candidate_table"] == record["sampled_candidate_table"]


def test_votes_emit_manifest_hashes_replayable(tmp_path: Path) -> None:
    state = _make_state(numel=64)
    votes = _votes_for_state(state)
    kwargs = dict(
        optimizer_step_index=2,
        tensor_states={"proj": state},
        votes_by_key={"proj": votes},
        vote_specs_by_key={"proj": _vote_spec()},
        max_abs_per_tensor=4096,
    )
    record_a = build_votes_emit_step_record(**kwargs)
    record_b = build_votes_emit_step_record(**kwargs)
    assert record_a["sampled_candidate_table"] == record_b["sampled_candidate_table"]
    assert record_a["source_table_hash"] == record_b["source_table_hash"]
    collector = VotesEmitCollector(tmp_path)
    first = collector.emit_step(record_a, optimizer_step_index=2)
    second = collector.emit_step(record_b, optimizer_step_index=2)
    assert first["step_hash"] == second["step_hash"]
    assert first["manifest_hash"] == second["manifest_hash"]


def test_votes_emit_disabled_hook_emits_nothing_and_preserves_state(tmp_path: Path) -> None:
    state = _make_state(numel=32)
    votes = _votes_for_state(state)
    before = copy.deepcopy(state.q_levels)
    result = maybe_emit_votes_step_record(
        root=tmp_path,
        enabled=False,
        optimizer_step_index=1,
        tensor_states={"proj": state},
        votes_by_key={"proj": votes},
        vote_specs_by_key={"proj": _vote_spec()},
        max_abs_per_tensor=4096,
    )
    assert result is None
    assert torch.equal(state.q_levels, before)
    assert not (tmp_path / "votes_emit").exists()


def test_votes_emit_enabled_hook_is_read_only_on_state(tmp_path: Path) -> None:
    state = _make_state(numel=32)
    votes = _votes_for_state(state)
    before = state.q_levels.detach().clone()
    maybe_emit_votes_step_record(
        root=tmp_path,
        enabled=True,
        optimizer_step_index=1,
        tensor_states={"proj": state},
        votes_by_key={"proj": votes},
        vote_specs_by_key={"proj": _vote_spec()},
        max_abs_per_tensor=4096,
    )
    assert torch.equal(state.q_levels, before)


def test_votes_emit_scale_smoke_receipt_fields_present_and_overhead_computed(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "tiny_parent.pt"
    torch.save(_tiny_parent_blob(batch_size=8), parent)
    receipt = run_votes_emit_scale_smoke(
        scratch_root=tmp_path / "scale_smoke_run",
        parent=parent,
        steps=3,
    )
    assert receipt["schema_version"] == "hrm_text_158_votes_emit_scale_smoke/v0"
    assert receipt["emit_sample_count"] > 0
    assert receipt["bytes_per_step_mean"] > 0.0
    assert receipt["per_step_file_count"] == 3
    assert receipt["manifest_exists"] is True
    assert receipt["emit_p50_ms"] >= 0.0
    assert receipt["emit_p95_ms"] >= receipt["emit_p50_ms"]
    manifest = json.loads(
        Path(receipt["manifest_path"]).read_text(encoding="utf-8")
    )
    manifest_timings = [float(value) for value in manifest["emit_timings_ms"]]
    assert receipt["emit_sample_count"] == len(manifest_timings)
    assert receipt["emit_p50_ms"] == sorted(manifest_timings)[len(manifest_timings) // 2]
    assert receipt["emit_p95_ms"] == max(manifest_timings)


def test_votes_emit_scale_smoke_receipt_fail_closed_on_empty_timings(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="emit_sample_count=0"):
        build_votes_emit_scale_smoke_receipt(
            steps=1,
            baseline_seconds=0.01,
            emit_enabled_seconds=0.011,
            manifest_path=tmp_path / "missing_manifest.json",
            per_step_dir=tmp_path / "per_step",
            emit_timings_ms=[],
        )


def test_measure_r4_and_r4b_unchanged_import_smoke() -> None:
    import calm.hrm_text_158.native_full_stack.persistent_state_budget as psb

    source = inspect.getsource(psb)
    assert "event_coded_acc_checkpoint_codec" not in source
    assert "def measure_r4_persistent_state_budget" in source
    assert "def measure_r4b_persistent_state_budget" in source
    assert "def measure_r4v_event_coded_acc_budget" in source
    assert measure_r4_persistent_state_budget is psb.measure_r4_persistent_state_budget
    assert measure_r4b_persistent_state_budget is psb.measure_r4b_persistent_state_budget
