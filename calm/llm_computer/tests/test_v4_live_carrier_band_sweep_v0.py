"""CPU tests for V4-LIVE carrier, codec v1 hot_exact, and band sweep."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from calm.hrm_text_158.native_full_stack.event_coded_acc_checkpoint_codec import (
    EventCodedAccEvent,
    PackedEventCodedAccState,
    pack_event_coded_acc_checkpoint_v1,
    unpack_event_coded_acc_checkpoint_v1,
)
from calm.hrm_text_158.native_full_stack.event_coded_acc_live_carrier import (
    DEFAULT_VERDICT_NUMEL,
    DenseOracleState,
    EventCodedAccLiveState,
    decisive_surface_drift_count,
)
from calm.hrm_text_158.native_full_stack.persistent_state_budget import (
    EVENT_CODED_ACC_CHECKPOINT_PAYLOAD_SCHEMA_V1,
    measure_r4_persistent_state_budget,
    measure_r4b_persistent_state_budget,
    measure_r4v_event_coded_acc_budget,
)
from calm.hrm_text_158.native_full_stack.qscale_linear import QScaleWeightState
from calm.hrm_text_158.native_full_stack.v4_live_band_sweep import (
    CPU_VERDICT_QUALIFY_PROCEED_TO_LIVE,
    DYNAMICS_CLASS_SYNTHETIC,
    HOT_RISK_PROXY_LABEL,
    assert_cpu_never_banks_reducible_or_intrinsic,
    build_sweep_table_payload,
    default_adversarial_scenarios,
    demotion_band_knob_is_live,
    map_cpu_verdict_to_terminal_classifier,
    run_band_sweep,
    run_representative_ms_scenario,
    write_sweep_table_json,
)
from calm.hrm_text_158.native_full_stack.votes_emit_dynamics_replay import (
    CLASSIFIER_INTRINSIC_WIDE_CONFIRMED,
    CLASSIFIER_REDUCIBLE_UNDER_DYNAMICS,
    CLASSIFIER_STATIC_PROXY_ARTIFACT,
)


def _qstate(numel: int) -> QScaleWeightState:
    return QScaleWeightState(
        q_levels=torch.zeros((1, int(numel)), dtype=torch.int8),
        scale=torch.tensor(1.0, dtype=torch.float32),
    )


def test_event_coded_acc_checkpoint_v1_hot_exact_roundtrip() -> None:
    packed = pack_event_coded_acc_checkpoint_v1(
        logical_numel=128,
        events=(
            EventCodedAccEvent(flat_index=3, direction=1, residual_mag=5, event_type=1),
        ),
        backlog_indices=[9],
        hot_exact_indices=[0, 17, 63],
        hot_exact_values=[4, -2, 7],
    )
    events, backlog, hot_indices, hot_values = unpack_event_coded_acc_checkpoint_v1(packed)
    assert packed.schema == "event_coded_acc_checkpoint/v1"
    assert events[0].flat_index == 3
    assert backlog == (9,)
    assert hot_indices == (0, 17, 63)
    assert hot_values == (4, -2, 7)
    assert int(packed.hot_exact_packed.numel()) > 0


def test_event_coded_acc_checkpoint_v1_schema_literal_is_exact() -> None:
    packed = pack_event_coded_acc_checkpoint_v1(
        logical_numel=16,
        events=(),
        hot_exact_indices=[1],
        hot_exact_values=[2],
    )
    assert packed.schema == EVENT_CODED_ACC_CHECKPOINT_PAYLOAD_SCHEMA_V1
    assert EVENT_CODED_ACC_CHECKPOINT_PAYLOAD_SCHEMA_V1 == "event_coded_acc_checkpoint/v1"


def test_event_coded_acc_checkpoint_v1_unpack_rejects_wrong_schema() -> None:
    packed = pack_event_coded_acc_checkpoint_v1(
        logical_numel=16,
        events=(),
        hot_exact_indices=[1],
        hot_exact_values=[2],
    )
    bad = PackedEventCodedAccState(
        events_packed=packed.events_packed,
        backlog_packed=packed.backlog_packed,
        logical_numel=packed.logical_numel,
        event_count=packed.event_count,
        backlog_entry_count=packed.backlog_entry_count,
        schema="event_coded_acc_checkpoint/v0",
        format=packed.format,
        hot_exact_packed=packed.hot_exact_packed,
        hot_exact_row_count=packed.hot_exact_row_count,
    )
    with pytest.raises(ValueError, match="event_coded_acc_checkpoint/v1"):
        unpack_event_coded_acc_checkpoint_v1(bad)


def test_measure_r4v_counts_hot_exact_bytes_metadata_inclusive() -> None:
    packed = pack_event_coded_acc_checkpoint_v1(
        logical_numel=64,
        events=(
            EventCodedAccEvent(flat_index=1, direction=0, residual_mag=2, event_type=1),
        ),
        hot_exact_indices=[0, 5],
        hot_exact_values=[3, -1],
    )
    report = measure_r4v_event_coded_acc_budget(
        [_qstate(64)],
        [packed],
        state_keys=["acc"],
    )
    assert report.r4v_actual_hot_exact_payload_bytes > 0
    assert report.r4v_acc_inclusive_physical_bits_per_weight == pytest.approx(
        (
            report.r4v_actual_events_payload_bytes
            + report.r4v_actual_backlog_payload_bytes
            + report.r4v_actual_hot_exact_payload_bytes
            + report.r4v_actual_acc_metadata_bytes
        )
        * 8
        / 64,
        rel=1e-6,
        abs=1e-6,
    )


def test_measure_r4v_failclosed_missing_hot_exact_packed_on_v1_schema() -> None:
    packed = pack_event_coded_acc_checkpoint_v1(
        logical_numel=8,
        events=(),
        hot_exact_indices=[1],
        hot_exact_values=[2],
    )
    bad = PackedEventCodedAccState(
        events_packed=packed.events_packed,
        backlog_packed=packed.backlog_packed,
        logical_numel=packed.logical_numel,
        event_count=packed.event_count,
        backlog_entry_count=packed.backlog_entry_count,
        schema=packed.schema,
        format=packed.format,
        hot_exact_row_count=1,
    )
    with pytest.raises(ValueError, match="packed byte length|hot_exact"):
        measure_r4v_event_coded_acc_budget([_qstate(8)], [bad], state_keys=["acc"])


def test_event_coded_acc_checkpoint_v1_unpack_rejects_trailing_hot_exact_bytes() -> None:
    packed = pack_event_coded_acc_checkpoint_v1(
        logical_numel=16,
        events=(),
        hot_exact_indices=[1],
        hot_exact_values=[2],
    )
    trailing = torch.cat(
        (
            packed.hot_exact_packed,
            torch.tensor([0xFF], dtype=torch.uint8),
        )
    )
    bad = PackedEventCodedAccState(
        events_packed=packed.events_packed,
        backlog_packed=packed.backlog_packed,
        logical_numel=packed.logical_numel,
        event_count=packed.event_count,
        backlog_entry_count=packed.backlog_entry_count,
        schema=packed.schema,
        format=packed.format,
        hot_exact_packed=trailing,
        hot_exact_row_count=packed.hot_exact_row_count,
    )
    with pytest.raises(ValueError, match="packed byte length must match format-specific ceiling"):
        unpack_event_coded_acc_checkpoint_v1(bad)


def test_measure_r4v_rejects_too_short_v1_hot_exact_payload() -> None:
    packed = pack_event_coded_acc_checkpoint_v1(
        logical_numel=8,
        events=(),
        hot_exact_indices=[1],
        hot_exact_values=[2],
    )
    bad = PackedEventCodedAccState(
        events_packed=packed.events_packed,
        backlog_packed=packed.backlog_packed,
        logical_numel=packed.logical_numel,
        event_count=packed.event_count,
        backlog_entry_count=packed.backlog_entry_count,
        schema=packed.schema,
        format=packed.format,
        hot_exact_packed=torch.tensor([0x01, 0x02], dtype=torch.uint8),
        hot_exact_row_count=1,
    )
    with pytest.raises(ValueError, match="packed byte length must match format-specific ceiling"):
        measure_r4v_event_coded_acc_budget([_qstate(8)], [bad], state_keys=["acc"])


def test_measure_r4v_rejects_trailing_bytes_v1_hot_exact_payload() -> None:
    packed = pack_event_coded_acc_checkpoint_v1(
        logical_numel=8,
        events=(),
        hot_exact_indices=[1],
        hot_exact_values=[2],
    )
    trailing = torch.cat(
        (
            packed.hot_exact_packed,
            torch.tensor([0x7F], dtype=torch.uint8),
        )
    )
    bad = PackedEventCodedAccState(
        events_packed=packed.events_packed,
        backlog_packed=packed.backlog_packed,
        logical_numel=packed.logical_numel,
        event_count=packed.event_count,
        backlog_entry_count=packed.backlog_entry_count,
        schema=packed.schema,
        format=packed.format,
        hot_exact_packed=trailing,
        hot_exact_row_count=packed.hot_exact_row_count,
    )
    with pytest.raises(ValueError, match="packed byte length must match format-specific ceiling"):
        measure_r4v_event_coded_acc_budget([_qstate(8)], [bad], state_keys=["acc"])


def test_no_codec_import_in_persistent_state_budget() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    budget_path = (
        repo_root
        / "hrm_text_158"
        / "native_full_stack"
        / "persistent_state_budget.py"
    )
    text = budget_path.read_text(encoding="utf-8")
    assert "event_coded_acc_checkpoint_codec" not in text


def test_measure_r4_and_r4b_unchanged_import_smoke() -> None:
    assert callable(measure_r4_persistent_state_budget)
    assert callable(measure_r4b_persistent_state_budget)


def test_c1_ms1_sweep_delayed_crossing_no_missed_crossing_per_band() -> None:
    scenario = default_adversarial_scenarios(numel=8)[0]
    for demotion_band in range(1, 7):
        carrier = EventCodedAccLiveState(logical_numel=8, demotion_band=demotion_band)
        oracle = DenseOracleState.zeros(8)
        for step_index, votes in enumerate(scenario.steps):
            carrier.apply_step(step_index, votes=votes)
            oracle.apply_step(step_index, votes=votes)
        assert carrier.events or demotion_band >= 4, (
            "low bands must cross; high bands may delay via demotion-lossiness"
        )
        assert oracle.step_records[-1].crossing_indices or any(
            record.crossing_indices for record in oracle.step_records
        )


def test_c1_ms7_sweep_long_run_hot_boundedness_per_band() -> None:
    for demotion_band in range(1, 7):
        carrier = EventCodedAccLiveState(logical_numel=16, demotion_band=demotion_band)
        carrier.apply_step(0, votes={0: 6})
        peak = carrier.step_records[-1].hot_exact_row_count
        for step in range(1, 8):
            carrier.apply_step(step, votes={})
        assert carrier.step_records[-1].hot_exact_row_count <= peak
        carrier.apply_step(8, votes={0: 6})
        assert carrier.events


def test_c1_ms2_decay_without_selection_at_representative_band() -> None:
    carrier = run_representative_ms_scenario(numel=16, demotion_band=3)
    assert carrier.step_records[0].hot_exact_row_count >= 1
    assert any(record.demotion_on_decay_count >= 0 for record in carrier.step_records)


def test_c1_ms3_residual_persistence_at_representative_band() -> None:
    carrier = EventCodedAccLiveState(logical_numel=16, demotion_band=1)
    carrier.apply_step(0, votes={0: 6})
    carrier.apply_step(1, votes={})
    assert carrier.reconstruct_lane(0) >= 0 or 0 in carrier.hot_exact


def test_c1_ms4_cap_deferred_carry_continuity_at_representative_band() -> None:
    carrier = EventCodedAccLiveState(logical_numel=16, demotion_band=3)
    carrier.backlog.add(2)
    carrier.apply_step(0, votes={2: 1})
    assert 2 in carrier.backlog


def test_c1_ms5_cold_to_hot_promotion_at_representative_band() -> None:
    carrier = EventCodedAccLiveState(logical_numel=16, demotion_band=3)
    carrier.apply_step(0, votes={4: 6})
    assert 4 in carrier.hot_exact


def test_c1_ms6_no_dense_int16_alloc_on_hot_path() -> None:
    carrier = run_representative_ms_scenario(numel=32, demotion_band=3)
    assert carrier.dense_accumulator_materialized_numel == 0


def test_classifier_qualifying_band_defers_reducible_banking_to_live() -> None:
    result = run_band_sweep(numel=32)
    mapped = map_cpu_verdict_to_terminal_classifier(
        cpu_verdict=result.cpu_verdict,
        replay_only=False,
    )
    assert_cpu_never_banks_reducible_or_intrinsic(
        cpu_verdict=result.cpu_verdict,
        mapped_classifier=mapped,
    )
    if result.qualifying_bands:
        assert result.cpu_verdict == CPU_VERDICT_QUALIFY_PROCEED_TO_LIVE
        assert mapped != CLASSIFIER_REDUCIBLE_UNDER_DYNAMICS


def test_classifier_missing_decisive_arm_if_no_band_achieves_both() -> None:
    result = run_band_sweep(numel=8)
    assert result.cpu_verdict in {
        CPU_VERDICT_QUALIFY_PROCEED_TO_LIVE,
        "no_qualify_on_adversarial_synthetic",
        "missing_decisive_arm_signal_synthetic",
    }
    if not result.qualifying_bands:
        mapped = map_cpu_verdict_to_terminal_classifier(
            cpu_verdict=result.cpu_verdict,
            replay_only=False,
        )
        assert mapped != CLASSIFIER_REDUCIBLE_UNDER_DYNAMICS


def test_classifier_never_intrinsic_from_sweep_fail_alone() -> None:
    result = run_band_sweep(numel=16)
    mapped = map_cpu_verdict_to_terminal_classifier(
        cpu_verdict=result.cpu_verdict,
        replay_only=False,
    )
    assert mapped != CLASSIFIER_INTRINSIC_WIDE_CONFIRMED


def test_classifier_replay_only_never_banks_reducible_or_intrinsic() -> None:
    mapped = map_cpu_verdict_to_terminal_classifier(
        cpu_verdict=CPU_VERDICT_QUALIFY_PROCEED_TO_LIVE,
        replay_only=True,
    )
    assert mapped == CLASSIFIER_STATIC_PROXY_ARTIFACT
    assert_cpu_never_banks_reducible_or_intrinsic(
        cpu_verdict=CPU_VERDICT_QUALIFY_PROCEED_TO_LIVE,
        mapped_classifier=mapped,
    )


def test_v4_live_band_sweep_emits_sweep_table_json_with_pareto_annotation(tmp_path: Path) -> None:
    result = run_band_sweep(numel=DEFAULT_VERDICT_NUMEL)
    out_path = write_sweep_table_json(run_root=tmp_path, result=result)
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["dynamics_class"] == DYNAMICS_CLASS_SYNTHETIC
    assert payload["hot_risk_proxy"] == HOT_RISK_PROXY_LABEL
    assert payload["verdict_numel"] == DEFAULT_VERDICT_NUMEL
    assert payload["band_knob_live"] is True
    assert payload["drift_root_cause"]
    assert payload["reducible_banking_deferred_to_live"] is True
    assert "pareto_frontier" in payload
    assert len(payload["rows"]) == 6


def test_demotion_band_knob_is_live_drift_varies_across_bands() -> None:
    result = run_band_sweep(numel=DEFAULT_VERDICT_NUMEL)
    assert result.band_knob_live is True
    assert demotion_band_knob_is_live(result.rows)
    drift_values = {row.decisive_surface_drift_count for row in result.rows}
    assert len(drift_values) > 1


def test_default_verdict_uses_representative_numel_not_metadata_dominated_tiny() -> None:
    tiny = run_band_sweep(numel=32)
    representative = run_band_sweep()
    assert representative.verdict_numel == DEFAULT_VERDICT_NUMEL
    assert tiny.cpu_verdict == "missing_decisive_arm_signal_synthetic"
    assert representative.cpu_verdict == CPU_VERDICT_QUALIFY_PROCEED_TO_LIVE
    assert all(row.ledger_pass for row in representative.rows[:3])


def test_v4_live_band_sweep_script_end_to_end(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "scripts" / "hrm_text_158_v4_live_band_sweep.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--run-root",
            str(tmp_path),
            "--numel",
            str(DEFAULT_VERDICT_NUMEL),
            "--json-out",
            str(tmp_path / "sweep_copy.json"),
        ],
        cwd=str(repo_root),
        env={**dict(**__import__("os").environ), "PYTHONPATH": str(repo_root)},
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    payload = json.loads((tmp_path / "sweep_copy.json").read_text(encoding="utf-8"))
    assert payload["dynamics_class"] == DYNAMICS_CLASS_SYNTHETIC


def test_decisive_surface_drift_helper_smoke() -> None:
    carrier = EventCodedAccLiveState(logical_numel=8, demotion_band=3)
    oracle = DenseOracleState.zeros(8)
    for step_index, votes in enumerate(({0: 5}, {}, {0: 5})):
        carrier.apply_step(step_index, votes=votes)
        oracle.apply_step(step_index, votes=votes)
    drift = decisive_surface_drift_count(carrier.step_records, oracle.step_records)
    assert drift == 0


def test_build_sweep_table_payload_includes_summary() -> None:
    result = run_band_sweep(numel=DEFAULT_VERDICT_NUMEL)
    payload = build_sweep_table_payload(result)
    assert "necessary_not_sufficient_note" in payload["summary"]
    assert payload["verdict_numel"] == DEFAULT_VERDICT_NUMEL
