from __future__ import annotations

import json
from pathlib import Path

import pytest

from calm.hrm_text_158.native_full_stack.d_recompute_window_emit import (
    D_RECOMPUTE_WINDOW_SCHEMA_VERSION,
    D_RECOMPUTE_WINDOW_SCHEMA_VERSION_V0,
    build_step_log_entry,
    default_production_replay_constants,
)
from calm.hrm_text_158.native_full_stack.d_recompute_window_slice5_bracket_analyzer import (
    BRACKET_ENVELOPE_TOO_TIGHT,
    BRACKET_INSUFFICIENT_LOGS_NEED_LIVE_SNAPSHOT,
    BRACKET_REAL_DENSITY_EXCEEDS_SUB2,
    _adversarial_hot_surface,
    analyze_slice5_density_bracket,
    analyze_slice5_density_bracket_from_run_root,
)


RUN_2189E72017 = Path(
    "/home/gabe/claw-code-creditdir/transient_fp_credit/"
    "d_recompute_window_feasibility_seed43_43_2189e72017"
)


def _replay():
    return default_production_replay_constants()


def _make_record(
    *,
    step: int,
    state_key: str,
    lane_indices: list[int],
    flip_positions: set[int] | None = None,
    schema_version: str = D_RECOMPUTE_WINDOW_SCHEMA_VERSION,
    global_cap: tuple[int, int] | None = (1, 0),
    backlog_depth: int = 0,
    acc_after: list[int] | None = None,
) -> dict:
    flip_positions = flip_positions or set()
    if acc_after is None:
        acc_after = [16 if index in flip_positions else 0 for index in range(len(lane_indices))]
    acc_before = [0 for _ in lane_indices]
    flip_lanes = [index in flip_positions for index in range(len(lane_indices))]
    entry = build_step_log_entry(
        step=int(step),
        state_key=str(state_key),
        replay_constants=_replay(),
        acc_before=acc_before,
        acc_after=acc_after,
        q_before=[0 for _ in lane_indices],
        q_after=[0 for _ in lane_indices],
        vote_lanes=[1 for _ in lane_indices],
        lane_indices=lane_indices,
        flip_residual_applied_lanes=flip_lanes,
        flip_direction_lanes=[1 if applied else None for applied in flip_lanes],
        backlog_depth=int(backlog_depth),
        global_rate_cap_accepted_count=None if global_cap is None else int(global_cap[0]),
        global_rate_cap_deferred_count=None if global_cap is None else int(global_cap[1]),
    )
    entry["schema_version"] = schema_version
    if global_cap is None:
        entry.pop("global_rate_cap_accepted_count", None)
        entry.pop("global_rate_cap_deferred_count", None)
    return entry


def _synthetic_records(
    *,
    steps: int,
    keys: int,
    backlog_depth: int,
    cap_accepted_per_step: int,
    flip: bool = False,
) -> list[dict]:
    records: list[dict] = []
    for step in range(1, int(steps) + 1):
        for key_index in range(int(keys)):
            records.append(
                _make_record(
                    step=step,
                    state_key=f"layer.{key_index}",
                    lane_indices=[0, 1],
                    flip_positions={0} if flip else set(),
                    backlog_depth=int(backlog_depth),
                    global_cap=(int(cap_accepted_per_step), 0),
                )
            )
    return records


def test_bracket_lower_exceeds_budget_synthetic() -> None:
    records = _synthetic_records(
        steps=1,
        keys=1,
        backlog_depth=2_000_000,
        cap_accepted_per_step=0,
    )
    result = analyze_slice5_density_bracket(
        records,
        numel_by_key={"layer.0": 2_000_000},
        sizing_horizon_h=1,
    )
    assert result["bracket_decision"] == BRACKET_REAL_DENSITY_EXCEEDS_SUB2
    assert result["lower_total_bytes"] > result["budget_bytes"]


def test_bracket_upper_under_budget_synthetic() -> None:
    records = _synthetic_records(
        steps=1,
        keys=1,
        backlog_depth=4,
        cap_accepted_per_step=0,
    )
    result = analyze_slice5_density_bracket(
        records,
        numel_by_key={"layer.0": 1_000_000},
        sizing_horizon_h=1,
    )
    assert result["bracket_decision"] == BRACKET_ENVELOPE_TOO_TIGHT
    assert (
        result["sample_limited_adversarial_upper_bytes"]
        < result["budget_bytes"]
    )


def test_bracket_straddle_synthetic() -> None:
    records = _synthetic_records(
        steps=1,
        keys=1,
        backlog_depth=130_000,
        cap_accepted_per_step=51_712,
    )
    result = analyze_slice5_density_bracket(
        records,
        numel_by_key={"layer.0": 16_000_000},
        sizing_horizon_h=1,
    )
    assert result["lower_total_bytes"] < result["budget_bytes"]
    assert (
        result["sample_limited_adversarial_upper_bytes"]
        > result["budget_bytes"]
    )
    assert result["bracket_decision"] == BRACKET_INSUFFICIENT_LOGS_NEED_LIVE_SNAPSHOT


def test_bracket_insufficient_missing_backlog_depth() -> None:
    record = _make_record(
        step=1,
        state_key="layer.0",
        lane_indices=[0],
        backlog_depth=0,
    )
    record.pop("backlog_depth")
    result = analyze_slice5_density_bracket(
        [record],
        numel_by_key={"layer.0": 128},
        sizing_horizon_h=1,
    )
    assert "missing_peak_backlog_depth" in result["honesty_fail_reasons"]
    assert result["bracket_decision"] == BRACKET_INSUFFICIENT_LOGS_NEED_LIVE_SNAPSHOT


def test_bracket_never_promotes_recommended_law() -> None:
    records = _synthetic_records(
        steps=1,
        keys=1,
        backlog_depth=4,
        cap_accepted_per_step=0,
    )
    result = analyze_slice5_density_bracket(
        records,
        numel_by_key={"layer.0": 1_000_000},
        sizing_horizon_h=1,
    )
    flags = result["honesty_flags"]
    assert flags["recommended_law_eligible"] is False
    assert flags["in_vivo_validated"] is False
    assert flags["live_carrier_bytes_exact"] is False


@pytest.mark.skipif(not RUN_2189E72017.is_dir(), reason="preserved run_root unavailable")
def test_bracket_2189e72017_insufficient_under_adversarial_upper() -> None:
    result = analyze_slice5_density_bracket_from_run_root(RUN_2189E72017)
    assert result["bracket_decision"] == BRACKET_INSUFFICIENT_LOGS_NEED_LIVE_SNAPSHOT
    assert result["lower_total_bytes"] < result["budget_bytes"]
    assert (
        result["sample_limited_adversarial_upper_bytes"]
        > result["budget_bytes"]
    )
    assert (
        result["observed_sample_upper_bytes"]
        < result["sample_limited_adversarial_upper_bytes"]
    )
    assert result["event_upper_count"] == 51712
    assert result["honesty_flags"]["live_carrier_bytes_exact"] is False
    assert "live_runtime_acc_working_set_bpw_proxy" in result
    source = result["source_artifacts"]
    assert source["selector_manifest_path"].endswith("calibrated_selector_manifest.json")
    assert "selector_manifest_sha256" in source
    assert source["postrun_input_manifest_path"].endswith("postrun_input_manifest.json")
    assert source["selector_manifest_sha256"] != source["postrun_input_manifest_sha256"]
    assert "postrun_input_manifest_sha256" not in result["artifact_hashes"]


def test_adversarial_hot_values_are_clip_valid() -> None:
    _, hot_values = _adversarial_hot_surface(numel=16_000_000, peak_backlog_depth=8)
    assert hot_values
    assert all(-127 <= value <= 127 for value in hot_values)
    assert -128 not in hot_values


def test_sampled_hot_undercoverage_flips_to_insufficient() -> None:
    records = _synthetic_records(
        steps=1,
        keys=1,
        backlog_depth=130_816,
        cap_accepted_per_step=0,
    )
    result = analyze_slice5_density_bracket(
        records,
        numel_by_key={"layer.0": 16_000_000},
        sizing_horizon_h=1,
    )
    assert result["observed_sample_upper_bytes"] < result["budget_bytes"]
    assert (
        result["sample_limited_adversarial_upper_bytes"]
        > result["budget_bytes"]
    )
    assert result["bracket_decision"] == BRACKET_INSUFFICIENT_LOGS_NEED_LIVE_SNAPSHOT


def test_cap_accepted_drives_event_upper_bytes() -> None:
    low_cap = _synthetic_records(
        steps=1,
        keys=1,
        backlog_depth=10,
        cap_accepted_per_step=1,
        flip=False,
    )
    high_cap = _synthetic_records(
        steps=1,
        keys=1,
        backlog_depth=10,
        cap_accepted_per_step=500,
        flip=False,
    )
    low = analyze_slice5_density_bracket(
        low_cap,
        numel_by_key={"layer.0": 10_000},
        sizing_horizon_h=1,
    )
    high = analyze_slice5_density_bracket(
        high_cap,
        numel_by_key={"layer.0": 10_000},
        sizing_horizon_h=1,
    )
    assert high["event_upper_count"] > low["event_upper_count"]
    assert (
        high["sample_limited_adversarial_upper"]["events_payload_bytes"]
        > low["sample_limited_adversarial_upper"]["events_payload_bytes"]
    )


def test_digest_only_v0_log_is_insufficient() -> None:
    record = _make_record(
        step=1,
        state_key="layer.0",
        lane_indices=[0],
        backlog_depth=8,
        schema_version=D_RECOMPUTE_WINDOW_SCHEMA_VERSION_V0,
    )
    result = analyze_slice5_density_bracket(
        [record],
        numel_by_key={"layer.0": 128},
        sizing_horizon_h=1,
    )
    assert "digest_only_v0_log" in result["honesty_fail_reasons"]
    assert result["bracket_decision"] == BRACKET_INSUFFICIENT_LOGS_NEED_LIVE_SNAPSHOT
