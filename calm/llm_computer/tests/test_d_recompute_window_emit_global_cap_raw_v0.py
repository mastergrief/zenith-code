from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    BoundedDeltaTensorState,
    make_bounded_tensor_state,
)
from calm.hrm_text_158.native_full_stack.d_recompute_window_stratified_selector import (
    STRESS_TAIL_POLICY_HORIZON_FIXED,
    StratifiedKeyEntry,
    StratifiedSelectorManifest,
)
from calm.hrm_text_158.native_full_stack.d_recompute_window_emit import (
    D_RECOMPUTE_WINDOW_SCHEMA_VERSION,
    D_RECOMPUTE_WINDOW_SCHEMA_VERSION_V0,
    ReplayConstants,
    _digest_mapping,
    _optional_raw_global_summary_int,
    build_step_log_entry,
    default_production_replay_constants,
    iter_recompute_window_log_records,
    maybe_emit_d_recompute_window_step_records,
    read_global_rate_cap_accepted_count,
    read_global_rate_cap_deferred_count,
)


def _replay() -> ReplayConstants:
    return default_production_replay_constants()


def _fresh_state(acc_values: list[list[int]]) -> BoundedDeltaTensorState:
    acc = torch.tensor(acc_values, dtype=torch.int16)
    q = torch.zeros_like(acc, dtype=torch.int8)
    return make_bounded_tensor_state("tiny.proj", q, 1.0, acc)


def _minimal_lane_args() -> dict:
    return {
        "acc_before": [0],
        "acc_after": [1],
        "q_before": [0],
        "q_after": [0],
        "vote_lanes": [1],
        "lane_indices": [0],
    }


def test_build_step_log_entry_includes_raw_global_cap_counts_when_present() -> None:
    entry = build_step_log_entry(
        step=1,
        state_key="tiny.proj",
        replay_constants=_replay(),
        global_rate_cap_accepted_count=7,
        global_rate_cap_deferred_count=2,
        **_minimal_lane_args(),
    )
    assert entry["schema_version"] == D_RECOMPUTE_WINDOW_SCHEMA_VERSION
    assert entry["global_rate_cap_accepted_count"] == 7
    assert entry["global_rate_cap_deferred_count"] == 2


def test_build_step_log_entry_raw_counts_none_when_not_passed() -> None:
    entry = build_step_log_entry(
        step=1,
        state_key="tiny.proj",
        replay_constants=_replay(),
        **_minimal_lane_args(),
    )
    assert entry["global_rate_cap_accepted_count"] is None
    assert entry["global_rate_cap_deferred_count"] is None


def test_digest_byte_identity_unchanged_when_raw_fields_added() -> None:
    replay = _replay()
    cap_digest = _digest_mapping(
        {
            "global_rate_cap_cap": 128,
            "global_rate_cap_saturated": False,
            "global_rate_cap_enabled": True,
        }
    )
    applied_digest = _digest_mapping(
        {
            "global_rate_cap_accepted_count": 12,
            "global_rate_cap_deferred_count": 3,
            "q_changed_count": 4,
        }
    )
    base_kwargs = {
        "step": 1,
        "state_key": "tiny.proj",
        "replay_constants": replay,
        "cap_order_digest": cap_digest,
        "applied_order_digest": applied_digest,
        **_minimal_lane_args(),
    }
    without_raw = build_step_log_entry(**base_kwargs)
    with_raw = build_step_log_entry(
        **base_kwargs,
        global_rate_cap_accepted_count=12,
        global_rate_cap_deferred_count=3,
    )
    assert without_raw["cap_order_digest"] == cap_digest
    assert without_raw["applied_order_digest"] == applied_digest
    assert with_raw["cap_order_digest"] == cap_digest
    assert with_raw["applied_order_digest"] == applied_digest


def test_optional_raw_global_summary_int_preserves_missingness() -> None:
    assert _optional_raw_global_summary_int(None, "global_rate_cap_accepted_count") is None
    assert (
        _optional_raw_global_summary_int({}, "global_rate_cap_accepted_count") is None
    )
    assert (
        _optional_raw_global_summary_int(
            {"global_rate_cap_accepted_count": None},
            "global_rate_cap_accepted_count",
        )
        is None
    )
    assert (
        _optional_raw_global_summary_int(
            {"global_rate_cap_deferred_count": 2},
            "global_rate_cap_accepted_count",
        )
        is None
    )
    assert (
        _optional_raw_global_summary_int(
            {"global_rate_cap_accepted_count": 9},
            "global_rate_cap_accepted_count",
        )
        == 9
    )


def test_maybe_emit_writes_raw_counts_from_global_summary(tmp_path: Path) -> None:
    replay = _replay()
    state = _fresh_state([[5, -9, 21, 88]])
    states = {"tiny.proj": state}
    votes = {"tiny.proj": torch.tensor([[1, -1, 2, 0]], dtype=torch.int32)}
    log_path = tmp_path / "recompute_window_log.jsonl"
    global_summary = {
        "global_rate_cap_cap": 128,
        "global_rate_cap_saturated": False,
        "global_rate_cap_enabled": True,
        "global_rate_cap_accepted_count": 11,
        "global_rate_cap_deferred_count": 4,
        "q_changed_count": 2,
        "deferred_backlog_size": 0,
    }
    maybe_emit_d_recompute_window_step_records(
        enabled=True,
        log_path=log_path,
        step=3,
        pre_update_states=states,
        post_update_states=states,
        votes_by_key=votes,
        replay_constants=replay,
        global_summary=global_summary,
    )
    records = iter_recompute_window_log_records(log_path)
    assert len(records) == 1
    record = records[0]
    assert record["schema_version"] == D_RECOMPUTE_WINDOW_SCHEMA_VERSION
    assert read_global_rate_cap_accepted_count(record) == 11
    assert read_global_rate_cap_deferred_count(record) == 4
    assert record["cap_order_digest"] is not None
    assert record["applied_order_digest"] is not None


def test_maybe_emit_missing_global_cap_keys_emit_none_not_zero(tmp_path: Path) -> None:
    replay = _replay()
    state = _fresh_state([[5, -9, 21, 88]])
    states = {"tiny.proj": state}
    votes = {"tiny.proj": torch.tensor([[1, -1, 2, 0]], dtype=torch.int32)}
    log_path = tmp_path / "recompute_window_log.jsonl"
    global_summary = {
        "global_rate_cap_cap": 128,
        "global_rate_cap_saturated": False,
        "global_rate_cap_enabled": True,
        "q_changed_count": 2,
        "deferred_backlog_size": 0,
    }
    maybe_emit_d_recompute_window_step_records(
        enabled=True,
        log_path=log_path,
        step=3,
        pre_update_states=states,
        post_update_states=states,
        votes_by_key=votes,
        replay_constants=replay,
        global_summary=global_summary,
    )
    record = iter_recompute_window_log_records(log_path)[0]
    assert "global_rate_cap_accepted_count" in record
    assert "global_rate_cap_deferred_count" in record
    assert record["global_rate_cap_accepted_count"] is None
    assert record["global_rate_cap_deferred_count"] is None
    assert read_global_rate_cap_accepted_count(record) is None
    assert read_global_rate_cap_deferred_count(record) is None


def test_v0_record_without_raw_fields_reads_as_none() -> None:
    v0_record = {
        "schema_version": D_RECOMPUTE_WINDOW_SCHEMA_VERSION_V0,
        "step": 1,
        "state_key": "tiny.proj",
        "applied_order_digest": "abc123",
    }
    assert read_global_rate_cap_accepted_count(v0_record) is None
    assert read_global_rate_cap_deferred_count(v0_record) is None


def test_old_v0_jsonl_remains_parseable(tmp_path: Path) -> None:
    v0_record = build_step_log_entry(
        step=1,
        state_key="tiny.proj",
        replay_constants=_replay(),
        **_minimal_lane_args(),
    )
    v0_record["schema_version"] = D_RECOMPUTE_WINDOW_SCHEMA_VERSION_V0
    v0_record.pop("global_rate_cap_accepted_count", None)
    v0_record.pop("global_rate_cap_deferred_count", None)
    log_path = tmp_path / "legacy.jsonl"
    log_path.write_text(json.dumps(v0_record) + "\n", encoding="utf-8")
    records = iter_recompute_window_log_records(log_path)
    assert len(records) == 1
    assert read_global_rate_cap_accepted_count(records[0]) is None
    assert read_global_rate_cap_deferred_count(records[0]) is None


def _horizon_fixed_manifest(*, state_key: str = "tiny.proj", numel: int = 4096) -> StratifiedSelectorManifest:
    uniform = list(range(0, numel, max(1, numel // 16)))[:16]
    stress = list(range(numel - 16, numel))
    lane_indices = list(uniform) + [index for index in stress if index not in set(uniform)]
    entry = StratifiedKeyEntry(
        state_key=state_key,
        level="L",
        layer_idx=0,
        role="proj",
        depth_tercile="low",
        numel_band="tiny",
        numel=int(numel),
        uniform_lanes=tuple(int(index) for index in uniform),
        stress_tail_lanes=tuple(int(index) for index in stress),
        lane_indices=tuple(int(index) for index in lane_indices),
        stratum_weight=1.0,
    )
    return StratifiedSelectorManifest(
        schema_version="hrm_text_158_d_recompute_window_stratified_selector/v0",
        manifest_sha256="test",
        coverage_tier="REPRESENTATIVE",
        selected_key_count=1,
        stratum_weights={"L_low": 1.0},
        entries=(entry,),
        manifest_spec={"stress_tail_policy": STRESS_TAIL_POLICY_HORIZON_FIXED},
    )


def test_horizon_fixed_emit_no_full_population_before_lane_sample(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    replay = _replay()
    numel = 10_000
    acc = torch.zeros(100, 100, dtype=torch.int16)
    q = torch.zeros_like(acc, dtype=torch.int8)
    state = make_bounded_tensor_state("tiny.proj", q, 1.0, acc)
    states = {"tiny.proj": state}
    votes = {"tiny.proj": torch.zeros(100, 100, dtype=torch.int32)}
    manifest = _horizon_fixed_manifest(numel=numel)
    log_path = tmp_path / "recompute_window_log.jsonl"
    largest_range = {"n": 0}
    original_range = range

    def tracking_range(*args: int, **kwargs: int):
        values = original_range(*args, **kwargs)
        length = len(values)
        if length > largest_range["n"]:
            largest_range["n"] = length
        return values

    monkeypatch.setattr("builtins.range", tracking_range)
    maybe_emit_d_recompute_window_step_records(
        enabled=True,
        log_path=log_path,
        step=1,
        pre_update_states=states,
        post_update_states=states,
        votes_by_key=votes,
        replay_constants=replay,
        selector_manifest=manifest,
    )
    assert largest_range["n"] <= 256


def test_horizon_fixed_emit_golden_schema_matches_banked_slice(tmp_path: Path) -> None:
    banked_log = Path(
        "/home/gabe/claw-code-creditdir/transient_fp_credit/"
        "d_recompute_window_feasibility_seed43_43_2189e72015/"
        "d_recompute_window_diagnostic/recompute_window_log.jsonl"
    )
    if not banked_log.is_file():
        pytest.skip("banked H=100 recompute_window_log unavailable")
    banked_record = json.loads(banked_log.read_text(encoding="utf-8").splitlines()[0])

    replay = _replay()
    numel = 4096
    acc = torch.zeros(64, 64, dtype=torch.int16)
    q = torch.zeros_like(acc, dtype=torch.int8)
    state = make_bounded_tensor_state("tiny.proj", q, 1.0, acc)
    states = {"tiny.proj": state}
    votes = {"tiny.proj": torch.zeros(64, 64, dtype=torch.int32)}
    manifest = _horizon_fixed_manifest(numel=numel)
    log_path = tmp_path / "recompute_window_log.jsonl"
    maybe_emit_d_recompute_window_step_records(
        enabled=True,
        log_path=log_path,
        step=1,
        pre_update_states=states,
        post_update_states=states,
        votes_by_key=votes,
        replay_constants=replay,
        selector_manifest=manifest,
    )
    emitted = iter_recompute_window_log_records(log_path)[0]
    assert set(emitted.keys()) == set(banked_record.keys())
    assert emitted["schema_version"] == banked_record["schema_version"]
    assert emitted["lane_indices"] == list(manifest.entries[0].lane_indices)

