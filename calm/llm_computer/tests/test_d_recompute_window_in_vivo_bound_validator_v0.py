from __future__ import annotations

import json

import pytest

from calm.hrm_text_158.native_full_stack.d_recompute_window_acc_sizing import (
    SIZING_VERDICT_RECOMMENDED_LAW,
    measure_envelope_inclusive_bpw,
)
from calm.hrm_text_158.native_full_stack.d_recompute_window_emit import (
    D_RECOMPUTE_WINDOW_SCHEMA_VERSION,
    D_RECOMPUTE_WINDOW_SCHEMA_VERSION_V0,
    build_step_log_entry,
    default_production_replay_constants,
)
from calm.hrm_text_158.native_full_stack.d_recompute_window_in_vivo_bound_validator import (
    IN_VIVO_DOMINANCE_PROVEN,
    IN_VIVO_ENVELOPE_NOT_SIZED,
    IN_VIVO_EXCEEDS,
    IN_VIVO_GLOBAL_CAP_INCONSISTENT,
    IN_VIVO_INCOMPLETE,
    IN_VIVO_MANIFEST_COVERAGE_DRIFT,
    IN_VIVO_MANIFEST_MISMATCH,
    VERDICT_SCOPE_IN_VIVO_VALIDATED,
    build_logged_equivalent_payload,
    extract_logged_density_surface,
    measure_packed_payload_total_bytes,
    validate_in_vivo_acc_bound,
)
from calm.hrm_text_158.native_full_stack.d_recompute_window_stratified_selector import (
    COVERAGE_TIER_REPRESENTATIVE,
    STRESS_TAIL_POLICY_HORIZON_FIXED,
    StratifiedSelectorManifest,
)


def _replay():
    return default_production_replay_constants()


def _make_record(
    *,
    step: int,
    state_key: str,
    lane_indices: list[int],
    flip_positions: set[int],
    schema_version: str = D_RECOMPUTE_WINDOW_SCHEMA_VERSION,
    global_cap: tuple[int, int] | None = (1, 0),
    backlog_depth: int = 0,
) -> dict:
    acc_before = [0 for _ in lane_indices]
    acc_after = [1 if index in flip_positions else 0 for index in range(len(lane_indices))]
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


def _minimal_manifest(*, lane_indices: tuple[int, ...] = (0,)) -> StratifiedSelectorManifest:
    body = {
        "schema_version": "hrm_text_158_stratified_selector_manifest/v0",
        "coverage_tier": COVERAGE_TIER_REPRESENTATIVE,
        "selected_key_count": 2,
        "stratum_weights": {"key.a": 0.5, "key.b": 0.5},
        "manifest_spec": {
            "stress_tail_policy": STRESS_TAIL_POLICY_HORIZON_FIXED,
            "measurement_start_step": 1,
        },
        "entries": [
            {
                "state_key": "key.a",
                "level": "H",
                "layer_idx": 0,
                "role": "attn_q",
                "depth_tercile": "early",
                "numel_band": "small",
                "numel": 8,
                "uniform_lanes": list(lane_indices),
                "stress_tail_lanes": list(lane_indices),
                "lane_indices": list(lane_indices),
                "stratum_weight": 0.5,
            },
            {
                "state_key": "key.b",
                "level": "H",
                "layer_idx": 1,
                "role": "attn_q",
                "depth_tercile": "early",
                "numel_band": "small",
                "numel": 8,
                "uniform_lanes": list(lane_indices),
                "stress_tail_lanes": list(lane_indices),
                "lane_indices": list(lane_indices),
                "stratum_weight": 0.5,
            },
        ],
    }
    digest = __import__("hashlib").sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return StratifiedSelectorManifest.from_dict(
        {
            **body,
            "manifest_sha256": digest,
        }
    )


def _envelope_sizing(*, window_k: int = 10) -> dict:
    row = measure_envelope_inclusive_bpw(
        window_k=int(window_k),
        decay_num=1,
        decay_den=1,
        numel=16,
    )
    return {
        "window_k": int(window_k),
        "best_grid_row": row,
        "sizing_verdict": SIZING_VERDICT_RECOMMENDED_LAW,
        "strict_less_than_budget": True,
    }


def test_v1_complete_low_density_validates_in_vivo() -> None:
    manifest = _minimal_manifest()
    records = [
        _make_record(step=1, state_key="key.a", lane_indices=[0], flip_positions={0}),
        _make_record(step=1, state_key="key.b", lane_indices=[0], flip_positions=set()),
    ]
    result = validate_in_vivo_acc_bound(
        records,
        manifest=manifest,
        envelope_sizing=_envelope_sizing(window_k=10),
        sizing_horizon_h=10,
        numel_for_bpw=16,
    )
    assert result["in_vivo_verdict"] == IN_VIVO_DOMINANCE_PROVEN
    assert result["verdict_scope"] == VERDICT_SCOPE_IN_VIVO_VALIDATED
    assert result["requires_slice5_live_validation"] is False


def test_v0_or_missing_raw_global_cap_is_inconclusive() -> None:
    manifest = _minimal_manifest()
    records = [
        _make_record(
            step=1,
            state_key="key.a",
            lane_indices=[0],
            flip_positions=set(),
            schema_version=D_RECOMPUTE_WINDOW_SCHEMA_VERSION_V0,
            global_cap=None,
        ),
        _make_record(
            step=1,
            state_key="key.b",
            lane_indices=[0],
            flip_positions=set(),
            schema_version=D_RECOMPUTE_WINDOW_SCHEMA_VERSION_V0,
            global_cap=None,
        ),
    ]
    result = validate_in_vivo_acc_bound(
        records,
        manifest=manifest,
        envelope_sizing=_envelope_sizing(window_k=10),
        sizing_horizon_h=10,
        numel_for_bpw=16,
    )
    assert result["in_vivo_verdict"] == IN_VIVO_INCOMPLETE


def test_multi_lane_total_exceeds_k_is_inconclusive_even_when_peak_le_k() -> None:
    manifest = _minimal_manifest()
    records = []
    for step in (1, 2):
        for key in ("key.a", "key.b"):
            records.append(
                _make_record(
                    step=step,
                    state_key=key,
                    lane_indices=[0],
                    flip_positions={0},
                )
            )
    surface = extract_logged_density_surface(records, sizing_horizon_h=10, manifest=manifest)
    assert surface.peak_flip_events_per_record == 1
    assert surface.total_flip_events == 4
    result = validate_in_vivo_acc_bound(
        records,
        manifest=manifest,
        envelope_sizing=_envelope_sizing(window_k=3),
        sizing_horizon_h=10,
        numel_for_bpw=16,
    )
    assert result["in_vivo_verdict"] == IN_VIVO_EXCEEDS
    assert result["reason"] == "total_flip_events_exceeds_k"


def test_manifest_lane_mismatch_fail_closed() -> None:
    manifest = _minimal_manifest(lane_indices=(0,))
    records = [
        _make_record(step=1, state_key="key.a", lane_indices=[1], flip_positions=set()),
        _make_record(step=1, state_key="key.b", lane_indices=[0], flip_positions=set()),
    ]
    result = validate_in_vivo_acc_bound(
        records,
        manifest=manifest,
        envelope_sizing=_envelope_sizing(window_k=10),
        sizing_horizon_h=10,
        numel_for_bpw=16,
    )
    assert result["in_vivo_verdict"] == IN_VIVO_MANIFEST_MISMATCH


def test_high_index_logged_payload_uses_observed_lane_index() -> None:
    records = [
        _make_record(step=1, state_key="key.a", lane_indices=[7], flip_positions={0}),
    ]
    payload = build_logged_equivalent_payload(records, numel=8)
    events = __import__(
        "calm.hrm_text_158.native_full_stack.d_recompute_window_in_vivo_bound_validator",
        fromlist=["_collect_logged_flip_events"],
    )._collect_logged_flip_events(records)
    assert events[0].flat_index == 7
    measured = measure_packed_payload_total_bytes(payload, numel=8)
    assert measured["total_payload_bytes"] > 0


def test_envelope_not_sized_is_inconclusive() -> None:
    manifest = _minimal_manifest()
    records = [
        _make_record(step=1, state_key="key.a", lane_indices=[0], flip_positions=set()),
        _make_record(step=1, state_key="key.b", lane_indices=[0], flip_positions=set()),
    ]
    result = validate_in_vivo_acc_bound(
        records,
        manifest=manifest,
        envelope_sizing={"sizing_verdict": "INCONCLUSIVE"},
        sizing_horizon_h=10,
        numel_for_bpw=16,
    )
    assert result["in_vivo_verdict"] == IN_VIVO_ENVELOPE_NOT_SIZED


def test_global_cap_step_aggregation_requires_cross_key_consistency() -> None:
    manifest = _minimal_manifest()
    records = [
        _make_record(step=1, state_key="key.a", lane_indices=[0], flip_positions=set(), global_cap=(2, 0)),
        _make_record(step=1, state_key="key.b", lane_indices=[0], flip_positions=set(), global_cap=(3, 0)),
    ]
    result = validate_in_vivo_acc_bound(
        records,
        manifest=manifest,
        envelope_sizing=_envelope_sizing(window_k=10),
        sizing_horizon_h=10,
        numel_for_bpw=16,
    )
    assert result["in_vivo_verdict"] == IN_VIVO_GLOBAL_CAP_INCONSISTENT


def test_unknown_observed_state_key_is_coverage_drift_not_dominance() -> None:
    manifest = _minimal_manifest()
    records = [
        _make_record(step=1, state_key="key.a", lane_indices=[0], flip_positions=set()),
        _make_record(step=1, state_key="key.b", lane_indices=[0], flip_positions=set()),
        _make_record(step=1, state_key="key.unknown", lane_indices=[0], flip_positions=set()),
    ]
    result = validate_in_vivo_acc_bound(
        records,
        manifest=manifest,
        envelope_sizing=_envelope_sizing(window_k=10),
        sizing_horizon_h=10,
        numel_for_bpw=16,
    )
    assert result["in_vivo_verdict"] == IN_VIVO_MANIFEST_COVERAGE_DRIFT
    assert result["reason"] == "unknown_observed_state_key"
    assert result["in_vivo_verdict"] != IN_VIVO_DOMINANCE_PROVEN


def test_missing_selected_manifest_key_is_inconclusive() -> None:
    manifest = _minimal_manifest()
    records = [
        _make_record(step=1, state_key="key.a", lane_indices=[0], flip_positions=set()),
    ]
    result = validate_in_vivo_acc_bound(
        records,
        manifest=manifest,
        envelope_sizing=_envelope_sizing(window_k=10),
        sizing_horizon_h=10,
        numel_for_bpw=16,
    )
    assert result["in_vivo_verdict"] == IN_VIVO_MANIFEST_COVERAGE_DRIFT
    assert result["reason"] == "missing_selected_manifest_key"


def test_empty_measurement_window_is_inconclusive_not_vacuous_dominance() -> None:
    manifest = _minimal_manifest()
    records = [
        _make_record(step=99, state_key="key.a", lane_indices=[0], flip_positions=set()),
        _make_record(step=99, state_key="key.b", lane_indices=[0], flip_positions=set()),
    ]
    result = validate_in_vivo_acc_bound(
        records,
        manifest=manifest,
        envelope_sizing=_envelope_sizing(window_k=10),
        sizing_horizon_h=10,
        numel_for_bpw=16,
    )
    assert result["in_vivo_verdict"] == IN_VIVO_INCOMPLETE
    assert result["reason"] == "empty_measurement_window"
    assert result["in_vivo_verdict"] != IN_VIVO_DOMINANCE_PROVEN
