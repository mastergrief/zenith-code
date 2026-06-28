"""In-vivo upper-bound validator for D recompute-window acc envelope sizing.

Authoritative dominance is TOTAL byte-level: build a logged-equivalent checkpoint
payload from the real total window density (all flip events summed across steps
and keys/lanes) and require its measured bytes <= the slice-4 envelope payload.

Per-record / peak checks are necessary pre-screens only — a multi-lane surface
can have total flip events > K while every per-record peak <= K.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import torch

from calm.hrm_text_158.native_full_stack.d_recompute_window_acc_sizing import (
    VERDICT_SCOPE_ENVELOPE_MODEL_ONLY,
    build_conservative_envelope_payload,
)
from calm.hrm_text_158.native_full_stack.d_recompute_window_emit import (
    D_RECOMPUTE_WINDOW_SCHEMA_VERSION,
    D_RECOMPUTE_WINDOW_SCHEMA_VERSION_V0,
    read_global_rate_cap_accepted_count,
    read_global_rate_cap_deferred_count,
)
from calm.hrm_text_158.native_full_stack.d_recompute_window_stratified_selector import (
    COVERAGE_TIER_PILOT,
    STRESS_TAIL_POLICY_HORIZON_FIXED,
    StratifiedSelectorManifest,
)
from calm.hrm_text_158.native_full_stack.event_coded_acc_checkpoint_codec import (
    EventCodedAccEvent,
    pack_event_coded_acc_checkpoint_v1,
)
from calm.hrm_text_158.native_full_stack.event_coded_acc_live_carrier import (
    promotion_carry_threshold,
)
from calm.hrm_text_158.native_full_stack.persistent_state_budget import (
    measure_r4v_event_coded_acc_budget,
)
from calm.hrm_text_158.native_full_stack.qscale_linear import QScaleWeightState
from calm.hrm_text_158.native_full_stack.two_tier_carry_reducers import (
    DEFAULT_CROSSING_THRESHOLD_ABS,
)

IN_VIVO_VALIDATOR_SCHEMA_VERSION = "hrm_text_158_d_recompute_in_vivo_bound/v0"

VERDICT_SCOPE_IN_VIVO_VALIDATED = "in_vivo_validated"

IN_VIVO_DOMINANCE_PROVEN = "DOMINANCE_PROVEN"
IN_VIVO_EXCEEDS = "INCONCLUSIVE_REAL_DENSITY_EXCEEDS_ENVELOPE"
IN_VIVO_INCOMPLETE = "INCONCLUSIVE_INCOMPLETE_OBSERVABLES"
IN_VIVO_MANIFEST_MISMATCH = "INCONCLUSIVE_MANIFEST_LANE_MISMATCH"
IN_VIVO_MANIFEST_COVERAGE_DRIFT = "INCONCLUSIVE_MANIFEST_COVERAGE_DRIFT"
IN_VIVO_ENVELOPE_NOT_SIZED = "INCONCLUSIVE_ENVELOPE_NOT_SIZED"
IN_VIVO_PILOT = "INCONCLUSIVE_PILOT_COVERAGE"
IN_VIVO_GLOBAL_CAP_INCONSISTENT = "INCONCLUSIVE_GLOBAL_CAP_INCONSISTENT"


@dataclass(frozen=True)
class LoggedDensitySurface:
    total_flip_events: int
    peak_flip_events_per_record: int
    peak_flip_events_per_step: int
    peak_backlog_depth: int | None
    total_global_rate_cap_accepted: int | None
    total_global_rate_cap_deferred: int | None
    max_observed_lane_index: int | None
    records_in_window: int
    steps_in_window: int
    raw_global_cap_complete: bool
    schema_version_min: str | None
    manifest_lane_mismatch_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _records_in_measurement_window(
    records: Sequence[Mapping[str, Any]],
    *,
    sizing_horizon_h: int,
    measurement_start_step: int,
) -> list[dict[str, Any]]:
    start = int(measurement_start_step)
    end = int(sizing_horizon_h)
    return [
        dict(record)
        for record in records
        if start <= int(record.get("step", 0)) <= end
    ]


def _flip_count(record: Mapping[str, Any]) -> int:
    flip_lanes = record.get("flip_residual_applied_lanes") or []
    return sum(1 for applied in flip_lanes if bool(applied))


def _aggregate_global_cap_by_step(
    records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any] | None, str | None]:
    by_step: dict[int, tuple[int, int]] = {}
    for record in records:
        step = int(record["step"])
        accepted = read_global_rate_cap_accepted_count(record)
        deferred = read_global_rate_cap_deferred_count(record)
        if accepted is None or deferred is None:
            return None, "missing_raw_global_cap"
        pair = (int(accepted), int(deferred))
        prior = by_step.get(step)
        if prior is None:
            by_step[step] = pair
        elif prior != pair:
            return None, "inconsistent_global_cap_across_keys"
    total_accepted = sum(pair[0] for pair in by_step.values())
    total_deferred = sum(pair[1] for pair in by_step.values())
    return {
        "by_step": {str(step): {"accepted": pair[0], "deferred": pair[1]} for step, pair in sorted(by_step.items())},
        "total_accepted": int(total_accepted),
        "total_deferred": int(total_deferred),
        "step_count": int(len(by_step)),
    }, None


def _verify_manifest_coverage(
    records: Sequence[Mapping[str, Any]],
    manifest: StratifiedSelectorManifest,
) -> str | None:
    entries = manifest.entry_by_key()
    manifest_keys = set(entries.keys())
    observed_keys = {str(record.get("state_key")) for record in records}
    if observed_keys - manifest_keys:
        return "unknown_observed_state_key"
    if manifest_keys - observed_keys:
        return "missing_selected_manifest_key"
    return None


def _verify_manifest_lanes(
    records: Sequence[Mapping[str, Any]],
    manifest: StratifiedSelectorManifest,
) -> int:
    entries = manifest.entry_by_key()
    mismatches = 0
    for record in records:
        state_key = str(record.get("state_key"))
        if state_key not in entries:
            continue
        expected = list(entries[state_key].lane_indices)
        actual = [int(index) for index in record.get("lane_indices") or []]
        if actual != expected:
            mismatches += 1
    return int(mismatches)


def extract_logged_density_surface(
    records: Sequence[Mapping[str, Any]],
    *,
    sizing_horizon_h: int = 100,
    measurement_start_step: int = 1,
    manifest: StratifiedSelectorManifest | None = None,
) -> LoggedDensitySurface:
    window_records = _records_in_measurement_window(
        records,
        sizing_horizon_h=int(sizing_horizon_h),
        measurement_start_step=int(measurement_start_step),
    )
    per_step_flips: dict[int, int] = {}
    peak_per_record = 0
    peak_backlog: int | None = None
    max_lane_index: int | None = None
    schema_versions: set[str] = set()
    raw_complete = True
    for record in window_records:
        schema_versions.add(str(record.get("schema_version") or ""))
        if read_global_rate_cap_accepted_count(record) is None:
            raw_complete = False
        if read_global_rate_cap_deferred_count(record) is None:
            raw_complete = False
        flip_count = _flip_count(record)
        peak_per_record = max(peak_per_record, flip_count)
        step = int(record["step"])
        per_step_flips[step] = per_step_flips.get(step, 0) + flip_count
        backlog = record.get("backlog_depth")
        if backlog is None:
            raw_complete = False
        elif peak_backlog is None or int(backlog) > peak_backlog:
            peak_backlog = int(backlog)
        for lane_index in record.get("lane_indices") or []:
            lane = int(lane_index)
            max_lane_index = lane if max_lane_index is None else max(max_lane_index, lane)

    cap_summary, _cap_error = _aggregate_global_cap_by_step(window_records)
    if cap_summary is None:
        raw_complete = False

    mismatch_count = 0
    if manifest is not None:
        mismatch_count = _verify_manifest_lanes(window_records, manifest)

    schema_min = min(schema_versions) if schema_versions else None
    return LoggedDensitySurface(
        total_flip_events=int(sum(_flip_count(record) for record in window_records)),
        peak_flip_events_per_record=int(peak_per_record),
        peak_flip_events_per_step=int(max(per_step_flips.values(), default=0)),
        peak_backlog_depth=peak_backlog,
        total_global_rate_cap_accepted=(
            None if cap_summary is None else int(cap_summary["total_accepted"])
        ),
        total_global_rate_cap_deferred=(
            None if cap_summary is None else int(cap_summary["total_deferred"])
        ),
        max_observed_lane_index=max_lane_index,
        records_in_window=int(len(window_records)),
        steps_in_window=int(len(per_step_flips)),
        raw_global_cap_complete=bool(raw_complete),
        schema_version_min=schema_min,
        manifest_lane_mismatch_count=int(mismatch_count),
    )


def _residual_mag(acc_before: int, acc_after: int) -> int:
    delta = abs(int(acc_after) - int(acc_before))
    return max(0, min(15, int(delta)))


def _collect_logged_flip_events(
    records: Sequence[Mapping[str, Any]],
) -> tuple[EventCodedAccEvent, ...]:
    events: list[EventCodedAccEvent] = []
    for record in records:
        lane_indices = [int(index) for index in record.get("lane_indices") or []]
        flip_lanes = record.get("flip_residual_applied_lanes") or []
        direction_lanes = record.get("flip_direction_lanes") or []
        acc_before = [int(value) for value in record.get("acc_before_lanes") or []]
        acc_after = [int(value) for value in record.get("acc_after_lanes") or []]
        for position, applied in enumerate(flip_lanes):
            if not bool(applied):
                continue
            direction_raw = (
                direction_lanes[position]
                if position < len(direction_lanes)
                else None
            )
            direction = 1 if direction_raw is None or int(direction_raw) >= 0 else 0
            before = acc_before[position] if position < len(acc_before) else 0
            after = acc_after[position] if position < len(acc_after) else 0
            events.append(
                EventCodedAccEvent(
                    flat_index=int(lane_indices[position]),
                    direction=int(direction),
                    residual_mag=_residual_mag(before, after),
                    event_type=1,
                )
            )
    return tuple(events)


def _collect_logged_backlog_hot(
    records: Sequence[Mapping[str, Any]],
    *,
    numel: int,
    threshold_abs: int = DEFAULT_CROSSING_THRESHOLD_ABS,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    promote_at = int(promotion_carry_threshold(threshold_abs=int(threshold_abs)))
    last_acc: dict[tuple[str, int], int] = {}
    for record in records:
        state_key = str(record.get("state_key"))
        lane_indices = [int(index) for index in record.get("lane_indices") or []]
        acc_after = [int(value) for value in record.get("acc_after_lanes") or []]
        for position, lane_index in enumerate(lane_indices):
            if position < len(acc_after):
                last_acc[(state_key, lane_index)] = int(acc_after[position])

    backlog_indices: list[int] = []
    hot_indices: list[int] = []
    hot_values: list[int] = []
    seen_backlog: set[int] = set()
    seen_hot: set[int] = set()
    for (_state_key, lane_index), acc_value in sorted(last_acc.items()):
        if int(acc_value) == 0:
            continue
        if lane_index not in seen_backlog and lane_index < int(numel):
            seen_backlog.add(lane_index)
            backlog_indices.append(int(lane_index))
        if abs(int(acc_value)) >= promote_at and lane_index not in seen_hot:
            seen_hot.add(lane_index)
            hot_indices.append(int(lane_index))
            hot_values.append(int(acc_value))
    return tuple(backlog_indices), tuple(hot_indices), tuple(hot_values)


def measure_packed_payload_total_bytes(
    payload: Any,
    *,
    numel: int,
    state_key: str = "in_vivo.measure",
) -> dict[str, int]:
    q = torch.zeros(int(numel), dtype=torch.int8)
    qstate = QScaleWeightState(
        q_levels=q.view(1, int(numel)),
        scale=torch.tensor(1.0, dtype=torch.float32),
    )
    report = measure_r4v_event_coded_acc_budget(
        [qstate],
        [payload],
        state_keys=[str(state_key)],
    )
    total = int(
        report.r4v_actual_events_payload_bytes
        + report.r4v_actual_backlog_payload_bytes
        + report.r4v_actual_hot_exact_payload_bytes
        + report.r4v_actual_acc_metadata_bytes
    )
    return {
        "total_payload_bytes": total,
        "events_payload_bytes": int(report.r4v_actual_events_payload_bytes),
        "backlog_payload_bytes": int(report.r4v_actual_backlog_payload_bytes),
        "hot_exact_payload_bytes": int(report.r4v_actual_hot_exact_payload_bytes),
        "metadata_bytes": int(report.r4v_actual_acc_metadata_bytes),
    }


def build_logged_equivalent_payload(
    records: Sequence[Mapping[str, Any]],
    *,
    numel: int,
) -> Any:
    events = _collect_logged_flip_events(records)
    backlog_indices, hot_indices, hot_values = _collect_logged_backlog_hot(
        records,
        numel=int(numel),
    )
    return pack_event_coded_acc_checkpoint_v1(
        logical_numel=int(numel),
        events=events,
        backlog_indices=backlog_indices,
        hot_exact_indices=hot_indices,
        hot_exact_values=hot_values,
    )


def _envelope_backlog_lane_count(
    *,
    window_k: int,
    decay_num: int,
    decay_den: int,
    numel: int,
) -> int:
    retention = float(int(decay_num)) / float(int(decay_den))
    return min(int(numel), max(1, int(math.ceil(float(window_k) * retention))))


def validate_in_vivo_acc_bound(
    records: Sequence[Mapping[str, Any]],
    *,
    manifest: StratifiedSelectorManifest | None,
    envelope_sizing: Mapping[str, Any],
    sizing_horizon_h: int = 100,
    measurement_start_step: int = 1,
    numel_for_bpw: int,
) -> dict[str, Any]:
    surface = extract_logged_density_surface(
        records,
        sizing_horizon_h=int(sizing_horizon_h),
        measurement_start_step=int(measurement_start_step),
        manifest=manifest,
    )
    base: dict[str, Any] = {
        "schema_version": IN_VIVO_VALIDATOR_SCHEMA_VERSION,
        "verdict_scope": VERDICT_SCOPE_ENVELOPE_MODEL_ONLY,
        "not_in_vivo_bound": True,
        "requires_slice5_live_validation": True,
        "logged_density_surface": surface.to_dict(),
        "measurement_start_step": int(measurement_start_step),
        "sizing_horizon_h": int(sizing_horizon_h),
    }

    if manifest is None:
        return base | {
            "in_vivo_verdict": IN_VIVO_INCOMPLETE,
            "reason": "missing_manifest",
        }

    if str(manifest.coverage_tier) == COVERAGE_TIER_PILOT:
        return base | {
            "in_vivo_verdict": IN_VIVO_PILOT,
            "reason": "pilot_coverage_tier",
        }

    policy = str(manifest.manifest_spec.get("stress_tail_policy") or "")
    if policy != STRESS_TAIL_POLICY_HORIZON_FIXED:
        return base | {
            "in_vivo_verdict": IN_VIVO_INCOMPLETE,
            "reason": "non_horizon_fixed_policy",
            "stress_tail_policy": policy,
        }

    if surface.records_in_window <= 0 or surface.steps_in_window <= 0:
        return base | {
            "in_vivo_verdict": IN_VIVO_INCOMPLETE,
            "reason": "empty_measurement_window",
        }

    window_records = _records_in_measurement_window(
        records,
        sizing_horizon_h=int(sizing_horizon_h),
        measurement_start_step=int(measurement_start_step),
    )
    coverage_error = _verify_manifest_coverage(window_records, manifest)
    if coverage_error is not None:
        return base | {
            "in_vivo_verdict": IN_VIVO_MANIFEST_COVERAGE_DRIFT,
            "reason": coverage_error,
        }

    if surface.manifest_lane_mismatch_count > 0:
        return base | {
            "in_vivo_verdict": IN_VIVO_MANIFEST_MISMATCH,
            "reason": "manifest_lane_mismatch",
        }

    cap_summary, cap_error = _aggregate_global_cap_by_step(window_records)
    if cap_summary is None and cap_error == "inconsistent_global_cap_across_keys":
        return base | {
            "in_vivo_verdict": IN_VIVO_GLOBAL_CAP_INCONSISTENT,
            "reason": cap_error,
        }

    if not surface.raw_global_cap_complete:
        return base | {
            "in_vivo_verdict": IN_VIVO_INCOMPLETE,
            "reason": "incomplete_raw_observables",
            "schema_version_min": surface.schema_version_min,
        }

    if surface.schema_version_min == D_RECOMPUTE_WINDOW_SCHEMA_VERSION_V0:
        return base | {
            "in_vivo_verdict": IN_VIVO_INCOMPLETE,
            "reason": "digest_only_v0_log",
        }

    if cap_summary is None:
        return base | {
            "in_vivo_verdict": IN_VIVO_GLOBAL_CAP_INCONSISTENT,
            "reason": cap_error,
        }

    best_row = envelope_sizing.get("best_grid_row")
    window_k = envelope_sizing.get("window_k")
    if not isinstance(best_row, Mapping) or window_k is None:
        return base | {
            "in_vivo_verdict": IN_VIVO_ENVELOPE_NOT_SIZED,
            "reason": "envelope_not_sized",
        }

    window_k_int = int(window_k)
    decay_num = int(best_row["decay_num"])
    decay_den = int(best_row["decay_den"])
    numel = int(numel_for_bpw)

    pre_screens: dict[str, Any] = {
        "peak_flip_events_per_record": surface.peak_flip_events_per_record,
        "peak_flip_events_per_step": surface.peak_flip_events_per_step,
        "total_flip_events": surface.total_flip_events,
        "window_k": window_k_int,
        "peak_backlog_depth": surface.peak_backlog_depth,
        "envelope_backlog_lane_count": _envelope_backlog_lane_count(
            window_k=window_k_int,
            decay_num=decay_num,
            decay_den=decay_den,
            numel=numel,
        ),
        "max_observed_lane_index": surface.max_observed_lane_index,
        "envelope_max_lane_index": int(numel) - 1,
    }

    if surface.peak_flip_events_per_record > window_k_int:
        return base | {
            "in_vivo_verdict": IN_VIVO_EXCEEDS,
            "reason": "peak_flip_events_per_record_exceeds_k",
            "pre_screens": pre_screens,
        }

    if surface.peak_backlog_depth is not None and surface.peak_backlog_depth > pre_screens["envelope_backlog_lane_count"]:
        return base | {
            "in_vivo_verdict": IN_VIVO_EXCEEDS,
            "reason": "peak_backlog_depth_exceeds_envelope",
            "pre_screens": pre_screens,
        }

    if surface.max_observed_lane_index is not None and surface.max_observed_lane_index > int(numel) - 1:
        return base | {
            "in_vivo_verdict": IN_VIVO_EXCEEDS,
            "reason": "max_lane_index_out_of_range",
            "pre_screens": pre_screens,
        }

    if surface.total_flip_events > window_k_int:
        return base | {
            "in_vivo_verdict": IN_VIVO_EXCEEDS,
            "reason": "total_flip_events_exceeds_k",
            "pre_screens": pre_screens,
        }

    logged_payload = build_logged_equivalent_payload(window_records, numel=numel)
    envelope_payload = build_conservative_envelope_payload(
        window_k=window_k_int,
        decay_num=decay_num,
        decay_den=decay_den,
        numel=numel,
    )
    logged_bytes = measure_packed_payload_total_bytes(
        logged_payload,
        numel=numel,
        state_key="in_vivo.logged",
    )
    envelope_bytes = measure_packed_payload_total_bytes(
        envelope_payload,
        numel=numel,
        state_key="in_vivo.envelope",
    )

    byte_comparison = {
        "logged_total_payload_bytes": int(logged_bytes["total_payload_bytes"]),
        "envelope_total_payload_bytes": int(envelope_bytes["total_payload_bytes"]),
        "logged": logged_bytes,
        "envelope": envelope_bytes,
        "logged_event_count": int(len(_collect_logged_flip_events(window_records))),
        "envelope_event_count": int(window_k_int),
    }

    if logged_bytes["total_payload_bytes"] > envelope_bytes["total_payload_bytes"]:
        return base | {
            "in_vivo_verdict": IN_VIVO_EXCEEDS,
            "reason": "logged_total_bytes_exceed_envelope",
            "pre_screens": pre_screens,
            "byte_comparison": byte_comparison,
            "global_cap_summary": cap_summary,
        }

    return base | {
        "in_vivo_verdict": IN_VIVO_DOMINANCE_PROVEN,
        "verdict_scope": VERDICT_SCOPE_IN_VIVO_VALIDATED,
        "not_in_vivo_bound": False,
        "requires_slice5_live_validation": False,
        "reason": "logged_total_bytes_le_envelope",
        "pre_screens": pre_screens,
        "byte_comparison": byte_comparison,
        "global_cap_summary": cap_summary,
        "dominating_density_evidence": {
            "manifest_sha256": str(manifest.manifest_sha256),
            "window_k": window_k_int,
            "decay_num": decay_num,
            "decay_den": decay_den,
            "total_flip_events": surface.total_flip_events,
            "peak_flip_events_per_record": surface.peak_flip_events_per_record,
            "logged_total_payload_bytes": logged_bytes["total_payload_bytes"],
            "envelope_total_payload_bytes": envelope_bytes["total_payload_bytes"],
        },
    }
