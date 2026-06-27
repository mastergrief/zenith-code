"""Offline CPU D recompute-window feasibility analyzer.

Measures per-lane K*, inclusive bpw, and classifies under the locked D
precedence without mutating banked run artifacts.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from calm.hrm_text_158.native_full_stack.d_recompute_window_emit import (
    BOOTSTRAP_KNOWN_SATURATED_NEGATIVE,
    BOOTSTRAP_KNOWN_SATURATED_POSITIVE,
    BOOTSTRAP_KNOWN_ZERO,
    D_RECOMPUTE_WINDOW_SCHEMA_VERSION,
    ReplayConstants,
    default_production_replay_constants,
    iter_recompute_window_log_records,
    validate_bootstrap_record,
)
from calm.hrm_text_158.native_full_stack.persistent_state_budget import (
    PACKED_BASE3_TERNARY_Q_FORMAT,
    R4B_Q_PHYSICAL_BITS_PER_WEIGHT_BASE3,
    TARGET_PHYSICAL_BITS_PER_WEIGHT,
)

ANALYZER_SCHEMA_VERSION = "hrm_text_158_d_recompute_window_feasibility/v0"

CLASSIFIER_MISSING_OBSERVABLES_OR_INVALID_WINDOW = "MISSING_OBSERVABLES_OR_INVALID_WINDOW"
CLASSIFIER_D_RECOMPUTE_WINDOW_LEAD = "D_RECOMPUTE_WINDOW_LEAD"
CLASSIFIER_D_RECOMPUTE_UNBOUNDED_OR_UNOBSERVABLE = "D_RECOMPUTE_UNBOUNDED_OR_UNOBSERVABLE"
CLASSIFIER_D_NEEDS_UPDATE_LAW_REDESIGN = "D_NEEDS_UPDATE_LAW_REDESIGN"
CLASSIFIER_NO_CARRIER_FAMILY_VIABLE = "NO_CARRIER_FAMILY_VIABLE_ON_EXISTING_ARTIFACTS"

CLASSIFIER_PRECEDENCE: tuple[str, ...] = (
    CLASSIFIER_MISSING_OBSERVABLES_OR_INVALID_WINDOW,
    CLASSIFIER_D_RECOMPUTE_WINDOW_LEAD,
    CLASSIFIER_D_RECOMPUTE_UNBOUNDED_OR_UNOBSERVABLE,
    CLASSIFIER_D_NEEDS_UPDATE_LAW_REDESIGN,
    CLASSIFIER_NO_CARRIER_FAMILY_VIABLE,
)

LANE_CLASS_SATURATED_AT_CLAMP = "saturated_at_clamp"
LANE_CLASS_FREQUENTLY_FLIPPING = "frequently_flipping"
LANE_CLASS_SLOW_SUB_SATURATION = "slow_sub_saturation"

W8_DENSE_ACC_TERM_BPW = 8.0
SUB2_INCLUSIVE_TARGET_BPW = float(TARGET_PHYSICAL_BITS_PER_WEIGHT)
DECLARED_Q_BPW_BASE3 = float(R4B_Q_PHYSICAL_BITS_PER_WEIGHT_BASE3)
ACC_BUDGET_BPW_UNDER_BASE3_Q = SUB2_INCLUSIVE_TARGET_BPW - DECLARED_Q_BPW_BASE3

REQUIRED_LOG_FIELDS: tuple[str, ...] = (
    "schema_version",
    "step",
    "state_key",
    "replay_constants",
    "lane_indices",
    "vote_lanes",
    "acc_before_lanes",
    "acc_after_lanes",
    "q_before_lanes",
    "q_after_lanes",
    "flip_residual_applied_lanes",
    "residual_authority_lanes",
    "cap_order_digest",
    "applied_order_digest",
    "vote_source_digest",
)

EXPLICIT_NON_CLAIMS: tuple[str, ...] = (
    "d_feasibility_hypothesis_not_winner",
    "no_sub2_total_claim",
    "no_held_rules_unlock",
    "w8_faithfulness_only_not_universal_transparency",
    "w7_negative_stands",
    "d_addresses_acc_leg_only_live_q_remains_int8",
    "b_annex_optional_non_authoritative",
    "no_carrier_family_viable_is_bounded_null_not_exhaustive_proof",
)

FORBIDDEN_B_TERMINAL_FROM_D_FIELDS = "B_APPROX_DENSE_LEAD"


@dataclass(frozen=True)
class LaneKStarMeasurement:
    lane_index: int
    target_step: int
    k_star: int | None
    lane_class: str
    parity_pass: bool
    bootstrap_used: str | None


def carry_after_scalar(
    acc: int,
    vote: int,
    *,
    replay: ReplayConstants,
) -> int:
    decayed = int(acc)
    if int(replay.decay_numerator) != 1 or int(replay.decay_denominator) != 1:
        decayed = (int(acc) * int(replay.decay_numerator)) // int(replay.decay_denominator)
    value = int(decayed) + int(vote)
    return max(int(replay.accumulator_clip_min), min(int(replay.accumulator_clip_max), value))


def apply_flip_residual_scalar(acc: int, *, direction: int, threshold: int) -> int:
    residual = int(acc) - int(direction) * int(threshold)
    low = -int(threshold) + 1
    high = int(threshold) - 1
    return max(low, min(high, residual))


def classify_lane_step(
    *,
    acc_before: int,
    acc_after: int,
    vote: int,
    replay: ReplayConstants,
    flip_residual_applied: bool,
) -> str:
    clip_min = int(replay.accumulator_clip_min)
    clip_max = int(replay.accumulator_clip_max)
    if acc_after in (clip_min, clip_max):
        return LANE_CLASS_SATURATED_AT_CLAMP
    if flip_residual_applied:
        return LANE_CLASS_FREQUENTLY_FLIPPING
    expected = carry_after_scalar(acc_before, vote, replay=replay)
    if expected != acc_after and not flip_residual_applied:
        return LANE_CLASS_FREQUENTLY_FLIPPING
    return LANE_CLASS_SLOW_SUB_SATURATION


def _valid_bootstrap_for_acc(
    acc_value: int,
    *,
    replay: ReplayConstants,
) -> list[str]:
    clip_min = int(replay.accumulator_clip_min)
    clip_max = int(replay.accumulator_clip_max)
    options = [BOOTSTRAP_KNOWN_ZERO]
    if int(acc_value) == clip_max:
        options.append(BOOTSTRAP_KNOWN_SATURATED_POSITIVE)
    if int(acc_value) == clip_min:
        options.append(BOOTSTRAP_KNOWN_SATURATED_NEGATIVE)
    return options


def bootstrap_starting_acc(bootstrap: str, *, replay: ReplayConstants) -> int:
    if bootstrap == BOOTSTRAP_KNOWN_ZERO:
        return 0
    if bootstrap == BOOTSTRAP_KNOWN_SATURATED_POSITIVE:
        return int(replay.accumulator_clip_max)
    if bootstrap == BOOTSTRAP_KNOWN_SATURATED_NEGATIVE:
        return int(replay.accumulator_clip_min)
    raise ValueError(f"invalid bootstrap {bootstrap!r}")


def reconstruct_lane_from_bootstrap(
    *,
    bootstrap: str,
    votes: Sequence[int],
    replay: ReplayConstants,
    flip_residual_flags: Sequence[bool] | None = None,
    flip_directions: Sequence[int | None] | None = None,
    flip_thresholds: Sequence[int | None] | None = None,
) -> int:
    acc = bootstrap_starting_acc(bootstrap, replay=replay)
    for index, vote in enumerate(votes):
        acc = carry_after_scalar(acc, int(vote), replay=replay)
        if flip_residual_flags and index < len(flip_residual_flags):
            if bool(flip_residual_flags[index]):
                direction = int(flip_directions[index]) if flip_directions else 1
                threshold = (
                    int(flip_thresholds[index])
                    if flip_thresholds and flip_thresholds[index] is not None
                    else int(replay.threshold_abs)
                )
                acc = apply_flip_residual_scalar(
                    acc,
                    direction=direction,
                    threshold=threshold,
                )
    return int(acc)


def _lane_flip_fields(
    entry: Mapping[str, Any],
    lane_position: int,
) -> tuple[bool, int | None, int | None, str]:
    applied_lanes = entry.get("flip_residual_applied_lanes")
    direction_lanes = entry.get("flip_direction_lanes")
    threshold_lanes = entry.get("flip_threshold_lanes")
    authority_lanes = entry.get("residual_authority_lanes")
    if isinstance(applied_lanes, list) and lane_position < len(applied_lanes):
        applied = bool(applied_lanes[lane_position])
        direction = (
            direction_lanes[lane_position]
            if isinstance(direction_lanes, list) and lane_position < len(direction_lanes)
            else entry.get("flip_direction")
        )
        threshold = (
            threshold_lanes[lane_position]
            if isinstance(threshold_lanes, list) and lane_position < len(threshold_lanes)
            else entry.get("flip_threshold")
        )
        authority = (
            str(authority_lanes[lane_position])
            if isinstance(authority_lanes, list) and lane_position < len(authority_lanes)
            else "absent"
        )
        return applied, None if direction is None else int(direction), (
            None if threshold is None else int(threshold)
        ), authority
    return (
        bool(entry.get("flip_residual_applied")),
        None if entry.get("flip_direction") is None else int(entry.get("flip_direction")),
        None if entry.get("flip_threshold") is None else int(entry.get("flip_threshold")),
        "absent",
    )


def measure_lane_k_star(
    *,
    lane_index: int,
    step_entries: Sequence[Mapping[str, Any]],
    lane_position: int,
    replay: ReplayConstants,
) -> LaneKStarMeasurement:
    if not step_entries:
        return LaneKStarMeasurement(
            lane_index=int(lane_index),
            target_step=0,
            k_star=None,
            lane_class=LANE_CLASS_SLOW_SUB_SATURATION,
            parity_pass=False,
            bootstrap_used=None,
        )
    target_step = int(step_entries[-1]["step"])
    target_acc = int(step_entries[-1]["acc_after_lanes"][lane_position])
    last_flip_applied, _, _, last_authority = _lane_flip_fields(
        step_entries[-1],
        lane_position,
    )
    lane_class = classify_lane_step(
        acc_before=int(step_entries[-1]["acc_before_lanes"][lane_position]),
        acc_after=target_acc,
        vote=int(step_entries[-1]["vote_lanes"][lane_position]),
        replay=replay,
        flip_residual_applied=bool(last_flip_applied),
    )
    if last_authority == "absent" and carry_after_scalar(
        int(step_entries[-1]["acc_before_lanes"][lane_position]),
        int(step_entries[-1]["vote_lanes"][lane_position]),
        replay=replay,
    ) != target_acc:
        return LaneKStarMeasurement(
            lane_index=int(lane_index),
            target_step=int(target_step),
            k_star=None,
            lane_class=str(lane_class),
            parity_pass=False,
            bootstrap_used=None,
        )
    best_k: int | None = None
    best_bootstrap: str | None = None
    for bootstrap in (
        BOOTSTRAP_KNOWN_ZERO,
        BOOTSTRAP_KNOWN_SATURATED_POSITIVE,
        BOOTSTRAP_KNOWN_SATURATED_NEGATIVE,
    ):
        saturated_start = bootstrap_starting_acc(bootstrap, replay=replay)
        if (
            bootstrap
            in (BOOTSTRAP_KNOWN_SATURATED_POSITIVE, BOOTSTRAP_KNOWN_SATURATED_NEGATIVE)
            and int(saturated_start) == int(target_acc)
        ):
            if best_k is None or 0 < best_k:
                best_k = 0
                best_bootstrap = bootstrap
        for k in range(1, len(step_entries) + 1):
            window = step_entries[-k:]
            start_acc = int(window[0]["acc_before_lanes"][lane_position])
            if bootstrap not in _valid_bootstrap_for_acc(start_acc, replay=replay):
                if not (bootstrap == BOOTSTRAP_KNOWN_ZERO and start_acc == 0):
                    continue
            votes = [int(entry["vote_lanes"][lane_position]) for entry in window]
            flip_flags: list[bool] = []
            flip_dirs: list[int | None] = []
            flip_thrs: list[int | None] = []
            for entry in window:
                applied, direction, threshold, authority = _lane_flip_fields(
                    entry,
                    lane_position,
                )
                if authority == "absent" and applied:
                    flip_flags = []
                    break
                flip_flags.append(bool(applied))
                flip_dirs.append(direction)
                flip_thrs.append(threshold)
            if not flip_flags and any(
                carry_after_scalar(
                    int(entry["acc_before_lanes"][lane_position]),
                    int(entry["vote_lanes"][lane_position]),
                    replay=replay,
                )
                != int(entry["acc_after_lanes"][lane_position])
                for entry in window
            ):
                break
            reconstructed = reconstruct_lane_from_bootstrap(
                bootstrap=bootstrap,
                votes=votes,
                replay=replay,
                flip_residual_flags=flip_flags,
                flip_directions=flip_dirs,
                flip_thresholds=flip_thrs,
            )
            if reconstructed == target_acc:
                if best_k is None or k < best_k:
                    best_k = int(k)
                    best_bootstrap = bootstrap
                break
    return LaneKStarMeasurement(
        lane_index=int(lane_index),
        target_step=int(target_step),
        k_star=best_k,
        lane_class=str(lane_class),
        parity_pass=best_k is not None,
        bootstrap_used=best_bootstrap,
    )


def _percentile(values: Sequence[int], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(int(value) for value in values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * (float(pct) / 100.0)
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return float(ordered[low])
    weight = rank - low
    return float(ordered[low] * (1.0 - weight) + ordered[high] * weight)


def inclusive_stream_bpw(*, numel: int, stream_bytes: int) -> float:
    if int(numel) <= 0:
        return float("inf")
    return 8.0 * float(stream_bytes) / float(numel)


def dual_budget_booleans(
    *,
    acc_term_bpw: float,
    byte_model_declared: bool,
) -> dict[str, bool]:
    beats_w8 = bool(byte_model_declared and float(acc_term_bpw) < W8_DENSE_ACC_TERM_BPW)
    sub2_total = bool(
        byte_model_declared and float(acc_term_bpw) < ACC_BUDGET_BPW_UNDER_BASE3_Q
    )
    return {
        "beats_w8_dense_acc_term": beats_w8,
        "sub2_total_candidate_under_named_q_basis": sub2_total,
        "byte_model_declared": bool(byte_model_declared),
        "named_q_basis": PACKED_BASE3_TERNARY_Q_FORMAT,
        "declared_q_bpw": DECLARED_Q_BPW_BASE3,
        "acc_budget_bpw_under_base3_q": ACC_BUDGET_BPW_UNDER_BASE3_Q,
    }


def _group_entries_by_lane(
    records: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for record in records:
        state_key = str(record["state_key"])
        lane_indices = record.get("lane_indices") or []
        for position, lane_index in enumerate(lane_indices):
            key = (state_key, int(lane_index))
            grouped.setdefault(key, []).append(dict(record) | {"_lane_position": position})
    for key in grouped:
        grouped[key].sort(key=lambda entry: int(entry["step"]))
    return grouped


def analyze_recompute_window_log(
    log_path: Path | str,
    *,
    numel_for_bpw: int | None = None,
) -> dict[str, Any]:
    path = Path(log_path)
    records = iter_recompute_window_log_records(path)
    if not records:
        return {
            "schema_version": ANALYZER_SCHEMA_VERSION,
            "primary_classifier": CLASSIFIER_MISSING_OBSERVABLES_OR_INVALID_WINDOW,
            "structural_fail": True,
            "structural_reason": "empty_recompute_window_log",
            "explicit_non_claims": list(EXPLICIT_NON_CLAIMS),
        }
    missing_fields = [
        field
        for field in REQUIRED_LOG_FIELDS
        if any(field not in record for record in records)
    ]
    if missing_fields:
        return {
            "schema_version": ANALYZER_SCHEMA_VERSION,
            "primary_classifier": CLASSIFIER_MISSING_OBSERVABLES_OR_INVALID_WINDOW,
            "structural_fail": True,
            "structural_reason": "missing_required_log_fields",
            "missing_fields": missing_fields,
            "explicit_non_claims": list(EXPLICIT_NON_CLAIMS),
        }
    replay = ReplayConstants(**dict(records[0]["replay_constants"]))
    grouped = _group_entries_by_lane(records)
    lane_measurements: list[dict[str, Any]] = []
    for (state_key, lane_index), entries in grouped.items():
        position = int(entries[0]["_lane_position"])
        measurement = measure_lane_k_star(
            lane_index=int(lane_index),
            step_entries=entries,
            lane_position=position,
            replay=replay,
        )
        lane_measurements.append(
            {
                "state_key": state_key,
                **asdict(measurement),
            }
        )
    k_values = [int(item["k_star"]) for item in lane_measurements if item["k_star"] is not None]
    available_history = max(
        (len(entries) for entries in grouped.values()),
        default=0,
    )
    k_star_p50 = _percentile(k_values, 50.0)
    k_star_p95 = _percentile(k_values, 95.0)
    k_star_p99 = _percentile(k_values, 99.0)
    k_star_worst = max(k_values) if k_values else None
    parity_pass_count = sum(1 for item in lane_measurements if item["parity_pass"])
    parity_fail_count = len(lane_measurements) - parity_pass_count
    plateau_signal = (
        k_star_worst is not None
        and available_history > 0
        and float(k_star_worst) < float(available_history)
    )
    worst_case_full_history = (
        k_star_worst is not None
        and available_history > 0
        and float(k_star_worst) >= float(available_history)
    )
    unbounded_at_scale_signal = bool(worst_case_full_history) or (
        k_star_p95 is not None
        and available_history > 0
        and float(k_star_p95) >= float(available_history)
    )
    stream_bytes = int(path.stat().st_size) if path.is_file() else 0
    effective_numel = int(numel_for_bpw) if numel_for_bpw is not None else max(
        1,
        len(grouped),
    )
    acc_term_bpw = inclusive_stream_bpw(numel=effective_numel, stream_bytes=stream_bytes)
    dual_booleans = dual_budget_booleans(
        acc_term_bpw=float(acc_term_bpw),
        byte_model_declared=True,
    )
    inclusive_pass = float(acc_term_bpw) < float(ACC_BUDGET_BPW_UNDER_BASE3_Q)
    bounded_k_star = (
        k_star_worst is not None
        and parity_fail_count == 0
        and plateau_signal
    )
    primary = CLASSIFIER_NO_CARRIER_FAMILY_VIABLE
    promoted_fork: str | None = None
    if bounded_k_star and inclusive_pass and parity_pass_count > 0:
        primary = CLASSIFIER_D_RECOMPUTE_WINDOW_LEAD
    elif unbounded_at_scale_signal or parity_fail_count > 0:
        primary = CLASSIFIER_D_RECOMPUTE_UNBOUNDED_OR_UNOBSERVABLE
        if unbounded_at_scale_signal:
            promoted_fork = CLASSIFIER_D_NEEDS_UPDATE_LAW_REDESIGN
    lane_class_counts: dict[str, int] = {}
    for item in lane_measurements:
        lane_class_counts[item["lane_class"]] = lane_class_counts.get(item["lane_class"], 0) + 1
    return {
        "schema_version": ANALYZER_SCHEMA_VERSION,
        "log_schema_version": D_RECOMPUTE_WINDOW_SCHEMA_VERSION,
        "primary_classifier": primary,
        "promoted_fork": promoted_fork,
        "structural_fail": primary == CLASSIFIER_MISSING_OBSERVABLES_OR_INVALID_WINDOW,
        "replay_constants": replay.to_dict(),
        "lane_measurement_count": len(lane_measurements),
        "parity_pass_count": parity_pass_count,
        "parity_fail_count": parity_fail_count,
        "k_star_distribution": {
            "p50": k_star_p50,
            "p95": k_star_p95,
            "p99": k_star_p99,
            "worst": k_star_worst,
            "available_history_steps": available_history,
            "plateau_signal": plateau_signal,
            "worst_case_full_history": worst_case_full_history,
            "unbounded_at_scale_signal": unbounded_at_scale_signal,
        },
        "lane_class_counts": lane_class_counts,
        "inclusive_bpw": {
            "stream_bytes": stream_bytes,
            "numel_basis": effective_numel,
            "acc_term_bpw": acc_term_bpw,
            "passes_acc_budget_under_base3_q": inclusive_pass,
        },
        "dual_booleans": dual_booleans,
        "b_annex": {
            "authoritative": False,
            "forbidden_terminal_from_d_fields": FORBIDDEN_B_TERMINAL_FROM_D_FIELDS,
        },
        "explicit_non_claims": list(EXPLICIT_NON_CLAIMS),
        "instrumentation_fork": {
            "law_redesign": (
                "add decay / forgettable vote history so old votes do not persist "
                "in the running sum under decay=1/1"
            )
        },
    }


def analyze_synthetic_lane_trajectory(
    *,
    votes: Sequence[int],
    acc_trajectory: Sequence[int],
    replay: ReplayConstants | None = None,
    flip_residual_flags: Sequence[bool] | None = None,
) -> dict[str, Any]:
    """Build minimal in-memory log records for unit tests and K* fixtures."""

    replay_constants = replay or default_production_replay_constants()
    records: list[dict[str, Any]] = []
    acc_before = 0
    for step, vote in enumerate(votes, start=1):
        acc_after = int(acc_trajectory[step - 1])
        flip = (
            bool(flip_residual_flags[step - 1])
            if flip_residual_flags is not None and step - 1 < len(flip_residual_flags)
            else False
        )
        record = {
            "schema_version": D_RECOMPUTE_WINDOW_SCHEMA_VERSION,
            "step": int(step),
            "state_key": "synthetic.lane",
            "resume_generation": 0,
            "replay_constants": replay_constants.to_dict(),
            "lane_indices": [0],
            "vote_lanes": [int(vote)],
            "acc_before_lanes": [int(acc_before)],
            "acc_after_lanes": [int(acc_after)],
            "q_before_lanes": [0],
            "q_after_lanes": [0],
            "flip_residual_applied": flip,
            "flip_direction": 1 if flip else None,
            "flip_threshold": int(replay_constants.threshold_abs) if flip else None,
            "flip_residual_applied_lanes": [flip],
            "flip_direction_lanes": [1 if flip else None],
            "flip_threshold_lanes": [
                int(replay_constants.threshold_abs) if flip else None
            ],
            "residual_authority_lanes": ["present" if flip else "not_applicable"],
            "cap_order_digest": "synthetic",
            "applied_order_digest": "synthetic",
            "vote_source_digest": "synthetic",
        }
        records.append(record)
        acc_before = acc_after
    grouped_measurement = measure_lane_k_star(
        lane_index=0,
        step_entries=records,
        lane_position=0,
        replay=replay_constants,
    )
    return {
        "records": records,
        "measurement": asdict(grouped_measurement),
    }
