"""Offline CPU horizon-prefix K*(H) growth analyzer for D recompute-window logs."""
from __future__ import annotations

from dataclasses import asdict
import math
import random
from typing import Any, Mapping, Sequence

from calm.hrm_text_158.native_full_stack.d_recompute_window_emit import (
    BOOTSTRAP_KNOWN_SATURATED_NEGATIVE,
    BOOTSTRAP_KNOWN_SATURATED_POSITIVE,
    BOOTSTRAP_KNOWN_ZERO,
    D_RECOMPUTE_WINDOW_SCHEMA_VERSION,
    ReplayConstants,
    default_production_replay_constants,
)
from calm.hrm_text_158.native_full_stack.d_recompute_window_feasibility_analyzer import (
    LaneKStarMeasurement,
    _lane_flip_fields,
    _percentile,
    _valid_bootstrap_for_acc,
    bootstrap_starting_acc,
    carry_after_scalar,
    classify_lane_step,
    reconstruct_lane_from_bootstrap,
)
from calm.hrm_text_158.native_full_stack.d_recompute_window_stratified_selector import (
    COVERAGE_TIER_PILOT,
    STRESS_TAIL_POLICY_HORIZON_FIXED,
)

HORIZON_ANALYZER_SCHEMA_VERSION = "hrm_text_158_d_recompute_window_horizon/v0"

GROWTH_INCONCLUSIVE_COST_OR_COVERAGE = "INCONCLUSIVE_COST_OR_COVERAGE"
GROWTH_RIGHT_CENSORED_LOWER_BOUND = "RIGHT_CENSORED_LOWER_BOUND"
GROWTH_PLATEAU_SIZED = "PLATEAU_SIZED"
GROWTH_LINEAR_SIZED_WITH_DECAY = "LINEAR_SIZED_WITH_DECAY"
GROWTH_ACCELERATING_OR_RIGHT_CENSORED = "ACCELERATING_OR_RIGHT_CENSORED"

GROWTH_BRANCH_PRECEDENCE: tuple[str, ...] = (
    GROWTH_INCONCLUSIVE_COST_OR_COVERAGE,
    GROWTH_RIGHT_CENSORED_LOWER_BOUND,
    GROWTH_PLATEAU_SIZED,
    GROWTH_LINEAR_SIZED_WITH_DECAY,
    GROWTH_ACCELERATING_OR_RIGHT_CENSORED,
)

DEFAULT_HORIZON_LADDER: tuple[int, ...] = (25, 50, 100)
RIGHT_CENSOR_RATE_THRESHOLD = 0.15
GAPPED_LANE_FRACTION_MAX = 0.0
BOOTSTRAP_SAMPLE_COUNT = 1000


def slope_threshold(delta_h: int) -> float:
    return max(2.0, 0.10 * float(int(delta_h)))


def resolve_lane_position(record: Mapping[str, Any], lane_index: int) -> int:
    lane_indices = record.get("lane_indices") or []
    for position, value in enumerate(lane_indices):
        if int(value) == int(lane_index):
            return int(position)
    raise KeyError(
        f"lane_index {lane_index} missing from record lane_indices for "
        f"{record.get('state_key')!r} step={record.get('step')!r}"
    )


def audit_lane_coverage(
    entries: Sequence[Mapping[str, Any]],
    *,
    horizon_h: int,
    measurement_start_step: int = 1,
) -> dict[str, Any]:
    steps = sorted(
        int(entry["step"])
        for entry in entries
        if int(entry["step"]) <= int(horizon_h)
    )
    if not steps:
        return {
            "observation_count": 0,
            "gap_count": 0,
            "contiguous": True,
            "right_censored": False,
        }
    expected_steps = list(range(int(measurement_start_step), int(horizon_h) + 1))
    observed = {step for step in steps}
    gap_count = sum(1 for step in expected_steps if step not in observed)
    return {
        "observation_count": len([step for step in expected_steps if step in observed]),
        "gap_count": int(gap_count),
        "contiguous": gap_count == 0,
        "right_censored": False,
    }


def measure_lane_k_star_per_record_positions(
    *,
    lane_index: int,
    step_entries: Sequence[Mapping[str, Any]],
    replay: ReplayConstants,
) -> LaneKStarMeasurement:
    if not step_entries:
        return LaneKStarMeasurement(
            lane_index=int(lane_index),
            target_step=0,
            k_star=None,
            lane_class="slow_sub_saturation",
            parity_pass=False,
            bootstrap_used=None,
        )

    def lane_position(entry: Mapping[str, Any]) -> int:
        return resolve_lane_position(entry, lane_index)

    target_step = int(step_entries[-1]["step"])
    target_acc = int(step_entries[-1]["acc_after_lanes"][lane_position(step_entries[-1])])
    last_flip_applied, _, _, last_authority = _lane_flip_fields(
        step_entries[-1],
        lane_position(step_entries[-1]),
    )
    lane_class = classify_lane_step(
        acc_before=int(
            step_entries[-1]["acc_before_lanes"][lane_position(step_entries[-1])]
        ),
        acc_after=target_acc,
        vote=int(step_entries[-1]["vote_lanes"][lane_position(step_entries[-1])]),
        replay=replay,
        flip_residual_applied=bool(last_flip_applied),
    )
    if last_authority == "absent" and carry_after_scalar(
        int(step_entries[-1]["acc_before_lanes"][lane_position(step_entries[-1])]),
        int(step_entries[-1]["vote_lanes"][lane_position(step_entries[-1])]),
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
            start_acc = int(window[0]["acc_before_lanes"][lane_position(window[0])])
            if bootstrap not in _valid_bootstrap_for_acc(start_acc, replay=replay):
                if not (bootstrap == BOOTSTRAP_KNOWN_ZERO and start_acc == 0):
                    continue
            votes = [
                int(entry["vote_lanes"][lane_position(entry)]) for entry in window
            ]
            flip_flags: list[bool] = []
            flip_dirs: list[int | None] = []
            flip_thrs: list[int | None] = []
            for entry in window:
                position = lane_position(entry)
                applied, direction, threshold, authority = _lane_flip_fields(
                    entry,
                    position,
                )
                if authority == "absent" and applied:
                    flip_flags = []
                    break
                flip_flags.append(bool(applied))
                flip_dirs.append(direction)
                flip_thrs.append(threshold)
            if not flip_flags and any(
                carry_after_scalar(
                    int(entry["acc_before_lanes"][lane_position(entry)]),
                    int(entry["vote_lanes"][lane_position(entry)]),
                    replay=replay,
                )
                != int(entry["acc_after_lanes"][lane_position(entry)])
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


def _group_lane_histories(
    records: Sequence[Mapping[str, Any]],
    *,
    horizon_h: int,
) -> dict[tuple[str, int], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for record in records:
        if int(record["step"]) > int(horizon_h):
            continue
        state_key = str(record["state_key"])
        for lane_index in record.get("lane_indices") or []:
            key = (state_key, int(lane_index))
            grouped.setdefault(key, []).append(dict(record))
    for entries in grouped.values():
        entries.sort(key=lambda item: int(item["step"]))
    return grouped


def _weighted_percentile(
    values: Sequence[tuple[int, float]],
    pct: float,
) -> float | None:
    if not values:
        return None
    total_weight = sum(weight for _, weight in values)
    if total_weight <= 0:
        return None
    ordered = sorted(values, key=lambda item: item[0])
    target = total_weight * (float(pct) / 100.0)
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += float(weight)
        if cumulative >= target:
            return float(value)
    return float(ordered[-1][0])


def summarize_k_star_at_horizon_prefix(
    records: Sequence[Mapping[str, Any]],
    horizon_h: int,
    *,
    stratum_weights: Mapping[str, float] | None = None,
    replay: ReplayConstants | None = None,
    measurement_start_step: int = 1,
) -> dict[str, Any]:
    replay_constants = replay or default_production_replay_constants()
    weights = {str(key): float(value) for key, value in dict(stratum_weights or {}).items()}
    grouped = _group_lane_histories(records, horizon_h=int(horizon_h))
    lane_rows: list[dict[str, Any]] = []
    gapped_lane_count = 0
    parity_fail_count = 0

    for (state_key, lane_index), entries in sorted(grouped.items()):
        coverage = audit_lane_coverage(
            entries,
            horizon_h=int(horizon_h),
            measurement_start_step=int(measurement_start_step),
        )
        gapped = not bool(coverage["contiguous"])
        if gapped:
            gapped_lane_count += 1
        measurement = measure_lane_k_star_per_record_positions(
            lane_index=int(lane_index),
            step_entries=entries,
            replay=replay_constants,
        )
        if not measurement.parity_pass:
            parity_fail_count += 1
        right_censored = (
            measurement.k_star is not None and int(measurement.k_star) == int(horizon_h)
        )
        lane_rows.append(
            {
                "state_key": state_key,
                "lane_index": int(lane_index),
                "k_star": measurement.k_star,
                "parity_pass": bool(measurement.parity_pass),
                "gapped": bool(gapped),
                "right_censored": bool(right_censored),
                "observation_count": int(coverage["observation_count"]),
                "gap_count": int(coverage["gap_count"]),
                "lane_position_by_step": {
                    str(entry["step"]): resolve_lane_position(entry, lane_index)
                    for entry in entries
                },
            }
        )

    eligible = [
        row
        for row in lane_rows
        if row["k_star"] is not None and not row["gapped"] and row["parity_pass"]
    ]
    unweighted_values = [int(row["k_star"]) for row in eligible]
    weighted_values = [
        (int(row["k_star"]), weights.get(str(row["state_key"]), 1.0)) for row in eligible
    ]
    censored_values = [row for row in eligible if row["right_censored"]]
    lane_count = len(lane_rows)
    right_censor_rate = (
        float(len(censored_values)) / float(len(eligible)) if eligible else 0.0
    )

    return {
        "horizon_h": int(horizon_h),
        "lane_count": int(lane_count),
        "eligible_lane_count": len(eligible),
        "gapped_lane_count": int(gapped_lane_count),
        "parity_fail_count": int(parity_fail_count),
        "right_censor_rate": float(right_censor_rate),
        "k95_unweighted": _percentile(unweighted_values, 95.0),
        "k99_unweighted": _percentile(unweighted_values, 99.0),
        "kworst_unweighted": max(unweighted_values) if unweighted_values else None,
        "k95_weighted": _weighted_percentile(weighted_values, 95.0),
        "k99_weighted": _weighted_percentile(weighted_values, 99.0),
        "kworst_weighted": (
            max(value for value, _ in weighted_values) if weighted_values else None
        ),
        "lane_rows": lane_rows,
    }


def _is_right_censored_at_horizon(summary: Mapping[str, Any]) -> bool:
    horizon_h = int(summary["horizon_h"])
    if summary.get("kworst_weighted") is not None and float(summary["kworst_weighted"]) >= horizon_h:
        return True
    if summary.get("k99_weighted") is not None and float(summary["k99_weighted"]) >= horizon_h:
        return True
    return float(summary.get("right_censor_rate", 0.0)) >= RIGHT_CENSOR_RATE_THRESHOLD


def _material_growth(
  low: float | None,
  high: float | None,
  *,
  threshold: float,
) -> bool:
    if low is None or high is None:
        return False
    return float(high) - float(low) > float(threshold)


def _bootstrap_slope_inconclusive(
    summaries_by_h: Mapping[int, Mapping[str, Any]],
    *,
    low_h: int,
    high_h: int,
    sample_count: int = BOOTSTRAP_SAMPLE_COUNT,
    seed: int = 17,
) -> bool:
    low_rows = [
        row
        for row in summaries_by_h[low_h]["lane_rows"]
        if row["k_star"] is not None and not row["gapped"] and row["parity_pass"]
    ]
    high_rows = [
        row
        for row in summaries_by_h[high_h]["lane_rows"]
        if row["k_star"] is not None and not row["gapped"] and row["parity_pass"]
    ]
    if not low_rows or not high_rows:
        return True
    rng = random.Random(int(seed))
    threshold = slope_threshold(int(high_h) - int(low_h))
    below = 0
    above = 0
    for _ in range(int(sample_count)):
        low_pick = rng.choice(low_rows)
        high_pick = rng.choice(high_rows)
        slope = float(high_pick["k_star"]) - float(low_pick["k_star"])
        if slope <= threshold:
            below += 1
        else:
            above += 1
    minority = min(below, above)
    return minority / float(sample_count) > 0.05


def classify_k_star_growth(
    summaries_by_h: Mapping[int, Mapping[str, Any]],
    *,
    stress_tail_policy: str | None = None,
    coverage_tier: str | None = None,
    gapped_lane_fraction_max: float = GAPPED_LANE_FRACTION_MAX,
) -> dict[str, Any]:
    horizons = sorted(int(h) for h in summaries_by_h.keys())
    h100 = summaries_by_h.get(100)
    if h100 is None:
        return {
            "growth_branch": GROWTH_INCONCLUSIVE_COST_OR_COVERAGE,
            "reason": "missing_h100_summary",
        }

    lane_count = int(h100.get("lane_count", 0))
    gapped_fraction = (
        float(h100.get("gapped_lane_count", 0)) / float(lane_count)
        if lane_count > 0
        else 1.0
    )
    if (
        stress_tail_policy is not None
        and str(stress_tail_policy) != STRESS_TAIL_POLICY_HORIZON_FIXED
    ):
        return {
            "growth_branch": GROWTH_INCONCLUSIVE_COST_OR_COVERAGE,
            "reason": "non_horizon_fixed_policy",
            "stress_tail_policy": str(stress_tail_policy),
        }
    if coverage_tier == COVERAGE_TIER_PILOT:
        return {
            "growth_branch": GROWTH_INCONCLUSIVE_COST_OR_COVERAGE,
            "reason": "pilot_coverage_tier",
        }
    if int(h100.get("parity_fail_count", 0)) > 0:
        return {
            "growth_branch": GROWTH_INCONCLUSIVE_COST_OR_COVERAGE,
            "reason": "parity_failures",
        }
    if gapped_fraction > float(gapped_lane_fraction_max):
        return {
            "growth_branch": GROWTH_INCONCLUSIVE_COST_OR_COVERAGE,
            "reason": "gapped_lane_fraction",
            "gapped_lane_fraction": float(gapped_fraction),
        }

    if _is_right_censored_at_horizon(h100):
        return {
            "growth_branch": GROWTH_RIGHT_CENSORED_LOWER_BOUND,
            "reason": "right_censored_at_h100",
        }

    if 50 in summaries_by_h and 100 in summaries_by_h:
        plateau_threshold = slope_threshold(50)
        if _bootstrap_slope_inconclusive(summaries_by_h, low_h=50, high_h=100):
            return {
                "growth_branch": GROWTH_INCONCLUSIVE_COST_OR_COVERAGE,
                "reason": "bootstrap_slope_threshold_crossing",
            }
        p99_50 = summaries_by_h[50].get("k99_weighted")
        p99_100 = summaries_by_h[100].get("k99_weighted")
        p95_100 = summaries_by_h[100].get("k95_weighted")
        worst_100 = summaries_by_h[100].get("kworst_weighted")
        if (
            p99_100 is not None
            and worst_100 is not None
            and float(p99_100) < 100.0
            and float(worst_100) < 100.0
            and p99_50 is not None
            and float(p99_100) - float(p99_50) <= plateau_threshold
        ):
            return {
                "growth_branch": GROWTH_PLATEAU_SIZED,
                "reason": "weighted_p99_plateau",
                "weighted_p99_slope_50_to_100": float(p99_100) - float(p99_50),
            }

    uncensored_horizons = [
        h
        for h in horizons
        if summaries_by_h[h].get("k99_weighted") is not None
        and float(summaries_by_h[h]["k99_weighted"]) < float(h)
    ]
    if len(uncensored_horizons) >= 3:
        points = [
            (float(h), float(summaries_by_h[h]["k99_weighted"]))
            for h in uncensored_horizons
        ]
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)
        num = sum((x - mean_x) * (y - mean_y) for x, y in points)
        den = sum((x - mean_x) ** 2 for x in xs)
        if den > 0:
            slope = num / den
            second_diff = ys[-1] - 2 * ys[-2] + ys[-3]
            if slope > 0 and second_diff > 0:
                return {
                    "growth_branch": GROWTH_ACCELERATING_OR_RIGHT_CENSORED,
                    "reason": "convex_weighted_p99_growth",
                }

    if 25 in summaries_by_h and 50 in summaries_by_h and 100 in summaries_by_h:
        p99_25 = summaries_by_h[25].get("k99_weighted")
        p99_50 = summaries_by_h[50].get("k99_weighted")
        p99_100 = summaries_by_h[100].get("k99_weighted")
        if (
            _material_growth(p99_25, p99_50, threshold=slope_threshold(25))
            and _material_growth(p99_50, p99_100, threshold=slope_threshold(50))
            and not _is_right_censored_at_horizon(h100)
        ):
            return {
                "growth_branch": GROWTH_LINEAR_SIZED_WITH_DECAY,
                "reason": "material_weighted_p99_growth",
            }

    return {
        "growth_branch": GROWTH_INCONCLUSIVE_COST_OR_COVERAGE,
        "reason": "no_branch_matched",
    }


def analyze_horizon_k_star_growth(
    records: Sequence[Mapping[str, Any]],
    *,
    stratum_weights: Mapping[str, float] | None = None,
    horizons: Sequence[int] = DEFAULT_HORIZON_LADDER,
    replay: ReplayConstants | None = None,
    measurement_start_step: int = 1,
    stress_tail_policy: str | None = None,
    coverage_tier: str | None = None,
) -> dict[str, Any]:
    summaries = {
        int(horizon_h): summarize_k_star_at_horizon_prefix(
            records,
            int(horizon_h),
            stratum_weights=stratum_weights,
            replay=replay,
            measurement_start_step=int(measurement_start_step),
        )
        for horizon_h in horizons
    }
    branch = classify_k_star_growth(
        summaries,
        stress_tail_policy=stress_tail_policy,
        coverage_tier=coverage_tier,
    )
    return {
        "schema_version": HORIZON_ANALYZER_SCHEMA_VERSION,
        "log_schema_version": D_RECOMPUTE_WINDOW_SCHEMA_VERSION,
        "horizons": [int(h) for h in horizons],
        "summaries_by_h": summaries,
        "growth_branch": branch["growth_branch"],
        "growth_branch_detail": branch,
        "stress_tail_policy": stress_tail_policy,
        "coverage_tier": coverage_tier,
        "slope_threshold_50_to_100": slope_threshold(50),
    }
