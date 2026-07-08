#!/usr/bin/env python3
"""Arc #2b Slice-5 discovery Arm B offline CPU harness (K* saturation over B1).

Frozen v6 plan (co_lead gate-2 PASS 1783512484577, +1 implement 1783526612437).
Arm B = OFFLINE CPU: K* saturation over B1 2189e72017 decay-1/1 lane data.
SEPARATE-AXIS likely-negative control. Deliverable = K* trend (saturates or not).
Envelope bpw (~0.0006) is NOT a point on the live decay-gap curve.
0 GPU. Fail-closed on missing lane fields.

Caveats: B1 decay 1/1 != 1/2 (law under test); censored@200 (right_censor_rate
may be nonzero). K* saturation criterion: |gap(n)-gap(n-1)|/gap(n-1) < 0.05 for
3 consecutive steps (per frozen v6 §4 materiality).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from calm.hrm_text_158.native_full_stack.arc2b_slice5_discovery_branch import (
    ARM_B_SOURCE_RUN_ID,
    ARM_B_SOURCE_RUN_ROOT,
    CLASSIFIER,
    EVIDENCE_ARM_B_OFFLINE,
    RECEIPT_SCHEMA,
    REQUIRED_LANE_FIELDS,
    validate_lane_fields,
)
from calm.hrm_text_158.native_full_stack.d_recompute_window_emit import (
    ReplayConstants,
    default_production_replay_constants,
)
from calm.hrm_text_158.native_full_stack.d_recompute_window_horizon_analyzer import (
    analyze_horizon_k_star_growth,
    summarize_k_star_at_horizon_prefix,
)

ACTIVE_TASK_ID = "1783272482268-052281aa"
B1_LOG_PATH = (
    Path(ARM_B_SOURCE_RUN_ROOT) / "d_recompute_window_diagnostic" / "recompute_window_log.jsonl"
)
DEFAULT_HORIZONS: tuple[int, ...] = (25, 50, 100, 200)
K_STAR_SATURATION_REL_DELTA = 0.05  # |gap(n)-gap(n-1)|/gap(n-1) < 5%
K_STAR_SATURATION_CONSECUTIVE_STEPS = 3


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _load_b1_records(log_path: Path = B1_LOG_PATH) -> list[dict[str, Any]]:
    """Load B1 2189e72017 recompute_window_log.jsonl records."""
    rows = _load_jsonl_rows(log_path)
    if not rows:
        return []
    return rows


def _validate_lane_fields_for_all_records(
    records: Sequence[Mapping[str, Any]],
) -> list[str]:
    """FAIL-CLOSED: every record must have all required lane fields."""
    failures: list[str] = []
    for idx, record in enumerate(records):
        record_failures = validate_lane_fields(record)
        for failure in record_failures:
            failures.append(f"record[{idx}]:{failure}")
    return failures


def _extract_replay_constants(
    records: Sequence[Mapping[str, Any]],
) -> ReplayConstants | None:
    """Extract replay_constants from first record (B1 decay 1/1)."""
    if not records:
        return None
    rc = dict(records[0].get("replay_constants") or {})
    if not rc:
        return None
    try:
        return default_production_replay_constants(
            decay_numerator=int(rc.get("decay_numerator", 1)),
            decay_denominator=int(rc.get("decay_denominator", 1)),
        )
    except Exception:
        return None


def _k_star_trend(
    records: Sequence[Mapping[str, Any]],
    *,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    replay: ReplayConstants | None = None,
) -> list[dict[str, Any]]:
    """Compute K* trend across horizons (kworst_weighted per horizon)."""
    replay_constants = replay or _extract_replay_constants(records) or default_production_replay_constants()
    trend: list[dict[str, Any]] = []
    for horizon_h in horizons:
        if horizon_h > len(records):
            break
        summary = summarize_k_star_at_horizon_prefix(
            records,
            int(horizon_h),
            replay=replay_constants,
        )
        trend.append(
            {
                "horizon_h": int(horizon_h),
                "kworst_weighted": summary.get("kworst_weighted"),
                "k99_weighted": summary.get("k99_weighted"),
                "k95_weighted": summary.get("k95_weighted"),
                "right_censor_rate": summary.get("right_censor_rate"),
                "eligible_lane_count": summary.get("eligible_lane_count"),
                "gapped_lane_count": summary.get("gapped_lane_count"),
            }
        )
    return trend


def _check_k_star_saturation(
    trend: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """K* saturation: |gap(n)-gap(n-1)|/gap(n-1) < 0.05 for 3 consecutive steps.

    'gap' here = kworst_weighted (the K* value at each horizon).
    Saturated AND gap>0 => K* NOT mechanism (window sufficient, density is problem).
    Saturated AND gap<=0 => recompute-window IS mechanism.
    """
    if len(trend) < K_STAR_SATURATION_CONSECUTIVE_STEPS + 1:
        return {
            "saturated": False,
            "reason": "insufficient_horizons_for_saturation_check",
            "consecutive_steps_required": K_STAR_SATURATION_CONSECUTIVE_STEPS,
        }

    consecutive = 0
    max_consecutive = 0
    for i in range(1, len(trend)):
        prev = trend[i - 1].get("kworst_weighted")
        curr = trend[i].get("kworst_weighted")
        if prev is None or curr is None or float(prev) <= 0:
            consecutive = 0
            continue
        rel_delta = abs(float(curr) - float(prev)) / float(prev)
        if rel_delta < K_STAR_SATURATION_REL_DELTA:
            consecutive += 1
            max_consecutive = max(max_consecutive, consecutive)
        else:
            consecutive = 0

    saturated = max_consecutive >= K_STAR_SATURATION_CONSECUTIVE_STEPS
    return {
        "saturated": bool(saturated),
        "max_consecutive_saturated_steps": int(max_consecutive),
        "consecutive_steps_required": K_STAR_SATURATION_CONSECUTIVE_STEPS,
        "rel_delta_threshold": K_STAR_SATURATION_REL_DELTA,
        "reason": "k_star_saturation_criterion_met" if saturated else "k_star_still_climbing",
    }


def build_arm_b_receipt(
    *,
    log_path: Path = B1_LOG_PATH,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
) -> dict[str, Any]:
    """Build the Arm B offline CPU receipt (K* saturation over B1, 0 GPU)."""
    records = _load_b1_records(log_path)

    if not records:
        return {
            "schema": RECEIPT_SCHEMA,
            "task_id": ACTIVE_TASK_ID,
            "classifier": CLASSIFIER,
            "evidence_source": EVIDENCE_ARM_B_OFFLINE,
            "arm_b_source_run_id": ARM_B_SOURCE_RUN_ID,
            "arm_b_source_log_path": str(log_path),
            "arm_b_record_count": 0,
            "arm_b_k_star_summary": {
                "trend": [],
                "saturation": {"saturated": False, "reason": "no_records_loaded"},
            },
            "arm_b_finding": "DISCOVERY_INCONCLUSIVE_LOG_COVERAGE: no B1 records loaded",
            "arm_b_lane_field_failures": [],
            "ready_for_main_science": False,
            "counts_as_sub2": False,
            "pre_full_stack_diagnostic": True,
            "autonomy_rung": "arm_b_offline_cpu",
            "generated_at_unix": int(time.time()),
        }

    # FAIL-CLOSED: validate lane fields on all records
    lane_failures = _validate_lane_fields_for_all_records(records)

    if lane_failures:
        return {
            "schema": RECEIPT_SCHEMA,
            "task_id": ACTIVE_TASK_ID,
            "classifier": CLASSIFIER,
            "evidence_source": EVIDENCE_ARM_B_OFFLINE,
            "arm_b_source_run_id": ARM_B_SOURCE_RUN_ID,
            "arm_b_source_log_path": str(log_path),
            "arm_b_record_count": len(records),
            "arm_b_k_star_summary": {
                "trend": [],
                "saturation": {"saturated": False, "reason": "lane_field_failures"},
            },
            "arm_b_finding": (
                "DISCOVERY_INCONCLUSIVE_LANE_FIELD_MISSING: "
                f"{len(lane_failures)} lane field failures across {len(records)} records"
            ),
            "arm_b_lane_field_failures": lane_failures[:20],  # cap for receipt size
            "arm_b_lane_field_failure_count": len(lane_failures),
            "ready_for_main_science": False,
            "counts_as_sub2": False,
            "pre_full_stack_diagnostic": True,
            "autonomy_rung": "arm_b_offline_cpu",
            "generated_at_unix": int(time.time()),
        }

    replay = _extract_replay_constants(records)
    trend = _k_star_trend(records, horizons=horizons, replay=replay)
    saturation = _check_k_star_saturation(trend)

    # B1 decay 1/1 caveat
    decay_num = int(records[0].get("replay_constants", {}).get("decay_numerator", -1))
    decay_den = int(records[0].get("replay_constants", {}).get("decay_denominator", -1))
    decay_caveat = (
        f"B1 decay {decay_num}/{decay_den} != law 1/2 under test; "
        "K* trend is separate-axis likely-negative control"
    )

    finding = (
        "k_star_saturated_window_sufficient_density_is_problem"
        if saturation.get("saturated")
        else "k_star_still_climbing_recompute_window_not_mechanism"
    )

    return {
        "schema": RECEIPT_SCHEMA,
        "task_id": ACTIVE_TASK_ID,
        "classifier": CLASSIFIER,
        "evidence_source": EVIDENCE_ARM_B_OFFLINE,
        "arm_b_source_run_id": ARM_B_SOURCE_RUN_ID,
        "arm_b_source_log_path": str(log_path),
        "arm_b_record_count": len(records),
        "arm_b_decay_caveat": decay_caveat,
        "arm_b_k_star_summary": {
            "trend": trend,
            "saturation": saturation,
            "horizons_checked": [int(h) for h in horizons if int(h) <= len(records)],
        },
        "arm_b_finding": finding,
        "arm_b_lane_field_failures": [],
        "ready_for_main_science": False,
        "counts_as_sub2": False,
        "pre_full_stack_diagnostic": True,
        "autonomy_rung": "arm_b_offline_cpu",
        "generated_at_unix": int(time.time()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--log-path",
        type=Path,
        default=B1_LOG_PATH,
        help=f"B1 recompute_window_log.jsonl path (default: {B1_LOG_PATH})",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output receipt path (default: stdout)",
    )
    ap.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=list(DEFAULT_HORIZONS),
        help="Horizons to check (default: 25 50 100 200)",
    )
    args = ap.parse_args()

    receipt = build_arm_b_receipt(
        log_path=args.log_path,
        horizons=tuple(args.horizons),
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
