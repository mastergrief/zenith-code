"""CPU-static tests for Arc #2b Slice-5 discovery Arm B offline harness."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from scripts.hrm_text_158_arc2b_slice5_discovery_arm_b_offline import (
    B1_LOG_PATH,
    K_STAR_SATURATION_CONSECUTIVE_STEPS,
    K_STAR_SATURATION_REL_DELTA,
    build_arm_b_receipt,
    _check_k_star_saturation,
    _k_star_trend,
    _load_b1_records,
    _validate_lane_fields_for_all_records,
)
from calm.hrm_text_158.native_full_stack.arc2b_slice5_discovery_branch import (
    ARM_B_SOURCE_RUN_ID,
    EVIDENCE_ARM_B_OFFLINE,
    RECEIPT_SCHEMA,
    REQUIRED_LANE_FIELDS,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _make_good_record(step: int = 1) -> dict:
    return {
        "step": step,
        "state_key": f"state_{step}",
        "lane_indices": [0, 65536, 131072],
        "acc_before_lanes": [0, 0, 0],
        "acc_after_lanes": [1, 1, 1],
        "vote_lanes": [1, 1, 1],
        "replay_constants": {
            "decay_numerator": 1,
            "decay_denominator": 1,
            "accumulator_clip_max": 127,
            "accumulator_clip_min": -127,
        },
        "backlog_depth": 100,
        "resume_generation": 0,
    }


def test_arm_b_receipt_schema() -> None:
    receipt = build_arm_b_receipt()
    assert receipt["schema"] == RECEIPT_SCHEMA
    assert receipt["evidence_source"] == EVIDENCE_ARM_B_OFFLINE
    assert receipt["arm_b_source_run_id"] == ARM_B_SOURCE_RUN_ID
    assert receipt["ready_for_main_science"] is False
    assert receipt["counts_as_sub2"] is False
    assert receipt["pre_full_stack_diagnostic"] is True
    assert receipt["autonomy_rung"] == "arm_b_offline_cpu"


def test_arm_b_receipt_has_k_star_summary() -> None:
    receipt = build_arm_b_receipt()
    summary = receipt["arm_b_k_star_summary"]
    assert "trend" in summary
    assert "saturation" in summary
    assert isinstance(summary["trend"], list)


def test_arm_b_receipt_decay_caveat() -> None:
    receipt = build_arm_b_receipt()
    caveat = receipt.get("arm_b_decay_caveat", "")
    assert "decay 1/1" in caveat
    assert "1/2" in caveat
    assert "separate-axis" in caveat


def test_arm_b_fail_closed_on_missing_lane_fields() -> None:
    """FAIL-CLOSED: missing lane fields => INCONCLUSIVE_LANE_FIELD_MISSING."""
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "recompute_window_log.jsonl"
        bad_record = _make_good_record()
        del bad_record["acc_after_lanes"]  # remove required field
        _write_jsonl(log_path, [bad_record])

        receipt = build_arm_b_receipt(log_path=log_path)
        finding = receipt["arm_b_finding"]
        assert "DISCOVERY_INCONCLUSIVE_LANE_FIELD_MISSING" in finding
        assert receipt["arm_b_lane_field_failure_count"] > 0
        assert len(receipt["arm_b_lane_field_failures"]) > 0


def test_arm_b_fail_closed_on_empty_log() -> None:
    """No records => INCONCLUSIVE_LOG_COVERAGE."""
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "nonexistent.jsonl"
        receipt = build_arm_b_receipt(log_path=log_path)
        assert receipt["arm_b_record_count"] == 0
        assert "DISCOVERY_INCONCLUSIVE_LOG_COVERAGE" in receipt["arm_b_finding"]


def test_validate_lane_fields_for_all_records() -> None:
    good_records = [_make_good_record(step=i) for i in range(1, 4)]
    assert _validate_lane_fields_for_all_records(good_records) == []

    bad_records = [_make_good_record(step=1)]
    del bad_records[0]["vote_lanes"]
    failures = _validate_lane_fields_for_all_records(bad_records)
    assert len(failures) == 1
    assert "missing_lane_field:vote_lanes" in failures[0]


def test_k_star_saturation_check_insufficient_horizons() -> None:
    """Fewer than 4 horizons => not saturated (insufficient data)."""
    trend = [{"kworst_weighted": 10}, {"kworst_weighted": 11}]
    result = _check_k_star_saturation(trend)
    assert result["saturated"] is False
    assert "insufficient_horizons" in result["reason"]


def test_k_star_saturation_check_saturated() -> None:
    """K* stable for 3+ consecutive steps => saturated."""
    trend = [
        {"kworst_weighted": 100},
        {"kworst_weighted": 101},  # delta=1, rel=0.01 < 0.05
        {"kworst_weighted": 101},  # delta=0, rel=0 < 0.05
        {"kworst_weighted": 101},  # delta=0, rel=0 < 0.05
    ]
    result = _check_k_star_saturation(trend)
    assert result["saturated"] is True
    assert result["max_consecutive_saturated_steps"] >= K_STAR_SATURATION_CONSECUTIVE_STEPS


def test_k_star_saturation_check_still_climbing() -> None:
    """K* growing >5% per step => not saturated."""
    trend = [
        {"kworst_weighted": 10},
        {"kworst_weighted": 20},  # delta=10, rel=1.0 > 0.05
        {"kworst_weighted": 40},  # delta=20, rel=1.0 > 0.05
        {"kworst_weighted": 80},  # delta=40, rel=1.0 > 0.05
    ]
    result = _check_k_star_saturation(trend)
    assert result["saturated"] is False
    assert "still_climbing" in result["reason"]


def test_arm_b_loads_b1_records_if_available() -> None:
    """If B1 log exists, load records (3600 expected)."""
    if not B1_LOG_PATH.is_file():
        return  # skip if B1 not available
    records = _load_b1_records()
    assert len(records) > 0
    # Verify required lane fields present
    failures = _validate_lane_fields_for_all_records(records)
    assert failures == [], f"B1 records have lane field failures: {failures[:3]}"


def test_k_star_trend_returns_list() -> None:
    """K* trend returns a list of per-horizon summaries."""
    if not B1_LOG_PATH.is_file():
        return
    records = _load_b1_records()
    trend = _k_star_trend(records, horizons=(25, 50))
    assert isinstance(trend, list)
    assert len(trend) <= 2
    if trend:
        assert "kworst_weighted" in trend[0]
        assert "horizon_h" in trend[0]
