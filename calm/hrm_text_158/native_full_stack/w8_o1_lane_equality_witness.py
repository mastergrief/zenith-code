"""W8-scoped O1 lane-equality witness (domain ±127, warmup-only skip)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from calm.hrm_text_158.native_full_stack.narrow_accumulator_codec import (
    W8_SIGNED_MAX,
    W8_SIGNED_MIN,
)
from calm.hrm_text_158.native_full_stack.s3bb_headroom_telemetry import (
    WARMUP_STEPS,
    _finalize_wiring_guard_stats,
    _index_sidecar_file,
    _shared_measured_step_ids,
    diagnose_sidecar_coverage,
)

O1_WITNESS_DOMAIN = "w8_signed_max_127"
O1_SKIP_POLICY = "warmup_only_not_w6_strict_raise"
STRUCTURAL_REASON_W8_LANE_OUT_OF_DOMAIN = "w8_lane_out_of_domain"


def _lane_out_of_w8_domain(value: int) -> bool:
    lane = int(value)
    return lane < int(W8_SIGNED_MIN) or lane > int(W8_SIGNED_MAX)


def compare_w8_o1_lane_equality_streaming(
    oracle_receipt: Mapping[str, Any],
    treatment_receipt: Mapping[str, Any],
    *,
    oracle_sidecar_path: Path | str,
    treatment_sidecar_path: Path | str,
    sidecar_coverage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Keyed streaming O1 witness: W8 domain ±127, skip warmup only."""

    oracle_path = Path(oracle_sidecar_path)
    treatment_path = Path(treatment_sidecar_path)
    if not oracle_path.is_file():
        raise FileNotFoundError(f"missing oracle wiring sidecar: {oracle_path}")
    if not treatment_path.is_file():
        raise FileNotFoundError(f"missing treatment wiring sidecar: {treatment_path}")

    measured = _shared_measured_step_ids(oracle_receipt, treatment_receipt)
    if sidecar_coverage is None:
        sidecar_coverage = diagnose_sidecar_coverage(oracle_path, treatment_path)
    else:
        sidecar_coverage = dict(sidecar_coverage)

    base_audit = {
        "o1_witness_domain": O1_WITNESS_DOMAIN,
        "o1_skip_policy": O1_SKIP_POLICY,
    }

    if sidecar_coverage.get("structural_fail"):
        stats = _finalize_wiring_guard_stats(
            l1_max=0.0,
            crossing_disagreements=0,
            equal_lanes=0,
            total_lanes=0,
            measured_step_count=len(measured),
        )
        stats["sidecar_coverage_diagnostics"] = dict(sidecar_coverage)
        stats["structural_compare_skipped"] = True
        stats.update(base_audit)
        return stats

    oracle_keyed, _, _ = _index_sidecar_file(oracle_path)
    treatment_keyed, _, _ = _index_sidecar_file(treatment_path)
    shared_keys = sorted(set(oracle_keyed).intersection(treatment_keyed))

    l1_max = 0.0
    total_lanes = 0
    equal_lanes = 0
    out_of_domain_lane_count = 0

    for key in shared_keys:
        step_id, _state_key = key
        if int(step_id) <= WARMUP_STEPS:
            continue
        oracle_record = oracle_keyed[key]
        treatment_record = treatment_keyed[key]
        o_vals = [int(v) for v in oracle_record["accumulator_lanes"]]
        t_vals = [int(v) for v in treatment_record["accumulator_lanes"]]
        if len(o_vals) != len(t_vals):
            return {
                **base_audit,
                "structural_fail": True,
                "structural_reason": "w8_lane_length_mismatch",
                "sidecar_coverage_diagnostics": dict(sidecar_coverage),
                "measured_step_count": len(measured),
                "total_lane_count": 0,
                "vote_update_state_accumulator_equality_rate": 0.0,
            }
        for o_val, t_val in zip(o_vals, t_vals, strict=True):
            if _lane_out_of_w8_domain(o_val) or _lane_out_of_w8_domain(t_val):
                out_of_domain_lane_count += 1
                continue
            total_lanes += 1
            delta = abs(int(o_val) - int(t_val))
            l1_max = max(l1_max, float(delta))
            if delta == 0:
                equal_lanes += 1

    if out_of_domain_lane_count > 0:
        return {
            **base_audit,
            "structural_fail": True,
            "structural_reason": STRUCTURAL_REASON_W8_LANE_OUT_OF_DOMAIN,
            "out_of_domain_lane_count": int(out_of_domain_lane_count),
            "sidecar_coverage_diagnostics": dict(sidecar_coverage),
            "measured_step_count": len(measured),
            "total_lane_count": 0,
            "vote_update_state_accumulator_equality_rate": 0.0,
        }

    stats = _finalize_wiring_guard_stats(
        l1_max=l1_max,
        crossing_disagreements=0,
        equal_lanes=equal_lanes,
        total_lanes=total_lanes,
        measured_step_count=len(measured),
    )
    sidecar_coverage = dict(sidecar_coverage)
    sidecar_coverage["matched_key_compared_lane_count"] = int(total_lanes)
    stats["sidecar_coverage_diagnostics"] = sidecar_coverage
    stats.update(base_audit)
    return stats


__all__ = [
    "O1_SKIP_POLICY",
    "O1_WITNESS_DOMAIN",
    "STRUCTURAL_REASON_W8_LANE_OUT_OF_DOMAIN",
    "compare_w8_o1_lane_equality_streaming",
]
