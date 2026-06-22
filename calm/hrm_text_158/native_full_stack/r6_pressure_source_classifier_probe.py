"""Read-only R6 pressure-source classifier over banked sidecar q/acc trajectories."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from calm.hrm_text_158.native_full_stack.acc_width_recorded_row_sweep import (
    CANONICAL_VOTE_UPDATE_THRESHOLD_ABS,
)
from calm.hrm_text_158.native_full_stack.r5_acc_term_measurement_probe import (
    cross_check_sidecar_against_receipt,
    extract_last_sidecar_records,
    file_sha256,
    min_lossless_width_for_tensor,
    modules_from_sidecar_records,
)
from calm.hrm_text_158.native_full_stack.s3bb_headroom_telemetry import (
    _iter_sidecar_records,
)

R6_PROBE_SCHEMA_VERSION = "hrm_text_158_r6_pressure_source_classifier_probe/v1"

RELIEF_THRESHOLD = 0.25
PERSIST_THRESHOLD = 0.50
SPARSE_Q_RATIO_MAX = 0.05
INTRINSIC_Q_CHANGE_PRESSURE_MIN = 0.50
HIGH_PRESSURE_ABS = int(CANONICAL_VOTE_UPDATE_THRESHOLD_ABS)

BRANCH_HARNESS_FAIL = "R6_HARNESS_FAIL"
BRANCH_READ_PATH_FAIL = "R6_READ_PATH_FAIL"
BRANCH_ARTIFACT_INSUFFICIENT = "R6_ARTIFACT_INSUFFICIENT"
BRANCH_PERSISTS = "R6_PRESSURE_PERSISTS_UNDER_Q_APPLY_PROXY"
BRANCH_RELIEVED = "R6_PRESSURE_RELIEVED_BY_Q_APPLY_PROXY"
BRANCH_INTRINSIC = "R6_INTRINSIC_SIGNAL_PROXY"

EXPLICIT_NON_CLAIMS: tuple[str, ...] = (
    "proxy_not_proof",
    "no_backlog_age_or_drainage_claim",
    "no_mechanism_reducible_claim",
    "no_stability_verdict",
    "no_sub2_claim",
    "no_readiness_claim",
    "no_hot_path_claim",
    "no_trainer_or_gpu",
    "no_pt_mutation_or_commit",
    "no_raw_per_lane_arrays",
    "no_decision_surface_claim",
)

NEXT_ACTION_BY_BRANCH: dict[str, str] = {
    BRANCH_PERSISTS: "from_clean_contiguous_instrumented_run",
    BRANCH_RELIEVED: "dynamics_run_before_mechanism_redesign",
    BRANCH_INTRINSIC: "lower_amplitude_decay_sign_compressed_mechanism_design",
    BRANCH_ARTIFACT_INSUFFICIENT: "instrumentation_not_interpretation",
    BRANCH_HARNESS_FAIL: "stop_and_fix_inputs",
    BRANCH_READ_PATH_FAIL: "stop_and_fix_read_authority",
}

REQUIRED_RECORD_FIELDS: tuple[str, ...] = (
    "step",
    "state_key",
    "accumulator_lanes",
    "q_lanes",
)


def _trend_sign(delta: int) -> int:
    if delta < 0:
        return -1
    if delta > 0:
        return 1
    return 0


def index_sidecar_records(sidecar_path: Path) -> dict[str, dict[int, dict[str, Any]]]:
    index: dict[str, dict[int, dict[str, Any]]] = {}
    for record in _iter_sidecar_records(sidecar_path):
        state_key = str(record["state_key"])
        step = int(record["step"])
        by_step = index.setdefault(state_key, {})
        if step in by_step:
            raise ValueError(f"duplicate sidecar record for {state_key} step {step}")
        by_step[step] = record
    return index


def validate_record(record: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    for field in REQUIRED_RECORD_FIELDS:
        if field not in record:
            failures.append(f"missing_field:{field}")
    if failures:
        return failures
    acc = record["accumulator_lanes"]
    q = record["q_lanes"]
    if not isinstance(acc, list) or not isinstance(q, list):
        failures.append("lanes_not_lists")
        return failures
    if len(acc) != len(q):
        failures.append("acc_q_length_mismatch")
    return failures


def validate_index(
    index: Mapping[str, Mapping[int, Mapping[str, Any]]],
) -> list[str]:
    failures: list[str] = []
    for state_key in sorted(index.keys()):
        for step in sorted(int(step) for step in index[state_key].keys()):
            record_failures = validate_record(index[state_key][step])
            for failure in record_failures:
                failures.append(f"{state_key}@step{step}:{failure}")
    return failures


def _pair_content_hash(summary: Mapping[str, Any]) -> str:
    payload = json.dumps(summary, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _empty_trajectory_metrics() -> dict[str, Any]:
    return {
        "pressure_mass_by_step": {},
        "pressure_mass_first": 0,
        "pressure_mass_last": 0,
        "pressure_mass_trend": 0,
        "pressure_mass_trend_sign": 0,
        "relief_fraction": 0.0,
        "persist_fraction": 0.0,
        "q_transition_mass_ratio": 0.0,
        "intrinsic_persistence_fraction_mean": None,
        "intrinsic_pair_count": 0,
        "steps_observed": 0,
        "modules_observed": 0,
        "adjacent_pairs_total": 0,
        "adjacent_pairs": [],
        "steps_per_module": {},
        "min_steps_per_module": 0,
        "first_step": None,
        "last_step": None,
        "per_step_module": [],
        "thresholds": {
            "RELIEF_THRESHOLD": RELIEF_THRESHOLD,
            "PERSIST_THRESHOLD": PERSIST_THRESHOLD,
            "SPARSE_Q_RATIO_MAX": SPARSE_Q_RATIO_MAX,
            "INTRINSIC_Q_CHANGE_PRESSURE_MIN": INTRINSIC_Q_CHANGE_PRESSURE_MIN,
            "HIGH_PRESSURE_ABS": HIGH_PRESSURE_ABS,
        },
    }


def _lanes_tensor(record: Mapping[str, Any], field: str) -> torch.Tensor:
    return torch.tensor(record[field], dtype=torch.int32).reshape(-1)


def _high_pressure_count(acc: torch.Tensor) -> int:
    return int(torch.sum(acc.abs().to(torch.int64) >= HIGH_PRESSURE_ABS).item())


def compute_trajectory_metrics(
    index: Mapping[str, Mapping[int, Mapping[str, Any]]],
) -> dict[str, Any]:
    pressure_mass_by_step: dict[int, int] = {}
    per_module_steps: dict[str, list[int]] = {}
    per_step_module: list[dict[str, Any]] = []
    adjacent_pairs: list[dict[str, Any]] = []

    q_transition_lane_total = 0
    pressure_mass_lane_steps_total = 0

    intrinsic_fractions: list[float] = []
    intrinsic_pair_count = 0

    observed_steps: set[int] = set()
    adjacent_pairs_total = 0

    for state_key in sorted(index.keys()):
        steps = sorted(int(step) for step in index[state_key].keys())
        per_module_steps[state_key] = steps
        observed_steps.update(steps)
        for step in steps:
            record = index[state_key][step]
            acc = _lanes_tensor(record, "accumulator_lanes").to(torch.int16)
            hp_count = _high_pressure_count(acc)
            pressure_mass_by_step[step] = pressure_mass_by_step.get(step, 0) + hp_count

            min_w = min_lossless_width_for_tensor(acc)
            lane_count = int(acc.numel())
            per_step_module.append(
                {
                    "state_key": state_key,
                    "step": step,
                    "high_pressure_lane_count": hp_count,
                    "signed_w_floor": int(min_w) if min_w is not None else None,
                    "count_abs_gte_7": int(torch.sum(acc.abs() >= 7).item()),
                    "count_abs_gte_10": int(torch.sum(acc.abs() >= HIGH_PRESSURE_ABS).item()),
                    "count_abs_gt_15": int(torch.sum(acc.abs() > 15).item()),
                    "count_abs_gte_25": int(torch.sum(acc.abs() >= 25).item()),
                    "max_abs": int(acc.abs().max().item()) if lane_count > 0 else 0,
                    "fraction_abs_gte_10": (
                        float(hp_count) / float(lane_count) if lane_count > 0 else 0.0
                    ),
                }
            )

        for prev_step, curr_step in zip(steps[:-1], steps[1:], strict=False):
            adjacent_pairs_total += 1
            prev = index[state_key][prev_step]
            curr = index[state_key][curr_step]
            acc_prev = _lanes_tensor(prev, "accumulator_lanes").to(torch.int16)
            acc_curr = _lanes_tensor(curr, "accumulator_lanes").to(torch.int16)
            q_prev = _lanes_tensor(prev, "q_lanes")
            q_curr = _lanes_tensor(curr, "q_lanes")
            q_changed = q_curr != q_prev
            q_transition_count = int(torch.sum(q_changed).item())
            lane_count = int(q_changed.numel())

            hp_prev = acc_prev.abs() >= HIGH_PRESSURE_ABS
            hp_curr = acc_curr.abs() >= HIGH_PRESSURE_ABS
            pressure_mass_prev = int(torch.sum(hp_prev).item())
            pressure_mass_curr = int(torch.sum(hp_curr).item())
            high_pressure_unchanged_q = int(torch.sum(hp_prev & (q_prev == q_curr)).item())
            high_pressure_after_q_change = int(torch.sum(hp_curr & q_changed).item())

            q_transition_lane_total += q_transition_count
            pressure_mass_lane_steps_total += pressure_mass_prev

            pair_summary = {
                "state_key": state_key,
                "step_from": prev_step,
                "step_to": curr_step,
                "q_transition_count": q_transition_count,
                "q_transition_fraction": (
                    float(q_transition_count) / float(lane_count) if lane_count > 0 else 0.0
                ),
                "high_pressure_unchanged_q": high_pressure_unchanged_q,
                "high_pressure_after_q_change": high_pressure_after_q_change,
                "pressure_mass_delta": pressure_mass_curr - pressure_mass_prev,
            }
            adjacent_pairs.append(
                {
                    **pair_summary,
                    "pair_content_hash": _pair_content_hash(pair_summary),
                }
            )

        for left, mid, right in zip(steps[:-2], steps[1:-1], steps[2:], strict=False):
            rec_left = index[state_key][left]
            rec_mid = index[state_key][mid]
            rec_right = index[state_key][right]
            acc_mid = _lanes_tensor(rec_mid, "accumulator_lanes").to(torch.int16)
            acc_right = _lanes_tensor(rec_right, "accumulator_lanes").to(torch.int16)
            q_left = _lanes_tensor(rec_left, "q_lanes")
            q_mid = _lanes_tensor(rec_mid, "q_lanes")
            q_changed_into_mid = q_mid != q_left
            with_q_change = (acc_mid.abs() >= HIGH_PRESSURE_ABS) & q_changed_into_mid
            with_count = int(torch.sum(with_q_change).item())
            if with_count <= 0:
                continue
            after_count = int(
                torch.sum(with_q_change & (acc_right.abs() >= HIGH_PRESSURE_ABS)).item()
            )
            intrinsic_fractions.append(after_count / float(with_count))
            intrinsic_pair_count += 1

    if not pressure_mass_by_step:
        ordered_steps: list[int] = []
    else:
        ordered_steps = sorted(pressure_mass_by_step.keys())

    if ordered_steps:
        first_step = ordered_steps[0]
        last_step = ordered_steps[-1]
        pressure_mass_first = int(pressure_mass_by_step[first_step])
        pressure_mass_last = int(pressure_mass_by_step[last_step])
    else:
        first_step = None
        last_step = None
        pressure_mass_first = 0
        pressure_mass_last = 0

    pressure_mass_trend = pressure_mass_last - pressure_mass_first
    pressure_mass_trend_sign = _trend_sign(pressure_mass_trend)
    relief_fraction = (
        float(pressure_mass_first - pressure_mass_last) / float(pressure_mass_first)
        if pressure_mass_first > 0
        else 0.0
    )
    persist_fraction = (
        float(pressure_mass_last) / float(pressure_mass_first)
        if pressure_mass_first > 0
        else 0.0
    )
    q_transition_mass_ratio = (
        float(q_transition_lane_total) / float(pressure_mass_lane_steps_total)
        if pressure_mass_lane_steps_total > 0
        else 0.0
    )
    intrinsic_persistence_fraction_mean = (
        float(sum(intrinsic_fractions) / len(intrinsic_fractions))
        if intrinsic_fractions
        else None
    )

    steps_per_module = {
        key: len(per_module_steps.get(key, [])) for key in sorted(per_module_steps.keys())
    }
    min_steps_per_module = min(steps_per_module.values()) if steps_per_module else 0

    return {
        "pressure_mass_by_step": {str(k): int(v) for k, v in pressure_mass_by_step.items()},
        "pressure_mass_first": pressure_mass_first,
        "pressure_mass_last": pressure_mass_last,
        "pressure_mass_trend": pressure_mass_trend,
        "pressure_mass_trend_sign": pressure_mass_trend_sign,
        "relief_fraction": relief_fraction,
        "persist_fraction": persist_fraction,
        "q_transition_mass_ratio": q_transition_mass_ratio,
        "intrinsic_persistence_fraction_mean": intrinsic_persistence_fraction_mean,
        "intrinsic_pair_count": intrinsic_pair_count,
        "steps_observed": len(observed_steps),
        "modules_observed": len(index),
        "adjacent_pairs_total": adjacent_pairs_total,
        "adjacent_pairs": adjacent_pairs,
        "steps_per_module": steps_per_module,
        "min_steps_per_module": int(min_steps_per_module),
        "first_step": first_step,
        "last_step": last_step,
        "per_step_module": per_step_module,
        "thresholds": {
            "RELIEF_THRESHOLD": RELIEF_THRESHOLD,
            "PERSIST_THRESHOLD": PERSIST_THRESHOLD,
            "SPARSE_Q_RATIO_MAX": SPARSE_Q_RATIO_MAX,
            "INTRINSIC_Q_CHANGE_PRESSURE_MIN": INTRINSIC_Q_CHANGE_PRESSURE_MIN,
            "HIGH_PRESSURE_ABS": HIGH_PRESSURE_ABS,
        },
    }


def select_branch(
    *,
    harness_fail: bool,
    cross_check_pass: bool,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    if harness_fail:
        return {"branch": BRANCH_HARNESS_FAIL, "next_action": NEXT_ACTION_BY_BRANCH[BRANCH_HARNESS_FAIL]}
    if not cross_check_pass:
        return {
            "branch": BRANCH_READ_PATH_FAIL,
            "next_action": NEXT_ACTION_BY_BRANCH[BRANCH_READ_PATH_FAIL],
        }
    if int(metrics["min_steps_per_module"]) < 2:
        return {
            "branch": BRANCH_ARTIFACT_INSUFFICIENT,
            "next_action": NEXT_ACTION_BY_BRANCH[BRANCH_ARTIFACT_INSUFFICIENT],
            "reason": "fewer_than_two_steps_per_module",
        }

    trend_sign = int(metrics["pressure_mass_trend_sign"])
    relief_fraction = float(metrics["relief_fraction"])
    persist_fraction = float(metrics["persist_fraction"])
    q_ratio = float(metrics["q_transition_mass_ratio"])
    intrinsic_mean = metrics["intrinsic_persistence_fraction_mean"]
    intrinsic_pair_count = int(metrics["intrinsic_pair_count"])

    if trend_sign < 0 and relief_fraction >= RELIEF_THRESHOLD:
        return {
            "branch": BRANCH_RELIEVED,
            "next_action": NEXT_ACTION_BY_BRANCH[BRANCH_RELIEVED],
            "relief_fraction": relief_fraction,
            "pressure_mass_trend_sign": trend_sign,
        }

    if (
        trend_sign >= 0
        and persist_fraction >= PERSIST_THRESHOLD
        and q_ratio <= SPARSE_Q_RATIO_MAX
    ):
        return {
            "branch": BRANCH_PERSISTS,
            "next_action": NEXT_ACTION_BY_BRANCH[BRANCH_PERSISTS],
            "persist_fraction": persist_fraction,
            "q_transition_mass_ratio": q_ratio,
            "pressure_mass_trend_sign": trend_sign,
        }

    if (
        trend_sign >= 0
        and intrinsic_pair_count > 0
        and intrinsic_mean is not None
        and float(intrinsic_mean) >= INTRINSIC_Q_CHANGE_PRESSURE_MIN
    ):
        return {
            "branch": BRANCH_INTRINSIC,
            "next_action": NEXT_ACTION_BY_BRANCH[BRANCH_INTRINSIC],
            "intrinsic_persistence_fraction_mean": float(intrinsic_mean),
            "intrinsic_pair_count": intrinsic_pair_count,
            "pressure_mass_trend_sign": trend_sign,
        }

    return {
        "branch": BRANCH_ARTIFACT_INSUFFICIENT,
        "next_action": NEXT_ACTION_BY_BRANCH[BRANCH_ARTIFACT_INSUFFICIENT],
        "reason": "no_science_predicate_fired",
    }


def build_classifier_from_index(
    *,
    index: Mapping[str, Mapping[int, Mapping[str, Any]]],
    cross_check: Mapping[str, Any],
    harness_fail: bool = False,
    run_root: str | None = None,
    head_sha256: str | None = None,
    input_artifact_hashes: Mapping[str, str] | None = None,
    r4_baseline: Mapping[str, Any] | None = None,
    cross_check_required: bool = True,
    validation_failures: Sequence[str] | None = None,
) -> dict[str, Any]:
    index_validation_failures = list(validate_index(index)) if index else []
    all_validation_failures = list(validation_failures or []) + index_validation_failures
    effective_harness_fail = harness_fail or bool(all_validation_failures)
    metrics = (
        compute_trajectory_metrics(index)
        if not effective_harness_fail and index
        else _empty_trajectory_metrics()
    )
    cross_check_pass = bool(cross_check.get("cross_check_pass"))
    if not cross_check_required:
        cross_check_pass = True
    branch = select_branch(
        harness_fail=effective_harness_fail,
        cross_check_pass=cross_check_pass,
        metrics=metrics,
    )
    result = {
        "schema_version": R6_PROBE_SCHEMA_VERSION,
        "raw_arrays_included": False,
        "run_root": run_root,
        "head_sha256": head_sha256,
        "input_artifact_hashes": dict(input_artifact_hashes or {}),
        "r4_baseline": dict(r4_baseline or {}),
        "cross_check": dict(cross_check),
        "cross_check_required": bool(cross_check_required),
        "trajectory_metrics": metrics,
        "branch_selection": branch,
        "explicit_non_claims": list(EXPLICIT_NON_CLAIMS),
    }
    if all_validation_failures:
        result["validation_failures"] = list(all_validation_failures)
    return result


def build_classifier_probe_receipt(
    *,
    run_root: Path,
    arm_dir: str = "w6_on_q_on_treatment",
    head_sha256: str,
    expected_receipt_sha256: str | None = None,
    expected_sidecar_sha256: str | None = None,
    cross_check_required: bool = True,
) -> dict[str, Any]:
    arm_path = run_root / arm_dir
    receipt_path = arm_path / "receipt.json"
    sidecar_path = arm_path / "headroom_wiring_sidecar.jsonl"
    if not receipt_path.is_file():
        raise FileNotFoundError(f"missing receipt: {receipt_path}")
    if not sidecar_path.is_file():
        raise FileNotFoundError(f"missing sidecar: {sidecar_path}")

    pre_receipt_sha = file_sha256(receipt_path)
    pre_sidecar_sha = file_sha256(sidecar_path)
    harness_fail = False
    harness_failures: list[str] = []

    if expected_receipt_sha256 and pre_receipt_sha != expected_receipt_sha256:
        harness_fail = True
        harness_failures.append("receipt_sha256_mismatch")
    if expected_sidecar_sha256 and pre_sidecar_sha != expected_sidecar_sha256:
        harness_fail = True
        harness_failures.append("sidecar_sha256_mismatch")

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    ledger = receipt.get("r4_persistent_ledger") or {}
    receipt_rows = ledger.get("r4_per_module_acc_rows") or []
    expected_content_sha = str(ledger.get("r4_acc_packed_content_sha256", ""))
    logical_shapes = {str(row["state_key"]): row["logical_shape"] for row in receipt_rows}

    cross_check: dict[str, Any] = {"cross_check_pass": False, "reason": "harness_fail_before_read"}
    observed_steps: dict[str, int] = {}
    index: dict[str, dict[int, dict[str, Any]]] = {}
    if not harness_fail:
        try:
            index = index_sidecar_records(sidecar_path)
            validation_failures = validate_index(index)
            if validation_failures:
                harness_fail = True
                harness_failures.extend(validation_failures)
        except (ValueError, json.JSONDecodeError, KeyError, TypeError) as exc:
            harness_fail = True
            harness_failures.append(f"sidecar_parse_error:{exc}")
        if not harness_fail:
            sidecar_records, observed_steps = extract_last_sidecar_records(sidecar_path)
            modules = modules_from_sidecar_records(sidecar_records, logical_shapes)
            cross_check = cross_check_sidecar_against_receipt(
                modules=modules,
                receipt_rows=receipt_rows,
                expected_content_sha256=expected_content_sha,
            )

    post_receipt_sha = file_sha256(receipt_path)
    post_sidecar_sha = file_sha256(sidecar_path)
    if post_receipt_sha != pre_receipt_sha or post_sidecar_sha != pre_sidecar_sha:
        harness_fail = True
        harness_failures.append("input_artifact_mutated_during_read")

    r4_baseline = {
        "r4_q_physical_bits_per_weight": ledger.get("r4_q_physical_bits_per_weight"),
        "r4_acc_physical_bits_per_weight": ledger.get("r4_acc_physical_bits_per_weight"),
        "r4_checkpoint_inclusive_physical_bits_per_weight": ledger.get(
            "r4_checkpoint_inclusive_physical_bits_per_weight"
        ),
        "r4_acc_packed_content_sha256": expected_content_sha,
    }

    result = build_classifier_from_index(
        index=index if not harness_fail else {},
        cross_check=cross_check,
        harness_fail=harness_fail,
        run_root=str(run_root),
        head_sha256=head_sha256,
        input_artifact_hashes={
            "receipt_sha256_pre": pre_receipt_sha,
            "receipt_sha256_post": post_receipt_sha,
            "sidecar_sha256_pre": pre_sidecar_sha,
            "sidecar_sha256_post": post_sidecar_sha,
        },
        r4_baseline=r4_baseline,
        cross_check_required=cross_check_required,
    )
    if harness_failures:
        result["harness_failures"] = harness_failures
    if observed_steps:
        result["observed_max_step_per_module"] = observed_steps
    return result
