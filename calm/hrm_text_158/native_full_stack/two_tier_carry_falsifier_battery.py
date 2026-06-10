"""CPU read-only two-tier carry falsifier battery (F1/F2/F3) over B2b recorded rows."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from calm.hrm_text_158.native_full_stack.acc_width_recorded_row_sweep import (
    CANONICAL_THRESHOLD_SOURCE,
    CANONICAL_VOTE_UPDATE_THRESHOLD_ABS,
    REQUIRED_TRACE_ROW_FIELDS,
    VoteSpecParsed,
    build_required_field_inventory,
    build_teacher_forced_applied_candidate_ids,
    crosses_threshold,
    decay_vote_clamp,
    effective_clip_bounds,
    load_acc_width_trace_steps,
    resolve_vote_spec,
)
from calm.hrm_text_158.native_full_stack.accumulator_policy_shadow_screen import (
    PRE_FULL_STACK_DIAGNOSTIC_ONLY,
    _file_sha256,
    _stable_hash16,
)

BATTERY_SCHEMA_VERSION = "hrm_text_158_two_tier_carry_falsifier_battery/v0"
BATTERY_CONTRACT_ID = "two_tier_carry_falsifier_battery_v0"
BATTERY_RECEIPT_KIND = "cpu_read_only_two_tier_carry_falsifier_battery"

W_REF = 16
W_TEST = 6
HELD_STEP_START = 26
HELD_STEP_END = 50
F1_JACCARD_BAR = 0.90
F1_HELD_PASS_RATE_BAR = 0.90
F2_TAU_B_BAR = 0.90
F2_MIN_COMPARABLE_PAIRS = 10
F2_SATURATION_VACUOUS_FRACTION = 0.80
F3_DIFFER_RATE_MAX = 0.05
MIN_HELD_QUALIFYING_STEPS = 10

BANKED_F4_AUDIT_RECEIPT_SHA = (
    "098649204e17c0a274f2191855867aa149ef16e2d635427c4885fdf5f0b093fe"
)
BANKED_F4_PRIMARY_LABEL = "transient_compute_control_only"
EXPECTED_TRACE_HASH = "cb373de78030c5a9"
REQUIRED_STORAGE_CLASS = "durable_not_tmp"

ALLOWED_PHASES = frozenset({"acc_width_sweep_v0", "two_tier_falsifier_battery_v0"})
UPSTREAM_EXIT_KEYS = (
    "b2c_replay",
    "audit_v0",
    "determinism_gate",
    "acc_width_sweep_v0",
)

LABEL_SCREEN_HARNESS_OR_GATE_FAIL = "screen_harness_or_gate_fail"
LABEL_CAP_PRIORITY_REQUIRES_FULL_MAGNITUDE = "cap_priority_requires_full_magnitude"
LABEL_SELECTION_MUST_STAY_TRANSIENT = "selection_must_stay_transient"
LABEL_SELECTION_MUST_STAY_TRANSIENT_BROAD = "selection_must_stay_transient_broad"
LABEL_CARRY_W6_FALSIFIERS_PASS = "carry_w6_falsifiers_pass_on_trace1"

CLASSIFIER_PRECEDENCE = (
    LABEL_SCREEN_HARNESS_OR_GATE_FAIL,
    LABEL_CAP_PRIORITY_REQUIRES_FULL_MAGNITUDE,
    LABEL_SELECTION_MUST_STAY_TRANSIENT,
    LABEL_SELECTION_MUST_STAY_TRANSIENT_BROAD,
    LABEL_CARRY_W6_FALSIFIERS_PASS,
)

MANIFEST_ROLE_ALIASES = {
    "stable_copy_00": ("stable_copy_00", "stable_trace"),
    "b2b_trace": ("b2b_trace", "original_trace"),
    "capture_receipt": ("capture_receipt",),
    "b2c_receipt": ("b2c_receipt",),
    "audit_receipt": ("audit_receipt",),
    "acc_width_receipt": ("acc_width_receipt", "acc_width_recorded_row_sweep_receipt"),
}
REQUIRED_MANIFEST_ROLES = frozenset(MANIFEST_ROLE_ALIASES)


@dataclass(frozen=True)
class LaneMaps:
    w_ref: dict[tuple[int, int], int]
    w_test: dict[tuple[int, int], int]
    w_ref_crossings: dict[tuple[int, int], bool]
    w_test_crossings: dict[tuple[int, int], bool]


def _row_key(step_index: int, flat_index: int) -> tuple[int, int]:
    return int(step_index), int(flat_index)


def recompute_new_acc(
    row: Mapping[str, Any],
    *,
    vote_spec: VoteSpecParsed,
    width: int,
) -> int:
    clip_min, clip_max = effective_clip_bounds(
        width,
        vote_spec.accumulator_clip_min,
        vote_spec.accumulator_clip_max,
    )
    return decay_vote_clamp(
        int(row["pre_accumulator_i16"]),
        int(row["vote_value"]),
        clip_min=clip_min,
        clip_max=clip_max,
        decay_numerator=vote_spec.decay_numerator,
        decay_denominator=vote_spec.decay_denominator,
    )


def build_lane_maps(
    steps: Sequence[Mapping[str, Any]],
    *,
    vote_spec: VoteSpecParsed,
) -> LaneMaps:
    w_ref: dict[tuple[int, int], int] = {}
    w_test: dict[tuple[int, int], int] = {}
    w_ref_crossings: dict[tuple[int, int], bool] = {}
    w_test_crossings: dict[tuple[int, int], bool] = {}
    for step in steps:
        step_index = int(step["optimizer_step_index"])
        for row in step.get("sampled_candidate_table") or ():
            if not isinstance(row, Mapping):
                continue
            flat_index = int(row["flat_index"])
            key = _row_key(step_index, flat_index)
            new_ref = recompute_new_acc(row, vote_spec=vote_spec, width=W_REF)
            new_test = recompute_new_acc(row, vote_spec=vote_spec, width=W_TEST)
            w_ref[key] = new_ref
            w_test[key] = new_test
            w_ref_crossings[key] = crosses_threshold(
                new_ref,
                current_q_level=int(row["current_q_level"]),
                threshold_abs=vote_spec.threshold_abs,
            )
            w_test_crossings[key] = crosses_threshold(
                new_test,
                current_q_level=int(row["current_q_level"]),
                threshold_abs=vote_spec.threshold_abs,
            )
    return LaneMaps(
        w_ref=w_ref,
        w_test=w_test,
        w_ref_crossings=w_ref_crossings,
        w_test_crossings=w_test_crossings,
    )


def _applied_flip_flat_indices(
    step: Mapping[str, Any],
    *,
    applied_candidate_ids_by_step: Mapping[int, str],
) -> list[int]:
    telemetry = dict(step.get("post_update_telemetry") or {})
    override = telemetry.get("applied_flip_flat_indices")
    if isinstance(override, list):
        return [int(value) for value in override]
    if int(telemetry.get("q_changed_count", 0)) <= 0:
        return []
    step_index = int(step["optimizer_step_index"])
    candidate_id = applied_candidate_ids_by_step.get(step_index)
    if not candidate_id:
        return []
    for row in step.get("sampled_candidate_table") or ():
        if not isinstance(row, Mapping):
            continue
        if str(row.get("candidate_id")) == str(candidate_id):
            return [int(row["flat_index"])]
    return []


def _crossing_flat_indices_for_step(
    step: Mapping[str, Any],
    *,
    lane_maps: LaneMaps,
) -> list[int]:
    step_index = int(step["optimizer_step_index"])
    indices = [
        int(row["flat_index"])
        for row in step.get("sampled_candidate_table") or ()
        if isinstance(row, Mapping)
        and lane_maps.w_ref_crossings.get(_row_key(step_index, int(row["flat_index"])), False)
    ]
    return sorted(indices)


def _top_k_crossing_set(
    crossing_flat_indices: Sequence[int],
    *,
    lane_map: Mapping[tuple[int, int], int],
    step_index: int,
    k: int,
) -> set[int]:
    if k <= 0:
        return set()
    ranked = sorted(
        crossing_flat_indices,
        key=lambda flat_index: (
            -abs(int(lane_map[_row_key(step_index, int(flat_index))])),
            int(flat_index),
        ),
    )
    return set(ranked[: int(k)])


def jaccard_similarity(left: set[int], right: set[int]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def kendall_tau_b(
    x: Sequence[int],
    y: Sequence[int],
) -> tuple[float, int, int]:
    """Knight (1966) tau-b with tie correction: (C-D)/sqrt((C+D+Tx)(C+D+Ty))."""

    n = len(x)
    if n < 2:
        return 0.0, 0, 0
    concordant = discordant = ties_x = ties_y = 0
    for left in range(n):
        for right in range(left + 1, n):
            if x[left] == x[right] and y[left] == y[right]:
                continue
            if x[left] == x[right]:
                ties_x += 1
                continue
            if y[left] == y[right]:
                ties_y += 1
                continue
            sign_x = (x[left] > x[right]) - (x[left] < x[right])
            sign_y = (y[left] > y[right]) - (y[left] < y[right])
            if sign_x == sign_y:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + ties_x) * (concordant + discordant + ties_y)
    )
    if denominator == 0.0:
        return 0.0, concordant + discordant, discordant
    tau_b = (concordant - discordant) / denominator
    return tau_b, concordant + discordant, discordant


def _argmax_abs_with_tiebreak(
    flat_indices: Sequence[int],
    *,
    lane_map: Mapping[tuple[int, int], int],
    step_index: int,
) -> int | None:
    if not flat_indices:
        return None
    best_flat = min(flat_indices)
    best_abs = -1
    for flat_index in flat_indices:
        value = abs(int(lane_map[_row_key(step_index, int(flat_index))]))
        if value > best_abs or (value == best_abs and int(flat_index) < best_flat):
            best_abs = value
            best_flat = int(flat_index)
    return best_flat


def summarize_quantiles(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "max": 0.0}

    ordered = sorted(float(value) for value in values)

    def _quantile(position: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        index = position * (len(ordered) - 1)
        lower = int(math.floor(index))
        upper = int(math.ceil(index))
        if lower == upper:
            return ordered[lower]
        weight = index - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "min": ordered[0],
        "p25": _quantile(0.25),
        "p50": _quantile(0.50),
        "p75": _quantile(0.75),
        "max": ordered[-1],
    }


def is_held_step(step_index: int) -> bool:
    return HELD_STEP_START <= int(step_index) <= HELD_STEP_END


def evaluate_f1_step(
    step: Mapping[str, Any],
    *,
    lane_maps: LaneMaps,
    applied_candidate_ids_by_step: Mapping[int, str],
) -> dict[str, Any]:
    step_index = int(step["optimizer_step_index"])
    crossing = _crossing_flat_indices_for_step(step, lane_maps=lane_maps)
    applied_indices = _applied_flip_flat_indices(
        step,
        applied_candidate_ids_by_step=applied_candidate_ids_by_step,
    )
    k = len(applied_indices)
    if k == 0:
        return {
            "optimizer_step_index": step_index,
            "qualifying": False,
            "skip_reason": "f1_skip_no_applied",
            "k": 0,
            "crossing_count": len(crossing),
        }
    if k > len(crossing):
        return {
            "optimizer_step_index": step_index,
            "qualifying": False,
            "skip_reason": "trace_policy_mismatch",
            "k": k,
            "crossing_count": len(crossing),
        }
    o_ref = _top_k_crossing_set(
        crossing,
        lane_map=lane_maps.w_ref,
        step_index=step_index,
        k=k,
    )
    o_test = _top_k_crossing_set(
        crossing,
        lane_map=lane_maps.w_test,
        step_index=step_index,
        k=k,
    )
    jaccard = jaccard_similarity(o_ref, o_test)
    return {
        "optimizer_step_index": step_index,
        "qualifying": True,
        "k": k,
        "crossing_count": len(crossing),
        "jaccard": jaccard,
        "pass": jaccard >= F1_JACCARD_BAR,
        "o_ref_size": len(o_ref),
        "o_test_size": len(o_test),
    }


def evaluate_f2_step(
    step: Mapping[str, Any],
    *,
    lane_maps: LaneMaps,
) -> dict[str, Any]:
    step_index = int(step["optimizer_step_index"])
    rows = [
        row
        for row in step.get("sampled_candidate_table") or ()
        if isinstance(row, Mapping)
    ]
    rows.sort(key=lambda row: int(row["flat_index"]))
    v_ref = [
        int(lane_maps.w_ref[_row_key(step_index, int(row["flat_index"]))]) for row in rows
    ]
    v_test = [
        int(lane_maps.w_test[_row_key(step_index, int(row["flat_index"]))]) for row in rows
    ]
    tau_b, comparable_pairs, discordant_pairs = kendall_tau_b(v_ref, v_test)
    qualifying = comparable_pairs >= F2_MIN_COMPARABLE_PAIRS
    return {
        "optimizer_step_index": step_index,
        "qualifying": qualifying,
        "skip_reason": None if qualifying else "f2_insufficient_pairs",
        "tau_b": tau_b,
        "n_comparable_pairs": comparable_pairs,
        "n_discordant_pairs": discordant_pairs,
        "pass": qualifying and tau_b >= F2_TAU_B_BAR,
    }


def evaluate_f3_step(
    step: Mapping[str, Any],
    *,
    lane_maps: LaneMaps,
) -> dict[str, Any]:
    step_index = int(step["optimizer_step_index"])
    crossing = _crossing_flat_indices_for_step(step, lane_maps=lane_maps)
    if not crossing:
        return {
            "optimizer_step_index": step_index,
            "qualifying": False,
            "skip_reason": "f3_skip_no_crossers",
        }
    a_ref = _argmax_abs_with_tiebreak(
        crossing,
        lane_map=lane_maps.w_ref,
        step_index=step_index,
    )
    a_test = _argmax_abs_with_tiebreak(
        crossing,
        lane_map=lane_maps.w_test,
        step_index=step_index,
    )
    agree = a_ref == a_test
    return {
        "optimizer_step_index": step_index,
        "qualifying": True,
        "a_ref": a_ref,
        "a_test": a_test,
        "pass": agree,
        "differ": not agree,
    }


def aggregate_f1(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    held = [item for item in results if is_held_step(int(item["optimizer_step_index"]))]
    qualifying = [item for item in held if item.get("qualifying")]
    pass_count = sum(1 for item in qualifying if item.get("pass"))
    fail_count = len(qualifying) - pass_count
    rate = pass_count / len(qualifying) if qualifying else 0.0
    return {
        "held_qualifying_steps": len(qualifying),
        "held_pass_count": pass_count,
        "held_fail_count": fail_count,
        "held_pass_rate": rate,
        "held_pass": len(qualifying) >= MIN_HELD_QUALIFYING_STEPS
        and rate >= F1_HELD_PASS_RATE_BAR,
        "per_step_jaccard": summarize_quantiles(
            [float(item["jaccard"]) for item in qualifying if "jaccard" in item]
        ),
    }


def aggregate_f2(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    held = [item for item in results if is_held_step(int(item["optimizer_step_index"]))]
    qualifying = [item for item in held if item.get("qualifying")]
    disqualified = len(held) - len(qualifying)
    vacuity_fraction = disqualified / len(held) if held else 0.0
    pass_count = sum(1 for item in qualifying if item.get("pass"))
    fail_count = len(qualifying) - pass_count
    mean_tau_b = (
        sum(float(item["tau_b"]) for item in qualifying) / len(qualifying)
        if qualifying
        else 0.0
    )
    vacuity_triggered = vacuity_fraction > F2_SATURATION_VACUOUS_FRACTION
    return {
        "held_qualifying_steps": len(qualifying),
        "held_pass_count": pass_count,
        "held_fail_count": fail_count,
        "held_mean_tau_b": mean_tau_b,
        "disqualified_held_fraction": vacuity_fraction,
        "vacuity_triggered": vacuity_triggered,
        "held_pass": not vacuity_triggered
        and len(qualifying) > 0
        and mean_tau_b >= F2_TAU_B_BAR,
        "per_step_tau_b": summarize_quantiles(
            [float(item["tau_b"]) for item in qualifying]
        ),
        "n_comparable_pairs": summarize_quantiles(
            [float(item["n_comparable_pairs"]) for item in held]
        ),
    }


def aggregate_f3(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    held = [item for item in results if is_held_step(int(item["optimizer_step_index"]))]
    qualifying = [item for item in held if item.get("qualifying")]
    differ_count = sum(1 for item in qualifying if item.get("differ"))
    differ_rate = differ_count / len(qualifying) if qualifying else 0.0
    return {
        "held_qualifying_steps": len(qualifying),
        "held_differ_count": differ_count,
        "held_differ_rate": differ_rate,
        "held_pass": len(qualifying) >= MIN_HELD_QUALIFYING_STEPS
        and differ_rate <= F3_DIFFER_RATE_MAX,
    }


def build_estimand_vacuity_guard(
    *,
    f1_summary: Mapping[str, Any],
    f2_summary: Mapping[str, Any],
    f3_summary: Mapping[str, Any],
    f1_trace_policy_mismatch: bool,
) -> dict[str, Any]:
    return {
        "f1_insufficient_qualifying": int(f1_summary["held_qualifying_steps"])
        < MIN_HELD_QUALIFYING_STEPS,
        "f2_saturation_vacuous": bool(f2_summary.get("vacuity_triggered")),
        "f3_insufficient_qualifying": int(f3_summary["held_qualifying_steps"])
        < MIN_HELD_QUALIFYING_STEPS,
        "trace_policy_mismatch": bool(f1_trace_policy_mismatch),
    }


def classify_battery(
    *,
    f1_pass: bool,
    f2_pass: bool,
    f3_pass: bool,
    vacuity_guard: Mapping[str, Any],
    harness_failures: Sequence[str],
) -> dict[str, Any]:
    failures = list(dict.fromkeys(harness_failures))
    screen_triggered = bool(failures) or any(
        bool(vacuity_guard.get(key))
        for key in (
            "f1_insufficient_qualifying",
            "f2_saturation_vacuous",
            "f3_insufficient_qualifying",
            "trace_policy_mismatch",
        )
    )
    if screen_triggered:
        return {
            "primary_label": LABEL_SCREEN_HARNESS_OR_GATE_FAIL,
            "branch_precedence": list(CLASSIFIER_PRECEDENCE),
            "matched_row": 1,
            "failure_reasons": failures,
        }
    if not f1_pass and f2_pass and f3_pass:
        return {
            "primary_label": LABEL_CAP_PRIORITY_REQUIRES_FULL_MAGNITUDE,
            "branch_precedence": list(CLASSIFIER_PRECEDENCE),
            "matched_row": 2,
            "failure_reasons": failures,
        }
    if (not f2_pass or not f3_pass) and f1_pass:
        return {
            "primary_label": LABEL_SELECTION_MUST_STAY_TRANSIENT,
            "branch_precedence": list(CLASSIFIER_PRECEDENCE),
            "matched_row": 3,
            "failure_reasons": failures,
        }
    if not f1_pass and (not f2_pass or not f3_pass):
        return {
            "primary_label": LABEL_SELECTION_MUST_STAY_TRANSIENT_BROAD,
            "branch_precedence": list(CLASSIFIER_PRECEDENCE),
            "matched_row": 4,
            "failure_reasons": failures,
        }
    if f1_pass and f2_pass and f3_pass:
        return {
            "primary_label": LABEL_CARRY_W6_FALSIFIERS_PASS,
            "branch_precedence": list(CLASSIFIER_PRECEDENCE),
            "matched_row": 5,
            "failure_reasons": failures,
        }
    return {
        "primary_label": LABEL_SCREEN_HARNESS_OR_GATE_FAIL,
        "branch_precedence": list(CLASSIFIER_PRECEDENCE),
        "matched_row": 1,
        "failure_reasons": failures + ["classifier_no_match"],
    }


def _manifest_entry_path(
    manifest: Mapping[str, Any],
    role: str,
) -> Path | None:
    entries = manifest.get("artifacts") or manifest.get("entries") or []
    aliases = MANIFEST_ROLE_ALIASES.get(role, (role,))
    if isinstance(entries, Mapping):
        for alias in aliases:
            value = entries.get(alias)
            if isinstance(value, Mapping) and value.get("path"):
                return Path(str(value["path"]))
            if isinstance(value, str):
                return Path(value)
        return None
    if isinstance(entries, list):
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            entry_role = str(entry.get("role") or entry.get("name") or "")
            if entry_role in aliases:
                path = entry.get("path")
                if path:
                    return Path(str(path))
    return None


def verify_manifest_preflight(
    manifest: Mapping[str, Any],
    *,
    fals_root: str | Path | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    phase = str(manifest.get("phase") or "")
    if phase not in ALLOWED_PHASES:
        failures.append(f"phase_not_allowed:{phase}")
    storage_class = str(manifest.get("storage_class") or "")
    if not storage_class:
        failures.append("missing_storage_class")
    elif storage_class != REQUIRED_STORAGE_CLASS:
        failures.append(f"storage_class_mismatch:{storage_class}")
    exit_codes = dict(manifest.get("exit_codes") or {})
    for key in UPSTREAM_EXIT_KEYS:
        if int(exit_codes.get(key, -1)) != 0:
            failures.append(f"upstream_exit_nonzero:{key}")
    determinism_raw = manifest.get("determinism_gate")
    trace_hash = ""
    if not isinstance(determinism_raw, Mapping) or not determinism_raw:
        failures.append("missing_determinism_gate")
    else:
        determinism = dict(determinism_raw)
        if not bool(determinism.get("pass", False)):
            failures.append("determinism_gate_failed")
        trace_hash = str(
            determinism.get("observed_trace_hash") or determinism.get("trace_hash") or ""
        )
        if not trace_hash:
            failures.append("missing_trace_hash")
        elif trace_hash != EXPECTED_TRACE_HASH:
            failures.append("trace_hash_mismatch")
    own_phase_exit = exit_codes.get("two_tier_falsifier_battery_v0")
    prior_own_phase_classification: str | None = None
    if phase == "two_tier_falsifier_battery_v0" and own_phase_exit is not None:
        own_exit = int(own_phase_exit)
        if own_exit == 0:
            prior_own_phase_classification = "rerun_over_prior_success"
        elif own_exit == 127:
            prior_own_phase_classification = "launcher_failed_previous_attempt"
        elif own_exit != 0:
            failures.append("stop_for_review")
            prior_own_phase_classification = "stop_for_review"
    bound_paths: dict[str, str | None] = {}
    for role in MANIFEST_ROLE_ALIASES:
        path = _manifest_entry_path(manifest, role)
        bound_paths[role] = str(path) if path is not None else None
        if path is None and role in REQUIRED_MANIFEST_ROLES:
            failures.append(f"missing_manifest_role:{role}")
    return {
        "phase": phase,
        "allowed_phases": sorted(ALLOWED_PHASES),
        "upstream_exit_keys": list(UPSTREAM_EXIT_KEYS),
        "exit_codes": exit_codes,
        "trace_hash": trace_hash or None,
        "bound_paths": bound_paths,
        "fals_root": str(fals_root) if fals_root is not None else None,
        "prior_own_phase_classification": prior_own_phase_classification,
        "passed": not failures,
        "failure_reasons": failures,
    }


def verify_battery_input_integrity(
    *,
    stable_trace_path: Path,
    b2b_trace_path: Path,
    capture_receipt_path: Path,
    b2c_receipt_path: Path,
    audit_receipt_path: Path,
    acc_width_receipt_path: Path,
    expected_shas: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    paths = {
        "stable_copy_00": stable_trace_path,
        "b2b_trace": b2b_trace_path,
        "capture_receipt": capture_receipt_path,
        "b2c_receipt": b2c_receipt_path,
        "audit_receipt": audit_receipt_path,
        "acc_width_receipt": acc_width_receipt_path,
    }
    expected = dict(expected_shas or {})
    observed: dict[str, str | None] = {}
    failures: list[str] = []
    for key, path in paths.items():
        if not path.exists():
            failures.append(f"missing_input:{key}")
            observed[key] = None
            continue
        digest = _file_sha256(path)
        observed[key] = digest
        expected_sha = expected.get(key)
        if expected_sha is not None and digest != expected_sha:
            failures.append(f"sha_mismatch:{key}")
    sha_before = observed.get("stable_copy_00")
    sha_after = _file_sha256(stable_trace_path) if stable_trace_path.exists() else None
    diff_ec = 0 if sha_before == sha_after else 1
    if diff_ec != 0:
        failures.append("stable_trace_diff_ec_nonzero")
    return {
        "paths": {key: str(path) for key, path in paths.items()},
        "stable_copy_00_sha": observed.get("stable_copy_00"),
        "b2b_trace_sha": observed.get("b2b_trace"),
        "capture_receipt_sha": observed.get("capture_receipt"),
        "b2c_receipt_sha": observed.get("b2c_receipt"),
        "acc_width_receipt_sha": observed.get("acc_width_receipt"),
        "sha_before": sha_before,
        "sha_after": sha_after,
        "diff_ec": diff_ec,
        "sha256": observed,
        "expected_sha256": expected,
        "passed": not failures,
        "failure_reasons": failures,
    }


def run_falsifier_battery(
    steps: Sequence[Mapping[str, Any]],
    *,
    vote_spec: VoteSpecParsed,
    applied_candidate_ids_by_step: Mapping[int, str] | None = None,
) -> dict[str, Any]:
    applied_ids = dict(applied_candidate_ids_by_step or {})
    if not applied_ids:
        applied_ids = build_teacher_forced_applied_candidate_ids(steps)
    lane_maps = build_lane_maps(steps, vote_spec=vote_spec)
    f1_steps = [
        evaluate_f1_step(
            step,
            lane_maps=lane_maps,
            applied_candidate_ids_by_step=applied_ids,
        )
        for step in steps
    ]
    f2_steps = [evaluate_f2_step(step, lane_maps=lane_maps) for step in steps]
    f3_steps = [evaluate_f3_step(step, lane_maps=lane_maps) for step in steps]
    f1_summary = aggregate_f1(f1_steps)
    f2_summary = aggregate_f2(f2_steps)
    f3_summary = aggregate_f3(f3_steps)
    trace_policy_mismatch = any(
        item.get("skip_reason") == "trace_policy_mismatch" for item in f1_steps
    )
    vacuity_guard = build_estimand_vacuity_guard(
        f1_summary=f1_summary,
        f2_summary=f2_summary,
        f3_summary=f3_summary,
        f1_trace_policy_mismatch=trace_policy_mismatch,
    )
    classifier = classify_battery(
        f1_pass=bool(f1_summary["held_pass"]),
        f2_pass=bool(f2_summary["held_pass"]),
        f3_pass=bool(f3_summary["held_pass"]),
        vacuity_guard=vacuity_guard,
        harness_failures=[],
    )
    clip_min, clip_max = effective_clip_bounds(
        W_TEST,
        vote_spec.accumulator_clip_min,
        vote_spec.accumulator_clip_max,
    )
    return {
        "f1_cap_priority": f1_summary,
        "f2_rank_tau_b": f2_summary,
        "f3_tiebreak": f3_summary,
        "estimand_vacuity_guard": vacuity_guard,
        "classifier": classifier,
        "vote_spec": {
            "threshold_abs": vote_spec.threshold_abs,
            "threshold_source": CANONICAL_THRESHOLD_SOURCE,
            "clip_global": vote_spec.accumulator_clip_max,
            "decay": {
                "numerator": vote_spec.decay_numerator,
                "denominator": vote_spec.decay_denominator,
            },
            "w_ref": W_REF,
            "w_test": W_TEST,
            "w_test_effective_clip": clip_max,
        },
        "crossing_set_provenance": {
            "membership_lane": "w_ref",
            "w6_crossing_equivalence_cited_from": "acc_width_receipt_3e3157af",
            "w6_crossing_mismatches": 0,
            "rederived_in_battery": False,
        },
        "tau_b_formula": "knight_1966_tau_b_tie_correction",
    }


def build_two_tier_carry_falsifier_battery(
    *,
    stable_trace_path: str | Path,
    b2b_trace_path: str | Path,
    capture_receipt_path: str | Path,
    b2c_receipt_path: str | Path,
    audit_receipt_path: str | Path,
    acc_width_receipt_path: str | Path,
    chain_manifest_path: str | Path | None = None,
    fals_root: str | Path | None = None,
    expected_shas: Mapping[str, str] | None = None,
    acc_width_crossing_mismatches: int = 0,
    trace_hash: str | None = None,
) -> dict[str, Any]:
    harness_failures: list[str] = []
    manifest_payload: dict[str, Any] | None = None
    manifest_preflight: dict[str, Any] | None = None
    if chain_manifest_path is not None:
        manifest_path = Path(chain_manifest_path)
        if not manifest_path.exists():
            harness_failures.append("missing_input:chain_manifest")
        else:
            try:
                manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                harness_failures.append(
                    f"chain_manifest_parse_error:{type(exc).__name__}"
                )
            else:
                manifest_preflight = verify_manifest_preflight(
                    manifest_payload,
                    fals_root=fals_root,
                )
                if not manifest_preflight.get("passed", False):
                    harness_failures.extend(
                        list(manifest_preflight.get("failure_reasons") or [])
                    )

    integrity = verify_battery_input_integrity(
        stable_trace_path=Path(stable_trace_path),
        b2b_trace_path=Path(b2b_trace_path),
        capture_receipt_path=Path(capture_receipt_path),
        b2c_receipt_path=Path(b2c_receipt_path),
        audit_receipt_path=Path(audit_receipt_path),
        acc_width_receipt_path=Path(acc_width_receipt_path),
        expected_shas=expected_shas,
    )
    if not integrity.get("passed", False):
        harness_failures.extend(list(integrity.get("failure_reasons") or []))

    raw_steps: list[dict[str, Any]] = []
    vote_spec: VoteSpecParsed | None = None
    vote_spec_provenance: dict[str, Any] | None = None
    capture_payload: dict[str, Any] | None = None
    if integrity.get("passed", False):
        try:
            capture_payload = json.loads(
                Path(capture_receipt_path).read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            harness_failures.append(f"capture_receipt_parse_error:{type(exc).__name__}")
    if integrity.get("passed", False):
        raw_steps, load_failures = load_acc_width_trace_steps(Path(stable_trace_path))
        harness_failures.extend(load_failures)
    if capture_payload is not None and raw_steps:
        vote_spec, vote_spec_provenance, spec_failures = resolve_vote_spec(
            capture_payload,
            raw_steps,
            manifest_payload=manifest_payload,
        )
        harness_failures.extend(spec_failures)

    field_inventory = (
        build_required_field_inventory(raw_steps)
        if raw_steps
        else {
            "required_fields": list(REQUIRED_TRACE_ROW_FIELDS),
            "present_fields": [],
            "missing_fields": list(REQUIRED_TRACE_ROW_FIELDS),
            "row_count": 0,
            "passed": False,
        }
    )
    if not field_inventory.get("passed", False):
        harness_failures.append("field_inventory_gate_fail")

    battery_core: dict[str, Any] | None = None
    if raw_steps and vote_spec is not None and field_inventory.get("passed", False):
        battery_core = run_falsifier_battery(raw_steps, vote_spec=vote_spec)
        if harness_failures:
            battery_core["classifier"] = classify_battery(
                f1_pass=bool(battery_core["f1_cap_priority"]["held_pass"]),
                f2_pass=bool(battery_core["f2_rank_tau_b"]["held_pass"]),
                f3_pass=bool(battery_core["f3_tiebreak"]["held_pass"]),
                vacuity_guard=battery_core["estimand_vacuity_guard"],
                harness_failures=harness_failures,
            )
    elif harness_failures:
        battery_core = {
            "f1_cap_priority": aggregate_f1([]),
            "f2_rank_tau_b": aggregate_f2([]),
            "f3_tiebreak": aggregate_f3([]),
            "estimand_vacuity_guard": build_estimand_vacuity_guard(
                f1_summary=aggregate_f1([]),
                f2_summary=aggregate_f2([]),
                f3_summary=aggregate_f3([]),
                f1_trace_policy_mismatch=False,
            ),
            "classifier": classify_battery(
                f1_pass=False,
                f2_pass=False,
                f3_pass=False,
                vacuity_guard={},
                harness_failures=harness_failures,
            ),
            "vote_spec": None,
            "crossing_set_provenance": {
                "membership_lane": "w_ref",
                "w6_crossing_equivalence_cited_from": "acc_width_receipt_3e3157af",
                "w6_crossing_mismatches": int(acc_width_crossing_mismatches),
                "rederived_in_battery": False,
            },
            "tau_b_formula": "knight_1966_tau_b_tie_correction",
        }

    resolved_trace_hash = trace_hash
    if resolved_trace_hash is None and manifest_preflight is not None:
        resolved_trace_hash = manifest_preflight.get("trace_hash")
    if resolved_trace_hash is None and raw_steps:
        resolved_trace_hash = _stable_hash16(
            [int(step["optimizer_step_index"]) for step in raw_steps]
        )

    primary_label = (
        battery_core["classifier"]["primary_label"]
        if battery_core is not None
        else LABEL_SCREEN_HARNESS_OR_GATE_FAIL
    )
    receipt: dict[str, Any] = {
        "schema_version": BATTERY_SCHEMA_VERSION,
        "contract_id": BATTERY_CONTRACT_ID,
        "receipt_kind": BATTERY_RECEIPT_KIND,
        "compact_receipt": True,
        "trace_hash": resolved_trace_hash,
        "held_split": {"start": HELD_STEP_START, "end": HELD_STEP_END, "inclusive": True},
        "width_lanes": {"w_ref": W_REF, "w_test": W_TEST},
        "threshold_abs": CANONICAL_VOTE_UPDATE_THRESHOLD_ABS,
        "input_integrity": integrity,
        "manifest_preflight": manifest_preflight,
        "field_inventory_gate": field_inventory,
        "vote_spec_provenance": vote_spec_provenance,
        "banked_f4": {
            "audit_receipt_sha": BANKED_F4_AUDIT_RECEIPT_SHA,
            "primary_label": BANKED_F4_PRIMARY_LABEL,
            "remeasured": False,
        },
        "claim_boundary": {
            "class": PRE_FULL_STACK_DIAGNOSTIC_ONLY,
            "measurement_only": True,
            "single_trace": True,
            "trace_hash": resolved_trace_hash,
            "no_build_claim": True,
            "no_runtime_claim": True,
            "no_training_claim": True,
        },
        "primary_label": primary_label,
        "failure_reasons": harness_failures,
        "seam_debt": {
            "acc_width_imports": [
                "decay_vote_clamp",
                "crosses_threshold",
                "effective_clip_bounds",
                "resolve_vote_spec",
                "build_required_field_inventory",
                "REQUIRED_TRACE_ROW_FIELDS",
                "CANONICAL_THRESHOLD_SOURCE",
            ],
            "screen_module_touched": False,
            "audit_module_touched": False,
        },
    }
    if battery_core is not None:
        receipt.update(battery_core)
        if battery_core.get("crossing_set_provenance") is not None:
            receipt["crossing_set_provenance"]["w6_crossing_mismatches"] = int(
                acc_width_crossing_mismatches
            )
    return receipt
