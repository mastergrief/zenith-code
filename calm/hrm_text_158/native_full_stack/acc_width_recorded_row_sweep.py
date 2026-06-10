"""CPU read-only accumulator width sweep over B2b recorded rows."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from calm.hrm_text_158.native_full_stack.accumulator_policy_shadow_screen import (
    PRE_FULL_STACK_DIAGNOSTIC_ONLY,
    _file_sha256,
    _stable_hash16,
)
from calm.hrm_text_158.native_full_stack.accumulator_policy_shadow_screen import (
    _current_repo_readiness_summary,
)
from calm.hrm_text_158.native_full_stack.accumulator_policy_shadow_screen import (
    B2B_SEQUENTIAL_TRACE_SCHEMA,
    FAIL_MIXED_SOURCE_KIND,
    FAIL_MISSING_OPTIMIZER_STEP_INDEX,
    FAIL_NO_REAL_CANDIDATE_TABLE,
    FAIL_NON_MONOTONIC_STEP_INDEX,
)
from calm.hrm_text_158.native_full_stack.transient_selection_information_audit import (
    reconstruct_transient_target,
)

ACC_WIDTH_RECORDED_ROW_SWEEP_SCHEMA_VERSION = (
    "hrm_text_158_acc_width_recorded_row_sweep/v0"
)
ACC_WIDTH_RECORDED_ROW_SWEEP_CONTRACT_ID = (
    "teacher_forced_recorded_row_acc_width_invariance_v0"
)
ACC_WIDTH_RECORDED_ROW_SWEEP_RECEIPT_KIND = (
    "cpu_read_only_acc_width_recorded_row_sweep"
)

SCOPE_STATEMENT = (
    "teacher_forced_recorded_row_acc_width_invariance: single-trace; "
    "recorded rows only (sampled in-band candidates plus applied-flip replay); "
    "tensor-wide band-entry deferred to second-capture gate"
)

DEFAULT_WIDTH_GRID = (16, 12, 10, 8, 6, 4, 3, 2)
DEFAULT_HEADROOM_FACTOR = 2.0
TRACE_FAMILY_SOURCE_CLIP_MIN = -127
TRACE_FAMILY_SOURCE_CLIP_MAX = 127

REQUIRED_TRACE_ROW_FIELDS = (
    "pre_accumulator_i16",
    "new_acc_i32_signed",
    "vote_value",
    "proposal_direction",
    "current_q_level",
    "in_target_tie_band",
    "flat_index",
    "threshold_residual_signed",
    "proximity_to_threshold",
)

LABEL_ACC_SHRINK_AGGRESSIVE = "acc_shrink_aggressive"
LABEL_ACC_SHRINK_TWO_TIER = "acc_shrink_two_tier"
LABEL_ACC_SHRINK_PARTIAL = "acc_shrink_partial"
LABEL_ACC_NOT_SHRINKABLE = "acc_not_shrinkable"
LABEL_SCREEN_HARNESS_OR_GATE_FAIL = "screen_harness_or_gate_fail"

CLASSIFIER_PRECEDENCE = (
    LABEL_SCREEN_HARNESS_OR_GATE_FAIL,
    LABEL_ACC_SHRINK_AGGRESSIVE,
    LABEL_ACC_SHRINK_TWO_TIER,
    LABEL_ACC_SHRINK_PARTIAL,
    LABEL_ACC_NOT_SHRINKABLE,
)


@dataclass(frozen=True)
class VoteSpecParsed:
    threshold_abs: int
    decay_numerator: int
    decay_denominator: int
    accumulator_clip_min: int
    accumulator_clip_max: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold_abs": int(self.threshold_abs),
            "decay": {
                "numerator": int(self.decay_numerator),
                "denominator": int(self.decay_denominator),
            },
            "accumulator_clip_min": int(self.accumulator_clip_min),
            "accumulator_clip_max": int(self.accumulator_clip_max),
        }


def signed_w_max(width: int) -> int:
    if width < 2:
        raise ValueError(f"width must be >= 2, got {width}")
    return (1 << (int(width) - 1)) - 1


def effective_clip_bounds(
    width: int,
    source_clip_min: int,
    source_clip_max: int,
) -> tuple[int, int]:
    """effective_clip(W)=±min(source_clip_max, 2^(W-1)-1) composed with source clip."""

    w_max = signed_w_max(width)
    w_min = -w_max
    eff_max = min(int(source_clip_max), w_max)
    eff_min = max(int(source_clip_min), w_min)
    eff_max = min(eff_max, w_max)
    eff_min = max(eff_min, w_min)
    return eff_min, eff_max


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(int(lo), min(int(hi), int(value)))


def _decay_accumulator(
    acc: int,
    *,
    decay_numerator: int,
    decay_denominator: int,
) -> int:
    if decay_denominator <= 0:
        raise ValueError("decay_denominator must be > 0")
    return (int(acc) * int(decay_numerator)) // int(decay_denominator)


def decay_vote_clamp(
    pre_accumulator: int,
    vote_value: int,
    *,
    clip_min: int,
    clip_max: int,
    decay_numerator: int,
    decay_denominator: int,
) -> int:
    decayed = _decay_accumulator(
        pre_accumulator,
        decay_numerator=decay_numerator,
        decay_denominator=decay_denominator,
    )
    return _clamp(decayed + int(vote_value), clip_min, clip_max)


def post_flip_residual_clamp(
    new_acc: int,
    *,
    proposal_direction: int,
    threshold_abs: int,
) -> int:
    direction = 1 if int(proposal_direction) >= 0 else -1
    residual = int(new_acc) - direction * int(threshold_abs)
    lo = -int(threshold_abs) + 1
    hi = int(threshold_abs) - 1
    return _clamp(residual, lo, hi)


def crosses_threshold(
    new_acc: int,
    *,
    current_q_level: int,
    threshold_abs: int,
) -> bool:
    q = int(current_q_level)
    acc = int(new_acc)
    threshold = int(threshold_abs)
    return (acc >= threshold and q < 1) or (acc <= -threshold and q > -1)


def vote_update_source_constants_at_pinned_head() -> dict[str, int]:
    """Canonical vote_update clip/decay constants for the pinned B2b trace family."""

    return {
        "accumulator_clip_min": TRACE_FAMILY_SOURCE_CLIP_MIN,
        "accumulator_clip_max": TRACE_FAMILY_SOURCE_CLIP_MAX,
        "decay_numerator": 1,
        "decay_denominator": 1,
    }


def _iter_capture_vote_spec_blocks(
    payload: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    candidates: list[Mapping[str, Any]] = []
    for key in ("vote_update_spec", "default_vote_update_spec"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            candidates.append(value)
    updater_config = payload.get("updater_config")
    if isinstance(updater_config, Mapping):
        nested = updater_config.get("vote_update_spec")
        if isinstance(nested, Mapping):
            candidates.append(nested)
    probe_config = payload.get("probe_config")
    if isinstance(probe_config, Mapping):
        nested = probe_config.get("vote_update_spec")
        if isinstance(nested, Mapping):
            candidates.append(nested)
    return candidates


def _vote_spec_from_spec_block(spec: Mapping[str, Any]) -> VoteSpecParsed | None:
    if "threshold_abs" not in spec:
        return None
    decay = spec.get("decay") or {}
    if isinstance(decay, Mapping) and "numerator" in decay:
        decay_num = int(decay["numerator"])
        decay_den = int(decay["denominator"])
    else:
        decay_num = int(spec.get("decay_numerator", 1))
        decay_den = int(spec.get("decay_denominator", 1))
    clip_min = spec.get("accumulator_clip_min")
    clip_max = spec.get("accumulator_clip_max")
    if clip_min is None or clip_max is None:
        acc = spec.get("accumulator")
        if isinstance(acc, Mapping):
            clip_min = acc.get("clip_min", clip_min)
            clip_max = acc.get("clip_max", clip_max)
    if clip_min is None or clip_max is None:
        return None
    return VoteSpecParsed(
        threshold_abs=int(spec["threshold_abs"]),
        decay_numerator=decay_num,
        decay_denominator=decay_den,
        accumulator_clip_min=int(clip_min),
        accumulator_clip_max=int(clip_max),
    )


def try_parse_vote_spec_from_capture_receipt(
    payload: Mapping[str, Any],
) -> VoteSpecParsed | None:
    for spec in _iter_capture_vote_spec_blocks(payload):
        parsed = _vote_spec_from_spec_block(spec)
        if parsed is not None:
            return parsed
    return None


def parse_vote_spec_from_capture_receipt(payload: Mapping[str, Any]) -> VoteSpecParsed:
    parsed = try_parse_vote_spec_from_capture_receipt(payload)
    if parsed is None:
        raise ValueError("vote_update_spec not found in capture receipt")
    return parsed


def derive_threshold_abs_from_recorded_rows(
    steps: Sequence[Mapping[str, Any]],
) -> tuple[int | None, dict[str, Any]]:
    derived: list[int] = []
    relation_mismatches: list[dict[str, Any]] = []
    for step in steps:
        for row in step.get("sampled_candidate_table") or ():
            if not isinstance(row, Mapping):
                continue
            new_acc = int(row["new_acc_i32_signed"])
            residual = int(row["threshold_residual_signed"])
            direction = int(row.get("proposal_direction", 1 if new_acc >= 0 else -1))
            if direction == 0:
                direction = 1 if new_acc >= 0 else -1
            threshold_abs = direction * (new_acc - residual)
            if threshold_abs <= 0:
                relation_mismatches.append(
                    {
                        "optimizer_step_index": int(step["optimizer_step_index"]),
                        "flat_index": int(row["flat_index"]),
                        "reason": "non_positive_threshold",
                        "derived_threshold_abs": threshold_abs,
                    }
                )
                continue
            proximity = int(row["proximity_to_threshold"])
            expected_proximity = abs(abs(new_acc) - threshold_abs)
            if proximity != expected_proximity:
                relation_mismatches.append(
                    {
                        "optimizer_step_index": int(step["optimizer_step_index"]),
                        "flat_index": int(row["flat_index"]),
                        "reason": "proximity_mismatch",
                        "observed_proximity": proximity,
                        "expected_proximity": expected_proximity,
                    }
                )
                continue
            derived.append(threshold_abs)
    unique = sorted(set(derived))
    if relation_mismatches:
        return None, {
            "failure": "row_threshold_relation_mismatch",
            "mismatches": relation_mismatches[:20],
            "row_count": len(derived),
        }
    if not unique:
        return None, {"failure": "no_threshold_samples"}
    if len(unique) != 1:
        return None, {
            "failure": "threshold_inconsistent_across_rows",
            "candidate_thresholds": unique,
            "row_count": len(derived),
        }
    return unique[0], {
        "threshold_source": "recorded_row_residual_proximity_relation",
        "threshold_abs": unique[0],
        "row_count": len(derived),
    }


def parse_manifest_parameters(
    manifest_payload: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if manifest_payload is None:
        return {}
    parameters = manifest_payload.get("parameters")
    if isinstance(parameters, Mapping):
        return dict(parameters)
    return {}


def compose_vote_spec_from_production_sources(
    steps: Sequence[Mapping[str, Any]],
    *,
    manifest_payload: Mapping[str, Any] | None = None,
) -> tuple[VoteSpecParsed | None, dict[str, Any], list[str]]:
    failures: list[str] = []
    provenance: dict[str, Any] = {"parse_path": "composed_production_fallback"}
    source_constants = vote_update_source_constants_at_pinned_head()
    provenance["clip_source"] = "vote_update_source_at_pinned_head"
    provenance["decay_source"] = "vote_update_source_at_pinned_head"
    provenance.update(source_constants)

    threshold_abs, threshold_provenance = derive_threshold_abs_from_recorded_rows(steps)
    provenance["threshold"] = threshold_provenance
    if threshold_abs is None:
        failures.append("threshold_derivation_fail")
        return None, provenance, failures

    manifest_parameters = parse_manifest_parameters(manifest_payload)
    if "max_abs_per_tensor" in manifest_parameters:
        provenance["max_abs_per_tensor"] = int(manifest_parameters["max_abs_per_tensor"])
        provenance["max_abs_source"] = "manifest_parameters"

    return (
        VoteSpecParsed(
            threshold_abs=int(threshold_abs),
            decay_numerator=int(source_constants["decay_numerator"]),
            decay_denominator=int(source_constants["decay_denominator"]),
            accumulator_clip_min=int(source_constants["accumulator_clip_min"]),
            accumulator_clip_max=int(source_constants["accumulator_clip_max"]),
        ),
        provenance,
        failures,
    )


def resolve_vote_spec(
    capture_payload: Mapping[str, Any],
    steps: Sequence[Mapping[str, Any]],
    *,
    manifest_payload: Mapping[str, Any] | None = None,
) -> tuple[VoteSpecParsed | None, dict[str, Any], list[str]]:
    parsed = try_parse_vote_spec_from_capture_receipt(capture_payload)
    if parsed is not None:
        return parsed, {"parse_path": "capture_receipt_spec_block"}, []

    return compose_vote_spec_from_production_sources(
        steps,
        manifest_payload=manifest_payload,
    )


def parse_observed_trace_family_clip_from_capture_receipt(
    payload: Mapping[str, Any],
) -> tuple[int, int] | None:
    """Parse observed trace-family clip bounds from capture receipt, if present."""

    candidate_keys = (
        "observed_trace_family_clip",
        "trace_family_clip",
        "observed_clip_bounds",
        "observed_clip",
    )
    for key in candidate_keys:
        value = payload.get(key)
        if isinstance(value, Sequence) and len(value) == 2:
            return int(value[0]), int(value[1])
        if isinstance(value, Mapping):
            clip_min = value.get("min", value.get("clip_min"))
            clip_max = value.get("max", value.get("clip_max"))
            if clip_min is not None and clip_max is not None:
                return int(clip_min), int(clip_max)
    saturation = payload.get("saturation_observed")
    if isinstance(saturation, Mapping):
        clip = saturation.get("clip") or saturation.get("clip_bounds")
        if isinstance(clip, Sequence) and len(clip) == 2:
            return int(clip[0]), int(clip[1])
    return None


def infer_observed_clip_bounds_from_recorded_rows(
    steps: Sequence[Mapping[str, Any]],
) -> tuple[int, int] | None:
    """Infer tight observed clip bounds from recorded row accumulator fields."""

    values: list[int] = []
    for step in steps:
        for row in step.get("sampled_candidate_table") or ():
            if not isinstance(row, Mapping):
                continue
            for field in ("pre_accumulator_i16", "new_acc_i32_signed"):
                if field in row and row.get(field) is not None:
                    values.append(int(row[field]))
    if not values:
        return None
    return min(values), max(values)


def assert_clip_bound_proof(
    vote_spec: VoteSpecParsed,
    *,
    recorded_row_bounds: tuple[int, int] | None,
) -> dict[str, Any]:
    expected = (TRACE_FAMILY_SOURCE_CLIP_MIN, TRACE_FAMILY_SOURCE_CLIP_MAX)
    declared = (vote_spec.accumulator_clip_min, vote_spec.accumulator_clip_max)
    declared_passed = declared == expected
    result: dict[str, Any] = {
        "declared_clip_min": vote_spec.accumulator_clip_min,
        "declared_clip_max": vote_spec.accumulator_clip_max,
        "expected_trace_family_clip_min": TRACE_FAMILY_SOURCE_CLIP_MIN,
        "expected_trace_family_clip_max": TRACE_FAMILY_SOURCE_CLIP_MAX,
        "declared_passed": declared_passed,
        "recorded_row_min": None,
        "recorded_row_max": None,
        "row_extrema_within_declared": None,
        "observed_clip_source": "recorded_row_extrema",
        "passed": declared_passed,
    }
    if recorded_row_bounds is not None:
        row_min, row_max = recorded_row_bounds
        within_declared = row_min >= declared[0] and row_max <= declared[1]
        result.update(
            {
                "recorded_row_min": int(row_min),
                "recorded_row_max": int(row_max),
                "row_extrema_within_declared": within_declared,
                "passed": declared_passed and within_declared,
            }
        )
    return result


def assert_trace_family_source_clip_is_pm127(
    vote_spec: VoteSpecParsed,
    *,
    observed_trace_family_clip: tuple[int, int] | None = None,
) -> dict[str, Any]:
    recorded_row_bounds = observed_trace_family_clip
    return assert_clip_bound_proof(
        vote_spec,
        recorded_row_bounds=recorded_row_bounds,
    )


def assert_observed_clip_matches_declared(
    vote_spec: VoteSpecParsed,
    *,
    observed_clip_min: int,
    observed_clip_max: int,
) -> dict[str, Any]:
    declared = [vote_spec.accumulator_clip_min, vote_spec.accumulator_clip_max]
    observed = [int(observed_clip_min), int(observed_clip_max)]
    passed = declared == observed
    return {
        "declared_clip_min": vote_spec.accumulator_clip_min,
        "declared_clip_max": vote_spec.accumulator_clip_max,
        "observed_clip_min": int(observed_clip_min),
        "observed_clip_max": int(observed_clip_max),
        "passed": passed,
    }


def load_acc_width_trace_steps(
    trace_path: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Load B2b trace steps preserving full sampled_candidate_table rows."""

    failure_reasons: list[str] = []
    steps: list[dict[str, Any]] = []
    expected_source_kind: str | None = None
    try:
        lines = trace_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return [], [f"trace_load_error:{type(exc).__name__}"]

    for line_index, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            failure_reasons.append(f"trace_json_error:line_{line_index}")
            continue
        if not isinstance(record, Mapping):
            failure_reasons.append(f"trace_shape_error:line_{line_index}")
            continue
        if record.get("schema") == B2B_SEQUENTIAL_TRACE_SCHEMA:
            continue

        source_kind = str(record.get("source_kind") or "")
        if not source_kind:
            failure_reasons.append(FAIL_MIXED_SOURCE_KIND)
            continue
        if expected_source_kind is None:
            expected_source_kind = source_kind
        elif source_kind != expected_source_kind:
            failure_reasons.append(FAIL_MIXED_SOURCE_KIND)
            continue

        if "optimizer_step_index" not in record:
            failure_reasons.append(FAIL_MISSING_OPTIMIZER_STEP_INDEX)
            continue
        optimizer_step_index = int(record["optimizer_step_index"])
        if optimizer_step_index <= 0:
            failure_reasons.append(FAIL_MISSING_OPTIMIZER_STEP_INDEX)
            continue

        rows = list(record.get("sampled_candidate_table") or record.get("candidates") or ())
        if not rows:
            failure_reasons.append(FAIL_NO_REAL_CANDIDATE_TABLE)
            continue

        steps.append(
            {
                "optimizer_step_index": optimizer_step_index,
                "source_kind": source_kind,
                "source_table_hash": record.get("source_table_hash"),
                "pre_update_state_hash": record.get("pre_update_state_hash"),
                "sampled_candidate_table": [dict(row) for row in rows],
                "post_update_telemetry": dict(record.get("post_update_telemetry") or {}),
            }
        )

    steps.sort(key=lambda step: int(step["optimizer_step_index"]))
    if len(steps) >= 2:
        indices = [int(step["optimizer_step_index"]) for step in steps]
        if indices != sorted(indices) or len(set(indices)) != len(indices):
            failure_reasons.append(FAIL_NON_MONOTONIC_STEP_INDEX)
            return [], list(dict.fromkeys(failure_reasons))
    return steps, list(dict.fromkeys(failure_reasons))


def build_required_field_inventory(steps: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    missing_by_field: dict[str, int] = {field: 0 for field in REQUIRED_TRACE_ROW_FIELDS}
    present_fields: set[str] = set()
    row_count = 0
    for step in steps:
        for row in step.get("sampled_candidate_table") or ():
            if not isinstance(row, Mapping):
                continue
            row_count += 1
            for field in REQUIRED_TRACE_ROW_FIELDS:
                if field not in row or row.get(field) is None:
                    missing_by_field[field] += 1
                else:
                    present_fields.add(field)
    missing_fields = tuple(
        field for field in REQUIRED_TRACE_ROW_FIELDS if missing_by_field[field] > 0
    )
    return {
        "required_fields": list(REQUIRED_TRACE_ROW_FIELDS),
        "present_fields": sorted(present_fields),
        "missing_fields": list(missing_fields),
        "row_count": row_count,
        "passed": not missing_fields,
    }


def _row_key(step_index: int, flat_index: int) -> tuple[int, int]:
    return int(step_index), int(flat_index)


def _teacher_forced_applied_flat_index(
    step: Mapping[str, Any],
    *,
    applied_candidate_id: str | None,
) -> int | None:
    if not applied_candidate_id:
        return None
    for row in step.get("sampled_candidate_table") or ():
        if not isinstance(row, Mapping):
            continue
        if str(row.get("candidate_id")) == str(applied_candidate_id):
            return int(row["flat_index"])
    return None


def replay_width_lane(
    steps: Sequence[Mapping[str, Any]],
    *,
    vote_spec: VoteSpecParsed,
    width: int,
    applied_candidate_ids_by_step: Mapping[int, str],
) -> dict[str, Any]:
    clip_min, clip_max = effective_clip_bounds(
        width,
        vote_spec.accumulator_clip_min,
        vote_spec.accumulator_clip_max,
    )
    carry_by_flat: dict[int, int] = {}
    row_crossings: dict[tuple[int, int], bool] = {}
    recorded_band_membership_echo: dict[tuple[int, int], bool] = {}
    row_recomputed_new_acc: dict[tuple[int, int], int] = {}
    applied_post_flip_abs: list[int] = []
    bit_identical_to_recorded = True
    mismatched_rows: list[dict[str, Any]] = []

    for step in steps:
        step_index = int(step["optimizer_step_index"])
        telemetry = dict(step.get("post_update_telemetry") or {})
        q_changed = int(telemetry.get("q_changed_count", 0)) > 0
        applied_candidate_id = applied_candidate_ids_by_step.get(step_index)
        applied_flat = _teacher_forced_applied_flat_index(
            step,
            applied_candidate_id=applied_candidate_id,
        )

        for row in step.get("sampled_candidate_table") or ():
            if not isinstance(row, Mapping):
                continue
            flat_index = int(row["flat_index"])
            key = _row_key(step_index, flat_index)
            pre_acc = int(row["pre_accumulator_i16"])
            vote_value = int(row["vote_value"])
            new_acc = decay_vote_clamp(
                pre_acc,
                vote_value,
                clip_min=clip_min,
                clip_max=clip_max,
                decay_numerator=vote_spec.decay_numerator,
                decay_denominator=vote_spec.decay_denominator,
            )
            row_recomputed_new_acc[key] = new_acc
            row_crossings[key] = crosses_threshold(
                new_acc,
                current_q_level=int(row["current_q_level"]),
                threshold_abs=vote_spec.threshold_abs,
            )
            recorded_band_membership_echo[key] = bool(row.get("in_target_tie_band"))

            recorded_new_acc = int(row["new_acc_i32_signed"])
            if width == 16 and new_acc != recorded_new_acc:
                bit_identical_to_recorded = False
                mismatched_rows.append(
                    {
                        "optimizer_step_index": step_index,
                        "flat_index": flat_index,
                        "recorded_new_acc_i32_signed": recorded_new_acc,
                        "recomputed_new_acc_i32_signed": new_acc,
                    }
                )

        if q_changed and applied_flat is not None:
            applied_row = next(
                (
                    row
                    for row in step.get("sampled_candidate_table") or ()
                    if isinstance(row, Mapping) and int(row["flat_index"]) == applied_flat
                ),
                None,
            )
            if applied_row is not None:
                pre_acc = carry_by_flat.get(
                    applied_flat,
                    int(applied_row["pre_accumulator_i16"]),
                )
                new_acc = decay_vote_clamp(
                    pre_acc,
                    int(applied_row["vote_value"]),
                    clip_min=clip_min,
                    clip_max=clip_max,
                    decay_numerator=vote_spec.decay_numerator,
                    decay_denominator=vote_spec.decay_denominator,
                )
                post_acc = post_flip_residual_clamp(
                    new_acc,
                    proposal_direction=int(applied_row["proposal_direction"]),
                    threshold_abs=vote_spec.threshold_abs,
                )
                carry_by_flat[applied_flat] = post_acc
                applied_post_flip_abs.append(abs(int(post_acc)))

    return {
        "width": int(width),
        "effective_clip_min": clip_min,
        "effective_clip_max": clip_max,
        "row_crossings": row_crossings,
        "recorded_band_membership_echo": recorded_band_membership_echo,
        "row_recomputed_new_acc": row_recomputed_new_acc,
        "applied_post_flip_abs_values": applied_post_flip_abs,
        "max_abs_acc_applied_flips": (
            max(applied_post_flip_abs) if applied_post_flip_abs else 0
        ),
        "bit_identical_to_recorded_new_acc": bit_identical_to_recorded,
        "w16_mismatch_rows": mismatched_rows,
    }


def coarse_invariant_vs_reference(
    lane: Mapping[str, Any],
    *,
    reference_lane: Mapping[str, Any],
) -> dict[str, Any]:
    ref_cross = dict(reference_lane.get("row_crossings") or {})
    lane_cross = dict(lane.get("row_crossings") or {})
    all_keys = sorted(set(ref_cross) | set(lane_cross))
    mismatches: list[dict[str, Any]] = []
    for key in all_keys:
        crossing_match = ref_cross.get(key) == lane_cross.get(key)
        if not crossing_match:
            mismatches.append(
                {
                    "optimizer_step_index": key[0],
                    "flat_index": key[1],
                    "reference_crossing": ref_cross.get(key),
                    "lane_crossing": lane_cross.get(key),
                }
            )
    return {
        "coarse_crossing_invariant": not mismatches,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:20],
        "band_membership_scope": "recorded_echo_not_width_selector",
    }


def headroom_passes(
    width: int,
    *,
    max_abs_acc_applied: int,
    headroom_factor: float,
) -> bool:
    limit = signed_w_max(width)
    return float(max_abs_acc_applied) * float(headroom_factor) <= float(limit)


def compute_w_min_invariant(
    width_grid: Sequence[int],
    *,
    lane_by_width: Mapping[int, Mapping[str, Any]],
    reference_width: int = 16,
) -> int | None:
    reference_lane = lane_by_width.get(reference_width)
    if reference_lane is None:
        return None
    if not reference_lane.get("bit_identical_to_recorded_new_acc", False):
        return None
    ordered = sorted(int(width) for width in width_grid)
    for width in ordered:
        lane = lane_by_width[width]
        inv = coarse_invariant_vs_reference(lane, reference_lane=reference_lane)
        if inv["coarse_crossing_invariant"]:
            return int(width)
    return None


def compute_w_min_headroom_safe(
    width_grid: Sequence[int],
    *,
    lane_by_width: Mapping[int, Mapping[str, Any]],
    reference_width: int = 16,
    headroom_factor: float = DEFAULT_HEADROOM_FACTOR,
) -> int | None:
    reference_lane = lane_by_width.get(reference_width)
    if reference_lane is None:
        return None
    if not reference_lane.get("bit_identical_to_recorded_new_acc", False):
        return None
    ordered = sorted(int(width) for width in width_grid)
    for width in ordered:
        lane = lane_by_width[width]
        inv = coarse_invariant_vs_reference(lane, reference_lane=reference_lane)
        if not inv["coarse_crossing_invariant"]:
            continue
        max_abs = int(lane.get("max_abs_acc_applied_flips", 0))
        if headroom_passes(
            int(width),
            max_abs_acc_applied=max_abs,
            headroom_factor=headroom_factor,
        ):
            return int(width)
    return None


def compute_w_min(
    width_grid: Sequence[int],
    *,
    lane_by_width: Mapping[int, Mapping[str, Any]],
    reference_width: int = 16,
) -> int | None:
    return compute_w_min_invariant(
        width_grid,
        lane_by_width=lane_by_width,
        reference_width=reference_width,
    )


def classify_w_min_label(
    w_min: int | None,
    *,
    harness_failures: Sequence[str],
    headroom_pass: bool,
    reference_width: int = 16,
) -> dict[str, Any]:
    failures = list(dict.fromkeys(harness_failures))
    if failures or w_min is None:
        return {
            "primary_label": LABEL_SCREEN_HARNESS_OR_GATE_FAIL,
            "sub_reasons": failures or ["w_min_undefined"],
            "branch_precedence": CLASSIFIER_PRECEDENCE,
            "failure_reasons": failures,
        }
    if w_min in {2, 3, 4} and headroom_pass:
        label = LABEL_ACC_SHRINK_AGGRESSIVE
    elif w_min in {6, 8} and headroom_pass:
        label = LABEL_ACC_SHRINK_TWO_TIER
    elif w_min in {10, 12} and headroom_pass:
        label = LABEL_ACC_SHRINK_PARTIAL
    elif w_min == reference_width:
        label = LABEL_ACC_NOT_SHRINKABLE
    else:
        return {
            "primary_label": LABEL_SCREEN_HARNESS_OR_GATE_FAIL,
            "sub_reasons": [f"w_min_out_of_disjoint_ranges:{w_min}"],
            "branch_precedence": CLASSIFIER_PRECEDENCE,
            "failure_reasons": failures
            + [f"w_min_out_of_disjoint_ranges:{w_min}"],
        }
    return {
        "primary_label": label,
        "sub_reasons": [f"w_min={w_min}"],
        "branch_precedence": CLASSIFIER_PRECEDENCE,
        "failure_reasons": failures,
    }


def verify_acc_width_input_integrity(
    *,
    stable_trace_path: Path,
    capture_receipt_path: Path,
    b2c_receipt_path: Path,
    audit_receipt_path: Path,
    expected_shas: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    expected = dict(expected_shas or {})
    paths = {
        "stable_trace": stable_trace_path,
        "capture_receipt": capture_receipt_path,
        "b2c_receipt": b2c_receipt_path,
        "audit_receipt": audit_receipt_path,
    }
    observed_shas: dict[str, str | None] = {}
    failure_reasons: list[str] = []
    for key, path in paths.items():
        if not path.exists():
            failure_reasons.append(f"missing_input:{key}")
            observed_shas[key] = None
            continue
        observed = _file_sha256(path)
        observed_shas[key] = observed
        expected_sha = expected.get(key)
        if expected_sha is not None and observed != expected_sha:
            failure_reasons.append(f"sha_mismatch:{key}")

    return {
        "paths": {key: str(path) for key, path in paths.items()},
        "sha256": observed_shas,
        "expected_sha256": dict(expected),
        "passed": not failure_reasons,
        "failure_reasons": failure_reasons,
    }


def build_teacher_forced_applied_candidate_ids(
    steps: Sequence[Mapping[str, Any]],
) -> dict[int, str]:
    selected_by_step, _ = reconstruct_transient_target(steps, rate_cap=1)
    mapping: dict[int, str] = {}
    for step, selected in zip(steps, selected_by_step, strict=False):
        if not selected:
            continue
        mapping[int(step["optimizer_step_index"])] = str(selected[0])
    return mapping


def build_acc_width_recorded_row_sweep(
    *,
    stable_trace_path: str | Path,
    capture_receipt_path: str | Path,
    b2c_receipt_path: str | Path,
    audit_receipt_path: str | Path,
    expected_shas: Mapping[str, str] | None = None,
    chain_manifest_path: str | Path | None = None,
    width_grid: Sequence[int] | None = None,
    headroom_factor: float = DEFAULT_HEADROOM_FACTOR,
) -> dict[str, Any]:
    widths = tuple(int(width) for width in (width_grid or DEFAULT_WIDTH_GRID))
    integrity = verify_acc_width_input_integrity(
        stable_trace_path=Path(stable_trace_path),
        capture_receipt_path=Path(capture_receipt_path),
        b2c_receipt_path=Path(b2c_receipt_path),
        audit_receipt_path=Path(audit_receipt_path),
        expected_shas=expected_shas,
    )

    harness_failures = list(integrity.get("failure_reasons") or [])
    raw_steps: list[dict[str, Any]] = []
    vote_spec: VoteSpecParsed | None = None
    vote_spec_provenance: dict[str, Any] | None = None
    clip_assertion: dict[str, Any] | None = None
    capture_payload: dict[str, Any] | None = None
    manifest_payload: dict[str, Any] | None = None

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

    recorded_row_value_bounds: tuple[int, int] | None = None
    if raw_steps:
        recorded_row_value_bounds = infer_observed_clip_bounds_from_recorded_rows(raw_steps)

    if capture_payload is not None and raw_steps:
        vote_spec, vote_spec_provenance, spec_failures = resolve_vote_spec(
            capture_payload,
            raw_steps,
            manifest_payload=manifest_payload,
        )
        harness_failures.extend(spec_failures)
    elif capture_payload is not None and not raw_steps:
        harness_failures.append("vote_spec_unresolved_without_trace_rows")

    if vote_spec is not None:
        clip_assertion = assert_clip_bound_proof(
            vote_spec,
            recorded_row_bounds=recorded_row_value_bounds,
        )
        if not clip_assertion["passed"]:
            harness_failures.append("source_clip_not_pm127")

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

    applied_candidate_ids: dict[int, str] = {}
    if raw_steps and field_inventory.get("passed", False):
        applied_candidate_ids = build_teacher_forced_applied_candidate_ids(raw_steps)

    lane_by_width: dict[int, dict[str, Any]] = {}
    width_results: list[dict[str, Any]] = []
    if raw_steps and vote_spec is not None and field_inventory.get("passed", False):
        for width in widths:
            lane = replay_width_lane(
                raw_steps,
                vote_spec=vote_spec,
                width=width,
                applied_candidate_ids_by_step=applied_candidate_ids,
            )
            lane_by_width[width] = lane
            reference_lane = lane_by_width.get(16)
            invariance = (
                coarse_invariant_vs_reference(lane, reference_lane=reference_lane)
                if reference_lane is not None and width != 16
                else {
                    "coarse_crossing_invariant": lane.get(
                        "bit_identical_to_recorded_new_acc", False
                    ),
                    "mismatch_count": len(lane.get("w16_mismatch_rows") or []),
                    "mismatches": lane.get("w16_mismatch_rows") or [],
                }
            )
            width_results.append(
                {
                    "width": int(width),
                    "effective_clip_min": lane["effective_clip_min"],
                    "effective_clip_max": lane["effective_clip_max"],
                    "coarse_crossing_invariant": bool(
                        invariance["coarse_crossing_invariant"]
                    ),
                    "mismatch_count": int(invariance["mismatch_count"]),
                    "bit_identical_to_recorded_new_acc": bool(
                        lane.get("bit_identical_to_recorded_new_acc", False)
                    ),
                    "max_abs_acc_applied_flips": int(
                        lane.get("max_abs_acc_applied_flips", 0)
                    ),
                }
            )
            if width == 16 and not lane.get("bit_identical_to_recorded_new_acc", False):
                harness_failures.append("w16_not_bit_identical_to_reference")
            if width == 8 and not invariance["coarse_crossing_invariant"]:
                harness_failures.append("w8_not_reference_invariant")

    w_min_invariant = (
        compute_w_min_invariant(widths, lane_by_width=lane_by_width, reference_width=16)
        if lane_by_width
        else None
    )
    w_min_headroom_safe = (
        compute_w_min_headroom_safe(
            widths,
            lane_by_width=lane_by_width,
            reference_width=16,
            headroom_factor=headroom_factor,
        )
        if lane_by_width
        else None
    )
    w_min = w_min_headroom_safe
    max_abs_acc_applied = 0
    if lane_by_width.get(16) is not None:
        max_abs_acc_applied = int(lane_by_width[16].get("max_abs_acc_applied_flips", 0))
    headroom_ok = w_min_headroom_safe is not None
    branch = classify_w_min_label(
        w_min_headroom_safe,
        harness_failures=harness_failures,
        headroom_pass=headroom_ok,
    )

    source_clip_pm127 = bool(clip_assertion and clip_assertion.get("passed", False))
    source_semantics_prereg = {
        "global_clip_pm127_implies_w_ge_8_tautology_expected": source_clip_pm127,
        "sixteen_to_eight_bpw_shrink": (
            "source_semantics_confirmed" if source_clip_pm127 else "not_confirmed"
        ),
        "not_measured_discovery": True,
        "open_measurement_widths": [6, 4, 3, 2],
    }
    if vote_spec is not None:
        source_semantics_prereg["declared_clip"] = [
            vote_spec.accumulator_clip_min,
            vote_spec.accumulator_clip_max,
        ]
    if clip_assertion is not None and clip_assertion.get("recorded_row_min") is not None:
        source_semantics_prereg["recorded_row_extrema"] = [
            clip_assertion["recorded_row_min"],
            clip_assertion["recorded_row_max"],
        ]
    if recorded_row_value_bounds is not None:
        source_semantics_prereg["recorded_row_value_bounds"] = list(
            recorded_row_value_bounds
        )

    receipt: dict[str, Any] = {
        "schema_version": ACC_WIDTH_RECORDED_ROW_SWEEP_SCHEMA_VERSION,
        "contract_id": ACC_WIDTH_RECORDED_ROW_SWEEP_CONTRACT_ID,
        "receipt_kind": ACC_WIDTH_RECORDED_ROW_SWEEP_RECEIPT_KIND,
        "compact_receipt": True,
        "scope_statement": SCOPE_STATEMENT,
        "claim_boundary": {
            "measurement_only": True,
            "pre_full_stack_diagnostic_only": True,
            "single_trace": True,
            "tensor_wide_deferred": True,
            "runtime_readiness_claim": False,
            "training_or_acquisition_claim": False,
            "full_sub2_claim": False,
        },
        "input_integrity": integrity,
        "field_inventory_gate": field_inventory,
        "vote_spec": vote_spec.to_dict() if vote_spec is not None else None,
        "vote_spec_provenance": vote_spec_provenance,
        "clip_bound_assertion": clip_assertion,
        "recorded_row_value_bounds": (
            list(recorded_row_value_bounds) if recorded_row_value_bounds else None
        ),
        "saturation_policy": "clamp_match_vote_update_reference",
        "saturation_detail": (
            "decay+vote clamp to effective_clip(W); post-flip residual clamp "
            "[-threshold+1, threshold-1]; NO wrap"
        ),
        "width_grid": list(widths),
        "width_results": width_results,
        "w_min_invariant": w_min_invariant,
        "w_min_headroom_safe": w_min_headroom_safe,
        "w_min": w_min,
        "band_membership_scope": "recorded_echo_not_width_selector",
        "headroom_factor": float(headroom_factor),
        "headroom_rule": (
            "max_recorded_abs_acc_on_applied_flips * headroom_factor <= (2^(W-1)-1)"
        ),
        "max_abs_acc_applied_flips": max_abs_acc_applied,
        "headroom_pass": bool(headroom_ok),
        "source_semantics_prereg": source_semantics_prereg,
        "primary_label": branch["primary_label"],
        "sub_reasons": branch.get("sub_reasons", []),
        "branch_precedence": branch.get("branch_precedence", list(CLASSIFIER_PRECEDENCE)),
        "failure_reasons": branch.get("failure_reasons", []),
        "readiness_current_repo": _current_repo_readiness_summary(),
        "taxonomy_labels": [
            PRE_FULL_STACK_DIAGNOSTIC_ONLY,
            branch["primary_label"],
        ],
        "seam_debt": {
            "private_helper_imports": [
                "_file_sha256",
                "_stable_hash16",
                "_current_repo_readiness_summary",
            ],
            "audit_module_imports": [
                "reconstruct_transient_target",
            ],
            "screen_module_touched": False,
            "audit_module_touched": False,
        },
        "trace_metadata": {
            "optimizer_step_count": len(raw_steps),
            "trace_hash": _stable_hash16(
                [
                    int(step["optimizer_step_index"])
                    for step in raw_steps
                ]
            )
            if raw_steps
            else None,
            "teacher_forced_applied_candidate_count": len(applied_candidate_ids),
        },
    }
    return receipt
