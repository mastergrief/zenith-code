"""CPU read-only transient selection information audit over B2b/B2c traces."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from calm.hrm_text_158.native_full_stack.accumulator_policy_shadow_screen import (
    PRE_FULL_STACK_DIAGNOSTIC_ONLY,
    _file_sha256,
    _load_b2b_sequential_trace_steps,
    _stable_hash16,
    build_real_sequential_trace_candidate_stream,
)
from calm.hrm_text_158.native_full_stack.accumulator_policy_shadow_screen import (
    _current_repo_readiness_summary,
)

TRANSIENT_SELECTION_INFORMATION_AUDIT_SCHEMA_VERSION = (
    "hrm_text_158_transient_selection_information_audit/v0"
)
TRANSIENT_SELECTION_INFORMATION_AUDIT_CONTRACT_ID = (
    "transient_selection_information_audit_v0/post_b2c_accumulator_null"
)
TRANSIENT_SELECTION_INFORMATION_AUDIT_RECEIPT_KIND = (
    "cpu_read_only_transient_selection_information_audit"
)

FIELD_ROLE_PERSISTENT = "persistent_candidate_feature"
FIELD_ROLE_TRANSIENT_REF = "transient_reference_feature"
FIELD_ROLE_OUTCOME = "outcome_scoring_field"
FIELD_ROLE_FORBIDDEN = "forbidden_selector_field"

PROVENANCE_CARRIED = "carried_persistent"
PROVENANCE_STEP_LOCAL = "step_local_cheap"
PROVENANCE_IDENTITY = "identity_locality"

SELECTION_CARRIED_ONLY = "carried_only"
SELECTION_CARRIED_PLUS_IDENTITY = "carried_plus_identity_locality"
SELECTION_CARRIED_PLUS_STEP_LOCAL = "carried_plus_step_local"
SELECTION_STEP_LOCAL_ONLY = "step_local_only"
SELECTION_IDENTITY_ONLY = "identity_locality_only"
SELECTION_INVALID = "invalid"

BRANCH_HARNESS_FAIL = "screen_harness_or_gate_fail"
BRANCH_COMPRESSIBLE = "transient_compressible_persistent_candidate"
BRANCH_BUDGET_BLOCKED = "transient_sparse_but_budget_blocked"
BRANCH_UPDATE_LAW = "update_law_predictive_candidate"
BRANCH_COMPUTE_CONTROL = "transient_compute_control_only"

HELD_THRESHOLD_JACCARD = 0.90
HELD_THRESHOLD_REGRET = 0.90
HELD_THRESHOLD_ORACLE_TOP1 = 0.90
DIAGNOSTIC_BUDGET_BPW_LIMIT = 2.0
FULL_PHYSICAL_BUDGET_BPW_LIMIT = 2.0
INT8_Q_BPW = 8.0
DEFAULT_SCALE_METADATA_BPW = 0.5

FIT_STEP_LO = 1
FIT_STEP_HI = 25
HELD_STEP_LO = 26
HELD_STEP_HI = 50

FORBIDDEN_SELECTOR_FIELDS = frozenset(
    {
        "candidate_loss",
        "regret_vs_target_tie_band_oracle_top1_local_loss_delta",
        "selected_candidate_ids",
        "primary_label",
        "arm_label",
        "source_table_hash",
        "pre_update_state_hash",
        "post_update_telemetry",
    }
)

STEP_METADATA_FIELDS = frozenset(
    {
        "optimizer_step_index",
        "source_kind",
        "source_table_hash",
        "pre_update_state_hash",
        "post_update_telemetry",
        "schema",
    }
)

PERSISTENT_FIELD_SPECS: dict[str, str] = {
    "current_q_level": PROVENANCE_CARRIED,
    "pre_accumulator_i16": PROVENANCE_CARRIED,
    "tie_band_id": PROVENANCE_CARRIED,
    "vote_value": PROVENANCE_STEP_LOCAL,
    "abs_vote_value": PROVENANCE_STEP_LOCAL,
    "new_acc_i32_signed": PROVENANCE_STEP_LOCAL,
    "proposal_direction": PROVENANCE_STEP_LOCAL,
    "proximity_to_threshold": PROVENANCE_STEP_LOCAL,
    "threshold_residual_signed": PROVENANCE_STEP_LOCAL,
    "current_margin_abs": PROVENANCE_STEP_LOCAL,
    "current_rank_position": PROVENANCE_STEP_LOCAL,
    "candidate_id": PROVENANCE_IDENTITY,
    "state_key": PROVENANCE_IDENTITY,
    "flat_index": PROVENANCE_IDENTITY,
    "flat_index_quartile": PROVENANCE_IDENTITY,
    "current_rank_quartile_within_state": PROVENANCE_IDENTITY,
    "in_target_tie_band": PROVENANCE_IDENTITY,
    "state_candidate_count": PROVENANCE_IDENTITY,
    "transition_class": PROVENANCE_IDENTITY,
}

TRANSIENT_REFERENCE_FIELDS = frozenset({"local_loss_delta"})
OUTCOME_FIELDS = frozenset(
    {
        "candidate_loss",
        "regret_vs_target_tie_band_oracle_top1_local_loss_delta",
    }
)

BRANCH_PRECEDENCE = (
    BRANCH_HARNESS_FAIL,
    BRANCH_COMPRESSIBLE,
    BRANCH_BUDGET_BLOCKED,
    BRANCH_UPDATE_LAW,
    BRANCH_COMPUTE_CONTROL,
)

SCORE_DEPENDENCY_ENFORCEMENT = "field_filtered_view"


@dataclass(frozen=True)
class SummaryFamilySpec:
    family_id: str
    selector_fields: tuple[str, ...]
    provenance_tags: frozenset[str]
    score_fields: tuple[str, ...]
    score_fn: Callable[[Mapping[str, Any]], float]


def _float_field(row: Mapping[str, Any], field: str, default: float = 0.0) -> float:
    value = row.get(field)
    if value is None:
        return default
    return float(value)


def _int_field(row: Mapping[str, Any], field: str, default: int = 0) -> int:
    value = row.get(field)
    if value is None:
        return default
    return int(value)


def _identity_score(row: Mapping[str, Any]) -> float:
    flat_index = row.get("flat_index")
    if flat_index is not None:
        return -float(flat_index)
    state_key = str(row.get("state_key") or "")
    return -float(sum(ord(char) for char in state_key) % 997)


SUMMARY_FAMILY_SPECS: tuple[SummaryFamilySpec, ...] = (
    SummaryFamilySpec(
        family_id="carried_persistent_bucket",
        selector_fields=(
            "pre_accumulator_i16",
            "current_q_level",
            "tie_band_id",
        ),
        provenance_tags=frozenset({PROVENANCE_CARRIED}),
        score_fields=("pre_accumulator_i16", "current_q_level"),
        score_fn=lambda row: (
            0.01 * _float_field(row, "pre_accumulator_i16")
            + 0.001 * _float_field(row, "current_q_level")
        ),
    ),
    SummaryFamilySpec(
        family_id="carried_persistent_flip",
        selector_fields=(
            "pre_accumulator_i16",
            "current_q_level",
            "proximity_to_threshold",
        ),
        provenance_tags=frozenset({PROVENANCE_CARRIED, PROVENANCE_STEP_LOCAL}),
        score_fields=(
            "proximity_to_threshold",
            "pre_accumulator_i16",
            "current_q_level",
        ),
        score_fn=lambda row: (
            -_float_field(row, "proximity_to_threshold")
            + 0.01 * _float_field(row, "pre_accumulator_i16")
            + 0.001 * _float_field(row, "current_q_level")
        ),
    ),
    SummaryFamilySpec(
        family_id="step_local_vote_proximity",
        selector_fields=(
            "vote_value",
            "abs_vote_value",
            "proximity_to_threshold",
            "current_rank_position",
            "current_margin_abs",
        ),
        provenance_tags=frozenset({PROVENANCE_STEP_LOCAL}),
        score_fields=(
            "vote_value",
            "abs_vote_value",
            "proximity_to_threshold",
            "current_rank_position",
        ),
        score_fn=lambda row: (
            _float_field(row, "vote_value")
            + _float_field(row, "abs_vote_value")
            - 0.1 * _float_field(row, "proximity_to_threshold")
            - 0.01 * _float_field(row, "current_rank_position")
        ),
    ),
    SummaryFamilySpec(
        family_id="identity_locality_flat",
        selector_fields=(
            "flat_index",
            "state_key",
            "flat_index_quartile",
            "current_rank_quartile_within_state",
        ),
        provenance_tags=frozenset({PROVENANCE_IDENTITY}),
        score_fields=("flat_index", "state_key"),
        score_fn=_identity_score,
    ),
    SummaryFamilySpec(
        family_id="carried_plus_identity",
        selector_fields=(
            "pre_accumulator_i16",
            "current_q_level",
            "flat_index",
            "state_key",
        ),
        provenance_tags=frozenset({PROVENANCE_CARRIED, PROVENANCE_IDENTITY}),
        score_fields=("pre_accumulator_i16", "flat_index", "state_key"),
        score_fn=lambda row: (
            0.01 * _float_field(row, "pre_accumulator_i16")
            + 0.001 * _identity_score(row)
        ),
    ),
)


def classify_field_role(field_name: str) -> tuple[str, str | None]:
    if field_name in STEP_METADATA_FIELDS:
        return "step_metadata", None
    if field_name in TRANSIENT_REFERENCE_FIELDS:
        return FIELD_ROLE_TRANSIENT_REF, None
    if field_name in OUTCOME_FIELDS:
        return FIELD_ROLE_OUTCOME, None
    if field_name in FORBIDDEN_SELECTOR_FIELDS:
        return FIELD_ROLE_FORBIDDEN, None
    if field_name in PERSISTENT_FIELD_SPECS:
        return FIELD_ROLE_PERSISTENT, PERSISTENT_FIELD_SPECS[field_name]
  # Unknown fields default to forbidden to stay fail-closed.
    return FIELD_ROLE_FORBIDDEN, None


def enumerate_trace_fields(steps: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    fields: set[str] = set()
    for step in steps:
        for row in step.get("sampled_candidate_table") or ():
            if isinstance(row, Mapping):
                fields.update(str(key) for key in row)
    return tuple(sorted(fields))


def build_field_inventory(
    steps: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    actual_fields = enumerate_trace_fields(steps)
    classification: dict[str, str] = {}
    provenance: dict[str, str] = {}
    forbidden_fields: list[str] = []
    persistent_fields: list[str] = []
    for field_name in actual_fields:
        role, tag = classify_field_role(field_name)
        classification[field_name] = role
        if role == FIELD_ROLE_PERSISTENT and tag is not None:
            provenance[field_name] = tag
            persistent_fields.append(field_name)
        elif role == FIELD_ROLE_FORBIDDEN:
            forbidden_fields.append(field_name)

    local_loss_role = classification.get("local_loss_delta")
    field_role_ambiguous = (
        local_loss_role != FIELD_ROLE_TRANSIENT_REF
        and "local_loss_delta" in classification
    )
    return {
        "actual_fields": list(actual_fields),
        "classification": classification,
        "provenance_tags": provenance,
        "persistent_candidate_fields": persistent_fields,
        "forbidden_fields": forbidden_fields,
        "field_role_ambiguous": field_role_ambiguous,
    }


def verify_input_integrity(
    *,
    stable_trace_path: Path,
    original_trace_path: Path,
    capture_receipt_path: Path,
    b2c_receipt_path: Path,
    expected_shas: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    expected = dict(expected_shas or {})
    paths = {
        "stable_trace": stable_trace_path,
        "original_trace": original_trace_path,
        "capture_receipt": capture_receipt_path,
        "b2c_receipt": b2c_receipt_path,
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

    stable_sha = observed_shas.get("stable_trace")
    original_sha = observed_shas.get("original_trace")
    stable_copy_equals_original = (
        stable_sha is not None
        and original_sha is not None
        and stable_sha == original_sha
    )
    if stable_sha is not None and original_sha is not None and not stable_copy_equals_original:
        failure_reasons.append("stable_copy_trace_sha_mismatch")

    transient_hash_expected: str | None = None
    if b2c_receipt_path.exists():
        try:
            b2c_payload = json.loads(b2c_receipt_path.read_text(encoding="utf-8"))
            transient_hash_expected = (
                b2c_payload.get("arms", {})
                .get("transient_resolver_only", {})
                .get("selected_candidate_ids_hash16")
            )
        except (OSError, json.JSONDecodeError):
            failure_reasons.append("b2c_receipt_parse_error")

    return {
        "paths": {key: str(path) for key, path in paths.items()},
        "sha256": observed_shas,
        "expected_sha256": dict(expected),
        "stable_copy_equals_original": stable_copy_equals_original,
        "transient_selected_candidate_ids_hash16_expected": transient_hash_expected,
        "passed": not failure_reasons,
        "failure_reasons": failure_reasons,
    }


def load_raw_sequential_steps(trace_path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    steps, failures = _load_b2b_sequential_trace_steps(trace_path)
    normalized: list[dict[str, Any]] = []
    for step in steps:
        record = dict(step)
        if "sampled_candidate_table" not in record and "candidates" in record:
            record["sampled_candidate_table"] = [
                {
                    "candidate_id": str(row["candidate_id"]),
                    "current_rank_position": int(row["current_rank_position"]),
                    "local_loss_delta": float(row["local_loss_delta"]),
                    "pre_accumulator_i16": int(row["pre_accumulator_i16"]),
                    "new_acc_i32_signed": int(row["new_acc_i32_signed"]),
                    "proximity_to_threshold": int(row["proximity_to_threshold"]),
                }
                for row in record["candidates"]
            ]
        normalized.append(record)
    return normalized, failures


def reconstruct_transient_target(
    steps: Sequence[Mapping[str, Any]],
    *,
    rate_cap: int = 1,
) -> tuple[list[tuple[str, ...]], str]:
    selected_by_step: list[tuple[str, ...]] = []
    for step in steps:
        rows = list(step.get("sampled_candidate_table") or ())
        ranked = sorted(
            rows,
            key=lambda row: (float(row["local_loss_delta"]), str(row["candidate_id"])),
        )
        selected_by_step.append(
            tuple(str(row["candidate_id"]) for row in ranked[:rate_cap])
        )
    return selected_by_step, _stable_hash16(selected_by_step)


def partition_steps(
    steps: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    fit_steps = [
        step
        for step in steps
        if FIT_STEP_LO <= int(step["optimizer_step_index"]) <= FIT_STEP_HI
    ]
    held_steps = [
        step
        for step in steps
        if HELD_STEP_LO <= int(step["optimizer_step_index"]) <= HELD_STEP_HI
    ]
    return {
        "fit_step_range": [FIT_STEP_LO, FIT_STEP_HI],
        "held_step_range": [HELD_STEP_LO, HELD_STEP_HI],
        "fit_step_count": len(fit_steps),
        "held_step_count": len(held_steps),
        "fit_steps": fit_steps,
        "held_steps": held_steps,
        "held_candidate_row_denominator": sum(
            len(step.get("sampled_candidate_table") or ()) for step in held_steps
        ),
    }


def _selector_field_view(
    row: Mapping[str, Any],
    selector_fields: Sequence[str],
) -> dict[str, Any]:
    """Expose only declared selector fields so score_fn cannot read undeclared data."""
    return {field_name: row.get(field_name) for field_name in selector_fields}


def _select_with_family(
    rows: Sequence[Mapping[str, Any]],
    family: SummaryFamilySpec,
    *,
    rate_cap: int,
) -> tuple[str, ...]:
    ranked = sorted(
        rows,
        key=lambda row: (
            family.score_fn(_selector_field_view(row, family.selector_fields)),
            str(row["candidate_id"]),
        ),
        reverse=True,
    )
    return tuple(str(row["candidate_id"]) for row in ranked[:rate_cap])


def feature_provenance_mix(family: SummaryFamilySpec) -> dict[str, int]:
    counts = {
        PROVENANCE_CARRIED: 0,
        PROVENANCE_STEP_LOCAL: 0,
        PROVENANCE_IDENTITY: 0,
    }
    for field_name in family.selector_fields:
        tag = PERSISTENT_FIELD_SPECS.get(field_name)
        if tag in counts:
            counts[tag] += 1
    return counts


def actual_provenance_tags(family: SummaryFamilySpec) -> frozenset[str]:
    mix = feature_provenance_mix(family)
    return frozenset(tag for tag, count in mix.items() if count > 0)


def provenance_declaration_matches_fields(family: SummaryFamilySpec) -> bool:
    return family.provenance_tags == actual_provenance_tags(family)


def score_fields_within_selector(family: SummaryFamilySpec) -> bool:
    selector_set = frozenset(family.selector_fields)
    return all(field_name in selector_set for field_name in family.score_fields)


def uses_outcome_or_forbidden_selector_field(family: SummaryFamilySpec) -> bool:
    checked_fields = set(family.selector_fields) | set(family.score_fields)
    for field_name in checked_fields:
        role, _ = classify_field_role(field_name)
        if role in {FIELD_ROLE_OUTCOME, FIELD_ROLE_FORBIDDEN}:
            return True
    return False


def score_field_provenance_tags(family: SummaryFamilySpec) -> frozenset[str]:
    tags: set[str] = set()
    for field_name in family.score_fields:
        tag = PERSISTENT_FIELD_SPECS.get(field_name)
        if tag is not None:
            tags.add(tag)
    return frozenset(tags)


def selection_semantics_for_tags(tags: frozenset[str]) -> str:
    if tags == frozenset({PROVENANCE_CARRIED}):
        return SELECTION_CARRIED_ONLY
    if tags == frozenset({PROVENANCE_STEP_LOCAL}):
        return SELECTION_STEP_LOCAL_ONLY
    if tags == frozenset({PROVENANCE_IDENTITY}):
        return SELECTION_IDENTITY_ONLY
    if tags == frozenset({PROVENANCE_CARRIED, PROVENANCE_IDENTITY}):
        return SELECTION_CARRIED_PLUS_IDENTITY
    if PROVENANCE_STEP_LOCAL in tags:
        return SELECTION_CARRIED_PLUS_STEP_LOCAL
    return SELECTION_INVALID


def selection_semantics_for_family(family: SummaryFamilySpec) -> str:
    if not provenance_declaration_matches_fields(family):
        return SELECTION_INVALID
    return selection_semantics_for_tags(actual_provenance_tags(family))


def compute_step_metrics(
    *,
    steps: Sequence[Mapping[str, Any]],
    selected_by_step: Sequence[tuple[str, ...]],
    transient_by_step: Sequence[tuple[str, ...]],
    rate_cap: int,
) -> dict[str, Any]:
    if not steps:
        return {
            "jaccard_vs_transient": 0.0,
            "regret_capture_vs_oracle": 0.0,
            "oracle_top1_in_selected_rate": 0.0,
            "step_denominator": 0,
        }

    jaccards: list[float] = []
    regret_capture: list[float] = []
    oracle_top1: list[float] = []
    for step, selected, transient in zip(steps, selected_by_step, transient_by_step):
        rows = list(step.get("sampled_candidate_table") or ())
        gains = {
            str(row["candidate_id"]): max(0.0, -float(row["local_loss_delta"]))
            for row in rows
        }
        oracle_best = max(gains.values()) if gains else 0.0
        oracle_top_id = max(gains.items(), key=lambda item: item[1])[0] if gains else ""
        selected_set = set(selected)
        transient_set = set(transient)
        union = selected_set | transient_set
        jaccards.append(len(selected_set & transient_set) / len(union) if union else 1.0)
        selected_gain = sum(gains.get(candidate_id, 0.0) for candidate_id in selected_set)
        regret_capture.append(
            min(1.0, selected_gain / max(oracle_best * rate_cap, 1e-12))
        )
        oracle_top1.append(1.0 if oracle_top_id in selected_set else 0.0)

    step_denominator = len(steps)
    return {
        "jaccard_vs_transient": sum(jaccards) / step_denominator,
        "regret_capture_vs_oracle": sum(regret_capture) / step_denominator,
        "oracle_top1_in_selected_rate": sum(oracle_top1) / step_denominator,
        "step_denominator": step_denominator,
    }


def threshold_triple_passes(metrics: Mapping[str, Any]) -> bool:
    return (
        float(metrics["jaccard_vs_transient"]) >= HELD_THRESHOLD_JACCARD
        and float(metrics["regret_capture_vs_oracle"]) >= HELD_THRESHOLD_REGRET
        and float(metrics["oracle_top1_in_selected_rate"]) >= HELD_THRESHOLD_ORACLE_TOP1
    )


def compute_budget_ledger(
  family: SummaryFamilySpec,
) -> dict[str, Any]:
    field_count = len(family.selector_fields)
    sidecar_bpw = min(2.0, 0.25 * field_count)
    q_bpw = INT8_Q_BPW
    scale_metadata_bpw = DEFAULT_SCALE_METADATA_BPW
    total_physical_persistent_bpw = sidecar_bpw + q_bpw + scale_metadata_bpw
    diagnostic_budget_pass = sidecar_bpw <= DIAGNOSTIC_BUDGET_BPW_LIMIT
    full_physical_budget_pass = (
        total_physical_persistent_bpw <= FULL_PHYSICAL_BUDGET_BPW_LIMIT
    )
    q_scale_budget_not_created = diagnostic_budget_pass and not full_physical_budget_pass
    return {
        "sidecar_bpw": sidecar_bpw,
        "q_bpw": q_bpw,
        "scale_metadata_bpw": scale_metadata_bpw,
        "total_physical_persistent_bpw": total_physical_persistent_bpw,
        "diagnostic_budget_pass": diagnostic_budget_pass,
        "full_physical_budget_pass": full_physical_budget_pass,
        "q_scale_budget_not_created": q_scale_budget_not_created,
        "algorithmic_proxy_not_physical_sub2": (
            diagnostic_budget_pass and not full_physical_budget_pass
        ),
    }


def evaluate_summary_families(
    steps: Sequence[Mapping[str, Any]],
    *,
    fit_steps: Sequence[Mapping[str, Any]],
    held_steps: Sequence[Mapping[str, Any]],
    transient_by_step: Sequence[tuple[str, ...]],
    rate_cap: int = 1,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for family in SUMMARY_FAMILY_SPECS:
        full_selected: list[tuple[str, ...]] = []
        fit_selected: list[tuple[str, ...]] = []
        held_selected: list[tuple[str, ...]] = []
        held_transient: list[tuple[str, ...]] = []
        for step in steps:
            rows = list(step.get("sampled_candidate_table") or ())
            selected = _select_with_family(rows, family, rate_cap=rate_cap)
            full_selected.append(selected)
            step_index = int(step["optimizer_step_index"])
            if FIT_STEP_LO <= step_index <= FIT_STEP_HI:
                fit_selected.append(selected)
            if HELD_STEP_LO <= step_index <= HELD_STEP_HI:
                held_selected.append(selected)
                held_transient.append(
                    transient_by_step[int(step_index) - 1]
                    if 0 < int(step_index) <= len(transient_by_step)
                    else ()
                )

        held_metrics = compute_step_metrics(
            steps=held_steps,
            selected_by_step=held_selected,
            transient_by_step=held_transient,
            rate_cap=rate_cap,
        )
        full_metrics = compute_step_metrics(
            steps=steps,
            selected_by_step=full_selected,
            transient_by_step=transient_by_step,
            rate_cap=rate_cap,
        )
        provenance_mix = feature_provenance_mix(family)
        declaration_match = provenance_declaration_matches_fields(family)
        score_dependency_ok = score_fields_within_selector(family)
        outcome_or_forbidden = uses_outcome_or_forbidden_selector_field(family)
        semantics = selection_semantics_for_family(family)
        budget = compute_budget_ledger(family)
        held_pass = threshold_triple_passes(held_metrics)
        full_pass = threshold_triple_passes(full_metrics)
        overfit_single_trace = full_pass and not held_pass
        results.append(
            {
                "family_id": family.family_id,
                "selector_fields": list(family.selector_fields),
                "score_fields": list(family.score_fields),
                "score_field_provenance_tags": sorted(score_field_provenance_tags(family)),
                "feature_provenance_mix": provenance_mix,
                "provenance_declaration_mismatch": not declaration_match,
                "score_field_dependency_mismatch": not score_dependency_ok,
                "selection_semantics": semantics,
                "step_local_selection_dependency": semantics
                in {SELECTION_STEP_LOCAL_ONLY, SELECTION_CARRIED_PLUS_STEP_LOCAL},
                "uses_outcome_or_forbidden_selector_field": outcome_or_forbidden,
                "fit_step_count": len(fit_steps),
                "held_metrics": held_metrics,
                "full_trace_metrics": full_metrics,
                "held_threshold_pass": held_pass,
                "full_trace_threshold_pass": full_pass,
                "summary_overfit_single_trace": overfit_single_trace,
                "budget": budget,
            }
        )
    return results


def classify_branch(
    *,
    harness_failures: Sequence[str],
    field_inventory: Mapping[str, Any],
    transient_target_reconstructed: bool,
    family_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    sub_reasons: list[str] = []
    if harness_failures:
        return {
            "primary_label": BRANCH_HARNESS_FAIL,
            "sub_reasons": list(harness_failures),
            "branch_precedence": BRANCH_PRECEDENCE,
            "failure_reasons": list(harness_failures),
        }
    if field_inventory.get("field_role_ambiguous"):
        sub_reasons.append("field_role_ambiguous")
        return {
            "primary_label": BRANCH_HARNESS_FAIL,
            "sub_reasons": sub_reasons,
            "branch_precedence": BRANCH_PRECEDENCE,
            "failure_reasons": sub_reasons,
        }
    if not transient_target_reconstructed:
        sub_reasons.append("transient_target_not_reconstructible")
        return {
            "primary_label": BRANCH_HARNESS_FAIL,
            "sub_reasons": sub_reasons,
            "branch_precedence": BRANCH_PRECEDENCE,
            "failure_reasons": sub_reasons,
        }

    provenance_mismatches = [
        result["family_id"]
        for result in family_results
        if bool(result.get("provenance_declaration_mismatch"))
    ]
    if provenance_mismatches:
        sub_reasons.append("summary_family_provenance_declaration_mismatch")
        return {
            "primary_label": BRANCH_HARNESS_FAIL,
            "sub_reasons": sub_reasons + provenance_mismatches[:3],
            "branch_precedence": BRANCH_PRECEDENCE,
            "failure_reasons": sub_reasons + provenance_mismatches[:3],
        }

    score_dependency_mismatches = [
        result["family_id"]
        for result in family_results
        if bool(result.get("score_field_dependency_mismatch"))
    ]
    if score_dependency_mismatches:
        sub_reasons.append("summary_family_score_field_dependency_mismatch")
        return {
            "primary_label": BRANCH_HARNESS_FAIL,
            "sub_reasons": sub_reasons + score_dependency_mismatches[:3],
            "branch_precedence": BRANCH_PRECEDENCE,
            "failure_reasons": sub_reasons + score_dependency_mismatches[:3],
        }

    outcome_or_forbidden_families = [
        result["family_id"]
        for result in family_results
        if bool(result.get("uses_outcome_or_forbidden_selector_field"))
    ]
    if outcome_or_forbidden_families:
        sub_reasons.append("summary_family_outcome_or_forbidden_selector_field")
        return {
            "primary_label": BRANCH_HARNESS_FAIL,
            "sub_reasons": sub_reasons + outcome_or_forbidden_families[:3],
            "branch_precedence": BRANCH_PRECEDENCE,
            "failure_reasons": sub_reasons + outcome_or_forbidden_families[:3],
        }

    overfit_families = [
        result["family_id"]
        for result in family_results
        if bool(result.get("summary_overfit_single_trace"))
    ]
    if overfit_families:
        sub_reasons.append("summary_overfit_single_trace")
        return {
            "primary_label": BRANCH_HARNESS_FAIL,
            "sub_reasons": sub_reasons,
            "branch_precedence": BRANCH_PRECEDENCE,
            "failure_reasons": sub_reasons,
        }

    compressible_candidates = [
        result
        for result in family_results
        if bool(result.get("held_threshold_pass"))
        and result.get("selection_semantics")
        in {SELECTION_CARRIED_ONLY, SELECTION_CARRIED_PLUS_IDENTITY}
        and bool(result.get("budget", {}).get("diagnostic_budget_pass"))
        and bool(result.get("budget", {}).get("full_physical_budget_pass"))
    ]
    if compressible_candidates:
        return {
            "primary_label": BRANCH_COMPRESSIBLE,
            "sub_reasons": [
                result["family_id"] for result in compressible_candidates[:1]
            ],
            "branch_precedence": BRANCH_PRECEDENCE,
            "failure_reasons": [],
        }

    budget_blocked = [
        result
        for result in family_results
        if bool(result.get("held_threshold_pass"))
        and result.get("selection_semantics")
        in {SELECTION_CARRIED_ONLY, SELECTION_CARRIED_PLUS_IDENTITY}
        and bool(result.get("budget", {}).get("diagnostic_budget_pass"))
        and not bool(result.get("budget", {}).get("full_physical_budget_pass"))
    ]
    if budget_blocked:
        return {
            "primary_label": BRANCH_BUDGET_BLOCKED,
            "sub_reasons": [result["family_id"] for result in budget_blocked[:1]],
            "branch_precedence": BRANCH_PRECEDENCE,
            "failure_reasons": [],
        }

    update_law_candidates = [
        result
        for result in family_results
        if bool(result.get("held_threshold_pass"))
        and result.get("selection_semantics")
        in {SELECTION_STEP_LOCAL_ONLY, SELECTION_CARRIED_PLUS_STEP_LOCAL}
    ]
    if update_law_candidates:
        return {
            "primary_label": BRANCH_UPDATE_LAW,
            "sub_reasons": [
                "step_local_selection_dependency",
                "selection_law_candidate",
            ],
            "branch_precedence": BRANCH_PRECEDENCE,
            "failure_reasons": [],
        }

    identity_only = [
        result
        for result in family_results
        if result.get("selection_semantics") == SELECTION_IDENTITY_ONLY
        and bool(result.get("held_threshold_pass"))
    ]
    if identity_only:
        return {
            "primary_label": BRANCH_COMPUTE_CONTROL,
            "sub_reasons": ["identity_locality_only_no_mechanism_route"],
            "branch_precedence": BRANCH_PRECEDENCE,
            "failure_reasons": [],
        }

    return {
        "primary_label": BRANCH_COMPUTE_CONTROL,
        "sub_reasons": ["persistent_features_do_not_reproduce_held_transient_choices"],
        "branch_precedence": BRANCH_PRECEDENCE,
        "failure_reasons": [],
    }


def build_transient_selection_information_audit(
    *,
    stable_trace_path: str | Path,
    original_trace_path: str | Path,
    capture_receipt_path: str | Path,
    b2c_receipt_path: str | Path,
    expected_shas: Mapping[str, str] | None = None,
    stable_copy_dir: str | Path | None = None,
    rate_cap: int = 1,
) -> dict[str, Any]:
    integrity = verify_input_integrity(
        stable_trace_path=Path(stable_trace_path),
        original_trace_path=Path(original_trace_path),
        capture_receipt_path=Path(capture_receipt_path),
        b2c_receipt_path=Path(b2c_receipt_path),
        expected_shas=expected_shas,
    )

    harness_failures = list(integrity.get("failure_reasons") or [])
    raw_steps: list[dict[str, Any]] = []
    if integrity.get("passed", False):
        raw_steps, load_failures = load_raw_sequential_steps(Path(stable_trace_path))
        harness_failures.extend(load_failures)

    stream, stream_metadata, stream_failures = build_real_sequential_trace_candidate_stream(
        [stable_trace_path],
        stable_copy_dir=stable_copy_dir,
    )
    harness_failures.extend(stream_failures)
    if not raw_steps and stream:
        raw_steps = [
            {
                "optimizer_step_index": int(step["optimizer_step_index"]),
                "sampled_candidate_table": [
                    {
                        "candidate_id": str(row["candidate_id"]),
                        "current_rank_position": int(row["current_rank_position"]),
                        "local_loss_delta": float(row["local_loss_delta"]),
                        "pre_accumulator_i16": int(row["pre_accumulator_i16"]),
                        "new_acc_i32_signed": int(row["new_acc_i32_signed"]),
                        "proximity_to_threshold": int(row["proximity_to_threshold"]),
                    }
                    for row in step["candidates"]
                ],
            }
            for step in stream
        ]

    field_inventory = build_field_inventory(raw_steps) if raw_steps else {
        "actual_fields": [],
        "classification": {},
        "provenance_tags": {},
        "persistent_candidate_fields": [],
        "forbidden_fields": [],
        "field_role_ambiguous": False,
    }

    transient_by_step: list[tuple[str, ...]] = []
    transient_hash_observed: str | None = None
    transient_target_reconstructed = False
    if raw_steps and not field_inventory.get("field_role_ambiguous"):
        transient_by_step, transient_hash_observed = reconstruct_transient_target(
            raw_steps,
            rate_cap=rate_cap,
        )
        expected_hash = integrity.get("transient_selected_candidate_ids_hash16_expected")
        transient_target_reconstructed = (
            expected_hash is None or transient_hash_observed == expected_hash
        )
        if expected_hash is not None and not transient_target_reconstructed:
            harness_failures.append("transient_target_not_reconstructible")

    split = partition_steps(raw_steps) if raw_steps else {
        "fit_step_range": [FIT_STEP_LO, FIT_STEP_HI],
        "held_step_range": [HELD_STEP_LO, HELD_STEP_HI],
        "fit_step_count": 0,
        "held_step_count": 0,
        "fit_steps": [],
        "held_steps": [],
        "held_candidate_row_denominator": 0,
    }

    family_results = (
        evaluate_summary_families(
            raw_steps,
            fit_steps=split["fit_steps"],
            held_steps=split["held_steps"],
            transient_by_step=transient_by_step,
            rate_cap=rate_cap,
        )
        if raw_steps and transient_by_step
        else []
    )

    branch = classify_branch(
        harness_failures=harness_failures,
        field_inventory=field_inventory,
        transient_target_reconstructed=transient_target_reconstructed,
        family_results=family_results,
    )

    winning_family = next(
        (
            result
            for result in family_results
            if result["family_id"] in branch.get("sub_reasons", ())
        ),
        family_results[0] if family_results else None,
    )

    receipt: dict[str, Any] = {
        "schema_version": TRANSIENT_SELECTION_INFORMATION_AUDIT_SCHEMA_VERSION,
        "contract_id": TRANSIENT_SELECTION_INFORMATION_AUDIT_CONTRACT_ID,
        "receipt_kind": TRANSIENT_SELECTION_INFORMATION_AUDIT_RECEIPT_KIND,
        "compact_receipt": True,
        "claim_boundary": {
            "measurement_only": True,
            "pre_full_stack_diagnostic_only": True,
            "runtime_readiness_claim": False,
            "training_or_acquisition_claim": False,
            "q_mutation_applied_to_model": False,
            "full_sub2_claim": False,
        },
        "input_integrity": integrity,
        "field_inventory_gate": {
            **field_inventory,
            "transient_target_reconstructed": transient_target_reconstructed,
            "transient_selected_candidate_ids_hash16_expected": integrity.get(
                "transient_selected_candidate_ids_hash16_expected"
            ),
            "transient_selected_candidate_ids_hash16_observed": transient_hash_observed,
        },
        "held_split": {
            "fit_step_range": split["fit_step_range"],
            "held_step_range": split["held_step_range"],
            "fit_step_denominator": split["fit_step_count"],
            "held_step_denominator": split["held_step_count"],
            "held_candidate_row_denominator": split["held_candidate_row_denominator"],
        },
        "summary_families": family_results,
        "held_metrics": (
            winning_family.get("held_metrics") if winning_family is not None else {}
        ),
        "budget": winning_family.get("budget") if winning_family is not None else {},
        "primary_label": branch["primary_label"],
        "sub_reasons": branch.get("sub_reasons", []),
        "branch_precedence": branch.get("branch_precedence", list(BRANCH_PRECEDENCE)),
        "failure_reasons": branch.get("failure_reasons", []),
        "readiness_current_repo": _current_repo_readiness_summary(),
        "stream_metadata": {
            "optimizer_step_count": stream_metadata.get("optimizer_step_count", 0),
            "trace_hash": stream_metadata.get("trace_hash"),
        },
        "taxonomy_labels": [
            PRE_FULL_STACK_DIAGNOSTIC_ONLY,
            branch["primary_label"],
        ],
        "score_dependency_enforcement": SCORE_DEPENDENCY_ENFORCEMENT,
        "seam_debt": {
            "private_helper_imports": [
                "_stable_hash16",
                "_file_sha256",
                "_load_b2b_sequential_trace_steps",
                "_current_repo_readiness_summary",
            ],
            "screen_module_touched": False,
        },
    }
    if winning_family is not None:
        receipt["winning_family"] = {
            "family_id": winning_family["family_id"],
            "feature_provenance_mix": winning_family["feature_provenance_mix"],
            "selection_semantics": winning_family["selection_semantics"],
            "step_local_selection_dependency": winning_family[
                "step_local_selection_dependency"
            ],
            "uses_outcome_or_forbidden_selector_field": winning_family[
                "uses_outcome_or_forbidden_selector_field"
            ],
        }
    return receipt
