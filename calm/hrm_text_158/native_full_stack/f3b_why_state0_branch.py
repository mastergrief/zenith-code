"""Fold-3B mechanism-diagnosis branch classifier (inert / not live-wired)."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Mapping, Sequence

CLASSIFIER = "F3B_WHY_STATE0_BRANCH_V1"

ANTI_OVERCLAIM_VERBATIM = (
    "Within the Fold-3B packet scope, state0-only crossing support classifies as one of "
    "the pre-registered branches. FORBIDDEN: candidate-C, CA/reduction eligibility, W/P, "
    "~430MB bank pin, universal all-state census, bank mutation, sub-2 readiness, "
    "full-stack readiness, implementation readiness."
)

ALLOWED_CLAIM = (
    "Within the Fold-3B packet scope, state0-only crossing support classifies as one of "
    "the pre-registered branches."
)

RECEIPT_SCHEMA = "hrm_text_158_fold3b_mechanism_diagnosis_receipt/v1"
PREFLIGHT_RECEIPT_SCHEMA = "hrm_text_158_fold3b_step1_feasibility_preflight_receipt/v1"
PREREG_PACKET_SCHEMA = "hrm_text_158_fold3b_step1_prereg_packet/v1"

# Single schema authority: must match prereg packet receipt_schema.required_fields.
REQUIRED_RECEIPT_FIELDS: tuple[str, ...] = (
    "sampled_state_set",
    "sampled_state_order",
    "order_rank_by_semantic_state",
    "semantic_state_id",
    "per_state",
    "dedup_reset_called",
    "dedup_session_scope",
    "wrapper_path",
    "primary_receipt_path",
    "fallback_receipt_path",
    "science_verdict_source",
    "parent_sha",
    "git_head_required",
    "variable_id",
    "control_reason",
    "f3b_branch",
    "f3b_branch_inputs",
    "ready_for_main_science",
    "counts_as_sub2",
    "pre_full_stack_diagnostic",
)

VALID_DEDUP_SESSION_SCOPES: frozenset[str] = frozenset({"probe_subprocess"})

REQUIRED_PER_STATE_FIELDS: tuple[str, ...] = (
    "state_index",
    "crossing_indices_len",
    "crossing_count",
    "mark_count",
)

REQUIRED_BRANCH_INPUT_FIELDS: tuple[str, ...] = (
    "operational_ok",
    "schema_ok",
    "sampled_state_set",
    "sampled_state_order",
    "order_rank_by_semantic_state",
    "exact_per_state_coverage",
    "dedup_reset_called",
    "dedup_session_scope",
    "identity_order_inertness_proven",
    "semantic_state0_crossing_indices_len",
    "cb_state_count",
    "first_measured_semantic_state",
    "first_measured_is_crossing_bearing",
    "semantic_state0_is_crossing_bearing",
    "sampled_set_changed",
    "mark_count_consistent",
    "variable_id",
    "control_reason",
)


class F3BWhyState0Branch(StrEnum):
    NO_VERDICT_OPERATIONAL = "F3B_NO_VERDICT_OPERATIONAL"
    NO_VERDICT_SCHEMA = "F3B_NO_VERDICT_SCHEMA"
    MEASUREMENT_ORDER_ARTIFACT = "F3B_MEASUREMENT_ORDER_ARTIFACT"
    SAMPLE_SET_OR_ELIGIBILITY_ARTIFACT = "F3B_SAMPLE_SET_OR_ELIGIBILITY_ARTIFACT"
    MARKING_OR_DEDUP_ARTIFACT = "F3B_MARKING_OR_DEDUP_ARTIFACT"
    STATE0_IDENTITY_STRUCTURE = "F3B_STATE0_IDENTITY_STRUCTURE"
    MIXED_OR_INCONCLUSIVE = "F3B_MIXED_OR_INCONCLUSIVE"


BRANCH_PRECEDENCE: tuple[F3BWhyState0Branch, ...] = (
    F3BWhyState0Branch.NO_VERDICT_OPERATIONAL,
    F3BWhyState0Branch.NO_VERDICT_SCHEMA,
    F3BWhyState0Branch.MEASUREMENT_ORDER_ARTIFACT,
    F3BWhyState0Branch.SAMPLE_SET_OR_ELIGIBILITY_ARTIFACT,
    F3BWhyState0Branch.MARKING_OR_DEDUP_ARTIFACT,
    F3BWhyState0Branch.STATE0_IDENTITY_STRUCTURE,
    F3BWhyState0Branch.MIXED_OR_INCONCLUSIVE,
)


def _dedup_field_status(receipt_or_inputs: Mapping[str, Any]) -> str:
    """Return dedup evidence status: ok | schema_missing | artifact."""

    if "dedup_reset_called" not in receipt_or_inputs:
        return "schema_missing"
    if "dedup_session_scope" not in receipt_or_inputs:
        return "schema_missing"

    scope = receipt_or_inputs.get("dedup_session_scope")
    if not isinstance(scope, str) or not scope.strip():
        return "schema_missing"

    reset_called = receipt_or_inputs.get("dedup_reset_called")
    if reset_called is not True:
        return "artifact"

    if scope not in VALID_DEDUP_SESSION_SCOPES:
        return "artifact"

    return "ok"


def _validate_branch_input_order_rank_consistency(
    mapping: Mapping[str, Any],
) -> list[str]:
    """Order/set/rank-map checks for branch inputs (no per_state required)."""

    failures: list[str] = []
    order = mapping.get("sampled_state_order")
    sampled_set = mapping.get("sampled_state_set")
    rank_map = mapping.get("order_rank_by_semantic_state")

    if not isinstance(order, list):
        failures.append("sampled_state_order_not_list")
        return failures
    if not isinstance(sampled_set, list):
        failures.append("sampled_state_set_not_list")
        return failures
    if not isinstance(rank_map, dict):
        failures.append("order_rank_by_semantic_state_not_dict")
        return failures

    order_ints = [int(state) for state in order]
    set_ints = {int(state) for state in sampled_set}
    if len(order_ints) != len(set(order_ints)):
        failures.append("sampled_state_order_has_duplicates")
    if set(order_ints) != set_ints:
        failures.append("sampled_state_order_set_mismatch")

    for rank, state in enumerate(order_ints):
        key = str(state)
        if key not in rank_map:
            failures.append(f"order_rank_missing_state:{state}")
            continue
        if int(rank_map[key]) != rank:
            failures.append(f"order_rank_mismatch_state:{state}")

    return failures


def _validate_order_rank_consistency(receipt: Mapping[str, Any]) -> list[str]:
    failures = _validate_branch_input_order_rank_consistency(receipt)
    per_state = receipt.get("per_state")
    if not isinstance(per_state, list):
        failures.append("per_state_not_list")
        return failures

    order = receipt.get("sampled_state_order")
    sampled_set = receipt.get("sampled_state_set")
    if not isinstance(order, list) or not isinstance(sampled_set, list):
        return failures

    order_ints = [int(state) for state in order]
    set_ints = {int(state) for state in sampled_set}

    observed_states: set[int] = set()
    for idx, row in enumerate(per_state):
        if not isinstance(row, Mapping) or "state_index" not in row:
            failures.append(f"per_state_row_{idx}_missing_state_index")
            continue
        observed_states.add(int(row["state_index"]))

    if set_ints != observed_states:
        failures.append("sampled_state_set_per_state_mismatch")
    if set(order_ints) != observed_states:
        failures.append("sampled_state_order_per_state_mismatch")

    return failures


def _branch_inputs_well_formed(branch_inputs: Mapping[str, Any]) -> bool:
    return all(field in branch_inputs for field in REQUIRED_BRANCH_INPUT_FIELDS)


def _validate_f3b_branch_value(receipt: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    branch = receipt.get("f3b_branch")
    try:
        F3BWhyState0Branch(str(branch))
    except (ValueError, TypeError):
        failures.append("f3b_branch_not_enum")
        return failures

    branch_inputs = receipt.get("f3b_branch_inputs")
    if not isinstance(branch_inputs, Mapping) or not _branch_inputs_well_formed(branch_inputs):
        return failures

    classified = classify_f3b_why_state0_branch(branch_inputs)["terminal_branch"]
    if branch != classified:
        failures.append("f3b_branch_mismatch")
    return failures


def validate_receipt_schema(receipt: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    if receipt.get("schema") != RECEIPT_SCHEMA:
        failures.append("schema_mismatch")
    for field in REQUIRED_RECEIPT_FIELDS:
        if field not in receipt:
            failures.append(f"missing:{field}")

    dedup_status = _dedup_field_status(receipt)
    if dedup_status == "schema_missing":
        if "dedup_reset_called" not in receipt:
            failures.append("missing:dedup_reset_called")
        elif receipt.get("dedup_reset_called") is not True:
            failures.append("dedup_reset_called_not_true")
        if "dedup_session_scope" not in receipt:
            failures.append("missing:dedup_session_scope")
        else:
            scope = receipt.get("dedup_session_scope")
            if not isinstance(scope, str) or not scope.strip():
                failures.append("dedup_session_scope_empty_or_invalid")
    elif dedup_status == "artifact":
        if receipt.get("dedup_reset_called") is not True:
            failures.append("dedup_reset_called_false")
        scope = receipt.get("dedup_session_scope")
        if not isinstance(scope, str) or scope not in VALID_DEDUP_SESSION_SCOPES:
            failures.append("dedup_session_scope_invalid")

    failures.extend(_validate_order_rank_consistency(receipt))

    per_state = receipt.get("per_state")
    if isinstance(per_state, list):
        for idx, row in enumerate(per_state):
            if not isinstance(row, Mapping):
                failures.append(f"per_state_row_{idx}_not_mapping")
                continue
            for field in REQUIRED_PER_STATE_FIELDS:
                if field not in row:
                    failures.append(f"per_state_row_{idx}_missing:{field}")

    branch_inputs = receipt.get("f3b_branch_inputs")
    if isinstance(branch_inputs, Mapping):
        for field in REQUIRED_BRANCH_INPUT_FIELDS:
            if field not in branch_inputs:
                failures.append(f"f3b_branch_inputs_missing:{field}")
    else:
        failures.append("f3b_branch_inputs_missing_or_not_mapping")

    if receipt.get("ready_for_main_science") is not False:
        failures.append("ready_for_main_science_not_false")
    if receipt.get("counts_as_sub2") is not False:
        failures.append("counts_as_sub2_not_false")
    if receipt.get("pre_full_stack_diagnostic") is not True:
        failures.append("pre_full_stack_diagnostic_not_true")
    failures.extend(_validate_f3b_branch_value(receipt))
    return failures


def validate_prereg_packet_schema(packet: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    if packet.get("schema") != PREREG_PACKET_SCHEMA:
        failures.append("schema_mismatch")
    for key in (
        "classifier",
        "branch_enum",
        "receipt_schema",
        "feasibility_preflight",
        "run_determinism",
        "gpu_ladder",
        "identity_order_inertness_precondition",
        "order_control_patch_scope",
        "sub2_first_launch_gate_exception",
        "claim_boundary",
    ):
        if key not in packet:
            failures.append(f"missing:{key}")
    inertness = packet.get("identity_order_inertness_precondition")
    if not isinstance(inertness, Mapping):
        failures.append("identity_order_inertness_precondition_missing")
    else:
        if inertness.get("blocks_variable_a_interpretation") is not True:
            failures.append("identity_inertness_blocks_variable_a_not_true")
        if not inertness.get("identity_order"):
            failures.append("identity_inertness_missing_identity_order")
        if not inertness.get("baseline_comparison_receipt_path"):
            failures.append("identity_inertness_missing_baseline_path")
    ladder = packet.get("gpu_ladder")
    if isinstance(ladder, Mapping):
        budget = ladder.get("budget")
        if isinstance(budget, Mapping):
            if budget.get("identity_inertness_in_fold3b_budget") is not False:
                failures.append("identity_inertness_must_not_be_in_fold3b_budget")
    receipt_schema = packet.get("receipt_schema") or {}
    packet_required = receipt_schema.get("required_fields")
    if isinstance(packet_required, list):
        if set(REQUIRED_RECEIPT_FIELDS) != set(packet_required):
            failures.append("required_receipt_fields_packet_drift")
    return failures


def validate_preflight_receipt_schema(receipt: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    if receipt.get("schema") != PREFLIGHT_RECEIPT_SCHEMA:
        failures.append("schema_mismatch")
    for key in (
        "feasibility_verdict",
        "run_determinism_classification",
        "source_trace_anchors",
        "upstream_receipt_grounding",
        "identity_order_inertness_precondition",
    ):
        if key not in receipt:
            failures.append(f"missing:{key}")
    if receipt.get("feasibility_verdict") not in {"EXISTS", "NEEDS_PATCH", "INFEASIBLE"}:
        failures.append("invalid_feasibility_verdict")
    if receipt.get("run_determinism_classification") not in {
        "DETERMINISTIC",
        "NON_DETERMINISTIC",
    }:
        failures.append("invalid_run_determinism_classification")
    return failures


def _cb_state_indices(per_state: Sequence[Mapping[str, Any]]) -> list[int]:
    indices: list[int] = []
    for row in per_state:
        crossing_len = int(row.get("crossing_indices_len") or 0)
        if crossing_len > 0:
            indices.append(int(row["state_index"]))
    return sorted(indices)


def _order_rank_consistent(inputs: Mapping[str, Any]) -> bool:
    return _validate_branch_input_order_rank_consistency(inputs) == []


def classify_f3b_why_state0_branch(
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Inert branch classifier contract (not wired into live launch acceptance)."""

    fired: list[F3BWhyState0Branch] = []

    if not bool(inputs.get("operational_ok", True)):
        fired.append(F3BWhyState0Branch.NO_VERDICT_OPERATIONAL)

    dedup_status = _dedup_field_status(inputs)
    if dedup_status == "schema_missing":
        fired.append(F3BWhyState0Branch.NO_VERDICT_SCHEMA)
    elif dedup_status == "artifact":
        fired.append(F3BWhyState0Branch.MARKING_OR_DEDUP_ARTIFACT)

    schema_ok = bool(inputs.get("schema_ok", False))
    exact_coverage = bool(inputs.get("exact_per_state_coverage", False))
    has_order = bool(inputs.get("sampled_state_order"))
    has_rank_map = bool(inputs.get("order_rank_by_semantic_state"))
    order_rank_ok = _order_rank_consistent(inputs)
    if (
        not schema_ok
        or not exact_coverage
        or not has_order
        or not has_rank_map
        or not order_rank_ok
    ):
        fired.append(F3BWhyState0Branch.NO_VERDICT_SCHEMA)

    dedup_ok = dedup_status == "ok"
    schema_gates_clear = (
        F3BWhyState0Branch.NO_VERDICT_SCHEMA not in fired
        and F3BWhyState0Branch.MARKING_OR_DEDUP_ARTIFACT not in fired
        and dedup_ok
        and order_rank_ok
    )

    variable_id = str(inputs.get("variable_id") or "")
    identity_inert = bool(inputs.get("identity_order_inertness_proven", False))
    semantic_state0_cb = bool(inputs.get("semantic_state0_is_crossing_bearing", False))
    first_measured_cb = bool(inputs.get("first_measured_is_crossing_bearing", False))
    first_measured = inputs.get("first_measured_semantic_state")
    cb_count = int(inputs.get("cb_state_count") or 0)
    sampled_set_changed = bool(inputs.get("sampled_set_changed", False))
    mark_consistent = bool(inputs.get("mark_count_consistent", True))

    if schema_gates_clear:
        if variable_id == "A_order_only" and not identity_inert:
            fired.append(F3BWhyState0Branch.MIXED_OR_INCONCLUSIVE)
        elif variable_id == "A_order_only" and identity_inert:
            first_measured_int = (
                int(first_measured) if first_measured is not None else None
            )
            if first_measured_cb and first_measured_int is not None and first_measured_int != 0:
                fired.append(F3BWhyState0Branch.MEASUREMENT_ORDER_ARTIFACT)
            elif semantic_state0_cb and cb_count == 1:
                fired.append(F3BWhyState0Branch.STATE0_IDENTITY_STRUCTURE)

        if variable_id == "B_state0_omission" and sampled_set_changed:
            if cb_count >= 1 and not semantic_state0_cb:
                fired.append(F3BWhyState0Branch.SAMPLE_SET_OR_ELIGIBILITY_ARTIFACT)

        if not mark_consistent:
            fired.append(F3BWhyState0Branch.MARKING_OR_DEDUP_ARTIFACT)

    unique: list[F3BWhyState0Branch] = []
    for branch in fired:
        if branch not in unique:
            unique.append(branch)

    if not unique:
        terminal = F3BWhyState0Branch.MIXED_OR_INCONCLUSIVE
    elif len(unique) == 1:
        terminal = unique[0]
    else:
        precedence_index = {branch: idx for idx, branch in enumerate(BRANCH_PRECEDENCE)}
        terminal = min(unique, key=lambda branch: precedence_index[branch])
        if terminal not in {
            F3BWhyState0Branch.NO_VERDICT_OPERATIONAL,
            F3BWhyState0Branch.NO_VERDICT_SCHEMA,
            F3BWhyState0Branch.MARKING_OR_DEDUP_ARTIFACT,
        }:
            terminal = F3BWhyState0Branch.MIXED_OR_INCONCLUSIVE

    return {
        "classifier": CLASSIFIER,
        "terminal_branch": terminal.value,
        "fired_branches": [branch.value for branch in unique],
        "f3b_branch_inputs": dict(inputs),
    }


def build_branch_input_contract_from_ca_receipt(
    receipt: Mapping[str, Any],
    *,
    variable_id: str,
    control_reason: str | None = None,
    identity_order_inertness_proven: bool = False,
    operational_ok: bool = True,
) -> dict[str, Any]:
    per_state = list(receipt.get("per_state") or [])
    sampled_states = list(receipt.get("sampled_state_order") or receipt.get("sampled_states") or [])
    sampled_set = list(receipt.get("sampled_state_set") or sampled_states)
    order_rank = receipt.get("order_rank_by_semantic_state")
    if not isinstance(order_rank, dict) and sampled_states:
        order_rank = {str(state): rank for rank, state in enumerate(sampled_states)}

    cb_indices = _cb_state_indices(per_state)
    first_measured = sampled_states[0] if sampled_states else None
    first_measured_row = next(
        (row for row in per_state if int(row.get("state_index", -1)) == int(first_measured)),
        None,
    ) if first_measured is not None else None

    synthetic = {
        "sampled_state_set": sampled_set,
        "sampled_state_order": sampled_states,
        "order_rank_by_semantic_state": order_rank or {},
        "per_state": per_state,
        "dedup_reset_called": receipt.get("dedup_reset_called"),
        "dedup_session_scope": receipt.get("dedup_session_scope"),
    }

    receipt_schema_failures = validate_receipt_schema(receipt)
    builder_schema_failures = [
        failure
        for failure in receipt_schema_failures
        if failure not in {"f3b_branch_not_enum", "f3b_branch_mismatch"}
        and not failure.startswith("missing:f3b_branch")
        and not failure.startswith("f3b_branch_inputs")
    ]

    return {
        "operational_ok": operational_ok,
        "schema_ok": len(builder_schema_failures) == 0,
        "sampled_state_set": sorted({int(x) for x in sampled_set}),
        "sampled_state_order": sampled_states,
        "order_rank_by_semantic_state": order_rank or {},
        "exact_per_state_coverage": _validate_order_rank_consistency(synthetic) == [],
        "dedup_reset_called": receipt.get("dedup_reset_called"),
        "dedup_session_scope": receipt.get("dedup_session_scope"),
        "identity_order_inertness_proven": identity_order_inertness_proven,
        "semantic_state0_crossing_indices_len": next(
            (
                int(row.get("crossing_indices_len") or 0)
                for row in per_state
                if int(row.get("state_index", -1)) == 0
            ),
            0,
        ),
        "cb_state_count": len(cb_indices),
        "first_measured_semantic_state": first_measured,
        "first_measured_is_crossing_bearing": bool(
            first_measured_row is not None
            and int(first_measured_row.get("crossing_indices_len") or 0) > 0
        ),
        "semantic_state0_is_crossing_bearing": 0 in cb_indices,
        "sampled_set_changed": bool(receipt.get("sampled_set_changed", False)),
        "mark_count_consistent": int(receipt.get("mark_count") or 0) == len(sampled_states),
        "variable_id": variable_id,
        "control_reason": control_reason,
    }
