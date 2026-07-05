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
CA_BRANCH_INPUT_SOURCE_SCHEMA = (
    "hrm_text_158_callsite_band_counter_ca_confirmation_receipt/v1"
)
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
    "ca_source_schema_failures",
)


class F3BWhyState0Branch(StrEnum):
    NO_VERDICT_OPERATIONAL = "F3B_NO_VERDICT_OPERATIONAL"
    NO_VERDICT_SCHEMA = "F3B_NO_VERDICT_SCHEMA"
    MEASUREMENT_ORDER_ARTIFACT = "F3B_MEASUREMENT_ORDER_ARTIFACT"
    SAMPLE_SET_OR_ELIGIBILITY_ARTIFACT = "F3B_SAMPLE_SET_OR_ELIGIBILITY_ARTIFACT"
    MARKING_OR_DEDUP_ARTIFACT = "F3B_MARKING_OR_DEDUP_ARTIFACT"
    STATE0_IDENTITY_STRUCTURE = "F3B_STATE0_IDENTITY_STRUCTURE"
    MIXED_OR_INCONCLUSIVE = "F3B_MIXED_OR_INCONCLUSIVE"


DECISIVE_F3B_BRANCHES: frozenset[str] = frozenset(
    {
        F3BWhyState0Branch.STATE0_IDENTITY_STRUCTURE.value,
        F3BWhyState0Branch.MEASUREMENT_ORDER_ARTIFACT.value,
    }
)


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


def _coerce_ca_int(value: Any) -> int | None:
    """Strict CA integer coercion: accept only genuine ``int`` values.

    Policy (Option B): ``type(value) is int`` — excludes ``bool`` (a subclass of
    ``int`` in Python), ``float``, ``str``, and all other types. No silent
    ``int()`` truncation or truthy coercion on external CA numerics.
    """

    if type(value) is int:
        return value
    return None


def _coerce_int_sequence(
    values: Sequence[Any],
    *,
    malformed_code: str,
) -> tuple[list[int] | None, list[str]]:
    ints: list[int] = []
    for value in values:
        coerced = _coerce_ca_int(value)
        if coerced is None:
            return None, [malformed_code]
        ints.append(coerced)
    return ints, []


def _coerce_int_set(
    values: Sequence[Any],
    *,
    malformed_code: str,
) -> tuple[set[int] | None, list[str]]:
    ints: set[int] = set()
    for value in values:
        coerced = _coerce_ca_int(value)
        if coerced is None:
            return None, [malformed_code]
        ints.add(coerced)
    return ints, []


def _safe_sequence(
    value: Any,
    *,
    not_sequence_code: str,
) -> tuple[list[Any] | None, list[str]]:
    """Convert an external CA sequence field to list without raising.

    Only list/tuple are accepted; truthy non-iterables (int, bool, str, dict, …)
    return ``not_sequence_code`` instead of raising TypeError.
    """

    if value is None:
        return [], []
    if isinstance(value, (list, tuple)):
        return list(value), []
    return None, [not_sequence_code]


def _extract_sampled_states_from_ca_receipt(
    receipt: Mapping[str, Any],
) -> tuple[list[Any] | None, list[str]]:
    """Mirror ``sampled_state_order or sampled_states or []`` without raising."""

    order_val = receipt.get("sampled_state_order")
    states_val = receipt.get("sampled_states")
    if order_val:
        return _safe_sequence(order_val, not_sequence_code="sampled_state_order_not_a_list")
    if states_val:
        return _safe_sequence(states_val, not_sequence_code="sampled_states_not_a_list")
    return [], []


def _extract_sampled_state_set_from_ca_receipt(
    receipt: Mapping[str, Any],
    sampled_states: Sequence[Any],
) -> tuple[list[Any] | None, list[str]]:
    """Mirror ``sampled_state_set or sampled_states`` without raising."""

    set_val = receipt.get("sampled_state_set")
    if set_val:
        return _safe_sequence(set_val, not_sequence_code="sampled_state_set_not_a_list")
    return list(sampled_states), []


def _extract_per_state_from_ca_receipt(
    receipt: Mapping[str, Any],
) -> tuple[list[Any], list[str]]:
    """Extract per_state rows without raising on non-iterable values."""

    per_state = receipt.get("per_state")
    if per_state is None:
        return [], []
    rows, failures = _safe_sequence(per_state, not_sequence_code="ca_per_state_not_a_list")
    if failures:
        return [], failures
    return rows or [], []


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

    order_ints, order_failures = _coerce_int_sequence(
        order,
        malformed_code="sampled_state_order_malformed",
    )
    failures.extend(order_failures)
    if order_ints is None:
        return failures

    set_ints, set_failures = _coerce_int_set(
        sampled_set,
        malformed_code="sampled_state_set_malformed",
    )
    failures.extend(set_failures)
    if set_ints is None:
        return failures

    if len(order_ints) != len(set(order_ints)):
        failures.append("sampled_state_order_has_duplicates")
    if set(order_ints) != set_ints:
        failures.append("sampled_state_order_set_mismatch")

    for rank, state in enumerate(order_ints):
        key = str(state)
        if key not in rank_map:
            failures.append(f"order_rank_missing_state:{state}")
            continue
        rank_value = _coerce_ca_int(rank_map[key])
        if rank_value is None:
            failures.append(f"order_rank_value_malformed:{key}")
            continue
        if rank_value != rank:
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

    if any(
        code in failures
        for code in (
            "sampled_state_order_malformed",
            "sampled_state_set_malformed",
            "order_rank_by_semantic_state_not_dict",
            "sampled_state_order_not_list",
            "sampled_state_set_not_list",
        )
    ):
        return failures

    order_ints, order_failures = _coerce_int_sequence(
        order,
        malformed_code="sampled_state_order_malformed",
    )
    failures.extend(order_failures)
    if order_ints is None:
        return failures

    set_ints, set_failures = _coerce_int_set(
        sampled_set,
        malformed_code="sampled_state_set_malformed",
    )
    failures.extend(set_failures)
    if set_ints is None:
        return failures

    observed_states: set[int] = set()
    for idx, row in enumerate(per_state):
        if not isinstance(row, Mapping) or "state_index" not in row:
            failures.append(f"per_state_row_{idx}_missing_state_index")
            continue
        state_index = _coerce_ca_int(row["state_index"])
        if state_index is None:
            failures.append(f"per_state_row_{idx}_state_index_malformed")
            continue
        observed_states.add(state_index)

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
    branch = receipt.get("f3b_branch")
    if branch in DECISIVE_F3B_BRANCHES:
        parent_sha = receipt.get("parent_sha")
        if not isinstance(parent_sha, str) or not parent_sha.strip():
            failures.append("missing:parent_sha_for_decisive_branch")
    failures.extend(_validate_f3b_branch_value(receipt))
    return failures


def validate_ca_branch_input_source(receipt: Mapping[str, Any]) -> list[str]:
    """Validate a raw CA confirmation receipt for Fold-3B branch-input construction."""

    failures: list[str] = []
    if receipt.get("schema") != CA_BRANCH_INPUT_SOURCE_SCHEMA:
        failures.append("ca_schema_mismatch")

    if receipt.get("infra_ok") is not True:
        failures.append("ca_infra_not_ok")
    if receipt.get("ok") is not True:
        failures.append("ca_ok_not_true")

    if "order_control_active" in receipt and receipt.get("order_control_active") is not True:
        failures.append("ca_order_control_inactive")

    per_state = receipt.get("per_state")
    if not isinstance(per_state, list):
        failures.append("ca_per_state_not_list")
    else:
        sampled_states, states_failures = _extract_sampled_states_from_ca_receipt(receipt)
        failures.extend(states_failures)
        if sampled_states is None:
            sampled_states = []
        sampled_set_raw, set_list_failures = _extract_sampled_state_set_from_ca_receipt(
            receipt,
            sampled_states,
        )
        failures.extend(set_list_failures)
        if sampled_set_raw is None:
            sampled_set_raw = []
        parsed_set, set_failures = _coerce_int_set(
            sampled_set_raw,
            malformed_code="ca_sampled_state_set_malformed",
        )
        failures.extend(set_failures)
        sampled_set = parsed_set if parsed_set is not None else set()
        for idx, row in enumerate(per_state):
            if not isinstance(row, Mapping):
                failures.append(f"ca_per_state_row_{idx}_not_mapping")
                continue
            if "state_index" not in row:
                failures.append(f"ca_per_state_row_{idx}_missing_state_index")
                continue
            state_index = _coerce_ca_int(row["state_index"])
            if state_index is None:
                failures.append(f"ca_per_state_row_{idx}_state_index_malformed")
                continue
            if state_index in sampled_set:
                if "semantic_state_id" not in row:
                    failures.append(f"ca_per_state_row_{idx}_missing_semantic_state_id")
                else:
                    semantic_id = _coerce_ca_int(row["semantic_state_id"])
                    if semantic_id is None:
                        failures.append(f"ca_per_state_row_{idx}_semantic_state_id_malformed")
                    elif semantic_id != state_index:
                        failures.append(f"ca_per_state_row_{idx}_semantic_state_id_mismatch")
            if "crossing_indices_len" not in row:
                failures.append(f"ca_per_state_row_{idx}_missing_crossing_indices_len")
            else:
                crossing_len = _coerce_ca_int(row["crossing_indices_len"])
                if crossing_len is None:
                    failures.append(f"ca_per_state_row_{idx}_crossing_indices_len_malformed")
                elif crossing_len < 0:
                    failures.append(f"ca_per_state_row_{idx}_crossing_indices_len_negative")

    failures.extend(_validate_order_rank_consistency(receipt))

    dedup_status = _dedup_field_status(receipt)
    if dedup_status == "schema_missing":
        if "dedup_reset_called" not in receipt:
            failures.append("ca_missing:dedup_reset_called")
        elif receipt.get("dedup_reset_called") is not True:
            failures.append("ca_dedup_reset_called_not_true")
        if "dedup_session_scope" not in receipt:
            failures.append("ca_missing:dedup_session_scope")
        else:
            scope = receipt.get("dedup_session_scope")
            if not isinstance(scope, str) or not scope.strip():
                failures.append("ca_dedup_session_scope_empty_or_invalid")
    elif dedup_status == "artifact":
        failures.append("ca_dedup_artifact")

    parent_sha = receipt.get("parent_sha")
    if not isinstance(parent_sha, str) or not parent_sha.strip():
        failures.append("ca_missing_parent_sha")

    sampled_states, states_failures = _extract_sampled_states_from_ca_receipt(receipt)
    failures.extend(states_failures)
    if sampled_states is None:
        sampled_states = []
    mark_count = receipt.get("mark_count")
    if mark_count is None:
        failures.append("ca_missing_mark_count")
    else:
        parsed_mark_count = _coerce_ca_int(mark_count)
        if parsed_mark_count is None:
            failures.append("ca_mark_count_malformed")
        elif parsed_mark_count != len(sampled_states):
            failures.append("ca_mark_count_mismatch")

    return failures


def normalize_per_state_for_mechanism_receipt(
    ca_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Derive mechanism-receipt per_state rows from CA-native band-counter rows.

    Pure field derivations only — no invented measurements:
    - crossing_indices_len: canonical CA field (unchanged).
    - crossing_count: alias identical to crossing_indices_len (the count IS the
      index-list length emitted by the band counter).
    - mark_count: 1 per sampled-state row (one s1d7_band_counter mark per semantic
      state; consistent with s1d7_band_counter_mark_count == sampled_state_count).
    """

    normalized: list[dict[str, Any]] = []
    for row in ca_rows:
        if not isinstance(row, Mapping):
            continue
        out = dict(row)
        crossing_len = int(row.get("crossing_indices_len") or 0)
        out["crossing_indices_len"] = crossing_len
        out["crossing_count"] = crossing_len
        out["mark_count"] = 1
        normalized.append(out)
    return normalized


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
        crossing_len = _coerce_ca_int(row.get("crossing_indices_len"))
        if crossing_len is None or crossing_len <= 0:
            continue
        state_index = _coerce_ca_int(row.get("state_index"))
        if state_index is None:
            continue
        indices.append(state_index)
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
    cb_count_raw = _coerce_ca_int(inputs.get("cb_state_count"))
    cb_count = cb_count_raw if cb_count_raw is not None else 0
    sampled_set_changed = bool(inputs.get("sampled_set_changed", False))
    mark_consistent = bool(inputs.get("mark_count_consistent", True))

    if schema_gates_clear:
        if variable_id == "A_order_only" and not identity_inert:
            fired.append(F3BWhyState0Branch.MIXED_OR_INCONCLUSIVE)
        elif variable_id == "A_order_only" and identity_inert:
            first_measured_int = _coerce_ca_int(first_measured)
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


def _schema_failed_branch_input_contract(
    receipt: Mapping[str, Any],
    *,
    receipt_schema_failures: list[str],
    variable_id: str,
    control_reason: str | None,
    identity_order_inertness_proven: bool,
    operational_ok: bool,
) -> dict[str, Any]:
    """Fail-closed branch inputs when CA-source validation fails (no int() on bad rows)."""

    sampled_states, states_failures = _extract_sampled_states_from_ca_receipt(receipt)
    if sampled_states is None:
        sampled_states = []
    sampled_set_raw, set_list_failures = _extract_sampled_state_set_from_ca_receipt(
        receipt,
        sampled_states,
    )
    if sampled_set_raw is None:
        sampled_set_raw = []
    order_rank = receipt.get("order_rank_by_semantic_state")
    if not isinstance(order_rank, dict) and sampled_states:
        order_rank = {str(state): rank for rank, state in enumerate(sampled_states)}
    sampled_set_sorted: list[int] = []
    for raw in sampled_set_raw:
        coerced = _coerce_ca_int(raw)
        if coerced is None:
            sampled_set_sorted = []
            break
        sampled_set_sorted.append(coerced)
    else:
        sampled_set_sorted = sorted(set(sampled_set_sorted))
    first_measured: Any = None
    if sampled_states:
        first_measured = _coerce_ca_int(sampled_states[0])
    return {
        "operational_ok": operational_ok,
        "schema_ok": False,
        "ca_source_schema_failures": list(receipt_schema_failures),
        "sampled_state_set": sampled_set_sorted,
        "sampled_state_order": sampled_states,
        "order_rank_by_semantic_state": order_rank if isinstance(order_rank, dict) else {},
        "exact_per_state_coverage": False,
        "dedup_reset_called": receipt.get("dedup_reset_called"),
        "dedup_session_scope": receipt.get("dedup_session_scope"),
        "identity_order_inertness_proven": identity_order_inertness_proven,
        "semantic_state0_crossing_indices_len": 0,
        "cb_state_count": 0,
        "first_measured_semantic_state": first_measured,
        "first_measured_is_crossing_bearing": False,
        "semantic_state0_is_crossing_bearing": False,
        "sampled_set_changed": bool(receipt.get("sampled_set_changed", False)),
        "mark_count_consistent": False,
        "variable_id": variable_id,
        "control_reason": control_reason,
    }


def build_branch_input_contract_from_ca_receipt(
    receipt: Mapping[str, Any],
    *,
    variable_id: str,
    control_reason: str | None = None,
    identity_order_inertness_proven: bool = False,
    operational_ok: bool = True,
) -> dict[str, Any]:
    receipt_schema_failures = validate_ca_branch_input_source(receipt)
    if receipt_schema_failures:
        return _schema_failed_branch_input_contract(
            receipt,
            receipt_schema_failures=receipt_schema_failures,
            variable_id=variable_id,
            control_reason=control_reason,
            identity_order_inertness_proven=identity_order_inertness_proven,
            operational_ok=operational_ok,
        )

    per_state, _per_state_failures = _extract_per_state_from_ca_receipt(receipt)
    sampled_states, _states_failures = _extract_sampled_states_from_ca_receipt(receipt)
    if sampled_states is None:
        sampled_states = []
    sampled_set, _set_failures = _extract_sampled_state_set_from_ca_receipt(
        receipt,
        sampled_states,
    )
    if sampled_set is None:
        sampled_set = []
    order_rank = receipt.get("order_rank_by_semantic_state")
    if not isinstance(order_rank, dict) and sampled_states:
        order_rank = {str(state): rank for rank, state in enumerate(sampled_states)}

    cb_indices = _cb_state_indices(per_state)
    first_measured = _coerce_ca_int(sampled_states[0]) if sampled_states else None
    first_measured_row = next(
        (
            row
            for row in per_state
            if _coerce_ca_int(row.get("state_index")) == first_measured
        ),
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

    return {
        "operational_ok": operational_ok,
        "schema_ok": True,
        "ca_source_schema_failures": [],
        "sampled_state_set": sorted(
            {
                coerced
                for x in sampled_set
                if (coerced := _coerce_ca_int(x)) is not None
            }
        ),
        "sampled_state_order": sampled_states,
        "order_rank_by_semantic_state": order_rank or {},
        "exact_per_state_coverage": _validate_order_rank_consistency(synthetic) == [],
        "dedup_reset_called": receipt.get("dedup_reset_called"),
        "dedup_session_scope": receipt.get("dedup_session_scope"),
        "identity_order_inertness_proven": identity_order_inertness_proven,
        "semantic_state0_crossing_indices_len": next(
            (
                crossing_len
                for row in per_state
                if _coerce_ca_int(row.get("state_index")) == 0
                and (crossing_len := _coerce_ca_int(row.get("crossing_indices_len")))
                is not None
            ),
            0,
        ),
        "cb_state_count": len(cb_indices),
        "first_measured_semantic_state": first_measured,
        "first_measured_is_crossing_bearing": bool(
            first_measured_row is not None
            and (
                (crossing_len := _coerce_ca_int(
                    first_measured_row.get("crossing_indices_len")
                ))
                is not None
                and crossing_len > 0
            )
        ),
        "semantic_state0_is_crossing_bearing": 0 in cb_indices,
        "sampled_set_changed": bool(receipt.get("sampled_set_changed", False)),
        "mark_count_consistent": (
            (parsed_mark := _coerce_ca_int(receipt.get("mark_count"))) is not None
            and parsed_mark == len(sampled_states)
        ),
        "variable_id": variable_id,
        "control_reason": control_reason,
    }
