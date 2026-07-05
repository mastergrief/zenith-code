"""CA-shaped Fold-3B branch-input and mechanism-receipt parity tests."""

from __future__ import annotations

from typing import Any

import pytest

from calm.hrm_text_158.native_full_stack.f3b_why_state0_branch import (
    CA_BRANCH_INPUT_SOURCE_SCHEMA,
    DECISIVE_F3B_BRANCHES,
    F3BWhyState0Branch,
    RECEIPT_SCHEMA,
    build_branch_input_contract_from_ca_receipt,
    classify_f3b_why_state0_branch,
    normalize_per_state_for_mechanism_receipt,
    validate_ca_branch_input_source,
    validate_receipt_schema,
)

PARENT_SHA = "9b4e311a22787e7d4808bde7bc2953568d767a2ee8ac648942a3f5dbb7b4d5ec"
DENSE_SAMPLED = list(range(10))
DENSE_ORDER_REVERSED = list(range(9, -1, -1))
EXPECTED_EFFECTIVE = DENSE_ORDER_REVERSED + list(range(10, 32))


def _ca_per_state_row(
    state_index: int,
    *,
    crossing_indices_len: int = 0,
) -> dict[str, Any]:
    """Mirror extract_band_counter_per_state_rows_from_marks (slice5 :6431-6463)."""
    band_a = 22640 if crossing_indices_len > 0 else 112
    band_c = 48640 if crossing_indices_len > 0 else 0
    band_e = 5408 if crossing_indices_len > 0 else 0
    state_total = band_a + band_c + band_e
    per_cb_ca_share = (
        float(band_a + band_c) / float(state_total)
        if crossing_indices_len > 0 and state_total > 0
        else None
    )
    return {
        "state_index": state_index,
        "semantic_state_id": state_index,
        "band_a_bytes": band_a,
        "band_c_bytes": band_c,
        "band_e_bytes": band_e,
        "crossing_indices_len": crossing_indices_len,
        "per_cb_ca_share": per_cb_ca_share,
        "is_crossing_bearing": crossing_indices_len > 0,
    }


def _minimal_valid_ca_confirmation_receipt(
    *,
    reversed_order: bool = True,
    cb_state_index: int = 0,
    crossing_len: int = 512,
) -> dict[str, Any]:
    order = list(DENSE_ORDER_REVERSED if reversed_order else DENSE_SAMPLED)
    rank_map = {str(state): rank for rank, state in enumerate(order)}
    per_state = [
        _ca_per_state_row(
            state,
            crossing_indices_len=crossing_len if state == cb_state_index else 0,
        )
        for state in range(10)
    ]
    return {
        "schema": CA_BRANCH_INPUT_SOURCE_SCHEMA,
        "sampled_states": list(DENSE_SAMPLED),
        "sampled_state_set": list(DENSE_SAMPLED),
        "sampled_state_order": order,
        "order_rank_by_semantic_state": rank_map,
        "order_control_active": True,
        "order_perturbation_kind": "sampled_block_order_perturbation",
        "effective_visit_order": list(EXPECTED_EFFECTIVE),
        "per_state": per_state,
        "mark_count": 10,
        "s1d7_band_counter_mark_count": 10,
        "dedup_reset_called": True,
        "dedup_session_scope": "probe_subprocess",
        "parent_sha": PARENT_SHA,
        "infra_ok": True,
        "ok": True,
        "checks": {
            "s1d7_band_counter_mark_count_eq_sampled_state_count": True,
        },
    }


def _classify_ca_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    inputs = build_branch_input_contract_from_ca_receipt(
        receipt,
        variable_id="A_order_only",
        control_reason="order_only_perturbation",
        identity_order_inertness_proven=True,
    )
    return classify_f3b_why_state0_branch(inputs)


def test_ca_source_validator_passes_real_shape() -> None:
    receipt = _minimal_valid_ca_confirmation_receipt()
    failures = validate_ca_branch_input_source(receipt)
    assert failures == []


def test_builder_to_classify_identity_structure_from_ca_shape() -> None:
    receipt = _minimal_valid_ca_confirmation_receipt(
        reversed_order=True,
        cb_state_index=0,
    )
    result = _classify_ca_receipt(receipt)
    assert result["terminal_branch"] == F3BWhyState0Branch.STATE0_IDENTITY_STRUCTURE.value
    inputs = result["f3b_branch_inputs"]
    assert inputs["schema_ok"] is True
    assert inputs["ca_source_schema_failures"] == []


def test_builder_to_classify_measurement_order_artifact_from_ca_shape() -> None:
    receipt = _minimal_valid_ca_confirmation_receipt(
        reversed_order=True,
        cb_state_index=9,
    )
    result = _classify_ca_receipt(receipt)
    assert result["terminal_branch"] == (
        F3BWhyState0Branch.MEASUREMENT_ORDER_ARTIFACT.value
    )
    inputs = result["f3b_branch_inputs"]
    assert inputs["schema_ok"] is True
    assert inputs["ca_source_schema_failures"] == []


@pytest.mark.parametrize(
    ("mutator", "expected_failure_substr"),
    [
        (lambda r: r.pop("parent_sha"), "ca_missing_parent_sha"),
        (lambda r: r.update({"dedup_reset_called": False}), "ca_dedup_artifact"),
        (lambda r: r.pop("dedup_reset_called"), "ca_missing:dedup_reset_called"),
        (
            lambda r: r.update(
                {"order_rank_by_semantic_state": {**r["order_rank_by_semantic_state"], "9": 5}}
            ),
            "order_rank_mismatch_state:9",
        ),
        (lambda r: r["per_state"].pop(), "sampled_state_set_per_state_mismatch"),
        (lambda r: r["per_state"][0].pop("crossing_indices_len"), "ca_per_state_row_0_missing_crossing_indices_len"),
    ],
)
def test_ca_source_adversaries_fail_closed(
    mutator,
    expected_failure_substr: str,
) -> None:
    receipt = _minimal_valid_ca_confirmation_receipt()
    mutator(receipt)
    failures = validate_ca_branch_input_source(receipt)
    assert any(expected_failure_substr in failure for failure in failures)

    result = _classify_ca_receipt(receipt)
    assert result["terminal_branch"] not in DECISIVE_F3B_BRANCHES
    inputs = result["f3b_branch_inputs"]
    assert inputs["schema_ok"] is False
    assert any(expected_failure_substr in failure for failure in inputs["ca_source_schema_failures"])


@pytest.mark.parametrize(
    ("mutator", "expected_failure_substr"),
    [
        (lambda r: r["per_state"][0].__setitem__("crossing_indices_len", "bad"), "crossing_indices_len_malformed"),
        (lambda r: r.__setitem__("mark_count", "bad"), "ca_mark_count_malformed"),
        (lambda r: r["per_state"][0].__setitem__("state_index", "bad"), "state_index_malformed"),
        (lambda r: r["sampled_state_order"].__setitem__(0, "bad"), "sampled_state_order_malformed"),
        (lambda r: r["sampled_state_set"].__setitem__(0, "bad"), "sampled_state_set_malformed"),
        (
            lambda r: r["order_rank_by_semantic_state"].__setitem__("9", "bad"),
            "order_rank_value_malformed:9",
        ),
    ],
)
def test_ca_source_malformed_numeric_fields_fail_closed_without_exception(
    mutator,
    expected_failure_substr: str,
) -> None:
    receipt = _minimal_valid_ca_confirmation_receipt()
    mutator(receipt)

    failures = validate_ca_branch_input_source(receipt)
    assert any(expected_failure_substr in failure for failure in failures)

    inputs = build_branch_input_contract_from_ca_receipt(
        receipt,
        variable_id="A_order_only",
        control_reason="order_only_perturbation",
        identity_order_inertness_proven=True,
    )
    assert inputs["schema_ok"] is False
    assert any(expected_failure_substr in failure for failure in inputs["ca_source_schema_failures"])

    result = classify_f3b_why_state0_branch(inputs)
    assert result["terminal_branch"] not in DECISIVE_F3B_BRANCHES
    assert result["terminal_branch"] == F3BWhyState0Branch.NO_VERDICT_SCHEMA.value


@pytest.mark.parametrize(
    ("mutator", "expected_failure_substr"),
    [
        (
            lambda r: r["per_state"][0].__setitem__("crossing_indices_len", True),
            "crossing_indices_len_malformed",
        ),
        (
            lambda r: r["per_state"][0].__setitem__("crossing_indices_len", 3.14),
            "crossing_indices_len_malformed",
        ),
        (lambda r: r.__setitem__("mark_count", 10.0), "ca_mark_count_malformed"),
        (
            lambda r: r["sampled_state_order"].__setitem__(0, 9.0),
            "sampled_state_order_malformed",
        ),
        (
            lambda r: r["sampled_state_set"].__setitem__(0, 0.0),
            "ca_sampled_state_set_malformed",
        ),
        (
            lambda r: r["order_rank_by_semantic_state"].__setitem__("9", 3.14),
            "order_rank_value_malformed:9",
        ),
        (
            lambda r: r["order_rank_by_semantic_state"].__setitem__("9", True),
            "order_rank_value_malformed:9",
        ),
    ],
)
def test_ca_source_strict_integer_false_green_state0_row_regressions(
    mutator,
    expected_failure_substr: str,
) -> None:
    """Bool/float on state0 (or peer int fields) must NOT yield decisive identity."""

    receipt = _minimal_valid_ca_confirmation_receipt(
        reversed_order=True,
        cb_state_index=0,
        crossing_len=512,
    )
    mutator(receipt)

    failures = validate_ca_branch_input_source(receipt)
    assert any(expected_failure_substr in failure for failure in failures)

    inputs = build_branch_input_contract_from_ca_receipt(
        receipt,
        variable_id="A_order_only",
        control_reason="order_only_perturbation",
        identity_order_inertness_proven=True,
    )
    assert inputs["schema_ok"] is False
    assert any(expected_failure_substr in failure for failure in inputs["ca_source_schema_failures"])

    result = classify_f3b_why_state0_branch(inputs)
    assert result["terminal_branch"] not in DECISIVE_F3B_BRANCHES
    assert result["terminal_branch"] == F3BWhyState0Branch.NO_VERDICT_SCHEMA.value


@pytest.mark.parametrize(
    ("mutator", "expected_failure_substr"),
    [
        (lambda r: r.__setitem__("sampled_state_order", 1), "sampled_state_order_not_a_list"),
        (lambda r: r.__setitem__("sampled_state_set", 1), "sampled_state_set_not_a_list"),
        (
            lambda r: (
                r.pop("sampled_state_order"),
                r.__setitem__("sampled_states", 1),
            ),
            "sampled_states_not_a_list",
        ),
        (lambda r: r.__setitem__("sampled_state_order", True), "sampled_state_order_not_a_list"),
        (lambda r: r.__setitem__("sampled_state_set", 3.14), "sampled_state_set_not_a_list"),
        (lambda r: r.__setitem__("per_state", 1), "ca_per_state_not_list"),
    ],
)
def test_ca_source_non_iterable_sequence_fields_fail_closed_without_exception(
    mutator,
    expected_failure_substr: str,
) -> None:
    receipt = _minimal_valid_ca_confirmation_receipt()
    mutator(receipt)

    failures = validate_ca_branch_input_source(receipt)
    assert any(expected_failure_substr in failure for failure in failures)

    inputs = build_branch_input_contract_from_ca_receipt(
        receipt,
        variable_id="A_order_only",
        control_reason="order_only_perturbation",
        identity_order_inertness_proven=True,
    )
    assert inputs["schema_ok"] is False
    assert any(expected_failure_substr in failure for failure in inputs["ca_source_schema_failures"])

    result = classify_f3b_why_state0_branch(inputs)
    assert result["terminal_branch"] not in DECISIVE_F3B_BRANCHES
    assert result["terminal_branch"] == F3BWhyState0Branch.NO_VERDICT_SCHEMA.value


def test_mechanism_receipt_normalized_from_ca_passes_validate_receipt_schema() -> None:
    ca_receipt = _minimal_valid_ca_confirmation_receipt()
    classified = _classify_ca_receipt(ca_receipt)
    normalized_per_state = normalize_per_state_for_mechanism_receipt(ca_receipt["per_state"])
    mechanism_receipt = {
        "schema": RECEIPT_SCHEMA,
        "sampled_state_set": ca_receipt["sampled_state_set"],
        "sampled_state_order": ca_receipt["sampled_state_order"],
        "order_rank_by_semantic_state": ca_receipt["order_rank_by_semantic_state"],
        "semantic_state_id": [row["semantic_state_id"] for row in normalized_per_state],
        "per_state": normalized_per_state,
        "dedup_reset_called": ca_receipt["dedup_reset_called"],
        "dedup_session_scope": ca_receipt["dedup_session_scope"],
        "wrapper_path": "/valid/wrapper.json",
        "primary_receipt_path": "/valid/primary.json",
        "fallback_receipt_path": None,
        "science_verdict_source": "primary",
        "parent_sha": ca_receipt["parent_sha"],
        "git_head_required": "bd23cc9",
        "variable_id": "A_order_only",
        "control_reason": "order_only_perturbation",
        "f3b_branch": classified["terminal_branch"],
        "f3b_branch_inputs": classified["f3b_branch_inputs"],
        "ready_for_main_science": False,
        "counts_as_sub2": False,
        "pre_full_stack_diagnostic": True,
    }
    failures = validate_receipt_schema(mechanism_receipt)
    assert failures == []
