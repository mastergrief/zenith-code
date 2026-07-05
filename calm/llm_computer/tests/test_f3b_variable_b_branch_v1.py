"""Variable B (state0-omission) classifier branch tests — real CA-shape fixtures."""

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
    validate_receipt_schema,
)

PARENT_SHA = "9b4e311a22787e7d4808bde7bc2953568d767a2ee8ac648942a3f5dbb7b4d5ec"
B_SAMPLED = list(range(1, 11))
B_ORDER = list(range(1, 11))
EXPECTED_EFFECTIVE = B_ORDER + list(range(11, 32))


def _ca_per_state_row(
    state_index: int,
    *,
    crossing_indices_len: int = 0,
) -> dict[str, Any]:
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


def _minimal_valid_b_ca_receipt(
    *,
    cb_state_index: int | None = None,
    crossing_len: int = 512,
    sampled_set: list[int] | None = None,
    sampled_order: list[int] | None = None,
    sampled_set_changed: bool = True,
) -> dict[str, Any]:
    sampled_set = list(sampled_set if sampled_set is not None else B_SAMPLED)
    sampled_order = list(sampled_order if sampled_order is not None else B_ORDER)
    rank_map = {str(state): rank for rank, state in enumerate(sampled_order)}
    per_state = [
        _ca_per_state_row(
            state,
            crossing_indices_len=crossing_len if state == cb_state_index else 0,
        )
        for state in sampled_set
    ]
    return {
        "schema": CA_BRANCH_INPUT_SOURCE_SCHEMA,
        "sampled_states": sampled_set,
        "sampled_state_set": sampled_set,
        "sampled_state_order": sampled_order,
        "sampled_set_changed": sampled_set_changed,
        "order_rank_by_semantic_state": rank_map,
        "order_control_active": True,
        "order_perturbation_kind": "sampled_block_order_perturbation",
        "effective_visit_order": list(EXPECTED_EFFECTIVE),
        "per_state": per_state,
        "mark_count": len(sampled_set),
        "s1d7_band_counter_mark_count": len(sampled_set),
        "dedup_reset_called": True,
        "dedup_session_scope": "probe_subprocess",
        "parent_sha": PARENT_SHA,
        "infra_ok": True,
        "ok": True,
        "checks": {
            "s1d7_band_counter_mark_count_eq_sampled_state_count": True,
        },
    }


def _classify_b_ca_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    inputs = build_branch_input_contract_from_ca_receipt(
        receipt,
        variable_id="B_state0_omission",
        control_reason="state0_omission_or_shifted_set",
        identity_order_inertness_proven=True,
    )
    return classify_f3b_why_state0_branch(inputs)


def _mechanism_receipt_from_ca(
    ca_receipt: dict[str, Any],
    *,
    f3b_branch: str | None = None,
) -> dict[str, Any]:
    classified = _classify_b_ca_receipt(ca_receipt)
    branch = f3b_branch if f3b_branch is not None else classified["terminal_branch"]
    normalized_per_state = (
        normalize_per_state_for_mechanism_receipt(ca_receipt["per_state"])
        if classified["f3b_branch_inputs"].get("schema_ok")
        else []
    )
    return {
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
        "git_head_required": "feb708f0a0533ee73370dfc73c100c213eb05849",
        "variable_id": "B_state0_omission",
        "control_reason": "state0_omission_or_shifted_set",
        "f3b_branch": branch,
        "f3b_branch_inputs": classified["f3b_branch_inputs"],
        "ready_for_main_science": False,
        "counts_as_sub2": False,
        "pre_full_stack_diagnostic": True,
    }


def test_variable_b_zero_cb_corroborated_by_omission() -> None:
    receipt = _minimal_valid_b_ca_receipt(cb_state_index=None)
    result = _classify_b_ca_receipt(receipt)
    assert result["terminal_branch"] == (
        F3BWhyState0Branch.STATE0_IDENTITY_CORROBORATED_BY_OMISSION.value
    )
    assert result["terminal_branch"] in DECISIVE_F3B_BRANCHES
    inputs = result["f3b_branch_inputs"]
    assert inputs["schema_ok"] is True
    assert inputs["sampled_set_changed"] is True
    assert inputs["cb_state_count"] == 0
    assert inputs["semantic_state0_is_crossing_bearing"] is False
    assert 0 not in inputs["sampled_state_set"]


def test_variable_b_replacement_sole_cb_sample_set_artifact() -> None:
    receipt = _minimal_valid_b_ca_receipt(cb_state_index=10, crossing_len=512)
    result = _classify_b_ca_receipt(receipt)
    assert result["terminal_branch"] == (
        F3BWhyState0Branch.SAMPLE_SET_OR_ELIGIBILITY_ARTIFACT.value
    )
    assert result["f3b_branch_inputs"]["cb_state_count"] == 1


def test_variable_b_set_including_zero_zero_cb_not_corroborated() -> None:
    receipt = _minimal_valid_b_ca_receipt(
        cb_state_index=None,
        sampled_set=list(range(10)),
        sampled_order=list(range(10)),
    )
    result = _classify_b_ca_receipt(receipt)
    assert result["terminal_branch"] not in DECISIVE_F3B_BRANCHES
    assert result["terminal_branch"] == (
        F3BWhyState0Branch.MIXED_OR_INCONCLUSIVE.value
    )


def test_variable_b_cb_count_gt_one_non_decisive() -> None:
    receipt = _minimal_valid_b_ca_receipt(cb_state_index=None)
    receipt["per_state"][0]["crossing_indices_len"] = 100
    receipt["per_state"][1]["crossing_indices_len"] = 200
    result = _classify_b_ca_receipt(receipt)
    assert result["terminal_branch"] not in DECISIVE_F3B_BRANCHES
    assert result["terminal_branch"] == (
        F3BWhyState0Branch.MIXED_OR_INCONCLUSIVE.value
    )


def test_variable_b_sampled_set_changed_missing_schema_fail() -> None:
    receipt = _minimal_valid_b_ca_receipt(cb_state_index=None)
    receipt["sampled_set_changed"] = False
    result = _classify_b_ca_receipt(receipt)
    assert result["terminal_branch"] == F3BWhyState0Branch.MIXED_OR_INCONCLUSIVE.value
    assert result["terminal_branch"] not in DECISIVE_F3B_BRANCHES


def test_variable_b_malformed_crossing_schema_fail() -> None:
    receipt = _minimal_valid_b_ca_receipt(cb_state_index=None)
    receipt["per_state"][0]["crossing_indices_len"] = True
    result = _classify_b_ca_receipt(receipt)
    assert result["terminal_branch"] == F3BWhyState0Branch.NO_VERDICT_SCHEMA.value
    assert result["terminal_branch"] not in DECISIVE_F3B_BRANCHES


def test_mechanism_receipt_schema_accepts_corroboration_enum() -> None:
    receipt = _minimal_valid_b_ca_receipt(cb_state_index=None)
    mechanism = _mechanism_receipt_from_ca(receipt)
    failures = validate_receipt_schema(mechanism)
    assert failures == []
    assert mechanism["f3b_branch"] == (
        F3BWhyState0Branch.STATE0_IDENTITY_CORROBORATED_BY_OMISSION.value
    )


def test_mechanism_receipt_schema_rejects_branch_mismatch() -> None:
    receipt = _minimal_valid_b_ca_receipt(cb_state_index=None)
    mechanism = _mechanism_receipt_from_ca(
        receipt,
        f3b_branch=F3BWhyState0Branch.SAMPLE_SET_OR_ELIGIBILITY_ARTIFACT.value,
    )
    failures = validate_receipt_schema(mechanism)
    assert any("f3b_branch_mismatch" in failure for failure in failures)


def test_variable_a_identity_unchanged_no_b_leakage() -> None:
    from calm.llm_computer.tests.test_f3b_ca_source_schema_v1 import (
        _classify_ca_receipt,
        _minimal_valid_ca_confirmation_receipt,
    )

    receipt = _minimal_valid_ca_confirmation_receipt(
        reversed_order=True,
        cb_state_index=0,
    )
    result = _classify_ca_receipt(receipt)
    assert result["terminal_branch"] == F3BWhyState0Branch.STATE0_IDENTITY_STRUCTURE.value
    assert result["terminal_branch"] != (
        F3BWhyState0Branch.STATE0_IDENTITY_CORROBORATED_BY_OMISSION.value
    )


@pytest.mark.parametrize(
    "sampled_set_changed",
    ["False", "true", 1],
)
def test_variable_b_truthy_non_bool_sampled_set_changed_not_b3(
    sampled_set_changed: object,
) -> None:
    receipt = _minimal_valid_b_ca_receipt(cb_state_index=None)
    receipt["sampled_set_changed"] = sampled_set_changed
    result = _classify_b_ca_receipt(receipt)
    assert result["terminal_branch"] != (
        F3BWhyState0Branch.STATE0_IDENTITY_CORROBORATED_BY_OMISSION.value
    )
    assert result["f3b_branch_inputs"]["schema_ok"] is False
    assert "ca_sampled_set_changed_not_bool" in result["f3b_branch_inputs"][
        "ca_source_schema_failures"
    ]


@pytest.mark.parametrize(
    "cb_state_count",
    [True, "0", None],
)
def test_variable_b_malformed_cb_state_count_not_b3(cb_state_count: object) -> None:
    receipt = _minimal_valid_b_ca_receipt(cb_state_index=None)
    inputs = build_branch_input_contract_from_ca_receipt(
        receipt,
        variable_id="B_state0_omission",
        control_reason="state0_omission_or_shifted_set",
        identity_order_inertness_proven=True,
    )
    inputs["cb_state_count"] = cb_state_count
    result = classify_f3b_why_state0_branch(inputs)
    assert result["terminal_branch"] != (
        F3BWhyState0Branch.STATE0_IDENTITY_CORROBORATED_BY_OMISSION.value
    )


def test_variable_b_absent_cb_state_count_not_b3() -> None:
    receipt = _minimal_valid_b_ca_receipt(cb_state_index=None)
    inputs = build_branch_input_contract_from_ca_receipt(
        receipt,
        variable_id="B_state0_omission",
        control_reason="state0_omission_or_shifted_set",
        identity_order_inertness_proven=True,
    )
    del inputs["cb_state_count"]
    result = classify_f3b_why_state0_branch(inputs)
    assert result["terminal_branch"] != (
        F3BWhyState0Branch.STATE0_IDENTITY_CORROBORATED_BY_OMISSION.value
    )


@pytest.mark.parametrize(
    "semantic_state0_is_crossing_bearing",
    [None, True, 0, "false"],
)
def test_variable_b_non_exact_false_semantic_state0_not_b3(
    semantic_state0_is_crossing_bearing: object,
) -> None:
    receipt = _minimal_valid_b_ca_receipt(cb_state_index=None)
    inputs = build_branch_input_contract_from_ca_receipt(
        receipt,
        variable_id="B_state0_omission",
        control_reason="state0_omission_or_shifted_set",
        identity_order_inertness_proven=True,
    )
    inputs["semantic_state0_is_crossing_bearing"] = semantic_state0_is_crossing_bearing
    result = classify_f3b_why_state0_branch(inputs)
    assert result["terminal_branch"] != (
        F3BWhyState0Branch.STATE0_IDENTITY_CORROBORATED_BY_OMISSION.value
    )


@pytest.mark.parametrize(
    "cb_state_count",
    [True, "0", None],
)
def test_mechanism_receipt_claiming_b3_with_malformed_cb_count_mismatch(
    cb_state_count: object,
) -> None:
    receipt = _minimal_valid_b_ca_receipt(cb_state_index=None)
    mechanism = _mechanism_receipt_from_ca(receipt)
    mechanism["f3b_branch"] = (
        F3BWhyState0Branch.STATE0_IDENTITY_CORROBORATED_BY_OMISSION.value
    )
    mechanism["f3b_branch_inputs"]["cb_state_count"] = cb_state_count
    failures = validate_receipt_schema(mechanism)
    assert any("f3b_branch_mismatch" in failure for failure in failures)
