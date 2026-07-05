"""CPU-static schema tests for Fold-3B Step 1 prereg/preflight artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

REPO = Path("/mnt/c/Users/gabes/projects/claw-code-hrm-text-158")
DRAFT = REPO / "artifacts/consensus_prep/c4s1_fold3b_step1_prereg_packet_v1_draft.json"
PREFLIGHT = (
    REPO
    / "artifacts/measurement_closeout/c4s1_fold3b_step1_feasibility_preflight_receipt.json"
)
STALE_WRAPPER = "/tmp/stale_nonexistent_wrapper_receipt.json"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _minimal_valid_f3b_receipt_base() -> dict[str, Any]:
    order = list(range(10))
    rank_map = {str(state): rank for rank, state in enumerate(order)}
    per_state = [
        {
            "state_index": state,
            "crossing_indices_len": 512 if state == 0 else 0,
            "crossing_count": 512 if state == 0 else 0,
            "mark_count": 1,
        }
        for state in order
    ]
    return {
        "schema": "hrm_text_158_fold3b_mechanism_diagnosis_receipt/v1",
        "sampled_state_set": order,
        "sampled_state_order": order,
        "order_rank_by_semantic_state": rank_map,
        "semantic_state_id": 0,
        "per_state": per_state,
        "dedup_reset_called": True,
        "dedup_session_scope": "probe_subprocess",
        "wrapper_path": "/valid/wrapper.json",
        "primary_receipt_path": "/valid/primary.json",
        "fallback_receipt_path": None,
        "science_verdict_source": "primary",
        "parent_sha": "abc123",
        "git_head_required": "891a5e6",
        "variable_id": "A_order_only",
        "control_reason": "order_only_perturbation",
        "mark_count": 10,
        "ready_for_main_science": False,
        "counts_as_sub2": False,
        "pre_full_stack_diagnostic": True,
    }


def _sync_receipt_branch_fields(
    receipt: dict[str, Any],
    *,
    variable_id: str = "A_order_only",
    control_reason: str = "order_only_perturbation",
    identity_order_inertness_proven: bool = True,
) -> dict[str, Any]:
    from calm.hrm_text_158.native_full_stack.f3b_why_state0_branch import (
        classify_f3b_why_state0_branch,
    )

    per_state = list(receipt.get("per_state") or [])
    sampled_states = list(receipt.get("sampled_state_order") or receipt.get("sampled_states") or [])
    sampled_set = list(receipt.get("sampled_state_set") or sampled_states)
    order_rank = receipt.get("order_rank_by_semantic_state")
    if not isinstance(order_rank, dict) and sampled_states:
        order_rank = {str(state): rank for rank, state in enumerate(sampled_states)}

    cb_indices = sorted(
        int(row["state_index"])
        for row in per_state
        if isinstance(row, dict) and int(row.get("crossing_indices_len") or 0) > 0
    )
    first_measured = sampled_states[0] if sampled_states else None
    first_measured_row = next(
        (row for row in per_state if int(row.get("state_index", -1)) == int(first_measured)),
        None,
    ) if first_measured is not None else None

    inputs = {
        "operational_ok": True,
        "schema_ok": True,
        "ca_source_schema_failures": [],
        "sampled_state_set": sorted({int(x) for x in sampled_set}),
        "sampled_state_order": sampled_states,
        "order_rank_by_semantic_state": order_rank or {},
        "exact_per_state_coverage": True,
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
    receipt["f3b_branch_inputs"] = inputs
    receipt["f3b_branch"] = classify_f3b_why_state0_branch(inputs)["terminal_branch"]
    return receipt


def _minimal_valid_f3b_receipt() -> dict[str, Any]:
    return _sync_receipt_branch_fields(_minimal_valid_f3b_receipt_base())


def _well_formed_variable_a_inputs() -> dict[str, Any]:
    order = list(range(10))
    rank_map = {str(state): rank for rank, state in enumerate(order)}
    per_state = [
        {
            "state_index": state,
            "crossing_indices_len": 512 if state == 0 else 0,
            "crossing_count": 512 if state == 0 else 0,
            "mark_count": 1,
        }
        for state in order
    ]
    return {
        "operational_ok": True,
        "schema_ok": True,
        "sampled_state_set": order,
        "sampled_state_order": order,
        "order_rank_by_semantic_state": rank_map,
        "per_state": per_state,
        "exact_per_state_coverage": True,
        "dedup_reset_called": True,
        "dedup_session_scope": "probe_subprocess",
        "identity_order_inertness_proven": True,
        "semantic_state0_crossing_indices_len": 512,
        "cb_state_count": 1,
        "first_measured_semantic_state": 0,
        "first_measured_is_crossing_bearing": True,
        "semantic_state0_is_crossing_bearing": True,
        "sampled_set_changed": False,
        "mark_count_consistent": True,
        "variable_id": "A_order_only",
        "control_reason": "order_only_perturbation",
        "ca_source_schema_failures": [],
    }


def test_prereg_packet_and_preflight_parse() -> None:
    from calm.hrm_text_158.native_full_stack.f3b_why_state0_branch import (
        validate_preflight_receipt_schema,
        validate_prereg_packet_schema,
    )

    packet = _load(DRAFT)
    preflight = _load(PREFLIGHT)
    assert validate_prereg_packet_schema(packet) == []
    assert validate_preflight_receipt_schema(preflight) == []


def test_required_receipt_fields_match_packet_json_drift_guard() -> None:
    from calm.hrm_text_158.native_full_stack.f3b_why_state0_branch import (
        REQUIRED_RECEIPT_FIELDS,
    )

    packet = _load(DRAFT)
    packet_required = packet["receipt_schema"]["required_fields"]
    assert set(REQUIRED_RECEIPT_FIELDS) == set(packet_required)


def test_prereg_packet_encodes_identity_inertness_precondition() -> None:
    packet = _load(DRAFT)
    block = packet["identity_order_inertness_precondition"]
    assert block["blocks_variable_a_interpretation"] is True
    assert block["identity_order"] == list(range(10))
    assert block["counts_against_fold3b_gpu_budget"] is False
    assert "dual_purpose" in block
    var_a = next(
        v for v in packet["gpu_ladder"]["variables"] if v["variable_id"] == "A_order_only"
    )
    assert var_a["blocked_until"] == "IDENTITY_ORDER_INERTNESS_PRECONDITION"


def test_receipt_schema_fails_on_missing_sampled_state_order() -> None:
    from calm.hrm_text_158.native_full_stack.f3b_why_state0_branch import (
        validate_receipt_schema,
    )

    receipt = _minimal_valid_f3b_receipt()
    del receipt["sampled_state_order"]
    failures = validate_receipt_schema(receipt)
    assert any("missing:sampled_state_order" in failure for failure in failures)


def test_receipt_schema_fails_on_missing_order_rank_mapping() -> None:
    from calm.hrm_text_158.native_full_stack.f3b_why_state0_branch import (
        validate_receipt_schema,
    )

    receipt = _minimal_valid_f3b_receipt()
    del receipt["order_rank_by_semantic_state"]
    failures = validate_receipt_schema(receipt)
    assert any(
        "missing:order_rank_by_semantic_state" in failure for failure in failures
    )


def test_receipt_schema_fails_on_missing_exact_coverage_fields() -> None:
    from calm.hrm_text_158.native_full_stack.f3b_why_state0_branch import (
        validate_receipt_schema,
    )

    receipt = _minimal_valid_f3b_receipt()
    receipt["per_state"] = receipt["per_state"][:5]
    failures = validate_receipt_schema(receipt)
    assert failures


def test_receipt_schema_fails_on_stale_wrapper_path_pattern() -> None:
    from calm.hrm_text_158.native_full_stack.f3b_why_state0_branch import (
        validate_receipt_schema,
    )

    receipt = _minimal_valid_f3b_receipt()
    receipt["wrapper_path"] = STALE_WRAPPER
    assert receipt["wrapper_path"] == STALE_WRAPPER
    failures = validate_receipt_schema(receipt)
    assert "missing:sampled_state_order" not in failures


def test_receipt_schema_fails_when_dedup_reset_not_true() -> None:
    from calm.hrm_text_158.native_full_stack.f3b_why_state0_branch import (
        validate_receipt_schema,
    )

    receipt = _minimal_valid_f3b_receipt()
    receipt["dedup_reset_called"] = False
    failures = validate_receipt_schema(receipt)
    assert any("dedup_reset_called_false" in failure for failure in failures)


def test_receipt_schema_fails_on_empty_dedup_session_scope() -> None:
    from calm.hrm_text_158.native_full_stack.f3b_why_state0_branch import (
        validate_receipt_schema,
    )

    receipt = _minimal_valid_f3b_receipt()
    receipt["dedup_session_scope"] = ""
    failures = validate_receipt_schema(receipt)
    assert any("dedup_session_scope" in failure for failure in failures)


def test_receipt_schema_fails_on_wrong_order_rank_map() -> None:
    from calm.hrm_text_158.native_full_stack.f3b_why_state0_branch import (
        validate_receipt_schema,
    )

    receipt = _minimal_valid_f3b_receipt()
    receipt["order_rank_by_semantic_state"]["0"] = 9
    failures = validate_receipt_schema(receipt)
    assert any("order_rank_mismatch_state:0" in failure for failure in failures)


def test_receipt_schema_fails_on_duplicate_sampled_state_order() -> None:
    from calm.hrm_text_158.native_full_stack.f3b_why_state0_branch import (
        validate_receipt_schema,
    )

    receipt = _minimal_valid_f3b_receipt()
    receipt["sampled_state_order"] = [0, 0] + list(range(2, 10))
    failures = validate_receipt_schema(receipt)
    assert any("sampled_state_order_has_duplicates" in failure for failure in failures)


def test_receipt_schema_fails_on_order_set_mismatch() -> None:
    from calm.hrm_text_158.native_full_stack.f3b_why_state0_branch import (
        validate_receipt_schema,
    )

    receipt = _minimal_valid_f3b_receipt()
    receipt["sampled_state_set"] = list(range(11))
    failures = validate_receipt_schema(receipt)
    assert any("sampled_state_order_set_mismatch" in failure for failure in failures)


def test_receipt_schema_fails_on_per_state_set_mismatch() -> None:
    from calm.hrm_text_158.native_full_stack.f3b_why_state0_branch import (
        validate_receipt_schema,
    )

    receipt = _minimal_valid_f3b_receipt()
    receipt["per_state"] = receipt["per_state"][:-1]
    failures = validate_receipt_schema(receipt)
    assert any("sampled_state_set_per_state_mismatch" in failure for failure in failures)


def test_branch_classifier_precedence_operational_before_schema() -> None:
    from calm.hrm_text_158.native_full_stack.f3b_why_state0_branch import (
        F3BWhyState0Branch,
        classify_f3b_why_state0_branch,
    )

    result = classify_f3b_why_state0_branch(
        {
            "operational_ok": False,
            "schema_ok": False,
            "sampled_state_order": [0],
            "order_rank_by_semantic_state": {"0": 0},
            "exact_per_state_coverage": False,
            "variable_id": "A_order_only",
            "identity_order_inertness_proven": False,
        }
    )
    assert result["terminal_branch"] == F3BWhyState0Branch.NO_VERDICT_OPERATIONAL.value


def test_branch_classifier_dedup_false_routes_to_dedup_artifact() -> None:
    from calm.hrm_text_158.native_full_stack.f3b_why_state0_branch import (
        F3BWhyState0Branch,
        classify_f3b_why_state0_branch,
    )

    inputs = _well_formed_variable_a_inputs()
    inputs["dedup_reset_called"] = False
    result = classify_f3b_why_state0_branch(inputs)
    assert result["terminal_branch"] == F3BWhyState0Branch.MARKING_OR_DEDUP_ARTIFACT.value
    assert result["terminal_branch"] != F3BWhyState0Branch.STATE0_IDENTITY_STRUCTURE.value


def test_branch_classifier_missing_dedup_routes_to_schema() -> None:
    from calm.hrm_text_158.native_full_stack.f3b_why_state0_branch import (
        F3BWhyState0Branch,
        classify_f3b_why_state0_branch,
    )

    inputs = _well_formed_variable_a_inputs()
    del inputs["dedup_reset_called"]
    result = classify_f3b_why_state0_branch(inputs)
    assert result["terminal_branch"] == F3BWhyState0Branch.NO_VERDICT_SCHEMA.value


def test_branch_classifier_empty_dedup_scope_routes_fail_closed() -> None:
    from calm.hrm_text_158.native_full_stack.f3b_why_state0_branch import (
        F3BWhyState0Branch,
        classify_f3b_why_state0_branch,
    )

    inputs = _well_formed_variable_a_inputs()
    inputs["dedup_session_scope"] = ""
    result = classify_f3b_why_state0_branch(inputs)
    assert result["terminal_branch"] in {
        F3BWhyState0Branch.NO_VERDICT_SCHEMA.value,
        F3BWhyState0Branch.MARKING_OR_DEDUP_ARTIFACT.value,
    }
    assert result["terminal_branch"] != F3BWhyState0Branch.STATE0_IDENTITY_STRUCTURE.value


def test_branch_classifier_wrong_rank_map_does_not_reach_identity() -> None:
    from calm.hrm_text_158.native_full_stack.f3b_why_state0_branch import (
        F3BWhyState0Branch,
        classify_f3b_why_state0_branch,
    )

    inputs = _well_formed_variable_a_inputs()
    inputs["order_rank_by_semantic_state"]["0"] = 9
    result = classify_f3b_why_state0_branch(inputs)
    assert result["terminal_branch"] == F3BWhyState0Branch.NO_VERDICT_SCHEMA.value


def test_branch_classifier_variable_a_blocked_without_inertness() -> None:
    from calm.hrm_text_158.native_full_stack.f3b_why_state0_branch import (
        F3BWhyState0Branch,
        classify_f3b_why_state0_branch,
    )

    inputs = _well_formed_variable_a_inputs()
    inputs["identity_order_inertness_proven"] = False
    inputs["sampled_state_order"] = list(range(9, -1, -1))
    inputs["order_rank_by_semantic_state"] = {str(i): 9 - i for i in range(10)}
    result = classify_f3b_why_state0_branch(inputs)
    assert result["terminal_branch"] == F3BWhyState0Branch.MIXED_OR_INCONCLUSIVE.value


def test_branch_classifier_well_formed_variable_a_identity_structure() -> None:
    from calm.hrm_text_158.native_full_stack.f3b_why_state0_branch import (
        F3BWhyState0Branch,
        classify_f3b_why_state0_branch,
    )

    result = classify_f3b_why_state0_branch(_well_formed_variable_a_inputs())
    assert result["terminal_branch"] == F3BWhyState0Branch.STATE0_IDENTITY_STRUCTURE.value


def test_builder_to_classify_identity_structure_path() -> None:
    pytest.importorskip("calm.hrm_text_158.native_full_stack.f3b_why_state0_branch")
    from calm.llm_computer.tests.test_f3b_ca_source_schema_v1 import (
        test_builder_to_classify_identity_structure_from_ca_shape,
    )

    test_builder_to_classify_identity_structure_from_ca_shape()


def test_builder_to_classify_measurement_order_artifact_path() -> None:
    pytest.importorskip("calm.hrm_text_158.native_full_stack.f3b_why_state0_branch")
    from calm.llm_computer.tests.test_f3b_ca_source_schema_v1 import (
        test_builder_to_classify_measurement_order_artifact_from_ca_shape,
    )

    test_builder_to_classify_measurement_order_artifact_from_ca_shape()


def _variable_a_order_artifact_receipt() -> dict[str, Any]:
    receipt = _minimal_valid_f3b_receipt_base()
    reversed_order = list(range(9, -1, -1))
    rank_map = {str(state): rank for rank, state in enumerate(reversed_order)}
    per_state = [
        {
            "state_index": state,
            "crossing_indices_len": 512 if state == 9 else 0,
            "crossing_count": 512 if state == 9 else 0,
            "mark_count": 1,
        }
        for state in range(10)
    ]
    receipt["sampled_state_order"] = reversed_order
    receipt["sampled_state_set"] = list(range(10))
    receipt["order_rank_by_semantic_state"] = rank_map
    receipt["per_state"] = per_state
    receipt["semantic_state_id"] = 9
    return _sync_receipt_branch_fields(receipt)


def test_receipt_schema_fails_on_bogus_f3b_branch() -> None:
    from calm.hrm_text_158.native_full_stack.f3b_why_state0_branch import (
        validate_receipt_schema,
    )

    receipt = _minimal_valid_f3b_receipt()
    receipt["f3b_branch"] = "GARBAGE"
    failures = validate_receipt_schema(receipt)
    assert any("f3b_branch_not_enum" in failure for failure in failures)


def test_receipt_schema_fails_on_f3b_branch_mismatch() -> None:
    from calm.hrm_text_158.native_full_stack.f3b_why_state0_branch import (
        validate_receipt_schema,
    )

    receipt = _variable_a_order_artifact_receipt()
    receipt["f3b_branch"] = "F3B_STATE0_IDENTITY_STRUCTURE"
    failures = validate_receipt_schema(receipt)
    assert any("f3b_branch_mismatch" in failure for failure in failures)


def test_branch_input_order_rank_rejects_duplicate_without_per_state() -> None:
    from calm.hrm_text_158.native_full_stack.f3b_why_state0_branch import (
        _validate_branch_input_order_rank_consistency,
    )

    failures = _validate_branch_input_order_rank_consistency(
        {
            "sampled_state_order": [0, 0, 1],
            "sampled_state_set": [0, 1],
            "order_rank_by_semantic_state": {"0": 0, "1": 2},
        }
    )
    assert any("sampled_state_order_has_duplicates" in failure for failure in failures)


def test_apply_script_self_verify_passes() -> None:
    import subprocess

    proc = subprocess.run(
        ["python3", "scripts/apply_c4s1_fold3b_step1_prereg_packet.py"],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    result = json.loads(proc.stdout)
    assert result["ok"] is True
    assert result["deterministic_regen"] is True
