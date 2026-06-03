"""C1.1b assumption-bound native-loop verdict tests."""
from __future__ import annotations

import json
from dataclasses import replace

import pytest

from calm.hrm_text_158.native_full_stack.accumulator_decision_density import (
    DECISION_EXACT_INFEASIBLE,
)
from calm.hrm_text_158.native_full_stack.accumulator_real_dynamics_verdict import (
    ASSUMPTION_BOUND_ENGINEERING_VERDICT,
    BINDING_FOR_IN_TREE_NATIVE_LOOP_DISTRIBUTION,
    BINDING_S1_REGIME_VERDICT,
    C1_1C_ROUTE_BOUNDED_DELTA_WITH_REPORT,
    C1_1C_ROUTE_SPARSE_EXACT,
    PARTIAL_EVIDENCE_ONLY,
    PARTIAL_FOR_S1_REAL_DYNAMICS,
    SOURCE_KIND_COMPACT_FULL_LOOP_ARTIFACT,
    SOURCE_KIND_GENERATED_NATIVE_LOOP,
    SOURCE_KIND_RAW_S1_TELEMETRY,
    SourceFieldCoverage,
    assert_compact_payload_has_no_tensors,
    capture_file_integrity,
    assert_file_integrity_unchanged,
    classify_compact_full_loop_artifact_payload,
    pre_register_source_bindingness,
    raw_state_capture_schema,
    run_pre_registered_native_loop_verdict,
    validate_accumulator_assumption_bound_verdict_report,
)


@pytest.fixture(scope="module")
def verdict_report():
    return run_pre_registered_native_loop_verdict()


def test_generated_source_keeps_assumption_bound_labels_and_no_forbidden_serialization(
    verdict_report,
):
    payload = verdict_report.to_dict()
    serialized = json.dumps(payload, sort_keys=True)

    validate_accumulator_assumption_bound_verdict_report(verdict_report)
    assert verdict_report.label == ASSUMPTION_BOUND_ENGINEERING_VERDICT
    assert verdict_report.source_kind == SOURCE_KIND_GENERATED_NATIVE_LOOP
    assert (
        verdict_report.source_bindingness.primary_bindingness
        == BINDING_FOR_IN_TREE_NATIVE_LOOP_DISTRIBUTION
    )
    assert verdict_report.source_bindingness.s1_bindingness == PARTIAL_FOR_S1_REAL_DYNAMICS
    assert BINDING_S1_REGIME_VERDICT not in serialized
    assert "real-dynamics verdict" not in serialized
    assert verdict_report.raw_arrays_included is False
    assert_compact_payload_has_no_tensors(payload)


def test_validation_rejects_generated_source_if_binding_s1_label_is_injected(verdict_report):
    bad_binding = replace(
        verdict_report.source_bindingness,
        primary_bindingness=BINDING_S1_REGIME_VERDICT,
        s1_bindingness=BINDING_S1_REGIME_VERDICT,
    )
    bad_report = replace(verdict_report, source_bindingness=bad_binding)

    with pytest.raises(ValueError, match="forbidden claim|in-tree only"):
        validate_accumulator_assumption_bound_verdict_report(bad_report)


def test_compact_full_loop_artifact_lacking_raw_state_is_partial_only():
    compact_payload = {
        "label": "native_full_loop_engineering_receipt_reference_stitch_only",
        "eligible_weight_count": 160,
        "step_receipts": [
            {
                "global_rate_cap_accepted_count": 2,
                "global_rate_cap_deferred_count": 14,
                "global_cap_deferred_backlog_size": 14,
                "global_cap_saturated": True,
            },
        ],
    }

    bindingness = classify_compact_full_loop_artifact_payload(compact_payload)

    assert bindingness.source_kind == SOURCE_KIND_COMPACT_FULL_LOOP_ARTIFACT
    assert bindingness.primary_bindingness == PARTIAL_EVIDENCE_ONLY
    assert bindingness.can_claim_binding_live_regime is False
    assert "raw_q_acc_state" in bindingness.missing_fields
    assert "raw_votes" in bindingness.missing_fields
    assert "cap_selected_deferred_rows" in bindingness.missing_fields
    assert "deferred_backlog" in bindingness.missing_fields


def test_only_full_raw_source_coverage_can_claim_binding_live_regime():
    incomplete = replace(
        SourceFieldCoverage.full_raw_s1_telemetry(),
        backlog_state_carry=False,
    )
    incomplete_binding = pre_register_source_bindingness(
        source_kind=SOURCE_KIND_RAW_S1_TELEMETRY,
        coverage=incomplete,
    )
    full_binding = pre_register_source_bindingness(
        source_kind=SOURCE_KIND_RAW_S1_TELEMETRY,
        coverage=SourceFieldCoverage.full_raw_s1_telemetry(),
    )
    generated_binding = pre_register_source_bindingness(
        source_kind=SOURCE_KIND_GENERATED_NATIVE_LOOP,
        coverage=SourceFieldCoverage.full_generated_native_loop(),
    )

    assert incomplete_binding.primary_bindingness == PARTIAL_EVIDENCE_ONLY
    assert incomplete_binding.can_claim_binding_live_regime is False
    assert "backlog_state_carry" in incomplete_binding.missing_fields
    assert full_binding.primary_bindingness == BINDING_S1_REGIME_VERDICT
    assert full_binding.can_claim_binding_live_regime is True
    assert (
        generated_binding.primary_bindingness
        == BINDING_FOR_IN_TREE_NATIVE_LOOP_DISTRIBUTION
    )
    assert generated_binding.can_claim_binding_live_regime is False


def test_pre_registered_schedule_stresses_hard_regime_and_routes_bounded_delta(
    verdict_report,
):
    steps = {step.schedule_name: step for step in verdict_report.per_step_reports}

    assert list(steps) == [
        "sparse_unsaturated",
        "moderate_unsaturated",
        "cap_saturated",
        "backlog_growth",
    ]
    assert steps["sparse_unsaturated"].sparse_classification.decision_exact_feasible is True
    assert steps["sparse_unsaturated"].c1_1c_step_route == C1_1C_ROUTE_SPARSE_EXACT
    assert steps["moderate_unsaturated"].sparse_classification.classification == (
        DECISION_EXACT_INFEASIBLE
    )
    assert steps["cap_saturated"].global_cap_saturated is True
    assert steps["cap_saturated"].global_cap_deferred_count > 0
    assert steps["backlog_growth"].global_cap_saturated is True
    assert steps["backlog_growth"].backlog_state_carry_count > (
        steps["cap_saturated"].backlog_state_carry_count
    )
    assert verdict_report.terminal_decision == DECISION_EXACT_INFEASIBLE
    assert verdict_report.c1_1c_route == C1_1C_ROUTE_BOUNDED_DELTA_WITH_REPORT
    assert verdict_report.vote_pressure_summary["eligible_weight_count"] == 16_384
    assert verdict_report.vote_pressure_summary["schedule_fixed_before_measurement"] is True
    assert verdict_report.vote_pressure_summary["post_hoc_tuning_allowed"] is False


def test_raw_state_capture_schema_names_upgrade_fields_without_raw_arrays():
    schema = raw_state_capture_schema()
    field_names = {field["name"] for field in schema["required_fields"]}
    serialized = json.dumps(schema, sort_keys=True)

    assert {
        "source_identity_integrity",
        "q_acc_state",
        "votes",
        "vote_update_spec",
        "global_cap_rows_or_inputs",
        "deferred_backlog",
        "n_scale",
    } <= field_names
    assert BINDING_S1_REGIME_VERDICT not in serialized
    assert_compact_payload_has_no_tensors(schema)


def test_read_only_integrity_snapshot_detects_mutation(tmp_path):
    path = tmp_path / "telemetry.json"
    path.write_text('{"compact": true}\n', encoding="utf-8")
    before = capture_file_integrity(path)
    after = capture_file_integrity(path)
    assert_file_integrity_unchanged(before, after)

    path.write_text('{"compact": false}\n', encoding="utf-8")
    changed = capture_file_integrity(path)
    with pytest.raises(ValueError, match="read-only telemetry integrity changed"):
        assert_file_integrity_unchanged(before, changed)
