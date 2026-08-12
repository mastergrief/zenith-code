from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from calm.hrm_text_158.native_full_stack.acc_width_recorded_row_sweep import (
    LABEL_ACC_SHRINK_TWO_TIER,
    LABEL_SCREEN_HARNESS_OR_GATE_FAIL,
)
from calm.hrm_text_158.native_full_stack.b0_recorded_state_inventory import (
    B0_CAPTURE2_BUNDLE,
    B0_MULTI_TRACE_BUNDLE_SPECS,
    B0_TRACE1_BUNDLE,
    BRANCH_HARNESS_OR_SCOPE_FAIL,
    BRANCH_MEASUREMENT_STATE_EXISTS_AND_HEADROOM,
    BRANCH_MEASUREMENT_STATE_EXISTS_NO_HEADROOM,
    CROSS_TRACE_HARNESS_FAIL,
    CROSS_TRACE_HOLDS_ACROSS_TRACES,
    CROSS_TRACE_TRACE_DEPENDENT_HEADROOM,
    THRESHOLD_MISMATCH_ID,
    bundle_sweep_inputs_available,
    emit_cross_trace_branch_classifier,
    emit_measurement_shape_branch,
    extract_threshold_mismatch_hazard,
    run_b0_multi_trace_recorded_state_inventory,
    run_b0_recorded_state_inventory_vote_acc_prize_sizing,
)


def _healthy_sweep_receipt() -> dict[str, object]:
    return {
        "input_integrity": {"passed": True, "failure_reasons": []},
        "field_inventory_gate": {"passed": True},
        "vote_spec": {"threshold_abs": 10},
        "vote_spec_provenance": {
            "threshold_crosscheck": {
                "threshold_crosscheck": THRESHOLD_MISMATCH_ID,
                "expected_threshold_abs": 10,
                "derived_threshold_abs": 1,
                "surfaced_loudly": True,
                "row_provenance": {"threshold_abs": 1, "row_count": 1600},
            },
            "threshold_row_derivation_mismatch": {
                "threshold_crosscheck": THRESHOLD_MISMATCH_ID,
                "expected_threshold_abs": 10,
                "derived_threshold_abs": 1,
                "surfaced_loudly": True,
            },
        },
        "primary_label": LABEL_ACC_SHRINK_TWO_TIER,
        "failure_reasons": [],
        "headroom_pass": True,
        "w_min_headroom_safe": 6,
        "w_min": 6,
        "max_abs_acc_applied_flips": 9,
    }


def test_extract_threshold_mismatch_hazard_surfaces_without_adopting_derived() -> None:
    hazard = extract_threshold_mismatch_hazard(_healthy_sweep_receipt())
    assert hazard["present"] is True
    assert hazard["expected_threshold_abs"] == 10
    assert hazard["derived_threshold_abs_from_rows"] == 1
    assert hazard["vote_spec_replay_threshold_abs"] == 10
    assert hazard["do_not_bank_row_derived_threshold"] is True
    vacuity = hazard["estimand_vacuity_guard"]
    assert vacuity["derived_threshold_assessment"]["passes_vacuity_guard"] is False
    assert vacuity["attested_threshold_assessment"]["passes_vacuity_guard"] is True


def test_emit_measurement_shape_branch_headroom_path() -> None:
    branch = emit_measurement_shape_branch(
        _healthy_sweep_receipt(),
        inventory_missing=[],
    )
    assert branch["primary_branch"] == BRANCH_MEASUREMENT_STATE_EXISTS_AND_HEADROOM
    assert branch["w_min_headroom_safe"] == 6
    assert branch["acc_shrink_primary_label"] == LABEL_ACC_SHRINK_TWO_TIER


def test_emit_measurement_shape_branch_harness_fail_on_integrity() -> None:
    receipt = dict(_healthy_sweep_receipt())
    receipt["input_integrity"] = {"passed": False}
    branch = emit_measurement_shape_branch(receipt, inventory_missing=[])
    assert branch["primary_branch"] == BRANCH_HARNESS_OR_SCOPE_FAIL
    assert "input_integrity_fail" in branch["harness_failures"]


def test_emit_measurement_shape_branch_no_headroom_when_headroom_false() -> None:
    receipt = dict(_healthy_sweep_receipt())
    receipt["headroom_pass"] = False
    branch = emit_measurement_shape_branch(receipt, inventory_missing=[])
    assert branch["primary_branch"] == BRANCH_MEASUREMENT_STATE_EXISTS_NO_HEADROOM


def test_emit_measurement_shape_branch_harness_on_missing_inventory() -> None:
    branch = emit_measurement_shape_branch(
        _healthy_sweep_receipt(),
        inventory_missing=["capture_receipt"],
    )
    assert branch["primary_branch"] == BRANCH_HARNESS_OR_SCOPE_FAIL
    assert "capture_receipt" in branch["harness_failures"]


@pytest.mark.skipif(
    not bundle_sweep_inputs_available(B0_CAPTURE2_BUNDLE),
    reason="frozen b2b_recapture capture2 bundle unavailable",
)
def test_b0_capture2_bundle_revalidation_matches_frozen_fingerprint() -> None:
    result = run_b0_recorded_state_inventory_vote_acc_prize_sizing()
    assert result["reuse_verdict"] == "REUSE_EXISTING_SWEEP_NO_NEW_MEASUREMENT_ENGINE"
    assert (
        result["measurement_shape_branch"]["primary_branch"]
        == BRANCH_MEASUREMENT_STATE_EXISTS_AND_HEADROOM
    )
    assert result["threshold_mismatch_hazard"]["present"] is True
    assert result["prize_sizing"]["w_min_headroom_safe"] == 6
    assert result["prize_sizing"]["headroom_pass"] is True
    assert result["frozen_fingerprint_compare"]["passed"] is True
    assert result["readiness_fixtures"]["neither_is_launch_pass"] is True
    scaffold = result["readiness_fixtures"]["embedded_current_repo_scaffold"]
    # current_repo_scaffold ledger is diagnostic-eligible once the two MISSING
    # rows carry justified exception fields; main-science still fail-closed.
    assert scaffold["ready_for_pre_full_stack_diagnostic"] is True
    assert scaffold["ready_for_main_science"] is False
    assert len(scaffold["blocker_surface_names"] or []) == 8


@pytest.mark.skipif(
    not bundle_sweep_inputs_available(B0_CAPTURE2_BUNDLE),
    reason="frozen b2b_recapture capture2 bundle unavailable",
)
def test_b0_source_inventory_classifies_missing_tensor_snapshot() -> None:
    result = run_b0_recorded_state_inventory_vote_acc_prize_sizing()
    by_id = {
        entry["artifact_id"]: entry for entry in result["source_inventory"]
    }
    assert by_id["stable_trace_canonical"]["source_kind"] == "stable_trace"
    assert by_id["stable_trace_canonical"]["sufficient_for_b0"] is True
    assert by_id["tensor_wide_persistent_qacc"]["source_kind"] == "missing"
    assert by_id["parent_checkpoint"]["sufficient_for_b0"] is False


def test_b0_harness_fail_when_bundle_missing(tmp_path: Path) -> None:
    missing_root = tmp_path / "missing_chain"
    missing_root.mkdir()
    bad_spec = replace(B0_CAPTURE2_BUNDLE, chain_root=missing_root)
    result = run_b0_recorded_state_inventory_vote_acc_prize_sizing(bundle_spec=bad_spec)
    assert (
        result["measurement_shape_branch"]["primary_branch"]
        == BRANCH_HARNESS_OR_SCOPE_FAIL
    )
    assert result["sweep_receipt"]["primary_label"] == LABEL_SCREEN_HARNESS_OR_GATE_FAIL


def _trace_result_stub(
    *,
    capture_id: str,
    primary_branch: str = BRANCH_MEASUREMENT_STATE_EXISTS_AND_HEADROOM,
    headroom_pass: bool = True,
    w_min_headroom_safe: int | None = 6,
    w_min: int | None = 6,
    fingerprint_passed: bool = True,
) -> dict[str, object]:
    return {
        "capture_id": capture_id,
        "measurement_shape_branch": {
            "primary_branch": primary_branch,
            "headroom_pass": headroom_pass,
            "w_min_headroom_safe": w_min_headroom_safe,
        },
        "prize_sizing": {
            "w_min_headroom_safe": w_min_headroom_safe,
            "w_min": w_min,
            "headroom_pass": headroom_pass,
        },
        "frozen_fingerprint_compare": {"passed": fingerprint_passed},
    }


def test_emit_cross_trace_branch_classifier_holds_across_traces() -> None:
    classifier = emit_cross_trace_branch_classifier(
        [
            _trace_result_stub(capture_id="trace1"),
            _trace_result_stub(capture_id="capture2"),
        ]
    )
    assert classifier["primary_branch"] == CROSS_TRACE_HOLDS_ACROSS_TRACES
    assert classifier["w_min_headroom_safe"] == 6


def test_emit_cross_trace_branch_classifier_trace_dependent_headroom() -> None:
    classifier = emit_cross_trace_branch_classifier(
        [
            _trace_result_stub(capture_id="trace1", w_min_headroom_safe=6),
            _trace_result_stub(capture_id="capture2", w_min_headroom_safe=8),
        ]
    )
    assert classifier["primary_branch"] == CROSS_TRACE_TRACE_DEPENDENT_HEADROOM


def test_emit_cross_trace_branch_classifier_harness_fail_on_fingerprint() -> None:
    classifier = emit_cross_trace_branch_classifier(
        [
            _trace_result_stub(capture_id="trace1", fingerprint_passed=False),
            _trace_result_stub(capture_id="capture2"),
        ]
    )
    assert classifier["primary_branch"] == CROSS_TRACE_HARNESS_FAIL
    assert "trace1" in classifier["harness_failures"]


def test_emit_cross_trace_branch_classifier_harness_fail_on_per_trace_branch() -> None:
    classifier = emit_cross_trace_branch_classifier(
        [
            _trace_result_stub(
                capture_id="trace1",
                primary_branch=BRANCH_HARNESS_OR_SCOPE_FAIL,
            ),
            _trace_result_stub(capture_id="capture2"),
        ]
    )
    assert classifier["primary_branch"] == CROSS_TRACE_HARNESS_FAIL


@pytest.mark.skipif(
    not all(bundle_sweep_inputs_available(spec) for spec in B0_MULTI_TRACE_BUNDLE_SPECS),
    reason="frozen dual b2b_recapture bundles unavailable",
)
def test_b0_multi_trace_inventory_holds_across_traces() -> None:
    result = run_b0_multi_trace_recorded_state_inventory()
    assert result["schema_version"].endswith("/v1")
    assert result["slice_id"].endswith("_v1")
    assert len(result["traces"]) == 2
    assert result["traces"][0]["capture_id"] == "trace1"
    assert result["traces"][1]["capture_id"] == "capture2"
    for trace in result["traces"]:
        assert (
            trace["measurement_shape_branch"]["primary_branch"]
            == BRANCH_MEASUREMENT_STATE_EXISTS_AND_HEADROOM
        )
        assert trace["frozen_fingerprint_compare"]["passed"] is True
        assert trace["prize_sizing"]["w_min_headroom_safe"] == 6
    assert (
        result["cross_trace_branch_classifier"]["primary_branch"]
        == CROSS_TRACE_HOLDS_ACROSS_TRACES
    )
    assert "multi_trace true" in result["explicit_non_claims"]


@pytest.mark.skipif(
    not bundle_sweep_inputs_available(B0_TRACE1_BUNDLE),
    reason="frozen trace1 bundle unavailable",
)
def test_b0_trace1_bundle_revalidation_matches_frozen_fingerprint() -> None:
    result = run_b0_recorded_state_inventory_vote_acc_prize_sizing(
        bundle_spec=B0_TRACE1_BUNDLE
    )
    assert (
        result["measurement_shape_branch"]["primary_branch"]
        == BRANCH_MEASUREMENT_STATE_EXISTS_AND_HEADROOM
    )
    assert result["frozen_fingerprint_compare"]["passed"] is True
    assert result["prize_sizing"]["w_min_headroom_safe"] == 6
