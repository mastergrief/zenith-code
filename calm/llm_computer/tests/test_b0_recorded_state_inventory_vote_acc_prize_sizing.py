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
    BRANCH_HARNESS_OR_SCOPE_FAIL,
    BRANCH_MEASUREMENT_STATE_EXISTS_AND_HEADROOM,
    BRANCH_MEASUREMENT_STATE_EXISTS_NO_HEADROOM,
    THRESHOLD_MISMATCH_ID,
    bundle_sweep_inputs_available,
    emit_measurement_shape_branch,
    extract_threshold_mismatch_hazard,
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
    assert scaffold["ready_for_pre_full_stack_diagnostic"] is False
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
