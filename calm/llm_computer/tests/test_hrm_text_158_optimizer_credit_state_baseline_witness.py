"""Focused tests for optimizer_credit_state baseline witness harness."""
from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from calm.hrm_text_158.native_full_stack.optimizer_credit_state import (
    BRANCH_3C_C_CAPTURE_LAUNDER,
    BRANCH_3C_C_MEASUREMENT_INVALID,
    BRANCH_3C_C_OPT_EXCL_FAIL,
)
from calm.hrm_text_158.native_full_stack.optimizer_credit_state_baseline_witness import (
    ALLOC_GUARD_DENSE_SURFACE_NAMES,
    BRANCH_BASELINE_WITNESS_A_GREEN,
    BRANCH_FEASIBILITY_WITNESS_FAIL,
    INVENTORY_ANCHOR_PROJECTED_MOVES,
    OptimizerCreditStateBaselineWitnessReceipt,
    ProjectionEquivalenceFeasibilityWitness,
    build_optimizer_credit_state_baseline_witness_receipt,
    classify_optimizer_credit_state_baseline_witness_branch,
    collect_baseline_current_debt_inventory,
    run_optimizer_credit_state_baseline_witness,
    run_optimizer_state_exclusion_observation,
    run_projection_equivalence_feasibility_witness,
    validate_optimizer_credit_state_baseline_witness_receipt,
)


def test_projected_moves_inventory_only_not_alloc_guard_surface():
    assert INVENTORY_ANCHOR_PROJECTED_MOVES not in ALLOC_GUARD_DENSE_SURFACE_NAMES


def test_baseline_current_debt_inventory_observes_dense_fp_anchors():
    inventory = collect_baseline_current_debt_inventory()
    assert inventory.inventory_complete is True
    assert inventory.observed_anchor_names == (
        "weighted_grad",
        "credit",
        "projected_moves",
        "dense_rank_votes_before_sparse_event_extraction",
    )
    row = inventory.per_module_rows[0].anchors
    for anchor in row:
        assert anchor["peak_transient_bytes"] > 0
        assert anchor["lifetime"] == "step_local_fp_path"


def test_optimizer_state_exclusion_observation_checkpoint_sha_stable():
    obs = run_optimizer_state_exclusion_observation()
    assert obs.checkpoint_sha256_before == obs.checkpoint_sha256_after
    assert obs.to_dict()["optimizer_state_exclusion_observation"] is True
    assert obs.observation_holds is True
    assert obs.eligible_state_summary["optimizer_checks"]["eligible_params_in_optimizer"] == 0


def test_projection_equivalence_feasibility_witness_passes_with_zero_revival():
    witness = run_projection_equivalence_feasibility_witness()
    assert witness.passed is True
    assert witness.zero_revival_exercised is True
    assert witness.mismatch_count == 0
    assert witness.checkpoint_sha256_before == witness.checkpoint_sha256_after


def test_run_baseline_witness_classifies_branch_a_green():
    receipt = run_optimizer_credit_state_baseline_witness()
    validate_optimizer_credit_state_baseline_witness_receipt(receipt)
    assert receipt.branch_classifier == BRANCH_BASELINE_WITNESS_A_GREEN
    assert receipt.br_3c_c_audit_pass_cpu is False
    assert receipt.optimizer_state_eligible_exclusion_proven is False
    assert receipt.ready_to_flip is False
    assert receipt.gpu_runtime_receipt_present is False


def test_validator_rejects_forbidden_flag_set():
    base = run_optimizer_credit_state_baseline_witness()
    forged = replace(base, br_3c_c_audit_pass_cpu=True)
    with pytest.raises(ValueError, match="forbidden receipt field"):
        validate_optimizer_credit_state_baseline_witness_receipt(forged)


def test_classifier_branch_d_on_incomplete_inventory():
    inventory = collect_baseline_current_debt_inventory()
    bad_inventory = replace(inventory, inventory_complete=False)
    exclusion = run_optimizer_state_exclusion_observation()
    projection = run_projection_equivalence_feasibility_witness()
    branch, _ = classify_optimizer_credit_state_baseline_witness_branch(
        inventory=bad_inventory,
        exclusion=exclusion,
        projection=projection,
    )
    assert branch == BRANCH_3C_C_MEASUREMENT_INVALID


def test_classifier_branch_c_on_projection_failure():
    inventory = collect_baseline_current_debt_inventory()
    exclusion = run_optimizer_state_exclusion_observation()
    projection = ProjectionEquivalenceFeasibilityWitness(
        passed=False,
        cases_run=1,
        zero_revival_exercised=False,
        checkpoint_sha256_before="a" * 64,
        checkpoint_sha256_after="a" * 64,
        mismatch_count=1,
    )
    branch, _ = classify_optimizer_credit_state_baseline_witness_branch(
        inventory=inventory,
        exclusion=exclusion,
        projection=projection,
    )
    assert branch == BRANCH_FEASIBILITY_WITNESS_FAIL


def test_classifier_branch_b_opt_excl_on_exclusion_failure():
    inventory = collect_baseline_current_debt_inventory()
    exclusion = replace(
        run_optimizer_state_exclusion_observation(),
        observation_holds=False,
        capture_laundering_signal=False,
    )
    projection = run_projection_equivalence_feasibility_witness()
    branch, _ = classify_optimizer_credit_state_baseline_witness_branch(
        inventory=inventory,
        exclusion=exclusion,
        projection=projection,
    )
    assert branch == BRANCH_3C_C_OPT_EXCL_FAIL


def test_classifier_branch_b_capture_launder_signal():
    inventory = collect_baseline_current_debt_inventory()
    exclusion = replace(
        run_optimizer_state_exclusion_observation(),
        observation_holds=False,
        capture_laundering_signal=True,
    )
    projection = run_projection_equivalence_feasibility_witness()
    branch, _ = classify_optimizer_credit_state_baseline_witness_branch(
        inventory=inventory,
        exclusion=exclusion,
        projection=projection,
    )
    assert branch == BRANCH_3C_C_CAPTURE_LAUNDER


def test_build_receipt_rejects_branch_mismatch():
    inventory = collect_baseline_current_debt_inventory()
    exclusion = run_optimizer_state_exclusion_observation()
    projection = run_projection_equivalence_feasibility_witness()
    receipt = build_optimizer_credit_state_baseline_witness_receipt(
        inventory=inventory,
        exclusion=exclusion,
        projection=projection,
    )
    forged = replace(receipt, branch_classifier=BRANCH_3C_C_MEASUREMENT_INVALID)
    with pytest.raises(ValueError, match="branch_classifier does not match"):
        validate_optimizer_credit_state_baseline_witness_receipt(forged)
