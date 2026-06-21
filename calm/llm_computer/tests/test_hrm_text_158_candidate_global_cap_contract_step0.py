"""B2-5c Step-0 candidate↔global-cap contract characterization tests (CPU-only)."""
from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import pytest

from calm.hrm_text_158.native_full_stack.candidate_global_cap_contract_step0_facade import (
    run_candidate_global_cap_contract_step0_suite,
)
from calm.hrm_text_158.native_full_stack.candidate_global_cap_contract_step0_measurement import (
    build_classifier_negative_measurements,
    build_representative_consumer_measurements,
    measure_paired_fixture,
    _PairedFixtureSpec,
)
from calm.hrm_text_158.native_full_stack.candidate_global_cap_contract_step0_receipt import (
    CANDIDATE_GLOBAL_CAP_CONTRACT_STEP0_HARD_FALSE_FIELDS,
    CANDIDATE_GLOBAL_CAP_CONTRACT_STEP0_NON_CLAIMS,
    COMPOSITION_GUARD_ANCHOR,
    PINNED_SURFACES_FULL_EXECUTION,
    CandidateGlobalCapContractBranchId,
    CandidateGlobalCapContractFixtureMeasurement,
    build_candidate_global_cap_contract_step0_receipt,
    classify_aggregate_branch,
    classify_fixture_branch_probe,
    composition_path_exists,
    validate_candidate_global_cap_contract_step0_receipt,
)


def _assert_real_sha256(value: str) -> None:
    assert len(value) == 64
    assert value != "0" * 64
    int(value, 16)


def test_default_suite_bridge_aggregate_with_real_execution():
    receipt = run_candidate_global_cap_contract_step0_suite()
    validate_candidate_global_cap_contract_step0_receipt(receipt)
    assert receipt.measurement_representative is True
    assert receipt.aggregate_branch_id == CandidateGlobalCapContractBranchId.BRIDGE_IMPLEMENTATION
    assert receipt.composition_path_exists is False
    assert receipt.composition_guard_anchor == COMPOSITION_GUARD_ANCHOR
    assert not receipt.include_classifier_negatives
    assert not receipt.classifier_negative_results
    for field in CANDIDATE_GLOBAL_CAP_CONTRACT_STEP0_HARD_FALSE_FIELDS:
        assert getattr(receipt, field) is False
    for claim in CANDIDATE_GLOBAL_CAP_CONTRACT_STEP0_NON_CLAIMS:
        assert claim in receipt.non_claims

    execution_rows = [
        row
        for row in receipt.representative_measurements
        if row.fixture_tier != "structural"
    ]
    assert len(execution_rows) == 2
    tiers = {row.fixture_tier for row in execution_rows}
    assert tiers == {"minimal", "saturated"}
    assert any(row.structural_candidate_global_cap_reject for row in receipt.representative_measurements)
    for row in execution_rows:
        assert row.total_sparse_event_count > 0
        assert set(PINNED_SURFACES_FULL_EXECUTION).issubset(set(row.pinned_surfaces))
        assert row.shadow_mutation_observed is False
        _assert_real_sha256(row.candidate_applied_row_identities_sha256)
        _assert_real_sha256(row.exact_local_applied_row_identities_sha256)
        _assert_real_sha256(row.candidate_residual_after_threshold_sha256)
        _assert_real_sha256(row.exact_local_residual_after_threshold_sha256)
        assert (
            row.candidate_applied_row_identities_sha256
            == row.exact_local_applied_row_identities_sha256
        )
        assert (
            row.candidate_residual_after_threshold_sha256
            == row.exact_local_residual_after_threshold_sha256
        )
        assert row.identity_set_match is True
        assert row.direction_match is True
        assert row.residual_hash_match is True
        assert row.ordering_match is True
        assert row.global_cap_pure_subset_of_local_universe is True
        assert row.deferred_backlog_authority_defined is True
    assert any(row.saturation_exercised for row in execution_rows)


def test_build_rejects_classifier_negatives_by_default():
    measurements = build_representative_consumer_measurements()
    with pytest.raises(ValueError, match="include_classifier_negatives=True"):
        build_candidate_global_cap_contract_step0_receipt(
            fixture_measurements=(*measurements, *build_classifier_negative_measurements()),
            include_classifier_negatives=False,
        )


def test_classifier_negatives_isolated_when_enabled():
    receipt = run_candidate_global_cap_contract_step0_suite(include_classifier_negatives=True)
    validate_candidate_global_cap_contract_step0_receipt(receipt)
    assert receipt.include_classifier_negatives is True
    assert receipt.aggregate_branch_id == CandidateGlobalCapContractBranchId.BRIDGE_IMPLEMENTATION
    negative_map = {
        result.fixture_name: result.branch_id for result in receipt.classifier_negative_results
    }
    assert negative_map["F_NEG_ZERO_SPARSE"] == CandidateGlobalCapContractBranchId.MEASUREMENT_INVALID
    assert (
        negative_map["F_NEG_FORCED_IDENTITY_DIVERGE"]
        == CandidateGlobalCapContractBranchId.RECONCILIATION_CONTRACT
    )


def test_classify_aggregate_branch_priority():
    representative = build_representative_consumer_measurements()
    assert classify_aggregate_branch(representative) == (
        CandidateGlobalCapContractBranchId.BRIDGE_IMPLEMENTATION
    )
    diverged = replace(
        representative[0],
        identity_set_match=False,
        direction_match=False,
        residual_hash_match=False,
    )
    assert classify_aggregate_branch((diverged, representative[1], representative[2])) == (
        CandidateGlobalCapContractBranchId.RECONCILIATION_CONTRACT
    )
    if composition_path_exists():
        assert classify_aggregate_branch(representative) == (
            CandidateGlobalCapContractBranchId.PROOF_EXTENSION
        )


def test_fixture_branch_probe_measurement_invalid_and_reconciliation():
    zero_sparse, _, forced = build_classifier_negative_measurements()
    assert classify_fixture_branch_probe(zero_sparse) == (
        CandidateGlobalCapContractBranchId.MEASUREMENT_INVALID
    )
    assert classify_fixture_branch_probe(forced) == (
        CandidateGlobalCapContractBranchId.RECONCILIATION_CONTRACT
    )


def test_no_native_dispatch_import_in_step0_modules():
    root = Path(__file__).resolve().parents[2] / "hrm_text_158" / "native_full_stack"
    module_names = (
        "candidate_global_cap_contract_step0_receipt.py",
        "candidate_global_cap_contract_step0_measurement.py",
        "candidate_global_cap_contract_step0_facade.py",
    )
    for module_name in module_names:
        tree = ast.parse((root / module_name).read_text(encoding="utf-8"))
        imports = {
            node.names[0].name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        assert "native_dispatch" not in str(imports)


def test_exact_local_hashes_do_not_echo_candidate_placeholders():
    row = measure_paired_fixture(
        _PairedFixtureSpec(
            fixture_name="probe.echo",
            fixture_role="representative_consumer",
            fixture_tier="minimal",
            state_key="probe.echo",
            numel=4,
            acc_overrides={0: 9, 2: -9},
            sparse_votes={0: 2, 2: -2},
            hot_exact_indices=(0, 2),
            cap=10,
            max_abs_per_tensor=4,
        ),
    )
    _assert_real_sha256(row.candidate_applied_row_identities_sha256)
    _assert_real_sha256(row.exact_local_applied_row_identities_sha256)
    assert row.candidate_applied_row_identities_sha256 == row.exact_local_applied_row_identities_sha256
    assert row.candidate_residual_after_threshold_sha256 == row.exact_local_residual_after_threshold_sha256


def test_saturated_fixture_exercises_defer_and_subset():
    measurements = build_representative_consumer_measurements()
    saturated = next(row for row in measurements if row.fixture_name == "F_PAIR_SATURATED_CAP")
    assert saturated.deferred_count > 0
    assert saturated.saturation_exercised is True
    assert saturated.cap < saturated.exact_local_pre_cap_demand_count
    assert saturated.accepted_count == saturated.cap


def test_hard_false_non_claims_on_receipt():
    receipt = run_candidate_global_cap_contract_step0_suite()
    for field in CANDIDATE_GLOBAL_CAP_CONTRACT_STEP0_HARD_FALSE_FIELDS:
        with pytest.raises(ValueError, match=field):
            validate_candidate_global_cap_contract_step0_receipt(
                replace(receipt, **{field: True}),
            )
