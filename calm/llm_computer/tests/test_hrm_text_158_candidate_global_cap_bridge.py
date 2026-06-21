"""B2-5c Step-1a candidate+global-cap bridge reference tests (CPU-only)."""
from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from calm.hrm_text_158.native_full_stack.bounded_delta_accumulator import (
    encode_budget_capped_hybrid_reference,
    execute_direct_bounded_local_vote_update_candidate,
)
from calm.hrm_text_158.native_full_stack.candidate_global_cap_bridge_facade import (
    run_candidate_global_cap_bridge_suite,
)
from calm.hrm_text_158.native_full_stack.candidate_global_cap_bridge_receipt import (
    CANDIDATE_GLOBAL_CAP_BRIDGE_HARD_FALSE_FIELDS,
    CANDIDATE_GLOBAL_CAP_BRIDGE_NON_CLAIMS,
    COMPOSITION_GUARD_ANCHOR,
    build_candidate_global_cap_bridge_receipt,
    validate_candidate_global_cap_bridge_receipt,
)
from calm.hrm_text_158.native_full_stack.candidate_global_cap_bridge_reference import (
    BridgeFixtureSpec,
    build_classifier_negative_bridge_measurements,
    build_representative_bridge_measurements,
    materialize_bridge_artifacts_from_candidate_result,
    run_bridge_fixture,
    verify_materialization_fidelity_lattice,
    _build_state,
    _vote_spec,
)


def _assert_real_sha256(value: str) -> None:
    assert len(value) == 64
    assert value != "0" * 64
    int(value, 16)


def test_default_suite_bridge_equivalent_with_real_execution():
    receipt = run_candidate_global_cap_bridge_suite()
    assert receipt.measurement_representative is True
    assert receipt.aggregate_bridge_equivalent is True
    assert receipt.composition_path_exists is False
    assert receipt.composition_guard_anchor == COMPOSITION_GUARD_ANCHOR
    for field in CANDIDATE_GLOBAL_CAP_BRIDGE_HARD_FALSE_FIELDS:
        assert getattr(receipt, field) is False
    for claim in CANDIDATE_GLOBAL_CAP_BRIDGE_NON_CLAIMS:
        assert claim in receipt.non_claims

    execution_rows = [
        row
        for row in receipt.representative_measurements
        if not row.structural_candidate_global_cap_reject
    ]
    assert len(execution_rows) == 2
    assert {row.fixture_name for row in execution_rows} == {
        "F_BRIDGE_MINIMAL",
        "F_BRIDGE_SATURATED",
    }
    assert any(
        row.structural_candidate_global_cap_reject
        for row in receipt.representative_measurements
    )
    for row in execution_rows:
        assert row.total_sparse_event_count > 0
        assert row.magnitude_regime == "no_clip_exact_add_back"
        assert row.add_back_clip_boundary_reconciliation is False
        assert row.fidelity_lattice_pass is True
        assert row.bridge_equivalent is True
        assert row.step1a_novel_claim_materialization_fidelity is True
        assert row.step1a_novel_claim_cap_api_composability is True
        _assert_real_sha256(row.candidate_applied_row_identities_sha256)
        _assert_real_sha256(row.bridge_accepted_identities_sha256)
        _assert_real_sha256(row.oracle_accepted_identities_sha256)
        assert (
            row.bridge_accepted_identities_sha256
            == row.oracle_accepted_identities_sha256
        )
    saturated = next(
        row for row in execution_rows if row.fixture_name == "F_BRIDGE_SATURATED"
    )
    assert saturated.step1a_novel_claim_saturated_margin_ordering_identity is True


def test_build_rejects_classifier_negatives_by_default():
    measurements = build_representative_bridge_measurements()
    with pytest.raises(ValueError, match="include_classifier_negatives=True"):
        build_candidate_global_cap_bridge_receipt(
            fixture_measurements=(*measurements, *build_classifier_negative_bridge_measurements()),
            include_classifier_negatives=False,
        )


def test_classifier_negatives_isolated_when_enabled():
    receipt = run_candidate_global_cap_bridge_suite(include_classifier_negatives=True)
    assert receipt.include_classifier_negatives is True
    assert receipt.aggregate_bridge_equivalent is True
    negative_map = {row.fixture_name: row for row in receipt.classifier_negative_results}
    assert negative_map["F_NEG_ZERO_SPARSE"].fidelity_lattice_pass is False
    assert negative_map["F_CLIP_BOUNDARY"].add_back_clip_boundary_reconciliation is True
    assert negative_map["F_CLIP_BOUNDARY"].bridge_equivalent is False


def test_materialization_tamper_fails_lattice():
    spec = BridgeFixtureSpec(
        fixture_name="F_TAMPER_PROBE",
        fixture_role="classifier_negative",
        state_key="F_TAMPER_PROBE",
        numel=4,
        acc_overrides={0: 9, 2: -9},
        sparse_votes={0: 2, 2: -2},
        hot_exact_indices=(0, 2),
        cap=10,
        max_abs_per_tensor=4,
    )
    state = _build_state(spec.numel, acc_overrides=spec.acc_overrides)
    vote_spec = _vote_spec(max_abs_per_tensor=spec.max_abs_per_tensor)
    bounded = encode_budget_capped_hybrid_reference(
        state,
        hot_exact_indices=spec.hot_exact_indices,
        cold_default_value=0,
    )
    candidate_result = execute_direct_bounded_local_vote_update_candidate(
        state_key=spec.state_key,
        q_levels=state.q_levels,
        bounded_accumulator=bounded,
        sparse_vote_events=spec.sparse_votes,
        vote_spec=vote_spec,
    )
    artifacts = materialize_bridge_artifacts_from_candidate_result(
        state_key=spec.state_key,
        prior_state=state,
        candidate_result=candidate_result,
        vote_spec=vote_spec,
    )
    with pytest.raises(ValueError, match="residual_after_threshold_sha256 mismatch"):
        verify_materialization_fidelity_lattice(
            artifacts,
            tamper_index=artifacts.applied_indices[0],
            tamper_delta=1,
        )


def test_forced_diverge_probe_not_bridge_equivalent():
    representative = build_representative_bridge_measurements()
    diverged = replace(
        representative[0],
        bridge_equivalent=False,
        accepted_identities_match=False,
    )
    receipt = build_candidate_global_cap_bridge_receipt(
        fixture_measurements=(diverged, *representative[1:]),
        include_classifier_negatives=False,
    )
    with pytest.raises(ValueError, match="bridge_equivalent"):
        validate_candidate_global_cap_bridge_receipt(receipt)


def test_pre_cap_mismatch_probe_fidelity_fail():
    representative = build_representative_bridge_measurements()
    mismatched = replace(
        representative[0],
        fidelity_lattice_pass=False,
        step1a_novel_claim_materialization_fidelity=True,
    )
    with pytest.raises(ValueError, match="materialization claim requires fidelity pass"):
        validate_candidate_global_cap_bridge_receipt(
            build_candidate_global_cap_bridge_receipt(
                fixture_measurements=(mismatched, *representative[1:]),
                include_classifier_negatives=False,
            ),
        )


def test_no_native_dispatch_import_in_bridge_modules():
    repo_root = Path(__file__).resolve().parents[2]
    module_dir = repo_root / "hrm_text_158" / "native_full_stack"
    targets = (
        "candidate_global_cap_bridge_receipt.py",
        "candidate_global_cap_bridge_reference.py",
        "candidate_global_cap_bridge_facade.py",
    )
    for name in targets:
        source = (module_dir / name).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = {
            node.names[0].name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        assert "native_dispatch" not in str(source)
        assert not any("native_dispatch" in (imp or "") for imp in imports)
