"""CPU tests for 3C readonly classification audit — evidence seam + hash contract."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest
import torch

from calm.hrm_text_158.native_full_stack.optimizer_credit_state import (
    BRANCH_3C_C_AUDIT_PASS_CPU,
    BRANCH_3C_C_DENSE_LEAK,
    OBSERVATION_PROBE_MODE_ALLOC_GUARD,
    OBSERVATION_PROBE_MODE_STATIC,
)
from calm.hrm_text_158.native_full_stack.optimizer_credit_state_no_hidden_fp_audit import (
    AUDIT_NO_DENSE_PROJECTED_MOVES,
    DENSE_SURFACE_PROJECTED_MOVES,
    IntegerPathDenseSurfaceObservationEvidence,
    ProjectedMovesObservationEvidence,
    assert_eligible_modules_owned_by_model,
    build_projected_moves_observation_evidence,
    compute_canonical_json_sha256,
    compute_tensor_canonical_sha256,
    compute_tensor_data_sha256,
    run_integer_path_dense_surface_observation_with_alloc_guard,
    run_optimizer_credit_state_no_hidden_fp_audit,
    validate_canonical_json_digest,
    validate_evidence_receipt_field_equality,
    validate_tensor_digest_matches,
)
from calm.hrm_text_158.bit_linear import BitLinear
from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    build_optimizer_excluding_eligible_masters,
)
from calm.llm_computer.tests.test_hrm_text_158_optimizer_credit_state_proof_contract import (
    _dry_run_fixture,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

def _evidence_bound_receipt(evidence: IntegerPathDenseSurfaceObservationEvidence) -> dict:
    pev = evidence.projected_moves_evidence
    return {
        "projected_moves_sha256": pev.projected_moves_sha256,
        "projected_moves_numel": pev.projected_moves_numel,
        "projected_moves_dtype": pev.projected_moves_dtype,
        "projected_moves_shape": list(pev.projected_moves_shape),
        "move_indices_sha256": pev.move_indices_sha256,
        "move_indices_dtype": pev.move_indices_dtype,
        "move_indices_numel": pev.move_indices_numel,
        "observed_dense_surfaces": list(evidence.observed_surfaces),
        "probe_mode": evidence.probe_mode,
        "observation_evidence_type": "IntegerPathDenseSurfaceObservationEvidence",
        "projected_moves_sha256_sourced_from_evidence_not_recomputed": True,
    }

HARNESS_PATH = REPO_ROOT / "scripts" / "optimizer_credit_state_3C_readonly_audit_run.py"


def _load_harness():
    spec = importlib.util.spec_from_file_location(
        "optimizer_credit_state_3C_readonly_audit_run", HARNESS_PATH
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_audit_no_dense_projected_moves_token_present():
    assert AUDIT_NO_DENSE_PROJECTED_MOVES == "AUDIT-NO-DENSE-PROJECTED-MOVES"
    assert DENSE_SURFACE_PROJECTED_MOVES == "projected_moves"


def test_seed158_alloc_guard_evidence_dense_leak():
    captures, q_flat, weight_shape, _eligible, _model = _dry_run_fixture()
    evidence = run_integer_path_dense_surface_observation_with_alloc_guard(
        captures=captures,
        weight_shape=weight_shape,
        q_flat=q_flat,
    )
    assert isinstance(evidence, IntegerPathDenseSurfaceObservationEvidence)
    assert evidence.probe_mode == OBSERVATION_PROBE_MODE_ALLOC_GUARD
    assert DENSE_SURFACE_PROJECTED_MOVES in evidence.observed_surfaces
    assert evidence.projected_moves_evidence.projected_moves_numel == 6
    assert evidence.projected_moves_evidence.projected_moves_dtype == "torch.int8"
    audit = run_optimizer_credit_state_no_hidden_fp_audit(
        optimizer_checks={
            "eligible_params_in_optimizer": 0,
            "eligible_optimizer_state_entries": 0,
        },
        observed_dense_surfaces=evidence.observed_surfaces,
        observation_probe_mode=evidence.probe_mode,
        audit_observation_complete=True,
    )
    assert audit.branch_id == BRANCH_3C_C_DENSE_LEAK
    # DENSE_LEAK_EXPECTED_SEED158


def test_pending_forbidden_as_terminal_constant():
    from calm.hrm_text_158.native_full_stack.optimizer_credit_state import (
        BRANCH_3C_C_AUDIT_PENDING,
    )

    assert BRANCH_3C_C_AUDIT_PENDING == "BR-3C-C-AUDIT-PENDING"


def test_canonical_hash_contract_envelope():
    t = torch.tensor([1, 2, 3], dtype=torch.int8)
    data = compute_tensor_data_sha256(t)
    outer = compute_tensor_canonical_sha256(t)
    validate_tensor_digest_matches(t, outer)
    # dtype drift
    t2 = t.to(torch.int16)
    with pytest.raises(ValueError, match="tensor digest mismatch"):
        validate_tensor_digest_matches(t2, outer)
    # shape drift
    t3 = t.reshape(3, 1)
    with pytest.raises(ValueError, match="tensor digest mismatch"):
        validate_tensor_digest_matches(t3, outer)
    # value drift
    t4 = t.clone()
    t4[0] = 9
    with pytest.raises(ValueError, match="tensor digest mismatch"):
        validate_tensor_digest_matches(t4, outer)
    # ordering drift on maps/lists
    obj = {"b": 1, "a": [2, 1]}
    digest = compute_canonical_json_sha256(obj)
    validate_canonical_json_digest(obj, digest)
    with pytest.raises(ValueError, match="canonical JSON digest mismatch"):
        validate_canonical_json_digest({"a": [1, 2], "b": 1}, digest)
    assert data != outer  # CANONICAL_HASH_CONTRACT — envelope wraps data


def test_hostile_hash_drift_reject_observed_surfaces_order():
    surfaces_a = ["projected_moves", "credit"]
    surfaces_b = ["credit", "projected_moves"]
    # canonical JSON of list is order-significant
    da = compute_canonical_json_sha256(surfaces_a)
    db = compute_canonical_json_sha256(surfaces_b)
    assert da != db


def test_hostile_recompute_reject_for_receipt():
    """Recomputing projected_moves for receipt hashing must be rejected."""
    captures, q_flat, weight_shape, _eligible, _model = _dry_run_fixture()
    evidence = run_integer_path_dense_surface_observation_with_alloc_guard(
        captures=captures,
        weight_shape=weight_shape,
        q_flat=q_flat,
    )
    receipt = _evidence_bound_receipt(evidence)
    receipt["projected_moves_sha256"] = "0" * 64  # forged recompute
    with pytest.raises(ValueError, match="evidence↔receipt inequality"):
        validate_evidence_receipt_field_equality(evidence, receipt)


def test_hostile_manual_injection_reject():
    captures, q_flat, weight_shape, _eligible, _model = _dry_run_fixture()
    evidence = run_integer_path_dense_surface_observation_with_alloc_guard(
        captures=captures,
        weight_shape=weight_shape,
        q_flat=q_flat,
    )
    receipt = _evidence_bound_receipt(evidence)
    receipt["projected_moves_numel"] = 999  # injected
    with pytest.raises(ValueError, match="evidence↔receipt inequality"):
        validate_evidence_receipt_field_equality(evidence, receipt)


def test_hostile_hand_authored_empty_observations_do_not_launder_via_evidence_equality():
    """Empty hand-authored surfaces cannot claim evidence-seam equality with live evidence."""
    captures, q_flat, weight_shape, _eligible, _model = _dry_run_fixture()
    evidence = run_integer_path_dense_surface_observation_with_alloc_guard(
        captures=captures,
        weight_shape=weight_shape,
        q_flat=q_flat,
    )
    receipt = _evidence_bound_receipt(evidence)
    receipt["observed_dense_surfaces"] = []  # hand-authored empty
    with pytest.raises(ValueError, match="evidence↔receipt inequality"):
        validate_evidence_receipt_field_equality(evidence, receipt)


def test_hostile_row_flip_still_false_on_dense_leak():
    captures, q_flat, weight_shape, _eligible, _model = _dry_run_fixture()
    evidence = run_integer_path_dense_surface_observation_with_alloc_guard(
        captures=captures,
        weight_shape=weight_shape,
        q_flat=q_flat,
    )
    audit = run_optimizer_credit_state_no_hidden_fp_audit(
        optimizer_checks={
            "eligible_params_in_optimizer": 0,
            "eligible_optimizer_state_entries": 0,
        },
        observed_dense_surfaces=evidence.observed_surfaces,
        observation_probe_mode=evidence.probe_mode,
        audit_observation_complete=True,
    )
    assert audit.branch_id == BRANCH_3C_C_DENSE_LEAK
    assert audit.ready_to_flip is False
    assert audit.optimizer_state_eligible_exclusion_proven is False


def test_static_empty_still_can_pass_cpu_unit_characterization():
    """Blindness is observation-layer; STATIC empty unit tests still AUDIT_PASS_CPU."""
    audit = run_optimizer_credit_state_no_hidden_fp_audit(
        optimizer_checks={
            "eligible_params_in_optimizer": 0,
            "eligible_optimizer_state_entries": 0,
        },
        observed_dense_surfaces=(),
        observation_probe_mode=OBSERVATION_PROBE_MODE_STATIC,
        audit_observation_complete=True,
    )
    assert audit.branch_id == BRANCH_3C_C_AUDIT_PASS_CPU


def test_evidence_receipt_equality_happy_path():
    captures, q_flat, weight_shape, _eligible, _model = _dry_run_fixture()
    evidence = run_integer_path_dense_surface_observation_with_alloc_guard(
        captures=captures,
        weight_shape=weight_shape,
        q_flat=q_flat,
    )
    receipt = _evidence_bound_receipt(evidence)
    validate_evidence_receipt_field_equality(evidence, receipt)
    receipt["evidence_to_receipt_field_equality_validated"] = True


def test_harness_source_has_no_second_attribution_call():
    """Implement-gate check: harness must not recompute projected_moves."""
    src = HARNESS_PATH.read_text(encoding="utf-8")
    # Allow the comment forbidding the call; forbid an import/call of the symbol.
    assert "from calm.hrm_text_158.native_full_stack.integer_marginal_attribution import" not in src
    assert "projected_moves_from_integer_attribution(" not in src
    assert "IntegerPathDenseSurfaceObservationEvidence" in src or "run_integer_path_dense_surface_observation_with_alloc_guard" in src


def test_build_projected_moves_evidence_from_tensors():
    moves = torch.tensor([1, 0, -1], dtype=torch.int8)
    indices = torch.tensor([0, 1, 2], dtype=torch.int64)
    pev = build_projected_moves_observation_evidence(
        projected_moves=moves, move_indices=indices
    )
    assert isinstance(pev, ProjectedMovesObservationEvidence)
    validate_tensor_digest_matches(moves, pev.projected_moves_sha256)

def test_hostile_move_indices_and_shape_field_census_reject():
    captures, q_flat, weight_shape, _eligible, _model = _dry_run_fixture()
    evidence = run_integer_path_dense_surface_observation_with_alloc_guard(
        captures=captures,
        weight_shape=weight_shape,
        q_flat=q_flat,
    )
    for field, bad in (
        ("projected_moves_shape", [999]),
        ("move_indices_sha256", "f" * 64),
        ("move_indices_dtype", "torch.int32"),
        ("move_indices_numel", -1),
    ):
        receipt = _evidence_bound_receipt(evidence)
        receipt[field] = bad
        with pytest.raises(ValueError, match="evidence↔receipt inequality"):
            validate_evidence_receipt_field_equality(evidence, receipt)


def test_hostile_mismatched_model_eligible_rejected():
    """Fresh model + fixture eligible must hard-fail object-identity assert."""
    _captures, _q, _ws, eligible, owning_model = _dry_run_fixture()
    assert_eligible_modules_owned_by_model(owning_model, eligible)

    class _Other(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = BitLinear(3, 2, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.proj(x)

    other = _Other()
    with pytest.raises(ValueError, match="object-identity mismatch"):
        assert_eligible_modules_owned_by_model(other, eligible)


def test_same_model_optimizer_checks_pass_with_ownership():
    captures, q_flat, weight_shape, eligible, model = _dry_run_fixture()
    assert_eligible_modules_owned_by_model(model, eligible)
    opt, checks = build_optimizer_excluding_eligible_masters(model, eligible, lr=0.0)
    # Tiny fixture is all-eligible → opt is None under honest same-model exclusion.
    assert opt is None
    assert checks["optimizer_created"] is False
    assert checks["eligible_params_in_optimizer"] == 0
    assert checks["eligible_optimizer_state_entries"] == 0
    assert checks["pass"] is True
    evidence = run_integer_path_dense_surface_observation_with_alloc_guard(
        captures=captures,
        weight_shape=weight_shape,
        q_flat=q_flat,
    )
    audit = run_optimizer_credit_state_no_hidden_fp_audit(
        optimizer_checks=checks,
        observed_dense_surfaces=evidence.observed_surfaces,
        observation_probe_mode=evidence.probe_mode,
        audit_observation_complete=True,
    )
    assert audit.branch_id == BRANCH_3C_C_DENSE_LEAK
