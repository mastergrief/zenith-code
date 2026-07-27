"""CPU tests for Fold C drain-path classification audit — PLAN_v3 evidence seam."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
HARNESS_PATH = (
    REPO_ROOT / "scripts" / "optimizer_credit_state_fold_c_drain_path_audit_run.py"
)
PLAN_V3 = (
    REPO_ROOT
    / "artifacts"
    / "acc_entropy"
    / "optimizer_credit_state_fold_c_drain_path_classification_audit_PLAN_v3.json"
)


def _load_harness():
    import sys

    spec = importlib.util.spec_from_file_location(
        "optimizer_credit_state_fold_c_drain_path_audit_run", HARNESS_PATH
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_pending_forbidden_as_terminal_constant():
    h = _load_harness()
    assert h.BRANCH_PENDING not in h.TERMINAL_ALLOWED


def test_seed158_expected_parity_holds_dense_remains():
    h = _load_harness()
    evidence = h.collect_fold_c_evidence(repo_root=REPO_ROOT)
    assert evidence.branch_id == h.BRANCH_PARITY_HOLDS_DENSE_REMAINS
    assert evidence.mismatch_count == 0
    assert "projected_moves" in evidence.observed_dense_surfaces
    assert evidence.credit_scheme == "full_magnitude_ceiling"
    assert evidence.probe_mode == h.PROBE_MODE
    assert evidence.observed_device == "cpu"
    assert dict(evidence.observed_devices_per_tensor) == {
        "weighted_grad": "cpu",
        "credit": "cpu",
        "fp_moves": "cpu",
        "integer_moves": "cpu",
    }


def test_mismatch_count_rejects_bool():
    h = _load_harness()
    with pytest.raises(ValueError, match="exact int"):
        h.validate_mismatch_count_exact_int(True)
    with pytest.raises(ValueError, match="exact int"):
        h.validate_mismatch_count_exact_int(False)


def test_hostile_credit_scheme_drift():
    h = _load_harness()
    with pytest.raises(ValueError, match="credit_scheme drift"):
        h.validate_credit_scheme_exact("pow2_bucket")


def test_hostile_device_nonuniformity():
    h = _load_harness()
    with pytest.raises(ValueError, match="device non-uniform"):
        h.validate_observed_device_uniformity(
            {"weighted_grad": "cpu", "credit": "cuda:0", "fp_moves": "cpu", "integer_moves": "cpu"}
        )


def test_hostile_literal_device_reject():
    h = _load_harness()
    devices = {
        "weighted_grad": "cpu",
        "credit": "cpu",
        "fp_moves": "cpu",
        "integer_moves": "cpu",
    }
    with pytest.raises(ValueError, match="observed_device must equal"):
        h.validate_observed_device_from_bound_tensors_not_literal(
            observed_device="cuda:0",  # literal / not from tensors
            devices_from_tensors=devices,
        )


def test_hostile_hand_authored_mismatch_zero_reject():
    h = _load_harness()
    evidence = h.collect_fold_c_evidence(repo_root=REPO_ROOT)
    receipt = {
        **evidence.to_receipt_fields(),
        "mismatch_count": 0 if evidence.mismatch_count != 0 else 1,
    }
    # Force forged mismatch independent of live evidence when live is 0:
    receipt["mismatch_count"] = 99
    with pytest.raises(ValueError, match="evidence↔receipt inequality"):
        h.validate_fold_c_evidence_receipt_field_equality(evidence, receipt)


def test_hostile_recompute_projection_for_receipt_fields():
    h = _load_harness()
    evidence = h.collect_fold_c_evidence(repo_root=REPO_ROOT)
    receipt = evidence.to_receipt_fields()
    receipt["fp_moves_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="evidence↔receipt inequality"):
        h.validate_fold_c_evidence_receipt_field_equality(evidence, receipt)


def test_hostile_observed_surfaces_order_drift():
    from calm.hrm_text_158.native_full_stack.optimizer_credit_state_no_hidden_fp_audit import (
        compute_canonical_json_sha256,
    )

    a = compute_canonical_json_sha256(["projected_moves", "credit"])
    b = compute_canonical_json_sha256(["credit", "projected_moves"])
    assert a != b


def test_hostile_literal_repo_head():
    h = _load_harness()
    with pytest.raises(ValueError, match="live repo HEAD mismatch"):
        h.validate_live_repo_head_matches_claim(
            claimed="0" * 40,
            live="1" * 40,
        )


def test_hostile_dependency_currency_drift():
    h = _load_harness()
    import json

    plan = json.loads(PLAN_V3.read_text(encoding="utf-8"))
    plan["dependency_currency_freeze"]["files"][0]["expected_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="dependency_currency drift"):
        h.validate_dependency_currency_against_plan_pins(
            plan=plan, repo_root=REPO_ROOT
        )


def test_classify_fractional_collision_branch():
    h = _load_harness()
    assert (
        h.classify_fold_c_branch(
            mismatch_count=3,
            observed_dense_surfaces=("projected_moves",),
            measurement_valid=True,
        )
        == h.BRANCH_FRACTIONAL_COLLISION
    )


def test_governing_receipt_binds_observed_device_and_debt_ceiling():
    h = _load_harness()
    argv = [
        "python",
        "scripts/optimizer_credit_state_fold_c_drain_path_audit_run.py",
        "--plan",
        "artifacts/acc_entropy/optimizer_credit_state_fold_c_drain_path_classification_audit_PLAN_v3.json",
        "--out",
        "artifacts/acc_entropy/optimizer_credit_state_fold_c_drain_path_classification_audit_receipt_v1.json",
    ]
    receipt = h.build_governing_receipt(plan_path=PLAN_V3, argv=argv)
    assert receipt["audit_branch_id"] == h.BRANCH_PARITY_HOLDS_DENSE_REMAINS
    assert receipt["observed_device"] == "cpu"
    assert receipt["backend"]["field_type"] == "declared_contract"
    assert receipt["transient_fp_debt_remains"] is True
    assert receipt["claim_ceiling"]["no_readiness_row_flip"] is True
    assert receipt["claim_ceiling"]["authorizes_readiness_row_flip"] is False
    assert receipt["evidence_to_receipt_field_equality_validated"] is True
    assert receipt["plan_sha256"] == h.PLAN_SHA256_EXPECTED
    # Ensure observed_device is not a plan-declaration copy path: devices map present
    assert receipt["observed_devices_per_tensor"]["weighted_grad"] == "cpu"


def test_device_str_from_tensor_uses_tensor_device():
    h = _load_harness()
    t = torch.zeros(2, dtype=torch.float32)  # cpu tensor
    assert h._device_str_from_tensor(t) == str(t.device)
    assert h._device_str_from_tensor(t) == "cpu"
