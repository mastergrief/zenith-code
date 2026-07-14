"""Option B schema v2 surfaces — characterization + reducers (CPU-only)."""
from __future__ import annotations

import pytest

from calm.hrm_text_158.native_full_stack.forgotten_accum_training_equivalence_bank_eval import (
    CloseSiblingReport,
    ParentFloorStatus,
    claim_blockers_from_close_sibling,
    e_must_match_u_bank,
    evaluate_arm_bank_gate,
    reduce_close_sibling_blockers,
)
from calm.hrm_text_158.native_full_stack.forgotten_accum_training_equivalence_bank_measure import (
    BASE_FORMAL_CLAIM_BLOCKERS,
    BankInputsRefuse,
    build_parent_consistency_mechanism_receipt,
    parse_required_arm_bank_blob,
)


def _report(**kw) -> CloseSiblingReport:
    base = dict(
        numeric_close_sibling_clear=False,
        same_surface_parent_relative_hole_count=0,
        parent_floor_status=ParentFloorStatus.UNRESOLVED_POLICY,
    )
    base.update(kw)
    return CloseSiblingReport(**base)


def test_v1_boolean_blob_schema_invalid_no_true_migration():
    with pytest.raises(BankInputsRefuse, match="SCHEMA_INVALID"):
        parse_required_arm_bank_blob(
            "U",
            {
                "acquire_pct": 95.0,
                "retain_pct_by_support": {"L0b": 91.0, "math_a0": 92.0},
                "clears_by_save": {250: True},
                "parent_consistency_ok": True,
                "close_sibling_ok": True,
            },
        )


def test_cluster_threshold_3_emits_CLOSE_SIBLING_BROAD_CLUSTER():
    report = _report(same_surface_parent_relative_hole_count=3)
    assert "CLOSE_SIBLING_BROAD_CLUSTER" in reduce_close_sibling_blockers(report)
    receipt = evaluate_arm_bank_gate(
        arm="U",
        acquire_pct=95.0,
        retain_pct_by_support={"L0b": 91.0, "math_a0": 92.0},
        clears_by_save={250: True},
        close_sibling_report=report,
    )
    assert not receipt.bank_clear
    assert "CLOSE_SIBLING_BROAD_CLUSTER" in receipt.bank_blockers


def test_sibling_local_OR_numeric_or_cluster_and_floor():
    numeric = _report(numeric_close_sibling_clear=True)
    assert numeric.sibling_local_clear is True
    parent_rel = _report(
        numeric_close_sibling_clear=False,
        same_surface_parent_relative_hole_count=0,
        parent_floor_status=ParentFloorStatus.PASS,
    )
    assert parent_rel.sibling_local_clear is True
    unresolved = _report(
        numeric_close_sibling_clear=False,
        same_surface_parent_relative_hole_count=0,
        parent_floor_status=ParentFloorStatus.UNRESOLVED_POLICY,
    )
    assert unresolved.sibling_local_clear is False


def test_parent_floor_blocker_only_when_floor_defined_else_unresolved_no_invention():
    unresolved = _report(parent_floor_status=ParentFloorStatus.UNRESOLVED_POLICY)
    assert reduce_close_sibling_blockers(unresolved) == []
    assert claim_blockers_from_close_sibling(unresolved) == [
        "PARENT_FLOOR_POLICY_UNRESOLVED"
    ]
    failed = _report(parent_floor_status=ParentFloorStatus.FAIL)
    assert "CLOSE_SIBLING_PARENT_FLOOR" in reduce_close_sibling_blockers(failed)
    assert claim_blockers_from_close_sibling(failed) == []


def test_global_retain_behavior_preserved_not_declared_canonical():
    """Preserve existing numeric >=90 path; do not declare it canonical."""

    weak = evaluate_arm_bank_gate(
        arm="U",
        acquire_pct=95.0,
        retain_pct_by_support={"L0b": 89.0, "math_a0": 92.0},
        clears_by_save={250: True},
        close_sibling_report=_report(numeric_close_sibling_clear=True),
    )
    assert weak.bank_clear is False
    strong = evaluate_arm_bank_gate(
        arm="U",
        acquire_pct=95.0,
        retain_pct_by_support={"L0b": 90.0, "math_a0": 90.0},
        clears_by_save={250: True},
        close_sibling_report=_report(numeric_close_sibling_clear=True),
    )
    assert strong.bank_clear is True


def test_category_separation_pc_mechanism_cannot_affect_bank_clear():
    receipt = evaluate_arm_bank_gate(
        arm="U",
        acquire_pct=95.0,
        retain_pct_by_support={"L0b": 91.0, "math_a0": 92.0},
        clears_by_save={250: True},
        close_sibling_report=_report(numeric_close_sibling_clear=True),
    )
    pc = build_parent_consistency_mechanism_receipt(
        expected_parent_sha256="aa" * 32,
        observed_parent_sha256="bb" * 32,
    )
    assert pc.match is False
    assert pc.as_dict()["may_raise_retain_ok"] is False
    assert pc.as_dict()["may_clear_bank_gate"] is False
    # Mismatched PC does not flip bank_clear.
    assert receipt.bank_clear is True


def test_UE_structured_equality_and_divergence_via_e_must_match_u_bank():
    clears = {250: True}
    sib = _report(numeric_close_sibling_clear=True)
    u = evaluate_arm_bank_gate(
        arm="U",
        acquire_pct=95.0,
        retain_pct_by_support={"L0b": 91.0, "math_a0": 92.0},
        clears_by_save=clears,
        close_sibling_report=sib,
    )
    e_ok = evaluate_arm_bank_gate(
        arm="E",
        acquire_pct=95.0,
        retain_pct_by_support={"L0b": 91.0, "math_a0": 92.0},
        clears_by_save=clears,
        close_sibling_report=sib,
    )
    assert e_must_match_u_bank(u, e_ok)
    e_bad = evaluate_arm_bank_gate(
        arm="E",
        acquire_pct=50.0,
        retain_pct_by_support={"L0b": 50.0, "math_a0": 50.0},
        clears_by_save={250: False},
        close_sibling_report=sib,
    )
    assert not e_must_match_u_bank(u, e_bad)


def test_RULE_CONFLICT_UNRESOLVED_in_base_blockers():
    assert "RULE_CONFLICT_UNRESOLVED" in BASE_FORMAL_CLAIM_BLOCKERS
    assert "A_LEDGER_SYNTHETIC" in BASE_FORMAL_CLAIM_BLOCKERS
