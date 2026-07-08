"""CPU-static tests for Arc #2b Slice-5 discovery Arm A static harness."""

from __future__ import annotations

from scripts.hrm_text_158_arc2b_slice5_discovery_arm_a_static import (
    ARM_A_W7_BPW,
    ARM_A_W8_BPW,
    build_arm_a_receipt,
    can_reach_sub2,
    compute_w7_bpw,
    compute_w7_gap_bpw,
    compute_w8_bpw,
    compute_w8_gap_bpw,
)
from calm.hrm_text_158.native_full_stack.arc2b_slice5_discovery_branch import (
    DEFAULT_EFFECTIVE_ACC_BUDGET_BPW,
    EVIDENCE_ARM_A_STATIC,
    RECEIPT_SCHEMA,
)


def test_w8_bpw_is_8() -> None:
    assert compute_w8_bpw() == 8.0
    assert ARM_A_W8_BPW == 8


def test_w7_bpw_is_7() -> None:
    assert compute_w7_bpw() == 7.0
    assert ARM_A_W7_BPW == 7


def test_w8_gap_over_budget() -> None:
    gap = compute_w8_gap_bpw()
    assert gap == 8.0 - DEFAULT_EFFECTIVE_ACC_BUDGET_BPW
    assert gap > 0  # way over budget


def test_w7_gap_over_budget() -> None:
    gap = compute_w7_gap_bpw()
    assert gap == 7.0 - DEFAULT_EFFECTIVE_ACC_BUDGET_BPW
    assert gap > 0  # way over budget


def test_neither_can_reach_sub2() -> None:
    """Fixed-width W8=8, W7=7 cannot reach sub-2 (0.4 bpw strict)."""
    assert can_reach_sub2(8.0) is False
    assert can_reach_sub2(7.0) is False
    assert can_reach_sub2(0.3) is True  # hypothetical sub-2 value


def test_build_arm_a_receipt_schema() -> None:
    receipt = build_arm_a_receipt()
    assert receipt["schema"] == RECEIPT_SCHEMA
    assert receipt["evidence_source"] == EVIDENCE_ARM_A_STATIC
    assert receipt["arm_a_bpw_w8"] == 8.0
    assert receipt["arm_a_bpw_w7"] == 7.0
    assert receipt["arm_a_w8_can_reach_sub2"] is False
    assert receipt["arm_a_w7_can_reach_sub2"] is False
    assert receipt["ready_for_main_science"] is False
    assert receipt["counts_as_sub2"] is False
    assert receipt["pre_full_stack_diagnostic"] is True
    assert receipt["autonomy_rung"] == "arm_a_static_cpu"


def test_build_arm_a_receipt_finding() -> None:
    receipt = build_arm_a_receipt()
    finding = receipt["arm_a_finding"]
    assert "fixed_width_dense_accumulator_cannot_reach_sub2_ceiling" in finding
    assert "W8=8.0" in finding
    assert "W7=7.0" in finding


def test_build_arm_a_receipt_custom_budget() -> None:
    receipt = build_arm_a_receipt(effective_acc_budget_bpw=0.5)
    assert receipt["effective_acc_budget_bpw"] == 0.5
    assert receipt["arm_a_gap_w8"] == 7.5
    assert receipt["arm_a_gap_w7"] == 6.5
