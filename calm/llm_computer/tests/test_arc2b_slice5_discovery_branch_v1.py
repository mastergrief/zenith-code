"""CPU-static tests for Arc #2b Slice-5 discovery 2x2 branch classifier."""

from __future__ import annotations

from calm.hrm_text_158.native_full_stack.arc2b_slice5_discovery_branch import (
    CLASSIFIER,
    DEFAULT_DIRECTION_TOL_FACTOR,
    DEFAULT_MATERIALITY_FACTOR,
    MECHANISM_TERMINALS,
    OPERATIONAL_TERMINALS,
    Arc2bSlice5DiscoveryBranch,
    arm_eligible,
    classify_arc2b_slice5_discovery_branch,
    compute_budget_gap_bpw,
    validate_discovery_receipt_schema,
    validate_lane_fields,
)


def _base_eligible_inputs(
    *,
    gap_c: float = 100.0,
    gap_d: float = 100.0,
    gap_e: float = 100.0,
    arm_d_bpw_strict: float | None = None,
) -> dict:
    return {
        "evidence_source": "live_decay_curve",
        "schema_ok": True,
        "arm_c_eligible": True,
        "arm_d_eligible": True,
        "arm_e_eligible": True,
        "gap_c_bpw": gap_c,
        "gap_d_bpw": gap_d,
        "gap_e_bpw": gap_e,
        "materiality_factor": DEFAULT_MATERIALITY_FACTOR,
        "direction_tol_factor": DEFAULT_DIRECTION_TOL_FACTOR,
        "arm_d_bpw_strict": arm_d_bpw_strict,
    }


def test_classifier_name() -> None:
    assert CLASSIFIER == "ARC2B_SLICE5_DISCOVERY_BRANCH_V1"


def test_cell1_representation_new_mechanism() -> None:
    """not MF and not LF -> REPRESENTATION_NEW_MECHANISM."""
    # gap(D)=100, threshold=80. gap_c=90 (>=80), gap_e=90 (>=80) => neither improves
    inputs = _base_eligible_inputs(gap_c=90.0, gap_d=100.0, gap_e=90.0)
    result = classify_arc2b_slice5_discovery_branch(inputs)
    assert result["terminal_branch"] == "REPRESENTATION_NEW_MECHANISM"
    assert result["mf_boolean"] is False
    assert result["lf_boolean"] is False
    assert result["terminal_branch"] in MECHANISM_TERMINALS


def test_cell1_sweet_spot_sublabel() -> None:
    """1/2-sweet-spot sub-label: gap(1/2) <= both neighbors."""
    # gap_d <= gap_c and gap_d <= gap_e
    inputs = _base_eligible_inputs(gap_c=100.0, gap_d=90.0, gap_e=100.0)
    result = classify_arc2b_slice5_discovery_branch(inputs)
    assert result["terminal_branch"] == "REPRESENTATION_NEW_MECHANISM"
    assert result["sweet_spot_sublabel"] is True


def test_cell2_more_forgetting_helps() -> None:
    """MF and not LF -> MORE_FORGETTING_HELPS (toward faster 1/4)."""
    # gap(D)=100, threshold=80. gap_c=70 (<80 => MF), gap_e=90 (>=80 => not LF)
    inputs = _base_eligible_inputs(gap_c=70.0, gap_d=100.0, gap_e=90.0)
    result = classify_arc2b_slice5_discovery_branch(inputs)
    assert result["terminal_branch"] == "MORE_FORGETTING_HELPS"
    assert result["mf_boolean"] is True
    assert result["lf_boolean"] is False
    assert result["decay_direction"] == "faster_1_over_4"


def test_cell3_less_forgetting_helps() -> None:
    """not MF and LF -> LESS_FORGETTING_HELPS (toward slower 9/10)."""
    # gap(D)=100, threshold=80. gap_c=90 (>=80 => not MF), gap_e=70 (<80 => LF)
    inputs = _base_eligible_inputs(gap_c=90.0, gap_d=100.0, gap_e=70.0)
    result = classify_arc2b_slice5_discovery_branch(inputs)
    assert result["terminal_branch"] == "LESS_FORGETTING_HELPS"
    assert result["mf_boolean"] is False
    assert result["lf_boolean"] is True
    assert result["decay_direction"] == "slower_9_over_10"


def test_cell4_both_improve_argmin() -> None:
    """MF and LF -> BOTH-IMPROVE; direction = argmin."""
    # gap(D)=100, threshold=80. gap_c=60 (<80 => MF), gap_e=70 (<80 => LF)
    # direction_diff = |60-70| = 10, tol = 0.1*100 = 10. 10 < 10 is False => BOTH_IMPROVE
    inputs = _base_eligible_inputs(gap_c=60.0, gap_d=100.0, gap_e=70.0)
    result = classify_arc2b_slice5_discovery_branch(inputs)
    assert result["terminal_branch"] == "BOTH_IMPROVE"
    assert result["mf_boolean"] is True
    assert result["lf_boolean"] is True
    assert result["decay_direction"] == "faster_1_over_4"  # gap_c=60 <= gap_e=70


def test_cell4_decay_direction_ambiguous() -> None:
    """MF and LF, |gap_c-gap_e| < tol => DECAY_DIRECTION_AMBIGUOUS."""
    # gap(D)=100, threshold=80, tol=10. gap_c=75 (<80 => MF), gap_e=76 (<80 => LF)
    # direction_diff = |75-76| = 1 < 10 => AMBIGUOUS
    inputs = _base_eligible_inputs(gap_c=75.0, gap_d=100.0, gap_e=76.0)
    result = classify_arc2b_slice5_discovery_branch(inputs)
    assert result["terminal_branch"] == "DECAY_DIRECTION_AMBIGUOUS"
    assert result["mf_boolean"] is True
    assert result["lf_boolean"] is True


def test_cell4_ambiguous_overridden_when_d_under_budget() -> None:
    """MF and LF, |gap_c-gap_e| < tol, BUT D already < 0.4 => argmin (not ambiguous)."""
    # gap(D)=100, threshold=80, tol=10. gap_c=75, gap_e=76, direction_diff=1 < 10
    # BUT arm_d_bpw_strict=0.3 < 0.4 => d_already_under_budget=True => BOTH_IMPROVE
    inputs = _base_eligible_inputs(
        gap_c=75.0, gap_d=100.0, gap_e=76.0, arm_d_bpw_strict=0.3
    )
    result = classify_arc2b_slice5_discovery_branch(inputs)
    assert result["terminal_branch"] == "BOTH_IMPROVE"
    assert result["d_already_under_budget"] is True


def test_operational_guard_missing_arm() -> None:
    """Any live arm ineligible => INCONCLUSIVE_MISSING_ARM."""
    inputs = _base_eligible_inputs()
    inputs["arm_c_eligible"] = False
    result = classify_arc2b_slice5_discovery_branch(inputs)
    assert result["terminal_branch"] == "DISCOVERY_INCONCLUSIVE_MISSING_ARM"
    assert result["terminal_branch"] not in MECHANISM_TERMINALS
    assert result["terminal_branch"] in OPERATIONAL_TERMINALS


def test_operational_guard_liveness_failure() -> None:
    """Liveness failure => INCONCLUSIVE_LIVENESS_FAILURE."""
    inputs = _base_eligible_inputs()
    inputs["liveness_failure"] = True
    result = classify_arc2b_slice5_discovery_branch(inputs)
    assert result["terminal_branch"] == "DISCOVERY_INCONCLUSIVE_LIVENESS_FAILURE"
    assert result["terminal_branch"] not in MECHANISM_TERMINALS


def test_operational_guard_schema_failure() -> None:
    """Schema not ok => NO_VERDICT_SCHEMA."""
    inputs = _base_eligible_inputs()
    inputs["schema_ok"] = False
    result = classify_arc2b_slice5_discovery_branch(inputs)
    assert result["terminal_branch"] == "DISCOVERY_NO_VERDICT_SCHEMA"
    assert result["terminal_branch"] not in MECHANISM_TERMINALS


def test_inconclusive_log_coverage_missing_gaps() -> None:
    """Missing gap values => INCONCLUSIVE_LOG_COVERAGE."""
    inputs = _base_eligible_inputs()
    inputs["gap_c_bpw"] = None
    result = classify_arc2b_slice5_discovery_branch(inputs)
    assert result["terminal_branch"] == "DISCOVERY_INCONCLUSIVE_LOG_COVERAGE"
    assert result["terminal_branch"] not in MECHANISM_TERMINALS


def test_mutual_exclusivity_all_four_cells() -> None:
    """Verify all 4 cells are mutually exclusive (exactly one terminal per tuple)."""
    test_cases = [
        (90.0, 100.0, 90.0, "REPRESENTATION_NEW_MECHANISM"),  # not MF, not LF
        (70.0, 100.0, 90.0, "MORE_FORGETTING_HELPS"),  # MF, not LF
        (90.0, 100.0, 70.0, "LESS_FORGETTING_HELPS"),  # not MF, LF
        (60.0, 100.0, 70.0, "BOTH_IMPROVE"),  # MF, LF, diff >= tol
    ]
    for gap_c, gap_d, gap_e, expected in test_cases:
        inputs = _base_eligible_inputs(gap_c=gap_c, gap_d=gap_d, gap_e=gap_e)
        result = classify_arc2b_slice5_discovery_branch(inputs)
        assert result["terminal_branch"] == expected, (
            f"gap_c={gap_c}, gap_d={gap_d}, gap_e={gap_e}: "
            f"expected {expected}, got {result['terminal_branch']}"
        )


def test_arm_eligible_all_conditions() -> None:
    """Arm eligible requires ALL conditions."""
    assert arm_eligible(
        operational_ok=True,
        live_carrier_bytes_exact=True,
        resume_generation=0,
        liveness_failure=False,
    ) is True
    assert arm_eligible(
        operational_ok=False,
        live_carrier_bytes_exact=True,
        resume_generation=0,
    ) is False
    assert arm_eligible(
        operational_ok=True,
        live_carrier_bytes_exact=False,
        resume_generation=0,
    ) is False
    assert arm_eligible(
        operational_ok=True,
        live_carrier_bytes_exact=True,
        resume_generation=1,
    ) is False
    assert arm_eligible(
        operational_ok=True,
        live_carrier_bytes_exact=True,
        resume_generation=0,
        liveness_failure=True,
    ) is False


def test_compute_budget_gap_bpw() -> None:
    """Selector: budget_gap_bpw = live_acc_carrier_bpw_max - effective_acc_budget_bpw."""
    gap = compute_budget_gap_bpw(live_acc_carrier_bpw_max=118.0, effective_acc_budget_bpw=0.4)
    assert gap == 117.6


def test_validate_lane_fields_fail_closed() -> None:
    """FAIL-CLOSED: missing lane fields are flagged."""
    good_record = {
        "lane_indices": [0, 1, 2],
        "acc_before_lanes": [0, 0, 0],
        "acc_after_lanes": [1, 1, 1],
        "vote_lanes": [1, 1, 1],
    }
    assert validate_lane_fields(good_record) == []

    bad_record = {
        "lane_indices": [0, 1, 2],
        "acc_before_lanes": [0, 0, 0],
        # acc_after_lanes missing
        "vote_lanes": [1, 1, 1],
    }
    failures = validate_lane_fields(bad_record)
    assert len(failures) == 1
    assert "missing_lane_field:acc_after_lanes" in failures[0]

    empty_record = {
        "lane_indices": [],
        "acc_before_lanes": [],
        "acc_after_lanes": [],
        "vote_lanes": [],
    }
    failures = validate_lane_fields(empty_record)
    assert len(failures) == 4  # all empty

    not_list_record = {
        "lane_indices": "not_a_list",
        "acc_before_lanes": [0],
        "acc_after_lanes": [1],
        "vote_lanes": [1],
    }
    failures = validate_lane_fields(not_list_record)
    assert len(failures) == 1
    assert "lane_field_not_list:lane_indices" in failures[0]


def test_validate_discovery_receipt_schema() -> None:
    """Receipt schema validation."""
    good_receipt = {
        "schema": "hrm_text_158_arc2b_slice5_discovery_branch_receipt/v1",
        "task_id": "test",
        "classifier": CLASSIFIER,
        "evidence_source": "live_decay_curve",
        "arm_c_eligible": True,
        "arm_d_eligible": True,
        "arm_e_eligible": True,
        "arm_a_bpw_w8": 8.0,
        "arm_a_bpw_w7": 7.0,
        "arm_b_k_star_summary": {},
        "gap_c_bpw": 90.0,
        "gap_d_bpw": 100.0,
        "gap_e_bpw": 90.0,
        "mf_boolean": False,
        "lf_boolean": False,
        "materiality_factor": 0.8,
        "direction_tol_factor": 0.1,
        "discovery_branch": "REPRESENTATION_NEW_MECHANISM",
        "discovery_branch_inputs": {},
        "ready_for_main_science": False,
        "counts_as_sub2": False,
        "pre_full_stack_diagnostic": True,
        "autonomy_rung": "discovery_h25_mechanism",
    }
    assert validate_discovery_receipt_schema(good_receipt) == []

    bad_receipt = dict(good_receipt)
    bad_receipt["schema"] = "wrong_schema"
    failures = validate_discovery_receipt_schema(bad_receipt)
    assert "schema_mismatch" in failures

    bad_receipt2 = dict(good_receipt)
    bad_receipt2["ready_for_main_science"] = True
    failures = validate_discovery_receipt_schema(bad_receipt2)
    assert "ready_for_main_science_not_false" in failures

    bad_receipt3 = dict(good_receipt)
    bad_receipt3["discovery_branch"] = "INVALID_TERMINAL"
    failures = validate_discovery_receipt_schema(bad_receipt3)
    assert any("invalid_terminal" in f for f in failures)


def test_branch_enum_values() -> None:
    """Verify all branch enum values are in the terminal sets."""
    for branch in Arc2bSlice5DiscoveryBranch:
        value = branch.value
        assert value in MECHANISM_TERMINALS or value in OPERATIONAL_TERMINALS, (
            f"branch {value} not in any terminal set"
        )
