from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from calm.hrm_text_158.native_full_stack.carry_w6_second_capture_regression import (
    AUDIT_WINNING_FAMILY_CARRIED_PERSISTENT_BUCKET,
    B2C_PRIMARY_LABEL_ACCUMULATOR_NO_TRACKING_NULL,
    CAPTURE2_CAPTURE,
    DUAL_CAPTURE_SPECS,
    F5_TAXONOMY_ACC_SHRINK_TWO_TIER,
    TRACE1_CAPTURE,
    VERDICT_CLASS,
    assert_verdict_class_only,
    chain_paths_available,
    compute_trace_hash_from_path,
    evaluate_capture_surfaces,
    evaluate_carry_w6_second_capture_confirmed,
    load_json_receipt_readonly,
    reject_science_overclaim_surface,
    sha256_canonical_regression_block,
    validate_b2c_accumulator_no_tracking_null,
    validate_determinism_trace_hash,
    validate_f5_acc_shrink_two_tier_w_min,
    validate_transient_compute_control_winning_family,
    validate_value_drift_non_gating_across_captures,
    validate_w6_floor_and_hard_break_below_w6,
)


def _canonical_json(data: dict) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _require_chain(spec=TRACE1_CAPTURE):
    if not chain_paths_available(spec):
        pytest.skip(f"chain root unavailable: {spec.chain_root}")


@pytest.fixture(params=DUAL_CAPTURE_SPECS, ids=[spec.capture_id for spec in DUAL_CAPTURE_SPECS])
def capture_spec(request):
    spec = request.param
    _require_chain(spec)
    return spec


@pytest.fixture
def capture_block(capture_spec):
    return evaluate_capture_surfaces(capture_spec)


def test_b8_verdict_class_only_no_science_overclaim():
    assert_verdict_class_only(verdict_class=VERDICT_CLASS)
    with pytest.raises(ValueError, match="B8 regression asserts carry_w6_second_capture_confirmed only"):
        assert_verdict_class_only(verdict_class="sub_2_runtime_ready")
    with pytest.raises(ValueError, match="science overclaim surface rejected"):
        reject_science_overclaim_surface(claim_surface="training_stable")


def test_b8_determinism_trace_hash_matches_recorded(capture_spec):
    trace_path = capture_spec.chain_root / capture_spec.trace_relpath
    result = validate_determinism_trace_hash(spec=capture_spec, trace_path=trace_path)
    assert result["trace_hash"] == capture_spec.trace_hash
    assert compute_trace_hash_from_path(trace_path) == capture_spec.trace_hash


def test_b8_b2c_accumulator_no_tracking_null_jaccard_zero(capture_spec):
    b2c_path = capture_spec.chain_root / capture_spec.b2c_relpath
    b2c_receipt, sha = load_json_receipt_readonly(b2c_path)
    result = validate_b2c_accumulator_no_tracking_null(
        spec=capture_spec,
        b2c_receipt=b2c_receipt,
    )
    assert result["primary_label"] == B2C_PRIMARY_LABEL_ACCUMULATOR_NO_TRACKING_NULL
    assert result["jaccard_vs_int16"] == 0.0
    assert len(sha) == 64


def test_b8_transient_compute_control_winning_family_carried_persistent_bucket(capture_spec):
    audit_path = capture_spec.chain_root / capture_spec.audit_relpath
    audit_receipt, _sha = load_json_receipt_readonly(audit_path)
    result = validate_transient_compute_control_winning_family(
        spec=capture_spec,
        audit_receipt=audit_receipt,
    )
    assert result["family_id"] == AUDIT_WINNING_FAMILY_CARRIED_PERSISTENT_BUCKET


def test_b8_f5_acc_shrink_two_tier_w_min_six(capture_spec):
    acc_width_path = capture_spec.chain_root / capture_spec.acc_width_relpath
    acc_width_receipt, _sha = load_json_receipt_readonly(acc_width_path)
    result = validate_f5_acc_shrink_two_tier_w_min(
        spec=capture_spec,
        acc_width_receipt=acc_width_receipt,
    )
    assert result["w_min"] == 6
    assert F5_TAXONOMY_ACC_SHRINK_TWO_TIER in result["taxonomy_labels"]


def test_b8_w6_floor_zero_crossings_hard_break_below_w6(capture_spec):
    acc_width_path = capture_spec.chain_root / capture_spec.acc_width_relpath
    acc_width_receipt, _sha = load_json_receipt_readonly(acc_width_path)
    result = validate_w6_floor_and_hard_break_below_w6(
        spec=capture_spec,
        acc_width_receipt=acc_width_receipt,
    )
    assert result["w6_crossing_mismatch_count_vs_w16"] == 0
    assert result["w6_mismatch_count_vs_w16_reference"] == capture_spec.w6_value_drift_mismatch
    for width, expected in capture_spec.below_w6_crossing_mismatch_by_width.items():
        assert result["below_w6_crossing_mismatch_by_width"][str(width)] == expected


def test_b8_anti_overclaim_sub_2_science_surface_rejected_fail_closed():
    """Guardrail: B8 module never emits physical/sub-2 verdict claims."""
    with pytest.raises(ValueError, match="science overclaim"):
        reject_science_overclaim_surface(claim_surface="sub_2_bit_physical_persistent")


def test_b8_value_drift_786_vs_838_explicitly_non_gating():
    _require_chain(TRACE1_CAPTURE)
    _require_chain(CAPTURE2_CAPTURE)
    trace1 = evaluate_capture_surfaces(TRACE1_CAPTURE)
    capture2 = evaluate_capture_surfaces(CAPTURE2_CAPTURE)
    result = validate_value_drift_non_gating_across_captures(
        observed_by_capture={
            "trace1": int(
                trace1["surfaces"]["w6_floor_break"]["w6_mismatch_count_vs_w16_reference"]
            ),
            "capture2": int(
                capture2["surfaces"]["w6_floor_break"]["w6_mismatch_count_vs_w16_reference"]
            ),
        },
    )
    assert result["trace1_w6_mismatch"] == 838
    assert result["capture2_w6_mismatch"] == 786
    assert result["gating_surface"] == "crossing_mismatch_count_vs_w16_at_w6"
    assert result["non_gating_surface"] == "mismatch_count_vs_w16_reference_at_w6"


def test_b8_capture2_parent_sha_pin():
    _require_chain(CAPTURE2_CAPTURE)
    assert CAPTURE2_CAPTURE.parent_sha256 == (
        "9b4e311a22787e7d4808bde7bc2953568d767a2ee8ac648942a3f5dbb7b4d5ec"
    )
    parent_path = CAPTURE2_CAPTURE.chain_root / "b2b_seed44" / "parent.sha256.after"
    parent_line = parent_path.read_text(encoding="utf-8").strip().split()[0]
    assert parent_line == CAPTURE2_CAPTURE.parent_sha256


def test_b8_dual_capture_joint_block_and_canonical_sha_stable():
    if not all(chain_paths_available(spec) for spec in DUAL_CAPTURE_SPECS):
        pytest.skip("dual chain roots unavailable")
    block_a = evaluate_carry_w6_second_capture_confirmed()
    block_b = evaluate_carry_w6_second_capture_confirmed()
    assert block_a["verdict_class"] == VERDICT_CLASS
    assert block_a["claim_boundary"]["no_sub_2_claim"] is True
    assert block_a["claim_boundary"]["no_stability_claim"] is True
    assert block_a["claim_boundary"]["no_training_claim"] is True
    assert len(block_a["captures"]) == 2
    payload_a = _canonical_json(block_a)
    payload_b = _canonical_json(block_b)
    assert payload_a == payload_b
    digest = sha256_canonical_regression_block(block_a)
    assert digest == hashlib.sha256(payload_a.encode("utf-8")).hexdigest()
    assert len(digest) == 64
