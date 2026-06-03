"""C1.0 accumulator-compression feasibility/semantic-contract tests."""
from __future__ import annotations

from dataclasses import replace
import math

import pytest

from calm.hrm_text_158.native_full_stack.accumulator_compression import (
    FIXED_Q_ACCUMULATOR_ONLY_NULL,
    IDENTITY_INT16_BASELINE_NAME,
    JOINT_Q_ENTROPY_PREVIEW,
    AccumulatorFeasibilityReport,
    CandidateClassification,
    candidate_assessment,
    default_fixed_q_feasibility_table,
    evaluate_accumulator_feasibility,
    identity_int16_baseline_assessment,
    joint_q_entropy_preview,
    packed_2bit_payload_bits_and_padding,
    required_decision_dimension_names,
    semantic_decision_surface_contract,
    validate_accumulator_feasibility_report,
)
from calm.hrm_text_158.native_full_stack.persistent_state_budget import (
    EFFECTIVE_FORWARD_TERNARY_BITS,
)


def _by_name(rows: tuple[AccumulatorFeasibilityReport, ...]) -> dict[str, AccumulatorFeasibilityReport]:
    return {row.regime_name: row for row in rows}


def test_fixed_q_feasibility_table_is_computed_from_source_ledger():
    rows = _by_name(default_fixed_q_feasibility_table())

    tiny = rows["tiny_two_projection_fixture_fixed_2bit_q"]
    assert tiny.eligible_weight_count == 160
    assert tiny.q_packed_data_bits_per_weight == pytest.approx(2.0)
    assert tiny.q_packed_padding_bits_per_weight == pytest.approx(0.0)
    assert tiny.q_packed_metadata_bits_per_weight == pytest.approx(3.2)
    assert tiny.frozen_scale_bits_per_weight == pytest.approx(0.4)
    assert tiny.remaining_accumulator_budget_bits_per_weight == pytest.approx(-3.6)
    assert tiny.packed_inclusive_physical_bits_per_weight == pytest.approx(21.6)
    assert tiny.target_achieved_with_reported_ledger is False
    assert tiny.accumulator_only_sub2_possible_under_current_q is False
    assert tiny.fixed_q_null_statement == FIXED_Q_ACCUMULATOR_ONLY_NULL

    large = rows["prior_large_fixture_fixed_2bit_q"]
    assert large.eligible_weight_count == 16384
    assert large.q_packed_data_bits_per_weight == pytest.approx(2.0)
    assert large.q_packed_metadata_bits_per_weight == pytest.approx(0.015625)
    assert large.frozen_scale_bits_per_weight == pytest.approx(0.001953125)
    assert large.remaining_accumulator_budget_bits_per_weight == pytest.approx(-0.017578125)
    assert large.packed_inclusive_physical_bits_per_weight == pytest.approx(18.017578125)
    validate_accumulator_feasibility_report(large)

    realistic = rows["illustrative_4096x4096_one_tensor_one_scale_fixed_2bit_q"]
    assert realistic.eligible_weight_count == 4096 * 4096
    assert realistic.q_packed_metadata_bits_per_weight == pytest.approx(256 / (4096 * 4096))
    assert realistic.frozen_scale_bits_per_weight == pytest.approx(32 / (4096 * 4096))
    assert realistic.remaining_accumulator_budget_bits_per_weight == pytest.approx(
        -(256 + 32) / (4096 * 4096),
    )

    per_row_scale = rows["illustrative_4096x4096_one_tensor_per_row_scale_fixed_2bit_q"]
    assert per_row_scale.frozen_scale_bits_per_weight == pytest.approx((4096 * 32) / (4096 * 4096))
    assert per_row_scale.remaining_accumulator_budget_bits_per_weight == pytest.approx(
        -(256 + 4096 * 32) / (4096 * 4096),
    )


def test_padding_is_visible_but_not_double_counted_in_source_total():
    payload_bits, padding_bits = packed_2bit_payload_bits_and_padding(5)
    report = evaluate_accumulator_feasibility(
        regime_name="non_multiple_of_four_padding_probe",
        eligible_weight_count=5,
        q_packed_data_bits_per_weight=payload_bits / 5,
        q_packed_padding_bits=padding_bits,
        q_packed_metadata_bits=192,
        frozen_scale_bits=32,
        accumulator_bits_per_weight=16.0,
    )

    assert payload_bits == 16
    assert padding_bits == 6
    assert report.q_packed_data_bits_per_weight == pytest.approx(3.2)
    assert report.q_packed_padding_bits_per_weight == pytest.approx(1.2)
    assert report.q_packed_metadata_bits_per_weight == pytest.approx(38.4)
    assert report.q_packed_total_bits_per_weight == pytest.approx(3.2 + 38.4)
    validate_accumulator_feasibility_report(report)


def test_false_claim_guards_reject_fixed_q_and_preview_only_sub2_claims():
    large = _by_name(default_fixed_q_feasibility_table())["prior_large_fixture_fixed_2bit_q"]

    with pytest.raises(ValueError, match="physical sub-2 claim"):
        validate_accumulator_feasibility_report(large, claimed_physical_sub2_achieved=True)

    with pytest.raises(ValueError, match="target flag"):
        validate_accumulator_feasibility_report(replace(large, target_achieved_with_reported_ledger=True))
    with pytest.raises(ValueError, match="inclusive physical"):
        validate_accumulator_feasibility_report(
            replace(
                large,
                packed_inclusive_physical_bits_per_weight=large.q_packed_total_bits_per_weight,
                target_achieved_with_reported_ledger=True,
                claimable_physical_sub2=True,
            ),
            claimed_physical_sub2_achieved=True,
        )

    preview = joint_q_entropy_preview(
        regime_name="prior_large_q1p6_preview",
        eligible_weight_count=16384,
        q_entropy_bits_per_weight=1.6,
        q_packed_metadata_bits=256,
        frozen_scale_bits=32,
        accumulator_bits_per_weight=0.3,
    )
    assert preview.remaining_accumulator_budget_bits_per_weight == pytest.approx(0.382421875)
    assert preview.target_achieved_with_reported_ledger is True
    assert preview.claimable_physical_sub2 is False
    assert preview.joint_q_entropy_route_status == JOINT_Q_ENTROPY_PREVIEW
    with pytest.raises(ValueError, match="preview-only q-entropy"):
        validate_accumulator_feasibility_report(preview, claimed_physical_sub2_achieved=True)


def test_joint_q_entropy_preview_numbers_are_labeled_hypothesis_not_feasibility_claim():
    q_log2_3 = joint_q_entropy_preview(
        regime_name="prior_large_log2_3_preview",
        eligible_weight_count=16384,
        q_entropy_bits_per_weight=EFFECTIVE_FORWARD_TERNARY_BITS,
        q_packed_metadata_bits=256,
        frozen_scale_bits=32,
        accumulator_bits_per_weight=0.0,
    )

    assert q_log2_3.remaining_accumulator_budget_bits_per_weight == pytest.approx(
        2.0 - math.log2(3.0) - 0.015625 - 0.001953125,
    )
    assert q_log2_3.q_entropy_code_overhead_accounted is False
    assert "preview only" in q_log2_3.joint_q_entropy_route_status
    assert q_log2_3.claimable_physical_sub2 is False


def test_semantic_surface_contract_requires_all_decision_dimensions():
    names = set(required_decision_dimension_names())
    assert names == {
        "candidate_mask",
        "direction",
        "threshold_crossing",
        "truncating_decay",
        "clip",
        "residual_after_threshold",
        "replay_veto_residual_rows",
        "abs_new_acc_ranking",
        "flat_index_tie_ordering",
        "cross_state_global_flat_ordering",
        "accepted_rows",
        "deferred_rows",
        "backlog_carry",
        "final_q_changes",
        "accumulator_residuals",
        "step_to_step_state_hashes",
    }
    assert {item.surface for item in semantic_decision_surface_contract()} == {
        "vote_preplan",
        "selection_global_cap",
        "q_acc_apply",
    }

    missing_backlog = tuple(name for name in names if name != "backlog_carry")
    with pytest.raises(ValueError, match="missing=.*backlog_carry"):
        candidate_assessment(
            candidate_name="lossless_but_unregistered_backlog",
            classification=CandidateClassification.DECISION_EXACT,
            covered_decision_dimensions=missing_backlog,
            compressed_representation=True,
        )
    with pytest.raises(ValueError, match="missing=.*backlog_carry"):
        candidate_assessment(
            candidate_name="bounded_delta_but_unregistered_backlog",
            classification=CandidateClassification.BOUNDED_DELTA_WITH_REPORT,
            covered_decision_dimensions=missing_backlog,
            compressed_representation=True,
            bounded_delta_hypothesis="bounded deltas stay below all cap margins",
            guardrail="must report changed candidate/order/accepted/deferred counts",
        )
    with pytest.raises(ValueError, match="missing=.*backlog_carry"):
        candidate_assessment(
            candidate_name="rejected_but_still_incomplete",
            classification=CandidateClassification.NOT_SAME_LEARNER,
            covered_decision_dimensions=missing_backlog,
            compressed_representation=True,
        )

    decision_exact = candidate_assessment(
        candidate_name="future_decision_exact_probe",
        classification=CandidateClassification.DECISION_EXACT,
        covered_decision_dimensions=required_decision_dimension_names(),
        compressed_representation=True,
    )
    assert decision_exact.c2_eligible_by_default is True


def test_candidate_taxonomy_keeps_identity_baseline_out_of_compression_progress():
    baseline = identity_int16_baseline_assessment()
    assert baseline.candidate_name == IDENTITY_INT16_BASELINE_NAME
    assert baseline.normalized_classification == CandidateClassification.BIT_EXACT
    assert baseline.compressed_representation is False
    assert baseline.c2_eligible_by_default is False
    assert "not a compressed representation" in baseline.note

    bounded = candidate_assessment(
        candidate_name="future_sparse_event_queue",
        classification=CandidateClassification.BOUNDED_DELTA_WITH_REPORT,
        covered_decision_dimensions=required_decision_dimension_names(),
        compressed_representation=True,
        bounded_delta_hypothesis="bounded ranking deltas do not change accepted rows under cap slack",
        guardrail="must report changed candidate/order/accepted/deferred counts before C2",
    )
    assert bounded.c2_eligible_by_default is False

    with pytest.raises(ValueError, match="hypothesis and guardrail"):
        candidate_assessment(
            candidate_name="bounded_without_guardrail",
            classification=CandidateClassification.BOUNDED_DELTA_WITH_REPORT,
            covered_decision_dimensions=required_decision_dimension_names(),
            compressed_representation=True,
        )
