"""Front-C CPU/static projection scaffold tests."""
from __future__ import annotations

from dataclasses import replace
import math
from typing import Any

import pytest

from calm.hrm_text_158.native_full_stack.front_c_fixtures import (
    front_c_count_only_timeline_artifact,
    front_c_overhead_failure_timeline_fixture,
    front_c_prior_large_q_ledger,
    front_c_timeline_churn_fixture,
    front_c_zero_drift_decision_paths,
)
from calm.hrm_text_158.native_full_stack.front_c_harness import front_c_report_from_mapping
from calm.hrm_text_158.native_full_stack.front_c_projection import (
    COUNT_ONLY_ARTIFACT_REJECTION,
    FRONT_C_PAYLOAD_ONLY_GATING_PHRASE,
    FrontCDecisionPath,
    build_front_c_projection_report,
    compare_front_c_decision_equivalence,
    front_c_physical_base3_q_gate_report,
    measure_front_c_timeline_density,
    validate_front_c_projection_report,
)
from calm.hrm_text_158.native_full_stack.q_entropy_packing import (
    BASE3_FIXED_PAYLOAD_BITS_PER_WEIGHT,
    BASE3_Q_FORMAT,
    validate_base3_q_entropy_ledger,
)


def _fake_entropy_floor_q_row(row):
    q_data_bits = int(round(math.log2(3.0) * float(row.eligible_weight_count)))
    q_data_bpw = q_data_bits / float(row.eligible_weight_count)
    q_total_bits = q_data_bits + int(row.packed_q_metadata_bits)
    q_total_bpw = q_total_bits / float(row.eligible_weight_count)
    scale_bpw = row.frozen_scale_fp32_bits_per_weight
    remaining = row.target_bits_per_weight - q_total_bpw - scale_bpw
    inclusive_bpw = q_total_bpw + scale_bpw
    inclusive_bits = inclusive_bpw * float(row.eligible_weight_count)
    return replace(
        row,
        regime_name="fake_entropy_floor_not_physical_base3",
        packed_q_data_bits=q_data_bits,
        packed_q_total_bits=q_total_bits,
        accumulator_bits=0.0,
        packed_inclusive_physical_bits=inclusive_bits,
        q_packed_data_bits_per_weight=q_data_bpw,
        q_packed_total_bits_per_weight=q_total_bpw,
        accumulator_bits_per_weight=0.0,
        packed_inclusive_physical_bits_per_weight=inclusive_bpw,
        remaining_accumulator_budget_bits_per_weight=remaining,
        target_achieved=inclusive_bpw < row.target_bits_per_weight,
        claimable_physical_sub2=inclusive_bpw < row.target_bits_per_weight,
    )


def _assert_no_tensors(value: Any) -> None:
    if value.__class__.__name__ == "Tensor":
        raise AssertionError("Front-C compact reports must not include raw tensors")
    if isinstance(value, dict):
        for child in value.values():
            _assert_no_tensors(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _assert_no_tensors(child)


def test_physical_base3_q_gate_rejects_entropy_floor_row_even_when_self_consistent():
    row = front_c_prior_large_q_ledger()
    valid = front_c_physical_base3_q_gate_report(row)

    assert valid.gate_valid is True
    assert valid.q_format == BASE3_Q_FORMAT
    assert valid.base3_fixed_payload_bits_per_weight == pytest.approx(8 / 5)
    assert valid.q_packed_data_bits_per_weight >= BASE3_FIXED_PAYLOAD_BITS_PER_WEIGHT
    assert valid.q_effective_entropy_floor_bits_per_weight == pytest.approx(math.log2(3.0))

    fake = _fake_entropy_floor_q_row(row)
    validate_base3_q_entropy_ledger(fake)
    fake_gate = front_c_physical_base3_q_gate_report(fake)

    assert fake.q_packed_data_bits_per_weight == pytest.approx(math.log2(3.0), abs=1e-4)
    assert fake.q_packed_data_bits_per_weight < BASE3_FIXED_PAYLOAD_BITS_PER_WEIGHT
    assert fake_gate.base3_validator_passed is True
    assert fake_gate.physical_base3_bounds_passed is False
    assert fake_gate.gate_valid is False
    assert fake_gate.diagnostic_only is True
    assert "entropy-floor" in fake_gate.rejection_reason

    dense, sparse = front_c_zero_drift_decision_paths()
    report = build_front_c_projection_report(
        timeline_steps=front_c_timeline_churn_fixture(
            eligible_weight_count=fake.eligible_weight_count,
        ),
        q_ledger_row=fake,
        dense_decision_path=dense,
        sparse_decision_path=sparse,
    )
    assert report.bounded_delta_inclusive_ledger.claimable_physical_sub2 is True
    assert report.decision_equivalence.zero_drift is True
    assert report.final_gate_passed is False
    validate_front_c_projection_report(report)


def test_timeline_density_reports_max_p95_union_and_churn_entry_exit():
    row = front_c_prior_large_q_ledger()
    summary = measure_front_c_timeline_density(
        front_c_timeline_churn_fixture(eligible_weight_count=row.eligible_weight_count),
    )

    assert summary.step_count == 2
    assert summary.eligible_weight_count == row.eligible_weight_count
    assert summary.step_densities[0].decision_relevant_exact_count == 3
    assert summary.step_densities[1].decision_relevant_exact_count == 3
    assert summary.union_decision_relevant_exact_count == 5
    assert summary.union_backlog_carry_count == 1
    assert summary.max_decision_relevant_exact_density == pytest.approx(3 / row.eligible_weight_count)
    assert summary.p95_decision_relevant_exact_density == pytest.approx(3 / row.eligible_weight_count)
    assert summary.transition_count == 1
    assert summary.total_entry_count == 2
    assert summary.total_exit_count == 2
    assert summary.max_churn_rate == pytest.approx(4 / row.eligible_weight_count)
    assert summary.raw_arrays_included is False
    _assert_no_tensors(summary.to_dict())


def test_payload_only_is_reported_but_overhead_inclusive_projection_is_the_only_gate():
    row = front_c_prior_large_q_ledger()
    dense, sparse = front_c_zero_drift_decision_paths()
    report = build_front_c_projection_report(
        timeline_steps=front_c_overhead_failure_timeline_fixture(
            eligible_weight_count=row.eligible_weight_count,
        ),
        q_ledger_row=row,
        dense_decision_path=dense,
        sparse_decision_path=sparse,
    )

    assert report.gate_basis_statement == FRONT_C_PAYLOAD_ONLY_GATING_PHRASE
    assert report.payload_only_gate_used is False
    assert report.overhead_inclusive_gate_used is True
    assert report.payload_only_would_fit_remaining_budget is True
    assert report.payload_only_bits_per_weight <= (
        report.bounded_delta_inclusive_ledger.remaining_accumulator_budget_bits_per_weight
    )
    assert report.overhead_inclusive_projected_bpw > (
        report.bounded_delta_inclusive_ledger.remaining_accumulator_budget_bits_per_weight
    )
    assert report.bounded_delta_inclusive_ledger.claimable_physical_sub2 is False
    assert report.final_gate_passed is False
    validate_front_c_projection_report(report)


def test_decision_equivalence_requires_zero_drift_on_all_locked_surfaces():
    dense, sparse = front_c_zero_drift_decision_paths()
    zero = compare_front_c_decision_equivalence(dense, sparse)
    assert zero.zero_drift is True
    assert zero.failed_surfaces == ()

    q_direction_mismatch = FrontCDecisionPath(
        label="sparse_bad_q_direction",
        q_flip_directions=(("fixture", 5, -1), ("fixture", 7, -1)),
        accepted_under_global_cap_keys=sparse.accepted_under_global_cap_keys,
        deferred_under_global_cap_keys=sparse.deferred_under_global_cap_keys,
        backlog_keys=sparse.backlog_keys,
        replay_veto_decision_keys=sparse.replay_veto_decision_keys,
    )
    q_report = compare_front_c_decision_equivalence(dense, q_direction_mismatch)
    assert q_report.zero_drift is False
    assert "q_flip_direction" in q_report.failed_surfaces

    mismatch_cases = (
        (
            "accepted_under_global_cap",
            replace(sparse, accepted_under_global_cap_keys=(("fixture", 6),)),
        ),
        (
            "deferred_under_global_cap",
            replace(sparse, deferred_under_global_cap_keys=(("fixture", 8),)),
        ),
        ("backlog_keys", replace(sparse, backlog_keys=(("fixture", 14),))),
        (
            "replay_veto_decisions",
            replace(sparse, replay_veto_decision_keys=(("fixture", 10),)),
        ),
    )
    for expected_surface, bad_sparse in mismatch_cases:
        report = compare_front_c_decision_equivalence(dense, bad_sparse)
        assert report.zero_drift is False
        assert expected_surface in report.failed_surfaces


def test_harness_rejects_count_only_artifacts_and_keeps_reports_compact():
    row = front_c_prior_large_q_ledger()
    dense, sparse = front_c_zero_drift_decision_paths()
    count_only = {
        "timeline": [front_c_count_only_timeline_artifact(eligible_weight_count=row.eligible_weight_count)],
        "dense_decision_path": dense.to_dict(),
        "sparse_decision_path": sparse.to_dict(),
    }

    with pytest.raises(ValueError, match=COUNT_ONLY_ARTIFACT_REJECTION):
        front_c_report_from_mapping(count_only, q_ledger_row=row)

    mixed_identity_plus_count = {
        "timeline": [
            {
                "step": 0,
                "eligible_weight_count": row.eligible_weight_count,
                "active_next_step_keys": [{"state_key": "mixed", "flat_index": 1}],
                "backlog_carry_count": 1,
            },
        ],
        "dense_decision_path": dense.to_dict(),
        "sparse_decision_path": sparse.to_dict(),
    }
    with pytest.raises(ValueError, match=COUNT_ONLY_ARTIFACT_REJECTION):
        front_c_report_from_mapping(mixed_identity_plus_count, q_ledger_row=row)

    valid = {
        "timeline": [
            step.to_dict()
            for step in front_c_timeline_churn_fixture(
                eligible_weight_count=row.eligible_weight_count,
            )
        ],
        "dense_decision_path": dense.to_dict(),
        "sparse_decision_path": sparse.to_dict(),
    }
    report = front_c_report_from_mapping(valid, q_ledger_row=row)

    assert report.final_gate_passed is True
    assert report.claimable_physical_sub2_with_decision_guard is True
    assert "no Front-C viability claim" in " ".join(report.non_claims)
    validate_front_c_projection_report(report)
    _assert_no_tensors(report.to_dict())
