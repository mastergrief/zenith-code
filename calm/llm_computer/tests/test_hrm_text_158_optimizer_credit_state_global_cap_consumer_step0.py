"""B2-5b Step-0 optimizer_credit_state consumer measurement tests (CPU-only)."""
from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import pytest

from calm.hrm_text_158.native_full_stack.optimizer_credit_state_global_cap_consumer_step0_facade import (
    run_optimizer_credit_state_global_cap_consumer_step0_suite,
)
from calm.hrm_text_158.native_full_stack.integer_optimizer_credit_path import (
    INTEGER_OPTIMIZER_CREDIT_PATH_WIRE_HARD_FALSE_FIELDS,
)
from calm.hrm_text_158.native_full_stack.optimizer_credit_state_global_cap_consumer_step0_measurement import (
    build_path_b_representative_inputs,
    measure_bdl_global_cap_reference,
    measure_iocp_receipt_flags,
    measure_iocp_sparse_emit_step,
    measure_margin_at_1280,
    project_global_cap_demand,
    _minimal_integer_wire_bundle,
)
from calm.hrm_text_158.native_full_stack.optimizer_credit_state_global_cap_consumer_step0_receipt import (
    OPTIMIZER_CREDIT_STATE_GLOBAL_CAP_CONSUMER_STEP0_HARD_FALSE_FIELDS,
    PINNED_CALL_SITE_IDS,
    ConsumerStep0BranchId,
    ConsumerStep0FixtureMeasurement,
    classify_aggregate_branch,
    validate_optimizer_credit_state_global_cap_consumer_step0_receipt,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap import GlobalRateCapSpec


def _row(
    *,
    fixture_name: str,
    fixture_role: str = "representative_consumer",
    pinned_call_site_id: str,
    max_row_count: int = 0,
    ordering_mode: str = "margin",
    candidate_rejects_global_cap: bool = False,
    seam_resolved: bool = False,
) -> ConsumerStep0FixtureMeasurement:
    return ConsumerStep0FixtureMeasurement(
        fixture_name=fixture_name,
        fixture_role=fixture_role,  # type: ignore[arg-type]
        pinned_call_site_id=pinned_call_site_id,
        source_anchor="test",
        consumer_path_class="PATH_B_GLOBAL_CAP_REFERENCE",
        candidate_mode_class="NON_CANDIDATE_GLOBAL_CAP_REFERENCE",
        total_sparse_event_count=0,
        projected_full_demand_count=max_row_count,
        projected_global_pre_cap_would_apply_count=max_row_count,
        max_row_count=max_row_count,
        ordering_mode=ordering_mode,
        ordering_mode_source="test",
        cap=256,
        deferred_count=0,
        saturation_observed=False,
        candidate_rejects_global_cap=candidate_rejects_global_cap,
        seam_resolved=seam_resolved,
    )


def _all_pinned_rows(**overrides) -> tuple[ConsumerStep0FixtureMeasurement, ...]:
    base = {
        PINNED_CALL_SITE_IDS[0]: _row(
            fixture_name="site0",
            pinned_call_site_id=PINNED_CALL_SITE_IDS[0],
            seam_resolved=True,
        ),
        PINNED_CALL_SITE_IDS[1]: _row(
            fixture_name="site1",
            pinned_call_site_id=PINNED_CALL_SITE_IDS[1],
            seam_resolved=True,
        ),
        PINNED_CALL_SITE_IDS[2]: _row(
            fixture_name="site2",
            pinned_call_site_id=PINNED_CALL_SITE_IDS[2],
            seam_resolved=True,
        ),
        PINNED_CALL_SITE_IDS[3]: _row(
            fixture_name="site3",
            pinned_call_site_id=PINNED_CALL_SITE_IDS[3],
            max_row_count=512,
            seam_resolved=True,
        ),
        PINNED_CALL_SITE_IDS[4]: _row(
            fixture_name="site4",
            pinned_call_site_id=PINNED_CALL_SITE_IDS[4],
            candidate_rejects_global_cap=True,
        ),
        PINNED_CALL_SITE_IDS[5]: _row(
            fixture_name="site5",
            pinned_call_site_id=PINNED_CALL_SITE_IDS[5],
            max_row_count=512,
            seam_resolved=True,
        ),
        PINNED_CALL_SITE_IDS[6]: _row(
            fixture_name="site6",
            pinned_call_site_id=PINNED_CALL_SITE_IDS[6],
            seam_resolved=True,
        ),
    }
    base.update(overrides)
    return tuple(base[site_id] for site_id in PINNED_CALL_SITE_IDS)


def test_default_suite_representative_aggregate_seam():
    receipt = run_optimizer_credit_state_global_cap_consumer_step0_suite()
    validate_optimizer_credit_state_global_cap_consumer_step0_receipt(receipt)
    assert receipt.measurement_representative is True
    assert set(receipt.sampled_call_sites) == set(PINNED_CALL_SITE_IDS)
    assert receipt.aggregate_branch_id == ConsumerStep0BranchId.CANDIDATE_GCAP_SEAM
    assert receipt.any_candidate_rejects_global_cap is True
    assert receipt.any_row_count_above_ceiling is False
    assert receipt.any_non_margin_ordering is False
    assert not receipt.include_classifier_negatives
    assert receipt.classifier_negative_results == ()


def test_iocp_sparse_step_real_nonzero_sparse_count():
    row = measure_iocp_sparse_emit_step()
    assert row.total_sparse_event_count > 0
    assert row.projected_full_demand_count == 0
    assert row.max_row_count == 0


def test_iocp_receipt_flags_propagates_wire_receipt_count_and_hard_false():
    sparse_row = measure_iocp_sparse_emit_step()
    receipt_row = measure_iocp_receipt_flags()
    assert receipt_row.total_sparse_event_count == sparse_row.total_sparse_event_count
    assert receipt_row.total_sparse_event_count > 0
    bundle = _minimal_integer_wire_bundle()
    assert receipt_row.total_sparse_event_count == bundle.wire_receipt.total_sparse_event_count
    for field in INTEGER_OPTIMIZER_CREDIT_PATH_WIRE_HARD_FALSE_FIELDS:
        assert getattr(bundle.wire_receipt, field) is False


def test_default_suite_excludes_classifier_negatives():
    receipt = run_optimizer_credit_state_global_cap_consumer_step0_suite()
    assert receipt.aggregate_branch_id not in {
        ConsumerStep0BranchId.CEILING_LIFT_FIRST,
        ConsumerStep0BranchId.ORDERING_MODE_FIRST,
    }


def test_reducer_coverage_ceiling_lift():
    rows = _all_pinned_rows()
    over_ceiling = _row(
        fixture_name="over",
        pinned_call_site_id=PINNED_CALL_SITE_IDS[3],
        max_row_count=3000,
        seam_resolved=True,
    )
    branch_rows = tuple(
        over_ceiling if row.pinned_call_site_id == PINNED_CALL_SITE_IDS[3] else row
        for row in rows
    )
    assert classify_aggregate_branch(branch_rows) == ConsumerStep0BranchId.CEILING_LIFT_FIRST


def test_reducer_coverage_ordering_mode():
    rows = _all_pinned_rows()
    non_margin = _row(
        fixture_name="hash",
        pinned_call_site_id=PINNED_CALL_SITE_IDS[5],
        max_row_count=512,
        ordering_mode="hash_shuffle",
        seam_resolved=True,
    )
    branch_rows = tuple(
        non_margin if row.pinned_call_site_id == PINNED_CALL_SITE_IDS[5] else row
        for row in rows
    )
    assert classify_aggregate_branch(branch_rows) == ConsumerStep0BranchId.ORDERING_MODE_FIRST


def test_reducer_coverage_candidate_seam_subset():
    rows = (
        _row(
            fixture_name="candidate",
            pinned_call_site_id=PINNED_CALL_SITE_IDS[4],
            candidate_rejects_global_cap=True,
        ),
        _row(
            fixture_name="path_a",
            pinned_call_site_id=PINNED_CALL_SITE_IDS[0],
            candidate_rejects_global_cap=True,
        ),
    )
    assert classify_aggregate_branch(rows) == ConsumerStep0BranchId.MEASUREMENT_INVALID


def test_reducer_coverage_integration_plan_path_b_only():
    rows = _all_pinned_rows()
    resolved_rows = tuple(
        replace(row, candidate_rejects_global_cap=False, seam_resolved=True)
        for row in rows
    )
    assert classify_aggregate_branch(resolved_rows) == ConsumerStep0BranchId.INTEGRATION_PLAN


def test_reducer_coverage_measurement_invalid_missing_site():
    rows = (_row(fixture_name="only", pinned_call_site_id=PINNED_CALL_SITE_IDS[0]),)
    assert classify_aggregate_branch(rows) == ConsumerStep0BranchId.MEASUREMENT_INVALID


def test_classifier_negative_probes_when_enabled():
    receipt = run_optimizer_credit_state_global_cap_consumer_step0_suite(
        include_classifier_negatives=True,
    )
    assert receipt.include_classifier_negatives is True
    branches = {probe.branch_id for probe in receipt.classifier_negative_results}
    assert ConsumerStep0BranchId.CEILING_LIFT_FIRST in branches
    assert ConsumerStep0BranchId.ORDERING_MODE_FIRST in branches
    assert receipt.aggregate_branch_id == ConsumerStep0BranchId.CANDIDATE_GCAP_SEAM


def test_hard_false_fields_remain_false():
    receipt = run_optimizer_credit_state_global_cap_consumer_step0_suite()
    for field in OPTIMIZER_CREDIT_STATE_GLOBAL_CAP_CONSUMER_STEP0_HARD_FALSE_FIELDS:
        assert getattr(receipt, field) is False


def test_no_native_dispatch_import_in_step0_modules():
    repo_root = Path(__file__).resolve().parents[2]
    module_dir = (
        repo_root
        / "hrm_text_158"
        / "native_full_stack"
    )
    offenders: list[str] = []
    for name in (
        "optimizer_credit_state_global_cap_consumer_step0_receipt.py",
        "optimizer_credit_state_global_cap_consumer_step0_measurement.py",
        "optimizer_credit_state_global_cap_consumer_step0_facade.py",
    ):
        source = (module_dir / name).read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if "native_dispatch" in alias.name:
                        offenders.append(f"{name}: import {alias.name}")
            if isinstance(node, ast.ImportFrom) and node.module:
                if "native_dispatch" in node.module:
                    offenders.append(f"{name}: from {node.module}")
    assert offenders == []


def test_path_b_projection_matches_select_surface():
    inputs = build_path_b_representative_inputs(target_row_count=1280)
    spec = GlobalRateCapSpec(cap=512, step=1)
    projection = project_global_cap_demand(inputs, spec)
    assert projection.max_row_count == 1280
    assert projection.ordering_mode == "margin"
    assert projection.projected_full_demand_count == 1280
    assert projection.saturation_observed is True
    assert projection.deferred_count == 1280 - 512
