"""Tests for read-only FP credit degeneracy grounding diagnostic (v4)."""
from __future__ import annotations

import pytest
import torch

from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    default_dry_run_rank_vote_spec,
    rank_bucketed_int16_votes,
)
from calm.hrm_text_158.native_full_stack.fp_credit_degeneracy_diagnostic import (
    BR_FP_DEG_AMPLIFIED_FIXABLE,
    BR_FP_DEG_MEASUREMENT_INVALID,
    BR_FP_DEG_NO_PARITY_ENRICHMENT,
    BR_FP_DEG_POOL_SELECTION_CONFOUND,
    BR_FP_DEG_UNRESOLVED,
    FP_CREDIT_DEGENERACY_SCHEMA_V2,
    FpCreditPoolMetrics,
    ParityEnrichmentMetrics,
    characterize_fp_credit_pool,
    classify_fp_credit_degeneracy_branch,
    compute_parity_enrichment_metrics,
    full_pool_fp4_anchor_in_band,
    run_anchored_fp_credit_degeneracy_diagnostic,
)
from calm.hrm_text_158.native_full_stack.realistic_gradient_parity_probe import (
    DEFAULT_T2_CHECKPOINT_REL,
    PerCandidateParityRecord,
    discover_t2_checkpoint,
)
from calm.hrm_text_158.native_full_stack.t2_fp_vs_s24_disambiguation import (
    FROZEN_T2_ANCHOR_BATCH_SIZE,
    FROZEN_T2_ANCHOR_CHECKPOINT_SHA256,
    FROZEN_T2_ANCHOR_CURRICULUM_SEED,
    FROZEN_T2_ANCHOR_KEY_SET_SHA256,
)


def _symmetric_fixture() -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.logspace(-8, -3, 1000, dtype=torch.float32)
    signs = torch.where(torch.arange(1000) % 2 == 0, 1.0, -1.0)
    fp_credit = (values * signs).reshape(40, 25)
    fp_moves = torch.ones_like(fp_credit, dtype=torch.int8)
    return fp_credit, fp_moves


def _pool_metrics(
    *,
    move_candidates: int,
    fp_vote4_count: int,
    fp_vote1_count: int,
    pool_id: str = "TRUE_FULL_FP_MOVE_POOL",
    module_key: str | None = None,
) -> FpCreditPoolMetrics:
    return FpCreditPoolMetrics(
        pool_id=pool_id,
        module_key=module_key,
        move_candidates=move_candidates,
        fp_vote4_count=fp_vote4_count,
        fp_vote1_count=fp_vote1_count,
        fp_vote0_count=0,
        rank_rf_ge_0p5_count=move_candidates // 2,
        credit_min=1e-8,
        credit_p25=None,
        credit_median=1e-5,
        credit_p75=None,
        credit_max=1e-3,
        distinct_abs_groups=100,
        largest_tie_group_size=10,
        median_tie_group_size=5.0,
        histogram_bins=tuple(0 for _ in range(8)),
    )


def _enrichment(
    *,
    full_pool_fp4: float,
    parity_fp4: float,
    full_candidates: int = 100_000,
    parity_candidates: int = 1_000,
    consistency_pass: bool = True,
) -> ParityEnrichmentMetrics:
    return ParityEnrichmentMetrics(
        full_pool_fp4_fraction=full_pool_fp4,
        parity_fp4_balance=parity_fp4,
        full_pool_fp4_at_parity_indices=parity_fp4,
        parity_index_enrichment_over_full_pool=parity_fp4 - full_pool_fp4,
        integer_move_coverage_vs_full_fp=parity_candidates / full_candidates,
        parity_move_candidates=parity_candidates,
        full_pool_move_candidates=full_candidates,
        consistency_anchor_pass=consistency_pass,
        consistency_anchor_delta=0.0 if consistency_pass else 0.5,
    )


def test_by_construction_anchor_symmetric_credits():
    fp_credit, fp_moves = _symmetric_fixture()
    spec = default_dry_run_rank_vote_spec()
    metrics = characterize_fp_credit_pool(
        pool_id="TRUE_FULL_FP_MOVE_POOL",
        module_key="toy",
        fp_credit=fp_credit,
        fp_moves=fp_moves,
        spec=spec,
    )
    assert metrics.fp_vote4_fraction == pytest.approx(0.5, abs=0.05)
    assert full_pool_fp4_anchor_in_band(metrics.fp_vote4_fraction)


def test_enrichment_confound_without_full_pool_below_half():
    full = _pool_metrics(move_candidates=100_000, fp_vote4_count=50_001, fp_vote1_count=49_999)
    enrichment = _enrichment(full_pool_fp4=0.50001, parity_fp4=0.96)
    branch = classify_fp_credit_degeneracy_branch(full_pool=full, enrichment=enrichment)
    assert branch == BR_FP_DEG_POOL_SELECTION_CONFOUND


def test_v3_gap_closed_half_boundary_routes_confound_not_unresolved():
    full = _pool_metrics(move_candidates=100_000, fp_vote4_count=50_001, fp_vote1_count=49_999)
    enrichment = _enrichment(full_pool_fp4=0.50001, parity_fp4=0.96, parity_candidates=1080)
    assert enrichment.parity_index_enrichment_over_full_pool == pytest.approx(0.45999, abs=0.01)
    branch = classify_fp_credit_degeneracy_branch(full_pool=full, enrichment=enrichment)
    assert branch == BR_FP_DEG_POOL_SELECTION_CONFOUND
    assert branch != BR_FP_DEG_UNRESOLVED


def test_full_pool_independence_aggregates_excluded_and_parity_keys():
    from calm.hrm_text_158.native_full_stack.fp_credit_degeneracy_diagnostic import (
        _aggregate_pool_metrics,
    )

    excluded = _pool_metrics(
        move_candidates=1_000_000,
        fp_vote4_count=275_000,
        fp_vote1_count=725_000,
        module_key="excluded_key",
    )
    parity_key = _pool_metrics(
        move_candidates=100_000,
        fp_vote4_count=50_000,
        fp_vote1_count=50_000,
        module_key="parity_key",
    )
    per_key = [excluded, parity_key]
    pooled = _aggregate_pool_metrics(
        pool_id="TRUE_FULL_FP_MOVE_POOL",
        per_key=per_key,
    )
    assert pooled.move_candidates == 1_100_000
    assert pooled.fp_vote4_fraction == pytest.approx(325_000 / 1_100_000)
    assert len(per_key) == 2


def test_consistency_anchor_failure_invalidates_branch():
    full = _pool_metrics(move_candidates=100_000, fp_vote4_count=50_000, fp_vote1_count=50_000)
    enrichment = _enrichment(full_pool_fp4=0.50, parity_fp4=0.96, consistency_pass=False)
    assert (
        classify_fp_credit_degeneracy_branch(full_pool=full, enrichment=enrichment)
        == BR_FP_DEG_MEASUREMENT_INVALID
    )


def test_enrichment_not_self_cancel_regression():
    full_pool_fp4 = 0.50
    parity_fp4 = 0.96
    enrichment = parity_fp4 - full_pool_fp4
    tautology_bias = parity_fp4 - parity_fp4
    assert enrichment == pytest.approx(0.46, abs=0.01)
    assert tautology_bias == pytest.approx(0.0, abs=1e-9)
    assert enrichment != tautology_bias


def test_coverage_metric_ratio_fixture():
    full = _pool_metrics(
        move_candidates=100_000,
        fp_vote4_count=50_000,
        fp_vote1_count=50_000,
        module_key="k",
    )
    parity = _pool_metrics(
        move_candidates=1_000,
        fp_vote4_count=900,
        fp_vote1_count=100,
        pool_id="PARITY_INTEGER_FILTERED_POOL",
        module_key="k",
    )
    enrichment = compute_parity_enrichment_metrics(
        full_pool=full,
        parity_pool=parity,
        per_key_full=[full],
        per_key_parity=[parity],
        parity_records_by_key={"k": []},
        dense_fp_votes_by_key={"k": torch.zeros(1, dtype=torch.int16)},
    )
    assert enrichment.integer_move_coverage_vs_full_fp == pytest.approx(0.01)


def test_coverage_not_a_classifier_gate():
    full = _pool_metrics(move_candidates=100_000, fp_vote4_count=27_500, fp_vote1_count=72_500)
    low_cov = _enrichment(full_pool_fp4=0.50, parity_fp4=0.96, parity_candidates=10)
    high_cov = _enrichment(full_pool_fp4=0.50, parity_fp4=0.96, parity_candidates=50_000)
    assert classify_fp_credit_degeneracy_branch(full_pool=full, enrichment=low_cov) == (
        classify_fp_credit_degeneracy_branch(full_pool=full, enrichment=high_cov)
    )


@pytest.mark.parametrize("full_pool_fp4", [0.30, 0.90])
def test_anchor_violation_measurement_invalid_not_confound(full_pool_fp4: float):
    vote4 = int(100_000 * full_pool_fp4)
    full = _pool_metrics(
        move_candidates=100_000,
        fp_vote4_count=vote4,
        fp_vote1_count=100_000 - vote4,
    )
    enrichment = _enrichment(full_pool_fp4=full_pool_fp4, parity_fp4=0.96)
    branch = classify_fp_credit_degeneracy_branch(full_pool=full, enrichment=enrichment)
    assert branch == BR_FP_DEG_MEASUREMENT_INVALID
    assert branch != BR_FP_DEG_POOL_SELECTION_CONFOUND


def test_amplified_fixable_low_enrichment_high_parity():
    full = _pool_metrics(move_candidates=100_000, fp_vote4_count=50_000, fp_vote1_count=50_000)
    enrichment = _enrichment(full_pool_fp4=0.50, parity_fp4=0.90)
    enrichment = ParityEnrichmentMetrics(
        full_pool_fp4_fraction=0.50,
        parity_fp4_balance=0.90,
        full_pool_fp4_at_parity_indices=0.90,
        parity_index_enrichment_over_full_pool=0.30,
        integer_move_coverage_vs_full_fp=0.01,
        parity_move_candidates=1000,
        full_pool_move_candidates=100_000,
        consistency_anchor_pass=True,
        consistency_anchor_delta=0.0,
    )
    branch = classify_fp_credit_degeneracy_branch(full_pool=full, enrichment=enrichment)
    assert branch == BR_FP_DEG_AMPLIFIED_FIXABLE


def test_no_parity_enrichment_when_parity_not_dominant():
    full = _pool_metrics(move_candidates=100_000, fp_vote4_count=50_000, fp_vote1_count=50_000)
    enrichment = _enrichment(full_pool_fp4=0.50, parity_fp4=0.50)
    branch = classify_fp_credit_degeneracy_branch(full_pool=full, enrichment=enrichment)
    assert branch == BR_FP_DEG_NO_PARITY_ENRICHMENT


@pytest.mark.skipif(
    not discover_t2_checkpoint().checkpoint_present,
    reason=f"missing checkpoint {DEFAULT_T2_CHECKPOINT_REL}",
)
def test_live_t2_v4_receipt_and_branch():
    discovery = discover_t2_checkpoint()
    payload = run_anchored_fp_credit_degeneracy_diagnostic(
        checkpoint_path=str(discovery.checkpoint_path),
        checkpoint_sha256=discovery.checkpoint_sha256,
    )
    assert payload["schema"] == FP_CREDIT_DEGENERACY_SCHEMA_V2
    assert payload["pass_receipt"] is False
    assert all(value is False for value in payload["hard_false"].values())
    assert payload["branch_id"] == BR_FP_DEG_POOL_SELECTION_CONFOUND
    assert payload["full_pool_fp4_anchor_in_band"] is True
    assert payload["intrinsic_via_full_pool_fp4_structurally_unreachable_under_locked_spec"] is True
    audit = payload["key_filter_audit"]
    assert audit["full_pool_key_count"] == audit["captured_key_count"] == 32
    assert audit["full_pool_key_count"] >= audit["parity_valid_key_count"]
    enrichment = payload["enrichment"]
    assert enrichment["consistency_anchor_pass"] is True
    assert enrichment["parity_index_enrichment_over_full_pool"] == pytest.approx(0.46, abs=0.05)
    anchor = payload["anchor"]
    assert anchor["curriculum_seed"] == FROZEN_T2_ANCHOR_CURRICULUM_SEED
    assert anchor["batch_size"] == FROZEN_T2_ANCHOR_BATCH_SIZE
    assert anchor["checkpoint_sha256"] == FROZEN_T2_ANCHOR_CHECKPOINT_SHA256
    assert anchor["key_set_sha256"] == FROZEN_T2_ANCHOR_KEY_SET_SHA256
    assert "BR-FP-DEG-INTRINSIC" not in payload["branch_id"]
