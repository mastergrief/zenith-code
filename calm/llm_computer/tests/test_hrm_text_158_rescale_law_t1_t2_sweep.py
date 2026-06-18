"""Tests for read-only rescale-law T1/T2 sweep (Phase 1 extraction + dual-tier selector)."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from calm.hrm_text_158.native_full_stack.integer_marginal_attribution import (
    INTEGER_MARGINAL_ATTRIBUTION_PRODUCTION_LAW_ID,
    ATTRIBUTION_RESCALE_SHIFT_V1,
)
from calm.hrm_text_158.native_full_stack.realistic_gradient_parity_probe import (
    DEFAULT_T2_CHECKPOINT_REL,
    MIN_MOVE_CANDIDATES,
    MIN_TIER_TOTAL_MOVE_CANDIDATES,
    VERDICT_BROAD_HOLDS,
    VERDICT_FRACTIONAL_COLLAPSE,
    VERDICT_NARROW_HOLDS,
    _measurement_validity_for_key,
    discover_t2_checkpoint,
)
from calm.hrm_text_158.native_full_stack.rescale_law_readonly_sweep import (
    DUAL_TIER_SWEEP_SCHEMA_V1,
    RESCALE_LAW_READONLY_SWEEP_SCHEMA_V1,
    ROUTING_STRENGTH_CAVEATED,
    ROUTING_STRENGTH_CLEAN,
    RescaleSaturationError,
    DualTierCandidateResult,
    SELECTOR_OUTCOME_PIN_COARSEST_T2_BROAD,
    SELECTOR_OUTCOME_PIN_COARSEST_T2_NARROW_CAVEATED,
    SELECTOR_OUTCOME_STOP_NO_SHIFT_CLEARS_VALID_T2,
    ShiftParityKeyResult,
    SweepCandidateResult,
    TierShiftParityResult,
    aggregate_tier_from_shift_parity_key_results,
    apply_valid_t2_selector,
    rescale_accumulator_to_attribution_q,
    run_dual_tier_readonly_sweep,
    run_readonly_sweep,
)


def _sweep_result(
    *,
    shift: int,
    verdict: str,
    rank_rate: float = 1.0,
    event_rate: float = 1.0,
    saturation_fail_count: int = 0,
    measurement_valid: bool = True,
) -> SweepCandidateResult:
    return SweepCandidateResult(
        candidate_id=f"rescale_q{shift}",
        rescale_shift=shift,
        move_candidate_count=32,
        rank_positions_match_rate=rank_rate,
        events_match_rate=event_rate,
        fractional_collision_share_of_mismatches=0.0,
        parity_verdict=verdict,
        measurement_valid=measurement_valid,
        rank_match_count=32,
        event_match_count=32,
        mismatch_count=0,
        fractional_collision_mismatch_count=0,
        saturation_fail_count=saturation_fail_count,
    )


def _tier_result(
    *,
    shift: int,
    verdict: str,
    measurement_valid: bool = True,
    saturation_fail_count: int = 0,
) -> TierShiftParityResult:
    return TierShiftParityResult(
        tier_id="T2",
        rescale_shift=shift,
        candidate_id=f"rescale_q{shift}",
        measurement_valid=measurement_valid,
        parity_verdict=verdict,
        rank_positions_match_rate=1.0 if verdict == VERDICT_BROAD_HOLDS else 0.85,
        events_match_rate=0.95 if verdict == VERDICT_BROAD_HOLDS else 0.75,
        fractional_collision_share_of_mismatches=0.0,
        total_move_candidates=32,
        valid_key_count=1,
        saturation_fail_count=saturation_fail_count,
        per_key_results=(),
    )


def _dual(
    shift: int,
    *,
    t1_verdict: str = VERDICT_BROAD_HOLDS,
    t2_verdict: str,
    t2_valid: bool = True,
    t1_saturation: int = 0,
    t2_saturation: int = 0,
) -> DualTierCandidateResult:
    return DualTierCandidateResult(
        candidate_id=f"rescale_q{shift}",
        rescale_shift=shift,
        t1=_sweep_result(
            shift=shift,
            verdict=t1_verdict,
            saturation_fail_count=t1_saturation,
        ),
        t2=_tier_result(
            shift=shift,
            verdict=t2_verdict,
            measurement_valid=t2_valid,
            saturation_fail_count=t2_saturation,
        ),
    )


def test_t1_characterization_gate_reproduces_post_extraction():
    payload = run_readonly_sweep()
    assert payload["schema"] == RESCALE_LAW_READONLY_SWEEP_SCHEMA_V1
    assert payload["production_module_unchanged"] is True
    by_id = {row["candidate_id"]: row for row in payload["candidate_results"]}
    assert by_id["law_v0"]["parity_verdict"] == VERDICT_FRACTIONAL_COLLAPSE
    assert by_id["law_v0"]["rank_positions_match_rate"] == pytest.approx(0.1875)
    for candidate_id in ("rescale_q24", "rescale_q16", "rescale_q8"):
        row = by_id[candidate_id]
        assert row["parity_verdict"] == VERDICT_BROAD_HOLDS
        assert row["rank_positions_match_rate"] == pytest.approx(1.0)
        assert row["events_match_rate"] == pytest.approx(1.0)


def test_production_law_id_unchanged():
    assert INTEGER_MARGINAL_ATTRIBUTION_PRODUCTION_LAW_ID.endswith("rescale_q24_v1")
    assert ATTRIBUTION_RESCALE_SHIFT_V1 == 24


def test_selector_s16_narrow_s8_broad_selects_s8():
    dual = {
        16: _dual(16, t2_verdict=VERDICT_NARROW_HOLDS),
        8: _dual(8, t2_verdict=VERDICT_BROAD_HOLDS),
    }
    outcome = apply_valid_t2_selector(dual)
    assert outcome["selector_outcome"] == SELECTOR_OUTCOME_PIN_COARSEST_T2_BROAD
    assert outcome["selected_shift"] == 8
    assert outcome["routing_strength"] == ROUTING_STRENGTH_CLEAN


def test_selector_narrow_only_yields_caveated():
    dual = {
        16: _dual(16, t2_verdict=VERDICT_NARROW_HOLDS),
        8: _dual(8, t2_verdict=VERDICT_NARROW_HOLDS),
    }
    outcome = apply_valid_t2_selector(dual)
    assert outcome["selector_outcome"] == SELECTOR_OUTCOME_PIN_COARSEST_T2_NARROW_CAVEATED
    assert outcome["selected_shift"] == 16
    assert outcome["routing_strength"] == ROUTING_STRENGTH_CAVEATED


def test_selector_rejects_saturation_fail_candidate():
    dual = {
        16: _dual(16, t2_verdict=VERDICT_BROAD_HOLDS, t2_saturation=1),
        8: _dual(8, t2_verdict=VERDICT_BROAD_HOLDS, t1_saturation=1),
    }
    outcome = apply_valid_t2_selector(dual)
    assert outcome["selector_outcome"] == SELECTOR_OUTCOME_STOP_NO_SHIFT_CLEARS_VALID_T2
    assert outcome["selected_shift"] is None


def test_per_key_validity_move_count_seven_invalid():
    valid, detail = _measurement_validity_for_key(
        captures_present=True,
        captures_finite=True,
        move_candidate_count=7,
        fp_credit_nonzero_count=10,
        fractional_diversity=5,
        rank_group_count=3,
    )
    assert valid is False
    assert detail["move_candidate_count"] == 7
    assert detail["min_move_candidates"] == MIN_MOVE_CANDIDATES


def test_tier_validity_requires_total_move_candidates():
    per_key = (
        ShiftParityKeyResult(
            state_key="k0",
            measurement_valid=True,
            validity_detail={"measurement_valid": True},
            move_candidate_count=10,
            rank_positions_match_rate=1.0,
            events_match_rate=1.0,
            fractional_collision_share_of_mismatches=0.0,
            saturation_failed=False,
        ),
    )
    tier = aggregate_tier_from_shift_parity_key_results(
        tier_id="T2",
        candidate_id="rescale_q16",
        rescale_shift=16,
        per_key_results=per_key,
    )
    assert tier.valid_key_count == 1
    assert tier.total_move_candidates == 10
    assert tier.total_move_candidates < MIN_TIER_TOTAL_MOVE_CANDIDATES
    assert tier.measurement_valid is False


def test_rescale_saturation_fail_closed():
    huge = torch.tensor([2**60], dtype=torch.int64)
    with pytest.raises(RescaleSaturationError, match="outside int32"):
        rescale_accumulator_to_attribution_q(huge, shift=8)


@pytest.mark.skipif(
    not Path(DEFAULT_T2_CHECKPOINT_REL).is_file()
    and not (
        Path(__file__).resolve().parents[3] / DEFAULT_T2_CHECKPOINT_REL
    ).is_file(),
    reason="anchored T2 checkpoint absent",
)
def test_live_dual_tier_sweep_on_anchor_checkpoint():
    discovery = discover_t2_checkpoint()
    assert discovery.checkpoint_present
    payload = run_dual_tier_readonly_sweep(checkpoint_path=str(discovery.checkpoint_path))
    assert payload["schema"] == DUAL_TIER_SWEEP_SCHEMA_V1
    assert payload["production_module_unchanged"] is True
    assert "selector" in payload
    assert len(payload["candidate_results"]) == 4


def test_dual_tier_cli_skips_when_checkpoint_absent(tmp_path, monkeypatch):
    monkeypatch.delenv("HRM_TEXT_158_PROBE_CHECKPOINT", raising=False)
    monkeypatch.setattr(
        "calm.hrm_text_158.native_full_stack.realistic_gradient_parity_probe._repo_root",
        lambda: tmp_path,
    )
    discovery = discover_t2_checkpoint()
    assert discovery.checkpoint_present is False
    assert discovery.absence_proof is not None
