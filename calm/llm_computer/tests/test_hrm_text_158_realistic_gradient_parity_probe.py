"""CPU tests for 3C-B realistic-gradient parity probe (v1.3)."""
from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from unittest import mock

import pytest
import torch
import torch.nn.functional as F

from calm.hrm_text_158.native_full_stack.realistic_gradient_parity_probe import (
    CONCURRENCE_CONCUR,
    CONCURRENCE_DISAGREE,
    CONCURRENCE_T2_ABSENT,
    CONCURRENCE_T2_REQUIRED_MISSING,
    DEFAULT_T2_CHECKPOINT_REL,
    FORBIDDEN_FIXTURE_WEIGHT_SHAPE,
    GPU_GATE_INSUFFICIENT,
    GPU_GATE_INVESTIGATE,
    GPU_GATE_PROCEED,
    GPU_GATE_PROCEED_NARROW,
    GPU_GATE_REOPEN_3C_A,
    MAX_PER_CANDIDATE_RECORDS_PER_KEY,
    MIN_FP_CREDIT_NONZERO,
    MIN_FRACTIONAL_DIVERSITY,
    FRACTIONAL_DIVERSITY_RELATIVE_BINS,
    MIN_MOVE_CANDIDATES,
    MIN_RANK_GROUPS,
    MIN_TIER_TOTAL_MOVE_CANDIDATES,
    REALISTIC_GRADIENT_PARITY_PROBE_HARD_FALSE_FIELDS,
    REALISTIC_GRADIENT_PARITY_PROBE_NON_CLAIMS,
    VERDICT_BROAD_HOLDS,
    VERDICT_FRACTIONAL_COLLAPSE,
    VERDICT_MEASUREMENT_INVALID,
    VERDICT_NARROW_HOLDS,
    RealisticGradientParityProbeReceipt,
    TierParityVerdict,
    T2CheckpointDiscovery,
    build_per_candidate_parity_records,
    build_trainer_16x16_capture_fixture,
    classify_gpu_gate_recommendation,
    classify_t1_t2_concurrence,
    classify_tier_parity_verdict,
    discover_t2_checkpoint,
    realistic_gradient_parity_probe_hard_false_snapshot,
    run_realistic_gradient_parity_probe,
    run_tier1_trainer_16x16_capture,
    run_tier2_checkpoint_capture,
    capture_tier2_checkpoint_raw_captures,
    validate_realistic_gradient_parity_probe_receipt,
    _fractional_diversity_count,
    _measurement_validity_for_key,
    _probe_key_from_captures,
)
from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    default_dry_run_rank_vote_spec,
)
from calm.hrm_text_158.native_full_stack.integer_marginal_attribution import (
    INTEGER_MARGINAL_ATTRIBUTION_LAW_ID_V0,
)
from calm.hrm_text_158.native_full_stack.integer_sparse_rank_votes import (
    CREDIT_LAW_NEG_ATTRIBUTION_Q31_V0,
)
from calm.hrm_text_158.native_full_stack.trainer_sub2_authority import (
    derive_trainer_sub2_authority_states,
    select_trainer_eligible_bitlinears,
    trainer_authoritative_forward_context,
)


def _tier_verdict(
    tier_id: str,
    *,
    measurement_valid: bool,
    parity_verdict: str,
    rank: float = 1.0,
    events: float = 1.0,
    fractional: float = 0.0,
) -> TierParityVerdict:
    return TierParityVerdict(
        tier_id=tier_id,
        measurement_valid=measurement_valid,
        parity_verdict=parity_verdict,
        rank_positions_match_rate=rank,
        events_match_rate=events,
        fractional_collision_share_of_mismatches=fractional,
    )


def test_fixture_is_not_bitlinear_3x2():
    fixture = build_trainer_16x16_capture_fixture()
    assert fixture.weight_shape() != FORBIDDEN_FIXTURE_WEIGHT_SHAPE
    assert fixture.weight_shape() == (16, 16)


def test_probe_module_does_not_import_tests_package():
    import calm.hrm_text_158.native_full_stack.realistic_gradient_parity_probe as probe

    source = inspect.getsource(probe)
    assert "calm.llm_computer.tests" not in source


def test_tier1_probe_run_measurement_valid_and_allowed_verdict():
    result = run_tier1_trainer_16x16_capture()
    assert result.measurement_valid is True
    verdict = classify_tier_parity_verdict(
        tier_id="T1",
        measurement_valid=result.measurement_valid,
        rank_positions_match_rate=result.aggregate_metrics["rank_positions_match_rate"],
        events_match_rate=result.aggregate_metrics["events_match_rate"],
        fractional_collision_share_of_mismatches=result.aggregate_metrics[
            "fractional_collision_share_of_mismatches"
        ],
    )
    assert verdict.parity_verdict in {
        VERDICT_BROAD_HOLDS,
        VERDICT_NARROW_HOLDS,
        VERDICT_FRACTIONAL_COLLAPSE,
        VERDICT_MEASUREMENT_INVALID,
    }
    assert result.aggregate_metrics["total_move_candidates"] >= MIN_TIER_TOTAL_MOVE_CANDIDATES


def test_per_candidate_records_nonzero_and_fractional_collision_detector():
    result = run_tier1_trainer_16x16_capture()
    records = result.per_key_metrics["proj"].per_candidate_records
    assert len(records) > 0
    assert any(item.fractional_collision_mismatch for item in records) or all(
        item.rank_match for item in records
    )


def test_validity_gate_negatives_fail_closed():
    valid, _ = _measurement_validity_for_key(
        captures_present=True,
        captures_finite=True,
        move_candidate_count=MIN_MOVE_CANDIDATES - 1,
        fp_credit_nonzero_count=MIN_FP_CREDIT_NONZERO,
        fractional_diversity=MIN_FRACTIONAL_DIVERSITY,
        rank_group_count=MIN_RANK_GROUPS,
    )
    assert valid is False
    verdict = classify_tier_parity_verdict(
        tier_id="T1",
        measurement_valid=False,
        rank_positions_match_rate=1.0,
        events_match_rate=1.0,
        fractional_collision_share_of_mismatches=0.0,
    )
    assert verdict.parity_verdict == VERDICT_MEASUREMENT_INVALID


def test_degenerate_capture_measurement_invalid_not_broad():
    verdict = classify_tier_parity_verdict(
        tier_id="T1",
        measurement_valid=False,
        rank_positions_match_rate=1.0,
        events_match_rate=1.0,
        fractional_collision_share_of_mismatches=0.0,
    )
    assert verdict.parity_verdict != VERDICT_BROAD_HOLDS


def _sub_1e3_credits(*relative_fractions: float) -> torch.Tensor:
    max_abs = 5.05e-4
    return torch.tensor(
        [float(fraction) * max_abs for fraction in relative_fractions],
        dtype=torch.float32,
    )


def test_fractional_diversity_t1_characterization_locked_at_sixteen():
    result = run_tier1_trainer_16x16_capture()
    assert result.per_key_metrics["proj"].validity_detail["fractional_diversity"] == 16


@pytest.mark.parametrize(
    ("credits", "expected"),
    [
        (_sub_1e3_credits(1.0, 1.0, 1.0), 1),
        (_sub_1e3_credits(0.31, 1.0), 2),
        (_sub_1e3_credits(0.21, 0.51, 1.0), 3),
    ],
)
def test_fractional_diversity_sub_1e3_level_counts(
    credits: torch.Tensor,
    expected: int,
) -> None:
    assert _fractional_diversity_count(credits) == expected


def test_fractional_diversity_sub_1e3_three_level_passes_validity_gate():
    diversity = _fractional_diversity_count(_sub_1e3_credits(0.21, 0.51, 1.0))
    valid, _ = _measurement_validity_for_key(
        captures_present=True,
        captures_finite=True,
        move_candidate_count=MIN_MOVE_CANDIDATES,
        fp_credit_nonzero_count=MIN_FP_CREDIT_NONZERO,
        fractional_diversity=diversity,
        rank_group_count=MIN_RANK_GROUPS,
    )
    assert diversity == MIN_FRACTIONAL_DIVERSITY
    assert valid is True


def test_fractional_diversity_sub_1e3_two_level_rejected_by_validity_gate():
    diversity = _fractional_diversity_count(_sub_1e3_credits(0.31, 1.0))
    valid, _ = _measurement_validity_for_key(
        captures_present=True,
        captures_finite=True,
        move_candidate_count=MIN_MOVE_CANDIDATES,
        fp_credit_nonzero_count=MIN_FP_CREDIT_NONZERO,
        fractional_diversity=diversity,
        rank_group_count=MIN_RANK_GROUPS,
    )
    assert diversity == 2
    assert valid is False


def test_fractional_diversity_rounding_avoids_half_bin_fixture_values():
    # torch.round half-bin ties are banker's rounding; fixtures use off-half fractions.
    bins = float(FRACTIONAL_DIVERSITY_RELATIVE_BINS)
    credits = _sub_1e3_credits(0.21, 0.51, 1.0)
    max_abs = float(credits.abs().max().item())
    masked = credits[credits.abs() > 0.05 * max_abs]
    scaled = (masked / max_abs) * bins
    frac_part = scaled - torch.floor(scaled)
    assert not torch.any(torch.abs(frac_part - 0.5) < 1e-6)
    assert _fractional_diversity_count(credits) == 3


def test_checkpoint_present_t2_not_run_rejects_proceed():
    tier_verdicts = {
        "T1": _tier_verdict("T1", measurement_valid=True, parity_verdict=VERDICT_BROAD_HOLDS),
    }
    parity, gate, _ = classify_gpu_gate_recommendation(
        t2_checkpoint_present=True,
        tiers_executed=("T1",),
        tier_verdicts=tier_verdicts,
        t1_t2_concurrence=CONCURRENCE_T2_REQUIRED_MISSING,
    )
    assert gate != GPU_GATE_PROCEED
    assert gate == GPU_GATE_INSUFFICIENT
    receipt = RealisticGradientParityProbeReceipt(
        schema_version="hrm_text_158_realistic_gradient_parity_probe/v1_3",
        target_name="step3c_b_realistic_gradient_parity_probe",
        parity_contract_mode="REALISTIC_GRADIENT_ORDERING_PROBE",
        pass_receipt=False,
        tiers_executed=("T1",),
        t2_checkpoint_present=True,
        t2_checkpoint_path="/tmp/fake.pt",
        t2_checkpoint_sha256="0" * 64,
        t2_absence_proof=None,
        tier_verdicts=tier_verdicts,
        t2_verdict=None,
        t1_t2_concurrence=CONCURRENCE_T2_REQUIRED_MISSING,
        hrm_representativeness_unconfirmed=False,
        capture_provenance={},
        measurement_valid=True,
        measurement_validity_detail={},
        tensor_key_sample_set={},
        per_key_metrics={},
        per_candidate_records={},
        aggregate_metrics={},
        mismatch_clusters={},
        fractional_collision_examples={},
        parity_verdict=parity,
        gpu_gate_recommendation=gate,
        **realistic_gradient_parity_probe_hard_false_snapshot(),
        non_claims=REALISTIC_GRADIENT_PARITY_PROBE_NON_CLAIMS,
    )
    validate_realistic_gradient_parity_probe_receipt(receipt)


def test_t1_t2_disagreement_investigate():
    tier_verdicts = {
        "T1": _tier_verdict("T1", measurement_valid=True, parity_verdict=VERDICT_BROAD_HOLDS),
        "T2": _tier_verdict("T2", measurement_valid=True, parity_verdict=VERDICT_FRACTIONAL_COLLAPSE),
    }
    concurrence = classify_t1_t2_concurrence(
        t2_checkpoint_present=True,
        tiers_executed=("T1", "T2"),
        tier_verdicts=tier_verdicts,
    )
    assert concurrence == CONCURRENCE_DISAGREE
    parity, gate, _ = classify_gpu_gate_recommendation(
        t2_checkpoint_present=True,
        tiers_executed=("T1", "T2"),
        tier_verdicts=tier_verdicts,
        t1_t2_concurrence=concurrence,
    )
    assert parity == "investigate"
    assert gate == GPU_GATE_INVESTIGATE
    assert gate != GPU_GATE_PROCEED


def test_t1_t2_concordant_broad_allows_proceed():
    tier_verdicts = {
        "T1": _tier_verdict("T1", measurement_valid=True, parity_verdict=VERDICT_BROAD_HOLDS),
        "T2": _tier_verdict("T2", measurement_valid=True, parity_verdict=VERDICT_BROAD_HOLDS),
    }
    concurrence = classify_t1_t2_concurrence(
        t2_checkpoint_present=True,
        tiers_executed=("T1", "T2"),
        tier_verdicts=tier_verdicts,
    )
    assert concurrence == CONCURRENCE_CONCUR
    parity, gate, unconfirmed = classify_gpu_gate_recommendation(
        t2_checkpoint_present=True,
        tiers_executed=("T1", "T2"),
        tier_verdicts=tier_verdicts,
        t1_t2_concurrence=concurrence,
    )
    assert parity == VERDICT_BROAD_HOLDS
    assert gate == GPU_GATE_PROCEED
    assert unconfirmed is False


def test_checkpoint_absent_t1_broad_insufficient_evidence():
    tier_verdicts = {
        "T1": _tier_verdict("T1", measurement_valid=True, parity_verdict=VERDICT_BROAD_HOLDS),
    }
    concurrence = classify_t1_t2_concurrence(
        t2_checkpoint_present=False,
        tiers_executed=("T1",),
        tier_verdicts=tier_verdicts,
    )
    assert concurrence == CONCURRENCE_T2_ABSENT
    _parity, gate, _ = classify_gpu_gate_recommendation(
        t2_checkpoint_present=False,
        tiers_executed=("T1",),
        tier_verdicts=tier_verdicts,
        t1_t2_concurrence=concurrence,
    )
    assert gate == GPU_GATE_INSUFFICIENT
    assert gate != GPU_GATE_PROCEED


def test_checkpoint_absent_t1_narrow_proceed_narrow_with_caveat():
    tier_verdicts = {
        "T1": _tier_verdict("T1", measurement_valid=True, parity_verdict=VERDICT_NARROW_HOLDS),
    }
    concurrence = classify_t1_t2_concurrence(
        t2_checkpoint_present=False,
        tiers_executed=("T1",),
        tier_verdicts=tier_verdicts,
    )
    _parity, gate, unconfirmed = classify_gpu_gate_recommendation(
        t2_checkpoint_present=False,
        tiers_executed=("T1",),
        tier_verdicts=tier_verdicts,
        t1_t2_concurrence=concurrence,
    )
    assert gate == GPU_GATE_PROCEED_NARROW
    assert unconfirmed is True


def test_hard_false_and_pass_receipt_validator_negatives():
    snapshot = realistic_gradient_parity_probe_hard_false_snapshot()
    assert len(snapshot) == 7
    assert all(value is False for value in snapshot.values())
    tier_verdicts = {
        "T1": _tier_verdict("T1", measurement_valid=True, parity_verdict=VERDICT_FRACTIONAL_COLLAPSE),
    }
    receipt = RealisticGradientParityProbeReceipt(
        schema_version="hrm_text_158_realistic_gradient_parity_probe/v1_3",
        target_name="step3c_b_realistic_gradient_parity_probe",
        parity_contract_mode="REALISTIC_GRADIENT_ORDERING_PROBE",
        pass_receipt=False,
        tiers_executed=("T1",),
        t2_checkpoint_present=False,
        t2_checkpoint_path=None,
        t2_checkpoint_sha256=None,
        t2_absence_proof="absent",
        tier_verdicts=tier_verdicts,
        t2_verdict=None,
        t1_t2_concurrence=CONCURRENCE_T2_ABSENT,
        hrm_representativeness_unconfirmed=False,
        capture_provenance={},
        measurement_valid=True,
        measurement_validity_detail={},
        tensor_key_sample_set={},
        per_key_metrics={},
        per_candidate_records={},
        aggregate_metrics={},
        mismatch_clusters={},
        fractional_collision_examples={},
        parity_verdict=VERDICT_FRACTIONAL_COLLAPSE,
        gpu_gate_recommendation=GPU_GATE_REOPEN_3C_A,
        **snapshot,
        non_claims=REALISTIC_GRADIENT_PARITY_PROBE_NON_CLAIMS,
    )
    validate_realistic_gradient_parity_probe_receipt(receipt)
    with pytest.raises(ValueError, match="pass_receipt"):
        validate_realistic_gradient_parity_probe_receipt(
            RealisticGradientParityProbeReceipt(
                **{
                    **receipt.__dict__,
                    "pass_receipt": True,
                }
            )
        )
    for field in REALISTIC_GRADIENT_PARITY_PROBE_HARD_FALSE_FIELDS:
        bad = dict(receipt.__dict__)
        bad[field] = True
        with pytest.raises(ValueError, match=field):
            validate_realistic_gradient_parity_probe_receipt(
                RealisticGradientParityProbeReceipt(**bad)
            )


def test_regression_banked_modules_importable():
    importlib.import_module("calm.hrm_text_158.native_full_stack.integer_marginal_attribution")
    importlib.import_module("calm.hrm_text_158.native_full_stack.integer_sparse_rank_votes")
    importlib.import_module("calm.hrm_text_158.native_full_stack.sparse_vote_authority_apply")
    importlib.import_module("calm.hrm_text_158.native_full_stack.integer_optimizer_credit_path")


@pytest.mark.skipif(
    not Path(DEFAULT_T2_CHECKPOINT_REL).is_file()
    and not (
        Path(__file__).resolve().parents[3] / DEFAULT_T2_CHECKPOINT_REL
    ).is_file(),
    reason="default T2 checkpoint absent on disk",
)
def test_t2_live_read_only_integration_when_checkpoint_present():
    discovery = discover_t2_checkpoint()
    assert discovery.checkpoint_present
    result = run_tier2_checkpoint_capture(checkpoint_path=str(discovery.checkpoint_path))
    assert result.tier_id == "T2"
    assert len(result.tensor_key_sample_set) > 0


@pytest.mark.skipif(
    not Path(DEFAULT_T2_CHECKPOINT_REL).is_file()
    and not (
        Path(__file__).resolve().parents[3] / DEFAULT_T2_CHECKPOINT_REL
    ).is_file(),
    reason="default T2 checkpoint absent on disk",
)
def test_capture_tier2_checkpoint_raw_captures_matches_probe_metrics():
    discovery = discover_t2_checkpoint()
    assert discovery.checkpoint_present
    checkpoint_path = str(discovery.checkpoint_path)
    bundle = capture_tier2_checkpoint_raw_captures(checkpoint_path=checkpoint_path)
    wrapper = run_tier2_checkpoint_capture(checkpoint_path=checkpoint_path)
    spec = default_dry_run_rank_vote_spec()
    assert bundle.provenance.get("capture_seam_id") == "capture_tier2_checkpoint_raw_captures"
    assert sorted(bundle.per_key_captures.keys()) == sorted(wrapper.per_key_metrics.keys())
    for key, capture in bundle.per_key_captures.items():
        direct = _probe_key_from_captures(
            state_key=key,
            inputs=capture.inputs,
            grad_outputs=capture.grad_outputs,
            weight_shape=capture.weight_shape,
            q_levels_flat=capture.q_levels_flat,
            spec=spec,
        )
        wrapped = wrapper.per_key_metrics[key]
        assert direct.move_candidate_count == wrapped.move_candidate_count
        assert direct.measurement_valid == wrapped.measurement_valid
        assert direct.rank_positions_match_rate == pytest.approx(
            wrapped.rank_positions_match_rate
        )
        assert direct.events_match_rate == pytest.approx(wrapped.events_match_rate)
        assert direct.branch_id == wrapped.branch_id
    assert wrapper.capture_provenance["tensor_keys_probed"] == bundle.provenance[
        "tensor_keys_probed"
    ]


def test_verdict_metrics_use_full_candidate_set_not_receipt_cap():
    fixture = build_trainer_16x16_capture_fixture()
    model = fixture.model
    eligible = select_trainer_eligible_bitlinears(model, use_ternary_bulk=True)
    states = derive_trainer_sub2_authority_states(eligible)
    model.train(True)
    model.zero_grad(set_to_none=True)
    with trainer_authoritative_forward_context(
        eligible,
        states,
        device="cpu",
        requires_grad=True,
    ) as handle:
        out = model(fixture.batch["x"])
        loss = F.mse_loss(out, fixture.batch["target"])
        loss.backward()
        capture = handle.captures["proj"]
        state = states["proj"]
        common = dict(
            inputs=capture["inputs"],
            grad_outputs=capture["grad_outputs"],
            weight_shape=tuple(state.q_levels.shape),
            q_levels_flat=state.q_levels.reshape(-1),
            spec=default_dry_run_rank_vote_spec(),
        )
        uncapped_records, uncapped_summary = build_per_candidate_parity_records(
            **common,
            attribution_law_id=INTEGER_MARGINAL_ATTRIBUTION_LAW_ID_V0,
            credit_law_id=CREDIT_LAW_NEG_ATTRIBUTION_Q31_V0,
            max_records=10_000,
        )
        capped_records, capped_summary = build_per_candidate_parity_records(
            **common,
            attribution_law_id=INTEGER_MARGINAL_ATTRIBUTION_LAW_ID_V0,
            credit_law_id=CREDIT_LAW_NEG_ATTRIBUTION_Q31_V0,
            max_records=8,
        )
    assert uncapped_summary["move_candidate_count"] > MAX_PER_CANDIDATE_RECORDS_PER_KEY
    assert len(capped_records) == 8
    assert len(uncapped_records) == uncapped_summary["move_candidate_count"]
    assert capped_summary["move_candidate_count"] == uncapped_summary["move_candidate_count"]
    assert capped_summary["rank_positions_match_rate"] == pytest.approx(
        uncapped_summary["rank_positions_match_rate"]
    )
    assert capped_summary["events_match_rate"] == pytest.approx(
        uncapped_summary["events_match_rate"]
    )
    assert capped_summary["fractional_collision_share_of_mismatches"] == pytest.approx(
        uncapped_summary["fractional_collision_share_of_mismatches"]
    )
    prefix_only_rate = sum(1 for item in capped_records if item.rank_match) / len(capped_records)
    assert prefix_only_rate != pytest.approx(uncapped_summary["rank_positions_match_rate"])


def test_full_probe_end_to_end_t1_only_when_t2_absent(tmp_path, monkeypatch):
    missing = tmp_path / "missing.pt"
    monkeypatch.setenv("HRM_TEXT_158_PROBE_CHECKPOINT", str(missing))
    receipt = run_realistic_gradient_parity_probe(run_t2=False)
    validate_realistic_gradient_parity_probe_receipt(receipt)
    assert receipt.tiers_executed == ("T1",)
    assert receipt.pass_receipt is False
