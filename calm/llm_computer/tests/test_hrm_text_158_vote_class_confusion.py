"""Tests for read-only T2 vote-class confusion diagnostic (v2)."""
from __future__ import annotations

from pathlib import Path

import pytest

from calm.hrm_text_158.native_full_stack.realistic_gradient_parity_probe import (
    DEFAULT_T2_CHECKPOINT_REL,
    PerCandidateParityRecord,
    discover_t2_checkpoint,
)
from calm.hrm_text_158.native_full_stack.vote_class_confusion_diagnostic import (
    OUTCOME_CLASS_IMBALANCE_CONFOUNDED,
    OUTCOME_SIGN_FLIP_MASKED,
    OUTCOME_VIABLE_CANDIDATE_PARITY,
    VOTE_CLASS_CONFUSION_T2_SCHEMA_V2,
    aggregate_tier_vote_class_confusion,
    classify_vote_class_confusion_outcome,
    compute_vote_class_confusion,
    run_anchored_t2_vote_class_confusion,
)


def _record(*, fp_vote: int, int_vote: int) -> PerCandidateParityRecord:
    return PerCandidateParityRecord(
        flat_index=0,
        fp_credit=1.0,
        int_credit_q31=1,
        fp_rank_position=1,
        int_rank_position=1,
        fp_vote=int(fp_vote),
        int_vote=int(int_vote),
        rank_match=True,
        event_match=fp_vote == int_vote,
        fractional_collision_mismatch=False,
    )


def test_toy_2x2_confusion_exact_arithmetic():
    records = (
        _record(fp_vote=+4, int_vote=+4),
        _record(fp_vote=+4, int_vote=-4),
        _record(fp_vote=+1, int_vote=+1),
        _record(fp_vote=+1, int_vote=-1),
    )
    rates = compute_vote_class_confusion(records)
    assert rates.counts.fp4_count == 2
    assert rates.counts.fp1_count == 2
    assert rates.vote4_abs_recall == 1.0
    assert rates.vote4_signed_recall == 0.5
    assert rates.vote4_abs_precision == 1.0
    assert rates.vote4_signed_precision == 0.5
    assert rates.counts.top4_sign_mismatch_count == 1
    assert rates.top4_sign_mismatch_rate == 0.5
    assert rates.signed_vote_agreement_rate == 0.5
    assert rates.vote4_abs_minus_signed_recall == 0.5


def test_abs_passes_signed_fails_fixture():
    records = (
        _record(fp_vote=+4, int_vote=-4),
        _record(fp_vote=+4, int_vote=+4),
    )
    rates = compute_vote_class_confusion(records)
    assert rates.vote4_abs_recall == 1.0
    assert rates.vote4_signed_recall == 0.5
    assert rates.counts.top4_sign_mismatch_count == 1
    outcome = classify_vote_class_confusion_outcome(rates)
    assert outcome["outcome"] == OUTCOME_SIGN_FLIP_MASKED


def test_signed_vote_agreement_matches_event_match():
    records = (
        _record(fp_vote=+4, int_vote=+4),
        _record(fp_vote=+1, int_vote=-1),
        _record(fp_vote=-1, int_vote=-1),
    )
    rates = compute_vote_class_confusion(records)
    expected = sum(1 for item in records if item.event_match) / len(records)
    assert rates.signed_vote_agreement_rate == pytest.approx(expected)


def test_class_balance_fifty_fifty():
    records = tuple(_record(fp_vote=+4, int_vote=+4) for _ in range(5)) + tuple(
        _record(fp_vote=+1, int_vote=+1) for _ in range(5)
    )
    rates = compute_vote_class_confusion(records)
    assert rates.fp4_balance == 0.5
    assert rates.fp1_balance == 0.5


def test_aggregate_micro_matches_pooled_records():
    key_a = (_record(fp_vote=+4, int_vote=+4),)
    key_b = (_record(fp_vote=+1, int_vote=+1),)
    tier = aggregate_tier_vote_class_confusion(
        rescale_shift=16,
        candidate_id="rescale_q16",
        per_key_records={"a": key_a, "b": key_b},
        valid_keys=("a", "b"),
    )
    pooled = compute_vote_class_confusion(list(key_a) + list(key_b))
    assert tier.micro.counts.n == pooled.counts.n
    assert tier.micro.signed_vote_agreement_rate == pooled.signed_vote_agreement_rate


def test_class_imbalanced_fixture_not_viable():
    """96/4 fp4-dominant: vote-4 signed preserved, vote-1 signed ~=0."""
    records = tuple(_record(fp_vote=+4, int_vote=+4) for _ in range(96)) + tuple(
        _record(fp_vote=+1, int_vote=-1) for _ in range(4)
    )
    rates = compute_vote_class_confusion(records)
    assert rates.fp4_balance == pytest.approx(0.96)
    assert rates.fp1_balance == pytest.approx(0.04)
    assert rates.vote4_signed_recall == pytest.approx(1.0)
    assert rates.vote1_signed_recall == pytest.approx(0.0)
    assert rates.class_balanced_signed_agreement == pytest.approx(0.5)
    outcome = classify_vote_class_confusion_outcome(rates)
    assert outcome["outcome"] == OUTCOME_CLASS_IMBALANCE_CONFOUNDED
    assert outcome["outcome"] != OUTCOME_VIABLE_CANDIDATE_PARITY


def test_balanced_both_classes_signed_preserved_is_viable():
    records = tuple(_record(fp_vote=+4, int_vote=+4) for _ in range(50)) + tuple(
        _record(fp_vote=+1, int_vote=+1) for _ in range(50)
    )
    rates = compute_vote_class_confusion(records)
    assert rates.fp4_balance == pytest.approx(0.5)
    assert rates.fp1_balance == pytest.approx(0.5)
    assert rates.vote4_signed_recall == pytest.approx(1.0)
    assert rates.vote1_signed_recall == pytest.approx(1.0)
    outcome = classify_vote_class_confusion_outcome(rates)
    assert outcome["outcome"] == OUTCOME_VIABLE_CANDIDATE_PARITY


def test_viable_candidate_classification_on_perfect_signed_top_bin():
    records = (
        _record(fp_vote=+4, int_vote=+4),
        _record(fp_vote=-4, int_vote=-4),
        _record(fp_vote=+1, int_vote=+1),
        _record(fp_vote=-1, int_vote=-1),
    )
    rates = compute_vote_class_confusion(records)
    outcome = classify_vote_class_confusion_outcome(rates)
    assert outcome["outcome"] == OUTCOME_VIABLE_CANDIDATE_PARITY


@pytest.mark.skipif(
    not Path(DEFAULT_T2_CHECKPOINT_REL).is_file()
    and not (
        Path(__file__).resolve().parents[3] / DEFAULT_T2_CHECKPOINT_REL
    ).is_file(),
    reason="anchored T2 checkpoint absent",
)
def test_live_anchored_t2_vote_class_confusion():
    discovery = discover_t2_checkpoint()
    assert discovery.checkpoint_present
    payload = run_anchored_t2_vote_class_confusion(
        checkpoint_path=str(discovery.checkpoint_path),
        checkpoint_sha256=discovery.checkpoint_sha256,
    )
    assert payload["schema"] == VOTE_CLASS_CONFUSION_T2_SCHEMA_V2
    assert payload["pass_receipt"] is False
    assert "rescale_q16" in payload["shift_tables"]
    assert "rescale_q8" in payload["shift_tables"]
    s16 = payload["shift_tables"]["rescale_q16"]["micro"]
    s8 = payload["shift_tables"]["rescale_q8"]["micro"]
    assert s16["signed_vote_agreement_rate"] is not None
    assert s8["signed_vote_agreement_rate"] is not None
    assert s16["signed_gate"]["vote4_signed_recall"] is not None
    assert s16["abs_contrast"]["vote4_abs_recall"] is not None
    primary = payload["interpretation"]["primary_outcome"]
    assert primary["gates_on_signed_metrics_only"] is True
    assert primary["requires_both_class_signed_preservation"] is True
    s16_gate = s16["signed_gate"]
    assert s16_gate["vote1_signed_recall"] is not None
    assert s16_gate["vote1_signed_recall"] < 0.10
    # Integer assigns ~100% vote-4 on real-HRM T2 -> no int1 mass -> balanced metric null.
    assert s16_gate["class_balanced_insufficient_mass"] is True
    assert s16_gate["class_balanced_signed_agreement"] is None
    assert s16["class_balance"]["fp4_balance"] >= 0.90
    assert "int4_balance_delta" in s16["class_balance"]
    assert primary["outcome"] == OUTCOME_CLASS_IMBALANCE_CONFOUNDED
    assert primary["class_imbalance_confounded"] is True
