"""Read-only vote-class confusion diagnostic for HRM-Text-1.58 Step 3C-A T2.

Decomposes aggregate events_match into abs-class contrast and signed top-bin
gate metrics on anchored real-HRM captures. Parity evidence only.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    RankVoteSpec,
    default_dry_run_rank_vote_spec,
)
from calm.hrm_text_158.native_full_stack.integer_marginal_attribution import (
    INTEGER_MARGINAL_ATTRIBUTION_PRODUCTION_LAW_ID,
)
from calm.hrm_text_158.native_full_stack.realistic_gradient_parity_probe import (
    PerCandidateParityRecord,
    Tier2RawCaptureBundle,
    capture_tier2_checkpoint_raw_captures,
    realistic_gradient_parity_probe_hard_false_snapshot,
    sha256_file,
)
from calm.hrm_text_158.native_full_stack.rescale_law_readonly_sweep import (
    RescaleSaturationError,
    build_full_parity_records_with_shift,
    measure_shift_key_at_capture,
)
from calm.hrm_text_158.native_full_stack.t2_fp_vs_s24_disambiguation import (
    FROZEN_T2_ANCHOR_BATCH_SIZE,
    FROZEN_T2_ANCHOR_CHECKPOINT_SHA256,
    FROZEN_T2_ANCHOR_CURRICULUM_SEED,
    FROZEN_T2_ANCHOR_KEY_SET_SHA256,
    T2_DISAMBIGUATION_CAPTURE_SEAM_ID,
    anchor_key_set_sha256,
)

VOTE_CLASS_CONFUSION_T2_SCHEMA_V2 = "hrm_text_158_vote_class_confusion_t2/v2"
VOTE_CLASS_CONFUSION_T2_TARGET_NAME = "vote_class_confusion_t2"

OUTCOME_VIABLE_CANDIDATE_PARITY = "VIABLE_CANDIDATE_PARITY"
OUTCOME_SIGN_FLIP_MASKED = "SIGN_FLIP_MASKED"
OUTCOME_BULK_MASKED_FAILURE = "BULK_MASKED_FAILURE"
OUTCOME_CLASS_IMBALANCE_CONFOUNDED = "CLASS_IMBALANCE_CONFOUNDED"
OUTCOME_MIXED = "MIXED"

SIGNED_HIGH_THRESHOLD = 0.90
SIGNED_LOW_THRESHOLD = 0.80
SIGN_FLIP_DELTA_THRESHOLD = 0.05
TOP4_SIGN_MISMATCH_HIGH_THRESHOLD = 0.05
VOTE1_AGREEMENT_HIGH_THRESHOLD = 0.90
CLASS_DOMINANCE_THRESHOLD = 0.90
MINORITY_SIGNED_NEAR_ZERO_THRESHOLD = 0.10

VOTE_CLASS_CONFUSION_NON_CLAIMS = (
    "vote-class confusion is CPU parity decomposition only; not optimizer viability",
    "signed top-bin metrics are parity evidence; not training-quality proof",
    "abs-class metrics are diagnostic contrast; interpretation gates on signed only",
    "pass_receipt is always false; no production re-pin or 3C-GPU claim",
    "update-effect parity explicitly deferred from this diagnostic",
)


def vote_abs_from_signed(signed_vote: int) -> int:
    return abs(int(signed_vote))


def _safe_rate(numerator: int, denominator: int) -> float | None:
    if int(denominator) == 0:
        return None
    return float(numerator) / float(denominator)


@dataclass(frozen=True)
class VoteClassConfusionCounts:
    n: int
    fp4_count: int
    fp1_count: int
    int4_count: int
    int1_count: int
    abs_hit4_count: int
    abs_hit1_count: int
    signed_hit4_count: int
    signed_hit1_count: int
    top4_sign_mismatch_count: int
    false4_count: int
    miss4_count: int
    signed_agree_count: int
    insufficient_class_mass: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": int(self.n),
            "fp4_count": int(self.fp4_count),
            "fp1_count": int(self.fp1_count),
            "int4_count": int(self.int4_count),
            "int1_count": int(self.int1_count),
            "abs_hit4_count": int(self.abs_hit4_count),
            "abs_hit1_count": int(self.abs_hit1_count),
            "signed_hit4_count": int(self.signed_hit4_count),
            "signed_hit1_count": int(self.signed_hit1_count),
            "top4_sign_mismatch_count": int(self.top4_sign_mismatch_count),
            "false4_count": int(self.false4_count),
            "miss4_count": int(self.miss4_count),
            "signed_agree_count": int(self.signed_agree_count),
            "insufficient_class_mass": bool(self.insufficient_class_mass),
        }


@dataclass(frozen=True)
class VoteClassConfusionRates:
    counts: VoteClassConfusionCounts
    vote4_signed_recall: float | None
    vote4_signed_precision: float | None
    top4_sign_mismatch_rate: float | None
    vote4_abs_recall: float | None
    vote4_abs_precision: float | None
    false4_rate: float | None
    missed4_rate: float | None
    vote1_agreement_rate: float | None
    signed_vote_agreement_rate: float | None
    fp4_balance: float | None
    fp1_balance: float | None
    int4_balance: float | None
    int1_balance: float | None
    vote4_abs_minus_signed_recall: float | None
    vote1_signed_recall: float | None
    vote1_signed_precision: float | None
    int4_balance_delta: float | None
    int1_balance_delta: float | None
    class_balanced_signed_agreement: float | None
    class_balanced_insufficient_mass: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "counts": self.counts.to_dict(),
            "signed_gate": {
                "vote4_signed_recall": self.vote4_signed_recall,
                "vote4_signed_precision": self.vote4_signed_precision,
                "vote1_signed_recall": self.vote1_signed_recall,
                "vote1_signed_precision": self.vote1_signed_precision,
                "signed_hit1_count": self.counts.signed_hit1_count,
                "top4_sign_mismatch_rate": self.top4_sign_mismatch_rate,
                "class_balanced_signed_agreement": self.class_balanced_signed_agreement,
                "class_balanced_insufficient_mass": self.class_balanced_insufficient_mass,
            },
            "abs_contrast": {
                "vote4_abs_recall": self.vote4_abs_recall,
                "vote4_abs_precision": self.vote4_abs_precision,
                "false4_rate": self.false4_rate,
                "missed4_rate": self.missed4_rate,
                "vote1_agreement_rate": self.vote1_agreement_rate,
                "vote4_abs_minus_signed_recall": self.vote4_abs_minus_signed_recall,
            },
            "signed_vote_agreement_rate": self.signed_vote_agreement_rate,
            "class_balance": {
                "fp4_balance": self.fp4_balance,
                "fp1_balance": self.fp1_balance,
                "int4_balance": self.int4_balance,
                "int1_balance": self.int1_balance,
                "int4_balance_delta": self.int4_balance_delta,
                "int1_balance_delta": self.int1_balance_delta,
            },
        }


def _counts_from_records(records: Sequence[PerCandidateParityRecord]) -> VoteClassConfusionCounts:
    fp4 = fp1 = int4 = int1 = 0
    abs_hit4 = abs_hit1 = signed_hit4 = signed_hit1 = top4_sign_mismatch = 0
    false4 = miss4 = signed_agree = 0
    for record in records:
        fp_vote = int(record.fp_vote)
        int_vote = int(record.int_vote)
        fp_is4 = vote_abs_from_signed(fp_vote) == 4
        fp_is1 = vote_abs_from_signed(fp_vote) == 1
        int_is4 = vote_abs_from_signed(int_vote) == 4
        int_is1 = vote_abs_from_signed(int_vote) == 1
        if fp_is4:
            fp4 += 1
        if fp_is1:
            fp1 += 1
        if int_is4:
            int4 += 1
        if int_is1:
            int1 += 1
        if fp_is4 and int_is4:
            abs_hit4 += 1
        if fp_is1 and int_is1:
            abs_hit1 += 1
        if fp_is4 and int_vote == fp_vote:
            signed_hit4 += 1
        if fp_is1 and int_vote == fp_vote:
            signed_hit1 += 1
        if fp_is4 and int_is4 and int_vote != fp_vote:
            top4_sign_mismatch += 1
        if int_is4 and fp_is1:
            false4 += 1
        if int_is1 and fp_is4:
            miss4 += 1
        if int_vote == fp_vote:
            signed_agree += 1
    n = len(records)
    insufficient = fp4 == 0 or int4 == 0 or fp1 == 0 or int1 == 0
    return VoteClassConfusionCounts(
        n=n,
        fp4_count=fp4,
        fp1_count=fp1,
        int4_count=int4,
        int1_count=int1,
        abs_hit4_count=abs_hit4,
        abs_hit1_count=abs_hit1,
        signed_hit4_count=signed_hit4,
        signed_hit1_count=signed_hit1,
        top4_sign_mismatch_count=top4_sign_mismatch,
        false4_count=false4,
        miss4_count=miss4,
        signed_agree_count=signed_agree,
        insufficient_class_mass=insufficient,
    )


def compute_vote_class_confusion(
    records: Sequence[PerCandidateParityRecord],
) -> VoteClassConfusionRates:
    counts = _counts_from_records(records)
    vote4_signed_recall = _safe_rate(counts.signed_hit4_count, counts.fp4_count)
    vote4_signed_precision = _safe_rate(counts.signed_hit4_count, counts.int4_count)
    top4_sign_mismatch_rate = _safe_rate(counts.top4_sign_mismatch_count, counts.fp4_count)
    vote4_abs_recall = _safe_rate(counts.abs_hit4_count, counts.fp4_count)
    vote4_abs_precision = _safe_rate(counts.abs_hit4_count, counts.int4_count)
    false4_rate = _safe_rate(counts.false4_count, counts.int4_count)
    missed4_rate = _safe_rate(counts.miss4_count, counts.fp4_count)
    vote1_agreement_rate = _safe_rate(counts.abs_hit1_count, counts.fp1_count)
    signed_vote_agreement_rate = _safe_rate(counts.signed_agree_count, counts.n)
    fp4_balance = _safe_rate(counts.fp4_count, counts.n)
    fp1_balance = _safe_rate(counts.fp1_count, counts.n)
    int4_balance = _safe_rate(counts.int4_count, counts.n)
    int1_balance = _safe_rate(counts.int1_count, counts.n)
    abs_minus_signed = None
    if vote4_abs_recall is not None and vote4_signed_recall is not None:
        abs_minus_signed = float(vote4_abs_recall) - float(vote4_signed_recall)
    vote1_signed_recall = (
        None if counts.fp1_count == 0 else _safe_rate(counts.signed_hit1_count, counts.fp1_count)
    )
    vote1_signed_precision = (
        None if counts.int1_count == 0 else _safe_rate(counts.signed_hit1_count, counts.int1_count)
    )
    int4_balance_delta = None
    int1_balance_delta = None
    if fp4_balance is not None and int4_balance is not None:
        int4_balance_delta = float(int4_balance) - float(fp4_balance)
    if fp1_balance is not None and int1_balance is not None:
        int1_balance_delta = float(int1_balance) - float(fp1_balance)
    class_balanced_insufficient = counts.fp1_count == 0 or counts.int1_count == 0
    class_balanced_signed_agreement = None
    if (
        not class_balanced_insufficient
        and vote4_signed_recall is not None
        and vote1_signed_recall is not None
    ):
        class_balanced_signed_agreement = (
            float(vote4_signed_recall) + float(vote1_signed_recall)
        ) / 2.0
    return VoteClassConfusionRates(
        counts=counts,
        vote4_signed_recall=vote4_signed_recall,
        vote4_signed_precision=vote4_signed_precision,
        top4_sign_mismatch_rate=top4_sign_mismatch_rate,
        vote4_abs_recall=vote4_abs_recall,
        vote4_abs_precision=vote4_abs_precision,
        false4_rate=false4_rate,
        missed4_rate=missed4_rate,
        vote1_agreement_rate=vote1_agreement_rate,
        signed_vote_agreement_rate=signed_vote_agreement_rate,
        fp4_balance=fp4_balance,
        fp1_balance=fp1_balance,
        int4_balance=int4_balance,
        int1_balance=int1_balance,
        vote4_abs_minus_signed_recall=abs_minus_signed,
        vote1_signed_recall=vote1_signed_recall,
        vote1_signed_precision=vote1_signed_precision,
        int4_balance_delta=int4_balance_delta,
        int1_balance_delta=int1_balance_delta,
        class_balanced_signed_agreement=class_balanced_signed_agreement,
        class_balanced_insufficient_mass=class_balanced_insufficient,
    )


def _mean_optional(values: Sequence[float | None]) -> float | None:
    present = [float(item) for item in values if item is not None]
    if not present:
        return None
    return float(sum(present) / len(present))


@dataclass(frozen=True)
class TierVoteClassConfusionResult:
    rescale_shift: int
    candidate_id: str
    valid_key_count: int
    total_move_candidates: int
    micro: VoteClassConfusionRates
    macro: dict[str, float | None]
    per_key: tuple[dict[str, Any], ...]
    events_match_rate_from_records: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rescale_shift": int(self.rescale_shift),
            "candidate_id": str(self.candidate_id),
            "valid_key_count": int(self.valid_key_count),
            "total_move_candidates": int(self.total_move_candidates),
            "micro": self.micro.to_dict(),
            "macro": dict(self.macro),
            "per_key": list(self.per_key),
            "events_match_rate_from_records": self.events_match_rate_from_records,
        }


def aggregate_tier_vote_class_confusion(
    *,
    rescale_shift: int,
    candidate_id: str,
    per_key_records: Mapping[str, Sequence[PerCandidateParityRecord]],
    valid_keys: Sequence[str],
) -> TierVoteClassConfusionResult:
    pooled: list[PerCandidateParityRecord] = []
    per_key_payload: list[dict[str, Any]] = []
    per_key_signed_recall: list[float | None] = []
    per_key_signed_precision: list[float | None] = []
    per_key_abs_recall: list[float | None] = []
    per_key_signed_agree: list[float | None] = []
    total_moves = 0
    for key in sorted(valid_keys):
        records = tuple(per_key_records[key])
        total_moves += len(records)
        pooled.extend(records)
        key_rates = compute_vote_class_confusion(records)
        per_key_signed_recall.append(key_rates.vote4_signed_recall)
        per_key_signed_precision.append(key_rates.vote4_signed_precision)
        per_key_abs_recall.append(key_rates.vote4_abs_recall)
        per_key_signed_agree.append(key_rates.signed_vote_agreement_rate)
        per_key_payload.append(
            {
                "state_key": key,
                "move_candidate_count": len(records),
                "rates": key_rates.to_dict(),
            }
        )
    micro = compute_vote_class_confusion(pooled)
    events_match = micro.signed_vote_agreement_rate
    macro = {
        "vote4_signed_recall": _mean_optional(per_key_signed_recall),
        "vote4_signed_precision": _mean_optional(per_key_signed_precision),
        "vote4_abs_recall": _mean_optional(per_key_abs_recall),
        "signed_vote_agreement_rate": _mean_optional(per_key_signed_agree),
    }
    return TierVoteClassConfusionResult(
        rescale_shift=int(rescale_shift),
        candidate_id=str(candidate_id),
        valid_key_count=len(valid_keys),
        total_move_candidates=int(total_moves),
        micro=micro,
        macro=macro,
        per_key=tuple(per_key_payload),
        events_match_rate_from_records=events_match,
    )


def classify_vote_class_confusion_outcome(
    rates: VoteClassConfusionRates,
) -> dict[str, Any]:
    signed_recall = rates.vote4_signed_recall
    signed_precision = rates.vote4_signed_precision
    vote1_signed_recall = rates.vote1_signed_recall
    vote1_signed_precision = rates.vote1_signed_precision
    abs_recall = rates.vote4_abs_recall
    mismatch_rate = rates.top4_sign_mismatch_rate
    vote1_agreement = rates.vote1_agreement_rate
    abs_minus_signed = rates.vote4_abs_minus_signed_recall
    fp4_balance = rates.fp4_balance
    fp1_balance = rates.fp1_balance

    vote4_signed_high = (
        signed_recall is not None
        and signed_precision is not None
        and signed_recall >= SIGNED_HIGH_THRESHOLD
        and signed_precision >= SIGNED_HIGH_THRESHOLD
        and (mismatch_rate is None or mismatch_rate <= TOP4_SIGN_MISMATCH_HIGH_THRESHOLD)
    )
    vote1_signed_high = (
        vote1_signed_recall is not None
        and vote1_signed_precision is not None
        and vote1_signed_recall >= SIGNED_HIGH_THRESHOLD
        and vote1_signed_precision >= SIGNED_HIGH_THRESHOLD
    )
    signed_high = vote4_signed_high and vote1_signed_high

    fp4_dominant = fp4_balance is not None and fp4_balance >= CLASS_DOMINANCE_THRESHOLD
    fp1_dominant = fp1_balance is not None and fp1_balance >= CLASS_DOMINANCE_THRESHOLD
    vote1_minority_near_zero = (
        vote1_signed_recall is not None
        and vote1_signed_recall < MINORITY_SIGNED_NEAR_ZERO_THRESHOLD
    )
    vote4_minority_near_zero = (
        signed_recall is not None
        and signed_recall < MINORITY_SIGNED_NEAR_ZERO_THRESHOLD
    )
    class_imbalance_confounded = (
        (fp4_dominant and vote1_minority_near_zero)
        or (fp1_dominant and vote4_minority_near_zero)
    )

    sign_flip_masked = False
    if abs_recall is not None and signed_recall is not None:
        delta = abs_minus_signed if abs_minus_signed is not None else (abs_recall - signed_recall)
        sign_flip_masked = (
            abs_recall >= SIGNED_HIGH_THRESHOLD
            and signed_recall < SIGNED_HIGH_THRESHOLD
            and delta >= SIGN_FLIP_DELTA_THRESHOLD
        ) or (
            mismatch_rate is not None
            and mismatch_rate > TOP4_SIGN_MISMATCH_HIGH_THRESHOLD
            and abs_recall is not None
            and abs_recall >= SIGNED_HIGH_THRESHOLD
        )
    bulk_masked = (
        not class_imbalance_confounded
        and signed_recall is not None
        and signed_precision is not None
        and signed_recall < SIGNED_LOW_THRESHOLD
        and signed_precision < SIGNED_LOW_THRESHOLD
        and vote1_agreement is not None
        and vote1_agreement >= VOTE1_AGREEMENT_HIGH_THRESHOLD
    )
    if signed_high:
        outcome = OUTCOME_VIABLE_CANDIDATE_PARITY
    elif class_imbalance_confounded:
        outcome = OUTCOME_CLASS_IMBALANCE_CONFOUNDED
    elif sign_flip_masked:
        outcome = OUTCOME_SIGN_FLIP_MASKED
    elif bulk_masked:
        outcome = OUTCOME_BULK_MASKED_FAILURE
    else:
        outcome = OUTCOME_MIXED
    return {
        "outcome": outcome,
        "gates_on_signed_metrics_only": True,
        "requires_both_class_signed_preservation": True,
        "vote4_signed_recall": signed_recall,
        "vote4_signed_precision": signed_precision,
        "vote1_signed_recall": vote1_signed_recall,
        "vote1_signed_precision": vote1_signed_precision,
        "class_balanced_signed_agreement": rates.class_balanced_signed_agreement,
        "top4_sign_mismatch_rate": mismatch_rate,
        "abs_recall_contrast": abs_recall,
        "abs_minus_signed_recall": abs_minus_signed,
        "vote1_agreement_contrast": vote1_agreement,
        "fp4_balance": fp4_balance,
        "fp1_balance": fp1_balance,
        "int4_balance_delta": rates.int4_balance_delta,
        "int1_balance_delta": rates.int1_balance_delta,
        "class_imbalance_confounded": class_imbalance_confounded,
    }


def _collect_shift_confusion(
    bundle: Tier2RawCaptureBundle,
    *,
    rescale_shift: int,
    candidate_id: str,
    spec: RankVoteSpec,
) -> TierVoteClassConfusionResult:
    per_key_records: dict[str, list[PerCandidateParityRecord]] = {}
    valid_keys: list[str] = []
    for key in sorted(bundle.per_key_captures.keys()):
        capture = bundle.per_key_captures[key]
        key_result = measure_shift_key_at_capture(
            state_key=key,
            inputs=capture.inputs,
            grad_outputs=capture.grad_outputs,
            weight_shape=capture.weight_shape,
            q_levels_flat=capture.q_levels_flat,
            spec=spec,
            rescale_shift=int(rescale_shift),
        )
        if key_result.saturation_failed:
            continue
        if not key_result.measurement_valid:
            continue
        try:
            records = build_full_parity_records_with_shift(
                inputs=capture.inputs,
                grad_outputs=capture.grad_outputs,
                weight_shape=capture.weight_shape,
                q_levels_flat=capture.q_levels_flat,
                spec=spec,
                rescale_shift=int(rescale_shift),
            )
        except RescaleSaturationError:
            continue
        per_key_records[key] = records
        valid_keys.append(key)
    return aggregate_tier_vote_class_confusion(
        rescale_shift=int(rescale_shift),
        candidate_id=str(candidate_id),
        per_key_records=per_key_records,
        valid_keys=valid_keys,
    )


def run_anchored_t2_vote_class_confusion(
    *,
    checkpoint_path: str | Path,
    checkpoint_sha256: str | None = None,
    curriculum_seed: int = FROZEN_T2_ANCHOR_CURRICULUM_SEED,
    batch_size: int = FROZEN_T2_ANCHOR_BATCH_SIZE,
    rank_spec: RankVoteSpec | None = None,
) -> dict[str, Any]:
    path = Path(checkpoint_path)
    blob_sha = checkpoint_sha256 or sha256_file(path)
    bundle = capture_tier2_checkpoint_raw_captures(
        checkpoint_path=str(path),
        checkpoint_sha256=str(blob_sha),
        curriculum_seed=int(curriculum_seed),
        batch_size=int(batch_size),
    )
    key_set_sha = anchor_key_set_sha256(sorted(bundle.per_key_captures.keys()))
    spec = rank_spec or default_dry_run_rank_vote_spec()
    shifts = (
        ("rescale_q16", 16),
        ("rescale_q8", 8),
    )
    shift_tables: dict[str, Any] = {}
    interpretations: dict[str, Any] = {}
    for candidate_id, shift in shifts:
        tier = _collect_shift_confusion(
            bundle,
            rescale_shift=int(shift),
            candidate_id=str(candidate_id),
            spec=spec,
        )
        shift_tables[candidate_id] = tier.to_dict()
        interpretations[candidate_id] = classify_vote_class_confusion_outcome(tier.micro)
    primary = interpretations["rescale_q16"]
    return {
        "schema": VOTE_CLASS_CONFUSION_T2_SCHEMA_V2,
        "target_name": VOTE_CLASS_CONFUSION_T2_TARGET_NAME,
        "pass_receipt": False,
        "hard_false": realistic_gradient_parity_probe_hard_false_snapshot(),
        "non_claims": list(VOTE_CLASS_CONFUSION_NON_CLAIMS),
        "anchor": {
            "checkpoint_path": str(path),
            "checkpoint_sha256": str(blob_sha),
            "curriculum_seed": int(curriculum_seed),
            "batch_size": int(batch_size),
            "key_count": len(bundle.per_key_captures),
            "key_set_sha256": str(key_set_sha),
            "frozen_checkpoint_sha256": FROZEN_T2_ANCHOR_CHECKPOINT_SHA256,
            "frozen_key_set_sha256": FROZEN_T2_ANCHOR_KEY_SET_SHA256,
            "capture_seam_id": T2_DISAMBIGUATION_CAPTURE_SEAM_ID,
        },
        "production_law_id": INTEGER_MARGINAL_ATTRIBUTION_PRODUCTION_LAW_ID,
        "shift_tables": shift_tables,
        "interpretation": {
            "primary_shift": 16,
            "primary_candidate_id": "rescale_q16",
            "primary_outcome": primary,
            "per_shift": interpretations,
            "classification_basis": "signed_top_bin_metrics_only",
        },
        "bundle_provenance": dict(bundle.provenance),
    }
