"""CPU reference integer sparse rank-vote emission for HRM-Text-1.58 Step 3C-B.

Replaces dense FP credit + dense int16 rank_bucketed_int16_votes with direct
SparseVoteEvents on the integer path. PRIMARY parity requires FP credit magnitudes
to remain rank-equivalent after 3C-A rescale and credit_q31 = -attribution_q31;
any fractional magnitude collision, including values >= 1 such as 2.7 vs 3.3
collapsing to the same rounded integer, is BR-3C-B-PARITY-FAIL.

Dense reference scratch is labeled cpu_reference_dense_int32_scratch and does NOT
clear optimizer_credit_state debt or authorize real_native_integer_credit_ranking_present.
"""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Callable, Sequence

import torch

from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    RankVoteSpec,
    _bisect_right_rank_positions_by_equal_value_group,
    _rank_bin_bounds,
    credit_from_weighted_grad,
    default_dry_run_rank_vote_spec,
    project_s1_gradient_to_moves,
    rank_bucketed_int16_votes,
    weighted_grad_from_captures,
)
from calm.hrm_text_158.native_full_stack.integer_marginal_attribution import (
    CPU_REFERENCE_DENSE_INT32_SCRATCH_LABEL,
    INT32_MAX,
    INT32_MIN,
    IntegerMarginalAttributionEvents,
    integer_marginal_attribution_from_captures,
    projected_moves_from_integer_attribution,
)
from calm.hrm_text_158.native_full_stack.sparse_vote_events import SparseVoteEvents

CREDIT_LAW_NEG_ATTRIBUTION_Q31_V0 = "credit_neg_attribution_q31_v0"
CREDIT_LAW_NEG_ATTRIBUTION_Q31_V1 = "credit_neg_attribution_q31_v1"
CREDIT_LAW_POW2_BUCKET_INTEGER_V0 = "credit_pow2_bucket_integer_v0"
INTEGER_SPARSE_RANK_PRODUCTION_CREDIT_LAW_ID = CREDIT_LAW_NEG_ATTRIBUTION_Q31_V1

BRANCH_3C_B_PARITY_PASS_CPU = "BR-3C-B-PARITY-PASS-CPU"
BRANCH_3C_B_PARITY_FAIL = "BR-3C-B-PARITY-FAIL"
BRANCH_3C_B_DENSE_LEAK = "BR-3C-B-DENSE-LEAK"
BRANCH_3C_B_MEASUREMENT_INVALID = "BR-3C-B-MEASUREMENT-INVALID"

INTEGER_SPARSE_RANK_VOTES_HARD_FALSE_FIELDS = (
    "ready_to_flip",
    "optimizer_credit_state_sub2_claim",
    "optimizer_credit_state_resolved",
    "readiness_row_flip_authorized",
    "real_native_integer_attribution_present",
    "real_native_integer_credit_ranking_present",
)


@dataclass(frozen=True)
class IntegerSparseRankVoteResult:
    events: SparseVoteEvents
    credit_law_id: str
    branch_id: str
    rank_positions_match: bool
    events_match: bool


def _validate_candidate_aligned(
    credit_q31: torch.Tensor,
    projected_moves: torch.Tensor,
    flat_indices: torch.Tensor,
) -> None:
    if credit_q31.dtype != torch.int32:
        raise ValueError(f"credit_q31 must be torch.int32, got {credit_q31.dtype}")
    if projected_moves.dtype != torch.int8:
        raise ValueError(f"projected_moves must be torch.int8, got {projected_moves.dtype}")
    if flat_indices.dtype != torch.int64:
        raise ValueError(f"flat_indices must be torch.int64, got {flat_indices.dtype}")
    if not (credit_q31.is_cpu and projected_moves.is_cpu and flat_indices.is_cpu):
        raise ValueError("credit_q31, projected_moves, and flat_indices must be CPU tensors")
    if not (
        int(credit_q31.numel()) == int(projected_moves.numel()) == int(flat_indices.numel())
    ):
        raise ValueError("credit_q31, projected_moves, and flat_indices length mismatch")
    if projected_moves.numel() > 0 and bool((projected_moves == 0).any().item()):
        raise ValueError("projected_moves must be nonzero on the integer sparse rank path")
    if flat_indices.numel() == 0:
        return
    if int(flat_indices.min().item()) < 0:
        raise ValueError("flat_indices must be non-negative")
    if flat_indices.numel() > 1:
        diffs = flat_indices[1:] - flat_indices[:-1]
        if not bool((diffs > 0).all().item()):
            raise ValueError("flat_indices must be strictly increasing and unique")


def _fail_closed_int32_cast(values: torch.Tensor) -> torch.Tensor:
    casted = values.to(torch.int64)
    if bool((casted < INT32_MIN).any().item()) or bool((casted > INT32_MAX).any().item()):
        raise ValueError("credit law produced values outside int32 range")
    return casted.to(torch.int32)


def credit_q31_from_attribution(
    attribution_q31: torch.Tensor,
    *,
    credit_law_id: str = INTEGER_SPARSE_RANK_PRODUCTION_CREDIT_LAW_ID,
) -> torch.Tensor:
    if credit_law_id in {
        CREDIT_LAW_NEG_ATTRIBUTION_Q31_V0,
        CREDIT_LAW_NEG_ATTRIBUTION_Q31_V1,
    }:
        if bool((attribution_q31 == INT32_MIN).any().item()):
            raise ValueError("attribution_q31 contains INT32_MIN; negation would overflow int32")
        return (-attribution_q31).to(torch.int32)
    if credit_law_id == CREDIT_LAW_POW2_BUCKET_INTEGER_V0:
        values = attribution_q31.to(torch.float32)
        out = torch.zeros_like(values)
        nonzero = values != 0
        if bool(nonzero.any().item()):
            abs_values = values[nonzero].abs()
            exponents = torch.log2(abs_values).round().clamp(-8.0, 8.0)
            out[nonzero] = values[nonzero].sign() * torch.pow(2.0, exponents)
        return _fail_closed_int32_cast(-out)
    raise ValueError(f"unsupported credit_law_id: {credit_law_id!r}")


def sparse_rank_bucketed_vote_events_from_integer_credit(
    credit_q31: torch.Tensor,
    projected_moves: torch.Tensor,
    flat_indices: torch.Tensor,
    spec: RankVoteSpec,
    *,
    credit_law_id: str = INTEGER_SPARSE_RANK_PRODUCTION_CREDIT_LAW_ID,
) -> SparseVoteEvents:
    spec.validate()
    _validate_candidate_aligned(credit_q31, projected_moves, flat_indices)
    if int(projected_moves.numel()) == 0:
        return SparseVoteEvents(
            indices=torch.empty(0, dtype=torch.int64),
            values=torch.empty(0, dtype=torch.int16),
        )

    # Rank contract: float32 abs magnitudes through grouped_bisect_right bit groups.
    abs_values = credit_q31.to(torch.float32).abs()
    if spec.rank_method != "grouped_bisect_right":
        raise ValueError("integer sparse rank path requires grouped_bisect_right")
    rank_positions = _bisect_right_rank_positions_by_equal_value_group(abs_values)
    candidate_count = int(projected_moves.numel())
    vote_abs = torch.zeros(candidate_count, dtype=torch.int16)
    matched = torch.zeros(candidate_count, dtype=torch.bool)
    for item in spec.rank_bins:
        lo_rank, hi_limit = _rank_bin_bounds(candidate_count, item)
        mask = (rank_positions >= lo_rank) & (rank_positions < hi_limit)
        vote_abs[mask] = int(item.vote_abs)
        matched |= mask
    if not bool(matched.all().item()):
        raise ValueError("rank-bucket vote mapping left unmatched candidates")
    votes = (projected_moves.to(torch.int16) * vote_abs).to(torch.int16)
    return SparseVoteEvents(indices=flat_indices.contiguous(), values=votes.contiguous())


def sparse_rank_votes_from_attribution_events(
    events: IntegerMarginalAttributionEvents,
    projected_move_indices: torch.Tensor,
    projected_moves: torch.Tensor,
    spec: RankVoteSpec,
    *,
    credit_law_id: str = INTEGER_SPARSE_RANK_PRODUCTION_CREDIT_LAW_ID,
) -> SparseVoteEvents:
    events.validate()
    if int(projected_move_indices.numel()) != int(projected_moves.numel()):
        raise ValueError("projected move indices/moves length mismatch")
    index_to_pos = {int(index): pos for pos, index in enumerate(events.flat_indices.tolist())}
    attribution_selected = torch.tensor(
        [int(events.attribution_q31[index_to_pos[int(index)]].item()) for index in projected_move_indices.tolist()],
        dtype=torch.int32,
    )
    credit_q31 = credit_q31_from_attribution(attribution_selected, credit_law_id=credit_law_id)
    return sparse_rank_bucketed_vote_events_from_integer_credit(
        credit_q31,
        projected_moves,
        projected_move_indices,
        spec,
        credit_law_id=credit_law_id,
    )


def _rank_positions_for_credit_values(credit_values: torch.Tensor, spec: RankVoteSpec) -> torch.Tensor:
    spec.validate()
    abs_values = credit_values.to(torch.float32).abs()
    if spec.rank_method != "grouped_bisect_right":
        raise ValueError("rank comparison requires grouped_bisect_right")
    return _bisect_right_rank_positions_by_equal_value_group(abs_values)


def classify_integer_sparse_rank_parity(
    integer_events: SparseVoteEvents,
    fp_reference_events: SparseVoteEvents,
    *,
    integer_rank_positions: torch.Tensor,
    fp_rank_positions: torch.Tensor,
) -> str:
    if integer_events.to_dict() != fp_reference_events.to_dict():
        return BRANCH_3C_B_PARITY_FAIL
    if not torch.equal(integer_rank_positions, fp_rank_positions):
        return BRANCH_3C_B_PARITY_FAIL
    return BRANCH_3C_B_PARITY_PASS_CPU


def compare_sparse_rank_to_fp_dense_reference(
    credit_q31: torch.Tensor,
    projected_moves: torch.Tensor,
    flat_indices: torch.Tensor,
    fp_credit_dense: torch.Tensor,
    fp_moves_dense: torch.Tensor,
    spec: RankVoteSpec,
    *,
    credit_law_id: str = INTEGER_SPARSE_RANK_PRODUCTION_CREDIT_LAW_ID,
) -> IntegerSparseRankVoteResult:
    spec.validate()
    _validate_candidate_aligned(credit_q31, projected_moves, flat_indices)
    integer_events = sparse_rank_bucketed_vote_events_from_integer_credit(
        credit_q31,
        projected_moves,
        flat_indices,
        spec,
        credit_law_id=credit_law_id,
    )
    fp_votes_dense = rank_bucketed_int16_votes(fp_credit_dense, fp_moves_dense, spec)
    fp_reference_events = SparseVoteEvents.from_dense_votes(fp_votes_dense)
    candidate_idx = torch.nonzero(fp_moves_dense.reshape(-1) != 0, as_tuple=False).flatten()
    fp_credit_sparse = fp_credit_dense.reshape(-1).index_select(0, candidate_idx)
    integer_rank = _rank_positions_for_credit_values(credit_q31.to(torch.float32), spec)
    fp_rank = _rank_positions_for_credit_values(fp_credit_sparse, spec)
    branch_id = classify_integer_sparse_rank_parity(
        integer_events,
        fp_reference_events,
        integer_rank_positions=integer_rank,
        fp_rank_positions=fp_rank,
    )
    return IntegerSparseRankVoteResult(
        events=integer_events,
        credit_law_id=credit_law_id,
        branch_id=branch_id,
        rank_positions_match=bool(torch.equal(integer_rank, fp_rank)),
        events_match=integer_events.to_dict() == fp_reference_events.to_dict(),
    )


def sparse_rank_votes_from_captures_reference(
    inputs: Sequence[torch.Tensor],
    grad_outputs: Sequence[torch.Tensor],
    *,
    weight_shape: Sequence[int],
    q_levels_flat: torch.Tensor,
    spec: RankVoteSpec | None = None,
    credit_law_id: str = INTEGER_SPARSE_RANK_PRODUCTION_CREDIT_LAW_ID,
) -> IntegerSparseRankVoteResult:
    if spec is None:
        spec = default_dry_run_rank_vote_spec()
    weight_dims = tuple(int(dim) for dim in weight_shape)
    attribution_events = integer_marginal_attribution_from_captures(
        inputs,
        grad_outputs,
        weight_shape=weight_dims,
    )
    move_indices, moves = projected_moves_from_integer_attribution(attribution_events, q_levels_flat)
    weighted_grad = weighted_grad_from_captures(inputs, grad_outputs, weight_shape=weight_dims)
    fp_credit = credit_from_weighted_grad(weighted_grad)
    fp_moves = project_s1_gradient_to_moves(weighted_grad, q_levels_flat.reshape(weight_dims))
    index_to_pos = {int(index): pos for pos, index in enumerate(attribution_events.flat_indices.tolist())}
    attribution_selected = torch.tensor(
        [int(attribution_events.attribution_q31[index_to_pos[int(index)]].item()) for index in move_indices.tolist()],
        dtype=torch.int32,
    )
    credit_q31 = credit_q31_from_attribution(attribution_selected, credit_law_id=credit_law_id)
    return compare_sparse_rank_to_fp_dense_reference(
        credit_q31,
        moves,
        move_indices,
        fp_credit,
        fp_moves,
        spec,
        credit_law_id=credit_law_id,
    )


def integer_sparse_rank_votes_hard_false_snapshot() -> dict[str, bool]:
    return {field: False for field in INTEGER_SPARSE_RANK_VOTES_HARD_FALSE_FIELDS}


def dense_scratch_is_reference_only_not_row_flip_evidence() -> bool:
    return True


# --- BR-3C-F strict-integer ranking subcontract ---

BR_F_RANKING_MEASUREMENT_INVALID = "BR-F-RANKING-MEASUREMENT-INVALID"
BR_F_RANKING_REPRESENTATION_LIMIT = "BR-F-RANKING-REPRESENTATION-LIMIT"
BR_F_RANKING_PARTIAL_COVERAGE = "BR-F-RANKING-PARTIAL-COVERAGE"
BR_F_RANKING_BIN_BOUNDARY_DIVERGENCE = "BR-F-RANKING-BIN-BOUNDARY-DIVERGENCE"
BR_F_RANKING_PRECISION_DIVERGENCE = "BR-F-RANKING-PRECISION-DIVERGENCE"
BR_F_RANKING_TIE_GROUP_DIVERGENCE = "BR-F-RANKING-TIE-GROUP-DIVERGENCE"
BR_F_RANKING_INTEGER_EXACT = "BR-F-RANKING-INTEGER-EXACT"

STRICT_INTEGER_RANK_BIN_DENOM_MAX_POWER = 20
_STANDARD_DECIMAL_REPR_RE = re.compile(r"^-?\d+(\.\d+)?$")

PRODUCTION_STRICT_INTEGER_CREDIT_LAW_IDS = frozenset(
    {
        CREDIT_LAW_NEG_ATTRIBUTION_Q31_V0,
        CREDIT_LAW_NEG_ATTRIBUTION_Q31_V1,
    }
)


@dataclass(frozen=True)
class CanonicalRankBoundary:
    numerator: int
    denominator: int


@dataclass(frozen=True)
class CanonicalRankVoteBin:
    lo: CanonicalRankBoundary
    hi: CanonicalRankBoundary
    vote_abs: int
    include_hi: bool


@dataclass(frozen=True)
class StrictIntegerRankingComparisonResult:
    branch_id: str
    strict_events: SparseVoteEvents
    float32_reference_events: SparseVoteEvents
    credit_law_id: str
    rank_method: str
    rank_bin_spec_canonical_tuple: tuple[tuple[int, int, int, int, int, bool], ...]
    rank_bin_spec_sha256: str
    candidate_count: int
    credit_q31_count: int
    projected_move_count: int
    flat_index_count: int
    emitted_event_count: int
    integer_vs_float_rank_mismatch_count: int
    vote_mismatch_count: int
    measurement_invalid_count: int
    representation_limit_count: int
    partial_coverage_count: int
    bin_boundary_divergence_count: int
    precision_divergence_count: int
    tie_group_divergence_count: int
    drop_in_float32_parity_pass: bool
    strict_integer_self_consistency_pass: bool


def _is_power_of_two(value: int) -> bool:
    return int(value) > 0 and (int(value) & (int(value) - 1)) == 0


def _reduce_coprime(numerator: int, denominator: int) -> tuple[int, int]:
    if denominator == 0:
        raise ValueError("rank boundary denominator must be non-zero")
    sign = -1 if numerator < 0 else 1
    p = abs(int(numerator))
    q = abs(int(denominator))
    divisor = math.gcd(p, q)
    p //= divisor
    q //= divisor
    if sign < 0:
        p = -p
    return p, q


def _decimal_repr_to_coprime(decimal_text: str) -> tuple[int, int]:
    if "e" in decimal_text.lower():
        raise ValueError("scientific-notation rank boundary repr is not allowed")
    if not _STANDARD_DECIMAL_REPR_RE.fullmatch(decimal_text):
        raise ValueError("rank boundary repr must be standard decimal notation")
    if "." in decimal_text:
        whole, frac = decimal_text.split(".", 1)
        frac_digits = len(frac)
        digit_body = whole + frac
    else:
        frac_digits = 0
        digit_body = decimal_text
    signed = int(digit_body)
    if decimal_text.startswith("-") and not digit_body.startswith("-"):
        signed = -signed
    return _reduce_coprime(signed, 10**frac_digits)


def canonicalize_rank_boundary_from_spec_float(value: float) -> CanonicalRankBoundary:
    if not math.isfinite(float(value)):
        raise ValueError("rank boundary must be finite")
    decimal_text = repr(float(value))
    numerator, denominator = _decimal_repr_to_coprime(decimal_text)
    if not _is_power_of_two(denominator):
        raise ValueError("rank boundary reduced denominator must be a power of two")
    if denominator > 2**STRICT_INTEGER_RANK_BIN_DENOM_MAX_POWER:
        raise ValueError("rank boundary denominator exceeds allowed power-of-two bound")
    return CanonicalRankBoundary(numerator=numerator, denominator=denominator)


def canonical_rank_vote_spec(spec: RankVoteSpec) -> tuple[CanonicalRankVoteBin, ...]:
    spec.validate()
    if spec.rank_method != "grouped_bisect_right":
        raise ValueError("strict integer rank path requires grouped_bisect_right")
    canonical_bins: list[CanonicalRankVoteBin] = []
    for item in spec.rank_bins:
        canonical_bins.append(
            CanonicalRankVoteBin(
                lo=canonicalize_rank_boundary_from_spec_float(float(item.lo_inclusive)),
                hi=canonicalize_rank_boundary_from_spec_float(float(item.hi_exclusive)),
                vote_abs=int(item.vote_abs),
                include_hi=bool(item.include_hi),
            )
        )
    return tuple(canonical_bins)


def canonical_rank_bin_spec_tuple(
    canonical_bins: tuple[CanonicalRankVoteBin, ...],
) -> tuple[tuple[int, int, int, int, int, bool], ...]:
    return tuple(
        (
            bin_spec.lo.numerator,
            bin_spec.lo.denominator,
            bin_spec.hi.numerator,
            bin_spec.hi.denominator,
            int(bin_spec.vote_abs),
            bool(bin_spec.include_hi),
        )
        for bin_spec in canonical_bins
    )


def canonical_rank_bin_spec_sha256(
    canonical_bins: tuple[CanonicalRankVoteBin, ...],
) -> str:
    payload = canonical_rank_bin_spec_tuple(canonical_bins)
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


def _rank_fraction_gte(rank_position: int, count: int, boundary: CanonicalRankBoundary) -> bool:
    return int(rank_position) * int(boundary.denominator) >= int(boundary.numerator) * int(count)


def _rank_fraction_gt(rank_position: int, count: int, boundary: CanonicalRankBoundary) -> bool:
    return int(rank_position) * int(boundary.denominator) > int(boundary.numerator) * int(count)


def _first_integer_rank_position_matching(
    count: int,
    predicate: Callable[[int, int], bool],
) -> int:
    lo = 1
    hi = count + 1
    while lo < hi:
        mid = (lo + hi) // 2
        if predicate(mid, count):
            hi = mid
        else:
            lo = mid + 1
    return lo


def integer_rank_bin_bounds(count: int, canonical_bin: CanonicalRankVoteBin) -> tuple[int, int]:
    lo_rank = _first_integer_rank_position_matching(
        count,
        lambda rank_position, candidate_count: _rank_fraction_gte(
            rank_position, candidate_count, canonical_bin.lo
        ),
    )
    if canonical_bin.include_hi:
        hi_limit = _first_integer_rank_position_matching(
            count,
            lambda rank_position, candidate_count: _rank_fraction_gt(
                rank_position, candidate_count, canonical_bin.hi
            ),
        )
    else:
        hi_limit = _first_integer_rank_position_matching(
            count,
            lambda rank_position, candidate_count: _rank_fraction_gte(
                rank_position, candidate_count, canonical_bin.hi
            ),
        )
    return lo_rank, hi_limit


def _fail_closed_direct_credit_q31(credit_q31: torch.Tensor) -> None:
    if bool((credit_q31 == INT32_MIN).any().item()):
        raise ValueError("credit_q31 contains INT32_MIN on strict integer rank path")


def integer_abs_magnitude_i64(credit_q31: torch.Tensor) -> torch.Tensor:
    _fail_closed_direct_credit_q31(credit_q31)
    return credit_q31.to(torch.int64).abs()


def grouped_bisect_right_rank_positions_integer_abs(abs_i64: torch.Tensor) -> torch.Tensor:
    sorted_values, order = torch.sort(abs_i64.contiguous())
    count = int(abs_i64.numel())
    if count == 0:
        return torch.empty(0, dtype=torch.int64)
    group_start = torch.ones(count, dtype=torch.bool)
    group_start[1:] = sorted_values[1:] != sorted_values[:-1]
    group_id = torch.cumsum(group_start.to(torch.int64), dim=0) - 1
    group_end = torch.ones(count, dtype=torch.bool)
    group_end[:-1] = sorted_values[:-1] != sorted_values[1:]
    group_end_ranks = (torch.nonzero(group_end, as_tuple=False).flatten() + 1).to(torch.int64)
    rank_positions_sorted = group_end_ranks[group_id]
    rank_positions = torch.empty_like(rank_positions_sorted)
    rank_positions[order] = rank_positions_sorted
    return rank_positions


def strict_integer_rank_positions_for_credit(
    credit_q31: torch.Tensor,
    canonical_bins: tuple[CanonicalRankVoteBin, ...],
) -> torch.Tensor:
    if len(canonical_bins) == 0:
        raise ValueError("canonical rank bins must be non-empty")
    abs_i64 = integer_abs_magnitude_i64(credit_q31)
    return grouped_bisect_right_rank_positions_integer_abs(abs_i64)


def _assign_bins_and_votes(
    rank_positions: torch.Tensor,
    projected_moves: torch.Tensor,
    *,
    count: int,
    canonical_bins: tuple[CanonicalRankVoteBin, ...] | None,
    float_bins: RankVoteSpec | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    vote_abs = torch.zeros(count, dtype=torch.int16)
    bin_ids = torch.full((count,), -1, dtype=torch.int64)
    matched = torch.zeros(count, dtype=torch.bool)
    if canonical_bins is not None:
        for bin_index, canonical_bin in enumerate(canonical_bins):
            lo_rank, hi_limit = integer_rank_bin_bounds(count, canonical_bin)
            mask = (rank_positions >= lo_rank) & (rank_positions < hi_limit)
            vote_abs[mask] = int(canonical_bin.vote_abs)
            bin_ids[mask] = int(bin_index)
            matched |= mask
    else:
        assert float_bins is not None
        for bin_index, item in enumerate(float_bins.rank_bins):
            lo_rank, hi_limit = _rank_bin_bounds(count, item)
            mask = (rank_positions >= lo_rank) & (rank_positions < hi_limit)
            vote_abs[mask] = int(item.vote_abs)
            bin_ids[mask] = int(bin_index)
            matched |= mask
    if not bool(matched.all().item()):
        raise ValueError("rank-bucket vote mapping left unmatched candidates")
    votes = (projected_moves.to(torch.int16) * vote_abs).to(torch.int16)
    return votes, bin_ids, matched


def strict_integer_sparse_rank_bucketed_vote_events_from_credit(
    credit_q31: torch.Tensor,
    projected_moves: torch.Tensor,
    flat_indices: torch.Tensor,
    canonical_bins: tuple[CanonicalRankVoteBin, ...],
    *,
    credit_law_id: str = INTEGER_SPARSE_RANK_PRODUCTION_CREDIT_LAW_ID,
) -> SparseVoteEvents:
    if credit_law_id not in PRODUCTION_STRICT_INTEGER_CREDIT_LAW_IDS:
        raise ValueError(f"strict integer rank path rejects credit_law_id: {credit_law_id!r}")
    _validate_candidate_aligned(credit_q31, projected_moves, flat_indices)
    if int(projected_moves.numel()) == 0:
        return SparseVoteEvents(
            indices=torch.empty(0, dtype=torch.int64),
            values=torch.empty(0, dtype=torch.int16),
        )
    rank_positions = strict_integer_rank_positions_for_credit(credit_q31, canonical_bins)
    votes, _bin_ids, _matched = _assign_bins_and_votes(
        rank_positions,
        projected_moves,
        count=int(projected_moves.numel()),
        canonical_bins=canonical_bins,
        float_bins=None,
    )
    return SparseVoteEvents(indices=flat_indices.contiguous(), values=votes.contiguous())


def float32_reference_sparse_rank_events_from_credit(
    credit_q31: torch.Tensor,
    projected_moves: torch.Tensor,
    flat_indices: torch.Tensor,
    spec: RankVoteSpec,
    *,
    credit_law_id: str = INTEGER_SPARSE_RANK_PRODUCTION_CREDIT_LAW_ID,
) -> SparseVoteEvents:
    return sparse_rank_bucketed_vote_events_from_integer_credit(
        credit_q31,
        projected_moves,
        flat_indices,
        spec,
        credit_law_id=credit_law_id,
    )


def _abs_order_equal(credit_q31: torch.Tensor) -> bool:
    abs_i64 = credit_q31.to(torch.int64).abs()
    abs_fp = credit_q31.to(torch.float32).abs()
    if int(abs_i64.numel()) <= 1:
        return True
    i64_order = torch.argsort(abs_i64, stable=True)
    fp_order = torch.argsort(abs_fp, stable=True)
    return bool(torch.equal(i64_order, fp_order))


def _precision_cause(credit_q31: torch.Tensor) -> bool:
    if int(credit_q31.numel()) <= 1:
        return False
    if not _abs_order_equal(credit_q31):
        return True
    abs_i64 = credit_q31.to(torch.int64).abs()
    abs_fp_bits = credit_q31.to(torch.float32).abs().contiguous().view(torch.int32)
    for left in range(int(credit_q31.numel())):
        for right in range(left + 1, int(credit_q31.numel())):
            if int(abs_i64[left].item()) != int(abs_i64[right].item()):
                if int(abs_fp_bits[left].item()) == int(abs_fp_bits[right].item()):
                    return True
    return False


def _branch_class_counts(branch_id: str) -> dict[str, int]:
    counts = {
        "measurement_invalid_count": 0,
        "representation_limit_count": 0,
        "partial_coverage_count": 0,
        "bin_boundary_divergence_count": 0,
        "precision_divergence_count": 0,
        "tie_group_divergence_count": 0,
    }
    mapping = {
        BR_F_RANKING_MEASUREMENT_INVALID: "measurement_invalid_count",
        BR_F_RANKING_REPRESENTATION_LIMIT: "representation_limit_count",
        BR_F_RANKING_PARTIAL_COVERAGE: "partial_coverage_count",
        BR_F_RANKING_BIN_BOUNDARY_DIVERGENCE: "bin_boundary_divergence_count",
        BR_F_RANKING_PRECISION_DIVERGENCE: "precision_divergence_count",
        BR_F_RANKING_TIE_GROUP_DIVERGENCE: "tie_group_divergence_count",
    }
    field = mapping.get(branch_id)
    if field is not None:
        counts[field] = 1
    return counts


def _dual_pass_booleans(
    branch_id: str,
    *,
    strict_self_consistent: bool,
    events_equal: bool,
    ranks_equal: bool,
) -> tuple[bool, bool]:
    if branch_id == BR_F_RANKING_INTEGER_EXACT:
        return bool(events_equal and ranks_equal), strict_self_consistent
    if branch_id in {
        BR_F_RANKING_BIN_BOUNDARY_DIVERGENCE,
        BR_F_RANKING_TIE_GROUP_DIVERGENCE,
        BR_F_RANKING_PRECISION_DIVERGENCE,
    }:
        return False, strict_self_consistent
    return False, False


def classify_strict_integer_ranking_branch(
    *,
    credit_q31: torch.Tensor,
    rank_int: torch.Tensor,
    rank_fp: torch.Tensor,
    bin_int: torch.Tensor,
    bin_fp: torch.Tensor,
    strict_events: SparseVoteEvents,
    float32_events: SparseVoteEvents,
    strict_self_consistent: bool,
) -> str:
    ranks_equal = bool(torch.equal(rank_int, rank_fp))
    bins_equal = bool(torch.equal(bin_int, bin_fp))
    events_equal = strict_events.to_dict() == float32_events.to_dict()
    if ranks_equal and bins_equal and not events_equal:
        return BR_F_RANKING_MEASUREMENT_INVALID
    if ranks_equal and bins_equal and events_equal:
        return BR_F_RANKING_INTEGER_EXACT
    if ranks_equal and not bins_equal:
        return BR_F_RANKING_BIN_BOUNDARY_DIVERGENCE
    if not ranks_equal:
        if _precision_cause(credit_q31):
            return BR_F_RANKING_PRECISION_DIVERGENCE
        return BR_F_RANKING_TIE_GROUP_DIVERGENCE
    return BR_F_RANKING_MEASUREMENT_INVALID


def compare_strict_integer_ranking_to_float32_reference(
    credit_q31: torch.Tensor,
    projected_moves: torch.Tensor,
    flat_indices: torch.Tensor,
    spec: RankVoteSpec,
    *,
    credit_law_id: str = INTEGER_SPARSE_RANK_PRODUCTION_CREDIT_LAW_ID,
) -> StrictIntegerRankingComparisonResult:
    if credit_law_id not in PRODUCTION_STRICT_INTEGER_CREDIT_LAW_IDS:
        raise ValueError(f"strict integer rank path rejects credit_law_id: {credit_law_id!r}")
    spec.validate()
    candidate_count = int(projected_moves.numel())
    canonical_bins = canonical_rank_vote_spec(spec)
    canonical_tuple = canonical_rank_bin_spec_tuple(canonical_bins)
    canonical_sha = canonical_rank_bin_spec_sha256(canonical_bins)
    if candidate_count == 0:
        empty = SparseVoteEvents(
            indices=torch.empty(0, dtype=torch.int64),
            values=torch.empty(0, dtype=torch.int16),
        )
        return StrictIntegerRankingComparisonResult(
            branch_id=BR_F_RANKING_INTEGER_EXACT,
            strict_events=empty,
            float32_reference_events=empty,
            credit_law_id=credit_law_id,
            rank_method=spec.rank_method,
            rank_bin_spec_canonical_tuple=canonical_tuple,
            rank_bin_spec_sha256=canonical_sha,
            candidate_count=0,
            credit_q31_count=0,
            projected_move_count=0,
            flat_index_count=0,
            emitted_event_count=0,
            integer_vs_float_rank_mismatch_count=0,
            vote_mismatch_count=0,
            measurement_invalid_count=0,
            representation_limit_count=0,
            partial_coverage_count=0,
            bin_boundary_divergence_count=0,
            precision_divergence_count=0,
            tie_group_divergence_count=0,
            drop_in_float32_parity_pass=True,
            strict_integer_self_consistency_pass=True,
        )
    try:
        _validate_candidate_aligned(credit_q31, projected_moves, flat_indices)
    except ValueError:
        empty = SparseVoteEvents(
            indices=torch.empty(0, dtype=torch.int64),
            values=torch.empty(0, dtype=torch.int16),
        )
        counts = _branch_class_counts(BR_F_RANKING_MEASUREMENT_INVALID)
        parity_pass, self_pass = _dual_pass_booleans(
            BR_F_RANKING_MEASUREMENT_INVALID,
            strict_self_consistent=False,
            events_equal=False,
            ranks_equal=False,
        )
        return StrictIntegerRankingComparisonResult(
            branch_id=BR_F_RANKING_MEASUREMENT_INVALID,
            strict_events=empty,
            float32_reference_events=empty,
            credit_law_id=credit_law_id,
            rank_method=spec.rank_method,
            rank_bin_spec_canonical_tuple=canonical_tuple,
            rank_bin_spec_sha256=canonical_sha,
            candidate_count=candidate_count,
            credit_q31_count=int(credit_q31.numel()),
            projected_move_count=candidate_count,
            flat_index_count=int(flat_indices.numel()),
            emitted_event_count=0,
            integer_vs_float_rank_mismatch_count=0,
            vote_mismatch_count=0,
            drop_in_float32_parity_pass=parity_pass,
            strict_integer_self_consistency_pass=self_pass,
            **counts,
        )
    if bool((credit_q31 == INT32_MIN).any().item()):
        empty = SparseVoteEvents(
            indices=torch.empty(0, dtype=torch.int64),
            values=torch.empty(0, dtype=torch.int16),
        )
        counts = _branch_class_counts(BR_F_RANKING_REPRESENTATION_LIMIT)
        parity_pass, self_pass = _dual_pass_booleans(
            BR_F_RANKING_REPRESENTATION_LIMIT,
            strict_self_consistent=False,
            events_equal=False,
            ranks_equal=False,
        )
        return StrictIntegerRankingComparisonResult(
            branch_id=BR_F_RANKING_REPRESENTATION_LIMIT,
            strict_events=empty,
            float32_reference_events=empty,
            credit_law_id=credit_law_id,
            rank_method=spec.rank_method,
            rank_bin_spec_canonical_tuple=canonical_tuple,
            rank_bin_spec_sha256=canonical_sha,
            candidate_count=candidate_count,
            credit_q31_count=int(credit_q31.numel()),
            projected_move_count=candidate_count,
            flat_index_count=int(flat_indices.numel()),
            emitted_event_count=0,
            integer_vs_float_rank_mismatch_count=0,
            vote_mismatch_count=0,
            drop_in_float32_parity_pass=parity_pass,
            strict_integer_self_consistency_pass=self_pass,
            **counts,
        )
    try:
        rank_int = strict_integer_rank_positions_for_credit(credit_q31, canonical_bins)
        strict_votes, bin_int, _matched = _assign_bins_and_votes(
            rank_int,
            projected_moves,
            count=candidate_count,
            canonical_bins=canonical_bins,
            float_bins=None,
        )
        strict_events = SparseVoteEvents(
            indices=flat_indices.contiguous(),
            values=strict_votes.contiguous(),
        )
        recomputed = strict_integer_sparse_rank_bucketed_vote_events_from_credit(
            credit_q31,
            projected_moves,
            flat_indices,
            canonical_bins,
            credit_law_id=credit_law_id,
        )
        strict_self_consistent = recomputed.to_dict() == strict_events.to_dict()
    except ValueError as exc:
        message = str(exc)
        branch_id = (
            BR_F_RANKING_PARTIAL_COVERAGE
            if "unmatched candidates" in message
            else BR_F_RANKING_MEASUREMENT_INVALID
        )
        empty = SparseVoteEvents(
            indices=torch.empty(0, dtype=torch.int64),
            values=torch.empty(0, dtype=torch.int16),
        )
        counts = _branch_class_counts(branch_id)
        parity_pass, self_pass = _dual_pass_booleans(
            branch_id,
            strict_self_consistent=False,
            events_equal=False,
            ranks_equal=False,
        )
        return StrictIntegerRankingComparisonResult(
            branch_id=branch_id,
            strict_events=empty,
            float32_reference_events=empty,
            credit_law_id=credit_law_id,
            rank_method=spec.rank_method,
            rank_bin_spec_canonical_tuple=canonical_tuple,
            rank_bin_spec_sha256=canonical_sha,
            candidate_count=candidate_count,
            credit_q31_count=int(credit_q31.numel()),
            projected_move_count=candidate_count,
            flat_index_count=int(flat_indices.numel()),
            emitted_event_count=0,
            integer_vs_float_rank_mismatch_count=0,
            vote_mismatch_count=0,
            drop_in_float32_parity_pass=parity_pass,
            strict_integer_self_consistency_pass=self_pass,
            **counts,
        )
    abs_fp = credit_q31.to(torch.float32).abs()
    rank_fp = _bisect_right_rank_positions_by_equal_value_group(abs_fp)
    _fp_votes, bin_fp, _fp_matched = _assign_bins_and_votes(
        rank_fp,
        projected_moves,
        count=candidate_count,
        canonical_bins=None,
        float_bins=spec,
    )
    float32_events = float32_reference_sparse_rank_events_from_credit(
        credit_q31,
        projected_moves,
        flat_indices,
        spec,
        credit_law_id=credit_law_id,
    )
    branch_id = classify_strict_integer_ranking_branch(
        credit_q31=credit_q31,
        rank_int=rank_int,
        rank_fp=rank_fp,
        bin_int=bin_int,
        bin_fp=bin_fp,
        strict_events=strict_events,
        float32_events=float32_events,
        strict_self_consistent=strict_self_consistent,
    )
    rank_mismatch_count = int((rank_int != rank_fp).sum().item())
    events_equal = strict_events.to_dict() == float32_events.to_dict()
    vote_mismatch_count = 0 if events_equal else 1
    counts = _branch_class_counts(branch_id)
    parity_pass, self_pass = _dual_pass_booleans(
        branch_id,
        strict_self_consistent=strict_self_consistent,
        events_equal=events_equal,
        ranks_equal=bool(torch.equal(rank_int, rank_fp)),
    )
    return StrictIntegerRankingComparisonResult(
        branch_id=branch_id,
        strict_events=strict_events,
        float32_reference_events=float32_events,
        credit_law_id=credit_law_id,
        rank_method=spec.rank_method,
        rank_bin_spec_canonical_tuple=canonical_tuple,
        rank_bin_spec_sha256=canonical_sha,
        candidate_count=candidate_count,
        credit_q31_count=int(credit_q31.numel()),
        projected_move_count=candidate_count,
        flat_index_count=int(flat_indices.numel()),
        emitted_event_count=strict_events.event_count(),
        integer_vs_float_rank_mismatch_count=rank_mismatch_count,
        vote_mismatch_count=1 if vote_mismatch_count else 0,
        drop_in_float32_parity_pass=parity_pass,
        strict_integer_self_consistency_pass=self_pass,
        **counts,
    )
