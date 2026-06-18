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

from dataclasses import dataclass
from typing import Sequence

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
CREDIT_LAW_POW2_BUCKET_INTEGER_V0 = "credit_pow2_bucket_integer_v0"

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
    credit_law_id: str = CREDIT_LAW_NEG_ATTRIBUTION_Q31_V0,
) -> torch.Tensor:
    if credit_law_id == CREDIT_LAW_NEG_ATTRIBUTION_Q31_V0:
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
    credit_law_id: str = CREDIT_LAW_NEG_ATTRIBUTION_Q31_V0,
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
    credit_law_id: str = CREDIT_LAW_NEG_ATTRIBUTION_Q31_V0,
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
    credit_law_id: str = CREDIT_LAW_NEG_ATTRIBUTION_Q31_V0,
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
    credit_law_id: str = CREDIT_LAW_NEG_ATTRIBUTION_Q31_V0,
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
