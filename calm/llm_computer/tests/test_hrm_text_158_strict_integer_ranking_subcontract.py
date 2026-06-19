"""CPU tests for BR-3C-F strict-integer ranking subcontract."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from typing import Iterator
from unittest import mock

import pytest
import torch

from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    RankVoteSpec,
    RankVoteBin,
    _rank_bin_bounds,
    default_dry_run_rank_vote_spec,
)
from calm.hrm_text_158.native_full_stack.integer_marginal_attribution import INT32_MIN
from calm.hrm_text_158.native_full_stack.integer_native_optimizer_credit_path_design import (
    FORBIDDEN_RANKING_SUBCONTRACT_FIELDS,
    RANKING_SUBCONTRACT_MODE_STRICT_INTEGER,
    RANKING_SUBCONTRACT_NON_CLAIMS,
    build_ranking_subcontract_receipt,
    prove_strict_integer_ranking_subcontract,
    ranking_subcontract_hard_false_snapshot,
    validate_ranking_subcontract_receipt,
)
from calm.hrm_text_158.native_full_stack.integer_sparse_rank_votes import (
    BR_F_RANKING_BIN_BOUNDARY_DIVERGENCE,
    BR_F_RANKING_INTEGER_EXACT,
    BR_F_RANKING_MEASUREMENT_INVALID,
    BR_F_RANKING_PRECISION_DIVERGENCE,
    BR_F_RANKING_REPRESENTATION_LIMIT,
    BR_F_RANKING_TIE_GROUP_DIVERGENCE,
    CREDIT_LAW_NEG_ATTRIBUTION_Q31_V0,
    CanonicalRankVoteBin,
    CanonicalRankBoundary,
    StrictIntegerRankingComparisonResult,
    canonical_rank_vote_spec,
    canonicalize_rank_boundary_from_spec_float,
    classify_strict_integer_ranking_branch,
    compare_strict_integer_ranking_to_float32_reference,
    grouped_bisect_right_rank_positions_integer_abs,
    integer_rank_bin_bounds,
    strict_integer_sparse_rank_bucketed_vote_events_from_credit,
)
from calm.hrm_text_158.native_full_stack.optimizer_credit_state import (
    OPTIMIZER_CREDIT_STATE_FAIL_CLOSED_NON_CLAIMS,
)
from calm.hrm_text_158.native_full_stack.sparse_vote_events import SparseVoteEvents


def _simple_spec() -> RankVoteSpec:
    return default_dry_run_rank_vote_spec()


def _green_credit_fixture() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    credit_q31 = torch.tensor([4, 2, 6], dtype=torch.int32)
    projected_moves = torch.tensor([1, -1, 1], dtype=torch.int8)
    flat_indices = torch.tensor([0, 1, 2], dtype=torch.int64)
    return credit_q31, projected_moves, flat_indices


@contextmanager
def _strict_path_forbidden_float_guard() -> Iterator[None]:
    original_to = torch.Tensor.to

    def guarded_to(self, dtype=None, *args, **kwargs):
        target = dtype if dtype is not None else (args[0] if args else None)
        if target == torch.float32:
            raise AssertionError("forbidden float32 in strict integer rank path")
        return original_to(self, dtype, *args, **kwargs)

    with (
        mock.patch.object(torch.Tensor, "to", guarded_to),
        mock.patch(
            "calm.hrm_text_158.native_full_stack.integer_sparse_rank_votes._rank_bin_bounds",
            side_effect=AssertionError("forbidden _rank_bin_bounds"),
        ),
    ):
        yield


def test_canonicalize_rank_boundary_rejects_one_tenth() -> None:
    with pytest.raises(ValueError, match="power of two"):
        canonicalize_rank_boundary_from_spec_float(0.1)


def test_canonicalize_rank_boundary_rejects_sci_notation_repr() -> None:
    with pytest.raises(ValueError, match="scientific-notation"):
        canonicalize_rank_boundary_from_spec_float(1e-05)


def test_canonicalize_rank_boundary_accepts_dry_run_bins() -> None:
    assert canonicalize_rank_boundary_from_spec_float(0.0) == CanonicalRankBoundary(0, 1)
    assert canonicalize_rank_boundary_from_spec_float(0.5) == CanonicalRankBoundary(1, 2)
    assert canonicalize_rank_boundary_from_spec_float(1.0) == CanonicalRankBoundary(1, 1)


@pytest.mark.parametrize("count", [1, 2, 3, 7, 16, 1024])
def test_rank_bin_boundary_integer_matches_incumbent_dry_run(count: int) -> None:
    spec = _simple_spec()
    canonical_bins = canonical_rank_vote_spec(spec)
    for float_bin, canonical_bin in zip(spec.rank_bins, canonical_bins):
        assert integer_rank_bin_bounds(count, canonical_bin) == _rank_bin_bounds(count, float_bin)


def test_integer_grouped_bisect_equal_magnitude_tie() -> None:
    abs_i64 = torch.tensor([3, 3, 1], dtype=torch.int64)
    ranks = grouped_bisect_right_rank_positions_integer_abs(abs_i64)
    assert ranks.tolist() == [3, 3, 1]


def test_strict_path_no_hidden_float() -> None:
    credit_q31, projected_moves, flat_indices = _green_credit_fixture()
    canonical_bins = canonical_rank_vote_spec(_simple_spec())
    with _strict_path_forbidden_float_guard():
        events = strict_integer_sparse_rank_bucketed_vote_events_from_credit(
            credit_q31,
            projected_moves,
            flat_indices,
            canonical_bins,
        )
    assert events.event_count() == 3


def test_gt_2pow24_collision_classified_precision_not_bin_boundary() -> None:
    credit_q31 = torch.tensor([16777216, 16777217, 1], dtype=torch.int32)
    projected_moves = torch.tensor([1, -1, 1], dtype=torch.int8)
    flat_indices = torch.tensor([0, 1, 2], dtype=torch.int64)
    result = compare_strict_integer_ranking_to_float32_reference(
        credit_q31,
        projected_moves,
        flat_indices,
        _simple_spec(),
        credit_law_id=CREDIT_LAW_NEG_ATTRIBUTION_Q31_V0,
    )
    assert result.branch_id == BR_F_RANKING_PRECISION_DIVERGENCE
    assert result.bin_boundary_divergence_count == 0
    assert result.drop_in_float32_parity_pass is False


def test_mixed_sign_abs_ordering() -> None:
    credit_q31 = torch.tensor([-5, 3, -1], dtype=torch.int32)
    projected_moves = torch.tensor([1, -1, 1], dtype=torch.int8)
    flat_indices = torch.tensor([0, 1, 2], dtype=torch.int64)
    result = compare_strict_integer_ranking_to_float32_reference(
        credit_q31,
        projected_moves,
        flat_indices,
        _simple_spec(),
        credit_law_id=CREDIT_LAW_NEG_ATTRIBUTION_Q31_V0,
    )
    assert result.branch_id == BR_F_RANKING_INTEGER_EXACT


def test_direct_credit_int32_min_fail_closed() -> None:
    credit_q31 = torch.tensor([INT32_MIN, 2], dtype=torch.int32)
    projected_moves = torch.tensor([1, -1], dtype=torch.int8)
    flat_indices = torch.tensor([0, 1], dtype=torch.int64)
    result = compare_strict_integer_ranking_to_float32_reference(
        credit_q31,
        projected_moves,
        flat_indices,
        _simple_spec(),
        credit_law_id=CREDIT_LAW_NEG_ATTRIBUTION_Q31_V0,
    )
    assert result.branch_id == BR_F_RANKING_REPRESENTATION_LIMIT


def test_zero_candidate_empty_events() -> None:
    result = compare_strict_integer_ranking_to_float32_reference(
        torch.empty(0, dtype=torch.int32),
        torch.empty(0, dtype=torch.int8),
        torch.empty(0, dtype=torch.int64),
        _simple_spec(),
        credit_law_id=CREDIT_LAW_NEG_ATTRIBUTION_Q31_V0,
    )
    assert result.branch_id == BR_F_RANKING_INTEGER_EXACT
    assert result.candidate_count == 0
    assert result.drop_in_float32_parity_pass is True
    assert result.strict_integer_self_consistency_pass is True


def test_drop_in_float32_parity_green_fixture() -> None:
    credit_q31, projected_moves, flat_indices = _green_credit_fixture()
    receipt = prove_strict_integer_ranking_subcontract(
        credit_q31,
        projected_moves,
        flat_indices,
        _simple_spec(),
        comparable_set_id="green-v1",
        reference_float32_run_id="ref-float-001",
        candidate_strict_run_id="candidate-strict-001",
        credit_law_id=CREDIT_LAW_NEG_ATTRIBUTION_Q31_V0,
    )
    assert receipt.branch_id == BR_F_RANKING_INTEGER_EXACT
    assert receipt.ranking_subcontract_mode == RANKING_SUBCONTRACT_MODE_STRICT_INTEGER
    assert receipt.drop_in_float32_parity_pass is True
    assert receipt.strict_integer_self_consistency_pass is True
    validate_ranking_subcontract_receipt(receipt)


def test_dual_pass_only_integer_exact_both_true() -> None:
    credit_q31, projected_moves, flat_indices = _green_credit_fixture()
    comparison = compare_strict_integer_ranking_to_float32_reference(
        credit_q31,
        projected_moves,
        flat_indices,
        _simple_spec(),
        credit_law_id=CREDIT_LAW_NEG_ATTRIBUTION_Q31_V0,
    )
    assert comparison.branch_id == BR_F_RANKING_INTEGER_EXACT
    assert comparison.drop_in_float32_parity_pass is True
    assert comparison.strict_integer_self_consistency_pass is True

    divergent = compare_strict_integer_ranking_to_float32_reference(
        torch.tensor([16777216, 16777217, 1], dtype=torch.int32),
        torch.tensor([1, -1, 1], dtype=torch.int8),
        torch.tensor([0, 1, 2], dtype=torch.int64),
        _simple_spec(),
        credit_law_id=CREDIT_LAW_NEG_ATTRIBUTION_Q31_V0,
    )
    assert divergent.drop_in_float32_parity_pass is False


def test_ranks_equal_bins_equal_events_differ_routes_measurement_invalid() -> None:
    credit_q31, projected_moves, flat_indices = _green_credit_fixture()
    rank_int = torch.tensor([1, 2, 3], dtype=torch.int64)
    rank_fp = rank_int.clone()
    bin_int = torch.tensor([0, 0, 1], dtype=torch.int64)
    bin_fp = bin_int.clone()
    strict_events = SparseVoteEvents(
        indices=flat_indices,
        values=torch.tensor([1, 2, 3], dtype=torch.int16),
    )
    float32_events = SparseVoteEvents(
        indices=flat_indices,
        values=torch.tensor([4, 8, 12], dtype=torch.int16),
    )
    branch_id = classify_strict_integer_ranking_branch(
        credit_q31=credit_q31,
        rank_int=rank_int,
        rank_fp=rank_fp,
        bin_int=bin_int,
        bin_fp=bin_fp,
        strict_events=strict_events,
        float32_events=float32_events,
        strict_self_consistent=True,
    )
    assert branch_id == BR_F_RANKING_MEASUREMENT_INVALID


def test_receipt_non_claims_superset() -> None:
    for claim in OPTIMIZER_CREDIT_STATE_FAIL_CLOSED_NON_CLAIMS:
        assert claim in RANKING_SUBCONTRACT_NON_CLAIMS


def test_receipt_forbidden_flags_default_false() -> None:
    snapshot = ranking_subcontract_hard_false_snapshot()
    for field in FORBIDDEN_RANKING_SUBCONTRACT_FIELDS:
        assert snapshot[field] is False


@pytest.mark.parametrize("field", FORBIDDEN_RANKING_SUBCONTRACT_FIELDS)
def test_receipt_validator_rejects_forbidden_true(field: str) -> None:
    credit_q31, projected_moves, flat_indices = _green_credit_fixture()
    receipt = prove_strict_integer_ranking_subcontract(
        credit_q31,
        projected_moves,
        flat_indices,
        _simple_spec(),
        comparable_set_id="forbidden-v1",
        reference_float32_run_id="ref-float-002",
        candidate_strict_run_id="candidate-strict-002",
        credit_law_id=CREDIT_LAW_NEG_ATTRIBUTION_Q31_V0,
    )
    with pytest.raises(ValueError, match=field):
        validate_ranking_subcontract_receipt(replace(receipt, **{field: True}))


def test_branch_id_integer_exact_iff_zero_mismatches() -> None:
    credit_q31, projected_moves, flat_indices = _green_credit_fixture()
    receipt = prove_strict_integer_ranking_subcontract(
        credit_q31,
        projected_moves,
        flat_indices,
        _simple_spec(),
        comparable_set_id="branch-v1",
        reference_float32_run_id="ref-float-003",
        candidate_strict_run_id="candidate-strict-003",
        credit_law_id=CREDIT_LAW_NEG_ATTRIBUTION_Q31_V0,
    )
    assert receipt.branch_id == BR_F_RANKING_INTEGER_EXACT
    with pytest.raises(ValueError, match="INTEGER-EXACT requires zero rank mismatches"):
        validate_ranking_subcontract_receipt(
            replace(receipt, integer_vs_float_rank_mismatch_count=1)
        )


def test_builder_rejects_mismatched_rank_bin_sha() -> None:
    credit_q31, projected_moves, flat_indices = _green_credit_fixture()
    comparison = compare_strict_integer_ranking_to_float32_reference(
        credit_q31,
        projected_moves,
        flat_indices,
        _simple_spec(),
        credit_law_id=CREDIT_LAW_NEG_ATTRIBUTION_Q31_V0,
    )
    bad = replace(comparison, rank_bin_spec_sha256="0" * 64)
    with pytest.raises(ValueError, match="rank_bin_spec_sha256"):
        build_ranking_subcontract_receipt(
            bad,
            comparable_set_id="tamper-v1",
            reference_float32_run_id="ref-float-004",
            candidate_strict_run_id="candidate-strict-004",
        )


def test_tie_group_fixture_not_bin_boundary() -> None:
    credit_q31 = torch.tensor([3, 3], dtype=torch.int32)
    projected_moves = torch.tensor([1, -1], dtype=torch.int8)
    flat_indices = torch.tensor([0, 1], dtype=torch.int64)
    result = compare_strict_integer_ranking_to_float32_reference(
        credit_q31,
        projected_moves,
        flat_indices,
        RankVoteSpec(
            rank_bins=(
                RankVoteBin(0.0, 0.5, 1),
                RankVoteBin(0.5, 1.0, 4, include_hi=True),
            )
        ),
        credit_law_id=CREDIT_LAW_NEG_ATTRIBUTION_Q31_V0,
    )
    if result.branch_id == BR_F_RANKING_TIE_GROUP_DIVERGENCE:
        assert result.bin_boundary_divergence_count == 0
    else:
        assert result.branch_id in {
            BR_F_RANKING_INTEGER_EXACT,
            BR_F_RANKING_PRECISION_DIVERGENCE,
            BR_F_RANKING_TIE_GROUP_DIVERGENCE,
        }
