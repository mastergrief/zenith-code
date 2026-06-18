"""CPU tests for 3C-B integer sparse rank-vote reference law."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator
from unittest import mock

import pytest
import torch
import torch.nn.functional as F

from calm.hrm_text_158.bit_linear import BitLinear
from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    RankVoteSpec,
    RankVoteBin,
    _bisect_right_rank_positions_by_equal_value_group,
    authoritative_forward_context,
    credit_from_weighted_grad,
    default_dry_run_rank_vote_spec,
    make_bounded_tensor_state,
    project_s1_gradient_to_moves,
    rank_bucketed_int16_votes,
    weighted_grad_from_captures,
)
from calm.hrm_text_158.native_full_stack.integer_marginal_attribution import INT32_MIN
from calm.hrm_text_158.native_full_stack.integer_sparse_rank_votes import (
    BRANCH_3C_B_PARITY_FAIL,
    BRANCH_3C_B_PARITY_PASS_CPU,
    CPU_REFERENCE_DENSE_INT32_SCRATCH_LABEL,
    CREDIT_LAW_NEG_ATTRIBUTION_Q31_V0,
    CREDIT_LAW_POW2_BUCKET_INTEGER_V0,
    compare_sparse_rank_to_fp_dense_reference,
    credit_q31_from_attribution,
    dense_scratch_is_reference_only_not_row_flip_evidence,
    integer_sparse_rank_votes_hard_false_snapshot,
    sparse_rank_bucketed_vote_events_from_integer_credit,
    sparse_rank_votes_from_captures_reference,
)
from calm.hrm_text_158.native_full_stack.sparse_vote_events import SparseVoteEvents


class _Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = BitLinear(3, 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def _dry_run_fixture_tensors() -> tuple[dict, torch.Tensor, dict]:
    torch.manual_seed(158)
    model = _Tiny()
    with torch.no_grad():
        model.proj.weight.zero_()
    eligible = {"proj": model.proj}
    q = torch.zeros_like(model.proj.weight.detach(), dtype=torch.int8)
    tensor_state = make_bounded_tensor_state(
        "proj",
        q,
        torch.tensor(1.0, dtype=torch.float32),
        hot_exact_indices=tuple(range(int(q.numel()))),
    )
    x = torch.tensor([[1.0, -2.0, 3.0]], dtype=torch.float32)
    target = torch.tensor([[2.0, -1.0]], dtype=torch.float32)
    model.zero_grad(set_to_none=True)
    with authoritative_forward_context(
        eligible,
        {"proj": tensor_state},
        device="cpu",
        requires_grad=True,
    ) as handle:
        out = model(x)
        loss = F.mse_loss(out, target)
        loss.backward()
        captures = handle.captures["proj"]
    return captures, q.reshape(-1), {"proj": tensor_state}


@contextmanager
def _dense_vote_alloc_guard(weight_shape: tuple[int, int]) -> Iterator[None]:
    original_zeros = torch.zeros

    def guarded_zeros(*size, **kwargs):
        dtype = kwargs.get("dtype", torch.get_default_dtype())
        if len(size) == 1 and isinstance(size[0], (tuple, list)):
            shape = tuple(int(dim) for dim in size[0])
        else:
            shape = tuple(int(dim) for dim in size)
        if len(shape) == 2 and shape == weight_shape:
            if dtype == torch.float32:
                raise AssertionError("dense FP32 credit allocation detected on integer path")
            if dtype == torch.int16:
                raise AssertionError("dense int16 vote allocation detected on integer path")
        return original_zeros(*size, **kwargs)

    with mock.patch("torch.zeros", side_effect=guarded_zeros):
        yield


def test_sparse_rank_votes_match_dense_fp_reference():
    captures, q_flat, states = _dry_run_fixture_tensors()
    weight_shape = tuple(int(dim) for dim in states["proj"].q_levels.shape)
    result = sparse_rank_votes_from_captures_reference(
        captures["inputs"],
        captures["grad_outputs"],
        weight_shape=weight_shape,
        q_levels_flat=q_flat,
        credit_law_id=CREDIT_LAW_NEG_ATTRIBUTION_Q31_V0,
    )
    assert result.branch_id == BRANCH_3C_B_PARITY_PASS_CPU
    assert result.events_match is True
    assert result.rank_positions_match is True
    assert result.credit_law_id == CREDIT_LAW_NEG_ATTRIBUTION_Q31_V0


def test_rank_uses_float32_abs_bits_not_raw_int32():
    spec = default_dry_run_rank_vote_spec()
    fp_abs = torch.tensor([2.7, 3.3], dtype=torch.float32)
    fp_rank = _bisect_right_rank_positions_by_equal_value_group(fp_abs)
    int_abs = torch.tensor([3.0, 3.0], dtype=torch.float32)
    int_rank = _bisect_right_rank_positions_by_equal_value_group(int_abs)
    assert fp_rank.tolist() == [1, 2]
    assert int_rank.tolist() == [2, 2]
    assert not torch.equal(fp_rank, int_rank)


def test_no_dense_credit_or_vote_buffer_allocated():
    captures, q_flat, states = _dry_run_fixture_tensors()
    weight_shape = tuple(int(dim) for dim in states["proj"].q_levels.shape)
    spec = default_dry_run_rank_vote_spec()
    from calm.hrm_text_158.native_full_stack.integer_marginal_attribution import (
        integer_marginal_attribution_from_captures,
        projected_moves_from_integer_attribution,
    )

    attribution_events = integer_marginal_attribution_from_captures(
        captures["inputs"],
        captures["grad_outputs"],
        weight_shape=weight_shape,
    )
    move_indices, moves = projected_moves_from_integer_attribution(attribution_events, q_flat)
    index_to_pos = {int(index): pos for pos, index in enumerate(attribution_events.flat_indices.tolist())}
    attribution_selected = torch.tensor(
        [
            int(attribution_events.attribution_q31[index_to_pos[int(index)]].item())
            for index in move_indices.tolist()
        ],
        dtype=torch.int32,
    )
    credit_q31 = credit_q31_from_attribution(
        attribution_selected,
        credit_law_id=CREDIT_LAW_NEG_ATTRIBUTION_Q31_V0,
    )
    with _dense_vote_alloc_guard(weight_shape):
        integer_events = sparse_rank_bucketed_vote_events_from_integer_credit(
            credit_q31,
            moves,
            move_indices,
            spec,
        )
    assert integer_events.event_count() > 0


def test_events_validate_fail_closed():
    spec = default_dry_run_rank_vote_spec()
    with pytest.raises(ValueError, match="credit_q31 must be torch.int32"):
        sparse_rank_bucketed_vote_events_from_integer_credit(
            torch.tensor([1], dtype=torch.int64),
            torch.tensor([1], dtype=torch.int8),
            torch.tensor([0], dtype=torch.int64),
            spec,
        )
    with pytest.raises(ValueError, match="projected_moves must be nonzero"):
        sparse_rank_bucketed_vote_events_from_integer_credit(
            torch.tensor([1], dtype=torch.int32),
            torch.tensor([0], dtype=torch.int8),
            torch.tensor([0], dtype=torch.int64),
            spec,
        )


def test_credit_law_metadata_present():
    attribution = torch.tensor([-2, 4], dtype=torch.int32)
    credit_q31 = credit_q31_from_attribution(
        attribution,
        credit_law_id=CREDIT_LAW_NEG_ATTRIBUTION_Q31_V0,
    )
    assert credit_q31.tolist() == [2, -4]


def test_dense_scratch_reference_only_not_row_flip():
    assert CPU_REFERENCE_DENSE_INT32_SCRATCH_LABEL == "cpu_reference_dense_int32_scratch"
    assert dense_scratch_is_reference_only_not_row_flip_evidence() is True
    hard_false = integer_sparse_rank_votes_hard_false_snapshot()
    assert hard_false["real_native_integer_credit_ranking_present"] is False
    assert hard_false["ready_to_flip"] is False


def test_fractional_magnitude_fixture_classified():
    spec = default_dry_run_rank_vote_spec()
    flat_indices = torch.tensor([0, 1], dtype=torch.int64)
    projected_moves = torch.tensor([1, -1], dtype=torch.int8)
    fp_credit_dense = torch.tensor([[2.7, -3.3]], dtype=torch.float32)
    fp_moves_dense = projected_moves.reshape(1, 2)
    # Integer path collapses 2.7/3.3 magnitudes to 3/3 after rescale.
    credit_q31 = torch.tensor([3, 3], dtype=torch.int32)
    result = compare_sparse_rank_to_fp_dense_reference(
        credit_q31,
        projected_moves,
        flat_indices,
        fp_credit_dense,
        fp_moves_dense,
        spec,
        credit_law_id=CREDIT_LAW_NEG_ATTRIBUTION_Q31_V0,
    )
    assert result.branch_id == BRANCH_3C_B_PARITY_FAIL
    assert result.rank_positions_match is False

    # Fallback is pre-registered only; not default and not assumed to recover precision.
    fallback_credit = credit_q31_from_attribution(
        torch.tensor([-3, -3], dtype=torch.int32),
        credit_law_id=CREDIT_LAW_POW2_BUCKET_INTEGER_V0,
    )
    fallback_result = compare_sparse_rank_to_fp_dense_reference(
        fallback_credit,
        projected_moves,
        flat_indices,
        fp_credit_dense,
        fp_moves_dense,
        spec,
        credit_law_id=CREDIT_LAW_POW2_BUCKET_INTEGER_V0,
    )
    assert fallback_result.credit_law_id == CREDIT_LAW_POW2_BUCKET_INTEGER_V0
    assert fallback_result.branch_id == BRANCH_3C_B_PARITY_FAIL


def test_credit_q31_primary_rejects_int32_min_negation_overflow():
    with pytest.raises(ValueError, match="INT32_MIN"):
        credit_q31_from_attribution(
            torch.tensor([INT32_MIN], dtype=torch.int32),
            credit_law_id=CREDIT_LAW_NEG_ATTRIBUTION_Q31_V0,
        )


def test_flat_indices_validate_fail_closed():
    spec = default_dry_run_rank_vote_spec()
    credit_q31 = torch.tensor([2, 3], dtype=torch.int32)
    moves = torch.tensor([1, -1], dtype=torch.int8)
    with pytest.raises(ValueError, match="strictly increasing"):
        sparse_rank_bucketed_vote_events_from_integer_credit(
            credit_q31,
            moves,
            torch.tensor([1, 0], dtype=torch.int64),
            spec,
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        sparse_rank_bucketed_vote_events_from_integer_credit(
            credit_q31,
            moves,
            torch.tensor([0, 0], dtype=torch.int64),
            spec,
        )
    with pytest.raises(ValueError, match="non-negative"):
        sparse_rank_bucketed_vote_events_from_integer_credit(
            credit_q31,
            moves,
            torch.tensor([-1, 2], dtype=torch.int64),
            spec,
        )
