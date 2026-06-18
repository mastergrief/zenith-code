"""CPU tests for 3C-A integer marginal attribution reference law."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator
from unittest import mock

import pytest
import torch
import torch.nn.functional as F

from calm.hrm_text_158.bit_linear import BitLinear
from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    authoritative_forward_context,
    make_bounded_tensor_state,
    project_s1_gradient_to_moves,
    weighted_grad_from_captures,
)
from calm.hrm_text_158.native_full_stack.integer_marginal_attribution import (
    CPU_REFERENCE_DENSE_INT32_SCRATCH_LABEL,
    INDEX_SET_ALL_STRUCTURALLY_TOUCHED,
    INDEX_SET_PROJECTED_MOVE_REFERENCE_ONLY,
    INTEGER_MARGINAL_ATTRIBUTION_LAW_ID,
    IntegerMarginalAttributionEvents,
    dense_int32_scratch_is_reference_only_not_row_flip_evidence,
    integer_marginal_attribution_from_captures,
    integer_marginal_attribution_hard_false_snapshot,
    projected_moves_from_integer_attribution,
)


class _Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = BitLinear(3, 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def _dry_run_fixture_tensors() -> tuple[dict, torch.Tensor, dict[str, torch.Tensor]]:
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
def _fp32_weight_shape_alloc_guard(weight_shape: tuple[int, int]) -> Iterator[None]:
    original_zeros = torch.zeros
    original_empty = torch.empty

    def guarded_zeros(*size, **kwargs):
        dtype = kwargs.get("dtype", torch.get_default_dtype())
        if len(size) == 1 and isinstance(size[0], (tuple, list)):
            shape = tuple(int(dim) for dim in size[0])
        else:
            shape = tuple(int(dim) for dim in size)
        if dtype == torch.float32 and len(shape) == 2 and shape == weight_shape:
            raise AssertionError("dense FP32 weighted_grad allocation detected on integer path")
        return original_zeros(*size, **kwargs)

    def guarded_empty(*size, **kwargs):
        dtype = kwargs.get("dtype", torch.get_default_dtype())
        if len(size) == 1 and isinstance(size[0], (tuple, list)):
            shape = tuple(int(dim) for dim in size[0])
        else:
            shape = tuple(int(dim) for dim in size)
        if dtype == torch.float32 and len(shape) == 2 and shape == weight_shape:
            raise AssertionError("dense FP32 weighted_grad allocation detected on integer path")
        return original_empty(*size, **kwargs)

    with mock.patch("torch.zeros", side_effect=guarded_zeros), mock.patch(
        "torch.empty",
        side_effect=guarded_empty,
    ):
        yield


def test_integer_attribution_matches_fp_moves_nonzero_set():
    captures, q_flat, states = _dry_run_fixture_tensors()
    weight_shape = tuple(int(dim) for dim in states["proj"].q_levels.shape)
    weighted_grad = weighted_grad_from_captures(
        captures["inputs"],
        captures["grad_outputs"],
        weight_shape=weight_shape,
    )
    fp_moves = project_s1_gradient_to_moves(weighted_grad, states["proj"].q_levels)
    fp_move_indices = set(torch.nonzero(fp_moves.reshape(-1) != 0, as_tuple=False).flatten().tolist())

    with _fp32_weight_shape_alloc_guard(weight_shape):
        events = integer_marginal_attribution_from_captures(
            captures["inputs"],
            captures["grad_outputs"],
            weight_shape=weight_shape,
            index_set_policy=INDEX_SET_ALL_STRUCTURALLY_TOUCHED,
        )
    move_indices, _moves = projected_moves_from_integer_attribution(events, q_flat)
    integer_move_indices = set(move_indices.tolist())

    assert integer_move_indices == fp_move_indices
    index_to_pos = {int(index): pos for pos, index in enumerate(events.flat_indices.tolist())}
    for index in integer_move_indices:
        fp_sign = int(torch.sign(weighted_grad.reshape(-1)[index]).item())
        int_sign = int(torch.sign(events.attribution_q31[index_to_pos[index]]).item())
        assert fp_sign == int_sign


def test_no_dense_weighted_grad_allocated():
    captures, q_flat, states = _dry_run_fixture_tensors()
    weight_shape = tuple(int(dim) for dim in states["proj"].q_levels.shape)
    with _fp32_weight_shape_alloc_guard(weight_shape):
        events = integer_marginal_attribution_from_captures(
            captures["inputs"],
            captures["grad_outputs"],
            weight_shape=weight_shape,
        )
        projected_moves_from_integer_attribution(events, q_flat)
    assert events.event_count() > 0


def test_events_validate_fail_closed():
    with pytest.raises(ValueError, match="flat_indices must be torch.int64"):
        IntegerMarginalAttributionEvents(
            flat_indices=torch.tensor([0], dtype=torch.int32),
            attribution_q31=torch.tensor([1], dtype=torch.int32),
            law_id=INTEGER_MARGINAL_ATTRIBUTION_LAW_ID,
            numel=2,
        ).validate()
    with pytest.raises(ValueError, match="strictly increasing"):
        IntegerMarginalAttributionEvents(
            flat_indices=torch.tensor([1, 1], dtype=torch.int64),
            attribution_q31=torch.tensor([2, 3], dtype=torch.int32),
            law_id=INTEGER_MARGINAL_ATTRIBUTION_LAW_ID,
            numel=4,
        ).validate()
    with pytest.raises(ValueError, match="out of range"):
        IntegerMarginalAttributionEvents(
            flat_indices=torch.tensor([4], dtype=torch.int64),
            attribution_q31=torch.tensor([1], dtype=torch.int32),
            law_id=INTEGER_MARGINAL_ATTRIBUTION_LAW_ID,
            numel=4,
        ).validate()


def test_law_metadata_present():
    captures, _q_flat, states = _dry_run_fixture_tensors()
    weight_shape = tuple(int(dim) for dim in states["proj"].q_levels.shape)
    events = integer_marginal_attribution_from_captures(
        captures["inputs"],
        captures["grad_outputs"],
        weight_shape=weight_shape,
    )
    assert events.law_id == INTEGER_MARGINAL_ATTRIBUTION_LAW_ID
    assert events.index_set_policy == INDEX_SET_ALL_STRUCTURALLY_TOUCHED
    assert events.is_production_oracle() is True


def test_projected_move_reference_only_is_not_production():
    captures, _q_flat, states = _dry_run_fixture_tensors()
    weight_shape = tuple(int(dim) for dim in states["proj"].q_levels.shape)
    weighted_grad = weighted_grad_from_captures(
        captures["inputs"],
        captures["grad_outputs"],
        weight_shape=weight_shape,
    )
    fp_moves = project_s1_gradient_to_moves(weighted_grad, states["proj"].q_levels)
    reference_indices = torch.nonzero(fp_moves.reshape(-1) != 0, as_tuple=False).flatten()
    events = integer_marginal_attribution_from_captures(
        captures["inputs"],
        captures["grad_outputs"],
        weight_shape=weight_shape,
        index_set_policy=INDEX_SET_PROJECTED_MOVE_REFERENCE_ONLY,
        reference_flat_indices=reference_indices,
    )
    assert events.reference_only is True
    assert events.is_production_oracle() is False


def test_dense_int32_scratch_is_reference_only_not_row_flip_evidence():
    assert CPU_REFERENCE_DENSE_INT32_SCRATCH_LABEL == "cpu_reference_dense_int32_scratch"
    assert dense_int32_scratch_is_reference_only_not_row_flip_evidence() is True
    hard_false = integer_marginal_attribution_hard_false_snapshot()
    assert hard_false["real_native_integer_attribution_present"] is False
    assert hard_false["ready_to_flip"] is False
