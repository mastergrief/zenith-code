"""CPU tests for 3C-C2b integer optimizer credit-path wire (Option A)."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from typing import Iterator
from unittest import mock

import pytest
import torch
import torch.nn.functional as F

from calm.hrm_text_158.bit_linear import BitLinear
from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    AuthoritativeForwardHandle,
    authoritative_forward_context,
    default_dry_run_rank_vote_spec,
    make_bounded_tensor_state,
)
from calm.hrm_text_158.native_full_stack.integer_optimizer_credit_path import (
    BRANCH_3C_C2B_DENSE_LEAK,
    BRANCH_3C_C2B_WIRE_VIABLE,
    INTEGER_OPTIMIZER_CREDIT_PATH_ENABLED,
    INTEGER_OPTIMIZER_CREDIT_PATH_WIRE_HARD_FALSE_FIELDS,
    INTEGER_OPTIMIZER_CREDIT_PATH_WIRE_NON_CLAIMS,
    IntegerOptimizerCreditPathWireReceipt,
    apply_integer_optimizer_credit_path_step,
    build_integer_optimizer_credit_path_wire_receipt,
    build_integer_optimizer_credit_path_wire_receipt_from_step,
    default_integer_optimizer_credit_path_vote_update_spec,
    emit_integer_sparse_vote_events_from_trainer_handle,
    integer_optimizer_credit_path_hard_false_snapshot,
    validate_integer_optimizer_credit_path_wire_receipt,
)
from calm.hrm_text_158.native_full_stack.sparse_vote_authority_apply import (
    PARITY_CONTRACT_MODE_SPARSE_EVENT_SHAPE_ONLY,
)
from calm.hrm_text_158.native_full_stack.sparse_vote_events import SparseVoteEvents
from calm.hrm_text_158.native_full_stack.trainer_sub2_authority import (
    trainer_local_update_builder_active_control_parameters,
)


class _Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = BitLinear(3, 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def _capture_handle() -> tuple[AuthoritativeForwardHandle, dict, torch.Tensor]:
    torch.manual_seed(158)
    model = _Tiny()
    with torch.no_grad():
        model.proj.weight.zero_()
    eligible = {"proj": model.proj}
    q = torch.zeros_like(model.proj.weight.detach(), dtype=torch.int8)
    state = make_bounded_tensor_state(
        "proj",
        q,
        torch.tensor(1.0, dtype=torch.float32),
        hot_exact_indices=tuple(range(int(q.numel()))),
    )
    states = {"proj": state}
    x = torch.tensor([[1.0, -2.0, 3.0]], dtype=torch.float32)
    target = torch.tensor([[2.0, -1.0]], dtype=torch.float32)
    model.zero_grad(set_to_none=True)
    with authoritative_forward_context(
        eligible,
        states,
        device="cpu",
        requires_grad=True,
    ) as handle:
        out = model(x)
        loss = F.mse_loss(out, target)
        loss.backward()
    return handle, states, x


@contextmanager
def _dense_o_i_alloc_guard(weight_shape: tuple[int, int]) -> Iterator[None]:
    original_zeros = torch.zeros

    def guarded_zeros(*size, **kwargs):
        dtype = kwargs.get("dtype", torch.get_default_dtype())
        if len(size) == 1 and isinstance(size[0], (tuple, list)):
            shape = tuple(int(dim) for dim in size[0])
        else:
            shape = tuple(int(dim) for dim in size)
        if len(shape) == 2 and shape == weight_shape:
            if dtype == torch.float32:
                raise AssertionError("dense FP32 [O,I] allocation detected on integer wire path")
            if dtype == torch.int16:
                raise AssertionError("dense int16 [O,I] allocation detected on integer wire path")
        return original_zeros(*size, **kwargs)

    with mock.patch("torch.zeros", side_effect=guarded_zeros):
        yield


def test_module_constant_default_off():
    assert INTEGER_OPTIMIZER_CREDIT_PATH_ENABLED is False


def test_emit_integer_sparse_events_from_captures():
    handle, states, _x = _capture_handle()
    rank_spec = default_dry_run_rank_vote_spec()
    weight_shape = tuple(int(dim) for dim in states["proj"].q_levels.shape)
    with _dense_o_i_alloc_guard(weight_shape):
        sparse_events = emit_integer_sparse_vote_events_from_trainer_handle(
            handle,
            states,
            rank_spec,
        )
    assert int(sparse_events["proj"].event_count()) > 0


def test_emit_integer_path_does_not_call_weighted_grad():
    handle, states, _x = _capture_handle()
    rank_spec = default_dry_run_rank_vote_spec()
    with mock.patch.object(
        AuthoritativeForwardHandle,
        "weighted_grad",
        autospec=True,
        side_effect=AssertionError(
            "handle.weighted_grad must not be called on integer wire path"
        ),
    ):
        sparse_events = emit_integer_sparse_vote_events_from_trainer_handle(
            handle,
            states,
            rank_spec,
        )
    assert int(sparse_events["proj"].event_count()) > 0


def test_apply_integer_optimizer_credit_path_step_uses_sparse_authority():
    handle, states, _x = _capture_handle()
    rank_spec = default_dry_run_rank_vote_spec()
    sparse_events = emit_integer_sparse_vote_events_from_trainer_handle(
        handle,
        states,
        rank_spec,
    )
    spec = default_integer_optimizer_credit_path_vote_update_spec()
    step = apply_integer_optimizer_credit_path_step(
        states,
        sparse_events,
        {"proj": spec},
    )
    summary = step.global_summary
    assert summary["sparse_vote_authority_only"] is True
    assert summary["parity_contract_mode"] == PARITY_CONTRACT_MODE_SPARSE_EVENT_SHAPE_ONLY
    assert summary["pass_receipt"] is False


def test_build_integer_optimizer_credit_path_wire_receipt_tiny_dry_run():
    torch.manual_seed(158)
    model = _Tiny()
    with torch.no_grad():
        model.proj.weight.zero_()
    batch = {
        "x": torch.tensor([[1.0, -2.0, 3.0]], dtype=torch.float32),
        "target": torch.tensor([[2.0, -1.0]], dtype=torch.float32),
    }

    def forward_loss_fn(mod: torch.nn.Module, data: dict) -> torch.Tensor:
        return F.mse_loss(mod(data["x"]), data["target"])

    receipt = build_integer_optimizer_credit_path_wire_receipt(
        model,
        batch=batch,
        forward_loss_fn=forward_loss_fn,
        use_ternary_bulk=True,
        device="cpu",
    )
    assert isinstance(receipt, IntegerOptimizerCreditPathWireReceipt)
    assert receipt.branch_id == BRANCH_3C_C2B_WIRE_VIABLE
    assert receipt.total_sparse_event_count > 0
    assert receipt.dense_credit_path_materialized is False
    assert receipt.oracle_parity_proof_executed is False
    assert receipt.parity_contract_mode == PARITY_CONTRACT_MODE_SPARSE_EVENT_SHAPE_ONLY
    assert receipt.pass_receipt is False
    for field in INTEGER_OPTIMIZER_CREDIT_PATH_WIRE_HARD_FALSE_FIELDS:
        assert getattr(receipt, field) is False
    validate_integer_optimizer_credit_path_wire_receipt(receipt)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("pass_receipt", True),
        ("dense_credit_path_materialized", True),
        ("oracle_parity_proof_executed", True),
        ("ready_to_flip", True),
        ("parity_contract_mode", "REFERENCE_DENSE_ORACLE"),
    ],
)
def test_wire_receipt_validator_rejects_forbidden_fields(field, value):
    handle, states, _x = _capture_handle()
    rank_spec = default_dry_run_rank_vote_spec()
    sparse_events = emit_integer_sparse_vote_events_from_trainer_handle(
        handle,
        states,
        rank_spec,
    )
    spec = default_integer_optimizer_credit_path_vote_update_spec()
    step = apply_integer_optimizer_credit_path_step(states, sparse_events, {"proj": spec})
    receipt = build_integer_optimizer_credit_path_wire_receipt_from_step(
        step_result=step,
        sparse_events_by_key=sparse_events,
    )
    bad = replace(receipt, **{field: value})
    with pytest.raises(ValueError):
        validate_integer_optimizer_credit_path_wire_receipt(bad)


def test_wire_receipt_validator_rejects_changed_non_claims():
    handle, states, _x = _capture_handle()
    sparse_events = emit_integer_sparse_vote_events_from_trainer_handle(
        handle,
        states,
        default_dry_run_rank_vote_spec(),
    )
    step = apply_integer_optimizer_credit_path_step(
        states,
        sparse_events,
        {"proj": default_integer_optimizer_credit_path_vote_update_spec()},
    )
    receipt = build_integer_optimizer_credit_path_wire_receipt_from_step(
        step_result=step,
        sparse_events_by_key=sparse_events,
    )
    bad = replace(receipt, non_claims=INTEGER_OPTIMIZER_CREDIT_PATH_WIRE_NON_CLAIMS + ("extra",))
    with pytest.raises(ValueError):
        validate_integer_optimizer_credit_path_wire_receipt(bad)


def test_classify_dense_leak_branch():
    handle, states, _x = _capture_handle()
    sparse_events = emit_integer_sparse_vote_events_from_trainer_handle(
        handle,
        states,
        default_dry_run_rank_vote_spec(),
    )
    step = apply_integer_optimizer_credit_path_step(
        states,
        sparse_events,
        {"proj": default_integer_optimizer_credit_path_vote_update_spec()},
    )
    receipt = build_integer_optimizer_credit_path_wire_receipt_from_step(
        step_result=step,
        sparse_events_by_key=sparse_events,
        dense_leak_detected=True,
    )
    assert receipt.branch_id == BRANCH_3C_C2B_DENSE_LEAK
    validate_integer_optimizer_credit_path_wire_receipt(receipt)


def test_hard_false_snapshot_all_false():
    assert integer_optimizer_credit_path_hard_false_snapshot() == {
        field: False for field in INTEGER_OPTIMIZER_CREDIT_PATH_WIRE_HARD_FALSE_FIELDS
    }


def test_trainer_active_control_parameters_unchanged():
    before = trainer_local_update_builder_active_control_parameters()
    after = trainer_local_update_builder_active_control_parameters()
    assert before == after
