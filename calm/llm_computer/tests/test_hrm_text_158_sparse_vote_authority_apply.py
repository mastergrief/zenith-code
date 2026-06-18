"""CPU tests for HRM-Text-1.58 Step 3C-C2a sparse vote authority apply shim."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator
from unittest import mock

import pytest
import torch

from calm.hrm_text_158.native_full_stack.bounded_delta_accumulator import (
    ACCUMULATOR_SUBSTITUTE_LOCAL_VOTE_UPDATE_EXECUTABLE,
    ALGORITHMIC_LOCAL_VOTE_UPDATE_EXECUTABLE_NOT_PHYSICAL_SUB2,
)
from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    apply_bounded_delta_vote_step,
    make_bounded_tensor_state,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap import GlobalRateCapSpec
from calm.hrm_text_158.native_full_stack.sparse_vote_authority_apply import (
    BRANCH_3C_C2_DENSE_LEAK,
    BRANCH_3C_C2_SPARSE_APPLY_VIABLE,
    PARITY_CONTRACT_MODE_SPARSE_EVENT_SHAPE_ONLY,
    SPARSE_VOTE_AUTHORITY_HARD_FALSE_FIELDS,
    SparseVoteAuthorityApplyReceipt,
    build_sparse_vote_authority_apply_receipt,
    classify_sparse_vote_authority_apply_receipt,
    sparse_vote_authority_hard_false_snapshot,
    validate_sparse_vote_authority_apply_receipt,
)
from calm.hrm_text_158.native_full_stack.vote_update import (
    LOCAL_SELECTION_ORDER_TRANSIENT_LOCAL_LOSS_DELTA,
    VoteUpdateSpec,
)


def _fixture_state_and_spec() -> tuple:
    state = make_bounded_tensor_state(
        "toy.proj",
        torch.tensor([0, 0], dtype=torch.int8),
        0.5,
        torch.zeros(2, dtype=torch.int16),
    )
    spec = VoteUpdateSpec(
        threshold_abs=10,
        accumulator_clip_min=-127,
        accumulator_clip_max=127,
        max_abs_per_tensor=2,
    )
    sparse_events = {"toy.proj": {0: 12, 1: -12}}
    return state, spec, sparse_events


def _sparse_authority_kwargs(
  state,
  spec,
  sparse_events,
  **overrides,
):
    base = dict(
        tensor_states={"toy.proj": state},
        votes_by_key=None,
        vote_specs_by_key={"toy.proj": spec},
        candidate_mode=ACCUMULATOR_SUBSTITUTE_LOCAL_VOTE_UPDATE_EXECUTABLE,
        candidate_sparse_vote_events_by_key=sparse_events,
        candidate_oracle_control_enabled=False,
        sparse_vote_authority_only=True,
    )
    base.update(overrides)
    return base


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
                raise AssertionError("dense FP32 [O,I] allocation detected on sparse-authority path")
            if dtype == torch.int16:
                raise AssertionError("dense int16 [O,I] allocation detected on sparse-authority path")
        return original_zeros(*size, **kwargs)

    with mock.patch("torch.zeros", side_effect=guarded_zeros):
        yield


def test_sparse_authority_succeeds_only_under_exact_five_conditions():
    state, spec, sparse_events = _fixture_state_and_spec()
    result = apply_bounded_delta_vote_step(**_sparse_authority_kwargs(state, spec, sparse_events))
    summary = result.global_summary
    assert summary["sparse_vote_authority_only"] is True
    assert summary["dense_vote_authority_skipped"] is True
    assert summary["parity_contract_mode"] == PARITY_CONTRACT_MODE_SPARSE_EVENT_SHAPE_ONLY
    assert summary["pass_receipt"] is False
    assert summary["candidate_dense_vote_authority_used"] is False


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"sparse_vote_authority_only": True, "votes_by_key": {"toy.proj": torch.zeros(2, dtype=torch.int16)}}, "sparse_vote_authority_only requires votes_by_key=None"),
        ({"sparse_vote_authority_only": True, "candidate_mode": ALGORITHMIC_LOCAL_VOTE_UPDATE_EXECUTABLE_NOT_PHYSICAL_SUB2}, "sparse_vote_authority_only requires"),
        ({"sparse_vote_authority_only": True, "candidate_oracle_control_enabled": True}, "sparse_vote_authority_only requires candidate_oracle_control_enabled=False"),
        ({"sparse_vote_authority_only": True, "candidate_sparse_vote_events_by_key": None}, "sparse_vote_authority_only requires candidate_sparse_vote_events_by_key"),
        ({"sparse_vote_authority_only": True, "candidate_sparse_vote_events_by_key": {}}, "candidate_sparse_vote_events_by_key keys must match"),
        ({"sparse_vote_authority_only": True, "candidate_sparse_vote_events_by_key": {"other": {0: 1}}}, "candidate_sparse_vote_events_by_key keys must match"),
    ],
)
def test_sparse_authority_gate_rejects_invalid_preconditions(overrides, match):
    state, spec, sparse_events = _fixture_state_and_spec()
    kwargs = _sparse_authority_kwargs(state, spec, sparse_events, **overrides)
    with pytest.raises(ValueError, match=match):
        apply_bounded_delta_vote_step(**kwargs)


def test_votes_by_key_none_rejected_without_sparse_authority_flag():
    state, spec, sparse_events = _fixture_state_and_spec()
    with pytest.raises(ValueError, match="votes_by_key is required unless sparse_vote_authority_only=True"):
        apply_bounded_delta_vote_step(
            tensor_states={"toy.proj": state},
            votes_by_key=None,
            vote_specs_by_key={"toy.proj": spec},
            candidate_mode=ACCUMULATOR_SUBSTITUTE_LOCAL_VOTE_UPDATE_EXECUTABLE,
            candidate_sparse_vote_events_by_key=sparse_events,
            candidate_oracle_control_enabled=False,
        )


def test_sparse_authority_rejects_active_controls_and_alternate_ordering():
    state, spec, sparse_events = _fixture_state_and_spec()
    base = _sparse_authority_kwargs(state, spec, sparse_events)
    active_cases = [
        ({"global_cap_spec": GlobalRateCapSpec(cap=1, step=0)}, "global cap"),
        ({"deferred_backlog": {"toy.proj": {0: {"defer_count": 1}}}}, "deferred backlog"),
        (
            {
                "replay_ce_veto_votes_by_key": {"toy.proj": torch.zeros(2, dtype=torch.int16)},
                "replay_ce_veto_moves_by_key": {"toy.proj": torch.zeros(2, dtype=torch.int8)},
            },
            "replay/pc auxiliary",
        ),
        (
            {
                "pc_aux_votes_by_key": {"toy.proj": torch.zeros(2, dtype=torch.int16)},
                "pc_aux_moves_by_key": {"toy.proj": torch.zeros(2, dtype=torch.int8)},
            },
            "replay/pc auxiliary",
        ),
        ({"front_c_identity_observer": lambda payload: payload}, "front_c live identity"),
        (
            {"local_selection_ordering_mode": LOCAL_SELECTION_ORDER_TRANSIENT_LOCAL_LOSS_DELTA},
            "alternate local ordering",
        ),
    ]
    for extra_kwargs, error in active_cases:
        with pytest.raises(ValueError, match=error):
            apply_bounded_delta_vote_step(**{**base, **extra_kwargs})


def test_sparse_authority_rejects_unsupported_candidate_mode():
    state, spec, sparse_events = _fixture_state_and_spec()
    with pytest.raises(ValueError, match="sparse_vote_authority_only requires"):
        apply_bounded_delta_vote_step(
            **_sparse_authority_kwargs(
                state,
                spec,
                sparse_events,
                candidate_mode=ALGORITHMIC_LOCAL_VOTE_UPDATE_EXECUTABLE_NOT_PHYSICAL_SUB2,
            )
        )


def test_default_dense_callers_unchanged_without_sparse_flag():
    state, spec, _sparse_events = _fixture_state_and_spec()
    votes = torch.tensor([12, -12], dtype=torch.int16)
    result = apply_bounded_delta_vote_step(
        {"toy.proj": state},
        {"toy.proj": votes},
        {"toy.proj": spec},
        candidate_mode=ACCUMULATOR_SUBSTITUTE_LOCAL_VOTE_UPDATE_EXECUTABLE,
        candidate_sparse_vote_events_by_key={"toy.proj": {0: 12, 1: -12}},
        candidate_oracle_control_enabled=False,
    )
    assert "sparse_vote_authority_only" not in result.global_summary
    assert result.global_summary["candidate_dense_vote_authority_used"] is False


def test_oracle_on_dense_path_still_requires_votes_by_key():
    state, spec, sparse_events = _fixture_state_and_spec()
    votes = torch.tensor([12, -12], dtype=torch.int16)
    result = apply_bounded_delta_vote_step(
        {"toy.proj": state},
        {"toy.proj": votes},
        {"toy.proj": spec},
        candidate_mode=ACCUMULATOR_SUBSTITUTE_LOCAL_VOTE_UPDATE_EXECUTABLE,
        candidate_sparse_vote_events_by_key=sparse_events,
        candidate_oracle_control_enabled=True,
    )
    proof = result.global_summary["candidate_local_update_proof_by_key"]["toy.proj"]
    assert proof["dense_oracle_control_used"] is True
    assert proof["oracle_dense_vote_sha256"] is not None


def test_no_dense_o_i_alloc_on_sparse_authority_path():
    state, spec, sparse_events = _fixture_state_and_spec()
    weight_shape = tuple(int(dim) for dim in state.q_levels.shape)
    if len(weight_shape) != 2:
        weight_shape = (1, int(state.q_levels.numel()))
    with _dense_vote_alloc_guard(weight_shape):
        result = apply_bounded_delta_vote_step(**_sparse_authority_kwargs(state, spec, sparse_events))
    assert result.global_summary["sparse_vote_authority_only"] is True


def test_sparse_authority_receipt_hard_false_and_parity_contract():
    state, spec, sparse_events = _fixture_state_and_spec()
    step = apply_bounded_delta_vote_step(**_sparse_authority_kwargs(state, spec, sparse_events))
    receipt = build_sparse_vote_authority_apply_receipt(
        step_result=step,
        sparse_events_by_key=sparse_events,
    )
    assert isinstance(receipt, SparseVoteAuthorityApplyReceipt)
    assert receipt.parity_contract_mode == PARITY_CONTRACT_MODE_SPARSE_EVENT_SHAPE_ONLY
    assert receipt.pass_receipt is False
    assert receipt.sparse_vote_authority_only is True
    for field in SPARSE_VOTE_AUTHORITY_HARD_FALSE_FIELDS:
        assert getattr(receipt, field) is False
    validate_sparse_vote_authority_apply_receipt(receipt)
    assert sparse_vote_authority_hard_false_snapshot() == {field: False for field in SPARSE_VOTE_AUTHORITY_HARD_FALSE_FIELDS}


def test_classify_dense_leak_branch():
    state, spec, sparse_events = _fixture_state_and_spec()
    step = apply_bounded_delta_vote_step(**_sparse_authority_kwargs(state, spec, sparse_events))
    assert classify_sparse_vote_authority_apply_receipt(
        step_result=step,
        sparse_events_by_key=sparse_events,
        dense_leak_detected=True,
    ) == BRANCH_3C_C2_DENSE_LEAK


def test_classify_viable_branch_when_local_update_passes():
    state, spec, sparse_events = _fixture_state_and_spec()
    step = apply_bounded_delta_vote_step(**_sparse_authority_kwargs(state, spec, sparse_events))
    if step.global_summary.get("candidate_local_update_pass"):
        assert classify_sparse_vote_authority_apply_receipt(
            step_result=step,
            sparse_events_by_key=sparse_events,
        ) == BRANCH_3C_C2_SPARSE_APPLY_VIABLE
