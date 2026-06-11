"""CPU tests for W6/T=10 grad-proxy audit (slice 3a)."""
from __future__ import annotations

from typing import Any

import pytest
import torch

from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    VoteUpdateSpec,
    make_bounded_tensor_state,
)
from calm.hrm_text_158.native_full_stack.grad_proxy_audit import (
    GRAD_PROXY_AUDIT_COMPARATOR_SPEC,
    GradProxyAuditAborted,
    compute_grad_proxy_pass_bars,
    derive_w6_t10_candidate_delta_weight,
    run_grad_proxy_audit_step1,
    w6_t10_base_spec,
)
from calm.hrm_text_158.native_full_stack.oracle_screen_runner import (
    _audit_sparse_singleton_identity_for_candidate,
    _candidate_delta_weight_from_one_flip,
    _single_flip_spec,
)
from calm.hrm_text_158.native_full_stack.vote_update import (
    LOCAL_SELECTION_ORDER_CURRENT_MARGIN_INDEX,
    VoteUpdateInputs,
    apply_integer_vote_update_reference,
)
from calm.llm_computer.tests.test_hrm_text_158_native_bounded_delta_acquisition_probe import (
    _tiny_forward_fixture,
)


def _delta_weight_from_spec(
    *,
    state: Any,
    votes: torch.Tensor,
    candidate: dict[str, Any],
    base_spec: VoteUpdateSpec,
) -> float:
    one_flip_spec = _single_flip_spec(base_spec)
    flat_index = int(candidate["flat_index"])
    sparse_votes = torch.zeros_like(votes, dtype=torch.int16)
    sparse_votes.view(-1)[flat_index] = votes.view(-1)[flat_index]
    result = apply_integer_vote_update_reference(
        state.vote_update_state(),
        VoteUpdateInputs(votes=sparse_votes),
        one_flip_spec,
        local_selection_ordering_mode=LOCAL_SELECTION_ORDER_CURRENT_MARGIN_INDEX,
    )
    return _candidate_delta_weight_from_one_flip(
        q_after_one_flip=result.q_levels,
        flat_index=flat_index,
        current_q_level=int(candidate["current_q_level"]),
        frozen_scale_scalar=float(state.frozen_scale.detach().cpu().item()),
    )


def _identity_hold_w6_t10_bundle(
    state_key: str,
) -> tuple[Any, torch.Tensor, dict[str, Any], VoteUpdateSpec, VoteUpdateSpec]:
    """Explicit 2-row fixture: singleton apply holds on flat_index=0 under T=10."""

    q = torch.tensor([0, 0], dtype=torch.int8)
    acc = torch.tensor([9, 0], dtype=torch.int16)
    state = make_bounded_tensor_state(state_key, q, 1.0, acc)
    votes = torch.tensor([12, 0], dtype=torch.int16)
    base_spec = w6_t10_base_spec(max_abs_per_tensor=4096)
    one_flip_spec = _single_flip_spec(base_spec)
    candidate = {
        "candidate_id": f"{state_key}:0",
        "state_key": state_key,
        "flat_index": 0,
        "current_q_level": 0,
        "proposal_direction": 1,
        "new_acc_i32_signed": 12,
        "deterministic_hash_rank_position": 0,
        "current_rank_position": 0,
        "vote_value": 12,
        "pre_accumulator_i16": 9,
    }
    audit = _audit_sparse_singleton_identity_for_candidate(
        tensor_state=state,
        votes=votes,
        candidate=candidate,
        one_flip_spec=one_flip_spec,
    )
    assert audit["drifted"] is False
    assert audit["applied_indices"] == [0]
    return state, votes, candidate, base_spec, one_flip_spec


def _singleton_drift_w6_t10_bundle(
    state_key: str,
) -> tuple[Any, torch.Tensor, dict[str, Any], VoteUpdateSpec, VoteUpdateSpec]:
    """Explicit 2-row fixture: singleton apply drifts to flat_index=1 under T=10."""

    q = torch.tensor([0, 0], dtype=torch.int8)
    acc = torch.tensor([0, 50], dtype=torch.int16)
    state = make_bounded_tensor_state(state_key, q, 1.0, acc)
    votes = torch.tensor([12, 0], dtype=torch.int16)
    base_spec = w6_t10_base_spec(max_abs_per_tensor=4096)
    one_flip_spec = _single_flip_spec(base_spec)
    candidate = {
        "candidate_id": f"{state_key}:0",
        "state_key": state_key,
        "flat_index": 0,
        "current_q_level": 0,
        "proposal_direction": 1,
        "new_acc_i32_signed": 12,
        "deterministic_hash_rank_position": 0,
        "current_rank_position": 0,
        "vote_value": 12,
        "pre_accumulator_i16": 0,
    }
    audit = _audit_sparse_singleton_identity_for_candidate(
        tensor_state=state,
        votes=votes,
        candidate=candidate,
        one_flip_spec=one_flip_spec,
    )
    assert audit["drifted"] is True
    assert audit["applied_indices"] == [1]
    return state, votes, candidate, base_spec, one_flip_spec


def test_proxy_delta_weight_uses_w6_t10_one_flip_spec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_key = "toy.proj"
    state, votes, candidate, expected_spec, _one_flip_spec = _identity_hold_w6_t10_bundle(
        state_key
    )
    observed_thresholds: list[int] = []
    original_apply = apply_integer_vote_update_reference

    def _spy_apply(vote_state, inputs, spec, **kwargs):
        observed_thresholds.append(int(spec.threshold_abs))
        return original_apply(vote_state, inputs, spec, **kwargs)

    monkeypatch.setattr(
        "calm.hrm_text_158.native_full_stack.grad_proxy_audit.apply_integer_vote_update_reference",
        _spy_apply,
    )
    weight = derive_w6_t10_candidate_delta_weight(
        tensor_state=state,
        votes=votes,
        candidate=candidate,
        max_abs_per_tensor=4096,
    )
    expected = _delta_weight_from_spec(
        state=state,
        votes=votes,
        candidate=candidate,
        base_spec=expected_spec,
    )
    assert observed_thresholds == [10]
    assert weight == pytest.approx(expected)


def test_grad_proxy_audit_fail_closed_on_singleton_identity_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, batch, eligible, _states = _tiny_forward_fixture(batch_size=8)
    state_key = next(iter(eligible))
    state, votes, candidate, base_spec, one_flip_spec = _singleton_drift_w6_t10_bundle(
        state_key
    )
    states = {state_key: state}
    votes_by_key = {state_key: votes}

    monkeypatch.setattr(
        "calm.hrm_text_158.native_full_stack.grad_proxy_audit.build_w6_t10_crossing_candidate_universe_from_votes",
        lambda **kwargs: {
            "base_spec": base_spec,
            "one_flip_spec": one_flip_spec,
            "votes_by_key": votes_by_key,
            "candidate_by_id": {candidate["candidate_id"]: candidate},
            "sampled_ids": [candidate["candidate_id"]],
            "crossing_eligible_count": 1,
        },
    )
    monkeypatch.setattr(
        "calm.hrm_text_158.native_full_stack.grad_proxy_audit._compute_baseline_votes",
        lambda *args, **kwargs: (1.0, votes_by_key),
    )

    with pytest.raises(GradProxyAuditAborted, match="singleton_identity_drift"):
        run_grad_proxy_audit_step1(
            model=model,
            batch=batch,
            tensor_states=states,
            eligible_modules=eligible,
            device=torch.device("cpu"),
            extras=model.compute_train_extra_args(1, 1),
            max_abs_per_tensor=4096,
            max_audit_candidates=1,
            launch_sha="testsha",
        )


def test_grad_proxy_audit_comparator_spec_mismatch_blocks_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, batch, eligible, _states = _tiny_forward_fixture(batch_size=8)
    state_key = next(iter(eligible))
    state, votes, candidate, base_spec, one_flip_spec = _identity_hold_w6_t10_bundle(
        state_key
    )
    states = {state_key: state}
    votes_by_key = {state_key: votes}

    monkeypatch.setattr(
        "calm.hrm_text_158.native_full_stack.grad_proxy_audit.build_w6_t10_crossing_candidate_universe_from_votes",
        lambda **kwargs: {
            "base_spec": base_spec,
            "one_flip_spec": one_flip_spec,
            "votes_by_key": votes_by_key,
            "candidate_by_id": {candidate["candidate_id"]: candidate},
            "sampled_ids": [candidate["candidate_id"]],
            "crossing_eligible_count": 1,
        },
    )
    monkeypatch.setattr(
        "calm.hrm_text_158.native_full_stack.grad_proxy_audit._compute_baseline_votes",
        lambda *args, **kwargs: (1.0, votes_by_key),
    )
    monkeypatch.setattr(
        "calm.hrm_text_158.native_full_stack.grad_proxy_audit._compute_activation_credit_candidate_proxies",
        lambda **kwargs: {
            "grad_proxy_by_candidate_id": {candidate["candidate_id"]: 0.25},
        },
    )
    monkeypatch.setattr(
        "calm.hrm_text_158.native_full_stack.grad_proxy_audit._evaluate_loss",
        lambda *args, **kwargs: 1.1,
    )
    identity_audits: list[dict[str, Any]] = []
    original_identity_audit = _audit_sparse_singleton_identity_for_candidate

    def _record_identity_audit(**kwargs):
        audit = original_identity_audit(**kwargs)
        identity_audits.append(dict(audit))
        return audit

    monkeypatch.setattr(
        "calm.hrm_text_158.native_full_stack.grad_proxy_audit._audit_sparse_singleton_identity_for_candidate",
        _record_identity_audit,
    )

    receipt = run_grad_proxy_audit_step1(
        model=model,
        batch=batch,
        tensor_states=states,
        eligible_modules=eligible,
        device=torch.device("cpu"),
        extras=model.compute_train_extra_args(1, 1),
        max_abs_per_tensor=4096,
        max_audit_candidates=1,
        launch_sha="testsha",
        comparator_spec="legacy_threshold_abs_1",
    )
    assert len(identity_audits) == 1
    assert identity_audits[0]["drifted"] is False
    assert identity_audits[0]["applied_indices"] == [0]
    assert receipt["comparator_spec"] != GRAD_PROXY_AUDIT_COMPARATOR_SPEC
    assert receipt["comparator_spec_mismatch"] is True
    assert receipt["pass_bars"] is None
    assert len(receipt["per_candidate"]) == 1
    row = receipt["per_candidate"][0]
    assert row["local_loss_delta_proxy"] == pytest.approx(0.25)
    assert row["local_loss_delta_shadow"] == pytest.approx(0.1)

    bars = compute_grad_proxy_pass_bars(
        per_candidate=[
            {
                "candidate_id": "a:0",
                "local_loss_delta_proxy": -0.2,
                "local_loss_delta_shadow": -0.1,
            },
            {
                "candidate_id": "a:1",
                "local_loss_delta_proxy": 0.3,
                "local_loss_delta_shadow": 0.4,
            },
            {
                "candidate_id": "a:2",
                "local_loss_delta_proxy": 0.1,
                "local_loss_delta_shadow": 0.0,
            },
        ]
    )
    assert "kendall_tau" in bars
    assert bars["top8_overlap"] == pytest.approx(1.0)
