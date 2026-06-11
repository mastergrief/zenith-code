"""CPU tests for W6/T=10 grad-proxy audit (slice 3a)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    VoteUpdateSpec,
    apply_bounded_delta_vote_step,
    make_bounded_tensor_state,
)
from calm.hrm_text_158.native_full_stack.grad_proxy_audit import (
    DRIFT_AUDIT_SAMPLE_COUNT,
    DRIFT_AUDIT_STEP_INTERVAL,
    GRAD_PROXY_AUDIT_ABORT_NAME,
    GRAD_PROXY_AUDIT_ABORT_REASON,
    GRAD_PROXY_AUDIT_COMPARATOR_SPEC,
    GRAD_PROXY_AUDIT_ESTIMAND,
    GRAD_PROXY_AUDIT_STATE_SOURCE,
    POPULATION_MODE_FULL_CROSSING_ELIGIBLE,
    POPULATION_MODE_SAMPLED_K64,
    GradProxyAuditAborted,
    GradProxyAuditWarmupCapAborted,
    assert_local_loss_delta_proxy_coverage,
    build_grad_proxy_local_loss_delta_by_key,
    build_w6_t10_crossing_candidate_universe_from_votes,
    compute_grad_proxy_pass_bars,
    count_w6_t10_crossing_eligible_from_votes,
    derive_probe_science_arm_votes,
    discover_probe_warmup_audit_anchor,
    run_grad_proxy_audit_at_anchor,
    run_grad_proxy_audit_with_warmup,
    run_proxy_oracle_drift_audit,
    w6_t10_base_spec,
    write_grad_proxy_audit_abort_receipt,
)
from calm.hrm_text_158.native_full_stack.two_tier_transient_selection import (
    LOCAL_SELECTION_ORDER_TRANSIENT_LOCAL_LOSS_DELTA,
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
from scripts.hrm_text_158_bounded_delta_acquisition_probe import (
    DEFAULT_PARENT,
    DEFAULT_PARENT_SHA256,
    build_identity_full_support_batches,
    build_model_from_checkpoint,
    derive_tensor_states_and_check_init_fidelity,
    load_parent_checkpoint,
    select_eligible_bitlinears,
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


def test_proxy_delta_weight_uses_w6_t10_one_flip_spec() -> None:
    state_key = "toy.proj"
    state, votes, candidate, expected_spec, _one_flip_spec = _identity_hold_w6_t10_bundle(
        state_key
    )
    from calm.hrm_text_158.native_full_stack.grad_proxy_audit import (
        derive_w6_t10_candidate_delta_weight,
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
    assert int(expected_spec.threshold_abs) == 10
    assert weight == expected


def test_full_pop_vectorized_ingress_scatter_scalar_parity() -> None:
    from calm.hrm_text_158.native_full_stack.grad_proxy_audit import (
        _crossing_flat_indices_by_state_key,
        _scatter_vectorized_grad_proxy_ingress_for_state,
        derive_w6_t10_candidate_delta_weight,
    )

    state_key = "toy.proj"
    q = torch.tensor([0, 0, 0, -1], dtype=torch.int8)
    acc = torch.tensor([0, 50, -50, 0], dtype=torch.int16)
    state = make_bounded_tensor_state(state_key, q, 1.0, acc)
    votes = torch.tensor([12, 12, -12, 12], dtype=torch.int16)
    tensor_states = {state_key: state}
    votes_by_key = {state_key: votes}
    flat_by_state = _crossing_flat_indices_by_state_key(
        tensor_states=tensor_states,
        votes_by_key=votes_by_key,
    )
    flat_indices = flat_by_state[state_key]
    grad_proxies = torch.tensor(
        [0.25, -0.5, 0.75, 0.125][: int(flat_indices.numel())],
        dtype=torch.float32,
    )
    vectorized = torch.full((4,), float("nan"), dtype=torch.float32)
    _scatter_vectorized_grad_proxy_ingress_for_state(
        local_loss_delta=vectorized,
        tensor_state=state,
        votes=votes,
        flat_indices=flat_indices,
        grad_proxies=grad_proxies,
        max_abs_per_tensor=4096,
    )
    scalar = torch.full((4,), float("nan"), dtype=torch.float32)
    universe = build_w6_t10_crossing_candidate_universe_from_votes(
        tensor_states=tensor_states,
        votes_by_key=votes_by_key,
        max_abs_per_tensor=4096,
        max_sampled_candidates=64,
        population_mode=POPULATION_MODE_FULL_CROSSING_ELIGIBLE,
    )
    candidate_by_id = universe["candidate_by_id"]
    for offset, flat_index in enumerate(flat_indices.tolist()):
        candidate_id = f"{state_key}:{int(flat_index)}"
        candidate = candidate_by_id[candidate_id]
        delta_weight = derive_w6_t10_candidate_delta_weight(
            tensor_state=state,
            votes=votes,
            candidate=candidate,
            max_abs_per_tensor=4096,
        )
        scalar.view(-1)[int(flat_index)] = float(grad_proxies[offset] * delta_weight)
    assert torch.equal(vectorized, scalar)


def test_vectorized_delta_weights_exact_scalar_parity() -> None:
    from calm.hrm_text_158.native_full_stack.grad_proxy_audit import (
        _vectorized_w6_t10_delta_weights_at_flat_indices,
    )

    for bundle in (_identity_hold_w6_t10_bundle, _singleton_drift_w6_t10_bundle):
        state_key = "toy.proj"
        state, votes, candidate, base_spec, _one_flip_spec = bundle(state_key)
        flat_index = int(candidate["flat_index"])
        flat_indices = torch.tensor([flat_index], dtype=torch.int64)
        vectorized = _vectorized_w6_t10_delta_weights_at_flat_indices(
            tensor_state=state,
            votes=votes,
            flat_indices=flat_indices,
            max_abs_per_tensor=4096,
        )
        scalar = torch.tensor(
            [
                _delta_weight_from_spec(
                    state=state,
                    votes=votes,
                    candidate=candidate,
                    base_spec=base_spec,
                )
            ],
            dtype=torch.float32,
        )
        assert torch.equal(vectorized, scalar)


def test_tensorized_crossing_count_matches_row_loop_reducer() -> None:
    from calm.hrm_text_158.native_full_stack.grad_proxy_audit import (
        count_w6_t10_crossing_eligible_from_votes,
        materialize_selector_rows,
    )
    from calm.hrm_text_158.native_full_stack.two_tier_transient_selection import (
        crossing_eligible_flat_indices,
    )

    state_key = "toy.proj"
    q = torch.tensor([0, 0, 0, -1], dtype=torch.int8)
    acc = torch.tensor([0, 50, -50, 0], dtype=torch.int16)
    state = make_bounded_tensor_state(state_key, q, 1.0, acc)
    votes = torch.tensor([12, 12, -12, 12], dtype=torch.int16)
    tensor_states = {state_key: state}
    votes_by_key = {state_key: votes}
    tensorized = count_w6_t10_crossing_eligible_from_votes(
        tensor_states=tensor_states,
        votes_by_key=votes_by_key,
    )
    rows = materialize_selector_rows(votes=votes, state=state)
    row_loop = len(crossing_eligible_flat_indices(rows))
    assert tensorized == row_loop
    assert tensorized == 4


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

    with pytest.raises(GradProxyAuditAborted, match="singleton_identity_drift"):
        run_grad_proxy_audit_at_anchor(
            model=model,
            batch=batch,
            tensor_states=states,
            votes_by_key=votes_by_key,
            baseline_loss=1.0,
            eligible_modules=eligible,
            device=torch.device("cpu"),
            extras=model.compute_train_extra_args(1, 1),
            max_abs_per_tensor=4096,
            max_audit_candidates=1,
            launch_sha="testsha",
            audit_step_index=3,
            audit_warmup_steps_run=2,
            crossing_eligible_count_by_step=[0, 0, 1],
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

    receipt = run_grad_proxy_audit_at_anchor(
        model=model,
        batch=batch,
        tensor_states=states,
        votes_by_key=votes_by_key,
        baseline_loss=1.0,
        eligible_modules=eligible,
        device=torch.device("cpu"),
        extras=model.compute_train_extra_args(1, 1),
        max_abs_per_tensor=4096,
        max_audit_candidates=1,
        launch_sha="testsha",
        audit_step_index=3,
        audit_warmup_steps_run=2,
        crossing_eligible_count_by_step=[0, 0, 1],
        comparator_spec="legacy_threshold_abs_1",
    )
    assert len(identity_audits) == 1
    assert identity_audits[0]["drifted"] is False
    assert identity_audits[0]["applied_indices"] == [0]
    assert receipt["comparator_spec"] != GRAD_PROXY_AUDIT_COMPARATOR_SPEC
    assert receipt["comparator_spec_mismatch"] is True
    assert receipt["pass_bars"] is None
    assert receipt["audit_state_source"] == GRAD_PROXY_AUDIT_STATE_SOURCE
    assert receipt["warmup_two_tier_enabled"] is False
    assert receipt["audit_step_index"] == 3
    assert receipt["audit_warmup_steps_run"] == 2
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


def test_fresh_parent_step1_has_zero_w6_crossings() -> None:
    model, batch, eligible, states = _tiny_forward_fixture(batch_size=8)
    device = torch.device("cpu")
    extras = model.compute_train_extra_args(1, 1)
    _baseline_loss, votes_by_key = derive_probe_science_arm_votes(
        model=model,
        batch=batch,
        tensor_states=states,
        eligible_modules=eligible,
        device=device,
        extras=extras,
        max_abs_per_tensor=4096,
    )
    assert (
        count_w6_t10_crossing_eligible_from_votes(
            tensor_states=states,
            votes_by_key=votes_by_key,
        )
        == 0
    )


def test_write_grad_proxy_audit_abort_receipt(tmp_path: Path) -> None:
    abort_path = write_grad_proxy_audit_abort_receipt(
        artifact_dir=tmp_path,
        crossing_eligible_count_by_step=[0, 0, 0],
        warmup_steps_run=3,
        launch_sha="launchsha",
        parent_sha256="parentsha",
    )
    assert abort_path.endswith(GRAD_PROXY_AUDIT_ABORT_NAME)
    payload = json.loads(Path(abort_path).read_text(encoding="utf-8"))
    assert payload["reason"] == GRAD_PROXY_AUDIT_ABORT_REASON
    assert payload["crossing_eligible_count_by_step"] == [0, 0, 0]
    assert payload["audit_state_source"] == GRAD_PROXY_AUDIT_STATE_SOURCE
    assert payload["warmup_steps_run"] == 3
    assert payload["warmup_two_tier_enabled"] is False
    assert payload["launch_sha"] == "launchsha"
    assert payload["parent_sha256"] == "parentsha"


def test_warmup_cap_abort_is_typed_with_telemetry() -> None:
    exc = GradProxyAuditWarmupCapAborted(
        crossing_eligible_count_by_step=[0, 0],
        warmup_steps_run=2,
        launch_sha="launchsha",
        parent_sha256="parentsha",
    )
    assert exc.crossing_eligible_count_by_step == [0, 0]
    assert exc.warmup_steps_run == 2
    assert exc.launch_sha == "launchsha"
    assert exc.parent_sha256 == "parentsha"


def test_warmup_cap_abort_when_no_crossings_within_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, batch, eligible, states = _tiny_forward_fixture(batch_size=8)
    monkeypatch.setattr(
        "calm.hrm_text_158.native_full_stack.grad_proxy_audit.count_w6_t10_crossing_eligible_from_votes",
        lambda **kwargs: 0,
    )

    with pytest.raises(GradProxyAuditWarmupCapAborted) as exc_info:
        run_grad_proxy_audit_with_warmup(
            model=model,
            batch=batch,
            tensor_states=states,
            eligible_modules=eligible,
            device=torch.device("cpu"),
            extras=model.compute_train_extra_args(1, 1),
            max_abs_per_tensor=4096,
            max_audit_candidates=1,
            launch_sha="launchsha",
            parent_sha256="parentsha",
            warmup_max_steps=2,
        )
    assert exc_info.value.crossing_eligible_count_by_step == [0, 0]
    assert exc_info.value.warmup_steps_run == 2


@pytest.mark.skipif(
    not Path(DEFAULT_PARENT).exists(),
    reason="pinned parent checkpoint required for warm-up determinism pin",
)
def test_warmup_determinism_pin_parent_seed44() -> None:
    device = torch.device("cpu")
    ckpt, _parent_sha = load_parent_checkpoint(
        Path(DEFAULT_PARENT),
        expected_sha256=DEFAULT_PARENT_SHA256,
    )
    model, tok, cfg = build_model_from_checkpoint(ckpt, device)
    support_batches, _support_proof = build_identity_full_support_batches(
        tok=tok,
        max_len=int(cfg.max_seq_len),
        batch_size=1,
        curriculum_seed=44,
        device=device,
        support_order_seed=44,
    )
    batch = support_batches[0]["batch"]
    eligible = select_eligible_bitlinears(model, eligible_scope="first-bitlinear")
    tensor_states, init_report = derive_tensor_states_and_check_init_fidelity(
        eligible,
        threshold=0.0,
    )
    assert init_report["all_pass"] is True
    model.train()
    extras = model.compute_train_extra_args(1, 1)

    def _discover_once():
        return discover_probe_warmup_audit_anchor(
            model=model,
            batch=batch,
            tensor_states=tensor_states,
            eligible_modules=eligible,
            device=device,
            extras=extras,
            max_abs_per_tensor=4096,
            launch_sha="determinism-pin",
            warmup_max_steps=8,
        )

    first = _discover_once()
    second = _discover_once()
    assert first.audit_step_index == 3
    assert second.audit_step_index == 3
    assert first.audit_warmup_steps_run == 2
    assert second.audit_warmup_steps_run == 2
    assert first.crossing_eligible_count == second.crossing_eligible_count
    assert first.crossing_eligible_count > 0
    assert first.crossing_eligible_count_by_step == second.crossing_eligible_count_by_step
    assert first.crossing_eligible_count_by_step[:2] == (0, 0)
    assert first.crossing_eligible_count_by_step[2] > 0


def test_population_mode_fork_full_vs_sampled_k64() -> None:
    state_key = "toy.proj"
    state, votes, candidate, _base_spec, _one_flip_spec = _identity_hold_w6_t10_bundle(
        state_key
    )
    tensor_states = {state_key: state}
    votes_by_key = {state_key: votes}
    full_universe = build_w6_t10_crossing_candidate_universe_from_votes(
        tensor_states=tensor_states,
        votes_by_key=votes_by_key,
        max_abs_per_tensor=4096,
        max_sampled_candidates=64,
        population_mode=POPULATION_MODE_FULL_CROSSING_ELIGIBLE,
    )
    sampled_universe = build_w6_t10_crossing_candidate_universe_from_votes(
        tensor_states=tensor_states,
        votes_by_key=votes_by_key,
        max_abs_per_tensor=4096,
        max_sampled_candidates=64,
        population_mode=POPULATION_MODE_SAMPLED_K64,
    )
    assert full_universe["population_mode"] == POPULATION_MODE_FULL_CROSSING_ELIGIBLE
    assert sampled_universe["population_mode"] == POPULATION_MODE_SAMPLED_K64
    assert full_universe["sampled_ids"] == [candidate["candidate_id"]]
    assert sampled_universe["sampled_ids"] == [candidate["candidate_id"]]


def test_assert_local_loss_delta_proxy_coverage_fail_closed() -> None:
    state_key = "toy.proj"
    q = torch.tensor([0, 0], dtype=torch.int8)
    acc = torch.zeros(2, dtype=torch.int16)
    state = make_bounded_tensor_state(state_key, q, 0.5, acc)
    votes = torch.tensor([12, 12], dtype=torch.int16)
    incomplete = {"toy.proj": torch.full((2,), float("nan"), dtype=torch.float32)}
    incomplete["toy.proj"][0] = -0.1
    with pytest.raises(ValueError, match="local_loss_delta_proxy_incomplete_coverage"):
        assert_local_loss_delta_proxy_coverage(
            local_loss_delta_by_key=incomplete,
            tensor_states={state_key: state},
            votes_by_key={state_key: votes},
        )


def test_build_grad_proxy_local_loss_delta_by_key_receipt_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, batch, eligible, _states = _tiny_forward_fixture(batch_size=8)
    state_key = next(iter(eligible))
    state, votes, candidate, _base_spec, _one_flip_spec = _identity_hold_w6_t10_bundle(
        state_key
    )
    tensor_states = {state_key: state}
    votes_by_key = {state_key: votes}
    monkeypatch.setattr(
        "calm.hrm_text_158.native_full_stack.grad_proxy_audit._compute_activation_credit_candidate_proxies",
        lambda **kwargs: {
            "grad_proxy_by_candidate_id": {candidate["candidate_id"]: 0.5},
        },
    )
    local_loss_delta_by_key, ingress_receipt = build_grad_proxy_local_loss_delta_by_key(
        model=model,
        batch=batch,
        tensor_states=tensor_states,
        eligible_modules=eligible,
        device=torch.device("cpu"),
        extras=model.compute_train_extra_args(1, 1),
        votes_by_key=votes_by_key,
        max_abs_per_tensor=4096,
        population_mode=POPULATION_MODE_FULL_CROSSING_ELIGIBLE,
        optimizer_step_index=3,
    )
    assert ingress_receipt["grad_proxy_ingress_enabled"] is True
    assert ingress_receipt["grad_proxy_ingress_estimand"] == GRAD_PROXY_AUDIT_ESTIMAND
    assert (
        ingress_receipt["grad_proxy_ingress_population_mode"]
        == POPULATION_MODE_FULL_CROSSING_ELIGIBLE
    )
    assert ingress_receipt["grad_proxy_ingress_crossing_eligible_count"] == 1
    assert ingress_receipt["grad_proxy_ingress_candidate_count_ingressed"] == 1
    assert ingress_receipt["candidate_count_ingressed"] == 1
    assert ingress_receipt["optimizer_step_index"] == 3
    assert torch.isfinite(local_loss_delta_by_key[state_key].view(-1)[0]).item()
    assert float(local_loss_delta_by_key[state_key].view(-1)[1].item()) == 0.0


def test_run_proxy_oracle_drift_audit_locked_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, batch, eligible, _states = _tiny_forward_fixture(batch_size=8)
    state_key = next(iter(eligible))
    state, votes, candidate, _base_spec, _one_flip_spec = _identity_hold_w6_t10_bundle(
        state_key
    )
    tensor_states = {state_key: state}
    votes_by_key = {state_key: votes}
    monkeypatch.setattr(
        "calm.hrm_text_158.native_full_stack.grad_proxy_audit._compute_activation_credit_candidate_proxies",
        lambda **kwargs: {
            "grad_proxy_by_candidate_id": {candidate["candidate_id"]: 0.5},
        },
    )
    local_loss_delta_by_key, _ingress = build_grad_proxy_local_loss_delta_by_key(
        model=model,
        batch=batch,
        tensor_states=tensor_states,
        eligible_modules=eligible,
        device=torch.device("cpu"),
        extras=model.compute_train_extra_args(1, 1),
        votes_by_key=votes_by_key,
        max_abs_per_tensor=4096,
        population_mode=POPULATION_MODE_FULL_CROSSING_ELIGIBLE,
    )
    monkeypatch.setattr(
        "calm.hrm_text_158.native_full_stack.grad_proxy_audit._shadow_local_loss_deltas_for_candidates",
        lambda **kwargs: [
            {
                "candidate_id": candidate["candidate_id"],
                "state_key": state_key,
                "flat_index": 0,
                "grad_proxy": 0.5,
                "candidate_delta_weight": 1.0,
                "local_loss_delta_proxy": 0.5,
                "local_loss_delta_shadow": 0.4,
            }
        ],
    )
    drift = run_proxy_oracle_drift_audit(
        model=model,
        batch=batch,
        tensor_states=tensor_states,
        eligible_modules=eligible,
        device=torch.device("cpu"),
        extras=model.compute_train_extra_args(1, 1),
        votes_by_key=votes_by_key,
        local_loss_delta_by_key=local_loss_delta_by_key,
        baseline_loss=1.0,
        max_abs_per_tensor=4096,
        optimizer_step_index=5,
        drift_sample_count=DRIFT_AUDIT_SAMPLE_COUNT,
    )
    assert drift["proxy_oracle_drift_step"] == 5
    assert drift["proxy_oracle_drift_sample_count"] <= DRIFT_AUDIT_SAMPLE_COUNT
    assert drift["proxy_oracle_drift_gating"] is False
    assert drift["proxy_oracle_drift_comparator_spec"] == GRAD_PROXY_AUDIT_COMPARATOR_SPEC
    assert drift["proxy_oracle_drift_estimand"] == GRAD_PROXY_AUDIT_ESTIMAND
    assert "proxy_oracle_drift_tau" in drift
    assert "proxy_oracle_drift_sign_agreement" in drift
    assert "proxy_oracle_drift_top8_overlap" in drift


def _tensor_state_snapshot(states: dict[str, Any]) -> dict[str, dict[str, list[int]]]:
    out: dict[str, dict[str, list[int]]] = {}
    for state_key, state in states.items():
        vote_state = state.vote_update_state()
        out[state_key] = {
            "q_levels": vote_state.q_levels.detach().cpu().flatten().tolist(),
            "accumulators": vote_state.accumulators.detach().cpu().flatten().tolist(),
        }
    return out


def test_drift_audit_trajectory_invariant_ten_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, batch, eligible, _states = _tiny_forward_fixture(batch_size=8)
    state_key = next(iter(eligible))
    state, votes, candidate, _base_spec, _one_flip_spec = _identity_hold_w6_t10_bundle(
        state_key
    )
    device = torch.device("cpu")
    extras = model.compute_train_extra_args(1, 1)
    vote_spec = w6_t10_base_spec(max_abs_per_tensor=4096)
    vote_specs = {state_key: vote_spec}
    fixed_local_loss_delta_by_key = {
        state_key: torch.tensor([-0.25, 0.0], dtype=torch.float32),
    }
    monkeypatch.setattr(
        "calm.hrm_text_158.native_full_stack.grad_proxy_audit._shadow_local_loss_deltas_for_candidates",
        lambda **kwargs: [
            {
                "candidate_id": candidate["candidate_id"],
                "state_key": state_key,
                "flat_index": 0,
                "grad_proxy": 0.25,
                "candidate_delta_weight": 1.0,
                "local_loss_delta_proxy": 0.25,
                "local_loss_delta_shadow": 0.2,
            }
        ],
    )

    def _run_trajectory(*, drift_enabled: bool) -> tuple[dict[str, Any], list[list[int]]]:
        tensor_states = {state_key: make_bounded_tensor_state(
            state_key,
            state.vote_update_state().q_levels.clone(),
            float(state.frozen_scale.detach().cpu().item()),
            state.vote_update_state().accumulators.clone(),
        )}
        votes_by_key = {state_key: votes.clone()}
        applied_flips: list[list[int]] = []
        for step in range(1, 11):
            local_loss_delta_by_key = {
                key: tensor.clone()
                for key, tensor in fixed_local_loss_delta_by_key.items()
            }
            if drift_enabled and int(step) % int(DRIFT_AUDIT_STEP_INTERVAL) == 0:
                run_proxy_oracle_drift_audit(
                    model=model,
                    batch=batch,
                    tensor_states=tensor_states,
                    eligible_modules=eligible,
                    device=device,
                    extras=extras,
                    votes_by_key=votes_by_key,
                    local_loss_delta_by_key=local_loss_delta_by_key,
                    baseline_loss=1.0,
                    max_abs_per_tensor=4096,
                    optimizer_step_index=int(step),
                )
            step_result = apply_bounded_delta_vote_step(
                tensor_states,
                votes_by_key,
                vote_specs,
                local_selection_ordering_mode=LOCAL_SELECTION_ORDER_TRANSIENT_LOCAL_LOSS_DELTA,
                local_selection_ordering_seed=17,
                local_selection_ordering_step=int(step),
                two_tier_carry_w6_enabled=True,
                local_loss_delta_by_key=local_loss_delta_by_key,
            )
            tensor_states = step_result.tensor_states
            applied = []
            for key, stats in step_result.tensor_stats.items():
                applied.extend(int(idx) for idx in stats.get("applied_indices", []))
            applied_flips.append(sorted(applied))
        return _tensor_state_snapshot(tensor_states), applied_flips

    off_snapshot, off_flips = _run_trajectory(drift_enabled=False)
    on_snapshot, on_flips = _run_trajectory(drift_enabled=True)
    assert on_snapshot == off_snapshot
    assert on_flips == off_flips
