"""M4 probe sparse vote ingress v1 — construction-seam sparse events + guards."""
from __future__ import annotations

import time
from typing import Any

import pytest
import torch

from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    apply_bounded_delta_vote_step,
    credit_from_weighted_grad,
    default_dry_run_rank_vote_spec,
    make_event_coded_live_tensor_state,
    project_s1_gradient_to_moves,
    rank_bucketed_int16_votes,
    rank_bucketed_int16_votes_and_sparse_events,
    sign_pressure_int16_votes,
    sign_pressure_int16_votes_and_sparse_events,
    sparse_rank_bucketed_int16_vote_events,
    sparse_sign_pressure_int16_vote_events,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap import GlobalRateCapSpec
from calm.hrm_text_158.native_full_stack.event_coded_vote_update_adapter import (
    C8_TRANSIENT_DENSE_COMPUTE_NUMEL_KEY,
    carrier_content_sha256,
)
from calm.hrm_text_158.native_full_stack.sparse_vote_events import SparseVoteEvents
from calm.hrm_text_158.native_full_stack.vote_update import VoteUpdateSpec


def _vote_spec(*, threshold_abs: int = 8) -> VoteUpdateSpec:
    return VoteUpdateSpec(
        threshold_abs=int(threshold_abs),
        accumulator_clip_min=-127,
        accumulator_clip_max=127,
        max_abs_per_tensor=16,
    )


def _rank_fixture():
    rank_spec = default_dry_run_rank_vote_spec()
    q = torch.tensor([[0, 1, -1, 0]], dtype=torch.int8)
    weighted_grad = torch.tensor([[-1.0, -2.0, 3.0, 4.0]])
    moves = project_s1_gradient_to_moves(weighted_grad, q)
    credit = credit_from_weighted_grad(weighted_grad)
    return credit, moves, rank_spec


def _sign_fixture(*, inverted: bool = False):
    vote_spec = _vote_spec(threshold_abs=1)
    q = torch.tensor([[0, 1, -1, 0]], dtype=torch.int8)
    weighted_grad = torch.tensor([[-1.0, -2.0, 3.0, 4.0]])
    moves = project_s1_gradient_to_moves(weighted_grad, q)
    return moves, vote_spec, inverted


def test_combined_rank_votes_and_sparse_match_separate_paths() -> None:
    credit, moves, rank_spec = _rank_fixture()
    combined_votes, combined_sparse = rank_bucketed_int16_votes_and_sparse_events(
        credit,
        moves,
        rank_spec,
    )
    separate_votes = rank_bucketed_int16_votes(credit, moves, rank_spec)
    separate_sparse = sparse_rank_bucketed_int16_vote_events(credit, moves, rank_spec)
    assert torch.equal(combined_votes, separate_votes)
    assert combined_sparse.to_dict() == separate_sparse.to_dict()


def test_combined_sign_votes_and_sparse_match_separate_paths() -> None:
    moves, vote_spec, inverted = _sign_fixture(inverted=False)
    combined_votes, combined_sparse = sign_pressure_int16_votes_and_sparse_events(
        moves,
        vote_spec,
        inverted=inverted,
    )
    separate_votes = sign_pressure_int16_votes(moves, vote_spec, inverted=inverted)
    separate_sparse = sparse_sign_pressure_int16_vote_events(
        moves,
        vote_spec,
        inverted=inverted,
    )
    assert torch.equal(combined_votes, separate_votes)
    assert combined_sparse.to_dict() == separate_sparse.to_dict()


def test_combined_rank_builder_single_candidate_compute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Combined path = one _compute_rank_bucketed_candidate_votes; separate = two."""
    import calm.hrm_text_158.native_full_stack.bounded_delta_learner as learner_mod

    credit, moves, rank_spec = _rank_fixture()
    calls = {"n": 0}
    original = learner_mod._compute_rank_bucketed_candidate_votes

    def _counting(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        learner_mod,
        "_compute_rank_bucketed_candidate_votes",
        _counting,
    )
    rank_bucketed_int16_votes_and_sparse_events(credit, moves, rank_spec)
    assert calls["n"] == 1

    calls["n"] = 0
    rank_bucketed_int16_votes(credit, moves, rank_spec)
    sparse_rank_bucketed_int16_vote_events(credit, moves, rank_spec)
    assert calls["n"] == 2


def test_probe_vote_construction_uses_combined_not_separate_builders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Probe hot path must call combined builders, not separate dense+sparse scans."""
    import scripts.hrm_text_158_bounded_delta_acquisition_probe as probe_mod
    from scripts.hrm_text_158_bounded_delta_acquisition_probe import (
        ARM_A0_RANK_BUCKET_CURRENT,
        _weighted_grads_to_science_arm_votes,
        default_vote_update_spec,
    )

    def _forbid_separate_dense(*args, **kwargs):
        raise AssertionError("separate rank_bucketed_int16_votes must not run on probe hot path")

    def _forbid_separate_sparse(*args, **kwargs):
        raise AssertionError(
            "separate sparse_rank_bucketed_int16_vote_events must not run on probe hot path"
        )

    monkeypatch.setattr(probe_mod, "rank_bucketed_int16_votes", _forbid_separate_dense)
    monkeypatch.setattr(
        probe_mod,
        "sparse_rank_bucketed_int16_vote_events",
        _forbid_separate_sparse,
    )
    q = torch.tensor([[0, 1, -1, 0]], dtype=torch.int8)
    weighted_grad = torch.tensor([[-1.0, -2.0, 3.0, 4.0]])
    tensor_states = {"toy.proj": type("State", (), {"q_levels": q})()}
    rank_spec = default_dry_run_rank_vote_spec()
    vote_spec = default_vote_update_spec(16)
    sparse_out: dict[str, Any] = {}
    _weighted_grads_to_science_arm_votes(
        {"toy.proj": weighted_grad},
        tensor_states,
        rank_spec=rank_spec,
        vote_spec=vote_spec,
        science_arm=str(ARM_A0_RANK_BUCKET_CURRENT),
        sparse_events_out=sparse_out,
    )
    assert "toy.proj" in sparse_out


def test_sparse_rank_events_match_dense_oracle() -> None:
    credit, moves, rank_spec = _rank_fixture()
    sparse = sparse_rank_bucketed_int16_vote_events(credit, moves, rank_spec)
    dense = rank_bucketed_int16_votes(credit, moves, rank_spec)
    oracle = SparseVoteEvents.from_dense_votes(dense)
    assert sparse.to_dict() == oracle.to_dict()


def test_sparse_sign_events_match_dense_oracle() -> None:
    moves, vote_spec, inverted = _sign_fixture(inverted=False)
    sparse = sparse_sign_pressure_int16_vote_events(moves, vote_spec, inverted=inverted)
    dense = sign_pressure_int16_votes(moves, vote_spec, inverted=inverted)
    oracle = SparseVoteEvents.from_dense_votes(dense)
    assert sparse.to_dict() == oracle.to_dict()


def test_sparse_sign_events_match_dense_oracle_inverted() -> None:
    moves, vote_spec, inverted = _sign_fixture(inverted=True)
    sparse = sparse_sign_pressure_int16_vote_events(moves, vote_spec, inverted=inverted)
    dense = sign_pressure_int16_votes(moves, vote_spec, inverted=inverted)
    oracle = SparseVoteEvents.from_dense_votes(dense)
    assert sparse.to_dict() == oracle.to_dict()


def _step_summary(result) -> dict[str, object]:
    state = result.tensor_states["toy.proj"]
    stats = result.tensor_stats["toy.proj"]
    return {
        "q": tuple(int(x) for x in state.q_levels.flatten().tolist()),
        "flip_count": int(stats.get("flip_count", -1)),
        "cap_enabled": bool(result.global_summary.get("global_rate_cap_enabled")),
    }


def test_apply_step_bit_exact_sparse_vs_fallback() -> None:
    q = torch.zeros((4, 4), dtype=torch.int8)
    state = make_event_coded_live_tensor_state("toy.proj", q, 0.25, demotion_band=1)
    votes = torch.zeros((4, 4), dtype=torch.int16)
    votes.view(-1)[[0, 3, 7]] = torch.tensor([12, -9, 6], dtype=torch.int16)
    sparse = SparseVoteEvents.from_dense_votes(votes)
    spec = _vote_spec(threshold_abs=10)
    cap = GlobalRateCapSpec(cap=1, step=1, mutate_outputs=True)
    kwargs = dict(
        tensor_states={"toy.proj": state},
        votes_by_key={"toy.proj": votes},
        vote_specs_by_key={"toy.proj": spec},
        global_cap_spec=cap,
    )
    with_sparse = apply_bounded_delta_vote_step(
        **kwargs,
        candidate_sparse_vote_events_by_key={"toy.proj": sparse},
    )
    without_sparse = apply_bounded_delta_vote_step(**kwargs)
    assert _step_summary(with_sparse) == _step_summary(without_sparse)


def test_weighted_grads_to_science_arm_votes_backward_compatible_3tuple() -> None:
    """Gate-1: 3-tuple return preserved; sparse via optional sparse_events_out."""
    from scripts.hrm_text_158_bounded_delta_acquisition_probe import (
        ARM_A0_RANK_BUCKET_CURRENT,
        _weighted_grads_to_science_arm_votes,
        default_dry_run_rank_vote_spec,
        default_vote_update_spec,
    )

    q = torch.tensor([[0, 1, -1, 0]], dtype=torch.int8)
    weighted_grad = torch.tensor([[-1.0, -2.0, 3.0, 4.0]])
    tensor_states = {
        "toy.proj": type("State", (), {"q_levels": q})(),
    }
    rank_spec = default_dry_run_rank_vote_spec()
    vote_spec = default_vote_update_spec(16)
    votes, pressure, finite = _weighted_grads_to_science_arm_votes(
        {"toy.proj": weighted_grad},
        tensor_states,
        rank_spec=rank_spec,
        vote_spec=vote_spec,
        science_arm=str(ARM_A0_RANK_BUCKET_CURRENT),
    )
    assert isinstance(votes, dict)
    assert isinstance(pressure, dict)
    assert finite is True

    sparse_out: dict[str, Any] = {}
    votes2, pressure2, finite2 = _weighted_grads_to_science_arm_votes(
        {"toy.proj": weighted_grad},
        tensor_states,
        rank_spec=rank_spec,
        vote_spec=vote_spec,
        science_arm=str(ARM_A0_RANK_BUCKET_CURRENT),
        sparse_events_out=sparse_out,
    )
    assert votes2.keys() == votes.keys()
    for key in votes:
        assert torch.equal(votes2[key], votes[key])
    assert pressure2 == pressure
    assert finite2 is finite
    assert "toy.proj" in sparse_out
    assert isinstance(sparse_out["toy.proj"], SparseVoteEvents)


def test_hot_path_no_from_dense_votes(monkeypatch: pytest.MonkeyPatch) -> None:
    import calm.hrm_text_158.native_full_stack.sparse_vote_events as sparse_mod

    def _forbid_from_dense(*args, **kwargs):
        raise AssertionError("from_dense_votes must not run on probe hot path")

    monkeypatch.setattr(sparse_mod.SparseVoteEvents, "from_dense_votes", _forbid_from_dense)
    credit, moves, rank_spec = _rank_fixture()
    sparse_events_by_key = {
        "toy.proj": sparse_rank_bucketed_int16_vote_events(credit, moves, rank_spec),
    }
    votes_by_key = {"toy.proj": rank_bucketed_int16_votes(credit, moves, rank_spec)}
    q = torch.zeros((1, 4), dtype=torch.int8)
    state = make_event_coded_live_tensor_state("toy.proj", q, 0.25, demotion_band=1)
    apply_bounded_delta_vote_step(
        {"toy.proj": state},
        votes_by_key,
        {"toy.proj": _vote_spec()},
        candidate_sparse_vote_events_by_key=sparse_events_by_key,
        global_cap_spec=GlobalRateCapSpec(cap=1, step=1, mutate_outputs=True),
    )


def test_hot_path_no_learner_votes_nonzero_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    import calm.hrm_text_158.native_full_stack.bounded_delta_learner as learner_mod

    original = learner_mod._vote_active_flat_indices_for_event_coded_inputs

    def _guard(votes, sparse_events):
        if sparse_events is None:
            return original(votes, sparse_events)
        flat = votes.detach().cpu().view(-1)
        if int(flat.numel()) > 16:
            raise AssertionError("learner votes nonzero fallback must not run when sparse provided")
        return original(votes, sparse_events)

    monkeypatch.setattr(
        learner_mod,
        "_vote_active_flat_indices_for_event_coded_inputs",
        _guard,
    )
    credit, moves, rank_spec = _rank_fixture()
    sparse_events_by_key = {
        "toy.proj": sparse_rank_bucketed_int16_vote_events(credit, moves, rank_spec),
    }
    votes_by_key = {"toy.proj": rank_bucketed_int16_votes(credit, moves, rank_spec)}
    q = torch.zeros((1, 4), dtype=torch.int8)
    state = make_event_coded_live_tensor_state("toy.proj", q, 0.25, demotion_band=1)
    apply_bounded_delta_vote_step(
        {"toy.proj": state},
        votes_by_key,
        {"toy.proj": _vote_spec()},
        candidate_sparse_vote_events_by_key=sparse_events_by_key,
        global_cap_spec=GlobalRateCapSpec(cap=1, step=1, mutate_outputs=True),
    )


def _cap_apply_fingerprint(result) -> dict[str, object]:
    per_key: dict[str, object] = {}
    for key, state in sorted(result.tensor_states.items()):
        stats = result.tensor_stats[key]
        carrier = state.event_coded_live_carrier
        per_key[key] = {
            "q": tuple(int(x) for x in state.q_levels.flatten().tolist()),
            "flip_count": int(stats.get("flip_count", stats.get("global_rate_cap_applied_count", -1))),
            "carrier_content_sha256": (
                carrier_content_sha256(carrier) if carrier is not None else None
            ),
        }
    summary = result.global_summary
    return {
        "per_key": per_key,
        "global_rate_cap_applied_count": int(summary.get("global_rate_cap_applied_count", -1)),
        "global_rate_cap_accepted_count": int(summary.get("global_rate_cap_accepted_count", -1)),
    }


def _apply_dense_oracle_cap(
    states: dict[str, Any],
    votes_by_key: dict[str, torch.Tensor],
    sparse_by_key: dict[str, Any],
    vote_specs: dict[str, VoteUpdateSpec],
    cap: GlobalRateCapSpec,
):
    return apply_bounded_delta_vote_step(
        states,
        votes_by_key,
        vote_specs,
        candidate_sparse_vote_events_by_key=sparse_by_key,
        global_cap_spec=cap,
    )


def _apply_sparse_authority_cap(
    states: dict[str, Any],
    sparse_by_key: dict[str, Any],
    vote_specs: dict[str, VoteUpdateSpec],
    cap: GlobalRateCapSpec,
):
    return apply_bounded_delta_vote_step(
        states,
        None,
        vote_specs,
        candidate_sparse_vote_events_by_key=sparse_by_key,
        global_cap_spec=cap,
        event_coded_sparse_vote_authority=True,
    )


def test_event_coded_sparse_abs_new_acc_lookup_miss_raises() -> None:
    from calm.hrm_text_158.native_full_stack.event_coded_vote_update_adapter import (
        event_coded_sparse_abs_new_acc_at,
    )
    from calm.hrm_text_158.native_full_stack.vote_update import VoteUpdatePlan

    plan = VoteUpdatePlan(
        q_i16=torch.zeros(4, dtype=torch.int16),
        new_acc_i32=torch.zeros(4, dtype=torch.int32),
        candidate_indices=torch.empty(0, dtype=torch.int64),
        pre_veto_selected_indices=torch.empty(0, dtype=torch.int64),
        applied_indices=torch.tensor([99], dtype=torch.int64),
        applied_directions=torch.tensor([1], dtype=torch.int16),
        applied_thresholds=torch.tensor([8], dtype=torch.int32),
        replay_ce_veto_indices=torch.empty(0, dtype=torch.int64),
        replay_veto_directions=torch.empty(0, dtype=torch.int16),
        replay_veto_thresholds=torch.empty(0, dtype=torch.int32),
        pc_aux_negative_indices=torch.empty(0, dtype=torch.int64),
        pc_aux_veto_indices=torch.empty(0, dtype=torch.int64),
        stats={},
        event_coded_sparse_active_idx=torch.tensor([0, 1], dtype=torch.int64),
        event_coded_sparse_post_active_i32=torch.tensor([8, -8], dtype=torch.int32),
    )
    with pytest.raises(ValueError, match="lookup miss"):
        event_coded_sparse_abs_new_acc_at(plan, 99)


def test_event_coded_sparse_cap_cold_default_unsafe_raises() -> None:
    from calm.hrm_text_158.native_full_stack.event_coded_acc_live_carrier import (
        EventCodedAccLiveState,
    )

    threshold = 10
    q = torch.zeros(8, dtype=torch.int8)
    carrier = EventCodedAccLiveState(
        logical_numel=8,
        cold_default=threshold,
        threshold_abs=threshold,
        demotion_band=1,
    )
    state = make_event_coded_live_tensor_state(
        "toy.proj",
        q,
        0.25,
        demotion_band=1,
        carrier=carrier,
    )
    sparse = SparseVoteEvents(
        indices=torch.tensor([1], dtype=torch.int64),
        values=torch.tensor([5], dtype=torch.int16),
    )
    spec = _vote_spec(threshold_abs=threshold)
    cap = GlobalRateCapSpec(cap=1, step=1, mutate_outputs=True)
    with pytest.raises(ValueError, match="unsafe cold_default"):
        apply_bounded_delta_vote_step(
            {"toy.proj": state},
            None,
            {"toy.proj": spec},
            candidate_sparse_vote_events_by_key={"toy.proj": sparse},
            global_cap_spec=cap,
            event_coded_sparse_vote_authority=True,
        )


def test_event_coded_sparse_global_cap_parity_f1_vote_only() -> None:
    credit, moves, rank_spec = _rank_fixture()
    sparse = sparse_rank_bucketed_int16_vote_events(credit, moves, rank_spec)
    votes = rank_bucketed_int16_votes(credit, moves, rank_spec)
    spec = _vote_spec(threshold_abs=8)
    cap = GlobalRateCapSpec(cap=1, step=1, mutate_outputs=True)
    q = torch.tensor([[0, 1, -1, 0]], dtype=torch.int8)
    dense_state = make_event_coded_live_tensor_state("toy.proj", q, 0.25, demotion_band=1)
    sparse_state = make_event_coded_live_tensor_state("toy.proj", q.clone(), 0.25, demotion_band=1)
    vote_specs = {"toy.proj": spec}
    dense = _apply_dense_oracle_cap(
        {"toy.proj": dense_state},
        {"toy.proj": votes},
        {"toy.proj": sparse},
        vote_specs,
        cap,
    )
    sparse_result = _apply_sparse_authority_cap(
        {"toy.proj": sparse_state},
        {"toy.proj": sparse},
        vote_specs,
        cap,
    )
    assert _cap_apply_fingerprint(dense) == _cap_apply_fingerprint(sparse_result)


def test_event_coded_sparse_global_cap_parity_f2_hot_lane_zero_vote() -> None:
    from calm.hrm_text_158.native_full_stack.event_coded_acc_live_carrier import (
        EventCodedAccLiveState,
    )

    threshold = 10
    q = torch.zeros(8, dtype=torch.int8)
    carrier_dense = EventCodedAccLiveState(
        logical_numel=8,
        cold_default=0,
        threshold_abs=threshold,
        demotion_band=1,
    )
    carrier_dense._hot.set_lane(0, threshold)
    carrier_sparse = EventCodedAccLiveState(
        logical_numel=8,
        cold_default=0,
        threshold_abs=threshold,
        demotion_band=1,
    )
    carrier_sparse._hot.set_lane(0, threshold)
    dense_state = make_event_coded_live_tensor_state(
        "toy.proj",
        q.clone(),
        0.25,
        demotion_band=1,
        carrier=carrier_dense,
    )
    sparse_state = make_event_coded_live_tensor_state(
        "toy.proj",
        q.clone(),
        0.25,
        demotion_band=1,
        carrier=carrier_sparse,
    )
    votes = torch.zeros(8, dtype=torch.int16)
    sparse = SparseVoteEvents(
        indices=torch.empty(0, dtype=torch.int64),
        values=torch.empty(0, dtype=torch.int16),
    )
    spec = _vote_spec(threshold_abs=threshold)
    cap = GlobalRateCapSpec(cap=1, step=1, mutate_outputs=True)
    vote_specs = {"toy.proj": spec}
    dense = _apply_dense_oracle_cap(
        {"toy.proj": dense_state},
        {"toy.proj": votes},
        {"toy.proj": sparse},
        vote_specs,
        cap,
    )
    sparse_result = _apply_sparse_authority_cap(
        {"toy.proj": sparse_state},
        {"toy.proj": sparse},
        vote_specs,
        cap,
    )
    assert _cap_apply_fingerprint(dense) == _cap_apply_fingerprint(sparse_result)


def test_event_coded_sparse_cap_carrier_state_parity_large_events() -> None:
    from calm.hrm_text_158.native_full_stack.event_coded_vote_update_adapter import (
        _SPARSE_CARRIER_BULK_VOTE_APPLY_MAX_EVENTS,
    )

    numel = 200_000
    n_events = int(_SPARSE_CARRIER_BULK_VOTE_APPLY_MAX_EVENTS) + 4_096
    idx = torch.arange(n_events, dtype=torch.int64)
    vals = torch.full((n_events,), 12, dtype=torch.int16)
    sparse = SparseVoteEvents(indices=idx, values=vals)
    votes = torch.zeros(numel, dtype=torch.int16)
    votes[idx] = vals
    spec = _vote_spec(threshold_abs=10)
    cap = GlobalRateCapSpec(cap=4, step=1, mutate_outputs=True)
    q = torch.zeros(numel, dtype=torch.int8)
    dense_state = make_event_coded_live_tensor_state("toy.proj", q.clone(), 0.25, demotion_band=1)
    sparse_state = make_event_coded_live_tensor_state("toy.proj", q.clone(), 0.25, demotion_band=1)
    vote_specs = {"toy.proj": spec}
    dense = _apply_dense_oracle_cap(
        {"toy.proj": dense_state},
        {"toy.proj": votes},
        {"toy.proj": sparse},
        vote_specs,
        cap,
    )
    sparse_result = _apply_sparse_authority_cap(
        {"toy.proj": sparse_state},
        {"toy.proj": sparse},
        vote_specs,
        cap,
    )
    assert sparse.event_count() > int(_SPARSE_CARRIER_BULK_VOTE_APPLY_MAX_EVENTS)
    assert _cap_apply_fingerprint(dense) == _cap_apply_fingerprint(sparse_result)


def test_event_coded_sparse_global_cap_parity_f3_mixed_modules() -> None:
    credit, moves, rank_spec = _rank_fixture()
    sparse_a = sparse_rank_bucketed_int16_vote_events(credit, moves, rank_spec)
    votes_a = rank_bucketed_int16_votes(credit, moves, rank_spec)
    q_a = torch.tensor([[0, 1, -1, 0]], dtype=torch.int8)
    q_b = torch.tensor([[1, 0, 0, -1]], dtype=torch.int8)
    credit_b, moves_b, _ = _rank_fixture()
    sparse_b = sparse_rank_bucketed_int16_vote_events(credit_b, moves_b, rank_spec)
    votes_b = rank_bucketed_int16_votes(credit_b, moves_b, rank_spec)
    spec = _vote_spec(threshold_abs=8)
    cap = GlobalRateCapSpec(cap=4, step=2, mutate_outputs=True)
    states_dense = {
        "mod.a": make_event_coded_live_tensor_state("mod.a", q_a, 0.25, demotion_band=1),
        "mod.b": make_event_coded_live_tensor_state("mod.b", q_b, 0.25, demotion_band=1),
    }
    states_sparse = {
        "mod.a": make_event_coded_live_tensor_state("mod.a", q_a.clone(), 0.25, demotion_band=1),
        "mod.b": make_event_coded_live_tensor_state("mod.b", q_b.clone(), 0.25, demotion_band=1),
    }
    vote_specs = {"mod.a": spec, "mod.b": spec}
    sparse_by_key = {"mod.a": sparse_a, "mod.b": sparse_b}
    votes_by_key = {"mod.a": votes_a, "mod.b": votes_b}
    dense = _apply_dense_oracle_cap(states_dense, votes_by_key, sparse_by_key, vote_specs, cap)
    sparse_result = _apply_sparse_authority_cap(states_sparse, sparse_by_key, vote_specs, cap)
    assert _cap_apply_fingerprint(dense) == _cap_apply_fingerprint(sparse_result)


def test_sparse_authority_forbids_dense_vote_and_cap_densify(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import calm.hrm_text_158.native_full_stack.bounded_delta_learner as learner_mod
    import calm.hrm_text_158.native_full_stack.event_coded_vote_update_adapter as adapter_mod
    from scripts.hrm_text_158_bounded_delta_acquisition_probe import (
        ARM_A0_RANK_BUCKET_CURRENT,
        _weighted_grads_to_science_arm_votes,
        default_vote_update_spec,
    )

    def _forbid_dense_votes(*args, **kwargs):
        raise AssertionError("_dense_int16_votes_from_candidates forbidden on sparse authority path")

    def _forbid_densify(*args, **kwargs):
        raise AssertionError("densify_new_acc_i32_at_cap_boundary forbidden on sparse authority path")

    monkeypatch.setattr(learner_mod, "_dense_int16_votes_from_candidates", _forbid_dense_votes)
    monkeypatch.setattr(adapter_mod, "densify_new_acc_i32_at_cap_boundary", _forbid_densify)
    numel = 4096
    q = torch.zeros(numel, dtype=torch.int8)
    q.view(-1)[:8] = 1
    states = {"toy.proj": make_event_coded_live_tensor_state("toy.proj", q, 0.25, demotion_band=1)}
    weighted_grads = {"toy.proj": torch.randn(numel)}
    sparse_events: dict[str, Any] = {}
    rank_spec = default_dry_run_rank_vote_spec()
    vote_spec = default_vote_update_spec(16)
    _votes, _pressure, _finite = _weighted_grads_to_science_arm_votes(
        weighted_grads,
        states,
        rank_spec=rank_spec,
        vote_spec=vote_spec,
        science_arm=str(ARM_A0_RANK_BUCKET_CURRENT),
        sparse_events_out=sparse_events,
        sparse_construction_only=True,
    )
    apply_bounded_delta_vote_step(
        states,
        None,
        {"toy.proj": _vote_spec(threshold_abs=10)},
        candidate_sparse_vote_events_by_key=sparse_events,
        global_cap_spec=GlobalRateCapSpec(cap=1, step=1, mutate_outputs=True),
        event_coded_sparse_vote_authority=True,
    )


def test_cap_on_module_loop_under_budget(capsys) -> None:
    """M2: sparse construction + sparse cap-ON apply at representative 7.34M x 32."""
    from scripts.hrm_text_158_bounded_delta_acquisition_probe import (
        ARM_A0_RANK_BUCKET_CURRENT,
        _weighted_grads_to_science_arm_votes,
        default_vote_update_spec,
    )

    numel = 7_340_000
    module_count = 32
    spec = _vote_spec(threshold_abs=10)
    cap = GlobalRateCapSpec(cap=1, step=1, mutate_outputs=True)
    states = {}
    sparse_events_by_key: dict[str, Any] = {}
    rank_spec = default_dry_run_rank_vote_spec()
    vote_spec = default_vote_update_spec(16)
    weighted_grads = {}
    for idx in range(module_count):
        key = f"mod.{idx:02d}"
        q = torch.zeros(numel, dtype=torch.int8)
        q.view(-1)[:8] = 1
        states[key] = make_event_coded_live_tensor_state(key, q, 0.25, demotion_band=1)
        weighted_grads[key] = torch.randn(numel)

    start = time.perf_counter()
    votes_by_key, _pressure, _finite = _weighted_grads_to_science_arm_votes(
        weighted_grads,
        states,
        rank_spec=rank_spec,
        vote_spec=vote_spec,
        science_arm=str(ARM_A0_RANK_BUCKET_CURRENT),
        sparse_events_out=sparse_events_by_key,
        sparse_construction_only=True,
    )
    construction_elapsed = time.perf_counter() - start
    assert votes_by_key is None

    vote_specs = {key: spec for key in states}
    start = time.perf_counter()
    result = apply_bounded_delta_vote_step(
        states,
        None,
        vote_specs,
        candidate_sparse_vote_events_by_key=sparse_events_by_key,
        global_cap_spec=cap,
        event_coded_sparse_vote_authority=True,
    )
    apply_elapsed = time.perf_counter() - start
    elapsed = construction_elapsed + apply_elapsed
    assert result.global_summary.get("event_coded_sparse_cap_enabled") is True
    assert int(result.global_summary.get(C8_TRANSIENT_DENSE_COMPUTE_NUMEL_KEY, -1)) == 0
    assert elapsed < 90.0, (
        "coarse CPU liveness ceiling for carrier-faithful sparse-authority M2 "
        f"(far below 421s dense wall; total={elapsed:.2f}s)"
    )
    with capsys.disabled():
        print(
            "M2_TIMING_DIAGNOSTIC_REPORT_ONLY "
            f"elapsed={elapsed:.2f}s construction={construction_elapsed:.2f}s "
            f"apply={apply_elapsed:.2f}s"
        )


def _rank_moves_credit_at_numel(numel: int):
    """Synthetic rank-arm fixture with fixed candidate count (10 active lanes)."""
    rank_spec = default_dry_run_rank_vote_spec()
    q = torch.zeros(numel, dtype=torch.int8)
    q.view(-1)[:10] = torch.tensor([1, -1, 1, -1, 1, -1, 1, -1, 1, -1], dtype=torch.int8)
    weighted_grad = torch.randn(numel)
    moves = project_s1_gradient_to_moves(weighted_grad, q)
    credit = credit_from_weighted_grad(weighted_grad)
    return credit, moves, rank_spec


def test_probe_hot_path_m3_combined_marginal_over_dense_sweep(capsys) -> None:
    """M3: {1e3..1e7} sweep — report-only wall-time ratios; hard 2→1 proof is structural."""
    sweep = [1_000, 10_000, 100_000, 1_000_000, 10_000_000]
    medians: dict[int, dict[str, float]] = {}
    for numel in sweep:
        credit, moves, rank_spec = _rank_moves_credit_at_numel(numel)
        combined_samples: list[float] = []
        dense_samples: list[float] = []
        separate_samples: list[float] = []
        for _ in range(3):
            start = time.perf_counter()
            rank_bucketed_int16_votes_and_sparse_events(credit, moves, rank_spec)
            combined_samples.append(time.perf_counter() - start)

            start = time.perf_counter()
            rank_bucketed_int16_votes(credit, moves, rank_spec)
            dense_samples.append(time.perf_counter() - start)

            start = time.perf_counter()
            rank_bucketed_int16_votes(credit, moves, rank_spec)
            sparse_rank_bucketed_int16_vote_events(credit, moves, rank_spec)
            separate_samples.append(time.perf_counter() - start)

        combined_median = float(sorted(combined_samples)[1])
        dense_median = float(sorted(dense_samples)[1])
        separate_median = float(sorted(separate_samples)[1])
        marginal_ratio = combined_median / max(dense_median, 1e-9)
        double_scan_ratio = combined_median / max(separate_median, 1e-9)
        medians[numel] = {
            "combined": combined_median,
            "dense_only": dense_median,
            "separate_dense_sparse": separate_median,
            "marginal_over_dense_ratio": marginal_ratio,
            "combined_over_separate_ratio": double_scan_ratio,
        }

    # Report-only timing diagnostic: wall-time ratios are noisy on shared dev boxes.
    # Hard 2→1 proof lives in test_combined_rank_builder_single_candidate_compute.
    diagnostic_lines = [
        "M3_TIMING_DIAGNOSTIC_REPORT_ONLY",
        *(
            f"numel={numel}: combined={medians[numel]['combined']:.6f}s "
            f"dense={medians[numel]['dense_only']:.6f}s "
            f"separate={medians[numel]['separate_dense_sparse']:.6f}s "
            f"marginal_ratio={medians[numel]['marginal_over_dense_ratio']:.3f} "
            f"combined_over_separate={medians[numel]['combined_over_separate_ratio']:.3f}"
            for numel in sweep
        ),
    ]
    with capsys.disabled():
        print("\n".join(diagnostic_lines))

    # Contract exit proof: medians recorded for receipt (no silent lowering).
    assert medians[1_000_000]["combined"] > 0.0
    assert medians[10_000_000]["dense_only"] > 0.0


def _event_coded_state_on_device(state, device: str):
    from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
        BoundedDeltaTensorState,
    )

    return BoundedDeltaTensorState(
        state_key=state.state_key,
        q_levels=state.q_levels.to(device),
        frozen_scale=state.frozen_scale,
        bounded_accumulator=state.bounded_accumulator,
        exact_accumulator_shadow=state.exact_accumulator_shadow,
        bounded_accumulator_fresh_for_exact_shadow=state.bounded_accumulator_fresh_for_exact_shadow,
        bounded_accumulator_rebuild_hot_exact_indices=state.bounded_accumulator_rebuild_hot_exact_indices,
        bounded_accumulator_rebuild_cold_default_value=state.bounded_accumulator_rebuild_cold_default_value,
        event_coded_live_carrier=state.event_coded_live_carrier,
    )


def _two_module_sparse_cap_fixture():
    credit, moves, rank_spec = _rank_fixture()
    sparse_a = sparse_rank_bucketed_int16_vote_events(credit, moves, rank_spec)
    sparse_b = sparse_rank_bucketed_int16_vote_events(credit, moves, rank_spec)
    q_a = torch.tensor([[0, 1, -1, 0]], dtype=torch.int8)
    q_b = torch.tensor([[1, 0, 0, -1]], dtype=torch.int8)
    spec = _vote_spec(threshold_abs=8)
    cap = GlobalRateCapSpec(cap=4, step=1, mutate_outputs=True)
    states = {
        "mod.a": make_event_coded_live_tensor_state("mod.a", q_a, 0.25, demotion_band=1),
        "mod.b": make_event_coded_live_tensor_state("mod.b", q_b, 0.25, demotion_band=1),
    }
    sparse_by_key = {"mod.a": sparse_a, "mod.b": sparse_b}
    vote_specs = {"mod.a": spec, "mod.b": spec}
    return states, sparse_by_key, vote_specs, cap


def test_sync_q_levels_tensor_sparse_matches_dense_oracle() -> None:
    from calm.hrm_text_158.native_full_stack.event_coded_acc_live_carrier import (
        EventCodedAccLiveState,
    )
    from calm.hrm_text_158.native_full_stack.event_coded_vote_update_adapter import (
        _sync_q_levels_tensor,
    )

    q = torch.tensor([[0, 1, -1, 0, 1, 0, -1, 0]], dtype=torch.int8)
    carrier = EventCodedAccLiveState(
        logical_numel=8,
        cold_default=0,
        threshold_abs=8,
        demotion_band=1,
    )
    carrier.q_levels[1] = 1
    carrier.q_levels[4] = -1
    dense_oracle = q.detach().cpu().clone().to(torch.int8)
    flat = dense_oracle.flatten()
    for flat_index, level in carrier.q_levels.items():
        flat[int(flat_index)] = int(level)
    dense_oracle = flat.view_as(q).contiguous()
    sparse_out = _sync_q_levels_tensor(carrier, q)
    assert torch.equal(sparse_out.cpu(), dense_oracle)


def test_sync_q_levels_tensor_sparse_no_full_numel_cpu_clone(monkeypatch) -> None:
    from calm.hrm_text_158.native_full_stack.event_coded_acc_live_carrier import (
        EventCodedAccLiveState,
    )
    from calm.hrm_text_158.native_full_stack.event_coded_vote_update_adapter import (
        _sync_q_levels_tensor,
    )

    numel = 10_000
    q = torch.zeros(numel, dtype=torch.int8)
    carrier = EventCodedAccLiveState(
        logical_numel=numel,
        cold_default=0,
        threshold_abs=8,
        demotion_band=1,
    )
    carrier.q_levels[3] = 1
    carrier.q_levels[9999] = -1
    full_numel_cpu_calls: list[int] = []
    original_cpu = torch.Tensor.cpu

    def _tracking_cpu(self, *args, **kwargs):
        if int(self.numel()) >= numel:
            full_numel_cpu_calls.append(int(self.numel()))
        return original_cpu(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", _tracking_cpu)
    out = _sync_q_levels_tensor(carrier, q)
    assert out.shape == q.shape
    assert full_numel_cpu_calls == []


def test_sparse_cap_apply_serial_on_cuda_device(monkeypatch) -> None:
    if not torch.cuda.is_available():
        pytest.fail("R2/F1 cuda regression requires a CUDA device (4070 dev box)")
    device = "cuda:0"
    states, sparse_by_key, vote_specs, cap = _two_module_sparse_cap_fixture()
    states_cuda = {key: _event_coded_state_on_device(state, device) for key, state in states.items()}
    constructed: list[int] = []
    import concurrent.futures

    class _TrackingExecutor:
        def __init__(self, *args, **kwargs):
            constructed.append(1)
            raise AssertionError("ThreadPoolExecutor must not be used on cuda sparse cap apply")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(concurrent.futures, "ThreadPoolExecutor", _TrackingExecutor)
    result = apply_bounded_delta_vote_step(
        states_cuda,
        None,
        vote_specs,
        candidate_sparse_vote_events_by_key=sparse_by_key,
        global_cap_spec=cap,
        event_coded_sparse_vote_authority=True,
    )
    assert constructed == []
    assert result.global_summary.get("sparse_cap_apply_parallel_mode") == "serial_cuda"


def test_sparse_cap_apply_parallel_on_cpu_device(monkeypatch) -> None:
    states, sparse_by_key, vote_specs, cap = _two_module_sparse_cap_fixture()
    credit, moves, rank_spec = _rank_fixture()
    votes_by_key = {
        key: rank_bucketed_int16_votes(credit, moves, rank_spec)
        for key in sorted(sparse_by_key)
    }
    states_dense = {
        key: make_event_coded_live_tensor_state(
            key,
            state.q_levels.clone(),
            0.25,
            demotion_band=1,
        )
        for key, state in states.items()
    }
    states_sparse = {
        key: make_event_coded_live_tensor_state(
            key,
            state.q_levels.clone(),
            0.25,
            demotion_band=1,
        )
        for key, state in states.items()
    }
    constructed: list[int] = []
    import concurrent.futures

    original_executor = concurrent.futures.ThreadPoolExecutor

    class _TrackingExecutor(original_executor):
        def __init__(self, *args, **kwargs):
            constructed.append(1)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(concurrent.futures, "ThreadPoolExecutor", _TrackingExecutor)
    result = apply_bounded_delta_vote_step(
        states_sparse,
        None,
        vote_specs,
        candidate_sparse_vote_events_by_key=sparse_by_key,
        global_cap_spec=cap,
        event_coded_sparse_vote_authority=True,
    )
    dense = _apply_dense_oracle_cap(
        states_dense,
        votes_by_key,
        sparse_by_key,
        vote_specs,
        cap,
    )
    assert constructed == []
    assert result.global_summary.get("sparse_cap_apply_parallel_mode") == "serial_cpu"
    assert set(result.tensor_states.keys()) == {"mod.a", "mod.b"}
    assert all(
        result.tensor_states[key].event_coded_live_carrier is not None
        for key in ("mod.a", "mod.b")
    )
    assert _cap_apply_fingerprint(dense) == _cap_apply_fingerprint(result)


def test_sparse_cap_apply_mixed_device_states_fail_closed() -> None:
    if not torch.cuda.is_available():
        pytest.fail("mixed-device fail-close test requires CUDA to construct cuda/cpu pair")
    states, sparse_by_key, vote_specs, cap = _two_module_sparse_cap_fixture()
    states_mixed = {
        "mod.a": _event_coded_state_on_device(states["mod.a"], "cuda:0"),
        "mod.b": states["mod.b"],
    }
    with pytest.raises(ValueError, match="consistent q_levels devices"):
        apply_bounded_delta_vote_step(
            states_mixed,
            None,
            vote_specs,
            candidate_sparse_vote_events_by_key=sparse_by_key,
            global_cap_spec=cap,
            event_coded_sparse_vote_authority=True,
        )


def test_sparse_cap_apply_q_out_downstream_boundary_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.fail("R2 downstream boundary test must EXECUTE on 4070 (cuda required, not skip)")
    from calm.hrm_text_158.native_full_stack.bounded_delta_learner import tensor_sha256

    device = "cuda:0"
    states, sparse_by_key, vote_specs, cap = _two_module_sparse_cap_fixture()
    states_cuda = {key: _event_coded_state_on_device(state, device) for key, state in states.items()}
    q_sha_before = {
        key: tensor_sha256(states_cuda[key].q_levels) for key in sorted(states_cuda)
    }
    result = apply_bounded_delta_vote_step(
        states_cuda,
        None,
        vote_specs,
        candidate_sparse_vote_events_by_key=sparse_by_key,
        global_cap_spec=cap,
        event_coded_sparse_vote_authority=True,
    )
    assert result.global_summary.get("sparse_cap_apply_parallel_mode") == "serial_cuda"
    for key, prior in states_cuda.items():
        next_state = result.tensor_states[key]
        assert next_state.q_levels.device.type == "cpu"
        assert next_state.q_levels.dtype == torch.int8
        assert next_state.q_levels.is_contiguous()
        assert tuple(next_state.q_levels.shape) == tuple(prior.q_levels.shape)
        stats = result.tensor_stats[key]
        assert stats.get("q_sha256_before")
        assert stats.get("q_sha256_after")
        assert stats["q_sha256_before"] == q_sha_before[key]
        assert tensor_sha256(next_state.q_levels)
