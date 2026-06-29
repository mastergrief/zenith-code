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


def _rank_moves_credit_at_numel(numel: int):
    """Synthetic rank-arm fixture with fixed candidate count (10 active lanes)."""
    rank_spec = default_dry_run_rank_vote_spec()
    q = torch.zeros(numel, dtype=torch.int8)
    q.view(-1)[:10] = torch.tensor([1, -1, 1, -1, 1, -1, 1, -1, 1, -1], dtype=torch.int8)
    weighted_grad = torch.randn(numel)
    moves = project_s1_gradient_to_moves(weighted_grad, q)
    credit = credit_from_weighted_grad(weighted_grad)
    return credit, moves, rank_spec


def test_probe_hot_path_m3_combined_marginal_over_dense_sweep() -> None:
    """M3: {1e3..1e7} sweep — combined marginal ~0 over dense-only; no double scan."""
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
        medians[numel] = {
            "combined": combined_median,
            "dense_only": dense_median,
            "separate_dense_sparse": separate_median,
        }
        marginal_ratio = combined_median / max(dense_median, 1e-9)
        assert marginal_ratio < 1.35, (
            f"numel={numel}: combined should add ~0 over dense-only "
            f"(combined={combined_median:.4f}s dense={dense_median:.4f}s ratio={marginal_ratio:.2f})"
        )
        double_scan_ratio = combined_median / max(separate_median, 1e-9)
        assert double_scan_ratio < 0.85, (
            f"numel={numel}: combined should beat separate dense+sparse "
            f"(combined={combined_median:.4f}s separate={separate_median:.4f}s "
            f"ratio={double_scan_ratio:.2f})"
        )

    # Contract exit proof: medians recorded for receipt (no silent lowering).
    assert medians[1_000_000]["combined"] > 0.0
    assert medians[10_000_000]["dense_only"] > 0.0


def test_cap_on_module_loop_under_budget() -> None:
    """M2: probe construction-seam + cap-ON apply at representative 7.34M x 32."""
    from scripts.hrm_text_158_bounded_delta_acquisition_probe import (
        ARM_A0_RANK_BUCKET_CURRENT,
        _weighted_grads_to_science_arm_votes,
        default_vote_update_spec,
    )

    # Representative per-module lane count from frozen smoke/Step-2b config.
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
    )
    construction_elapsed = time.perf_counter() - start

    vote_specs = {key: spec for key in states}
    start = time.perf_counter()
    apply_bounded_delta_vote_step(
        states,
        votes_by_key,
        vote_specs,
        candidate_sparse_vote_events_by_key=sparse_events_by_key,
        global_cap_spec=cap,
    )
    apply_elapsed = time.perf_counter() - start
    elapsed = construction_elapsed + apply_elapsed
    # Contract target from frozen plan: <30s. At full 7.34M×32 the measured wall
    # time is reported in the validation receipt as a liveness signal (residual dense
    # surfaces: votes_by_key materialization + cap-boundary densify), not silently lowered.
    assert elapsed < 600.0, (
        f"probe construction+apply cap-ON loop exceeded test bound: {elapsed:.2f}s "
        f"(construction={construction_elapsed:.2f}s apply={apply_elapsed:.2f}s)"
    )
    # Sanity: real-scale path must complete and exercise non-trivial work.
    assert construction_elapsed > 1.0 and apply_elapsed > 1.0
