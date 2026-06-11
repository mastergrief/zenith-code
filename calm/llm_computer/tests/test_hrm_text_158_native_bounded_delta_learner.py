"""C2.0 bounded-delta learner integration tests.

These tests are CPU-only. They prove the default-off learner seams without
launching a GPU acquisition probe or touching the dirty curriculum trainer.
"""
from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json

import pytest
import torch
import torch.nn.functional as F

import calm.hrm_text_158.native_full_stack.bounded_delta_learner as bounded_delta_learner
from calm.hrm_text_158.bit_linear import BitLinear
from calm.hrm_text_158.native_full_stack.bounded_delta_accumulator import (
    ACCUMULATOR_SUBSTITUTE_LOCAL_VOTE_UPDATE_EXECUTABLE,
    ALGORITHMIC_LOCAL_VOTE_UPDATE_EXECUTABLE_NOT_PHYSICAL_SUB2,
)
from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    AUTHORITATIVE_STATE_SOURCE,
    BOUNDED_DELTA_CHECKPOINT_SCHEMA_VERSION,
    BOUNDED_UPDATE_ATTRIBUTION,
    RUN_BOUNDED_DELTA_LEARNER_ENV,
    RankVoteBin,
    RankVoteSpec,
    SourcePointer,
    apply_bounded_delta_vote_step,
    authoritative_forward_context,
    build_authoritative_checkpoint_payload,
    build_optimizer_excluding_eligible_masters,
    credit_from_weighted_grad,
    file_sha256,
    make_bounded_tensor_state,
    project_s1_gradient_to_moves,
    prove_eligible_master_identity_after_optimizer_step,
    rank_bucketed_int16_votes,
    reanchor_s1_oracle_hash,
    run_c2_bounded_delta_cpu_dry_run,
    sign_pressure_int16_votes,
    validate_authoritative_resume_payload,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap import (
    C1_BANKED_FAITHFUL_LONG_RUN_GLOBAL_CAP_CONTRACT_NAME,
    DEFER_ALL_NO_BACKFILL_TIE_RULE_MODE,
    EXACT_GLOBAL_CAP_TIE_RULE_MODE,
    GlobalRateCapSpec,
)
from calm.hrm_text_158.native_full_stack.vote_update import (
    VoteUpdateInputs,
    VoteUpdateSpec,
    VoteUpdateState,
    apply_integer_vote_update_reference,
)


def _assert_no_tensors(value):
    if isinstance(value, torch.Tensor):
        raise AssertionError("receipt payload must not contain raw tensors")
    if isinstance(value, dict):
        for child in value.values():
            _assert_no_tensors(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _assert_no_tensors(child)


def _rank_spec() -> RankVoteSpec:
    return RankVoteSpec(
        rank_bins=(
            RankVoteBin(0.0, 0.5, 1),
            RankVoteBin(0.5, 1.0, 4, include_hi=True),
        ),
    )


def test_s1_projection_and_rank_bucket_votes_port_sign_and_rank_law():
    q = torch.tensor([[-1, 0, 0, 1, -1, 1]], dtype=torch.int8)
    grad = torch.tensor([[-1.0, -2.0, 3.0, 4.0, 5.0, -6.0]])

    moves = project_s1_gradient_to_moves(grad, q)
    credit = credit_from_weighted_grad(grad)
    votes = rank_bucketed_int16_votes(credit, moves, _rank_spec())

    assert moves.tolist() == [[1, 1, -1, -1, 0, 0]]
    assert votes.dtype == torch.int16
    # Four candidates: smallest abs credit gets vote 1, the rest land in the
    # inclusive upper rank bucket with vote 4, signed by the projected move.
    assert votes.tolist() == [[1, 4, -4, -4, 0, 0]]


def test_sign_pressure_votes_use_constant_threshold_and_no_credit_rank():
    q = torch.tensor([[-1, 0, 0, 1, -1, 1]], dtype=torch.int8)
    grad = torch.tensor([[-1.0, -2.0, 3.0, 4.0, 5.0, -6.0]])
    moves = project_s1_gradient_to_moves(grad, q)
    spec = VoteUpdateSpec(
        threshold_abs=7,
        accumulator_clip_min=-127,
        accumulator_clip_max=127,
    )

    votes = sign_pressure_int16_votes(moves, spec)
    inverted = sign_pressure_int16_votes(moves, spec, inverted=True)

    assert moves.tolist() == [[1, 1, -1, -1, 0, 0]]
    assert votes.dtype == torch.int16
    assert votes.tolist() == [[7, 7, -7, -7, 0, 0]]
    assert inverted.tolist() == [[-7, -7, 7, 7, 0, 0]]


def test_bounded_delta_step_updates_q_acc_backlog_and_attributes_bounded_updates():
    state = make_bounded_tensor_state(
        "toy.proj",
        torch.tensor([0, 0, 0, 0], dtype=torch.int8),
        0.5,
        torch.zeros(4, dtype=torch.int16),
        hot_exact_indices=(0, 2),
    )
    votes = torch.tensor([3, 0, -3, 0], dtype=torch.int16)
    spec = VoteUpdateSpec(
        threshold_abs=2,
        accumulator_clip_min=-127,
        accumulator_clip_max=127,
        max_abs_per_tensor=8,
    )

    result = apply_bounded_delta_vote_step(
        {"toy.proj": state},
        {"toy.proj": votes},
        {"toy.proj": spec},
        parity_check=True,
    )
    next_state = result.tensor_states["toy.proj"]
    stats = result.tensor_stats["toy.proj"]

    assert next_state.q_levels.tolist() == [1, 0, -1, 0]
    assert next_state.exact_accumulator_shadow.tolist() == [1, 0, -1, 0]
    assert next_state.decoded_accumulators().tolist() == [1, 0, -1, 0]
    assert stats["q_changed_count"] == 2
    assert stats["bounded_accumulator_fresh_for_exact_shadow"] is True
    assert stats["bounded_accumulator_rebuilt_for_parity"] is True
    assert stats["bounded_decode_matches_exact_shadow"] is True
    assert stats["bounded_update_attribution"] == BOUNDED_UPDATE_ATTRIBUTION
    assert result.to_dict()["bounded_update_attribution"] == BOUNDED_UPDATE_ATTRIBUTION


def test_bounded_delta_step_passes_replay_and_pc_aux_maps_to_vote_update():
    state = make_bounded_tensor_state(
        "toy.proj",
        torch.tensor([0], dtype=torch.int8),
        0.5,
        torch.zeros(1, dtype=torch.int16),
    )
    votes = torch.tensor([12], dtype=torch.int16)
    replay_votes = torch.tensor([0], dtype=torch.int16)
    replay_moves = torch.tensor([0], dtype=torch.int8)
    pc_votes = torch.tensor([-1], dtype=torch.int16)
    pc_moves = torch.tensor([0], dtype=torch.int8)
    spec = VoteUpdateSpec(
        threshold_abs=10,
        accumulator_clip_min=-127,
        accumulator_clip_max=127,
        max_abs_per_tensor=1,
    )

    result = apply_bounded_delta_vote_step(
        {"toy.proj": state},
        {"toy.proj": votes},
        {"toy.proj": spec},
        replay_ce_veto_votes_by_key={"toy.proj": replay_votes},
        replay_ce_veto_moves_by_key={"toy.proj": replay_moves},
        pc_aux_votes_by_key={"toy.proj": pc_votes},
        pc_aux_moves_by_key={"toy.proj": pc_moves},
        pc_aux_mode="veto",
    )
    stats = result.tensor_stats["toy.proj"]

    assert result.tensor_states["toy.proj"].q_levels.tolist() == [0]
    assert result.tensor_states["toy.proj"].exact_accumulator_shadow.tolist() == [12]
    assert stats["pc_aux_mode"] == "veto"
    assert stats["pc_aux_negative_count"] == 1
    assert stats["pc_aux_veto_count"] == 1
    assert stats["post_veto_applied_flip_count"] == 0


def test_bounded_delta_step_global_cap_tie_rule_wiring_is_opt_in_and_shadowed():
    state = make_bounded_tensor_state(
        "toy.proj",
        torch.tensor([0, 0], dtype=torch.int8),
        0.5,
        torch.zeros(2, dtype=torch.int16),
    )
    votes = torch.tensor([12, 12], dtype=torch.int16)
    spec = VoteUpdateSpec(
        threshold_abs=10,
        accumulator_clip_min=-127,
        accumulator_clip_max=127,
        max_abs_per_tensor=2,
    )
    cap_spec = GlobalRateCapSpec(cap=1, step=3)

    exact = apply_bounded_delta_vote_step(
        {"toy.proj": state},
        {"toy.proj": votes},
        {"toy.proj": spec},
        global_cap_spec=cap_spec,
        global_cap_tie_rule_mode=EXACT_GLOBAL_CAP_TIE_RULE_MODE,
        global_cap_contract_name=C1_BANKED_FAITHFUL_LONG_RUN_GLOBAL_CAP_CONTRACT_NAME,
    )
    defer_all = apply_bounded_delta_vote_step(
        {"toy.proj": state},
        {"toy.proj": votes},
        {"toy.proj": spec},
        global_cap_spec=cap_spec,
        global_cap_tie_rule_mode=DEFER_ALL_NO_BACKFILL_TIE_RULE_MODE,
        global_cap_contract_name=C1_BANKED_FAITHFUL_LONG_RUN_GLOBAL_CAP_CONTRACT_NAME,
    )

    assert exact.tensor_states["toy.proj"].q_levels.tolist() == [1, 0]
    assert defer_all.tensor_states["toy.proj"].q_levels.tolist() == [0, 0]
    assert exact.global_summary["global_tie_rule_mode"] == EXACT_GLOBAL_CAP_TIE_RULE_MODE
    assert defer_all.global_summary["global_tie_rule_mode"] == DEFER_ALL_NO_BACKFILL_TIE_RULE_MODE
    assert defer_all.global_summary["global_rate_cap_contract_name"] == (
        C1_BANKED_FAITHFUL_LONG_RUN_GLOBAL_CAP_CONTRACT_NAME
    )
    assert defer_all.global_summary["drop_exercised_basis"] == "same_step_same_pre_state_shadow"
    assert defer_all.global_summary["exact_shadow_full_demand_sha256"] == defer_all.global_summary["defer_full_demand_sha256"]
    assert defer_all.global_summary["dropped_mass_count"] == 1
    assert defer_all.global_summary["mixed_class_count"] == 1
    assert exact.global_summary["global_rate_cap_accepted_count"] == 1
    assert defer_all.global_summary["global_rate_cap_accepted_count"] == 0
    json.dumps(defer_all.global_summary, sort_keys=True)

    for result in (exact, defer_all):
        receipt_fragment = {
            "device": "cpu",
            "dry_run": True,
            "checkpoint_written": False,
            "creditdir_mutated": False,
            "banked_pt_mutated": False,
            "steps_completed": 1,
            "step_reports": {"1": {"step_result": result.to_compact_dict()}},
            "phase_telemetry": {"event_count": 0, "events": []},
        }
        json.dumps(receipt_fragment, sort_keys=True)


def test_candidate_mode_rejects_active_controls_and_deferred_backlog():
    state = make_bounded_tensor_state(
        "toy.proj",
        torch.tensor([0], dtype=torch.int8),
        0.5,
        torch.zeros(1, dtype=torch.int16),
    )
    votes = torch.tensor([12], dtype=torch.int16)
    spec = VoteUpdateSpec(
        threshold_abs=10,
        accumulator_clip_min=-127,
        accumulator_clip_max=127,
        max_abs_per_tensor=1,
    )
    base_kwargs = dict(
        candidate_mode=ACCUMULATOR_SUBSTITUTE_LOCAL_VOTE_UPDATE_EXECUTABLE,
        candidate_sparse_vote_events_by_key={"toy.proj": {0: 12}},
        candidate_oracle_control_enabled=False,
    )

    active_cases = [
        (
            {"global_cap_spec": GlobalRateCapSpec(cap=1, step=0)},
            "global cap",
        ),
        (
            {"deferred_backlog": {"toy.proj": {0: {"defer_count": 1}}}},
            "deferred backlog",
        ),
        (
            {
                "replay_ce_veto_votes_by_key": {
                    "toy.proj": torch.zeros(1, dtype=torch.int16)
                },
                "replay_ce_veto_moves_by_key": {
                    "toy.proj": torch.zeros(1, dtype=torch.int8)
                },
            },
            "replay/pc auxiliary",
        ),
        (
            {
                "pc_aux_votes_by_key": {
                    "toy.proj": torch.zeros(1, dtype=torch.int16)
                },
                "pc_aux_moves_by_key": {
                    "toy.proj": torch.zeros(1, dtype=torch.int8)
                },
            },
            "replay/pc auxiliary",
        ),
        (
            {"front_c_identity_observer": lambda payload: payload},
            "front_c live identity",
        ),
    ]

    for extra_kwargs, error in active_cases:
        with pytest.raises(ValueError, match=error):
            apply_bounded_delta_vote_step(
                {"toy.proj": state},
                {"toy.proj": votes},
                {"toy.proj": spec},
                **base_kwargs,
                **extra_kwargs,
            )


def test_candidate_mode_success_keeps_active_controls_inactive():
    state = make_bounded_tensor_state(
        "toy.proj",
        torch.tensor([0], dtype=torch.int8),
        0.5,
        torch.zeros(1, dtype=torch.int16),
    )
    votes = torch.tensor([12], dtype=torch.int16)
    spec = VoteUpdateSpec(
        threshold_abs=10,
        accumulator_clip_min=-127,
        accumulator_clip_max=127,
        max_abs_per_tensor=1,
    )

    result = apply_bounded_delta_vote_step(
        {"toy.proj": state},
        {"toy.proj": votes},
        {"toy.proj": spec},
        candidate_mode=ACCUMULATOR_SUBSTITUTE_LOCAL_VOTE_UPDATE_EXECUTABLE,
        candidate_sparse_vote_events_by_key={"toy.proj": {0: 12}},
        candidate_oracle_control_enabled=False,
    )

    assert result.global_summary["global_rate_cap_enabled"] is False
    assert result.deferred_backlog == {}


def test_bounded_delta_step_validates_aux_map_keys_and_dtypes():
    state = make_bounded_tensor_state("toy.proj", torch.tensor([0], dtype=torch.int8), 0.5)
    votes = torch.tensor([12], dtype=torch.int16)
    spec = VoteUpdateSpec(
        threshold_abs=10,
        accumulator_clip_min=-127,
        accumulator_clip_max=127,
        max_abs_per_tensor=1,
    )

    with pytest.raises(ValueError, match="keys must match tensor_states"):
        apply_bounded_delta_vote_step(
            {"toy.proj": state},
            {"toy.proj": votes},
            {"toy.proj": spec},
            replay_ce_veto_votes_by_key={"other": votes},
        )
    with pytest.raises(ValueError, match="must be torch.int8"):
        apply_bounded_delta_vote_step(
            {"toy.proj": state},
            {"toy.proj": votes},
            {"toy.proj": spec},
            replay_ce_veto_votes_by_key={"toy.proj": votes},
            replay_ce_veto_moves_by_key={"toy.proj": votes},
        )


def test_live_bounded_delta_step_uses_exact_shadow_without_decode(monkeypatch):
    q = torch.zeros(64, dtype=torch.int8)
    acc = torch.ones(64, dtype=torch.int16)
    state = make_bounded_tensor_state("toy.proj", q, 0.5, acc)
    votes = torch.zeros(64, dtype=torch.int16)
    spec = VoteUpdateSpec(
        threshold_abs=3,
        accumulator_clip_min=-127,
        accumulator_clip_max=127,
        max_abs_per_tensor=64,
    )

    def fail_decode(_state):
        raise AssertionError("live path must not decode bounded accumulators")

    monkeypatch.setattr(
        bounded_delta_learner,
        "decode_bounded_accumulator_to_i16",
        fail_decode,
    )

    result = apply_bounded_delta_vote_step(
        {"toy.proj": state},
        {"toy.proj": votes},
        {"toy.proj": spec},
    )
    stats = result.tensor_stats["toy.proj"]

    assert result.tensor_states["toy.proj"].exact_accumulator_shadow.tolist() == acc.tolist()
    assert result.tensor_states["toy.proj"].bounded_accumulator_fresh_for_exact_shadow is False
    assert stats["bounded_accumulator_fresh_for_exact_shadow"] is False
    assert stats["bounded_accumulator_rebuilt_for_parity"] is False
    assert stats["bounded_decode_parity_checked"] is False
    assert "bounded_decode_matches_exact_shadow" not in stats
    compact = result.to_compact_dict()
    assert compact["tensor_state_summaries_included"] is False
    assert compact["tensor_state_keys"] == ["toy.proj"]


def test_live_bounded_delta_step_skips_bounded_encode_and_marks_stale(monkeypatch):
    q = torch.zeros(64, dtype=torch.int8)
    acc = torch.ones(64, dtype=torch.int16)
    state = make_bounded_tensor_state("toy.proj", q, 0.5, acc)
    votes = torch.zeros(64, dtype=torch.int16)
    spec = VoteUpdateSpec(
        threshold_abs=3,
        accumulator_clip_min=-127,
        accumulator_clip_max=127,
        max_abs_per_tensor=64,
    )

    def fail_live_rebuild(*_args, **_kwargs):
        raise AssertionError("live path must not rebuild bounded accumulators")

    monkeypatch.setattr(bounded_delta_learner, "make_bounded_tensor_state", fail_live_rebuild)
    monkeypatch.setattr(
        bounded_delta_learner,
        "encode_budget_capped_hybrid_reference",
        fail_live_rebuild,
    )

    result = apply_bounded_delta_vote_step(
        {"toy.proj": state},
        {"toy.proj": votes},
        {"toy.proj": spec},
    )
    next_state = result.tensor_states["toy.proj"]

    assert next_state.exact_accumulator_shadow.tolist() == acc.tolist()
    assert next_state.bounded_accumulator_fresh_for_exact_shadow is False
    assert result.tensor_stats["toy.proj"]["bounded_accumulator_fresh_for_exact_shadow"] is False
    assert result.tensor_stats["toy.proj"]["bounded_decode_parity_checked"] is False


def test_shadow_direct_live_update_matches_decode_baseline_on_dense_exceptions():
    q = torch.zeros(1024, dtype=torch.int8)
    acc = torch.arange(1024, dtype=torch.int16).remainder(5) - 2
    votes = torch.where(torch.arange(1024) % 2 == 0, 3, -3).to(torch.int16)
    state = make_bounded_tensor_state("toy.proj", q, 0.5, acc)
    assert len(state.bounded_accumulator.cold_exception_indices) > 700
    spec = VoteUpdateSpec(
        threshold_abs=3,
        accumulator_clip_min=-127,
        accumulator_clip_max=127,
        max_abs_per_tensor=1024,
    )
    decoded_baseline = VoteUpdateState(
        q_levels=state.q_levels,
        accumulators=state.decoded_accumulators(),
    )
    expected = apply_integer_vote_update_reference(
        decoded_baseline,
        VoteUpdateInputs(votes=votes),
        spec,
    )

    result = apply_bounded_delta_vote_step(
        {"toy.proj": state},
        {"toy.proj": votes},
        {"toy.proj": spec},
    )
    next_state = result.tensor_states["toy.proj"]
    stats = result.tensor_stats["toy.proj"]
    fresh_baseline = make_bounded_tensor_state(
        "toy.proj",
        expected.q_levels,
        0.5,
        expected.accumulators,
    )

    torch.testing.assert_close(next_state.q_levels, expected.q_levels, atol=0, rtol=0)
    torch.testing.assert_close(
        next_state.exact_accumulator_shadow,
        expected.accumulators,
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        next_state.exact_accumulator_shadow,
        fresh_baseline.exact_accumulator_shadow,
        atol=0,
        rtol=0,
    )
    assert next_state.with_fresh_bounded_accumulator().bounded_accumulator.to_dict() == (
        fresh_baseline.bounded_accumulator.to_dict()
    )
    assert result.global_summary["q_changed_count"] == expected.stats["q_changed_count"]
    for key in (
        "candidate_count",
        "vote_nonzero_count",
        "q_changed_count",
        "acc_abs_max_after",
    ):
        assert stats[key] == expected.stats[key]
    assert stats["bounded_decode_parity_checked"] is False
    assert stats["bounded_accumulator_fresh_for_exact_shadow"] is False


def test_explicit_bounded_decode_parity_catches_shadow_mismatch():
    q = torch.zeros(8, dtype=torch.int8)
    acc = torch.arange(8, dtype=torch.int16)
    state = make_bounded_tensor_state("toy.proj", q, 0.5, acc)
    corrupted = replace(
        state,
        exact_accumulator_shadow=state.exact_accumulator_shadow.clone().add(1),
    )

    with pytest.raises(ValueError, match="decode does not match exact shadow"):
        corrupted.bounded_decode_parity_report(fail_on_mismatch=True)
    with pytest.raises(ValueError, match="decode does not match exact shadow"):
        corrupted.to_schema_dict(parity_check=True)


def test_candidate_local_vote_update_mode_matches_dense_oracle_and_emits_scoped_proof():
    state = make_bounded_tensor_state(
        "toy.proj",
        torch.tensor([0, 0, 0, 0], dtype=torch.int8),
        0.5,
        torch.tensor([9, 0, -9, 0], dtype=torch.int16),
        hot_exact_indices=(0, 2),
    )
    votes = torch.tensor([2, 0, -2, 0], dtype=torch.int16)
    spec = VoteUpdateSpec(
        threshold_abs=10,
        accumulator_clip_min=-127,
        accumulator_clip_max=127,
        max_abs_per_tensor=4,
    )

    result = apply_bounded_delta_vote_step(
        {"toy.proj": state},
        {"toy.proj": votes},
        {"toy.proj": spec},
        candidate_mode=ACCUMULATOR_SUBSTITUTE_LOCAL_VOTE_UPDATE_EXECUTABLE,
        candidate_sparse_vote_events_by_key={"toy.proj": {0: 2, 2: -2}},
        candidate_oracle_control_enabled=True,
    )
    next_state = result.tensor_states["toy.proj"]
    proof = result.global_summary["candidate_local_update_proof_by_key"]["toy.proj"]
    stats = result.tensor_stats["toy.proj"]

    assert next_state.exact_accumulator_shadow is None
    assert next_state.q_levels.tolist() == [1, 0, -1, 0]
    assert proof["pass"] is True
    assert proof["scoped_label"] == ALGORITHMIC_LOCAL_VOTE_UPDATE_EXECUTABLE_NOT_PHYSICAL_SUB2
    assert proof["terminal_classification"] == ALGORITHMIC_LOCAL_VOTE_UPDATE_EXECUTABLE_NOT_PHYSICAL_SUB2
    assert proof["parity_pass"] is True
    assert proof["candidate_dense_decode_used"] is False
    assert proof["candidate_accumulator_transient_over2_used"] is False
    assert proof["candidate_vote_transient_over2_used"] is False
    assert proof["candidate_dense_vote_authority_used"] is False
    assert proof["dense_oracle_control_used"] is True
    assert proof["scoped_physical_budget_claim"] == "algorithmic_only_not_physical_sub2"
    assert proof["accumulator_physical_sub2_pass"] is False
    assert proof["candidate_bounded_decode_sha256_after"] == proof["oracle_acc_sha256_after"]
    assert proof["candidate_q_sha256_after"] == proof["oracle_q_sha256_after"]
    assert result.global_summary["candidate_local_update_pass"] is True
    assert result.global_summary["candidate_dense_vote_authority_used"] is False
    assert result.global_summary["q_changed_count"] == 2
    assert stats["candidate_local_update_pass"] is True
    assert (
        stats["candidate_terminal_classification"]
        == ALGORITHMIC_LOCAL_VOTE_UPDATE_EXECUTABLE_NOT_PHYSICAL_SUB2
    )


def test_candidate_local_vote_update_mode_rejects_dense_vote_authority_without_sparse_events():
    state = make_bounded_tensor_state(
        "toy.proj",
        torch.tensor([0], dtype=torch.int8),
        0.5,
        torch.tensor([9], dtype=torch.int16),
        hot_exact_indices=(0,),
    )
    votes = torch.tensor([2], dtype=torch.int16)
    spec = VoteUpdateSpec(
        threshold_abs=10,
        accumulator_clip_min=-127,
        accumulator_clip_max=127,
        max_abs_per_tensor=1,
    )

    with pytest.raises(ValueError, match="dense vote authority is unsupported"):
        apply_bounded_delta_vote_step(
            {"toy.proj": state},
            {"toy.proj": votes},
            {"toy.proj": spec},
            candidate_mode=ACCUMULATOR_SUBSTITUTE_LOCAL_VOTE_UPDATE_EXECUTABLE,
        )


def test_candidate_local_vote_update_mode_avoids_dense_escape_hatches_when_oracle_disabled(monkeypatch):
    state = make_bounded_tensor_state(
        "toy.proj",
        torch.tensor([0, 0, 0, 0], dtype=torch.int8),
        0.5,
        torch.tensor([9, 0, -9, 0], dtype=torch.int16),
        hot_exact_indices=(0, 2),
    )
    object.__setattr__(state, "exact_accumulator_shadow", object())
    votes = torch.tensor([2, 0, -2, 0], dtype=torch.int16)
    spec = VoteUpdateSpec(
        threshold_abs=10,
        accumulator_clip_min=-127,
        accumulator_clip_max=127,
        max_abs_per_tensor=4,
    )

    def fail_decode(_state):
        raise AssertionError("candidate path must not dense-decode bounded accumulators")

    def fail_vote_update_state(*_args, **_kwargs):
        raise AssertionError("candidate path must not request dense vote_update_state")

    def fail_dense_oracle(*_args, **_kwargs):
        raise AssertionError("candidate path must not route through the dense oracle branch")

    monkeypatch.setattr(bounded_delta_learner, "decode_bounded_accumulator_to_i16", fail_decode)
    monkeypatch.setattr(bounded_delta_learner.BoundedDeltaTensorState, "vote_update_state", fail_vote_update_state)
    monkeypatch.setattr(bounded_delta_learner, "apply_integer_vote_update_reference", fail_dense_oracle)

    result = apply_bounded_delta_vote_step(
        {"toy.proj": state},
        {"toy.proj": votes},
        {"toy.proj": spec},
        candidate_mode=ACCUMULATOR_SUBSTITUTE_LOCAL_VOTE_UPDATE_EXECUTABLE,
        candidate_sparse_vote_events_by_key={"toy.proj": {0: 2, 2: -2}},
        candidate_oracle_control_enabled=False,
    )
    proof = result.global_summary["candidate_local_update_proof_by_key"]["toy.proj"]

    assert proof["pass"] is True
    assert proof["dense_oracle_control_used"] is False
    assert proof["candidate_dense_decode_used"] is False
    assert proof["candidate_dense_vote_authority_used"] is False


def test_stale_state_guard_and_checkpoint_rebuild_serializes_fresh_bounded_payload():
    q = torch.zeros(32, dtype=torch.int8)
    acc = torch.full((32,), 7, dtype=torch.int16)
    acc[5:12] = torch.arange(7, dtype=torch.int16) - 3
    state = make_bounded_tensor_state(
        "toy.proj",
        q,
        0.5,
        acc,
        hot_exact_indices=(1, 3),
        cold_default_value=7,
    )
    votes = torch.zeros(32, dtype=torch.int16)
    spec = VoteUpdateSpec(
        threshold_abs=3,
        accumulator_clip_min=-127,
        accumulator_clip_max=127,
        max_abs_per_tensor=32,
    )

    result = apply_bounded_delta_vote_step(
        {"toy.proj": state},
        {"toy.proj": votes},
        {"toy.proj": spec},
    )
    stale = result.tensor_states["toy.proj"]

    assert stale.bounded_accumulator_fresh_for_exact_shadow is False
    assert stale.rebuild_hot_exact_indices() == (1, 3)
    assert stale.rebuild_cold_default_value() == 7
    with pytest.raises(ValueError, match="bounded accumulator is stale"):
        stale.decoded_accumulators()
    stale_summary = stale.to_schema_dict(parity_check=False)
    assert stale_summary["bounded_accumulator_fresh_for_exact_shadow"] is False
    assert stale_summary["bounded_accumulator_authority"] == "stale_optional_not_live_authority"
    assert stale_summary["bounded_decode_parity_checked"] is False
    assert "exact_shadow_matches_bounded_decode" not in stale_summary

    payload = build_authoritative_checkpoint_payload(
        {"toy.proj": stale},
        step=1,
        updater_config={"rank_vote_spec": _rank_spec().to_live_dict()},
        dry_run=True,
        checkpoint_written=False,
    )
    validate_authoritative_resume_payload(payload)
    checkpoint_summary = payload["tensor_summaries"]["toy.proj"]
    fresh = stale.with_fresh_bounded_accumulator()

    assert checkpoint_summary["bounded_accumulator_fresh_for_exact_shadow"] is True
    assert checkpoint_summary["bounded_accumulator_rebuilt_for_parity"] is True
    assert checkpoint_summary["exact_shadow_matches_bounded_decode"] is True
    assert checkpoint_summary["bounded_accumulator"] == fresh.bounded_accumulator.to_dict()
    assert checkpoint_summary["bounded_accumulator"]["cold_default_value"] == 7
    assert checkpoint_summary["bounded_accumulator"]["hot_exact_row_count"] == 2
    torch.testing.assert_close(
        fresh.decoded_accumulators(),
        stale.exact_accumulator_shadow,
        atol=0,
        rtol=0,
    )


def test_optimizer_excludes_bitlinear_masters_and_identity_snapshot():
    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = BitLinear(3, 2, bias=True)
            self.noneligible = torch.nn.Linear(2, 1)

    model = Tiny()
    eligible = {"proj": model.proj}
    opt, checks = build_optimizer_excluding_eligible_masters(model, eligible)
    model.proj.weight.grad = torch.ones_like(model.proj.weight)
    for param in model.noneligible.parameters():
        param.grad = torch.ones_like(param)

    proof = prove_eligible_master_identity_after_optimizer_step(
        opt,
        eligible,
        optimizer_checks=checks,
    )

    assert checks["eligible_params_in_optimizer"] == 0
    assert checks["eligible_optimizer_state_entries"] == 0
    assert proof["eligible_master_identity_pass"] is True
    assert proof["pass"] is True
    assert proof["eligible_master_sha256_before"] == proof["eligible_master_sha256_after"]


def test_authoritative_forward_context_uses_q_state_not_fp_master_and_captures_grad():
    module = BitLinear(3, 2, bias=False)
    with torch.no_grad():
        module.weight.fill_(99.0)
    q = torch.tensor([[1, 0, -1], [0, 1, 0]], dtype=torch.int8)
    state = make_bounded_tensor_state(
        "proj",
        q,
        2.0,
        hot_exact_indices=tuple(range(q.numel())),
    )
    x = torch.tensor([[1.0, 2.0, 3.0]])
    target = torch.tensor([[0.0, 1.0]])

    with authoritative_forward_context({"proj": module}, {"proj": state}, requires_grad=True) as handle:
        assert handle.capture_enabled is True
        out = module(x)
        expected = F.linear(x, q.to(torch.float32) * 2.0, None)
        torch.testing.assert_close(out, expected, atol=0.0, rtol=0.0)
        assert len(handle.captures["proj"]["inputs"]) == 1
        with pytest.raises(RuntimeError, match="captured inputs and grad_outputs"):
            handle.weighted_grad("proj")
        with torch.no_grad():
            module.weight.fill_(-123.0)
        out_after_master_mutation = module(x)
        torch.testing.assert_close(out_after_master_mutation, expected, atol=0.0, rtol=0.0)
        loss = F.mse_loss(out, target)
        loss.backward()
        weighted_grad = handle.weighted_grad("proj")

    assert weighted_grad.shape == q.shape
    torch.testing.assert_close(weighted_grad, handle.current_weights["proj"].grad, atol=0.0, rtol=0.0)


def test_authoritative_forward_context_no_grad_disables_capture_without_changing_output():
    module = BitLinear(3, 2, bias=False)
    with torch.no_grad():
        module.weight.fill_(42.0)
    q = torch.tensor([[1, 0, -1], [0, 1, 0]], dtype=torch.int8)
    state = make_bounded_tensor_state("proj", q, 2.0)
    x = torch.tensor([[1.0, 2.0, 3.0]])

    with authoritative_forward_context({"proj": module}, {"proj": state}, requires_grad=False) as handle:
        out = module(x)
        expected = F.linear(x, q.to(torch.float32) * 2.0, None)
        torch.testing.assert_close(out, expected, atol=0.0, rtol=0.0)
        assert handle.capture_enabled is False
        assert handle.captures["proj"]["inputs"] == []
        assert handle.captures["proj"]["grad_outputs"] == []
        with pytest.raises(RuntimeError, match="capture is disabled"):
            handle.weighted_grad("proj")


def test_checkpoint_schema_resume_refusal_and_reanchored_oracle_hash_receipt(tmp_path):
    oracle = tmp_path / "transient_fp_credit_science_train.py"
    oracle.write_text("def oracle_semantics():\n    return 'current'\n", encoding="utf-8")
    pointer = SourcePointer(
        label="tmp_s1_oracle",
        root=str(tmp_path),
        relative_path=oracle.name,
        expected_sha256="0" * 64,
        reason="unit test re-anchor fixture",
        reanchor_note="refresh when file-content sha256 changes",
    )
    receipt = reanchor_s1_oracle_hash(pointer)
    q = torch.tensor([0, 1], dtype=torch.int8)
    state = make_bounded_tensor_state("proj", q, 0.25, hot_exact_indices=(0, 1))

    payload = build_authoritative_checkpoint_payload(
        {"proj": state},
        step=3,
        updater_config={"rank_vote_spec": _rank_spec().to_live_dict()},
        oracle_receipt=receipt,
        dry_run=True,
        checkpoint_written=False,
    )

    assert receipt["current_sha256"] == file_sha256(oracle)
    assert receipt["expected_matches_current"] is False
    assert receipt["reanchored"] is True
    assert payload["schema"] == BOUNDED_DELTA_CHECKPOINT_SCHEMA_VERSION
    assert payload["authoritative_state_source"] == AUTHORITATIVE_STATE_SOURCE
    assert payload["source_oracle_receipt"]["current_sha256"] == hashlib.sha256(
        oracle.read_bytes(),
    ).hexdigest()
    assert payload["checkpoint_written"] is False
    _assert_no_tensors(payload)
    validate_authoritative_resume_payload(payload)

    eval_export = copy.deepcopy(payload)
    eval_export["artifact_role"] = "eval_export"
    with pytest.raises(ValueError, match="authoritative train state"):
        validate_authoritative_resume_payload(eval_export)


def test_default_off_cpu_dry_run_smoke_no_checkpoint_written(monkeypatch, tmp_path):
    monkeypatch.delenv(RUN_BOUNDED_DELTA_LEARNER_ENV, raising=False)
    with pytest.raises(RuntimeError, match=RUN_BOUNDED_DELTA_LEARNER_ENV):
        run_c2_bounded_delta_cpu_dry_run()

    oracle = tmp_path / "transient_fp_credit_science_train.py"
    oracle.write_text("def oracle_semantics():\n    return 'dry-run-current'\n", encoding="utf-8")
    pointer = SourcePointer(
        label="tmp_s1_oracle",
        root=str(tmp_path),
        relative_path=oracle.name,
        expected_sha256="f" * 64,
        reason="unit test dry-run oracle fixture",
        reanchor_note="refresh when file-content sha256 changes",
    )

    receipt = run_c2_bounded_delta_cpu_dry_run(enabled=True, oracle_pointer=pointer).to_dict()

    assert receipt["dry_run"] is True
    assert receipt["gpu_launched"] is False
    assert receipt["checkpoint_written"] is False
    assert receipt["first_forward_backward_update_finite"] is True
    assert receipt["parent_hash_unchanged"] is True
    assert receipt["optimizer_identity_proof"]["eligible_master_identity_pass"] is True
    assert receipt["oracle_receipt"]["current_sha256"] == hashlib.sha256(oracle.read_bytes()).hexdigest()
    assert receipt["step_result"]["global_summary"]["q_changed_count"] > 0
    assert receipt["bounded_update_attribution"] == BOUNDED_UPDATE_ATTRIBUTION
    assert receipt["checkpoint_payload"]["checkpoint_written"] is False
    _assert_no_tensors(receipt)

# --- B6 fold-5: two_tier_carry_w6 flag-OFF golden + ON-path coverage ---

from dataclasses import asdict, is_dataclass

from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    _front_c_cloned_observation,
)
from calm.hrm_text_158.native_full_stack.two_tier_carry_reducers import carry_self_update_row
from calm.hrm_text_158.native_full_stack.grad_proxy_audit import (
    count_w6_t10_crossing_eligible_from_votes,
)
from calm.hrm_text_158.native_full_stack.two_tier_step_orchestrator import (
    plan_two_tier_step,
    run_two_tier_optimizer_step,
)
from calm.hrm_text_158.native_full_stack.two_tier_threshold_semantics import (
    CROSSING_THRESHOLD_ABS,
)
from calm.hrm_text_158.native_full_stack.two_tier_transient_selection import (
    LOCAL_SELECTION_ORDER_TRANSIENT_LOCAL_LOSS_DELTA,
)
from calm.hrm_text_158.native_full_stack.vote_update import (
    LOCAL_SELECTION_ORDER_CURRENT_MARGIN_INDEX,
    plan_integer_vote_update_reference,
    plan_two_tier_vote_update_reference,
)

B6_OFF_GOLDEN_SHA256 = "0b3a999592469c3b8b5e43644891e5d4c39dadf2b52cd5d72494fa95cd29756b"
B6_OFF_GOLDEN_FIXTURE = {
    "capture_head_sha": "2ec3d4df79b7e4d5b41498eab77439fc908d5315",
    "cases": [
        {
            "S10_global_cap": {
                "deferred_backlog": {},
                "global_rate_cap_enabled": False,
                "q_levels": [
                    1,
                    0,
                    -1,
                    0
                ]
            },
            "S11_front_c_clone": {
                "deferred_backlog": {},
                "global_cap_used": False,
                "inputs_by_key": {
                    "toy.proj": {
                        "pc_aux_mode": "telemetry",
                        "pc_aux_moves": None,
                        "pc_aux_votes": None,
                        "replay_ce_veto_moves": None,
                        "replay_ce_veto_votes": None,
                        "vote_format": "int16_votes",
                        "votes": [
                            3,
                            0,
                            -3,
                            0
                        ]
                    }
                },
                "live_mutation_inputs_exposed": False,
                "plans_by_key": {
                    "toy.proj": {
                        "applied_directions": [
                            1,
                            -1
                        ],
                        "applied_indices": [
                            0,
                            2
                        ],
                        "applied_thresholds": [
                            2,
                            2
                        ],
                        "candidate_indices": [
                            0,
                            2
                        ],
                        "new_acc_i32": [
                            3,
                            0,
                            -3,
                            0
                        ],
                        "pc_aux_negative_indices": [],
                        "pc_aux_veto_indices": [],
                        "pre_veto_selected_indices": [
                            0,
                            2
                        ],
                        "q_i16": [
                            0,
                            0,
                            0,
                            0
                        ],
                        "replay_ce_veto_indices": [],
                        "replay_veto_directions": [],
                        "replay_veto_thresholds": [],
                        "stats": {
                            "acc_abs_max_after_decay_vote": 3,
                            "candidate_count": 2,
                            "global_cap_policy": "deferred_global_cap",
                            "local_selection_ordering_mode": "current_abs_new_acc_then_index",
                            "local_selection_ordering_seed": 0,
                            "local_selection_ordering_step": 0,
                            "max_flips": 4,
                            "pc_aux_mode": "telemetry",
                            "pc_aux_negative_count": 0,
                            "pc_aux_veto_accumulator_residual_policy": "not_enabled",
                            "pc_aux_veto_count": 0,
                            "pc_aux_veto_enabled": False,
                            "post_veto_acceptance_ratio_pre_cap": 1.0,
                            "post_veto_would_apply_pre_cap_count": 2,
                            "pre_veto_selected_flip_count": 2,
                            "replay_ce_veto_consumes_threshold_event": False,
                            "replay_ce_veto_count": 0,
                            "scope": "per_tensor_local_update",
                            "threshold_jitter_policy": "deferred_reject",
                            "vetoed_accumulator_clamp_count": 0,
                            "vetoed_accumulator_residual_policy": "not_enabled",
                            "vote_nonzero_count": 2
                        }
                    }
                },
                "q_acc_by_key": {
                    "toy.proj": {
                        "accumulators": [
                            1,
                            0,
                            -1,
                            0
                        ],
                        "q_levels": [
                            1,
                            0,
                            -1,
                            0
                        ],
                        "stats": {
                            "acc_abs_max_after": 1,
                            "acc_abs_max_after_decay_vote": 3,
                            "candidate_count": 2,
                            "flip_count": 2,
                            "global_cap_policy": "deferred_global_cap",
                            "local_selection_ordering_mode": "current_abs_new_acc_then_index",
                            "local_selection_ordering_seed": 0,
                            "local_selection_ordering_step": 0,
                            "max_flips": 4,
                            "pc_aux_mode": "telemetry",
                            "pc_aux_negative_count": 0,
                            "pc_aux_veto_accumulator_residual_policy": "not_enabled",
                            "pc_aux_veto_count": 0,
                            "pc_aux_veto_enabled": False,
                            "post_veto_acceptance_ratio_pre_cap": 1.0,
                            "post_veto_applied_flip_count": 2,
                            "post_veto_would_apply_pre_cap_count": 2,
                            "pre_veto_selected_flip_count": 2,
                            "q_changed_count": 2,
                            "replay_ce_veto_consumes_threshold_event": False,
                            "replay_ce_veto_count": 0,
                            "scope": "per_tensor_local_update",
                            "threshold_jitter_policy": "deferred_reject",
                            "vetoed_accumulator_clamp_count": 0,
                            "vetoed_accumulator_residual_policy": "not_enabled",
                            "vote_nonzero_count": 2
                        }
                    }
                },
                "schema": "hrm_text_158_front_c/v0.live_identity_observation_cloned_cpu",
                "specs_by_key": {
                    "toy.proj": {
                        "accumulator_clip_max": 127,
                        "accumulator_clip_min": -127,
                        "decay_denominator": 1,
                        "decay_numerator": 1,
                        "fraction_per_tensor": 1.0,
                        "global_cap_policy": "deferred_global_cap",
                        "max_abs_per_tensor": 8,
                        "threshold_abs": 2,
                        "threshold_jitter_enabled": False
                    }
                },
                "states_by_key": {
                    "toy.proj": {
                        "accumulator_format": "int16_accumulators",
                        "accumulators": [
                            0,
                            0,
                            0,
                            0
                        ],
                        "q_format": "int8_levels",
                        "q_levels": [
                            0,
                            0,
                            0,
                            0
                        ]
                    }
                }
            },
            "S12_learner_summary": {
                "global_summary": {
                    "global_rate_cap_enabled": False,
                    "local_selection_ordering_mode": "current_abs_new_acc_then_index",
                    "local_selection_ordering_seed": 0,
                    "local_selection_ordering_step": 0,
                    "q_changed_count": 2
                },
                "tensor_stats": {
                    "toy.proj": {
                        "acc_abs_max_after": 1,
                        "acc_abs_max_after_decay_vote": 3,
                        "bounded_accumulator_decoded_sha256_after": "b5505448e6dd95b473d7dc1befa335524ffc4d87c91fbd6f7e82b4c1582a5509",
                        "bounded_accumulator_fresh_for_exact_shadow": True,
                        "bounded_accumulator_rebuilt_for_parity": True,
                        "bounded_decode_matches_exact_shadow": True,
                        "bounded_decode_parity_checked": True,
                        "bounded_update_attribution": "q_acc_backlog_changed_by_bounded_delta_vote_update_only",
                        "candidate_count": 2,
                        "exact_accumulator_shadow_sha256_after": "b5505448e6dd95b473d7dc1befa335524ffc4d87c91fbd6f7e82b4c1582a5509",
                        "flip_count": 2,
                        "global_cap_policy": "deferred_global_cap",
                        "local_selection_ordering_mode": "current_abs_new_acc_then_index",
                        "local_selection_ordering_seed": 0,
                        "local_selection_ordering_step": 0,
                        "max_flips": 4,
                        "pc_aux_mode": "telemetry",
                        "pc_aux_negative_count": 0,
                        "pc_aux_veto_accumulator_residual_policy": "not_enabled",
                        "pc_aux_veto_count": 0,
                        "pc_aux_veto_enabled": False,
                        "post_veto_acceptance_ratio_pre_cap": 1.0,
                        "post_veto_applied_flip_count": 2,
                        "post_veto_would_apply_pre_cap_count": 2,
                        "pre_veto_selected_flip_count": 2,
                        "projection_law": "ported_s1_gradient_sign_to_ternary_move",
                        "q_changed_count": 2,
                        "q_sha256_after": "6a699f0aa48245d7fac028c84bba5c712d821832f80f1f4f6c36d2614ffe07d7",
                        "q_sha256_before": "de3087fc684851bb32d76597811c0f0615448c9b5b6cdac7c27950786608963a",
                        "replay_ce_veto_consumes_threshold_event": False,
                        "replay_ce_veto_count": 0,
                        "scope": "per_tensor_local_update",
                        "state_key": "toy.proj",
                        "threshold_jitter_policy": "deferred_reject",
                        "vetoed_accumulator_clamp_count": 0,
                        "vetoed_accumulator_residual_policy": "not_enabled",
                        "vote_law": "ported_s1_rank_bucketed_integer_votes",
                        "vote_nonzero_count": 2,
                        "votes_sha256": "7e4493a0e5f62184fd6bcd91677be3c7074db4708010711d5f57c62ae369103c"
                    }
                }
            },
            "S1_q_levels": [
                1,
                0,
                -1,
                0
            ],
            "S2_accumulators": [
                1,
                0,
                -1,
                0
            ],
            "S3_plan_candidate_indices": [
                0,
                2
            ],
            "S4_plan_pre_veto_selected": [
                0,
                2
            ],
            "S5_plan_applied_triple": {
                "applied_directions": [
                    1,
                    -1
                ],
                "applied_indices": [
                    0,
                    2
                ],
                "applied_thresholds": [
                    2,
                    2
                ]
            },
            "S6_plan_veto_tensors": {
                "pc_aux_negative_indices": [],
                "pc_aux_veto_indices": [],
                "replay_ce_veto_indices": [],
                "replay_veto_directions": [],
                "replay_veto_thresholds": []
            },
            "S7_plan_stats": {
                "acc_abs_max_after_decay_vote": 3,
                "candidate_count": 2,
                "global_cap_policy": "deferred_global_cap",
                "local_selection_ordering_mode": "current_abs_new_acc_then_index",
                "local_selection_ordering_seed": 0,
                "local_selection_ordering_step": 0,
                "max_flips": 4,
                "pc_aux_mode": "telemetry",
                "pc_aux_negative_count": 0,
                "pc_aux_veto_accumulator_residual_policy": "not_enabled",
                "pc_aux_veto_count": 0,
                "pc_aux_veto_enabled": False,
                "post_veto_acceptance_ratio_pre_cap": 1.0,
                "post_veto_would_apply_pre_cap_count": 2,
                "pre_veto_selected_flip_count": 2,
                "replay_ce_veto_consumes_threshold_event": False,
                "replay_ce_veto_count": 0,
                "scope": "per_tensor_local_update",
                "threshold_jitter_policy": "deferred_reject",
                "vetoed_accumulator_clamp_count": 0,
                "vetoed_accumulator_residual_policy": "not_enabled",
                "vote_nonzero_count": 2
            },
            "S8_replay_ce_veto_partition": {
                "applied_indices": [
                    0,
                    2
                ],
                "replay_ce_veto_indices": []
            },
            "S9_pc_aux_veto": {
                "pc_aux_mode": "telemetry",
                "pc_aux_negative_indices": [],
                "pc_aux_veto_indices": []
            },
            "name": "backlog_parity"
        },
        {
            "S10_global_cap": {
                "deferred_backlog": {},
                "global_rate_cap_enabled": False,
                "q_levels": [
                    0
                ]
            },
            "S11_front_c_clone": {
                "deferred_backlog": {},
                "global_cap_used": False,
                "inputs_by_key": {
                    "toy.proj": {
                        "pc_aux_mode": "veto",
                        "pc_aux_moves": [
                            0
                        ],
                        "pc_aux_votes": [
                            -1
                        ],
                        "replay_ce_veto_moves": [
                            0
                        ],
                        "replay_ce_veto_votes": [
                            0
                        ],
                        "vote_format": "int16_votes",
                        "votes": [
                            12
                        ]
                    }
                },
                "live_mutation_inputs_exposed": False,
                "plans_by_key": {
                    "toy.proj": {
                        "applied_directions": [],
                        "applied_indices": [],
                        "applied_thresholds": [],
                        "candidate_indices": [
                            0
                        ],
                        "new_acc_i32": [
                            12
                        ],
                        "pc_aux_negative_indices": [
                            0
                        ],
                        "pc_aux_veto_indices": [
                            0
                        ],
                        "pre_veto_selected_indices": [
                            0
                        ],
                        "q_i16": [
                            0
                        ],
                        "replay_ce_veto_indices": [],
                        "replay_veto_directions": [],
                        "replay_veto_thresholds": [],
                        "stats": {
                            "acc_abs_max_after_decay_vote": 12,
                            "candidate_count": 1,
                            "global_cap_policy": "deferred_global_cap",
                            "local_selection_ordering_mode": "current_abs_new_acc_then_index",
                            "local_selection_ordering_seed": 0,
                            "local_selection_ordering_step": 0,
                            "max_flips": 1,
                            "pc_aux_mode": "veto",
                            "pc_aux_negative_count": 1,
                            "pc_aux_veto_accumulator_residual_policy": "q_mutation_veto_only_accumulator_retained",
                            "pc_aux_veto_count": 1,
                            "pc_aux_veto_enabled": True,
                            "post_veto_acceptance_ratio_pre_cap": 0.0,
                            "post_veto_would_apply_pre_cap_count": 0,
                            "pre_veto_selected_flip_count": 1,
                            "replay_ce_veto_consumes_threshold_event": True,
                            "replay_ce_veto_count": 0,
                            "scope": "per_tensor_local_update",
                            "threshold_jitter_policy": "deferred_reject",
                            "vetoed_accumulator_clamp_count": 0,
                            "vetoed_accumulator_residual_policy": "subtract_threshold_then_clamp_without_q_mutation",
                            "vote_nonzero_count": 1
                        }
                    }
                },
                "q_acc_by_key": {
                    "toy.proj": {
                        "accumulators": [
                            12
                        ],
                        "q_levels": [
                            0
                        ],
                        "stats": {
                            "acc_abs_max_after": 12,
                            "acc_abs_max_after_decay_vote": 12,
                            "candidate_count": 1,
                            "flip_count": 0,
                            "global_cap_policy": "deferred_global_cap",
                            "local_selection_ordering_mode": "current_abs_new_acc_then_index",
                            "local_selection_ordering_seed": 0,
                            "local_selection_ordering_step": 0,
                            "max_flips": 1,
                            "pc_aux_mode": "veto",
                            "pc_aux_negative_count": 1,
                            "pc_aux_veto_accumulator_residual_policy": "q_mutation_veto_only_accumulator_retained",
                            "pc_aux_veto_count": 1,
                            "pc_aux_veto_enabled": True,
                            "post_veto_acceptance_ratio_pre_cap": 0.0,
                            "post_veto_applied_flip_count": 0,
                            "post_veto_would_apply_pre_cap_count": 0,
                            "pre_veto_selected_flip_count": 1,
                            "q_changed_count": 0,
                            "replay_ce_veto_consumes_threshold_event": True,
                            "replay_ce_veto_count": 0,
                            "scope": "per_tensor_local_update",
                            "threshold_jitter_policy": "deferred_reject",
                            "vetoed_accumulator_clamp_count": 0,
                            "vetoed_accumulator_residual_policy": "subtract_threshold_then_clamp_without_q_mutation",
                            "vote_nonzero_count": 1
                        }
                    }
                },
                "schema": "hrm_text_158_front_c/v0.live_identity_observation_cloned_cpu",
                "specs_by_key": {
                    "toy.proj": {
                        "accumulator_clip_max": 127,
                        "accumulator_clip_min": -127,
                        "decay_denominator": 1,
                        "decay_numerator": 1,
                        "fraction_per_tensor": 1.0,
                        "global_cap_policy": "deferred_global_cap",
                        "max_abs_per_tensor": 1,
                        "threshold_abs": 10,
                        "threshold_jitter_enabled": False
                    }
                },
                "states_by_key": {
                    "toy.proj": {
                        "accumulator_format": "int16_accumulators",
                        "accumulators": [
                            0
                        ],
                        "q_format": "int8_levels",
                        "q_levels": [
                            0
                        ]
                    }
                }
            },
            "S12_learner_summary": {
                "global_summary": {
                    "global_rate_cap_enabled": False,
                    "local_selection_ordering_mode": "current_abs_new_acc_then_index",
                    "local_selection_ordering_seed": 0,
                    "local_selection_ordering_step": 0,
                    "q_changed_count": 0
                },
                "tensor_stats": {
                    "toy.proj": {
                        "acc_abs_max_after": 12,
                        "acc_abs_max_after_decay_vote": 12,
                        "bounded_accumulator_fresh_for_exact_shadow": False,
                        "bounded_accumulator_rebuilt_for_parity": False,
                        "bounded_decode_parity_checked": False,
                        "bounded_update_attribution": "q_acc_backlog_changed_by_bounded_delta_vote_update_only",
                        "candidate_count": 1,
                        "exact_accumulator_shadow_sha256_after": "0f6b227e1793628aff62f7477fd18e3524a03d475dc5bfa01400e9a3ffb53698",
                        "flip_count": 0,
                        "global_cap_policy": "deferred_global_cap",
                        "local_selection_ordering_mode": "current_abs_new_acc_then_index",
                        "local_selection_ordering_seed": 0,
                        "local_selection_ordering_step": 0,
                        "max_flips": 1,
                        "pc_aux_mode": "veto",
                        "pc_aux_negative_count": 1,
                        "pc_aux_veto_accumulator_residual_policy": "q_mutation_veto_only_accumulator_retained",
                        "pc_aux_veto_count": 1,
                        "pc_aux_veto_enabled": True,
                        "post_veto_acceptance_ratio_pre_cap": 0.0,
                        "post_veto_applied_flip_count": 0,
                        "post_veto_would_apply_pre_cap_count": 0,
                        "pre_veto_selected_flip_count": 1,
                        "projection_law": "ported_s1_gradient_sign_to_ternary_move",
                        "q_changed_count": 0,
                        "q_sha256_after": "a7e2e06ec3ceb587f752d70d546f1dc116f0b40d1b7ca15098023acbd33865f9",
                        "q_sha256_before": "a7e2e06ec3ceb587f752d70d546f1dc116f0b40d1b7ca15098023acbd33865f9",
                        "replay_ce_veto_consumes_threshold_event": True,
                        "replay_ce_veto_count": 0,
                        "scope": "per_tensor_local_update",
                        "state_key": "toy.proj",
                        "threshold_jitter_policy": "deferred_reject",
                        "vetoed_accumulator_clamp_count": 0,
                        "vetoed_accumulator_residual_policy": "subtract_threshold_then_clamp_without_q_mutation",
                        "vote_law": "ported_s1_rank_bucketed_integer_votes",
                        "vote_nonzero_count": 1,
                        "votes_sha256": "0f6b227e1793628aff62f7477fd18e3524a03d475dc5bfa01400e9a3ffb53698"
                    }
                }
            },
            "S1_q_levels": [
                0
            ],
            "S2_accumulators": [
                12
            ],
            "S3_plan_candidate_indices": [
                0
            ],
            "S4_plan_pre_veto_selected": [
                0
            ],
            "S5_plan_applied_triple": {
                "applied_directions": [],
                "applied_indices": [],
                "applied_thresholds": []
            },
            "S6_plan_veto_tensors": {
                "pc_aux_negative_indices": [
                    0
                ],
                "pc_aux_veto_indices": [
                    0
                ],
                "replay_ce_veto_indices": [],
                "replay_veto_directions": [],
                "replay_veto_thresholds": []
            },
            "S7_plan_stats": {
                "acc_abs_max_after_decay_vote": 12,
                "candidate_count": 1,
                "global_cap_policy": "deferred_global_cap",
                "local_selection_ordering_mode": "current_abs_new_acc_then_index",
                "local_selection_ordering_seed": 0,
                "local_selection_ordering_step": 0,
                "max_flips": 1,
                "pc_aux_mode": "veto",
                "pc_aux_negative_count": 1,
                "pc_aux_veto_accumulator_residual_policy": "q_mutation_veto_only_accumulator_retained",
                "pc_aux_veto_count": 1,
                "pc_aux_veto_enabled": True,
                "post_veto_acceptance_ratio_pre_cap": 0.0,
                "post_veto_would_apply_pre_cap_count": 0,
                "pre_veto_selected_flip_count": 1,
                "replay_ce_veto_consumes_threshold_event": True,
                "replay_ce_veto_count": 0,
                "scope": "per_tensor_local_update",
                "threshold_jitter_policy": "deferred_reject",
                "vetoed_accumulator_clamp_count": 0,
                "vetoed_accumulator_residual_policy": "subtract_threshold_then_clamp_without_q_mutation",
                "vote_nonzero_count": 1
            },
            "S8_replay_ce_veto_partition": {
                "applied_indices": [],
                "replay_ce_veto_indices": []
            },
            "S9_pc_aux_veto": {
                "pc_aux_mode": "veto",
                "pc_aux_negative_indices": [
                    0
                ],
                "pc_aux_veto_indices": [
                    0
                ]
            },
            "name": "replay_pc_veto"
        },
        {
            "S10_global_cap": {
                "deferred_backlog": {
                    "toy.proj": {
                        "1": {
                            "defer_count": 1,
                            "first_step": 3,
                            "last_deferred_step": 3
                        }
                    }
                },
                "global_rate_cap_enabled": True,
                "q_levels": [
                    1,
                    0
                ]
            },
            "S11_front_c_clone": {
                "deferred_backlog": {
                    "toy.proj": {
                        "1": {
                            "defer_count": 1,
                            "first_step": 3,
                            "last_deferred_step": 3
                        }
                    }
                },
                "global_cap_used": True,
                "inputs_by_key": {
                    "toy.proj": {
                        "pc_aux_mode": "telemetry",
                        "pc_aux_moves": None,
                        "pc_aux_votes": None,
                        "replay_ce_veto_moves": None,
                        "replay_ce_veto_votes": None,
                        "vote_format": "int16_votes",
                        "votes": [
                            12,
                            12
                        ]
                    }
                },
                "live_mutation_inputs_exposed": False,
                "plans_by_key": {
                    "toy.proj": {
                        "applied_directions": [
                            1,
                            1
                        ],
                        "applied_indices": [
                            0,
                            1
                        ],
                        "applied_thresholds": [
                            10,
                            10
                        ],
                        "candidate_indices": [
                            0,
                            1
                        ],
                        "new_acc_i32": [
                            12,
                            12
                        ],
                        "pc_aux_negative_indices": [],
                        "pc_aux_veto_indices": [],
                        "pre_veto_selected_indices": [
                            0,
                            1
                        ],
                        "q_i16": [
                            0,
                            0
                        ],
                        "replay_ce_veto_indices": [],
                        "replay_veto_directions": [],
                        "replay_veto_thresholds": [],
                        "stats": {
                            "acc_abs_max_after_decay_vote": 12,
                            "candidate_count": 2,
                            "global_cap_policy": "deferred_global_cap",
                            "local_selection_ordering_mode": "current_abs_new_acc_then_index",
                            "local_selection_ordering_seed": 0,
                            "local_selection_ordering_step": 0,
                            "max_flips": 2,
                            "pc_aux_mode": "telemetry",
                            "pc_aux_negative_count": 0,
                            "pc_aux_veto_accumulator_residual_policy": "not_enabled",
                            "pc_aux_veto_count": 0,
                            "pc_aux_veto_enabled": False,
                            "post_veto_acceptance_ratio_pre_cap": 1.0,
                            "post_veto_would_apply_pre_cap_count": 2,
                            "pre_veto_selected_flip_count": 2,
                            "replay_ce_veto_consumes_threshold_event": False,
                            "replay_ce_veto_count": 0,
                            "scope": "per_tensor_local_update",
                            "threshold_jitter_policy": "deferred_reject",
                            "vetoed_accumulator_clamp_count": 0,
                            "vetoed_accumulator_residual_policy": "not_enabled",
                            "vote_nonzero_count": 2
                        }
                    }
                },
                "q_acc_by_key": {
                    "toy.proj": {
                        "accumulators": [
                            2,
                            2
                        ],
                        "q_levels": [
                            1,
                            1
                        ],
                        "stats": {
                            "acc_abs_max_after": 2,
                            "acc_abs_max_after_decay_vote": 12,
                            "candidate_count": 2,
                            "flip_count": 2,
                            "global_cap_policy": "deferred_global_cap",
                            "local_selection_ordering_mode": "current_abs_new_acc_then_index",
                            "local_selection_ordering_seed": 0,
                            "local_selection_ordering_step": 0,
                            "max_flips": 2,
                            "pc_aux_mode": "telemetry",
                            "pc_aux_negative_count": 0,
                            "pc_aux_veto_accumulator_residual_policy": "not_enabled",
                            "pc_aux_veto_count": 0,
                            "pc_aux_veto_enabled": False,
                            "post_veto_acceptance_ratio_pre_cap": 1.0,
                            "post_veto_applied_flip_count": 2,
                            "post_veto_would_apply_pre_cap_count": 2,
                            "pre_veto_selected_flip_count": 2,
                            "q_changed_count": 2,
                            "replay_ce_veto_consumes_threshold_event": False,
                            "replay_ce_veto_count": 0,
                            "scope": "per_tensor_local_update",
                            "threshold_jitter_policy": "deferred_reject",
                            "vetoed_accumulator_clamp_count": 0,
                            "vetoed_accumulator_residual_policy": "not_enabled",
                            "vote_nonzero_count": 2
                        }
                    }
                },
                "schema": "hrm_text_158_front_c/v0.live_identity_observation_cloned_cpu",
                "specs_by_key": {
                    "toy.proj": {
                        "accumulator_clip_max": 127,
                        "accumulator_clip_min": -127,
                        "decay_denominator": 1,
                        "decay_numerator": 1,
                        "fraction_per_tensor": 1.0,
                        "global_cap_policy": "deferred_global_cap",
                        "max_abs_per_tensor": 2,
                        "threshold_abs": 10,
                        "threshold_jitter_enabled": False
                    }
                },
                "states_by_key": {
                    "toy.proj": {
                        "accumulator_format": "int16_accumulators",
                        "accumulators": [
                            0,
                            0
                        ],
                        "q_format": "int8_levels",
                        "q_levels": [
                            0,
                            0
                        ]
                    }
                }
            },
            "S12_learner_summary": {
                "global_summary": {
                    "accepted_fresh_count": 1,
                    "accepted_from_prior_deferred_count": 0,
                    "bad_pressure_drain_policy": "deferred_non_scope",
                    "cpu_glue_not_kernel": True,
                    "cpu_glue_not_kernel_note": "global-rate-cap selection is thin cross-tensor CPU/control-flow glue; it has no GPU receipt by design in Slice 2B",
                    "deferred_backlog_max_age_steps": 0,
                    "deferred_backlog_max_defer_count": 1,
                    "deferred_backlog_size": 1,
                    "drop_exercised": False,
                    "dropped_mass_count": 0,
                    "dropped_mass_identities_sha256": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
                    "exact_shadow_accepted_sha256": "c773b27f4c5b93a8227463ca4116357e78645021908de6fbace90c9ff684fe39",
                    "exact_shadow_deferred_sha256": "0eebb09249ec06c9a0c57a4bc7c0c65cfb1d43d061053519feea2079ef706e10",
                    "exact_shadow_full_demand_sha256": "21316f108eca8475cedba9d43e10aeee4b5f7fe8f2f18718744415e566fb1e97",
                    "functional_veto_policy": "deferred_non_scope",
                    "global_deferred_ratio": 0.5,
                    "global_pre_cap_would_apply_count": 2,
                    "global_rate_cap_accepted_count": 1,
                    "global_rate_cap_applied_count": 1,
                    "global_rate_cap_cap": 1,
                    "global_rate_cap_contract_name": "c1_banked_faithful_long_run_global_cap",
                    "global_rate_cap_deferred_count": 1,
                    "global_rate_cap_enabled": True,
                    "global_rate_cap_fill_ratio": 1.0,
                    "global_rate_cap_ordering_mode": "margin",
                    "global_rate_cap_ordering_seed": 17,
                    "global_rate_cap_ordering_summary": {
                        "default_margin_behavior_equivalent": True,
                        "deferred_count": 1,
                        "full_demand_count": 2,
                        "global_indices_sha256": {
                            "cap_deferred": "0eebb09249ec06c9a0c57a4bc7c0c65cfb1d43d061053519feea2079ef706e10",
                            "cap_selected": "c773b27f4c5b93a8227463ca4116357e78645021908de6fbace90c9ff684fe39",
                            "full_demand": "21316f108eca8475cedba9d43e10aeee4b5f7fe8f2f18718744415e566fb1e97"
                        },
                        "global_step": 3,
                        "mode": "margin",
                        "order_key": "highest_abs_new_acc_then_lower_global_flat_index",
                        "schema_version": "global_rate_cap_ordering/v1",
                        "seed": 17,
                        "selected_count": 1
                    },
                    "global_rate_cap_saturated": True,
                    "global_tie_rule_mode": "exact_global_cap",
                    "local_selection_ordering_mode": "current_abs_new_acc_then_index",
                    "local_selection_ordering_seed": 0,
                    "local_selection_ordering_step": 0,
                    "max_mixed_class_cardinality": 0,
                    "mixed_class_count": 0,
                    "mixed_class_row_count": 0,
                    "pre_cap_demand_sha256": "21316f108eca8475cedba9d43e10aeee4b5f7fe8f2f18718744415e566fb1e97",
                    "q_changed_count": 1,
                    "ternary_mutation_enabled": True,
                    "ternary_mutation_frozen": False
                },
                "tensor_stats": {
                    "toy.proj": {
                        "acc_abs_max_after_decay_vote": 12,
                        "bad_pressure_drain_policy": "deferred_non_scope",
                        "bounded_accumulator_fresh_for_exact_shadow": False,
                        "bounded_accumulator_rebuilt_for_parity": False,
                        "bounded_decode_parity_checked": False,
                        "bounded_update_attribution": "q_acc_backlog_changed_by_bounded_delta_vote_update_only",
                        "candidate_count": 2,
                        "cpu_glue_not_kernel": True,
                        "exact_accumulator_shadow_sha256_after": "201799a63bcada705a62d4aefb7bc6750ce76bd1614afb2e2d954aa6454c132a",
                        "flip_count": 1,
                        "functional_veto_policy": "deferred_non_scope",
                        "global_cap_policy": "deferred_global_cap",
                        "global_rate_cap_accepted_count": 1,
                        "global_rate_cap_accepted_indices": [
                            0
                        ],
                        "global_rate_cap_accepted_indices_sample": [
                            0
                        ],
                        "global_rate_cap_accepted_indices_sha256": "c773b27f4c5b93a8227463ca4116357e78645021908de6fbace90c9ff684fe39",
                        "global_rate_cap_applied_count": 1,
                        "global_rate_cap_cap": 1,
                        "global_rate_cap_deferred_count": 1,
                        "global_rate_cap_deferred_indices": [
                            1
                        ],
                        "global_rate_cap_deferred_indices_sample": [
                            1
                        ],
                        "global_rate_cap_deferred_indices_sha256": "0eebb09249ec06c9a0c57a4bc7c0c65cfb1d43d061053519feea2079ef706e10",
                        "global_rate_cap_enabled": True,
                        "global_rate_cap_ordering_mode": "margin",
                        "global_rate_cap_ordering_seed": 17,
                        "global_rate_cap_would_accept_count": 1,
                        "local_selection_ordering_mode": "current_abs_new_acc_then_index",
                        "local_selection_ordering_seed": 0,
                        "local_selection_ordering_step": 0,
                        "max_flips": 2,
                        "pc_aux_mode": "telemetry",
                        "pc_aux_negative_count": 0,
                        "pc_aux_veto_accumulator_residual_policy": "not_enabled",
                        "pc_aux_veto_count": 0,
                        "pc_aux_veto_enabled": False,
                        "post_veto_acceptance_ratio_pre_cap": 1.0,
                        "post_veto_applied_flip_count": 1,
                        "post_veto_applied_indices": [
                            0
                        ],
                        "post_veto_would_apply_pre_cap_count": 2,
                        "pre_veto_selected_flip_count": 2,
                        "projection_law": "ported_s1_gradient_sign_to_ternary_move",
                        "q_changed_count": 1,
                        "q_sha256_after": "9706a41a0b415de3ee46d21ed08c2db29c97aeb1109008c3bc7667360eac817d",
                        "q_sha256_before": "6ea281ada06034249ad191989928cfca8b69020f82aef00111325ed5f0c8a30e",
                        "replay_ce_veto_consumes_threshold_event": False,
                        "replay_ce_veto_count": 0,
                        "scope": "global_rate_cap_reference",
                        "state_key": "toy.proj",
                        "ternary_mutation_enabled": True,
                        "ternary_mutation_frozen": False,
                        "threshold_jitter_policy": "deferred_reject",
                        "two_b_input_name": "2A applied_indices are local_post_veto_pre_global_cap_candidates",
                        "vetoed_accumulator_clamp_count": 0,
                        "vetoed_accumulator_residual_policy": "not_enabled",
                        "vote_law": "ported_s1_rank_bucketed_integer_votes",
                        "vote_nonzero_count": 2,
                        "votes_sha256": "90b410161b3257c3414cd20a456102333bf6db87e1e9138738fe4bc36d3fb5bb"
                    }
                }
            },
            "S1_q_levels": [
                1,
                0
            ],
            "S2_accumulators": [
                2,
                12
            ],
            "S3_plan_candidate_indices": [
                0,
                1
            ],
            "S4_plan_pre_veto_selected": [
                0,
                1
            ],
            "S5_plan_applied_triple": {
                "applied_directions": [
                    1,
                    1
                ],
                "applied_indices": [
                    0,
                    1
                ],
                "applied_thresholds": [
                    10,
                    10
                ]
            },
            "S6_plan_veto_tensors": {
                "pc_aux_negative_indices": [],
                "pc_aux_veto_indices": [],
                "replay_ce_veto_indices": [],
                "replay_veto_directions": [],
                "replay_veto_thresholds": []
            },
            "S7_plan_stats": {
                "acc_abs_max_after_decay_vote": 12,
                "candidate_count": 2,
                "global_cap_policy": "deferred_global_cap",
                "local_selection_ordering_mode": "current_abs_new_acc_then_index",
                "local_selection_ordering_seed": 0,
                "local_selection_ordering_step": 0,
                "max_flips": 2,
                "pc_aux_mode": "telemetry",
                "pc_aux_negative_count": 0,
                "pc_aux_veto_accumulator_residual_policy": "not_enabled",
                "pc_aux_veto_count": 0,
                "pc_aux_veto_enabled": False,
                "post_veto_acceptance_ratio_pre_cap": 1.0,
                "post_veto_would_apply_pre_cap_count": 2,
                "pre_veto_selected_flip_count": 2,
                "replay_ce_veto_consumes_threshold_event": False,
                "replay_ce_veto_count": 0,
                "scope": "per_tensor_local_update",
                "threshold_jitter_policy": "deferred_reject",
                "vetoed_accumulator_clamp_count": 0,
                "vetoed_accumulator_residual_policy": "not_enabled",
                "vote_nonzero_count": 2
            },
            "S8_replay_ce_veto_partition": {
                "applied_indices": [
                    0,
                    1
                ],
                "replay_ce_veto_indices": []
            },
            "S9_pc_aux_veto": {
                "pc_aux_mode": "telemetry",
                "pc_aux_negative_indices": [],
                "pc_aux_veto_indices": []
            },
            "name": "global_cap_exact"
        }
    ],
    "fixture_id": "B6_OFF_GOLDEN_DFDB5CFE_V0",
    "off_semantics_basis_sha": "dfdb5cfe5301cd3a7b58a183ddff56c353d2761c"
}


def _b6_tensor_list(tensor: torch.Tensor) -> list:
    return tensor.detach().cpu().reshape(-1).tolist()


def _b6_sanitize(value):
    if isinstance(value, torch.Tensor):
        return _b6_tensor_list(value)
    if is_dataclass(value):
        return _b6_sanitize(asdict(value))
    if isinstance(value, dict):
        out = {str(k): _b6_sanitize(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
        if out.get("local_loss_delta") is None:
            out.pop("local_loss_delta", None)
        return out
    if isinstance(value, (list, tuple)):
        return [_b6_sanitize(v) for v in value]
    if hasattr(value, "value") and type(value).__name__ in {"PcAuxMode", "VoteUpdateVoteFormat"}:
        return value.value
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _b6_backlog_case_kwargs(*, two_tier_carry_w6_enabled: bool = False, local_loss_delta=None):
    kwargs = {
        "tensor_states": {
            "toy.proj": make_bounded_tensor_state(
                "toy.proj",
                torch.tensor([0, 0, 0, 0], dtype=torch.int8),
                0.5,
                torch.zeros(4, dtype=torch.int16),
                hot_exact_indices=(0, 2),
            )
        },
        "votes_by_key": {"toy.proj": torch.tensor([3, 0, -3, 0], dtype=torch.int16)},
        "vote_specs_by_key": {
            "toy.proj": VoteUpdateSpec(
                threshold_abs=2,
                accumulator_clip_min=-127,
                accumulator_clip_max=127,
                max_abs_per_tensor=8,
            )
        },
        "parity_check": True,
        "two_tier_carry_w6_enabled": two_tier_carry_w6_enabled,
    }
    if local_loss_delta is not None:
        kwargs["local_loss_delta_by_key"] = {"toy.proj": local_loss_delta}
    return kwargs


def _b6_replay_pc_case_kwargs(*, two_tier_carry_w6_enabled: bool = False, local_loss_delta=None):
    kwargs = {
        "tensor_states": {
            "toy.proj": make_bounded_tensor_state(
                "toy.proj",
                torch.tensor([0], dtype=torch.int8),
                0.5,
                torch.zeros(1, dtype=torch.int16),
            )
        },
        "votes_by_key": {"toy.proj": torch.tensor([12], dtype=torch.int16)},
        "vote_specs_by_key": {
            "toy.proj": VoteUpdateSpec(
                threshold_abs=10,
                accumulator_clip_min=-127,
                accumulator_clip_max=127,
                max_abs_per_tensor=1,
            )
        },
        "replay_ce_veto_votes_by_key": {"toy.proj": torch.tensor([0], dtype=torch.int16)},
        "replay_ce_veto_moves_by_key": {"toy.proj": torch.tensor([0], dtype=torch.int8)},
        "pc_aux_votes_by_key": {"toy.proj": torch.tensor([-1], dtype=torch.int16)},
        "pc_aux_moves_by_key": {"toy.proj": torch.tensor([0], dtype=torch.int8)},
        "pc_aux_mode": "veto",
        "two_tier_carry_w6_enabled": two_tier_carry_w6_enabled,
    }
    if local_loss_delta is not None:
        kwargs["local_loss_delta_by_key"] = {"toy.proj": local_loss_delta}
    return kwargs


def _b6_global_cap_case_kwargs(*, two_tier_carry_w6_enabled: bool = False, local_loss_delta=None):
    kwargs = {
        "tensor_states": {
            "toy.proj": make_bounded_tensor_state(
                "toy.proj",
                torch.tensor([0, 0], dtype=torch.int8),
                0.5,
                torch.zeros(2, dtype=torch.int16),
            )
        },
        "votes_by_key": {"toy.proj": torch.tensor([12, 12], dtype=torch.int16)},
        "vote_specs_by_key": {
            "toy.proj": VoteUpdateSpec(
                threshold_abs=10,
                accumulator_clip_min=-127,
                accumulator_clip_max=127,
                max_abs_per_tensor=2,
            )
        },
        "global_cap_spec": GlobalRateCapSpec(cap=1, step=3),
        "global_cap_tie_rule_mode": EXACT_GLOBAL_CAP_TIE_RULE_MODE,
        "global_cap_contract_name": C1_BANKED_FAITHFUL_LONG_RUN_GLOBAL_CAP_CONTRACT_NAME,
        "two_tier_carry_w6_enabled": two_tier_carry_w6_enabled,
    }
    if local_loss_delta is not None:
        kwargs["local_loss_delta_by_key"] = {"toy.proj": local_loss_delta}
    return kwargs


_B6_CASE_BUILDERS = {
    "backlog_parity": _b6_backlog_case_kwargs,
    "replay_pc_veto": _b6_replay_pc_case_kwargs,
    "global_cap_exact": _b6_global_cap_case_kwargs,
}


def _b6_run_case(name: str, **overrides):
    builder = _B6_CASE_BUILDERS[name]
    return apply_bounded_delta_vote_step(**builder(**overrides))


def _b6_capture_live_surfaces(name: str, **overrides) -> dict:
    kwargs = _B6_CASE_BUILDERS[name](**overrides)
    result = apply_bounded_delta_vote_step(**kwargs)
    key = next(iter(result.tensor_states))
    state = kwargs["tensor_states"][key]
    vu_state = state.vote_update_state()
    votes = kwargs["votes_by_key"][key].detach().cpu().to(torch.int16).contiguous()
    inputs = VoteUpdateInputs(votes=votes)
    if kwargs.get("replay_ce_veto_votes_by_key"):
        inputs = VoteUpdateInputs(
            votes=votes,
            replay_ce_veto_votes=kwargs["replay_ce_veto_votes_by_key"][key],
            replay_ce_veto_moves=kwargs["replay_ce_veto_moves_by_key"][key],
            pc_aux_votes=kwargs.get("pc_aux_votes_by_key", {}).get(key),
            pc_aux_moves=kwargs.get("pc_aux_moves_by_key", {}).get(key),
            pc_aux_mode=kwargs.get("pc_aux_mode", "telemetry"),
        )
    spec = kwargs["vote_specs_by_key"][key]
    plan = plan_integer_vote_update_reference(
        vu_state,
        inputs,
        spec,
        local_selection_ordering_mode=str(
            kwargs.get("local_selection_ordering_mode", LOCAL_SELECTION_ORDER_CURRENT_MARGIN_INDEX)
        ),
    )
    applied = apply_integer_vote_update_reference(
        vu_state,
        inputs,
        spec,
        local_selection_ordering_mode=str(
            kwargs.get("local_selection_ordering_mode", LOCAL_SELECTION_ORDER_CURRENT_MARGIN_INDEX)
        ),
    )
    front_c = _front_c_cloned_observation(
        vote_update_states={key: vu_state},
        inputs_by_key={key: inputs},
        vote_specs_by_key={key: spec},
        plans_by_key={key: plan},
        q_acc_by_key={key: (applied.q_levels, applied.accumulators, dict(applied.stats))},
        deferred_backlog=result.deferred_backlog,
        global_cap_used=kwargs.get("global_cap_spec") is not None,
    )
    return {
        "S1_q_levels": _b6_tensor_list(result.tensor_states[key].q_levels),
        "S2_accumulators": _b6_tensor_list(result.tensor_states[key].exact_accumulator_shadow),
        "S3_plan_candidate_indices": _b6_tensor_list(plan.candidate_indices),
        "S4_plan_pre_veto_selected": _b6_tensor_list(plan.pre_veto_selected_indices),
        "S5_plan_applied_triple": {
            "applied_indices": _b6_tensor_list(plan.applied_indices),
            "applied_directions": _b6_tensor_list(plan.applied_directions),
            "applied_thresholds": _b6_tensor_list(plan.applied_thresholds),
        },
        "S6_plan_veto_tensors": {
            "replay_ce_veto_indices": _b6_tensor_list(plan.replay_ce_veto_indices),
            "replay_veto_directions": _b6_tensor_list(plan.replay_veto_directions),
            "replay_veto_thresholds": _b6_tensor_list(plan.replay_veto_thresholds),
            "pc_aux_negative_indices": _b6_tensor_list(plan.pc_aux_negative_indices),
            "pc_aux_veto_indices": _b6_tensor_list(plan.pc_aux_veto_indices),
        },
        "S7_plan_stats": {k: plan.stats[k] for k in sorted(plan.stats)},
        "S8_replay_ce_veto_partition": {
            "replay_ce_veto_indices": _b6_tensor_list(plan.replay_ce_veto_indices),
            "applied_indices": _b6_tensor_list(plan.applied_indices),
        },
        "S9_pc_aux_veto": {
            "pc_aux_negative_indices": _b6_tensor_list(plan.pc_aux_negative_indices),
            "pc_aux_veto_indices": _b6_tensor_list(plan.pc_aux_veto_indices),
            "pc_aux_mode": inputs.normalized_pc_aux_mode.value,
        },
        "S10_global_cap": {
            "global_rate_cap_enabled": result.global_summary.get("global_rate_cap_enabled"),
            "deferred_backlog": _b6_sanitize(result.deferred_backlog),
            "q_levels": _b6_tensor_list(result.tensor_states[key].q_levels),
        },
        "S11_front_c_clone": _b6_sanitize(front_c),
        "S12_learner_summary": {
            "global_summary": _b6_sanitize(result.global_summary),
            "tensor_stats": {k: _b6_sanitize(v) for k, v in sorted(result.tensor_stats.items())},
        },
    }


def test_b6_flag_off_matches_frozen_golden_dfdb5cfe_v0():
    fixture_payload = json.dumps(B6_OFF_GOLDEN_FIXTURE, sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(fixture_payload.encode()).hexdigest() == B6_OFF_GOLDEN_SHA256
    for case in B6_OFF_GOLDEN_FIXTURE["cases"]:
        live = _b6_capture_live_surfaces(case["name"], two_tier_carry_w6_enabled=False)
        for surface_key, expected in case.items():
            if surface_key == "name":
                continue
            assert live[surface_key] == expected, f"surface {surface_key} for case {case['name']}"


@pytest.mark.parametrize("case_name", ["backlog_parity", "replay_pc_veto", "global_cap_exact"])
def test_b6_flag_off_q_levels_match_baseline(case_name):
    off = _b6_run_case(case_name, two_tier_carry_w6_enabled=False)
    key = next(iter(off.tensor_states))
    idx = ["backlog_parity", "replay_pc_veto", "global_cap_exact"].index(case_name)
    assert _b6_tensor_list(off.tensor_states[key].q_levels) == B6_OFF_GOLDEN_FIXTURE["cases"][idx]["S1_q_levels"]


@pytest.mark.parametrize("case_name", ["backlog_parity", "replay_pc_veto", "global_cap_exact"])
def test_b6_flag_off_accumulators_match_baseline(case_name):
    off = _b6_run_case(case_name, two_tier_carry_w6_enabled=False)
    key = next(iter(off.tensor_states))
    idx = ["backlog_parity", "replay_pc_veto", "global_cap_exact"].index(case_name)
    assert _b6_tensor_list(off.tensor_states[key].exact_accumulator_shadow) == B6_OFF_GOLDEN_FIXTURE["cases"][idx]["S2_accumulators"]


def _b6_build_vote_update_inputs(key: str, kwargs: dict) -> VoteUpdateInputs:
    votes = kwargs["votes_by_key"][key].detach().cpu().to(torch.int16).contiguous()
    local_loss_delta = None
    if kwargs.get("local_loss_delta_by_key") is not None:
        local_loss_delta = kwargs["local_loss_delta_by_key"][key]
    return VoteUpdateInputs(
        votes=votes,
        replay_ce_veto_votes=(
            None
            if kwargs.get("replay_ce_veto_votes_by_key") is None
            else kwargs["replay_ce_veto_votes_by_key"][key]
        ),
        replay_ce_veto_moves=(
            None
            if kwargs.get("replay_ce_veto_moves_by_key") is None
            else kwargs["replay_ce_veto_moves_by_key"][key]
        ),
        pc_aux_votes=(
            None if kwargs.get("pc_aux_votes_by_key") is None else kwargs["pc_aux_votes_by_key"][key]
        ),
        pc_aux_moves=(
            None if kwargs.get("pc_aux_moves_by_key") is None else kwargs["pc_aux_moves_by_key"][key]
        ),
        pc_aux_mode=kwargs.get("pc_aux_mode", "telemetry"),
        local_loss_delta=local_loss_delta,
    )


def _b6_plan_from_step_kwargs(kwargs: dict):
    key = next(iter(kwargs["tensor_states"]))
    state = kwargs["tensor_states"][key]
    inputs = _b6_build_vote_update_inputs(key, kwargs)
    return plan_integer_vote_update_reference(
        state.vote_update_state(),
        inputs,
        kwargs["vote_specs_by_key"][key],
        local_selection_ordering_mode=str(
            kwargs.get("local_selection_ordering_mode", LOCAL_SELECTION_ORDER_CURRENT_MARGIN_INDEX)
        ),
        two_tier_carry_w6_enabled=bool(kwargs.get("two_tier_carry_w6_enabled", False)),
    )


def _b6_plan_for_case(case_name: str, **overrides):
    kwargs = _B6_CASE_BUILDERS[case_name](**overrides)
    return _b6_plan_from_step_kwargs(kwargs)


@pytest.mark.parametrize("case_name", ["backlog_parity", "replay_pc_veto", "global_cap_exact"])
def test_b6_flag_off_plan_candidate_indices(case_name):
    plan = _b6_plan_for_case(case_name, two_tier_carry_w6_enabled=False)
    idx = ["backlog_parity", "replay_pc_veto", "global_cap_exact"].index(case_name)
    assert _b6_tensor_list(plan.candidate_indices) == B6_OFF_GOLDEN_FIXTURE["cases"][idx]["S3_plan_candidate_indices"]


@pytest.mark.parametrize("case_name", ["backlog_parity", "replay_pc_veto", "global_cap_exact"])
def test_b6_flag_off_plan_pre_veto_selected(case_name):
    plan = _b6_plan_for_case(case_name, two_tier_carry_w6_enabled=False)
    idx = ["backlog_parity", "replay_pc_veto", "global_cap_exact"].index(case_name)
    assert _b6_tensor_list(plan.pre_veto_selected_indices) == B6_OFF_GOLDEN_FIXTURE["cases"][idx]["S4_plan_pre_veto_selected"]


@pytest.mark.parametrize("case_name", ["backlog_parity", "replay_pc_veto", "global_cap_exact"])
def test_b6_flag_off_plan_applied_triple(case_name):
    plan = _b6_plan_for_case(case_name, two_tier_carry_w6_enabled=False)
    idx = ["backlog_parity", "replay_pc_veto", "global_cap_exact"].index(case_name)
    live = {
        "applied_indices": _b6_tensor_list(plan.applied_indices),
        "applied_directions": _b6_tensor_list(plan.applied_directions),
        "applied_thresholds": _b6_tensor_list(plan.applied_thresholds),
    }
    assert live == B6_OFF_GOLDEN_FIXTURE["cases"][idx]["S5_plan_applied_triple"]


@pytest.mark.parametrize("case_name", ["backlog_parity", "replay_pc_veto", "global_cap_exact"])
def test_b6_flag_off_plan_veto_tensors(case_name):
    plan = _b6_plan_for_case(case_name, two_tier_carry_w6_enabled=False)
    idx = ["backlog_parity", "replay_pc_veto", "global_cap_exact"].index(case_name)
    live = {
        "replay_ce_veto_indices": _b6_tensor_list(plan.replay_ce_veto_indices),
        "replay_veto_directions": _b6_tensor_list(plan.replay_veto_directions),
        "replay_veto_thresholds": _b6_tensor_list(plan.replay_veto_thresholds),
        "pc_aux_negative_indices": _b6_tensor_list(plan.pc_aux_negative_indices),
        "pc_aux_veto_indices": _b6_tensor_list(plan.pc_aux_veto_indices),
    }
    assert live == B6_OFF_GOLDEN_FIXTURE["cases"][idx]["S6_plan_veto_tensors"]


@pytest.mark.parametrize("case_name", ["backlog_parity", "replay_pc_veto", "global_cap_exact"])
def test_b6_flag_off_plan_stats_full_dict(case_name):
    plan = _b6_plan_for_case(case_name, two_tier_carry_w6_enabled=False)
    idx = ["backlog_parity", "replay_pc_veto", "global_cap_exact"].index(case_name)
    assert {k: plan.stats[k] for k in sorted(plan.stats)} == B6_OFF_GOLDEN_FIXTURE["cases"][idx]["S7_plan_stats"]


@pytest.mark.parametrize("case_name", ["replay_pc_veto"])
def test_b6_flag_off_replay_ce_veto_partition(case_name):
    plan = _b6_plan_for_case(case_name, two_tier_carry_w6_enabled=False)
    idx = ["backlog_parity", "replay_pc_veto", "global_cap_exact"].index(case_name)
    live = {
        "replay_ce_veto_indices": _b6_tensor_list(plan.replay_ce_veto_indices),
        "applied_indices": _b6_tensor_list(plan.applied_indices),
    }
    assert live == B6_OFF_GOLDEN_FIXTURE["cases"][idx]["S8_replay_ce_veto_partition"]


@pytest.mark.parametrize("case_name", ["replay_pc_veto"])
def test_b6_flag_off_pc_aux_veto_modes(case_name):
    live = _b6_capture_live_surfaces(case_name, two_tier_carry_w6_enabled=False)["S9_pc_aux_veto"]
    idx = ["backlog_parity", "replay_pc_veto", "global_cap_exact"].index(case_name)
    assert live == B6_OFF_GOLDEN_FIXTURE["cases"][idx]["S9_pc_aux_veto"]


@pytest.mark.parametrize("case_name", ["global_cap_exact"])
def test_b6_flag_off_global_cap_and_backlog(case_name):
    live = _b6_capture_live_surfaces(case_name, two_tier_carry_w6_enabled=False)["S10_global_cap"]
    idx = ["backlog_parity", "replay_pc_veto", "global_cap_exact"].index(case_name)
    assert live == B6_OFF_GOLDEN_FIXTURE["cases"][idx]["S10_global_cap"]


@pytest.mark.parametrize("case_name", ["backlog_parity", "replay_pc_veto", "global_cap_exact"])
def test_b6_flag_off_front_c_clone_parity(case_name):
    live = _b6_capture_live_surfaces(case_name, two_tier_carry_w6_enabled=False)["S11_front_c_clone"]
    idx = ["backlog_parity", "replay_pc_veto", "global_cap_exact"].index(case_name)
    assert live == B6_OFF_GOLDEN_FIXTURE["cases"][idx]["S11_front_c_clone"]


@pytest.mark.parametrize("case_name", ["backlog_parity", "replay_pc_veto", "global_cap_exact"])
def test_b6_flag_off_learner_summary_and_tensor_stats(case_name):
    live = _b6_capture_live_surfaces(case_name, two_tier_carry_w6_enabled=False)["S12_learner_summary"]
    idx = ["backlog_parity", "replay_pc_veto", "global_cap_exact"].index(case_name)
    assert live == B6_OFF_GOLDEN_FIXTURE["cases"][idx]["S12_learner_summary"]


def test_b6_flag_off_inert_optional_local_loss_delta_preserves_s1_s12():
    delta = torch.zeros(4, dtype=torch.float32)
    baseline = _b6_capture_live_surfaces("backlog_parity", two_tier_carry_w6_enabled=False)
    with_delta = _b6_capture_live_surfaces(
        "backlog_parity",
        two_tier_carry_w6_enabled=False,
        local_loss_delta=delta,
    )
    for surface_key in baseline:
        assert with_delta[surface_key] == baseline[surface_key], f"surface {surface_key}"


def test_b6_flag_on_fail_closed_missing_local_loss_delta_map():
    with pytest.raises(ValueError, match="local_loss_delta_by_key_required_when_two_tier_enabled"):
        apply_bounded_delta_vote_step(
            **_b6_backlog_case_kwargs(two_tier_carry_w6_enabled=True),
            local_selection_ordering_mode=LOCAL_SELECTION_ORDER_TRANSIENT_LOCAL_LOSS_DELTA,
        )


def test_b6_flag_on_fail_closed_wrong_ordering_mode():
    with pytest.raises(ValueError, match="two_tier_carry_w6_enabled requires"):
        apply_bounded_delta_vote_step(
            **_b6_backlog_case_kwargs(
                two_tier_carry_w6_enabled=True,
                local_loss_delta=torch.zeros(4, dtype=torch.float32),
            ),
            local_selection_ordering_mode=LOCAL_SELECTION_ORDER_CURRENT_MARGIN_INDEX,
        )


def test_b6_flag_on_fail_closed_bad_delta_dtype():
    with pytest.raises(ValueError, match="local_loss_delta_bad_dtype"):
        apply_bounded_delta_vote_step(
            **_b6_backlog_case_kwargs(
                two_tier_carry_w6_enabled=True,
                local_loss_delta=torch.zeros(4, dtype=torch.float64),
            ),
            local_selection_ordering_mode=LOCAL_SELECTION_ORDER_TRANSIENT_LOCAL_LOSS_DELTA,
        )


def test_b6_flag_on_fail_closed_delta_shape_mismatch():
    with pytest.raises(ValueError, match="local_loss_delta_shape_mismatch"):
        apply_bounded_delta_vote_step(
            **_b6_backlog_case_kwargs(
                two_tier_carry_w6_enabled=True,
                local_loss_delta=torch.zeros(3, dtype=torch.float32),
            ),
            local_selection_ordering_mode=LOCAL_SELECTION_ORDER_TRANSIENT_LOCAL_LOSS_DELTA,
        )


def test_b6_flag_on_fail_closed_non_finite_delta():
    bad = torch.zeros(4, dtype=torch.float32)
    bad[0] = float("nan")
    with pytest.raises(ValueError, match="local_loss_delta_non_finite"):
        apply_bounded_delta_vote_step(
            **_b6_backlog_case_kwargs(two_tier_carry_w6_enabled=True, local_loss_delta=bad),
            local_selection_ordering_mode=LOCAL_SELECTION_ORDER_TRANSIENT_LOCAL_LOSS_DELTA,
        )


def test_b6_flag_on_enabled_pin_fields_present_only_when_on():
    off = _b6_run_case("backlog_parity", two_tier_carry_w6_enabled=False)
    for key in (
        "two_tier_carry_w6_enabled",
        "two_tier_enabled_pin_count",
        "two_tier_enabled_pin_sha256",
    ):
        assert key not in off.global_summary
    on = apply_bounded_delta_vote_step(
        **_b6_backlog_case_kwargs(
            two_tier_carry_w6_enabled=True,
            local_loss_delta=torch.tensor([-0.9, -0.1, -0.8, 0.0], dtype=torch.float32),
        ),
        local_selection_ordering_mode=LOCAL_SELECTION_ORDER_TRANSIENT_LOCAL_LOSS_DELTA,
    )
    assert on.global_summary["two_tier_carry_w6_enabled"] is True
    assert on.global_summary["local_selection_ordering_mode"] == LOCAL_SELECTION_ORDER_TRANSIENT_LOCAL_LOSS_DELTA
    assert on.global_summary["two_tier_enabled_pin_count"] == 1
    assert len(on.global_summary["two_tier_enabled_pin_sha256"]) == 64


def test_b6_flag_on_carry_all_rows_non_applied_evolve():
    deltas = torch.tensor([-0.9, -0.1, -0.8, 0.0], dtype=torch.float32)
    result = apply_bounded_delta_vote_step(
        **_b6_backlog_case_kwargs(two_tier_carry_w6_enabled=True, local_loss_delta=deltas),
        local_selection_ordering_mode=LOCAL_SELECTION_ORDER_TRANSIENT_LOCAL_LOSS_DELTA,
    )
    key = "toy.proj"
    acc = result.tensor_states[key].exact_accumulator_shadow.tolist()
    assert acc[1] == carry_self_update_row(0, 0)
    assert acc[3] == carry_self_update_row(0, 0)


def test_b6_flag_on_matches_orchestrator_fixture():
    rows = [
        {
            "candidate_id": "0",
            "flat_index": 0,
            "vote_value": 12,
            "pre_accumulator_i16": 0,
            "current_q_level": 0,
            "local_loss_delta": -0.9,
            "in_target_tie_band": True,
        },
        {
            "candidate_id": "2",
            "flat_index": 2,
            "vote_value": -12,
            "pre_accumulator_i16": 0,
            "current_q_level": 0,
            "local_loss_delta": -0.8,
            "in_target_tie_band": True,
        },
    ]
    orchestrator = run_two_tier_optimizer_step(
        rows,
        carry_by_flat_index={0: 0, 2: 0},
        q_level_by_flat_index={0: 0, 2: 0},
        rate_cap=8,
        warmup=False,
        threshold_abs=CROSSING_THRESHOLD_ABS,
    )
    kwargs = _b6_backlog_case_kwargs(two_tier_carry_w6_enabled=True)
    kwargs["votes_by_key"] = {"toy.proj": torch.tensor([12, 0, -12, 0], dtype=torch.int16)}
    kwargs["local_loss_delta_by_key"] = {
        "toy.proj": torch.tensor([-0.9, 0.0, -0.8, 0.0], dtype=torch.float32),
    }
    learner = apply_bounded_delta_vote_step(
        **kwargs,
        local_selection_ordering_mode=LOCAL_SELECTION_ORDER_TRANSIENT_LOCAL_LOSS_DELTA,
    )
    key = "toy.proj"
    q = learner.tensor_states[key].q_levels.tolist()
    assert q[0] == orchestrator.q_level_after_by_flat_index[0]
    assert q[2] == orchestrator.q_level_after_by_flat_index[2]


def test_b6_flag_on_fail_closed_missing_local_loss_delta_key():
    state_a = make_bounded_tensor_state(
        "tensor.a",
        torch.tensor([0], dtype=torch.int8),
        0.5,
        torch.zeros(1, dtype=torch.int16),
    )
    state_b = make_bounded_tensor_state(
        "tensor.b",
        torch.tensor([0], dtype=torch.int8),
        0.5,
        torch.zeros(1, dtype=torch.int16),
    )
    spec = VoteUpdateSpec(
        threshold_abs=10,
        accumulator_clip_min=-127,
        accumulator_clip_max=127,
        max_abs_per_tensor=1,
    )
    with pytest.raises(ValueError, match=r"local_loss_delta_by_key.*missing=\['tensor.b'\]"):
        apply_bounded_delta_vote_step(
            tensor_states={"tensor.a": state_a, "tensor.b": state_b},
            votes_by_key={
                "tensor.a": torch.tensor([12], dtype=torch.int16),
                "tensor.b": torch.tensor([12], dtype=torch.int16),
            },
            vote_specs_by_key={"tensor.a": spec, "tensor.b": spec},
            local_loss_delta_by_key={
                "tensor.a": torch.tensor([-0.1], dtype=torch.float32),
            },
            two_tier_carry_w6_enabled=True,
            local_selection_ordering_mode=LOCAL_SELECTION_ORDER_TRANSIENT_LOCAL_LOSS_DELTA,
        )


def test_b6_flag_on_fresh_parent_subthreshold_vote_spec_does_not_apply_at_t10():
    numel = 8192
    state = make_bounded_tensor_state(
        "toy.proj",
        torch.zeros(numel, dtype=torch.int8),
        0.5,
        torch.zeros(numel, dtype=torch.int16),
    )
    votes = torch.full((numel,), 4, dtype=torch.int16)
    spec = VoteUpdateSpec(
        threshold_abs=1,
        accumulator_clip_min=-127,
        accumulator_clip_max=127,
        max_abs_per_tensor=4096,
    )
    tensor_states = {"toy.proj": state}
    votes_by_key = {"toy.proj": votes}
    assert count_w6_t10_crossing_eligible_from_votes(
        tensor_states=tensor_states,
        votes_by_key=votes_by_key,
    ) == 0
    result = apply_bounded_delta_vote_step(
        tensor_states,
        votes_by_key,
        {"toy.proj": spec},
        local_loss_delta_by_key={
            "toy.proj": torch.zeros(numel, dtype=torch.float32),
        },
        two_tier_carry_w6_enabled=True,
        local_selection_ordering_mode=LOCAL_SELECTION_ORDER_TRANSIENT_LOCAL_LOSS_DELTA,
    )
    stats = result.tensor_stats["toy.proj"]
    assert stats["candidate_count"] == 0
    assert stats["post_veto_applied_flip_count"] == 0
    assert stats["two_tier_threshold_abs"] == stats["two_tier_canonical_threshold_abs"] == 10
    assert stats["two_tier_vote_spec_threshold_abs"] == 1
    assert result.tensor_states["toy.proj"].q_levels.eq(0).all()


def test_b6_flag_on_plan_two_tier_step_noncanonical_threshold_selects_subthreshold_rows():
    rows = [
        {
            "candidate_id": "0",
            "flat_index": 0,
            "vote_value": 4,
            "pre_accumulator_i16": 0,
            "current_q_level": 0,
            "local_loss_delta": 0.0,
            "in_target_tie_band": True,
        },
    ]
    subthreshold_plan = plan_two_tier_step(
        rows,
        carry_by_flat_index={0: 0},
        q_level_by_flat_index={0: 0},
        rate_cap=1,
        warmup=False,
        threshold_abs=1,
    )
    canonical_plan = plan_two_tier_step(
        rows,
        carry_by_flat_index={0: 0},
        q_level_by_flat_index={0: 0},
        rate_cap=1,
        warmup=False,
        threshold_abs=CROSSING_THRESHOLD_ABS,
    )
    assert subthreshold_plan.pre_veto_flat_indices == (0,)
    assert canonical_plan.pre_veto_flat_indices == ()


def test_b6_flag_on_vote_update_overrides_noncanonical_spec_threshold():
    state = make_bounded_tensor_state(
        "toy.proj",
        torch.tensor([0], dtype=torch.int8),
        0.5,
        torch.zeros(1, dtype=torch.int16),
    )
    votes = torch.tensor([4], dtype=torch.int16)
    spec = VoteUpdateSpec(
        threshold_abs=1,
        accumulator_clip_min=-127,
        accumulator_clip_max=127,
        max_abs_per_tensor=1,
    )
    plan = plan_two_tier_vote_update_reference(
        state.vote_update_state(),
        VoteUpdateInputs(
            votes=votes,
            local_loss_delta=torch.zeros(1, dtype=torch.float32),
        ),
        spec,
        local_selection_ordering_mode=LOCAL_SELECTION_ORDER_TRANSIENT_LOCAL_LOSS_DELTA,
    )
    assert int(plan.stats["candidate_count"]) == 0
    assert int(plan.stats["pre_veto_selected_flip_count"]) == 0
    assert plan.stats["two_tier_threshold_abs"] == plan.stats["two_tier_canonical_threshold_abs"]


def test_b6_flag_on_replay_veto_excludes_applied_writeback():
    kwargs = {
        "tensor_states": {
            "toy.proj": make_bounded_tensor_state(
                "toy.proj",
                torch.tensor([0], dtype=torch.int8),
                0.5,
                torch.zeros(1, dtype=torch.int16),
            )
        },
        "votes_by_key": {"toy.proj": torch.tensor([12], dtype=torch.int16)},
        "vote_specs_by_key": {
            "toy.proj": VoteUpdateSpec(
                threshold_abs=10,
                accumulator_clip_min=-127,
                accumulator_clip_max=127,
                max_abs_per_tensor=1,
            )
        },
        "replay_ce_veto_votes_by_key": {"toy.proj": torch.tensor([-12], dtype=torch.int16)},
        "replay_ce_veto_moves_by_key": {"toy.proj": torch.tensor([-1], dtype=torch.int8)},
        "local_loss_delta_by_key": {"toy.proj": torch.tensor([-0.1], dtype=torch.float32)},
        "two_tier_carry_w6_enabled": True,
        "local_selection_ordering_mode": LOCAL_SELECTION_ORDER_TRANSIENT_LOCAL_LOSS_DELTA,
    }
    result = apply_bounded_delta_vote_step(**kwargs)
    stats = result.tensor_stats["toy.proj"]
    plan = _b6_plan_from_step_kwargs(kwargs)
    assert result.tensor_states["toy.proj"].q_levels.tolist() == [0]
    assert stats["post_veto_applied_flip_count"] == 0
    assert stats["replay_ce_veto_count"] == 1
    assert stats["vetoed_accumulator_clamp_count"] == 1
    assert stats["vetoed_accumulator_residual_policy"] == (
        "subtract_threshold_then_clamp_without_q_mutation"
    )
    assert result.tensor_states["toy.proj"].exact_accumulator_shadow.tolist() == [2]
    assert _b6_tensor_list(plan.replay_ce_veto_indices) == [0]
    assert _b6_tensor_list(plan.applied_indices) == []


def _b6_on_pc_aux_staging_case_kwargs(*, pc_aux_mode: str):
    return {
        "tensor_states": {
            "toy.proj": make_bounded_tensor_state(
                "toy.proj",
                torch.tensor([0], dtype=torch.int8),
                0.5,
                torch.zeros(1, dtype=torch.int16),
            )
        },
        "votes_by_key": {"toy.proj": torch.tensor([12], dtype=torch.int16)},
        "vote_specs_by_key": {
            "toy.proj": VoteUpdateSpec(
                threshold_abs=10,
                accumulator_clip_min=-127,
                accumulator_clip_max=127,
                max_abs_per_tensor=1,
            )
        },
        "replay_ce_veto_votes_by_key": {"toy.proj": torch.tensor([12], dtype=torch.int16)},
        "replay_ce_veto_moves_by_key": {"toy.proj": torch.tensor([1], dtype=torch.int8)},
        "pc_aux_votes_by_key": {"toy.proj": torch.tensor([-1], dtype=torch.int16)},
        "pc_aux_moves_by_key": {"toy.proj": torch.tensor([0], dtype=torch.int8)},
        "pc_aux_mode": pc_aux_mode,
        "local_loss_delta_by_key": {"toy.proj": torch.tensor([-0.1], dtype=torch.float32)},
        "two_tier_carry_w6_enabled": True,
        "local_selection_ordering_mode": LOCAL_SELECTION_ORDER_TRANSIENT_LOCAL_LOSS_DELTA,
    }


def test_b6_flag_on_pc_aux_veto_modes_match_off_staging():
    for pc_aux_mode in ("telemetry", "veto"):
        on_kwargs = _b6_on_pc_aux_staging_case_kwargs(pc_aux_mode=pc_aux_mode)
        off_kwargs = dict(on_kwargs)
        off_kwargs["two_tier_carry_w6_enabled"] = False
        off_kwargs.pop("local_loss_delta_by_key")
        off_kwargs.pop("local_selection_ordering_mode", None)
        on_result = apply_bounded_delta_vote_step(**on_kwargs)
        off_result = apply_bounded_delta_vote_step(**off_kwargs)
        on_plan = _b6_plan_from_step_kwargs(on_kwargs)
        off_plan = _b6_plan_from_step_kwargs(off_kwargs)
        assert _b6_tensor_list(on_plan.applied_indices) == _b6_tensor_list(off_plan.applied_indices), (
            f"applied_indices surface mismatch for pc_aux_mode={pc_aux_mode!r}"
        )
        assert _b6_tensor_list(on_plan.pc_aux_negative_indices) == _b6_tensor_list(
            off_plan.pc_aux_negative_indices
        ), f"pc_aux_negative_indices surface mismatch for pc_aux_mode={pc_aux_mode!r}"
        assert _b6_tensor_list(on_plan.pc_aux_veto_indices) == _b6_tensor_list(
            off_plan.pc_aux_veto_indices
        ), f"pc_aux_veto_indices surface mismatch for pc_aux_mode={pc_aux_mode!r}"
        assert on_result.tensor_states["toy.proj"].q_levels.tolist() == (
            off_result.tensor_states["toy.proj"].q_levels.tolist()
        ), f"q_levels surface mismatch for pc_aux_mode={pc_aux_mode!r}"
        assert on_result.tensor_stats["toy.proj"]["pc_aux_mode"] == pc_aux_mode
        assert on_result.tensor_stats["toy.proj"]["post_veto_applied_flip_count"] == (
            1 if pc_aux_mode == "telemetry" else 0
        )


def _b6_on_global_cap_after_veto_case_kwargs():
    return {
        "tensor_states": {
            "toy.proj": make_bounded_tensor_state(
                "toy.proj",
                torch.tensor([0, 0, 0], dtype=torch.int8),
                0.5,
                torch.zeros(3, dtype=torch.int16),
            )
        },
        "votes_by_key": {"toy.proj": torch.tensor([12, 12, 12], dtype=torch.int16)},
        "vote_specs_by_key": {
            "toy.proj": VoteUpdateSpec(
                threshold_abs=10,
                accumulator_clip_min=-127,
                accumulator_clip_max=127,
                max_abs_per_tensor=3,
            )
        },
        "replay_ce_veto_votes_by_key": {
            "toy.proj": torch.tensor([0, -12, 0], dtype=torch.int16),
        },
        "replay_ce_veto_moves_by_key": {
            "toy.proj": torch.tensor([0, -1, 0], dtype=torch.int8),
        },
        "local_loss_delta_by_key": {
            "toy.proj": torch.tensor([-0.9, -0.8, -0.7], dtype=torch.float32),
        },
        "global_cap_spec": GlobalRateCapSpec(cap=1, step=3),
        "global_cap_tie_rule_mode": EXACT_GLOBAL_CAP_TIE_RULE_MODE,
        "global_cap_contract_name": C1_BANKED_FAITHFUL_LONG_RUN_GLOBAL_CAP_CONTRACT_NAME,
        "two_tier_carry_w6_enabled": True,
        "local_selection_ordering_mode": LOCAL_SELECTION_ORDER_TRANSIENT_LOCAL_LOSS_DELTA,
    }


def test_b6_flag_on_global_cap_operates_after_veto_pre_veto_selection():
    kwargs = _b6_on_global_cap_after_veto_case_kwargs()
    on_result = apply_bounded_delta_vote_step(**kwargs)
    plan = _b6_plan_from_step_kwargs(kwargs)
    stats = on_result.tensor_stats["toy.proj"]
    assert on_result.global_summary["global_rate_cap_enabled"] is True
    assert on_result.global_summary["global_rate_cap_accepted_count"] == 1
    assert _b6_tensor_list(plan.pre_veto_selected_indices) == [0, 1, 2]
    assert _b6_tensor_list(plan.replay_ce_veto_indices) == [1]
    assert _b6_tensor_list(plan.applied_indices) == [0, 2]
    assert stats["post_veto_would_apply_pre_cap_count"] == 2
    assert stats["global_rate_cap_accepted_indices"] == [0]
    assert stats["global_rate_cap_deferred_indices"] == [2]
    assert 1 not in stats["global_rate_cap_accepted_indices"]
    assert 1 not in stats["global_rate_cap_deferred_indices"]
    assert on_result.tensor_states["toy.proj"].q_levels.tolist() == [1, 0, 0]
    assert on_result.tensor_states["toy.proj"].exact_accumulator_shadow.tolist() == [2, 2, 12]
    assert on_result.deferred_backlog["toy.proj"][2]["defer_count"] == 1
