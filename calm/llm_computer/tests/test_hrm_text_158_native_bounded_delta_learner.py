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
