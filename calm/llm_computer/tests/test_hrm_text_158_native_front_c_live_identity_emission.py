"""Front-C live identity emission default-off seam tests."""
from __future__ import annotations

import json

import pytest
import torch

import calm.hrm_text_158.native_full_stack.front_c_live_identity_emission as front_c_emission
from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    apply_bounded_delta_vote_step,
    make_bounded_tensor_state,
    tensor_sha256,
)
from calm.hrm_text_158.native_full_stack.front_c_identity_emitter import (
    FRONT_C_IDENTITY_EXTRACTABLE,
    FRONT_C_RUN_DERIVED_ARTIFACT,
    classify_front_c_saved_audit_root,
    front_c_report_from_identity_artifact,
    validate_front_c_identity_artifact,
)
from calm.hrm_text_158.native_full_stack.front_c_live_identity_emission import (
    FrontCLiveIdentityCollector,
)
from calm.hrm_text_158.native_full_stack.vote_update import VoteUpdateSpec


def _spec() -> VoteUpdateSpec:
    return VoteUpdateSpec(
        threshold_abs=1,
        accumulator_clip_min=-127,
        accumulator_clip_max=127,
        max_abs_per_tensor=4,
    )


def _state():
    return make_bounded_tensor_state(
        "toy.weight",
        torch.zeros(8, dtype=torch.int8),
        1.0,
        torch.zeros(8, dtype=torch.int16),
    )


def _state_with_accumulators(*values: int):
    acc = torch.tensor(list(values), dtype=torch.int16)
    return make_bounded_tensor_state(
        "toy.weight",
        torch.zeros_like(acc, dtype=torch.int8),
        1.0,
        acc,
    )


def _votes():
    return _votes_for((0, 2), (3, -2))


def _votes_for(*entries: tuple[int, int]):
    out = torch.zeros(8, dtype=torch.int16)
    for flat_index, vote in entries:
        out[int(flat_index)] = int(vote)
    return out


def test_front_c_identity_observer_is_logging_only_and_cloned(tmp_path):
    states = {"toy.weight": _state()}
    votes = {"toy.weight": _votes()}
    specs = {"toy.weight": _spec()}

    off = apply_bounded_delta_vote_step(states, votes, specs)
    collector = FrontCLiveIdentityCollector(
        artifact_path=tmp_path / "front_c_identity_artifact.json",
        emission_interval=1,
    )
    collector.record_step0(states)
    observations = []

    def observer(observation):
        observations.append(observation)
        collector.record_step_observation(step=1, observation=observation)
        return {"ignored": True}

    on = apply_bounded_delta_vote_step(
        states,
        votes,
        specs,
        front_c_identity_observer=observer,
    )

    assert off.to_compact_dict() == on.to_compact_dict()
    assert off.global_summary == on.global_summary
    assert off.deferred_backlog == on.deferred_backlog
    assert {
        key: tensor_sha256(state.q_levels)
        for key, state in off.tensor_states.items()
    } == {
        key: tensor_sha256(state.q_levels)
        for key, state in on.tensor_states.items()
    }
    assert observations[0]["live_mutation_inputs_exposed"] is False
    assert set(observations[0]["plans_by_key"]) == {"toy.weight"}
    assert set(observations[0]["q_acc_by_key"]) == {"toy.weight"}
    observations[0]["states_by_key"]["toy.weight"].q_levels[0] = -1
    observations[0]["plans_by_key"]["toy.weight"].applied_indices[0] = 7
    observations[0]["q_acc_by_key"]["toy.weight"]["q_levels"][0] = -1
    assert int(on.tensor_states["toy.weight"].q_levels.flatten()[0].item()) == 1


def test_front_c_live_identity_artifact_is_run_derived_and_extractable(tmp_path):
    states = {"toy.weight": _state()}
    votes = {"toy.weight": _votes()}
    specs = {"toy.weight": _spec()}
    collector = FrontCLiveIdentityCollector(
        artifact_path=tmp_path / "front_c_identity_artifact.json",
        emission_interval=1,
    )
    collector.record_step0(states)
    apply_bounded_delta_vote_step(
        states,
        votes,
        specs,
        front_c_identity_observer=lambda observation: collector.record_step_observation(
            step=1,
            observation=observation,
        ),
    )

    receipt = collector.finalize(
        audit_reports={"1": {"acquired": True, "strict_exact_count": 90}},
        prior_audit_start_reports={"L0b": {"strict_exact": "230/230"}},
        prior_audit_final_reports={"L0b": {"strict_exact": "230/230"}},
        steps_completed=1,
        stop_reason="unit_test_terminal",
    )
    payload = json.loads((tmp_path / "front_c_identity_artifact.json").read_text())
    validation = validate_front_c_identity_artifact(payload)
    report = front_c_report_from_identity_artifact(payload)
    inventory = classify_front_c_saved_audit_root(tmp_path / "front_c_identity_artifact.json")

    assert receipt["global_cap_used"] is False
    assert validation.status == FRONT_C_IDENTITY_EXTRACTABLE
    assert validation.synthetic_fixture is False
    assert validation.independent_sparse_derivation is True
    assert payload["decision_path_derivation"]["artifact_class"] == FRONT_C_RUN_DERIVED_ARTIFACT
    assert (
        payload["decision_path_derivation"]["sparse_active_set_source"]
        == "dense_oracle_active_ids"
    )
    assert (
        payload["diagnostics"]["step_diagnostics"]["1"]["sparse"][
            "sparse_decision_equivalence_scope"
        ]
        == "conditional_on_dense_oracle_active_set_encode_decode"
    )
    assert payload["diagnostics"]["metadata_bit_receipt"]["tensor_metadata_bits"] > 0
    assert payload["diagnostics"]["metadata_bit_receipt"]["bucket_metadata_bits"] > 0
    assert payload["diagnostics"]["metadata_bit_receipt"]["guardrail_metadata_bits"] > 0
    assert inventory.identity_extractable is True
    assert report.decision_equivalence.zero_drift is True


def test_front_c_exact_claim_path_touches_sparse_encode_decode(monkeypatch, tmp_path):
    calls = []
    original = front_c_emission.encode_sparse_active_set_accumulator

    def spy_sparse_encode_decode(state, *, hot_exact_indices):
        calls.append(tuple(int(index) for index in hot_exact_indices))
        return original(state, hot_exact_indices=hot_exact_indices)

    monkeypatch.setattr(
        front_c_emission,
        "encode_sparse_active_set_accumulator",
        spy_sparse_encode_decode,
    )
    states = {"toy.weight": _state()}
    votes = {"toy.weight": _votes()}
    specs = {"toy.weight": _spec()}
    collector = FrontCLiveIdentityCollector(
        artifact_path=tmp_path / "front_c_identity_artifact.json",
        emission_interval=1,
    )
    collector.record_step0(states)
    apply_bounded_delta_vote_step(
        states,
        votes,
        specs,
        front_c_identity_observer=lambda observation: collector.record_step_observation(
            step=1,
            observation=observation,
        ),
    )

    receipt = collector.finalize(
        audit_reports={"1": {"acquired": True, "strict_exact_count": 90}},
        prior_audit_start_reports={"L0b": {"strict_exact": "230/230"}},
        prior_audit_final_reports={"L0b": {"strict_exact": "230/230"}},
        steps_completed=1,
        stop_reason="unit_test_terminal",
    )
    payload = json.loads((tmp_path / "front_c_identity_artifact.json").read_text())

    assert calls == [(0, 3)]
    assert "front_c_report" in receipt
    assert payload["decision_path_derivation"]["full_sparse_equivalence_claimed"] is True
    assert payload["diagnostics"]["step_diagnostics"]["1"]["sparse"]["sparse_exact_oracle_ran"] is True
    assert receipt["front_c_report"]["decision_equivalence"]["zero_drift"] is True


def test_front_c_observer_reuses_passed_plan_without_dense_path_recompute(monkeypatch, tmp_path):
    original_path_from_inputs = front_c_emission._path_from_inputs

    def fail_dense_reference_recompute(*args, **kwargs):
        if kwargs.get("label") == "front_c_dense_int16_reference":
            raise AssertionError("collector must reuse observer plans/q_acc for dense path")
        return original_path_from_inputs(*args, **kwargs)

    monkeypatch.setattr(
        front_c_emission,
        "_path_from_inputs",
        fail_dense_reference_recompute,
    )
    states = {"toy.weight": _state()}
    votes = {"toy.weight": _votes()}
    specs = {"toy.weight": _spec()}
    collector = FrontCLiveIdentityCollector(
        artifact_path=tmp_path / "front_c_identity_artifact.json",
        emission_interval=1,
    )
    collector.record_step0(states)

    apply_bounded_delta_vote_step(
        states,
        votes,
        specs,
        front_c_identity_observer=lambda observation: collector.record_step_observation(
            step=1,
            observation=observation,
        ),
    )
    payload = collector.build_payload(
        audit_reports={"1": {"acquired": True, "strict_exact_count": 90}},
        prior_audit_start_reports={"L0b": {"strict_exact": "230/230"}},
        prior_audit_final_reports={"L0b": {"strict_exact": "230/230"}},
        steps_completed=1,
        stop_reason="unit_test_terminal",
    )

    step_diag = payload["diagnostics"]["step_diagnostics"]["1"]
    assert step_diag["dense"]["dense_source"] == "reused_vote_update_plan"
    assert step_diag["dense_q_flip_directions"] == [
        {"state_key": "toy.weight", "flat_index": 0, "direction": 1},
        {"state_key": "toy.weight", "flat_index": 3, "direction": -1},
    ]


def test_front_c_dense_replay_veto_direction_comes_from_reused_plan(tmp_path):
    states = {"toy.weight": _state()}
    votes = {"toy.weight": _votes_for((0, 2), (2, 2))}
    replay_votes = {"toy.weight": _votes_for((2, -1))}
    replay_moves = {"toy.weight": torch.zeros(8, dtype=torch.int8)}
    specs = {"toy.weight": _spec()}
    collector = FrontCLiveIdentityCollector(
        artifact_path=tmp_path / "front_c_identity_artifact.json",
        emission_interval=1,
    )
    collector.record_step0(states)

    result = apply_bounded_delta_vote_step(
        states,
        votes,
        specs,
        replay_ce_veto_votes_by_key=replay_votes,
        replay_ce_veto_moves_by_key=replay_moves,
        front_c_identity_observer=lambda observation: collector.record_step_observation(
            step=1,
            observation=observation,
        ),
    )
    payload = collector.build_payload(
        audit_reports={"1": {"acquired": True, "strict_exact_count": 90}},
        prior_audit_start_reports={"L0b": {"strict_exact": "230/230"}},
        prior_audit_final_reports={"L0b": {"strict_exact": "230/230"}},
        steps_completed=1,
        stop_reason="unit_test_terminal",
    )
    dense = payload["dense_decision_path"]

    assert int(result.tensor_states["toy.weight"].q_levels.flatten()[0].item()) == 1
    assert int(result.tensor_states["toy.weight"].q_levels.flatten()[2].item()) == 0
    assert dense["q_flip_directions"] == [
        {"state_key": "toy.weight", "flat_index": 0, "direction": 1},
    ]
    assert dense["replay_veto_decision_keys"] == [
        {"state_key": "toy.weight", "flat_index": 2},
    ]


def test_front_c_bounded_identity_artifact_is_structurally_nonclaimable(tmp_path):
    states = {"toy.weight": _state_with_accumulators(1, 1, 0, 0, 0, 0, 0, 0)}
    votes = {"toy.weight": torch.zeros(8, dtype=torch.int16)}
    specs = {
        "toy.weight": VoteUpdateSpec(
            threshold_abs=1,
            accumulator_clip_min=-127,
            accumulator_clip_max=127,
            max_abs_per_tensor=0,
        ),
    }
    collector = FrontCLiveIdentityCollector(
        artifact_path=tmp_path / "front_c_identity_artifact.json",
        emission_interval=1,
        max_exact_identity_keys=1,
        sparse_oracle_max_active_ids=1,
    )
    collector.record_step0(states)
    apply_bounded_delta_vote_step(
        states,
        votes,
        specs,
        front_c_identity_observer=lambda observation: collector.record_step_observation(
            step=1,
            observation=observation,
        ),
    )

    receipt = collector.finalize(
        audit_reports={"1": {"acquired": True, "strict_exact_count": 90}},
        prior_audit_start_reports={"L0b": {"strict_exact": "230/230"}},
        prior_audit_final_reports={"L0b": {"strict_exact": "230/230"}},
        steps_completed=1,
        stop_reason="unit_test_bounded_nonclaim",
    )
    payload = json.loads((tmp_path / "front_c_identity_artifact.json").read_text())
    validation = validate_front_c_identity_artifact(payload)

    assert validation.status == FRONT_C_IDENTITY_EXTRACTABLE
    assert "front_c_report" not in receipt
    assert receipt["front_c_report_skipped_bounded_nonclaim"]["identity_emission_scope"].startswith(
        "bounded_",
    )
    assert payload["decision_path_derivation"]["identity_emission_scope"].startswith("bounded_")
    assert payload["decision_path_derivation"]["full_identity_emission_claimed"] is False
    assert payload["decision_path_derivation"]["full_sparse_equivalence_claimed"] is False
    assert payload["timeline"][1]["current_magnitude_threshold_keys"] == [
        {"state_key": "toy.weight", "flat_index": 0},
    ]
    assert payload["diagnostics"]["step_diagnostics"]["1"]["surface"][
        "current_magnitude_threshold_keys"
    ]["full_identity_count"] == 2
    assert (
        payload["dense_decision_path"]["q_flip_directions"]
        == payload["sparse_decision_path"]["q_flip_directions"]
    )
    with pytest.raises(ValueError, match="bounded/non-claim"):
        front_c_report_from_identity_artifact(payload)


def test_front_c_event_delta_count_uses_selected_timeline_not_latest(tmp_path):
    states = {"toy.weight": _state()}
    specs = {"toy.weight": _spec()}
    collector = FrontCLiveIdentityCollector(
        artifact_path=tmp_path / "front_c_identity_artifact.json",
        emission_interval=1,
    )
    collector.record_step0(states)

    step1 = apply_bounded_delta_vote_step(
        states,
        {"toy.weight": _votes_for((0, 2), (3, -2))},
        specs,
        front_c_identity_observer=lambda observation: collector.record_step_observation(
            step=1,
            observation=observation,
        ),
    )
    apply_bounded_delta_vote_step(
        step1.tensor_states,
        {"toy.weight": _votes_for((4, 2))},
        specs,
        front_c_identity_observer=lambda observation: collector.record_step_observation(
            step=2,
            observation=observation,
        ),
    )

    receipt = collector.finalize(
        audit_reports={
            "1": {"acquired": False, "strict_exact_count": 2},
            "2": {"acquired": False, "strict_exact_count": 3},
        },
        prior_audit_start_reports={"L0b": {"strict_exact": "230/230"}},
        prior_audit_final_reports={"L0b": {"strict_exact": "230/230"}},
        steps_completed=2,
        stop_reason="unit_test_terminal",
    )
    payload = json.loads((tmp_path / "front_c_identity_artifact.json").read_text())
    bit_receipt = payload["diagnostics"]["metadata_bit_receipt"]
    latest_only_count = len(payload["dense_decision_path"]["q_flip_directions"])

    assert receipt["global_cap_used"] is False
    assert latest_only_count == 1
    assert bit_receipt["event_delta_count"] == 3
    assert bit_receipt["event_delta_count"] > latest_only_count
    assert bit_receipt["event_delta_unique_identity_count"] == 3
    assert bit_receipt["selected_timeline_steps"] == [0, 1, 2]
    assert bit_receipt["selected_step_q_flip_receipts"]["1"]["q_flip_count"] == 2
    assert bit_receipt["selected_step_q_flip_receipts"]["2"]["q_flip_count"] == 1


def test_front_c_selected_timeline_keeps_full_audit_cadence(tmp_path):
    states = {"toy.weight": _state()}
    specs = {"toy.weight": _spec()}
    collector = FrontCLiveIdentityCollector(
        artifact_path=tmp_path / "front_c_identity_artifact.json",
        emission_interval=20,
        audit_interval=20,
    )
    collector.record_step0(states)

    current_states = states
    step_votes = (
        (1, (0, 2)),
        (20, (1, 2)),
        (40, (2, 2)),
        (60, (3, -2)),
        (80, (4, 2)),
        (100, (5, -2)),
        (120, (6, 2)),
    )
    for step, vote_entry in step_votes:
        result = apply_bounded_delta_vote_step(
            current_states,
            {"toy.weight": _votes_for(vote_entry)},
            specs,
            front_c_identity_observer=lambda observation, step=step: (
                collector.record_step_observation(
                    step=step,
                    observation=observation,
                )
            ),
        )
        current_states = result.tensor_states

    payload = collector.build_payload(
        audit_reports={
            "20": {"acquired": False, "strict_exact_count": 20},
            "40": {"acquired": False, "strict_exact_count": 40},
            "60": {"acquired": False, "strict_exact_count": 60},
            "80": {"acquired": True, "strict_exact_count": 90},
            "100": {"acquired": False, "strict_exact_count": 88},
            "120": {"acquired": True, "strict_exact_count": 90},
        },
        prior_audit_start_reports={"L0b": {"strict_exact": "230/230"}},
        prior_audit_final_reports={"L0b": {"strict_exact": "230/230"}},
        steps_completed=120,
        stop_reason="unit_test_terminal",
    )
    bit_receipt = payload["diagnostics"]["metadata_bit_receipt"]
    expected_steps = [0, 1, 20, 40, 60, 80, 100, 120]

    assert bit_receipt["selected_timeline_steps"] == expected_steps
    assert bit_receipt["selected_step_q_flip_receipts"]["120"]["q_flip_count"] == 1
    assert bit_receipt["event_delta_count"] == len(step_votes)
    assert [row["step"] for row in payload["timeline"]] == expected_steps


def test_front_c_rejects_acquired_audit_step_not_collected(tmp_path):
    states = {"toy.weight": _state()}
    specs = {"toy.weight": _spec()}
    collector = FrontCLiveIdentityCollector(
        artifact_path=tmp_path / "front_c_identity_artifact.json",
        emission_interval=0,
    )
    collector.record_step0(states)
    apply_bounded_delta_vote_step(
        states,
        {"toy.weight": _votes()},
        specs,
        front_c_identity_observer=lambda observation: collector.record_step_observation(
            step=1,
            observation=observation,
        ),
    )

    with pytest.raises(ValueError, match="acquired audit step was not collected"):
        collector.build_payload(
            audit_reports={"2": {"acquired": True, "strict_exact_count": 90}},
            prior_audit_start_reports={"L0b": {"strict_exact": "230/230"}},
            prior_audit_final_reports={"L0b": {"strict_exact": "230/230"}},
            steps_completed=2,
            stop_reason="unit_test_missing_acquired_row",
        )
