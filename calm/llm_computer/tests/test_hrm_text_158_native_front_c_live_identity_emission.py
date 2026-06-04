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
from calm.hrm_text_158.native_full_stack.vote_update import (
    VoteUpdatePlan,
    VoteUpdateSpec,
    VoteUpdateState,
)
from scripts.hrm_text_158_bounded_delta_acquisition_probe import (
    build_arg_parser,
    make_front_c_identity_observer_for_step,
)


STEP_TIMING_KEYS = {
    "surface_from_reused_plans",
    "dense_from_reused_plans",
    "sparse_active_set_select_or_bound",
    "sparse_encode_decode_or_bounded_filter",
    "sparse_path_materialize",
    "record_step_observation_total",
}
FINALIZE_TIMING_KEYS = {
    "build_payload",
    "artifact_write",
    "identity_validate",
    "front_c_report_or_skip",
    "finalize_total",
}
ARTIFACT_FINALIZE_TIMING_KEYS = {
    "build_payload",
    "identity_validate",
    "front_c_report_or_skip",
}


def _assert_nonnegative_timing(timing: dict, required_keys: set[str]) -> None:
    assert timing["schema"] == front_c_emission.FRONT_C_LIVE_TIMING_SCHEMA_VERSION
    durations = timing["durations_seconds"]
    assert required_keys <= set(durations)
    for key in required_keys:
        assert durations[key] >= 0.0


def _assert_authoritative_finalize_timing(timing: dict) -> None:
    _assert_nonnegative_timing(timing, FINALIZE_TIMING_KEYS)
    assert timing["authoritative"] is True
    assert (
        timing["authoritative_timing_location"]
        == "front_c_finalize_receipt.front_c_finalize_timing"
    )
    assert (
        timing["artifact_write_position"]
        == "post_identity_validate_and_front_c_report_or_skip"
    )
    assert timing["durations_seconds"]["finalize_total"] >= (
        timing["durations_seconds"]["artifact_write"]
    )


def _assert_artifact_finalize_timing_is_caveated(timing: dict) -> None:
    _assert_nonnegative_timing(timing, ARTIFACT_FINALIZE_TIMING_KEYS)
    assert timing["authoritative"] is False
    assert (
        timing["authoritative_timing_location"]
        == "front_c_finalize_receipt.front_c_finalize_timing"
    )
    assert set(timing["excluded_duration_keys_due_to_self_reference"]) == {
        "artifact_write",
        "finalize_total",
    }
    assert "artifact_write" not in timing["durations_seconds"]
    assert "finalize_total" not in timing["durations_seconds"]
    assert "cannot embed the cost of the write" in timing["artifact_embedded_timing_caveat"]


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


def _spec_with_max_flips(max_abs_per_tensor: int) -> VoteUpdateSpec:
    return VoteUpdateSpec(
        threshold_abs=1,
        accumulator_clip_min=-127,
        accumulator_clip_max=127,
        max_abs_per_tensor=int(max_abs_per_tensor),
    )


def _fake_vote_update_plan(
    numel: int,
    *,
    candidate_indices: torch.Tensor,
) -> VoteUpdatePlan:
    empty_i64 = torch.empty(0, dtype=torch.int64)
    empty_i16 = torch.empty(0, dtype=torch.int16)
    return VoteUpdatePlan(
        q_i16=torch.zeros(int(numel), dtype=torch.int16),
        new_acc_i32=torch.zeros(int(numel), dtype=torch.int32),
        candidate_indices=candidate_indices.to(torch.int64).contiguous(),
        pre_veto_selected_indices=empty_i64,
        applied_indices=empty_i64,
        applied_directions=empty_i16,
        applied_thresholds=empty_i16,
        replay_ce_veto_indices=empty_i64,
        replay_veto_directions=empty_i16,
        replay_veto_thresholds=empty_i16,
        pc_aux_negative_indices=empty_i64,
        pc_aux_veto_indices=empty_i64,
        stats={},
    )


def _legacy_bounded_identity_selection(
    identities: set[tuple[str, int]],
    *,
    max_keys: int,
    priority_ids: set[tuple[str, int]] | None = None,
) -> list[dict[str, int | str]]:
    full = set(identities)
    limit = max(0, int(max_keys))
    emitted = full
    if len(full) > limit:
        selected: list[tuple[str, int]] = []
        seen: set[tuple[str, int]] = set()
        for identity in sorted(set(priority_ids or set()) & full):
            if len(selected) >= limit:
                break
            selected.append(identity)
            seen.add(identity)
        for identity in sorted(full):
            if len(selected) >= limit:
                break
            if identity in seen:
                continue
            selected.append(identity)
            seen.add(identity)
        emitted = set(selected)
    return front_c_emission._identity_dicts(emitted)


@pytest.mark.parametrize(
    ("identities", "max_keys", "priority_ids"),
    (
        ({("only", 3), ("only", 1), ("only", 2)}, 2, set()),
        ({("b", 0), ("a", 5), ("a", 1), ("b", 1)}, 3, set()),
        ({("a", 0), ("a", 1), ("a", 9), ("a", 10)}, 2, {("a", 10)}),
        ({("a", 0), ("a", 1)}, 8, {("a", 1)}),
        (set(), 1, {("missing", 0)}),
    ),
)
def test_front_c_bounded_identity_set_preserves_legacy_contract(
    identities,
    max_keys,
    priority_ids,
):
    emitted, diagnostics = front_c_emission._bounded_identity_set(
        "unit_surface",
        identities,
        max_keys=max_keys,
        priority_ids=priority_ids,
    )

    assert front_c_emission._identity_dicts(emitted) == _legacy_bounded_identity_selection(
        identities,
        max_keys=max_keys,
        priority_ids=priority_ids,
    )
    assert diagnostics["full_identity_count"] == len(identities)
    assert diagnostics["emitted_identity_count"] == len(emitted)
    assert diagnostics["bounded"] is (len(identities) > max(0, int(max_keys)))


def test_front_c_source_bounded_selection_matches_tuple_legacy_ordering():
    universe = front_c_emission._identity_universe_from_sources(
        {
            "b": torch.tensor([2, 0, 2, 1], dtype=torch.int64),
            "a": torch.tensor([5, 1], dtype=torch.int64),
        },
        extra_identities={("a", 0), ("b", 9), ("c", 3), ("a", 5)},
    )
    full_reference = {
        ("a", 0),
        ("a", 1),
        ("a", 5),
        ("b", 0),
        ("b", 1),
        ("b", 2),
        ("b", 9),
        ("c", 3),
    }

    selection = front_c_emission._select_bounded_identity_universe(
        "unit_surface",
        universe,
        max_keys=4,
        priority_ids={("b", 9), ("c", 3), ("missing", 0)},
    )

    assert front_c_emission._identity_dicts(selection.identities) == (
        _legacy_bounded_identity_selection(
            full_reference,
            max_keys=4,
            priority_ids={("b", 9), ("c", 3), ("missing", 0)},
        )
    )
    assert selection.diagnostics["full_identity_count"] == len(full_reference)
    assert selection.diagnostics["emitted_identity_count"] == 4
    assert selection.diagnostics["bounded"] is True


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
    step_timing = payload["diagnostics"]["step_diagnostics"]["1"]["timing"]
    _assert_nonnegative_timing(step_timing, STEP_TIMING_KEYS)
    assert step_timing["path_source"] == "reused_observer_plan"
    assert step_timing["sparse_path_mode"] == "exact_sparse_encode_decode"
    _assert_artifact_finalize_timing_is_caveated(
        payload["diagnostics"]["finalize_timing"],
    )
    _assert_authoritative_finalize_timing(receipt["front_c_finalize_timing"])
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
    step_timing = payload["diagnostics"]["step_diagnostics"]["1"]["timing"]
    _assert_nonnegative_timing(step_timing, STEP_TIMING_KEYS)
    assert step_timing["path_source"] == "reused_observer_plan"
    assert step_timing["sparse_path_mode"] == "bounded_reused_plan_filter"
    _assert_artifact_finalize_timing_is_caveated(
        payload["diagnostics"]["finalize_timing"],
    )
    _assert_authoritative_finalize_timing(receipt["front_c_finalize_timing"])
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


def test_front_c_collection_cadence_rebuild_skips_uncollected_observe_tax(tmp_path):
    states = {"toy.weight": _state()}
    specs = {"toy.weight": _spec_with_max_flips(1)}
    collector = FrontCLiveIdentityCollector(
        artifact_path=tmp_path / "front_c_identity_artifact.json",
        emission_interval=0,
        independent_oracle_compare=True,
    )
    collector.record_step0(states)

    current_states = states
    observer_calls = []
    for step, votes, collect in (
        (1, _votes_for((0, 2), (1, 2), (2, 2)), True),
        (2, _votes_for(), False),
        (3, _votes_for(), True),
    ):
        observer = make_front_c_identity_observer_for_step(
            collector,
            step=step,
            total_steps=3,
        )
        assert (observer is not None) is collect

        def wrapped_observer(observation, *, observer=observer, step=step):
            observer_calls.append(step)
            assert observer is not None
            observer(observation)

        result = apply_bounded_delta_vote_step(
            current_states,
            {"toy.weight": votes},
            specs,
            front_c_identity_observer=wrapped_observer if observer is not None else None,
        )
        current_states = result.tensor_states

    payload = collector.build_payload(
        audit_reports={
            "1": {"acquired": False, "strict_exact_count": 1},
            "3": {"acquired": False, "strict_exact_count": 3},
        },
        prior_audit_start_reports={"L0b": {"strict_exact": "230/230"}},
        prior_audit_final_reports={"L0b": {"strict_exact": "230/230"}},
        steps_completed=3,
        stop_reason="unit_test_terminal",
    )
    diagnostics = payload["diagnostics"]
    step3_row = next(row for row in payload["timeline"] if row["step"] == 3)

    assert observer_calls == [1, 3]
    assert sorted(diagnostics["step_diagnostics"]) == ["1", "3"]
    assert "2" not in diagnostics["step_diagnostics"]
    assert diagnostics["independent_oracle_compare_enabled"] is True
    assert diagnostics["full_active_hash_oracle_enabled"] is False
    assert diagnostics["observe_only_diagnostics"] == {}
    assert diagnostics["touched_count_by_step"] == {}
    assert diagnostics["carried_threshold_count_by_step"] == {}
    assert diagnostics["touch_ratio_alarm_by_step"] == {}
    assert sorted(diagnostics["collection_current_threshold_rebuild_by_step"]) == ["1", "3"]
    assert step3_row["current_magnitude_threshold_keys"] == [
        {"state_key": "toy.weight", "flat_index": 2},
    ]
    assert (
        diagnostics["step_diagnostics"]["3"]["surface"]["current_magnitude_threshold_keys"][
            "full_identity_count"
        ]
        == 1
    )
    assert diagnostics["step_diagnostics"]["3"]["legacy_oracle"]["enabled"] is True
    assert (
        diagnostics["step_diagnostics"]["3"]["legacy_oracle"][
            "full_active_hash_computed"
        ]
        is False
    )
    assert (
        diagnostics["step_diagnostics"]["3"]["legacy_oracle"]["oracle_source"]
        == "exact_reference_recompute_no_reused_plan"
    )
    assert (
        diagnostics["step_diagnostics"]["3"]["legacy_oracle"]["oracle_path_source"]
        == "reference_recompute_compatibility_fallback"
    )
    assert all(diagnostics["step_diagnostics"]["3"]["legacy_oracle"]["checks"].values())
    assert diagnostics["step_diagnostics"]["3"]["legacy_oracle"]["checks"][
        "sparse_active_set_full_hash_not_computed"
    ] is True
    assert diagnostics["step_diagnostics"]["3"]["legacy_oracle"]["checks"][
        "q_flip_receipt_parity"
    ] is True
    assert diagnostics["step_diagnostics"]["3"]["sparse"][
        "sparse_active_set_full_hash_computed"
    ] is False
    assert (
        diagnostics["step_diagnostics"]["3"]["collection_current_threshold_rebuild"][
            "source"
        ]
        == "pre_step_q_acc_scan"
    )
    assert (
        diagnostics["step_diagnostics"]["3"]["carried_index_update"]["enabled"]
        is False
    )


def test_front_c_collection_only_on_off_parity(tmp_path):
    specs = {"toy.weight": _spec_with_max_flips(1)}
    step_votes = (
        _votes_for((0, 2), (1, 2), (2, 2)),
        _votes_for(),
        _votes_for(),
    )

    off_states = {"toy.weight": _state()}
    off_results = []
    for votes in step_votes:
        result = apply_bounded_delta_vote_step(
            off_states,
            {"toy.weight": votes},
            specs,
        )
        off_results.append(result)
        off_states = result.tensor_states

    on_states = {"toy.weight": _state()}
    on_results = []
    collector = FrontCLiveIdentityCollector(
        artifact_path=tmp_path / "front_c_identity_artifact.json",
        emission_interval=0,
        legacy_oracle_compare=True,
    )
    collector.record_step0(on_states)
    for step, votes in enumerate(step_votes, start=1):
        observer = make_front_c_identity_observer_for_step(
            collector,
            step=step,
            total_steps=len(step_votes),
        )
        result = apply_bounded_delta_vote_step(
            on_states,
            {"toy.weight": votes},
            specs,
            front_c_identity_observer=observer,
        )
        on_results.append(result)
        on_states = result.tensor_states

    assert [result.to_compact_dict() for result in off_results] == [
        result.to_compact_dict() for result in on_results
    ]
    assert [result.global_summary for result in off_results] == [
        result.global_summary for result in on_results
    ]
    assert [result.deferred_backlog for result in off_results] == [
        result.deferred_backlog for result in on_results
    ]
    assert {
        key: tensor_sha256(state.q_levels)
        for key, state in off_states.items()
    } == {
        key: tensor_sha256(state.q_levels)
        for key, state in on_states.items()
    }
    payload = collector.build_payload(
        audit_reports={"1": {"acquired": False}, "3": {"acquired": False}},
        prior_audit_start_reports={"L0b": {"strict_exact": "230/230"}},
        prior_audit_final_reports={"L0b": {"strict_exact": "230/230"}},
        steps_completed=3,
        stop_reason="unit_test_terminal",
    )
    assert payload["diagnostics"]["observe_only_diagnostics"] == {}
    assert sorted(payload["diagnostics"]["step_diagnostics"]) == ["1", "3"]
    assert sorted(payload["diagnostics"]["collection_current_threshold_rebuild_by_step"]) == [
        "1",
        "3",
    ]


def test_front_c_collected_rebuild_does_not_call_carried_index(monkeypatch, tmp_path):
    def fail_carried_index(*args, **kwargs):
        raise AssertionError("collection-cadence rebuild must not use carried index")

    monkeypatch.setattr(
        FrontCLiveIdentityCollector,
        "_ensure_current_threshold_index",
        fail_carried_index,
    )
    monkeypatch.setattr(
        FrontCLiveIdentityCollector,
        "_materialize_current_threshold_indices_for_collect",
        fail_carried_index,
    )
    monkeypatch.setattr(
        FrontCLiveIdentityCollector,
        "_update_current_threshold_index",
        fail_carried_index,
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

    assert step_diag["collection_current_threshold_rebuild"]["source"] == (
        "pre_step_q_acc_scan"
    )
    assert (
        step_diag["timing"]["durations_seconds"]["current_threshold_scan"]
        >= 0.0
    )
    assert step_diag["carried_index_update"]["enabled"] is False
    assert step_diag["carried_index_materialize_for_collect"]["materialized_for_collect"] is False
    assert payload["diagnostics"]["observe_only_diagnostics"] == {}
    assert payload["diagnostics"]["touched_count_by_step"] == {}


def test_front_c_default_collection_skips_full_active_universe_hash(monkeypatch, tmp_path):
    def fail_full_universe_hash(*args, **kwargs):
        raise AssertionError("full active universe hash must be oracle-only")

    monkeypatch.setattr(
        front_c_emission,
        "_identity_universe_sha256",
        fail_full_universe_hash,
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
    sparse_diag = payload["diagnostics"]["step_diagnostics"]["1"]["sparse"]
    step_diag = payload["diagnostics"]["step_diagnostics"]["1"]

    assert payload["diagnostics"]["independent_oracle_compare_enabled"] is False
    assert payload["diagnostics"]["full_active_hash_oracle_enabled"] is False
    assert step_diag["legacy_oracle"]["enabled"] is False
    assert sparse_diag["sparse_active_set_full_hash_computed"] is False
    assert sparse_diag["sparse_active_set_full_sha256"] == ""


def test_front_c_independent_oracle_does_not_compute_full_active_hash(
    monkeypatch,
    tmp_path,
):
    def fail_full_universe_hash(*args, **kwargs):
        raise AssertionError("independent oracle receipts must not compute full hash")

    monkeypatch.setattr(
        front_c_emission,
        "_identity_universe_sha256",
        fail_full_universe_hash,
    )
    states = {"toy.weight": _state()}
    votes = {"toy.weight": _votes()}
    specs = {"toy.weight": _spec()}
    collector = FrontCLiveIdentityCollector(
        artifact_path=tmp_path / "front_c_identity_artifact.json",
        emission_interval=1,
        independent_oracle_compare=True,
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
    oracle = step_diag["legacy_oracle"]

    assert payload["diagnostics"]["independent_oracle_compare_enabled"] is True
    assert payload["diagnostics"]["full_active_hash_oracle_enabled"] is False
    assert oracle["enabled"] is True
    assert oracle["oracle_source"] == "exact_reference_recompute_no_reused_plan"
    assert oracle["oracle_path_source"] == "reference_recompute_compatibility_fallback"
    assert oracle["live_path_source"] == "reused_observer_plan"
    assert oracle["full_active_hash_computed"] is False
    assert oracle["full_active_count"] == oracle["oracle_full_active_count"]
    assert all(oracle["checks"].values())
    assert step_diag["sparse"]["sparse_active_set_full_hash_computed"] is False
    assert "independent_oracle_reference_recompute" in step_diag["timing"][
        "durations_seconds"
    ]
    assert "oracle_full_rebuild" not in step_diag["timing"]["durations_seconds"]


@pytest.mark.parametrize(
    ("pc_aux_mode", "expected_q_flip_indices"),
    (
        ("telemetry", (0, 2, 3)),
        ("veto", (0,)),
    ),
)
def test_front_c_exact_reference_oracle_covers_replay_and_pc_branches(
    tmp_path,
    pc_aux_mode,
    expected_q_flip_indices,
):
    states = {"toy.weight": _state()}
    specs = {"toy.weight": _spec()}
    replay_moves = {"toy.weight": torch.zeros(8, dtype=torch.int8)}
    pc_moves = {"toy.weight": torch.zeros(8, dtype=torch.int8)}
    collector = FrontCLiveIdentityCollector(
        artifact_path=tmp_path / f"front_c_identity_artifact.{pc_aux_mode}.json",
        emission_interval=1,
        independent_oracle_compare=True,
    )
    collector.record_step0(states)

    apply_bounded_delta_vote_step(
        states,
        {"toy.weight": _votes_for((0, 2), (1, 2), (2, 2), (3, 2))},
        specs,
        replay_ce_veto_votes_by_key={"toy.weight": _votes_for((1, -1))},
        replay_ce_veto_moves_by_key=replay_moves,
        pc_aux_votes_by_key={"toy.weight": _votes_for((2, -1), (3, -1))},
        pc_aux_moves_by_key=pc_moves,
        pc_aux_mode=pc_aux_mode,
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
    q_flip_indices = tuple(
        int(row["flat_index"])
        for row in step_diag["dense_q_flip_directions"]
    )

    assert step_diag["legacy_oracle"]["enabled"] is True
    assert step_diag["legacy_oracle"]["full_active_hash_computed"] is False
    assert (
        step_diag["legacy_oracle"]["oracle_source"]
        == "exact_reference_recompute_no_reused_plan"
    )
    assert (
        step_diag["legacy_oracle"]["oracle_path_source"]
        == "reference_recompute_compatibility_fallback"
    )
    assert (
        step_diag["legacy_oracle"]["live_path_source"]
        == "reused_observer_plan"
    )
    assert all(step_diag["legacy_oracle"]["checks"].values())
    assert step_diag["legacy_oracle"]["checks"]["q_flip_receipt_hash_parity"] is True
    assert step_diag["legacy_oracle"]["checks"][
        "sparse_active_set_full_hash_not_computed"
    ] is True
    assert q_flip_indices == expected_q_flip_indices
    assert step_diag["carried_index_update"]["enabled"] is False
    assert payload["diagnostics"]["carried_threshold_count_by_step"] == {}
    assert payload["diagnostics"]["collection_current_threshold_rebuild_by_step"]["1"][
        "source"
    ] == "pre_step_q_acc_scan"


def test_front_c_independent_oracle_cli_flag_parses():
    default_args = build_arg_parser().parse_args([])
    enabled_args = build_arg_parser().parse_args(["--front-c-independent-oracle"])

    assert default_args.front_c_independent_oracle is False
    assert enabled_args.front_c_independent_oracle is True
