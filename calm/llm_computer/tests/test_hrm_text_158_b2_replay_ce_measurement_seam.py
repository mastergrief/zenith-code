"""CPU tests for the B2 replay-CE measurement-only seam.

Does not edit five of the six VETO compatibility files. The sixth,
test_hrm_text_158_native_bounded_delta_learner.py, was edited only for
the advisor-authorized B6 golden refresh (route 1786740132444-a7586361).
"""
from __future__ import annotations

import torch

from calm.hrm_text_158.native_full_stack.vote_update import (
    ReplayCeMode,
    VoteUpdateInputs,
    VoteUpdateSpec,
    VoteUpdateState,
    apply_integer_vote_update_reference,
    apply_two_tier_vote_update_reference,
    plan_integer_vote_update_reference,
    plan_two_tier_vote_update_reference,
)
from scripts.hrm_text_158_bounded_delta_acquisition_probe import (
    B2_FULL_VERDICT_SCHEMA_VERSION,
    build_b2_full_prior_snapshot,
    finalize_b2_full_verdict_state,
    new_b2_full_verdict_state,
    summarize_b2_full_prior_snapshot,
)


def _spec() -> VoteUpdateSpec:
    return VoteUpdateSpec(
        threshold_abs=10,
        accumulator_clip_min=-127,
        accumulator_clip_max=127,
        decay_numerator=1,
        decay_denominator=1,
        max_abs_per_tensor=64,
        fraction_per_tensor=1.0,
    )


def _state() -> VoteUpdateState:
    return VoteUpdateState(
        q_levels=torch.tensor([0], dtype=torch.int8),
        accumulators=torch.tensor([0], dtype=torch.int16),
    )


def _veto_inputs(*, mode: str) -> VoteUpdateInputs:
    return VoteUpdateInputs(
        votes=torch.tensor([12], dtype=torch.int16),
        replay_ce_veto_votes=torch.tensor([-1], dtype=torch.int16),
        replay_ce_veto_moves=torch.tensor([0], dtype=torch.int8),
        replay_ce_mode=mode,
        local_loss_delta=torch.tensor([0.0], dtype=torch.float32),
    )


def _unarmed_inputs() -> VoteUpdateInputs:
    return VoteUpdateInputs(
        votes=torch.tensor([12], dtype=torch.int16),
        local_loss_delta=torch.tensor([0.0], dtype=torch.float32),
    )


def _prior_report(step: int, *, failures: tuple[str, ...] = ()) -> dict:
    n_fail = len(failures)
    return {
        "step": int(step),
        "strict_exact": f"{10 - n_fail}/10",
        "strict_exact_count": 10 - n_fail,
        "strict_exact_total": 10,
        "parsed_exact": f"{10 - n_fail}/10",
        "parsed_exact_count": 10 - n_fail,
        "parsed_exact_total": 10,
        "duration_seconds": 0.0,
        "strict_failure_row_ids": list(failures),
        "parsed_failure_row_ids": list(failures),
        "strict_failure_sources_by_row_id": {row: "fixture" for row in failures},
        "parsed_failure_sources_by_row_id": {row: "fixture" for row in failures},
    }


def _snapshot(**report_overrides):
    start = {
        "L0b": _prior_report(0),
        "math_a0": _prior_report(0),
        "L0c1": _prior_report(0),
    }
    current = {
        "L0b": _prior_report(40),
        "math_a0": _prior_report(40),
        "L0c1": _prior_report(40),
    }
    current.update(report_overrides)
    return build_b2_full_prior_snapshot(
        snapshot_name="fixture",
        step=40,
        target_audit={
            "step": 40,
            "strict_exact": "90/90",
            "strict_exact_count": 90,
            "parsed_exact": "90/90",
            "parsed_exact_count": 90,
            "acquired": True,
        },
        coverage_by_support={
            "L0b": {"coverage_cycles": 0, "rows_total": 4},
            "math_a0": {"coverage_cycles": 0, "rows_total": 4},
        },
        start_reports=start,
        current_reports=current,
    )


def test_it1_present_hold_emits_true():
    snap = _snapshot()
    assert snap["schema"] == B2_FULL_VERDICT_SCHEMA_VERSION
    assert snap["retained_true_priors_no_new_broad_cluster"] is True
    assert snap["stop_supports_present"] is True


def test_it1_present_regression_emits_false():
    snap = _snapshot(
        L0b=_prior_report(40, failures=("a", "b", "c")),
    )
    assert snap["retained_true_priors_no_new_broad_cluster"] is False
    assert snap["stop_supports_present"] is True


def test_it1_missing_support_emits_none_not_false():
    snap = _snapshot()
    # drop L0b from current by rebuilding without it
    start = {
        "math_a0": _prior_report(0),
        "L0c1": _prior_report(0),
    }
    current = {
        "math_a0": _prior_report(40),
        "L0c1": _prior_report(40),
    }
    snap = build_b2_full_prior_snapshot(
        snapshot_name="gap",
        step=40,
        target_audit={
            "step": 40,
            "strict_exact": "90/90",
            "strict_exact_count": 90,
            "parsed_exact": "90/90",
            "parsed_exact_count": 90,
            "acquired": True,
        },
        coverage_by_support={
            "L0b": {"coverage_cycles": 0, "rows_total": 4},
            "math_a0": {"coverage_cycles": 0, "rows_total": 4},
        },
        start_reports=start,
        current_reports=current,
    )
    assert snap["stop_supports_present"] is False
    assert snap["retained_true_priors_no_new_broad_cluster"] is None
    assert snap["retained_true_priors_no_new_broad_cluster"] is not False
    summarized = summarize_b2_full_prior_snapshot(snap)
    assert summarized["retained_true_priors_no_new_broad_cluster"] is None


def test_finalize_null_floor_is_artifact_insufficient_never_no_retain():
    snap = build_b2_full_prior_snapshot(
        snapshot_name="terminal",
        step=40,
        target_audit={
            "step": 40,
            "strict_exact": "90/90",
            "strict_exact_count": 90,
            "parsed_exact": "90/90",
            "parsed_exact_count": 90,
            "acquired": True,
        },
        coverage_by_support={
            "L0b": {"coverage_cycles": 1, "rows_total": 4},
            "math_a0": {"coverage_cycles": 1, "rows_total": 4},
        },
        start_reports={"math_a0": _prior_report(0)},
        current_reports={"math_a0": _prior_report(40)},
    )
    state = new_b2_full_verdict_state()
    state["first_audited_target_ge_90"] = {"acquired": True}
    out = finalize_b2_full_verdict_state(state, terminal_snapshot=snap)
    assert out["verdict"] == "artifact_insufficient"
    assert out["verdict"] != "no-retain"
    assert out["terminal"]["retained_true_priors_no_new_broad_cluster"] is None


def _digests(mode: str | None):
    state = _state()
    inputs = _unarmed_inputs() if mode is None else _veto_inputs(mode=mode)
    plan = plan_integer_vote_update_reference(state, inputs, _spec())
    result = apply_integer_vote_update_reference(state, inputs, _spec())
    return (
        plan.applied_indices.tolist(),
        result.q_levels.tolist(),
        result.accumulators.tolist(),
        plan.replay_ce_veto_indices.tolist(),
    )


def test_it4_off_and_telemetry_match_unarmed_integer_path():
    unarmed = _digests(None)
    off = _digests("off")
    telemetry = _digests("telemetry")
    veto = _digests("veto")
    assert off[:3] == unarmed[:3]
    assert telemetry[:3] == unarmed[:3]
    assert off[3] == []
    assert telemetry[3] == []
    assert veto[0] == []
    assert veto[3] == [0]
    assert veto[1] == [0]


def test_it4_two_tier_apply_clamp_site_gated():
    state = _state()
    unarmed = apply_two_tier_vote_update_reference(state, _unarmed_inputs(), _spec())
    telemetry = apply_two_tier_vote_update_reference(
        state, _veto_inputs(mode="telemetry"), _spec()
    )
    off = apply_two_tier_vote_update_reference(state, _veto_inputs(mode="off"), _spec())
    assert telemetry.q_levels.tolist() == unarmed.q_levels.tolist()
    assert telemetry.accumulators.tolist() == unarmed.accumulators.tolist()
    assert off.q_levels.tolist() == unarmed.q_levels.tolist()
    assert off.accumulators.tolist() == unarmed.accumulators.tolist()


def test_clamp_and_apply_mask_sites_source_proven_empty_under_non_veto():
    """Every enumerated residual-clamp / apply_mask site is inert under off and telemetry.

    Sites: partition apply_mask; plan assignments that feed clamp (integer plan,
    two-tier plan); apply_two_tier clamp; apply_integer frozen-plan clamp.
    Event-coded plan assignment uses the same _partition helper.
    """
    for mode in ("off", "telemetry"):
        inputs = _veto_inputs(mode=mode)
        state = _state()
        int_plan = plan_integer_vote_update_reference(state, inputs, _spec())
        two_plan = plan_two_tier_vote_update_reference(state, inputs, _spec())
        assert int_plan.applied_indices.tolist() == _digests(None)[0]
        assert int_plan.replay_ce_veto_indices.tolist() == []
        assert two_plan.replay_ce_veto_indices.tolist() == []
        assert ReplayCeMode(mode) is not ReplayCeMode.VETO


def test_default_mode_is_legacy_veto():
    inputs = VoteUpdateInputs(votes=torch.tensor([12], dtype=torch.int16))
    assert inputs.normalized_replay_ce_mode is ReplayCeMode.VETO


def _plan_stats(mode: str, *, negative: bool):
    votes = torch.tensor([-1 if negative else 1], dtype=torch.int16)
    inputs = VoteUpdateInputs(
        votes=torch.tensor([12], dtype=torch.int16),
        replay_ce_veto_votes=votes,
        replay_ce_veto_moves=torch.tensor([0], dtype=torch.int8),
        replay_ce_mode=mode,
        local_loss_delta=torch.tensor([0.0], dtype=torch.float32),
    )
    plan = plan_integer_vote_update_reference(_state(), inputs, _spec())
    result = apply_integer_vote_update_reference(_state(), inputs, _spec())
    return plan, result


def test_telemetry_records_replay_negative_and_stays_non_acting():
    off_plan, off_result = _plan_stats("off", negative=True)
    tel_plan, tel_result = _plan_stats("telemetry", negative=True)
    veto_plan, veto_result = _plan_stats("veto", negative=True)

    assert off_plan.applied_indices.tolist() == [0]
    assert off_plan.replay_ce_veto_indices.tolist() == []
    assert off_plan.stats["replay_ce_veto_count"] == 0
    assert "replay_ce_mode" not in off_plan.stats
    assert "replay_ce_negative_count" not in off_plan.stats

    assert tel_plan.applied_indices.tolist() == [0]
    assert tel_plan.replay_ce_veto_indices.tolist() == []
    assert tel_plan.stats["replay_ce_veto_count"] == 0
    assert tel_plan.stats["vetoed_accumulator_clamp_count"] == 0
    assert tel_plan.stats["replay_ce_mode"] == "telemetry"
    assert tel_plan.stats["replay_ce_negative_count"] == 1
    assert tel_result.q_levels.tolist() == off_result.q_levels.tolist()
    assert tel_result.accumulators.tolist() == off_result.accumulators.tolist()

    assert veto_plan.applied_indices.tolist() == []
    assert veto_plan.replay_ce_veto_indices.tolist() == [0]
    assert veto_plan.stats["replay_ce_veto_count"] == 1
    assert "replay_ce_mode" not in veto_plan.stats
    assert "replay_ce_negative_count" not in veto_plan.stats
    assert veto_result.q_levels.tolist() == [0]


def test_telemetry_known_good_row_is_silent():
    tel_plan, _ = _plan_stats("telemetry", negative=False)
    assert tel_plan.stats["replay_ce_mode"] == "telemetry"
    assert tel_plan.stats["replay_ce_negative_count"] == 0
    assert tel_plan.replay_ce_veto_indices.tolist() == []
    assert tel_plan.applied_indices.tolist() == [0]


def _finalize_with(snap_updates=None, delete_keys=()):
    snap = _snapshot()
    snap["target_gate_met"] = True
    snap["coverage_gate_met"] = True
    if snap_updates:
        snap.update(snap_updates)
    for key in delete_keys:
        snap.pop(key, None)
    state = new_b2_full_verdict_state()
    state["first_audited_target_ge_90"] = {"acquired": True}
    return finalize_b2_full_verdict_state(state, terminal_snapshot=snap)["verdict"]


def test_item4_target_gate_missing_false_true():
    missing = _finalize_with(delete_keys=("target_gate_met",))
    explicit_false = _finalize_with({"target_gate_met": False})
    explicit_true = _finalize_with({"target_gate_met": True})
    assert missing == "artifact_insufficient"
    assert explicit_false == "acquire-then-forget"
    assert explicit_true == "RETAINS"


def test_item4_coverage_gate_missing_false_true():
    missing = _finalize_with(delete_keys=("coverage_gate_met",))
    explicit_false = _finalize_with({"coverage_gate_met": False})
    explicit_true = _finalize_with({"coverage_gate_met": True})
    assert missing == "artifact_insufficient"
    assert explicit_false == "no-retain"
    assert explicit_true == "RETAINS"
