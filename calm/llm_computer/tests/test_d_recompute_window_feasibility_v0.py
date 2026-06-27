from __future__ import annotations

import json
from pathlib import Path

import pytest

from calm.hrm_text_158.native_full_stack.d_recompute_window_emit import (
    BOOTSTRAP_KNOWN_SATURATED_POSITIVE,
    D_RECOMPUTE_WINDOW_LOG_FILENAME,
    ReplayConstants,
    _derive_lane_flip_residual,
    append_recompute_window_log_chunk,
    build_step_log_entry,
    default_production_replay_constants,
    initialize_recompute_window_log_for_probe_session,
    validate_bootstrap_record,
)
from calm.hrm_text_158.native_full_stack.d_recompute_window_feasibility_analyzer import (
    ACC_BUDGET_BPW_UNDER_BASE3_Q,
    CLASSIFIER_D_NEEDS_UPDATE_LAW_REDESIGN,
    CLASSIFIER_D_RECOMPUTE_UNBOUNDED_OR_UNOBSERVABLE,
    CLASSIFIER_D_RECOMPUTE_WINDOW_LEAD,
    CLASSIFIER_MISSING_OBSERVABLES_OR_INVALID_WINDOW,
    LANE_CLASS_SATURATED_AT_CLAMP,
    analyze_recompute_window_log,
    analyze_synthetic_lane_trajectory,
    apply_flip_residual_scalar,
    carry_after_scalar,
    inclusive_stream_bpw,
    measure_lane_k_star,
    reconstruct_lane_from_bootstrap,
)


def _replay() -> ReplayConstants:
    return default_production_replay_constants()


def _full_step_entry(
    *,
    step: int,
    state_key: str,
    replay: ReplayConstants,
    acc_before: list[int],
    acc_after: list[int],
    vote_lanes: list[int],
    lane_indices: list[int] | None = None,
    q_before: list[int] | None = None,
    q_after: list[int] | None = None,
    flip_residual_applied_lanes: list[bool] | None = None,
    flip_direction_lanes: list[int | None] | None = None,
    flip_threshold_lanes: list[int | None] | None = None,
    residual_authority_lanes: list[str] | None = None,
    resume_generation: int = 0,
) -> dict:
    lane_indices = lane_indices or list(range(len(acc_before)))
    q_before = q_before if q_before is not None else [0] * len(acc_before)
    q_after = q_after if q_after is not None else [0] * len(acc_before)
    return build_step_log_entry(
        step=step,
        state_key=state_key,
        replay_constants=replay,
        acc_before=acc_before,
        acc_after=acc_after,
        q_before=q_before,
        q_after=q_after,
        vote_lanes=vote_lanes,
        lane_indices=lane_indices,
        resume_generation=resume_generation,
        cap_order_digest="test",
        applied_order_digest="test",
        vote_source_digest="test",
        flip_residual_applied_lanes=flip_residual_applied_lanes,
        flip_direction_lanes=flip_direction_lanes,
        flip_threshold_lanes=flip_threshold_lanes,
        residual_authority_lanes=residual_authority_lanes,
    )


def test_slow_growth_k_star_equals_full_history() -> None:
    replay = _replay()
    result = analyze_synthetic_lane_trajectory(
        votes=[24, 24, 24],
        acc_trajectory=[24, 48, 72],
        replay=replay,
    )
    measurement = result["measurement"]
    assert measurement["k_star"] == 3
    assert measurement["parity_pass"] is True


def test_saturated_known_bootstrap_k_star_one_from_zero() -> None:
    replay = _replay()
    result = analyze_synthetic_lane_trajectory(
        votes=[72],
        acc_trajectory=[72],
        replay=replay,
    )
    measurement = result["measurement"]
    assert measurement["k_star"] == 1
    assert measurement["bootstrap_used"] == "known_zero"
    assert measurement["parity_pass"] is True


def test_saturated_free_k_star_zero_with_sign_proof() -> None:
    replay = _replay()
    entry = _full_step_entry(
        step=1,
        state_key="tiny.proj",
        replay=replay,
        acc_before=[127],
        acc_after=[127],
        vote_lanes=[0],
        q_before=[10],
        q_after=[10],
        residual_authority_lanes=["not_applicable"],
    )
    measurement = measure_lane_k_star(
        lane_index=0,
        step_entries=[entry],
        lane_position=0,
        replay=replay,
    )
    assert measurement.k_star == 0
    assert measurement.bootstrap_used == "known_saturated_positive"
    assert measurement.parity_pass is True
    failures = validate_bootstrap_record(
        {
            "bootstrap_state": BOOTSTRAP_KNOWN_SATURATED_POSITIVE,
            "saturated_sign_proof": "positive",
        }
    )
    assert failures == []


def test_flip_bounded_trajectory_reconstructs_with_residual() -> None:
    replay = _replay()
    votes = [24, 24, 7]
    accs = [24, 48, 9]
    result = analyze_synthetic_lane_trajectory(
        votes=votes,
        acc_trajectory=accs,
        replay=replay,
        flip_residual_flags=[False, False, True],
    )
    reconstructed = reconstruct_lane_from_bootstrap(
        bootstrap="known_zero",
        votes=votes,
        replay=replay,
        flip_residual_flags=[False, False, True],
        flip_directions=[None, None, 1],
        flip_thresholds=[None, None, replay.threshold_abs],
    )
    assert reconstructed == 9
    assert result["measurement"]["parity_pass"] is True


def test_slow_sub_saturation_unbounded_k_star_risk() -> None:
    replay = _replay()
    votes = [24, 24, 24, 24]
    accs = [24, 48, 72, 96]
    result = analyze_synthetic_lane_trajectory(votes=votes, acc_trajectory=accs, replay=replay)
    measurement = result["measurement"]
    assert measurement["k_star"] == len(votes)
    naive_wrong = 1
    assert measurement["k_star"] != naive_wrong


def test_bootstrap_forbidden_acc_before_smuggling_rejected() -> None:
    failures = validate_bootstrap_record(
        {
            "bootstrap_state": BOOTSTRAP_KNOWN_SATURATED_POSITIVE,
            "saturated_sign_proof": "positive",
            "acc_before_stored": 72,
        }
    )
    assert any("forbidden_bootstrap_field" in item for item in failures)


def test_resume_divergence_fails_classifier(tmp_path: Path) -> None:
    replay = _replay()
    log_path = tmp_path / D_RECOMPUTE_WINDOW_LOG_FILENAME
    initialize_recompute_window_log_for_probe_session(log_path)
    entry = _full_step_entry(
        step=1,
        state_key="tiny.proj",
        replay=replay,
        acc_before=[0],
        acc_after=[24],
        vote_lanes=[24],
    )
    append_recompute_window_log_chunk(log_path, entry)
    diverged = dict(entry)
    diverged["step"] = 2
    diverged["resume_generation"] = 1
    diverged["acc_before_lanes"] = [999]
    diverged["acc_after_lanes"] = [48]
    append_recompute_window_log_chunk(log_path, diverged)
    receipt = analyze_recompute_window_log(log_path, numel_for_bpw=1)
    assert receipt["primary_classifier"] in {
        CLASSIFIER_D_RECOMPUTE_UNBOUNDED_OR_UNOBSERVABLE,
        CLASSIFIER_MISSING_OBSERVABLES_OR_INVALID_WINDOW,
    }


def test_parity_mismatch_not_lead(tmp_path: Path) -> None:
    replay = _replay()
    log_path = tmp_path / D_RECOMPUTE_WINDOW_LOG_FILENAME
    initialize_recompute_window_log_for_probe_session(log_path)
    append_recompute_window_log_chunk(
        log_path,
        _full_step_entry(
            step=1,
            state_key="tiny.proj",
            replay=replay,
            acc_before=[0],
            acc_after=[24],
            vote_lanes=[24],
        ),
    )
    append_recompute_window_log_chunk(
        log_path,
        _full_step_entry(
            step=2,
            state_key="tiny.proj",
            replay=replay,
            acc_before=[24],
            acc_after=[99],
            vote_lanes=[24],
        ),
    )
    receipt = analyze_recompute_window_log(log_path, numel_for_bpw=1)
    assert receipt["primary_classifier"] == CLASSIFIER_D_RECOMPUTE_UNBOUNDED_OR_UNOBSERVABLE
    assert receipt["parity_fail_count"] >= 1


def test_byte_budget_failure_at_or_above_acc_margin() -> None:
    bpw = inclusive_stream_bpw(numel=1, stream_bytes=int(ACC_BUDGET_BPW_UNDER_BASE3_Q / 8.0) + 1)
    assert bpw >= ACC_BUDGET_BPW_UNDER_BASE3_Q


def test_bounded_fixture_can_classify_lead(tmp_path: Path) -> None:
    replay = _replay()
    log_path = tmp_path / D_RECOMPUTE_WINDOW_LOG_FILENAME
    initialize_recompute_window_log_for_probe_session(log_path)
    acc = 127
    for step, vote in enumerate([0, 0, 0], start=1):
        append_recompute_window_log_chunk(
            log_path,
            _full_step_entry(
                step=step,
                state_key="tiny.proj",
                replay=replay,
                acc_before=[acc],
                acc_after=[acc],
                vote_lanes=[vote],
                q_before=[10],
                q_after=[10],
                residual_authority_lanes=["not_applicable"],
            ),
        )
    receipt = analyze_recompute_window_log(log_path, numel_for_bpw=1_000_000)
    assert receipt["primary_classifier"] == CLASSIFIER_D_RECOMPUTE_WINDOW_LEAD
    assert receipt["dual_booleans"]["beats_w8_dense_acc_term"] is True
    assert receipt["inclusive_bpw"]["passes_acc_budget_under_base3_q"] is True


def test_unbounded_promotes_law_redesign_fork(tmp_path: Path) -> None:
    replay = _replay()
    log_path = tmp_path / D_RECOMPUTE_WINDOW_LOG_FILENAME
    initialize_recompute_window_log_for_probe_session(log_path)
    acc = 0
    for step in range(1, 5):
        vote = 24
        acc_after = carry_after_scalar(acc, vote, replay=replay)
        append_recompute_window_log_chunk(
            log_path,
            _full_step_entry(
                step=step,
                state_key="tiny.proj",
                replay=replay,
                acc_before=[acc],
                acc_after=[acc_after],
                vote_lanes=[vote],
            ),
        )
        acc = acc_after
    receipt = analyze_recompute_window_log(log_path, numel_for_bpw=1)
    assert receipt["primary_classifier"] == CLASSIFIER_D_RECOMPUTE_UNBOUNDED_OR_UNOBSERVABLE
    assert receipt["promoted_fork"] == CLASSIFIER_D_NEEDS_UPDATE_LAW_REDESIGN


def test_naive_nonzero_overlap_metric_would_fail_changed_transition_discriminator() -> None:
    replay = _replay()
    votes = [24, 24, 24, 24]
    accs = [24, 48, 72, 96]
    records = []
    acc_before = 0
    for step, (vote, acc_after) in enumerate(zip(votes, accs, strict=True), start=1):
        records.append(
            {
                "step": step,
                "acc_before_lanes": [acc_before],
                "acc_after_lanes": [acc_after],
                "vote_lanes": [vote],
                "flip_residual_applied": False,
            }
        )
        acc_before = acc_after
    measurement = measure_lane_k_star(
        lane_index=0,
        step_entries=records,
        lane_position=0,
        replay=replay,
    )
    assert measurement.k_star == 4
    assert measurement.k_star != 1


def test_multi_lane_worst_case_full_history_blocks_lead(tmp_path: Path) -> None:
    replay = _replay()
    log_path = tmp_path / D_RECOMPUTE_WINDOW_LOG_FILENAME
    initialize_recompute_window_log_for_probe_session(log_path)
    saturated_acc = 127
    slow_acc = 0
    for step in range(1, 5):
        slow_vote = 24
        slow_after = carry_after_scalar(slow_acc, slow_vote, replay=replay)
        append_recompute_window_log_chunk(
            log_path,
            _full_step_entry(
                step=step,
                state_key="tiny.proj",
                replay=replay,
                acc_before=[saturated_acc, slow_acc],
                acc_after=[saturated_acc, slow_after],
                vote_lanes=[0, slow_vote],
                lane_indices=[0, 1],
                q_before=[10, 0],
                q_after=[10, 0],
                residual_authority_lanes=["not_applicable", "not_applicable"],
            ),
        )
        slow_acc = slow_after
    receipt = analyze_recompute_window_log(log_path, numel_for_bpw=1_000_000)
    assert receipt["primary_classifier"] == CLASSIFIER_D_RECOMPUTE_UNBOUNDED_OR_UNOBSERVABLE
    assert receipt["k_star_distribution"]["worst_case_full_history"] is True
    assert receipt["k_star_distribution"]["plateau_signal"] is False
    p95_only_would_mask = (
        receipt["k_star_distribution"]["p95"] is not None
        and float(receipt["k_star_distribution"]["p95"]) < float(
            receipt["k_star_distribution"]["available_history_steps"]
        )
    )
    assert p95_only_would_mask is True


def test_production_q_delta_derivation_0_to_positive_flip() -> None:
    replay = _replay()
    votes = [24, 24, 7]
    accs = [24, 48, 9]
    q_pairs = [(0, 0), (0, 0), (0, 1)]
    records = []
    acc_before = 0
    for step, (vote, acc_after, (q_b, q_a)) in enumerate(
        zip(votes, accs, q_pairs, strict=True),
        start=1,
    ):
        applied, direction, threshold, authority = _derive_lane_flip_residual(
            q_before=q_b,
            q_after=q_a,
            acc_before=acc_before,
            acc_after=acc_after,
            vote=vote,
            replay=replay,
        )
        if step < 3:
            assert authority == "not_applicable"
            assert applied is False
        else:
            assert applied is True
            assert direction == 1
            assert threshold == replay.threshold_abs
            assert authority == "present"
        records.append(
            _full_step_entry(
                step=step,
                state_key="tiny.proj",
                replay=replay,
                acc_before=[acc_before],
                acc_after=[acc_after],
                vote_lanes=[vote],
                q_before=[q_b],
                q_after=[q_a],
                flip_residual_applied_lanes=[applied],
                flip_direction_lanes=[direction],
                flip_threshold_lanes=[threshold],
                residual_authority_lanes=[authority],
            )
        )
        acc_before = acc_after
    parity_lane = measure_lane_k_star(
        lane_index=0,
        step_entries=records,
        lane_position=0,
        replay=replay,
    )
    assert parity_lane.parity_pass is True


def test_production_q_delta_derivation_0_to_negative_flip() -> None:
    replay = _replay()
    acc_before = 48
    vote = 7
    expected_carry = carry_after_scalar(acc_before, vote, replay=replay)
    acc_after = apply_flip_residual_scalar(
        expected_carry,
        direction=-1,
        threshold=replay.threshold_abs,
    )
    applied, direction, threshold, authority = _derive_lane_flip_residual(
        q_before=0,
        q_after=-1,
        acc_before=acc_before,
        acc_after=acc_after,
        vote=vote,
        replay=replay,
    )
    assert applied is True
    assert direction == -1
    assert authority == "present"
    entry = _full_step_entry(
        step=1,
        state_key="tiny.proj",
        replay=replay,
        acc_before=[0],
        acc_after=[24],
        vote_lanes=[24],
        q_before=[0],
        q_after=[0],
        flip_residual_applied_lanes=[False],
        flip_direction_lanes=[None],
        flip_threshold_lanes=[None],
        residual_authority_lanes=["not_applicable"],
    )
    flip_entry = _full_step_entry(
        step=2,
        state_key="tiny.proj",
        replay=replay,
        acc_before=[24],
        acc_after=[48],
        vote_lanes=[24],
        q_before=[0],
        q_after=[0],
        flip_residual_applied_lanes=[False],
        flip_direction_lanes=[None],
        flip_threshold_lanes=[None],
        residual_authority_lanes=["not_applicable"],
    )
    flip_step = _full_step_entry(
        step=3,
        state_key="tiny.proj",
        replay=replay,
        acc_before=[acc_before],
        acc_after=[acc_after],
        vote_lanes=[vote],
        q_before=[0],
        q_after=[-1],
        flip_residual_applied_lanes=[applied],
        flip_direction_lanes=[direction],
        flip_threshold_lanes=[threshold],
        residual_authority_lanes=[authority],
    )
    parity_lane = measure_lane_k_star(
        lane_index=0,
        step_entries=[entry, flip_entry, flip_step],
        lane_position=0,
        replay=replay,
    )
    assert parity_lane.parity_pass is True


def test_production_q_delta_derivation_q_unchanged_veto_is_absent() -> None:
    replay = _replay()
    applied, direction, threshold, authority = _derive_lane_flip_residual(
        q_before=1,
        q_after=1,
        acc_before=24,
        acc_after=99,
        vote=24,
        replay=replay,
    )
    assert applied is False
    assert direction is None
    assert threshold is None
    assert authority == "absent"


def test_old_sign_cross_condition_would_false_null_on_0_to_1() -> None:
    """0→+1 is a real production flip; the old nonzero-both-signs-differ rule missed it."""
    replay = _replay()
    acc_before = 48
    vote = 7
    expected_carry = carry_after_scalar(acc_before, vote, replay=replay)
    acc_after = apply_flip_residual_scalar(
        expected_carry,
        direction=1,
        threshold=replay.threshold_abs,
    )
    applied, _, _, authority = _derive_lane_flip_residual(
        q_before=0,
        q_after=1,
        acc_before=acc_before,
        acc_after=acc_after,
        vote=vote,
        replay=replay,
    )
    old_crossed_rule = (
        0 != 0 and 1 != 0 and (0 > 0) != (1 > 0)
    )
    assert old_crossed_rule is False
    assert applied is True
    assert authority == "present"


def test_impossible_ternary_jump_minus1_to_plus1_is_absent() -> None:
    replay = _replay()
    applied, _, _, authority = _derive_lane_flip_residual(
        q_before=-1,
        q_after=1,
        acc_before=48,
        acc_after=9,
        vote=7,
        replay=replay,
    )
    assert applied is False
    assert authority == "absent"


def test_real_emit_shape_flip_residual_present_and_absent(tmp_path: Path) -> None:
    replay = _replay()
    log_path = tmp_path / D_RECOMPUTE_WINDOW_LOG_FILENAME
    initialize_recompute_window_log_for_probe_session(log_path)
    acc_before = 0
    for step, (vote, acc_after, flip) in enumerate(
        zip([24, 24, 7], [24, 48, 9], [False, False, True], strict=True),
        start=1,
    ):
        append_recompute_window_log_chunk(
            log_path,
            _full_step_entry(
                step=step,
                state_key="tiny.proj",
                replay=replay,
                acc_before=[acc_before],
                acc_after=[acc_after],
                vote_lanes=[vote],
                q_before=[-10 if flip else 10],
                q_after=[10],
                flip_residual_applied_lanes=[flip],
                flip_direction_lanes=[1 if flip else None],
                flip_threshold_lanes=[replay.threshold_abs if flip else None],
                residual_authority_lanes=["present" if flip else "not_applicable"],
            ),
        )
        acc_before = acc_after
    first_three = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()[:3]
    ]
    parity_lane = measure_lane_k_star(
        lane_index=0,
        step_entries=first_three,
        lane_position=0,
        replay=replay,
    )
    assert parity_lane.parity_pass is True
    append_recompute_window_log_chunk(
        log_path,
        _full_step_entry(
            step=4,
            state_key="tiny.proj",
            replay=replay,
            acc_before=[9],
            acc_after=[99],
            vote_lanes=[24],
            q_before=[10],
            q_after=[10],
            residual_authority_lanes=["absent"],
        ),
    )
    absent_lane = measure_lane_k_star(
        lane_index=0,
        step_entries=[json.loads(log_path.read_text(encoding="utf-8").splitlines()[-1])],
        lane_position=0,
        replay=replay,
    )
    assert absent_lane.parity_pass is False
    receipt = analyze_recompute_window_log(log_path, numel_for_bpw=1)
    assert receipt["primary_classifier"] in {
        CLASSIFIER_D_RECOMPUTE_UNBOUNDED_OR_UNOBSERVABLE,
        CLASSIFIER_MISSING_OBSERVABLES_OR_INVALID_WINDOW,
    }


def test_log_inventory_per_key_numel(tmp_path: Path) -> None:
    replay = _replay()
    log_path = tmp_path / D_RECOMPUTE_WINDOW_LOG_FILENAME
    initialize_recompute_window_log_for_probe_session(log_path)
    for step in (1, 2):
        append_recompute_window_log_chunk(
            log_path,
            _full_step_entry(
                step=step,
                state_key="tiny.proj",
                replay=replay,
                acc_before=[10],
                acc_after=[10],
                vote_lanes=[0],
            ),
        )
        append_recompute_window_log_chunk(
            log_path,
            _full_step_entry(
                step=step,
                state_key="tiny.other",
                replay=replay,
                acc_before=[5],
                acc_after=[5],
                vote_lanes=[0],
            ),
        )
    state_numel_by_key = {"tiny.other": 2048, "tiny.proj": 4096}
    receipt = analyze_recompute_window_log(
        log_path,
        numel_for_bpw=sum(state_numel_by_key.values()),
        numel_basis_source="parent_checkpoint_tensor_state_numel",
        state_numel_by_key=state_numel_by_key,
    )
    inventory = receipt["log_inventory"]
    assert inventory["per_key_numel"] == state_numel_by_key
    assert set(inventory["per_key_numel"]) == set(inventory["selected_state_keys"])
    assert inventory["jsonl_row_count"] == 4
    assert inventory["sampled_lane_count_by_key"]["tiny.proj"] == 1
