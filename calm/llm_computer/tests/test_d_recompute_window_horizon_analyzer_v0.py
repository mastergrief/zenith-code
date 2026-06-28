from __future__ import annotations

import copy

import pytest

from calm.hrm_text_158.native_full_stack.d_recompute_window_emit import (
    ReplayConstants,
    build_step_log_entry,
    default_production_replay_constants,
)
from calm.hrm_text_158.native_full_stack.d_recompute_window_horizon_analyzer import (
    GROWTH_ACCELERATING_OR_RIGHT_CENSORED,
    GROWTH_INCONCLUSIVE_COST_OR_COVERAGE,
    GROWTH_LINEAR_SIZED_WITH_DECAY,
    GROWTH_PLATEAU_SIZED,
    GROWTH_RIGHT_CENSORED_LOWER_BOUND,
    analyze_horizon_k_star_growth,
    audit_lane_coverage,
    classify_k_star_growth,
    resolve_lane_position,
    summarize_k_star_at_horizon_prefix,
)
from calm.hrm_text_158.native_full_stack.d_recompute_window_stratified_selector import (
    COVERAGE_TIER_PILOT,
    COVERAGE_TIER_REPRESENTATIVE,
    STRESS_TAIL_POLICY_HORIZON_FIXED,
)


def _replay() -> ReplayConstants:
    return default_production_replay_constants()


def _linear_lane_records(
    *,
    state_key: str,
    steps: int,
    lane_indices: list[int],
    vote: int = 1,
) -> list[dict]:
    replay = _replay()
    records: list[dict] = []
    acc = 0
    for step in range(1, int(steps) + 1):
        acc_before = acc
        acc_after = acc_before + int(vote)
        acc = acc_after
        lane_count = len(lane_indices)
        records.append(
            build_step_log_entry(
                step=step,
                state_key=state_key,
                replay_constants=replay,
                acc_before=[acc_before] * lane_count,
                acc_after=[acc_after] * lane_count,
                q_before=[0] * lane_count,
                q_after=[0] * lane_count,
                vote_lanes=[int(vote)] * lane_count,
                lane_indices=lane_indices,
            )
        )
    return records


def test_prefix_truncation_reproduces_native_prefix_k_star() -> None:
    state_key = "model.H_level.core.layers.0.attn.o_proj"
    full = _linear_lane_records(
        state_key=state_key,
        steps=100,
        lane_indices=[0],
        vote=1,
    )
    native_25 = _linear_lane_records(
        state_key=state_key,
        steps=25,
        lane_indices=[0],
        vote=1,
    )
    prefix_25 = summarize_k_star_at_horizon_prefix(full, 25)
    native_summary = summarize_k_star_at_horizon_prefix(native_25, 25)
    assert prefix_25["k99_unweighted"] == native_summary["k99_unweighted"] == 25.0


def test_k_star_measurement_has_no_future_step_leakage() -> None:
    state_key = "model.H_level.core.layers.0.attn.o_proj"
    records = _linear_lane_records(state_key=state_key, steps=100, lane_indices=[0], vote=1)
    before = summarize_k_star_at_horizon_prefix(records, 50)
    mutated = copy.deepcopy(records)
    for record in mutated:
        if int(record["step"]) > 50:
            record["vote_lanes"] = [99]
            record["acc_after_lanes"] = [999]
    after = summarize_k_star_at_horizon_prefix(mutated, 50)
    assert before["k99_unweighted"] == after["k99_unweighted"]


def test_per_record_lane_position_not_first_observed() -> None:
    replay = _replay()
    step1 = build_step_log_entry(
        step=1,
        state_key="key.a",
        replay_constants=replay,
        acc_before=[0, 0],
        acc_after=[1, 1],
        q_before=[0, 0],
        q_after=[0, 0],
        vote_lanes=[1, 1],
        lane_indices=[5, 7],
    )
    step2 = build_step_log_entry(
        step=2,
        state_key="key.a",
        replay_constants=replay,
        acc_before=[1, 1],
        acc_after=[2, 2],
        q_before=[0, 0],
        q_after=[0, 0],
        vote_lanes=[1, 1],
        lane_indices=[7, 5],
    )
    assert resolve_lane_position(step1, 7) == 1
    assert resolve_lane_position(step2, 7) == 0
    summary = summarize_k_star_at_horizon_prefix([step1, step2], 2)
    lane7 = next(row for row in summary["lane_rows"] if row["lane_index"] == 7)
    assert lane7["lane_position_by_step"] == {"1": 1, "2": 0}
    assert lane7["k_star"] == 2


def test_contiguous_coverage_guard_flags_gap_and_excludes_from_distribution() -> None:
    replay = _replay()
    records = [
        build_step_log_entry(
            step=1,
            state_key="key.a",
            replay_constants=replay,
            acc_before=[0],
            acc_after=[1],
            q_before=[0],
            q_after=[0],
            vote_lanes=[1],
            lane_indices=[0],
        ),
        build_step_log_entry(
            step=3,
            state_key="key.a",
            replay_constants=replay,
            acc_before=[1],
            acc_after=[2],
            q_before=[0],
            q_after=[0],
            vote_lanes=[1],
            lane_indices=[0],
        ),
    ]
    coverage = audit_lane_coverage(records, horizon_h=3)
    assert coverage["observation_count"] == 2
    assert coverage["gap_count"] == 1
    assert coverage["contiguous"] is False
    summary = summarize_k_star_at_horizon_prefix(records, 3)
    assert summary["gapped_lane_count"] == 1
    assert summary["eligible_lane_count"] == 0


def test_weighted_quantile_uses_stratum_weights() -> None:
    state_key_heavy = "heavy.key"
    state_key_light = "light.key"
    heavy = _linear_lane_records(
        state_key=state_key_heavy,
        steps=20,
        lane_indices=[0],
        vote=1,
    )
    light = _linear_lane_records(
        state_key=state_key_light,
        steps=5,
        lane_indices=[0],
        vote=1,
    )
    records = heavy + light
    unweighted = summarize_k_star_at_horizon_prefix(records, 20, stratum_weights=None)
    weighted = summarize_k_star_at_horizon_prefix(
        records,
        20,
        stratum_weights={state_key_heavy: 10.0, state_key_light: 0.01},
    )
    assert unweighted["k99_unweighted"] is not None
    assert weighted["k99_weighted"] is not None
    assert float(weighted["k99_weighted"]) > float(unweighted["k99_unweighted"]) / 2.0


def test_branch_routing_plateau_fixture() -> None:
    records = _linear_lane_records(
        state_key="plateau.key",
        steps=100,
        lane_indices=[0],
        vote=0,
    )
    result = analyze_horizon_k_star_growth(
        records,
        stratum_weights={"plateau.key": 1.0},
        stress_tail_policy=STRESS_TAIL_POLICY_HORIZON_FIXED,
        coverage_tier=COVERAGE_TIER_REPRESENTATIVE,
    )
    assert result["growth_branch"] == GROWTH_PLATEAU_SIZED


def test_branch_routing_linear_fixture() -> None:
    records: list[dict] = []
    for lane_index, vote in ((0, 1), (1, 2), (2, 3)):
        records.extend(
            _linear_lane_records(
                state_key=f"linear.{lane_index}",
                steps=100,
                lane_indices=[lane_index],
                vote=vote,
            )
        )
    result = analyze_horizon_k_star_growth(
        records,
        stratum_weights={f"linear.{lane_index}": 1.0 for lane_index in (0, 1, 2)},
        stress_tail_policy=STRESS_TAIL_POLICY_HORIZON_FIXED,
        coverage_tier=COVERAGE_TIER_REPRESENTATIVE,
    )
    assert result["growth_branch"] in {
        GROWTH_LINEAR_SIZED_WITH_DECAY,
        GROWTH_PLATEAU_SIZED,
        GROWTH_INCONCLUSIVE_COST_OR_COVERAGE,
    }


def test_branch_routing_right_censored_fixture() -> None:
    records = _linear_lane_records(
        state_key="censored.key",
        steps=100,
        lane_indices=[0],
        vote=1,
    )
    result = analyze_horizon_k_star_growth(
        records,
        stratum_weights={"censored.key": 1.0},
        stress_tail_policy=STRESS_TAIL_POLICY_HORIZON_FIXED,
        coverage_tier=COVERAGE_TIER_REPRESENTATIVE,
    )
    assert result["growth_branch"] == GROWTH_RIGHT_CENSORED_LOWER_BOUND


def test_branch_routing_inconclusive_for_pilot_coverage_tier() -> None:
    records = _linear_lane_records(
        state_key="pilot.key",
        steps=100,
        lane_indices=[0],
        vote=1,
    )
    result = analyze_horizon_k_star_growth(
        records,
        stratum_weights={"pilot.key": 1.0},
        stress_tail_policy=STRESS_TAIL_POLICY_HORIZON_FIXED,
        coverage_tier=COVERAGE_TIER_PILOT,
    )
    assert result["growth_branch"] == GROWTH_INCONCLUSIVE_COST_OR_COVERAGE


def test_branch_routing_inconclusive_for_gapped_lane_fraction() -> None:
    good = _linear_lane_records(
        state_key="good.key",
        steps=100,
        lane_indices=[0],
        vote=1,
    )
    replay = _replay()
    gapped = [
        build_step_log_entry(
            step=1,
            state_key="gapped.key",
            replay_constants=replay,
            acc_before=[0],
            acc_after=[1],
            q_before=[0],
            q_after=[0],
            vote_lanes=[1],
            lane_indices=[1],
        ),
        build_step_log_entry(
            step=3,
            state_key="gapped.key",
            replay_constants=replay,
            acc_before=[1],
            acc_after=[2],
            q_before=[0],
            q_after=[0],
            vote_lanes=[1],
            lane_indices=[1],
        ),
    ]
    padded = _linear_lane_records(
        state_key="gapped.key",
        steps=100,
        lane_indices=[1],
        vote=1,
    )
    gapped_records = gapped + padded[2:]
    result = analyze_horizon_k_star_growth(
        good + gapped_records,
        stratum_weights={"good.key": 1.0, "gapped.key": 1.0},
        stress_tail_policy=STRESS_TAIL_POLICY_HORIZON_FIXED,
        coverage_tier=COVERAGE_TIER_REPRESENTATIVE,
    )
    assert result["growth_branch"] == GROWTH_INCONCLUSIVE_COST_OR_COVERAGE
    assert result["growth_branch_detail"]["reason"] == "gapped_lane_fraction"


def test_branch_routing_accelerating_requires_three_uncensored_horizons() -> None:
    summaries = {
        25: {
            "horizon_h": 25,
            "lane_count": 4,
            "gapped_lane_count": 0,
            "parity_fail_count": 0,
            "right_censor_rate": 0.0,
            "k99_weighted": 8.0,
            "kworst_weighted": 8.0,
            "lane_rows": [{"k_star": 8, "gapped": False, "parity_pass": True}],
        },
        50: {
            "horizon_h": 50,
            "lane_count": 4,
            "gapped_lane_count": 0,
            "parity_fail_count": 0,
            "right_censor_rate": 0.0,
            "k99_weighted": 22.0,
            "kworst_weighted": 22.0,
            "lane_rows": [{"k_star": 22, "gapped": False, "parity_pass": True}],
        },
        100: {
            "horizon_h": 100,
            "lane_count": 4,
            "gapped_lane_count": 0,
            "parity_fail_count": 0,
            "right_censor_rate": 0.0,
            "k99_weighted": 70.0,
            "kworst_weighted": 85.0,
            "lane_rows": [{"k_star": 70, "gapped": False, "parity_pass": True}],
        },
    }
    result = classify_k_star_growth(
        summaries,
        stress_tail_policy=STRESS_TAIL_POLICY_HORIZON_FIXED,
        coverage_tier=COVERAGE_TIER_REPRESENTATIVE,
    )
    assert result["growth_branch"] == GROWTH_ACCELERATING_OR_RIGHT_CENSORED
