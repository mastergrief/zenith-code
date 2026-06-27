"""Unit tests for dense accumulator width parity screen (C3 Phase-0)."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from calm.hrm_text_158.native_full_stack.dense_accumulator_width_parity_screen import (
    BELOW_THRESHOLD_TRIVIAL_WIDTH,
    BOUNDARY_OVERSHOOT_SCENARIO_CLASS,
    BOUNDARY_TESTED_WIDTH,
    CANONICAL_MAX_RANK_VOTE_ABS,
    CLASSIFIER_C3_DENSE_WIDTH_REDUCIBLE,
    CLASSIFIER_C3_DENSE_WIDTH_REDUCIBLE_TO_FLOOR,
    CLASSIFIER_C3_INT16_STORAGE_OVERWIDE,
    CLASSIFIER_C3_MISSING_OBSERVABLES,
    CLASSIFIER_C3_NARROW_WIDTH_BREAKS_DECISION_PARITY,
    CLASSIFIER_C3_RUN_HEALTH_FAIL,
    MANDATORY_WIDTH_GRID,
    O5FixtureResult,
    SUB_FLOOR_BOUNDARY_TESTED_WIDTHS,
    SyntheticWidthScenario,
    WidthDriftRow,
    classify_c3_dense_width_screen,
    clip_abs_for_width,
    default_mandatory_scenarios,
    effective_clip_for_width,
    is_structurally_lossless_width,
    reachable_pre_crossing_accumulator_peak,
    run_width_parity_screen,
    simulate_lane_trajectory,
    structural_clip_floor_width,
    width_regime_label,
)


def test_w4_labeled_below_threshold_trivial_and_excluded_from_parity_failure_count() -> None:
    result = run_width_parity_screen()
    w4_rows = [row for row in result.drift_rows if row.width == BELOW_THRESHOLD_TRIVIAL_WIDTH]
    assert w4_rows, "expected W4 drift rows in mandatory grid"
    assert all(row.width_regime == "below_threshold_trivial" for row in w4_rows)
    assert all(row.excluded_from_parity_failure for row in w4_rows)
    assert result.parity_failure_count == sum(
        1 for row in result.drift_rows if row.drift and not row.excluded_from_parity_failure
    )


def test_w5_w6_sub_floor_boundary_tested_regimes() -> None:
    peak = reachable_pre_crossing_accumulator_peak(max_vote_abs=CANONICAL_MAX_RANK_VOTE_ABS)
    assert peak == 33
    assert SUB_FLOOR_BOUNDARY_TESTED_WIDTHS == (5, 6)
    assert width_regime_label(5, reachable_peak=peak) == "sub_floor_boundary_tested"
    assert width_regime_label(6, reachable_peak=peak) == "sub_floor_boundary_tested"
    clip_min, clip_max = effective_clip_for_width(5)
    assert clip_min == -15 and clip_max == 15
    assert clip_abs_for_width(6) == 31
    assert clip_abs_for_width(6) < peak
    assert not is_structurally_lossless_width(6, reachable_peak=peak)


def test_w7_structurally_lossless_w6_not() -> None:
    peak = reachable_pre_crossing_accumulator_peak(max_vote_abs=CANONICAL_MAX_RANK_VOTE_ABS)
    assert is_structurally_lossless_width(7, reachable_peak=peak)
    assert width_regime_label(7, reachable_peak=peak) == "clip_exceeds_reachable_range"
    assert width_regime_label(8, reachable_peak=peak) == "source_clip_lossless"
    assert structural_clip_floor_width(MANDATORY_WIDTH_GRID, reachable_peak=peak) == 7


def test_w8_storage_overwide_only_when_narrow_not_reducible() -> None:
    boundary_probe_rows = (
        WidthDriftRow(
            scenario_name="boundary_overshoot_positive",
            scenario_class=BOUNDARY_OVERSHOOT_SCENARIO_CLASS,
            width=5,
            width_regime="sub_floor_boundary_tested",
            drift=True,
            o1_crossing_mismatch=False,
            o2_flip_mismatch=False,
            o3_acc_mismatch=True,
            o4_q_mismatch=False,
            excluded_from_parity_failure=False,
        ),
        WidthDriftRow(
            scenario_name="boundary_overshoot_positive",
            scenario_class=BOUNDARY_OVERSHOOT_SCENARIO_CLASS,
            width=6,
            width_regime="sub_floor_boundary_tested",
            drift=False,
            o1_crossing_mismatch=False,
            o2_flip_mismatch=False,
            o3_acc_mismatch=False,
            o4_q_mismatch=False,
            excluded_from_parity_failure=False,
        ),
    )
    drift_rows = (
        WidthDriftRow(
            scenario_name="delayed_crossing_sparse_votes",
            scenario_class="baseline",
            width=8,
            width_regime="source_clip_lossless",
            drift=False,
            o1_crossing_mismatch=False,
            o2_flip_mismatch=False,
            o3_acc_mismatch=False,
            o4_q_mismatch=False,
            excluded_from_parity_failure=False,
        ),
        *boundary_probe_rows,
    )
    classifier, basis, storage_overwide, decisive_safe = classify_c3_dense_width_screen(
        drift_rows=drift_rows,
        width_grid=MANDATORY_WIDTH_GRID,
        screen_complete=True,
        o5=O5FixtureResult(
            observed=True,
            reason="test",
            reference_width=16,
            surfaces_by_width=(),
            o5_drift_vs_reference=(),
        ),
        bpw_by_width={str(width): float(width) for width in MANDATORY_WIDTH_GRID},
        boundary_probe_rows=boundary_probe_rows,
        structural_floor_width=7,
        minimum_safe_width_empirical=6,
        sub_floor_breaking_widths=(5,),
    )
    assert storage_overwide is True
    assert decisive_safe is False
    assert classifier == CLASSIFIER_C3_DENSE_WIDTH_REDUCIBLE_TO_FLOOR
    assert classifier != CLASSIFIER_C3_DENSE_WIDTH_REDUCIBLE
    assert "structural_clip_floor_w7" in basis


def test_w8_lossless_does_not_by_itself_select_dense_width_reducible() -> None:
    boundary_probe_rows = (
        WidthDriftRow(
            scenario_name="boundary_overshoot_positive",
            scenario_class=BOUNDARY_OVERSHOOT_SCENARIO_CLASS,
            width=5,
            width_regime="sub_floor_boundary_tested",
            drift=True,
            o1_crossing_mismatch=False,
            o2_flip_mismatch=False,
            o3_acc_mismatch=True,
            o4_q_mismatch=False,
            excluded_from_parity_failure=False,
        ),
    )
    drift_rows = (
        *boundary_probe_rows,
        WidthDriftRow(
            scenario_name="delayed_crossing_sparse_votes",
            scenario_class="baseline",
            width=8,
            width_regime="source_clip_lossless",
            drift=False,
            o1_crossing_mismatch=False,
            o2_flip_mismatch=False,
            o3_acc_mismatch=False,
            o4_q_mismatch=False,
            excluded_from_parity_failure=False,
        ),
    )
    classifier, _, _, _ = classify_c3_dense_width_screen(
        drift_rows=drift_rows,
        width_grid=MANDATORY_WIDTH_GRID,
        screen_complete=True,
        o5=O5FixtureResult(
            observed=True,
            reason="test",
            reference_width=16,
            surfaces_by_width=(),
            o5_drift_vs_reference=(),
        ),
        bpw_by_width={str(width): float(width) for width in MANDATORY_WIDTH_GRID},
        boundary_probe_rows=boundary_probe_rows,
        structural_floor_width=7,
        minimum_safe_width_empirical=6,
        sub_floor_breaking_widths=(5,),
    )
    assert classifier == CLASSIFIER_C3_DENSE_WIDTH_REDUCIBLE_TO_FLOOR
    assert classifier != CLASSIFIER_C3_DENSE_WIDTH_REDUCIBLE


def test_incomplete_screen_selects_run_health_fail_not_intrinsic() -> None:
    result = run_width_parity_screen(screen_complete=False)
    assert result.classifier == CLASSIFIER_C3_RUN_HEALTH_FAIL
    assert "intrinsic" not in result.classifier.lower()


def test_missing_o5_fixture_selects_missing_observables_not_full_contract_pass() -> None:
    with patch(
        "calm.hrm_text_158.native_full_stack.dense_accumulator_width_parity_screen.build_o5_fixture_result",
        return_value=O5FixtureResult(
            observed=False,
            reason="fixture_not_constructible",
            reference_width=16,
            surfaces_by_width=(),
            o5_drift_vs_reference=(),
        ),
    ):
        result = run_width_parity_screen()
    assert result.classifier == CLASSIFIER_C3_MISSING_OBSERVABLES
    assert result.classifier != CLASSIFIER_C3_DENSE_WIDTH_REDUCIBLE


def test_classifier_keys_prefixed_c3() -> None:
    result = run_width_parity_screen()
    assert result.classifier.startswith("C3_")


def test_mandatory_scenarios_include_sub_threshold_staircase_and_overshoot() -> None:
    names = {scenario.name for scenario in default_mandatory_scenarios()}
    assert "sub_threshold_staircase" in names
    assert "boundary_overshoot_positive" in names
    assert "boundary_overshoot_canonical_vote24" in names
    overshoot = [
        scenario
        for scenario in default_mandatory_scenarios()
        if scenario.scenario_class == BOUNDARY_OVERSHOOT_SCENARIO_CLASS
    ]
    assert len(overshoot) == 4


def test_full_screen_reducible_to_structural_floor_w7_w5_breaks_w6_passes() -> None:
    result = run_width_parity_screen()
    assert result.classifier == CLASSIFIER_C3_DENSE_WIDTH_REDUCIBLE_TO_FLOOR
    assert result.classifier != CLASSIFIER_C3_DENSE_WIDTH_REDUCIBLE
    assert result.structural_clip_floor_width == 7
    assert result.minimum_safe_width_empirical == 6
    assert result.sub_floor_breaking_widths == (5,)
    assert result.sub_floor_parity_safe_widths == (6,)
    assert result.reachable_pre_crossing_peak == 33
    assert result.bpw_reduction_at_structural_floor == pytest.approx(0.5625)
    assert result.persistent_state_bpw_at_structural_floor_estimate == pytest.approx(15.0)
    assert "vote-accumulator term only" in result.sub2_scope_caveat
    assert "not sub-2 inclusive" in result.sub2_scope_caveat.lower() or "does not achieve sub-2" in result.sub2_scope_caveat.lower()
    assert "vote-acc term only" in result.classifier_label
    w5_overshoot_rows = [
        row
        for row in result.drift_rows
        if row.width == BOUNDARY_TESTED_WIDTH
        and row.scenario_class == BOUNDARY_OVERSHOOT_SCENARIO_CLASS
    ]
    assert len(w5_overshoot_rows) == 4
    assert all(row.drift for row in w5_overshoot_rows)
    assert all(row.o3_acc_mismatch for row in w5_overshoot_rows)
    assert not any(row.o1_crossing_mismatch for row in w5_overshoot_rows)
    w6_overshoot_rows = [
        row
        for row in result.drift_rows
        if row.width == 6 and row.scenario_class == BOUNDARY_OVERSHOOT_SCENARIO_CLASS
    ]
    assert len(w6_overshoot_rows) == 4
    assert all(not row.drift for row in w6_overshoot_rows)


def test_baseline_scenarios_do_not_count_sub_floor_toward_parity_failure() -> None:
    result = run_width_parity_screen()
    baseline_sub_floor = [
        row
        for row in result.drift_rows
        if row.width in SUB_FLOOR_BOUNDARY_TESTED_WIDTHS and row.scenario_class == "baseline"
    ]
    assert baseline_sub_floor
    assert all(row.excluded_from_parity_failure for row in baseline_sub_floor)


def test_w8_lossless_matches_w16_trajectory_on_delayed_crossing() -> None:
    scenario = SyntheticWidthScenario(name="delayed_crossing_sparse_votes", votes=(5, 0, 0, 5))
    ref = simulate_lane_trajectory(scenario, width=16)
    w8 = simulate_lane_trajectory(scenario, width=8)
    assert ref.crossing_steps == w8.crossing_steps
    assert ref.q_history == w8.q_history
    assert ref.acc_history == w8.acc_history


def test_w8_in_vivo_reachable_peak_127_structural_floor() -> None:
    """Distinct from peak=33 structural-floor claim — in-vivo accumulator ceiling."""

    in_vivo_accumulator_peak = 127
    assert structural_clip_floor_width(
        MANDATORY_WIDTH_GRID, reachable_peak=in_vivo_accumulator_peak
    ) == 8
    assert is_structurally_lossless_width(8, reachable_peak=in_vivo_accumulator_peak)
    assert (
        width_regime_label(8, reachable_peak=in_vivo_accumulator_peak)
        == "source_clip_lossless"
    )
    assert not is_structurally_lossless_width(7, reachable_peak=in_vivo_accumulator_peak)
    # Preserve canonical crossing-peak structural floor (separate claim).
    canonical_peak = reachable_pre_crossing_accumulator_peak(
        max_vote_abs=CANONICAL_MAX_RANK_VOTE_ABS
    )
    assert canonical_peak == 33
    assert structural_clip_floor_width(MANDATORY_WIDTH_GRID, reachable_peak=canonical_peak) == 7
