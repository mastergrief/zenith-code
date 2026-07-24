"""Characterization tests for fork-2 scale seams (fixtures + projection reducers)."""
from __future__ import annotations

import torch

from calm.hrm_text_158.native_full_stack.pressure_metric_fork2_scale_fixtures import (
    compute_close_fixture_invariants,
    cross_arm_full_topk_applied,
    seed_open_applied_for_close,
    split_residual_zero_on_applied,
)
from calm.hrm_text_158.native_full_stack.pressure_metric_fork2_scale_timing import (
    decide_stop_go,
    lifecycle_projected_ms,
    project_observer_windows,
    writeback_projected_ms,
)
from calm.hrm_text_158.native_full_stack.pressure_metric_gpu_lifecycle_derisk import (
    DeviceLifecycleStore,
)
from calm.hrm_text_158.native_full_stack.two_tier_threshold_semantics import (
    CROSSING_THRESHOLD_ABS,
)


def test_cross_arm_full_topk_distributes_exactly():
    # 4 arms × 8 = 32 topk
    shapes = {f"arm{i}": (64, 8) for i in range(4)}
    thr = int(CROSSING_THRESHOLD_ABS)
    acc = {n: torch.zeros(s, dtype=torch.int16) for n, s in shapes.items()}
    _c, applied, n_cand, n_app, _o = cross_arm_full_topk_applied(acc, thr=thr, topk=32)
    assert n_cand == 32
    assert n_app == 32
    arms_hit = sum(1 for m in applied.values() if bool(m.any()))
    assert arms_hit == 4
    for m in applied.values():
        assert int(m.sum().item()) == 8


def test_close_fixture_invariants_representative():
    shapes = {"a": (32, 16), "b": (16, 16)}
    store = DeviceLifecycleStore.from_arm_shapes(shapes, steps=25, device="cpu")
    thr = int(CROSSING_THRESHOLD_ABS)
    applied = {n: torch.zeros(s, dtype=torch.bool) for n, s in shapes.items()}
    applied["a"].reshape(-1)[:8] = True
    applied["b"].reshape(-1)[:4] = True
    residual_zero = split_residual_zero_on_applied(applied)
    open_n = seed_open_applied_for_close(store, applied, first_step=1, after_step=0)
    assert open_n == 12
    first_before = {n: store.first_deferral_step[n].clone() for n in applied}
    agg_before = store.aggregates_t.clone()
    store.close_before_writeback_resets(
        applied_masks=applied, step=5, residual_zero=residual_zero
    )
    inv = compute_close_fixture_invariants(
        store,
        applied=applied,
        residual_zero=residual_zero,
        aggregates_before=agg_before,
        first_before=first_before,
    )
    assert inv["open_applied_count"] == 12
    assert inv["residual_clear_count"] > 0
    assert inv["residual_restart_count"] > 0
    assert inv["trackers_cleared_on_close"] > 0
    assert inv["aggregates_mutated"] is True
    assert inv["representative"] is True
    del thr


def test_projection_math_matches_gate1_rederive():
    # Gate-1 display used rounded inputs → ~8321ms; reducer uses exact arithmetic.
    life = lifecycle_projected_ms(
        process_ms=74.4, close_ms=129.0, roll_ms=15.5, full_sequence_ms=158.5
    )
    assert abs(life["lifecycle_sum_of_phase_medians"] - 218.9) < 1e-9
    assert abs(life["lifecycle_projected"] - 218.9) < 1e-9
    wb = writeback_projected_ms(17.7, 24.5)
    assert wb["writeback_projection_uses"] == "cross_arm_full_topk"
    assert abs(float(wb["writeback_projection_median_ms"]) - 24.5) < 1e-9
    windows = project_observer_windows(
        select_ms=69.7,
        lifecycle_ms=218.9,
        update_full_ms=31.3,
        update_index_ms=17.9,
        writeback_ms=24.5,
        finalize_ms=47.3,
        steps=25,
    )
    expect = (69.7 + 218.9 + 17.9 + 24.5) * 25 + 47.3
    assert abs(windows["window_index_ms"] - expect) < 1e-9
    decision = decide_stop_go(
        window_full_ms=windows["window_full_ms"],
        window_index_ms=windows["window_index_ms"],
        need_s=9.3,
        r6_status="PASS",
        full_mask_transfer=False,
        publish_ok=True,
    )
    assert decision["stop_go"] == "GO"
    assert abs(decision["projected_observer_delta_s"] - expect / 1000.0) < 1e-12


def test_decide_stop_go_red_when_over_bound():
    decision = decide_stop_go(
        window_full_ms=10000.0,
        window_index_ms=9500.0,
        need_s=9.3,
        r6_status="PASS",
        full_mask_transfer=False,
        publish_ok=True,
    )
    assert decision["stop_go"] == "RED_STOP"
    assert decision["go_idx"] is False
