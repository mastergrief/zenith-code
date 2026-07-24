"""Fork-2 scale-smoke STOP/GO artifact contract assembly (smoke-only)."""
from __future__ import annotations

from typing import Any, Mapping


R6_TEST_ID = "test_r6_index_only_q_shadow_25step_invariant"
SCHEMA = "pressure_metric_fork2_derisk/v3"
NEED_S = 9.3
LIFECYCLE_PHASES = [
    "process_pre_writeback",
    "close_before_writeback_resets",
    "roll_tracker_after_writeback",
]


def base_incomplete_artifact(
    *,
    plan_numel: int,
    r6_status: str,
    geometry: Mapping[str, Any],
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "status": "INCOMPLETE",
        "stop_go": "INCOMPLETE_BLOCKED",
        "plan_numel_flag": int(plan_numel),
        "plan_numel_note": (
            "18158319 was run-3 N_censored (events), not tensor numel; "
            "R5 requires real multi-arm shapes (total 29360128)."
        ),
        "r6_status": str(r6_status),
        "r6_test_id": R6_TEST_ID,
        "geometry": dict(geometry),
        "measurement": {
            "warmup": int(warmup),
            "repeats": int(repeats),
            "aggregator": "median",
        },
    }


def assemble_measured_artifact(
    *,
    base: Mapping[str, Any],
    phase_ms_median: Mapping[str, float],
    phase_samples: Mapping[str, Any],
    windows: Mapping[str, float],
    decision: Mapping[str, Any],
    sync_inventory: Mapping[str, Any],
    close_fixture_invariants: Mapping[str, Any],
    n_candidates_sample: int,
    n_applied_sample: int,
    vram_peak_mb: float,
    publish_ok: bool,
) -> dict[str, Any]:
    out = dict(base)
    out.update(
        {
            "status": "MEASURED",
            "stop_go": decision["stop_go"],
            "extrapolation_method": (
                "median(per_step_ms) × steps + median(finalize_window_ms) as fixed "
                "per-window cost; per_step = select + FULL lifecycle "
                "(process_pre + close_before_writeback_resets + "
                "roll_tracker_after_writeback) + update + writeback; "
                "writeback uses max(one_arm, cross_arm_full_topk=1024@32arms); "
                "lifecycle uses max(full_sequence_median, sum_of_phase_medians); "
                "close/reset timed on representative open_applied>0 fixture"
            ),
            "run3_need_observer_delta_s": NEED_S,
            "projected_observer_delta_s": decision["projected_observer_delta_s"],
            "phase_ms_median": dict(phase_ms_median),
            "phase_samples": dict(phase_samples),
            "close_fixture_invariants": dict(close_fixture_invariants),
            "n_candidates_sample": int(n_candidates_sample),
            "n_applied_sample": int(n_applied_sample),
            "vram_peak_mb": float(vram_peak_mb),
            "sync_inventory": dict(sync_inventory),
            "r4_q_shadow_variants": {
                "full_h2d": {
                    "per_step_ms": windows["per_step_full_ms"],
                    "window_ms": windows["window_full_ms"],
                    "go": decision["go_full"],
                    "includes_lifecycle_phases": list(LIFECYCLE_PHASES),
                    "includes_finalize_fixed": True,
                },
                "index_only": {
                    "per_step_ms": windows["per_step_index_ms"],
                    "window_ms": windows["window_index_ms"],
                    "go": decision["go_idx"],
                    "correct": decision["index_only_correct"],
                    "r6_status": base["r6_status"],
                    "r6_test_id": R6_TEST_ID,
                    "includes_lifecycle_phases": list(LIFECYCLE_PHASES),
                    "includes_finalize_fixed": True,
                },
                "disagreement_on_stop_go": decision["disagreement_on_stop_go"],
                "projection_uses": decision["projection_uses"],
            },
            "device_residency": {
                "acc": "cuda int16",
                "episode_start": "cuda int32",
                "trackers": "cuda int32×3",
                "aggregates": "cuda int64×8",
                "q_levels": "cpu int8 authoritative",
                "q_shadow": "cuda int8 mirror",
            },
            "hot_loop_residency": {
                "full_mask_h2d_d2h": False,
                "writeback_scalar_publishes_le_2": bool(publish_ok),
                "writeback_batched_global_d2h": True,
                "close_fixture_representative": bool(
                    close_fixture_invariants.get("representative", False)
                ),
            },
        }
    )
    return out
