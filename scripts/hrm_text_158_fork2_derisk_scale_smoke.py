#!/usr/bin/env python3
"""Thin fork-2 scale-smoke launcher (fixtures/timing/artifact live in importable seams)."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from calm.hrm_text_158.native_full_stack.pressure_metric_fork2_geometry import (
    RUN3_REAL_ARM_SHAPES,
    make_zero_arms,
    run3_total_numel,
)
from calm.hrm_text_158.native_full_stack.pressure_metric_fork2_scale_artifact import (
    NEED_S,
    assemble_measured_artifact,
    base_incomplete_artifact,
)
from calm.hrm_text_158.native_full_stack.pressure_metric_fork2_scale_fixtures import (
    compute_close_fixture_invariants,
    cross_arm_full_topk_applied,
    episode_rollover_pair,
    one_arm_favoring_applied,
    seed_open_applied_for_close,
    seed_open_applied_for_full_step,
    split_residual_zero_on_applied,
    writeback_distribution_inventory,
)
from calm.hrm_text_158.native_full_stack.pressure_metric_fork2_scale_timing import (
    cuda_sync,
    decide_stop_go,
    lifecycle_projected_ms,
    median_ms,
    project_observer_windows,
    time_phase,
    time_phase_with_setup,
    writeback_projected_ms,
)
from calm.hrm_text_158.native_full_stack.pressure_metric_gpu_lifecycle_derisk import (
    DeviceLifecycleStore,
    run_full_per_step_lifecycle,
)
from calm.hrm_text_158.native_full_stack.pressure_metric_gpu_selection_derisk import (
    project_and_update_acc_episode,
    select_topk_masks_deterministic,
    writeback_bridge_cpu_q,
)
from calm.hrm_text_158.native_full_stack.two_tier_threshold_semantics import (
    CROSSING_THRESHOLD_ABS,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--topk", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--horizon", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument(
        "--out",
        default="artifacts/acc_entropy/pressure_metric_fork2_derisk_stop_go.json",
    )
    ap.add_argument("--numel", type=int, default=18_158_319)
    ap.add_argument("--r6-status", default="PASS", choices=["PASS", "FAIL", "UNKNOWN"])
    args = ap.parse_args()

    geometry = {
        "n_arms": len(RUN3_REAL_ARM_SHAPES),
        "total_numel": run3_total_numel(),
        "arm_shapes": [
            {"name": n, "shape": list(s), "numel": int(s[0] * s[1])}
            for n, s in RUN3_REAL_ARM_SHAPES
        ],
        "order": "canonical _param_to_module / q_levels insertion",
    }
    out = base_incomplete_artifact(
        plan_numel=int(args.numel),
        r6_status=str(args.r6_status),
        geometry=geometry,
        warmup=int(args.warmup),
        repeats=int(args.repeats),
    )

    if not torch.cuda.is_available():
        out["error"] = "cuda_unavailable"
        Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
        print(json.dumps({k: out[k] for k in out if k != "geometry"}, indent=2))
        return 2

    device = torch.device(args.device)
    thr = int(CROSSING_THRESHOLD_ABS)
    shapes = {n: s for n, s in RUN3_REAL_ARM_SHAPES}
    steps = int(args.steps)
    topk = int(args.topk)

    try:
        try:
            torch.cuda.reset_peak_memory_stats(0)
        except RuntimeError:
            pass

        acc = make_zero_arms(device=device, dtype=torch.int16)
        ep = make_zero_arms(device=device, dtype=torch.int32)
        q_cpu = {n: torch.randint(-1, 2, s, dtype=torch.int8) for n, s in shapes.items()}
        q_shadow = {n: t.to(device) for n, t in q_cpu.items()}
        life = DeviceLifecycleStore.from_arm_shapes(
            shapes, steps=steps, device=device, follow_up_horizon=int(args.horizon)
        )

        cand, applied, n_cand, n_app, _ord = one_arm_favoring_applied(
            acc, thr=thr, topk=topk
        )
        cuda_sync()

        def phase_select():
            nonlocal cand, applied, n_cand, n_app
            cand, applied, n_cand, n_app, _o = select_topk_masks_deterministic(
                acc, topk=topk, threshold=thr
            )

        sel_stats = time_phase(
            phase_select, warmup=args.warmup, repeats=args.repeats
        )

        residual_zero = split_residual_zero_on_applied(applied)
        ep_before, ep_after = episode_rollover_pair(
            applied, device=device, before_val=1, after_val=steps
        )

        def phase_process():
            life.process_pre_writeback(
                candidate_masks=cand,
                applied_masks=applied,
                step=5,
                n_candidates=n_cand,
                n_applied=n_app,
            )

        process_stats = time_phase(
            phase_process, warmup=args.warmup, repeats=args.repeats
        )

        # Representative close: reseed open_applied outside timed region each sample.
        def setup_close():
            seed_open_applied_for_close(life, applied, first_step=1, after_step=0)

        def phase_close():
            life.close_before_writeback_resets(
                applied_masks=applied,
                step=5,
                residual_zero=residual_zero,
            )

        close_stats = time_phase_with_setup(
            setup_close,
            phase_close,
            warmup=args.warmup,
            repeats=args.repeats,
        )

        # Capture fixture invariants on one fresh seeded close (assert-boundary).
        seed_open_applied_for_close(life, applied, first_step=1, after_step=0)
        first_before = {n: life.first_deferral_step[n].clone() for n in applied}
        agg_before = life.aggregates_t.clone()
        life.close_before_writeback_resets(
            applied_masks=applied, step=5, residual_zero=residual_zero
        )
        close_inv = compute_close_fixture_invariants(
            life,
            applied=applied,
            residual_zero=residual_zero,
            aggregates_before=agg_before,
            first_before=first_before,
        )
        if not close_inv["representative"]:
            raise RuntimeError(f"close fixture not representative: {close_inv}")

        def phase_roll():
            life.roll_tracker_after_writeback(
                applied_masks=applied,
                episode_start_before=ep_before,
                episode_start_after=ep_after,
                step=5,
            )

        roll_stats = time_phase(phase_roll, warmup=args.warmup, repeats=args.repeats)

        life_seq = DeviceLifecycleStore.from_arm_shapes(
            shapes, steps=steps, device=device, follow_up_horizon=int(args.horizon)
        )

        def setup_full():
            seed_open_applied_for_full_step(
                life_seq, applied, first_step=1, after_step=1
            )

        def phase_lifecycle_full_step():
            run_full_per_step_lifecycle(
                life_seq,
                candidate_masks=cand,
                applied_masks=applied,
                episode_start_before=ep_before,
                episode_start_after=ep_after,
                step=5,
                n_candidates=n_cand,
                n_applied=n_app,
                residual_zero=residual_zero,
            )

        life_full_stats = time_phase_with_setup(
            setup_full,
            phase_lifecycle_full_step,
            warmup=args.warmup,
            repeats=args.repeats,
        )

        grads = {
            n: torch.randn(s, dtype=torch.float32, device=device) for n, s in shapes.items()
        }

        def make_update(mode: str):
            def _fn():
                project_and_update_acc_episode(
                    grads=grads,
                    q_auth_cpu=q_cpu,
                    q_shadow=q_shadow,
                    acc=acc,
                    episode_start=ep,
                    step=1,
                    q_shadow_mode=mode,
                )

            return _fn

        for n in q_shadow:
            q_shadow[n].copy_(q_cpu[n].to(device))
        cuda_sync()
        full_stats = time_phase(
            make_update("full_h2d"), warmup=args.warmup, repeats=args.repeats
        )
        for n in q_shadow:
            q_shadow[n].copy_(q_cpu[n].to(device))
        cuda_sync()
        idx_stats = time_phase(
            make_update("index_only"), warmup=args.warmup, repeats=args.repeats
        )

        # Writeback one-arm
        wb_one_holder: dict = {}

        def phase_wb_one():
            wb_one_holder["stats"] = writeback_bridge_cpu_q(
                acc=acc,
                episode_start=ep,
                q_auth_cpu=q_cpu,
                q_shadow=q_shadow,
                applied_masks=applied,
                step=1,
                refresh_shadow_index_only=True,
            )

        wb_one_t = time_phase(phase_wb_one, warmup=args.warmup, repeats=args.repeats)
        wb_one_sync = writeback_distribution_inventory(
            applied, wb_one_holder.get("stats", {})
        )
        wb_one_sync["median_ms"] = wb_one_t["median_ms"]
        wb_one_sync["samples"] = wb_one_t

        # Legal combined cross-arm: n_applied=topk distributed across ALL arms.
        acc_x = make_zero_arms(device=device, dtype=torch.int16)
        _cx, applied_x, n_cand_x, n_app_x, _ox = cross_arm_full_topk_applied(
            acc_x, thr=thr, topk=topk
        )
        if int(n_app_x) != topk:
            raise RuntimeError(f"cross-arm n_applied={n_app_x} != topk={topk}")
        arms_x = sum(1 for m in applied_x.values() if bool(m.any()))
        if arms_x != len(shapes):
            raise RuntimeError(f"cross-arm arms_hit={arms_x} != {len(shapes)}")

        acc_wx = make_zero_arms(device=device, dtype=torch.int16)
        for _n, t in acc_wx.items():
            t.random_(-20, 21)
        ep_wx = make_zero_arms(device=device, dtype=torch.int32)
        q_cpu_x = {n: torch.randint(-1, 2, s, dtype=torch.int8) for n, s in shapes.items()}
        q_shadow_x = {n: t.to(device) for n, t in q_cpu_x.items()}
        cuda_sync()
        wb_x_holder: dict = {}

        def phase_wb_cross():
            wb_x_holder["stats"] = writeback_bridge_cpu_q(
                acc=acc_wx,
                episode_start=ep_wx,
                q_auth_cpu=q_cpu_x,
                q_shadow=q_shadow_x,
                applied_masks=applied_x,
                step=1,
                refresh_shadow_index_only=True,
            )

        wb_x_t = time_phase(phase_wb_cross, warmup=args.warmup, repeats=args.repeats)
        wb_x_sync = writeback_distribution_inventory(
            applied_x, wb_x_holder.get("stats", {})
        )
        wb_x_sync["median_ms"] = wb_x_t["median_ms"]
        wb_x_sync["samples"] = wb_x_t
        wb_x_sync["n_candidates"] = int(n_cand_x)

        wb_proj = writeback_projected_ms(wb_one_t["median_ms"], wb_x_t["median_ms"])

        # Finalize fixed cost
        fin_samples: list[float] = []
        for i in range(int(args.warmup) + int(args.repeats)):
            life_f = DeviceLifecycleStore.from_arm_shapes(
                shapes, steps=steps, device=device, follow_up_horizon=int(args.horizon)
            )
            for step in range(1, 6):
                life_f.process_pre_writeback(
                    candidate_masks=cand,
                    applied_masks=applied,
                    step=step,
                    n_candidates=n_cand,
                    n_applied=n_app,
                )
            cuda_sync()
            if i < int(args.warmup):
                life_f.finalize_window(final_step=steps)
                cuda_sync()
                continue
            t0 = time.perf_counter()
            life_f.finalize_window(final_step=steps)
            cuda_sync()
            fin_samples.append((time.perf_counter() - t0) * 1000.0)
        fin_stats = {
            "samples_ms": fin_samples,
            "median_ms": median_ms(fin_samples),
            "n": len(fin_samples),
        }

        try:
            peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
        except RuntimeError:
            peak_mb = torch.cuda.memory_allocated() / (1024**2)

        life_proj = lifecycle_projected_ms(
            process_ms=process_stats["median_ms"],
            close_ms=close_stats["median_ms"],
            roll_ms=roll_stats["median_ms"],
            full_sequence_ms=life_full_stats["median_ms"],
        )
        windows = project_observer_windows(
            select_ms=sel_stats["median_ms"],
            lifecycle_ms=life_proj["lifecycle_projected"],
            update_full_ms=full_stats["median_ms"],
            update_index_ms=idx_stats["median_ms"],
            writeback_ms=float(wb_proj["writeback_projection_median_ms"]),
            finalize_ms=fin_stats["median_ms"],
            steps=steps,
        )

        publish_ok = (
            int(wb_one_sync["scalar_item_publishes"]) <= 2
            and int(wb_x_sync["scalar_item_publishes"]) <= 2
            and int(wb_one_sync["idx_d2h_count"]) == 1
            and int(wb_one_sync["dir_d2h_count"]) == 1
            and int(wb_x_sync["idx_d2h_count"]) == 1
            and int(wb_x_sync["dir_d2h_count"]) == 1
            and int(wb_x_sync["n_applied"]) == topk
            and int(wb_x_sync["n_arms_hit_from_masks"]) == len(shapes)
            and bool(close_inv["representative"])
        )
        decision = decide_stop_go(
            window_full_ms=windows["window_full_ms"],
            window_index_ms=windows["window_index_ms"],
            need_s=NEED_S,
            r6_status=str(args.r6_status),
            full_mask_transfer=False,
            publish_ok=publish_ok,
        )

        sync_inventory = {
            "selection_n_applied_item": 1,
            "writeback_batched_global_d2h": True,
            "writeback_one_arm": wb_one_sync,
            "writeback_cross_arm_full_topk": wb_x_sync,
            "writeback_projection_uses": wb_proj["writeback_projection_uses"],
            "writeback_projection_median_ms": wb_proj["writeback_projection_median_ms"],
            "full_mask_transfer": False,
            "duplicate_idx_d2h_for_shadow": False,
        }

        phase_ms_median = {
            "select_topk": sel_stats["median_ms"],
            "lifecycle_process_pre": process_stats["median_ms"],
            "lifecycle_close_before_writeback_resets": close_stats["median_ms"],
            "lifecycle_roll_tracker_after_writeback": roll_stats["median_ms"],
            "lifecycle_full_per_step_sequence": life_proj["lifecycle_full_per_step_sequence"],
            "lifecycle_sum_of_phase_medians": life_proj["lifecycle_sum_of_phase_medians"],
            "lifecycle_projected": life_proj["lifecycle_projected"],
            "update_full_h2d": full_stats["median_ms"],
            "update_index_only": idx_stats["median_ms"],
            "writeback_bridge_one_arm": wb_one_t["median_ms"],
            "writeback_bridge_cross_arm_full_topk": wb_x_t["median_ms"],
            "writeback_bridge_projected": float(wb_proj["writeback_projection_median_ms"]),
            "finalize_window_fixed": fin_stats["median_ms"],
        }
        phase_samples = {
            "select_topk": sel_stats,
            "lifecycle_process_pre": process_stats,
            "lifecycle_close_before_writeback_resets": close_stats,
            "lifecycle_roll_tracker_after_writeback": roll_stats,
            "lifecycle_full_per_step_sequence": life_full_stats,
            "update_full_h2d": full_stats,
            "update_index_only": idx_stats,
            "writeback_bridge_one_arm": wb_one_t,
            "writeback_bridge_cross_arm_full_topk": wb_x_t,
            "finalize_window": fin_stats,
        }

        out = assemble_measured_artifact(
            base=out,
            phase_ms_median=phase_ms_median,
            phase_samples=phase_samples,
            windows=windows,
            decision=decision,
            sync_inventory=sync_inventory,
            close_fixture_invariants=close_inv,
            n_candidates_sample=int(n_cand),
            n_applied_sample=int(n_app),
            vram_peak_mb=float(peak_mb),
            publish_ok=publish_ok,
        )
    except Exception as e:
        out["status"] = "INCOMPLETE"
        out["stop_go"] = "INCOMPLETE_BLOCKED"
        out["error"] = f"{type(e).__name__}: {e}"
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
        print(json.dumps({k: out[k] for k in out if k != "geometry"}, indent=2))
        return 2

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    slim = {k: v for k, v in out.items() if k != "geometry"}
    slim["geometry_summary"] = {
        "n_arms": out["geometry"]["n_arms"],
        "total_numel": out["geometry"]["total_numel"],
    }
    print(json.dumps(slim, indent=2))
    return 0 if out["stop_go"] in ("GO", "RED_STOP") else 2


if __name__ == "__main__":
    raise SystemExit(main())
