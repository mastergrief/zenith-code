#!/usr/bin/env python3
"""Fork-2 integration additive-cost smoke (representative production loop glue).

Prices wiring / publish / receipt_hotpath at real multi-arm geometry on cuda:0.
Wiring measures the ACTUAL promoted glue: project_credit_shared, select,
B-only episode_start clone, residual_zero, lifetimes, drained flips,
writeback (identity via writeback host idx), and persists A/B sync inventories.
CUDA unavailable / timeout / unpriceable → INCOMPLETE/STOP (never GO).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from calm.hrm_text_158.native_full_stack.pressure_metric_fork2_geometry import (
    RUN3_REAL_ARM_SHAPES,
    make_zero_arms,
    run3_total_numel,
)
from calm.hrm_text_158.native_full_stack.pressure_metric_fork2_scale_timing import (
    cuda_sync,
    time_phase,
)
from calm.hrm_text_158.native_full_stack.pressure_metric_gpu_lifecycle_derisk import (
    DeviceLifecycleStore,
)
from calm.hrm_text_158.native_full_stack.pressure_metric_gpu_loop_bridge import (
    apply_drained_flip_counts,
    hotpath_sync_inventory_from_writeback,
    init_gpu_loop_residency,
    lifetimes_before_writeback,
    ordered_selection_frame,
    predict_residual_zero,
    project_credit_shared,
    select_shared,
    writeback_shared,
)
from calm.hrm_text_158.native_full_stack.pressure_metric_gpu_selection_derisk import (
    select_topk_masks_deterministic,
)
from calm.hrm_text_158.native_full_stack.pressure_metric_receipt import (
    build_diagnostic_receipt,
)
from calm.hrm_text_158.native_full_stack.pressure_metric_telemetry import (
    active_episode_stats,
    global_margin_quantiles,
)
from calm.hrm_text_158.native_full_stack.two_tier_threshold_semantics import (
    CROSSING_THRESHOLD_ABS,
)


def _price_line(name: str, median: float | None, *, window_steps: int = 25) -> dict:
    if median is None or not (median == median) or median < 0:
        return {
            "name": name,
            "status": "UNPRICEABLE",
            "median_ms": None,
            "window_ms": None,
        }
    return {
        "name": name,
        "status": "MEASURED",
        "median_ms": float(median),
        "window_ms": float(median) * float(window_steps),
    }


def _seed_residency(res, acc_seed, thr: int, step: int = 5):
    """Copy seeded acc and give applied rows a non-zero episode_start."""
    for n in res.acc:
        res.acc[n].copy_(acc_seed[n])
        res.episode_start[n].zero_()
        # Mark a subset as previously deferred so lifetimes/close are non-empty.
        hit = res.acc[n].abs() >= thr
        res.episode_start[n][hit] = max(1, int(step) - 3)


def _fake_grads(res, thr: int) -> dict[str, torch.Tensor]:
    """Device grads that keep projection non-vacuous without a model forward."""
    out = {}
    for n, a in res.acc.items():
        g = torch.zeros(a.shape, dtype=torch.float32, device=a.device)
        # Push a sparse set of rows across threshold via credit projection bias.
        flat = g.reshape(-1)
        n_hit = min(48, int(flat.numel()))
        flat[:n_hit] = float(thr + 2)
        out[n] = g
    return out


def _run_representative_glue(
    *,
    q_cpu: dict[str, torch.Tensor],
    acc_seed: dict[str, torch.Tensor],
    device: torch.device,
    topk: int,
    thr: int,
    step: int,
    instrumented: bool,
    life: DeviceLifecycleStore | None,
) -> dict:
    """Mirror production loop glue for one step (A=uninstrumented, B=instrumented)."""
    res = init_gpu_loop_residency(q_cpu, device=device)
    _seed_residency(res, acc_seed, thr, step=step)
    grads = _fake_grads(res, thr)
    eligible = list(res.acc.keys())
    _moves, _credited = project_credit_shared(
        res, credit_grads=grads, eligible=eligible, step=step
    )
    cand, applied, n_cand, n_app, ordered = select_shared(
        res, topk=topk, threshold=thr
    )
    inventory: dict = {
        "flavor": "B_instrumented" if instrumented else "A_uninstrumented",
        "n_candidates": int(n_cand),
        "n_applied": int(n_app),
        "phases": [
            "project_credit_shared",
            "select_shared",
            "ordered_selection_frame_via_writeback_host_idx",
            "lifetimes_before_writeback",
            "apply_drained_flip_counts",
            "writeback_shared",
        ],
    }
    if instrumented:
        assert life is not None
        inventory["phases"].extend(
            [
                "episode_start_clone",
                "predict_residual_zero",
                "close_before_writeback_resets",
                "roll_tracker_after_writeback",
            ]
        )
        life.process_pre_writeback(
            candidate_masks=cand,
            applied_masks=applied,
            step=step,
            n_candidates=int(n_cand),
            n_applied=int(n_app),
        )
        ep_before = {
            n: res.episode_start[n].clone() for n in res.episode_start
        }
        residual_zero = predict_residual_zero(res, applied, threshold=thr)
        life.close_before_writeback_resets(
            applied_masks=applied, step=step, residual_zero=residual_zero
        )
    else:
        ep_before = None

    _lt = lifetimes_before_writeback(res, applied, step=step)
    _drained = apply_drained_flip_counts(res, applied)
    wb = writeback_shared(
        res,
        applied,
        step=step,
        threshold=thr,
        ordered_flat_idx=ordered,
    )
    _frame = ordered_selection_frame(
        step=step,
        ordered_flat_idx=wb.get(
            "selection_idx_host", torch.empty(0, dtype=torch.int64)
        ),
    )
    inventory["writeback"] = hotpath_sync_inventory_from_writeback(wb)
    inventory["selection_frame_bytes"] = len(_frame)
    inventory["lifetimes_len"] = len(_lt)
    inventory["drained"] = int(_drained)
    inventory["duplicate_idx_d2h_for_identity"] = bool(
        wb.get("duplicate_idx_d2h_for_identity", False)
    )

    if instrumented and ep_before is not None and life is not None:
        life.roll_tracker_after_writeback(
            applied_masks=applied,
            episode_start_before=ep_before,
            episode_start_after=res.episode_start,
            step=step,
        )
    return inventory


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--topk", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument(
        "--out",
        default="artifacts/acc_entropy/pressure_metric_fork2_integration_additive_costs.json",
    )
    args = ap.parse_args()

    out: dict = {
        "schema": "pressure_metric_fork2_integration_additive_costs/v2",
        "status": "INCOMPLETE",
        "device": str(args.device),
        "geometry": {
            "n_arms": len(RUN3_REAL_ARM_SHAPES),
            "total_numel": run3_total_numel(),
        },
        "lines": [],
        "verdict": "INCOMPLETE",
        "representative": True,
    }

    if not torch.cuda.is_available():
        out["error"] = "cuda_unavailable"
        out["verdict"] = "INCOMPLETE"
        Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
        print(json.dumps(out, indent=2))
        return 2

    device = torch.device(args.device)
    thr = int(CROSSING_THRESHOLD_ABS)
    shapes = {n: s for n, s in RUN3_REAL_ARM_SHAPES}
    steps = int(args.steps)
    topk = int(args.topk)

    try:
        acc = make_zero_arms(device=device, dtype=torch.int16)
        for i, (n, _s) in enumerate(RUN3_REAL_ARM_SHAPES):
            flat = acc[n].reshape(-1)
            n_hit = min(64, int(flat.numel()))
            flat[:n_hit] = thr + 1 + (i % 3)
        ep = make_zero_arms(device=device, dtype=torch.int32)
        q_cpu = {n: torch.randint(-1, 2, s, dtype=torch.int8) for n, s in shapes.items()}
        life = DeviceLifecycleStore.from_arm_shapes(
            shapes, steps=steps, device=device
        )
        cuda_sync()

        cand, applied, n_cand, n_app, _o = select_topk_masks_deterministic(
            acc, topk=topk, threshold=thr
        )

        # --- A wiring (uninstrumented shared path glue) ---
        def phase_wiring_a():
            _run_representative_glue(
                q_cpu=q_cpu,
                acc_seed=acc,
                device=device,
                topk=topk,
                thr=thr,
                step=5,
                instrumented=False,
                life=None,
            )

        wire_a = time_phase(phase_wiring_a, warmup=args.warmup, repeats=args.repeats)

        # --- B wiring (instrumented: + episode clone + residual_zero + lifecycle) ---
        def phase_wiring_b():
            _run_representative_glue(
                q_cpu=q_cpu,
                acc_seed=acc,
                device=device,
                topk=topk,
                thr=thr,
                step=5,
                instrumented=True,
                life=life,
            )

        wire_b = time_phase(phase_wiring_b, warmup=args.warmup, repeats=args.repeats)

        # Persist end-to-end sync inventories (one sample each, outside timer).
        inv_a = _run_representative_glue(
            q_cpu=q_cpu,
            acc_seed=acc,
            device=device,
            topk=topk,
            thr=thr,
            step=7,
            instrumented=False,
            life=None,
        )
        inv_b = _run_representative_glue(
            q_cpu=q_cpu,
            acc_seed=acc,
            device=device,
            topk=topk,
            thr=thr,
            step=7,
            instrumented=True,
            life=life,
        )

        # Named B-only episode_start clone line (isolated).
        res_clone = init_gpu_loop_residency(q_cpu, device=device)
        _seed_residency(res_clone, acc, thr, step=5)

        def phase_ep_clone():
            _ = {n: res_clone.episode_start[n].clone() for n in res_clone.episode_start}

        ep_clone_stats = time_phase(
            phase_ep_clone, warmup=args.warmup, repeats=args.repeats
        )

        # --- publish_cadence ---
        def phase_publish():
            _ = global_margin_quantiles(acc, cand, threshold=thr)
            _ = global_margin_quantiles(acc, applied, threshold=thr)
            _ = active_episode_stats(acc, ep, step=25)

        pub_stats = time_phase(
            phase_publish, warmup=args.warmup, repeats=args.repeats
        )

        # --- receipt_hotpath ---
        life.process_pre_writeback(
            candidate_masks=cand,
            applied_masks=applied,
            step=1,
            n_candidates=int(n_cand),
            n_applied=int(n_app),
        )
        life.two_tier_threshold_assert_pass = True
        meas = {
            "n_flips": int(n_app),
            "q_changed_count": 0,
            "credited_mass": 0,
            "lifetime_censored_frac": 0.0,
            "p50_flip_lifetime": None,
            "H_bits_per_weight": 0.0,
            "H_trajectory": [{"step": 25, "H_bits_per_weight": 0.0}],
            "n_applied_drains": int(n_app),
            "margin_trajectory": [],
            "episode_trajectory": [],
        }

        def phase_receipt():
            _ = build_diagnostic_receipt(
                store=life,
                measurements=meas,
                probes={"retention_ok": True, "ret_final_count": 1, "ret_step0_count": 1},
                require_probes=False,
                schema_only=True,
            )

        recv_stats = time_phase(
            phase_receipt, warmup=args.warmup, repeats=args.repeats
        )

        def phase_demand_pub():
            life.per_step_ratios.append(
                {
                    "step": 2,
                    "candidate_crossers_before_cap": int(n_cand),
                    "applied_count": int(n_app),
                    "demand_applied_ratio": float(n_cand) / float(max(1, n_app)),
                    "deferred_count": max(0, int(n_cand) - int(n_app)),
                }
            )
            if len(life.per_step_ratios) > 64:
                del life.per_step_ratios[:-32]

        demand_stats = time_phase(
            phase_demand_pub, warmup=args.warmup, repeats=args.repeats
        )

        # Conservative wiring = max(A, B) so instrumented cost is not under-counted.
        wire_median = max(float(wire_a["median_ms"]), float(wire_b["median_ms"]))
        lines = [
            _price_line("wiring_shell", wire_median, window_steps=steps),
            _price_line(
                "wiring_shell_A_uninstrumented",
                wire_a["median_ms"],
                window_steps=steps,
            ),
            _price_line(
                "wiring_shell_B_instrumented",
                wire_b["median_ms"],
                window_steps=steps,
            ),
            _price_line(
                "episode_start_clone_B_only",
                ep_clone_stats["median_ms"],
                window_steps=steps,
            ),
            _price_line(
                "publish_cadence", pub_stats["median_ms"], window_steps=max(1, steps // 25)
            ),
            _price_line("receipt_hotpath", recv_stats["median_ms"], window_steps=1),
            _price_line(
                "receipt_hotpath_demand_scalar_pub",
                demand_stats["median_ms"],
                window_steps=steps,
            ),
        ]
        rh = next(x for x in lines if x["name"] == "receipt_hotpath")
        dem = next(x for x in lines if x["name"] == "receipt_hotpath_demand_scalar_pub")
        if rh["status"] == "MEASURED" and dem["status"] == "MEASURED":
            rh["window_ms"] = float(rh["median_ms"]) + float(dem["window_ms"])
            rh["includes_demand_scalar_window_ms"] = dem["window_ms"]

        # Budget table uses conservative wiring_shell (max A/B) + publish + receipt.
        budget_names = ("wiring_shell", "publish_cadence", "receipt_hotpath")
        unpriceable = [
            x["name"]
            for x in lines
            if x["name"] in budget_names and x["status"] == "UNPRICEABLE"
        ]
        # Also fail closed if either flavor inventory shows duplicate idx D2H.
        if inv_a.get("duplicate_idx_d2h_for_identity") or inv_b.get(
            "duplicate_idx_d2h_for_identity"
        ):
            unpriceable.append("duplicate_idx_d2h_for_identity")

        out["lines"] = lines
        out["sync_inventory"] = {"A": inv_a, "B": inv_b}
        out["raw"] = {
            "wiring_shell_A": wire_a,
            "wiring_shell_B": wire_b,
            "episode_start_clone_B_only": ep_clone_stats,
            "publish_cadence": pub_stats,
            "receipt_hotpath": recv_stats,
            "receipt_hotpath_demand_scalar_pub": demand_stats,
        }
        out["peak_cuda_bytes"] = int(torch.cuda.max_memory_allocated(device))
        if unpriceable:
            out["status"] = "STOP"
            out["verdict"] = "STOP"
            out["unpriceable"] = unpriceable
            rc = 3
        else:
            out["status"] = "MEASURED"
            out["verdict"] = "PRICED"
            out["additive_window_ms_total"] = float(
                sum(
                    float(x["window_ms"])
                    for x in lines
                    if x["name"] in budget_names
                )
            )
            out["wiring_shell_uses"] = "max(A_uninstrumented, B_instrumented)"
            rc = 0
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
        print(json.dumps({k: out[k] for k in out if k != "raw"}, indent=2))
        return rc
    except Exception as exc:  # noqa: BLE001 — smoke must never claim GO on failure
        out["error"] = f"{type(exc).__name__}: {exc}"
        out["status"] = "INCOMPLETE"
        out["verdict"] = "INCOMPLETE"
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
        print(json.dumps(out, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
