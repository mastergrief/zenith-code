"""Authoritative per-step q/acc/episode train loop (r6c + fork-2 shared GPU path).

ONLY owner of train-state mutation for the forgetting-mechanism screen.
Calls into fixed_qscale_credit / forgetting_laws / phase_probe_sets via
model-runtime loss/credit primitives. Does NOT import JSON-IO / receipt assembly.

Shared A/B path: deterministic GPU (or CPU) selection + update + writeback via
pressure_metric_gpu_loop_bridge. B-only: DeviceLifecycleStore after selection.
Bound by PLAN_v9 + fork-2 PLAN_v2.

Selection identity reuses writeback's host idx payload (no separate idx D2H).
Drained/lifetime/credit host publishes are batched in the bridge helpers.
"""
from __future__ import annotations

import random
import time
from typing import Any, Callable

import torch

from calm.hrm_text_158.native_full_stack.family_classifier import (
    ARM0,
    ARM1,
    ARM2,
    ARM3,
)
from calm.hrm_text_158.native_full_stack.forgetting_laws import (
    apply_decay_leak,
    apply_sparse_hot,
    apply_ttl_age_drain_with_count,
    entropy_bits,
    should_record_h_trajectory,
)
from calm.hrm_text_158.native_full_stack.fixed_qscale_credit import (
    snapshot_route_counters,
)
from calm.hrm_text_158.native_full_stack.forgetting_screen_pre_post_telemetry import (
    PrePostTransformAccumulator,
)
from calm.hrm_text_158.native_full_stack.phase_probe_sets import (
    sample_batch_excluding_acquisition,
)
from calm.hrm_text_158.native_full_stack.pressure_metric_gpu_loop_bridge import (
    apply_drained_flip_counts,
    hotpath_sync_inventory_from_writeback,
    init_gpu_loop_residency,
    lifetimes_before_writeback,
    ordered_selection_frame,
    project_credit_shared,
    select_shared,
    writeback_shared,
)
from calm.hrm_text_158.native_full_stack.screen_model_runtime import (
    _loss_and_credit,
)
from calm.hrm_text_158.native_full_stack.two_tier_threshold_semantics import (
    CROSSING_THRESHOLD_ABS,
)

CLIP = 127
TOPK_PER_STEP = 1024


def _run_phase(timer: Any | None, name: str, fn: Callable[[], Any]) -> Any:
    """Direct call when timer is None (true OFF no-op); timed wrapper only if provided."""
    if timer is None:
        return fn()
    with timer.time(name):
        return fn()


def run_train_loop(
    *,
    m,
    tok,
    eligible: list[str],
    q_levels: dict[str, torch.Tensor],
    pool: list,
    acq_set: set,
    arm: str,
    steps: int,
    batch: int,
    topk: int,
    max_seq_len: int,
    device: str,
    correctness_smoke: bool = False,
    pressure_telemetry: Any | None = None,
    phase_timer: Any | None = None,
    pre_post_telemetry: bool = True,
) -> dict[str, Any]:
    """Mutate q/acc/episode for `steps`; return telemetry + final state tensors.

    Shared A/B: residency + deterministic selection/update/writeback.
    If `pressure_telemetry` is set, run event lifecycle around pre-writeback masks.
    `phase_timer` is diagnostic-only (default None/OFF = true no-op seams).
    `pre_post_telemetry` gates ARM1 PrePostTransformAccumulator (default ON).
    """
    residency = init_gpu_loop_residency(q_levels, device=device)
    _timer = phase_timer

    lifetimes: list[int] = []
    credited_mass = 0
    n_flips = 0
    q_changed_count = 0
    n_applied_drains = 0
    n_ttl_force_zero_drains = 0
    excluded_hit_count = 0
    H_trajectory: list[dict[str, Any]] = []
    margin_traj: list[dict[str, Any]] = []
    episode_traj: list[dict[str, Any]] = []
    selection_frames: list[bytes] = []
    sync_inventory_steps: list[dict[str, Any]] = []
    pre_post_on = bool(pre_post_telemetry)
    pre_post_acc: PrePostTransformAccumulator | None = (
        PrePostTransformAccumulator(device=device)
        if arm == ARM1 and pre_post_on
        else None
    )

    t0 = time.time()
    last_store = None
    for step in range(1, int(steps) + 1):
        rng = random.Random(1000 + step)
        batch_tuples, n_excl = sample_batch_excluding_acquisition(
            pool,
            batch=int(batch),
            rng=rng,
            acquisition_set=acq_set,
        )
        excluded_hit_count += int(n_excl)
        batch_rows = [
            {"rung": sr, "question": q, "expected": e} for q, e, sr in batch_tuples
        ]
        _loss, credit_grads, store = _loss_and_credit(
            m, tok, batch_rows, max_seq_len=max_seq_len, device=device, eligible=eligible
        )
        last_store = store

        _moves, credited = project_credit_shared(
            residency,
            credit_grads=credit_grads,
            eligible=eligible,
            step=step,
        )
        credited_mass += int(credited)

        acc_pre_decay = None
        if arm == ARM1 and pre_post_acc is not None:
            # Shallow name→tensor map of post-projection / pre-decay state (no cat).
            acc_pre_decay = residency.acc

        if arm == ARM1:
            residency.acc = {n: apply_decay_leak(a) for n, a in residency.acc.items()}
        elif arm == ARM2:
            for n in list(residency.acc.keys()):
                (
                    residency.acc[n],
                    residency.episode_start[n],
                    n_ttl,
                ) = apply_ttl_age_drain_with_count(
                    residency.acc[n], residency.episode_start[n], step=step, ttl=32
                )
                n_ttl_force_zero_drains += int(n_ttl)
        elif arm == ARM3:
            residency.acc = apply_sparse_hot(residency.acc, hot_h=8192)
        elif arm != ARM0:
            raise SystemExit(f"unknown --arm {arm}")

        cand_masks, masks, n_cand, n_app, ordered = select_shared(
            residency, topk=int(topk), threshold=CROSSING_THRESHOLD_ABS
        )

        if arm == ARM1 and pre_post_acc is not None and acc_pre_decay is not None:
            pre_post_acc.accumulate_step(
                moves=_moves,
                acc_pre_decay=acc_pre_decay,
                acc_post_decay=residency.acc,
                n_cand_after_decay=int(n_cand),
            )

        if pressure_telemetry is not None:
            from calm.hrm_text_158.native_full_stack.pressure_metric_telemetry import (
                active_episode_stats,
                assert_two_tier_threshold_pass,
                global_margin_quantiles,
            )

            pressure_telemetry.two_tier_threshold_assert_pass = (
                assert_two_tier_threshold_pass(
                    effective=CROSSING_THRESHOLD_ABS,
                    canonical=CROSSING_THRESHOLD_ABS,
                )
            )

            def _process_pre() -> None:
                pressure_telemetry.process_pre_writeback(
                    candidate_masks=cand_masks,
                    applied_masks=masks,
                    step=step,
                    n_candidates=n_cand,
                    n_applied=n_app,
                )

            _run_phase(_timer, "process_pre", _process_pre)
            if step % 25 == 0 or step == int(steps):

                def _publish() -> None:
                    margin_traj.append(
                        {
                            "step": int(step),
                            "residual_margin_pre_cap_crossers": global_margin_quantiles(
                                residency.acc,
                                cand_masks,
                                threshold=CROSSING_THRESHOLD_ABS,
                            ),
                            "residual_margin_applied_topk": global_margin_quantiles(
                                residency.acc,
                                masks,
                                threshold=CROSSING_THRESHOLD_ABS,
                            ),
                            "n_candidates": int(n_cand),
                            "n_applied": int(n_app),
                        }
                    )
                    ep_stats = active_episode_stats(
                        residency.acc, residency.episode_start, step=step
                    )
                    episode_traj.append({"step": int(step), **ep_stats})

                _run_phase(_timer, "publish", _publish)

        # B-only: full episode_start clone before writeback mutation.
        if pressure_telemetry is not None:

            def _episode_snapshot() -> dict[str, torch.Tensor]:
                return {
                    n: residency.episode_start[n].clone()
                    for n in residency.episode_start
                }

            ep_before = _run_phase(_timer, "episode_snapshot", _episode_snapshot)
        else:
            ep_before = None

        if pressure_telemetry is not None:

            def _close_before() -> None:
                # Branch-A F1: fused close no longer needs residual_zero split;
                # skip predict_residual_zero (was inside this timed phase).
                pressure_telemetry.close_before_writeback_resets(
                    applied_masks=masks,
                    step=step,
                    residual_zero=None,
                )

            _run_phase(_timer, "close_before", _close_before)

        step_lifetimes = lifetimes_before_writeback(
            residency, masks, step=step
        )
        lifetimes.extend(step_lifetimes)

        drained = apply_drained_flip_counts(residency, masks)
        n_applied_drains += drained
        n_flips += drained

        # Writeback owns the sole idx D2H; identity frames reuse selection_idx_host.
        wb = writeback_shared(
            residency,
            masks,
            step=step,
            threshold=CROSSING_THRESHOLD_ABS,
            ordered_flat_idx=ordered,
        )
        q_changed_count += int(wb.get("n_q_transitions_total", 0) or 0)
        selection_frames.append(
            ordered_selection_frame(
                step=step,
                ordered_flat_idx=wb.get(
                    "selection_idx_host", torch.empty(0, dtype=torch.int64)
                ),
            )
        )
        sync_inventory_steps.append(
            {"step": int(step), **hotpath_sync_inventory_from_writeback(wb)}
        )

        if pressure_telemetry is not None and ep_before is not None:

            def _roll() -> None:
                pressure_telemetry.roll_tracker_after_writeback(
                    applied_masks=masks,
                    episode_start_before=ep_before,
                    episode_start_after=residency.episode_start,
                    step=step,
                )

            _run_phase(_timer, "roll", _roll)

        if should_record_h_trajectory(step, int(steps)):
            H_now = entropy_bits(
                torch.cat([a.flatten() for a in residency.acc.values()])
            )
            H_trajectory.append(
                {
                    "step": int(step),
                    "H_bits_per_weight": float(H_now),
                    "support": "pooled_named_parameter_acc_flatten",
                    "denominator": "acc.numel()",
                    "estimator": "shannon_unique_counts",
                }
            )

        if step % 10 == 0 or step == int(steps) or correctness_smoke:
            print(
                f"[forget-mech] step {step:4d} ({time.time()-t0:.1f}s) "
                f"flips={n_flips} q_changed={q_changed_count} "
                f"credited_mass={credited_mass} excl_hits={excluded_hit_count}",
                flush=True,
            )

    if pressure_telemetry is not None:

        def _finalize() -> None:
            pressure_telemetry.finalize_window(final_step=int(steps))

        _run_phase(_timer, "finalize", _finalize)

    if last_store is None:
        train_route_counters = {
            "n_fixed_qscale_forwards": 0,
            "n_bitlinear_dynamic_forwards": -1,
            "n_eligible_keys": len(eligible),
            "n_credit_grads_present": 0,
        }
    else:
        train_route_counters = snapshot_route_counters(last_store)
        train_route_counters["n_eligible_keys"] = len(eligible)

    out = {
        "acc": residency.acc,
        "episode_start": residency.episode_start,
        "flip_count": residency.flip_count,
        "q_levels": residency.q_auth_cpu,
        "lifetimes": lifetimes,
        "credited_mass": credited_mass,
        "n_flips": n_flips,
        "q_changed_count": q_changed_count,
        "n_applied_drains": n_applied_drains,
        "n_ttl_force_zero_drains": n_ttl_force_zero_drains,
        "excluded_hit_count": excluded_hit_count,
        "H_trajectory": H_trajectory,
        "train_route_counters": train_route_counters,
        "selection_frames": selection_frames,
        "sync_inventory_steps": sync_inventory_steps,
    }
    if pre_post_acc is not None:
        out["pre_post_transform"] = pre_post_acc.finalize()
    if pressure_telemetry is not None:
        out["pressure_telemetry"] = pressure_telemetry
        out["margin_trajectory"] = margin_traj
        out["episode_trajectory"] = episode_traj
    return out
