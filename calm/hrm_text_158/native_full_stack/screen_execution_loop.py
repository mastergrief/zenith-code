"""Authoritative per-step q/acc/episode train loop (r6c).

ONLY owner of train-state mutation for the forgetting-mechanism screen.
Calls into fixed_qscale_credit / forgetting_laws / phase_probe_sets via
model-runtime loss/credit primitives. Does NOT import JSON-IO / receipt assembly.

Bound by PLAN_v9 sha 07a02aff… + authority 1784812148229.
"""
from __future__ import annotations

import random
import time
from typing import Any

import torch

from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    project_s1_gradient_to_moves,
)
from calm.hrm_text_158.native_full_stack.family_classifier import (
    ARM0,
    ARM1,
    ARM2,
    ARM3,
)
from calm.hrm_text_158.native_full_stack.forgetting_laws import (
    apply_decay_leak,
    apply_live_flip_writeback,
    apply_sparse_hot,
    apply_ttl_age_drain,
    entropy_bits,
    should_record_h_trajectory,
)
from calm.hrm_text_158.native_full_stack.fixed_qscale_credit import (
    snapshot_route_counters,
)
from calm.hrm_text_158.native_full_stack.phase_probe_sets import (
    sample_batch_excluding_acquisition,
)
from calm.hrm_text_158.native_full_stack.screen_model_runtime import (
    _loss_and_credit,
)
from calm.hrm_text_158.native_full_stack.two_tier_threshold_semantics import (
    CROSSING_THRESHOLD_ABS,
)
from calm.hrm_text_158.native_full_stack.vote_lifetime_screen_reducers import (
    update_episode_starts,
)

CLIP = 127
TOPK_PER_STEP = 1024


def _arm_topk_masks(
    arms_acc: dict[str, torch.Tensor], *, topk: int
) -> dict[str, torch.Tensor]:
    """Applied topK masks — selection bit-identical to pressure_metric_telemetry helper."""
    from calm.hrm_text_158.native_full_stack.pressure_metric_telemetry import (
        compute_topk_masks_and_counts,
    )

    _cand, applied, _nc, _na = compute_topk_masks_and_counts(arms_acc, topk=int(topk))
    return applied


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
) -> dict[str, Any]:
    """Mutate q/acc/episode for `steps`; return telemetry + final state tensors.

    If `pressure_telemetry` is set (PressureTelemetryStore), run PLAN_v6 event
    lifecycle stages 1-6 around pre-writeback masks. Uninstrumented path
    (pressure_telemetry is None) is unchanged aside from shared mask helper.
    """
    acc = {n: torch.zeros_like(q, dtype=torch.int16) for n, q in q_levels.items()}
    episode_start = {
        n: torch.zeros_like(q, dtype=torch.int32) for n, q in q_levels.items()
    }
    flip_count = {
        n: torch.zeros_like(q, dtype=torch.int32) for n, q in q_levels.items()
    }

    lifetimes: list[int] = []
    credited_mass = 0
    n_flips = 0
    q_changed_count = 0
    n_applied_drains = 0
    excluded_hit_count = 0
    H_trajectory: list[dict[str, Any]] = []
    margin_traj: list[dict[str, Any]] = []
    episode_traj: list[dict[str, Any]] = []

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

        moves = {}
        for n in eligible:
            g_cpu = credit_grads[n].detach().cpu()
            q_cpu = q_levels[n]
            if g_cpu.shape != q_cpu.shape:
                raise RuntimeError(
                    f"credit/q shape mismatch for {n}: {tuple(g_cpu.shape)} vs "
                    f"{tuple(q_cpu.shape)}"
                )
            moves[n] = project_s1_gradient_to_moves(g_cpu, q_cpu)
        credited_mass += int(sum(int(mv.abs().sum().item()) for mv in moves.values()))

        for n, mv in moves.items():
            prev = acc[n]
            new = (
                (prev.to(torch.int32) + mv.to(torch.int32))
                .clamp(-CLIP, CLIP)
                .to(torch.int16)
            )
            episode_start[n] = update_episode_starts(prev, new, episode_start[n], step)
            acc[n] = new

        if arm == ARM1:
            acc = {n: apply_decay_leak(a) for n, a in acc.items()}
        elif arm == ARM2:
            for n in list(acc.keys()):
                acc[n], episode_start[n] = apply_ttl_age_drain(
                    acc[n], episode_start[n], step=step, ttl=32
                )
        elif arm == ARM3:
            acc = apply_sparse_hot(acc, hot_h=8192)
        elif arm != ARM0:
            raise SystemExit(f"unknown --arm {arm}")

        # Stage 1: pre-writeback candidate + applied masks
        if pressure_telemetry is not None:
            from calm.hrm_text_158.native_full_stack.pressure_metric_telemetry import (
                active_episode_stats,
                assert_two_tier_threshold_pass,
                compute_topk_masks_and_counts,
                global_margin_quantiles,
            )

            cand_masks, masks, n_cand, n_app = compute_topk_masks_and_counts(
                acc, topk=int(topk)
            )
            pressure_telemetry.two_tier_threshold_assert_pass = assert_two_tier_threshold_pass(
                effective=CROSSING_THRESHOLD_ABS,
                canonical=CROSSING_THRESHOLD_ABS,
            )
            # Stages 2-4
            pressure_telemetry.process_pre_writeback(
                candidate_masks=cand_masks,
                applied_masks=masks,
                step=step,
                n_candidates=n_cand,
                n_applied=n_app,
            )
            if step % 25 == 0 or step == int(steps):
                # Dual-population GLOBAL margins on CPU acc (PLAN_v6; never n_parts)
                margin_traj.append(
                    {
                        "step": int(step),
                        "residual_margin_pre_cap_crossers": global_margin_quantiles(
                            acc, cand_masks, threshold=CROSSING_THRESHOLD_ABS
                        ),
                        "residual_margin_applied_topk": global_margin_quantiles(
                            acc, masks, threshold=CROSSING_THRESHOLD_ABS
                        ),
                        "n_candidates": int(n_cand),
                        "n_applied": int(n_app),
                    }
                )
                # Episode telemetry from existing CPU acc/episode_start seam
                ep_stats = active_episode_stats(acc, episode_start, step=step)
                episode_traj.append({"step": int(step), **ep_stats})
        else:
            masks = _arm_topk_masks(acc, topk=int(topk))

        # Snapshot episode_start before writeback for stage-6 rollover detect
        ep_before = (
            {n: episode_start[n].clone() for n in episode_start}
            if pressure_telemetry is not None
            else None
        )

        # Close open events on applied indices BEFORE writeback mutates episodes
        if pressure_telemetry is not None:
            # Predict residual-zero from current acc + applied (same law as writeback)
            from calm.hrm_text_158.native_full_stack.forgetting_laws import (
                threshold_residual_writeback,
            )

            residual_zero = {}
            for n in masks:
                if not bool(masks[n].any()):
                    residual_zero[n] = torch.zeros_like(masks[n], dtype=torch.bool)
                    continue
                dir_ = torch.where(
                    acc[n] >= 0, torch.ones_like(acc[n]), -torch.ones_like(acc[n])
                ).to(torch.int8)
                res = threshold_residual_writeback(
                    acc[n][masks[n]], dir_[masks[n]], threshold=CROSSING_THRESHOLD_ABS
                )
                rz = torch.zeros_like(masks[n], dtype=torch.bool)
                rz[masks[n]] = res == 0
                residual_zero[n] = rz
            pressure_telemetry.close_before_writeback_resets(
                applied_masks=masks,
                step=step,
                residual_zero=residual_zero,
            )

        for n in list(acc.keys()):
            drained = int(masks[n].sum().item())
            n_applied_drains += drained
            # Stage 5: writeback (may restart/clear episodes)
            new_acc, new_ep, new_q, lt, n_q = apply_live_flip_writeback(
                acc[n], episode_start[n], q_levels[n], masks[n], step=step
            )
            if drained:
                flip_count[n] = flip_count[n] + masks[n].to(torch.int32)
                n_flips += drained
            q_changed_count += int(n_q)
            if lt:
                lifetimes.extend(lt)
            acc[n] = new_acc
            episode_start[n] = new_ep
            q_levels[n] = new_q

        # Stage 6: roll tracker after writeback episode changes
        if pressure_telemetry is not None and ep_before is not None:
            pressure_telemetry.roll_tracker_after_writeback(
                applied_masks=masks,
                episode_start_before=ep_before,
                episode_start_after=episode_start,
                step=step,
            )

        if should_record_h_trajectory(step, int(steps)):
            H_now = entropy_bits(torch.cat([a.flatten() for a in acc.values()]))
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
        pressure_telemetry.finalize_window(final_step=int(steps))

    # Snapshot credit-step route counters BEFORE final probes (probes call
    # begin_credit_step and would overwrite the module-global store).
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
        "acc": acc,
        "episode_start": episode_start,
        "flip_count": flip_count,
        "q_levels": q_levels,
        "lifetimes": lifetimes,
        "credited_mass": credited_mass,
        "n_flips": n_flips,
        "q_changed_count": q_changed_count,
        "n_applied_drains": n_applied_drains,
        "excluded_hit_count": excluded_hit_count,
        "H_trajectory": H_trajectory,
        "train_route_counters": train_route_counters,
    }
    if pressure_telemetry is not None:
        out["pressure_telemetry"] = pressure_telemetry
        out["margin_trajectory"] = margin_traj
        out["episode_trajectory"] = episode_traj
    return out
