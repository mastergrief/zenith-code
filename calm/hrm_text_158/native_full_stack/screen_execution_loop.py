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
    flat_abs = []
    shapes = []
    for n, a in arms_acc.items():
        flat_abs.append(a.abs().flatten())
        shapes.append((n, a.numel(), a.shape))
    allabs = torch.cat(flat_abs)
    crosser_idx = torch.nonzero(
        allabs >= CROSSING_THRESHOLD_ABS, as_tuple=False
    ).flatten()
    sel = torch.zeros_like(allabs, dtype=torch.bool)
    if crosser_idx.numel():
        k = min(int(topk), int(crosser_idx.numel()))
        top = crosser_idx[allabs[crosser_idx].argsort(descending=True)[:k]]
        sel[top] = True
    masks: dict[str, torch.Tensor] = {}
    off = 0
    for n, nn, shape in shapes:
        masks[n] = sel[off : off + nn].view(shape)
        off += nn
    return masks


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
) -> dict[str, Any]:
    """Mutate q/acc/episode for `steps`; return telemetry + final state tensors."""
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

        masks = _arm_topk_masks(acc, topk=int(topk))
        for n in list(acc.keys()):
            drained = int(masks[n].sum().item())
            n_applied_drains += drained
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

    return {
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
