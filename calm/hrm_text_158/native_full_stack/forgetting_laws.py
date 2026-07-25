"""Live q/acc write-back + arm-local forgetting transforms (PLAN_v9 screen-local).

Extracted behavior-preservingly from forgetting_mechanism_screen_reducers.
"""
from __future__ import annotations

from typing import Mapping

import torch

from calm.hrm_text_158.native_full_stack.two_tier_threshold_semantics import (
    CROSSING_THRESHOLD_ABS,
)

H_TRAJECTORY_EVERY = 25


def threshold_residual_writeback(
    acc: torch.Tensor,
    direction: torch.Tensor,
    *,
    threshold: int = CROSSING_THRESHOLD_ABS,
) -> torch.Tensor:
    """acc' := clamp(acc - dir*T, -(T-1), T-1). Never FORCE-zero."""
    t = int(threshold)
    residual = acc.to(torch.int32) - direction.to(torch.int32) * t
    lo = -(t - 1)
    hi = t - 1
    return residual.clamp(lo, hi).to(acc.dtype)


def apply_live_flip_writeback(
    acc: torch.Tensor,
    episode_start: torch.Tensor,
    q_levels: torch.Tensor,
    flip_mask: torch.Tensor,
    *,
    step: int,
    threshold: int = CROSSING_THRESHOLD_ABS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int], int]:
    """Apply live residual write-back + q flip on flip_mask.

    Returns: new_acc, new_episode_start, new_q, lifetimes, n_q_transitions
    """
    if acc.shape != episode_start.shape or acc.shape != q_levels.shape or acc.shape != flip_mask.shape:
        raise ValueError("acc/episode_start/q/flip_mask shapes must match")
    lifetimes: list[int] = []
    n_q_transitions = 0
    if not bool(flip_mask.any()):
        return acc, episode_start, q_levels, lifetimes, 0

    new_acc = acc.clone()
    new_ep = episode_start.clone()
    new_q = q_levels.clone()

    # Direction from pre-flip acc sign (crossing direction)
    dir_ = torch.where(acc >= 0, torch.ones_like(acc), -torch.ones_like(acc)).to(torch.int8)
    # Only applied indices
    applied = flip_mask
    # Record lifetimes for active episodes before restart
    starts = episode_start[applied]
    active = starts > 0
    if bool(active.any()):
        ages = (int(step) - starts[active].to(torch.int64)).tolist()
        lifetimes.extend(int(x) for x in ages)

    dirs = dir_[applied]
    residual = threshold_residual_writeback(acc[applied], dirs, threshold=threshold)
    new_acc[applied] = residual

    q_before = new_q[applied].to(torch.int16)
    q_after = (q_before + dirs.to(torch.int16)).clamp(-1, 1).to(torch.int8)
    n_q_transitions = int((q_after.to(torch.int16) != q_before).sum().item())
    new_q[applied] = q_after

    # Residual-aware episode restart
    zero_res = residual == 0
    # clear where residual==0
    ep_vals = new_ep[applied]
    ep_vals = torch.where(zero_res, torch.zeros_like(ep_vals), torch.full_like(ep_vals, int(step)))
    new_ep[applied] = ep_vals

    return new_acc, new_ep, new_q, lifetimes, n_q_transitions


# --------------------------------------------------------------------------- #
# Arm-local forgetting transforms (screen-local; not shipped)
# --------------------------------------------------------------------------- #


def apply_decay_leak(acc: torch.Tensor, *, lam: float = 1.0 / 32.0) -> torch.Tensor:
    """trunc_toward_zero(acc * (1-lam))."""
    scaled = acc.to(torch.float32) * (1.0 - float(lam))
    # trunc toward zero
    out = torch.trunc(scaled)
    return out.to(acc.dtype)


def apply_ttl_age_drain(
    acc: torch.Tensor,
    episode_start: torch.Tensor,
    *,
    step: int,
    ttl: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Force-zero any active episode with age > T."""
    age = int(step) - episode_start
    old = (episode_start > 0) & (age > int(ttl))
    new_acc = acc.clone()
    new_ep = episode_start.clone()
    new_acc[old] = 0
    new_ep[old] = 0
    return new_acc, new_ep


def apply_ttl_age_drain_with_count(
    acc: torch.Tensor,
    episode_start: torch.Tensor,
    *,
    step: int,
    ttl: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Observation wrapper: call unchanged apply_ttl_age_drain; return drained_count.

    Count is derived from the primitive's OUTPUTS (not a recomputed drain predicate):
    an active episode (episode_start > 0) whose new_ep is 0 was force-zeroed by the
    primitive — the only zeroing path in apply_ttl_age_drain. Law outputs remain
    byte-identical; this does not alter transfer-law semantics.
    """
    new_acc, new_ep = apply_ttl_age_drain(
        acc, episode_start, step=step, ttl=ttl
    )
    drained_count = int(((episode_start > 0) & (new_ep == 0)).sum().item())
    return new_acc, new_ep, drained_count


def apply_sparse_hot(
    arms_acc: Mapping[str, torch.Tensor],
    *,
    hot_h: int = 8192,
) -> dict[str, torch.Tensor]:
    """Retain global top-H |acc|; zero cold."""
    flat = []
    meta = []
    for n, a in arms_acc.items():
        flat.append(a.abs().flatten())
        meta.append((n, a.numel(), a.shape))
    allabs = torch.cat(flat)
    k = min(int(hot_h), int(allabs.numel()))
    if k <= 0:
        return {n: torch.zeros_like(a) for n, a in arms_acc.items()}
    thresh = torch.topk(allabs, k).values.min()
    keep = allabs >= thresh
    # if ties make >H, keep arbitrary first H
    if int(keep.sum()) > k:
        idx = torch.nonzero(keep, as_tuple=False).flatten()
        drop = idx[k:]
        keep[drop] = False
    out: dict[str, torch.Tensor] = {}
    off = 0
    for n, nn, shape in meta:
        mask = keep[off : off + nn].view(shape)
        out[n] = torch.where(mask, arms_acc[n], torch.zeros_like(arms_acc[n]))
        off += nn
    return out



def entropy_bits(acc: torch.Tensor) -> float:
    vals, counts = torch.unique(acc, return_counts=True)
    p = counts.double() / max(1, int(acc.numel()))
    return float(-(p * p.log2()).sum())


def should_record_h_trajectory(step: int, total_steps: int, *, every: int = H_TRAJECTORY_EVERY) -> bool:
    """True at every `every` steps and at final (PLAN_v9 trajectory)."""
    s = int(step)
    t = int(total_steps)
    e = int(every)
    if s <= 0 or t <= 0:
        return False
    return s == t or (e > 0 and s % e == 0)
