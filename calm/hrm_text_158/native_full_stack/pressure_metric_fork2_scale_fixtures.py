"""Fork-2 scale-smoke fixtures (smoke-only).

Selection distributions, close/reset open-state seeding, and fixture invariants.
"""
from __future__ import annotations

from typing import Any, Mapping

import torch

from calm.hrm_text_158.native_full_stack.pressure_metric_gpu_lifecycle_derisk import (
    DeviceLifecycleStore,
)
from calm.hrm_text_158.native_full_stack.pressure_metric_gpu_selection_derisk import (
    select_topk_masks_deterministic,
)


def one_arm_favoring_applied(
    acc: dict[str, torch.Tensor],
    *,
    thr: int,
    topk: int,
) -> tuple[dict, dict, int, int, torch.Tensor]:
    """Saturated equal-demand → index-asc packs early flat indices (often 1 arm)."""
    for _n, t in acc.items():
        t.fill_(thr)
        t.reshape(-1)[::128] = thr + 5
    return select_topk_masks_deterministic(acc, topk=topk, threshold=thr)


def cross_arm_full_topk_applied(
    acc: dict[str, torch.Tensor],
    *,
    thr: int,
    topk: int = 1024,
) -> tuple[dict, dict, int, int, torch.Tensor]:
    """Exactly `topk` crossers evenly across all arms → n_applied=topk, all arms hit.

    Legal combined worst case: topk=1024 across 32 arms → 32 rows/arm.
    Only boosted rows cross the threshold so selection cannot collapse to early arms.
    """
    names = list(acc.keys())
    n_arms = len(names)
    if n_arms == 0:
        raise ValueError("acc must be non-empty")
    if int(topk) % n_arms != 0:
        raise ValueError(f"topk={topk} must be divisible by n_arms={n_arms}")
    per_arm = int(topk) // n_arms
    for n, t in acc.items():
        t.zero_()
    for n in names:
        flat = acc[n].reshape(-1)
        if flat.numel() < per_arm:
            raise ValueError(f"arm {n} numel {flat.numel()} < per_arm {per_arm}")
        flat[:per_arm] = int(thr) + 5
    cand, applied, n_cand, n_app, ordered = select_topk_masks_deterministic(
        acc, topk=int(topk), threshold=thr
    )
    return cand, applied, n_cand, n_app, ordered


def split_residual_zero_on_applied(
    applied: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Half of applied rows → residual_clear; remainder → residual_restart."""
    residual_zero = {n: torch.zeros_like(m) for n, m in applied.items()}
    for n, m in applied.items():
        if not bool(m.any()):
            continue
        idxs = torch.nonzero(m.reshape(-1), as_tuple=False).flatten()
        residual_zero[n].reshape(-1)[idxs[::2]] = True
    return residual_zero


def episode_rollover_pair(
    applied: Mapping[str, torch.Tensor],
    *,
    device: torch.device | str,
    before_val: int = 1,
    after_val: int = 25,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    ep_before: dict[str, torch.Tensor] = {}
    ep_after: dict[str, torch.Tensor] = {}
    for n, m in applied.items():
        ep_before[n] = torch.zeros_like(m, dtype=torch.int32, device=device)
        ep_after[n] = torch.zeros_like(m, dtype=torch.int32, device=device)
        ep_before[n][m] = int(before_val)
        ep_after[n][m] = int(after_val)
    return ep_before, ep_after


def seed_open_applied_for_close(
    store: DeviceLifecycleStore,
    applied: Mapping[str, torch.Tensor],
    *,
    first_step: int = 1,
    after_step: int = 0,
) -> int:
    """Seed trackers so close_before sees open_applied = applied & (first > 0).

    Returns open_applied_count (host int; assert-boundary only).
    """
    open_count = 0
    fs = int(first_step)
    a_s = int(after_step)
    for n, m in applied.items():
        first = store.first_deferral_step[n]
        after = store.applied_after_deferral_step[n]
        # Clear then seed only applied rows as previously-deferred open events.
        first.zero_()
        after.zero_()
        first[m] = fs
        if a_s != 0:
            after[m] = a_s
        open_count += int((m & (first > 0)).sum().item())
    return open_count


def seed_open_applied_for_full_step(
    store: DeviceLifecycleStore,
    applied: Mapping[str, torch.Tensor],
    *,
    first_step: int = 1,
    after_step: int = 1,
) -> int:
    """Seed so process_pre does not survive-close applied rows (after already set).

    close_before still sees open_applied because first stays > 0.
    """
    return seed_open_applied_for_close(
        store, applied, first_step=first_step, after_step=after_step
    )


def compute_close_fixture_invariants(
    store: DeviceLifecycleStore,
    *,
    applied: Mapping[str, torch.Tensor],
    residual_zero: Mapping[str, torch.Tensor],
    aggregates_before: torch.Tensor,
    first_before: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Prove representative close did real work (call after one seeded close)."""
    open_applied = 0
    clear_n = 0
    restart_n = 0
    cleared_now = 0
    for n, m in applied.items():
        first_b = first_before[n]
        open_m = m & (first_b > 0)
        open_applied += int(open_m.sum().item())
        rz = residual_zero[n].bool()
        clear_n += int((open_m & rz).sum().item())
        restart_n += int((open_m & ~rz).sum().item())
        # After close, those open rows should be cleared on trackers.
        first_a = store.first_deferral_step[n]
        cleared_now += int((open_m & (first_a == 0)).sum().item())
    agg_delta = int((store.aggregates_t - aggregates_before).abs().sum().item())
    return {
        "open_applied_count": open_applied,
        "residual_clear_count": clear_n,
        "residual_restart_count": restart_n,
        "trackers_cleared_on_close": cleared_now,
        "aggregates_mutated": agg_delta > 0,
        "aggregates_abs_delta": agg_delta,
        "representative": (
            open_applied > 0
            and clear_n > 0
            and restart_n > 0
            and cleared_now > 0
            and agg_delta > 0
        ),
    }


def writeback_distribution_inventory(
    applied: Mapping[str, torch.Tensor],
    wb_stats: Mapping[str, Any],
) -> dict[str, Any]:
    arms_hit = sum(1 for m in applied.values() if bool(m.any()))
    n_applied = int(sum(int(m.to(torch.int64).sum().item()) for m in applied.values()))
    return {
        "n_applied": n_applied,
        "n_arms_hit_from_masks": arms_hit,
        "n_arms_with_applied": wb_stats.get("n_arms_with_applied", 0),
        "idx_d2h_count": wb_stats.get("idx_d2h_count", 0),
        "dir_d2h_count": wb_stats.get("dir_d2h_count", 0),
        "idx_d2h_bytes": wb_stats.get("applied_nonzero_d2h_bytes", 0),
        "dir_d2h_bytes": wb_stats.get("dir_d2h_bytes", 0),
        "q_index_h2d_bytes": wb_stats.get("q_index_h2d_bytes", 0),
        "scalar_item_publishes": wb_stats.get("scalar_item_publishes", 0),
        "batched_global_d2h": wb_stats.get("batched_global_d2h", False),
    }
