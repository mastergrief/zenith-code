"""Device-invariant topK + q-shadow update + writeback bridge (production path).

Registered tiebreak: abs_acc desc, flat index asc (composite int64 key with inversion).
Shared A/B entrypoint used by screen_execution_loop via pressure_metric_gpu_loop_bridge.

Lifecycle reducer + geometry live in sibling modules (one-way imports):
  pressure_metric_fork2_geometry.py
  pressure_metric_gpu_lifecycle_derisk.py
"""
from __future__ import annotations

from typing import Any, Mapping

import torch

from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    project_s1_gradient_to_moves,
)
from calm.hrm_text_158.native_full_stack.forgetting_laws import apply_live_flip_writeback
from calm.hrm_text_158.native_full_stack.pressure_metric_fork2_geometry import (
    CLIP,
    INDEX_BITS,
    INDEX_MASK,
    RUN3_REAL_ARM_SHAPES,
    make_zero_arms,
    run3_total_numel,
)
from calm.hrm_text_158.native_full_stack.pressure_metric_gpu_lifecycle_derisk import (
    AGG_KEYS,
    DeviceLifecycleStore,
    cpu_store_from_shapes,
    run_full_per_step_lifecycle,
)
from calm.hrm_text_158.native_full_stack.two_tier_threshold_semantics import (
    CROSSING_THRESHOLD_ABS,
)
from calm.hrm_text_158.native_full_stack.vote_lifetime_screen_reducers import (
    update_episode_starts,
)

# Re-export packet surface used by tests/scale script.
__all__ = [
    "AGG_KEYS",
    "CLIP",
    "DeviceLifecycleStore",
    "INDEX_BITS",
    "INDEX_MASK",
    "RUN3_REAL_ARM_SHAPES",
    "composite_rank_key",
    "cpu_oracle_project_and_update",
    "cpu_store_from_shapes",
    "make_zero_arms",
    "project_and_update_acc_episode",
    "run3_total_numel",
    "run_full_per_step_lifecycle",
    "select_topk_masks_deterministic",
    "writeback_bridge_cpu_q",
    "writeback_cpu_oracle",
]


def composite_rank_key(abs_acc: torch.Tensor, flat_index: torch.Tensor) -> torch.Tensor:
    """int64 key: larger key sorts earlier under descending argsort.

    Encodes (abs_acc desc, flat_index asc) via
    key = (abs << INDEX_BITS) | (INDEX_MASK - flat_index).
    """
    abs_i = abs_acc.to(torch.int64)
    idx = flat_index.to(torch.int64)
    return (abs_i << INDEX_BITS) | (INDEX_MASK - (idx & INDEX_MASK))


def select_topk_masks_deterministic(
    arms_acc: Mapping[str, torch.Tensor],
    *,
    topk: int,
    threshold: int = CROSSING_THRESHOLD_ABS,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], int, int, torch.Tensor]:
    """Device-follows topK with registered composite-key tiebreak.

    Returns cand_masks, applied_masks, n_candidates, n_applied, ordered_selected_flat_idx.
    """
    flat_abs = []
    shapes: list[tuple[str, int, torch.Size]] = []
    for n, a in arms_acc.items():
        flat_abs.append(a.abs().reshape(-1))
        shapes.append((n, int(a.numel()), a.shape))
    allabs = (
        torch.cat(flat_abs)
        if flat_abs
        else torch.empty(0, dtype=torch.int64, device="cpu")
    )
    device = allabs.device
    crosser_idx = torch.nonzero(allabs >= int(threshold), as_tuple=False).flatten()
    n_candidates = int(crosser_idx.numel())
    cand = torch.zeros(allabs.shape, dtype=torch.bool, device=device)
    sel = torch.zeros_like(cand)
    ordered = torch.empty(0, dtype=torch.int64, device=device)
    if n_candidates:
        cand[crosser_idx] = True
        k = min(int(topk), n_candidates)
        keys = composite_rank_key(allabs[crosser_idx], crosser_idx)
        order = torch.argsort(keys, descending=True)
        ordered = crosser_idx[order[:k]]
        sel[ordered] = True
    n_applied = int(sel.sum().item())
    candidate_masks: dict[str, torch.Tensor] = {}
    applied_masks: dict[str, torch.Tensor] = {}
    off = 0
    for n, nn, shape in shapes:
        candidate_masks[n] = cand[off : off + nn].view(shape)
        applied_masks[n] = sel[off : off + nn].view(shape)
        off += nn
    return candidate_masks, applied_masks, n_candidates, n_applied, ordered


def project_and_update_acc_episode(
    *,
    grads: Mapping[str, torch.Tensor],
    q_auth_cpu: Mapping[str, torch.Tensor],
    q_shadow: dict[str, torch.Tensor],
    acc: Mapping[str, torch.Tensor],
    episode_start: Mapping[str, torch.Tensor],
    step: int,
    q_shadow_mode: str = "index_only",
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, Any]]:
    """Device grad→moves→acc/episode with q-shadow for projection.

    q_auth_cpu remains authoritative. q_shadow is device mirror.
    q_shadow_mode:
      - 'full_h2d': refresh entire shadow from CPU each call
      - 'index_only': assume shadow already coherent (no full refresh)
    """
    sync_counts: dict[str, Any] = {
        "full_q_h2d_bytes": 0,
        "scalar_item_publishes": 0,
        "mode": q_shadow_mode,
    }
    new_acc: dict[str, torch.Tensor] = {}
    new_ep: dict[str, torch.Tensor] = {}
    moves_out: dict[str, torch.Tensor] = {}
    for n in acc.keys():
        q_cpu = q_auth_cpu[n]
        if q_shadow_mode == "full_h2d":
            q_shadow[n].copy_(q_cpu.to(device=q_shadow[n].device, non_blocking=False))
            sync_counts["full_q_h2d_bytes"] += int(q_cpu.numel()) * int(q_cpu.element_size())
        g = grads[n]
        if g.device != q_shadow[n].device:
            g = g.to(device=q_shadow[n].device)
        mv = project_s1_gradient_to_moves(g, q_shadow[n])
        moves_out[n] = mv
        prev = acc[n]
        nxt = (
            (prev.to(torch.int32) + mv.to(torch.int32))
            .clamp(-CLIP, CLIP)
            .to(torch.int16)
        )
        new_ep[n] = update_episode_starts(prev, nxt, episode_start[n], step)
        new_acc[n] = nxt
    return new_acc, new_ep, moves_out, sync_counts


def cpu_oracle_project_and_update(
    *,
    grads_cpu: Mapping[str, torch.Tensor],
    q_cpu: Mapping[str, torch.Tensor],
    acc_cpu: Mapping[str, torch.Tensor],
    episode_cpu: Mapping[str, torch.Tensor],
    step: int,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    new_acc: dict[str, torch.Tensor] = {}
    new_ep: dict[str, torch.Tensor] = {}
    moves_out: dict[str, torch.Tensor] = {}
    for n in acc_cpu.keys():
        mv = project_s1_gradient_to_moves(grads_cpu[n], q_cpu[n])
        moves_out[n] = mv
        prev = acc_cpu[n]
        nxt = (
            (prev.to(torch.int32) + mv.to(torch.int32))
            .clamp(-CLIP, CLIP)
            .to(torch.int16)
        )
        new_ep[n] = update_episode_starts(prev, nxt, episode_cpu[n], step)
        new_acc[n] = nxt
    return new_acc, new_ep, moves_out


def _batched_cuda_writeback(
    *,
    acc: dict[str, torch.Tensor],
    episode_start: dict[str, torch.Tensor],
    q_auth_cpu: dict[str, torch.Tensor],
    q_shadow: dict[str, torch.Tensor] | None,
    applied_masks: Mapping[str, torch.Tensor],
    step: int,
    threshold: int,
    refresh_shadow_index_only: bool,
    ordered_flat_idx: torch.Tensor | None = None,
) -> dict[str, Any]:
    """ONE global idx D2H + ONE global dir D2H for all selected rows across arms.

    When ``ordered_flat_idx`` is provided (composite-key ordered selection from
    select_topk), that tensor is the sole idx source: the host copy is reused for
    identity framing so the production loop does not perform a second idx D2H.
    """
    from calm.hrm_text_158.native_full_stack.forgetting_laws import (
        threshold_residual_writeback,
    )

    names = list(applied_masks.keys())
    flat_masks = [applied_masks[n].reshape(-1) for n in names]
    flat_applied = torch.cat(flat_masks) if flat_masks else torch.empty(0, dtype=torch.bool)
    if ordered_flat_idx is not None:
        global_idx = ordered_flat_idx.reshape(-1).to(dtype=torch.int64)
    else:
        global_idx = torch.nonzero(flat_applied, as_tuple=False).flatten()
    stats: dict[str, Any] = {
        "applied_nonzero_d2h_bytes": 0,
        "dir_d2h_bytes": 0,
        "q_index_h2d_bytes": 0,
        "scalar_item_publishes": 0,
        "n_applied_total": 0,
        "n_arms_with_applied": 0,
        "idx_d2h_count": 0,
        "dir_d2h_count": 0,
        "batched_global_d2h": True,
        "selection_idx_host": torch.empty(0, dtype=torch.int64),
        "duplicate_idx_d2h_for_identity": False,
    }
    if global_idx.numel() == 0:
        return stats
    if int(global_idx.numel()) > 1024:
        raise RuntimeError(f"applied count {int(global_idx.numel())} exceeds topk bridge")

    # ONE idx D2H for the entire selected set (all arms); host payload also
    # feeds ordered selection identity (no separate per-step identity D2H).
    idx_cpu = global_idx.detach().cpu()
    stats["applied_nonzero_d2h_bytes"] = int(idx_cpu.numel()) * 8
    stats["idx_d2h_count"] = 1
    stats["n_applied_total"] = int(idx_cpu.numel())
    stats["selection_idx_host"] = idx_cpu.to(dtype=torch.int64).contiguous()

    offsets: list[tuple[str, int, int]] = []
    off = 0
    for n in names:
        nn = int(acc[n].numel())
        offsets.append((n, off, nn))
        off += nn
    # Count arms that contribute ≥1 applied row (host inventory on idx_cpu; no D2H).
    arm_hits = 0
    for _n, start, nn in offsets:
        if int(((idx_cpu >= start) & (idx_cpu < start + nn)).sum()) > 0:
            arm_hits += 1
    stats["n_arms_with_applied"] = arm_hits

    flat_acc = torch.cat([acc[n].reshape(-1) for n in names])
    flat_ep = torch.cat([episode_start[n].reshape(-1) for n in names])
    a_at = flat_acc[global_idx]
    dir_ = torch.where(a_at >= 0, torch.ones_like(a_at), -torch.ones_like(a_at)).to(torch.int8)
    residual = threshold_residual_writeback(a_at, dir_, threshold=threshold)
    flat_acc[global_idx] = residual
    # ONE dir D2H for the entire selected set.
    dir_cpu = dir_.detach().cpu()
    stats["dir_d2h_bytes"] = int(dir_cpu.numel()) * int(dir_cpu.element_size())
    stats["dir_d2h_count"] = 1

    flat_q_parts = []
    for n in names:
        flat_q_parts.append(q_auth_cpu[n].reshape(-1))
    # CPU q is already host; index with idx_cpu
    flat_q = torch.cat(flat_q_parts)
    q_before = flat_q[idx_cpu].to(torch.int16)
    q_after = (q_before + dir_cpu.to(torch.int16)).clamp(-1, 1).to(torch.int8)
    n_q_total = (q_after.to(torch.int16) != q_before).to(torch.int64).sum()
    flat_q[idx_cpu] = q_after
    # Write CPU q back per arm
    for n, start, nn in offsets:
        q_auth_cpu[n] = flat_q[start : start + nn].view_as(q_auth_cpu[n]).contiguous()

    zero_res = residual == 0
    ep_at = flat_ep[global_idx]
    ep_at = torch.where(zero_res, torch.zeros_like(ep_at), torch.full_like(ep_at, int(step)))
    flat_ep[global_idx] = ep_at
    # Scatter device acc/ep back into per-arm tensors (views into cat are copies).
    for n, start, nn in offsets:
        acc[n].reshape(-1).copy_(flat_acc[start : start + nn])
        episode_start[n].reshape(-1).copy_(flat_ep[start : start + nn])

    if q_shadow is not None and refresh_shadow_index_only:
        for n, start, nn in offsets:
            local_mask = (idx_cpu >= start) & (idx_cpu < start + nn)
            if int(local_mask.sum()) == 0:
                continue
            local_idx_cpu = idx_cpu[local_mask] - start
            local_idx_dev = global_idx[local_mask.to(device=global_idx.device)] - start
            flat_shadow = q_shadow[n].reshape(-1)
            flat_shadow[local_idx_dev] = (
                q_auth_cpu[n].reshape(-1)[local_idx_cpu].to(device=flat_shadow.device)
            )
            stats["q_index_h2d_bytes"] += int(local_idx_cpu.numel()) * int(
                q_auth_cpu[n].element_size()
            )

    # At most ONE host publish for n_q total.
    _ = int(n_q_total.detach().cpu().item())
    stats["scalar_item_publishes"] = 1
    stats["n_q_transitions_total"] = _
    # Inventory any() calls above are NOT counted as hot-path publishes —
    # they run only to fill n_arms_with_applied / empty-arm skip. Bound stays 1.
    return stats


def writeback_bridge_cpu_q(
    *,
    acc: dict[str, torch.Tensor],
    episode_start: dict[str, torch.Tensor],
    q_auth_cpu: dict[str, torch.Tensor],
    q_shadow: dict[str, torch.Tensor] | None,
    applied_masks: Mapping[str, torch.Tensor],
    step: int,
    refresh_shadow_index_only: bool = True,
    threshold: int = CROSSING_THRESHOLD_ABS,
    ordered_flat_idx: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Device acc/ep writeback + ≤1024-index CPU q bridge.

    CUDA path: ONE global idx D2H + ONE global dir D2H for all selected rows
    (kills per-arm transfer multiplier). n_q_transitions: at most ONE .item().
    Pass ``ordered_flat_idx`` from select_topk so the same host idx payload
    feeds ordered identity framing (no duplicate per-step identity D2H).
    """
    sample = next(iter(acc.values()))
    if sample.device.type == "cuda":
        # Confirm q is host-authoritative for bridge path.
        q0 = next(iter(q_auth_cpu.values()))
        if q0.device.type == "cpu":
            return _batched_cuda_writeback(
                acc=acc,
                episode_start=episode_start,
                q_auth_cpu=q_auth_cpu,
                q_shadow=q_shadow,
                applied_masks=applied_masks,
                step=step,
                threshold=int(threshold),
                refresh_shadow_index_only=refresh_shadow_index_only,
                ordered_flat_idx=ordered_flat_idx,
            )

    stats: dict[str, Any] = {
        "applied_nonzero_d2h_bytes": 0,
        "dir_d2h_bytes": 0,
        "q_index_h2d_bytes": 0,
        "scalar_item_publishes": 0,
        "n_applied_total": 0,
        "n_arms_with_applied": 0,
        "idx_d2h_count": 0,
        "dir_d2h_count": 0,
        "batched_global_d2h": False,
        "n_q_transitions_total": 0,
        "selection_idx_host": torch.empty(0, dtype=torch.int64),
        "duplicate_idx_d2h_for_identity": False,
    }
    if ordered_flat_idx is not None:
        host = ordered_flat_idx.detach()
        if host.device.type != "cpu":
            host = host.cpu()
            stats["idx_d2h_count"] = 1
            stats["applied_nonzero_d2h_bytes"] = int(host.numel()) * 8
        stats["selection_idx_host"] = host.to(dtype=torch.int64).contiguous()
        stats["n_applied_total"] = int(stats["selection_idx_host"].numel())
    for n, mask in applied_masks.items():
        a = acc[n]
        ep = episode_start[n]
        q = q_auth_cpu[n]
        m = mask
        new_acc, new_ep, new_q, _lt, n_q = apply_live_flip_writeback(
            a, ep, q, m, step=step, threshold=threshold
        )
        acc[n] = new_acc
        episode_start[n] = new_ep
        q_auth_cpu[n] = new_q
        n_app = int(m.to(torch.int64).sum().item())
        if ordered_flat_idx is None:
            stats["n_applied_total"] += n_app
        stats["n_q_transitions_total"] += int(n_q)
        stats["scalar_item_publishes"] += 1
        if n_app > 0:
            stats["n_arms_with_applied"] += 1
        if q_shadow is not None:
            q_shadow[n].copy_(new_q.to(device=q_shadow[n].device))
    return stats


def writeback_cpu_oracle(
    *,
    acc_cpu: dict[str, torch.Tensor],
    episode_cpu: dict[str, torch.Tensor],
    q_cpu: dict[str, torch.Tensor],
    applied_masks_cpu: Mapping[str, torch.Tensor],
    step: int,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Production apply_live_flip_writeback oracle (unchanged function)."""
    out_a: dict[str, torch.Tensor] = {}
    out_e: dict[str, torch.Tensor] = {}
    out_q: dict[str, torch.Tensor] = {}
    for n, mask in applied_masks_cpu.items():
        na, ne, nq, _lt, _nq = apply_live_flip_writeback(
            acc_cpu[n], episode_cpu[n], q_cpu[n], mask, step=step
        )
        out_a[n] = na
        out_e[n] = ne
        out_q[n] = nq
    return out_a, out_e, out_q
