"""Shared A/B GPU loop residency + residual/lifetime helpers (production path).

Owns device acc/episode/q_shadow init and thin glue so screen_execution_loop
stays under the LOC budget. Selection/update/writeback live in
pressure_metric_gpu_selection_derisk; lifecycle in
pressure_metric_gpu_lifecycle_derisk.

Hot-loop host syncs are batched: credited mass and drained counts publish via
ONE .item() each; lifetimes via ONE .tolist(); selection identity reuses the
writeback's already-required host idx payload (no separate idx D2H).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from calm.hrm_text_158.native_full_stack.forgetting_laws import (
    threshold_residual_writeback,
)
from calm.hrm_text_158.native_full_stack.pressure_metric_gpu_selection_derisk import (
    project_and_update_acc_episode,
    select_topk_masks_deterministic,
    writeback_bridge_cpu_q,
)
from calm.hrm_text_158.native_full_stack.two_tier_threshold_semantics import (
    CROSSING_THRESHOLD_ABS,
)

# Closed allowlist for static sync-class audit (bridge + loop hot path).
# Counts are maximum permitted source occurrences of each sync pattern.
HOTPATH_SYNC_ALLOWLIST = {
    "pressure_metric_gpu_loop_bridge.py": {
        ".item(": 2,  # credited_mass + drained (one each)
        ".tolist(": 1,  # lifetimes ONE boundary publish
        ".any(": 0,  # no per-arm .any() host syncs
    },
    "screen_execution_loop.py": {
        ".item(": 0,  # drained/credit publishes live in bridge helpers
        ".tolist(": 0,
        ".any(": 0,
    },
}


def count_hotpath_sync_patterns(source: str) -> dict[str, int]:
    """Count Attribute calls `.item` / `.tolist` / `.any` via AST (not substrings)."""
    import ast

    counts = {".item(": 0, ".tolist(": 0, ".any(": 0}
    attr_to_key = {"item": ".item(", "tolist": ".tolist(", "any": ".any("}
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            key = attr_to_key.get(node.func.attr)
            if key is not None:
                counts[key] += 1
    return counts


def assert_hotpath_sync_allowlist() -> dict[str, dict[str, int]]:
    """Fail closed if bridge/loop exceed HOTPATH_SYNC_ALLOWLIST counts."""
    from pathlib import Path

    bridge_dir = Path(__file__).resolve().parent
    observed: dict[str, dict[str, int]] = {}
    for fname, limits in HOTPATH_SYNC_ALLOWLIST.items():
        text = (bridge_dir / fname).read_text(encoding="utf-8")
        counts = count_hotpath_sync_patterns(text)
        observed[fname] = counts
        for pat, limit in limits.items():
            if counts[pat] > int(limit):
                raise AssertionError(
                    f"{fname}: sync pattern {pat!r} count={counts[pat]} "
                    f"exceeds allowlist {limit}"
                )
    return observed


@dataclass
class GpuLoopResidency:
    """Shared A/B residency: device acc/episode (+ optional q_shadow); q CPU-auth."""

    device: torch.device
    acc: dict[str, torch.Tensor]
    episode_start: dict[str, torch.Tensor]
    flip_count: dict[str, torch.Tensor]
    q_auth_cpu: dict[str, torch.Tensor]
    q_shadow: dict[str, torch.Tensor] | None
    use_device_path: bool


def init_gpu_loop_residency(
    q_levels: dict[str, torch.Tensor],
    *,
    device: str | torch.device,
) -> GpuLoopResidency:
    """Build shared residency. CUDA → device acc/ep + q_shadow; else CPU tensors."""
    dev = torch.device(device)
    use_device = dev.type == "cuda" and torch.cuda.is_available()
    target = dev if use_device else torch.device("cpu")
    acc = {
        n: torch.zeros(q.shape, dtype=torch.int16, device=target) for n, q in q_levels.items()
    }
    episode_start = {
        n: torch.zeros(q.shape, dtype=torch.int32, device=target) for n, q in q_levels.items()
    }
    flip_count = {
        n: torch.zeros(q.shape, dtype=torch.int32, device=target) for n, q in q_levels.items()
    }
    q_shadow = None
    if use_device:
        q_shadow = {n: q.to(device=target) for n, q in q_levels.items()}
    return GpuLoopResidency(
        device=target,
        acc=acc,
        episode_start=episode_start,
        flip_count=flip_count,
        q_auth_cpu=q_levels,
        q_shadow=q_shadow,
        use_device_path=use_device,
    )


def _credited_mass_one_publish(moves: Mapping[str, torch.Tensor]) -> int:
    """Sum |moves| across arms with ONE host scalar publish."""
    if not moves:
        return 0
    masses = torch.stack([mv.detach().abs().sum() for mv in moves.values()])
    return int(masses.sum().item())


def project_credit_shared(
    residency: GpuLoopResidency,
    *,
    credit_grads: Mapping[str, torch.Tensor],
    eligible: list[str],
    step: int,
) -> tuple[dict[str, torch.Tensor], int]:
    """Project grads into acc/episode; return moves + credited_mass delta."""
    from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
        project_s1_gradient_to_moves,
    )
    from calm.hrm_text_158.native_full_stack.vote_lifetime_screen_reducers import (
        update_episode_starts,
    )

    if residency.use_device_path and residency.q_shadow is not None:
        grads = {}
        for n in eligible:
            g = credit_grads[n].detach()
            if g.device != residency.device:
                g = g.to(device=residency.device)
            grads[n] = g
        new_acc, new_ep, moves, _sync = project_and_update_acc_episode(
            grads=grads,
            q_auth_cpu=residency.q_auth_cpu,
            q_shadow=residency.q_shadow,
            acc=residency.acc,
            episode_start=residency.episode_start,
            step=step,
            q_shadow_mode="index_only",
        )
        residency.acc = new_acc
        residency.episode_start = new_ep
        return moves, _credited_mass_one_publish(moves)

    moves: dict[str, torch.Tensor] = {}
    for n in eligible:
        g_cpu = credit_grads[n].detach().cpu()
        q_cpu = residency.q_auth_cpu[n]
        if g_cpu.shape != q_cpu.shape:
            raise RuntimeError(
                f"credit/q shape mismatch for {n}: {tuple(g_cpu.shape)} vs "
                f"{tuple(q_cpu.shape)}"
            )
        moves[n] = project_s1_gradient_to_moves(g_cpu, q_cpu)
    credited = _credited_mass_one_publish(moves)
    for n, mv in moves.items():
        prev = residency.acc[n]
        new = (
            (prev.to(torch.int32) + mv.to(torch.int32))
            .clamp(-127, 127)
            .to(torch.int16)
        )
        residency.episode_start[n] = update_episode_starts(
            prev, new, residency.episode_start[n], step
        )
        residency.acc[n] = new
    return moves, credited


def select_shared(
    residency: GpuLoopResidency,
    *,
    topk: int,
    threshold: int = CROSSING_THRESHOLD_ABS,
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    int,
    int,
    torch.Tensor,
]:
    return select_topk_masks_deterministic(
        residency.acc, topk=int(topk), threshold=int(threshold)
    )


def predict_residual_zero(
    residency: GpuLoopResidency,
    applied_masks: Mapping[str, torch.Tensor],
    *,
    threshold: int = CROSSING_THRESHOLD_ABS,
) -> dict[str, torch.Tensor]:
    """Residual-zero prediction matching writeback law (pre-mutation).

    No per-arm ``.any()`` host sync: empty masks yield empty gathers naturally.
    """
    out: dict[str, torch.Tensor] = {}
    for n, mask in applied_masks.items():
        acc_n = residency.acc[n]
        dir_ = torch.where(
            acc_n >= 0, torch.ones_like(acc_n), -torch.ones_like(acc_n)
        ).to(torch.int8)
        res = threshold_residual_writeback(
            acc_n[mask], dir_[mask], threshold=int(threshold)
        )
        rz = torch.zeros_like(mask, dtype=torch.bool)
        # scatter only when gather was non-empty (numel is metadata; no D2H)
        if res.numel() > 0:
            rz[mask] = res == 0
        out[n] = rz
    return out


def lifetimes_before_writeback(
    residency: GpuLoopResidency,
    applied_masks: Mapping[str, torch.Tensor],
    *,
    step: int,
) -> list[int]:
    """Collect episode ages for applied rows (ONE boundary .tolist() publish)."""
    age_parts: list[torch.Tensor] = []
    for n, mask in applied_masks.items():
        starts = residency.episode_start[n][mask]
        active = starts > 0
        ages = (int(step) - starts[active].to(torch.int64))
        if ages.numel() > 0:
            age_parts.append(ages)
    if not age_parts:
        return []
    return torch.cat(age_parts).detach().cpu().tolist()


def apply_drained_flip_counts(
    residency: GpuLoopResidency,
    applied_masks: Mapping[str, torch.Tensor],
) -> int:
    """Update flip_count from applied masks; ONE .item() for step drained total."""
    if not applied_masks:
        return 0
    drained_parts: list[torch.Tensor] = []
    for n, mask in applied_masks.items():
        m_i = mask.to(torch.int32)
        residency.flip_count[n] = residency.flip_count[n] + m_i
        drained_parts.append(m_i.reshape(-1).sum())
    return int(torch.stack(drained_parts).sum().item())


def writeback_shared(
    residency: GpuLoopResidency,
    applied_masks: Mapping[str, torch.Tensor],
    *,
    step: int,
    threshold: int = CROSSING_THRESHOLD_ABS,
    ordered_flat_idx: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Shared writeback; mutates residency; returns bridge stats + host idx."""
    stats = writeback_bridge_cpu_q(
        acc=residency.acc,
        episode_start=residency.episode_start,
        q_auth_cpu=residency.q_auth_cpu,
        q_shadow=residency.q_shadow,
        applied_masks=applied_masks,
        step=int(step),
        refresh_shadow_index_only=True,
        threshold=int(threshold),
        ordered_flat_idx=ordered_flat_idx,
    )
    return stats


def ordered_selection_frame(
    *,
    step: int,
    ordered_flat_idx: torch.Tensor,
) -> bytes:
    """Canonical step-framed selection bytes from a HOST idx payload.

    Callers must pass the writeback-returned ``selection_idx_host`` (or an
    already-host tensor). Device tensors are rejected — no identity D2H here.
    """
    if ordered_flat_idx.device.type != "cpu":
        raise RuntimeError(
            "ordered_selection_frame requires host idx "
            "(reuse writeback selection_idx_host; no separate identity D2H)"
        )
    idx = ordered_flat_idx.detach().to(dtype=torch.int64).contiguous()
    header = int(step).to_bytes(4, "little", signed=False)
    return header + idx.numpy().tobytes()


def hotpath_sync_inventory_from_writeback(stats: Mapping[str, Any]) -> dict[str, Any]:
    """Compact transfer/sync inventory from one writeback stats dict."""
    return {
        "idx_d2h_count": int(stats.get("idx_d2h_count", 0) or 0),
        "dir_d2h_count": int(stats.get("dir_d2h_count", 0) or 0),
        "batched_global_d2h": bool(stats.get("batched_global_d2h", False)),
        "duplicate_idx_d2h_for_identity": bool(
            stats.get("duplicate_idx_d2h_for_identity", False)
        ),
        "scalar_item_publishes": int(stats.get("scalar_item_publishes", 0) or 0),
        "applied_nonzero_d2h_bytes": int(stats.get("applied_nonzero_d2h_bytes", 0) or 0),
        "dir_d2h_bytes": int(stats.get("dir_d2h_bytes", 0) or 0),
        "n_applied_total": int(stats.get("n_applied_total", 0) or 0),
        "n_arms_with_applied": int(stats.get("n_arms_with_applied", 0) or 0),
    }
