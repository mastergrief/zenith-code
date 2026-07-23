"""Pure observation reducers for pressure/metric diagnostic (PLAN_v6).

Owns: selection/masks, margin quantiles, demand summaries, episode-age stats,
JSON sanitize helpers, prereg constants. NO mutable lifecycle store
(see pressure_metric_lifecycle.py). Dependency: telemetry → ∅ (two_tier only).
Bound by PLAN_v6 sha 346b67d8…; rev3 re-scope 1784828063166.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

import torch

from calm.hrm_text_158.native_full_stack.two_tier_threshold_semantics import (
    CROSSING_THRESHOLD_ABS,
)

# --- prereg constants (PLAN_v6 branch_classifier_prereg.constants) ---
EPS = 1e-9
HIGH_DEMAND_RATIO = 2.0
SUSTAINED_HIGH_DEMAND_FRAC_STEPS = 0.50
GROWING_DEFERRED_SURVIVAL_DELTA = 0.10
LOW_MODERATE_DEMAND_RATIO_MAX = 1.25
HIGH_LCF = 0.90
MATERIAL_H_MOTION_BPW = 1.0
REPRESENTATION_IMMOVABLE_H_DELTA_MAX = 0.25
STABLE_HIGH_DEFERRED_NEVER_APPLY_FLOOR = 0.50
FOLLOW_UP_HORIZON_STEPS = 32
MIN_COHORT_N = 100

LABEL_R0 = "F_underspecified"
LABEL_R1 = "pressure_source_backlog"
LABEL_R2 = "metric_mismatch"
LABEL_R3 = "representation_unresolved__low_pressure_low_H_motion"
LABEL_R4 = "metric_insufficient__pressure_partial"

PLAN_SHA256 = "346b67d878bc9ef4eda58d96002e8e820f36bdfc811a35db72aca03344382d29"
AUTHORITY_DISPATCH = "1784826002707-a339eb22"
PARENT_SHA256 = (
    "2d9b9f6746e66cec9e7e39d65e8171151e836daca99df6b56fb488d8a6f2403b"
)

# Formal geometry (PLAN_v6)
FORMAL_STEPS = 150
FORMAL_BATCH = 8
FORMAL_TOPK = 1024
PAIRED_STEPS = 25
PAIRED_N = 3
OVERHEAD_BOUND = 0.15


def expected_trajectory_boundaries(steps: int) -> list[int]:
    """Exact expected 25-step boundaries including final for a given window."""
    steps = int(steps)
    out = list(range(25, steps + 1, 25))
    if steps not in out and steps > 0:
        out.append(steps)
    return out


def optional_json_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x


def sanitize_receipt_for_strict_json(obj: Any) -> Any:
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: sanitize_receipt_for_strict_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_receipt_for_strict_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [sanitize_receipt_for_strict_json(v) for v in obj]
    return obj


def assert_two_tier_threshold_pass(
    *, effective: int, canonical: int = CROSSING_THRESHOLD_ABS
) -> bool:
    return int(effective) == int(canonical)


def compute_topk_masks_and_counts(
    arms_acc: Mapping[str, torch.Tensor],
    *,
    topk: int,
    threshold: int = CROSSING_THRESHOLD_ABS,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], int, int]:
    """Pre-topK candidate masks + applied topK masks; counts are PRE-topK aware."""
    flat_abs = []
    shapes = []
    for n, a in arms_acc.items():
        flat_abs.append(a.abs().flatten())
        shapes.append((n, a.numel(), a.shape))
    allabs = torch.cat(flat_abs)
    crosser_idx = torch.nonzero(allabs >= int(threshold), as_tuple=False).flatten()
    n_candidates = int(crosser_idx.numel())
    cand = torch.zeros_like(allabs, dtype=torch.bool)
    sel = torch.zeros_like(allabs, dtype=torch.bool)
    if n_candidates:
        cand[crosser_idx] = True
        k = min(int(topk), n_candidates)
        top = crosser_idx[allabs[crosser_idx].argsort(descending=True)[:k]]
        sel[top] = True
    n_applied = int(sel.sum().item())
    candidate_masks: dict[str, torch.Tensor] = {}
    applied_masks: dict[str, torch.Tensor] = {}
    off = 0
    for n, nn, shape in shapes:
        candidate_masks[n] = cand[off : off + nn].view(shape)
        applied_masks[n] = sel[off : off + nn].view(shape)
        off += nn
    return candidate_masks, applied_masks, n_candidates, n_applied


def _safe_quantiles_p10_p50_p90(values: torch.Tensor) -> torch.Tensor:
    """p10/p50/p90 without torch.quantile's ~2^24-element hard failure."""
    t = values.detach().reshape(-1).to(torch.float32)
    # torch.quantile refuses tensors larger than 2**24 elements.
    max_n = 16_777_216
    n = int(t.numel())
    if n > max_n:
        step = (n + max_n - 1) // max_n
        t = t[::step]
    try:
        return torch.quantile(t, torch.tensor([0.10, 0.50, 0.90], dtype=t.dtype))
    except RuntimeError:
        import numpy as np

        arr = t.detach().cpu().numpy()
        return torch.tensor(
            np.quantile(arr, [0.10, 0.50, 0.90]), dtype=torch.float32
        )


def margin_quantiles(
    acc: torch.Tensor,
    mask: torch.Tensor,
    *,
    threshold: int = CROSSING_THRESHOLD_ABS,
) -> dict[str, float | None]:
    if not bool(mask.any()):
        return {"p10": None, "p50": None, "p90": None, "n": 0}
    margins = (acc[mask].abs().to(torch.float32) - float(threshold)).flatten()
    qs = _safe_quantiles_p10_p50_p90(margins)
    return {
        "p10": float(qs[0].item()),
        "p50": float(qs[1].item()),
        "p90": float(qs[2].item()),
        "n": int(margins.numel()),
    }


def global_margin_quantiles(
    acc_dict: dict[str, torch.Tensor],
    masks_dict: dict[str, torch.Tensor],
    *,
    threshold: int = CROSSING_THRESHOLD_ABS,
) -> dict[str, float | None]:
    parts: list[torch.Tensor] = []
    for name, mask in masks_dict.items():
        acc = acc_dict.get(name)
        if acc is None or not bool(mask.any()):
            continue
        parts.append((acc[mask].abs().to(torch.float32) - float(threshold)).flatten())
    if not parts:
        return {"p10": None, "p50": None, "p90": None, "n": 0}
    margins = torch.cat(parts)
    qs = _safe_quantiles_p10_p50_p90(margins)
    return {
        "p10": float(qs[0].item()),
        "p50": float(qs[1].item()),
        "p90": float(qs[2].item()),
        "n": int(margins.numel()),
    }


def summarize_demand_totals(per_step: list[dict[str, Any]]) -> dict[str, Any]:
    if not per_step:
        return {
            "mean_ratio": None,
            "max_ratio": None,
            "frac_steps_ratio_gt_1": None,
            "frac_steps_ratio_ge_2": None,
            "n_steps": 0,
        }
    ratios = [float(r["demand_applied_ratio"]) for r in per_step]
    n = len(ratios)
    return {
        "mean_ratio": float(sum(ratios) / n),
        "max_ratio": float(max(ratios)),
        "frac_steps_ratio_gt_1": float(sum(1 for x in ratios if x > 1.0) / n),
        "frac_steps_ratio_ge_2": float(
            sum(1 for x in ratios if x >= HIGH_DEMAND_RATIO) / n
        ),
        "n_steps": n,
    }


def active_episode_stats(
    acc: Mapping[str, torch.Tensor],
    episode_start: Mapping[str, torch.Tensor],
    *,
    step: int,
) -> dict[str, Any]:
    ages: list[torch.Tensor] = []
    n_active = 0
    for n, a in acc.items():
        ep = episode_start[n]
        active = (a != 0) & (ep > 0)
        n_here = int(active.sum().item())
        n_active += n_here
        if n_here:
            ages.append((int(step) - ep[active].to(torch.int32)).to(torch.float32))
    if not ages:
        return {
            "active_episode_count": 0,
            "episode_age_quantiles_p10_p50_p90": {
                "p10": None,
                "p50": None,
                "p90": None,
                "n": 0,
            },
        }
    cat = torch.cat(ages)
    qs = _safe_quantiles_p10_p50_p90(cat)
    return {
        "active_episode_count": int(n_active),
        "episode_age_quantiles_p10_p50_p90": {
            "p10": float(qs[0].item()),
            "p50": float(qs[1].item()),
            "p90": float(qs[2].item()),
            "n": int(cat.numel()),
        },
    }


def hash_scale_dict(frozen_scales: Mapping[str, torch.Tensor]) -> str:
    """Re-hash live frozen scales (CPU tensors) — never copy a prior sha."""
    import hashlib

    parts = []
    for n in sorted(frozen_scales):
        t = frozen_scales[n].detach().cpu().contiguous()
        parts.append(hashlib.sha256(t.numpy().tobytes()).hexdigest().encode())
    return hashlib.sha256(b"".join(parts)).hexdigest()


def hash_q_dict(q_levels: Mapping[str, torch.Tensor]) -> str:
    import hashlib

    parts = []
    for n in sorted(q_levels):
        t = q_levels[n].detach().cpu().contiguous()
        parts.append(hashlib.sha256(t.numpy().tobytes()).hexdigest().encode())
    return hashlib.sha256(b"".join(parts)).hexdigest()
