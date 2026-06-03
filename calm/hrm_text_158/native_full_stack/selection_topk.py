"""Selection/top-k/tie-break reference seam for the native learner loop.

This is an honest torch reference surface for the control-flow selection seam.
It is not a custom Triton argsort kernel, q/acc mutation, global cap, packed
state, or full learner-loop receipt.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from calm.hrm_text_158.native_full_stack.vote_update import VoteUpdatePlan


RUN_GPU_SELECTION_TOPK_ENV = "HRM_TEXT_158_RUN_GPU_SELECTION_TOPK"
SELECTION_TOPK_TIEBREAK_CUDA_REFERENCE_SCOPE = "selection_topk_tiebreak_cuda_reference_only"
SELECTION_TOPK_CONTROL_FLOW_NOTE = (
    "selection/top-k/tie-break is a control-flow sort seam implemented with "
    "torch CUDA primitives for the Phase-1 receipt, not a custom Triton argsort"
)


@dataclass(frozen=True)
class SelectionTopKTieBreakResult:
    pre_veto_selected_indices: torch.Tensor
    selected_directions: torch.Tensor
    selected_thresholds: torch.Tensor
    selected_composite_scores: torch.Tensor
    scope: str
    backend: str


def _empty_selection(
    *,
    device: torch.device,
    scope: str,
    backend: str,
) -> SelectionTopKTieBreakResult:
    empty_i64 = torch.empty(0, dtype=torch.int64, device=device)
    return SelectionTopKTieBreakResult(
        pre_veto_selected_indices=empty_i64,
        selected_directions=torch.empty(0, dtype=torch.int16, device=device),
        selected_thresholds=torch.empty(0, dtype=torch.int32, device=device),
        selected_composite_scores=empty_i64,
        scope=scope,
        backend=backend,
    )


def selection_topk_tiebreak_reference_on_device(
    *,
    new_acc_i32: torch.Tensor,
    candidate_indices: torch.Tensor,
    threshold_abs: int,
    max_flips: int,
    scope: str = SELECTION_TOPK_TIEBREAK_CUDA_REFERENCE_SCOPE,
) -> SelectionTopKTieBreakResult:
    """Select local candidates using the live CPU law on the tensor's device."""

    threshold = int(threshold_abs)
    if threshold <= 0:
        raise ValueError(f"threshold_abs must be > 0, got {threshold_abs}")
    max_flips_i = int(max_flips)
    if max_flips_i < 0:
        raise ValueError(f"max_flips must be >= 0, got {max_flips}")
    if not candidate_indices.dtype in (torch.int32, torch.int64):
        raise ValueError(f"candidate_indices must be int32/int64, got {candidate_indices.dtype}")
    if not new_acc_i32.dtype in (torch.int16, torch.int32, torch.int64):
        raise ValueError(f"new_acc_i32 must be an integer tensor, got {new_acc_i32.dtype}")

    flat_acc = new_acc_i32.flatten().to(torch.int32)
    device = flat_acc.device
    backend = device.type
    candidate_idx = candidate_indices.flatten().to(device=device, dtype=torch.int64)
    if candidate_idx.numel() == 0 or max_flips_i == 0:
        return _empty_selection(device=device, scope=scope, backend=backend)
    if bool((candidate_idx < 0).any().item()) or bool((candidate_idx >= flat_acc.numel()).any().item()):
        raise ValueError("candidate_indices contain out-of-range flat indices")

    numel = int(flat_acc.numel())
    abs_score = flat_acc[candidate_idx].abs().to(torch.int64)
    composite = abs_score * (numel + 1) + (numel - candidate_idx)
    order = torch.argsort(composite, descending=True)
    take = min(max_flips_i, int(candidate_idx.numel()))
    selected_order = order[:take]
    selected = candidate_idx[selected_order].to(torch.int64)
    selected_scores = composite[selected_order].to(torch.int64)
    selected_acc = flat_acc[selected]
    directions = torch.where(
        selected_acc >= threshold,
        torch.ones_like(selected_acc, dtype=torch.int16),
        -torch.ones_like(selected_acc, dtype=torch.int16),
    )
    thresholds = torch.full((take,), threshold, dtype=torch.int32, device=device)
    return SelectionTopKTieBreakResult(
        pre_veto_selected_indices=selected,
        selected_directions=directions,
        selected_thresholds=thresholds,
        selected_composite_scores=selected_scores,
        scope=scope,
        backend=backend,
    )


def select_pre_veto_candidates_from_plan(
    plan: VoteUpdatePlan,
    *,
    threshold_abs: int,
    max_flips: int | None = None,
    scope: str = SELECTION_TOPK_TIEBREAK_CUDA_REFERENCE_SCOPE,
) -> SelectionTopKTieBreakResult:
    """Reproduce ``VoteUpdatePlan.pre_veto_selected_*`` on the plan device."""

    if max_flips is None:
        max_flips = int(plan.stats["max_flips"])
    return selection_topk_tiebreak_reference_on_device(
        new_acc_i32=plan.new_acc_i32,
        candidate_indices=plan.candidate_indices,
        threshold_abs=threshold_abs,
        max_flips=int(max_flips),
        scope=scope,
    )
