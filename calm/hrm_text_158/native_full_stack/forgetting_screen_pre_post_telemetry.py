"""Bounded GPU pre/post-transform telemetry for ARM1 decay discriminator (PLAN_v10.1r8).

Pack three 128-bin int64 hists + scalar counters into ONE device tensor.
finalize() performs EXACTLY one .cpu() D2H; CPU-side split into compact dict.
Move bin 0 is tracked on device but excluded from published move_abs_bins.
Quantile CDF uses bins 1..127 only (nonzero-magnitude population).
Forbidden on hotpath: full-surface torch.cat, per-step .item()/.tolist(), sampling.
No CLI/launch imports.
"""
from __future__ import annotations

from typing import Any, Mapping

import torch

from calm.hrm_text_158.native_full_stack.forgetting_screen_v10_1_contract import (
    PRE_POST_SCHEMA,
    TRANSFER_LAW,
)

ABS_HIST_BINS = 128  # 0..127 inclusive
# Packed layout: 3×128 hists + 6 scalars
_OFF_MOVE = 0
_OFF_PROJ = ABS_HIST_BINS
_OFF_DECAY = ABS_HIST_BINS * 2
_OFF_MOVE_NZ = ABS_HIST_BINS * 3
_OFF_PROJ_NZ = _OFF_MOVE_NZ + 1
_OFF_DECAY_NZ = _OFF_MOVE_NZ + 2
_OFF_ERASED = _OFF_MOVE_NZ + 3
_OFF_CAND = _OFF_MOVE_NZ + 4
_OFF_MISMATCH = _OFF_MOVE_NZ + 5
_PACKED_LEN = _OFF_MISMATCH + 1


def _accumulate_abs_hist_into(hist_slice: torch.Tensor, flat: torch.Tensor) -> None:
    """Add abs values clamped to 0..127 into hist_slice (int64, len 128) on device."""
    abs_i = flat.abs().to(torch.int64).clamp(0, ABS_HIST_BINS - 1)
    hist_slice.add_(torch.bincount(abs_i, minlength=ABS_HIST_BINS))


def _hist_quantile_nonzero_from_cpu(hist_np, q: float) -> float:
    """CDF quantile over bins 1..127 only (excludes zero-magnitude bin)."""
    total = int(hist_np[1:].sum()) if len(hist_np) > 1 else 0
    if total <= 0:
        return 0.0
    target = float(q) * float(total)
    cum = 0
    last = 1
    for i in range(1, int(hist_np.shape[0])):
        cum += int(hist_np[i])
        last = i
        if cum >= target:
            return float(i)
    return float(last)


def _hist_abs_max_nonzero_from_cpu(hist_np) -> int:
    for i in range(len(hist_np) - 1, 0, -1):
        if int(hist_np[i]) > 0:
            return int(i)
    return 0


class PrePostTransformAccumulator:
    """On-GPU packed accumulator; publish once via finalize() (single .cpu() D2H)."""

    def __init__(self, device: torch.device | str) -> None:
        self.device = torch.device(device)
        self._packed = torch.zeros(_PACKED_LEN, dtype=torch.int64, device=self.device)
        self._steps = 0  # python counter — no tensor.item

    def accumulate_step(
        self,
        *,
        moves: Mapping[str, torch.Tensor],
        acc_pre_decay: Mapping[str, torch.Tensor],
        acc_post_decay: Mapping[str, torch.Tensor],
        n_cand_after_decay: int,
    ) -> None:
        """Device-side per-tensor reductions into the packed buffer. No D2H."""
        self._steps += 1
        move_h = self._packed[_OFF_MOVE : _OFF_MOVE + ABS_HIST_BINS]
        proj_h = self._packed[_OFF_PROJ : _OFF_PROJ + ABS_HIST_BINS]
        decay_h = self._packed[_OFF_DECAY : _OFF_DECAY + ABS_HIST_BINS]
        for name, pre_t in acc_pre_decay.items():
            pre = pre_t.detach().reshape(-1)
            post = acc_post_decay[name].detach().reshape(-1)
            if pre.device != self.device:
                pre = pre.to(self.device)
            if post.device != self.device:
                post = post.to(self.device)
            pre_i = pre.to(torch.int32)
            post_i = post.to(torch.int32)
            _accumulate_abs_hist_into(proj_h, pre_i)
            _accumulate_abs_hist_into(decay_h, post_i)
            pre_nz = pre_i != 0
            self._packed[_OFF_PROJ_NZ].add_(pre_nz.sum())
            self._packed[_OFF_DECAY_NZ].add_((post_i != 0).sum())
            self._packed[_OFF_ERASED].add_((pre_nz & (post_i == 0)).sum())
            expected = torch.trunc(pre_i.to(torch.float32) * (31.0 / 32.0)).to(torch.int32)
            self._packed[_OFF_MISMATCH].add_((post_i != expected).sum())
        for mv in moves.values():
            flat = mv.detach().reshape(-1)
            if flat.device != self.device:
                flat = flat.to(self.device)
            flat_i = flat.to(torch.int32)
            nz = flat_i != 0
            self._packed[_OFF_MOVE_NZ].add_(nz.sum())
            _accumulate_abs_hist_into(move_h, flat_i)
        self._packed[_OFF_CAND].add_(
            torch.tensor(int(n_cand_after_decay), dtype=torch.int64, device=self.device)
        )

    def finalize(self) -> dict[str, Any]:
        """EXACTLY one .cpu() D2H; CPU-side split into compact schema dict."""
        host = self._packed.detach().cpu()  # sole D2H transfer
        arr = host.numpy()
        move_h = arr[_OFF_MOVE : _OFF_MOVE + ABS_HIST_BINS]
        proj_h = arr[_OFF_PROJ : _OFF_PROJ + ABS_HIST_BINS]
        decay_h = arr[_OFF_DECAY : _OFF_DECAY + ABS_HIST_BINS]
        move_nz = int(arr[_OFF_MOVE_NZ])
        proj_nz = int(arr[_OFF_PROJ_NZ])
        decay_nz = int(arr[_OFF_DECAY_NZ])
        erased = int(arr[_OFF_ERASED])
        cand = int(arr[_OFF_CAND])
        mismatch = int(arr[_OFF_MISMATCH])
        # Publish nonzero-magnitude bins only (exclude bin 0).
        bins = {
            str(i): int(move_h[i])
            for i in range(1, ABS_HIST_BINS)
            if int(move_h[i]) > 0
        }
        return {
            "schema": PRE_POST_SCHEMA,
            "law": TRANSFER_LAW,
            "move_abs_bins": bins,
            "move_nonzero_count": move_nz,
            "post_projection": {
                "nonzero": proj_nz,
                "abs_max": _hist_abs_max_nonzero_from_cpu(proj_h),
                "abs_p50": _hist_quantile_nonzero_from_cpu(proj_h, 0.50),
                "abs_p90": _hist_quantile_nonzero_from_cpu(proj_h, 0.90),
            },
            "post_decay": {
                "nonzero": decay_nz,
                "abs_max": _hist_abs_max_nonzero_from_cpu(decay_h),
                "abs_p50": _hist_quantile_nonzero_from_cpu(decay_h, 0.50),
                "abs_p90": _hist_quantile_nonzero_from_cpu(decay_h, 0.90),
            },
            "pre_nonzero_to_post_zero_count": erased,
            "pre_nonzero_to_post_zero_frac": (
                float(erased) / float(proj_nz) if proj_nz > 0 else None
            ),
            "post_decay_candidate_count": cand,
            "law_mismatch_count": mismatch,
            "steps_accumulated": int(self._steps),
        }
