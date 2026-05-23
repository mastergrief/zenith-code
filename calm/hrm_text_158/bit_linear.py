"""HRM-Text-1.58 Phase 2 Slice 1: BitLinear (native 1.58-bit bulk linear).

Per task #51, codex msg 1779457170889 (Phase 2 Slice 1 +1 implement).

D2.1 from RESEARCH/HRM-Text-1.58/01_DEVIATIONS.md:
  Replace bulk LinearInit with ternary BitLinear. Forward: quantize master
  weight to ternary {-1, 0, +1} via per-tensor absmean, scale via that
  absmean. STE for backward (gradient flows through master weight directly).
  FP/BF16 master weights persisted; quantized weights computed forward-only.

Bounded scope: ONLY gqkv_proj, o_proj, gate_up_proj, down_proj in
TransformerBlock attention + SwiGLU. NOT lm_head, NOT embed_tokens, NOT
norms, NOT zL_init (per D2.2).

Per-tensor absmean quantization is the BitNet b1.58 convention. STE
implemented via the standard `w + sg(w_q - w)` trick (Bengio et al.):
forward value = quantized*scale; backward gradient = identity to master.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from calm.hrm_text_158.layers import trunc_normal_init_


class BitLinear(nn.Module):
    """Ternary BitLinear with STE backward.

    Drop-in replacement for `LinearInit` in HRM-Text bulk projections per
    D2.1 / D2.3 (RESEARCH/HRM-Text-1.58/01_DEVIATIONS.md).

    - Master weight: FP/BF16 `nn.Parameter`, shape identical to LinearInit
    - Forward: quantize master → {-1, 0, +1} × per-tensor absmean scale
    - Backward: STE — gradient flows through master weight as identity

    Per-tensor absmean quantization (BitNet b1.58, arxiv:2402.17764).
    No activation quantization (FP activations preserved per D2.1
    bounded scope).
    """

    # Numerical floor for the absmean scale; prevents division-by-zero
    # when all weights happen to be exactly zero.
    _SCALE_EPS = 1e-5

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        batch_out_features: Sequence[int] = (),
        init_std: Optional[float] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        if init_std is None:
            init_std = 1.0 / (in_features ** 0.5)

        # Master weight (FP/BF16) — identical shape/init to LinearInit
        self.weight = nn.Parameter(
            trunc_normal_init_(
                torch.empty(
                    (math.prod(batch_out_features) * out_features, in_features),
                    **kwargs,
                ),
                std=init_std,
            )
        )
        self.bias = None
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(
                    (math.prod(batch_out_features) * out_features,),
                    **kwargs,
                )
            )

        # T1 (α) cached-ternary inference path. Runtime-only attributes;
        # NOT registered as buffers/parameters so they don't enter state_dict
        # (codex bound: no .pt schema/format change).
        self._cached_weight: Optional[Tensor] = None
        self._cached_active: bool = False

    def quantize_weight(self) -> tuple[Tensor, Tensor]:
        """Quantize master weight to ternary + per-tensor scale.

        Returns:
            (w_q_ste, scale): w_q_ste is the STE-wrapped quantized weight
            (forward value = quantized*scale, backward gradient = identity
            to master). scale is the per-tensor absmean used.
        """
        scale = self.weight.abs().mean().clamp(min=self._SCALE_EPS)
        # Ternary quantization: round to {-1, 0, +1} after scaling by 1/scale
        w_q = (self.weight / scale).round().clamp(-1.0, 1.0)
        # STE: forward uses w_q * scale; backward gradient flows to self.weight
        # via identity. Standard trick: w + sg(w_q*scale - w).
        w_q_ste = self.weight + (w_q * scale - self.weight).detach()
        return w_q_ste, scale

    def forward(self, input: Tensor) -> Tensor:
        # T1 (α) cached-ternary inference path: if frozen AND not training,
        # use cached `w_q * scale` directly. Defense in depth — the
        # `self.training` guard ensures the cached path never runs under
        # training even if cache was left active from a prior eval pass.
        # `train()` override below also clears `_cached_active` on mode flip,
        # so two independent checks both must agree before bypassing
        # re-quantize.
        if self._cached_active and self._cached_weight is not None and not self.training:
            return F.linear(input, self._cached_weight, self.bias)
        w_q_ste, _ = self.quantize_weight()
        return F.linear(input, w_q_ste, self.bias)

    @torch.no_grad()
    def freeze_for_inference(self) -> None:
        """Cache `w_q * scale` once for inference-only use.

        Computes the same value the STE forward materializes (forward value
        of `self.weight + (w_q*scale - self.weight).detach()` equals
        `w_q*scale` since the residual is wrapped in `detach()`), stores it
        as a detached non-Parameter tensor (no autograd, no state_dict
        entry), and flips `_cached_active=True` so subsequent eval-mode
        forwards skip re-quantization.

        **Must be called in eval mode** — raises `RuntimeError` otherwise.
        Codex msg 1779529701708-b4564ba8 closed a load-bearing hole: if
        freeze ran in training mode, `_cached_active=True` would set
        immediately (the `self.training` forward guard masks it then), but
        subsequent training-step weight mutations would NOT invalidate the
        cache (`train(False)` doesn't touch the flag — only `train(True)`
        clears). The next `.eval()` would then consume a stale cached
        weight. Fail-fast at freeze time eliminates the failure mode.

        Codex msg 1779528934673-1c8bedf3 scope:
        - Runtime-only cache, NOT in state_dict (no .pt schema change).
        - Master weight untouched; can re-freeze any time.
        - `train()` override invalidates the cache to prevent stale-cache
          training paths (defense in depth alongside `forward()` guard).
        """
        if self.training:
            raise RuntimeError(
                "BitLinear.freeze_for_inference() must be called in eval mode; "
                "training-mode freeze creates a stale-cache vulnerability "
                "after weight updates (codex msg 1779529701708-b4564ba8). "
                "Call `module.eval()` before `freeze_for_inference()`."
            )
        scale = self.weight.abs().mean().clamp(min=self._SCALE_EPS)
        w_q = (self.weight / scale).round().clamp(-1.0, 1.0)
        # Detached, same dtype/device as master. NOT an nn.Parameter and
        # NOT registered as a buffer — keeps state_dict bit-identical.
        self._cached_weight = (w_q * scale).detach().contiguous()
        self._cached_active = True

    def unfreeze(self) -> None:
        """Drop the cached inference weight; next forward re-quantizes.

        Use when switching back to training or when master weights have
        been mutated since the freeze.
        """
        self._cached_weight = None
        self._cached_active = False

    def train(self, mode: bool = True):
        """Invalidate the inference cache when entering training mode.

        Defense-in-depth alongside the `self.training` guard in `forward()`:
        both must agree before the cached path runs. This guarantees that
        a user calling `model.train()` after `freeze_bitlinears_for_inference()`
        cannot accidentally run the cached weight under autograd / STE.
        """
        super().train(mode)
        if mode:
            self._cached_active = False
        return self

    @torch.no_grad()
    def get_ternary_levels(self) -> Tensor:
        """Return the ternary levels {-1, 0, +1} of the quantized weight.

        Useful for the type-check test: assert all values ∈ {-1, 0, +1}
        AFTER division by scale + round + clamp. Not part of the
        forward path; backward-safe (no_grad).
        """
        scale = self.weight.abs().mean().clamp(min=self._SCALE_EPS)
        return (self.weight / scale).round().clamp(-1.0, 1.0)


def freeze_bitlinears_for_inference(module: nn.Module) -> int:
    """Walk `module` and call `freeze_for_inference()` on every BitLinear.

    Returns the count of BitLinear modules frozen — caller can assert
    this matches the expected count (e.g. 128 for HRM-Text-1.58 with
    n_layers=8, half_layers=True, 2 H-cycles × 1 + 2 × L_cycles=3 = 8
    iters × 4 layers × 4 BL/block = 128).

    Codex msg 1779528934673-1c8bedf3: must be called AFTER `model.eval()`
    so the `train()` cache invalidation doesn't fire post-freeze.
    """
    count = 0
    for m in module.modules():
        if isinstance(m, BitLinear):
            m.freeze_for_inference()
            count += 1
    return count
