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
        w_q_ste, _ = self.quantize_weight()
        return F.linear(input, w_q_ste, self.bias)

    @torch.no_grad()
    def get_ternary_levels(self) -> Tensor:
        """Return the ternary levels {-1, 0, +1} of the quantized weight.

        Useful for the type-check test: assert all values ∈ {-1, 0, +1}
        AFTER division by scale + round + clamp. Not part of the
        forward path; backward-safe (no_grad).
        """
        scale = self.weight.abs().mean().clamp(min=self._SCALE_EPS)
        return (self.weight / scale).round().clamp(-1.0, 1.0)
