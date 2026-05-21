"""TernaryLinear — BitNet b1.58-style ternary-weight Linear with FP master + STE.

Per TRM-1.58 (Slice 13) locked contract: native W1.58A8 BitNet-style RDT-v2,
trained natively from step zero (NOT post-training quantization).

Forward weights are constrained to {-1, 0, +1} via absmean quantization:
    scale = mean(|W|)
    W_q = clip(round(W / (scale + eps)), -1, +1) * scale

Backward uses straight-through estimator (STE): gradient flows to the FP/BF16
master weight `self.weight` as if quantization were identity. This is the
standard BitNet b1.58 training contract (Ma 2024).

Activations are NOT quantized in this primitive — Gate A keeps activations
in BF16/FP32 for finite-falsifier purposes. Activation int8 absmax (the "A8"
in W1.58A8) lands in a separate gate-B/C slice once the kernel arc proves
out.

Drop-in nn.Linear replacement: same constructor signature, same forward
contract. Bias stays FP.

Wired via `DeltaNetConfig.use_ternary_bulk` flag in delta_rule.py — when set,
W_qkv / W_out / ff_in / ff_out + H-bank mirrors swap to TernaryLinear at
build time. Mechanism-critical projections (copy_gate / beta_head /
attn_gate / RMSNorm / halt / embeddings / head) stay FP per the locked
TRM-1.58 scope-out.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def absmean_ternary_quantize(weight: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Absmean ternary quantization per BitNet b1.58 (Ma 2024).

    Returns the quantized weight tensor (same shape, same dtype) where every
    element is one of {-scale, 0, +scale} with scale = mean(|W|).
    """
    scale = weight.abs().mean().clamp_min(eps)
    w_norm = weight / scale
    w_clip = torch.clamp(torch.round(w_norm), -1.0, 1.0)
    return w_clip * scale


class TernaryLinear(nn.Module):
    """nn.Linear-equivalent with ternary forward weights + FP master + STE.

    Forward:
        w_q = absmean_ternary_quantize(self.weight)
        w_ste = self.weight + (w_q - self.weight).detach()  # STE
        y = F.linear(x, w_ste, self.bias)

    The STE ensures gradients flow into `self.weight` (FP master) as if
    quantization were identity, while the forward output uses the
    quantized weight.

    Bias stays FP (small, mechanism-affecting in some PT/DT contexts).

    Drop-in for nn.Linear: same constructor signature, same `.weight` /
    `.bias` Parameter attributes, same state_dict keys.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Match nn.Linear default init so checkpoint loads from FP run-in
        # (if ever needed) work. Kaiming uniform with a=sqrt(5).
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_q = absmean_ternary_quantize(self.weight)
        # Straight-through estimator: forward uses w_q, backward passes
        # gradient through to self.weight as if quantization were identity.
        w_ste = self.weight + (w_q - self.weight).detach()
        return F.linear(x, w_ste, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, ternary=True"
        )

    @torch.no_grad()
    def quantized_weight(self) -> torch.Tensor:
        """Return the current ternary forward weight (no grad)."""
        return absmean_ternary_quantize(self.weight)

    @torch.no_grad()
    def sparsity(self) -> float:
        """Fraction of weights at the zero ternary level. Telemetry only."""
        w_q = absmean_ternary_quantize(self.weight)
        return (w_q == 0).float().mean().item()
