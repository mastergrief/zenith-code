"""LoRA adapters for tq4-frozen linear layers.

Standard low-rank adaptation: for a frozen linear layer with weight W,
the effective transformation becomes W + (alpha/rank) * B @ A, where
A ∈ R^(rank, in) and B ∈ R^(out, rank) are trainable rank-r matrices.

Since tq4 weights don't accumulate gradient (straight-through dequant),
LoRA adapters provide the ONLY trainable surface for QLoRA-style
fine-tuning.

This module ships:
  LoRAAdapter(in, out, rank, alpha) — the trainable low-rank addon
  LoRATq4Linear(base: Tq4Linear, rank, alpha) — wraps a frozen tq4
    linear with a LoRA adapter; forward returns base(x) + adapter(x)
  merge_lora_into_base(wrapper) — fold learned LoRA into the base
    weight tensor (FP32) and re-quantize. Optional but useful for
    deployment when you want a single tq4 artifact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from calm.llm_computer.tq4_torch import Tq4Linear, quantize_tq4


class LoRAAdapter(nn.Module):
    """Low-rank adapter: out = (alpha/rank) * B @ A @ x.

    `A` initialized with Kaiming uniform, `B` zero-initialized. That
    init gives LoRA output = 0 at the start of training (no disruption
    to the base model). Gradient flows to both A and B.
    """

    def __init__(self, in_features: int, out_features: int,
                 rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        # A: (rank, in), B: (out, rank)
        self.A = nn.Parameter(torch.empty(rank, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, rank))
        # Kaiming uniform on A (same init as nn.Linear)
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    @property
    def scaling(self) -> float:
        return self.alpha / self.rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., in_features) → (..., out_features)."""
        # Decompose: (x @ A.T) @ B.T, avoids materializing full W
        mid = x @ self.A.T   # (..., rank)
        return self.scaling * (mid @ self.B.T)  # (..., out_features)


class LoRATq4Linear(nn.Module):
    """Wraps a frozen Tq4Linear with a trainable LoRA adapter.

    Base weight is quantized (tq4, frozen codes). LoRA adapter adds
    a trainable low-rank correction. Forward:
        y = base(x) + adapter(x)
    """

    def __init__(self, base: Tq4Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base = base
        # Ensure base is frozen
        for p in self.base.parameters():
            p.requires_grad = False
        self.adapter = LoRAAdapter(
            in_features=base.in_features,
            out_features=base.out_features,
            rank=rank, alpha=alpha,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.adapter(x)

    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def merge_lora_into_base(wrapper: LoRATq4Linear) -> None:
    """Merge the LoRA adapter into the frozen tq4 base, re-quantizing
    the result. After merging, the adapter is zeroed (further training
    would need a fresh adapter).

    Uses current Pi and centroids of the base layer. Result may have
    slightly different numerics than a live LoRATq4Linear forward due
    to re-quantization.
    """
    base = wrapper.base
    adapter = wrapper.adapter
    # Dequantize base to FP32
    from calm.llm_computer.tq4_torch import Tq4Tensor, dequantize_tq4
    q = Tq4Tensor(
        qs=base._qs, d=base._d, shape=(base.out_features, base.in_features),
    )
    base_fp = dequantize_tq4(q, pi=base._pi, centroids=base._centroids)
    # Add scaled LoRA product: (alpha/rank) * B @ A
    lora_delta = adapter.scaling * (adapter.B @ adapter.A)
    merged = base_fp + lora_delta.detach()
    # Re-quantize
    base.load_weight(merged)
    # Zero the adapter (A keeps Kaiming init for future training; B → 0)
    with torch.no_grad():
        adapter.B.zero_()
