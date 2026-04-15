"""Gemma-style architectural upgrades for Small2DTransformer substrate.

Adopts Gemma 4 E2B's key architectural innovations as OPT-IN features
on our `CombinedSmall2DTransformer`. Each feature is orthogonal to the
`d_head=2` invariant — they modify attention-masking, activation
functions, and normalization, not head dimensions. Compiled programs
(which depend on ReGLU + no norm + full attention) stay compatible by
default; trained streams can opt into the upgrades per-layer.

Upgrades:

  1. GeGLU (vs ReGLU): gate uses GELU instead of ReLU. Smoother
     gradients at the cost of slightly more compute. Standard in modern
     LLMs (PaLM, Gemma, Llama-style GeGLU/SwiGLU family).

  2. Sliding window attention: each query attends only to the last W
     tokens. Lets us scale context without quadratic memory blowup.
     Gemma 4 alternates sliding (window=4096) and full attention
     layers; here we expose per-layer window config.

  3. RMSNorm: `x / sqrt(mean(x²) + eps) * weight`. Simpler than
     LayerNorm, no mean-centering, often better training stability.
     Used in Gemma, Llama, T5. Pre-norm architecture.

All three are config-flag opt-ins. With defaults (None/False), the
CombinedSmall2DTransformer's behavior is unchanged — compiled programs
continue to work exactly as before.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----- RMSNorm -----

class RMSNorm(nn.Module):
    """Root mean square layer normalization.

    `y = x / sqrt(mean(x²) + eps) * weight`

    No mean-centering (vs LayerNorm). Cheaper. Commonly used in Gemma,
    Llama, T5. Parameter count: just a `d_model`-sized weight vector.
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute RMS along the last dimension
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


# ----- Sliding window attention mask -----

def sliding_window_mask(
    seq_len: int, window: int,
    device: torch.device,
) -> torch.Tensor:
    """Build a causal mask where each query attends only to the previous
    `window` positions (inclusive of self). Returns a (S, S) bool tensor
    where True = masked-out (not attended).

    Example (seq_len=5, window=2):
        q=0 can see: {0}          (standard causal)
        q=1 can see: {0, 1}
        q=2 can see: {1, 2}        (window kicks in, 0 is beyond)
        q=3 can see: {2, 3}
        q=4 can see: {3, 4}
    """
    # Standard causal: mask out upper triangle
    causal = torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
        diagonal=1,
    )
    # Window: mask out positions farther back than `window - 1`
    # For query i, key j < i - window + 1 → masked
    # That is: i - j > window - 1
    i = torch.arange(seq_len, device=device).unsqueeze(1)  # (S, 1)
    j = torch.arange(seq_len, device=device).unsqueeze(0)  # (1, S)
    beyond_window = (i - j) > (window - 1)
    return causal | beyond_window


# ----- GeGLU activation -----

def geglu(gate: torch.Tensor, val: torch.Tensor) -> torch.Tensor:
    """GeGLU: `GELU(gate) * val`. Smoother gradient than ReGLU's
    `ReLU(gate) * val` at approximately the same compute cost.
    Used in Gemma, PaLM, and LLaMA-style GeGLU/SwiGLU family.
    """
    return F.gelu(gate) * val


# ----- Config -----

@dataclass
class GemmaStyleConfig:
    """Per-layer opt-in flags for Gemma-style architectural upgrades.

    Each list is length n_layers; a None entry means "use default
    (ReGLU / full attention / no norm)". This lets you mix compiled
    layers (defaults) with trained layers (upgrades) in one model.

    Attributes:
        use_geglu_per_layer: tuple of bool per layer. True → GeGLU,
            False → ReGLU. Length must match n_layers. None (default)
            means all ReGLU (backward compatible).
        attention_windows: tuple of Optional[int] per layer. Int →
            sliding window of that size. None → full causal attention.
        rmsnorm_per_layer: tuple of bool per layer. True → apply
            RMSNorm before attention + before FFN (pre-norm). False →
            no normalization.
    """
    use_geglu_per_layer: Optional[tuple[bool, ...]] = None
    attention_windows: Optional[tuple[Optional[int], ...]] = None
    rmsnorm_per_layer: Optional[tuple[bool, ...]] = None

    def validate(self, n_layers: int) -> None:
        if self.use_geglu_per_layer is not None:
            assert len(self.use_geglu_per_layer) == n_layers
        if self.attention_windows is not None:
            assert len(self.attention_windows) == n_layers
        if self.rmsnorm_per_layer is not None:
            assert len(self.rmsnorm_per_layer) == n_layers

    def geglu_at(self, layer: int) -> bool:
        if self.use_geglu_per_layer is None:
            return False
        return self.use_geglu_per_layer[layer]

    def window_at(self, layer: int) -> Optional[int]:
        if self.attention_windows is None:
            return None
        return self.attention_windows[layer]

    def rmsnorm_at(self, layer: int) -> bool:
        if self.rmsnorm_per_layer is None:
            return False
        return self.rmsnorm_per_layer[layer]


# ----- Per-layer RMSNorm bank -----

class LayerwiseRMSNorm(nn.Module):
    """One RMSNorm per layer index, accessed by `norms[layer_idx](x)`.
    Allocated only for layers where rmsnorm_at(layer) is True; others
    get identity (no-op).
    """

    def __init__(self, n_layers: int, d_model: int,
                 enabled_per_layer: list[bool]):
        super().__init__()
        self.n_layers = n_layers
        # Create an RMSNorm only for enabled layers; others are identity
        self._norms = nn.ModuleList([
            RMSNorm(d_model) if enabled_per_layer[i] else nn.Identity()
            for i in range(n_layers)
        ])

    def forward(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        return self._norms[layer_idx](x)
