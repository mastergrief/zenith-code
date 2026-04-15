"""Small2DTransformer with per-layer attention mode — compiled programs
and Gemma-equivalent attention in ONE d_head=2 tensor.

Extends `Small2DTransformer` with a per-layer mode flag:
  - "single": standard d_head=2 per-sub-head softmax (compiled programs)
  - "grouped": n_groups heads of (group_size * 2) equivalent d_head,
    with scores summed pre-softmax (Gemma / Llama-equivalent)

The SAME Q/K/V weight tensors feed both modes. A layer configured as
"grouped" reinterprets its sub-heads as n_groups × group_size × 2
structure; a "single" layer treats every sub-head independently.

This is the architectural path to putting Gemma INSIDE a single
Small2DTransformer tensor without sacrificing compiled-card
functionality. One state_dict. One d_head=2 invariant. One file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from calm.llm_computer.grouped_attention import (
    grouped_attention, grouped_attention_single_head_mode,
)
from calm.llm_computer.model import Small2DConfig, Small2DTransformer


@dataclass
class GroupedSmall2DConfig(Small2DConfig):
    """Extends Small2DConfig with per-layer attention mode + grouping.

    Attributes:
        layer_modes: tuple of "single" | "grouped" per layer (length n_layers).
            None → all layers use "single" (backward compatible).
        layer_n_groups: tuple of int per layer (length n_layers). For
            "single" layers this is ignored; for "grouped" layers it's
            the equivalent n_heads of a larger-d_head attention.
        layer_group_sizes: tuple of int per layer. For grouped layers,
            group_size × 2 = equivalent d_head. n_heads / n_groups must
            equal group_size (by the decomposition math).
    """
    layer_modes: Optional[tuple[str, ...]] = None
    layer_n_groups: Optional[tuple[int, ...]] = None
    layer_group_sizes: Optional[tuple[int, ...]] = None
    layer_hard_max: Optional[tuple[bool, ...]] = None

    def __post_init__(self):
        if self.layer_modes is None:
            object.__setattr__(self, "layer_modes",
                                 tuple(["single"] * self.n_layers))
        if self.layer_n_groups is None:
            object.__setattr__(self, "layer_n_groups",
                                 tuple([1] * self.n_layers))
        if self.layer_group_sizes is None:
            object.__setattr__(self, "layer_group_sizes",
                                 tuple([self.n_heads] * self.n_layers))
        if self.layer_hard_max is None:
            object.__setattr__(self, "layer_hard_max",
                                 tuple([False] * self.n_layers))
        # Validate per-layer config
        assert len(self.layer_modes) == self.n_layers
        assert len(self.layer_n_groups) == self.n_layers
        assert len(self.layer_group_sizes) == self.n_layers
        assert len(self.layer_hard_max) == self.n_layers
        for i, (mode, n_g, g_s) in enumerate(zip(
            self.layer_modes, self.layer_n_groups, self.layer_group_sizes,
        )):
            assert mode in ("single", "grouped"), (
                f"layer {i}: unknown mode {mode!r}"
            )
            if mode == "grouped":
                assert n_g * g_s == self.n_heads, (
                    f"layer {i}: grouped needs n_groups {n_g} × "
                    f"group_size {g_s} = n_heads {self.n_heads}"
                )


class GroupedSmall2DTransformer(Small2DTransformer):
    """Small2DTransformer where each layer picks 'single' or 'grouped'
    attention at forward time. Compiled programs use single; Gemma-
    equivalent layers use grouped.

    Weight shapes are identical to the parent — attention mode only
    affects the score-computation + softmax step.
    """

    def __init__(self, config: GroupedSmall2DConfig):
        # Instantiate parent with the base Small2DConfig fields
        super().__init__(config)
        # Store extended config separately
        self._grouped_config = config

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        cfg = self._grouped_config
        B, S = idx.shape
        pos_idx = torch.arange(S, device=idx.device)
        x = self.tok(idx) + self.pos(pos_idx)

        mask = torch.triu(
            torch.ones(S, S, dtype=torch.bool, device=x.device), diagonal=1,
        )

        for layer in range(cfg.n_layers):
            qkv = self.W_qkv[layer](x)
            qkv = qkv.reshape(B, S, 3, cfg.n_heads, cfg.d_head)
            q, k, v = qkv.permute(2, 0, 3, 1, 4)
            # q, k, v: (B, n_heads, S, d_head=2)
            # Convert to (B, S, n_heads, 2) for grouped_attention signature
            q_bh = q.transpose(1, 2)
            k_bh = k.transpose(1, 2)
            v_bh = v.transpose(1, 2)

            mode = cfg.layer_modes[layer]
            if mode == "grouped":
                # Grouped mode IS Gemma-equivalent — scale by equivalent d_head
                attn = grouped_attention(
                    q_bh, k_bh, v_bh,
                    n_groups=cfg.layer_n_groups[layer],
                    group_size=cfg.layer_group_sizes[layer],
                    mask=mask,
                )
            else:  # single
                # Single mode matches vanilla Small2DTransformer: NO scale
                # (vanilla's `_attention` doesn't divide by sqrt(d_head)).
                # See calm/llm_computer/model.py:_attention
                attn = grouped_attention_single_head_mode(
                    q_bh, k_bh, v_bh, mask=mask, scale=1.0,
                    hard_max=cfg.layer_hard_max[layer],
                )
            # attn is already (B, S, n_heads, d_head=2); flatten directly.
            attn = attn.reshape(B, S, cfg.d_model)
            x = x + self.W_out[layer](attn)
            gate, val = self.ff_in[layer](x).chunk(2, dim=-1)
            x = x + self.ff_out[layer](F.relu(gate) * val)

        return self.head(x)
