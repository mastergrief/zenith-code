"""Combined-extension Small2DTransformer — D2 + D3 + D5 in one class.

The three separate subclasses (TracedSmall2DTransformer,
MixedGeometrySmall2DTransformer, RecurrentSmall2DTransformer) each add
one capability via override-forward. To use all three at once we need
one combined override.

This class is the "Hybrid v2" substrate target: per-layer geometry
selection (D3), variable iteration depth (D5), optional trace emission
(D2), all on the same Small2DTransformer base.

Backward compatible: with default config (uniform Euclidean, 1
iteration, no trace) it's equivalent to the parent class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from calm.llm_computer.computation_trace import (
    ComputationTrace, capture_layer,
)
from calm.llm_computer.mixed_geometry import GEOMETRY_DISPATCH
from calm.llm_computer.model import Small2DConfig, Small2DTransformer


@dataclass
class CombinedConfig(Small2DConfig):
    """Small2DConfig + D3 (geometries) + D5 (recurrence)."""
    layer_geometries: Optional[list[str]] = None
    default_iterations: int = 1
    max_iterations: int = 8


class CombinedSmall2DTransformer(Small2DTransformer):
    """Substrate with mixed-geometry attention, recurrent iteration,
    and optional computation trace."""

    def __init__(self, config: CombinedConfig):
        super().__init__(config)
        if config.layer_geometries is not None:
            assert len(config.layer_geometries) == config.n_layers, (
                f"layer_geometries must have length {config.n_layers}, "
                f"got {len(config.layer_geometries)}"
            )
            for g in config.layer_geometries:
                assert g in GEOMETRY_DISPATCH, f"unknown geometry {g!r}"

    def _attention_with_geometry(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        hard_max: bool, geometry: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (attention_output, attention_weights) — weights kept for trace."""
        B, H, S, Dh = q.shape
        scores = GEOMETRY_DISPATCH[geometry](q, k)
        mask = torch.triu(
            torch.ones(S, S, dtype=torch.bool, device=q.device), diagonal=1,
        )
        scores = scores.masked_fill(mask, float("-inf"))
        if hard_max:
            idx = scores.argmax(dim=-1, keepdim=True)
            weights = torch.zeros_like(scores)
            weights.scatter_(-1, idx, 1.0)
        else:
            weights = F.softmax(scores, dim=-1)
        out = torch.einsum("bhij,bhjd->bhid", weights, v)
        return out, weights

    def forward(self, idx: torch.Tensor,
                n_iterations: int | None = None,
                trace: Optional[ComputationTrace] = None) -> torch.Tensor:
        cfg: CombinedConfig = self.config  # type: ignore[assignment]

        # Resolve iteration count
        n = n_iterations if n_iterations is not None else cfg.default_iterations
        n = max(1, min(n, cfg.max_iterations))

        # Resolve geometries
        geoms = cfg.layer_geometries or ["euclidean"] * cfg.n_layers

        B, S = idx.shape
        pos_idx = torch.arange(S, device=idx.device)
        x = self.tok(idx) + self.pos(pos_idx)

        if trace is not None:
            trace.sequence_length = S
            trace.iterations = n

        for _iteration in range(n):
            for layer in range(cfg.n_layers):
                qkv = self.W_qkv[layer](x)
                qkv = qkv.reshape(B, S, 3, cfg.n_heads, cfg.d_head)
                q, k, v = qkv.permute(2, 0, 3, 1, 4)
                attn, attn_weights = self._attention_with_geometry(
                    q, k, v,
                    hard_max=cfg.use_hard_max,
                    geometry=geoms[layer],
                )
                attn = attn.transpose(1, 2).reshape(B, S, cfg.d_model)
                x = x + self.W_out[layer](attn)
                gate, val = self.ff_in[layer](x).chunk(2, dim=-1)
                ffn_pre = F.relu(gate) * val
                x = x + self.ff_out[layer](ffn_pre)

                if trace is not None and _iteration == n - 1:
                    # Capture only the last iteration to avoid trace bloat.
                    capture_layer(
                        trace,
                        layer_idx=layer,
                        attention_weights=attn_weights,
                        ffn_pre_activation=ffn_pre,
                        fast_weight_state=None,
                        geometry=geoms[layer],
                    )

        return self.head(x)
