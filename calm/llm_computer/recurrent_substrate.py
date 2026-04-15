"""Direction 5 — recurrent substrate with iteration budget.

The substrate currently runs each layer once per forward pass. This
mirrors a standard transformer. But there's no architectural reason a
2-layer model can't run its 2 layers MULTIPLE times on the same residual
stream within one forward pass — that's exactly what HRM's L/H nested
loops do (Wang et al. 2025) and what the Universal Transformer
(Dehghani et al. 2018) does.

For our substrate: expose `n_iterations` as a forward-time parameter.
Each iteration is "another pass through the layers, refining the residual
stream." More iterations = more thinking time on the same input. Default
n_iterations=1 preserves all current behavior.

Future extension (not in MVP): the model emits a `<|think_more|>` token
that triggers an extra iteration without producing visible output. The
inference loop catches the token and re-iterates instead of advancing.
That requires training the model to use the token wisely (RL or supervised
on labeled "needed more thought" examples). MVP just exposes the knob.

Why this is structurally different from "more layers": with n_iterations,
the SAME layers are re-applied with shared weights. The residual stream
accumulates refinement. With more layers, weights are independent. Shared
weights = inductive bias toward "iterate to convergence" (HRM-style),
independent weights = inductive bias toward "stack of distinct
transformations" (standard transformer).

Composes with Direction 2 (computation traces): trace.iterations records
how many were actually run.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from calm.llm_computer.model import Small2DConfig, Small2DTransformer


@dataclass
class RecurrentConfig(Small2DConfig):
    """Small2DConfig + max iteration budget.

    `default_iterations` is the iteration count when forward is called
    without an explicit `n_iterations` kwarg. Default 1 = parent behavior.
    `max_iterations` caps any explicit request — prevents runaway compute
    if a future learned controller emits too many `<|think_more|>` tokens.
    """
    default_iterations: int = 1
    max_iterations: int = 16


class RecurrentSmall2DTransformer(Small2DTransformer):
    """Iterates the layer stack n_iterations times within one forward pass.

    The same `W_qkv[layer]`, `W_out[layer]`, `ff_in[layer]`, `ff_out[layer]`
    matrices are applied iteratively — weights are shared across iterations
    (the iteration count doesn't multiply parameter count).

    Backward compatible: `forward(idx)` with `default_iterations=1` matches
    parent class exactly.
    """

    def __init__(self, config: RecurrentConfig):
        super().__init__(config)

    def forward(self, idx: torch.Tensor,
                n_iterations: int | None = None) -> torch.Tensor:
        cfg: RecurrentConfig = self.config  # type: ignore[assignment]
        n = n_iterations if n_iterations is not None else cfg.default_iterations
        n = max(1, min(n, cfg.max_iterations))

        # n=1 → exactly parent behavior (early return preserves bit-equality).
        if n == 1:
            return super().forward(idx)

        B, S = idx.shape
        pos_idx = torch.arange(S, device=idx.device)
        x = self.tok(idx) + self.pos(pos_idx)

        for _iteration in range(n):
            for layer in range(cfg.n_layers):
                qkv = self.W_qkv[layer](x)
                qkv = qkv.reshape(B, S, 3, cfg.n_heads, cfg.d_head)
                q, k, v = qkv.permute(2, 0, 3, 1, 4)
                attn = self._attention(q, k, v, hard_max=cfg.use_hard_max)
                attn = attn.transpose(1, 2).reshape(B, S, cfg.d_model)
                x = x + self.W_out[layer](attn)
                gate, val = self.ff_in[layer](x).chunk(2, dim=-1)
                x = x + self.ff_out[layer](F.relu(gate) * val)

        return self.head(x)
