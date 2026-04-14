"""Small 2D-head transformer matching the Percepta paper.

Vanilla PyTorch. The architectural constraint is `d_head=2`; everything
else is standard `nn.MultiheadAttention` + gated-ReLU FFN + causal mask +
learned positional embeddings. What makes this a computer is the weights,
not the architecture — the weights are compiled from a gate graph (see
`compile.py`), not trained.

Supports `use_hard_max=True` to replace softmax attention with argmax
attention. Hard-max is what the paper's analytical correctness proofs
assume; softmax with large scaling temperature approximates it to
arbitrary precision but is slower to reason about.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Small2DConfig:
    """Configuration for Small2DTransformer.

    Paper defaults: d_model=36, n_heads=18, n_layers=7, d_ffn=36.
    We use tiny defaults (d_model=8, n_heads=4, n_layers=2) for the
    first prototype so we can test compile/execute cycles in ms on CPU.
    """
    vocab_size: int = 32
    d_model: int = 8
    n_heads: int = 4
    n_layers: int = 2
    d_ffn: int = 8
    max_len: int = 256
    use_hard_max: bool = True

    @property
    def d_head(self) -> int:
        assert self.d_model % self.n_heads == 0, "d_model must divide n_heads"
        return self.d_model // self.n_heads


class Small2DTransformer(nn.Module):
    """Minimal transformer compatible with compiled-weight execution.

    Matches the paper's `VanillaTransformer` snippet (doc 02 §8): learned
    token + position embeddings, MultiheadAttention layers, gated ReLU
    FFN, causal mask, linear head.

    `use_hard_max=True` replaces softmax with argmax — each query picks
    exactly one past position, matching the paper's analytical-correctness
    regime.
    """

    def __init__(self, config: Small2DConfig):
        super().__init__()
        self.config = config
        self.tok = nn.Embedding(config.vocab_size, config.d_model)
        self.pos = nn.Embedding(config.max_len, config.d_model)

        # Split QKV/out/FFN into pieces so a compiler can populate them per-head.
        self.W_qkv = nn.ModuleList([
            nn.Linear(config.d_model, 3 * config.d_model, bias=False)
            for _ in range(config.n_layers)
        ])
        self.W_out = nn.ModuleList([
            nn.Linear(config.d_model, config.d_model, bias=False)
            for _ in range(config.n_layers)
        ])
        self.ff_in = nn.ModuleList([
            nn.Linear(config.d_model, 2 * config.d_ffn, bias=False)
            for _ in range(config.n_layers)
        ])
        self.ff_out = nn.ModuleList([
            nn.Linear(config.d_ffn, config.d_model, bias=False)
            for _ in range(config.n_layers)
        ])
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def _attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   hard_max: bool) -> torch.Tensor:
        """Causal multi-head attention. q,k,v: (B, H, S, D_head)."""
        B, H, S, Dh = q.shape
        scores = torch.einsum("bhid,bhjd->bhij", q, k)  # (B, H, S, S)
        # Causal mask: position i can only see j <= i.
        mask = torch.triu(
            torch.ones(S, S, dtype=torch.bool, device=q.device), diagonal=1
        )
        scores = scores.masked_fill(mask, float("-inf"))
        if hard_max:
            # Replace softmax with one-hot over argmax. Tie-breaks: first match.
            idx = scores.argmax(dim=-1, keepdim=True)  # (B, H, S, 1)
            weights = torch.zeros_like(scores)
            weights.scatter_(-1, idx, 1.0)
        else:
            weights = F.softmax(scores, dim=-1)
        out = torch.einsum("bhij,bhjd->bhid", weights, v)  # (B, H, S, D_head)
        return out

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """idx: (B, S). Returns logits (B, S, vocab)."""
        B, S = idx.shape
        cfg = self.config
        pos_idx = torch.arange(S, device=idx.device)
        x = self.tok(idx) + self.pos(pos_idx)
        for layer in range(cfg.n_layers):
            qkv = self.W_qkv[layer](x)  # (B, S, 3*D)
            qkv = qkv.reshape(B, S, 3, cfg.n_heads, cfg.d_head)
            q, k, v = qkv.permute(2, 0, 3, 1, 4)  # 3 × (B, H, S, Dh)
            attn = self._attention(q, k, v, hard_max=cfg.use_hard_max)
            attn = attn.transpose(1, 2).reshape(B, S, cfg.d_model)  # (B, S, D)
            x = x + self.W_out[layer](attn)
            gate, val = self.ff_in[layer](x).chunk(2, dim=-1)
            x = x + self.ff_out[layer](F.relu(gate) * val)
        return self.head(x)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
