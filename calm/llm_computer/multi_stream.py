"""Multi-stream residual — Level 2 bus redesign.

The single-d_model residual stream is a flat namespace. Every card
shares one channel budget. Channel masking (shipped in commit ec39687)
enforces separation by gradient hook, but the streams still share the
same physical d_model vector — interference is convention + hook
enforcement, not physical isolation.

Multi-stream fixes this. A `MultiStreamTransformer` has K parallel
residual streams, each with its own d_model, n_heads, layer stack.
Cards on stream 0 can't corrupt cards on stream 1 by construction
because their parameters and activations are in physically separate
tensors.

Minimal MVP design (one join layer shape: concat-at-head):
  - Each stream has its own tok_embed, pos_embed, per-layer W_qkv/
    W_out/ff_in/ff_out.
  - Input tokens go to every stream (each embeds them independently).
  - After all layers, per-stream residuals at the final position are
    concatenated and fed through a shared LM head.
  - Joins between streams (cross-stream data flow at mid-layer) are
    NOT in the MVP — parallel processing only.

This is sufficient for the "isolation" hypothesis: stream 0 adder +
stream 1 trained LM, train the LM without breaking the adder, no
channel mask needed. Joins come in a later round if/when composition
across streams is required.

Save/reload: single `.pt` — all streams share the same state_dict
file. They're named sub-modules inside one `nn.Module`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class StreamSpec:
    """One stream's shape. d_head is derived as d_model // n_heads; must
    equal 2 per substrate invariant."""
    name: str
    d_model: int
    n_heads: int
    d_ffn: int

    @property
    def d_head(self) -> int:
        assert self.d_model % self.n_heads == 0, (
            f"d_model {self.d_model} not divisible by n_heads {self.n_heads}"
        )
        return self.d_model // self.n_heads


@dataclass
class MultiStreamConfig:
    """Config for a multi-stream transformer."""
    streams: tuple[StreamSpec, ...]
    n_layers: int
    vocab_size: int
    max_len: int
    use_hard_max: bool = False

    def __post_init__(self):
        # Validate substrate invariant
        for s in self.streams:
            assert s.d_head == 2, (
                f"stream {s.name!r}: d_head must be 2, got {s.d_head}"
            )
        # Validate unique stream names
        names = [s.name for s in self.streams]
        assert len(set(names)) == len(names), (
            f"duplicate stream names: {names}"
        )

    @property
    def total_d(self) -> int:
        return sum(s.d_model for s in self.streams)

    def stream_by_name(self, name: str) -> StreamSpec:
        for s in self.streams:
            if s.name == name:
                return s
        raise KeyError(f"no stream named {name!r}")


class _StreamStack(nn.Module):
    """Per-stream parameters: token embedding, position embedding, and
    n_layers of (W_qkv, W_out, ff_in, ff_out)."""

    def __init__(self, spec: StreamSpec, n_layers: int,
                 vocab_size: int, max_len: int, use_hard_max: bool):
        super().__init__()
        self.spec = spec
        self.use_hard_max = use_hard_max
        self.tok = nn.Embedding(vocab_size, spec.d_model)
        self.pos = nn.Embedding(max_len, spec.d_model)
        self.W_qkv = nn.ModuleList([
            nn.Linear(spec.d_model, 3 * spec.d_model, bias=False)
            for _ in range(n_layers)
        ])
        self.W_out = nn.ModuleList([
            nn.Linear(spec.d_model, spec.d_model, bias=False)
            for _ in range(n_layers)
        ])
        self.ff_in = nn.ModuleList([
            nn.Linear(spec.d_model, 2 * spec.d_ffn, bias=False)
            for _ in range(n_layers)
        ])
        self.ff_out = nn.ModuleList([
            nn.Linear(spec.d_ffn, spec.d_model, bias=False)
            for _ in range(n_layers)
        ])

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Single-stream forward pass. Returns final residual (B, S, d_model)."""
        B, S = idx.shape
        pos_idx = torch.arange(S, device=idx.device)
        x = self.tok(idx) + self.pos(pos_idx)
        spec = self.spec
        for layer in range(len(self.W_qkv)):
            qkv = self.W_qkv[layer](x)
            qkv = qkv.reshape(B, S, 3, spec.n_heads, spec.d_head)
            q, k, v = qkv.permute(2, 0, 3, 1, 4)
            scores = torch.einsum("bhid,bhjd->bhij", q, k) / (spec.d_head ** 0.5)
            mask = torch.triu(
                torch.ones(S, S, dtype=torch.bool, device=x.device),
                diagonal=1,
            )
            scores = scores.masked_fill(mask, float("-inf"))
            if self.use_hard_max:
                idx_max = scores.argmax(dim=-1, keepdim=True)
                weights = torch.zeros_like(scores)
                weights.scatter_(-1, idx_max, 1.0)
            else:
                weights = F.softmax(scores, dim=-1)
            attn = torch.einsum("bhij,bhjd->bhid", weights, v)
            attn = attn.transpose(1, 2).reshape(B, S, spec.d_model)
            x = x + self.W_out[layer](attn)
            gate, val = self.ff_in[layer](x).chunk(2, dim=-1)
            x = x + self.ff_out[layer](F.relu(gate) * val)
        return x


class MultiStreamTransformer(nn.Module):
    """Parallel residual streams sharing input tokens and output head.

    Each stream has its own tok/pos embeddings and layer stack, all
    living in this module's state_dict. At forward time, each stream
    processes the input independently; final residuals concatenate and
    feed a shared LM head.
    """

    def __init__(self, config: MultiStreamConfig):
        super().__init__()
        self.config = config
        self.streams = nn.ModuleDict({
            s.name: _StreamStack(
                s, config.n_layers, config.vocab_size, config.max_len,
                config.use_hard_max,
            )
            for s in config.streams
        })
        # Shared head: takes concatenation of all stream final residuals
        self.head = nn.Linear(config.total_d, config.vocab_size, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns (B, S, vocab) logits.

        Each stream processes `idx` independently via its own parameters.
        Final per-stream residuals are concatenated along the channel
        dimension, then projected to vocab via the shared head.
        """
        # Run each stream in declaration order and concat per-position residuals
        stream_outs = [
            self.streams[s.name](idx) for s in self.config.streams
        ]
        concat = torch.cat(stream_outs, dim=-1)  # (B, S, total_d)
        return self.head(concat)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def stream_params(self, stream_name: str) -> list[nn.Parameter]:
        """All parameters belonging to a named stream (not the head)."""
        return list(self.streams[stream_name].parameters())


def build_empty_multistream(cfg: MultiStreamConfig) -> MultiStreamTransformer:
    """Zero-initialize a multi-stream transformer. Useful when you want
    to state-dict-transfer compiled weights into a specific stream."""
    m = MultiStreamTransformer(cfg)
    with torch.no_grad():
        for p in m.parameters():
            p.zero_()
    return m


def freeze_stream(model: MultiStreamTransformer, stream_name: str) -> int:
    """Set requires_grad=False on all parameters of a named stream.

    Does NOT freeze the shared head — use `freeze_head` or custom hooks
    for per-row head freezing.

    Returns: number of parameters frozen.
    """
    frozen = 0
    for p in model.streams[stream_name].parameters():
        p.requires_grad = False
        frozen += p.numel()
    return frozen


def freeze_head(model: MultiStreamTransformer) -> int:
    """Freeze the shared output head entirely."""
    frozen = 0
    for p in model.head.parameters():
        p.requires_grad = False
        frozen += p.numel()
    return frozen


def trainable_param_count(model: MultiStreamTransformer) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
