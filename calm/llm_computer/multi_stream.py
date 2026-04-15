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


@dataclass(frozen=True)
class JoinSpec:
    """Cross-stream data flow: after layer `at_layer` completes in every
    stream, project `from_stream`'s residual into `to_stream`'s channel
    space and add it to `to_stream`'s residual before layer `at_layer+1`.

    Join is a learned `nn.Linear(from_d_model, to_d_model, bias=False)`.
    Multiple joins can target the same stream at the same layer; they
    accumulate additively.

    Attributes:
        from_stream: name of the stream supplying data.
        to_stream: name of the stream receiving data.
        at_layer: layer index AFTER which the join fires (0..n_layers-1).
            Join at layer k means "read from_stream's post-layer-k
            residual, add to to_stream's residual before layer k+1".
        name: optional unique identifier for this join. Defaults to
            f"{from_stream}_to_{to_stream}_at{at_layer}". Different
            joins must have different names.
    """
    from_stream: str
    to_stream: str
    at_layer: int
    name: str = ""

    @property
    def key(self) -> str:
        if self.name:
            return self.name
        return f"{self.from_stream}_to_{self.to_stream}_at{self.at_layer}"


@dataclass
class MultiStreamConfig:
    """Config for a multi-stream transformer."""
    streams: tuple[StreamSpec, ...]
    n_layers: int
    vocab_size: int
    max_len: int
    use_hard_max: bool = False
    joins: tuple[JoinSpec, ...] = field(default_factory=tuple)

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
        # Validate joins reference existing streams and valid layers
        stream_set = set(names)
        join_keys = set()
        for j in self.joins:
            assert j.from_stream in stream_set, (
                f"join references unknown from_stream {j.from_stream!r}"
            )
            assert j.to_stream in stream_set, (
                f"join references unknown to_stream {j.to_stream!r}"
            )
            assert j.from_stream != j.to_stream, (
                f"self-join not allowed: {j.from_stream!r}"
            )
            assert 0 <= j.at_layer < self.n_layers, (
                f"join at_layer {j.at_layer} out of range "
                f"[0, {self.n_layers})"
            )
            assert j.key not in join_keys, (
                f"duplicate join key {j.key!r}"
            )
            join_keys.add(j.key)

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

    def embed(self, idx: torch.Tensor) -> torch.Tensor:
        """Initial residual = token + pos embedding. Separated from
        forward so a parent module can interleave cross-stream joins
        between layer steps."""
        B, S = idx.shape
        pos_idx = torch.arange(S, device=idx.device)
        return self.tok(idx) + self.pos(pos_idx)

    def process_layer(self, layer: int, x: torch.Tensor) -> torch.Tensor:
        """One layer of attention + FFN. Exposed so joins can fire
        between layers."""
        B, S, D = x.shape
        spec = self.spec
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

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Single-stream forward pass. Returns final residual
        (B, S, d_model). Equivalent to embed + process_layer loop."""
        x = self.embed(idx)
        for layer in range(len(self.W_qkv)):
            x = self.process_layer(layer, x)
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
        # Cross-stream join projections: from_stream's d_model →
        # to_stream's d_model, no bias. Keys are JoinSpec.key.
        self.joins = nn.ModuleDict({
            j.key: nn.Linear(
                config.stream_by_name(j.from_stream).d_model,
                config.stream_by_name(j.to_stream).d_model,
                bias=False,
            )
            for j in config.joins
        })
        # Shared head: takes concatenation of all stream final residuals
        self.head = nn.Linear(config.total_d, config.vocab_size, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns (B, S, vocab) logits.

        Layer-interleaved schedule so cross-stream joins can fire
        between layers:
          1. Each stream embeds the input (tok + pos).
          2. For each layer k:
             a. Every stream processes its layer k in parallel.
             b. Any JoinSpec with at_layer=k fires: add
                join_projection(from_stream_at_k) to to_stream.
          3. Per-stream final residuals concatenate; shared head projects.

        With empty `joins`, this behaves identically to the naive
        parallel forward (joins-between-layers are a no-op).
        """
        # Step 1: initial embeddings per stream
        states: dict[str, torch.Tensor] = {
            s.name: self.streams[s.name].embed(idx)
            for s in self.config.streams
        }
        # Step 2: layer-by-layer with joins fired after each layer
        joins_by_layer: dict[int, list[JoinSpec]] = {}
        for j in self.config.joins:
            joins_by_layer.setdefault(j.at_layer, []).append(j)

        for layer in range(self.config.n_layers):
            # All streams advance one layer
            for s in self.config.streams:
                states[s.name] = self.streams[s.name].process_layer(
                    layer, states[s.name],
                )
            # Apply joins scheduled after this layer
            for j in joins_by_layer.get(layer, []):
                projection = self.joins[j.key]
                states[j.to_stream] = (
                    states[j.to_stream] + projection(states[j.from_stream])
                )

        # Step 3: concat and project
        concat = torch.cat(
            [states[s.name] for s in self.config.streams], dim=-1,
        )  # (B, S, total_d)
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
