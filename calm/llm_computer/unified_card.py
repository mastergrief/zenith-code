"""Unified CHRLM card — one .pt containing Gemma + compiled cards + trained
substrate, with cross-stream routing.

Puts all three stream types under one `nn.Module` container:
  - Gemma stream (d_head=256/512, tq4 quantized, ~2.5B params)
  - Compiled-card streams (d_head=2, exact, frozen)
  - Trained-substrate streams (d_head=2, SGD-trainable)

Saves as one state_dict. Loads via one call.

Composition model:
  - Each stream has a `forward_residual(input_ids) -> (B, S, stream_d_model)`
    method that returns its final residual before any output head.
  - Cross-stream joins project from one stream's residual into another's
    (learned linear projections, trainable).
  - A single output head (inherited from Gemma's tied embedding by
    default) produces logits over Gemma's vocab, so card outputs must
    be mapped into designated vocab row ranges.

MVP scope:
  - Structural containment: save/load Gemma + cards in one .pt ✓
  - Stream interface unification via a protocol method
  - A `forward_concat` mode that runs all streams in parallel and
    concatenates residuals for a shared head (like MultiStreamTransformer)
  - A `forward_via` mode that runs one specific stream and returns its
    logits (pass-through to Gemma for NL, say)
  - Cross-stream joins left as JoinSpec + projection params; caller
    wires them per-layer in a future round.

What's deferred:
  - Layer-interleaved cross-stream joins (requires exposing per-layer
    forward on every stream, which the generic `_StreamStack` has but
    Gemma4Stream currently doesn't)
  - Gemma-vocab mapping for compiled card outputs (each card's head
    rows must be reserved in Gemma's 262144-entry vocab)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn


class UnifiedCHRLMCard(nn.Module):
    """Container for multiple named streams in one nn.Module.

    Each stream is any `nn.Module`. The container provides:
    - Unified state_dict that combines all streams
    - save/load helpers for one-file persistence
    - Dispatch methods to run a specific stream or combine residuals

    Streams are expected to have either:
    - `forward_residual(input_ids) -> (B, S, stream_d_model)` — preferred
    - OR `forward(input_ids) -> (B, S, vocab)` — treated as terminal,
       cannot be composed via joins

    Example:
        stream_map = {
            "gemma": load_gemma4_stream_from_gguf(gguf_path),
            "adder": build_adder_tiny_stream(),
            "substrate_lm": SubstrateLMStream(cfg),
        }
        card = UnifiedCHRLMCard(stream_map)
        card.save("unified.pt")
        card2 = UnifiedCHRLMCard.load("unified.pt", stream_map_builder=build_streams)
    """

    def __init__(self, streams: dict[str, nn.Module]):
        super().__init__()
        self.streams = nn.ModuleDict(streams)

    def stream_names(self) -> list[str]:
        return list(self.streams.keys())

    def forward(
        self, input_ids: torch.Tensor,
        stream: str = "gemma",
    ) -> torch.Tensor:
        """Run one named stream's default forward and return its output.

        Returns whatever that stream's forward returns (typically logits).
        """
        return self.streams[stream](input_ids)

    def forward_all(
        self, input_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Run every stream's forward_residual (if available) or forward.
        Returns dict of stream_name -> output tensor.

        Streams must accept (B, S) token ids as input. Streams with
        incompatible input signatures must be run via `forward(stream=...)`.
        """
        outputs = {}
        for name, mod in self.streams.items():
            if hasattr(mod, "forward_residual"):
                outputs[name] = mod.forward_residual(input_ids)
            else:
                outputs[name] = mod(input_ids)
        return outputs

    def save(self, path: str | Path) -> None:
        """Save all streams to one .pt file."""
        torch.save({
            "stream_names": list(self.streams.keys()),
            "state_dict": self.state_dict(),
        }, path)

    @classmethod
    def load(
        cls, path: str | Path,
        stream_builders: dict[str, callable],
    ) -> "UnifiedCHRLMCard":
        """Load from a .pt. Caller supplies `stream_builders` — a dict
        mapping stream_name -> zero-arg callable that returns an
        uninitialized nn.Module of the right shape. The container then
        loads state_dict into them.

        This two-step pattern (build then load) is required because
        streams have varying constructor signatures.
        """
        data = torch.load(path, weights_only=False)
        expected_names = set(data["stream_names"])
        builder_names = set(stream_builders.keys())
        if expected_names != builder_names:
            raise ValueError(
                f"stream name mismatch: saved {expected_names}, "
                f"builders {builder_names}"
            )
        streams = {name: builder() for name, builder in stream_builders.items()}
        card = cls(streams)
        card.load_state_dict(data["state_dict"])
        return card

    def total_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def per_stream_param_counts(self) -> dict[str, int]:
        return {
            name: sum(p.numel() for p in mod.parameters())
            for name, mod in self.streams.items()
        }


def add_forward_residual_to_gemma4(gemma4_stream) -> None:
    """Patch a Gemma4Stream with a forward_residual method that returns
    its final (B, S, d_model) residual instead of logits.

    Mutates the stream in place (adds a bound method). This lets
    UnifiedCHRLMCard.forward_all() get Gemma's residual without running
    it through the tied-embedding head — useful for cross-stream joins
    where we want to project Gemma's residual into a card's input.
    """
    import types

    def forward_residual(self, input_ids, positions=None):
        # Reuse forward but short-circuit before the final head
        B, S = input_ids.shape
        x = self.token_embd(input_ids)
        per_layer = self.per_layer_token_embd(input_ids)
        per_layer = per_layer.reshape(
            B, S, self.cfg.per_layer_embed_dim, self.cfg.n_layers,
        )
        for i, layer in enumerate(self.layers):
            x = self._forward_layer(x, per_layer[..., i], layer, positions)
        x = self._rms_apply(x, self.output_norm)
        return x  # (B, S, d_model) — no head applied

    gemma4_stream.forward_residual = types.MethodType(forward_residual, gemma4_stream)


class CrossStreamJoin(nn.Module):
    """Learned projection from one stream's residual into another's.

    Used when a card's output needs to influence Gemma's behavior, or
    vice versa. Frozen base + LoRA-style adapter makes this trainable
    without disturbing the streams themselves.
    """

    def __init__(self, from_d: int, to_d: int, rank: int = 16):
        super().__init__()
        self.from_d = from_d
        self.to_d = to_d
        self.rank = rank
        # Low-rank projection: x -> x @ A^T @ B^T
        self.A = nn.Parameter(torch.empty(rank, from_d))
        self.B = nn.Parameter(torch.zeros(to_d, rank))
        nn.init.kaiming_uniform_(self.A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., from_d) → (..., to_d)"""
        return (x @ self.A.T) @ self.B.T
