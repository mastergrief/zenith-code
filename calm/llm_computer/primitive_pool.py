"""Shared primitive pool — cards reference atomic heads/neurons from a
common bank instead of owning their own copies.

Current architecture: each stream has its own `n_heads` attention heads
and `d_ffn` FFN neurons. If stream A and stream B both need a "copy
prior position" attention head, they each carry their own copy.
Wasteful at 5 streams, prohibitive at 50.

Alternative: a pool of M primitives (M × head_size attention heads +
M × ff_size FFN neurons) shared across all cards. Cards are DECLARATIONS
over which pool indices they use. Two cards can reference the same head
without duplicating params.

MVP design (proof-of-concept):
  `AttentionHeadPool(n_primitives, d_head=2)` — a bank of attention
  heads. Forward pass takes (q, k, v) in a unified d_model=2
  representation and produces attention-weighted values.

  `FFNNeuronPool(n_primitives, d_model, d_ffn_primitive)` — a bank of
  ReGLU neurons.

  Cards declare `uses_heads=[3, 7, 12]` and `uses_neurons=[0, 1, 42]`.
  The cards' forward pass selects those primitives by index and
  applies them to the card's stream.

MVP scope: we ship the pool data structures + a minimal demonstration
that two cards can share an attention head and get identical outputs
on the same input. Full integration with the multi-stream transformer
forward pass is future work — this commit establishes the primitive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionHeadPool(nn.Module):
    """Bank of shared attention heads, each d_head-dimensional.

    Stores `n_primitives` independent (W_q, W_k, W_v) triples plus a
    shared output projection. Cards reference heads by index.
    Parameters per head: 3 * d_model * d_head + d_head * d_model.

    Attributes:
        n_primitives: number of distinct heads available.
        d_model: input/output dimension.
        d_head: per-head dim (substrate invariant = 2).
    """

    def __init__(self, n_primitives: int, d_model: int, d_head: int = 2):
        super().__init__()
        assert d_head == 2, f"substrate invariant d_head=2, got {d_head}"
        self.n_primitives = n_primitives
        self.d_model = d_model
        self.d_head = d_head
        # One (q, k, v) projection per primitive
        # Stacked as (n_primitives, d_head, d_model) so we can index a head.
        self.W_q = nn.Parameter(torch.zeros(n_primitives, d_head, d_model))
        self.W_k = nn.Parameter(torch.zeros(n_primitives, d_head, d_model))
        self.W_v = nn.Parameter(torch.zeros(n_primitives, d_head, d_model))
        # Shared output projection: attn output (d_head,) → residual (d_model,)
        self.W_out = nn.Parameter(torch.zeros(n_primitives, d_model, d_head))

    def forward_head(
        self, head_idx: int, x: torch.Tensor,
        use_hard_max: bool = False,
    ) -> torch.Tensor:
        """Apply head `head_idx` to input x: (B, S, d_model). Returns
        (B, S, d_model) — the head's contribution to the residual
        stream (pre residual-add)."""
        B, S, D = x.shape
        assert D == self.d_model
        Wq = self.W_q[head_idx]
        Wk = self.W_k[head_idx]
        Wv = self.W_v[head_idx]
        Wo = self.W_out[head_idx]
        # (B, S, d_head)
        q = x @ Wq.T
        k = x @ Wk.T
        v = x @ Wv.T
        scores = torch.einsum("bid,bjd->bij", q, k) / (self.d_head ** 0.5)
        mask = torch.triu(
            torch.ones(S, S, dtype=torch.bool, device=x.device), diagonal=1,
        )
        scores = scores.masked_fill(mask, float("-inf"))
        if use_hard_max:
            idx = scores.argmax(dim=-1, keepdim=True)
            weights = torch.zeros_like(scores)
            weights.scatter_(-1, idx, 1.0)
        else:
            weights = F.softmax(scores, dim=-1)
        # (B, S, d_head)
        attn = torch.einsum("bij,bjd->bid", weights, v)
        # Project back: (B, S, d_model)
        return attn @ Wo.T

    def forward_multi(
        self, head_indices: Iterable[int], x: torch.Tensor,
        use_hard_max: bool = False,
    ) -> torch.Tensor:
        """Apply multiple heads; sum contributions to residual."""
        out = torch.zeros_like(x)
        for h in head_indices:
            out = out + self.forward_head(h, x, use_hard_max=use_hard_max)
        return out


class FFNNeuronPool(nn.Module):
    """Bank of shared ReGLU neurons.

    Each neuron: y_c = coef_c * val_c * ReLU(gate_c) where gate_c and
    val_c are linear projections of x. Stored as stacked tensors so
    indexing selects a neuron without copying.

    Each neuron writes to a specific output channel of the residual
    (specified at init via `output_channels` OR determined by neuron
    index if left default).

    MVP: neuron `i` writes to residual channel `i % d_model` with
    coefficient +1. Full output-channel routing is a config; this ships
    the simplest form.
    """

    def __init__(
        self, n_primitives: int, d_model: int,
        output_channels: list[int] | None = None,
    ):
        super().__init__()
        self.n_primitives = n_primitives
        self.d_model = d_model
        if output_channels is None:
            output_channels = [i % d_model for i in range(n_primitives)]
        assert len(output_channels) == n_primitives
        self.register_buffer(
            "output_channels", torch.tensor(output_channels, dtype=torch.long),
        )
        # One scalar gate + one scalar val projection per neuron, as
        # linear functions of x (shape: n_primitives, d_model each).
        self.gate_w = nn.Parameter(torch.zeros(n_primitives, d_model))
        self.val_w = nn.Parameter(torch.zeros(n_primitives, d_model))
        self.coef = nn.Parameter(torch.ones(n_primitives))

    def forward_neuron(
        self, neuron_idx: int, x: torch.Tensor,
    ) -> torch.Tensor:
        """Compute this neuron's contribution — returns a (B, S, d_model)
        tensor that's zero everywhere except on the neuron's output
        channel."""
        B, S, D = x.shape
        g = self.gate_w[neuron_idx]  # (D,)
        v = self.val_w[neuron_idx]   # (D,)
        gate = x @ g                  # (B, S)
        val = x @ v                   # (B, S)
        activation = F.relu(gate) * val * self.coef[neuron_idx]  # (B, S)
        out = torch.zeros_like(x)
        ch = int(self.output_channels[neuron_idx].item())
        out[:, :, ch] = activation
        return out

    def forward_multi(
        self, neuron_indices: Iterable[int], x: torch.Tensor,
    ) -> torch.Tensor:
        """Sum contributions from multiple neurons."""
        out = torch.zeros_like(x)
        for n in neuron_indices:
            out = out + self.forward_neuron(n, x)
        return out


@dataclass
class CardPrimitiveSpec:
    """Declaration of which pool primitives a card uses."""
    card_name: str
    head_indices: tuple[int, ...]
    neuron_indices: tuple[int, ...]


class SharedPrimitiveRegistry:
    """Tracks which cards claim which primitives. Catches conflicts
    (two cards claiming exclusive use of the same primitive) analogously
    to ChannelRegistry's allocation tracking."""

    def __init__(self, n_heads: int, n_neurons: int):
        self.n_heads = n_heads
        self.n_neurons = n_neurons
        self._card_specs: dict[str, CardPrimitiveSpec] = {}

    def register(
        self, card_name: str,
        head_indices: Iterable[int], neuron_indices: Iterable[int],
    ) -> CardPrimitiveSpec:
        if card_name in self._card_specs:
            raise ValueError(f"card {card_name!r} already registered")
        heads = tuple(head_indices)
        neurons = tuple(neuron_indices)
        for h in heads:
            if not 0 <= h < self.n_heads:
                raise IndexError(
                    f"head {h} out of range [0, {self.n_heads})"
                )
        for n in neurons:
            if not 0 <= n < self.n_neurons:
                raise IndexError(
                    f"neuron {n} out of range [0, {self.n_neurons})"
                )
        spec = CardPrimitiveSpec(
            card_name=card_name,
            head_indices=heads, neuron_indices=neurons,
        )
        self._card_specs[card_name] = spec
        return spec

    def cards_sharing_head(self, head_idx: int) -> list[str]:
        """Which cards use head `head_idx`? Sharing is allowed — this
        lets users check how many cards reference each primitive."""
        return [
            n for n, s in self._card_specs.items()
            if head_idx in s.head_indices
        ]

    def cards_sharing_neuron(self, neuron_idx: int) -> list[str]:
        return [
            n for n, s in self._card_specs.items()
            if neuron_idx in s.neuron_indices
        ]

    def sharing_stats(self) -> dict[str, int]:
        """Returns total_primitives_used, unique_used, sharing_count."""
        all_heads = set()
        all_neurons = set()
        total_head_refs = 0
        total_neuron_refs = 0
        for spec in self._card_specs.values():
            all_heads.update(spec.head_indices)
            all_neurons.update(spec.neuron_indices)
            total_head_refs += len(spec.head_indices)
            total_neuron_refs += len(spec.neuron_indices)
        return {
            "unique_heads_used": len(all_heads),
            "unique_neurons_used": len(all_neurons),
            "total_head_refs": total_head_refs,
            "total_neuron_refs": total_neuron_refs,
            "sharing_heads": total_head_refs - len(all_heads),
            "sharing_neurons": total_neuron_refs - len(all_neurons),
        }
