"""Direction 2 — computation traces as first-class outputs.

Most neural networks are opaque: they emit logits, you have no record of
which neurons fired, which attention heads attended where, or which
fast-weight bindings were retrieved. The substrate already has the
*structure* to be transparent (gate-graph IR for compiled programs); we
just haven't surfaced it as output.

This module adds an opt-in `ComputationTrace` that any
`Small2DTransformer.forward` call can emit alongside logits. Cost is zero
when not requested; when requested, the trace captures per-layer
attention patterns, FFN activation magnitudes, and any compiled-program
contributions. Downstream uses:

  - Self-introspection: model reads its own trace, decides to retry
  - Auditability: every output traceable to the compute that produced it
  - Targeted online learning (Direction 4): gradient updates on specific
    layers/heads identified as wrong by an external verifier
  - Debugging: see exactly which residual channel a specific output came
    from

Backward-compatible: `forward(idx)` unchanged, `forward(idx, trace=t)`
populates the trace in place. None of the 15 existing compiled programs
or the SubstrateLM MVP checkpoint require any modification — they keep
working with `trace=None`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class LayerTrace:
    """Per-layer trace of compute that contributed to the residual stream.

    All tensors are detached + moved to CPU on capture so the trace is
    safe to keep around without holding GPU memory or autograd state.
    """
    layer_idx: int
    attention_weights: torch.Tensor              # (B, H, S, S) — softmax or hard-max output
    attention_argmax: torch.Tensor               # (B, H, S) — which past position each query selected
    ffn_active_count: torch.Tensor               # (B, S) — # of ReGLU neurons whose gate > 0
    ffn_max_activation: torch.Tensor             # (B, S) — peak activation magnitude per position
    fast_weight_norm: Optional[torch.Tensor]     # (B,) frobenius norm of W_fast at end of layer (None if no fast weights)
    geometry: str = "euclidean"                  # which attention geometry produced these weights


@dataclass
class ComputationTrace:
    """Full forward-pass trace. One LayerTrace per layer + global metadata."""
    layers: list[LayerTrace] = field(default_factory=list)
    sequence_length: int = 0
    iterations: int = 1                          # for Direction 5 (recurrent substrate)
    compiled_program_hits: list[str] = field(default_factory=list)
    notes: dict = field(default_factory=dict)

    def attention_to(self, layer: int, head: int, query_pos: int) -> int:
        """Return the past position that attention head selected at query_pos."""
        return int(self.layers[layer].attention_argmax[0, head, query_pos].item())

    def summary(self) -> str:
        """Compact human-readable summary of the trace."""
        lines = [f"ComputationTrace(seq_len={self.sequence_length}, "
                 f"iterations={self.iterations}, "
                 f"layers={len(self.layers)})"]
        for lt in self.layers:
            n_active = int(lt.ffn_active_count.float().mean().item())
            peak = float(lt.ffn_max_activation.max().item())
            fw = (f", W_fast_norm={float(lt.fast_weight_norm.mean()):.4f}"
                  if lt.fast_weight_norm is not None else "")
            lines.append(
                f"  layer {lt.layer_idx} [{lt.geometry}]: "
                f"avg_active_neurons={n_active}  peak_act={peak:.3f}{fw}"
            )
        if self.compiled_program_hits:
            lines.append(f"  compiled hits: {', '.join(self.compiled_program_hits)}")
        return "\n".join(lines)


def make_trace_collector() -> ComputationTrace:
    """Construct an empty trace ready to be populated during forward."""
    return ComputationTrace()


def capture_layer(
    trace: ComputationTrace,
    *,
    layer_idx: int,
    attention_weights: torch.Tensor,
    ffn_pre_activation: torch.Tensor,
    fast_weight_state: Optional[torch.Tensor] = None,
    geometry: str = "euclidean",
) -> None:
    """Append a LayerTrace to `trace` from raw tensors.

    Args:
        attention_weights: (B, H, S, S) — post-softmax / post-hardmax weights
        ffn_pre_activation: (B, S, d_ffn) — gate * val before output projection
            (used to derive active-neuron count and peak activation)
        fast_weight_state: (B, d_model, d_model) end-of-layer W_fast tensor.
    """
    with torch.no_grad():
        argmax = attention_weights.argmax(dim=-1)             # (B, H, S)
        active = (ffn_pre_activation > 0).sum(dim=-1)          # (B, S)
        peak = ffn_pre_activation.abs().max(dim=-1).values     # (B, S)
        fw_norm = None
        if fast_weight_state is not None:
            fw_norm = fast_weight_state.flatten(1).norm(dim=-1).cpu()
        trace.layers.append(LayerTrace(
            layer_idx=layer_idx,
            attention_weights=attention_weights.detach().cpu(),
            attention_argmax=argmax.detach().cpu(),
            ffn_active_count=active.detach().cpu(),
            ffn_max_activation=peak.detach().cpu(),
            fast_weight_norm=fw_norm,
            geometry=geometry,
        ))


# ---- Traced subclass of Small2DTransformer ----

import torch.nn.functional as F

from calm.llm_computer.model import Small2DTransformer


class TracedSmall2DTransformer(Small2DTransformer):
    """Small2DTransformer that emits a ComputationTrace alongside logits.

    Backward compatible: `forward(idx)` returns logits as before.
    `forward(idx, trace=...)` populates the trace in place AND returns
    logits — call sites that want introspection pass a trace; call sites
    that don't are completely unaffected (including the 15 existing
    compiled programs that subclass-cast through this won't see any
    behavior change).

    State_dict matches parent exactly — no new nn.Parameters. Existing
    Small2DTransformer checkpoints load directly into this subclass.
    """

    def forward(self, idx: torch.Tensor,
                trace: Optional[ComputationTrace] = None) -> torch.Tensor:
        if trace is None:
            return super().forward(idx)

        B, S = idx.shape
        cfg = self.config
        pos_idx = torch.arange(S, device=idx.device)
        x = self.tok(idx) + self.pos(pos_idx)

        trace.sequence_length = S

        for layer in range(cfg.n_layers):
            qkv = self.W_qkv[layer](x)
            qkv = qkv.reshape(B, S, 3, cfg.n_heads, cfg.d_head)
            q, k, v = qkv.permute(2, 0, 3, 1, 4)
            # Replicate parent's _attention but capture intermediate weights.
            scores = torch.einsum("bhid,bhjd->bhij", q, k)
            mask = torch.triu(torch.ones(S, S, dtype=torch.bool, device=q.device),
                              diagonal=1)
            scores = scores.masked_fill(mask, float("-inf"))
            if cfg.use_hard_max:
                argmax = scores.argmax(dim=-1, keepdim=True)
                attn_weights = torch.zeros_like(scores)
                attn_weights.scatter_(-1, argmax, 1.0)
            else:
                attn_weights = F.softmax(scores, dim=-1)
            attn = torch.einsum("bhij,bhjd->bhid", attn_weights, v)
            attn = attn.transpose(1, 2).reshape(B, S, cfg.d_model)
            x = x + self.W_out[layer](attn)
            gate, val = self.ff_in[layer](x).chunk(2, dim=-1)
            ffn_pre = F.relu(gate) * val
            x = x + self.ff_out[layer](ffn_pre)

            capture_layer(
                trace,
                layer_idx=layer,
                attention_weights=attn_weights,
                ffn_pre_activation=ffn_pre,
                fast_weight_state=None,
                geometry="euclidean",
            )

        return self.head(x)
