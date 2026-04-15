"""Gemma-compatible stream — tq4 + LoRA + RoPE + RMSNorm + SwiGLU.

Wires together every primitive from the past 3 commits into a single
forward-pass module that can host Gemma-format weights. Validation
against real Gemma inference is a separate step (requires HF access
and model download); this module provides the STRUCTURE so that
validation is a downstream activity, not a blocker.

Architecture of one layer (matches Gemma 2B):

  x → input_norm (RMSNorm) → q,k,v projections (tq4)
                             → RoPE on q, k
                             → attention (with GQA n_kv_heads)
                             → o_proj (tq4)
                             → residual add

  x → post_attn_norm (RMSNorm) → gate_proj, up_proj (tq4)
                                 → SiLU(gate) * up (SwiGLU)
                                 → down_proj (tq4)
                                 → residual add

This module ships the forward pass; actual Gemma weight loading is a
caller responsibility (tensors come from HF safetensors, GGUF, etc.).
The caller constructs `GemmaLayer` instances, populates weights via
`layer.load_weights(GemmaLayerWeights(...))`, wraps with LoRA via
`LoRATq4Linear`, and composes into a full forward pass.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from calm.llm_computer.gemma_style import RMSNorm, swiglu
from calm.llm_computer.hf_gemma_loader import GemmaConfig, GemmaLayer
from calm.llm_computer.lora import LoRATq4Linear
from calm.llm_computer.rope import apply_rope, build_rope_cache


def gemma_attention(
    x: torch.Tensor,
    q_proj, k_proj, v_proj, o_proj,
    cfg: GemmaConfig,
    cos: torch.Tensor, sin: torch.Tensor,
    positions: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """One attention operation.

    Supports GQA (n_kv_heads < n_heads): K/V are broadcast to match Q.
    """
    B, S, D = x.shape
    n_heads = cfg.n_heads
    n_kv = cfg.n_kv_heads
    head_dim = cfg.head_dim

    q = q_proj(x).reshape(B, S, n_heads, head_dim)
    k = k_proj(x).reshape(B, S, n_kv, head_dim)
    v = v_proj(x).reshape(B, S, n_kv, head_dim)

    # Apply RoPE to q, k
    q = apply_rope(q.transpose(1, 2), cos, sin, positions)  # (B, n_heads, S, D)
    k = apply_rope(k.transpose(1, 2), cos, sin, positions)  # (B, n_kv, S, D)
    v = v.transpose(1, 2)

    # GQA: repeat kv to match n_heads
    if n_kv < n_heads:
        repeats = n_heads // n_kv
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)

    # Scaled dot-product attention with causal mask
    scores = (q @ k.transpose(-1, -2)) / (head_dim ** 0.5)
    mask = torch.triu(
        torch.ones(S, S, dtype=torch.bool, device=x.device), diagonal=1,
    )
    scores = scores.masked_fill(mask, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    attn = weights @ v  # (B, n_heads, S, head_dim)

    # Concatenate heads and project
    attn = attn.transpose(1, 2).reshape(B, S, n_heads * head_dim)
    return o_proj(attn)


def gemma_ffn(x, gate_proj, up_proj, down_proj):
    """Gemma FFN: down_proj(SiLU(gate_proj(x)) * up_proj(x))."""
    gate = gate_proj(x)
    up = up_proj(x)
    return down_proj(swiglu(gate, up))


class GemmaStream(nn.Module):
    """Full Gemma-style stream with tq4 weights + optional LoRA.

    Usage:
        cfg = GemmaConfig(...)
        stream = GemmaStream(cfg)
        # Load weights from HF tensors
        for i, layer_w in enumerate(hf_layer_weights):
            stream.layers[i].load_weights(layer_w)
        stream.embed.weight.data.copy_(embed_weight)
        stream.final_norm.weight.data.copy_(final_norm_weight)
        # Optionally wrap layers with LoRA for fine-tuning
        stream.enable_lora(rank=8, alpha=16)
        # Freeze base, train only LoRA
        stream.freeze_base()
    """

    def __init__(self, cfg: GemmaConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([
            GemmaLayer(cfg) for _ in range(cfg.n_layers)
        ])
        self.final_norm = RMSNorm(cfg.d_model)
        # RoPE cache
        cos, sin = build_rope_cache(
            head_dim=cfg.head_dim,
            max_len=cfg.max_position,
            base=cfg.rope_base,
        )
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)
        self._lora_enabled = False

    def enable_lora(self, rank: int = 8, alpha: float = 16.0,
                    targets: Optional[list[str]] = None) -> int:
        """Wrap selected tq4 projections with LoRA adapters. Returns
        number of trainable parameters added.

        `targets` is a list of attribute names to wrap. Default:
        ["q_proj", "v_proj"] (most common LoRA targets).
        """
        if targets is None:
            targets = ["q_proj", "v_proj"]
        added = 0
        for layer in self.layers:
            for name in targets:
                base = getattr(layer, name)
                wrapped = LoRATq4Linear(base, rank=rank, alpha=alpha)
                setattr(layer, name, wrapped)
                added += wrapped.adapter.A.numel() + wrapped.adapter.B.numel()
        self._lora_enabled = True
        return added

    def freeze_base(self) -> int:
        """Freeze everything except LoRA adapters. Returns frozen count."""
        n = 0
        for p in self.parameters():
            p.requires_grad = False
            n += p.numel()
        # Unfreeze LoRA adapters
        if self._lora_enabled:
            for layer in self.layers:
                for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    m = getattr(layer, name, None)
                    if isinstance(m, LoRATq4Linear):
                        for p in m.adapter.parameters():
                            p.requires_grad = True
                            n -= p.numel()
        return n

    def forward(
        self, input_ids: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass. Input: (B, S) token ids. Output: (B, S, vocab) logits."""
        x = self.embed(input_ids)

        for layer in self.layers:
            # Attention block (pre-norm)
            h = x  # residual
            x = layer.input_norm * x / (x.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt())
            x = gemma_attention(
                x, layer.q_proj, layer.k_proj, layer.v_proj, layer.o_proj,
                self.cfg, self.rope_cos, self.rope_sin, positions,
            )
            x = h + x

            # FFN block (pre-norm)
            h = x
            x = layer.post_attn_norm * x / (x.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt())
            x = gemma_ffn(x, layer.gate_proj, layer.up_proj, layer.down_proj)
            x = h + x

        x = self.final_norm(x)

        # Tied output head
        if self.cfg.tie_embeddings:
            logits = x @ self.embed.weight.T
        else:
            # Separate head (not implemented for Gemma 2B)
            raise NotImplementedError("untied head not supported yet")
        return logits
