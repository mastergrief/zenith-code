"""HF Gemma weight → substrate Tq4Linear mapping.

Maps Gemma's HuggingFace `safetensors` tensor names to our substrate's
Tq4Linear layout. Does NOT actually download Gemma — the caller must
provide FP32/FP16 tensors from any source (HF, GGUF, fresh download).
Our job: quantize them to tq4 and wire up a stream.

Gemma 2B (E2B) tensor naming:
  model.embed_tokens.weight                       (vocab, d_model)
  model.layers.{N}.self_attn.q_proj.weight        (d_model, d_model) or (n_heads * d_head, d_model)
  model.layers.{N}.self_attn.k_proj.weight        (n_kv_heads * d_head, d_model)
  model.layers.{N}.self_attn.v_proj.weight        (n_kv_heads * d_head, d_model)
  model.layers.{N}.self_attn.o_proj.weight        (d_model, n_heads * d_head)
  model.layers.{N}.mlp.gate_proj.weight           (d_ffn, d_model)
  model.layers.{N}.mlp.up_proj.weight             (d_ffn, d_model)
  model.layers.{N}.mlp.down_proj.weight           (d_model, d_ffn)
  model.layers.{N}.input_layernorm.weight         (d_model,)
  model.layers.{N}.post_attention_layernorm.weight (d_model,)
  model.norm.weight                                (d_model,)

Output head: weights are TIED with embed_tokens in Gemma; we can
share the tensor or copy.

This module provides:
  GemmaLayerWeights(dataclass) — named tensors for one layer
  gemma_tensor_map(n_layers) → dict of expected tensor names
  build_gemma_stream_from_weights(spec, layer_weights) — construct a
    stream-like module with Tq4Linear instances populated
  validate_gemma_weight_shapes(weights, config) — sanity-check shapes
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from calm.llm_computer.tq4_torch import (
    HEAD_DIM, Tq4Linear, quantize_tq4,
)


@dataclass
class GemmaConfig:
    """Architectural config for a Gemma-family model.

    Defaults match Gemma 2B (E2B).
    """
    d_model: int = 2048
    n_heads: int = 8
    n_kv_heads: int = 1       # grouped-query attention; 1 for Gemma 2B
    head_dim: int = 256       # NOTE: Gemma 2B uses 256, not d_model/n_heads
    n_layers: int = 18
    d_ffn: int = 16384
    vocab_size: int = 256000
    max_position: int = 8192
    rope_base: float = 10000.0
    tie_embeddings: bool = True

    @property
    def q_proj_out(self) -> int:
        return self.n_heads * self.head_dim

    @property
    def kv_proj_out(self) -> int:
        return self.n_kv_heads * self.head_dim


@dataclass
class GemmaLayerWeights:
    """Named tensors for one transformer layer, as pulled from HF."""
    q_proj: torch.Tensor
    k_proj: torch.Tensor
    v_proj: torch.Tensor
    o_proj: torch.Tensor
    gate_proj: torch.Tensor
    up_proj: torch.Tensor
    down_proj: torch.Tensor
    input_norm: torch.Tensor
    post_attn_norm: torch.Tensor


def gemma_tensor_names(n_layers: int) -> dict[str, str]:
    """Return a canonical map of logical names → HF safetensors keys."""
    names = {
        "embed_tokens": "model.embed_tokens.weight",
        "final_norm":   "model.norm.weight",
    }
    for i in range(n_layers):
        p = f"model.layers.{i}"
        names[f"layer_{i}.q_proj"]        = f"{p}.self_attn.q_proj.weight"
        names[f"layer_{i}.k_proj"]        = f"{p}.self_attn.k_proj.weight"
        names[f"layer_{i}.v_proj"]        = f"{p}.self_attn.v_proj.weight"
        names[f"layer_{i}.o_proj"]        = f"{p}.self_attn.o_proj.weight"
        names[f"layer_{i}.gate_proj"]     = f"{p}.mlp.gate_proj.weight"
        names[f"layer_{i}.up_proj"]       = f"{p}.mlp.up_proj.weight"
        names[f"layer_{i}.down_proj"]     = f"{p}.mlp.down_proj.weight"
        names[f"layer_{i}.input_norm"]    = f"{p}.input_layernorm.weight"
        names[f"layer_{i}.post_attn_norm"] = f"{p}.post_attention_layernorm.weight"
    return names


def validate_gemma_weight_shapes(
    layer_weights: GemmaLayerWeights, cfg: GemmaConfig,
) -> None:
    """Assert each tensor has the right shape for this config."""
    D = cfg.d_model
    assert layer_weights.q_proj.shape == (cfg.q_proj_out, D), (
        f"q_proj shape {layer_weights.q_proj.shape} != "
        f"({cfg.q_proj_out}, {D})"
    )
    assert layer_weights.k_proj.shape == (cfg.kv_proj_out, D)
    assert layer_weights.v_proj.shape == (cfg.kv_proj_out, D)
    assert layer_weights.o_proj.shape == (D, cfg.q_proj_out)
    assert layer_weights.gate_proj.shape == (cfg.d_ffn, D)
    assert layer_weights.up_proj.shape == (cfg.d_ffn, D)
    assert layer_weights.down_proj.shape == (D, cfg.d_ffn)
    assert layer_weights.input_norm.shape == (D,)
    assert layer_weights.post_attn_norm.shape == (D,)


class GemmaLayer(nn.Module):
    """One Gemma transformer layer: tq4-quantized projections + RMSNorm.

    Forward pass is built in `gemma_stream.py` (next commit). This
    class provides the parameter scaffolding only.
    """

    def __init__(self, cfg: GemmaConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.d_model

        # Attention projections (tq4-quantized)
        self.q_proj = Tq4Linear(D, cfg.q_proj_out)
        self.k_proj = Tq4Linear(D, cfg.kv_proj_out)
        self.v_proj = Tq4Linear(D, cfg.kv_proj_out)
        self.o_proj = Tq4Linear(cfg.q_proj_out, D)

        # FFN projections (tq4)
        self.gate_proj = Tq4Linear(D, cfg.d_ffn)
        self.up_proj   = Tq4Linear(D, cfg.d_ffn)
        self.down_proj = Tq4Linear(cfg.d_ffn, D)

        # Norms (FP32 params — small, worth leaving unquantized)
        self.input_norm = nn.Parameter(torch.ones(D))
        self.post_attn_norm = nn.Parameter(torch.ones(D))

    def load_weights(self, w: GemmaLayerWeights) -> None:
        validate_gemma_weight_shapes(w, self.cfg)
        self.q_proj.load_weight(w.q_proj.float())
        self.k_proj.load_weight(w.k_proj.float())
        self.v_proj.load_weight(w.v_proj.float())
        self.o_proj.load_weight(w.o_proj.float())
        self.gate_proj.load_weight(w.gate_proj.float())
        self.up_proj.load_weight(w.up_proj.float())
        self.down_proj.load_weight(w.down_proj.float())
        with torch.no_grad():
            self.input_norm.copy_(w.input_norm.float())
            self.post_attn_norm.copy_(w.post_attn_norm.float())


def freeze_gemma_base(model: nn.Module) -> int:
    """Freeze all parameters of a Gemma layer/stream. Returns param count frozen.

    LoRA adapters added afterwards remain trainable.
    """
    count = 0
    for p in model.parameters():
        p.requires_grad = False
        count += p.numel()
    return count
