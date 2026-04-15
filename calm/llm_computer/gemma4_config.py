"""Gemma 4 E4B heterogeneous architecture config.

The `GemmaConfig` in `hf_gemma_loader.py` assumes uniform attention
across layers. Gemma 4 violates this:

  - 35 of 42 layers use sliding window attention (SWA):
      head_dim=256, window=512, rope_freq_base=10000, dim_count=256
  - 7 of 42 layers use full attention:
      head_dim=512, no window, rope_freq_base=1e6, dim_count=512
  - Pattern: every 6th layer is full (indices 5, 11, 17, 23, 29, 35, 41)

Also Gemma 4 specific:
  - Per-layer token embeddings (per_layer_token_embd.weight) that
    project through per_layer_model_proj per layer; added to residual
  - Per-layer layer_output_scale.weight scalar applied after attention
  - attn_q_norm, attn_k_norm, post_attention_norm — extra RMSNorms vs
    standard Gemma

This module encodes the GGUF metadata into a Python config so the
GemmaStream forward pass can route correctly per layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Gemma4LayerConfig:
    """Config for one layer of Gemma 4."""
    layer_idx: int
    is_full_attention: bool   # False → SWA
    head_dim: int             # 256 (SWA) or 512 (full)
    rope_freq_base: float     # 1e4 (SWA) or 1e6 (full)
    rope_dim_count: int       # 256 (SWA) or 512 (full)
    sliding_window: Optional[int]  # None (full) or 512 (SWA)

    @property
    def attention_type(self) -> str:
        return "full" if self.is_full_attention else "swa"


@dataclass
class Gemma4Config:
    """Full Gemma 4 E4B architectural config derived from GGUF metadata."""
    d_model: int = 2560
    n_heads: int = 8
    n_kv_heads: int = 2
    n_layers: int = 42
    d_ffn: int = 10240
    vocab_size: int = 262144
    max_position: int = 131072

    # SWA defaults (applies to ~5/6 of layers)
    swa_head_dim: int = 256
    swa_rope_base: float = 10000.0
    swa_rope_dim_count: int = 256
    swa_window: int = 512

    # Full attention defaults (applies to every 6th layer)
    full_head_dim: int = 512
    full_rope_base: float = 1000000.0
    full_rope_dim_count: int = 512

    # Full attention layer indices (every 6th, starting at index 5)
    full_attention_layers: tuple[int, ...] = field(
        default_factory=lambda: tuple(range(5, 42, 6)),
    )

    # Gemma 4 special: per-layer embedding dim
    per_layer_embed_dim: int = 256

    rms_norm_eps: float = 1e-6
    tie_embeddings: bool = True

    def layer_config(self, i: int) -> Gemma4LayerConfig:
        """Return the per-layer config for layer `i`."""
        is_full = i in self.full_attention_layers
        return Gemma4LayerConfig(
            layer_idx=i,
            is_full_attention=is_full,
            head_dim=self.full_head_dim if is_full else self.swa_head_dim,
            rope_freq_base=self.full_rope_base if is_full else self.swa_rope_base,
            rope_dim_count=self.full_rope_dim_count if is_full else self.swa_rope_dim_count,
            sliding_window=None if is_full else self.swa_window,
        )

    def q_proj_out(self, layer_idx: int) -> int:
        """Output size of q_proj for a given layer."""
        lc = self.layer_config(layer_idx)
        return self.n_heads * lc.head_dim

    def kv_proj_out(self, layer_idx: int) -> int:
        """Output size of k/v projections."""
        lc = self.layer_config(layer_idx)
        return self.n_kv_heads * lc.head_dim

    def all_layer_configs(self) -> list[Gemma4LayerConfig]:
        return [self.layer_config(i) for i in range(self.n_layers)]


def gemma4_e4b_config() -> Gemma4Config:
    """Canonical Gemma 4 E4B config matching the tq4-aligned GGUF."""
    return Gemma4Config()


def derive_config_from_gguf(reader) -> Gemma4Config:
    """Pull config values from GGUF metadata. Useful for validating that
    our hardcoded defaults match the real file."""
    meta = {}
    for f in reader.fields.values():
        if f.name.startswith("gemma4."):
            try:
                val = f.parts[-1].tolist()
                if isinstance(val, list) and len(val) == 1:
                    val = val[0]
                meta[f.name] = val
            except Exception:
                pass
    cfg = Gemma4Config(
        d_model=meta.get("gemma4.embedding_length", 2560),
        n_heads=meta.get("gemma4.attention.head_count", 8),
        n_kv_heads=meta.get("gemma4.attention.head_count_kv", 2),
        n_layers=meta.get("gemma4.block_count", 42),
        d_ffn=meta.get("gemma4.feed_forward_length", 10240),
        max_position=meta.get("gemma4.context_length", 131072),
        swa_head_dim=meta.get("gemma4.attention.key_length_swa", 256),
        swa_rope_base=meta.get("gemma4.rope.freq_base_swa", 10000.0),
        swa_rope_dim_count=meta.get("gemma4.rope.dimension_count_swa", 256),
        swa_window=meta.get("gemma4.attention.sliding_window", 512),
        full_head_dim=meta.get("gemma4.attention.key_length", 512),
        full_rope_base=meta.get("gemma4.rope.freq_base", 1e6),
        full_rope_dim_count=meta.get("gemma4.rope.dimension_count", 512),
        per_layer_embed_dim=meta.get(
            "gemma4.embedding_length_per_layer_input", 256,
        ),
        rms_norm_eps=meta.get(
            "gemma4.attention.layer_norm_rms_epsilon", 1e-6,
        ),
    )
    return cfg
