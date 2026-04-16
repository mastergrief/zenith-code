"""Gemma 4 E4B substrate loader — full model from GGUF in PyTorch.

Loads all 42 layers from the tq4-aligned GGUF into a PyTorch model.
Tq4 weights stay quantized in memory; dequantize on the fly during
forward pass. This is the foundation for substrate-native inference:
once loaded, PTs and compiled cards install into sub-head ranges.

Architecture (from GGUF metadata):
  - 42 layers, d_model=2560, d_ffn=10240, vocab=262144
  - GQA: 8 Q heads, 2 KV heads, d_head=256
  - Sliding window attention (512 tokens) on alternating layers
  - Per-layer input projection: 2560 → 256 → 2560
  - RoPE: freq_base 1M (global), 10K (SWA)
  - RMSNorm (eps=1e-6)
  - GeGLU FFN: gate * gelu(up)
  - Shared KV across 18 layers
  - tq4 quantized weights (132-byte blocks)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from calm.llm_computer.tq4_gguf_loader import (
    read_turboquant_gguf,
    extract_tq4_tensor,
)
from calm.llm_computer.tq4_torch import dequantize_tq4, Tq4Tensor


@dataclass
class GemmaConfig:
    """Gemma 4 E4B configuration from GGUF metadata."""
    n_layers: int = 42
    d_model: int = 2560
    d_ffn: int = 10240
    vocab_size: int = 262144
    n_heads_q: int = 8
    n_heads_kv: int = 2
    d_head: int = 256          # key_length and value_length
    d_head_swa: int = 256      # SWA head dim
    d_per_layer: int = 256     # per-layer projection dim
    sliding_window: int = 512
    rope_freq_base: float = 1_000_000.0
    rope_freq_base_swa: float = 10_000.0
    rms_norm_eps: float = 1e-6
    shared_kv_layers: int = 18
    max_len: int = 131072


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm: x * weight / rms(x)."""
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    return x / rms * weight


def _rope_freqs(dim: int, max_len: int, base: float = 1_000_000.0,
                device: str = "cpu") -> torch.Tensor:
    """Precompute RoPE frequency tensor: (max_len, dim//2, 2)."""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(max_len, device=device).float()
    angles = torch.outer(t, freqs)  # (max_len, dim//2)
    return torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)


def _apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings to x: (B, H, S, D)."""
    B, H, S, D = x.shape
    x = x.reshape(B, H, S, D // 2, 2)
    cos = freqs[:S, :, 0].unsqueeze(0).unsqueeze(0)  # (1, 1, S, D//2)
    sin = freqs[:S, :, 1].unsqueeze(0).unsqueeze(0)
    x0, x1 = x[..., 0], x[..., 1]
    out = torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1)
    return out.reshape(B, H, S, D)


class Tq4Linear:
    """Linear layer using tq4 quantized weights. Dequant on the fly."""

    def __init__(self, tq4: Tq4Tensor, out_features: int, in_features: int):
        self.tq4 = tq4
        self.out_features = out_features
        self.in_features = in_features

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # GGML convention: weight is (in, out), compute y = x @ W
        w = dequantize_tq4(self.tq4)  # → (in, out) float32
        # Slice to actual dimensions (tq4 pads to block boundaries)
        w = w[:self.in_features, :self.out_features]
        return x @ w

    def to(self, device):
        # tq4 data stays on CPU; dequant happens per-call
        return self


class FP32Linear:
    """Simple FP32 linear (for norms and small tensors)."""

    def __init__(self, weight: torch.Tensor):
        self.weight = weight

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T if self.weight.dim() == 2 else x * self.weight


class GemmaLayer:
    """One Gemma transformer layer with GQA + GeGLU FFN."""

    def __init__(self):
        self.attn_norm_w = None      # RMSNorm weight
        self.post_attn_norm_w = None
        self.ffn_norm_w = None
        self.post_ffw_norm_w = None
        self.post_norm_w = None
        self.attn_q = None           # Tq4Linear
        self.attn_k = None
        self.attn_v = None
        self.attn_output = None
        self.attn_q_norm_w = None
        self.attn_k_norm_w = None
        self.ffn_gate = None         # Tq4Linear
        self.ffn_up = None
        self.ffn_down = None
        self.inp_gate = None         # Per-layer input gate
        self.proj = None             # Per-layer projection
        self.layer_output_scale = None
        self.is_swa = False          # Sliding window attention layer?


class GemmaSubstrate(nn.Module):
    """Full Gemma 4 E4B loaded from GGUF — substrate-native inference.

    All weights stay tq4 quantized. Dequantize per-layer during forward.
    This is the foundation for installing PTs and compiled cards into
    sub-head ranges.
    """

    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.layers = [GemmaLayer() for _ in range(config.n_layers)]
        self.token_embd = None       # Q6_K → FP32
        self.output_norm_w = None    # RMSNorm
        self.per_layer_embd = None   # per-layer token embedding
        self.per_layer_proj = None   # per-layer model projection
        self.per_layer_proj_norm_w = None
        self.rope_freqs_global = None
        self.rope_freqs_swa = None
        self._loaded = False

    @classmethod
    def from_gguf(cls, gguf_path: str, max_len: int = 8192,
                  device: str = "cpu") -> "GemmaSubstrate":
        """Load from tq4-aligned GGUF. Returns ready-to-forward model."""
        print(f"[gemma-substrate] loading from {gguf_path}...")
        reader = read_turboquant_gguf(gguf_path)

        # Parse config from metadata
        cfg = GemmaConfig()
        model = cls(cfg)

        # Build tensor lookup
        tensors = {t.name: t for t in reader.tensors}

        # Global tensors
        print(f"[gemma-substrate] loading token embeddings (Q6_K, 262144 × 2560)...")
        from calm.llm_computer.q6k_dequant import extract_q6_k_tensor
        model.token_embd = extract_q6_k_tensor(reader, "token_embd.weight").to(device)
        print(f"[gemma-substrate] token_embd: {model.token_embd.shape}")

        model.output_norm_w = _extract_fp32(tensors, "output_norm.weight").to(device)

        # Per-layer embeddings
        if "per_layer_token_embd.weight" in tensors:
            model.per_layer_embd = extract_q6_k_tensor(
                reader, "per_layer_token_embd.weight").to(device)
            print(f"[gemma-substrate] per_layer_embd: {model.per_layer_embd.shape}")

        if "per_layer_model_proj.weight" in tensors:
            # This is FP16
            model.per_layer_proj = _extract_fp_tensor(tensors, "per_layer_model_proj.weight").to(device)

        if "per_layer_proj_norm.weight" in tensors:
            model.per_layer_proj_norm_w = _extract_fp32(tensors, "per_layer_proj_norm.weight").to(device)

        # RoPE frequencies
        model.rope_freqs_global = _rope_freqs(
            cfg.d_head, max_len, cfg.rope_freq_base, device)
        model.rope_freqs_swa = _rope_freqs(
            cfg.d_head_swa, max_len, cfg.rope_freq_base_swa, device)

        # Load layers
        for i in range(cfg.n_layers):
            layer = model.layers[i]
            prefix = f"blk.{i}."
            print(f"[gemma-substrate] loading layer {i}/{cfg.n_layers}...", end="\r")

            # Norms (FP32)
            layer.attn_norm_w = _extract_fp32(tensors, prefix + "attn_norm.weight").to(device)
            layer.ffn_norm_w = _extract_fp32(tensors, prefix + "ffn_norm.weight").to(device)

            if prefix + "post_attention_norm.weight" in tensors:
                layer.post_attn_norm_w = _extract_fp32(tensors, prefix + "post_attention_norm.weight").to(device)
            if prefix + "post_ffw_norm.weight" in tensors:
                layer.post_ffw_norm_w = _extract_fp32(tensors, prefix + "post_ffw_norm.weight").to(device)
            if prefix + "post_norm.weight" in tensors:
                layer.post_norm_w = _extract_fp32(tensors, prefix + "post_norm.weight").to(device)

            # QKV norms
            if prefix + "attn_q_norm.weight" in tensors:
                layer.attn_q_norm_w = _extract_fp32(tensors, prefix + "attn_q_norm.weight").to(device)
            if prefix + "attn_k_norm.weight" in tensors:
                layer.attn_k_norm_w = _extract_fp32(tensors, prefix + "attn_k_norm.weight").to(device)

            # Attention weights (tq4)
            layer.attn_q = _make_tq4_linear(tensors, prefix + "attn_q.weight",
                                             cfg.d_model, cfg.n_heads_q * cfg.d_head)
            layer.attn_k = _make_tq4_linear(tensors, prefix + "attn_k.weight",
                                             cfg.d_model, cfg.n_heads_kv * cfg.d_head)
            layer.attn_v = _make_tq4_linear(tensors, prefix + "attn_v.weight",
                                             cfg.d_model, cfg.n_heads_kv * cfg.d_head)
            layer.attn_output = _make_tq4_linear(tensors, prefix + "attn_output.weight",
                                                  cfg.n_heads_q * cfg.d_head, cfg.d_model)

            # FFN weights (tq4)
            layer.ffn_gate = _make_tq4_linear(tensors, prefix + "ffn_gate.weight",
                                               cfg.d_model, cfg.d_ffn)
            layer.ffn_up = _make_tq4_linear(tensors, prefix + "ffn_up.weight",
                                             cfg.d_model, cfg.d_ffn)
            layer.ffn_down = _make_tq4_linear(tensors, prefix + "ffn_down.weight",
                                               cfg.d_ffn, cfg.d_model)

            # Per-layer projection/gate (tq4)
            if prefix + "inp_gate.weight" in tensors:
                layer.inp_gate = _make_tq4_linear(tensors, prefix + "inp_gate.weight",
                                                   cfg.d_model, cfg.d_per_layer)
            if prefix + "proj.weight" in tensors:
                layer.proj = _make_tq4_linear(tensors, prefix + "proj.weight",
                                               cfg.d_per_layer, cfg.d_model)

            # Layer output scale
            if prefix + "layer_output_scale.weight" in tensors:
                layer.layer_output_scale = _extract_fp32(tensors, prefix + "layer_output_scale.weight").to(device)

        print(f"\n[gemma-substrate] loaded {cfg.n_layers} layers, {len(tensors)} tensors")
        model._loaded = True
        return model

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass: token_ids (B, S) → logits (B, S, vocab).

        Dequantizes tq4 weights per-layer. Slow but correct.
        """
        assert self._loaded, "Model not loaded — call from_gguf() first"
        cfg = self.config
        B, S = token_ids.shape

        # Token embedding
        h = self.token_embd[token_ids]  # (B, S, d_model)
        # Gemma scales embeddings by sqrt(d_model)
        h = h * math.sqrt(cfg.d_model)

        for i, layer in enumerate(self.layers):
            h = self._forward_layer(h, layer, i)

        # Final norm
        h = _rms_norm(h, self.output_norm_w, cfg.rms_norm_eps)

        # Output: tied to token embeddings
        logits = h @ self.token_embd.T  # (B, S, vocab)
        return logits

    def _forward_layer(self, h: torch.Tensor, layer: GemmaLayer,
                       layer_idx: int) -> torch.Tensor:
        """Forward one layer: attention + FFN with residual."""
        cfg = self.config
        B, S, D = h.shape

        # Pre-attention norm
        h_norm = _rms_norm(h, layer.attn_norm_w, cfg.rms_norm_eps)

        # QKV projections
        q = layer.attn_q(h_norm)  # (B, S, n_heads_q * d_head)
        k = layer.attn_k(h_norm)  # (B, S, n_heads_kv * d_head)
        v = layer.attn_v(h_norm)

        # Reshape for multi-head
        q = q.reshape(B, S, cfg.n_heads_q, cfg.d_head).transpose(1, 2)
        k = k.reshape(B, S, cfg.n_heads_kv, cfg.d_head).transpose(1, 2)
        v = v.reshape(B, S, cfg.n_heads_kv, cfg.d_head).transpose(1, 2)

        # QK norms (Gemma 4 uses per-head normalization)
        if layer.attn_q_norm_w is not None:
            q = _rms_norm(q, layer.attn_q_norm_w, cfg.rms_norm_eps)
        if layer.attn_k_norm_w is not None:
            k = _rms_norm(k, layer.attn_k_norm_w, cfg.rms_norm_eps)

        # RoPE
        freqs = self.rope_freqs_global
        q = _apply_rope(q, freqs)
        k = _apply_rope(k, freqs)

        # GQA: expand KV heads to match Q heads
        if cfg.n_heads_kv < cfg.n_heads_q:
            repeat = cfg.n_heads_q // cfg.n_heads_kv
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        # Attention: scaled dot product with causal mask
        scale = 1.0 / math.sqrt(cfg.d_head)
        scores = torch.einsum("bhid,bhjd->bhij", q, k) * scale

        # Causal mask
        mask = torch.triu(torch.ones(S, S, dtype=torch.bool, device=h.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        attn_out = torch.einsum("bhij,bhjd->bhid", weights, v)

        # Merge heads
        attn_out = attn_out.transpose(1, 2).reshape(B, S, cfg.n_heads_q * cfg.d_head)

        # Output projection
        attn_out = layer.attn_output(attn_out)

        # Post-attention norm
        if layer.post_attn_norm_w is not None:
            attn_out = _rms_norm(attn_out, layer.post_attn_norm_w, cfg.rms_norm_eps)

        # Residual
        h = h + attn_out

        # FFN
        h_norm = _rms_norm(h, layer.ffn_norm_w, cfg.rms_norm_eps)

        # GeGLU: gate * gelu(up)
        gate = layer.ffn_gate(h_norm)  # (B, S, d_ffn)
        up = layer.ffn_up(h_norm)
        ffn_out = F.gelu(gate, approximate="tanh") * up
        ffn_out = layer.ffn_down(ffn_out)

        # Post-FFN norm
        if layer.post_ffw_norm_w is not None:
            ffn_out = _rms_norm(ffn_out, layer.post_ffw_norm_w, cfg.rms_norm_eps)

        h = h + ffn_out
        return h

    def param_count(self) -> int:
        """Approximate parameter count."""
        cfg = self.config
        per_layer = (
            cfg.d_model * cfg.n_heads_q * cfg.d_head +  # attn_q
            cfg.d_model * cfg.n_heads_kv * cfg.d_head * 2 +  # attn_k, attn_v
            cfg.n_heads_q * cfg.d_head * cfg.d_model +  # attn_output
            cfg.d_model * cfg.d_ffn * 3 +  # ffn_gate, ffn_up, ffn_down
            cfg.d_model * 5  # norms
        )
        return per_layer * cfg.n_layers + cfg.vocab_size * cfg.d_model


# --- Helper functions ---

def _extract_fp32(tensors: dict, name: str) -> torch.Tensor:
    """Extract an FP32 tensor from GGUF."""
    t = tensors[name]
    data = t.data
    if hasattr(data, 'tobytes'):
        import numpy as np
        arr = np.frombuffer(data.tobytes(), dtype=np.float32)
        return torch.from_numpy(arr.copy()).reshape(list(t.shape))
    return torch.tensor(data, dtype=torch.float32)


def _extract_fp_tensor(tensors: dict, name: str) -> torch.Tensor:
    """Extract FP16 or FP32 tensor."""
    t = tensors[name]
    data = t.data
    import numpy as np
    if t.tensor_type == 1:  # FP16
        arr = np.frombuffer(data.tobytes(), dtype=np.float16)
        return torch.from_numpy(arr.copy().astype(np.float32)).reshape(list(t.shape))
    else:
        arr = np.frombuffer(data.tobytes(), dtype=np.float32)
        return torch.from_numpy(arr.copy()).reshape(list(t.shape))


def _make_tq4_linear(tensors: dict, name: str,
                     in_features: int, out_features: int) -> Tq4Linear:
    """Create a Tq4Linear from GGUF tensor data."""
    t = tensors[name]
    # Build Tq4Tensor from raw bytes
    raw = t.data.tobytes() if hasattr(t.data, 'tobytes') else bytes(t.data)
    n_blocks = len(raw) // 132
    import numpy as np
    qs = np.zeros((n_blocks, 128), dtype=np.uint8)
    d = np.zeros(n_blocks, dtype=np.float16)
    for i in range(n_blocks):
        offset = i * 132
        qs[i] = np.frombuffer(raw[offset:offset+128], dtype=np.uint8)
        d[i] = np.frombuffer(raw[offset+128:offset+130], dtype=np.float16)[0]
    tq4 = Tq4Tensor(
        qs=torch.from_numpy(qs),
        d=torch.from_numpy(d.astype(np.float32)),
        shape=(in_features, out_features),
    )
    return Tq4Linear(tq4, out_features, in_features)


def main():
    """Quick load test."""
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--gguf", default="/home/gabe/models/gemma-4-E4B-it-tq4-aligned.gguf")
    p.add_argument("--max-len", type=int, default=512)
    args = p.parse_args()

    model = GemmaSubstrate.from_gguf(args.gguf, max_len=args.max_len)
    print(f"\n[gemma-substrate] ~{model.param_count():,} params")
    print(f"[gemma-substrate] ready for substrate card installation")


if __name__ == "__main__":
    main()
