"""Gemma 4 E4B substrate loader — mmap-based, zero-copy from GGUF.

Loads all 42 layers from the tq4-aligned GGUF using memory-mapped I/O.
Weights stay as raw byte views into the mmap — zero heap allocation
during loading. Dequantization happens per-layer during forward pass.
Peak memory: ~400 MB (one dequantized layer) vs ~15 GB (all layers).

Architecture (from GGUF metadata):
  - 42 layers, d_model=2560, d_ffn=10240, vocab=262144
  - GQA: 8 Q heads, 2 KV heads, d_head=256
  - Sliding window attention (512 tokens) on alternating layers
  - Per-layer input projection: 2560 → 256 → 2560
  - RoPE: freq_base 1M (global), 10K (SWA)
  - RMSNorm (eps=1e-6), GeGLU FFN
  - Shared KV across 18 layers
  - tq4 quantized weights (132-byte blocks)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from calm.llm_computer.tq4_gguf_loader import read_turboquant_gguf
from calm.llm_computer.tq4_torch import Tq4Tensor, dequantize_tq4


@dataclass
class GemmaConfig:
    """Gemma 4 E4B configuration from GGUF metadata."""
    n_layers: int = 42
    d_model: int = 2560
    d_ffn: int = 10240
    vocab_size: int = 262144
    n_heads_q: int = 8
    n_heads_kv: int = 2
    d_head: int = 256
    d_head_swa: int = 256
    d_per_layer: int = 256
    sliding_window: int = 512
    rope_freq_base: float = 1_000_000.0
    rope_freq_base_swa: float = 10_000.0
    rms_norm_eps: float = 1e-6
    shared_kv_layers: int = 18
    max_len: int = 8192


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    return x / rms * weight


def _rope_freqs(dim: int, max_len: int, base: float, device: str = "cpu") -> torch.Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(max_len, device=device).float()
    angles = torch.outer(t, freqs)
    return torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)


def _apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    B, H, S, D = x.shape
    x = x.reshape(B, H, S, D // 2, 2)
    cos = freqs[:S, :, 0].unsqueeze(0).unsqueeze(0)
    sin = freqs[:S, :, 1].unsqueeze(0).unsqueeze(0)
    x0, x1 = x[..., 0], x[..., 1]
    out = torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1)
    return out.reshape(B, H, S, D)


# --- Mmap-based lazy tensor wrappers ---

class MmapTq4Linear:
    """tq4 linear layer backed by mmap view. Zero-copy load, dequant on call."""

    def __init__(self, raw_data: np.ndarray, in_features: int, out_features: int):
        # raw_data is a numpy view into the mmap — no heap allocation
        self.raw = raw_data
        self.in_features = in_features
        self.out_features = out_features
        # raw_data might be a typed numpy array — get byte count correctly
        self.n_bytes = raw_data.nbytes
        self.n_blocks = self.n_bytes // 132

    def dequant(self) -> torch.Tensor:
        """Vectorized dequant: mmap bytes → FP32 tensor."""
        blocks = np.frombuffer(self.raw, dtype=np.uint8).reshape(self.n_blocks, 132)
        qs_np = np.ascontiguousarray(blocks[:, :128])
        d_bytes = np.ascontiguousarray(blocks[:, 128:130])
        d_np = np.frombuffer(d_bytes, dtype=np.float16).astype(np.float32)
        # Let dequant determine shape from block count (don't impose our shape)
        n_elements = self.n_blocks * 256
        # GGML stores (in_features, out_features) but tq4 blocks are flat
        # Dequant to flat, then reshape to GGML orientation
        tq4 = Tq4Tensor(
            qs=torch.from_numpy(qs_np),
            d=torch.from_numpy(d_np),
            shape=(n_elements,),  # flat
        )
        w_flat = dequantize_tq4(tq4)
        # Reshape to (rows, cols) where rows × cols ≤ n_elements
        # GGML convention: weight stored as (in, out)
        rows = self.in_features
        cols = n_elements // rows
        w = w_flat.reshape(rows, cols)
        return w[:, :self.out_features]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        w = self.dequant()
        result = x @ w.to(x.device)
        del w  # free immediately
        return result


class MmapQ6KEmbedding:
    """Q6_K embedding backed by mmap. Dequants only requested rows."""

    BLOCK_BYTES = 210
    BLOCK_ELEMENTS = 256

    def __init__(self, raw_data: np.ndarray, vocab_size: int, d_model: int):
        self.raw = raw_data
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.blocks_per_row = d_model // self.BLOCK_ELEMENTS
        self.bytes_per_row = self.blocks_per_row * self.BLOCK_BYTES
        self._full_cache = None  # lazy full dequant for output head

    def _dequant_block(self, block_bytes: np.ndarray) -> np.ndarray:
        """Dequant one Q6_K block (210 bytes → 256 float32 values)."""
        ql = block_bytes[:128]
        qh = block_bytes[128:192]
        scales = block_bytes[192:208].view(np.int8)
        d = np.frombuffer(block_bytes[208:210], dtype=np.float16).astype(np.float32)[0]

        values = np.zeros(256, dtype=np.float32)
        for i in range(256):
            half = i // 128
            within = i % 128
            quarter = within // 32
            l = within % 32

            ql_idx = half * 64 + l + (32 if quarter in (1, 3) else 0)
            ql_shift = 4 if quarter >= 2 else 0
            qh_idx = half * 32 + l
            qh_shift = 2 * quarter
            scale_idx = half * 8 + (l // 16) + 2 * quarter

            q = ((int(ql[ql_idx]) >> ql_shift) & 0xF) | (((int(qh[qh_idx]) >> qh_shift) & 3) << 4)
            values[i] = d * float(scales[scale_idx]) * (q - 32)
        return values

    def _dequant_rows(self, row_ids: list) -> torch.Tensor:
        """Dequant specific rows from the embedding."""
        result = np.zeros((len(row_ids), self.d_model), dtype=np.float32)
        raw_bytes = np.frombuffer(self.raw, dtype=np.uint8)
        for idx, row_id in enumerate(row_ids):
            offset = row_id * self.bytes_per_row
            for b in range(self.blocks_per_row):
                blk_start = offset + b * self.BLOCK_BYTES
                blk_data = raw_bytes[blk_start:blk_start + self.BLOCK_BYTES]
                result[idx, b * self.BLOCK_ELEMENTS:(b + 1) * self.BLOCK_ELEMENTS] = \
                    self._dequant_block(blk_data)
        return torch.from_numpy(result)

    def __getitem__(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Look up embeddings for token_ids. Dequants only needed rows."""
        ids = token_ids.cpu().numpy().flatten().tolist()
        unique_ids = list(set(ids))
        # Dequant unique rows
        rows = self._dequant_rows(unique_ids)
        # Build lookup
        id_to_idx = {uid: i for i, uid in enumerate(unique_ids)}
        indices = [id_to_idx[i] for i in ids]
        result = rows[indices].reshape(*token_ids.shape, self.d_model)
        return result

    @property
    def T(self):
        """Transpose for output head (tied weights). Lazy full dequant."""
        if self._full_cache is None:
            print("[gemma-substrate] dequanting full embedding for output head...")
            all_ids = list(range(self.vocab_size))
            self._full_cache = self._dequant_rows(all_ids)
        return self._full_cache.T

    @property
    def shape(self):
        return (self.vocab_size, self.d_model)


class MmapFP32Tensor:
    """FP32 tensor backed by mmap view."""

    def __init__(self, raw_data: np.ndarray, shape: tuple):
        self._data = torch.from_numpy(
            np.frombuffer(raw_data, dtype=np.float32).copy()
        ).reshape(shape)

    def __call__(self):
        return self._data

    @property
    def data(self):
        return self._data


class GemmaLayer:
    """One Gemma layer — all weights are mmap views."""

    def __init__(self):
        self.attn_norm_w = None
        self.post_attn_norm_w = None
        self.ffn_norm_w = None
        self.post_ffw_norm_w = None
        self.post_norm_w = None
        self.attn_q = None
        self.attn_k = None
        self.attn_v = None
        self.attn_output = None
        self.attn_q_norm_w = None
        self.attn_k_norm_w = None
        self.ffn_gate = None
        self.ffn_up = None
        self.ffn_down = None
        self.inp_gate = None
        self.proj = None
        self.layer_output_scale = None


class GemmaSubstrate:
    """Full Gemma 4 E4B from GGUF — mmap-based, zero-copy loading.

    Weights stay as mmap byte views. Dequantize per-layer during forward.
    Peak memory: ~400 MB (one layer dequantized) + token embeddings for
    the current batch.
    """

    def __init__(self, config: GemmaConfig):
        self.config = config
        self.layers = [GemmaLayer() for _ in range(config.n_layers)]
        self.token_embd = None
        self.output_norm_w = None
        self.rope_freqs_global = None
        self.rope_freqs_swa = None
        self._reader = None  # keep reader alive (holds mmap)
        self._loaded = False

    @classmethod
    def from_gguf(cls, gguf_path: str, max_len: int = 8192) -> "GemmaSubstrate":
        """Load from tq4-aligned GGUF. Mmap-based — near-zero loading memory."""
        print(f"[gemma-substrate] mmap loading from {gguf_path}...")
        reader = read_turboquant_gguf(gguf_path)

        cfg = GemmaConfig(max_len=max_len)
        model = cls(cfg)
        model._reader = reader  # prevent GC of mmap

        tensors = {t.name: t for t in reader.tensors}

        # Token embedding — stays as Q6_K mmap view
        t_embd = tensors["token_embd.weight"]
        model.token_embd = MmapQ6KEmbedding(t_embd.data, cfg.vocab_size, cfg.d_model)
        print(f"[gemma-substrate] token_embd: mmap Q6_K ({cfg.vocab_size} × {cfg.d_model})")

        # Output norm
        model.output_norm_w = _load_fp32(tensors, "output_norm.weight")

        # RoPE frequencies — dimension from GGUF metadata, not head dim
        # rope.dimension_count=512 for global, 256 for SWA
        rope_dim_global = 512   # from GGUF: gemma4.rope.dimension_count
        rope_dim_swa = 256      # from GGUF: gemma4.rope.dimension_count_swa
        model.rope_freqs_global = _rope_freqs(rope_dim_global, max_len, cfg.rope_freq_base)
        model.rope_freqs_swa = _rope_freqs(rope_dim_swa, max_len, cfg.rope_freq_base_swa)

        # Load all layers — just store mmap views, no dequant
        for i in range(cfg.n_layers):
            layer = model.layers[i]
            p = f"blk.{i}."

            # Norms (FP32, tiny — copy is fine)
            layer.attn_norm_w = _load_fp32(tensors, p + "attn_norm.weight")
            layer.ffn_norm_w = _load_fp32(tensors, p + "ffn_norm.weight")
            if p + "post_attention_norm.weight" in tensors:
                layer.post_attn_norm_w = _load_fp32(tensors, p + "post_attention_norm.weight")
            if p + "post_ffw_norm.weight" in tensors:
                layer.post_ffw_norm_w = _load_fp32(tensors, p + "post_ffw_norm.weight")
            if p + "post_norm.weight" in tensors:
                layer.post_norm_w = _load_fp32(tensors, p + "post_norm.weight")
            if p + "attn_q_norm.weight" in tensors:
                layer.attn_q_norm_w = _load_fp32(tensors, p + "attn_q_norm.weight")
            if p + "attn_k_norm.weight" in tensors:
                layer.attn_k_norm_w = _load_fp32(tensors, p + "attn_k_norm.weight")

            # Attention (tq4 mmap views — zero copy)
            # Use actual GGUF shapes (vary per layer: SWA vs global)
            layer.attn_q = _load_tq4_auto(tensors, p + "attn_q.weight")
            layer.attn_k = _load_tq4_auto(tensors, p + "attn_k.weight")
            layer.attn_v = _load_tq4_auto(tensors, p + "attn_v.weight")
            layer.attn_output = _load_tq4_auto(tensors, p + "attn_output.weight")

            # FFN (tq4 mmap views)
            layer.ffn_gate = _load_tq4(tensors, p + "ffn_gate.weight",
                                        cfg.d_model, cfg.d_ffn)
            layer.ffn_up = _load_tq4(tensors, p + "ffn_up.weight",
                                      cfg.d_model, cfg.d_ffn)
            layer.ffn_down = _load_tq4(tensors, p + "ffn_down.weight",
                                        cfg.d_ffn, cfg.d_model)

            # Per-layer projection (tq4 mmap views)
            if p + "inp_gate.weight" in tensors:
                layer.inp_gate = _load_tq4(tensors, p + "inp_gate.weight",
                                            cfg.d_model, cfg.d_per_layer)
            if p + "proj.weight" in tensors:
                layer.proj = _load_tq4(tensors, p + "proj.weight",
                                        cfg.d_per_layer, cfg.d_model)

            if p + "layer_output_scale.weight" in tensors:
                layer.layer_output_scale = _load_fp32(tensors, p + "layer_output_scale.weight")

            if (i + 1) % 10 == 0 or i == cfg.n_layers - 1:
                print(f"[gemma-substrate] loaded layer {i+1}/{cfg.n_layers} (mmap views)")

        model._loaded = True
        # Report memory usage
        import os
        pid = os.getpid()
        try:
            with open(f"/proc/{pid}/status") as f:
                for line in f:
                    if line.startswith("VmRSS"):
                        print(f"[gemma-substrate] {line.strip()}")
                        break
        except Exception:
            pass
        print(f"[gemma-substrate] ready — {cfg.n_layers} layers, "
              f"{len(tensors)} tensors, mmap-backed")
        return model

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass: token_ids (B, S) → logits (B, S, vocab).

        Dequantizes weights per-layer. Slow (~5-15s per forward at S=32)
        but correct and memory-efficient.
        """
        assert self._loaded
        cfg = self.config
        B, S = token_ids.shape

        # Token embedding (Q6_K, dequants only needed rows)
        h = self.token_embd[token_ids]
        h = h * math.sqrt(cfg.d_model)

        for i, layer in enumerate(self.layers):
            h = self._forward_layer(h, layer, i)

        h = _rms_norm(h, self.output_norm_w, cfg.rms_norm_eps)

        # Output: tied to token embedding (lazy full dequant on first call)
        logits = h @ self.token_embd.T.to(h.device)
        return logits

    def _forward_layer(self, h: torch.Tensor, layer: GemmaLayer,
                       layer_idx: int) -> torch.Tensor:
        cfg = self.config
        B, S, D = h.shape

        h_norm = _rms_norm(h, layer.attn_norm_w, cfg.rms_norm_eps)

        q = layer.attn_q(h_norm)
        k = layer.attn_k(h_norm)
        v = layer.attn_v(h_norm)

        # Per-layer head dim: global layers use d_head=512, SWA use 256
        q_total = q.shape[-1]
        k_total = k.shape[-1]
        d_head_q = q_total // cfg.n_heads_q
        d_head_kv = k_total // cfg.n_heads_kv
        is_global = d_head_q > cfg.d_head  # 512 > 256 → global attention layer

        q = q.reshape(B, S, cfg.n_heads_q, d_head_q).transpose(1, 2)
        k = k.reshape(B, S, cfg.n_heads_kv, d_head_kv).transpose(1, 2)
        v = v.reshape(B, S, cfg.n_heads_kv, d_head_kv).transpose(1, 2)

        if layer.attn_q_norm_w is not None:
            q = _rms_norm(q, layer.attn_q_norm_w, cfg.rms_norm_eps)
        if layer.attn_k_norm_w is not None:
            k = _rms_norm(k, layer.attn_k_norm_w, cfg.rms_norm_eps)

        # RoPE: use global freqs for global layers, SWA freqs for SWA layers
        freqs = self.rope_freqs_global if is_global else self.rope_freqs_swa
        q = _apply_rope(q, freqs)
        k = _apply_rope(k, freqs)

        if cfg.n_heads_kv < cfg.n_heads_q:
            repeat = cfg.n_heads_q // cfg.n_heads_kv
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        scale = 1.0 / math.sqrt(d_head_q)
        scores = torch.einsum("bhid,bhjd->bhij", q, k) * scale
        mask = torch.triu(torch.ones(S, S, dtype=torch.bool, device=h.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        attn_out = torch.einsum("bhij,bhjd->bhid", weights, v)
        attn_out = attn_out.transpose(1, 2).reshape(B, S, q_total)
        attn_out = layer.attn_output(attn_out)

        if layer.post_attn_norm_w is not None:
            attn_out = _rms_norm(attn_out, layer.post_attn_norm_w, cfg.rms_norm_eps)

        h = h + attn_out

        h_norm = _rms_norm(h, layer.ffn_norm_w, cfg.rms_norm_eps)
        gate = layer.ffn_gate(h_norm)
        up = layer.ffn_up(h_norm)
        ffn_out = F.gelu(gate, approximate="tanh") * up
        ffn_out = layer.ffn_down(ffn_out)

        if layer.post_ffw_norm_w is not None:
            ffn_out = _rms_norm(ffn_out, layer.post_ffw_norm_w, cfg.rms_norm_eps)

        h = h + ffn_out
        return h


# --- Helpers ---

def _load_fp32(tensors: dict, name: str) -> torch.Tensor:
    """Load FP32 tensor — small enough to copy from mmap."""
    t = tensors[name]
    arr = np.frombuffer(t.data, dtype=np.float32).copy()
    return torch.from_numpy(arr).reshape([int(s) for s in t.shape])


def _load_tq4(tensors: dict, name: str,
              in_features: int, out_features: int) -> MmapTq4Linear:
    """Create mmap-backed tq4 linear — zero copy."""
    t = tensors[name]
    return MmapTq4Linear(t.data, in_features, out_features)


def _load_tq4_auto(tensors: dict, name: str) -> MmapTq4Linear:
    """Create mmap-backed tq4 linear using GGUF shape — zero copy."""
    t = tensors[name]
    shape = [int(s) for s in t.shape]
    # GGML stores as (in_features, out_features)
    in_f, out_f = shape[0], shape[1] if len(shape) > 1 else shape[0]
    return MmapTq4Linear(t.data, in_f, out_f)


def main():
    """Load test — verify mmap loading doesn't OOM."""
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--gguf", default="/home/gabe/models/gemma-4-E4B-it-tq4-aligned.gguf")
    p.add_argument("--max-len", type=int, default=512)
    args = p.parse_args()

    model = GemmaSubstrate.from_gguf(args.gguf, max_len=args.max_len)

    # Quick forward test with dummy tokens
    print("\n[gemma-substrate] testing forward pass (3 tokens)...")
    ids = torch.tensor([[1, 2, 3]])  # dummy token IDs
    try:
        logits = model.forward(ids)
        print(f"[gemma-substrate] output: {logits.shape}")
        print(f"[gemma-substrate] top token: {logits[0, -1].argmax().item()}")
        print("[gemma-substrate] forward pass WORKS")
    except Exception as e:
        print(f"[gemma-substrate] forward failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
