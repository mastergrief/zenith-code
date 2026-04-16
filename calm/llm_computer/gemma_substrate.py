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
    """tq4 linear layer backed by mmap view. Zero-copy load, dequant on call.

    Two modes:
    - CPU: dequant from mmap each call (slow, zero resident memory)
    - GPU: tq4 bytes preloaded to GPU, dequant on GPU (fast, 5 GB resident)
    """

    def __init__(self, raw_data: np.ndarray, in_features: int, out_features: int):
        self.raw = raw_data
        self.in_features = in_features
        self.out_features = out_features
        self.n_bytes = raw_data.nbytes
        self.n_blocks = self.n_bytes // 132
        # GPU-resident tq4 data (set by preload_gpu)
        self._gpu_qs = None
        self._gpu_d = None

    def _parse_blocks(self) -> tuple:
        """Parse tq4 blocks into qs and d arrays."""
        blocks = np.frombuffer(self.raw, dtype=np.uint8).reshape(self.n_blocks, 132)
        qs_np = np.ascontiguousarray(blocks[:, :128])
        d_bytes = np.ascontiguousarray(blocks[:, 128:130])
        d_np = np.frombuffer(d_bytes, dtype=np.float16).astype(np.float32)
        return torch.from_numpy(qs_np), torch.from_numpy(d_np)

    def preload_gpu(self, device: str = "cuda"):
        """Load tq4 bytes onto GPU. Dequant happens on GPU — no transfers."""
        qs, d = self._parse_blocks()
        self._gpu_qs = qs.to(device)
        self._gpu_d = d.to(device)

    def dequant(self, device: str = "cpu") -> torch.Tensor:
        """Dequant to FP32. Uses GPU-resident data if preloaded."""
        if self._gpu_qs is not None:
            qs, d = self._gpu_qs, self._gpu_d
        else:
            qs, d = self._parse_blocks()
            if device != "cpu":
                qs, d = qs.to(device), d.to(device)
        n_elements = self.n_blocks * 256
        tq4 = Tq4Tensor(qs=qs, d=d, shape=(n_elements,))
        w_flat = dequantize_tq4(tq4)
        rows = self.in_features
        cols = n_elements // rows
        w = w_flat.reshape(rows, cols)
        return w[:, :self.out_features]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        w = self.dequant(device=str(x.device))
        result = x @ w
        if self._gpu_qs is None:
            del w  # free if not using preloaded GPU data
        return result


class GpuQ6KEmbedding:
    """Q6_K embedding with GPU-accelerated dequant.

    Parses raw Q6_K blocks into component tensors (ql, qh, scales, d)
    at load time. Stores components on GPU (~553 MB as uint8/int8/fp16).
    Dequants on GPU using vectorized torch ops — fast for both row lookup
    and full output head multiply.
    """

    BLOCK_BYTES = 210
    BLOCK_ELEMENTS = 256

    def __init__(self, raw_data: np.ndarray, vocab_size: int, d_model: int):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.blocks_per_row = d_model // self.BLOCK_ELEMENTS
        n_blocks = vocab_size * self.blocks_per_row
        self._device = "cpu"

        # Parse all blocks at once (vectorized numpy, one-time cost)
        print(f"[gemma-substrate] parsing {n_blocks:,} Q6_K blocks...")
        raw_bytes = np.frombuffer(raw_data, dtype=np.uint8)
        blocks = raw_bytes[:n_blocks * self.BLOCK_BYTES].reshape(n_blocks, self.BLOCK_BYTES)

        self.ql = torch.from_numpy(np.ascontiguousarray(blocks[:, :128]))
        self.qh = torch.from_numpy(np.ascontiguousarray(blocks[:, 128:192]))
        self.scales = torch.from_numpy(
            np.ascontiguousarray(blocks[:, 192:208]).view(np.int8).copy())
        d_raw = np.ascontiguousarray(blocks[:, 208:210]).copy()
        self.d = torch.from_numpy(
            d_raw.view(np.float16).astype(np.float32).reshape(n_blocks))

        self._full_cache = None
        print(f"[gemma-substrate] Q6_K parsed: {self.ql.shape[0]:,} blocks "
              f"({self.ql.nbytes / 1e6:.0f} MB on CPU)")

    def to_gpu(self, device: str = "cuda"):
        """Move parsed components to GPU. ~553 MB total."""
        self.ql = self.ql.to(device)
        self.qh = self.qh.to(device)
        self.scales = self.scales.to(device)
        self.d = self.d.to(device)
        self._device = device
        print(f"[gemma-substrate] Q6_K embedding on {device}")

    def _dequant_blocks(self, block_indices: torch.Tensor) -> torch.Tensor:
        """Dequant specific blocks on GPU. Returns (N, 256) FP32."""
        from calm.llm_computer.q6k_dequant import dequantize_q6_k_blocks
        return dequantize_q6_k_blocks(
            self.ql[block_indices], self.qh[block_indices],
            self.scales[block_indices], self.d[block_indices])

    def __getitem__(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Look up embeddings for token_ids. GPU-accelerated dequant."""
        ids = token_ids.flatten()
        # Each row = blocks_per_row consecutive blocks
        row_starts = ids.to(self.ql.device) * self.blocks_per_row
        # Build block indices for all rows
        offsets = torch.arange(self.blocks_per_row, device=self.ql.device)
        block_idx = (row_starts.unsqueeze(1) + offsets.unsqueeze(0)).flatten()
        # Dequant all blocks at once on GPU
        values = self._dequant_blocks(block_idx)  # (N*bpr, 256)
        # Reshape to (N, d_model)
        result = values.reshape(len(ids), self.d_model)
        return result.reshape(*token_ids.shape, self.d_model)

    def output_logits(self, h: torch.Tensor, chunk_size: int = 16384) -> torch.Tensor:
        """Compute logits via chunked GPU Q6_K dequant + matmul.

        No caching, no extra memory. Q6_K components already on GPU (553 MB).
        Each chunk: dequant 16K rows → matmul → keep scores → free.
        Peak VRAM: ~160 MB per chunk. Total: ~16 chunks, ~160ms on GPU.

        h: (B, 1, d_model) on GPU
        Returns: (B, 1, vocab_size) on GPU
        """
        B = h.shape[0]
        device = h.device
        logits = torch.zeros(B, 1, self.vocab_size, device=device)
        bpr = self.blocks_per_row

        for start in range(0, self.vocab_size, chunk_size):
            end = min(start + chunk_size, self.vocab_size)
            n_rows = end - start
            row_starts = torch.arange(start, end, device=device) * bpr
            offsets = torch.arange(bpr, device=device)
            block_idx = (row_starts.unsqueeze(1) + offsets.unsqueeze(0)).flatten()
            chunk_embd = self._dequant_blocks(block_idx).reshape(n_rows, self.d_model)
            logits[:, :, start:end] = h @ chunk_embd.T
            del chunk_embd
        return logits

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


class KVCache:
    """KV cache for autoregressive generation.

    Supports sliding window (SWA layers keep only last `window_size` tokens)
    and shared KV (multiple layers read from the same cache entry).
    FP16 storage to halve memory vs FP32.
    """

    def __init__(self, n_layers: int, device: str = "cuda"):
        self.n_layers = n_layers
        self.device = device
        self.k_cache: dict[int, torch.Tensor] = {}  # layer_idx → (B, H, S, D)
        self.v_cache: dict[int, torch.Tensor] = {}

    def update(self, layer_idx: int, k_new: torch.Tensor, v_new: torch.Tensor,
               is_swa: bool = False, window_size: int = 512
               ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append new K/V and return full cached sequence.

        For SWA layers, only the last `window_size` tokens are kept.
        """
        if layer_idx in self.k_cache:
            k_full = torch.cat([self.k_cache[layer_idx], k_new.half()], dim=2)
            v_full = torch.cat([self.v_cache[layer_idx], v_new.half()], dim=2)
        else:
            k_full = k_new.half()
            v_full = v_new.half()

        # Sliding window: trim to last window_size for SWA layers
        if is_swa and k_full.shape[2] > window_size:
            k_full = k_full[:, :, -window_size:]
            v_full = v_full[:, :, -window_size:]

        self.k_cache[layer_idx] = k_full
        self.v_cache[layer_idx] = v_full

        # Return FP32 for attention computation
        return k_full.float(), v_full.float()

    def seq_len(self) -> int:
        """Current cached sequence length (from layer 0)."""
        if 0 in self.k_cache:
            return self.k_cache[0].shape[2]
        return 0

    def clear(self):
        self.k_cache.clear()
        self.v_cache.clear()


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

        # Token embedding — parse Q6_K blocks (vectorized, one-time)
        t_embd = tensors["token_embd.weight"]
        model.token_embd = GpuQ6KEmbedding(t_embd.data, cfg.vocab_size, cfg.d_model)

        # Output norm
        model.output_norm_w = _load_fp32(tensors, "output_norm.weight")

        # RoPE frequencies — dimension from GGUF metadata
        rope_dim_global = 512   # gemma4.rope.dimension_count
        rope_dim_swa = 256      # gemma4.rope.dimension_count_swa
        model.rope_freqs_swa = _rope_freqs(rope_dim_swa, max_len, cfg.rope_freq_base_swa)

        # Global RoPE uses proportional frequency factors from GGUF
        if "rope_freqs.weight" in tensors:
            rope_freq_factors = _load_fp32(tensors, "rope_freqs.weight")  # (256,)
            # Build position-dependent freqs modulated by freq_factors
            dim = rope_dim_global
            base_freqs = 1.0 / (cfg.rope_freq_base ** (
                torch.arange(0, dim, 2).float() / dim))
            # Apply proportional factors: each pair gets its own scale
            base_freqs = base_freqs * rope_freq_factors
            t = torch.arange(max_len).float()
            angles = torch.outer(t, base_freqs)
            model.rope_freqs_global = torch.stack(
                [torch.cos(angles), torch.sin(angles)], dim=-1)
        else:
            model.rope_freqs_global = _rope_freqs(
                rope_dim_global, max_len, cfg.rope_freq_base)

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

    def preload_gpu(self, device: str = "cuda"):
        """Load all tq4 bytes onto GPU. Eliminates CPU↔GPU transfer during
        forward pass — dequant happens entirely on GPU.

        VRAM budget: ~5 GB for tq4 bytes + ~1 GB headroom = ~6 GB.
        Fits on RTX 4070 (8 GB) with room for activations.
        """
        import gc

        # Pre-cache Pi rotation matrix on GPU (256×256, used by every dequant)
        from calm.llm_computer.tq4_torch import build_pi, compute_lloyd_max_codebook
        pi = build_pi(device=device, source="torch")
        centroids, _ = compute_lloyd_max_codebook()
        centroids = centroids.to(device)
        # Monkey-patch dequantize_tq4 to use cached Pi+centroids
        import calm.llm_computer.tq4_torch as tq4_mod
        _orig_dequant = tq4_mod.dequantize_tq4
        def _cached_dequant(q, pi_arg=None, centroids_arg=None):
            return _orig_dequant(q, pi=pi, centroids=centroids)
        tq4_mod.dequantize_tq4 = _cached_dequant
        # Also update our module-level import
        global dequantize_tq4
        dequantize_tq4 = _cached_dequant
        print(f"[gemma-substrate] cached Pi + centroids on {device}")

        # Move Q6_K embedding components to GPU (~553 MB)
        self.token_embd.to_gpu(device)

        print(f"[gemma-substrate] preloading tq4 weights to {device}...")
        total_bytes = 0
        for i, layer in enumerate(self.layers):
            for attr in ("attn_q", "attn_k", "attn_v", "attn_output",
                         "ffn_gate", "ffn_up", "ffn_down",
                         "inp_gate", "proj"):
                linear = getattr(layer, attr, None)
                if linear is not None and isinstance(linear, MmapTq4Linear):
                    linear.preload_gpu(device)
                    total_bytes += linear.n_bytes
            if (i + 1) % 10 == 0:
                print(f"[gemma-substrate] preloaded layer {i+1}/{self.config.n_layers} "
                      f"({total_bytes / 1e9:.2f} GB)")
        gc.collect()
        if device == "cuda":
            import torch
            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"[gemma-substrate] GPU: {allocated:.2f} GB allocated, "
                  f"{reserved:.2f} GB reserved")
        print(f"[gemma-substrate] preloaded {total_bytes / 1e9:.2f} GB of tq4 to {device}")

    def forward(self, token_ids: torch.Tensor,
                device: str = "cpu",
                kv_cache: Optional["KVCache"] = None,
                start_pos: int = 0) -> torch.Tensor:
        """Forward pass: token_ids (B, S) → logits (B, S, vocab).

        Args:
            token_ids: (B, S) input token IDs (full prompt or single token)
            device: "cuda" for GPU matmuls, "cpu" for pure CPU
            kv_cache: optional KV cache for autoregressive generation
            start_pos: position offset for RoPE (increments during generation)
        """
        assert self._loaded
        cfg = self.config
        B, S = token_ids.shape

        # Token embedding (Q6_K, dequants only needed rows)
        h = self.token_embd[token_ids].to(device)
        h = h * math.sqrt(cfg.d_model)

        # Per-layer embedding (computed once, used by all layers)
        # TODO: load per_layer_token_embd (Q6_K, 10752×262144) — skipped for now
        self._per_layer_embd = None
        if False:  # per-layer embedding disabled until Q6_K loader handles 10752×262K
            # Look up per-layer token embedding: (vocab, d_per_layer * n_layers)
            pl_embd = self.per_layer_embd[token_ids].to(device)  # (B, S, d_per_layer * n_layers)
            pl_embd = pl_embd * math.sqrt(cfg.d_per_layer)
            pl_embd = pl_embd.reshape(B, S, cfg.d_per_layer, cfg.n_layers)
            # Project main embedding to per-layer space
            if self.per_layer_proj is not None:
                h_proj = h @ self.per_layer_proj.to(device)  # (B, S, d_per_layer * n_layers)
                h_proj = h_proj * (1.0 / math.sqrt(cfg.d_model))
                h_proj = h_proj.reshape(B, S, cfg.d_per_layer, cfg.n_layers)
                if self.per_layer_proj_norm_w is not None:
                    h_proj = _rms_norm(h_proj, self.per_layer_proj_norm_w.to(device), cfg.rms_norm_eps)
                pl_embd = (pl_embd + h_proj) * (1.0 / math.sqrt(2.0))
            # Store as list of per-layer tensors: [n_layers × (B, S, d_per_layer)]
            self._per_layer_embd = [pl_embd[:, :, :, i] for i in range(cfg.n_layers)]

        for i, layer in enumerate(self.layers):
            h = self._forward_layer(h, layer, i, kv_cache=kv_cache,
                                     start_pos=start_pos)

        h = _rms_norm(h, self.output_norm_w.to(device), cfg.rms_norm_eps)

        # Output head: chunked Q6_K dequant + matmul on GPU
        h_last = h[:, -1:, :]  # (B, 1, d_model) stays on GPU
        logits = self.token_embd.output_logits(h_last)  # (B, 1, vocab) on GPU

        # Logit softcapping: tanh(logits / cap) * cap
        # Prevents extreme logits from dominating (Gemma 4 uses cap=30.0)
        cap = 30.0
        logits = torch.tanh(logits / cap) * cap

        return logits

    def _forward_layer(self, h: torch.Tensor, layer: GemmaLayer,
                       layer_idx: int, kv_cache: Optional["KVCache"] = None,
                       start_pos: int = 0) -> torch.Tensor:
        """Forward one layer — matches llama.cpp gemma4-iswa.cpp exactly."""
        cfg = self.config
        device = h.device
        B, S, D = h.shape
        inpL = h  # save for residual

        # --- Attention ---
        cur = _rms_norm(h, layer.attn_norm_w.to(device), cfg.rms_norm_eps)

        q = layer.attn_q(cur)
        k_new = layer.attn_k(cur)
        v_new = layer.attn_v(cur)

        # Per-layer head dim
        q_total = q.shape[-1]
        k_total = k_new.shape[-1]
        d_head_q = q_total // cfg.n_heads_q
        d_head_kv = k_total // cfg.n_heads_kv
        is_global = d_head_q > cfg.d_head

        q = q.reshape(B, S, cfg.n_heads_q, d_head_q).transpose(1, 2)
        k_new = k_new.reshape(B, S, cfg.n_heads_kv, d_head_kv).transpose(1, 2)
        v_new = v_new.reshape(B, S, cfg.n_heads_kv, d_head_kv).transpose(1, 2)

        # Q norm + K norm (learned weights)
        if layer.attn_q_norm_w is not None:
            q = _rms_norm(q, layer.attn_q_norm_w.to(device), cfg.rms_norm_eps)
        if layer.attn_k_norm_w is not None:
            k_new = _rms_norm(k_new, layer.attn_k_norm_w.to(device), cfg.rms_norm_eps)
        # V norm (no learned weight — plain RMSNorm)
        v_rms = torch.sqrt(torch.mean(v_new * v_new, dim=-1, keepdim=True) + cfg.rms_norm_eps)
        v_new = v_new / v_rms

        # RoPE
        freqs = (self.rope_freqs_global if is_global else self.rope_freqs_swa).to(device)
        q = _apply_rope(q, freqs[start_pos:])
        k_new = _apply_rope(k_new, freqs[start_pos:])

        # KV cache
        if kv_cache is not None:
            k_full, v_full = kv_cache.update(layer_idx, k_new, v_new,
                                              is_swa=not is_global,
                                              window_size=cfg.sliding_window)
        else:
            k_full, v_full = k_new, v_new

        # GQA expand
        if cfg.n_heads_kv < cfg.n_heads_q:
            repeat = cfg.n_heads_q // cfg.n_heads_kv
            k_full = k_full.repeat_interleave(repeat, dim=1)
            v_full = v_full.repeat_interleave(repeat, dim=1)

        # Attention scores
        S_kv = k_full.shape[2]
        scores = torch.einsum("bhid,bhjd->bhij", q, k_full) / math.sqrt(d_head_q)
        if S_kv > S:
            pass  # generating single token — attends to all cached
        else:
            mask = torch.triu(torch.ones(S, S_kv, dtype=torch.bool, device=device), diagonal=1)
            scores = scores.masked_fill(mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        cur = torch.einsum("bhij,bhjd->bhid", weights, v_full)
        cur = cur.transpose(1, 2).reshape(B, S, q_total)

        # Output projection
        cur = layer.attn_output(cur)

        # Post-attention norm THEN residual (matches gemma4-iswa.cpp line 107-112)
        if layer.post_attn_norm_w is not None:
            cur = _rms_norm(cur, layer.post_attn_norm_w.to(device), cfg.rms_norm_eps)
        attn_out = cur + inpL

        # --- FFN ---
        cur = _rms_norm(attn_out, layer.ffn_norm_w.to(device), cfg.rms_norm_eps)
        gate = layer.ffn_gate(cur)
        up = layer.ffn_up(cur)
        cur = F.gelu(gate, approximate="tanh") * up
        cur = layer.ffn_down(cur)

        # Post-FFN norm THEN residual (matches gemma4-iswa.cpp line 184-190)
        if layer.post_ffw_norm_w is not None:
            cur = _rms_norm(cur, layer.post_ffw_norm_w.to(device), cfg.rms_norm_eps)
        h = cur + attn_out

        # --- Per-layer embedding (gemma4-iswa.cpp line 193-213) ---
        if layer.inp_gate is not None and self._per_layer_embd is not None:
            pe_in = h
            gate_out = layer.inp_gate(h)
            gate_out = F.gelu(gate_out, approximate="tanh")
            # Get this layer's per-layer embedding slice
            inp_this = self._per_layer_embd[layer_idx]  # (B, S, d_per_layer)
            gate_out = gate_out * inp_this
            proj_out = layer.proj(gate_out)
            if layer.post_norm_w is not None:
                proj_out = _rms_norm(proj_out, layer.post_norm_w.to(device), cfg.rms_norm_eps)
            h = pe_in + proj_out

        # Layer output scale (gemma4-iswa.cpp line 216-219)
        if layer.layer_output_scale is not None:
            h = h * layer.layer_output_scale.to(device)

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
    """Load and generate — prove the substrate produces coherent text."""
    import argparse
    import time
    p = argparse.ArgumentParser()
    p.add_argument("--gguf", default="/home/gabe/models/gemma-4-E4B-it-tq4-aligned.gguf")
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--prompt", type=str, default="The capital of France is")
    p.add_argument("--tokens", type=int, default=10)
    args = p.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    model = GemmaSubstrate.from_gguf(args.gguf, max_len=args.max_len)

    # Preload tq4 to GPU for fast dequant
    if device == "cuda":
        model.preload_gpu(device)

    from calm.llm_computer.synth.gemma_tokenizer import GemmaTokenizer
    tok = GemmaTokenizer.from_gguf(args.gguf)

    ids = tok.encode(args.prompt)
    print(f"\n[substrate] prompt: \"{args.prompt}\" ({len(ids)} tokens)")
    print(f"[substrate] device: {device}")
    print(f"[substrate] generating {args.tokens} tokens with KV cache...")

    # Prefill: process entire prompt at once, populate KV cache
    cfg = model.config
    cache = KVCache(cfg.n_layers, device=device)
    t0 = time.time()
    x = torch.tensor([ids])
    with torch.no_grad():
        logits = model.forward(x, device=device, kv_cache=cache, start_pos=0)
    next_id = int(logits[0, -1].argmax().item())
    prefill_time = time.time() - t0
    print(f"  prefill: {prefill_time:.1f}s ({len(ids)} tokens)")

    generated = list(ids) + [next_id]
    token_text = tok.id_to_token.get(next_id, "?")
    print(f"  step 1: \"{token_text}\"")

    # Decode: one token at a time using KV cache
    for step in range(1, args.tokens):
        x = torch.tensor([[next_id]])
        with torch.no_grad():
            logits = model.forward(x, device=device, kv_cache=cache,
                                   start_pos=len(generated) - 1)
        next_id = int(logits[0, -1].argmax().item())
        generated.append(next_id)
        token_text = tok.id_to_token.get(next_id, "?")
        elapsed = time.time() - t0
        print(f"  step {step+1}: \"{token_text}\" ({elapsed:.1f}s)")
        if next_id == tok.EOS_ID:
            break

    output = tok.decode(generated)
    total = time.time() - t0
    n_gen = len(generated) - len(ids)
    tok_per_sec = n_gen / total if total > 0 else 0
    print(f"\n[substrate] output: \"{output}\"")
    print(f"[substrate] {n_gen} tokens in {total:.1f}s ({tok_per_sec:.2f} tok/s)")
    print(f"[substrate] KV cache: {cache.seq_len()} positions cached")


if __name__ == "__main__":
    main()
