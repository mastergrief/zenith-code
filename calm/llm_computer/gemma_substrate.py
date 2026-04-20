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

    @property
    def n_layer_kv_from_start(self) -> int:
        return self.n_layers - self.shared_kv_layers

    def kv_source_layer(self, il: int, is_swa: bool) -> int:
        """Layer index whose KV cache to read for attention. Layers >=
        n_layer_kv_from_start reuse: SWA → last SWA layer below the cutoff
        (cutoff - 2), global → last global layer below (cutoff - 1).
        Mirrors llama-model.cpp:8358 reuse callback."""
        if il < self.n_layer_kv_from_start:
            return il
        return self.n_layer_kv_from_start - (2 if is_swa else 1)


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    return x / rms * weight


def _rope_freqs(dim: int, max_len: int, base: float, device: str = "cpu") -> torch.Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(max_len, device=device).float()
    angles = torch.outer(t, freqs)
    return torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)


def _apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    # NEOX-style: pair is (x[i], x[i + D/2]), not (x[2i], x[2i+1]).
    # Gemma 4 uses LLAMA_ROPE_TYPE_NEOX (llama-model.cpp:9133).
    B, H, S, D = x.shape
    half = D // 2
    cos = freqs[:S, :, 0].unsqueeze(0).unsqueeze(0)  # (1, 1, S, D/2)
    sin = freqs[:S, :, 1].unsqueeze(0).unsqueeze(0)
    x0 = x[..., :half]
    x1 = x[..., half:]
    return torch.cat([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1)


# --- Mmap-based lazy tensor wrappers ---

def _tq4_linear_kernel(x: torch.Tensor, qs: torch.Tensor, d: torch.Tensor,
                       out_features: int, in_features: int,
                       pi: torch.Tensor, centroids: torch.Tensor
                       ) -> torch.Tensor:
    """Standalone tq4 mat-vec/mat-mat: rotates x by Pi.T, gathers centroid
    weights, then F.linear. Same math as dequant + x@W but skips the per-call
    Pi matmul. Pulled out as a free function so torch.compile can compile
    it ONCE and share the optimized kernel across all 378 linears."""
    *batch, in_f = x.shape
    bpr = in_f // 256
    x_rot = (x.reshape(*batch, bpr, 256) @ pi.T).reshape(*batch, in_f)
    low = qs & 0x0F
    high = (qs >> 4) & 0x0F
    codes = torch.stack([low, high], dim=-1).reshape(qs.shape[0], 256)
    w_flat = centroids[codes.long()] * d.unsqueeze(-1)
    w = w_flat.reshape(out_features, in_features)
    return F.linear(x_rot, w)


_compiled_tq4_linear = None  # set by enable_compile_tq4()
_use_triton = False           # set by enable_triton_tq4()
_use_fused_flash_attn = True   # Phase 2 fused tq4 flash-attn dispatch.
                               # NON-MONOTONIC perf curve vs Phase 1 memo on
                               # RTX 4070M (bench 2026-04-20 median-of-5):
                               #   N=64   -18% (launch overhead dominates)
                               #   N=256  +14%  <-- sweet spot
                               #   N=1024 +6%
                               #   N=4096 -7%  (cuBLAS-on-memo wins asymptotic)
                               # Flag enables the FUSED path; the runtime
                               # conditional in `_forward_layer` further
                               # restricts it to `128 < cached_kv_len < 2048`
                               # (the measured winning band). Outside that
                               # band the gate falls back to Phase 1 memo.
                               # Math is correct (cos=1.00000 vs fp32 ref).
                               # Full bench + gate rationale:
                               # `.claude/rules/turboquant.md` §"Fused
                               # flash-attention decode" and tracing_roadmap
                               # Round 53.34 reconciliation row.
                               # Disable entirely via
                               # `enable_fused_flash_attn(False)`.


def enable_triton_tq4(enabled: bool = True):
    """Toggle the Triton fused dequant-matvec kernel for tq4 linears.
    Triton wins ~5-17x per linear (ffn_up: 4.66 ms → 0.28 ms on RTX 4070M)."""
    global _use_triton
    _use_triton = bool(enabled)


def enable_fused_flash_attn(enabled: bool = True):
    """Toggle the Phase 2 fused tq4 flash-attn dispatch in `_forward_layer`.
    Orthogonal to `enable_triton_tq4` — keeps FFN/attn weight kernels on
    while A/B testing the KV-side fused path."""
    global _use_fused_flash_attn
    _use_fused_flash_attn = bool(enabled)


def enable_compile_tq4():
    """Wrap _tq4_linear_kernel in torch.compile once. Reused across all
    MmapTq4Linear instances — only one compile, not 378."""
    global _compiled_tq4_linear
    if _compiled_tq4_linear is None:
        torch.set_float32_matmul_precision("high")
        _compiled_tq4_linear = torch.compile(
            _tq4_linear_kernel, mode="default", dynamic=True)
    return _compiled_tq4_linear


class MmapTq4Linear:
    """tq4 linear layer backed by mmap view. Zero-copy load, dequant on call.

    Two modes:
    - CPU: dequant from mmap each call (slow, zero resident memory)
    - GPU: tq4 bytes preloaded to GPU, dequant on GPU (fast, 5 GB resident)
    """

    # Class-level GPU-resident Pi rotation + centroids — shared across all
    # MmapTq4Linear instances. Set by preload_gpu().
    _shared_pi: Optional[torch.Tensor] = None         # (256, 256), rotation
    _shared_centroids: Optional[torch.Tensor] = None  # (16,), Lloyd-Max levels

    def __init__(self, raw_data: np.ndarray, in_features: int, out_features: int):
        self.raw = raw_data
        self.in_features = in_features
        self.out_features = out_features
        self.n_bytes = raw_data.nbytes
        self.n_blocks = self.n_bytes // 132
        # GPU-resident tq4 data (set by preload_gpu)
        self._gpu_qs = None
        self._gpu_d = None
        # Cached unrotated weight (centroids[codes] * d) reshaped to (out, in).
        # Materialized lazily on first call when fast_path is enabled.
        self._w_unrot_cache = None

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
        """Dequant to (in, out) for `x @ w`. GGUF stores as math (out, in)
        row-major; we reshape to that and transpose."""
        if self._gpu_qs is not None:
            qs, d = self._gpu_qs, self._gpu_d
        else:
            qs, d = self._parse_blocks()
            if device != "cpu":
                qs, d = qs.to(device), d.to(device)
        n_elements = self.n_blocks * 256
        tq4 = Tq4Tensor(qs=qs, d=d, shape=(n_elements,))
        w_flat = dequantize_tq4(tq4)
        w_math = w_flat.reshape(self.out_features, self.in_features)
        return w_math.T.contiguous()

    def _w_unrotated(self) -> torch.Tensor:
        """Materialize centroids[codes] * d as a (out, in) tensor — same layout
        as the Pi-rotated dequant but skips the @Pi step. Combined with input
        pre-rotation by Pi.T this is mathematically equivalent to standard
        dequant + matmul, but ~250× fewer FLOPs (the dequant Pi-matmul was
        the dominant cost — 6.7 GFlops on ffn_up vs 0.026 GFlops for the
        actual mat-vec). Recomputed each call — caching it would defeat
        the tq4 memory savings."""
        assert self._gpu_qs is not None, "fast path requires GPU preload"
        qs, d = self._gpu_qs, self._gpu_d
        low = qs & 0x0F
        high = (qs >> 4) & 0x0F
        codes = torch.stack([low, high], dim=-1).reshape(self.n_blocks, 256)
        centroids = MmapTq4Linear._shared_centroids
        w_flat = centroids[codes.long()] * d.unsqueeze(-1)
        return w_flat.reshape(self.out_features, self.in_features)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self._gpu_qs is not None and MmapTq4Linear._shared_pi is not None:
            if _use_triton:
                from calm.llm_computer.tq4_triton import tq4_linear_triton
                return tq4_linear_triton(
                    x, self._gpu_qs, self._gpu_d,
                    MmapTq4Linear._shared_pi,
                    MmapTq4Linear._shared_centroids,
                    self.out_features, self.in_features)
            # PyTorch fast path: pre-rotate x, gather + matmul, no @Pi
            kernel = _compiled_tq4_linear or _tq4_linear_kernel
            return kernel(x, self._gpu_qs, self._gpu_d,
                          self.out_features, self.in_features,
                          MmapTq4Linear._shared_pi,
                          MmapTq4Linear._shared_centroids)
        # Slow path (CPU or non-preloaded): full dequant via Pi.
        w = self.dequant(device=str(x.device))
        result = x @ w
        if self._gpu_qs is None:
            del w
        return result

    def __getstate__(self):
        # Drop the numpy mmap view (self.raw points into the 5 GB GGUF file).
        # Once _gpu_qs/_gpu_d are preloaded, raw is redundant. Class-level
        # _shared_pi/_shared_centroids are NOT instance state.
        return {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "n_bytes": self.n_bytes,
            "n_blocks": self.n_blocks,
            "_gpu_qs": self._gpu_qs,
            "_gpu_d": self._gpu_d,
        }

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.raw = None
        self._w_unrot_cache = None


class FP32GemmaLinear:
    """FP32 weight tensor with the same call interface as MmapTq4Linear.
    Used after convert_layer_to_fp32 for layers hosting in-attention card
    installs — surgical weight edits are lossless in FP32 (no tq4 quant
    noise on installed card weights; see Round 11 hybrid finding).

    Weight is stored in (in, out) orientation so __call__ does x @ w
    directly (matches MmapTq4Linear's dequant() output convention)."""

    def __init__(self, weight: torch.Tensor, in_features: int, out_features: int):
        assert weight.shape == (in_features, out_features), (
            f"weight shape {weight.shape} != ({in_features}, {out_features})")
        self.weight = weight
        self.in_features = in_features
        self.out_features = out_features
        # MmapTq4Linear API compat (so existing checks like
        # `lin._gpu_qs is not None` short-circuit cleanly to the FP32 path)
        self._gpu_qs = None
        self._gpu_d = None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight


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
        """Look up embeddings for token_ids. Triton-accelerated dequant."""
        try:
            from calm.llm_computer.tq4_triton import q6k_lookup_triton
            result = q6k_lookup_triton(
                token_ids, self.ql, self.qh, self.scales, self.d,
                self.vocab_size, self.d_model)
            return result.reshape(*token_ids.shape, self.d_model)
        except Exception:
            pass
        # Fallback: PyTorch dequant
        ids = token_ids.flatten()
        row_starts = ids.to(self.ql.device) * self.blocks_per_row
        offsets = torch.arange(self.blocks_per_row, device=self.ql.device)
        block_idx = (row_starts.unsqueeze(1) + offsets.unsqueeze(0)).flatten()
        values = self._dequant_blocks(block_idx)
        result = values.reshape(len(ids), self.d_model)
        return result.reshape(*token_ids.shape, self.d_model)

    def output_logits(self, h: torch.Tensor, chunk_size: int = 16384) -> torch.Tensor:
        """Compute logits via chunked GPU Q6_K dequant + matmul.

        Default path uses the Triton fused dequant-matvec kernel
        (~125x faster than chunked PyTorch: 543 ms → 4.3 ms on RTX 4070M
        for the 262K-vocab head). Falls back to chunked PyTorch if the
        kernel can't run.

        h: (B, S, d_model) on GPU
        Returns: (B, S, vocab_size) on GPU
        """
        B, S, D = h.shape
        device = h.device

        try:
            from calm.llm_computer.tq4_triton import q6k_matvec_triton
            # Per-token Triton matvec — fast for B=S=1 (decode), loops for B*S>1.
            flat = h.reshape(-1, D)
            outs = []
            for i in range(flat.shape[0]):
                outs.append(q6k_matvec_triton(
                    flat[i].contiguous().float(), self.ql, self.qh,
                    self.scales, self.d, self.vocab_size, self.d_model))
            return torch.stack(outs, dim=0).reshape(B, S, self.vocab_size)
        except Exception:
            pass

        # Fallback: chunked PyTorch dequant
        logits = torch.zeros(B, S, self.vocab_size, device=device)
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
    Default FP16 storage (halves memory vs FP32). With `use_tq4=True`,
    runs cached K/V through a tq4 quantize-dequantize roundtrip on every
    return — proves tq4 quant noise (~4-bit, Lloyd-Max + Pi rotation)
    is tolerable for attention. Storage stays FP16 for now; switching to
    real tq4 storage is a follow-up that preserves this output behaviour.
    """

    def __init__(self, n_layers: int, device: str = "cuda",
                 use_tq4: bool = False):
        self.n_layers = n_layers
        self.device = device
        self.use_tq4 = use_tq4
        self.k_cache: dict[int, torch.Tensor] = {}  # layer_idx → (B, H, S, D)
        self.v_cache: dict[int, torch.Tensor] = {}
        # Track per-layer SWA-ness for post-forward trim. Set by
        # update() based on its is_swa argument.
        self._is_swa: dict[int, bool] = {}
        self._swa_window: int = 512

    def _tq4_roundtrip(self, x: torch.Tensor) -> torch.Tensor:
        """Inject tq4 quant noise. numel must be divisible by 256
        (always true for Gemma 4 K/V: 2*256 SWA, 2*512 global per token)."""
        from calm.llm_computer.tq4_torch import (
            Tq4Tensor, dequantize_tq4, quantize_tq4)
        flat = x.float().flatten()
        assert flat.numel() % 256 == 0
        q = quantize_tq4(flat)
        return dequantize_tq4(q).reshape(x.shape).to(x.dtype)

    def update(self, layer_idx: int, k_new: torch.Tensor, v_new: torch.Tensor,
               is_swa: bool = False, window_size: int = 512
               ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append new K/V. Returns the FULL concatenated K/V for this
        attention call. STORES only the last `window_size` for SWA
        layers (production-equivalent of llama.cpp's behavior).

        Compute-full / store-trimmed pattern:
          - Multi-position prefill needs full K/V to compute attention
            for each Q position's correct window (returned full)
          - Long-term storage for SWA only needs the last window_size
            (subsequent Q positions can't attend further back anyway)

        Math is offset-invariant — the windowed attention mask in
        _forward_layer uses j-vs-(S_kv-S+i) which cancels any storage
        offset. So callers don't need offset tracking.

        Memory at long context: SWA layers cap at window_size storage,
        global layers grow with sequence length. With KVCacheTq4 +
        proper window storage on global layers, matches llama.cpp's
        ~7 GB at 512K context.
        """
        if layer_idx in self.k_cache:
            k_full = torch.cat([self.k_cache[layer_idx], k_new.half()], dim=2)
            v_full = torch.cat([self.v_cache[layer_idx], v_new.half()], dim=2)
        else:
            k_full = k_new.half()
            v_full = v_new.half()

        if self.use_tq4:
            # Store the noised tensor so shared-KV reads see same noise.
            k_full = self._tq4_roundtrip(k_full)
            v_full = self._tq4_roundtrip(v_full)

        # Store FULL during this forward (don't trim mid-forward —
        # shared-KV consumer layers later in the same forward need
        # the full window). Track is_swa for post-forward trim.
        self.k_cache[layer_idx] = k_full
        self.v_cache[layer_idx] = v_full
        self._is_swa[layer_idx] = is_swa
        if is_swa:
            self._swa_window = window_size

        return k_full.float(), v_full.float()

    def trim_swa_storage(self) -> None:
        """Trim SWA layers' stored K/V to last `window_size` tokens.
        Call AFTER all layers in a forward have run, so shared-KV
        consumers got the full window before this trim.

        Invariant: SWA attention is offset-invariant in the windowed
        mask, so trim doesn't affect future attention correctness —
        only memory. Long-context support depends on this trim
        running after every forward."""
        window = self._swa_window
        for layer_idx, is_swa in self._is_swa.items():
            if is_swa and layer_idx in self.k_cache:
                k = self.k_cache[layer_idx]
                if k.shape[2] > window:
                    self.k_cache[layer_idx] = k[:, :, -window:].contiguous()
                    self.v_cache[layer_idx] = (
                        self.v_cache[layer_idx][:, :, -window:].contiguous())

    def seq_len(self) -> int:
        """Current cached sequence length (from layer 0)."""
        if 0 in self.k_cache:
            return self.k_cache[0].shape[2]
        return 0

    def clear(self):
        self.k_cache.clear()
        self.v_cache.clear()


class KVCacheStatic:
    """Pre-allocated KV cache with fixed-shape buffers and GPU-resident
    position tracker — for CUDA Graphs capture/replay (5.83x measured win
    over the dynamic-shape KVCache).

    All KV writes go through index_copy_ at `pos` (a 0-d GPU tensor) and
    reads return the FULL max_len buffer; attention masking handles the
    valid range. The graph captures these fixed-shape ops once; the caller
    updates `pos` and `valid_mask` between replays via .copy_().
    """

    def __init__(self, model: "GemmaSubstrate", max_len: int = 1024,
                 device: str = "cuda"):
        cfg = model.config
        self.cfg = cfg
        self.max_len = max_len
        self.device = device
        # Per-layer buffers — sized by per-layer head dim.
        self.k_buf: list[torch.Tensor] = []
        self.v_buf: list[torch.Tensor] = []
        for il in range(cfg.n_layers):
            # Inspect each layer's actual K/V shape via attn_k.out_features.
            kv_total = model.layers[il].attn_k.out_features
            d_head_kv = kv_total // cfg.n_heads_kv
            self.k_buf.append(torch.zeros(
                1, cfg.n_heads_kv, max_len, d_head_kv,
                dtype=torch.float16, device=device))
            self.v_buf.append(torch.zeros(
                1, cfg.n_heads_kv, max_len, d_head_kv,
                dtype=torch.float16, device=device))
        # Position is a 0-d GPU tensor — updated between replays via copy_.
        self.pos = torch.zeros((), dtype=torch.long, device=device)
        # Mask: True means position is INVALID (future or beyond pos).
        self.valid_mask = torch.ones(max_len, dtype=torch.bool, device=device)
        # Pre-allocated arange for GPU-side mask recomputation (avoids
        # Python-int slicing in the hot loop).
        self._arange = torch.arange(max_len, dtype=torch.long, device=device)

    def set_pos(self, p):
        """Update GPU pos tensor + valid_mask. p may be a Python int OR
        a 0-d GPU tensor; mask recomputation is purely GPU-side either way."""
        if isinstance(p, int):
            self.pos.fill_(p)
        else:
            self.pos.copy_(p)
        # valid_mask[i] = True if i > pos (mask out future positions)
        torch.gt(self._arange, self.pos, out=self.valid_mask)

    def update(self, layer_idx: int, k_new: torch.Tensor, v_new: torch.Tensor,
               is_swa: bool = False, window_size: int = 512):
        """Write k_new, v_new at position [pos] (single token, S=1).
        Returns the full pre-allocated K/V buffers (caller masks).
        is_swa/window_size accepted for API compat with KVCache; the
        static buffer relies on valid_mask for filtering instead."""
        self.k_buf[layer_idx].index_copy_(
            2, self.pos.unsqueeze(0), k_new.half())
        self.v_buf[layer_idx].index_copy_(
            2, self.pos.unsqueeze(0), v_new.half())
        return self.k_buf[layer_idx].float(), self.v_buf[layer_idx].float()

    @property
    def k_cache(self):
        """Dict-like access for shared-KV reads (matches KVCache API)."""
        return self.k_buf

    @property
    def v_cache(self):
        return self.v_buf


class KVCacheTq4:
    """KV cache with REAL tq4 byte storage — ~4x memory savings vs FP16.
    The noise-injection KVCache(use_tq4=True) MVP proved tq4 quant noise
    is tolerable for attention; this stores actual tq4 bytes.

    Phase 2 storage layout (HEAD-MAJOR):
      k_qs[il]: (n_kv_h, max_len * bpr, 128) uint8 — packed nybbles
      k_d[il]:  (n_kv_h, max_len * bpr) fp32      — per-block scale
      v_qs/v_d: same shapes as k_qs/k_d

    where bpr = d_head // 256 (blocks per row, per head). Gemma E4B:
    d_head=256 → bpr=1, so each (head, position) slot is exactly one
    tq4 block of 256 elements.

    Why head-major: the fused tq4 flash-attn kernel
    (`tq4_flash_attn.fused_tq4_flash_attn_decode`) iterates per Q head
    over N positions, with one program per head. Head-major puts each
    head's N tq4 blocks contiguously so byte loads coalesce stride-1.

    Memory at 512K context (gemma-4-E4B):
      FP16: ~14 GB total KV across own-KV layers
      tq4:  ~3.9 GB total KV (~3.6x reduction)

    Multi-token write supported (S>=1) — quantize is run on a single
    flat batch and the resulting blocks are scattered into per-head
    slots in one indexed write. Per-layer position tracking — each
    layer's write pos is tracked independently, so shared-KV read
    layers see the correct stored length without a shared step_done
    barrier.
    """

    def __init__(self, model: "GemmaSubstrate", max_len: int = 1024,
                 device: str = "cuda"):
        from calm.llm_computer.tq4_torch import build_pi, compute_lloyd_max_codebook
        cfg = model.config
        self.cfg = cfg
        self.max_len = max_len
        self.device = device
        self._pi = build_pi(device=device, source="torch")
        centroids, boundaries = compute_lloyd_max_codebook()
        self._centroids = centroids.to(device)
        self._boundaries = boundaries.to(device)

        self.k_qs: list[torch.Tensor] = []
        self.k_d: list[torch.Tensor] = []
        self.v_qs: list[torch.Tensor] = []
        self.v_d: list[torch.Tensor] = []
        self._d_head: list[int] = []
        self._bpr: list[int] = []  # blocks per row, per head (d_head // 256)
        self.layer_pos: list[int] = [0] * cfg.n_layers
        self._is_swa: dict[int, bool] = {}
        self._swa_window: int = 512
        # Per-step dequant memo. Key: (which: "k"|"v", layer_idx).
        # Value: (valid_for_pos, fp32_tensor of shape (1, n_kv_h, valid_for_pos, d_head)).
        # Hit when called repeatedly within one decode step (own-KV update +
        # shared-KV consumer reads from the same source layer). Invalidated
        # by update()/trim_swa_storage()/clear() on the affected layer.
        self._memo: dict[tuple[str, int], tuple[int, torch.Tensor]] = {}
        for il in range(cfg.n_layers):
            kv_total = model.layers[il].attn_k.out_features
            d_head = kv_total // cfg.n_heads_kv
            self._d_head.append(d_head)
            assert d_head % 256 == 0, (
                f"layer {il} d_head={d_head} not a multiple of 256")
            bpr = d_head // 256
            self._bpr.append(bpr)
            # Head-major: per head, max_len * bpr blocks of 128 bytes.
            self.k_qs.append(torch.zeros(cfg.n_heads_kv, max_len * bpr, 128,
                                         dtype=torch.uint8, device=device))
            self.k_d.append(torch.zeros(cfg.n_heads_kv, max_len * bpr,
                                        dtype=torch.float32, device=device))
            self.v_qs.append(torch.zeros(cfg.n_heads_kv, max_len * bpr, 128,
                                         dtype=torch.uint8, device=device))
            self.v_d.append(torch.zeros(cfg.n_heads_kv, max_len * bpr,
                                        dtype=torch.float32, device=device))

    def memory_bytes(self) -> int:
        """Total bytes in the tq4 KV cache (qs + d for all layers)."""
        total = 0
        for il in range(self.cfg.n_layers):
            total += self.k_qs[il].numel() + self.v_qs[il].numel()
            total += self.k_d[il].numel() * 4 + self.v_d[il].numel() * 4
        return total

    def _dequant_layer(self, layer_idx: int, which: str) -> torch.Tensor:
        """Dequant the full stored sequence for one (layer, which) to fp32,
        with per-step memoization. Returns shape (1, n_kv_h, layer_pos, d_head).

        Memo key is (which, layer_idx); value carries the layer_pos it was
        computed at. Stale entries (pos changed via update/trim/clear) are
        recomputed and overwritten. Within a single decode step the entry
        is hit by both the own-KV update path AND every shared-KV consumer
        reading this layer through `_Tq4ReadProxy`, eliminating the
        previous behavior where each shared-KV consumer re-dequanted the
        entire prefix from scratch.
        """
        from calm.llm_computer.tq4_torch import Tq4Tensor, dequantize_tq4
        pos = self.layer_pos[layer_idx]
        key = (which, layer_idx)
        cached = self._memo.get(key)
        if cached is not None and cached[0] == pos:
            return cached[1]

        bpr = self._bpr[layer_idx]
        d_head = self._d_head[layer_idx]
        n_kv_h = self.cfg.n_heads_kv
        if pos == 0:
            out = torch.zeros(1, n_kv_h, 0, d_head,
                              dtype=torch.float32, device=self.device)
            self._memo[key] = (pos, out)
            return out

        # Head-major: slice per-head [:, :pos*bpr, :], flatten over heads in
        # head-major order (already contiguous), single dequant call.
        n_blocks_per_head = pos * bpr
        qs_buf = self.k_qs[layer_idx] if which == "k" else self.v_qs[layer_idx]
        d_buf = self.k_d[layer_idx] if which == "k" else self.v_d[layer_idx]
        qs_used = qs_buf[:, :n_blocks_per_head, :].contiguous()  # (n_kv_h, pos*bpr, 128)
        d_used = d_buf[:, :n_blocks_per_head].contiguous()        # (n_kv_h, pos*bpr)
        total_blocks = n_kv_h * n_blocks_per_head
        flat = dequantize_tq4(Tq4Tensor(
            qs=qs_used.reshape(total_blocks, 128),
            d=d_used.reshape(total_blocks),
            shape=(total_blocks * 256,),
        ), pi=self._pi, centroids=self._centroids)
        # Output is head-major flat. Reshape to (n_kv_h, pos, d_head) then
        # add batch dim to match the (1, n_kv_h, pos, d_head) contract.
        out = flat.reshape(n_kv_h, pos, d_head).unsqueeze(0).contiguous().float()
        self._memo[key] = (pos, out)
        return out

    def write_only(self, layer_idx: int, k_new: torch.Tensor,
                   v_new: torch.Tensor,
                   is_swa: bool = False, window_size: int = 512) -> None:
        """Write S new tokens' tq4 bytes WITHOUT paying the full-prefix
        dequant. Returns None. Used by the fused tq4 flash-attn path
        which consumes raw bytes via `k_qs[il]` / `v_qs[il]` and does
        not need the fp32 materialization.

        Updates layer_pos and invalidates the memo so a subsequent
        `_dequant_layer` call (e.g. from a shared-KV consumer) sees the
        new bytes.
        """
        self._write_bytes(layer_idx, k_new, v_new, is_swa, window_size)
        # Invalidate memo even though we didn't repopulate it; the next
        # _dequant_layer read will rebuild against the new pos+S bytes.
        self._memo.pop(("k", layer_idx), None)
        self._memo.pop(("v", layer_idx), None)

    def update(self, layer_idx: int, k_new: torch.Tensor, v_new: torch.Tensor,
               is_swa: bool = False, window_size: int = 512):
        """Quantize S new tokens' K/V and append to this layer's cache.

        Args:
          k_new, v_new: (B=1, n_kv_h, S, d_head). S>=1 supported.
          is_swa, window_size: recorded for post-forward trim.

        Returns:
          (k_full, v_full) as fp32 tensors of shape
          (1, n_kv_h, pos+S, d_head) — the full cached sequence for
          attention score compute. Routed through `_dequant_layer` so
          subsequent reads via `_Tq4ReadProxy` in the same decode step
          hit the memo instead of re-dequanting (Phase 1 perf fix).

        For the fused tq4 flash-attn path (decode, no partitions,
        d_head==256), prefer `write_only` — it skips this method's
        eager dequant entirely.
        """
        self._write_bytes(layer_idx, k_new, v_new, is_swa, window_size)
        # Drop the stale entry; _dequant_layer will recompute and rememoize.
        self._memo.pop(("k", layer_idx), None)
        self._memo.pop(("v", layer_idx), None)
        return self._dequant_layer(layer_idx, "k"), self._dequant_layer(layer_idx, "v")

    def _write_bytes(self, layer_idx: int, k_new: torch.Tensor,
                     v_new: torch.Tensor, is_swa: bool,
                     window_size: int) -> None:
        """Quantize + scatter S new tokens' bytes into per-head slots.
        Shared by `write_only` and `update`."""
        from calm.llm_computer.tq4_torch import quantize_tq4
        bpr = self._bpr[layer_idx]
        d_head = self._d_head[layer_idx]
        n_kv_h = self.cfg.n_heads_kv
        S = k_new.shape[2]
        pos = self.layer_pos[layer_idx]
        assert pos + S <= self.max_len, (
            f"layer {layer_idx}: pos {pos} + S {S} exceeds max_len {self.max_len}")

        # Head-major flatten: (1, n_kv_h, S, d_head) → (n_kv_h, S * bpr * 256,)
        # → flat (n_kv_h * S * bpr * 256,). Quantize once. Output blocks come
        # out in head-major order, i.e. [head0_blocks..., head1_blocks..., ...].
        k_flat = k_new[0].contiguous().float().reshape(-1)
        v_flat = v_new[0].contiguous().float().reshape(-1)
        k_q = quantize_tq4(k_flat, pi=self._pi, boundaries=self._boundaries)
        v_q = quantize_tq4(v_flat, pi=self._pi, boundaries=self._boundaries)

        # Reshape quantized output to (n_kv_h, S * bpr, 128) and (n_kv_h, S * bpr),
        # then scatter into the per-head slot range [pos*bpr : (pos+S)*bpr].
        s_blocks = S * bpr
        k_qs_view = k_q.qs.reshape(n_kv_h, s_blocks, 128)
        k_d_view = k_q.d.reshape(n_kv_h, s_blocks)
        v_qs_view = v_q.qs.reshape(n_kv_h, s_blocks, 128)
        v_d_view = v_q.d.reshape(n_kv_h, s_blocks)
        slot_start = pos * bpr
        slot_end = (pos + S) * bpr
        self.k_qs[layer_idx][:, slot_start:slot_end, :] = k_qs_view
        self.k_d[layer_idx][:, slot_start:slot_end] = k_d_view
        self.v_qs[layer_idx][:, slot_start:slot_end, :] = v_qs_view
        self.v_d[layer_idx][:, slot_start:slot_end] = v_d_view

        # Advance this layer's position. Memo invalidation is the caller's
        # responsibility (both write_only and update do it).
        self.layer_pos[layer_idx] = pos + S
        self._is_swa[layer_idx] = is_swa
        if is_swa:
            self._swa_window = window_size

    def trim_swa_storage(self) -> None:
        """For SWA layers, trim back to last `window_size` tokens.
        Direct byte copy — no re-quantization. tq4 blocks are per-256-elt
        with independent scales; head-major storage means the last `window`
        positions' bytes for each head sit at slots
        [(pos-window)*bpr : pos*bpr] and copy verbatim to [0 : window*bpr]."""
        window = self._swa_window
        for layer_idx, is_swa in self._is_swa.items():
            if not is_swa:
                continue
            pos = self.layer_pos[layer_idx]
            if pos <= window:
                continue
            bpr = self._bpr[layer_idx]
            keep_blocks = window * bpr
            src_start = (pos - window) * bpr
            src_end = pos * bpr
            # Per-head copy (slice along the per-head block dim).
            self.k_qs[layer_idx][:, :keep_blocks, :] = (
                self.k_qs[layer_idx][:, src_start:src_end, :].clone())
            self.k_d[layer_idx][:, :keep_blocks] = (
                self.k_d[layer_idx][:, src_start:src_end].clone())
            self.v_qs[layer_idx][:, :keep_blocks, :] = (
                self.v_qs[layer_idx][:, src_start:src_end, :].clone())
            self.v_d[layer_idx][:, :keep_blocks] = (
                self.v_d[layer_idx][:, src_start:src_end].clone())
            self.layer_pos[layer_idx] = window
            # Bytes 0..keep_blocks now hold a different sequence; invalidate.
            self._memo.pop(("k", layer_idx), None)
            self._memo.pop(("v", layer_idx), None)

    @property
    def k_cache(self):
        # For shared-KV reads: return the dequantized k_buf via a getter
        # — but kv_cache.k_cache[kv_src] in _forward_layer is dict-style.
        # Implement as a list of CURRENT-state tensors. Slow path; the
        # update() call also returns the dequantized form so own-KV
        # layers don't pay the read cost twice.
        return _Tq4ReadProxy(self, "k")

    @property
    def v_cache(self):
        return _Tq4ReadProxy(self, "v")

    def seq_len(self) -> int:
        """Effective sequence length (layer 0's write position)."""
        return self.layer_pos[0] if self.layer_pos else 0

    def clear(self):
        """Reset all layers to pos=0. Byte buffers preserved (overwritten on next update)."""
        for il in range(len(self.layer_pos)):
            self.layer_pos[il] = 0
        self._is_swa.clear()
        self._memo.clear()


class _Tq4ReadProxy:
    """Lazy-dequant proxy so kv_cache.k_cache[kv_src] returns the
    dequantized buffer for that source layer."""

    def __init__(self, cache: "KVCacheTq4", which: str):
        self.cache = cache
        self.which = which  # "k" or "v"

    def __getitem__(self, layer_idx: int):
        # Route through the memoized helper. _forward_layer's shared-KV path
        # immediately casts back to .float() at the consumer (see :1343-1344),
        # so returning fp32 here saves a half→float round-trip vs the old
        # eager .half() cast. The memo gives us O(1) instead of O(N) when
        # multiple consumer layers share the same source.
        c = self.cache
        if c.layer_pos[layer_idx] == 0:
            d_head = c._d_head[layer_idx]
            return torch.zeros(1, c.cfg.n_heads_kv, 0, d_head,
                                dtype=torch.float32, device=c.device)
        return c._dequant_layer(layer_idx, self.which)


class CardSlot:
    """Reserved (layer, channel-range) slot in Gemma's residual stream
    where a compiled card can compute alongside Gemma's attention.

    MVP install model: at the END of layer `layer_idx`, the slot's card
    receives the residual values at channels [ch_off:ch_off+d_card],
    computes its forward pass, and ADDS the result back to the same
    channels. This is the simplest "card alongside Gemma" pattern —
    full Round 29 per-sub-head attention partition is a follow-up.

    For preserving card output across subsequent layers, the substrate's
    full vision requires zeroing FFN_DOWN rows at the reserved channels
    on subsequent layers (so Gemma's FFN doesn't overwrite). For now,
    cards are installed at the LAST useful layer (e.g., the layer right
    before output_norm) so their output reaches the head intact.
    """

    def __init__(self, layer_idx: int, ch_off: int, card,
                 d_card: Optional[int] = None,
                 card_input_fn=None,
                 use_full_residual: bool = False,
                 output_fn=None):
        self.layer_idx = layer_idx
        self.ch_off = ch_off
        self.card = card
        # card_input_fn(input) → tensor for card.forward(). When
        # use_full_residual=False (default), input is h[..., ch_off:ch_off+d_card].
        # When True, input is the full residual h (lets chained cards read
        # from prior cards' output channels).
        self.card_input_fn = card_input_fn or (lambda x: x)
        self.use_full_residual = use_full_residual
        # output_fn(h, card_out, ch_lo, ch_hi) → updated h. If None, default
        # is to add card_out (shape-matched) into h[..., ch_lo:ch_hi].
        self.output_fn = output_fn
        # d_card = number of OUTPUT channels in Gemma's residual that the
        # card's output writes to. For Small2DTransformer compiled cards,
        # this is typically vocab_size (their forward returns logits).
        self.d_card = d_card

    def attach(self, model: "GemmaSubstrate", preserve: bool = True):
        """Resolve d_card if not given and register on the target layer.

        If `preserve=True` (default), the channel range is added to the
        model's reserved_channels list so subsequent layers' attn / ffn /
        per-layer-embed contributions are masked out — card output then
        flows through to output_norm + head intact, even when installed at
        an earlier layer.
        """
        if self.d_card is None:
            cfg = getattr(self.card, "config", None)
            self.d_card = (getattr(cfg, "vocab_size", None)
                           or getattr(cfg, "d_model", None))
            if self.d_card is None:
                raise ValueError("d_card not provided and card has no "
                                  "config.vocab_size/d_model to infer from")
        if not hasattr(model.layers[self.layer_idx], "card_slots"):
            model.layers[self.layer_idx].card_slots = []
        model.layers[self.layer_idx].card_slots.append(self)
        if preserve:
            model.reserved_channels.append(
                (self.ch_off, self.ch_off + self.d_card, self.layer_idx))


class VerificationHook:
    """Closes the verification loop: takes a card's verified output (e.g.,
    adder's argmax), maps it to a Gemma BPE token via a vocab mapping,
    and adds `boost` to that token's logit.

    Effect: when the card has high confidence in answer N, Gemma's final
    output is biased toward emitting the Gemma token corresponding to N.
    Run after the head + softcapping (so it can override Gemma's natural
    answer when the card disagrees).

    `min_margin` gates the bias: only fire when
      (max card logit) - (median card logit) >= min_margin
    This prevents the hook from firing on "no confident answer" signals
    (e.g., a recall card whose key doesn't match any stored correction
    returns all-zero logits → argmax=0 spuriously boosts Gemma's '0'
    token). Default 0 matches session 32 behavior (always fire).
    """

    def __init__(self, card_slot: "CardSlot", vocab_mapping: dict,
                 boost: float = 10.0, min_margin: float = 0.0):
        self.card_slot = card_slot
        self.vocab_mapping = vocab_mapping  # card_token_id → gemma_token_id
        self.boost = boost
        self.min_margin = min_margin

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        out = getattr(self.card_slot, "last_output", None)
        if out is None:
            return logits
        # Confidence gate: card's argmax must stand out above the median
        # logit by min_margin. Prevents boosting on flat (no-match) output.
        last = out[0, -1].float()
        peak = last.max().item()
        med = last.median().item()
        if (peak - med) < self.min_margin:
            return logits
        # Take the LAST position's argmax over the card's vocab.
        verified = int(out[0, -1].argmax().item())
        gemma_token = self.vocab_mapping.get(verified)
        if gemma_token is not None:
            logits[..., -1, gemma_token] = logits[..., -1, gemma_token] + self.boost
        return logits


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
        # Reserved channel ranges for card preservation. Each entry:
        # (ch_off, ch_hi, install_layer). For any layer strictly AFTER
        # install_layer, attn_output and ffn_output contributions to these
        # channels are zeroed so the card's value flows through intact.
        self.reserved_channels: list[tuple[int, int, int]] = []
        # Verification hooks — applied AFTER the head + softcapping. Each
        # hook reads a CardSlot.last_output and biases specific Gemma
        # vocab logits to inject the verified value back into Gemma's output.
        self.verification_hooks: list = []
        # Per-layer attention-mode partition. Maps layer_idx → list of
        # (sh_lo, sh_hi, mode) entries where mode ∈ {'softmax', 'hard_max'}.
        # Sub-heads outside any partition use Gemma's default grouped softmax.
        # Set by install_card_in_attention(mode=...).
        self.attention_partition: dict[int, list] = {}

    def __getstate__(self):
        # Drop the GGUF mmap reader (non-picklable, and mmap-resident raw
        # bytes are already redundant with preloaded _gpu_qs/_gpu_d). Also
        # drop transient per-forward state.
        state = self.__dict__.copy()
        state["_reader"] = None
        state["_per_layer_embd"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Subsequent forward passes run on the loaded GPU weights; the mmap
        # reader is not needed to keep them alive.
        # Rebuild the class-level Pi + centroids cache if missing — the tq4
        # kernel path and dequant path both read these, but they're class
        # attributes so they don't pickle with the instance.
        if MmapTq4Linear._shared_pi is None:
            from calm.llm_computer.tq4_torch import (
                build_pi, compute_lloyd_max_codebook,
            )
            # Infer device from any loaded tq4 buffer.
            device = "cuda"
            for lyr in self.layers:
                lin = getattr(lyr, "attn_q", None)
                if isinstance(lin, MmapTq4Linear) and lin._gpu_qs is not None:
                    device = str(lin._gpu_qs.device)
                    break
            pi = build_pi(device=device, source="torch")
            centroids, _ = compute_lloyd_max_codebook()
            MmapTq4Linear._shared_pi = pi
            MmapTq4Linear._shared_centroids = centroids.to(device)
            print(f"[gemma-substrate] restored Pi + centroids on {device}")

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

        # Per-layer token embedding (Q6_K, 10752×262144)
        # d_per_layer * n_layers = 256 * 42 = 10752 per token
        model.per_layer_token_embd = None
        model.per_layer_model_proj = None
        model.per_layer_proj_norm_w = None
        if "per_layer_token_embd.weight" in tensors:
            d_pl = cfg.d_per_layer * cfg.n_layers  # 10752
            print(f"[gemma-substrate] loading per-layer embedding (Q6_K, {cfg.vocab_size}×{d_pl})...")
            t_pl = tensors["per_layer_token_embd.weight"]
            model.per_layer_token_embd = GpuQ6KEmbedding(t_pl.data, cfg.vocab_size, d_pl)
        if "per_layer_model_proj.weight" in tensors:
            t_proj = tensors["per_layer_model_proj.weight"]
            # GGUF shape is (in, out) but bytes are math (out, in) row-major.
            import numpy as _np
            raw = _np.frombuffer(t_proj.data, dtype=_np.float16).copy()
            in_f, out_f = int(t_proj.shape[0]), int(t_proj.shape[1])
            w_math = torch.from_numpy(raw.astype(_np.float32)).reshape(out_f, in_f)
            model.per_layer_model_proj = w_math.T.contiguous()  # (in, out) for h @ proj
            print(f"[gemma-substrate] per_layer_model_proj: {model.per_layer_model_proj.shape}")
        if "per_layer_proj_norm.weight" in tensors:
            model.per_layer_proj_norm_w = _load_fp32(tensors, "per_layer_proj_norm.weight")

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
            # llama.cpp applies as theta_base / freq_factor (rope.cu:107).
            base_freqs = base_freqs / rope_freq_factors
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

    def preload_gpu(self, device: str = "cuda", compile_linears: bool = False):
        """Load all tq4 bytes onto GPU. Eliminates CPU↔GPU transfer during
        forward pass — dequant happens entirely on GPU.

        With `compile_linears=True`, also wraps every MmapTq4Linear's
        __call__ in torch.compile after preload — gives ~6x per-linear
        speedup by fusing the gather + matmul (microbenchmark on
        ffn_up: 6.81 ms → 1.11 ms).

        VRAM budget: ~5 GB for tq4 bytes + ~1 GB headroom = ~6 GB.
        Fits on RTX 4070 (8 GB) with room for activations.
        """
        import gc

        # Pre-cache Pi rotation matrix on GPU (256×256, used by every dequant)
        from calm.llm_computer.tq4_torch import build_pi, compute_lloyd_max_codebook
        pi = build_pi(device=device, source="torch")
        centroids, _ = compute_lloyd_max_codebook()
        centroids = centroids.to(device)
        # Monkey-patch dequantize_tq4 to use cached Pi+centroids by default,
        # but pass through caller-supplied kwargs (e.g. KVCacheTq4 supplies
        # its own pi/centroids).
        import calm.llm_computer.tq4_torch as tq4_mod
        _orig_dequant = tq4_mod.dequantize_tq4
        def _cached_dequant(q, pi=None, centroids=None, **kw):
            return _orig_dequant(q,
                                  pi=pi if pi is not None else MmapTq4Linear._shared_pi,
                                  centroids=centroids if centroids is not None
                                  else MmapTq4Linear._shared_centroids)
        tq4_mod.dequantize_tq4 = _cached_dequant
        # Also update our module-level import
        global dequantize_tq4
        dequantize_tq4 = _cached_dequant
        # And expose to MmapTq4Linear's fast path (skips the @Pi dequant
        # matmul, rotates x once instead — see _w_unrotated docstring).
        MmapTq4Linear._shared_pi = pi
        MmapTq4Linear._shared_centroids = centroids
        print(f"[gemma-substrate] cached Pi + centroids on {device}")

        # Move Q6_K embedding components to GPU
        self.token_embd.to_gpu(device)  # ~553 MB
        if self.per_layer_token_embd is not None:
            self.per_layer_token_embd.to_gpu(device)  # ~553 MB more
        if self.per_layer_model_proj is not None:
            self.per_layer_model_proj = self.per_layer_model_proj.to(device)

        # Move ALL small FP32 weights (norms, scales, rope freqs) to GPU once.
        # Without this, every _forward_layer call does ~7 .to(device) calls
        # for norm weights — measured at 603 ms / 3.2 sec of decode time.
        if self.output_norm_w is not None:
            self.output_norm_w = self.output_norm_w.to(device)
        if self.per_layer_proj_norm_w is not None:
            self.per_layer_proj_norm_w = self.per_layer_proj_norm_w.to(device)
        if self.rope_freqs_global is not None:
            self.rope_freqs_global = self.rope_freqs_global.to(device)
        if self.rope_freqs_swa is not None:
            self.rope_freqs_swa = self.rope_freqs_swa.to(device)
        for layer in self.layers:
            for attr in ("attn_norm_w", "post_attn_norm_w", "ffn_norm_w",
                         "post_ffw_norm_w", "post_norm_w",
                         "attn_q_norm_w", "attn_k_norm_w",
                         "layer_output_scale"):
                t = getattr(layer, attr, None)
                if t is not None:
                    setattr(layer, attr, t.to(device))

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
        if compile_linears:
            enable_compile_tq4()
            print(f"[gemma-substrate] torch.compile enabled (one shared "
                  f"compiled kernel for all 378 tq4 linears)")

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
        # Required by torch.compile(mode="reduce-overhead") CUDAGraphs to
        # release prior step's output buffers before this step writes to them.
        if device == "cuda":
            try:
                torch.compiler.cudagraph_mark_step_begin()
            except (AttributeError, RuntimeError):
                pass

        # Token embedding (Q6_K, dequants only needed rows)
        h = self.token_embd[token_ids].to(device)
        h = h * math.sqrt(cfg.d_model)

        # Per-layer embedding (computed once from input tokens, used by all layers)
        # Matches gemma4-iswa.cpp project_per_layer_inputs()
        self._per_layer_embd = None
        if self.per_layer_token_embd is not None:
            d_pl = cfg.d_per_layer * cfg.n_layers  # 10752
            # Look up per-layer token embedding
            pl_embd = self.per_layer_token_embd[token_ids]  # (B, S, 10752) on GPU
            pl_embd = pl_embd * math.sqrt(cfg.d_per_layer)
            # Reshape to (B, S, n_layers, d_per_layer) — norm applies to last dim
            pl_embd = pl_embd.reshape(B, S, cfg.n_layers, cfg.d_per_layer)
            # Project main embedding to per-layer space
            if self.per_layer_model_proj is not None:
                h_proj = h @ self.per_layer_model_proj  # (B, S, 10752)
                h_proj = h_proj * (1.0 / math.sqrt(cfg.d_model))
                h_proj = h_proj.reshape(B, S, cfg.n_layers, cfg.d_per_layer)
                if self.per_layer_proj_norm_w is not None:
                    h_proj = _rms_norm(h_proj, self.per_layer_proj_norm_w, cfg.rms_norm_eps)
                pl_embd = (pl_embd + h_proj) * (1.0 / math.sqrt(2.0))
            # Store as list indexed by layer: each is (B, S, d_per_layer)
            self._per_layer_embd = [pl_embd[:, :, i, :] for i in range(cfg.n_layers)]

        for i, layer in enumerate(self.layers):
            h = self._forward_layer(h, layer, i, kv_cache=kv_cache,
                                     start_pos=start_pos)

        # Post-forward SWA storage trim — keep cache memory bounded
        # by window_size for SWA layers. Safe to do here because all
        # shared-KV consumers have already run with the full window.
        if (isinstance(kv_cache, (KVCache, KVCacheTq4))
                and hasattr(kv_cache, "trim_swa_storage")):
            kv_cache.trim_swa_storage()

        h = _rms_norm(h, self.output_norm_w, cfg.rms_norm_eps)

        # Output head: chunked Q6_K dequant + matmul on GPU
        h_last = h[:, -1:, :]  # (B, 1, d_model) stays on GPU
        logits = self.token_embd.output_logits(h_last)  # (B, 1, vocab) on GPU

        # Logit softcapping: tanh(logits / cap) * cap
        # Prevents extreme logits from dominating (Gemma 4 uses cap=30.0)
        cap = 30.0
        logits = torch.tanh(logits / cap) * cap

        # Verification loop: card outputs feed back into Gemma's logits.
        # Each hook reads a CardSlot.last_output (saved during _forward_layer),
        # picks a verified token, and adds a bias to Gemma's vocab logit
        # for the corresponding Gemma BPE token. Closes the loop:
        #   Gemma residual → card → Gemma logits
        for hook in self.verification_hooks:
            logits = hook(logits)

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
        cur = _rms_norm(h, layer.attn_norm_w, cfg.rms_norm_eps)

        q = layer.attn_q(cur)
        # Per-layer head dim — Q tells us SWA (256) vs global (512)
        q_total = q.shape[-1]
        d_head_q = q_total // cfg.n_heads_q
        is_global = d_head_q > cfg.d_head

        # Determine where this layer's K/V comes from. Layers 24-41 reuse
        # the cache of an earlier layer (last SWA or last global before the
        # shared block) and skip their own K/V projection entirely.
        kv_src = cfg.kv_source_layer(layer_idx, is_swa=not is_global)
        own_kv = (kv_src == layer_idx)

        q = q.reshape(B, S, cfg.n_heads_q, d_head_q).transpose(1, 2)
        if layer.attn_q_norm_w is not None:
            q = _rms_norm(q, layer.attn_q_norm_w, cfg.rms_norm_eps)

        # RoPE on Q (always — every layer rotates its own Q with current pos).
        # For KVCacheStatic (CUDA Graph capture), use a GPU-tensor index_select
        # so the slice shape stays fixed across positions.
        freqs = self.rope_freqs_global if is_global else self.rope_freqs_swa
        if isinstance(kv_cache, KVCacheStatic):
            freqs_used = torch.index_select(freqs, 0, kv_cache.pos.unsqueeze(0))
        else:
            freqs_used = freqs[start_pos:]
        q = _apply_rope(q, freqs_used)

        # Phase 2 fused tq4 flash-attn eligibility — checked BEFORE the cache
        # write so own-KV layers can use `write_only` (skip the eager dequant
        # in `KVCacheTq4.update`). Shared-KV layers read raw bytes directly.
        partitions = self.attention_partition.get(layer_idx, [])
        # N-gate: fused wins empirically at 128 < cached_kv_len < 2048
        # (bench 2026-04-20). Outside that band, Phase 1 memo is faster.
        fused_tq4 = (
            _use_triton
            and _use_fused_flash_attn
            and isinstance(kv_cache, KVCacheTq4)
            and S == 1
            and not partitions
            and kv_cache._d_head[kv_src] == 256  # MVP: BPR=1 only
            and 128 < kv_cache.layer_pos[kv_src] < 2048  # sweet-spot gate
        )

        if own_kv:
            k_new = layer.attn_k(cur)
            v_new = layer.attn_v(cur)
            d_head_kv = k_new.shape[-1] // cfg.n_heads_kv
            k_new = k_new.reshape(B, S, cfg.n_heads_kv, d_head_kv).transpose(1, 2)
            v_new = v_new.reshape(B, S, cfg.n_heads_kv, d_head_kv).transpose(1, 2)
            if layer.attn_k_norm_w is not None:
                k_new = _rms_norm(k_new, layer.attn_k_norm_w, cfg.rms_norm_eps)
            v_rms = torch.sqrt(torch.mean(v_new * v_new, dim=-1, keepdim=True) + cfg.rms_norm_eps)
            v_new = v_new / v_rms
            k_new = _apply_rope(k_new, freqs_used)
            if kv_cache is not None:
                if fused_tq4:
                    kv_cache.write_only(layer_idx, k_new, v_new,
                                         is_swa=not is_global,
                                         window_size=cfg.sliding_window)
                    k_full = v_full = None  # not needed — fused path uses raw bytes
                else:
                    k_full, v_full = kv_cache.update(layer_idx, k_new, v_new,
                                                      is_swa=not is_global,
                                                      window_size=cfg.sliding_window)
            else:
                k_full, v_full = k_new, v_new
        else:
            # Shared-KV layer — read source layer's cache, no own projection.
            assert kv_cache is not None
            if fused_tq4:
                k_full = v_full = None
            else:
                k_full = kv_cache.k_cache[kv_src].float()
                v_full = kv_cache.v_cache[kv_src].float()

        # GQA expand (slow path only — fused kernel does GQA per-program).
        if not fused_tq4 and cfg.n_heads_kv < cfg.n_heads_q:
            repeat = cfg.n_heads_q // cfg.n_heads_kv
            k_full = k_full.repeat_interleave(repeat, dim=1)
            v_full = v_full.repeat_interleave(repeat, dim=1)

        # Attention scores — Gemma 4 uses f_attention_scale = 1.0
        # (no /sqrt(d_head)). See llama-model.cpp:1273.
        if fused_tq4:
            S_kv = kv_cache.layer_pos[kv_src]
        else:
            S_kv = k_full.shape[2]

        # Build attention mask once, used by both paths.
        # For SWA layers (not global), apply BOTH causal mask AND
        # sliding-window mask. K cache stores full sequence (per the
        # update() change); attention here masks each Q position's
        # window to the last `sliding_window` K positions.
        attn_mask = None
        if isinstance(kv_cache, KVCacheStatic):
            attn_mask = kv_cache.valid_mask[None, None, None, :]
        elif not is_global and S_kv > 0:
            # SWA layer — build (S, S_kv) mask with causal + window.
            # Q index i has absolute position (S_kv - S) + i.
            # K index j has absolute position j.
            # Causal: K_abs > Q_abs → mask.
            # Window: K_abs < Q_abs - window + 1 → mask.
            offset = S_kv - S
            i_idx = torch.arange(S, device=device).unsqueeze(1)  # (S, 1)
            j_idx = torch.arange(S_kv, device=device).unsqueeze(0)  # (1, S_kv)
            i_abs = i_idx + offset
            window = cfg.sliding_window
            causal = j_idx > i_abs
            out_of_window = j_idx < (i_abs - window + 1)
            attn_mask = (causal | out_of_window)[None, None, :, :]
        elif S_kv == S:
            # Global layer prefill — causal mask only
            attn_mask = torch.triu(
                torch.ones(S, S_kv, dtype=torch.bool, device=device),
                diagonal=1)[None, None, :, :]
        # Global layer decode (S_kv > S, S=1): no mask, attend to all K

        if fused_tq4:
            # Fused tq4 flash-attn path. Slice the head-major bytes for the
            # active prefix length, pre-rotate Q, run kernel, reshape back.
            from calm.llm_computer.tq4_flash_attn import (
                fused_tq4_flash_attn_decode,
            )
            bpr_kv = kv_cache._bpr[kv_src]
            n_blocks_used = S_kv * bpr_kv
            k_qs = kv_cache.k_qs[kv_src][:, :n_blocks_used, :].contiguous()
            k_d = kv_cache.k_d[kv_src][:, :n_blocks_used].contiguous()
            v_qs = kv_cache.v_qs[kv_src][:, :n_blocks_used, :].contiguous()
            v_d = kv_cache.v_d[kv_src][:, :n_blocks_used].contiguous()

            # Squeeze (B=1, n_heads_q, S=1, d_head) → (n_heads_q, d_head)
            q_2d = q[0, :, 0, :].contiguous()
            # Pre-rotate (Pi.T applied here so the kernel can skip per-block
            # inverse rotation; out is post-rotated by Pi back to normal).
            q_rot = q_2d @ kv_cache._pi.T

            # Build (S_kv,) additive mask from the existing bool mask.
            if attn_mask is None:
                fused_mask = torch.zeros(S_kv, dtype=torch.float32, device=device)
            else:
                # attn_mask shape (1, 1, 1, S_kv) — squeeze to (S_kv,)
                bool_mask = attn_mask.reshape(-1)[:S_kv]
                fused_mask = torch.where(
                    bool_mask, torch.full_like(bool_mask, float("-inf"),
                                                dtype=torch.float32),
                    torch.zeros_like(bool_mask, dtype=torch.float32),
                )

            out_fused = fused_tq4_flash_attn_decode(
                q_rot, k_qs, k_d, v_qs, v_d,
                kv_cache._centroids, kv_cache._pi, fused_mask,
                softcap=0.0,  # Gemma 4 doesn't softcap attention scores
            )
            # (n_heads_q, d_head) → (B=1, n_heads_q, S=1, d_head)
            cur = out_fused.unsqueeze(0).unsqueeze(2)
        elif not partitions:
            # Fast path: pure Gemma grouped softmax.
            scores = torch.einsum("bhid,bhjd->bhij", q, k_full)
            if attn_mask is not None:
                scores = scores.masked_fill(attn_mask, float("-inf"))
            weights = F.softmax(scores, dim=-1)
            cur = torch.einsum("bhij,bhjd->bhid", weights, v_full)
        else:
            # Per-sub-head dispatch. Carve out card sub-heads from the
            # grouped-softmax sum (zero their Q contribution), compute
            # Gemma's standard attention on the rest, then add per-sub-head
            # attention for the card sub-heads with their own mode.
            q_gemma = q.clone()
            for (sh_lo, sh_hi, _mode) in partitions:
                q_gemma[..., sh_lo:sh_hi] = 0
            scores = torch.einsum("bhid,bhjd->bhij", q_gemma, k_full)
            if attn_mask is not None:
                scores = scores.masked_fill(attn_mask, float("-inf"))
            weights = F.softmax(scores, dim=-1)
            cur = torch.einsum("bhij,bhjd->bhid", weights, v_full)

            for (sh_lo, sh_hi, mode) in partitions:
                d_part = sh_hi - sh_lo
                assert d_part % 2 == 0
                n_sub = d_part // 2
                # (B, H, S, n_sub, 2) per sub-head views
                q_c = q[..., sh_lo:sh_hi].reshape(*q.shape[:-1], n_sub, 2)
                k_c = k_full[..., sh_lo:sh_hi].reshape(*k_full.shape[:-1], n_sub, 2)
                v_c = v_full[..., sh_lo:sh_hi].reshape(*v_full.shape[:-1], n_sub, 2)
                # Per-sub-head scores: (B, H, n_sub, S_q, S_kv)
                sc = torch.einsum("bhqni,bhkni->bhnqk", q_c, k_c)
                if attn_mask is not None:
                    # attn_mask shape (1, 1, S_q, S_kv) → broadcast over (n_sub)
                    sc = sc.masked_fill(attn_mask.unsqueeze(2), float("-inf"))
                if mode == "hard_max":
                    am = sc.argmax(dim=-1, keepdim=True)
                    w_c = torch.zeros_like(sc)
                    w_c.scatter_(-1, am, 1.0)
                elif mode == "softmax":
                    w_c = F.softmax(sc, dim=-1)
                else:
                    raise ValueError(f"unknown sub-head mode {mode!r}")
                # Apply: (B, H, S_q, n_sub, 2)
                out_c = torch.einsum("bhnqk,bhkni->bhqni", w_c, v_c)
                cur[..., sh_lo:sh_hi] = out_c.reshape(*out_c.shape[:-2], d_part)

        cur = cur.transpose(1, 2).reshape(B, S, q_total)

        # Output projection
        cur = layer.attn_output(cur)

        # Post-attention norm THEN residual (matches gemma4-iswa.cpp line 107-112)
        if layer.post_attn_norm_w is not None:
            cur = _rms_norm(cur, layer.post_attn_norm_w, cfg.rms_norm_eps)
        # Mask reserved channels: zero attn contribution so card output
        # in inpL (set by an earlier layer's card slot) is preserved.
        for ch_lo, ch_hi, install_layer in self.reserved_channels:
            if layer_idx > install_layer:
                cur[..., ch_lo:ch_hi] = 0
        attn_out = cur + inpL

        # --- FFN ---
        cur = _rms_norm(attn_out, layer.ffn_norm_w, cfg.rms_norm_eps)
        # gate+up share input — fuse into one Triton call when possible
        if (_use_triton
            and isinstance(layer.ffn_gate, MmapTq4Linear)
            and isinstance(layer.ffn_up, MmapTq4Linear)
            and layer.ffn_gate._gpu_qs is not None
            and layer.ffn_up._gpu_qs is not None):
            from calm.llm_computer.tq4_triton import tq4_linear_dual_triton
            gate, up = tq4_linear_dual_triton(
                cur,
                layer.ffn_gate._gpu_qs, layer.ffn_gate._gpu_d,
                layer.ffn_up._gpu_qs, layer.ffn_up._gpu_d,
                MmapTq4Linear._shared_pi, MmapTq4Linear._shared_centroids,
                out_features=layer.ffn_gate.out_features,
                in_features=layer.ffn_gate.in_features,
            )
        else:
            gate = layer.ffn_gate(cur)
            up = layer.ffn_up(cur)
        cur = F.gelu(gate, approximate="tanh") * up
        cur = layer.ffn_down(cur)

        # Post-FFN norm THEN residual (matches gemma4-iswa.cpp line 184-190)
        if layer.post_ffw_norm_w is not None:
            cur = _rms_norm(cur, layer.post_ffw_norm_w, cfg.rms_norm_eps)
        # Mask reserved channels: zero FFN contribution so card output
        # in attn_out is preserved.
        for ch_lo, ch_hi, install_layer in self.reserved_channels:
            if layer_idx > install_layer:
                cur[..., ch_lo:ch_hi] = 0
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
                proj_out = _rms_norm(proj_out, layer.post_norm_w, cfg.rms_norm_eps)
            # Mask reserved channels: zero per-layer-embed contribution.
            for ch_lo, ch_hi, install_layer in self.reserved_channels:
                if layer_idx > install_layer:
                    proj_out[..., ch_lo:ch_hi] = 0
            h = pe_in + proj_out

        # Layer output scale (gemma4-iswa.cpp line 216-219)
        if layer.layer_output_scale is not None:
            h = h * layer.layer_output_scale

        # Card slot dispatch — additive residual write at reserved channels.
        # The card sees the FULL residual h (so it can read from anywhere,
        # not just its own write range — enables chained cards that read
        # from an earlier card's output channels).
        slots = getattr(layer, "card_slots", None)
        if slots:
            for slot in slots:
                ch_lo, ch_hi = slot.ch_off, slot.ch_off + slot.d_card
                # Default: pass h_slice for backward compat; chained cards
                # set use_full_residual=True to receive h instead.
                if getattr(slot, "use_full_residual", False):
                    card_input = slot.card_input_fn(h)
                else:
                    card_input = slot.card_input_fn(h[..., ch_lo:ch_hi])
                with torch.no_grad():
                    card_out = slot.card(card_input)
                # Optional output_fn lets the card map its output shape
                # (e.g., (B, S_card, V)) back to the residual write shape
                # (B, S_residual, d_card).
                if getattr(slot, "output_fn", None) is not None:
                    h = slot.output_fn(h, card_out, ch_lo, ch_hi)
                elif card_out.shape == h[..., ch_lo:ch_hi].shape:
                    h[..., ch_lo:ch_hi] = h[..., ch_lo:ch_hi] + card_out
                # Save for downstream verification hooks (closes the loop:
                # card output → Gemma logit bias at end of forward).
                slot.last_output = card_out

        return h

    def convert_layer_to_fp32(self, layer_idx: int, attrs: tuple = None,
                               device: str = "cuda") -> dict:
        """Replace selected MmapTq4Linear in `layer_idx` with FP32GemmaLinear.
        Required before install_card_in_attention so card weights can be
        written without tq4 quant noise (Round 11 hybrid finding).

        Returns a dict mapping attr → new FP32GemmaLinear, useful for
        the caller to inspect or modify the FP32 weights directly.
        """
        if attrs is None:
            attrs = ("attn_q", "attn_k", "attn_v", "attn_output",
                     "ffn_gate", "ffn_up", "ffn_down")
        layer = self.layers[layer_idx]
        out = {}
        for attr in attrs:
            lin = getattr(layer, attr, None)
            if lin is None or not isinstance(lin, MmapTq4Linear):
                continue
            w_fp32 = lin.dequant(device=device).contiguous()
            new_lin = FP32GemmaLinear(w_fp32,
                                       in_features=lin.in_features,
                                       out_features=lin.out_features)
            setattr(layer, attr, new_lin)
            out[attr] = new_lin
        return out

    def install_card_in_attention(
        self,
        card,
        layer_idx: int,
        sub_head_offset: int,
        ch_off: int,
        d_card: int,
        card_layer: int = 0,
        mode: str = "grouped",
    ) -> dict:
        """Install a Small2DTransformer card's attention weights INTO
        Gemma's attn_q/k/v/output tensors at a specific sub-head slot.

        Round 29 in-attention install. Requires the layer to be FP32
        (call convert_layer_to_fp32 first). The card occupies:
          - INPUT channels [ch_off:ch_off+d_card] of Gemma's residual
          - sub_head_offset .. sub_head_offset + d_card // 2 SUB-HEADS
            of Gemma's first attention head (each sub-head = d_head=2)
          - OUTPUT channels [ch_off:ch_off+d_card] of Gemma's residual

        Card layout assumed (Small2DTransformer):
          - W_qkv.weight: (3*d_card, d_card)   — Q, K, V stacked
          - W_out.weight: (d_card, d_card)
          - d_head=2 (substrate invariant)

        Gemma layout (FP32 after conversion):
          - attn_q.weight: (in=d_model, out=H*d_head)  # SWA: 2560×2048
          - attn_k.weight: (in=d_model, out=H_kv*d_head_kv)
          - attn_v.weight: (in=d_model, out=H_kv*d_head_kv)
          - attn_output.weight: (in=H*d_head, out=d_model)

        Surgically zeros and writes:
          attn_q[ch_off:ch_off+d_card, sh_lo:sh_hi] = card.W_qkv[Q part].T
          attn_k[ch_off:ch_off+d_card, sh_lo:sh_hi] = card.W_qkv[K part].T
          attn_v[ch_off:ch_off+d_card, sh_lo:sh_hi] = card.W_qkv[V part].T
          attn_output[sh_lo:sh_hi, ch_off:ch_off+d_card] = card.W_out.T
        Other rows/cols at those slots are zeroed so the card's input
        comes ONLY from the reserved residual channels and its output
        goes ONLY to the reserved channels.
        """
        layer = self.layers[layer_idx]
        for attr in ("attn_q", "attn_k", "attn_v", "attn_output"):
            assert isinstance(getattr(layer, attr), FP32GemmaLinear), (
                f"layer {layer_idx} {attr} must be FP32 — "
                f"call convert_layer_to_fp32({layer_idx}) first")

        assert d_card % 2 == 0, "d_card must be even (d_head=2 invariant)"
        n_sub_heads = d_card // 2
        sh_lo = sub_head_offset * 2
        sh_hi = sh_lo + d_card

        # Small2DTransformer uses ModuleList for per-layer weights.
        # card.W_qkv is nn.ModuleList of Linear; pick the requested layer.
        if hasattr(card.W_qkv, "__getitem__"):
            c_w = card.W_qkv[card_layer].weight
            c_out = card.W_out[card_layer].weight
        else:
            c_w = card.W_qkv.weight
            c_out = card.W_out.weight

        # Card Q/K/V slices
        c_Q = c_w[0:d_card, :]              # (d_card, d_card)
        c_K = c_w[d_card:2*d_card, :]
        c_V = c_w[2*d_card:3*d_card, :]

        with torch.no_grad():
            # attn_q: zero target columns then write the d_card×d_card block.
            # Gemma stores (in, out); card needs to be at rows[ch_off:+d_card],
            # cols[sh_lo:sh_hi].
            for attr, c_M in (("attn_q", c_Q), ("attn_k", c_K), ("attn_v", c_V)):
                lin = getattr(layer, attr)
                # Zero ALL rows for these columns (so card input comes only
                # from reserved channels)
                lin.weight[:, sh_lo:sh_hi] = 0
                # Write card weights — transpose card (out, in) to (in, out)
                lin.weight[ch_off:ch_off+d_card, sh_lo:sh_hi] = c_M.T

            # attn_output: card's output at rows[sh_lo:sh_hi] (input dim),
            # cols[ch_off:ch_off+d_card] (output dim).
            # Zero ALL columns for these input rows so the card output goes
            # only to the reserved channels.
            attn_o = layer.attn_output
            attn_o.weight[sh_lo:sh_hi, :] = 0
            attn_o.weight[sh_lo:sh_hi, ch_off:ch_off+d_card] = c_out.T

        # Register attention mode for these sub-heads (mode='grouped' = default
        # Gemma softmax, no partition needed). 'hard_max' / 'softmax' enable
        # per-sub-head dispatch in _forward_layer.
        if mode in ("hard_max", "softmax"):
            self.attention_partition.setdefault(layer_idx, []).append(
                (sh_lo, sh_hi, mode))
        elif mode != "grouped":
            raise ValueError(f"unknown attention mode {mode!r}")

        return {
            "layer": layer_idx, "sub_heads": list(range(sub_head_offset,
                                                          sub_head_offset + n_sub_heads)),
            "ch_in": (ch_off, ch_off + d_card),
            "ch_out": (ch_off, ch_off + d_card),
            "card_d_model": d_card,
            "mode": mode,
        }

    def warmup(self, device: str = "cuda", seq_lens: tuple = (1, 6, 16)):
        """Trigger Triton kernel compilation for the listed sequence lengths
        so the first real generate() call doesn't pay the compile cost.
        Each unique (out_features, in_features, n_seq) shape needs one
        Triton compile (~50-300 ms). Without warmup, the first prefill at
        S=6 takes ~3.4 sec; after warmup it takes ~0.12 sec."""
        import time
        t0 = time.time()
        for s_len in seq_lens:
            dummy_ids = torch.zeros((1, s_len), dtype=torch.long, device=device)
            cache = KVCache(self.config.n_layers, device=device)
            with torch.no_grad():
                _ = self.forward(dummy_ids, device=device, kv_cache=cache, start_pos=0)
        torch.cuda.synchronize()
        print(f"[gemma-substrate] warmup compiled kernels for "
              f"S={list(seq_lens)} in {time.time()-t0:.1f}s")

    def generate_with_graph(self, prompt: str, tokenizer, max_tokens: int = 64,
                            device: str = "cuda", stop_on_eos: bool = True,
                            max_len: int = 1024) -> dict:
        """Greedy generation accelerated by CUDA Graphs.

        Prefill runs on the dynamic KVCache (variable shape), then state is
        transferred to a KVCacheStatic (fixed shape) and a CUDA Graph
        captures one decode step. Each subsequent token is one graph
        replay — eliminates ~95% of Python and kernel-launch overhead.

        Measured on RTX 4070M with the Triton kernels: 4.68x over
        non-graph decode, 38 tok/s vs 8 tok/s. Returns
        {'text', 'token_ids', 'prefill_s', 'decode_s'}.
        """
        import time
        ids = tokenizer.encode(prompt)

        # Prefill on dynamic cache.
        dyn = KVCache(self.config.n_layers, device=device)
        t0 = time.time()
        with torch.no_grad():
            logits = self.forward(torch.tensor([ids]), device=device,
                                   kv_cache=dyn, start_pos=0)
        next_id = int(logits[0, -1].argmax().item())
        prefill_s = time.time() - t0

        # Transfer to static cache.
        static = KVCacheStatic(self, max_len=max_len, device=device)
        prefill_len = dyn.seq_len()
        for il in dyn.k_cache:
            L = dyn.k_cache[il].shape[2]
            static.k_buf[il][:, :, :L, :] = dyn.k_cache[il].half()
            static.v_buf[il][:, :, :L, :] = dyn.v_cache[il].half()
        static.set_pos(prefill_len)

        # Side-stream warmup before capture (CUDA Graphs requirement).
        input_buf = torch.tensor([[next_id]], dtype=torch.long, device=device)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                with torch.no_grad():
                    _ = self.forward(input_buf, device=device,
                                      kv_cache=static, start_pos=0)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        # Reset static cache state — warmup wrote into it.
        for il in dyn.k_cache:
            L = dyn.k_cache[il].shape[2]
            static.k_buf[il][:, :, :L, :] = dyn.k_cache[il].half()
            static.v_buf[il][:, :, :L, :] = dyn.v_cache[il].half()
        static.set_pos(prefill_len)

        # Capture graph
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            with torch.no_grad():
                captured = self.forward(input_buf, device=device,
                                         kv_cache=static, start_pos=0)
        # Capture executed the forward — reset state again.
        for il in dyn.k_cache:
            L = dyn.k_cache[il].shape[2]
            static.k_buf[il][:, :, :L, :] = dyn.k_cache[il].half()
            static.v_buf[il][:, :, :L, :] = dyn.v_cache[il].half()
        static.set_pos(prefill_len)
        torch.cuda.synchronize()

        # Decode loop via graph replay — GPU-only state updates,
        # one CPU sync at the end for the whole token batch.
        n_steps = max_tokens - 1
        tokens_buf = torch.zeros(n_steps, dtype=torch.long, device=device)
        eos_id = tokenizer.EOS_ID

        t1 = time.time()
        for i in range(n_steps):
            static.set_pos(prefill_len + i)
            g.replay()
            next_id_t = captured[0, -1].argmax()  # 0-d GPU tensor
            tokens_buf[i] = next_id_t
            input_buf.copy_(next_id_t.reshape(1, 1))
        torch.cuda.synchronize()
        decode_s = time.time() - t1

        # One CPU sync: pull all decoded tokens at once
        gen_after_first = tokens_buf.tolist()
        generated = [next_id]
        for tid in gen_after_first:
            if stop_on_eos and tid == eos_id:
                break
            generated.append(tid)
        text = tokenizer.decode(generated)
        return {"text": text, "token_ids": generated,
                "prefill_s": prefill_s, "decode_s": decode_s}

    def generate(self, prompt: str, tokenizer, max_tokens: int = 64,
                 device: str = "cuda", stop_on_eos: bool = True,
                 use_tq4_kv: bool = False) -> dict:
        """Greedy generation. Returns {'text', 'token_ids', 'prefill_s', 'decode_s'}.
        Fresh KV cache per call — caller manages multi-turn state.

        With use_tq4_kv=True, uses KVCacheTq4 (real tq4 byte storage,
        ~3.6x smaller KV memory). Supports multi-token prefill.
        """
        import time
        ids = tokenizer.encode(prompt)
        if use_tq4_kv:
            # Allocate enough for prompt + max_tokens decode.
            cache = KVCacheTq4(self, max_len=len(ids) + max_tokens + 8,
                                device=device)
        else:
            cache = KVCache(self.config.n_layers, device=device)
        t0 = time.time()
        with torch.no_grad():
            logits = self.forward(torch.tensor([ids]), device=device,
                                   kv_cache=cache, start_pos=0)
        next_id = int(logits[0, -1].argmax().item())
        prefill_s = time.time() - t0
        generated = [next_id]
        for _ in range(max_tokens - 1):
            if stop_on_eos and next_id == tokenizer.EOS_ID:
                break
            with torch.no_grad():
                logits = self.forward(
                    torch.tensor([[next_id]]), device=device,
                    kv_cache=cache, start_pos=len(ids) + len(generated) - 1)
            next_id = int(logits[0, -1].argmax().item())
            generated.append(next_id)
        decode_s = time.time() - t0 - prefill_s
        text = tokenizer.decode(generated)
        return {"text": text, "token_ids": generated,
                "prefill_s": prefill_s, "decode_s": decode_s}


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
    p.add_argument("--tq4-kv", action="store_true",
                   help="Inject tq4 quant noise into KV cache (correctness check)")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile the tq4 kernel (modest help; recompiles per shape)")
    p.add_argument("--triton", action="store_true",
                   help="Use Triton fused dequant-matvec kernel (~10-17x per ffn linear)")
    p.add_argument("--cuda-graph", action="store_true",
                   help="Use CUDA Graph capture/replay for decode (~5x per step)")
    args = p.parse_args()
    if args.triton:
        enable_triton_tq4(True)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    model = GemmaSubstrate.from_gguf(args.gguf, max_len=args.max_len)

    # Preload tq4 to GPU for fast dequant
    if device == "cuda":
        model.preload_gpu(device, compile_linears=args.compile)

    from calm.llm_computer.synth.gemma_tokenizer import GemmaTokenizer
    tok = GemmaTokenizer.from_gguf(args.gguf)

    ids = tok.encode(args.prompt)
    print(f"\n[substrate] prompt: \"{args.prompt}\" ({len(ids)} tokens)")
    print(f"[substrate] device: {device}")
    print(f"[substrate] generating {args.tokens} tokens with KV cache...")

    if args.cuda_graph:
        # Warm up Triton compile cache so prefill timing reflects steady state
        prompt_len = len(tok.encode(args.prompt))
        model.warmup(device=device, seq_lens=(1, prompt_len))
        t_total = time.time()
        out = model.generate_with_graph(args.prompt, tok,
                                         max_tokens=args.tokens,
                                         device=device, max_len=args.max_len)
        total = time.time() - t_total
        full_text = args.prompt + out["text"]
        print(f"  prefill: {out['prefill_s']:.1f}s ({len(ids)} tokens)")
        print(f"  decode:  {out['decode_s']:.1f}s ({len(out['token_ids'])} tokens, "
              f"{len(out['token_ids']) / out['decode_s']:.2f} tok/s steady)")
        print(f"\n[substrate] output: \"{full_text}\"")
        n_gen = len(out["token_ids"])
        print(f"[substrate] {n_gen} tokens in {total:.1f}s ({n_gen / total:.2f} tok/s)")
        return

    # Prefill: process entire prompt at once, populate KV cache
    cfg = model.config
    cache = KVCache(cfg.n_layers, device=device, use_tq4=args.tq4_kv)
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
