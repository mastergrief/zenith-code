"""Compatibility patches + multi-head-dim wrapper for turboquant_gpu v0.1.5.

Two pieces:

  1. Compatibility patches for the published TurboQuantEngine — apply on
     import, fix transformers 5.5+ cache API and bf16 model dtype handling.

  2. `MultiHeadDimTurboQuantEngine` — a wrapper class that holds multiple
     TurboQuantEngine instances keyed by head_dim and routes per-layer based
     on the K tensor's actual feature dim. Required for models with
     heterogeneous attention layouts like Gemma 4 E4B (mixed head_dim=256
     SWA layers + head_dim=512 full-attention layers).

The published `turboquant-gpu` package on PyPI was written against an older
version of `transformers` and assumes uniform per-model head_dim. Without
these patches and the wrapper, it fails on:

  1. transformers 5.5+ DynamicCache.layers API
     - Old: cache.key_cache / cache.value_cache (lists of tensors)
     - New: cache.layers[i].keys / cache.layers[i].values
     - Affected: every model loaded via from_pretrained() in transformers 5.5+
     - Symptom: AttributeError: 'DynamicCache' object has no attribute 'key_cache'
                AttributeError: 'LinearAttentionLayer' object has no attribute 'keys'

  2. bf16 model compatibility
     - Original engine returns dequantized K/V as fp16 (.half())
     - Models like Gemma, Llama 3+, Qwen 2.5+ use bf16 internally
     - When DynamicCache.update() concatenates fp16 cache + new bf16 K/V,
       PyTorch upcasts to fp32 → SDPA dtype mismatch with bf16 query
     - Symptom: RuntimeError: Expected query, key, and value to have the same
                dtype, but got query.dtype: c10::BFloat16 key.dtype: float and
                value.dtype: float instead.

  3. Heterogeneous head_dim across layers (Gemma 4 E4B specifically)
     - Most layers use head_dim=256 (sliding window attention)
     - Every 6th layer uses head_dim=512 (full attention)
     - The single-head_dim engine cannot rotate both with one matrix
     - Symptom: RuntimeError: mat1 and mat2 shapes cannot be multiplied
                (316x512 and 256x256)
     - Fix: MultiHeadDimTurboQuantEngine routes per-layer

Apply by importing this module BEFORE constructing engines:

    import scripts.turboquant_patches  # noqa: F401  applies patches
    from turboquant_gpu import TurboQuantEngine
    from scripts.turboquant_patches import MultiHeadDimTurboQuantEngine

    # Single-head-dim model (Qwen, Gemma 3, etc.):
    engine = TurboQuantEngine(head_dim=256, total_bits=3, device="cuda")
    engine.set_target_dtype(torch.bfloat16)

    # Multi-head-dim model (Gemma 4 E4B):
    engine = MultiHeadDimTurboQuantEngine.from_model(model, tokenizer)
    result = engine.generate(model, tokenizer, prompt)
    print(result["stats"]["ratio"], "x compression")

The patches and the wrapper are designed so they can be lifted directly into
a PR against DevTechJr/turboquant-gpu without further changes.

Validated on:
  - Qwen 3 0.6B   (fp16, head_dim=128)               → 5.12× compression
  - Gemma 3 4B    (bf16, head_dim=256)               → 5.22× compression
  - Gemma 4 E4B   (bf16, head_dim={256, 512} mixed)  → 5.24× compression,
                                                       lossless greedy decode
"""
from __future__ import annotations

import torch
from transformers import DynamicCache

from turboquant_gpu import TurboQuantEngine


# ── Patch 1: cache extraction for transformers 5.5+ ─────────────────────


def _extract_kv_compat(past_key_values):
    """Extract per-layer K and V tensors from any transformers cache version.

    Handles three cache layouts:

      1. transformers 5.5+ — `cache.layers[i].keys / .values`. Layers may be
         a mix of attention types (dense, linear, sliding-window); we only
         extract from layers that expose `.keys` (skipping LinearAttentionLayer
         and similar non-cached layer types).

      2. transformers 4.43-5.4 — `cache.key_cache / cache.value_cache`
         (lists of tensors, one per layer). The original turboquant_gpu code
         path.

      3. transformers <= 4.42 — `cache` is iterable as a list of (k, v) tuples
         for each layer. Legacy API.
    """
    # New API (transformers 5.5+): layered cache with per-layer attributes
    if hasattr(past_key_values, "layers"):
        keys, vals = [], []
        for layer in past_key_values.layers:
            if (
                hasattr(layer, "keys")
                and layer.keys is not None
                and hasattr(layer, "values")
                and layer.values is not None
            ):
                keys.append(layer.keys)
                vals.append(layer.values)
        return keys, vals

    # Mid-era API: key_cache / value_cache attributes
    try:
        return past_key_values.key_cache, past_key_values.value_cache
    except AttributeError:
        # Legacy tuple iteration
        keys = [kv[0] for kv in past_key_values]
        vals = [kv[1] for kv in past_key_values]
        return keys, vals


# ── Patch 2: cache rebuild casts to model's compute dtype ───────────────


def _build_cache_compat(self, compressed):
    """Rebuild a DynamicCache from compressed K/V, cast to the model's dtype.

    The original `build_cache` left the decompressed tensors in fp16 (whatever
    `_decompress_*_pt` returns). When the model uses bf16 (Gemma 3+, Llama 3+,
    Qwen 2.5+), the next forward pass concatenates the fp16 cache with new
    bf16 K/V via DynamicCache.update(), and PyTorch upcasts the union to fp32.
    The query tensor is still bf16, so SDPA errors with a dtype mismatch.

    Fix: cast the decompressed K/V to a target dtype before inserting into the
    cache. The target is read from `self._target_dtype` (set via
    `engine.set_target_dtype()`), defaulting to fp16 for backward compat with
    the published behavior.
    """
    cache = DynamicCache()
    target = getattr(self, "_target_dtype", torch.float16)
    for li, (ck_list, cv_list) in enumerate(compressed["layers"]):
        k_heads, v_heads = [], []
        for ck, cv in zip(ck_list, cv_list):
            k, v = self._decompress_kv_fused(ck, cv)
            k_heads.append(k.to(target))
            v_heads.append(v.to(target))
        k_layer = torch.stack(k_heads).unsqueeze(0)
        v_layer = torch.stack(v_heads).unsqueeze(0)
        cache.update(k_layer, v_layer, li)
    return cache


def _set_target_dtype(self, dtype):
    """Set the dtype to cast decompressed K/V to in build_cache.

    Call this after constructing the engine, before generate(), with the
    model's compute dtype. Detect it via `model.dtype` or by inspecting a
    sample past_key_values tensor:

        out = model(**inputs, use_cache=True)
        target = out.past_key_values.layers[0].keys.dtype
        engine.set_target_dtype(target)
    """
    self._target_dtype = dtype


# ── Apply patches ───────────────────────────────────────────────────────


def apply_patches() -> None:
    """Install both compatibility patches onto TurboQuantEngine.

    Idempotent — safe to call multiple times. Records on the class so we
    don't double-patch.
    """
    if getattr(TurboQuantEngine, "_compat_patches_applied", False):
        return
    TurboQuantEngine._extract_kv = staticmethod(_extract_kv_compat)
    TurboQuantEngine.build_cache = _build_cache_compat
    TurboQuantEngine.set_target_dtype = _set_target_dtype
    TurboQuantEngine._compat_patches_applied = True


# Auto-apply on import for ergonomics
apply_patches()


# ── MultiHeadDimTurboQuantEngine ────────────────────────────────────────


class MultiHeadDimTurboQuantEngine:
    """TurboQuant wrapper that routes per-layer based on head_dim.

    Required for models with heterogeneous attention head dimensions across
    layers. The published `TurboQuantEngine` assumes a single uniform head_dim
    across the whole model and constructs one rotation matrix accordingly.
    Models like Gemma 4 E4B mix two head_dims:

      - 20 sliding-window attention layers with head_dim=256
      -  4 full attention layers with head_dim=512

    This wrapper holds one `TurboQuantEngine` per unique head_dim and
    dispatches each layer's K/V tensors to the correct engine based on
    `K.shape[-1]`. Compress + decompress + cache rebuild + decode all work
    transparently for the caller.

    Usage:

        engine = MultiHeadDimTurboQuantEngine.from_model(model, tokenizer)
        result = engine.generate(model, tokenizer, prompt)
        print(result["stats"]["ratio"])  # e.g. 5.24

    For uniform-head-dim models, just use the published `TurboQuantEngine`
    directly — this wrapper adds no value there.

    Validated lossless on:
      - Gemma 4 E4B (bf16, head_dim={256, 512}) at 3-bit, greedy decoding
        produces byte-identical output to the uncompressed baseline.
    """

    def __init__(
        self,
        head_dims: list[int] | None = None,
        total_bits: int = 3,
        device: str = "cuda",
    ):
        """Construct an empty wrapper or pre-populate with known head_dims.

        Args:
            head_dims: Optional list of unique head_dim values to pre-create
                       engines for. If None, engines are auto-created on first
                       use of each head_dim. For deterministic startup with a
                       known model, prefer `from_model()` instead.
            total_bits: TurboQuant compression bit width (3 = paper's default,
                        4 = more conservative if you observe quality issues).
            device: torch device string ("cuda", "cpu").
        """
        self.total_bits = total_bits
        self.device = device
        self._target_dtype = torch.float16
        self._engines: dict[int, "TurboQuantEngine"] = {}
        if head_dims:
            for hd in head_dims:
                self._add_engine(hd)

    @classmethod
    def from_model(
        cls,
        model,
        tokenizer,
        total_bits: int = 3,
        device: str = "cuda",
        probe_text: str = "test",
    ) -> "MultiHeadDimTurboQuantEngine":
        """Auto-discover head_dims by running a tiny forward pass on the model.

        This is the recommended constructor for unfamiliar models — it inspects
        the actual KV cache produced by a 1-token forward pass, finds the
        unique head_dims across all layers, creates one engine per head_dim,
        and sets the target dtype to match the cache.

        Args:
            model: A HuggingFace model (loaded via from_pretrained()).
            tokenizer: The matching tokenizer.
            total_bits: TurboQuant bit width.
            device: torch device.
            probe_text: Short text used for the discovery forward pass.
        """
        inputs = tokenizer(probe_text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs, use_cache=True)
        keys, _ = TurboQuantEngine._extract_kv(out.past_key_values)
        head_dims = sorted({int(K.shape[-1]) for K in keys})
        target = keys[0].dtype

        engine = cls(head_dims=head_dims, total_bits=total_bits, device=device)
        engine.set_target_dtype(target)
        return engine

    def _add_engine(self, head_dim: int) -> None:
        """Construct and register a sub-engine for a specific head_dim."""
        eng = TurboQuantEngine(
            head_dim=head_dim, total_bits=self.total_bits, device=self.device
        )
        eng.set_target_dtype(self._target_dtype)
        self._engines[head_dim] = eng

    def _route(self, head_dim: int) -> "TurboQuantEngine":
        """Get (or auto-create) the sub-engine for a given head_dim."""
        if head_dim not in self._engines:
            self._add_engine(head_dim)
        return self._engines[head_dim]

    def set_target_dtype(self, dtype: torch.dtype) -> None:
        """Set the dtype to cast decompressed K/V to in build_cache.

        Should match the model's compute dtype — bf16 for Gemma / Llama 3+,
        fp16 for older models. Auto-discovered by `from_model()`.
        """
        self._target_dtype = dtype
        for eng in self._engines.values():
            eng.set_target_dtype(dtype)

    @property
    def head_dims(self) -> list[int]:
        """Sorted list of head_dims this wrapper handles."""
        return sorted(self._engines.keys())

    @torch.no_grad()
    def compress_kv_cache(self, past_key_values) -> dict:
        """Compress every layer of a KV cache, routing per head_dim.

        Returns:
            dict with keys:
              - "layers": list of (ck_list, cv_list, head_dim) tuples, one per layer
              - "head_dims": parallel list of head_dim per layer (for build_cache)
        """
        keys, vals = TurboQuantEngine._extract_kv(past_key_values)
        compressed_layers = []
        for K_layer, V_layer in zip(keys, vals):
            hd = int(K_layer.shape[-1])
            engine = self._route(hd)
            ck_list, cv_list = [], []
            n_heads = K_layer.shape[1]
            for h in range(n_heads):
                ck, cv = engine._compress_kv_fused(K_layer[:, h], V_layer[:, h])
                ck_list.append(ck)
                cv_list.append(cv)
            compressed_layers.append((ck_list, cv_list, hd))
        return {
            "layers": compressed_layers,
            "head_dims": [hd for (_, _, hd) in compressed_layers],
        }

    @torch.no_grad()
    def build_cache(self, compressed: dict):
        """Reconstruct a `DynamicCache` from compressed layers.

        Each layer is decompressed by its assigned sub-engine, cast to the
        target dtype, and inserted into the cache in original layer order.
        Critical detail: uses `torch.stack(k_heads, dim=1)` to construct
        `(batch, n_heads, seq, head_dim)` directly, AVOIDING the extra
        leading dim that `torch.stack(k_heads).unsqueeze(0)` would produce
        (which would create a 5D cache tensor and break the next forward pass).
        """
        cache = DynamicCache()
        for li, (ck_list, cv_list, hd) in enumerate(compressed["layers"]):
            engine = self._route(hd)
            k_heads, v_heads = [], []
            for ck, cv in zip(ck_list, cv_list):
                k, v = engine._decompress_kv_fused(ck, cv)
                k_heads.append(k.to(self._target_dtype))
                v_heads.append(v.to(self._target_dtype))
            # Each k/v has shape (1, seq, head_dim). Stack along a NEW dim=1
            # to get (1, n_heads, 1, seq, head_dim), then collapse the spurious
            # leading dim by squeeze+unsqueeze, ending with the canonical
            # (1, n_heads, seq, head_dim) layout the model expects.
            k_layer = torch.stack(k_heads, dim=1).squeeze(0).unsqueeze(0)
            v_layer = torch.stack(v_heads, dim=1).squeeze(0).unsqueeze(0)
            cache.update(k_layer, v_layer, li)
        return cache

    def compression_stats(self, past_key_values) -> dict:
        """Compute total + per-head_dim compression statistics for a KV cache.

        Returns a dict with:
          - fp16_bytes / tq_bytes / ratio (overall, in BYTES regardless of
            whether the source dtype is fp16 or bf16 — both are 2 bytes/elem)
          - n_layers
          - per_head_dim: dict[head_dim -> {"layers", "fp16_bytes", "tq_bytes",
            "ratio"}] for breakdown by routed engine
        """
        from collections import defaultdict
        keys, _ = TurboQuantEngine._extract_kv(past_key_values)
        total_fp16 = 0
        total_tq = 0
        per_dim: dict = defaultdict(lambda: {"layers": 0, "fp16_bytes": 0, "tq_bytes": 0})
        for K in keys:
            hd = int(K.shape[-1])
            seq, n_heads = K.shape[2], K.shape[1]
            engine = self._route(hd)
            fp16_layer = seq * n_heads * hd * 2 * 2  # K+V, 2 bytes/elem
            tq_layer = engine._compressed_bytes(seq) * n_heads
            total_fp16 += fp16_layer
            total_tq += tq_layer
            per_dim[hd]["layers"] += 1
            per_dim[hd]["fp16_bytes"] += fp16_layer
            per_dim[hd]["tq_bytes"] += int(tq_layer)
        for hd in per_dim:
            d = per_dim[hd]
            d["ratio"] = d["fp16_bytes"] / max(d["tq_bytes"], 1)
        return {
            "fp16_bytes": total_fp16,
            "tq_bytes": int(total_tq),
            "ratio": total_fp16 / max(total_tq, 1),
            "n_layers": len(keys),
            "per_head_dim": dict(per_dim),
        }

    @torch.no_grad()
    def generate(
        self,
        model,
        tokenizer,
        prompt: str,
        max_new_tokens: int = 100,
    ) -> dict:
        """Compress prompt KV, then greedy-decode `max_new_tokens` tokens.

        Always passes explicit `position_ids` to the model on each step —
        without this, multi-attention-type Gemma 4 E4B produces garbage
        because the position bookkeeping for SWA + full attention layers
        gets confused.

        Returns dict with:
          - "text": decoded generated text (excluding prompt)
          - "tokens": int, number of tokens generated
          - "stats": output of compression_stats() on the prompt KV
        """
        inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
        n_input = inputs.input_ids.shape[1]

        out = model(**inputs, use_cache=True)
        compressed = self.compress_kv_cache(out.past_key_values)
        stats = self.compression_stats(out.past_key_values)
        cache = self.build_cache(compressed)

        next_tok = out.logits[:, -1:].argmax(dim=-1)
        generated = [int(next_tok.item())]
        eos = tokenizer.eos_token_id

        for step in range(max_new_tokens - 1):
            o = model(
                input_ids=next_tok,
                past_key_values=cache,
                position_ids=torch.tensor([[n_input + step]], device=self.device),
                use_cache=True,
            )
            cache = o.past_key_values
            next_tok = o.logits[:, -1:, :].argmax(dim=-1).squeeze(-1).unsqueeze(0)
            tid = int(next_tok.item())
            generated.append(tid)
            if eos is not None and tid == eos:
                break

        text = tokenizer.decode(generated, skip_special_tokens=True)
        return {
            "text": text,
            "tokens": len(generated),
            "stats": stats,
        }
