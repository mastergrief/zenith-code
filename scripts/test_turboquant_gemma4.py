#!/usr/bin/env python3
"""Validate per-layer routed TurboQuant on Gemma 4 E4B.

Loads Gemma 4 E4B with bitsandbytes 4-bit weights, runs the
`MultiHeadDimTurboQuantEngine` wrapper from `scripts.turboquant_patches`,
and verifies that compressed-cache decoding produces output token-for-token
identical to the uncompressed baseline. Reports compression ratio per layer
type (sliding-window vs full attention).

Why this script exists:

  Gemma 4 E4B has TWO different attention head dimensions across layers:
  20 sliding-window layers with head_dim=256, and 4 full-attention layers
  with head_dim=512. The published `turboquant_gpu` package assumes a
  single uniform head_dim and fails on Gemma 4 with a shape mismatch error.
  The `MultiHeadDimTurboQuantEngine` wrapper holds one TurboQuantEngine per
  unique head_dim and routes per-layer based on the actual K tensor shape.

  This was validated lossless on 2026-04-08 — see CLAUDE.md / session
  history for the original investigation. This script makes the result
  reproducible.

Prerequisites:

  - HF_TOKEN (or HUGGING_FACE_READ_TOKEN) in .env.local with read access
  - Gemma 4 E4B accessible at google/gemma-4-E4B-it (no longer gated as of
    2026-04-08, but the token is still needed for higher rate limits)
  - llama-server NOT running (frees ~6 GB VRAM for the test)
  - turboquant-gpu installed (`pip install turboquant-gpu`)
  - bitsandbytes installed (already in the project deps via Unsloth)

Usage from repo root:

    # Stop llama-server if running:
    #   pkill -f "llama-server"
    # Then:
    PYTHONPATH=. python3 scripts/test_turboquant_gemma4.py

  Optional flags:
    --bits 4          # use 4-bit instead of 3-bit (default 3)
    --new-tokens 20   # how many tokens to generate after the prompt (default 20)
    --prompt "..."    # custom prompt (default: short repetitive AI text)

After the test, restart llama-server with:

    bin/zenith   # or your normal launch command

Exit code:
  0 if compressed output matches baseline byte-for-byte
  1 if any tokens diverge (compression introduced quality loss)
  2 if the test setup itself fails (download, OOM, etc.)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def load_env_local() -> None:
    """Load .env.local into os.environ. Maps HUGGING_FACE_READ_TOKEN -> HF_TOKEN."""
    env_path = REPO_ROOT / ".env.local"
    if not env_path.exists():
        print(f"warning: {env_path} not found — assuming HF_TOKEN is set elsewhere")
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip()
    if "HUGGING_FACE_READ_TOKEN" in os.environ and "HF_TOKEN" not in os.environ:
        os.environ["HF_TOKEN"] = os.environ["HUGGING_FACE_READ_TOKEN"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--bits", type=int, default=3, choices=[2, 3, 4],
                        help="TurboQuant bit width (default: 3)")
    parser.add_argument("--new-tokens", type=int, default=20,
                        help="Tokens to generate after the prompt (default: 20)")
    parser.add_argument(
        "--prompt",
        default=(
            "The history of artificial intelligence research is a fascinating "
            "journey through decades of breakthroughs. " * 10
        ),
        help="Test prompt (default: short repetitive AI history text)",
    )
    parser.add_argument("--model", default="google/gemma-4-E4B-it",
                        help="HF model id (default: google/gemma-4-E4B-it)")
    args = parser.parse_args()

    load_env_local()

    # Defer heavy imports until after env setup
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    # Apply patches and import the wrapper
    import scripts.turboquant_patches  # noqa: F401  applies compat patches
    from scripts.turboquant_patches import MultiHeadDimTurboQuantEngine

    print("=" * 70)
    print("TurboQuant per-layer routing validation on Gemma 4 E4B")
    print("=" * 70)
    print(f"model:        {args.model}")
    print(f"bits:         {args.bits}")
    print(f"new tokens:   {args.new_tokens}")
    print(f"prompt chars: {len(args.prompt)}")

    # Load model with bitsandbytes 4-bit weights (necessary for the 8 GB 4070)
    print(f"\nloading {args.model} (bitsandbytes 4-bit weights, bf16 compute)...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    t0 = time.time()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=bnb, device_map="cuda"
        )
        tok = AutoTokenizer.from_pretrained(args.model)
    except Exception as e:
        print(f"ERROR loading model: {type(e).__name__}: {e}")
        return 2
    print(f"loaded in {time.time()-t0:.1f}s")

    cfg = model.config
    tcfg = getattr(cfg, "text_config", cfg)
    print(f"\nmodel class: {type(model).__name__}")
    print(f"text_config.num_hidden_layers: {getattr(tcfg, 'num_hidden_layers', '?')}")
    print(f"text_config.head_dim:          {getattr(tcfg, 'head_dim', '?')} (per text_config)")
    print(f"text_config.num_kv_heads:      {getattr(tcfg, 'num_key_value_heads', '?')}")

    torch.cuda.synchronize()
    print(f"VRAM after load: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Tokenize prompt
    inputs = tok(args.prompt, return_tensors="pt").to("cuda")
    n_input = int(inputs.input_ids.shape[1])
    print(f"\ntokenized prompt: {n_input} tokens")

    # ── Auto-construct the engine via from_model() ──────────────────────
    print(f"\n=== constructing MultiHeadDimTurboQuantEngine.from_model() ===")
    engine = MultiHeadDimTurboQuantEngine.from_model(
        model, tok, total_bits=args.bits, device="cuda"
    )
    print(f"discovered head_dims: {engine.head_dims}")
    print(f"target dtype:         {engine._target_dtype}")

    # ── Capture FIRST past_key_values (will be compressed) ──────────────
    print(f"\n=== capturing past_key_values for compression ===")
    with torch.no_grad():
        out_for_compression = model(**inputs, use_cache=True)
    pkv_for_compression = out_for_compression.past_key_values

    # Per-layer breakdown for context — snapshot SHAPES not refs (defensive)
    from turboquant_gpu import TurboQuantEngine
    keys_compress, _vals_compress = TurboQuantEngine._extract_kv(pkv_for_compression)
    snapshot_shapes = [tuple(K.shape) for K in keys_compress]  # immutable snapshot
    from collections import defaultdict
    layers_by_dim: dict[int, list[int]] = defaultdict(list)
    for i, K in enumerate(keys_compress):
        layers_by_dim[int(K.shape[-1])].append(i)
    for hd in sorted(layers_by_dim):
        idxs = layers_by_dim[hd]
        print(f"  head_dim={hd}: {len(idxs)} layers, indices={idxs}")

    # ── Compression stats + compress + build_cache (BEFORE any decode) ──
    print(f"\n=== compression stats ({args.bits}-bit) ===")
    stats = engine.compression_stats(pkv_for_compression)
    print(f"  total fp16 bytes: {stats['fp16_bytes']:,}")
    print(f"  total tq bytes:   {stats['tq_bytes']:,}")
    print(f"  overall ratio:    {stats['ratio']:.2f}x")
    for hd in sorted(stats["per_head_dim"]):
        d = stats["per_head_dim"][hd]
        print(f"  head_dim={hd}: {d['layers']:>2} layers, "
              f"{d['fp16_bytes']:>12,} → {d['tq_bytes']:>11,} bytes "
              f"({d['ratio']:.2f}x)")

    # IMPORTANT: compress + build_cache MUST run before any decode loop,
    # because DynamicCache decoding mutates the underlying tensors in place.
    # If we compressed AFTER decoding, we'd be operating on the extended
    # cache state and the verification would compare different things.
    print(f"\n=== compress + rebuild (before any decoding) ===")
    t0 = time.time()
    compressed = engine.compress_kv_cache(pkv_for_compression)
    compress_time = time.time() - t0
    print(f"  compress time:    {compress_time:.3f}s")

    t0 = time.time()
    new_cache = engine.build_cache(compressed)
    rebuild_time = time.time() - t0
    print(f"  rebuild time:     {rebuild_time:.3f}s")

    # Verify rebuilt shapes match the snapshotted (immutable) shapes
    mismatch_count = 0
    for li in range(len(new_cache.layers)):
        orig = snapshot_shapes[li]
        new = tuple(new_cache.layers[li].keys.shape)
        if orig != new:
            mismatch_count += 1
            if mismatch_count <= 3:
                print(f"  ⚠️  layer[{li}] shape mismatch: orig={orig} new={new}")
    if mismatch_count == 0:
        print(f"  ✅ all {len(new_cache.layers)} layer shapes match originals")
    else:
        print(f"  ❌ {mismatch_count}/{len(new_cache.layers)} layers have shape mismatches")

    # ── Run BASELINE decode using a SECOND, fresh forward pass ──────────
    # (Cannot reuse pkv_for_compression because the next decode loop would
    # mutate it, AND we need an untouched 151-token state to start from.)
    print(f"\n=== BASELINE: greedy decode {args.new_tokens} tokens (second forward pass, uncompressed cache) ===")
    with torch.no_grad():
        out_baseline = model(**inputs, use_cache=True)
    pkv_baseline = out_baseline.past_key_values

    next_tok = out_baseline.logits[:, -1:].argmax(dim=-1)
    cache = pkv_baseline
    baseline_ids: list[int] = []
    t0 = time.time()
    for step in range(args.new_tokens):
        o = model(
            input_ids=next_tok,
            past_key_values=cache,
            position_ids=torch.tensor([[n_input + step]], device="cuda"),
            use_cache=True,
        )
        cache = o.past_key_values
        next_tok = o.logits[:, -1:, :].argmax(dim=-1).squeeze(-1).unsqueeze(0)
        baseline_ids.append(int(next_tok.item()))
    baseline_decode_time = time.time() - t0
    baseline_text = tok.decode(baseline_ids, skip_special_tokens=True)
    print(f"  decode time:      {baseline_decode_time:.3f}s ({args.new_tokens/baseline_decode_time:.1f} tok/s)")
    print(f"  ids:  {baseline_ids}")
    print(f"  text: {baseline_text!r}")

    # ── Run COMPRESSED decode from the rebuilt cache ────────────────────
    print(f"\n=== COMPRESSED: greedy decode {args.new_tokens} tokens via per-layer routing ===")
    next_tok = out_for_compression.logits[:, -1:].argmax(dim=-1)
    cache = new_cache
    compressed_ids: list[int] = []
    t0 = time.time()
    for step in range(args.new_tokens):
        o = model(
            input_ids=next_tok,
            past_key_values=cache,
            position_ids=torch.tensor([[n_input + step]], device="cuda"),
            use_cache=True,
        )
        cache = o.past_key_values
        next_tok = o.logits[:, -1:, :].argmax(dim=-1).squeeze(-1).unsqueeze(0)
        compressed_ids.append(int(next_tok.item()))
    decode_time = time.time() - t0
    compressed_text = tok.decode(compressed_ids, skip_special_tokens=True)
    print(f"  decode time:      {decode_time:.3f}s ({args.new_tokens/decode_time:.1f} tok/s)")
    print(f"  ids:  {compressed_ids}")
    print(f"  text: {compressed_text!r}")

    # ── Verify lossless ─────────────────────────────────────────────────
    print(f"\n=== VERDICT ===")
    if baseline_ids == compressed_ids:
        print(f"  ✅ LOSSLESS: all {args.new_tokens} tokens identical to baseline")
        print(f"  → TurboQuant {args.bits}-bit per-layer routing on Gemma 4 E4B is byte-perfect")
        verdict = 0
    else:
        # Find first divergence
        first_diff = next(
            (i for i in range(min(len(baseline_ids), len(compressed_ids)))
             if baseline_ids[i] != compressed_ids[i]),
            -1,
        )
        diff_count = sum(1 for a, b in zip(baseline_ids, compressed_ids) if a != b)
        print(f"  ❌ DIVERGENCE at token {first_diff}")
        print(f"  baseline[{first_diff}] = {baseline_ids[first_diff]} ({tok.decode([baseline_ids[first_diff]])!r})")
        print(f"  comp.[{first_diff}]    = {compressed_ids[first_diff]} ({tok.decode([compressed_ids[first_diff]])!r})")
        print(f"  total diverging tokens: {diff_count}/{args.new_tokens}")
        verdict = 1

    torch.cuda.synchronize()
    print(f"\nVRAM peak: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    return verdict


if __name__ == "__main__":
    sys.exit(main())
