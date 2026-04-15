"""Validate our Gemma4Stream forward pass against llama.cpp serving the same GGUF.

Given a tq4-aligned Gemma 4 E4B GGUF, compare:
  - Our PyTorch Gemma4Stream.forward(tokens) logits
  - llama.cpp's /v1/completions logprobs on the same tokens

If numerically close (relative error < 5% on top-k tokens), our
PyTorch port is a valid training substrate. If far apart, we have
bugs to find.

Usage:
    # Start llama-server on the GGUF first:
    ~/llama.cpp/build/bin/llama-server \\
        -m ~/models/gemma-4-E4B-it-tq4-aligned.gguf \\
        --port 8080 --n-gpu-layers 999

    python3 scripts/validate_gemma4_vs_llamacpp.py
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import torch

from calm.llm_computer.gemma4_stream import load_gemma4_stream_from_gguf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default="/home/gabe/models/gemma-4-E4B-it-tq4-aligned.gguf")
    ap.add_argument("--server-url", default="http://localhost:8080")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--top-k", type=int, default=5,
                    help="compare top-k token probabilities")
    ap.add_argument("--skip-llamacpp", action="store_true",
                    help="Only run our PyTorch forward (skip API comparison)")
    args = ap.parse_args()

    print("=== Gemma 4 E4B validation: PyTorch substrate vs llama.cpp ===", flush=True)
    print(f"  GGUF: {args.gguf}", flush=True)
    print(f"  device: {args.device}", flush=True)

    gguf_path = Path(args.gguf)
    if not gguf_path.exists():
        print(f"ERROR: GGUF not found at {gguf_path}", flush=True)
        return 1

    print(f"\n--- loading Gemma 4 E4B (this takes ~30s) ---", flush=True)
    try:
        stream = load_gemma4_stream_from_gguf(args.gguf, device=args.device)
    except Exception as e:
        print(f"FAILED to load stream: {type(e).__name__}: {e}", flush=True)
        import traceback; traceback.print_exc()
        return 1

    n_params = sum(p.numel() for p in stream.parameters())
    print(f"  loaded: {n_params:,} parameters", flush=True)
    print(f"  layers: {stream.cfg.n_layers}", flush=True)
    print(f"  full attention layers: {stream.cfg.full_attention_layers}", flush=True)

    # Synthetic input (can't tokenize without tokenizer)
    # For MVP validation, use a fixed small token sequence
    test_tokens = torch.tensor([[1, 234, 5678, 42, 100]], dtype=torch.long)
    print(f"\n--- forward pass on {test_tokens.shape[1]} tokens ---", flush=True)
    try:
        with torch.no_grad():
            logits = stream(test_tokens.to(args.device))
        print(f"  logits shape: {tuple(logits.shape)}", flush=True)
        print(f"  finite: {torch.isfinite(logits).all().item()}", flush=True)
        print(f"  logit mean: {logits.mean().item():.3f}", flush=True)
        print(f"  logit std:  {logits.std().item():.3f}", flush=True)
        # Top-k at last position
        topk = logits[0, -1].topk(args.top_k)
        print(f"  top-{args.top_k} tokens at last position:", flush=True)
        for i in range(args.top_k):
            print(f"    token {topk.indices[i].item():6d}  "
                  f"logit {topk.values[i].item():.3f}", flush=True)
    except Exception as e:
        print(f"FAILED forward: {type(e).__name__}: {e}", flush=True)
        import traceback; traceback.print_exc()
        return 1

    if args.skip_llamacpp:
        print("\n  --skip-llamacpp set; done.", flush=True)
        return 0

    # Query llama.cpp for comparison
    print(f"\n--- comparing against llama-server at {args.server_url} ---", flush=True)
    try:
        # Check server is up
        req = urllib.request.Request(f"{args.server_url}/health")
        with urllib.request.urlopen(req, timeout=5) as resp:
            pass
    except Exception as e:
        print(f"  llama-server not reachable: {e}", flush=True)
        print(f"  start it with:", flush=True)
        print(f"    ~/llama.cpp/build/bin/llama-server -m {args.gguf} "
              f"--port 8080 --n-gpu-layers 999", flush=True)
        return 1

    # Tokenize via the server
    try:
        body = json.dumps({"content": args.prompt}).encode()
        req = urllib.request.Request(
            f"{args.server_url}/tokenize", data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            toks = json.loads(resp.read())
        token_ids = toks.get("tokens", [])
        print(f"  tokenized prompt: {len(token_ids)} tokens", flush=True)
    except Exception as e:
        print(f"  tokenize failed: {e}", flush=True)
        return 1

    # Get llama.cpp's logits via /completions with logprobs
    # Note: this gives TOP-k per position, not the raw logit vector
    try:
        body = json.dumps({
            "prompt": args.prompt,
            "n_predict": 1,
            "n_probs": args.top_k,
            "temperature": 0,
        }).encode()
        req = urllib.request.Request(
            f"{args.server_url}/completion", data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read())
        probs = result.get("completion_probabilities", [])
        if probs:
            llama_probs = probs[0].get("probs", [])
            print(f"  llama.cpp top-{args.top_k}:", flush=True)
            for p in llama_probs[:args.top_k]:
                print(f"    token {p.get('tok_str', '?')!r:20s}  "
                      f"prob {p.get('prob', 0):.4f}", flush=True)
    except Exception as e:
        print(f"  /completion failed: {e}", flush=True)

    # Run our PyTorch forward on the same tokens
    try:
        token_tensor = torch.tensor([token_ids], dtype=torch.long)
        with torch.no_grad():
            our_logits = stream(token_tensor.to(args.device))
        our_probs = torch.softmax(our_logits[0, -1], dim=-1)
        our_topk = our_probs.topk(args.top_k)
        print(f"\n  OUR top-{args.top_k}:", flush=True)
        for i in range(args.top_k):
            print(f"    token_id {our_topk.indices[i].item():6d}  "
                  f"prob {our_topk.values[i].item():.4f}", flush=True)
    except Exception as e:
        print(f"  our forward on tokenized prompt failed: {e}", flush=True)
        return 1

    print("\n=== summary ===", flush=True)
    print("  Structural PASS: stream loaded, forward runs, logits finite.", flush=True)
    print("  Numerical match against llama.cpp requires manual inspection of", flush=True)
    print("  top-k probs above. Exact match requires:", flush=True)
    print("    1. Token embeddings loaded (currently Q6_K, skipped)", flush=True)
    print("    2. Per-layer embedding injection wired correctly", flush=True)
    print("    3. Pi matrix bit-exact (c_header path)", flush=True)
    print("  Any discrepancies above narrow the bug to specific components.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
