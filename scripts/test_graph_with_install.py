"""CUDA Graph × FP32 layers × in-attention install compatibility.

H1: After convert_layer_to_fp32 + install_card_in_attention, a fresh
generate_with_graph call captures the new FP32 tensors and replays
correctly.

H2: The per-sub-head attention dispatch (attention_partition path)
does not break CUDA Graph capture — static Python control flow only.

H3: Graph speedup still beats non-graph decode with install present.

Measurement:
  - Correctness: token-by-token equality between graph and non-graph
    decode on the same prompt, with install active in both.
  - Perf: decode tok/s with graph vs without.
"""

import argparse
import os
import sys
import time

import torch

GGUF_PATH = os.path.expanduser("~/models/gemma-4-E4B-it-tq4-aligned.gguf")


def load_substrate():
    from calm.llm_computer.gemma_substrate import (
        GemmaSubstrate, enable_triton_tq4,
    )
    enable_triton_tq4(True)
    m = GemmaSubstrate.from_gguf(GGUF_PATH, max_len=512)
    m.preload_gpu("cuda")
    m.warmup(seq_lens=(1, 6))
    return m


def install_card(m):
    """Install add_one at layer 41 with hard_max partition — exercises
    both FP32 conversion and per-sub-head attention dispatch."""
    from calm.llm_computer.programs.add_one import build_add_one
    m.convert_layer_to_fp32(41)
    card = build_add_one()
    info = m.install_card_in_attention(
        card, layer_idx=41, sub_head_offset=0,
        ch_off=2400, d_card=8, mode="hard_max",
    )
    print(f"[install] add_one at layer 41: {info}")
    return card


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--tokens", type=int, default=30)
    ap.add_argument("--no-install", action="store_true",
                    help="baseline: skip install (should be 42 tok/s target)")
    args = ap.parse_args()

    m = load_substrate()
    if not args.no_install:
        install_card(m)

    from calm.llm_computer.synth.gemma_tokenizer import GemmaTokenizer
    tok = GemmaTokenizer.from_gguf(GGUF_PATH)

    # Path A: generate_with_graph (CUDA graph + static KV).
    print("\n[A] generate_with_graph (CUDA graph path)...")
    t0 = time.time()
    out_g = m.generate_with_graph(args.prompt, tok, max_tokens=args.tokens,
                                   max_len=256)
    t_g = time.time() - t0
    g_tok_s = (args.tokens - 1) / out_g["decode_s"]
    print(f"  tokens: {out_g['token_ids']}")
    print(f"  text:   {out_g['text']!r}")
    print(f"  prefill: {out_g['prefill_s']:.3f}s")
    print(f"  decode:  {out_g['decode_s']:.3f}s ({g_tok_s:.1f} tok/s)")

    # Path B: generate (no graph, dynamic KV). Same prompt and greedy
    # argmax — tokens must match exactly.
    print("\n[B] generate (non-graph baseline)...")
    t0 = time.time()
    out_n = m.generate(args.prompt, tok, max_tokens=args.tokens)
    t_n = time.time() - t0
    n_tok_s = (args.tokens - 1) / out_n["decode_s"]
    print(f"  tokens: {out_n['token_ids']}")
    print(f"  text:   {out_n['text']!r}")
    print(f"  prefill: {out_n['prefill_s']:.3f}s")
    print(f"  decode:  {out_n['decode_s']:.3f}s ({n_tok_s:.1f} tok/s)")

    # Compare
    print("\n[compare]")
    ids_g = out_g["token_ids"]
    ids_n = out_n["token_ids"]
    n_compare = min(len(ids_g), len(ids_n))
    match_n = sum(1 for i in range(n_compare) if ids_g[i] == ids_n[i])
    first_div = next(
        (i for i in range(n_compare) if ids_g[i] != ids_n[i]), None)
    print(f"  match:        {match_n}/{n_compare}")
    print(f"  first diverge: {first_div}")
    print(f"  speedup:      {n_tok_s:.1f} → {g_tok_s:.1f} tok/s "
          f"({g_tok_s/n_tok_s:.2f}x)")

    ok = (match_n == n_compare and g_tok_s >= n_tok_s * 0.9)
    # Allow 10% slack on graph perf (warmup variance on first call)
    print(f"\n  verdict: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
