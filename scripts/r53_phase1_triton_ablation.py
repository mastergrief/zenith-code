"""Triton-on vs triton-off parity diagnostic.

Runs the same 16-step Gemma decode with tq4 KV cache TWICE:
  1. _use_triton=True  — Phase 2 fused flash-attn kernel fires for SWA
     layers that pass the dispatch gate (d_head==256, S==1, no partitions)
  2. _use_triton=False — dispatch skips fused, runs tq4 through the
     existing dequant+einsum path

Both compared against fp16 KV baseline. If the fused path introduces
larger divergence than the dequant-path tq4, the fused kernel has a
bug. If both paths show ~0.97-0.99 min cosine, the divergence is just
tq4 quantization noise (4-bit Lloyd-Max) and Phase 2 is correct.

Usage:
  bin/gemma-run scripts/r53_phase1_triton_ablation.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _run_parity_with_triton(triton_on: bool, m, tok, prompt: str, n_decode: int):
    """Run lockstep fp16 vs tq4 decode and return (min_cos, mean_cos,
    argmax_matches) for this triton setting."""
    from calm.llm_computer import gemma_substrate as gs
    from calm.llm_computer.gemma_substrate import KVCache, KVCacheTq4

    # Toggle the module-level flag (what `_use_triton` checks at the
    # dispatch site in _forward_layer).
    gs._use_triton = triton_on

    ids = tok.encode(prompt)
    cache_fp16 = KVCache(m.config.n_layers, device="cuda")
    cache_tq4 = KVCacheTq4(m, max_len=len(ids) + n_decode + 8, device="cuda")

    with torch.no_grad():
        l_fp16 = m.forward(torch.tensor([ids]), device="cuda",
                           kv_cache=cache_fp16, start_pos=0)
        l_tq4 = m.forward(torch.tensor([ids]), device="cuda",
                          kv_cache=cache_tq4, start_pos=0)

    cosines = []
    argmax_matches = 0
    cur_id = int(l_fp16[0, -1].argmax().item())

    for step in range(n_decode):
        with torch.no_grad():
            r_fp16 = m.forward(torch.tensor([[cur_id]]), device="cuda",
                               kv_cache=cache_fp16,
                               start_pos=len(ids) + step)
            r_tq4 = m.forward(torch.tensor([[cur_id]]), device="cuda",
                              kv_cache=cache_tq4,
                              start_pos=len(ids) + step)
        v_fp16 = r_fp16[0, -1].float().flatten()
        v_tq4 = r_tq4[0, -1].float().flatten()
        cos = F.cosine_similarity(v_fp16, v_tq4, dim=0).item()
        cosines.append(cos)
        if v_fp16.argmax().item() == v_tq4.argmax().item():
            argmax_matches += 1
        cur_id = int(v_fp16.argmax().item())

    return min(cosines), sum(cosines) / len(cosines), argmax_matches, cosines


def _load_gemma_standalone():
    """Fallback loader when not running under the gemma daemon."""
    import os
    from calm.llm_computer.gemma_substrate import GemmaSubstrate
    from calm.llm_computer.synth.gemma_tokenizer import GemmaTokenizer
    gguf = os.path.expanduser("~/models/gemma-4-E4B-it-tq4-aligned.gguf")
    print("[ablation] loading Gemma (standalone — daemon not detected)...")
    model = GemmaSubstrate.from_gguf(gguf, max_len=128)
    model.preload_gpu("cuda")
    tokenizer = GemmaTokenizer.from_gguf(gguf)
    return model, tokenizer


def main():
    prompt = "The capital of France is"
    n_decode = 16

    # Use daemon globals if present, else load standalone.
    try:
        model = m           # noqa: F821 — daemon-injected
        tokenizer = tok     # noqa: F821
    except NameError:
        model, tokenizer = _load_gemma_standalone()

    print(f"[ablation] prompt={prompt!r}, n_decode={n_decode}")

    # triton OFF baseline first — isolates quantization-only effect
    print("\n[ablation] run 1: triton=OFF (dequant+einsum path on tq4)")
    off_min, off_mean, off_argmax, off_cosines = _run_parity_with_triton(
        False, model, tokenizer, prompt, n_decode)
    print(f"  min cosine: {off_min:.4f}  mean: {off_mean:.4f}  "
          f"argmax: {off_argmax}/{n_decode}")
    print(f"  per-step: {[f'{c:.4f}' for c in off_cosines]}")

    # triton ON — fused path fires for d_head==256 SWA layers
    print("\n[ablation] run 2: triton=ON (fused tq4 flash-attn)")
    on_min, on_mean, on_argmax, on_cosines = _run_parity_with_triton(
        True, model, tokenizer, prompt, n_decode)
    print(f"  min cosine: {on_min:.4f}  mean: {on_mean:.4f}  "
          f"argmax: {on_argmax}/{n_decode}")
    print(f"  per-step: {[f'{c:.4f}' for c in on_cosines]}")

    # Delta: does fused path introduce MORE divergence than dequant path?
    print(f"\n[ablation] Δ(triton_on - triton_off):")
    print(f"  min cosine: {on_min - off_min:+.4f}")
    print(f"  mean cosine: {on_mean - off_mean:+.4f}")
    print(f"  argmax matches: {on_argmax - off_argmax:+d}")

    if on_mean >= off_mean - 0.005 and on_argmax >= off_argmax - 1:
        print("\n[ablation] VERDICT: fused kernel neutral or better — correct.")
    else:
        print(f"\n[ablation] VERDICT: fused kernel introduces extra "
              f"divergence. Check the kernel against test_tq4_flash_attn.py.")


main()
