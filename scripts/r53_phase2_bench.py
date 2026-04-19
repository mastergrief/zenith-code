"""Decode-throughput benchmark: fp16 KV vs tq4 KV dequant-path vs tq4 KV fused.

Apples-to-apples perf isolation: uses `enable_fused_flash_attn(False)` to
disable the Phase 2 kernel WITHOUT disabling the unrelated Triton weight
kernels (ffn_gate/ffn_up/ffn_down/attn_q/k/v/output — all still fused).
The prior bench conflated "no fused flash-attn" with "no weight Triton"
and produced a 1.25 tok/s straw-man.

Runs each path at N ∈ {64, 256, 1024, 2048} to show the crossover where
fused flash-attn's streaming O(1) byte-load replaces the dequant path's
O(N) full-prefix dequant per step.
"""

from __future__ import annotations

import os
import time

import torch


def _load_gemma(max_len: int):
    from calm.llm_computer.gemma_substrate import GemmaSubstrate
    from calm.llm_computer.synth.gemma_tokenizer import GemmaTokenizer
    gguf = os.path.expanduser("~/models/gemma-4-E4B-it-tq4-aligned.gguf")
    print(f"[bench] loading Gemma (max_len={max_len})...")
    m = GemmaSubstrate.from_gguf(gguf, max_len=max_len)
    m.preload_gpu("cuda")
    tok = GemmaTokenizer.from_gguf(gguf)
    return m, tok


def _bench_path(name, m, tok, cache_factory, fused_on, prompt, n_decode):
    """Run prefill + n_decode decode steps with the given cache.
    Triton weight kernels stay ON; `fused_on` only flips the Phase 2
    KV-side flash-attn dispatch.
    Returns (decode_s, tok_per_s)."""
    from calm.llm_computer.gemma_substrate import enable_fused_flash_attn
    enable_fused_flash_attn(fused_on)

    ids = tok.encode(prompt)
    cache = cache_factory(m, len(ids) + n_decode + 8)

    with torch.no_grad():
        logits = m.forward(torch.tensor([ids]), device="cuda",
                           kv_cache=cache, start_pos=0)
    torch.cuda.synchronize()
    cur_id = int(logits[0, -1].argmax().item())

    torch.cuda.synchronize()
    t0 = time.time()
    for step in range(n_decode):
        with torch.no_grad():
            logits = m.forward(torch.tensor([[cur_id]]), device="cuda",
                               kv_cache=cache,
                               start_pos=len(ids) + step)
        cur_id = int(logits[0, -1].argmax().item())
    torch.cuda.synchronize()
    decode_s = time.time() - t0
    tps = n_decode / decode_s
    print(f"  {name:>36s}: {decode_s:7.2f}s / {n_decode:4d} tok = {tps:6.2f} tok/s")
    return decode_s, tps


def main():
    from calm.llm_computer.gemma_substrate import (
        KVCache, KVCacheTq4, enable_triton_tq4,
    )

    enable_triton_tq4(True)  # weight kernels ON throughout

    configs = [
        (64,   "n_decode=64 — baseline short"),
        (256,  "n_decode=256 — medium"),
        (1024, "n_decode=1024 — long"),
        (4096, "n_decode=4096 — very long (KV-bandwidth regime)"),
    ]

    # Load Gemma once with max_len sized for longest run
    max_decode = max(c[0] for c in configs)
    m, tok = _load_gemma(max_len=max_decode + 128)

    # Warm-up for both paths (Triton JIT)
    print("\n[bench] warm-up...")
    for fused in (True, False):
        from calm.llm_computer.gemma_substrate import enable_fused_flash_attn
        enable_fused_flash_attn(fused)
        cache = KVCacheTq4(m, max_len=32, device="cuda")
        ids = tok.encode("Hi")
        with torch.no_grad():
            l = m.forward(torch.tensor([ids]), device="cuda",
                          kv_cache=cache, start_pos=0)
            cur_id = int(l[0, -1].argmax())
            m.forward(torch.tensor([[cur_id]]), device="cuda",
                      kv_cache=cache, start_pos=len(ids))
        torch.cuda.synchronize()

    prompt = "The quick brown fox jumps over the lazy dog. " * 4  # ~40 tokens

    results = []
    for n_decode, label in configs:
        print(f"\n[bench] {label}")
        _, fp16_tps = _bench_path(
            "fp16 KV (baseline)", m, tok,
            lambda model, n: KVCache(model.config.n_layers, device="cuda"),
            fused_on=True, prompt=prompt, n_decode=n_decode)
        _, dq_tps = _bench_path(
            "tq4 KV — Phase 1 memo only", m, tok,
            lambda model, n: KVCacheTq4(model, max_len=n, device="cuda"),
            fused_on=False, prompt=prompt, n_decode=n_decode)
        _, fused_tps = _bench_path(
            "tq4 KV — Phase 2 fused flash-attn", m, tok,
            lambda model, n: KVCacheTq4(model, max_len=n, device="cuda"),
            fused_on=True, prompt=prompt, n_decode=n_decode)

        speedup = fused_tps / dq_tps if dq_tps > 0 else 0.0
        fp16_ratio = fused_tps / fp16_tps if fp16_tps > 0 else 0.0
        results.append((n_decode, fp16_tps, dq_tps, fused_tps, speedup, fp16_ratio))
        print(f"  Phase-2 speedup vs Phase-1-only: {speedup:.2f}×")
        print(f"  Fused tq4 / fp16: {fp16_ratio*100:.1f}%")

    print("\n[bench] SUMMARY")
    print(f"{'N':>6}  {'fp16':>8}  {'tq4 memo':>10}  {'tq4 fused':>10}  {'speedup':>8}  {'% fp16':>7}")
    for n_decode, fp16_tps, dq_tps, fused_tps, speedup, fp16_ratio in results:
        print(f"{n_decode:>6}  {fp16_tps:>7.2f}  {dq_tps:>9.2f}  {fused_tps:>9.2f}"
              f"  {speedup:>7.2f}×  {fp16_ratio*100:>6.1f}%")


main()
