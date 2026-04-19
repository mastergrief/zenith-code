"""Phase 1 parity validator — runs through bin/gemma-run on the loaded
Gemma daemon. Compares fp16 KVCache vs KVCacheTq4 decode logits in
lockstep and reports cosine + argmax preservation per step.

Asserts the same gates as calm/llm_computer/tests/test_kvcache_tq4_parity.py
(which is pytest-runnable when GPU is free; this script is for the
daemon-loaded path).

Usage:
  bin/gemma-run scripts/r53_phase1_parity.py
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F


def main():  # noqa: D401 — daemon convention
    from calm.llm_computer.gemma_substrate import KVCache, KVCacheTq4

    # m, tok injected by the daemon.
    n_decode = 16
    prompt = "The capital of France is"
    ids = tok.encode(prompt)  # noqa: F821 — daemon-bound

    print(f"[r53.phase1] prompt={prompt!r}, prompt_len={len(ids)}, "
          f"n_decode={n_decode}")

    cache_fp16 = KVCache(m.config.n_layers, device="cuda")  # noqa: F821
    cache_tq4 = KVCacheTq4(m, max_len=len(ids) + n_decode + 8,  # noqa: F821
                           device="cuda")

    t0 = time.time()
    with torch.no_grad():
        l_fp16 = m.forward(torch.tensor([ids]), device="cuda",  # noqa: F821
                           kv_cache=cache_fp16, start_pos=0)
        l_tq4 = m.forward(torch.tensor([ids]), device="cuda",  # noqa: F821
                          kv_cache=cache_tq4, start_pos=0)
    print(f"[r53.phase1] prefill done in {time.time() - t0:.1f}s")

    cosines = []
    argmax_matches = 0
    cur_id = int(l_fp16[0, -1].argmax().item())
    fp16_decode_s = 0.0
    tq4_decode_s = 0.0

    for step in range(n_decode):
        t = time.time()
        with torch.no_grad():
            r_fp16 = m.forward(torch.tensor([[cur_id]]), device="cuda",  # noqa: F821
                               kv_cache=cache_fp16,
                               start_pos=len(ids) + step)
        torch.cuda.synchronize()
        fp16_decode_s += time.time() - t

        t = time.time()
        with torch.no_grad():
            r_tq4 = m.forward(torch.tensor([[cur_id]]), device="cuda",  # noqa: F821
                              kv_cache=cache_tq4,
                              start_pos=len(ids) + step)
        torch.cuda.synchronize()
        tq4_decode_s += time.time() - t

        v_fp16 = r_fp16[0, -1].float().flatten()
        v_tq4 = r_tq4[0, -1].float().flatten()
        cos = F.cosine_similarity(v_fp16, v_tq4, dim=0).item()
        cosines.append(cos)
        a_fp16 = v_fp16.argmax().item()
        a_tq4 = v_tq4.argmax().item()
        if a_fp16 == a_tq4:
            argmax_matches += 1
        cur_id = int(a_fp16)

    min_cos = min(cosines)
    mean_cos = sum(cosines) / len(cosines)
    fp16_tps = n_decode / fp16_decode_s
    tq4_tps = n_decode / tq4_decode_s

    print(f"[r53.phase1] per-step cosine: min={min_cos:.4f} mean={mean_cos:.4f}")
    print(f"[r53.phase1] argmax matches: {argmax_matches}/{n_decode}")
    print(f"[r53.phase1] fp16 decode tok/s: {fp16_tps:.2f}")
    print(f"[r53.phase1] tq4  decode tok/s: {tq4_tps:.2f}")
    print(f"[r53.phase1] tq4 / fp16 ratio: {tq4_tps / fp16_tps:.3f}")

    parity_pass = (min_cos >= 0.999) and (argmax_matches >= int(0.9 * n_decode))
    print(f"[r53.phase1] PARITY: {'PASS' if parity_pass else 'FAIL'}")
    if not parity_pass:
        for i, c in enumerate(cosines):
            print(f"  step {i:2d}: cos={c:.4f}")


main()
