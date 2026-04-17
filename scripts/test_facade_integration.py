"""Integration test: reproduce Round 7 via the MathAdditionFacade API.

Must match Round 7's outcome (0/7 → 4/7 domain, 0 regressions).
Proves the class encapsulation is behaviorally equivalent to the
inline wiring.

Key API usage:
    facade = MathAdditionFacade(pt_ckpt_path=...)
    facade.install(gemma)

    # Baseline (no install state): detach first
    facade.detach(gemma)
    score_baseline(gemma, ...)

    facade.install(gemma)
    for prompt in prompts:
        facade.set_prompt(prompt)
        out = gemma.forward(...)
"""

from __future__ import annotations

import os
import sys

import torch


GGUF_PATH = os.path.expanduser("~/models/gemma-4-E4B-it-tq4-aligned.gguf")

DIGIT_TO_GEMMA = {
    0: 236771, 1: 236770, 2: 236778, 3: 236800, 4: 236812,
    5: 236810, 6: 236825, 7: 236832, 8: 236828, 9: 236819,
}

DOMAIN_PROMPTS = [
    "what is 2 plus 3",
    "what is 4 plus 1",
    "what is 3 plus 2",
    "what is 5 plus 1",
    "what is 2 plus 4",
    "what is 1 plus 6",
    "what is 3 plus 4",
]

REGRESSION_PROMPTS = [
    "The capital of France is",
    "The capital of Germany is",
    "The capital of Italy is",
]


def gemma_last_argmax(m, tok, prompt):
    from calm.llm_computer.gemma_substrate import KVCache
    ids = tok.encode(prompt)
    cache = KVCache(m.config.n_layers, device="cuda")
    with torch.no_grad():
        logits = m.forward(torch.tensor([ids]), device="cuda",
                            kv_cache=cache, start_pos=0)
    return int(logits[0, -1].argmax().item())


def decode_label(tok, tok_id):
    return tok.id_to_token.get(tok_id, f"?{tok_id}")


def main():
    from calm.llm_computer.gemma_substrate import (
        GemmaSubstrate, enable_triton_tq4,
    )
    from calm.llm_computer.synth.gemma_tokenizer import GemmaTokenizer
    from calm.llm_computer.facades import MathAdditionFacade

    enable_triton_tq4(True)
    print("[integration] loading substrate...")
    m = GemmaSubstrate.from_gguf(GGUF_PATH, max_len=256)
    m.preload_gpu("cuda")
    m.warmup(seq_lens=(1, 6, 8))
    tok = GemmaTokenizer.from_gguf(GGUF_PATH)

    print("[integration] constructing MathAdditionFacade...")
    facade = MathAdditionFacade()  # defaults: layer 33, ch_base 2400
    print(f"  alloc: layer={facade.alloc.layer} "
          f"PT ch{facade.alloc.pt_ch} adder ch{facade.alloc.adder_ch}")

    # ----- Baseline (facade NOT installed) -----
    print("\n=== baseline (no facade) ===")
    base_domain = []
    for p in DOMAIN_PROMPTS:
        got = gemma_last_argmax(m, tok, p + " equals")
        a, b = [int(s) for s in p.split() if s.isdigit()]
        expected = DIGIT_TO_GEMMA.get(a + b)
        match = got == expected
        base_domain.append((p, got, expected, match))
        print(f"  {'✓' if match else '✗'} {p!r:<24} got={decode_label(tok, got)!r}")

    base_reg = [(p, gemma_last_argmax(m, tok, p)) for p in REGRESSION_PROMPTS]

    # ----- Install facade -----
    print("\n=== installing facade ===")
    facade.install(m)

    post_domain = []
    for p in DOMAIN_PROMPTS:
        facade.set_prompt(p)
        got = gemma_last_argmax(m, tok, p + " equals")
        a, b = [int(s) for s in p.split() if s.isdigit()]
        expected = DIGIT_TO_GEMMA.get(a + b)
        match = got == expected
        post_domain.append((p, got, expected, match))
        print(f"  {'✓' if match else '✗'} {p!r:<24} got={decode_label(tok, got)!r}")

    post_reg = []
    for p in REGRESSION_PROMPTS:
        facade.set_prompt(p)
        got = gemma_last_argmax(m, tok, p)
        post_reg.append((p, got))

    print("\n  regression:")
    regressed = 0
    for (p, base), (_, post) in zip(base_reg, post_reg):
        ch = base != post
        regressed += ch
        mark = "✗" if ch else "✓"
        print(f"  {mark} {p!r:<32} base={decode_label(tok, base)!r} "
              f"post={decode_label(tok, post)!r}")

    # ----- detach() + re-run baseline → must match original baseline -----
    print("\n=== detach sanity check ===")
    facade.detach(m)
    post_detach = []
    for p in DOMAIN_PROMPTS:
        got = gemma_last_argmax(m, tok, p + " equals")
        post_detach.append(got)
    detach_matches_base = [
        post_detach[i] == base_domain[i][1]
        for i in range(len(DOMAIN_PROMPTS))
    ]
    print(f"  post-detach tokens match baseline: "
          f"{sum(detach_matches_base)}/{len(DOMAIN_PROMPTS)}")

    # ----- Verdict -----
    base_ok = sum(1 for _, _, _, m_ in base_domain if m_)
    post_ok = sum(1 for _, _, _, m_ in post_domain if m_)
    print("\n========== SUMMARY ==========")
    print(f"  baseline domain:   {base_ok}/{len(DOMAIN_PROMPTS)}")
    print(f"  post-install:      {post_ok}/{len(DOMAIN_PROMPTS)}")
    print(f"  fixes:             +{post_ok - base_ok}")
    print(f"  regressions:       {regressed}")
    print(f"  detach reversible: {all(detach_matches_base)}")
    ok = (post_ok - base_ok >= 4
          and regressed == 0
          and all(detach_matches_base))
    print(f"  verdict: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
