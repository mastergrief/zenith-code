"""Round 10b: strength sweep for token-embedding projection.

Hypothesis: projection strength is causally doing the work. There's
an operating range; too low and nothing moves, too high and the
injected embedding overrides other prompt signals → regressions.

Test at layer 33, position -1, MathAdditionFacade. Sweep α ∈ {0.0,
0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0}.

Expect: U-shape. α=0 matches baseline (nothing injected). Some low
α fails (too weak). Range around α=1 works (matches Round 9). High α
plausibly regresses regression prompts by overpowering e.g. Paris.
"""

from __future__ import annotations

import math
import os
import sys

import torch


GGUF_PATH = os.path.expanduser("~/models/gemma-4-E4B-it-tq4-aligned.gguf")

DIGIT_TO_GEMMA = {
    0: 236771, 1: 236770, 2: 236778, 3: 236800, 4: 236812,
    5: 236810, 6: 236825, 7: 236832, 8: 236828, 9: 236819,
}

DOMAIN = [
    "what is 2 plus 3",
    "what is 4 plus 1",
    "what is 3 plus 2",
    "what is 5 plus 1",
    "what is 2 plus 4",
    "what is 1 plus 6",
    "what is 3 plus 4",
]
REGRESSION = [
    "The capital of France is",
    "The capital of Germany is",
    "The capital of Italy is",
]

STRENGTHS = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]


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


def make_writer(gemma, facade_holder, strength):
    d_model = gemma.config.d_model
    scale = math.sqrt(d_model)

    def writer(h, card_out, ch_lo, ch_hi):
        h[..., ch_lo:ch_hi] = 0.0
        if facade_holder["f"]._parse_ok:
            slot = int(card_out[0, -1].argmax())
            gemma_tok = DIGIT_TO_GEMMA.get(slot)
            if gemma_tok is not None:
                tok_ids = torch.tensor([gemma_tok], device=h.device)
                embd = gemma.token_embd[tok_ids].to(h.device) * scale
                h[..., -1, :] = h[..., -1, :] + strength * embd.squeeze(0)
        else:
            card_out.zero_()
        return h
    return writer


def run(m, tok, strength, gemma_ref):
    from calm.llm_computer.facades import MathAdditionFacade
    facade = MathAdditionFacade(layer=33)
    facade.install(m)
    holder = {"f": facade}
    facade._adder_slot.output_fn = make_writer(gemma_ref, holder, strength)
    m.verification_hooks = [
        h for h in m.verification_hooks if h is not facade._hook
    ]

    dom_correct = 0
    dom_rows = []
    for p in DOMAIN:
        facade.set_prompt(p)
        got = gemma_last_argmax(m, tok, p + " equals")
        a, b = [int(s) for s in p.split() if s.isdigit()]
        expected = DIGIT_TO_GEMMA.get(a + b)
        match = got == expected
        if match:
            dom_correct += 1
        dom_rows.append((p, got, match))

    reg_tokens = []
    for p in REGRESSION:
        facade.set_prompt(p)
        reg_tokens.append(gemma_last_argmax(m, tok, p))

    facade.detach(m)
    return dom_correct, reg_tokens, dom_rows


def main():
    from calm.llm_computer.gemma_substrate import (
        GemmaSubstrate, enable_triton_tq4,
    )
    from calm.llm_computer.synth.gemma_tokenizer import GemmaTokenizer

    enable_triton_tq4(True)
    print("[strength-sweep] loading substrate...")
    m = GemmaSubstrate.from_gguf(GGUF_PATH, max_len=256)
    m.preload_gpu("cuda")
    m.warmup(seq_lens=(1, 6, 8))
    tok = GemmaTokenizer.from_gguf(GGUF_PATH)

    # True baseline for regressions
    base_reg = [gemma_last_argmax(m, tok, p) for p in REGRESSION]
    print(f"baseline reg: {[decode_label(tok, t) for t in base_reg]}")

    results = {}
    for s in STRENGTHS:
        print(f"\n--- α = {s} ---")
        dom, reg, rows = run(m, tok, s, m)
        regs = sum(1 for b, r in zip(base_reg, reg) if b != r)
        results[s] = (dom, regs, reg)
        print(f"  domain: {dom}/7 regs: {regs}")
        print(f"  reg: {[decode_label(tok, t) for t in reg]}")

    print("\n========== STRENGTH SWEEP ==========")
    print(f"{'α':>6} {'domain':>8} {'regs':>6}  reg_tokens")
    for s in STRENGTHS:
        dom, regs, reg = results[s]
        reg_labels = [decode_label(tok, t) for t in reg]
        print(f"{s:>6} {dom:>4}/7    {regs:>3}  {reg_labels}")


if __name__ == "__main__":
    sys.exit(main())
