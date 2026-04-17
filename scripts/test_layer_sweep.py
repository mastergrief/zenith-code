"""Round 10a: install-layer sweep with token-embedding projection.

Hypothesis: installing the projection early (layer 1) gives Gemma's
downstream layers more opportunity to integrate the verified signal.
If earlier-install ≥ later-install on domain and doesn't increase
regressions, the "inject verified context early, let Gemma reason
forward" story holds.

Counter-prediction: injecting a token embedding at position -1 very
early might confuse Gemma's layers 2-41, which expect that position
to carry the prompt's last BPE token ("equals"), not a superposition
of that plus a digit embedding. Downstream attention could distort.

Test: same facade, same PT, same adder, same projection strength.
Only `host_layer` varies across {1, 5, 15, 25, 33, 40}.

Measure: argmax on 7 domain + 3 regression prompts per layer.
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

LAYERS_TO_TEST = [1, 5, 15, 25, 33, 40]


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


def make_projection_writer(gemma, facade_ref):
    d_model = gemma.config.d_model
    scale = math.sqrt(d_model)

    def writer(h, card_out, ch_lo, ch_hi):
        h[..., ch_lo:ch_hi] = 0.0
        if facade_ref["f"]._parse_ok:
            slot = int(card_out[0, -1].argmax())
            gemma_tok = DIGIT_TO_GEMMA.get(slot)
            if gemma_tok is not None:
                tok_ids = torch.tensor([gemma_tok], device=h.device)
                embd = gemma.token_embd[tok_ids].to(h.device) * scale
                h[..., -1, :] = h[..., -1, :] + embd.squeeze(0)
        else:
            card_out.zero_()
        return h
    return writer


def run_config(m, tok, layer_idx, gemma_ref):
    """Install facade at `layer_idx` with projection-only writer (no hook).
    Returns (domain_correct, reg_tokens)."""
    from calm.llm_computer.facades import MathAdditionFacade
    facade = MathAdditionFacade(layer=layer_idx)
    facade.install(m)
    # Swap writer for projection; remove hook
    facade_ref = {"f": facade}
    proj_writer = make_projection_writer(gemma_ref, facade_ref)
    facade._adder_slot.output_fn = proj_writer
    m.verification_hooks = [
        h for h in m.verification_hooks if h is not facade._hook
    ]

    domain_correct = 0
    dom_rows = []
    for p in DOMAIN:
        facade.set_prompt(p)
        got = gemma_last_argmax(m, tok, p + " equals")
        a, b = [int(s) for s in p.split() if s.isdigit()]
        expected = DIGIT_TO_GEMMA.get(a + b)
        match = got == expected
        if match:
            domain_correct += 1
        dom_rows.append((p, got, expected, match))

    reg_tokens = []
    for p in REGRESSION:
        facade.set_prompt(p)
        reg_tokens.append(gemma_last_argmax(m, tok, p))

    facade.detach(m)
    return domain_correct, dom_rows, reg_tokens


def main():
    from calm.llm_computer.gemma_substrate import (
        GemmaSubstrate, enable_triton_tq4,
    )
    from calm.llm_computer.synth.gemma_tokenizer import GemmaTokenizer

    enable_triton_tq4(True)
    print("[layer-sweep] loading substrate...")
    m = GemmaSubstrate.from_gguf(GGUF_PATH, max_len=256)
    m.preload_gpu("cuda")
    m.warmup(seq_lens=(1, 6, 8))
    tok = GemmaTokenizer.from_gguf(GGUF_PATH)

    # Baseline regression tokens (used to detect regressions per config)
    print("\n=== baseline (no facade) ===")
    base_reg = []
    for p in REGRESSION:
        t = gemma_last_argmax(m, tok, p)
        base_reg.append(t)
        print(f"  {p!r:<32} -> {decode_label(tok, t)!r}")

    results = {}
    for layer in LAYERS_TO_TEST:
        print(f"\n=== layer {layer} ===")
        dom_n, dom_rows, reg_tokens = run_config(m, tok, layer, m)
        regressions = sum(1 for b, r in zip(base_reg, reg_tokens) if b != r)
        results[layer] = (dom_n, regressions, dom_rows, reg_tokens)
        for p, got, exp, match in dom_rows:
            print(f"  {'✓' if match else '✗'} {p!r:<24} "
                  f"got={decode_label(tok, got)!r}")
        print(f"  domain: {dom_n}/7, regressions: {regressions}")
        print(f"  reg: {[decode_label(tok, t) for t in reg_tokens]}")

    # Summary
    print("\n========== SWEEP SUMMARY ==========")
    print(f"{'layer':>6} {'domain':>8} {'regressions':>12}")
    for layer in LAYERS_TO_TEST:
        dom_n, regs, _, _ = results[layer]
        print(f"{layer:>6} {dom_n:>4}/7    {regs:>6}")

    # Hypothesis check
    dom_at_1 = results[1][0]
    dom_at_33 = results[33][0]
    reg_at_1 = results[1][1]
    reg_at_33 = results[33][1]
    print(f"\n  layer 1 domain ≥ layer 33 domain?  {dom_at_1} ≥ {dom_at_33}: "
          f"{'YES' if dom_at_1 >= dom_at_33 else 'NO'}")
    print(f"  layer 1 regressions ≤ layer 33?   {reg_at_1} ≤ {reg_at_33}: "
          f"{'YES' if reg_at_1 <= reg_at_33 else 'NO'}")


if __name__ == "__main__":
    sys.exit(main())
