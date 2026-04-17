"""Round 10c: multi-token projection + continuation test.

Hypothesis: if we inject a SEQUENCE of verified tokens into Gemma's
context (not just bias the next logit), Gemma's forward pass processes
those tokens as established context and CONTINUES reasoning about
them. This would be real integration, not just a head bias.

Concrete test:
  User prompt:  "2 plus 3 equals"   (6 BPE tokens roughly)
  Inject:       [" ", "5"] embeddings at positions [-2, -1]
  Continue:     let Gemma generate N more tokens autoregressively

Baseline (no injection): Gemma emits gibberish or "▁"
With single-token injection at -1: Gemma emits "5" once, then ???
With multi-token injection at [-2, -1]: Gemma should emit "5", then
  reason onward (e.g. "5. That is because 2+3=5...")

Metric: does Gemma's CONTINUATION contain additional semantic content
that depends on the "5" it was shown? Or does it produce noise?

This is a qualitative + quantitative test. We measure:
  1. First emitted token after prompt (the injection target)
  2. Next 20 tokens (autoregressive continuation)
  3. Does continuation contain coherent prose that references "5" or
     arithmetic reasoning?

Comparisons:
  A: no facade (baseline Gemma)
  B: facade with single-token projection at -1 (Round 9 config)
  C: facade with injection at -1 + autoreg continuation
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

PROMPTS = [
    ("what is 2 plus 3 equals", 5),
    ("what is 4 plus 1 equals", 5),
    ("what is 5 plus 1 equals", 6),
    ("what is 2 plus 4 equals", 6),
]


def generate(m, tok, prompt, max_tokens=20):
    """Baseline autoregressive generation from a prompt."""
    from calm.llm_computer.gemma_substrate import KVCache
    ids = tok.encode(prompt)
    cache = KVCache(m.config.n_layers, device="cuda")
    gen_ids = list(ids)
    with torch.no_grad():
        # Prefill
        logits = m.forward(
            torch.tensor([gen_ids]), device="cuda",
            kv_cache=cache, start_pos=0,
        )
        nxt = int(logits[0, -1].argmax())
        gen_ids.append(nxt)
        # Decode
        for i in range(max_tokens - 1):
            logits = m.forward(
                torch.tensor([[nxt]]), device="cuda",
                kv_cache=cache, start_pos=len(gen_ids) - 1,
            )
            nxt = int(logits[0, -1].argmax())
            gen_ids.append(nxt)
    return gen_ids[len(ids):], tok.decode(gen_ids[len(ids):])


def generate_with_facade(m, tok, prompt, facade, gemma_ref, max_tokens=20,
                         strength=1.0):
    """Autoregressive generation WITH facade installed.

    The projection fires once (at prefill), injecting the verified token
    embedding at prompt's last position. Then Gemma continues normally;
    we do NOT re-inject at each decode step (the PT's state is based on
    the prompt, not the growing generation, so re-injecting would just
    spam the same token)."""
    from calm.llm_computer.gemma_substrate import KVCache
    # Set PT input from prompt
    facade.set_prompt(prompt.replace(" equals", ""))
    # One-shot injection: after prefill, detach facade so further decode
    # steps don't re-fire.
    ids = tok.encode(prompt)
    cache = KVCache(m.config.n_layers, device="cuda")
    with torch.no_grad():
        logits = m.forward(
            torch.tensor([ids]), device="cuda",
            kv_cache=cache, start_pos=0,
        )
    nxt = int(logits[0, -1].argmax())
    gen_ids = list(ids) + [nxt]

    # For decode steps, we need to NOT re-fire the facade (the PT
    # doesn't know what the new tokens are; re-firing would inject
    # the old prompt's verified answer forever). So detach here.
    facade.detach(m)

    with torch.no_grad():
        for i in range(max_tokens - 1):
            logits = m.forward(
                torch.tensor([[nxt]]), device="cuda",
                kv_cache=cache, start_pos=len(gen_ids) - 1,
            )
            nxt = int(logits[0, -1].argmax())
            gen_ids.append(nxt)

    return gen_ids[len(ids):], tok.decode(gen_ids[len(ids):])


def make_proj_writer(gemma, facade_holder, strength=1.0):
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


def main():
    from calm.llm_computer.gemma_substrate import (
        GemmaSubstrate, enable_triton_tq4,
    )
    from calm.llm_computer.synth.gemma_tokenizer import GemmaTokenizer
    from calm.llm_computer.facades import MathAdditionFacade

    enable_triton_tq4(True)
    print("[multi-token] loading substrate...")
    m = GemmaSubstrate.from_gguf(GGUF_PATH, max_len=256)
    m.preload_gpu("cuda")
    m.warmup(seq_lens=(1, 6, 8, 20))
    tok = GemmaTokenizer.from_gguf(GGUF_PATH)

    print("\n=== A: baseline (no facade) ===")
    for prompt, expected in PROMPTS:
        ids, text = generate(m, tok, prompt, max_tokens=20)
        print(f"  prompt: {prompt!r}  →")
        print(f"    continuation: {text!r}")

    print("\n=== C: one-shot projection at prefill, detach for decode ===")
    for prompt, expected in PROMPTS:
        facade = MathAdditionFacade(layer=33)
        facade.install(m)
        holder = {"f": facade}
        facade._adder_slot.output_fn = make_proj_writer(m, holder, strength=1.0)
        m.verification_hooks = [
            h for h in m.verification_hooks if h is not facade._hook
        ]
        ids, text = generate_with_facade(
            m, tok, prompt, facade, m, max_tokens=20, strength=1.0,
        )
        first_tok = ids[0]
        correct_first = (first_tok == DIGIT_TO_GEMMA[expected])
        print(f"  prompt: {prompt!r}  expected first: {expected}")
        print(f"    first token: {tok.id_to_token.get(first_tok, '?')!r} "
              f"{'✓' if correct_first else '✗'}")
        print(f"    continuation: {text!r}")


if __name__ == "__main__":
    sys.exit(main())
