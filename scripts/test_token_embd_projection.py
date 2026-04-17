"""Round 9: does token-embedding projection replace the VerificationHook?

Hypothesis: writing Gemma's own token_embd[verified_answer_id] into the
residual at position -1 (mid-forward) causes Gemma's downstream layers
to process the signal as a legitimate token pattern they were trained
on, so the correct token wins the head argmax without any head-level
logit bias.

If the theory holds, a facade installed with:
  - PT CardSlot (same as Round 7/8)
  - adder CardSlot whose writer projects argmax → Gemma token_embd →
    residual position -1, additive
  - NO VerificationHook

...should match Round 8's 4/7 domain + 0 regressions result.

Comparison configs (same 10 prompts):
  A (baseline, no facade)
  B (projection-only, no hook)
  C (hook-only, Round 8 behavior) — sanity check we match 4/7
  D (both) — does combining help or hurt?
"""

from __future__ import annotations

import math
import os
import sys
import types

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


def make_projection_writer(gemma, projection_strength: float):
    """Return a new adder writer that replaces channel-bias writes with
    token-embedding projection. Bound to the facade instance via closure."""
    d_model = gemma.config.d_model
    scale = math.sqrt(d_model)

    def writer(h, card_out, ch_lo, ch_hi):
        # Still zero the reserved adder channels for consistency/debug.
        h[..., ch_lo:ch_hi] = 0.0
        # Access parse_state from the facade via the writer's bound attr.
        parse_ok = writer._facade._parse_ok
        if parse_ok:
            slot = int(card_out[0, -1].argmax())
            gemma_tok = DIGIT_TO_GEMMA.get(slot)
            if gemma_tok is not None:
                tok_ids = torch.tensor([gemma_tok], device=h.device)
                # (1, d_model), already scale-consistent with Gemma's
                # token_embd lookup (which multiplies by sqrt(d_model)
                # in forward() — we replicate that here).
                embd = gemma.token_embd[tok_ids].to(h.device)
                embd = embd * scale
                h[..., -1, :] = h[..., -1, :] + projection_strength * embd.squeeze(0)
        else:
            card_out.zero_()  # silence hook if also active
        return h
    return writer


def score(m, tok, facade, prompts_domain, prompts_reg, label):
    """Run both prompt sets and return (domain_correct, regressions)."""
    print(f"\n--- {label} ---")
    domain_correct = 0
    for p in prompts_domain:
        if facade is not None:
            facade.set_prompt(p)
        got = gemma_last_argmax(m, tok, p + " equals")
        a, b = [int(s) for s in p.split() if s.isdigit()]
        expected = DIGIT_TO_GEMMA.get(a + b)
        match = got == expected
        if match:
            domain_correct += 1
        print(f"  {'✓' if match else '✗'} {p!r:<24} got={decode_label(tok, got)!r}")

    print("  regression:")
    reg_tokens = []
    for p in prompts_reg:
        if facade is not None:
            facade.set_prompt(p)
        got = gemma_last_argmax(m, tok, p)
        reg_tokens.append(got)
        print(f"    {p!r:<32} -> {decode_label(tok, got)!r}")
    return domain_correct, reg_tokens


def main():
    from calm.llm_computer.gemma_substrate import (
        GemmaSubstrate, enable_triton_tq4,
    )
    from calm.llm_computer.synth.gemma_tokenizer import GemmaTokenizer
    from calm.llm_computer.facades import MathAdditionFacade

    enable_triton_tq4(True)
    print("[round-9] loading substrate...")
    m = GemmaSubstrate.from_gguf(GGUF_PATH, max_len=256)
    m.preload_gpu("cuda")
    m.warmup(seq_lens=(1, 6, 8))
    tok = GemmaTokenizer.from_gguf(GGUF_PATH)

    # ---- Config A: baseline (no facade) ----
    base_correct, base_reg = score(
        m, tok, None, DOMAIN_PROMPTS, REGRESSION_PROMPTS,
        "A — baseline (no facade)",
    )

    # ---- Config C: hook-only (Round 8 sanity check) ----
    print("\n[round-9] installing Round-8 facade (hook-only)...")
    facade = MathAdditionFacade()
    facade.install(m)
    c_correct, c_reg = score(
        m, tok, facade, DOMAIN_PROMPTS, REGRESSION_PROMPTS,
        "C — Round 8 hook-only",
    )
    facade.detach(m)

    # ---- Config B: projection-only, no hook ----
    print("\n[round-9] installing facade with token-embd projection (no hook)...")
    facade_b = MathAdditionFacade()
    facade_b.install(m)
    # Replace the adder writer with the projection variant and remove the
    # verification hook — this isolates the projection's effect.
    proj_writer = make_projection_writer(m, projection_strength=1.0)
    proj_writer._facade = facade_b
    facade_b._adder_slot.output_fn = proj_writer
    # Remove the VerificationHook we registered during install.
    m.verification_hooks = [
        h for h in m.verification_hooks if h is not facade_b._hook
    ]
    b_correct, b_reg = score(
        m, tok, facade_b, DOMAIN_PROMPTS, REGRESSION_PROMPTS,
        "B — projection-only (no hook), strength=1.0",
    )
    facade_b.detach(m)

    # ---- Config D: both ----
    print("\n[round-9] installing facade with projection + hook...")
    facade_d = MathAdditionFacade()
    facade_d.install(m)
    proj_writer_d = make_projection_writer(m, projection_strength=1.0)
    proj_writer_d._facade = facade_d
    facade_d._adder_slot.output_fn = proj_writer_d
    d_correct, d_reg = score(
        m, tok, facade_d, DOMAIN_PROMPTS, REGRESSION_PROMPTS,
        "D — projection + hook, strength=1.0",
    )
    facade_d.detach(m)

    # ---- Summary ----
    print("\n========== SUMMARY ==========")
    print(f"  A baseline            : {base_correct}/7 "
          f"reg={[decode_label(tok, t) for t in base_reg]}")
    print(f"  C hook-only (R8)      : {c_correct}/7 "
          f"reg={[decode_label(tok, t) for t in c_reg]}")
    print(f"  B projection-only     : {b_correct}/7 "
          f"reg={[decode_label(tok, t) for t in b_reg]}")
    print(f"  D both                : {d_correct}/7 "
          f"reg={[decode_label(tok, t) for t in d_reg]}")

    # Hypothesis test: B should match or exceed C with 0 regressions.
    b_regressed = sum(1 for bt, at in zip(b_reg, base_reg) if bt != at)
    c_regressed = sum(1 for ct, at in zip(c_reg, base_reg) if ct != at)
    print(f"\n  Hypothesis: projection replaces hook?")
    print(f"    B domain ≥ C domain?     {b_correct} ≥ {c_correct}: "
          f"{'YES' if b_correct >= c_correct else 'NO'}")
    print(f"    B regressions ≤ C?       {b_regressed} ≤ {c_regressed}: "
          f"{'YES' if b_regressed <= c_regressed else 'NO'}")
    verdict = (b_correct >= c_correct and b_regressed <= c_regressed)
    print(f"    verdict: {'SUPPORTED' if verdict else 'NOT SUPPORTED'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
