"""End-to-end reasoning facade on prod Gemma (Round 6).

Composes:
  - Gemma 4 E4B (the brain)
  - CalmVerifier (the oracle — uses CALM's 1002-function registry)
  - KnowledgeStore (persistence — corrections compile to a recall card)
  - CardSlot + VerificationHook (install path into prod Gemma)

Flow:
  1. Baseline Gemma on 10 mixed-domain prompts (arithmetic, primality,
     gcd) + 5 regression prompts (capitals, facts Gemma already knows)
  2. For each wrong domain answer, CalmVerifier computes ground truth;
     log (key=hash(prompt), value=truth) to KnowledgeStore
  3. Compile KnowledgeStore → recall card → install via CardSlot with
     VerificationHook biasing the corresponding Gemma digit logit
  4. Re-run all 15 prompts; count fixes on domain, regressions on
     non-domain

Result: Round 6 PASS if domain fixes > 0 and regressions == 0.
"""

from __future__ import annotations

import os
import sys
import time

import torch


GGUF_PATH = os.path.expanduser("~/models/gemma-4-E4B-it-tq4-aligned.gguf")


# Gemma 4 E4B BPE token IDs for single digits 0..9.
DIGIT_TO_GEMMA = {
    0: 236771, 1: 236770, 2: 236778, 3: 236800, 4: 236812,
    5: 236810, 6: 236825, 7: 236832, 8: 236828, 9: 236819,
}

MAX_KEY = 1024
MAX_VALUE = 10  # single-digit answers; fits DIGIT_TO_GEMMA


# 10 domain prompts. Each has a phrasing CALM can parse and answer
# within [0, 10). Some Gemma gets right, some wrong — the mix is
# deliberate to test both improvement AND non-regression.
DOMAIN_PROMPTS = [
    "2 plus 3 equals",
    "4 plus 1 equals",
    "3 plus 2 equals",
    "5 plus 1 equals",
    "2 plus 4 equals",
    "Is 5 prime?",
    "Is 9 prime?",
    "Is 7 prime?",
    "gcd of 6 and 9 equals",
    "gcd of 4 and 8 equals",
]

# 5 non-domain prompts — CalmVerifier returns None for these; they
# should pass through Gemma unchanged. Regression check: the install
# must not corrupt Gemma's behavior on untouched prompts.
REGRESSION_PROMPTS = [
    "The capital of France is",
    "The capital of Germany is",
    "The capital of Italy is",
    "The largest ocean is the",
    "Water freezes at",
]


def gemma_last_argmax(m, tok, prompt: str) -> int:
    """Return Gemma's argmax at the last prompt position. Fresh cache
    per call so each prompt is independent."""
    from calm.llm_computer.gemma_substrate import KVCache
    ids = tok.encode(prompt)
    cache = KVCache(m.config.n_layers, device="cuda")
    with torch.no_grad():
        logits = m.forward(torch.tensor([ids]), device="cuda",
                            kv_cache=cache, start_pos=0)
    return int(logits[0, -1].argmax().item())


def decode_token_label(tok, tok_id: int) -> str:
    """Human-readable label for a Gemma token id."""
    return tok.id_to_token.get(tok_id, f"?{tok_id}")


def score_prompts(m, tok, prompts, expected_tokens):
    """Return (correct_count, results). results is a list of
    (prompt, got_id, expected_id, match)."""
    results = []
    correct = 0
    for prompt, expected in zip(prompts, expected_tokens):
        got = gemma_last_argmax(m, tok, prompt)
        match = (got == expected) if expected is not None else None
        if match is True:
            correct += 1
        results.append((prompt, got, expected, match))
    return correct, results


def print_results(header, results, tok):
    print(f"\n{header}")
    for prompt, got, expected, match in results:
        got_label = decode_token_label(tok, got)
        if expected is None:
            mark = "  "
            exp_str = "-"
        else:
            mark = "✓ " if match else "✗ "
            exp_str = decode_token_label(tok, expected)
        print(f"  {mark}{prompt!r:<40} got={got_label!r} "
              f"expected={exp_str!r}")


def main():
    from calm.llm_computer.gemma_substrate import (
        GemmaSubstrate, enable_triton_tq4,
        CardSlot, VerificationHook,
    )
    from calm.llm_computer.synth.gemma_tokenizer import GemmaTokenizer
    from calm.llm_computer.calm_verifier import CalmVerifier, make_key
    from calm.llm_computer.persistent_knowledge import KnowledgeStore

    enable_triton_tq4(True)
    print("[facade] loading substrate...")
    m = GemmaSubstrate.from_gguf(GGUF_PATH, max_len=256)
    m.preload_gpu("cuda")
    m.warmup(seq_lens=(1, 6))
    tok = GemmaTokenizer.from_gguf(GGUF_PATH)

    verifier = CalmVerifier(max_value=MAX_VALUE)

    # ---------- Phase 1: CALM computes ground truth for domain prompts ----------
    print("\n=== PHASE 1: CALM-as-oracle ===")
    domain_expected = []
    for prompt in DOMAIN_PROMPTS:
        expr, value = verifier.verify_nl(prompt)
        if value is None:
            domain_expected.append(None)
            print(f"  {prompt!r:<40} CALM: UNVERIFIABLE")
            continue
        domain_expected.append(DIGIT_TO_GEMMA[value])
        print(f"  {prompt!r:<40} CALM: {expr!r} = {value}")

    # ---------- Phase 2: Gemma baseline ----------
    print("\n=== PHASE 2: Gemma baseline ===")
    base_domain_correct, base_domain = score_prompts(
        m, tok, DOMAIN_PROMPTS, domain_expected)

    # For regression prompts we expected_token = Gemma's baseline argmax
    # (because we want to verify install doesn't change them).
    regression_baseline = []
    for prompt in REGRESSION_PROMPTS:
        regression_baseline.append(gemma_last_argmax(m, tok, prompt))

    print_results("baseline domain:", base_domain, tok)
    print(f"  → {base_domain_correct}/{len(DOMAIN_PROMPTS)} domain correct")
    print("\nbaseline regression prompts (these Gemma tokens must NOT change):")
    for prompt, tok_id in zip(REGRESSION_PROMPTS, regression_baseline):
        print(f"     {prompt!r:<40} → {decode_token_label(tok, tok_id)!r}")

    # ---------- Phase 3: log corrections for wrong domain answers ----------
    print("\n=== PHASE 3: log corrections + compile recall card ===")
    store = KnowledgeStore(max_key=MAX_KEY, max_value=MAX_VALUE)
    logged = 0
    for (prompt, got, expected_tok, match), verifier_value in zip(
            base_domain,
            [verifier.verify_nl(p)[1] for p in DOMAIN_PROMPTS]):
        if verifier_value is None:
            continue
        if not match:
            key = make_key(prompt, max_key=MAX_KEY)
            store.add_correction(key, verifier_value)
            logged += 1
            print(f"  + key={key:4d} value={verifier_value} "
                  f"for {prompt!r}")

    print(f"\n  {logged} correction(s) logged to KnowledgeStore")
    if logged == 0:
        print("  Gemma baseline was perfect on domain — nothing to install.")
        return 0

    recall = store.build_recall_model(d_model=16, max_len=4, min_d_ffn=4)
    recall = recall.cuda().eval()
    n_params = sum(p.numel() for p in recall.parameters())
    print(f"  recall card: d_model={recall.config.d_model}, "
          f"vocab={recall.config.vocab_size}, params={n_params}")

    # ---------- Phase 4: install via CardSlot + VerificationHook ----------
    print("\n=== PHASE 4: install into Gemma ===")
    current_query = {"key": 0}

    def recall_input(h):
        return torch.tensor([[current_query["key"]]], device="cuda")

    def recall_output(h, logits, ch_lo, ch_hi):
        # Recall card's LinearHead has `vocab_size` outputs but only
        # the first MAX_VALUE slots are meaningful (digit answers). Pad
        # slots are always 0 so ignoring them is safe.
        ans = logits[:, -1:, :MAX_VALUE]
        n = min(ans.shape[-1], ch_hi - ch_lo)
        h[..., -1:, ch_lo:ch_lo + n] = (
            h[..., -1:, ch_lo:ch_lo + n] + ans[..., :n])
        return h

    slot = CardSlot(
        layer_idx=35, ch_off=2480, card=recall,
        d_card=MAX_VALUE,
        card_input_fn=recall_input,
        use_full_residual=True,
        output_fn=recall_output,
    )
    slot.attach(m, preserve=True)
    # min_margin guards against unmatched keys: recall card returns
    # all-zero logits for unknown keys; the hook must not fire then.
    hook = VerificationHook(slot, vocab_mapping=DIGIT_TO_GEMMA,
                             boost=50.0, min_margin=0.5)
    m.verification_hooks.append(hook)
    print(f"  installed CardSlot at layer 35 ch=[2480:{2480 + MAX_VALUE}]")
    print(f"  VerificationHook boost=50.0 min_margin=0.5 on DIGIT_TO_GEMMA")

    # ---------- Phase 5: re-run domain + regression ----------
    print("\n=== PHASE 5: re-run with facade active ===")
    post_domain_results = []
    for prompt, expected in zip(DOMAIN_PROMPTS, domain_expected):
        current_query["key"] = make_key(prompt, max_key=MAX_KEY)
        got = gemma_last_argmax(m, tok, prompt)
        match = (got == expected) if expected is not None else None
        post_domain_results.append((prompt, got, expected, match))

    post_regression_results = []
    for prompt, baseline_tok in zip(REGRESSION_PROMPTS, regression_baseline):
        current_query["key"] = make_key(prompt, max_key=MAX_KEY)
        got = gemma_last_argmax(m, tok, prompt)
        match = (got == baseline_tok)
        post_regression_results.append((prompt, got, baseline_tok, match))

    post_domain_correct = sum(
        1 for _, _, _, m_ in post_domain_results if m_ is True)

    print_results("post-install domain:", post_domain_results, tok)
    print(f"  → {post_domain_correct}/{len(DOMAIN_PROMPTS)} domain correct "
          f"(was {base_domain_correct})")

    regressions = [
        r for r in post_regression_results if r[3] is False
    ]
    print_results("post-install regression:", post_regression_results, tok)
    print(f"  → {len(regressions)} regressions (want 0)")

    # ---------- Verdict ----------
    print("\n========== SUMMARY ==========")
    fixed = post_domain_correct - base_domain_correct
    print(f"  baseline domain: {base_domain_correct}/{len(DOMAIN_PROMPTS)}")
    print(f"  post-install:    {post_domain_correct}/{len(DOMAIN_PROMPTS)}")
    print(f"  fixes:           +{fixed}")
    print(f"  regressions:     {len(regressions)}")
    ok = fixed > 0 and len(regressions) == 0
    print(f"  verdict: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
