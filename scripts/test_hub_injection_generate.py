"""Round 45: exercise HubInjectionCard.generate() end-to-end.

Hypothesis: hub injection at prefill preserves argmax on the first
answer token (R28/R42/R43 numbers) AND does not corrupt downstream
decode — i.e., continuation after the first token remains coherent.

Raw path: baseline GemmaSubstrate.generate vs facade.generate(inject=False)
should produce identical output (facade's no-inject path just adds a
thin wrapper around gemma.forward).

User-facing path: facade.generate(inject=True) vs baseline on prompts
from R42/R43. First token must match baseline (R43 argmax preservation
carries over to the generation first-step). Continuation must parse
as coherent text (no garbling).

Gate: first-token match 5/5, continuations non-empty and non-repeating.
"""

from __future__ import annotations

import os
import sys

PROMPTS = [
    # Comparison (R43a: 18/18 argmax match)
    ("Which is larger, 17 or 23? Answer: ", "comparison"),
    ("Which is larger, 4 or 9? Answer: ", "comparison"),
    # Counting (R43b: 6/6 argmax match)
    ("Count: 2, 3, 4, 5, ", "counting"),
    # SV agreement (R42: 8/10 argmax match)
    ("The cat that sits near the window", "sv"),
    ("The keys to the cabinet", "sv"),
]


def main():
    import torch  # noqa: F401
    from calm.llm_computer.gemma_substrate import (
        GemmaSubstrate, enable_triton_tq4)
    from calm.llm_computer.synth.gemma_tokenizer import GemmaTokenizer
    from calm.llm_computer.facades import HubInjectionCard

    gguf = os.path.expanduser("~/models/gemma-4-E4B-it-tq4-aligned.gguf")
    enable_triton_tq4(True)
    print("[r45] loading substrate...")
    m = GemmaSubstrate.from_gguf(gguf, max_len=256)
    m.preload_gpu("cuda")
    m.warmup(seq_lens=(1, 16))
    tok = GemmaTokenizer.from_gguf(gguf)

    card = HubInjectionCard()
    card.install(m)
    print(f"[r45] installed L{card.target_layer}, heads={card.heads}\n")

    max_tokens = 12
    first_match = 0
    continuations_ok = 0

    for prompt, cap in PROMPTS:
        print(f"--- {cap}: {prompt!r} ---")

        base = m.generate(prompt, tok, max_tokens=max_tokens)
        facade_clean = card.generate(prompt, tok, max_tokens=max_tokens,
                                     inject=False)
        facade_hub = card.generate(prompt, tok, max_tokens=max_tokens,
                                    inject=True)

        b_first = base["token_ids"][0]
        fc_first = facade_clean["token_ids"][0]
        fh_first = facade_hub["token_ids"][0]

        print(f"  baseline     : {base['text']!r}")
        print(f"  facade clean : {facade_clean['text']!r}")
        print(f"  facade hub   : {facade_hub['text']!r}")

        # Raw-path: no-inject facade must match baseline token-for-token
        clean_match = (base["token_ids"] == facade_clean["token_ids"])
        if not clean_match:
            print(f"  ✗ clean-path divergence (facade infrastructure bug)")

        # First-token match (R43 argmax preservation through generate)
        if fh_first == b_first:
            first_match += 1
            print(f"  ✓ first-token match (hub preserves baseline answer)")
        else:
            print(f"  ~ first-token diverges: "
                  f"base={tok.id_to_token.get(b_first, '?')!r} "
                  f"hub={tok.id_to_token.get(fh_first, '?')!r}")

        # Continuation coherence: facade_hub output is non-empty and has
        # diversity (not a single repeated token across all max_tokens).
        unique_toks = len(set(facade_hub["token_ids"]))
        if len(facade_hub["token_ids"]) >= 3 and unique_toks >= 2:
            continuations_ok += 1
        else:
            print(f"  ✗ degenerate continuation "
                  f"({unique_toks} unique / "
                  f"{len(facade_hub['token_ids'])} tokens)")
        print()

    n = len(PROMPTS)
    print(f"\n=== ROUND 45 SUMMARY ===")
    print(f"  first-token match (hub vs baseline): {first_match}/{n}")
    print(f"  coherent continuations:              {continuations_ok}/{n}")

    if first_match == n and continuations_ok == n:
        print(f"\n  ✓ HubInjectionCard.generate() VALIDATED.")
        print(f"    Prefill-only injection preserves baseline first token")
        print(f"    AND downstream decode remains coherent.")
        return 0
    else:
        print(f"\n  ✗ Partial. Investigate divergence.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
