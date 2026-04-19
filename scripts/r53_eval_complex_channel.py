"""R53.7 — re-run R53.0 complex eval with channel-code-hybrid retrieval.

Original R53.0 eval (`r53_eval_complex.py`) used CodeVerifierFacade
which called db.retrieve(prompt) — combined-hybrid mode. Result:
retrieval-attributable gain = +0.0pp.

This variant swaps to db.retrieve_channel('code', mode='hybrid'). The
R53.6 comparison showed channel-code-hybrid produces meaningfully
different + qualitatively cleaner top-3 on at least token_bucket
(avoided dense-similarity junk). This eval answers: does cleaner
retrieval translate to better Gemma pass rate?

Hypothesis: channel-code-hybrid moves hinted-vs-stock by at least
+5pp over what combined-hybrid achieved. Anything less means
retrieval mode wasn't the bottleneck.

Daemon-only:
  bin/gemma-run scripts/r53_eval_complex_channel.py
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path


CACHE_DIR = "/mnt/c/Users/gabes/projects/claw-code/.cache/r53_code_db"


def run_eval_channel(m, tok, max_tokens: int = 16384, seed: int = 0) -> None:
    # Force reimport so we pick up any code_example_db / code_verifier
    # edits from this session.
    import sys
    for mod_name in list(sys.modules.keys()):
        if (mod_name.startswith("calm.llm_computer.facades.")
                or mod_name == "calm.llm_computer.facades"):
            del sys.modules[mod_name]
    # Also drop the original eval script if cached so the helpers we
    # import below see fresh facade modules.
    for mod_name in list(sys.modules.keys()):
        if mod_name == "scripts.r53_eval_complex":
            del sys.modules[mod_name]

    # Reuse all generation + scoring machinery from the original eval
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from r53_eval_complex import (
        CORPUS, gen_stock, gen_hinted, score, _build_hints,
        BASE_SYSTEM, HINTED_PROMPT, _trim_markers,
    )
    import r53_eval_complex as orig
    from calm.llm_computer.facades.code_example_db import (
        CodeExampleDB, RetrievalHit,
    )
    from calm.llm_computer.facades.code_verifier import (
        CodeVerifierFacade, CodeHints,
    )

    # Load DB with all indices including channels
    db = CodeExampleDB.load_default()
    db.load_indices(CACHE_DIR)
    rng = random.Random(seed)

    print(f"[r53.7-ch] DB: {len(db)} examples", flush=True)
    print(f"[r53.7-ch]   combined: tfidf={db.has_tfidf()}"
          f" dense={db.has_dense()}", flush=True)
    print(f"[r53.7-ch]   code:     tfidf={db.has_channel('code', 'tfidf')}"
          f" dense={db.has_channel('code', 'dense')}", flush=True)
    if not db.has_channel('code', 'dense'):
        print("[r53.7-ch] ERROR: code dense not loaded — run "
              "scripts/r53_build_dense_channels.py first", flush=True)
        return

    # Monkey-patch _build_hints to use channel-code-hybrid retrieval
    # instead of CodeVerifierFacade's combined-hybrid default.
    def _build_hints_channel(db, rng, p, sanity_random):
        facade = CodeVerifierFacade(db=db, top_k=2)
        # Compute everything except retrieval the normal way (we want
        # intent classification, security flags, arithmetic precompute
        # exactly the same as the original)
        hints = facade.compute_hints(p.prompt)
        if sanity_random:
            n = len(db.examples)
            if n > 0:
                random_indices = rng.sample(range(n), min(2, n))
                hints.retrieved_examples = [
                    RetrievalHit(example=db.examples[i], score=0.0)
                    for i in random_indices
                ]
        else:
            # OVERRIDE: use channel-code-hybrid instead of combined
            channel_hits = db.retrieve_channel(
                p.prompt, channel="code", k=2, mode="hybrid",
                dense_m=m, dense_tok=tok)
            hints.retrieved_examples = channel_hits
        block = hints.to_system_prefix(max_example_chars=240)
        if len(block) > 2400:
            block = block[:2400] + "\n..."
        return block

    orig._build_hints = _build_hints_channel
    print("[r53.7-ch] retrieval mode: channel-code-hybrid (k=2, RRF)",
          flush=True)
    print(f"[r53.7-ch] corpus: {len(CORPUS)} complex problems", flush=True)
    print()

    rows = []
    totals = {"stock": [0, 0], "hinted": [0, 0], "sanity": [0, 0]}

    for i, p in enumerate(CORPUS):
        print(f"[{i + 1}/{len(CORPUS)}] {p.name} ({p.category})",
              flush=True)

        # STOCK
        raw_s = gen_stock(m, tok, p, max_tokens)
        s_pass, s_tot, s_diag = score(raw_s, p)
        totals["stock"][0] += s_pass
        totals["stock"][1] += s_tot

        # HINTED — channel-code-hybrid
        raw_h = gen_hinted(m, tok, p, db, rng, sanity_random=False,
                            max_tokens=max_tokens)
        h_pass, h_tot, h_diag = score(raw_h, p)
        totals["hinted"][0] += h_pass
        totals["hinted"][1] += h_tot

        # SANITY (random) — same length-control as original
        raw_r = gen_hinted(m, tok, p, db, rng, sanity_random=True,
                            max_tokens=max_tokens)
        r_pass, r_tot, r_diag = score(raw_r, p)
        totals["sanity"][0] += r_pass
        totals["sanity"][1] += r_tot

        print(f"  stock   {s_pass}/{s_tot}  {s_diag[:60]}", flush=True)
        print(f"  hinted  {h_pass}/{h_tot}  {h_diag[:60]}  (channel-code-hybrid)",
              flush=True)
        print(f"  sanity  {r_pass}/{r_tot}  {r_diag[:60]}", flush=True)
        rows.append((p.name, p.category,
                      (s_pass, s_tot), (h_pass, h_tot), (r_pass, r_tot)))

    print()
    print("=" * 80, flush=True)
    print(f"  {'name':<28} {'cat':<12} {'stock':>8} {'hinted':>8} {'sanity':>8}",
          flush=True)
    print("-" * 80, flush=True)
    for r in rows:
        name, cat, s, h, rr = r
        print(f"  {name:<28} {cat:<12} "
              f"{s[0]:>3}/{s[1]:<3} "
              f"{h[0]:>3}/{h[1]:<3} "
              f"{rr[0]:>3}/{rr[1]:<3}", flush=True)
    print("-" * 80, flush=True)
    sp, st = totals["stock"]
    hp, ht = totals["hinted"]
    rp, rt = totals["sanity"]
    print(f"  TOTAL: stock {sp}/{st}  hinted {hp}/{ht}  sanity {rp}/{rt}",
          flush=True)
    print()
    delta_real = (hp / ht if ht else 0) - (sp / st if st else 0)
    delta_sanity = (rp / rt if rt else 0) - (sp / st if st else 0)
    print(f"  Δ hinted-vs-stock : {delta_real * 100:+.1f}pp", flush=True)
    print(f"  Δ sanity-vs-stock : {delta_sanity * 100:+.1f}pp  "
          f"(control for prompt length)", flush=True)
    real_gain = delta_real - delta_sanity
    print(f"  retrieval-attributable gain: {real_gain * 100:+.1f}pp",
          flush=True)
    print()
    print("Compare to R53.0 baseline (combined-hybrid retrieval):", flush=True)
    print("  stock 25/27, hinted 21/21, sanity 23/23, gain +0.0pp",
          flush=True)


if __name__ == "__main__":
    print("Daemon-only. Use: bin/gemma-run scripts/r53_eval_complex_channel.py",
          flush=True)
    sys.exit(1)
elif "m" in globals() and "tok" in globals():
    run_eval_channel(m, tok)                                  # noqa: F821
