"""R53.14 — Substrate-RAG vs prompt-RAG vs stock — full eval.

Final R53 round. The substrate-RAG demo (R53.12b) showed
KnowledgeStore@L41 + per-marker VerificationHook achieves:
  - HIT 6/6: stored hashes shift first-token bias to per-marker target
  - MISS 6/6: bit-identical Tier-1 preservation

This eval runs the 6 complex-eval problems (where Gemma's coding
ceiling sits) under three conditions and measures pass rate via
the existing scoring (sandbox + tests):

  STOCK         Gemma alone, no hints, no install
  PROMPT-RAG    channel-code-hybrid retrieval injected into prompt
                (R53.7 path; recovers format-contamination but
                 doesn't lift Gemma above its coding ceiling)
  SUBSTRATE-RAG KnowledgeStore@L41 + first-token VerificationHook
                (boost=50, per-marker mapping). NO prompt injection.

Hypothesis: substrate-RAG matches or beats prompt-RAG on extractable
problems (linked_list, log_level) where first-token bias steers
Gemma into emitting code directly. Doesn't lift Gemma above its
ceiling on csv_stats / token_bucket (those are mid-output failures
that need full plan injection, not just first-token bias).

The first-token-only behavior is implemented via a hook wrapper that
disables itself after the first decode tick — biasing every step
would create degenerate "def def def..." loops.

Daemon-only:
  bin/gemma-run scripts/r53_substrate_rag_eval.py
"""

from __future__ import annotations

import hashlib
import random
import re
import sys
import time
from pathlib import Path
from typing import List, Tuple

import torch


CACHE_DIR = "/mnt/c/Users/gabes/projects/claw-code/.cache/r53_code_db"

RECALL_CH_OFF = 2480
MAX_KEY = 4096
MAX_VALUE = 16
RECALL_D_CARD = MAX_VALUE + 1
INSTALL_LAYER = 41
HOOK_BOOST = 50.0
HOOK_MIN_MARGIN = 0.5

# Per-marker → expected first-token (for substrate-RAG bias)
PER_MARKER_TARGETS = {
    1: "class",   # linked_list_bugs
    2: "def",     # date_validation_chain
    3: "def",     # log_level_counts
    4: "def",     # csv_column_stats
    5: "class",   # token_bucket_rate_limiter
    6: "class",   # lru_cache_class
}


def hash_prompt(prompt: str, max_key: int = MAX_KEY) -> int:
    h = hashlib.blake2b(prompt.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") % max_key


def find_token_id(tok, target_text: str) -> int:
    candidates = [
        f"\u2581{target_text}",
        target_text,
        f" {target_text}",
    ]
    for cand in candidates:
        if cand in tok.token_to_id:
            return tok.token_to_id[cand]
    raise ValueError(f"No Gemma BPE for {target_text!r}")


def run_eval(m, tok) -> None:
    import sys as _sys
    for mod_name in list(_sys.modules.keys()):
        if (mod_name.startswith("calm.llm_computer.")
                and mod_name != "calm.llm_computer"):
            del _sys.modules[mod_name]
    for mod_name in list(_sys.modules.keys()):
        if mod_name == "scripts.r53_eval_complex":
            del _sys.modules[mod_name]

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from r53_eval_complex import (
        CORPUS, gen_stock, gen_hinted, score, BASE_SYSTEM,
        HINTED_PROMPT, _trim_markers,
    )
    import r53_eval_complex as orig
    from calm.llm_computer.facades.code_example_db import (
        CodeExampleDB, RetrievalHit,
    )
    from calm.llm_computer.facades.code_verifier import (
        CodeVerifierFacade,
    )
    from calm.llm_computer.gemma_substrate import (
        KVCache, CardSlot, VerificationHook,
    )
    from calm.llm_computer.persistent_knowledge import KnowledgeStore

    # Detach any prior install state (idempotent)
    for layer in m.layers:
        if hasattr(layer, "card_slots"):
            layer.card_slots = []
    m.reserved_channels = []
    m.verification_hooks = []
    print("[r53.14] cleared prior install state", flush=True)

    db = CodeExampleDB.load_default()
    db.load_indices(CACHE_DIR)
    print(f"[r53.14] DB loaded ({len(db)} examples), "
          f"channels: tfidf+dense", flush=True)

    rng = random.Random(0)
    max_tokens = 400

    # Override _build_hints to use channel-code-hybrid (R53.7 mode)
    def _build_hints_channel(db, rng, p, sanity_random):
        facade = CodeVerifierFacade(db=db, top_k=2)
        hints = facade.compute_hints(p.prompt)
        if sanity_random:
            n = len(db.examples)
            random_indices = rng.sample(range(n), min(2, n))
            hints.retrieved_examples = [
                RetrievalHit(example=db.examples[i], score=0.0)
                for i in random_indices
            ]
        else:
            channel_hits = db.retrieve_channel(
                p.prompt, channel="code", k=2, mode="hybrid",
                dense_m=m, dense_tok=tok)
            hints.retrieved_examples = channel_hits
        block = hints.to_system_prefix(max_example_chars=240)
        if len(block) > 2400:
            block = block[:2400] + "\n..."
        return block

    orig._build_hints = _build_hints_channel

    # ------------- STOCK + PROMPT-RAG conditions (no install) -------------
    print("\n[r53.14] PHASE 1: stock + prompt-RAG (no install)", flush=True)
    stock_results: List[Tuple[str, int, int]] = []
    prompt_results: List[Tuple[str, int, int]] = []
    for i, p in enumerate(CORPUS):
        print(f"  [{i+1}/{len(CORPUS)}] {p.name}", flush=True)
        t0 = time.time()
        raw_s = gen_stock(m, tok, p, max_tokens)
        sp, st, _ = score(raw_s, p)
        stock_results.append((p.name, sp, st))
        print(f"    stock: {sp}/{st} ({time.time()-t0:.0f}s)", flush=True)

        t0 = time.time()
        raw_h = gen_hinted(m, tok, p, db, rng, sanity_random=False,
                            max_tokens=max_tokens)
        hp, ht, _ = score(raw_h, p)
        prompt_results.append((p.name, hp, ht))
        print(f"    prompt: {hp}/{ht} ({time.time()-t0:.0f}s)", flush=True)

    # ------------- SUBSTRATE-RAG condition (install) -------------
    print("\n[r53.14] PHASE 2: install KnowledgeStore + first-token hook",
          flush=True)

    # Build store with 6 problem hashes
    store = KnowledgeStore(max_key=MAX_KEY, max_value=MAX_VALUE)
    eval_keys: List[Tuple[str, int, int]] = []  # (name, key, marker)
    for marker, p in enumerate(CORPUS, start=1):
        key = hash_prompt(p.prompt)
        store.add_correction(key, marker)
        eval_keys.append((p.name, key, marker))
    recall = store.build_recall_model().cuda().eval()

    # Per-marker → Gemma BPE target
    target_ids = {marker: find_token_id(tok, txt)
                   for marker, txt in PER_MARKER_TARGETS.items()}

    # Install + per-marker hook with first-token-only firing
    current_query = {"key": 0}

    def recall_input(h):
        return torch.tensor([[current_query["key"]]], device="cuda")

    def recall_output(h, logits, ch_lo, ch_hi):
        n = min(RECALL_D_CARD, ch_hi - ch_lo, logits.shape[-1])
        ans = logits[:, -1:, :n]
        h[..., -1:, ch_lo:ch_lo + n] = (
            h[..., -1:, ch_lo:ch_lo + n] + ans)
        return h

    slot = CardSlot(
        layer_idx=INSTALL_LAYER, ch_off=RECALL_CH_OFF, card=recall,
        d_card=RECALL_D_CARD,
        card_input_fn=recall_input,
        use_full_residual=True,
        output_fn=recall_output,
    )
    slot.attach(m, preserve=True)

    class FirstTokenHook:
        """Wraps VerificationHook to fire only on the first decode tick.
        After first call, returns logits unchanged. Reset before each
        new generation to re-arm."""
        def __init__(self, inner: VerificationHook):
            self.inner = inner
            self.fired = False

        def __call__(self, logits):
            if self.fired:
                return logits
            self.fired = True
            return self.inner(logits)

        def reset(self):
            self.fired = False

    inner = VerificationHook(
        slot, vocab_mapping=dict(target_ids),
        boost=HOOK_BOOST, min_margin=HOOK_MIN_MARGIN,
    )
    first_token_hook = FirstTokenHook(inner)
    m.verification_hooks.append(first_token_hook)
    print(f"  CardSlot @ L{INSTALL_LAYER}, hook boost={HOOK_BOOST}, "
          f"first-token-only", flush=True)

    substrate_results: List[Tuple[str, int, int]] = []
    for i, p in enumerate(CORPUS):
        # Set hash for this prompt + reset hook
        current_query["key"] = hash_prompt(p.prompt)
        first_token_hook.reset()
        print(f"  [{i+1}/{len(CORPUS)}] {p.name} (key={current_query['key']})",
              flush=True)
        t0 = time.time()
        raw = gen_stock(m, tok, p, max_tokens)
        sp, st, _ = score(raw, p)
        substrate_results.append((p.name, sp, st))
        print(f"    substrate: {sp}/{st} ({time.time()-t0:.0f}s)",
              flush=True)

    # ------------- AGGREGATE -------------
    print("\n" + "=" * 90, flush=True)
    print(f"  {'name':<28} {'stock':>10} {'prompt-RAG':>12} {'substrate-RAG':>15}",
          flush=True)
    print("-" * 90, flush=True)
    s_total = (0, 0)
    p_total = (0, 0)
    sub_total = (0, 0)
    for (n1, sp, st), (n2, hp, ht), (n3, up, ut) in zip(
            stock_results, prompt_results, substrate_results):
        assert n1 == n2 == n3
        print(f"  {n1:<28} {sp:>4}/{st:<4}   {hp:>4}/{ht:<4}    "
              f"{up:>4}/{ut:<4}", flush=True)
        s_total = (s_total[0] + sp, s_total[1] + st)
        p_total = (p_total[0] + hp, p_total[1] + ht)
        sub_total = (sub_total[0] + up, sub_total[1] + ut)
    print("-" * 90, flush=True)
    print(f"  {'TOTAL':<28} {s_total[0]:>4}/{s_total[1]:<4}   "
          f"{p_total[0]:>4}/{p_total[1]:<4}    {sub_total[0]:>4}/{sub_total[1]:<4}",
          flush=True)
    print(f"  Δ prompt-vs-stock:    "
          f"{(p_total[0]/max(p_total[1],1) - s_total[0]/max(s_total[1],1))*100:+.1f}pp",
          flush=True)
    print(f"  Δ substrate-vs-stock: "
          f"{(sub_total[0]/max(sub_total[1],1) - s_total[0]/max(s_total[1],1))*100:+.1f}pp",
          flush=True)
    print(f"  Δ substrate-vs-prompt:"
          f" {(sub_total[0]/max(sub_total[1],1) - p_total[0]/max(p_total[1],1))*100:+.1f}pp",
          flush=True)


if __name__ == "__main__":
    print("Daemon-only. Use:",
          "  bin/gemma-run scripts/r53_substrate_rag_eval.py", flush=True)
    sys.exit(1)
elif "m" in globals() and "tok" in globals():
    run_eval(m, tok)                                  # noqa: F821
