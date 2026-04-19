"""R53.15 — Substrate-RAG with confidence-aware hook.

R53.14 showed first-token bias regresses -9.3pp because forcing
"▁def" / "▁class" disrupts Gemma's natural prelude on problems it
solves cleanly (log_level 6/6→0/0, lru_cache 9/9→0/0).

Hypothesis: gating the hook on Gemma's natural top-logit margin
fixes this. If Gemma is CONFIDENT (margin >> threshold), don't
fire — Gemma knows what it's doing. If Gemma is UNCERTAIN (small
margin), fire — bias might help.

Mechanism: ConfidenceAwareHook reads Gemma's top-logit and second-
top-logit BEFORE biasing. Skip bias if (top - second) > threshold.

This is a Tier-1-preservation refinement on top of hash gating.
The substrate now preserves Gemma's behavior in TWO ways:
  1. Hash gating (R53.12b): card silent on miss → no logit change
  2. Confidence gating (this round): hook silent when Gemma confident
     → no first-token override

Daemon-only:
  bin/gemma-run scripts/r53_substrate_rag_confidence.py
"""

from __future__ import annotations

import hashlib
import random
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

# Confidence threshold: Gemma top - second-top must be BELOW this for
# the hook to fire. Larger value = hook fires more often (less Tier-1
# preservation). Smaller value = hook fires less often.
# Pick a few thresholds to scan.
CONFIDENCE_THRESHOLDS = [1.0, 3.0, 5.0]

PER_MARKER_TARGETS = {
    1: "class", 2: "def", 3: "def",
    4: "def",   5: "class", 6: "class",
}


def hash_prompt(prompt: str, max_key: int = MAX_KEY) -> int:
    h = hashlib.blake2b(prompt.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") % max_key


def find_token_id(tok, target_text: str) -> int:
    candidates = [
        f"\u2581{target_text}", target_text, f" {target_text}",
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
    from r53_eval_complex import CORPUS, gen_stock, score
    from calm.llm_computer.gemma_substrate import (
        KVCache, CardSlot, VerificationHook,
    )
    from calm.llm_computer.persistent_knowledge import KnowledgeStore

    # Idempotent cleanup
    for layer in m.layers:
        if hasattr(layer, "card_slots"):
            layer.card_slots = []
    m.reserved_channels = []
    m.verification_hooks = []
    print("[r53.15] cleared prior install state", flush=True)

    max_tokens = 16384

    # Build store + recall card
    store = KnowledgeStore(max_key=MAX_KEY, max_value=MAX_VALUE)
    for marker, p in enumerate(CORPUS, start=1):
        store.add_correction(hash_prompt(p.prompt), marker)
    recall = store.build_recall_model().cuda().eval()

    target_ids = {marker: find_token_id(tok, txt)
                   for marker, txt in PER_MARKER_TARGETS.items()}

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

    class ConfidenceAwareFirstTokenHook:
        """Fires only on the first decode step AND only when Gemma's
        natural top-logit margin is below `confidence_threshold` (i.e.,
        Gemma is uncertain). After firing once or first-call, marks
        itself dormant for the rest of the generation."""

        def __init__(self, inner: VerificationHook, threshold: float):
            self.inner = inner
            self.threshold = threshold
            self.consumed = False
            self.last_decision = None  # for diagnostic

        def __call__(self, logits):
            if self.consumed:
                return logits
            self.consumed = True
            # Compute Gemma's natural top-logit margin BEFORE bias
            last = logits[0, -1].float() if logits.dim() == 3 else logits[0].float()
            top2 = torch.topk(last, k=2)
            margin = (top2.values[0] - top2.values[1]).item()
            if margin > self.threshold:
                # Gemma is confident — don't disrupt
                self.last_decision = ("skip-confident", margin)
                return logits
            # Gemma uncertain → bias
            self.last_decision = ("fire-uncertain", margin)
            return self.inner(logits)

        def reset(self):
            self.consumed = False
            self.last_decision = None

    inner = VerificationHook(
        slot, vocab_mapping=dict(target_ids),
        boost=HOOK_BOOST, min_margin=HOOK_MIN_MARGIN,
    )

    # Run a stock baseline to know the truth
    print("\n[r53.15] PHASE 1: baseline stock pass rates", flush=True)
    stock_results: List[Tuple[str, int, int]] = []
    for i, p in enumerate(CORPUS):
        # Make sure no hook is active
        m.verification_hooks = []
        t0 = time.time()
        raw = gen_stock(m, tok, p, max_tokens)
        sp, st, _ = score(raw, p)
        stock_results.append((p.name, sp, st))
        print(f"  [{i+1}/6] {p.name:<28} stock {sp}/{st} ({time.time()-t0:.0f}s)",
              flush=True)
    s_total = (sum(r[1] for r in stock_results),
               sum(r[2] for r in stock_results))

    # Sweep confidence thresholds
    all_results = []  # (threshold, [per-problem (name, pass, total, decision)])
    for thr in CONFIDENCE_THRESHOLDS:
        print(f"\n[r53.15] PHASE 2.{thr}: confidence-aware hook (threshold={thr})",
              flush=True)
        hook = ConfidenceAwareFirstTokenHook(inner, threshold=thr)
        m.verification_hooks = [hook]

        results: List[Tuple[str, int, int, str]] = []
        for i, p in enumerate(CORPUS):
            current_query["key"] = hash_prompt(p.prompt)
            hook.reset()
            t0 = time.time()
            raw = gen_stock(m, tok, p, max_tokens)
            sp, st, _ = score(raw, p)
            decision = (f"{hook.last_decision[0]} (margin={hook.last_decision[1]:.2f})"
                         if hook.last_decision else "n/a")
            results.append((p.name, sp, st, decision))
            print(f"  [{i+1}/6] {p.name:<28} sub {sp}/{st}  "
                  f"[{decision}]  ({time.time()-t0:.0f}s)", flush=True)
        all_results.append((thr, results))

    # ------------- AGGREGATE -------------
    print("\n" + "=" * 100, flush=True)
    thr_headers = "  ".join(f"thr={t:.0f}".rjust(13)
                              for t in CONFIDENCE_THRESHOLDS)
    print(f"  {'name':<28} {'stock':>9}  {thr_headers}", flush=True)
    print("-" * 100, flush=True)
    for i, (name, sp, st) in enumerate(stock_results):
        row = f"  {name:<28} {sp:>3}/{st:<4}"
        for thr, results in all_results:
            up = results[i][1]
            ut = results[i][2]
            row += f"  {up:>3}/{ut:<4}      "
        print(row, flush=True)
    print("-" * 100, flush=True)
    total_row = f"  {'TOTAL':<28} {s_total[0]:>3}/{s_total[1]:<4}"
    for thr, results in all_results:
        tot = (sum(r[1] for r in results), sum(r[2] for r in results))
        delta = ((tot[0]/max(tot[1],1) - s_total[0]/max(s_total[1],1)) * 100
                  if s_total[1] else 0.0)
        total_row += f"  {tot[0]:>3}/{tot[1]:<4} ({delta:+.1f}pp)"
    print(total_row, flush=True)


if __name__ == "__main__":
    print("Daemon-only. Use:",
          "  bin/gemma-run scripts/r53_substrate_rag_confidence.py",
          flush=True)
    sys.exit(1)
elif "m" in globals() and "tok" in globals():
    run_eval(m, tok)                                  # noqa: F821
