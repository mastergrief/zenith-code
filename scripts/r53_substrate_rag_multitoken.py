"""R53.16 — Substrate-RAG with multi-token step-through bias.

R53.14 showed first-token bias regresses (-9.3pp). Diagnosis: forcing
"▁def" or "▁class" at position 0 disrupts Gemma's natural prelude on
problems it solves cleanly. R53.15 confirmed Gemma is uniformly
confident at first-token (margins 6.8-9.2), so confidence-gating
just suppresses the bias entirely — preserves stock at best.

This round adapts R46.2's MultiStepReasoningFacade pattern: instead
of biasing position 0 toward one token, bias EACH decode step toward
the next token of a hand-crafted signature template per problem.

Templates per problem:
  linked_list_bugs        → "class LinkedList:"
  date_validation_chain   → "def validate_date(s):"
  log_level_counts        → "def count_levels(log_text):"
  csv_column_stats        → "def column_stats(csv_text):"
  token_bucket_rate_limit → "class TokenBucket:"
  lru_cache_class         → "class LRUCache:"

Hypothesis: full-signature guidance steers Gemma into emitting
correct code structure, recovering csv/token_bucket where first-token
couldn't help and where Gemma's mid-output failures are.

Daemon-only:
  bin/gemma-run scripts/r53_substrate_rag_multitoken.py
"""

from __future__ import annotations

import hashlib
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

# Per-problem signature templates. The hook biases Gemma to emit
# these as the FIRST N tokens of its response. After the template
# tokens are biased, hook stops firing and Gemma continues naturally.
PER_MARKER_TEMPLATES = {
    1: "class LinkedList:",
    2: "def validate_date(s):",
    3: "def count_levels(log_text):",
    4: "def column_stats(csv_text):",
    5: "class TokenBucket:",
    6: "class LRUCache:",
}


def hash_prompt(prompt: str, max_key: int = MAX_KEY) -> int:
    h = hashlib.blake2b(prompt.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") % max_key


def encode_template(tok, template: str) -> List[int]:
    """Tokenize a code template into Gemma BPE ids, dropping BOS.

    The Gemma tokenizer prepends BOS for non-empty input. We don't
    want BOS in the bias sequence — strip it. Result is the actual
    decode-step sequence we want to bias toward.
    """
    ids = tok.encode(template)
    bos_id = tok.token_to_id.get("<bos>")
    if ids and ids[0] == bos_id:
        ids = ids[1:]
    return ids


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
    print("[r53.16] cleared prior install state", flush=True)

    from calm.llm_computer.eval_defaults import EVAL_MAX_TOKENS
    max_tokens = EVAL_MAX_TOKENS  # was 400 — bumped to centralized 16K ceiling
                                   # per workflow.md §"MAX_TOKENS budget discipline"

    # Encode all templates upfront, print for verification
    template_ids: dict[int, List[int]] = {}
    print("[r53.16] templates:", flush=True)
    for marker, tpl in PER_MARKER_TEMPLATES.items():
        ids = encode_template(tok, tpl)
        template_ids[marker] = ids
        token_strs = [tok.id_to_token.get(i, '?') for i in ids]
        print(f"  marker={marker} {tpl!r:<35} → {ids} ({token_strs})",
              flush=True)

    # Build store + recall card (markers 1..6)
    store = KnowledgeStore(max_key=MAX_KEY, max_value=MAX_VALUE)
    for marker, p in enumerate(CORPUS, start=1):
        store.add_correction(hash_prompt(p.prompt), marker)
    recall = store.build_recall_model().cuda().eval()

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

    class StepThroughHook:
        """Multi-token bias that walks through a per-problem template.

        At each decode step where step < len(template):
          - Read the recall card's argmax (marker)
          - Look up template[step] for that marker
          - Add boost to that token's logit
          - Increment step

        After step >= len(template), hook stops firing — Gemma continues
        natively. Reset before each new generation.

        On miss (card argmax = 0, no marker for 0 in template_ids),
        hook silently no-ops — same as VerificationHook's min_margin gate.
        """
        def __init__(self, card_slot, template_ids: dict, boost: float):
            self.card_slot = card_slot
            self.template_ids = template_ids
            self.boost = boost
            self.step = 0
            self.last_marker = None

        def __call__(self, logits):
            # Read card output
            out = getattr(self.card_slot, "last_output", None)
            if out is None:
                return logits
            last = out[0, -1].float()
            peak = last.max().item()
            med = last.median().item()
            if (peak - med) < HOOK_MIN_MARGIN:
                # Card silent — pass through
                return logits
            marker = int(last.argmax())
            self.last_marker = marker
            if marker not in self.template_ids:
                return logits
            tpl = self.template_ids[marker]
            if self.step >= len(tpl):
                return logits  # template exhausted
            target_tok = tpl[self.step]
            self.step += 1
            # Add boost in-place
            if logits.dim() == 3:
                logits[0, -1, target_tok] = (
                    logits[0, -1, target_tok] + self.boost)
            else:
                logits[0, target_tok] = (
                    logits[0, target_tok] + self.boost)
            return logits

        def reset(self):
            self.step = 0
            self.last_marker = None

    hook = StepThroughHook(slot, template_ids, boost=HOOK_BOOST)
    m.verification_hooks = [hook]
    print(f"\n[r53.16] StepThroughHook installed (boost={HOOK_BOOST})",
          flush=True)

    # ------------- STOCK BASELINE (no install) -------------
    # Detach for stock pass — no card, no hook
    print("\n[r53.16] PHASE 1: stock baseline (no install)", flush=True)
    saved_slots = []
    for layer in m.layers:
        if hasattr(layer, "card_slots"):
            saved_slots.append((layer, list(layer.card_slots)))
            layer.card_slots = []
    saved_hooks = m.verification_hooks
    m.verification_hooks = []

    stock_results: List[Tuple[str, int, int]] = []
    for i, p in enumerate(CORPUS):
        t0 = time.time()
        raw = gen_stock(m, tok, p, max_tokens)
        sp, st, _ = score(raw, p)
        stock_results.append((p.name, sp, st))
        print(f"  [{i+1}/6] {p.name:<28} stock {sp}/{st} "
              f"({time.time()-t0:.0f}s)", flush=True)

    # Re-attach
    for layer, slots in saved_slots:
        layer.card_slots = slots
    m.verification_hooks = saved_hooks

    # ------------- SUBSTRATE STEP-THROUGH -------------
    print("\n[r53.16] PHASE 2: substrate-RAG with multi-token step-through",
          flush=True)
    sub_results: List[Tuple[str, int, int, int]] = []
    for i, p in enumerate(CORPUS):
        current_query["key"] = hash_prompt(p.prompt)
        hook.reset()
        t0 = time.time()
        raw = gen_stock(m, tok, p, max_tokens)
        sp, st, _ = score(raw, p)
        sub_results.append((p.name, sp, st, hook.step))
        print(f"  [{i+1}/6] {p.name:<28} sub {sp}/{st}  "
              f"steps_biased={hook.step}/{len(template_ids[i+1])}  "
              f"({time.time()-t0:.0f}s)", flush=True)

    # ------------- AGGREGATE -------------
    print("\n" + "=" * 90, flush=True)
    print(f"  {'name':<28} {'stock':>10}  {'substrate (multi-token)':>25}",
          flush=True)
    print("-" * 90, flush=True)
    s_total = (0, 0)
    sub_total = (0, 0)
    for (n1, sp, st), (n2, up, ut, steps) in zip(stock_results, sub_results):
        print(f"  {n1:<28} {sp:>4}/{st:<4}    "
              f"{up:>4}/{ut:<4}  (biased {steps} steps)", flush=True)
        s_total = (s_total[0] + sp, s_total[1] + st)
        sub_total = (sub_total[0] + up, sub_total[1] + ut)
    print("-" * 90, flush=True)
    print(f"  {'TOTAL':<28} {s_total[0]:>4}/{s_total[1]:<4}    "
          f"{sub_total[0]:>4}/{sub_total[1]:<4}", flush=True)
    if s_total[1] and sub_total[1]:
        delta = (sub_total[0]/sub_total[1] - s_total[0]/s_total[1]) * 100
        print(f"  Δ substrate-vs-stock: {delta:+.1f}pp", flush=True)


if __name__ == "__main__":
    print("Daemon-only. Use:",
          "  bin/gemma-run scripts/r53_substrate_rag_multitoken.py",
          flush=True)
    sys.exit(1)
elif "m" in globals() and "tok" in globals():
    run_eval(m, tok)                                  # noqa: F821
