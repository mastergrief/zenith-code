"""R53.12 — Wire VerificationHook on L41 install (Phase 2, Round 4).

Builds on R53.11 (L41 install + strict Tier-1 PASS) by adding a
VerificationHook that converts the HIT marker into a directed bias
on Gemma's vocab logits.

Hypothesis: with VerificationHook active and per-marker mapping to a
single Gemma token (" def"), stored prompts emit " def" as their top
token (driven by hook bias), while near-miss prompts emit baseline
top tokens (hook silent because card argmax=0 + min_margin gate).

This closes the substrate-RAG loop: hash gating + verified-token bias
on hit + native pass-through on miss = the structural answer to
R53.0's +0.0pp prompt-RAG ceiling.

For first round we map ALL markers → same Gemma token (" def") just
to demonstrate the hook mechanics. R53.13 would extend to per-marker
mapping (e.g. marker 1,5,6 → " class" for class-typed problems).

Daemon-only:
  bin/gemma-run scripts/r53_knowledgestore_hook_demo.py
"""

from __future__ import annotations

import hashlib
import sys
import time
from typing import List, Tuple

import torch


RECALL_CH_OFF = 2480
MAX_KEY = 4096
MAX_VALUE = 16
RECALL_D_CARD = MAX_VALUE + 1
INSTALL_LAYER = 41
HOOK_BOOST = 20.0
HOOK_MIN_MARGIN = 0.5


EVAL_PROMPTS = [
    ("linked_list_bugs",
     "The following Python LinkedList implementation has three bugs that "
     "make one or more methods incorrect. Fix ALL bugs."),
    ("date_validation_chain",
     "Write a function validate_date(s) that takes a string in format "
     "'YYYY-MM-DD' and returns True if it's a valid calendar date."),
    ("log_level_counts",
     "Parse a log file and count occurrences of each log level "
     "(DEBUG, INFO, WARNING, ERROR, CRITICAL). Return a dict mapping "
     "level to count."),
    ("csv_column_stats",
     "Parse a CSV string and return a dict mapping each numeric "
     "column name to a sub-dict of {'mean', 'stdev', 'min', 'max'}."),
    ("token_bucket_rate_limiter",
     "Implement a token bucket rate limiter as a class with consume(n) "
     "method. Constructor takes capacity and refill_rate."),
    ("lru_cache_class",
     "Implement an LRU cache class with get(key) and put(key, value) "
     "methods. Constructor takes capacity. Evict least-recently-used "
     "when full. Both ops O(1)."),
]

NEAR_MISS_PROMPTS = [
    ("similar_doubly_linked",
     "Write a Python DoublyLinkedList class with append(v), remove(v), "
     "and to_list() methods."),
    ("similar_time_validation",
     "Write a function validate_time(s) for HH:MM:SS strings. Reject "
     "invalid hours/minutes/seconds."),
    ("similar_log_filter",
     "Filter a log file to lines matching a given log level. Return list."),
    ("similar_csv_parse",
     "Parse a CSV file into a list of dicts using the first row as keys."),
    ("similar_leaky_bucket",
     "Implement a leaky bucket rate limiter as a class with allow() "
     "method."),
    ("similar_lfu_cache",
     "Implement an LFU (least-frequently-used) cache class with get and "
     "put methods."),
]


def hash_prompt(prompt: str, max_key: int = MAX_KEY) -> int:
    h = hashlib.blake2b(prompt.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") % max_key


def gemma_logits(m, tok, prompt: str, KVCache) -> torch.Tensor:
    ids = tok.encode(prompt)
    cache = KVCache(m.config.n_layers, device="cuda")
    with torch.no_grad():
        logits = m.forward(torch.tensor([ids]), device="cuda",
                            kv_cache=cache, start_pos=0)
    return logits[0, -1].cpu()


def find_token_id(tok, target_text: str) -> int:
    """Find Gemma BPE id for the target string. Skip ALL special
    tokens (anything bracketed like <bos>, <eos>, <pad>, <unk>) and
    return the first content BPE token in the encoding."""
    for variant in (target_text, target_text.strip()):
        ids = tok.encode(variant)
        for tid in ids:
            tok_str = tok.id_to_token.get(tid, "")
            stripped = tok_str.strip()
            if not stripped:
                continue
            # Skip special tokens (Gemma uses <bos>, <eos>, etc.)
            if stripped.startswith("<") and stripped.endswith(">"):
                continue
            return tid
    raise ValueError(f"Could not find content token for {target_text!r}")


def run_hook_demo(m, tok) -> None:
    import sys as _sys
    for mod_name in list(_sys.modules.keys()):
        if (mod_name.startswith("calm.llm_computer.")
                and mod_name != "calm.llm_computer"):
            del _sys.modules[mod_name]

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
    print("[r53.12] cleared prior CardSlots / hooks", flush=True)

    # ----- Find target Gemma BPE for " def" -----
    target_text = " def"
    target_token_id = find_token_id(tok, target_text)
    target_token_str = tok.id_to_token.get(target_token_id, "?")
    print(f"[r53.12] target Gemma BPE: id={target_token_id} "
          f"str={target_token_str!r}", flush=True)

    # ----- Build store (same as R53.9/10/11) -----
    store = KnowledgeStore(max_key=MAX_KEY, max_value=MAX_VALUE)
    eval_keys: List[Tuple[str, str, int, int]] = []
    print("[r53.12] storing eval prompts:", flush=True)
    for marker, (name, prompt) in enumerate(EVAL_PROMPTS, start=1):
        key = hash_prompt(prompt)
        store.add_correction(key, marker)
        eval_keys.append((name, prompt, key, marker))
        print(f"  marker={marker} key={key} {name}", flush=True)

    recall = store.build_recall_model().cuda().eval()

    # ----- BASELINE -----
    print("\n[r53.12] capturing BASELINE Gemma logits...", flush=True)
    t0 = time.time()
    baseline_eval, baseline_miss = [], []
    for _, prompt in EVAL_PROMPTS:
        baseline_eval.append(gemma_logits(m, tok, prompt, KVCache))
    for _, prompt in NEAR_MISS_PROMPTS:
        baseline_miss.append(gemma_logits(m, tok, prompt, KVCache))
    print(f"[r53.12] baseline captured in {time.time()-t0:.1f}s", flush=True)

    # ----- INSTALL: CardSlot at L41 + VerificationHook -----
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

    # All markers map to the same target Gemma token; on miss the card
    # outputs zeros so VerificationHook's min_margin gate suppresses
    # the boost (peak-median < min_margin → no fire).
    vocab_mapping = {marker: target_token_id
                      for marker in range(1, MAX_VALUE + 1)}
    hook = VerificationHook(
        slot, vocab_mapping=vocab_mapping,
        boost=HOOK_BOOST, min_margin=HOOK_MIN_MARGIN,
    )
    m.verification_hooks.append(hook)
    print(f"\n[r53.12] CardSlot @ L{INSTALL_LAYER} + VerificationHook",
          flush=True)
    print(f"           markers 1..{MAX_VALUE} → Gemma token "
          f"{target_token_id} ({target_token_str!r})", flush=True)
    print(f"           boost={HOOK_BOOST}, min_margin={HOOK_MIN_MARGIN}",
          flush=True)

    # ----- HIT TEST -----
    print("\n[r53.12] HIT TEST — expect Gemma top token shifts to "
          f"{target_token_str!r}:", flush=True)
    hit_results = []
    for i, (name, prompt, key, marker) in enumerate(eval_keys):
        current_query["key"] = key
        new_logits = gemma_logits(m, tok, prompt, KVCache)
        baseline_top = int(baseline_eval[i].argmax())
        new_top = int(new_logits.argmax())
        hit_target = (new_top == target_token_id)
        baseline_str = tok.id_to_token.get(baseline_top, "?")
        new_str = tok.id_to_token.get(new_top, "?")
        print(f"  {name:<26}  baseline={baseline_str!r:>14}  "
              f"new={new_str!r:>14}  "
              f"hit_target={'✓' if hit_target else '✗'}", flush=True)
        hit_results.append(hit_target)

    # ----- MISS TEST -----
    print("\n[r53.12] MISS TEST — expect Gemma top token UNCHANGED "
          "(hook silent):", flush=True)
    miss_results = []
    for i, (name, prompt) in enumerate(NEAR_MISS_PROMPTS):
        key = hash_prompt(prompt)
        current_query["key"] = key
        new_logits = gemma_logits(m, tok, prompt, KVCache)
        baseline_top = int(baseline_miss[i].argmax())
        new_top = int(new_logits.argmax())
        preserved = (baseline_top == new_top)
        baseline_str = tok.id_to_token.get(baseline_top, "?")
        new_str = tok.id_to_token.get(new_top, "?")
        diff_max = float((new_logits - baseline_miss[i]).abs().max())
        print(f"  {name:<26}  baseline={baseline_str!r:>14}  "
              f"new={new_str!r:>14}  "
              f"preserved={'✓' if preserved else '✗'}  "
              f"|Δ|_max={diff_max:.6f}", flush=True)
        miss_results.append(preserved)

    # ----- VERDICT -----
    print("\n" + "=" * 72, flush=True)
    print("R53.12 VERDICT", flush=True)
    print("=" * 72, flush=True)
    n_hit = sum(hit_results)
    n_miss = sum(miss_results)
    print(f"  HIT  top token shifted to target ({target_token_str!r}): "
          f"{n_hit}/{len(hit_results)}", flush=True)
    print(f"  MISS top token preserved (hook stayed silent):     "
          f"{n_miss}/{len(miss_results)}", flush=True)
    if n_hit == len(hit_results) and n_miss == len(miss_results):
        print("\n  PASS — hash-gated VerificationHook closes the loop.",
              flush=True)
        print("         Substrate-RAG: hit → directed bias, miss → native.",
              flush=True)
    elif n_miss == len(miss_results):
        print(f"\n  PARTIAL — gate works (misses preserved) but not all hits",
              flush=True)
        print("            converted to target. Likely boost too low",
              flush=True)
        print("            relative to Gemma's natural top-logit margin.",
              flush=True)
    elif n_hit == len(hit_results):
        print(f"\n  PARTIAL — hook fires on hits but ALSO leaks on misses.",
              flush=True)
        print("            min_margin too low — recall card's noise floor",
              flush=True)
        print("            exceeded the gate.", flush=True)
    else:
        print(f"\n  FAIL — investigate hook config + min_margin tuning.",
              flush=True)


if __name__ == "__main__":
    print("Daemon-only. Use:",
          "  bin/gemma-run scripts/r53_knowledgestore_hook_demo.py",
          flush=True)
    sys.exit(1)
elif "m" in globals() and "tok" in globals():
    run_hook_demo(m, tok)                                  # noqa: F821
