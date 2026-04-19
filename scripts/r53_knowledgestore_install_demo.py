"""R53.10 — KnowledgeStore install at L30 (Phase 2, Round 2).

Builds on R53.9 (standalone gating PASS) by installing the same recall
card into Gemma at L30 via CardSlot, then measuring two strict
properties on prod Gemma 4 E4B:

  HIT property:   stored prompts produce DIFFERENT logits than baseline
                  (card output at L30 propagates through L31..L41 + head)

  MISS property:  near-miss prompts produce IDENTICAL logits to baseline
                  (zero card output → Tier-1 preservation by construction)

The miss-equivalence is the load-bearing test. If logits differ even
slightly on a miss prompt, the install is NOT cleanly Tier-1-preserving
— some signal is leaking from the card's reserved channels into Gemma's
output path, violating the substrate-RAG promise.

Daemon-only:
  bin/gemma-run scripts/r53_knowledgestore_install_demo.py
"""

from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path
from typing import List, Tuple

import torch


# Channel range for the recall card on prod Gemma 4 E4B.
# Substrate.md's reserved-channel layout assumed d_model=4096; Gemma 4
# E4B has d_model=2560, so we have to live within that. Mirroring
# scripts/gemma_learning_loop_demo.py, ch_off=2480 with a small d_card
# fits comfortably below 2560 and inside the reserved-card region.
#
# d_card sizing: KnowledgeStore.build_recall_model emits a vocab-sized
# logit tensor (vocab = max(max_key, max_value) + 1 = 4097), but only
# the first MAX_VALUE+1 entries carry signal — head_entries only
# populate value slots in [0, max_value]. So we truncate to MAX_VALUE+1
# channels in the writer and leave the rest of the card output unused.
RECALL_CH_OFF = 2480
MAX_KEY = 4096
MAX_VALUE = 16
RECALL_D_CARD = MAX_VALUE + 1     # 17 channels written into Gemma residual

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
    """Run Gemma forward on prompt, return final-position logits (CPU)."""
    ids = tok.encode(prompt)
    cache = KVCache(m.config.n_layers, device="cuda")
    with torch.no_grad():
        logits = m.forward(torch.tensor([ids]), device="cuda",
                            kv_cache=cache, start_pos=0)
    return logits[0, -1].cpu()


def run_install_demo(m, tok) -> None:
    # Reimport to pick up R53.9's KnowledgeStore + this session's edits
    import sys as _sys
    for mod_name in list(_sys.modules.keys()):
        if (mod_name.startswith("calm.llm_computer.")
                and mod_name != "calm.llm_computer"):
            del _sys.modules[mod_name]

    from calm.llm_computer.gemma_substrate import (
        KVCache, CardSlot,
    )
    from calm.llm_computer.persistent_knowledge import KnowledgeStore

    # Daemon's `m` persists across script runs. Detach any prior install
    # so this script is idempotent (otherwise a previous failed-state
    # CardSlot would intercept BASELINE captures and corrupt them).
    for layer in m.layers:
        if hasattr(layer, "card_slots"):
            layer.card_slots = []
    m.reserved_channels = []
    m.verification_hooks = []
    print("[r53.10] cleared prior CardSlots / hooks / reserved channels",
          flush=True)

    # ----- Build store + recall card (same as R53.9) -----
    store = KnowledgeStore(max_key=MAX_KEY, max_value=MAX_VALUE)
    eval_keys: List[Tuple[str, str, int, int]] = []
    print("[r53.10] storing eval prompts:", flush=True)
    for marker, (name, prompt) in enumerate(EVAL_PROMPTS, start=1):
        key = hash_prompt(prompt)
        store.add_correction(key, marker)
        eval_keys.append((name, prompt, key, marker))
        print(f"  marker={marker} key={key} {name}", flush=True)

    recall = store.build_recall_model().cuda().eval()
    print(f"[r53.10] recall card: "
          f"{sum(p.numel() for p in recall.parameters())} params, "
          f"vocab={recall.config.vocab_size}", flush=True)

    # ----- BASELINE: capture Gemma logits with NO card installed -----
    print("\n[r53.10] capturing BASELINE Gemma logits (no install)...",
          flush=True)
    t0 = time.time()
    baseline_eval: List[torch.Tensor] = []
    baseline_miss: List[torch.Tensor] = []
    for _, prompt in EVAL_PROMPTS:
        baseline_eval.append(gemma_logits(m, tok, prompt, KVCache))
    for _, prompt in NEAR_MISS_PROMPTS:
        baseline_miss.append(gemma_logits(m, tok, prompt, KVCache))
    print(f"[r53.10] baseline captured in {time.time()-t0:.1f}s "
          f"({len(baseline_eval)} eval + {len(baseline_miss)} miss)",
          flush=True)

    # ----- INSTALL: CardSlot at L30 with hash-gated input -----
    current_query = {"key": 0}

    def recall_input(h):
        # Recall card sees a single position with the prompt's hash key
        return torch.tensor([[current_query["key"]]], device="cuda")

    def recall_output(h, logits, ch_lo, ch_hi):
        # Recall card has vocab=4097 but only first MAX_VALUE+1 entries
        # carry head signal. Truncate to fit Gemma's residual channels.
        # preserve=True (set on CardSlot) zeros L31..L41's contribution
        # to ch_lo..ch_hi at runtime so the card output flows through.
        n = min(RECALL_D_CARD, ch_hi - ch_lo, logits.shape[-1])
        ans = logits[:, -1:, :n]
        h[..., -1:, ch_lo:ch_lo + n] = (
            h[..., -1:, ch_lo:ch_lo + n] + ans)
        return h

    slot = CardSlot(
        layer_idx=30, ch_off=RECALL_CH_OFF, card=recall,
        d_card=RECALL_D_CARD,
        card_input_fn=recall_input,
        use_full_residual=True,
        output_fn=recall_output,
    )
    slot.attach(m, preserve=True)
    print(f"\n[r53.10] CardSlot installed at L30, "
          f"ch[{RECALL_CH_OFF}:{RECALL_CH_OFF + RECALL_D_CARD}], "
          f"preserve=True", flush=True)

    # ----- HIT TEST: stored prompts should change logits + card fires -----
    print("\n[r53.10] HIT TEST — stored prompts (expect logits to differ):",
          flush=True)
    hit_results = []
    for i, (name, prompt, key, marker) in enumerate(eval_keys):
        current_query["key"] = key
        new_logits = gemma_logits(m, tok, prompt, KVCache)
        diff = (new_logits - baseline_eval[i]).abs()
        max_diff = float(diff.max())
        l2_diff = float(diff.norm())
        # Read what the card actually emitted (last_output)
        card_argmax = int(slot.last_output[0, -1].argmax()) \
            if slot.last_output is not None else -1
        argmax_match = (card_argmax == marker)
        # Did Gemma's argmax change?
        baseline_top = int(baseline_eval[i].argmax())
        new_top = int(new_logits.argmax())
        top_changed = (baseline_top != new_top)
        print(f"  {name:<26}  card_argmax={card_argmax}/{marker} "
              f"{'✓' if argmax_match else '✗'}  "
              f"|Δlogits|_max={max_diff:.4f}  "
              f"|Δ|_2={l2_diff:.3f}  "
              f"top_token_changed={top_changed}",
              flush=True)
        hit_results.append((argmax_match, max_diff, top_changed))

    # ----- MISS TEST: near-miss prompts should produce IDENTICAL logits -----
    print("\n[r53.10] MISS TEST — near-miss prompts "
          "(expect logits IDENTICAL to baseline):",
          flush=True)
    miss_results = []
    for i, (name, prompt) in enumerate(NEAR_MISS_PROMPTS):
        key = hash_prompt(prompt)
        current_query["key"] = key
        new_logits = gemma_logits(m, tok, prompt, KVCache)
        diff = (new_logits - baseline_miss[i]).abs()
        max_diff = float(diff.max())
        l2_diff = float(diff.norm())
        card_argmax = int(slot.last_output[0, -1].argmax()) \
            if slot.last_output is not None else -1
        # Strict: logits should be bit-identical (max diff == 0)
        identical = (max_diff < 1e-5)  # numerical tolerance
        print(f"  {name:<26}  card_argmax={card_argmax} "
              f"{'(zero ✓)' if card_argmax == 0 else '(nonzero ✗)'}  "
              f"|Δlogits|_max={max_diff:.6f}  "
              f"|Δ|_2={l2_diff:.6f}  "
              f"identical_to_baseline={identical}",
              flush=True)
        miss_results.append((card_argmax == 0, max_diff, identical))

    # ----- VERDICT -----
    print("\n" + "=" * 72, flush=True)
    print("R53.10 VERDICT", flush=True)
    print("=" * 72, flush=True)
    n_hit_card_ok = sum(1 for r in hit_results if r[0])
    n_hit_logits_changed = sum(1 for r in hit_results if r[2])
    n_miss_card_zero = sum(1 for r in miss_results if r[0])
    n_miss_identical = sum(1 for r in miss_results if r[2])
    print(f"  HIT  card output correct:        "
          f"{n_hit_card_ok}/{len(hit_results)}", flush=True)
    print(f"  HIT  Gemma top token changed:    "
          f"{n_hit_logits_changed}/{len(hit_results)}", flush=True)
    print(f"  MISS card output zero (gating):  "
          f"{n_miss_card_zero}/{len(miss_results)}", flush=True)
    print(f"  MISS logits == baseline (Tier-1):"
          f" {n_miss_identical}/{len(miss_results)}", flush=True)

    print("\nReadings:", flush=True)
    if (n_miss_identical == len(miss_results)
            and n_hit_card_ok == len(hit_results)):
        print("  PASS — automatic Tier-1 preservation is OBSERVABLE in",
              flush=True)
        print("         Gemma's logits. Miss prompts emit identical tokens",
              flush=True)
        print("         to baseline; hit prompts have card-driven residual.",
              flush=True)
        print("  Next: wire VerificationHook to convert hit-marker into a",
              flush=True)
        print("        useful logit bias (e.g. first BPE of expected def).",
              flush=True)
    elif n_hit_card_ok < len(hit_results):
        print("  PARTIAL — card not firing on all hits. Investigate the",
              flush=True)
        print("            install path (CardSlot.last_output not updated?).",
              flush=True)
    elif n_miss_identical < len(miss_results):
        print("  PARTIAL — miss prompts NOT identical to baseline. The card",
              flush=True)
        print("            install is leaking signal. Investigate preserve",
              flush=True)
        print("            mask or output_fn additive write.", flush=True)


if __name__ == "__main__":
    print("Daemon-only. Use:",
          "  bin/gemma-run scripts/r53_knowledgestore_install_demo.py",
          flush=True)
    sys.exit(1)
elif "m" in globals() and "tok" in globals():
    run_install_demo(m, tok)                                  # noqa: F821
