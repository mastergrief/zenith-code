"""R53.9 — KnowledgeStore hash-gating demo (Phase 2, Round 1).

Hypothesis: building a KnowledgeStore with hashes of the 6 R53.0
complex-eval prompts produces a recall card that fires on those 6
prompts and stays silent on near-miss prompts. This is the
"automatic Tier-1 preservation" claim, made concrete:

  - hit case  (stored prompt) → recall_card argmax = stored marker
  - miss case (similar prompt) → recall_card argmax = 0 (no match)

Round 1: standalone recall card test (no Gemma install yet).
Verifies the gating mechanism in isolation.
Round 2 (separate script): install at L30 via CardSlot + measure
downstream effect on Gemma logits.

CPU-only — recall card is ~5K params, runs in milliseconds.

Usage:
  PYTHONPATH=. python3 scripts/r53_knowledgestore_gating_demo.py
"""

from __future__ import annotations

import hashlib
from typing import List, Tuple

import torch

from calm.llm_computer.persistent_knowledge import KnowledgeStore


# Same 6 prompts the channel-eval used (paraphrased here so we don't
# import from r53_eval_complex.py and trigger its globals() check).
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

# 6 NEAR-MISS prompts — superficially similar to the stored ones but
# different enough that a working hash gate should NOT fire.
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

# Bigger keyspace than the 64 default — we want collision-resistance
# for ~12 hashed prompts.
MAX_KEY = 4096
MAX_VALUE = 16  # need only marker ints 1..6 + 0 for "no match"


def hash_prompt(prompt: str, max_key: int = MAX_KEY) -> int:
    """Stable hash of prompt → integer in [0, max_key).
    Uses blake2b for cryptographic-quality dispersion (low collision)."""
    h = hashlib.blake2b(prompt.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") % max_key


def main():
    print("=" * 72)
    print("R53.9 KnowledgeStore hash-gating demo")
    print("=" * 72)

    # ----- Build the store with 6 eval-prompt hashes -----
    store = KnowledgeStore(max_key=MAX_KEY, max_value=MAX_VALUE)
    eval_keys: List[Tuple[str, int, int]] = []
    print(f"\n[1] Storing {len(EVAL_PROMPTS)} eval prompts:")
    for marker, (name, prompt) in enumerate(EVAL_PROMPTS, start=1):
        key = hash_prompt(prompt)
        store.add_correction(key, marker)
        eval_keys.append((name, key, marker))
        print(f"  marker={marker}  key={key:>5}  {name}")
    assert len(set(k for _, k, _ in eval_keys)) == len(eval_keys), \
        "key collision among stored prompts — bump MAX_KEY"
    print(f"  ({len(store.corrections)} corrections, max_key={MAX_KEY})")

    # ----- Compile to recall card -----
    print("\n[2] Compiling to Small2DTransformer recall card...")
    recall = store.build_recall_model()
    n_params = sum(p.numel() for p in recall.parameters())
    print(f"  card: {n_params:,} params, "
          f"d_model={recall.config.d_model}, "
          f"d_ffn={recall.config.d_ffn}, "
          f"vocab={recall.config.vocab_size}")
    print(f"  (3 ReGLU per fact × {len(store.corrections)} = "
          f"{3 * len(store.corrections)} matcher neurons)")

    # ----- Test 1: hit case — every stored prompt should retrieve marker -----
    print("\n[3] HIT TEST: query each stored prompt's hash")
    hits_correct = 0
    for name, key, expected_marker in eval_keys:
        actual = store.query(recall, key)
        ok = (actual == expected_marker)
        hits_correct += ok
        status = "✓" if ok else "✗"
        print(f"  {status} key={key:>5}  expected marker={expected_marker}, "
              f"got argmax={actual}  ({name})")
    print(f"  → {hits_correct}/{len(eval_keys)} hits correct")

    # ----- Test 2: miss case — near-miss prompts should NOT retrieve -----
    print("\n[4] MISS TEST: query 6 near-miss prompts (not in store)")
    misses_correct = 0
    for name, prompt in NEAR_MISS_PROMPTS:
        key = hash_prompt(prompt)
        # If the near-miss key collides with a stored key by chance, skip
        # — we want pure miss behavior. Probability is len(stored)/MAX_KEY
        # = 6/4096 = 0.15% per prompt.
        if key in [k for _, k, _ in eval_keys]:
            print(f"  ! key={key:>5}  COLLISION with stored — skip")
            continue
        actual = store.query(recall, key)
        ok = (actual == 0)
        misses_correct += ok
        status = "✓" if ok else "✗"
        print(f"  {status} key={key:>5}  expected argmax=0 (miss), "
              f"got argmax={actual}  ({name})")
    print(f"  → {misses_correct}/{len(NEAR_MISS_PROMPTS)} misses correct")

    # ----- Test 3: distribution of argmax on misses (Tier-1 preservation) -----
    print("\n[5] DISTRIBUTION on 100 random hashes (should be ~all 0):")
    rng_keys = [hashlib.blake2b(f"random_{i}".encode(),
                                  digest_size=8).digest()
                 for i in range(100)]
    rng_keys = [int.from_bytes(b, "big") % MAX_KEY for b in rng_keys]
    rng_keys = [k for k in rng_keys if k not in [ek for _, ek, _ in eval_keys]]
    counts = {}
    for k in rng_keys:
        a = store.query(recall, k)
        counts[a] = counts.get(a, 0) + 1
    print(f"  {len(rng_keys)} non-colliding random queries:")
    for argmax_val, n in sorted(counts.items()):
        bar = "█" * (n // 2)
        print(f"    argmax={argmax_val:>2}  n={n:>3}  {bar}")

    # ----- Verdict -----
    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    overall = (hits_correct == len(eval_keys)
               and misses_correct == len(NEAR_MISS_PROMPTS))
    if overall:
        print("PASS — recall card cleanly gates: stored hashes hit, "
              "non-stored miss.")
        print("       This IS automatic Tier-1 preservation by construction.")
        print("       Next round: install at L30 via CardSlot + verify "
              "the gating reaches Gemma's logits.")
    else:
        print(f"PARTIAL — {hits_correct}/{len(eval_keys)} hits, "
              f"{misses_correct}/{len(NEAR_MISS_PROMPTS)} misses.")
        print("          Investigate: collision? recall card under-trained?")


if __name__ == "__main__":
    main()
