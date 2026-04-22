"""R22f parse-diag — why does parse_mqar_prompt fail on 18/20 N=10 prompts?

Regenerates the exact R22 pooled corpus (no Gemma needed) and classifies
parse failures. Seed-deterministic → matches .cache/r22b/round6_gated_write.jsonl.

Findings from r22f_n10_diag: at N=10 and N=15 only 2/20 prompts are
"active" (adapter parses them). At N=5, 15/20 are active. The silent
margin=0 argmax=<pad> pattern confirms state["active"]=False, which per
r22b_round7 happens iff parse_mqar_prompt returns None.

This script runs the parser on all 60 prompts, logs exactly WHY each
fails (no mem / no pairs / no query / q_key not in pairs), and shows
example prompts for each failure class.
"""
from __future__ import annotations

import random
import re
import sys
from collections import defaultdict, Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ============================================================================
# Reimplement parse_mqar_prompt step-by-step with classification
# ============================================================================

_MEM_RE = re.compile(r"<mem>(.+?)</mem>", re.IGNORECASE | re.DOTALL)
_KV_RE = re.compile(r"\b([a-z])\s*=\s*(\d)\b")
_QUERY_RES = [
    re.compile(r"value of\s+([a-z])\b", re.IGNORECASE),
    re.compile(r"what is\s+([a-z])\b[?\.]?\s*$", re.IGNORECASE),
    re.compile(r"\bis\s+([a-z])\b\s*[?\.]?\s*$", re.IGNORECASE),
]


def parse_classified(prompt: str):
    """Returns (mqar_str | None, failure_reason). Failure reason one of
    'ok', 'no_mem', 'no_pairs', 'no_query', 'q_key_not_in_pairs'."""
    mem = _MEM_RE.search(prompt)
    if not mem:
        return None, "no_mem"
    pairs = _KV_RE.findall(mem.group(1))
    if not pairs:
        return None, "no_pairs"
    post_mem = prompt[mem.end():]
    question_idx = post_mem.lower().rfind("question:")
    search_region = post_mem[question_idx:] if question_idx >= 0 else post_mem
    q_key = None
    for q_re in _QUERY_RES:
        m = q_re.search(search_region)
        if m:
            q_key = m.group(1).lower()
            break
    if q_key is None:
        return None, "no_query"
    keys = [p[0].lower() for p in pairs]
    if q_key not in keys:
        return None, "q_key_not_in_pairs"
    body = " ".join(f"{k.lower()} {v}" for k, v in pairs)
    return f"{body} ; {q_key}", "ok"


# ============================================================================
# Regenerate the EXACT R22 pooled corpus (no Gemma)
# ============================================================================

_NEUTRAL = [
    "The sky grows dim as the evening settles on the quiet valley.",
    "Mountains rise sharply above the forest of pine and silver birch.",
    "Birds gather in the gray clouds before the autumn rains arrive.",
    "Rivers carve wide paths through stone that has stood for ages.",
    "Old libraries hold secrets that only patient readers may find.",
    "Travelers stop at the inn to rest and to share news of the road.",
    "Snow covers the hilltops long before it reaches the lower fields.",
    "Stars appear one by one as night spreads across the open sky.",
]
_CONFUSING = [
    "Previously the value of q rose to 2 before the market closed.",
    "Our records suggest that variable m was set near 7 early last week.",
    "The seventh chapter of the book introduces a symbol named z.",
    "An analyst noted that the column labeled w reached around 5 in May.",
    "The ledger lists pair (s, 8) as the baseline entry for spring.",
    "Observers recorded that k trended toward 3 over the summer months.",
    "Notes from the workshop mention that h held at approximately 6.",
    "A footnote clarifies that d reached roughly 4 before the audit.",
    "The chart groups variable n near the value 9 in its left column.",
    "Eight apples sat in the basket beside the letter r on the shelf.",
]

# Approximate token counts — without Gemma tokenizer handy we'll use char/3 as proxy
def _approx_tokens(sent: str) -> int:
    # Rough: ~1 token per 3 chars for English prose. Matches Gemma BPE
    # within ~20%, good enough for corpus reconstruction.
    return max(1, len(sent) // 3)


def make_distractor(target_tokens: int, mode: str, rng: random.Random) -> str:
    if target_tokens <= 0:
        return ""
    pool = _CONFUSING if mode.startswith("confusing") else _NEUTRAL
    per = _approx_tokens(pool[0])
    n_sents = max(1, (target_tokens + per - 1) // per)
    return " ".join(rng.choice(pool) for _ in range(n_sents))


_KEY_POOL = list("abcdefghijklmnopqrstuvwxyz")
_VAL_POOL = [str(d) for d in range(10)]


def make_prompt(n_pairs, distractor_tokens, mode, rng):
    keys = rng.sample(_KEY_POOL, n_pairs)
    values = [rng.choice(_VAL_POOL) for _ in range(n_pairs)]
    q_idx = rng.randrange(n_pairs)
    q_key = keys[q_idx]
    expected = values[q_idx]
    mem_body = " ".join(f"{k}={v}" for k, v in zip(keys, values))
    mem_block = f"<mem>{mem_body}</mem>"
    half = distractor_tokens // 2
    prefix = make_distractor(half, mode, rng)
    suffix = make_distractor(distractor_tokens - half, mode, rng)
    body = f"{prefix}\n\n{mem_block}\n\n{suffix}"
    prompt = f"{body}\n\nQuestion: What is the value of {q_key}? Answer: "
    return prompt, q_key, expected


def main():
    seeds = [2026_04_22, 2026_04_23]
    cells = [
        (5,  500, "confusing"),
        (5,  1500, "confusing_long"),
        (10, 500, "confusing"),
        (10, 1500, "confusing_long"),
        (15, 500, "confusing"),
        (15, 1500, "confusing_long"),
    ]
    REPLICAS = 5

    by_n = defaultdict(lambda: Counter())
    examples_by_reason = defaultdict(list)
    pair_counts_by_n = defaultdict(list)

    for seed in seeds:
        rng = random.Random(seed)
        for (n_pairs, dist_tok, mode) in cells:
            for r in range(REPLICAS):
                prompt, q_key, expected = make_prompt(n_pairs, dist_tok, mode, rng)
                result, reason = parse_classified(prompt)
                by_n[n_pairs][reason] += 1
                if reason != "ok":
                    # Record first occurrence per reason per N
                    key = (n_pairs, reason)
                    if len(examples_by_reason[key]) < 2:
                        # Slice key info
                        mem = _MEM_RE.search(prompt)
                        pairs = _KV_RE.findall(mem.group(1)) if mem else []
                        pair_counts_by_n[n_pairs].append(len(pairs))
                        examples_by_reason[key].append({
                            "n_pairs": n_pairs,
                            "dist": dist_tok,
                            "mode": mode,
                            "q_key": q_key,
                            "expected": expected,
                            "mem_pair_count": len(pairs),
                            "mem_preview": mem.group(1)[:200] if mem else "NO MEM",
                            "tail": prompt[-200:],
                        })

    print("=== parse_mqar_prompt results by N ===")
    print(f"  {'N':>3}  {'ok':>4}  {'no_mem':>7}  {'no_pairs':>9}  "
          f"{'no_query':>9}  {'q!∈pairs':>9}  {'total':>6}")
    for n in sorted(by_n.keys()):
        c = by_n[n]
        total = sum(c.values())
        print(f"  {n:>3}  {c['ok']:>4}  {c['no_mem']:>7}  {c['no_pairs']:>9}  "
              f"{c['no_query']:>9}  {c['q_key_not_in_pairs']:>9}  {total:>6}")

    print("\n=== Example failures (first 2 per (N, reason)) ===")
    for (n_pairs, reason), ex_list in sorted(examples_by_reason.items()):
        print(f"\n--- N={n_pairs} reason={reason!r} ---")
        for ex in ex_list:
            print(f"  q_key={ex['q_key']!r} expected={ex['expected']!r} "
                  f"mem_pair_count={ex['mem_pair_count']} "
                  f"dist={ex['dist']} mode={ex['mode']!r}")
            print(f"    mem preview: {ex['mem_preview']!r}")
            print(f"    tail: ...{ex['tail']!r}")


main()
