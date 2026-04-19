"""Compare channel-specific vs combined retrieval on the R53.0 complex
eval corpus (6 multi-step coding problems).

Hypothesis: for problems where Gemma fails (token_bucket_rate_limiter,
csv_column_stats), channel-code retrieval surfaces meaningfully
different — and arguably better — implementation patterns than combined
retrieval, because channel-dense encodes (problem + code_fragment).

Measures, per problem × {5 retrieval modes}:
  - top-3 hits with score
  - top-3 example.key set
Then per problem:
  - pairwise Jaccard overlap of top-k example sets across modes
Then aggregate:
  - mean overlap (combined-dense vs channel-code-dense)
  - mean overlap (combined-dense vs channel-code-hybrid)

Low overlap = channel mode produces meaningfully different rankings,
which is the prerequisite for it being USEFUL. High overlap (>0.8) =
no point in channel mode for these queries.

Daemon-only entry point.

  bin/gemma-run scripts/r53_compare_channel_retrieval.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List, Tuple


CACHE_DIR = Path(".cache/r53_code_db")


# Re-implement just the corpus prompts here — importing
# r53_eval_complex.py would also pull its Gemma + sandbox machinery.
EVAL_PROBLEMS = [
    ("linked_list_bugs",
     "multi_bug",
     "Fix bugs in this LinkedList class. The class must support "
     "append(v), remove(v), to_list(). Three bugs: tail never updated "
     "after append, advance missing when not matched in remove, "
     "infinite loop in to_list because cur never advances."),

    ("date_validation_chain",
     "multi_bug",
     "Write a function validate_date(s) that takes a string in format "
     "'YYYY-MM-DD' and returns True if it's a valid calendar date. "
     "Handle leap years, month lengths, range checks. Multi-step "
     "validation chain."),

    ("log_level_counts",
     "lib_compose",
     "Parse a log file and count occurrences of each log level "
     "(DEBUG, INFO, WARNING, ERROR, CRITICAL). Return a dict mapping "
     "level to count. Use Python stdlib only."),

    ("csv_column_stats",
     "lib_compose",
     "Parse a CSV string and return a dict mapping each numeric "
     "column name to a sub-dict of {'mean', 'stdev', 'min', 'max'} "
     "over the column values. Skip non-numeric columns."),

    ("token_bucket_rate_limiter",
     "plan_code",
     "Implement a token bucket rate limiter as a class with "
     "consume(n) method. Constructor takes capacity and refill_rate. "
     "Refills tokens proportional to time since last call."),

    ("lru_cache_class",
     "plan_code",
     "Implement an LRU cache class with get(key) and put(key, value) "
     "methods. Constructor takes capacity. Evict least-recently-used "
     "when full. Both ops O(1)."),
]


MODES = [
    ("combined-tfidf",       "combined", "tfidf",  None),
    ("combined-dense",       "combined", "dense",  None),
    ("combined-hybrid",      "combined", "hybrid", None),
    ("channel-code-tfidf",   "channel",  "tfidf",  "code"),
    ("channel-code-dense",   "channel",  "dense",  "code"),
    ("channel-code-hybrid",  "channel",  "hybrid", "code"),
]


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not (a | b):
        return 0.0
    return len(a & b) / len(a | b)


def short(s: str, n: int = 75) -> str:
    s = " ".join(s.split())
    return s[:n] + ("…" if len(s) > n else "")


def run_compare(m, tok) -> None:
    # Force reimport so we pick up this session's edits to code_example_db.
    import sys
    for mod_name in list(sys.modules.keys()):
        if (mod_name.startswith("calm.llm_computer.facades.")
                or mod_name == "calm.llm_computer.facades"):
            del sys.modules[mod_name]

    from calm.llm_computer.facades.code_example_db import CodeExampleDB

    t0 = time.time()
    db = CodeExampleDB.load_default()
    db.load_indices(CACHE_DIR)
    print(f"[compare] DB + indices loaded ({len(db)} examples) "
          f"in {time.time() - t0:.1f}s", flush=True)
    print(f"[compare]   combined: tfidf={db.has_tfidf()} dense={db.has_dense()}",
          flush=True)
    print(f"[compare]   code:     tfidf={db.has_channel('code', 'tfidf')}"
          f" dense={db.has_channel('code', 'dense')}", flush=True)
    print(f"[compare]   reason:   tfidf={db.has_channel('reasoning', 'tfidf')}"
          f" dense={db.has_channel('reasoning', 'dense')}", flush=True)

    K = 3

    # Per-problem detail + per-mode top-k example.key sets
    per_problem_keys: List[dict] = []  # mode -> set of keys

    for name, cat, query in EVAL_PROBLEMS:
        print("\n" + "=" * 78, flush=True)
        print(f"PROBLEM: {name} ({cat})", flush=True)
        print(f"  query: {short(query, 200)}", flush=True)
        print("=" * 78, flush=True)

        per_mode_keys: dict = {}

        for mode_name, scope, mode, channel in MODES:
            t0 = time.time()
            if scope == "combined":
                hits = db.retrieve(
                    query, k=K, mode=mode, dense_m=m, dense_tok=tok)
            else:
                hits = db.retrieve_channel(
                    query, channel=channel, k=K, mode=mode,
                    dense_m=m, dense_tok=tok)
            dt = (time.time() - t0) * 1000

            print(f"\n  [{mode_name}]  ({dt:.0f} ms)", flush=True)
            keys = set()
            for h in hits:
                print(f"    {h.score:.3f}  {short(h.example.problem)}",
                      flush=True)
                keys.add(h.example.key)
            per_mode_keys[mode_name] = keys

        per_problem_keys.append({"name": name, "modes": per_mode_keys})

    # Aggregate analysis
    print("\n" + "=" * 78, flush=True)
    print("AGGREGATE — pairwise Jaccard overlap of top-3 example sets",
          flush=True)
    print("=" * 78, flush=True)

    pairs: List[Tuple[str, str]] = [
        ("combined-tfidf",      "combined-dense"),
        ("combined-tfidf",      "combined-hybrid"),
        ("combined-dense",      "combined-hybrid"),
        ("combined-dense",      "channel-code-dense"),
        ("combined-dense",      "channel-code-hybrid"),
        ("combined-hybrid",     "channel-code-hybrid"),
        ("channel-code-tfidf",  "channel-code-dense"),
        ("channel-code-tfidf",  "channel-code-hybrid"),
        ("channel-code-dense",  "channel-code-hybrid"),
    ]

    print(f"\n  {'pair':<55} {'mean Jacc':>10}  per-problem", flush=True)
    for a, b in pairs:
        scores = []
        per_prob_strs = []
        for entry in per_problem_keys:
            mka = entry["modes"][a]
            mkb = entry["modes"][b]
            j = jaccard(mka, mkb)
            scores.append(j)
            per_prob_strs.append(f"{j:.2f}")
        mean = sum(scores) / len(scores) if scores else 0.0
        pair_str = f"{a} vs {b}"
        print(f"  {pair_str:<55} {mean:>10.3f}  "
              f"[{', '.join(per_prob_strs)}]", flush=True)

    # Headline: how different is channel-code-dense from combined-dense?
    print("\n" + "=" * 78, flush=True)
    print("HEADLINE", flush=True)
    print("=" * 78, flush=True)

    cd_vs_chd = []
    cd_vs_chh = []
    for entry in per_problem_keys:
        cd_vs_chd.append(jaccard(entry["modes"]["combined-dense"],
                                  entry["modes"]["channel-code-dense"]))
        cd_vs_chh.append(jaccard(entry["modes"]["combined-dense"],
                                  entry["modes"]["channel-code-hybrid"]))
    print(f"  combined-dense vs channel-code-dense:   "
          f"mean Jaccard = {sum(cd_vs_chd)/len(cd_vs_chd):.3f}", flush=True)
    print(f"  combined-dense vs channel-code-hybrid:  "
          f"mean Jaccard = {sum(cd_vs_chh)/len(cd_vs_chh):.3f}", flush=True)
    print("\n  Read: 1.0 = identical hits, 0.0 = fully disjoint.", flush=True)
    print("  <0.3 → channel mode produces meaningfully different signal.",
          flush=True)
    print("  >0.8 → channel mode is essentially the same; no value-add.",
          flush=True)


if __name__ == "__main__":
    print("This script must be run inside the Gemma daemon. Use:",
          flush=True)
    print("  bin/gemma-run scripts/r53_compare_channel_retrieval.py",
          flush=True)
    sys.exit(1)
elif "m" in globals() and "tok" in globals():
    run_compare(m, tok)                                  # noqa: F821
