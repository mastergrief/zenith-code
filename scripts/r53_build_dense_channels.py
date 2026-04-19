"""Build channel-specific dense indices for CodeExampleDB (R53 dual-path).

Daemon-only entry point. Mirrors `scripts/r53_build_dense.py` but encodes
each channel separately. Each vector represents (problem + channel_text)
mean-pooled through Gemma's token_embd, L2-normalized.

  bin/gemma-run scripts/r53_build_dense_channels.py

Code channel: ~6240 examples × ~50ms/text ≈ 5 min
Reasoning channel: ~3993 examples × ~50ms/text ≈ 3 min

Output: dense_code.pt (+.tq4.pt), dense_reasoning.pt (+.tq4.pt) plus
channel_maps.json (already there from build_tfidf_channels but rewritten).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path


CACHE_DIR = Path(".cache/r53_code_db")


def build_channels(m, tok, batch_size: int = 4) -> None:
    # Force reimport so a stale cached module from an earlier daemon
    # invocation doesn't shadow this session's code_example_db edits.
    import sys
    for mod_name in list(sys.modules.keys()):
        if (mod_name.startswith("calm.llm_computer.facades.")
                or mod_name == "calm.llm_computer.facades"):
            del sys.modules[mod_name]

    from calm.llm_computer.facades.code_example_db import CodeExampleDB

    t0 = time.time()
    db = CodeExampleDB.load_default()
    print(f"[dense-ch] DB loaded: {len(db)} unique examples "
          f"in {time.time() - t0:.1f}s", flush=True)

    # Reload existing combined indices + channel TF-IDFs (they're cheap
    # to keep around; we're only adding channel dense)
    db.load_indices(CACHE_DIR)
    print(f"[dense-ch] reloaded indices from {CACHE_DIR}", flush=True)
    print(f"[dense-ch]   combined: tfidf={db.has_tfidf()} dense={db.has_dense()}",
          flush=True)
    print(f"[dense-ch]   code:     tfidf={db.has_channel('code', 'tfidf')}"
          f" dense={db.has_channel('code', 'dense')}", flush=True)
    print(f"[dense-ch]   reason:   tfidf={db.has_channel('reasoning', 'tfidf')}"
          f" dense={db.has_channel('reasoning', 'dense')}", flush=True)

    # If channel TF-IDFs aren't built yet (clean cache), build them too —
    # they share the doc-to-ex mapping and are cheap.
    if not db.has_channel("code", "tfidf"):
        print("[dense-ch] building channel TF-IDFs first (~5s)...", flush=True)
        t0 = time.time()
        db.build_tfidf_channels()
        print(f"[dense-ch] channel TF-IDFs built in {time.time()-t0:.1f}s",
              flush=True)

    # Build channel dense indices
    print(f"[dense-ch] est time: ~"
          f"{(len(db._code_doc_to_ex) + len(db._reasoning_doc_to_ex)) * 0.05 / 60:.1f}"
          f" min total", flush=True)
    t0 = time.time()
    db.build_dense_channels(m, tok, batch_size=batch_size)
    dt = time.time() - t0
    print(f"[dense-ch] built in {dt:.1f}s", flush=True)
    print(f"[dense-ch]   code dense:      {tuple(db._dense_code.vectors.shape)}",
          flush=True)
    print(f"[dense-ch]   reasoning dense: {tuple(db._dense_reasoning.vectors.shape)}",
          flush=True)

    db.save_indices(CACHE_DIR)
    print(f"[dense-ch] saved to {CACHE_DIR}/", flush=True)

    # Sanity check — compare channel vs combined for the same query.
    # Different rankings would prove channel encoding adds signal beyond
    # problem-only encoding.
    print("\n[dense-ch] sanity 1 — code-channel vs combined for "
          "'implement caching with TTL':", flush=True)
    code_hits = db.retrieve_channel(
        "implement caching with TTL", channel="code", k=3, mode="dense",
        dense_m=m, dense_tok=tok)
    print("  CHANNEL CODE DENSE:", flush=True)
    for h in code_hits:
        print(f"    {h.score:.3f}  {h.example.problem[:75]}", flush=True)
    combined = db.retrieve(
        "implement caching with TTL", k=3, mode="dense",
        dense_m=m, dense_tok=tok)
    print("  COMBINED DENSE:", flush=True)
    for h in combined:
        print(f"    {h.score:.3f}  {h.example.problem[:75]}", flush=True)

    print("\n[dense-ch] sanity 2 — reasoning-channel for "
          "'how should I architect a rate limiter':", flush=True)
    reason_hits = db.retrieve_channel(
        "how should I architect a rate limiter", channel="reasoning",
        k=3, mode="dense", dense_m=m, dense_tok=tok)
    print("  CHANNEL REASONING DENSE:", flush=True)
    for h in reason_hits:
        print(f"    {h.score:.3f}  {h.example.problem[:75]}", flush=True)


# Daemon entrypoint
if __name__ == "__main__":
    print("This script must be run inside the Gemma daemon. Use:",
          flush=True)
    print("  bin/gemma-run scripts/r53_build_dense_channels.py", flush=True)
    sys.exit(1)
elif "m" in globals() and "tok" in globals():
    build_channels(m, tok)                                  # noqa: F821
