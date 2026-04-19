"""Build dense Gemma-encoded index for CodeExampleDB.

Daemon-only entry point. Assumes `m` (GemmaSubstrate) and `tok`
(GemmaTokenizer) are pre-loaded in globals by bin/gemma_daemon.py.

  bin/gemma-run scripts/r53_build_dense.py

Loads the current DB (merged from all DEFAULT_CORPORA + generated/),
loads existing TF-IDF index from cache, encodes every problem through
Gemma at a middle layer, mean-pools via KV-cache, L2-normalizes, saves
to .cache/r53_code_db/dense.pt.

On first call, may take ~15 min for ~9K examples. Second-run reloads
from cache in <1s.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path


CACHE_DIR = Path(".cache/r53_code_db")


def build_dense(m, tok, batch_size: int = 4) -> None:
    # Daemon may have cached stale modules from a pre-fetch import.
    # Force reimport so we pick up: updated DEFAULT_CORPORA, new
    # CodeExampleDB.load_indices / build_dense methods, retrieval.py
    # indices, and the generators package.
    import importlib, sys
    for mod_name in list(sys.modules.keys()):
        if (mod_name.startswith("calm.llm_computer.facades.")
                or mod_name == "calm.llm_computer.facades"):
            del sys.modules[mod_name]

    from calm.llm_computer.facades.code_example_db import CodeExampleDB

    # Load DB from all registered sources (default corpora includes
    # generated/*.jsonl). No regeneration — just load + dedup.
    t0 = time.time()
    db = CodeExampleDB.load_default()
    print(f"[dense] DB loaded: {len(db)} unique examples "
          f"in {time.time() - t0:.1f}s", flush=True)

    # Reuse cached TF-IDF index (don't rebuild)
    tfidf_path = CACHE_DIR / "tfidf.json"
    if tfidf_path.exists():
        db.load_indices(CACHE_DIR)
        print(f"[dense] TF-IDF reloaded from {tfidf_path}", flush=True)

    # Build dense index
    t0 = time.time()
    print(f"[dense] encoding {len(db)} problems through Gemma...",
          flush=True)
    print(f"[dense] est time: "
          f"{len(db) * 0.1 / 60:.1f} min at ~100ms/prefill",
          flush=True)
    db.build_dense(m, tok, batch_size=batch_size)
    dt = time.time() - t0
    print(f"[dense] built in {dt:.1f}s "
          f"({len(db) * 1000 / dt:.0f} prefills/sec)", flush=True)

    db.save_indices(CACHE_DIR)
    print(f"[dense] saved to {CACHE_DIR}/dense.pt", flush=True)
    print(f"[dense] vector shape: {db._dense.vectors.shape}, "
          f"dtype={db._dense.vectors.dtype}", flush=True)

    # Quick validation with a test query
    print("\n[dense] sanity check — query 'write is_prime function':",
          flush=True)
    hits = db.retrieve("write is_prime function", k=3, mode="dense",
                        dense_m=m, dense_tok=tok)
    for h in hits:
        print(f"  {h.score:.3f}  {h.example.problem[:80]}", flush=True)

    print("\n[dense] sanity check — hybrid query:",
          flush=True)
    hits = db.retrieve("SSRF URL validation block private IPs",
                        k=3, mode="hybrid",
                        dense_m=m, dense_tok=tok)
    for h in hits:
        print(f"  {h.score:.3f}  {h.example.problem[:80]}", flush=True)


# Daemon entrypoint
if __name__ == "__main__":
    print("This script must be run inside the Gemma daemon. Use:",
          flush=True)
    print("  bin/gemma-run scripts/r53_build_dense.py", flush=True)
    sys.exit(1)
elif "m" in globals() and "tok" in globals():
    build_dense(m, tok)                                  # noqa: F821
