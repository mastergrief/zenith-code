"""R53 data-generator pipeline — one-command rebuild of the corpus + indices.

Pipeline:

  1. Invoke every registered DomainDataGenerator
  2. Write per-generator JSONL to agents/distill/data/generated/<name>.jsonl
  3. Rebuild CodeExampleDB (default corpora + generated)
  4. Build global TF-IDF index (CPU, ~5 s)
  5. Build global dense index (GPU via daemon, ~2-10 min for 10K — optional)
  6. Persist indices to .cache/r53_code_db/

Usage:
    PYTHONPATH=. python3 scripts/r53_run_data_generators.py
    PYTHONPATH=. python3 scripts/r53_run_data_generators.py --n 100 --skip-dense
    # Inside the daemon (for dense indexing with `m`, `tok` pre-bound):
    bin/gemma-run scripts/r53_run_data_generators.py

Flags:
    --n N          max examples per generator (default: 1000)
    --skip-dense   skip Gemma-encoded dense index build (CPU-only run)
    --dense-only   only rebuild dense index (skip regeneration + TF-IDF)
    --out DIR      output directory for indices (default: .cache/r53_code_db)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

from calm.llm_computer.facades.data_generators import (
    get_generator, list_generators,
)
from calm.llm_computer.facades.data_generators.base import VerifiedExample


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "agents/distill/data"
GEN_DIR = DATA_DIR / "generated"
DEFAULT_CACHE = REPO_ROOT / ".cache/r53_code_db"


def _write_examples(gen_name: str, examples: List[VerifiedExample],
                    out_dir: Path) -> Path:
    """Write one generator's output as JSONL in messages schema."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{gen_name}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_messages_jsonl_record(),
                               ensure_ascii=False) + "\n")
    return path


def _write_pt_training(gen_name: str, examples: List[VerifiedExample],
                       out_dir: Path) -> Path:
    """Write PT training records (prompt → target pairs) to JSONL."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"pt_{gen_name}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_pt_training_record(),
                               ensure_ascii=False) + "\n")
    return path


def run_generators(n: int) -> dict:
    """Invoke every registered generator. Return {name: example_count}."""
    totals: dict = {}
    for name in list_generators():
        t0 = time.time()
        print(f"[{name}] generating up to {n}...", flush=True)
        gen_cls = get_generator(name)
        gen = gen_cls()
        examples = gen.generate(n)
        msg_path = _write_examples(name, examples, GEN_DIR)
        pt_path = _write_pt_training(name, examples, GEN_DIR)
        dt = time.time() - t0
        totals[name] = len(examples)
        print(f"[{name}] wrote {len(examples)} examples in {dt:.1f}s → "
              f"{msg_path.name}, {pt_path.name}",
              flush=True)
    return totals


def rebuild_db() -> "CodeExampleDB":
    """Load all default corpora + generated JSONL into one deduped DB."""
    from calm.llm_computer.facades.code_example_db import (
        CodeExampleDB, DEFAULT_CORPORA,
    )
    # DEFAULT_CORPORA already includes `generated/*.jsonl` explicitly —
    # don't re-add via glob (we'd just duplicate paths; dedup handles
    # it but the printed log is noisier).
    all_paths = [p for p in DEFAULT_CORPORA if not p.name.startswith("pt_")]
    print(f"rebuilding DB from {len(all_paths)} corpora:", flush=True)
    for p in all_paths:
        exists = "✓" if Path(p).exists() else "✗ (skip)"
        print(f"  {exists}  {p}", flush=True)
    db = CodeExampleDB.load_paths(all_paths)
    print(f"DB: {len(db)} unique examples", flush=True)
    return db


def build_tfidf_and_save(db, cache_dir: Path) -> None:
    t0 = time.time()
    db.build_tfidf()
    print(f"TF-IDF built in {time.time() - t0:.1f}s "
          f"(vocab ≈ {len(db._tfidf._idf)} terms)", flush=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    db.save_indices(cache_dir)
    print(f"TF-IDF saved to {cache_dir}", flush=True)


def build_dense_and_save(db, cache_dir: Path, m, tok,
                         batch_size: int = 8) -> None:
    t0 = time.time()
    db.build_dense(m, tok, batch_size=batch_size)
    dt = time.time() - t0
    print(f"Dense built in {dt:.1f}s "
          f"({len(db)} vectors, dim={db._dense.d_model})",
          flush=True)
    db.save_indices(cache_dir)
    print(f"Dense saved to {cache_dir}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000,
                    help="max examples per generator")
    ap.add_argument("--skip-dense", action="store_true",
                    help="skip Gemma dense index (CPU-only run)")
    ap.add_argument("--dense-only", action="store_true",
                    help="only rebuild dense index; skip regeneration + TF-IDF")
    ap.add_argument("--out", type=Path, default=DEFAULT_CACHE)
    args = ap.parse_args()

    # Daemon context?
    have_gemma = "m" in globals() and "tok" in globals()

    if not args.dense_only:
        # --- Regenerate ---
        t0 = time.time()
        totals = run_generators(args.n)
        print(f"\nGenerators total: {sum(totals.values())} examples in "
              f"{time.time() - t0:.1f}s", flush=True)
        for name, count in totals.items():
            print(f"  {name:<20} {count}", flush=True)

        # --- Rebuild DB ---
        db = rebuild_db()

        # --- TF-IDF ---
        build_tfidf_and_save(db, args.out)
    else:
        # Dense-only: load existing DB + indices
        from calm.llm_computer.facades.code_example_db import CodeExampleDB
        db = CodeExampleDB.load_default()
        if GEN_DIR.exists():
            for p in sorted(GEN_DIR.glob("*.jsonl")):
                if not p.name.startswith("pt_"):
                    db.ingest_jsonl(p)
        db.load_indices(args.out)
        print(f"DB: {len(db)} examples (dense-only rebuild)", flush=True)

    # --- Dense ---
    if args.skip_dense:
        print("skipping dense index (flag set)", flush=True)
    elif not have_gemma:
        print("\n[!] `m` and `tok` not in scope — run inside daemon:",
              flush=True)
        print("    bin/gemma-run scripts/r53_run_data_generators.py",
              flush=True)
        print("or pass --skip-dense for CPU-only pipeline.", flush=True)
    else:
        build_dense_and_save(db, args.out,
                             m, tok,                              # noqa: F821
                             batch_size=8)

    print("\n=== DONE ===", flush=True)


# Daemon entrypoint — if we have `m`, `tok` already loaded, call main
# after re-parsing empty argv (daemon doesn't pass CLI flags).
if __name__ == "__main__":
    main()
elif "m" in globals() and "tok" in globals():
    # When exec'd by the daemon, argv isn't parsed — run with defaults.
    sys.argv = ["r53_run_data_generators.py"]
    main()
