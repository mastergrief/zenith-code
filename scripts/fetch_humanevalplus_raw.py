"""Fetch raw HumanEvalPlus rows from HuggingFace datasets-server.

Distinct from scripts/r53_fetch_corpora.py:fetch_humaneval which
converts + clips tests to 1500 chars for retrieval-corpus use.
This script caches the RAW rows (full `test` field, `canonical_solution`
separate from `prompt`, `entry_point` function name) for evaluation
harnesses that need the complete test code to run `check(candidate)`.

Cache shape: `.cache/humanevalplus_raw/humanevalplus.jsonl`, one
JSON object per line, each a raw HF row dict with keys:
  - task_id     (e.g. "HumanEval/0")
  - prompt      (signature + docstring)
  - canonical_solution (body only)
  - entry_point (function name, e.g. "has_close_elements")
  - test        (full test code with def check(candidate) harness)

Usage:
    PYTHONPATH=. python3 scripts/fetch_humanevalplus_raw.py
    PYTHONPATH=. python3 scripts/fetch_humanevalplus_raw.py --force  # ignore cache
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import List


CACHE_DIR = Path(".cache/humanevalplus_raw")
CACHE_FILE = CACHE_DIR / "humanevalplus.jsonl"
BATCH = 100
UA = {"User-Agent": "Mozilla/5.0"}
DATASET = "evalplus%2Fhumanevalplus"
CONFIG = "default"
SPLIT = "test"
TOTAL = 164


def _fetch_rows(sleep_s: float = 0.5) -> List[dict]:
    """Paginate the HF datasets-server rows API. Handles 429 with backoff.

    Adapted from scripts/r53_fetch_corpora.py:59-85 but returns each
    row's raw `.row` payload instead of the full envelope.
    """
    rows: List[dict] = []
    for offset in range(0, TOTAL, BATCH):
        url = (
            f"https://datasets-server.huggingface.co/rows?"
            f"dataset={DATASET}&config={CONFIG}&split={SPLIT}"
            f"&offset={offset}&length={BATCH}"
        )
        for attempt in range(5):
            try:
                req = urllib.request.Request(url, headers=UA)
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = json.loads(resp.read())
                batch = data.get("rows", [])
                for entry in batch:
                    r = entry.get("row", {})
                    if r:
                        rows.append(r)
                break
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    wait = 5 * (attempt + 1)
                    print(f"  429 rate-limited; sleeping {wait}s...", flush=True)
                    time.sleep(wait)
                    continue
                raise
        time.sleep(sleep_s)
        print(f"  fetched {len(rows)}/{TOTAL}", flush=True)
    return rows


def _verify_rows(rows: List[dict]) -> None:
    """Sanity-check the fetched rows match the expected shape."""
    required = ("task_id", "prompt", "canonical_solution", "entry_point", "test")
    missing_any = 0
    for r in rows:
        for k in required:
            if k not in r or r[k] in (None, ""):
                missing_any += 1
                break
    if missing_any:
        print(f"  WARN: {missing_any}/{len(rows)} rows missing required fields", file=sys.stderr)
    else:
        print(f"  OK: all {len(rows)} rows have task_id/prompt/canonical_solution/entry_point/test")

    # Quick parse check — how many raw `test` bodies are valid Python?
    import ast
    parse_ok = 0
    for r in rows:
        try:
            ast.parse(r["test"])
            parse_ok += 1
        except SyntaxError:
            pass
    print(f"  raw `test` parse-ok: {parse_ok}/{len(rows)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="Re-fetch even if cache exists")
    args = ap.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if CACHE_FILE.exists() and not args.force:
        lines = CACHE_FILE.read_text().splitlines()
        print(f"Cache hit: {CACHE_FILE} ({len(lines)} rows). Use --force to re-fetch.")
        with CACHE_FILE.open() as f:
            rows = [json.loads(ln) for ln in f if ln.strip()]
        _verify_rows(rows)
        return 0

    print(f"Fetching {TOTAL} raw HumanEvalPlus rows from evalplus/humanevalplus...")
    rows = _fetch_rows()
    if len(rows) != TOTAL:
        print(f"WARN: fetched {len(rows)} rows, expected {TOTAL}", file=sys.stderr)

    with CACHE_FILE.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {CACHE_FILE} ({len(rows)} rows)")
    _verify_rows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
