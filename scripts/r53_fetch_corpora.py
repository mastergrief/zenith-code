"""Fetch code-focused corpora for R53 DB.

Targets ~10K quality-filtered code examples by pulling:

  - MBPP (all splits, ~974)             — every example has test cases
  - HumanEvalPlus (164)                 — canonical Python benchmarks
  - Nohurry code-only rows (~150-300)   — Opus-reasoning, filtered to code
  - Crownelius (~2100) with retry      — Opus-reasoning, mixed domains

Each source converts to our `{messages: [system, user, assistant]}`
schema and passes through a quality gate:

  - Problem >= 20 chars (signal, not a title fragment)
  - Solution >= 100 chars (has substance beyond a one-liner)
  - Solution contains code markers (```, `def `, `class `, `import `,
    or 'function' with a `{`) OR includes verified test cases
  - Skip math word problems without programmatic solutions

Output: one JSONL per source under agents/distill/data/. The DB
default corpora list is updated in code_example_db.py to ingest
them — dedup handles overlaps.

Usage:
    PYTHONPATH=. python3 scripts/r53_fetch_corpora.py
    PYTHONPATH=. python3 scripts/r53_fetch_corpora.py --sources mbpp humaneval
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional


DATA_DIR = Path("agents/distill/data")
BATCH = 100
UA = {"User-Agent": "Mozilla/5.0"}

# Quality gate — shared by all converters
_CODE_RE = re.compile(r"```|(^|\n)(def |class |import |from \w+ import )|function\s*\w*\s*\(.*\)\s*\{")


def passes_quality(problem: str, solution: str) -> bool:
    if len(problem) < 20 or len(solution) < 100:
        return False
    if _CODE_RE.search(solution):
        return True
    # No code markers — reject unless this is a verified algorithm trace
    # (which always mentions 'step' or 'algorithm' explicitly; skip those
    # as low-value for R53).
    return False


def _fetch_rows(dataset_id_urlenc: str, config: str, split: str,
                total: int, sleep_s: float = 0.5) -> List[dict]:
    """Paginate the HF datasets-server rows API. Handles 429 with backoff."""
    rows: List[dict] = []
    for offset in range(0, total, BATCH):
        url = (
            f"https://datasets-server.huggingface.co/rows?"
            f"dataset={dataset_id_urlenc}&config={config}&split={split}"
            f"&offset={offset}&length={BATCH}"
        )
        for attempt in range(5):
            try:
                req = urllib.request.Request(url, headers=UA)
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = json.loads(resp.read())
                rows.extend(data.get("rows", []))
                break
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    wait = 5 * (attempt + 1)
                    print(f"    429 rate-limited; sleeping {wait}s...", flush=True)
                    time.sleep(wait)
                    continue
                raise
        time.sleep(sleep_s)
        print(f"    fetched {len(rows)}/{total}", flush=True)
    return rows


def _write_jsonl(path: Path, records: Iterable[dict]) -> int:
    """Write records as JSONL. Returns count written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


# -------------------------------------------------------------
# MBPP
# -------------------------------------------------------------

def convert_mbpp(row: dict) -> Optional[dict]:
    r = row.get("row", {})
    problem = r.get("text", "").strip()
    code = r.get("code", "").strip()
    tests = r.get("test_list") or []
    if not problem or not code:
        return None
    tests_block = "\n".join(tests)
    # Inline tests as the first part of the assistant turn so the
    # retrieval index treats them as content (the "verified tests"
    # signature matters for security/correctness hits).
    solution = f"```python\n{code}\n```\n\n**Verified test cases:**\n```python\n{tests_block}\n```"
    if not passes_quality(problem, solution):
        return None
    return {"messages": [
        {"role": "system", "content": "You are a careful Python coding assistant."},
        {"role": "user", "content": problem},
        {"role": "assistant", "content": solution},
    ]}


def fetch_mbpp(out: Path) -> int:
    print(f"Fetching MBPP (all splits)...", flush=True)
    all_rows: List[dict] = []
    for split, n in (("train", 374), ("validation", 90), ("test", 500), ("prompt", 10)):
        print(f"  split {split} ({n})", flush=True)
        rows = _fetch_rows(
            "google-research-datasets%2Fmbpp", "full", split, n)
        all_rows.extend(rows)
    converted = [c for c in (convert_mbpp(r) for r in all_rows) if c is not None]
    n = _write_jsonl(out, converted)
    print(f"  MBPP: wrote {n} / fetched {len(all_rows)}", flush=True)
    return n


# -------------------------------------------------------------
# HumanEvalPlus
# -------------------------------------------------------------

def convert_humaneval(row: dict) -> Optional[dict]:
    r = row.get("row", {})
    prompt = r.get("prompt", "").strip()
    canonical = r.get("canonical_solution", "").strip()
    tests = r.get("test", "").strip()
    if not prompt or not canonical:
        return None
    # HumanEval prompt = signature + docstring. Combine with canonical
    # as the full solution.
    full_code = prompt + canonical
    solution = f"```python\n{full_code}\n```"
    if tests:
        solution += f"\n\n**Test harness:**\n```python\n{tests[:1500]}\n```"
    problem = prompt
    if not passes_quality(problem, solution):
        return None
    return {"messages": [
        {"role": "system", "content": "You are a careful Python coding assistant."},
        {"role": "user", "content": problem},
        {"role": "assistant", "content": solution},
    ]}


def fetch_humaneval(out: Path) -> int:
    print(f"Fetching HumanEvalPlus...", flush=True)
    rows = _fetch_rows(
        "evalplus%2Fhumanevalplus", "default", "test", 164)
    converted = [c for c in (convert_humaneval(r) for r in rows) if c is not None]
    n = _write_jsonl(out, converted)
    print(f"  HumanEvalPlus: wrote {n} / fetched {len(rows)}", flush=True)
    return n


# -------------------------------------------------------------
# Nohurry (code-only filter)
# -------------------------------------------------------------

def convert_nohurry_code(row: dict) -> Optional[dict]:
    r = row.get("row", {})
    if r.get("category") != "code":
        return None
    problem = (r.get("problem") or "").strip()
    thinking = (r.get("thinking") or "").strip()
    solution = (r.get("solution") or "").strip()
    if not problem or not solution:
        return None
    full_assistant = (
        f"<think>\n{thinking}\n</think>\n\n{solution}"
        if thinking else solution
    )
    if not passes_quality(problem, full_assistant):
        return None
    return {"messages": [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": problem},
        {"role": "assistant", "content": full_assistant},
    ]}


def fetch_nohurry_code(out: Path) -> int:
    print(f"Fetching Nohurry (code-only filter)...", flush=True)
    rows = _fetch_rows(
        "nohurry%2FOpus-4.6-Reasoning-3000x-filtered",
        "default", "train", 3000)
    converted = [c for c in (convert_nohurry_code(r) for r in rows) if c is not None]
    n = _write_jsonl(out, converted)
    print(f"  Nohurry code: wrote {n} / fetched {len(rows)}", flush=True)
    return n


# -------------------------------------------------------------
# CodeContests (Google DeepMind)
# -------------------------------------------------------------

def convert_codecontests(row: dict) -> Optional[dict]:
    """Extract Python3 solution + description + public tests."""
    r = row.get("row", {})
    desc = (r.get("description") or "").strip()
    name = (r.get("name") or "").strip()
    sols = r.get("solutions", {})
    langs = sols.get("language") or []
    codes = sols.get("solution") or []
    if not desc or not langs or not codes:
        return None
    # Language 3 = Python3 per empirical probe
    py3_idx = next((i for i, l in enumerate(langs) if l == 3), None)
    if py3_idx is None:
        return None
    solution_code = (codes[py3_idx] or "").strip()
    if not solution_code:
        return None
    # Optional test preview
    public = r.get("public_tests") or {}
    inputs = public.get("input") or []
    outputs = public.get("output") or []
    test_lines: List[str] = []
    for inp, out in list(zip(inputs, outputs))[:2]:
        test_lines.append(f"# Input:\n{inp.strip()}\n# Output:\n{out.strip()}")
    test_block = "\n\n".join(test_lines)

    problem = f"**{name}**\n\n{desc}"
    assistant = f"```python\n{solution_code}\n```"
    if test_block:
        assistant += f"\n\n**Sample I/O:**\n```\n{test_block}\n```"
    if not passes_quality(problem, assistant):
        return None
    return {"messages": [
        {"role": "system", "content": "You are a careful competitive-programming Python assistant."},
        {"role": "user", "content": problem},
        {"role": "assistant", "content": assistant},
    ]}


def fetch_codecontests(out: Path) -> int:
    print(f"Fetching CodeContests (train + valid + test, Python3 only)...", flush=True)
    all_rows: List[dict] = []
    for split, n in (("train", 3762), ("valid", 117), ("test", 165)):
        print(f"  split {split} ({n})", flush=True)
        rows = _fetch_rows(
            "deepmind%2Fcode_contests", "default", split, n, sleep_s=0.8)
        all_rows.extend(rows)
    converted = [c for c in (convert_codecontests(r) for r in all_rows) if c is not None]
    n = _write_jsonl(out, converted)
    print(f"  CodeContests: wrote {n} Python3 solutions / fetched {len(all_rows)} total", flush=True)
    return n


# -------------------------------------------------------------
# BigCodeBench
# -------------------------------------------------------------

def convert_bigcodebench(row: dict) -> Optional[dict]:
    r = row.get("row", {})
    prompt = (r.get("instruct_prompt") or r.get("complete_prompt") or "").strip()
    canonical = (r.get("canonical_solution") or "").strip()
    code_prompt = (r.get("code_prompt") or "").strip()
    test = (r.get("test") or "").strip()
    if not prompt or not canonical:
        return None
    # Full solution = code_prompt (imports + def line) + canonical body
    full_code = (code_prompt + "\n" + canonical) if code_prompt else canonical
    solution = f"```python\n{full_code}\n```"
    if test:
        solution += f"\n\n**Unit tests:**\n```python\n{test[:1800]}\n```"
    if not passes_quality(prompt, solution):
        return None
    return {"messages": [
        {"role": "system", "content": "You are a careful Python coding assistant."},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": solution},
    ]}


def fetch_bigcodebench(out: Path) -> int:
    print(f"Fetching BigCodeBench (v0.1.4)...", flush=True)
    rows = _fetch_rows(
        "bigcode%2Fbigcodebench", "default", "v0.1.4", 1140, sleep_s=0.5)
    converted = [c for c in (convert_bigcodebench(r) for r in rows) if c is not None]
    n = _write_jsonl(out, converted)
    print(f"  BigCodeBench: wrote {n} / fetched {len(rows)}", flush=True)
    return n


# -------------------------------------------------------------
# Crownelius (retry with backoff)
# -------------------------------------------------------------

def convert_crownelius(row: dict) -> Optional[dict]:
    r = row.get("row", {})
    msgs = r.get("messages", [])
    if not msgs:
        return None
    user = next((m.get("content") for m in msgs if m.get("role") == "user"), "")
    asst = next((m.get("content") for m in msgs if m.get("role") == "assistant"), "")
    user = (user or "").strip()
    asst = (asst or "").strip()
    if not user or not asst:
        return None
    if not passes_quality(user, asst):
        return None
    return {"messages": [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": user},
        {"role": "assistant", "content": asst},
    ]}


def fetch_crownelius(out: Path) -> int:
    print(f"Fetching Crownelius (with retry)...", flush=True)
    rows = _fetch_rows(
        "Crownelius%2FOpus-4.6-Reasoning-2100x-formatted",
        "default", "train", 2160, sleep_s=2.0)
    converted = [c for c in (convert_crownelius(r) for r in rows) if c is not None]
    n = _write_jsonl(out, converted)
    print(f"  Crownelius: wrote {n} / fetched {len(rows)}", flush=True)
    return n


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------

SOURCES: Dict[str, tuple[Callable[[Path], int], str]] = {
    "mbpp":          (fetch_mbpp,          "mbpp.jsonl"),
    "humaneval":     (fetch_humaneval,     "humanevalplus.jsonl"),
    "nohurry":       (fetch_nohurry_code,  "nohurry_code.jsonl"),
    "crownelius":    (fetch_crownelius,    "crownelius.jsonl"),
    "codecontests":  (fetch_codecontests,  "codecontests.jsonl"),
    "bigcodebench":  (fetch_bigcodebench,  "bigcodebench.jsonl"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="*", default=list(SOURCES.keys()),
                    choices=list(SOURCES.keys()))
    args = ap.parse_args()

    totals: Dict[str, int] = {}
    for src in args.sources:
        fn, fname = SOURCES[src]
        out = DATA_DIR / fname
        try:
            totals[src] = fn(out)
        except Exception as e:
            print(f"  {src}: FAILED {e}", flush=True)
            totals[src] = 0

    print("\n=== SUMMARY ===")
    grand = 0
    for src, n in totals.items():
        print(f"  {src:<12} {n:>6}  ({SOURCES[src][1]})")
        grand += n
    print(f"  {'TOTAL':<12} {grand:>6}")


if __name__ == "__main__":
    main()
