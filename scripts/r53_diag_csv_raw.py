"""R53.35 diagnostic — dump Gemma's raw output on csv_column_stats.

R53_21 showed csv_column_stats produces 0/0 NoCode across attempts.
User hypothesis: budget (8K) isn't big enough. Numbers rule out
pure truncation (Gemma emits 2500-3800 tokens before EOS, well
under 8K ceiling).

This script dumps the RAW output — every character Gemma emits —
so we can eyeball what 2500+ tokens of non-extractable content
actually look like. Goal: identify whether it's:

1. `<think>` block that runs to EOS without a code fence
2. Prose explanation without fencing
3. Malformed fence (opens ```python but never closes)
4. Wrong function name (def compute_stats instead of csv_column_stats)
5. Something else

Writes to /tmp/r53_csv_raw.txt + prints head/tail to stdout.

Daemon-only:
  bin/gemma-run scripts/r53_diag_csv_raw.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch


def run_diag(m, tok) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from r53_eval_complex import (
        CORPUS, gen_hinted, extract_code, BASE_SYSTEM,
    )
    from calm.llm_computer.facades.code_example_db import CodeExampleDB
    import random as _rng_mod

    CACHE_DIR = "/mnt/c/Users/gabes/projects/claw-code/.cache/r53_code_db"

    csv_problem = [p for p in CORPUS if p.name == "csv_column_stats"][0]
    print(f"[diag] problem: {csv_problem.name}", flush=True)
    print(f"[diag] prompt len: {len(csv_problem.prompt)} chars", flush=True)

    db = CodeExampleDB.load_default()
    db.load_indices(CACHE_DIR)
    print(f"[diag] DB loaded ({len(db)} examples)", flush=True)

    rng = _rng_mod.Random(0)

    # Generate with centralized EVAL_MAX_TOKENS ceiling so we see what
    # Gemma would emit if not truncated at all. gen_hinted passes
    # kv_max_len=EVAL_CTX_SIZE internally when use_tq4_kv=True.
    from calm.llm_computer.eval_defaults import EVAL_MAX_TOKENS
    t0 = time.time()
    raw = gen_hinted(m, tok, csv_problem, db, rng, sanity_random=False,
                     max_tokens=EVAL_MAX_TOKENS, use_tq4_kv=True)
    wall = time.time() - t0

    # Stats
    raw_len = len(raw)
    code = extract_code(raw, csv_problem.required)
    code_len = len(code) if code else 0

    # Classify shape
    has_fence_open = "```python" in raw or "```py" in raw
    has_fence_close = raw.count("```") >= 2
    has_think = "<think>" in raw or "```think" in raw
    has_required = any(f"def {r}" in raw or f"class {r}" in raw
                       for r in csv_problem.required)
    has_def = "def " in raw
    has_class = "class " in raw

    print(f"\n[diag] wall: {wall:.1f}s", flush=True)
    print(f"[diag] raw length: {raw_len} chars", flush=True)
    # Token count via tokenizer (approximate — counts all output tokens)
    try:
        tok_ids = tok.encode(raw)
        print(f"[diag] raw tokens: ~{len(tok_ids)}", flush=True)
    except Exception as e:
        print(f"[diag] tok.encode failed: {e}", flush=True)
    print(f"[diag] extracted code length: {code_len} chars", flush=True)
    print(f"[diag] shape markers:", flush=True)
    print(f"  has ```python fence open:   {has_fence_open}", flush=True)
    print(f"  has ``` close (any 2+):     {has_fence_close}", flush=True)
    print(f"  has <think> block:          {has_think}", flush=True)
    print(f"  has required name:          {has_required} "
          f"(required={csv_problem.required})", flush=True)
    print(f"  has any 'def ':             {has_def}", flush=True)
    print(f"  has any 'class ':           {has_class}", flush=True)

    # Write full raw output
    out_path = "/tmp/r53_csv_raw.txt"
    with open(out_path, "w") as f:
        f.write(f"# csv_column_stats raw output\n")
        f.write(f"# wall: {wall:.1f}s, raw_len: {raw_len} chars, "
                f"extracted_code_len: {code_len} chars\n")
        f.write(f"# has_fence_open={has_fence_open}, "
                f"has_fence_close={has_fence_close}, "
                f"has_think={has_think}, has_required={has_required}\n")
        f.write("-" * 60 + "\n")
        f.write(raw)
    print(f"\n[diag] full output written to {out_path}", flush=True)

    # Head + tail preview
    print(f"\n[diag] -- FIRST 1500 chars --", flush=True)
    print(raw[:1500], flush=True)
    print(f"\n[diag] -- LAST 800 chars --", flush=True)
    print(raw[-800:] if raw_len > 800 else raw, flush=True)

    if code:
        print(f"\n[diag] -- EXTRACTED CODE --", flush=True)
        print(code[:1000], flush=True)
    else:
        print(f"\n[diag] NO CODE EXTRACTED", flush=True)


if __name__ == "__main__":
    print("Daemon-only. Use: bin/gemma-run scripts/r53_diag_csv_raw.py",
          flush=True)
    sys.exit(1)
elif "m" in globals() and "tok" in globals():
    run_diag(m, tok)                                  # noqa: F821
