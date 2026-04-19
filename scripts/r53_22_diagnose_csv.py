"""R53.22 — Diagnostic: capture one csv_column_stats run end-to-end.

Answers: after mechanical import injection, WHY is csv still 0/0?
Prints:
  1. Gemma's raw output
  2. Extracted code
  3. Sandbox stdout/error on raw code
  4. Code after import injection
  5. Sandbox stdout/error on injected code

If the final error is a different runtime issue (AttributeError on
csv.DictReader, wrong function signature, etc.), we know the ceiling
is deeper than imports. If extract_code+fence-wrap has a bug, we'll
see it here.

Daemon-only:
  bin/gemma-run scripts/r53_22_diagnose_csv.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List


CACHE_DIR = "/mnt/c/Users/gabes/projects/claw-code/.cache/r53_code_db"

# Minimum imports we'd auto-inject for csv domain
AUTO_IMPORTS = [
    "from io import StringIO",
    "import csv",
    "import statistics",
    "from statistics import mean, stdev",
]


def run_eval(m, tok) -> None:
    import sys as _sys
    for mod_name in list(_sys.modules.keys()):
        if (mod_name.startswith("calm.llm_computer.")
                and mod_name != "calm.llm_computer"):
            del _sys.modules[mod_name]
    for mod_name in list(_sys.modules.keys()):
        if mod_name == "scripts.r53_eval_complex":
            del _sys.modules[mod_name]

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from r53_eval_complex import (
        CORPUS, gen_hinted, score, extract_code,
        BASE_SYSTEM, _trim_markers,
    )
    import r53_eval_complex as orig
    from calm.llm_computer.facades.code_example_db import (
        CodeExampleDB, RetrievalHit,
    )
    from calm.llm_computer.facades.code_verifier import (
        CodeVerifierFacade,
    )
    from calm.sandbox import run_python
    import random as _rng_mod

    # Clear state
    for layer in m.layers:
        if hasattr(layer, "card_slots"):
            layer.card_slots = []
    m.reserved_channels = []
    m.verification_hooks = []

    db = CodeExampleDB.load_default()
    db.load_indices(CACHE_DIR)
    rng = _rng_mod.Random(0)

    def _build_hints_channel(db, rng_, p, sanity_random):
        facade = CodeVerifierFacade(db=db, top_k=2)
        hints = facade.compute_hints(p.prompt)
        channel_hits = db.retrieve_channel(
            p.prompt, channel="code", k=2, mode="hybrid",
            dense_m=m, dense_tok=tok)
        hints.retrieved_examples = channel_hits
        block = hints.to_system_prefix(max_example_chars=160)
        if len(block) > 1200:
            block = block[:1200] + "\n..."
        return block

    orig._build_hints = _build_hints_channel

    # Find csv_column_stats
    csv_problem = None
    for p in CORPUS:
        if p.name == "csv_column_stats":
            csv_problem = p
            break
    if csv_problem is None:
        print("ERROR: csv_column_stats not in corpus", flush=True)
        return

    print("=" * 80, flush=True)
    print("R53.22 diagnostic — csv_column_stats", flush=True)
    print("=" * 80, flush=True)
    print(f"Problem prompt (first 200 chars):", flush=True)
    print(f"  {csv_problem.prompt[:200]}...", flush=True)
    print(f"Required: {csv_problem.required}", flush=True)
    print(f"Test code (first 200 chars):", flush=True)
    print(f"  {csv_problem.test_code[:200]}...", flush=True)

    print("\n" + "=" * 80, flush=True)
    print("STEP 1 — Gemma generation (channel-code-hybrid hints)", flush=True)
    print("=" * 80, flush=True)
    t0 = time.time()
    raw_output = gen_hinted(m, tok, csv_problem, db, rng,
                            sanity_random=False, max_tokens=400)
    print(f"Gemma output ({time.time()-t0:.0f}s):", flush=True)
    print("-" * 80, flush=True)
    print(raw_output, flush=True)
    print("-" * 80, flush=True)

    print("\n" + "=" * 80, flush=True)
    print("STEP 2 — Extracted code", flush=True)
    print("=" * 80, flush=True)
    code = extract_code(raw_output, list(csv_problem.required))
    if not code:
        print("EXTRACT FAILED — no valid AST candidate with "
              "csv_column_stats", flush=True)
        print("\nTrying extract with relaxed required=[]...", flush=True)
        code2 = extract_code(raw_output, [])
        print(f"Relaxed extract: {len(code2)} chars", flush=True)
        if code2:
            print(code2[:400], flush=True)
        return
    print(f"Extracted {len(code)} chars:", flush=True)
    print(code, flush=True)

    print("\n" + "=" * 80, flush=True)
    print("STEP 3 — Sandbox run (no import injection)", flush=True)
    print("=" * 80, flush=True)
    script = code + "\n\n" + csv_problem.test_code + "\npass\n"
    r1 = run_python(script, timeout=8.0)
    print(f"stdout: {r1.stdout!r}", flush=True)
    print(f"error: {r1.error!r}", flush=True)

    print("\n" + "=" * 80, flush=True)
    print("STEP 4 — With auto-imports prepended", flush=True)
    print("=" * 80, flush=True)
    imports_block = "\n".join(AUTO_IMPORTS) + "\n"
    injected_code = imports_block + code
    print(f"Injected {len(AUTO_IMPORTS)} imports:", flush=True)
    for imp in AUTO_IMPORTS:
        print(f"  {imp}", flush=True)
    script2 = injected_code + "\n\n" + csv_problem.test_code + "\npass\n"
    r2 = run_python(script2, timeout=8.0)
    print(f"stdout: {r2.stdout!r}", flush=True)
    print(f"error: {r2.error!r}", flush=True)

    # Count PASS/FAIL
    if r2.stdout:
        passed = r2.stdout.count("PASS")
        failed = r2.stdout.count("FAIL")
        print(f"\n  PASS: {passed}, FAIL: {failed}", flush=True)

    print("\n" + "=" * 80, flush=True)
    print("STEP 5 — Diagnostic conclusion", flush=True)
    print("=" * 80, flush=True)
    if r2.error:
        print(f"Post-injection still errors: {type(r2.error).__name__ if not isinstance(r2.error, str) else 'str'}",
              flush=True)
        err_text = str(r2.error)
        if "NameError" in err_text:
            print("DIAGNOSIS: another NameError after injection — import table incomplete",
                  flush=True)
        elif "AttributeError" in err_text:
            print("DIAGNOSIS: AttributeError — Gemma used wrong API (e.g. csv.reader vs csv.DictReader)",
                  flush=True)
        elif "TypeError" in err_text:
            print("DIAGNOSIS: TypeError — Gemma's code calls something with wrong args",
                  flush=True)
        elif "ValueError" in err_text:
            print("DIAGNOSIS: ValueError — Gemma's parse logic fails on test input",
                  flush=True)
        else:
            print(f"DIAGNOSIS: other error class — {err_text[:200]}", flush=True)
    elif r2.stdout and "PASS" in r2.stdout:
        print(f"DIAGNOSIS: injection FIXED IT — some tests pass! "
              f"(Why did R53.21 get 0/0?)", flush=True)
    elif not r2.stdout:
        print("DIAGNOSIS: post-injection code runs silently — tests didn't "
              "execute. Likely function signature mismatch or missing "
              "function definition.", flush=True)
    else:
        print(f"DIAGNOSIS: all tests FAIL (semantic bug in logic)",
              flush=True)


if __name__ == "__main__":
    print("Daemon-only. Use: bin/gemma-run scripts/r53_22_diagnose_csv.py",
          flush=True)
    sys.exit(1)
elif "m" in globals() and "tok" in globals():
    run_eval(m, tok)                                  # noqa: F821
