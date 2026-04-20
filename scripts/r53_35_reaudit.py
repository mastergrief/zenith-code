"""R53.35 re-audit — did mechanical extractor issues mask Gemma's capability?

Hypothesis: several "Gemma failed" rounds on the R53.0 6-problem
corpus attributed to capability gaps were actually mechanical output
issues that the strict AST-validate extractor rejected. Today's csv
diagnostic confirmed the pattern: 3 unbalanced parens in otherwise-
reasonable code. The walker's syntax_repair auto-heals 3/3, yielding
extractable code.

Protocol per problem:
  1. gen_hinted, medium budget (8K). Capture raw output.
  2. Save raw to /tmp/r53_reaudit/<name>.txt for post-hoc inspection.
  3. Classify pre-repair:
     - Has fence? has required name? has def/class?
     - Does ast.parse(fenced_code) succeed?
  4. If code extracts cleanly and runs → score.
  5. If syntax error → run syntax_repair. Re-check.
  6. Walker chain (shadow_rename, dict_synonym) on extracted code.
  7. Score final code.
  8. Emit a single-line summary with pre-repair / post-repair / fix kind.

No import injection or LLM repair loop this round — purely measures
walker applicability. For full pipeline see r53_21_import_inject.py.

Daemon-only:
  bin/gemma-run scripts/r53_35_reaudit.py
"""

from __future__ import annotations

import ast
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import torch


MAX_TOKENS = 8192
OUT_DIR = "/tmp/r53_reaudit"
USE_TQ4_KV = True

# Problems to skip (known-good baselines, not audit targets).
# R53.33: lru_cache_class passes 9/9 at medium budget (117s).
# Today: token_bucket passes 0/0 → 5/5 via walker (79s + 0.9s).
# linked_list_bugs has consistently been a long-decode problem
# (37+ min at 8K, 45+ min at 16K in today's runs) — skip to keep
# session budget sane; re-run separately if needed.
SKIP = {
    "linked_list_bugs", "lru_cache_class", "token_bucket_rate_limiter",
    # Already confirmed clean in run3; skip for this iteration to
    # focus on csv re-test after ast_repair fix.
    "date_validation_chain", "log_level_counts",
}


def classify_shape(raw: str, required_names) -> dict:
    """Static structural inspection before any repair."""
    return {
        "len": len(raw),
        "has_fence_open": "```python" in raw or "```py" in raw,
        "has_fence_close": raw.count("```") >= 2,
        "has_think": "<think>" in raw,
        "has_required": any(f"def {r}" in raw or f"class {r}" in raw
                            for r in required_names),
        "has_def": "def " in raw,
    }


def try_parse_fenced(raw: str) -> Tuple[Optional[str], Optional[str]]:
    """Pull out the first ```python ... ``` block. Returns
    (code_str or None, parse_error_msg or None)."""
    import re
    m = re.search(r"```(?:python|py)\n([\s\S]*?)\n```", raw)
    if not m:
        return None, "no_fence"
    code = m.group(1)
    try:
        ast.parse(code)
        return code, None
    except SyntaxError as e:
        return code, f"{e.msg}@line{e.lineno}"


def run_eval(m, tok) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    # Force fresh module loads. Daemon caches sys.modules across
    # script invocations; sys.modules clearing alone isn't always
    # enough (parent-package refs can pin the old object). Use
    # importlib.reload for hard refresh.
    import sys as _sys
    import importlib
    for mod_name in list(_sys.modules.keys()):
        if (mod_name.startswith("calm.llm_computer.facades.")
                or mod_name == "calm.llm_computer.facades"
                or mod_name == "r53_eval_complex"):
            del _sys.modules[mod_name]

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    # Import then explicitly reload — guarantees disk re-read
    import calm.llm_computer.facades.ast_repair as _ast_repair_mod
    importlib.reload(_ast_repair_mod)
    print(f"[reaudit] ast_repair reloaded from "
          f"{_ast_repair_mod.__file__}", flush=True)
    # Quick smoke-test: verify the trailing-colon fix is live
    _smoke = "def f(:\n    pass\n"
    _smoke_r = _ast_repair_mod.repair_syntax(_smoke)
    print(f"[reaudit] smoke-test trailing-colon repair: "
          f"applied={_smoke_r.applied}", flush=True)

    from r53_eval_complex import (
        CORPUS, gen_hinted, score, extract_code,
    )
    from calm.llm_computer.facades.code_example_db import CodeExampleDB
    repair = _ast_repair_mod.repair
    repair_syntax = _ast_repair_mod.repair_syntax
    from calm.sandbox import run_python
    import random as _rng_mod

    CACHE_DIR = "/mnt/c/Users/gabes/projects/claw-code/.cache/r53_code_db"
    db = CodeExampleDB.load_default()
    # Inline load — we need prefer_tq4=False because the cached-dequant
    # path shipped in R53.28 expects shared tq4 storage/centroids on
    # the same device, and disk-loaded tq4 is CPU at load time.
    # db.load_indices() doesn't expose prefer_tq4, so bypass it.
    from calm.llm_computer.facades.retrieval import (
        DenseIndex, TfidfIndex,
    )
    tfidf_path = Path(CACHE_DIR) / "tfidf.json"
    dense_path = Path(CACHE_DIR) / "dense.pt"
    if tfidf_path.exists():
        db._tfidf = TfidfIndex.load(tfidf_path)
    if dense_path.exists():
        db._dense = DenseIndex.load(dense_path, prefer_tq4=False)
    print(f"[reaudit] DB loaded ({len(db)} examples)", flush=True)
    print(f"[reaudit] MAX_TOKENS={MAX_TOKENS}, corpus={len(CORPUS)} problems",
          flush=True)

    rng = _rng_mod.Random(0)

    def run_sandbox(code: str, test_code: str) -> str:
        if not code:
            return "no extractable code"
        combined = code + "\n\n" + test_code + "\npass\n"
        result = run_python(combined, timeout=5.0)
        if result.error:
            return f"Runtime error: {result.error}\n{result.stdout or ''}"
        return result.stdout or "(no test output)"

    def score_code(code, problem):
        if not code:
            return 0, 0
        wrapper = f"```python\n{code}\n```"
        sp, st, _ = score(wrapper, problem)
        return sp, st

    results = []
    # (name, raw_tokens, shape_flags, parse_ok, syntax_fixes,
    #  walker_kind, pre_score, post_score, notes)

    target = [p for p in CORPUS if p.name not in SKIP]
    skipped = [p.name for p in CORPUS if p.name in SKIP]
    print(f"[reaudit] skipping known-good: {skipped}", flush=True)
    print(f"[reaudit] re-auditing {len(target)} problems", flush=True)

    for i, p in enumerate(target):
        print(f"\n[{i+1}/{len(target)}] {p.name}", flush=True)
        t0 = time.time()

        raw = gen_hinted(m, tok, p, db, rng, sanity_random=False,
                         max_tokens=MAX_TOKENS, use_tq4_kv=USE_TQ4_KV)
        wall = time.time() - t0

        # Save raw
        raw_path = f"{OUT_DIR}/{p.name}.txt"
        with open(raw_path, "w") as f:
            f.write(raw)

        # Shape + parse check
        shape = classify_shape(raw, p.required)
        code, parse_err = try_parse_fenced(raw)

        # Pre-repair score via normal pipeline
        pre_sp, pre_st, _ = score(raw, p)

        syntax_fixes = 0
        walker_kind = ""
        post_code = None
        note = ""

        if code is None:
            note = "no_fence_in_output"
        elif parse_err is None:
            # Parses — try walker chain on existing extraction
            extracted = extract_code(raw, p.required)
            if extracted:
                r = repair(extracted, "")  # shadow runs unconditionally
                if r.applied:
                    walker_kind = r.kind
                    post_code = r.new_code
        else:
            # SyntaxError — try auto-heal
            syn = repair_syntax(code)
            if syn.applied:
                syntax_fixes = len(syn.notes)
                post_code = syn.new_code
                # Now try walker on the healed code
                r = repair(post_code, "")
                if r.applied:
                    walker_kind = r.kind
                    post_code = r.new_code
            else:
                note = f"syntax_unfixable:{parse_err[:40]}"

        # Post-repair score if we have improved code
        if post_code:
            post_sp, post_st = score_code(post_code, p)
        else:
            post_sp, post_st = pre_sp, pre_st

        results.append((
            p.name, shape, wall, parse_err, syntax_fixes, walker_kind,
            pre_sp, pre_st, post_sp, post_st, note,
        ))

        # Per-problem one-liner
        parse_s = "OK" if parse_err is None else f"SyntaxError@{parse_err[:30]}"
        fix_s = f"syn={syntax_fixes}" if syntax_fixes else ""
        walk_s = f"walk={walker_kind}" if walker_kind else ""
        print(f"  wall={wall:.0f}s  parse={parse_s}  "
              f"pre={pre_sp}/{pre_st}  post={post_sp}/{post_st}  "
              f"{fix_s} {walk_s} {note}", flush=True)

    # Aggregate + reframe
    print("\n" + "=" * 130, flush=True)
    print(f"  {'name':<28} {'parse':<22} {'pre':>7} {'post':>7} "
          f"{'syn':>4} {'walker':<14} note", flush=True)
    print("-" * 130, flush=True)
    pre_total = (0, 0)
    post_total = (0, 0)
    reaudit_wins = 0
    for (name, shape, wall, parse_err, syn_n, walk_k,
         psp, pst, fsp, fst, note) in results:
        parse_disp = "OK" if parse_err is None else f"SyntaxError@L{parse_err.split('@line')[-1]}"
        improved = " ✓" if (fsp/max(fst, 1) > psp/max(pst, 1)) else ""
        print(f"  {name:<28} {parse_disp:<22} {psp:>3}/{pst:<3} "
              f"{fsp:>3}/{fst:<3}{improved}  {syn_n:>3}  "
              f"{walk_k:<14} {note}", flush=True)
        pre_total = (pre_total[0] + psp, pre_total[1] + pst)
        post_total = (post_total[0] + fsp, post_total[1] + fst)
        if fsp > psp or fst > pst:
            reaudit_wins += 1
    print("-" * 130, flush=True)
    print(f"  {'TOTAL':<28} {'':<22} {pre_total[0]:>3}/{pre_total[1]:<3} "
          f"{post_total[0]:>3}/{post_total[1]:<3}", flush=True)
    print(f"\n[reaudit] problems lifted by walker: {reaudit_wins}/{len(results)}",
          flush=True)
    print(f"[reaudit] raw outputs saved to {OUT_DIR}/", flush=True)


if __name__ == "__main__":
    print("Daemon-only: bin/gemma-run scripts/r53_35_reaudit.py",
          flush=True)
    sys.exit(1)
elif "m" in globals() and "tok" in globals():
    run_eval(m, tok)                                  # noqa: F821
