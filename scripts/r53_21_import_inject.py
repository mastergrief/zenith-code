"""R53.21 — Mechanical import injection on NameError.

R53.19's structured repair told Gemma "add `from io import StringIO`"
and Gemma ignored it. Stop asking. Inject the import ourselves.

Pipeline per problem:
  1. gen_hinted (channel-code-hybrid, R53.7 pattern)
  2. Extract code, run tests in sandbox
  3. If pass → done
  4. If NameError on a symbol in COMMON_IMPORTS:
     prepend the import, re-run, iterate up to 4 times
  5. If still failing → fall back to R53.19 structured repair

Hypothesis: csv_column_stats at the Gemma ceiling is often
"emits correct csv+StringIO code but forgot the from-io import".
Mechanical injection lifts that without Gemma cooperation.

Baseline: R53.19 v3 = 26/26 (tests that ran, all passed).
csv_column_stats contributes 0/0 to that total (no code extracts or
code crashes pre-first-test). If injection makes csv's code run at
all, we get a lift.

No substrate install. Pure retrieval + mechanical + optional repair.

Daemon-only:
  bin/gemma-run scripts/r53_21_import_inject.py
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch


CACHE_DIR = "/mnt/c/Users/gabes/projects/claw-code/.cache/r53_code_db"

MAX_ATTEMPTS = 3
MAX_TOKENS_CEILING = 16384  # cap; AdaptiveBudget picks per-prompt
MAX_IMPORT_INJECTIONS = 4
USE_TQ4_KV = False  # tq4 KV storage works but has O(N²) dequant per step
                    # until fused flash-attn kernel lands (tracked as
                    # R53.34). At 16K ctx fp16 KVCache (~1.4 GB) fits
                    # comfortably on 8 GB; flip to True only when
                    # 256K+ ctx becomes the bottleneck AND the fused
                    # kernel is in place.


COMMON_IMPORTS = {
    "StringIO": "from io import StringIO",
    "BytesIO": "from io import BytesIO",
    "Dict": "from typing import Dict",
    "List": "from typing import List",
    "Optional": "from typing import Optional",
    "Tuple": "from typing import Tuple",
    "Any": "from typing import Any",
    "Callable": "from typing import Callable",
    "Iterable": "from typing import Iterable",
    "Iterator": "from typing import Iterator",
    "Set": "from typing import Set",
    "Counter": "from collections import Counter",
    "defaultdict": "from collections import defaultdict",
    "OrderedDict": "from collections import OrderedDict",
    "deque": "from collections import deque",
    "namedtuple": "from collections import namedtuple",
    "datetime": "from datetime import datetime",
    "timedelta": "from datetime import timedelta",
    "date": "from datetime import date",
    "timezone": "from datetime import timezone",
    "time": "import time",
    "math": "import math",
    "json": "import json",
    "re": "import re",
    "os": "import os",
    "sys": "import sys",
    "csv": "import csv",
    "io": "import io",
    "statistics": "import statistics",
    "mean": "from statistics import mean",
    "median": "from statistics import median",
    "stdev": "from statistics import stdev",
    "variance": "from statistics import variance",
    "Path": "from pathlib import Path",
    "reduce": "from functools import reduce",
    "partial": "from functools import partial",
    "lru_cache": "from functools import lru_cache",
    "wraps": "from functools import wraps",
    "chain": "from itertools import chain",
    "combinations": "from itertools import combinations",
    "permutations": "from itertools import permutations",
    "product": "from itertools import product",
    "groupby": "from itertools import groupby",
    "copy": "import copy",
    "deepcopy": "from copy import deepcopy",
    "ABC": "from abc import ABC",
    "abstractmethod": "from abc import abstractmethod",
    "dataclass": "from dataclasses import dataclass",
    "field": "from dataclasses import field",
    "Enum": "from enum import Enum",
    "IntEnum": "from enum import IntEnum",
    "hashlib": "import hashlib",
    "base64": "import base64",
    "random": "import random",
    "heapq": "import heapq",
    "bisect": "import bisect",
}


class FailureCategory:
    __slots__ = ("kind", "detail", "repair_hint")

    def __init__(self, kind: str, detail: str, repair_hint: str):
        self.kind = kind
        self.detail = detail
        self.repair_hint = repair_hint


def extract_nameerror_symbol(output: str) -> Optional[str]:
    """Return the undefined symbol name from a NameError trace, or None."""
    mm = re.search(r"NameError: name '(\w+)' is not defined", output)
    return mm.group(1) if mm else None


def categorize_failure(test_output: str, prev_code: str) -> FailureCategory:
    out = test_output or ""
    if "no extractable code" in out.lower() or not prev_code:
        return FailureCategory(
            "NoCode", "Output contained no extractable Python code",
            "Re-output JUST the Python code in a single ```python``` "
            "fenced block. NO prose, NO explanation — only the function "
            "or class definition.",
        )
    sym = extract_nameerror_symbol(out)
    if sym:
        suggested = COMMON_IMPORTS.get(sym, f"# add appropriate import for {sym}")
        return FailureCategory(
            "NameError", f"NameError: '{sym}' is not defined",
            f"Your code uses `{sym}` but doesn't import it. "
            f"Add this line at the top: `{suggested}`. "
            f"Output the complete corrected code.",
        )
    mm = re.search(r"TypeError: '(\w+)' object is not callable", out)
    if mm:
        shadow_type = mm.group(1)
        return FailureCategory(
            "TypeError", f"TypeError: '{shadow_type}' object is not callable",
            f"You're calling a {shadow_type} value as if it were a function. "
            f"A method/function name was overwritten by a {shadow_type} value "
            f"(e.g. `self.consume = capacity` shadows method `consume`). "
            f"Rename the {shadow_type} attribute (e.g. `self.tokens = capacity`) "
            f"and use the new name everywhere you assigned the value. "
            f"Output complete code.",
        )
    mm = re.search(
        r"AttributeError: 'NoneType' object has no attribute '(\w+)'", out)
    if mm:
        attr = mm.group(1)
        return FailureCategory(
            "AttributeError",
            f"AttributeError: NoneType has no attribute '{attr}'",
            f"An object is None when `.{attr}` is accessed. "
            f"Add None-check. Output complete code.",
        )
    mm = re.search(r"SyntaxError: ([^\n]+)", out)
    if mm:
        return FailureCategory(
            "SyntaxError", f"SyntaxError: {mm.group(1)}",
            f"Python syntax error: {mm.group(1)}. Output complete code.",
        )
    fail_lines = [l for l in out.splitlines() if l.startswith("FAIL")]
    if fail_lines:
        f = fail_lines[0][:120]
        return FailureCategory(
            "FAIL", f,
            f"Test assertion failed: '{f}'. Output complete corrected code.",
        )
    if "Runtime error" in out or "Traceback" in out:
        return FailureCategory(
            "Other", out[:200],
            f"Runtime error: {out[:200]}. Output complete code.",
        )
    return FailureCategory(
        "Unknown", out[:200],
        f"Tests failed: {out[:200]}. Output complete code.",
    )


def inject_imports_if_possible(
    code: str,
    test_code: str,
    run_fn,
    score_fn,
    problem,
) -> Tuple[str, int, int, List[str]]:
    """Iteratively inject imports when NameError on COMMON_IMPORTS symbol.
    Returns (final_code, passed, total, injected_imports).
    Stops when: tests pass, NameError on unknown symbol, or
    MAX_IMPORT_INJECTIONS reached."""
    current = code
    injected: List[str] = []
    sp, st, _ = score_fn(current, problem)
    if st > 0 and sp == st:
        return current, sp, st, injected

    for i in range(MAX_IMPORT_INJECTIONS):
        output = run_fn(current, test_code)
        sym = extract_nameerror_symbol(output)
        if not sym:
            break
        if sym not in COMMON_IMPORTS:
            break
        import_line = COMMON_IMPORTS[sym]
        if import_line in current:
            # Already present — infinite-loop guard
            break
        current = import_line + "\n" + current
        injected.append(import_line)
        sp, st, _ = score_fn(current, problem)
        if st > 0 and sp == st:
            return current, sp, st, injected

    return current, sp, st, injected


REPAIR_PROMPT_TEMPLATE = """\
Fix this Python code based on the SPECIFIC issue identified.

Problem: {prompt}

Your code:
```python
{prev_code}
```

Issue: {repair_hint}

Output the complete corrected ```python``` block:
"""


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
        CORPUS, gen_stock, gen_hinted, score, extract_code,
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
    from calm.adaptive import AdaptiveBudget
    import random as _rng_mod

    # Detach any prior install state (idempotent)
    for layer in m.layers:
        if hasattr(layer, "card_slots"):
            layer.card_slots = []
    m.reserved_channels = []
    m.verification_hooks = []
    print("[r53.21] cleared prior install state", flush=True)
    print(f"[r53.21] MAX_ATTEMPTS={MAX_ATTEMPTS}, "
          f"MAX_TOKENS_CEILING={MAX_TOKENS_CEILING} (adaptive), "
          f"MAX_IMPORT_INJECTIONS={MAX_IMPORT_INJECTIONS}", flush=True)

    budgeter = AdaptiveBudget()

    def budget_for(prompt: str) -> int:
        est = budgeter.estimate(prompt)
        return min(est.budget, MAX_TOKENS_CEILING), est

    db = CodeExampleDB.load_default()
    db.load_indices(CACHE_DIR)
    print(f"[r53.21] DB loaded ({len(db)} examples)", flush=True)

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

    def run_sandbox(code: str, test_code: str) -> str:
        if not code:
            return "no extractable code"
        combined = code + "\n\n" + test_code + "\npass\n"
        result = run_python(combined, timeout=5.0)
        if result.error:
            return f"Runtime error: {result.error}\n{result.stdout or ''}"
        return result.stdout or "(no test output)"

    def score_code(code: str, problem) -> Tuple[int, int, str]:
        """Score already-extracted code (not raw_output). Reuses score()
        by synthesizing a raw_output wrapper."""
        if not code:
            return 0, 0, "no extractable code"
        # Fastest path: synthesize a ```python fenced wrapper so score's
        # extractor finds it without modification.
        wrapper = f"```python\n{code}\n```"
        return score(wrapper, problem)

    def gen_repair(p, prev_code: str, hint: str, budget: int) -> str:
        problem_trim = p.prompt[:200]
        code_trim = prev_code[:280]
        repair_prompt = REPAIR_PROMPT_TEMPLATE.format(
            prompt=problem_trim,
            prev_code=code_trim,
            repair_hint=hint,
        )
        out = m.generate(repair_prompt, tok, max_tokens=budget,
                         device="cuda", stop_on_eos=True,
                         use_tq4_kv=USE_TQ4_KV)
        return _trim_markers(out["text"])

    print(f"\n[r53.21] running {len(CORPUS)} problems...", flush=True)

    # Track per-problem breakdown
    results: List[Tuple[str, int, int, int, int, int, str, List[str]]] = []
    # (name, att1_pass, att1_total, final_pass, final_total, n_attempts,
    #  last_kind, injected_imports)

    for i, p in enumerate(CORPUS):
        print(f"\n[{i+1}/{len(CORPUS)}] {p.name}", flush=True)
        t0 = time.time()

        budget, est = budget_for(p.prompt)
        print(f"  budget: {est.tier} ({budget} tok) — {est.reasoning}",
              flush=True)

        # ATTEMPT 1: gen_hinted (channel-code-hybrid)
        raw = gen_hinted(m, tok, p, db, rng, sanity_random=False,
                         max_tokens=budget, use_tq4_kv=USE_TQ4_KV)
        code = extract_code(raw, p.required)
        sp1, st1, _ = score(raw, p)
        print(f"  attempt 1 (hinted): {sp1}/{st1} ({time.time()-t0:.0f}s)",
              flush=True)

        # IMPORT INJECTION (if code extracted but tests failed)
        injected_imports: List[str] = []
        best_pass, best_total = sp1, st1
        best_code = code
        if code and not (st1 > 0 and sp1 == st1):
            t1 = time.time()
            new_code, new_pass, new_total, injected_imports = (
                inject_imports_if_possible(
                    code, p.test_code, run_sandbox, score_code, p))
            if injected_imports:
                print(f"  injected {len(injected_imports)} import(s): "
                      f"{injected_imports} → {new_pass}/{new_total} "
                      f"({time.time()-t1:.0f}s)", flush=True)
                if new_total > best_total or (new_total == best_total
                                              and new_pass > best_pass):
                    best_pass, best_total = new_pass, new_total
                    best_code = new_code

        last_kind = "ok" if (best_total > 0 and best_pass == best_total) else "n/a"
        n_attempts = 1
        prev_raw = raw

        # FALLBACK: structured repair (R53.19 style) if still failing
        for attempt_idx in range(2, MAX_ATTEMPTS + 1):
            if best_total > 0 and best_pass == best_total:
                break
            prev_code = extract_code(prev_raw, p.required)
            test_output = run_sandbox(prev_code, p.test_code)
            cat = categorize_failure(test_output, prev_code)
            last_kind = cat.kind
            hint = cat.repair_hint
            code_for_prompt = prev_code if prev_code else "(no code emitted)"
            t1 = time.time()
            new_raw = gen_repair(p, code_for_prompt, hint, budget)
            new_code = extract_code(new_raw, p.required)
            new_pass, new_total, _ = score(new_raw, p)
            n_attempts += 1
            print(f"  attempt {attempt_idx} ({cat.kind}): "
                  f"{new_pass}/{new_total} ({time.time()-t1:.0f}s) "
                  f"[hint: {cat.detail[:60]}]", flush=True)

            # Try import injection on the repaired code too
            if new_code and not (new_total > 0 and new_pass == new_total):
                inj_code, inj_pass, inj_total, inj_imports = (
                    inject_imports_if_possible(
                        new_code, p.test_code, run_sandbox, score_code, p))
                if inj_imports and (
                        inj_total > new_total or
                        (inj_total == new_total and inj_pass > new_pass)):
                    print(f"    + injected {len(inj_imports)} import(s): "
                          f"{inj_imports} → {inj_pass}/{inj_total}",
                          flush=True)
                    new_pass, new_total = inj_pass, inj_total
                    injected_imports.extend(inj_imports)

            if new_total > best_total or (new_total == best_total
                                          and new_pass > best_pass):
                best_pass, best_total = new_pass, new_total
                prev_raw = new_raw
            else:
                break

        results.append((p.name, sp1, st1, best_pass, best_total,
                        n_attempts, last_kind, injected_imports))

    # ------------- AGGREGATE -------------
    print("\n" + "=" * 130, flush=True)
    print(f"  {'name':<28} {'attempt 1':>11} {'final':>11} "
          f"{'attempts':>9} {'last err':>16} {'injected'}", flush=True)
    print("-" * 130, flush=True)
    s_total = (0, 0)
    f_total = (0, 0)
    for name, sp, st, fp, ft, na, kind, inj in results:
        improved = "✓" if (fp/max(ft, 1) > sp/max(st, 1)) else (
            "=" if fp/max(ft, 1) == sp/max(st, 1) else "↓")
        inj_str = ",".join(inj)[:40] if inj else "—"
        print(f"  {name:<28} {sp:>4}/{st:<4}    {fp:>4}/{ft:<4}    "
              f"{na:>4}      {kind:>15}  {improved}  {inj_str}",
              flush=True)
        s_total = (s_total[0] + sp, s_total[1] + st)
        f_total = (f_total[0] + fp, f_total[1] + ft)
    print("-" * 130, flush=True)
    print(f"  {'TOTAL':<28} {s_total[0]:>4}/{s_total[1]:<4}    "
          f"{f_total[0]:>4}/{f_total[1]:<4}", flush=True)
    if s_total[1] and f_total[1]:
        delta = (f_total[0]/f_total[1] - s_total[0]/s_total[1]) * 100
        print(f"  Δ final-vs-attempt-1: {delta:+.1f}pp", flush=True)
    print(f"  Baseline (R53.19 v3): 26/26", flush=True)


if __name__ == "__main__":
    print("Daemon-only. Use:",
          "  bin/gemma-run scripts/r53_21_import_inject.py", flush=True)
    sys.exit(1)
elif "m" in globals() and "tok" in globals():
    run_eval(m, tok)                                  # noqa: F821
