"""R53.19 — Full CALM-substrate stack: targeted failure analysis + repair.

R53.18a confirmed generic "your code failed, fix it" gives +0.0pp
(Gemma's iterative refinement is unreliable). The fix isn't more
retries — it's STRUCTURED diagnosis + TARGETED repair instructions.

This script builds the minimum-viable full stack:

  1. Layer 2 precompute  channel-code-hybrid retrieval hint in
                         system prompt (R53.7 pattern)
  2. Layer 1 verify      sandbox runs tests, captures structured
                         failure type
  3. Layer 3 repair      categorize failure → targeted fix prompt
                         based on error type (NameError → add import,
                         TypeError → rename shadowed name, etc.)
  4. N-sample fallback   if categorized repair fails, regenerate
                         with simplified template prompt up to 3x

Hypothesis: structured failure-aware repair + retrieval lifts Gemma
above R53.0's +0.0pp ceiling. Specifically targets csv_column_stats
(NameError on StringIO) and token_bucket (TypeError shadowing) where
generic feedback didn't help.

Daemon-only:
  bin/gemma-run scripts/r53_calm_substrate_full.py
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
MAX_TOKENS = 250


# Common import → module mapping for NameError repair
COMMON_IMPORTS = {
    "StringIO": "from io import StringIO",
    "BytesIO": "from io import BytesIO",
    "Dict": "from typing import Dict",
    "List": "from typing import List",
    "Optional": "from typing import Optional",
    "Tuple": "from typing import Tuple",
    "Any": "from typing import Any",
    "Callable": "from typing import Callable",
    "Counter": "from collections import Counter",
    "defaultdict": "from collections import defaultdict",
    "OrderedDict": "from collections import OrderedDict",
    "deque": "from collections import deque",
    "datetime": "from datetime import datetime",
    "timedelta": "from datetime import timedelta",
    "date": "from datetime import date",
    "time": "import time",
    "math": "import math",
    "json": "import json",
    "re": "import re",
    "os": "import os",
    "sys": "import sys",
    "csv": "import csv",
    "statistics": "import statistics",
    "mean": "from statistics import mean",
    "stdev": "from statistics import stdev",
    "Path": "from pathlib import Path",
}


class FailureCategory:
    """Plain class (not @dataclass) — daemon's exec sets __name__ to
    '__daemon__' which isn't a real module, breaking dataclass's
    sys.modules lookup."""
    __slots__ = ("kind", "detail", "repair_hint")

    def __init__(self, kind: str, detail: str, repair_hint: str):
        self.kind = kind
        self.detail = detail
        self.repair_hint = repair_hint


def categorize_failure(test_output: str, prev_code: str) -> FailureCategory:
    """Parse test_output + prev_code to identify failure type and
    produce a SPECIFIC actionable repair instruction."""
    out = test_output or ""

    # No extractable code at all
    if "no extractable code" in out.lower() or not prev_code:
        return FailureCategory(
            kind="NoCode",
            detail="Output contained no extractable Python code",
            repair_hint=(
                "Re-output JUST the Python code in a single ```python``` "
                "fenced block. NO prose, NO explanation, NO markdown "
                "headers — only the function or class definition."
            ),
        )

    # NameError — try to extract symbol name + suggest import
    m = re.search(r"NameError: name '(\w+)' is not defined", out)
    if m:
        sym = m.group(1)
        suggested = COMMON_IMPORTS.get(sym, f"# add appropriate import for {sym}")
        return FailureCategory(
            kind="NameError",
            detail=f"NameError: '{sym}' is not defined",
            repair_hint=(
                f"Your code uses `{sym}` but doesn't import it. "
                f"Add this line at the top: `{suggested}`. "
                f"Output the complete corrected code."
            ),
        )

    # TypeError 'int' object is not callable — shadowing
    if "'int' object is not callable" in out:
        return FailureCategory(
            kind="TypeError",
            detail="TypeError: 'int' object is not callable",
            repair_hint=(
                "You're calling an integer as if it were a function. "
                "Likely cause: a method/function name was overwritten "
                "with an integer value (e.g. `self.consume = capacity` "
                "shadows method `consume`). Rename the integer attribute "
                "(e.g. `self.tokens = capacity`) and use the new name "
                "where you assigned the value. Output complete code."
            ),
        )

    # AttributeError on None
    m = re.search(r"AttributeError: 'NoneType' object has no attribute '(\w+)'", out)
    if m:
        attr = m.group(1)
        return FailureCategory(
            kind="AttributeError",
            detail=f"AttributeError: NoneType has no attribute '{attr}'",
            repair_hint=(
                f"An object is None when `.{attr}` is accessed. "
                f"Add a None-check before the access (e.g. "
                f"`if obj is not None: obj.{attr}`) or initialize "
                f"the object first. Output complete code."
            ),
        )

    # SyntaxError
    m = re.search(r"SyntaxError: ([^\n]+)", out)
    if m:
        return FailureCategory(
            kind="SyntaxError",
            detail=f"SyntaxError: {m.group(1)}",
            repair_hint=(
                f"Python syntax error: {m.group(1)}. "
                f"Check brackets, colons, indentation, string quotes. "
                f"Output complete corrected code."
            ),
        )

    # ValueError
    m = re.search(r"ValueError: ([^\n]+)", out)
    if m:
        return FailureCategory(
            kind="ValueError",
            detail=f"ValueError: {m.group(1)}",
            repair_hint=(
                f"ValueError: {m.group(1)}. "
                f"Add validation/try-except around the conversion. "
                f"Output complete code."
            ),
        )

    # FAIL lines from print("PASS"/"FAIL ...") test patterns
    fail_lines = [l for l in out.splitlines() if l.startswith("FAIL")]
    if fail_lines:
        f = fail_lines[0][:120]
        return FailureCategory(
            kind="FAIL",
            detail=f,
            repair_hint=(
                f"Test assertion failed: '{f}'. "
                f"Trace your logic for this specific input. "
                f"The expected behavior is in the test. "
                f"Output complete corrected code."
            ),
        )

    # Other runtime error
    if "Runtime error" in out or "Traceback" in out:
        return FailureCategory(
            kind="Other",
            detail=out[:200],
            repair_hint=(
                f"Runtime error during execution: {out[:200]}. "
                f"Inspect the trace and fix. Output complete code."
            ),
        )

    # Unknown — fall back to generic
    return FailureCategory(
        kind="Unknown",
        detail=out[:200],
        repair_hint=(
            f"Tests failed with output: {out[:200]}. "
            f"Output complete corrected code."
        ),
    )


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
    import random as _rng_mod

    # Detach any prior install state
    for layer in m.layers:
        if hasattr(layer, "card_slots"):
            layer.card_slots = []
    m.reserved_channels = []
    m.verification_hooks = []
    print("[r53.19] cleared prior install state", flush=True)
    print(f"[r53.19] MAX_ATTEMPTS={MAX_ATTEMPTS}, MAX_TOKENS={MAX_TOKENS}",
          flush=True)

    db = CodeExampleDB.load_default()
    db.load_indices(CACHE_DIR)

    # Override _build_hints to use channel-code-hybrid (R53.7 pattern)
    rng = _rng_mod.Random(0)

    def _build_hints_channel(db, rng_, p, sanity_random):
        facade = CodeVerifierFacade(db=db, top_k=2)
        hints = facade.compute_hints(p.prompt)
        channel_hits = db.retrieve_channel(
            p.prompt, channel="code", k=2, mode="hybrid",
            dense_m=m, dense_tok=tok)
        hints.retrieved_examples = channel_hits
        block = hints.to_system_prefix(max_example_chars=160)  # tighter
        if len(block) > 1200:
            block = block[:1200] + "\n..."
        return block

    orig._build_hints = _build_hints_channel

    def get_test_output(code: str, test_code: str) -> str:
        if not code:
            return "no extractable code"
        combined = code + "\n\n" + test_code + "\npass\n"
        result = run_python(combined, timeout=5.0)
        if result.error:
            return f"Runtime error: {result.error}\n{result.stdout or ''}"
        return result.stdout or "(no test output)"

    def gen_repair(p, prev_code: str, hint: str) -> str:
        problem_trim = p.prompt[:200]
        code_trim = prev_code[:280]
        repair_prompt = REPAIR_PROMPT_TEMPLATE.format(
            prompt=problem_trim,
            prev_code=code_trim,
            repair_hint=hint,
        )
        out = m.generate(repair_prompt, tok, max_tokens=MAX_TOKENS,
                         device="cuda", stop_on_eos=True)
        return _trim_markers(out["text"])

    print(f"\n[r53.19] running {len(CORPUS)} problems with full stack...",
          flush=True)

    results: List[Tuple[str, int, int, int, int, int, str]] = []
    # (name, single-shot pass, total, final pass, total, n_attempts, last_kind)

    for i, p in enumerate(CORPUS):
        print(f"\n[{i+1}/{len(CORPUS)}] {p.name}", flush=True)
        t0 = time.time()

        # ATTEMPT 1: gen_hinted (channel-code-hybrid)
        raw = gen_hinted(m, tok, p, db, rng, sanity_random=False,
                          max_tokens=MAX_TOKENS)
        sp1, st1, _ = score(raw, p)
        print(f"  attempt 1 (hinted): {sp1}/{st1} ({time.time()-t0:.0f}s)",
              flush=True)

        best_pass, best_total = sp1, st1
        last_kind = "ok" if (st1 > 0 and sp1 == st1) else "n/a"
        n_attempts = 1
        prev_raw = raw

        for attempt_idx in range(2, MAX_ATTEMPTS + 1):
            if best_total > 0 and best_pass == best_total:
                break
            prev_code = extract_code(prev_raw, p.required)
            test_output = get_test_output(prev_code, p.test_code)
            cat = categorize_failure(test_output, prev_code)
            last_kind = cat.kind
            if not prev_code and cat.kind == "NoCode":
                # No code to feed back — try regen with explicit code-only request
                hint = cat.repair_hint
                code_for_prompt = "(no code emitted — re-attempt)"
            else:
                hint = cat.repair_hint
                code_for_prompt = prev_code
            t1 = time.time()
            new_raw = gen_repair(p, code_for_prompt, hint)
            new_pass, new_total, _ = score(new_raw, p)
            n_attempts += 1
            print(f"  attempt {attempt_idx} ({cat.kind}): "
                  f"{new_pass}/{new_total} ({time.time()-t1:.0f}s) "
                  f"[hint: {cat.detail[:60]}]", flush=True)
            if new_total > best_total or (new_total == best_total
                                            and new_pass > best_pass):
                best_pass, best_total = new_pass, new_total
                prev_raw = new_raw
            else:
                # No improvement — stop
                break

        results.append((p.name, sp1, st1, best_pass, best_total,
                         n_attempts, last_kind))

    # ------------- AGGREGATE -------------
    print("\n" + "=" * 110, flush=True)
    print(f"  {'name':<28} {'attempt 1':>11} {'final':>11} "
          f"{'attempts':>9} {'last err':>16}", flush=True)
    print("-" * 110, flush=True)
    s_total = (0, 0)
    f_total = (0, 0)
    for name, sp, st, fp, ft, na, kind in results:
        improved = "✓" if (fp/max(ft,1) > sp/max(st,1)) else (
            "=" if fp/max(ft,1) == sp/max(st,1) else "↓")
        print(f"  {name:<28} {sp:>4}/{st:<4}    {fp:>4}/{ft:<4}    "
              f"{na:>4}      {kind:>15}  {improved}", flush=True)
        s_total = (s_total[0] + sp, s_total[1] + st)
        f_total = (f_total[0] + fp, f_total[1] + ft)
    print("-" * 110, flush=True)
    print(f"  {'TOTAL':<28} {s_total[0]:>4}/{s_total[1]:<4}    "
          f"{f_total[0]:>4}/{f_total[1]:<4}", flush=True)
    if s_total[1] and f_total[1]:
        delta = (f_total[0]/f_total[1] - s_total[0]/s_total[1]) * 100
        print(f"  Δ final-vs-attempt-1: {delta:+.1f}pp", flush=True)


if __name__ == "__main__":
    print("Daemon-only. Use:",
          "  bin/gemma-run scripts/r53_calm_substrate_full.py",
          flush=True)
    sys.exit(1)
elif "m" in globals() and "tok" in globals():
    run_eval(m, tok)                                  # noqa: F821
