"""End-to-end DT install eval — Gemma vs Gemma+DT on MBPP.

This is the test that was pending from the 2026-04-23 handoff:
does the v14 code-skeleton DT (0.20 greedy honest val) actually
improve Gemma's code correctness when installed as a decode-path
bias layer?

Procedure:
  1. Parse N MBPP problems: prompt + expected function name +
     test assertions.
  2. For each, generate two outputs via `CodeDtSkeletonFacade`:
     - STOCK   : use_bias=False (Gemma alone)
     - DT      : use_bias=True  (DT predicts args, biases Gemma's decode)
  3. Extract Python code from each output, run against test assertions
     in the shared sandbox.
  4. Report A/B pass rate, regressions, examples where each wins.

Runs via gemma-run daemon. Output at DONE marker.

Expected signals:
  - If DT helps: pass rate rises, especially on problems where
    Gemma's arg list mismatched the test's expected signature.
  - If null: exact-match ≠ correctness, as hypothesized in handoff.
  - Regressions are informative — DT may bias wrong signature on
    some problems and prevent Gemma from self-correcting.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import List, NamedTuple, Optional

import torch  # noqa: F401  (sanity import for daemon env)


# Centralized eval defaults
from calm.llm_computer.eval_defaults import EVAL_CTX_SIZE, EVAL_MAX_TOKENS

import os
EVAL_N = int(os.environ.get("DT_EVAL_N", "50"))  # how many MBPP problems
GENERATE_MAX_TOKENS = min(800, EVAL_MAX_TOKENS)    # per-problem output budget


# ---------------------------------------------------------------
# MBPP loader
# ---------------------------------------------------------------

class MbppProblem(NamedTuple):
    idx: int
    prompt: str
    fn_name: str
    tests: List[str]
    ref_code: str


def load_mbpp(limit: int) -> List[MbppProblem]:
    path = Path("agents/distill/data/mbpp.jsonl")
    probs: List[MbppProblem] = []
    with path.open() as f:
        for i, line in enumerate(f):
            if len(probs) >= limit:
                break
            d = json.loads(line)
            msgs = {m["role"]: m["content"] for m in d["messages"]}
            user = msgs.get("user", "").strip()
            asst = msgs.get("assistant", "")

            # Reference code block
            code_m = re.search(r"```python\n(.*?)\n```", asst, re.DOTALL)
            ref_code = code_m.group(1) if code_m else ""

            # Tests (assert statements after "Verified test cases:")
            tests_section = asst.split("**Verified test cases:**")
            if len(tests_section) < 2:
                continue
            tests_block = tests_section[1]
            tests_m = re.search(r"```python\n(.*?)\n```", tests_block, re.DOTALL)
            if not tests_m:
                continue
            tests_raw = tests_m.group(1).strip()
            tests = [ln.strip() for ln in tests_raw.splitlines()
                     if ln.strip().startswith("assert ")]
            if not tests:
                continue

            # Function name from first assert: `assert foo(...)` or
            # `assert foo(...) == X`
            fn_m = re.search(r"assert\s+(\w+)\s*\(", tests[0])
            if not fn_m:
                continue
            fn_name = fn_m.group(1)

            probs.append(MbppProblem(
                idx=i, prompt=user, fn_name=fn_name, tests=tests,
                ref_code=ref_code,
            ))
    return probs


# ---------------------------------------------------------------
# Extraction + scoring
# ---------------------------------------------------------------

_CODE_FENCE_RE = re.compile(r"```(?:python)?\n(.*?)\n```", re.DOTALL)


def _trim_to_first_def(code: str) -> str:
    """Trim extracted code to end at the first top-level def/class body.

    Fixes the "toxic trailer" failure mode: Gemma often emits
    `# Example Usage:` + print(...) calls after the function body.
    Those prints can contain syntax errors (truncation, wrong
    quoting) that fail the entire sandbox exec, even when the
    function itself is perfect.

    Two-pass strategy:
      1. Textual strip — drop all lines from the first col-0 `print(`
         (or `# Example` comment) onward. These are always example
         calls outside the function, never part of the body.
      2. AST trim — parse the remaining code; keep only up to the
         end of the first FunctionDef/ClassDef.

    Pass 1 fires when AST parse would fail on the raw trailer
    (typical — Gemma truncates mid-line); pass 2 cleans up
    well-formed trailers like `print(foo())` that parse fine but
    aren't part of the target function.
    """
    # Pass 1: textual strip
    lines = code.splitlines(keepends=True)
    cut = None
    for i, ln in enumerate(lines):
        stripped = ln.lstrip()
        if ln and not ln.startswith((" ", "\t")):
            # col-0 line. Drop example trailers + demo prints.
            if stripped.startswith("print(") or \
               stripped.startswith("# Example") or \
               stripped.startswith("# Test") or \
               stripped.startswith("# Usage"):
                cut = i
                break
    if cut is not None:
        code = "".join(lines[:cut]).rstrip() + "\n"

    # Pass 2: AST trim (if parseable)
    import ast as _ast
    try:
        tree = _ast.parse(code)
    except SyntaxError:
        return code
    if not tree.body:
        return code
    first = tree.body[0]
    if not isinstance(first, (_ast.FunctionDef, _ast.ClassDef,
                              _ast.AsyncFunctionDef)):
        return code
    end_line = getattr(first, "end_lineno", None)
    if end_line is None:
        return code
    lines2 = code.splitlines(keepends=True)
    return "".join(lines2[:end_line])


def extract_code(output: str, fn_name: str) -> Optional[str]:
    """Extract Python code from Gemma output.

    The facade prompt opens a ```python fence BEFORE Gemma generates,
    so the generated text typically contains only the CLOSING fence.
    Preferred path: everything before the first ``` in the output.

    Only fall to full fence-pair if the output starts with code that
    doesn't include a def (i.e. the opening fence IS in the generated
    text). Without this, an `fence-pair` search can match an interior
    example/demo block after the function, returning the wrong slice.
    """
    # 1. Close-only (preferred): everything up to the first ``` is the
    #    code. Fires when output starts with `def` and has ≥1 ``` later.
    end_fence = output.find("```")
    if end_fence > 0:
        candidate = output[:end_fence].rstrip()
        if "def " in candidate or "class " in candidate:
            return _trim_to_first_def(candidate)
    # 2. Full fence pair (generated text opened its own fence)
    m = _CODE_FENCE_RE.search(output)
    if m:
        return _trim_to_first_def(m.group(1))
    # 3. Bare def: slice from def <fn_name> to end of that function block
    m = re.search(rf"def\s+{re.escape(fn_name)}\s*\(", output)
    if m is None:
        m = re.search(r"def\s+\w+\s*\(", output)
    if m is None:
        return None
    start = m.start()
    # Stop at first line starting at col-0 with non-code prose (e.g.
    # "This function returns..."), or at first ``` (if present later).
    tail = output[start:]
    lines = tail.splitlines()
    out_lines = [lines[0]]
    for ln in lines[1:]:
        if ln.startswith("```"):
            break
        # Block ends when we hit a non-indented, non-blank line that
        # isn't a decorator / def / class / @-block continuation.
        if ln and not ln.startswith((" ", "\t", "#", "@")) \
                and not ln.lstrip().startswith(("def ", "class ", "import ",
                                                "from ", "if __name__")):
            break
        out_lines.append(ln)
    return "\n".join(out_lines).rstrip()


def score(output: str, p: MbppProblem) -> tuple[int, int, str]:
    """Run extracted code + asserts. Return (passed, total, diagnostic)."""
    from calm.sandbox import run_python

    code = extract_code(output, p.fn_name)
    if not code:
        return 0, len(p.tests), "no_code"

    # Wrap each assert in try/except to count partial credit
    test_harness_lines = []
    for i, assertion in enumerate(p.tests):
        test_harness_lines.append(
            f"try:\n"
            f"    {assertion}\n"
            f"    print('PASS {i}')\n"
            f"except Exception as _e:\n"
            f"    print('FAIL {i}: ' + type(_e).__name__)\n"
        )
    # Trailing `pass` at module-scope: calm.sandbox splits on last line and
    # tries eval() on it; if the last line is inside an except block the
    # split yields `except: <body>` without the body and IndentationErrors.
    # A module-scope `pass` keeps the sandbox's last-line split harmless.
    script = code + "\n\n" + "\n".join(test_harness_lines) + "\npass\n"

    r = run_python(script, timeout=8.0)
    out = r.stdout or ""
    if r.error:
        return 0, len(p.tests), f"err:{str(r.error)[:60]}"
    passed = out.count("PASS ")
    total = len(p.tests)
    diag = ""
    fail_lines = [ln for ln in out.splitlines() if ln.startswith("FAIL")]
    if fail_lines:
        diag = fail_lines[0][:80]
    return passed, total, diag


# ---------------------------------------------------------------
# Main eval
# ---------------------------------------------------------------

def run_eval():
    if "m" not in globals() or "tok" not in globals():
        print("ERROR: expected m, tok globals from gemma_daemon")
        sys.exit(1)

    from calm.llm_computer.facades.code_dt_skeleton import CodeDtSkeletonFacade

    problems = load_mbpp(EVAL_N)
    print(f"[dt-eval] loaded {len(problems)} MBPP problems", flush=True)

    facade = CodeDtSkeletonFacade(
        checkpoint_path="calm/hrm/checkpoints/dt_code_skel_best.pt",
        max_tokens=GENERATE_MAX_TOKENS,
        device="cuda",
    )
    facade.install(m, tok)
    print(f"[dt-eval] DT loaded, meta={getattr(facade, '_ckpt_meta', None)}",
          flush=True)

    totals = {"stock": [0, 0], "dt": [0, 0]}
    per_problem_rows = []

    for i, p in enumerate(problems):
        # DT prediction once per problem (shared across conditions for inspection)
        dt_raw = facade.predict_skeleton(p.prompt)
        dt_args = facade.parse_skeleton(dt_raw) if dt_raw else None

        # STOCK
        r_stock = facade.solve(p.prompt, p.fn_name, use_bias=False)
        s_pass, s_tot, s_diag = score(r_stock.generated, p)
        totals["stock"][0] += s_pass
        totals["stock"][1] += s_tot

        # DT-biased
        r_dt = facade.solve(p.prompt, p.fn_name, use_bias=True)
        d_pass, d_tot, d_diag = score(r_dt.generated, p)
        totals["dt"][0] += d_pass
        totals["dt"][1] += d_tot

        row = {
            "idx": p.idx, "fn": p.fn_name,
            "dt_raw": (dt_raw or "")[:80],
            "dt_args": dt_args,
            "skel_used": r_dt.skeleton,
            "stock": f"{s_pass}/{s_tot}",
            "dt": f"{d_pass}/{d_tot}",
            "stock_diag": s_diag, "dt_diag": d_diag,
            "stock_output": r_stock.generated[:2400],
            "dt_output": r_dt.generated[:2400],
        }
        per_problem_rows.append(row)

        delta = d_pass - s_pass
        marker = " +" if delta > 0 else (" -" if delta < 0 else "  ")
        print(f"[{i+1}/{len(problems)}] {p.fn_name:30s} "
              f"stock={s_pass}/{s_tot} dt={d_pass}/{d_tot}{marker} "
              f"args={dt_args}", flush=True)

    # Summary
    s_p, s_t = totals["stock"]
    d_p, d_t = totals["dt"]
    print()
    print(f"=== DT install A/B — {len(problems)} MBPP problems ===")
    print(f"  stock:    {s_p}/{s_t} = {s_p/max(s_t,1):.2%}")
    print(f"  dt-bias:  {d_p}/{d_t} = {d_p/max(d_t,1):.2%}")
    print(f"  delta:    {d_p - s_p:+d} ({(d_p - s_p)/max(s_t,1):+.2%})")

    # Win / regress breakdown
    wins = [r for r in per_problem_rows if _pass_count(r["dt"]) > _pass_count(r["stock"])]
    regressions = [r for r in per_problem_rows
                   if _pass_count(r["dt"]) < _pass_count(r["stock"])]
    print(f"  wins:        {len(wins)}")
    print(f"  regressions: {len(regressions)}")

    if wins:
        print("\n=== Sample wins ===")
        for r in wins[:5]:
            print(f"  {r['fn']}: stock={r['stock']} dt={r['dt']} "
                  f"skel={r['skel_used']}")
    if regressions:
        print("\n=== Sample regressions ===")
        for r in regressions[:5]:
            print(f"  {r['fn']}: stock={r['stock']} dt={r['dt']} "
                  f"skel={r['skel_used']} stock_diag={r['stock_diag']}")

    # Dump JSON for forensic analysis
    out_path = Path("/tmp/dt_install_eval_results.json")
    with out_path.open("w") as fh:
        json.dump({"totals": totals, "rows": per_problem_rows}, fh, indent=2)
    print(f"\n[dt-eval] forensic dump → {out_path}")
    print("DONE")


def _pass_count(s: str) -> int:
    """Parse '3/5' → 3."""
    try:
        return int(s.split("/")[0])
    except (ValueError, AttributeError):
        return 0


run_eval()
