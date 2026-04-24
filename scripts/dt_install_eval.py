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

import ast
import json
import re
import sys
from pathlib import Path
from typing import List, NamedTuple, Optional

import torch  # noqa: F401  (sanity import for daemon env)


# Centralized eval defaults
from calm.llm_computer.eval_defaults import EVAL_CTX_SIZE, EVAL_MAX_TOKENS

import os

# Benchmark dispatch: "mbpp" (default) or "humanevalplus"
EVAL_BENCHMARK = os.environ.get("EVAL_BENCHMARK", "mbpp").lower()
if EVAL_BENCHMARK not in ("mbpp", "humanevalplus"):
    raise SystemExit(f"EVAL_BENCHMARK must be 'mbpp' or 'humanevalplus', got {EVAL_BENCHMARK!r}")

# Default N depends on benchmark: MBPP=50 (full MBPP sweep), HE+=164 (all rows).
# DT_EVAL_N still wins if set (back-compat for MBPP).
_DEFAULT_N = {"mbpp": 50, "humanevalplus": 164}[EVAL_BENCHMARK]
EVAL_N = int(os.environ.get("DT_EVAL_N", str(_DEFAULT_N)))
GENERATE_MAX_TOKENS = min(800, EVAL_MAX_TOKENS)    # per-problem output budget

# Forensic dump path dispatched by benchmark so MBPP (/tmp/dt_install_eval_results.json)
# and HE+ dumps don't clobber each other.
DUMP_PATH = {
    "mbpp": "/tmp/dt_install_eval_results.json",
    "humanevalplus": "/tmp/he_install_eval_results.json",
}[EVAL_BENCHMARK]


# ---------------------------------------------------------------
# MBPP loader
# ---------------------------------------------------------------

class MbppProblem(NamedTuple):
    idx: int
    prompt: str
    fn_name: str
    tests: List[str]
    ref_code: str


class HumanEvalPlusProblem(NamedTuple):
    idx: int
    task_id: str          # e.g. "HumanEval/0"
    prompt: str           # signature + docstring (raw HF `prompt` field)
    fn_name: str          # entry_point (function name)
    ref_code: str         # prompt + canonical_solution
    test_code: str        # raw HF `test` field (full untruncated, 164/164 parse)
    inputs: list          # extracted `inputs = [...]` literal (164/164)
    results: Optional[list]  # extracted `results = [...]` literal (158/164)
    has_ref_func: bool    # True for the 6 rows using `ref_func(*inp)` instead


def _extract_literal(test_code: str, name: str):
    """Extract a top-level-or-nested `name = <literal>` assignment via AST.

    HumanEvalPlus test code defines `inputs`/`results` inside `def check(...)`;
    walking the AST handles either scope.
    """
    try:
        tree = ast.parse(test_code)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == name:
                    try:
                        return ast.literal_eval(node.value)
                    except (ValueError, TypeError):
                        return None
    return None


def _has_ref_func(test_code: str) -> bool:
    """Detect the 6 HE+ rows that use `ref_func(*inp)` for expected values."""
    try:
        tree = ast.parse(test_code)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in ("ref_func", "reference"):
                return True
    return False


def load_humaneval_plus(limit: int) -> List[HumanEvalPlusProblem]:
    """Load raw HumanEvalPlus rows from .cache/humanevalplus_raw/.

    Prerequisite: `scripts/fetch_humanevalplus_raw.py` must have run
    (creates the cache). Falls back to an informative error otherwise.
    """
    cache = Path(".cache/humanevalplus_raw/humanevalplus.jsonl")
    if not cache.exists():
        raise SystemExit(
            f"HumanEvalPlus raw cache missing at {cache}. "
            f"Run: PYTHONPATH=. python3 scripts/fetch_humanevalplus_raw.py"
        )
    probs: List[HumanEvalPlusProblem] = []
    with cache.open() as f:
        for i, line in enumerate(f):
            if len(probs) >= limit:
                break
            r = json.loads(line)
            inputs = _extract_literal(r["test"], "inputs")
            results = _extract_literal(r["test"], "results")
            if inputs is None:
                # All 164 rows should have literal `inputs`; skip any that don't
                # rather than crash, so the eval can still run.
                continue
            probs.append(HumanEvalPlusProblem(
                idx=i,
                task_id=r["task_id"],
                prompt=r["prompt"],
                fn_name=r["entry_point"],
                ref_code=r["prompt"] + r["canonical_solution"],
                test_code=r["test"],
                inputs=inputs,
                results=results,
                has_ref_func=_has_ref_func(r["test"]) if results is None else False,
            ))
    return probs


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


def _trim_keep_top_level_code(code: str) -> str:
    """HE+-aware variant of _trim_to_first_def: keeps ALL consecutive
    top-level imports/defs/classes from the start until the first
    non-code-definition statement. Required for HumanEvalPlus problems
    where the ref code contains a helper function before the target
    (HumanEval/10, /32, /38, /50 — is_palindrome / poly / encode_cyclic
    / encode_shift precede the target fn). MBPP path keeps using
    `_trim_to_first_def` (byte-identical behavior).
    """
    # Pass 1: textual strip (same as _trim_to_first_def)
    lines = code.splitlines(keepends=True)
    cut = None
    for i, ln in enumerate(lines):
        stripped = ln.lstrip()
        if ln and not ln.startswith((" ", "\t")):
            if stripped.startswith("print(") or \
               stripped.startswith("# Example") or \
               stripped.startswith("# Test") or \
               stripped.startswith("# Usage"):
                cut = i
                break
    if cut is not None:
        code = "".join(lines[:cut]).rstrip() + "\n"

    # Pass 2: AST-based trim. Keep nodes until a non-def/class/import node.
    import ast as _ast
    try:
        tree = _ast.parse(code)
    except SyntaxError:
        return code
    if not tree.body:
        return code
    _KEEP = (_ast.Import, _ast.ImportFrom, _ast.FunctionDef,
             _ast.AsyncFunctionDef, _ast.ClassDef)
    last_end = None
    for node in tree.body:
        if isinstance(node, _KEEP):
            end_line = getattr(node, "end_lineno", None)
            if end_line is None:
                continue
            last_end = end_line
        else:
            break
    if last_end is None:
        return code
    lines2 = code.splitlines(keepends=True)
    return "".join(lines2[:last_end])


def extract_code_he_plus(output: str, fn_name: str) -> Optional[str]:
    """HE+ variant of extract_code: uses _trim_keep_top_level_code so
    helper functions before the target are preserved.
    """
    end_fence = output.find("```")
    if end_fence > 0:
        candidate = output[:end_fence].rstrip()
        if "def " in candidate or "class " in candidate:
            return _trim_keep_top_level_code(candidate)
    m = _CODE_FENCE_RE.search(output)
    if m:
        return _trim_keep_top_level_code(m.group(1))
    # Bare def fallback — reuse the MBPP path since it already searches
    # for `def <fn_name>(` and the resulting slice usually contains only
    # the target fn (helpers-before-target in HE+ are always inside a
    # fenced block when Gemma emits them). If this ever misses helpers,
    # replace with a multi-def AST walk.
    return extract_code(output, fn_name)


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


def _test_defines_assertion(test_code: str) -> bool:
    """Detect the standard HumanEvalPlus test shape: `def assertion` at
    module scope. A handful of problems (HumanEval/32 at minimum) use a
    bespoke harness with a direct `assert _poly(...) <= eps` pattern
    instead. Those fall back to black-box `check(candidate)` scoring.
    """
    try:
        tree = ast.parse(test_code)
    except SyntaxError:
        return False
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == "assertion":
                return True
    return False


def score_humaneval_plus(output: str, p: HumanEvalPlusProblem) -> tuple[int, int, str, List[str]]:
    """Score HumanEvalPlus generated output with per-input partial credit
    where the test shape supports it, black-box `check(candidate)` fallback
    otherwise.

    Returns (pass_count, total_count, diag, per_input) where per_input[i]
    is "PASS" or "FAIL: <ExceptionName>" for the i-th input (or an N-long
    synthesized PASS/FAIL array for the black-box fallback rows).

    Receipt aggregates at macro mean of pass_count/total_count across
    problems, not micro (cell-weighted), because raw HumanEvalPlus cell
    count is dominated by a few high-input problems (max 1,100 cells for
    HumanEval/50; codex probe 2026-04-24).
    """
    from calm.sandbox import run_python

    code = extract_code_he_plus(output, p.fn_name)
    total = len(p.inputs)
    if not code:
        # HE+ prompts are signature+docstring; facade formats with trailing
        # ```python\n so Gemma's natural continuation is body-only (no `def`
        # re-emission). extract_code_he_plus requires `def ` / `class ` to
        # fire, so body-only output extracts as None. Retry with the prompt
        # prepended — signature+docstring+body is a valid def. Observed on
        # HumanEval/0/1/3/4 in N=5 smoke; HumanEval/2 happened to re-emit
        # `def` naturally.
        code = extract_code_he_plus(p.prompt + "\n" + output, p.fn_name)
    if not code:
        return 0, total, "no_code", ["FAIL: no_code"] * total

    use_per_input = _test_defines_assertion(p.test_code)

    # Script: extracted candidate + raw test harness + scoring loop.
    parts = [code, "", p.test_code, ""]

    if use_per_input:
        # Re-declare inputs at module scope so the loop sees them (they
        # live inside check()'s local scope in the original test).
        parts.append(f"_HE_INPUTS = {p.inputs!r}")
        if p.results is not None:
            parts.append(f"_HE_RESULTS = {p.results!r}")
        parts.append("")
        if p.results is not None:
            loop = (
                f"for _i, _inp in enumerate(_HE_INPUTS):\n"
                f"    try:\n"
                f"        _out = {p.fn_name}(*_inp)\n"
                f"        assertion(_out, _HE_RESULTS[_i], 0)\n"
                f"        print('PASS ' + str(_i))\n"
                f"    except Exception as _e:\n"
                f"        print('FAIL ' + str(_i) + ': ' + type(_e).__name__)"
            )
        else:
            loop = (
                f"for _i, _inp in enumerate(_HE_INPUTS):\n"
                f"    try:\n"
                f"        _exp = ref_func(*_inp)\n"
                f"        _out = {p.fn_name}(*_inp)\n"
                f"        assertion(_out, _exp, 0)\n"
                f"        print('PASS ' + str(_i))\n"
                f"    except Exception as _e:\n"
                f"        print('FAIL ' + str(_i) + ': ' + type(_e).__name__)"
            )
        parts.append(loop)
    else:
        # Black-box: run check(candidate). All-or-nothing score.
        # Binary result reported as total/total (pass) or 0/total (fail),
        # synthesized per_input mirrors the binary outcome so downstream
        # processing stays uniform.
        parts.append(
            f"try:\n"
            f"    check({p.fn_name})\n"
            f"    print('CHECK_PASS')\n"
            f"except AssertionError as _e:\n"
            f"    print('CHECK_FAIL: AssertionError')\n"
            f"except Exception as _e:\n"
            f"    print('CHECK_ERROR: ' + type(_e).__name__)"
        )
    parts.append("pass")  # sandbox last-line-eval guard
    script = "\n".join(parts)

    # Longer timeout than MBPP (up to 1100 inputs vs ~3 asserts).
    # numpy pre-import required: HE+ test harnesses import numpy as np and
    # use np.allclose() inside assertion(). Sandbox blocks `os`; numpy
    # transitively imports os so it must be pre-imported before the hook.
    r = run_python(script, timeout=30.0, extra_preimports=["numpy"])
    out = r.stdout or ""
    if r.error:
        return 0, total, f"err:{str(r.error)[:60]}", [f"FAIL: {type(r.error).__name__}"] * total

    # Parse scoring output. Two shapes depending on use_per_input branch.
    if use_per_input:
        per_input: List[str] = ["FAIL: NoOutput"] * total
        for line in out.splitlines():
            line = line.strip()
            m = re.match(r"^(PASS|FAIL)\s+(\d+)(?::\s*(.*))?$", line)
            if not m:
                continue
            kind, idx_s, tail = m.group(1), m.group(2), m.group(3)
            try:
                i = int(idx_s)
            except ValueError:
                continue
            if 0 <= i < total:
                per_input[i] = kind if kind == "PASS" else f"FAIL: {tail or ''}"
        passed = sum(1 for x in per_input if x == "PASS")
        first_fail = next((x for x in per_input if x.startswith("FAIL")), "")
        return passed, total, first_fail[:80], per_input
    else:
        # Black-box shape: look for CHECK_PASS / CHECK_FAIL / CHECK_ERROR.
        # All-or-nothing → synthesize per_input uniformly.
        ok = False
        diag = "FAIL: NoOutput"
        for line in out.splitlines():
            line = line.strip()
            if line == "CHECK_PASS":
                ok = True
                diag = ""
                break
            if line.startswith("CHECK_FAIL") or line.startswith("CHECK_ERROR"):
                diag = f"FAIL: {line.split(':', 1)[-1].strip()}"
                break
        if ok:
            return total, total, "", ["PASS"] * total
        return 0, total, diag[:80], [diag] * total


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


def run_eval_humaneval_plus():
    """HumanEvalPlus A/B: stock Gemma vs Gemma+DT, per-input partial credit.

    Dump at DUMP_PATH (/tmp/he_install_eval_results.json) carries:
      - totals + macro aggregates (all_pass, any_pass, macro mean fraction)
      - per-row: full stock_output/dt_output (no truncation), per-input
        PASS/FAIL arrays, and enough test metadata (test_code, inputs,
        results_or_ref_func flag) for offline RENAME replay without
        re-fetching HF.
    """
    if "m" not in globals() or "tok" not in globals():
        print("ERROR: expected m, tok globals from gemma_daemon")
        sys.exit(1)

    from calm.llm_computer.facades.code_dt_skeleton import CodeDtSkeletonFacade

    problems = load_humaneval_plus(EVAL_N)
    print(f"[he-eval] loaded {len(problems)} HumanEvalPlus problems", flush=True)

    facade = CodeDtSkeletonFacade(
        checkpoint_path="calm/hrm/checkpoints/dt_code_skel_best.pt",
        max_tokens=GENERATE_MAX_TOKENS,
        device="cuda",
    )
    facade.install(m, tok)
    print(f"[he-eval] DT loaded, meta={getattr(facade, '_ckpt_meta', None)}", flush=True)

    totals = {"stock": [0, 0], "dt": [0, 0]}
    all_pass = {"stock": 0, "dt": 0}
    any_pass = {"stock": 0, "dt": 0}
    macro_sum = {"stock": 0.0, "dt": 0.0}
    per_problem_rows = []

    for i, p in enumerate(problems):
        dt_raw = facade.predict_skeleton(p.prompt)
        dt_args = facade.parse_skeleton(dt_raw) if dt_raw else None

        # STOCK
        r_stock = facade.solve(p.prompt, p.fn_name, use_bias=False)
        s_pass, s_tot, s_diag, s_per = score_humaneval_plus(r_stock.generated, p)
        totals["stock"][0] += s_pass
        totals["stock"][1] += s_tot
        if s_pass == s_tot and s_tot > 0:
            all_pass["stock"] += 1
        if s_pass > 0:
            any_pass["stock"] += 1
        macro_sum["stock"] += (s_pass / max(s_tot, 1))

        # DT-biased
        r_dt = facade.solve(p.prompt, p.fn_name, use_bias=True)
        d_pass, d_tot, d_diag, d_per = score_humaneval_plus(r_dt.generated, p)
        totals["dt"][0] += d_pass
        totals["dt"][1] += d_tot
        if d_pass == d_tot and d_tot > 0:
            all_pass["dt"] += 1
        if d_pass > 0:
            any_pass["dt"] += 1
        macro_sum["dt"] += (d_pass / max(d_tot, 1))

        row = {
            "idx": p.idx,
            "task_id": p.task_id,
            "fn": p.fn_name,
            "dt_raw": (dt_raw or "")[:80],
            "dt_args": dt_args,
            "skel_used": r_dt.skeleton,
            "stock": f"{s_pass}/{s_tot}",
            "dt": f"{d_pass}/{d_tot}",
            "stock_diag": s_diag,
            "dt_diag": d_diag,
            "stock_output": r_stock.generated,   # FULL, no [:2400] truncation
            "dt_output":    r_dt.generated,       # FULL, no [:2400] truncation
            "stock_per_input": s_per,
            "dt_per_input":    d_per,
            # Test metadata for offline RENAME replay (no HF re-fetch)
            "test_code": p.test_code,
            "inputs":    p.inputs,
            "results":   p.results,               # None for the 6 ref_func rows
            "has_ref_func": p.has_ref_func,
            "prompt":    p.prompt,
            "ref_code":  p.ref_code,
        }
        per_problem_rows.append(row)

        delta = d_pass - s_pass
        marker = " +" if delta > 0 else (" -" if delta < 0 else "  ")
        print(f"[{i+1}/{len(problems)}] {p.task_id:16s} {p.fn_name:30s} "
              f"stock={s_pass:4d}/{s_tot:4d} dt={d_pass:4d}/{d_tot:4d}{marker} "
              f"args={dt_args}", flush=True)

    # Summary
    n = max(len(problems), 1)
    s_p, s_t = totals["stock"]
    d_p, d_t = totals["dt"]
    macro_s = macro_sum["stock"] / n
    macro_d = macro_sum["dt"] / n
    print()
    print(f"=== DT install A/B — {len(problems)} HumanEvalPlus problems ===")
    print(f"  all_pass:  stock={all_pass['stock']}/{n} ({all_pass['stock']/n:.2%})  "
          f"dt={all_pass['dt']}/{n} ({all_pass['dt']/n:.2%})  "
          f"delta={all_pass['dt']-all_pass['stock']:+d}")
    print(f"  any_pass:  stock={any_pass['stock']}/{n} ({any_pass['stock']/n:.2%})  "
          f"dt={any_pass['dt']}/{n} ({any_pass['dt']/n:.2%})  "
          f"delta={any_pass['dt']-any_pass['stock']:+d}")
    print(f"  macro_mean fraction:  stock={macro_s:.4f}  dt={macro_d:.4f}  "
          f"delta={macro_d-macro_s:+.4f}")
    print(f"  micro (cell-weighted, FYI not headline): "
          f"stock={s_p}/{s_t}={s_p/max(s_t,1):.2%} "
          f"dt={d_p}/{d_t}={d_p/max(d_t,1):.2%}")

    # Win / regress breakdown at problem granularity (pass_count delta)
    wins = [r for r in per_problem_rows if _pass_count(r["dt"]) > _pass_count(r["stock"])]
    regressions = [r for r in per_problem_rows
                   if _pass_count(r["dt"]) < _pass_count(r["stock"])]
    print(f"  wins (per-problem pass_count up):        {len(wins)}")
    print(f"  regressions (per-problem pass_count dn): {len(regressions)}")

    if wins:
        print("\n=== Sample wins ===")
        for r in wins[:5]:
            print(f"  {r['task_id']} {r['fn']}: stock={r['stock']} dt={r['dt']} "
                  f"skel={r['skel_used']}")
    if regressions:
        print("\n=== Sample regressions ===")
        for r in regressions[:5]:
            print(f"  {r['task_id']} {r['fn']}: stock={r['stock']} dt={r['dt']} "
                  f"skel={r['skel_used']} stock_diag={r['stock_diag']}")

    out_path = Path(DUMP_PATH)
    with out_path.open("w") as fh:
        json.dump({
            "benchmark": "humanevalplus",
            "n": n,
            "totals": totals,
            "all_pass": all_pass,
            "any_pass": any_pass,
            "macro_mean_fraction": {"stock": macro_s, "dt": macro_d},
            "rows": per_problem_rows,
        }, fh, indent=2)
    print(f"\n[he-eval] forensic dump → {out_path}")
    print("DONE")


# Dispatch — runs under daemon only (which pre-binds m/tok into globals).
# Guarding on globals lets offline/smoke scripts import this module to access
# load_mbpp / load_humaneval_plus / score_humaneval_plus without triggering
# the full eval or a SystemExit from the inner m/tok check.
if "m" in globals() and "tok" in globals():
    if EVAL_BENCHMARK == "humanevalplus":
        run_eval_humaneval_plus()
    else:
        run_eval()
