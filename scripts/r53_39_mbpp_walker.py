"""R53.39 — MBPP corpus walker test.

Runs the tier-2 AST walker chain (ast_repair.py) against N MBPP
problems. For each failure, tries to recover via mechanical rewrite.
Counts extractor-artifact lifts vs genuine capability gaps.

Protocol (per capability_gain.md §"Failure-surface gate as hard
precondition"):
1. Parse N problems from agents/distill/data/mbpp.jsonl — user
   content is the problem, assistant's trailing "Verified test cases"
   section has the assert lines.
2. For each problem: generate stock Gemma output; extract code;
   run user code + asserts in sandbox.
3. Partition:
   - clean     — all asserts pass on first try (skip for walker test)
   - walker_fixable — initial fail, walker rewrite makes it pass
   - genuine_fail   — initial fail, walker can't help
   - format_fail    — initial extract fails (no code)
4. Report per-problem outcome + aggregate lift count.

Cost: each Gemma run ~60-180s depending on AdaptiveBudget tier.
N=5 problems ≈ 10 min; N=20 ≈ 40-60 min; N=100 ≈ 3-5 hours.

Default: MBPP_N=5 for spot-check. Bump to 20-50 for proper
failure-surface pass. Much larger corpus runs need a long-run window.

Run via daemon (m, tok pre-bound):
  bin/gemma-run scripts/r53_39_mbpp_walker.py
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import List, Optional


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from calm.llm_computer.eval_defaults import (
    EVAL_CTX_SIZE, EVAL_MAX_TOKENS, ITERATION_N, FINAL_N,
    resolve_problem_window,
)

# ---- Bench config ----
MBPP_PATH = ROOT / "agents/distill/data/mbpp.jsonl"
# Problem window + N resolved from /tmp/substrate_eval_rotation.json
# (see calm/llm_computer/eval_defaults.py). Default ⇒ window 0, N=5.
# For generalization check: write {"window": 1} to rotate to next slice.
# For commit baseline: write {"final": true} to use N=20.
MBPP_N, MBPP_SKIP = resolve_problem_window(
    default_n=ITERATION_N, final_n=FINAL_N)
# MBPP-specific cap: 8K instead of EVAL_MAX_TOKENS=16K. Rationale:
# R13v2 showed clean problems (reverse_words, token-bucket style) finish
# at 1-3K tokens; rambling problems (max_chain_length, get_ludic)
# produce no code at 16K either way. Capping at 8K halves worst-case
# round time without losing any clean results. If MBPP problems ever
# legitimately need >8K (unlikely — MBPP test harnesses are small),
# bump via MBPP_MAX_TOKENS env var.
import os as _os_mbpp
MAX_TOKENS = int(_os_mbpp.environ.get("MBPP_MAX_TOKENS", "8192"))
USE_TQ4_KV = True


class MbppProblem:
    """Plain container (not @dataclass — Python 3.13 + exec-in-globals
    hits `sys.modules.get(cls.__module__).__dict__` AttributeError)."""

    def __init__(self, idx: int, prompt: str, tests: List[str],
                 fn_name: Optional[str]):
        self.idx = idx
        self.prompt = prompt
        self.tests = tests
        self.fn_name = fn_name


def parse_mbpp_problem(rec: dict, idx: int) -> Optional[MbppProblem]:
    """Extract (prompt, tests[, fn_name]) from an MBPP messages record.

    Returns None if the record lacks the expected shape (assistant
    message missing, no fenced test block).
    """
    msgs = rec.get("messages", [])
    user = next((m for m in msgs if m["role"] == "user"), None)
    asst = next((m for m in msgs if m["role"] == "assistant"), None)
    if not (user and asst):
        return None

    # Find the trailing "Verified test cases:" or similar block
    body = asst["content"]
    test_sec = None
    for marker in (r"Verified test cases:", r"\*\*Verified test cases:",
                   r"Test cases:", r"\*\*Test cases:"):
        m = re.search(marker, body, re.IGNORECASE)
        if m:
            test_sec = body[m.end():]
            break
    if test_sec is None:
        # Fallback: look for a trailing fenced block with asserts
        test_sec = body

    # Extract assert lines from any fenced block
    asserts: List[str] = []
    for block in re.finditer(r"```(?:python)?\s*\n(.*?)```", test_sec,
                             re.DOTALL):
        for line in block.group(1).splitlines():
            if line.strip().startswith("assert "):
                asserts.append(line.strip())
    if not asserts:
        return None

    # Parse the function name from the first assert: `assert <name>(...)`
    fn_name = None
    m_fn = re.match(r"assert\s+(\w+)\s*\(", asserts[0])
    if m_fn:
        fn_name = m_fn.group(1)

    return MbppProblem(idx=idx, prompt=user["content"].strip(),
                       tests=asserts, fn_name=fn_name)


def load_mbpp(limit: int = 5, skip: int = 0) -> List[MbppProblem]:
    problems: List[MbppProblem] = []
    skipped = 0
    with open(MBPP_PATH) as f:
        for idx, line in enumerate(f):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            p = parse_mbpp_problem(rec, idx)
            if p is None or not p.fn_name:
                continue
            if skipped < skip:
                skipped += 1
                continue
            problems.append(p)
            if len(problems) >= limit:
                break
    return problems


# ---- Prompt + generate ----

STOCK_PROMPT = (
    "<start_of_turn>user\n"
    "You are a careful, correct Python coding assistant.\n\n"
    "{prompt}\n\n"
    "Write ONLY the Python function in a ```python fenced block. "
    "No tests, no prose.\n"
    "<end_of_turn>\n"
    "<start_of_turn>model\n"
)


# Force-fence variant: prepends a code-fence + signature to the prompt
# tail so Gemma's emission is indented-body by construction.
FORCED_PROMPT = (
    "<start_of_turn>user\n"
    "You are a careful, correct Python coding assistant.\n\n"
    "{prompt}\n\n"
    "Write ONLY the Python function in a ```python fenced block. "
    "No tests, no prose.\n"
    "<end_of_turn>\n"
    "<start_of_turn>model\n"
    "```python\n"
    "{signature}\n"
)


def gen_stock(m_ref, tok_ref, p: MbppProblem, max_tokens: int = MAX_TOKENS) -> str:
    prompt = STOCK_PROMPT.format(prompt=p.prompt)
    out = m_ref.generate(prompt, tok_ref, max_tokens=max_tokens,
                         device="cuda", stop_on_eos=True,
                         use_tq4_kv=USE_TQ4_KV,
                         kv_max_len=EVAL_CTX_SIZE if USE_TQ4_KV else None)
    text = out["text"]
    for mark in ("<end_of_turn>", "<start_of_turn>"):
        i = text.find(mark)
        if i >= 0:
            text = text[:i]
    return text


def gen_forced(m_ref, tok_ref, p: MbppProblem, signature: str,
               max_tokens: int = MAX_TOKENS) -> str:
    """Force-fence fallback: same as gen_stock but with the code-fence
    and signature already in context. Returns the reconstructed output
    (sig prepended so extractor sees complete function)."""
    prompt = FORCED_PROMPT.format(prompt=p.prompt, signature=signature)
    out = m_ref.generate(prompt, tok_ref, max_tokens=max_tokens,
                         device="cuda", stop_on_eos=True,
                         use_tq4_kv=USE_TQ4_KV,
                         kv_max_len=EVAL_CTX_SIZE if USE_TQ4_KV else None)
    text = out["text"]
    for mark in ("<end_of_turn>", "<start_of_turn>"):
        i = text.find(mark)
        if i >= 0:
            text = text[:i]
    # Prepend fence + sig back — that was in Gemma's context but not
    # in its output stream; extractor needs the full sequence.
    return f"```python\n{signature}\n{text}"


def _derive_mbpp_signature(p: MbppProblem) -> str:
    """Derive a signature for an MBPP problem. Simpler than R53.38v3
    because MBPP fn_names are all function-based (lowercase). Try to
    peek at the first assert for arg count:
      `assert foo(arg1, arg2) == ...`
    Infer `def foo(a, b):` from that. Fallback: `def foo(*args):`.
    """
    if not p.tests:
        return f"def {p.fn_name}(*args):"
    first = p.tests[0]
    m = re.match(rf"assert\s+{re.escape(p.fn_name)}\s*\((.*?)\)\s*(?:==|!=|\s)",
                 first)
    if not m:
        return f"def {p.fn_name}(*args):"
    args_str = m.group(1)
    # Rough arg count — count top-level commas (doesn't handle nested
    # tuples/calls perfectly, but good enough for derivation)
    depth = 0
    n_commas = 0
    for ch in args_str:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "," and depth == 0:
            n_commas += 1
    n_args = n_commas + 1 if args_str.strip() else 0
    # Generate placeholder arg names
    if n_args == 0:
        return f"def {p.fn_name}():"
    if n_args == 1:
        return f"def {p.fn_name}(x):"
    if n_args == 2:
        return f"def {p.fn_name}(x, y):"
    if n_args == 3:
        return f"def {p.fn_name}(x, y, z):"
    # 4+: generic
    args = ", ".join(f"a{i}" for i in range(n_args))
    return f"def {p.fn_name}({args}):"


def _defines_function(code: str, name: str) -> bool:
    """True iff `code` parses AND defines a FunctionDef named `name`.

    Stricter than `name in code` — prevents docstring/comment
    substring false-positives that book as NameError at test time
    (R13 MBPP#1 first_repeated_char receipt).
    """
    import ast
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == name:
                return True
        if isinstance(node, ast.ClassDef) and node.name == name:
            return True  # MBPP class problems (rare)
    return False


def extract_code(raw: str, required_name: Optional[str]) -> Optional[str]:
    """Format-agnostic extractor. Try:
    1. Fenced ```python block whose AST actually defines required_name
    2. Fenced ```python block that parses (any)
    3. Bare code starting with `def <required_name>`
    4. Whole-output AST parse
    """
    import ast

    def _parses(s: str) -> bool:
        try:
            ast.parse(s)
            return True
        except SyntaxError:
            return False

    for block in re.finditer(r"```(?:python)?\s*\n(.*?)```", raw, re.DOTALL):
        code = block.group(1).strip()
        if not _parses(code):
            continue
        # Gate on actual FunctionDef, not substring — prevents
        # docstring/comment false-positives (R13v2 receipt).
        if required_name is None or _defines_function(code, required_name):
            return code

    if required_name:
        m_def = re.search(rf"^def\s+{re.escape(required_name)}\b", raw, re.M)
        if m_def:
            tail = raw[m_def.start():]
            if _parses(tail):
                return tail

    if _parses(raw):
        return raw
    return None


def score_code(code: str, tests: List[str]) -> tuple[int, int, str]:
    """Run the user function + each assert in a sandbox. Returns
    (passed, total, first_error)."""
    from calm.sandbox import run_python

    # Harness: each test wrapped in try/except, print PASS/FAIL
    lines = ["try:", f"    {'; '.join(tests)}", "    print('PASS')",
             "except Exception as e:", "    print(f'FAIL {type(e).__name__}: {e}')"]
    # Actually do one per assert for finer granularity
    harness = []
    for i, t in enumerate(tests):
        harness.append(
            f"try:\n    {t}\n    print('PASS {i}')\n"
            f"except Exception as e:\n    print(f'FAIL {i} {{type(e).__name__}}: {{e}}')")
    # Trailing "pass" protects the sandbox's last-line expr-eval wrapper
    # from stripping our final harness print out of its except: body
    # (leaving an empty block and triggering IndentationError).
    script = code + "\n\n" + "\n".join(harness) + "\npass\n"
    r = run_python(script, timeout=8.0)
    out = r.stdout or ""
    err = r.error or ""
    passed = out.count("PASS ")
    failed_count = out.count("FAIL ")
    total = passed + failed_count
    if total == 0 and err:
        return 0, len(tests), f"err: {str(err)[:80]}"
    fail_line = next((l for l in out.splitlines() if l.startswith("FAIL")),
                     "")
    return passed, total, fail_line[:100]


def run_mbpp_walker():
    # Reload ast_repair in case daemon has a stale version cached
    # (sys.modules persists across RESET_GLOBALS; new symbols like
    # repair_cascade added this session won't be visible without reload).
    import importlib
    import calm.llm_computer.facades.ast_repair as _ar
    importlib.reload(_ar)
    from calm.llm_computer.facades.ast_repair import repair_cascade

    # m, tok are daemon globals
    global m, tok

    problems = load_mbpp(limit=MBPP_N, skip=MBPP_SKIP)
    if not problems:
        print("[r53.39] ERROR: no MBPP problems parsed", flush=True)
        return

    print(f"[r53.39] MBPP walker test: N={len(problems)} problems "
          f"(skip={MBPP_SKIP}, max_tokens={MAX_TOKENS})", flush=True)
    print("=" * 80, flush=True)

    stats = {"clean": 0, "walker_fixable": 0, "genuine_fail": 0,
             "format_fail": 0, "force_fence_lift": 0}
    per_problem = []

    for pi, p in enumerate(problems):
        print(f"\n[{pi+1}/{len(problems)}] MBPP#{p.idx} fn={p.fn_name} "
              f"tests={len(p.tests)}", flush=True)
        print(f"  prompt: {p.prompt[:80]}...", flush=True)

        t0 = time.time()
        raw = gen_stock(m, tok, p)
        dt = time.time() - t0

        code = extract_code(raw, p.fn_name)
        used_force_fence = False

        # Force-fence fallback when stock extraction fails (R16)
        if not code:
            sig = _derive_mbpp_signature(p)
            print(f"  [{dt:.0f}s] stock NO CODE — trying force-fence "
                  f"with sig={sig!r}", flush=True)
            t1 = time.time()
            raw_forced = gen_forced(m, tok, p, sig)
            dt_forced = time.time() - t1
            code = extract_code(raw_forced, p.fn_name)
            dt += dt_forced
            used_force_fence = True
            if not code:
                stats["format_fail"] += 1
                print(f"  [+{dt_forced:.0f}s] force-fence ALSO NO CODE",
                      flush=True)
                per_problem.append((p.idx, "format_fail", 0, len(p.tests),
                                    None, ""))
                continue
            else:
                print(f"  [+{dt_forced:.0f}s] force-fence EXTRACTED "
                      f"{len(code)} chars", flush=True)

        passed, total, diag = score_code(code, p.tests)
        if passed == total:
            stats["clean"] += 1
            if used_force_fence:
                stats["force_fence_lift"] += 1
            tag = "CLEAN (force-fence)" if used_force_fence else "CLEAN"
            print(f"  [{dt:.0f}s] {tag} {passed}/{total}", flush=True)
            per_problem.append(
                (p.idx, "clean_force_fence" if used_force_fence else "clean",
                 passed, total, "force_fence" if used_force_fence else None,
                 ""))
            continue

        # Try walker cascade (R10) — allows empty_block → AST-walker
        # chaining when e.g. syntax_repair fires first then downstream
        # rewrites catch additional bugs.
        error_text = diag
        rr = repair_cascade(code, error_text, max_passes=4)
        if rr.applied:
            # Re-score
            p2, t2, d2 = score_code(rr.new_code, p.tests)
            if p2 > passed:
                stats["walker_fixable"] += 1
                print(f"  [{dt:.0f}s] WALKER LIFT  {passed}/{total} -> "
                      f"{p2}/{t2} via {rr.kind}", flush=True)
                per_problem.append(
                    (p.idx, "walker_fixable", p2, t2, rr.kind, diag))
                continue
            else:
                print(f"  [{dt:.0f}s] walker applied {rr.kind} but no "
                      f"improvement ({passed}->{p2})", flush=True)

        stats["genuine_fail"] += 1
        print(f"  [{dt:.0f}s] GENUINE FAIL {passed}/{total}  "
              f"diag={diag[:60]!r}", flush=True)
        per_problem.append(
            (p.idx, "genuine_fail", passed, total, None, diag))

    # Summary
    print("\n" + "=" * 80, flush=True)
    print("[r53.39] AGGREGATE", flush=True)
    print(f"  clean             : {stats['clean']:3d}/{len(problems)}  "
          f"(force-fence-lift: {stats['force_fence_lift']})",
          flush=True)
    print(f"  walker_fixable    : {stats['walker_fixable']:3d}/{len(problems)}",
          flush=True)
    print(f"  genuine_fail      : {stats['genuine_fail']:3d}/{len(problems)}",
          flush=True)
    print(f"  format_fail       : {stats['format_fail']:3d}/{len(problems)}",
          flush=True)
    print(f"\n  walker-attributable lift:      {stats['walker_fixable']} problems",
          flush=True)
    print(f"  force-fence-attributable lift: {stats['force_fence_lift']} problems",
          flush=True)

    print("\n[r53.39] per-problem:", flush=True)
    for idx, cat, p, t, kind, diag in per_problem:
        ext = f"  via {kind}" if kind else ""
        print(f"  MBPP#{idx:3d}  {cat:15s} {p}/{t}{ext}  {diag[:50]}",
              flush=True)

    print("[r53.39] DONE", flush=True)


# Daemon exec at top-level — skip when imported (no m/tok globals).
if "m" in globals() and "tok" in globals():
    run_mbpp_walker()
