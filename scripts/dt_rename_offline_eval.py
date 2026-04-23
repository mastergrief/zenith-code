"""Offline A/B/C: score stock Gemma's outputs AS-IS, after AST rename,
and compare to DT numbers already on disk.

No Gemma calls. Reads stock_output from /tmp/dt_install_eval_results.json
(populated by dt_install_eval.py), runs each through rename_first_def,
re-scores via the same sandbox harness. Validates Option 2 (AST rename)
as an alternative to Option 1 (DT bias) without consuming additional
daemon time.

Usage:
  python3 scripts/dt_rename_offline_eval.py

Assumes:
  - /tmp/dt_install_eval_results.json exists (run dt_install_eval.py first)
  - agents/distill/data/mbpp.jsonl exists (for re-loading tests)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def load_mbpp_tests(limit: int):
    """Copy of dt_install_eval.load_mbpp but importable without daemon."""
    path = Path("agents/distill/data/mbpp.jsonl")
    out = []
    with path.open() as f:
        for i, line in enumerate(f):
            if len(out) >= limit:
                break
            d = json.loads(line)
            msgs = {m["role"]: m["content"] for m in d["messages"]}
            asst = msgs.get("assistant", "")
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
            fn_m = re.search(r"assert\s+(\w+)\s*\(", tests[0])
            if not fn_m:
                continue
            out.append({"idx": i, "fn_name": fn_m.group(1), "tests": tests})
    return out


_CODE_FENCE_RE = re.compile(r"```(?:python)?\n(.*?)\n```", re.DOTALL)


def _trim_to_first_def(code):
    # Pass 1: textual strip — drop col-0 `print(` or `# Example` trailers
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
    # Pass 2: AST-based trim (if parseable)
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


def extract_code(output, fn_name):
    """Same as scripts/dt_install_eval.py:extract_code — kept inline
    so this script is self-contained."""
    end_fence = output.find("```")
    if end_fence > 0:
        candidate = output[:end_fence].rstrip()
        if "def " in candidate or "class " in candidate:
            return _trim_to_first_def(candidate)
    m = _CODE_FENCE_RE.search(output)
    if m:
        return _trim_to_first_def(m.group(1))
    m = re.search(rf"def\s+{re.escape(fn_name)}\s*\(", output)
    if m is None:
        m = re.search(r"def\s+\w+\s*\(", output)
    if m is None:
        return None
    start = m.start()
    tail = output[start:]
    lines = tail.splitlines()
    out_lines = [lines[0]]
    for ln in lines[1:]:
        if ln.startswith("```"):
            break
        if ln and not ln.startswith((" ", "\t", "#", "@")) \
                and not ln.lstrip().startswith(("def ", "class ", "import ",
                                                "from ", "if __name__")):
            break
        out_lines.append(ln)
    return "\n".join(out_lines).rstrip()


def score(code, tests):
    from calm.sandbox import run_python
    if not code:
        return 0, len(tests), "no_code"
    harness = []
    for i, a in enumerate(tests):
        harness.append(
            f"try:\n"
            f"    {a}\n"
            f"    print('PASS {i}')\n"
            f"except Exception as _e:\n"
            f"    print('FAIL {i}: ' + type(_e).__name__)\n"
        )
    script = code + "\n\n" + "\n".join(harness) + "\npass\n"
    r = run_python(script, timeout=8.0)
    out = r.stdout or ""
    if r.error:
        return 0, len(tests), f"err:{str(r.error)[:60]}"
    passed = out.count("PASS ")
    fail_lines = [ln for ln in out.splitlines() if ln.startswith("FAIL")]
    diag = fail_lines[0][:80] if fail_lines else ""
    return passed, len(tests), diag


def main():
    from calm.llm_computer.facades.code_rename import rename_first_def

    results_path = Path("/tmp/dt_install_eval_results.json")
    if not results_path.exists():
        print(f"ERROR: {results_path} missing. Run dt_install_eval.py first.")
        sys.exit(1)

    dump = json.loads(results_path.read_text())
    problems = load_mbpp_tests(len(dump["rows"]))
    print(f"[offline] {len(dump['rows'])} stock outputs from {results_path}")
    print(f"[offline] {len(problems)} MBPP tests loaded")
    print()

    tot_stock = [0, 0]
    tot_rename = [0, 0]
    tot_dt = [0, 0]
    per_row = []

    for row, p in zip(dump["rows"], problems):
        assert row["fn"] == p["fn_name"], f"mismatch {row['fn']} vs {p['fn_name']}"
        fn_name = p["fn_name"]
        tests = p["tests"]

        stock_raw = row["stock_output"]
        # Re-score stock
        code_stock = extract_code(stock_raw, fn_name)
        s_pass, s_tot, s_diag = score(code_stock, tests)
        tot_stock[0] += s_pass
        tot_stock[1] += s_tot

        # RENAME condition: extract + rename + score
        # Apply rename to the RAW output first, then extract (so rename
        # sees the full text incl. any surrounding context).
        renamed_raw, orig = rename_first_def(stock_raw, fn_name)
        code_renamed = extract_code(renamed_raw, fn_name)
        r_pass, r_tot, r_diag = score(code_renamed, tests)
        tot_rename[0] += r_pass
        tot_rename[1] += r_tot

        # DT condition — reuse already-recorded score
        d_pass = int(row["dt"].split("/")[0])
        d_tot = int(row["dt"].split("/")[1])
        tot_dt[0] += d_pass
        tot_dt[1] += d_tot

        per_row.append({
            "fn": fn_name,
            "stock": f"{s_pass}/{s_tot}",
            "rename": f"{r_pass}/{r_tot}",
            "dt": f"{d_pass}/{d_tot}",
            "orig_name": orig,
            "did_rename": orig is not None and orig != fn_name,
            "rename_diag": r_diag,
        })
        delta_marker = " +" if r_pass > s_pass else (" -" if r_pass < s_pass else "  ")
        print(f"{fn_name:30s} stock={s_pass}/{s_tot} rename={r_pass}/{r_tot}{delta_marker} "
              f"dt={d_pass}/{d_tot} (orig={orig!r}, renamed={orig != fn_name if orig else 'n/a'})")

    print()
    print(f"=== Offline A/B/C on {len(per_row)} MBPP problems ===")
    s_p, s_t = tot_stock
    r_p, r_t = tot_rename
    d_p, d_t = tot_dt
    print(f"  stock:   {s_p}/{s_t} = {s_p/max(s_t,1):.2%}")
    print(f"  rename:  {r_p}/{r_t} = {r_p/max(r_t,1):.2%}  (delta vs stock: {r_p-s_p:+d})")
    print(f"  dt-bias: {d_p}/{d_t} = {d_p/max(d_t,1):.2%}  (delta vs stock: {d_p-s_p:+d})")
    print()

    rename_wins = [r for r in per_row
                   if int(r["rename"].split("/")[0]) > int(r["stock"].split("/")[0])]
    rename_reg = [r for r in per_row
                  if int(r["rename"].split("/")[0]) < int(r["stock"].split("/")[0])]
    print(f"  RENAME wins:        {len(rename_wins)}")
    print(f"  RENAME regressions: {len(rename_reg)}")

    if rename_wins:
        print("\n=== RENAME wins ===")
        for r in rename_wins:
            print(f"  {r['fn']}: stock={r['stock']} rename={r['rename']} "
                  f"(orig={r['orig_name']!r} → {r['fn']})")
    if rename_reg:
        print("\n=== RENAME regressions ===")
        for r in rename_reg:
            print(f"  {r['fn']}: stock={r['stock']} rename={r['rename']} "
                  f"(orig={r['orig_name']!r}, diag={r['rename_diag'][:60]})")

    print("\nDONE")


if __name__ == "__main__":
    main()
