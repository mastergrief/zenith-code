"""Offline A/B/C scoring for DT install dumps.

MBPP mode preserves the original workflow: read stock_output from
/tmp/dt_install_eval_results.json, apply deterministic RENAME to stock output,
and compare stock / rename / recorded DT scores.

HumanEvalPlus mode reads /tmp/he_install_eval_results.json (or HE_RESULTS_PATH),
uses the raw test metadata serialized by scripts/dt_install_eval.py, and scores
stock / rename / dt outputs with the same prompt-prepend retry needed for HE+
body-only generations. No Gemma calls are made.

Usage:
  python3 scripts/dt_rename_offline_eval.py
  EVAL_BENCHMARK=humanevalplus python3 scripts/dt_rename_offline_eval.py
  EVAL_BENCHMARK=humanevalplus HE_RESULTS_PATH=/tmp/he_install_eval_n5_smoke_prefix_results.json python3 scripts/dt_rename_offline_eval.py
"""
from __future__ import annotations

import ast
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


EVAL_BENCHMARK = os.environ.get("EVAL_BENCHMARK", "mbpp").lower()
if EVAL_BENCHMARK not in ("mbpp", "humanevalplus"):
    raise SystemExit(
        f"EVAL_BENCHMARK must be 'mbpp' or 'humanevalplus', got {EVAL_BENCHMARK!r}"
    )

MBPP_RESULTS_PATH = Path(os.environ.get(
    "DT_RESULTS_PATH", "/tmp/dt_install_eval_results.json"))
HE_RESULTS_PATH = Path(os.environ.get(
    "HE_RESULTS_PATH", "/tmp/he_install_eval_results.json"))


# ---------------------------------------------------------------
# MBPP loader + scorer (original path)
# ---------------------------------------------------------------

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


def _trim_to_first_def(code: str) -> str:
    # Pass 1: textual strip: drop col-0 print/example trailers.
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

    # Pass 2: AST trim (if parseable).
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    if not tree.body:
        return code
    first = tree.body[0]
    if not isinstance(first, (ast.FunctionDef, ast.ClassDef,
                              ast.AsyncFunctionDef)):
        return code
    end_line = getattr(first, "end_lineno", None)
    if end_line is None:
        return code
    lines2 = code.splitlines(keepends=True)
    return "".join(lines2[:end_line])


def _trim_keep_top_level_code(code: str) -> str:
    """HE+-aware trim: keep consecutive imports/defs/classes.

    Some HE+ tasks need helper functions before the target function. The MBPP
    first-def trim would drop those helpers, so HE+ uses this wider trim.
    """
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

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    if not tree.body:
        return code
    keep = (ast.Import, ast.ImportFrom, ast.FunctionDef,
            ast.AsyncFunctionDef, ast.ClassDef)
    last_end = None
    for node in tree.body:
        if isinstance(node, keep):
            end_line = getattr(node, "end_lineno", None)
            if end_line is not None:
                last_end = end_line
        else:
            break
    if last_end is None:
        return code
    lines2 = code.splitlines(keepends=True)
    return "".join(lines2[:last_end])


def extract_code(output: str, fn_name: str) -> Optional[str]:
    """Extract Python code from Gemma output (MBPP/original behavior)."""
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


def extract_code_he_plus(output: str, fn_name: str) -> Optional[str]:
    """Extract HE+ code while preserving helper defs before target."""
    end_fence = output.find("```")
    if end_fence > 0:
        candidate = output[:end_fence].rstrip()
        if "def " in candidate or "class " in candidate:
            return _trim_keep_top_level_code(candidate)
    m = _CODE_FENCE_RE.search(output)
    if m:
        return _trim_keep_top_level_code(m.group(1))
    return extract_code(output, fn_name)


def score_mbpp(code: Optional[str], tests: List[str]):
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


# ---------------------------------------------------------------
# HumanEvalPlus scorer (self-contained, no daemon imports)
# ---------------------------------------------------------------

def _test_defines_assertion(test_code: str) -> bool:
    try:
        tree = ast.parse(test_code)
    except SyntaxError:
        return False
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == "assertion":
                return True
    return False


def _score_humaneval_code(code: Optional[str], row: dict):
    from calm.sandbox import run_python

    inputs = row.get("inputs") or []
    total = len(inputs)
    fn_name = row["fn"]
    if not code:
        return 0, total, "no_code", ["FAIL: no_code"] * total

    use_per_input = _test_defines_assertion(row["test_code"])
    parts = [code, "", row["test_code"], ""]

    if use_per_input:
        parts.append(f"_HE_INPUTS = {inputs!r}")
        if row.get("results") is not None:
            parts.append(f"_HE_RESULTS = {row['results']!r}")
        parts.append("")
        if row.get("results") is not None:
            parts.append(
                f"for _i, _inp in enumerate(_HE_INPUTS):\n"
                f"    try:\n"
                f"        _out = {fn_name}(*_inp)\n"
                f"        assertion(_out, _HE_RESULTS[_i], 0)\n"
                f"        print('PASS ' + str(_i))\n"
                f"    except Exception as _e:\n"
                f"        print('FAIL ' + str(_i) + ': ' + type(_e).__name__)"
            )
        else:
            parts.append(
                f"for _i, _inp in enumerate(_HE_INPUTS):\n"
                f"    try:\n"
                f"        _exp = ref_func(*_inp)\n"
                f"        _out = {fn_name}(*_inp)\n"
                f"        assertion(_out, _exp, 0)\n"
                f"        print('PASS ' + str(_i))\n"
                f"    except Exception as _e:\n"
                f"        print('FAIL ' + str(_i) + ': ' + type(_e).__name__)"
            )
    else:
        parts.append(
            f"try:\n"
            f"    check({fn_name})\n"
            f"    print('CHECK_PASS')\n"
            f"except AssertionError as _e:\n"
            f"    print('CHECK_FAIL: AssertionError')\n"
            f"except Exception as _e:\n"
            f"    print('CHECK_ERROR: ' + type(_e).__name__)"
        )
    parts.append("pass")
    script = "\n".join(parts)

    r = run_python(script, timeout=30.0, extra_preimports=["numpy"])
    out = r.stdout or ""
    if r.error:
        return 0, total, f"err:{str(r.error)[:60]}", [f"FAIL: {str(r.error)[:60]}"] * total

    if use_per_input:
        per_input = ["FAIL: NoOutput"] * total
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


def score_humaneval_output(output: str, row: dict):
    """Score HE+ output with prompt-prepend retry for body-only generations."""
    code = extract_code_he_plus(output, row["fn"])
    if not code:
        code = extract_code_he_plus(row["prompt"] + "\n" + output, row["fn"])
    return _score_humaneval_code(code, row)


def rename_humaneval_output(output: str, row: dict, rename_first_def):
    """Apply RENAME in the same context HE+ scoring uses.

    If the model emitted a def, rename raw output. If it emitted body-only,
    rename prompt+output so the prompt-carried signature is measured as a
    deliberate no-op instead of causing a no-code zero.
    """
    fn_name = row["fn"]
    renamed, orig = rename_first_def(output, fn_name)
    used_prompt_context = False
    if orig is None:
        used_prompt_context = True
        renamed, orig = rename_first_def(row["prompt"] + "\n" + output, fn_name)
    did_rename = orig is not None and orig != fn_name
    return renamed, orig, did_rename, used_prompt_context


def _parse_cell(cell: str):
    try:
        p, t = cell.split("/", 1)
        return int(p), int(t)
    except (ValueError, AttributeError):
        return 0, 0


def _add_metrics(metrics: dict, key: str, passed: int, total: int) -> None:
    metrics[key]["pass"] += passed
    metrics[key]["total"] += total
    metrics[key]["all_pass"] += int(total > 0 and passed == total)
    metrics[key]["any_pass"] += int(passed > 0)
    metrics[key]["macro_sum"] += passed / max(total, 1)


def _print_he_summary(metrics: dict, n: int) -> None:
    print()
    print(f"=== Offline A/B/C on {n} HumanEvalPlus problems ===")
    for key in ("stock", "rename", "dt"):
        m = metrics[key]
        macro = m["macro_sum"] / max(n, 1)
        micro = m["pass"] / max(m["total"], 1)
        print(
            f"  {key:7s} all={m['all_pass']}/{n} "
            f"any={m['any_pass']}/{n} macro={macro:.4f} "
            f"micro={m['pass']}/{m['total']}={micro:.2%}"
        )
    print(f"  rename delta: all={metrics['rename']['all_pass'] - metrics['stock']['all_pass']:+d} "
          f"any={metrics['rename']['any_pass'] - metrics['stock']['any_pass']:+d} "
          f"macro={(metrics['rename']['macro_sum'] - metrics['stock']['macro_sum']) / max(n, 1):+.4f}")
    print(f"  dt delta:     all={metrics['dt']['all_pass'] - metrics['stock']['all_pass']:+d} "
          f"any={metrics['dt']['any_pass'] - metrics['stock']['any_pass']:+d} "
          f"macro={(metrics['dt']['macro_sum'] - metrics['stock']['macro_sum']) / max(n, 1):+.4f}")


# ---------------------------------------------------------------
# Entrypoints
# ---------------------------------------------------------------

def run_mbpp_offline() -> None:
    from calm.llm_computer.facades.code_rename import rename_first_def

    results_path = MBPP_RESULTS_PATH
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
        code_stock = extract_code(stock_raw, fn_name)
        s_pass, s_tot, s_diag = score_mbpp(code_stock, tests)
        tot_stock[0] += s_pass
        tot_stock[1] += s_tot

        renamed_raw, orig = rename_first_def(stock_raw, fn_name)
        code_renamed = extract_code(renamed_raw, fn_name)
        r_pass, r_tot, r_diag = score_mbpp(code_renamed, tests)
        tot_rename[0] += r_pass
        tot_rename[1] += r_tot

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
                  f"(orig={r['orig_name']!r} -> {r['fn']})")
    if rename_reg:
        print("\n=== RENAME regressions ===")
        for r in rename_reg:
            print(f"  {r['fn']}: stock={r['stock']} rename={r['rename']} "
                  f"(orig={r['orig_name']!r}, diag={r['rename_diag'][:60]})")

    print("\nDONE")


def run_humanevalplus_offline() -> None:
    from calm.llm_computer.facades.code_rename import rename_first_def

    results_path = HE_RESULTS_PATH
    if not results_path.exists():
        print(f"ERROR: {results_path} missing. Run HE+ dt_install_eval first.")
        sys.exit(1)

    dump = json.loads(results_path.read_text())
    if dump.get("benchmark") != "humanevalplus":
        print(f"ERROR: {results_path} is not a HumanEvalPlus dump")
        sys.exit(1)

    rows = dump.get("rows", [])
    print(f"[offline-he] {len(rows)} rows from {results_path}")
    print(f"[offline-he] live totals: {dump.get('totals')}")
    print()

    metrics = {
        "stock": {"pass": 0, "total": 0, "all_pass": 0, "any_pass": 0, "macro_sum": 0.0},
        "rename": {"pass": 0, "total": 0, "all_pass": 0, "any_pass": 0, "macro_sum": 0.0},
        "dt": {"pass": 0, "total": 0, "all_pass": 0, "any_pass": 0, "macro_sum": 0.0},
    }
    per_row = []
    mismatches = []

    for i, row in enumerate(rows):
        fn_name = row["fn"]

        s_pass, s_tot, s_diag, _s_per = score_humaneval_output(row["stock_output"], row)

        renamed_input, orig, did_rename, used_prompt_context = rename_humaneval_output(
            row["stock_output"], row, rename_first_def)
        r_pass, r_tot, r_diag, _r_per = score_humaneval_output(renamed_input, row)

        d_pass, d_tot, d_diag, _d_per = score_humaneval_output(row["dt_output"], row)
        recorded_d = _parse_cell(row.get("dt", "0/0"))
        if recorded_d != (d_pass, d_tot):
            mismatches.append((row["task_id"], row.get("dt"), f"{d_pass}/{d_tot}"))

        _add_metrics(metrics, "stock", s_pass, s_tot)
        _add_metrics(metrics, "rename", r_pass, r_tot)
        _add_metrics(metrics, "dt", d_pass, d_tot)

        per_row.append({
            "task_id": row["task_id"],
            "fn": fn_name,
            "stock": f"{s_pass}/{s_tot}",
            "rename": f"{r_pass}/{r_tot}",
            "dt": f"{d_pass}/{d_tot}",
            "orig_name": orig,
            "did_rename": did_rename,
            "used_prompt_context": used_prompt_context,
            "rename_diag": r_diag,
            "stock_diag": s_diag,
            "dt_diag": d_diag,
        })

        marker = " +" if r_pass > s_pass else (" -" if r_pass < s_pass else "  ")
        print(f"[{i+1}/{len(rows)}] {row['task_id']:12s} {fn_name:30s} "
              f"stock={s_pass:4d}/{s_tot:4d} rename={r_pass:4d}/{r_tot:4d}{marker} "
              f"dt={d_pass:4d}/{d_tot:4d} "
              f"orig={orig!r} prompt_ctx={used_prompt_context}")

    _print_he_summary(metrics, len(rows))

    rename_wins = [_r for _r in per_row
                   if _parse_cell(_r["rename"])[0] > _parse_cell(_r["stock"])[0]]
    rename_reg = [_r for _r in per_row
                  if _parse_cell(_r["rename"])[0] < _parse_cell(_r["stock"])[0]]
    dt_wins = [_r for _r in per_row
               if _parse_cell(_r["dt"])[0] > _parse_cell(_r["stock"])[0]]
    dt_reg = [_r for _r in per_row
              if _parse_cell(_r["dt"])[0] < _parse_cell(_r["stock"])[0]]
    no_op = sum(1 for _r in per_row if not _r["did_rename"])

    print()
    print(f"  RENAME wins:        {len(rename_wins)}")
    print(f"  RENAME regressions: {len(rename_reg)}")
    print(f"  RENAME no-op rows:  {no_op}/{len(per_row)}")
    print(f"  DT wins:            {len(dt_wins)}")
    print(f"  DT regressions:     {len(dt_reg)}")
    if mismatches:
        print(f"  recorded DT mismatches after offline re-score: {len(mismatches)}")
        for task_id, old, new in mismatches[:10]:
            print(f"    {task_id}: recorded={old} offline={new}")

    if rename_wins:
        print("\n=== RENAME wins ===")
        for r in rename_wins[:10]:
            print(f"  {r['task_id']} {r['fn']}: stock={r['stock']} rename={r['rename']} "
                  f"orig={r['orig_name']!r}")
    if rename_reg:
        print("\n=== RENAME regressions ===")
        for r in rename_reg[:10]:
            print(f"  {r['task_id']} {r['fn']}: stock={r['stock']} rename={r['rename']} "
                  f"orig={r['orig_name']!r} diag={r['rename_diag'][:60]}")

    print("\nDONE")


def main() -> None:
    if EVAL_BENCHMARK == "humanevalplus":
        run_humanevalplus_offline()
    else:
        run_mbpp_offline()


if __name__ == "__main__":
    main()
