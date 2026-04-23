"""Offline A/B: score stock Gemma's saved outputs AS-IS vs
RETRIEVAL-SIGNATURE-renamed. Validates Option 1 (retrieval) as an
alternative to Option 2 (caller-supplied name rename), using the
same stock output pool as dt_rename_offline_eval.py.

Needs the daemon for dense retrieval (dense-encoding the query
requires loaded Gemma + tokenizer). Sends ONE prompt-encoding call
per problem, no fresh Gemma code generation.

Usage (via daemon):
  bin/gemma-run scripts/dt_retrieval_offline_eval.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def load_mbpp_tests(limit: int):
    path = Path("agents/distill/data/mbpp.jsonl")
    out = []
    with path.open() as f:
        for i, line in enumerate(f):
            if len(out) >= limit:
                break
            d = json.loads(line)
            msgs = {m["role"]: m["content"] for m in d["messages"]}
            user = msgs.get("user", "").strip()
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
            out.append({
                "idx": i,
                "prompt": user,
                "expected_fn_name": fn_m.group(1),
                "tests": tests,
            })
    return out


# Copied from dt_install_eval.py — self-contained
_CODE_FENCE_RE = re.compile(r"```(?:python)?\n(.*?)\n```", re.DOTALL)


def _trim_to_first_def(code):
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
            f"try:\n    {a}\n    print('PASS {i}')\n"
            f"except Exception as _e:\n    print('FAIL {i}: ' + type(_e).__name__)\n"
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
    if "m" not in globals() or "tok" not in globals():
        print("ERROR: expected m, tok globals from gemma_daemon")
        sys.exit(1)

    # Purge cached facade modules so edits on disk take effect in the
    # long-running daemon. Without this, sys.modules holds the module
    # loaded at first import.
    for modname in list(sys.modules.keys()):
        if modname.startswith("calm.llm_computer.facades.code_retrieval") \
                or modname.startswith("calm.llm_computer.facades.code_rename") \
                or modname.startswith("calm.llm_computer.facades.code_example_db") \
                or modname.startswith("calm.llm_computer.facades.retrieval"):
            del sys.modules[modname]

    from calm.llm_computer.facades.code_retrieval_signature import (
        CodeRetrievalSignatureFacade,
    )
    from calm.llm_computer.facades.code_rename import rename_first_def

    results_path = Path("/tmp/dt_install_eval_results.json")
    if not results_path.exists():
        print(f"ERROR: {results_path} missing. Run dt_install_eval.py first.")
        sys.exit(1)

    dump = json.loads(results_path.read_text())
    problems = load_mbpp_tests(len(dump["rows"]))
    print(f"[retrieval-offline] {len(dump['rows'])} stock outputs")
    print(f"[retrieval-offline] {len(problems)} MBPP tests loaded")

    facade = CodeRetrievalSignatureFacade()
    facade.install(m, tok)
    print(f"[retrieval-offline] DB ready "
          f"(tfidf={facade._db.has_tfidf() if hasattr(facade._db,'has_tfidf') else '?'}, "
          f"dense=?)", flush=True)
    print()

    tot_stock = [0, 0]
    tot_oracle = [0, 0]      # rename using ground-truth fn_name (ceiling)
    tot_retr_on = [0, 0]     # retrieval WITH self-match allowed
    tot_retr_off = [0, 0]    # retrieval with self-match EXCLUDED
    tot_dt = [0, 0]

    per_row = []

    for row, p in zip(dump["rows"], problems):
        assert row["fn"] == p["expected_fn_name"]
        fn_name = p["expected_fn_name"]
        tests = p["tests"]
        prompt = p["prompt"]
        stock_raw = row["stock_output"]

        # Stock
        sp, st, _ = score(extract_code(stock_raw, fn_name), tests)
        tot_stock[0] += sp; tot_stock[1] += st

        # Oracle (caller-supplied fn_name) — upper bound for rename
        renamed_oracle, _ = rename_first_def(stock_raw, fn_name)
        op, ot, _ = score(extract_code(renamed_oracle, fn_name), tests)
        tot_oracle[0] += op; tot_oracle[1] += ot

        # Retrieval: predict name w/ self-match allowed
        pred_on, args_on, score_on, src_on, snip_on = facade.predict_signature(
            prompt, exclude_self_match=False)
        if pred_on:
            renamed_on, _ = rename_first_def(stock_raw, pred_on)
            r_on_p, r_on_t, _ = score(extract_code(renamed_on, pred_on), tests)
        else:
            r_on_p, r_on_t, _ = sp, st, "no_retrieval"
        tot_retr_on[0] += r_on_p; tot_retr_on[1] += r_on_t

        # Retrieval: predict name w/ self-match EXCLUDED (honest mode)
        pred_off, args_off, score_off, src_off, snip_off = facade.predict_signature(
            prompt, exclude_self_match=True)
        if pred_off:
            renamed_off, _ = rename_first_def(stock_raw, pred_off)
            r_off_p, r_off_t, _ = score(extract_code(renamed_off, pred_off), tests)
        else:
            r_off_p, r_off_t, _ = sp, st, "no_retrieval"
        tot_retr_off[0] += r_off_p; tot_retr_off[1] += r_off_t

        # DT (already scored)
        d_p = int(row["dt"].split("/")[0]); d_t = int(row["dt"].split("/")[1])
        tot_dt[0] += d_p; tot_dt[1] += d_t

        per_row.append({
            "fn": fn_name, "stock": f"{sp}/{st}", "oracle": f"{op}/{ot}",
            "retr_on": f"{r_on_p}/{r_on_t}", "retr_off": f"{r_off_p}/{r_off_t}",
            "dt": f"{d_p}/{d_t}",
            "pred_on": (pred_on, float(score_on)),
            "pred_off": (pred_off, float(score_off)),
            "src_off": src_off, "snip_off": (snip_off or "")[:60],
        })

        delta_markers = (
            " +" if r_off_p > sp else (" -" if r_off_p < sp else "  "))
        pred_str = f"{pred_off!r}@{score_off:.2f}" if pred_off else "<none>"
        print(f"{fn_name:30s} stock={sp}/{st} oracle={op}/{ot} "
              f"retr_self={r_on_p}/{r_on_t} retr_ex={r_off_p}/{r_off_t}{delta_markers} "
              f"pred_ex={pred_str}", flush=True)

    print()
    print(f"=== Retrieval A/B/C/D on {len(per_row)} MBPP problems ===")
    for name, tot in [("stock    ", tot_stock), ("oracle-rename", tot_oracle),
                      ("retr-self-ok ", tot_retr_on),
                      ("retr-self-skip", tot_retr_off),
                      ("dt-bias  ", tot_dt)]:
        p, t = tot
        delta = p - tot_stock[0]
        print(f"  {name}: {p}/{t} = {p/max(t,1):.2%}   delta vs stock: {delta:+d}")

    dump_out = Path("/tmp/dt_retrieval_offline_eval.json")
    with dump_out.open("w") as f:
        json.dump({"totals": {
            "stock": tot_stock, "oracle": tot_oracle,
            "retr_self_ok": tot_retr_on, "retr_self_skip": tot_retr_off,
            "dt": tot_dt,
        }, "rows": per_row}, f, indent=2)
    print(f"\n[retrieval-offline] forensic dump → {dump_out}")
    print("DONE")


main()
