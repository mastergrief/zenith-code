"""R53.38 — Force-fence prefix test.

Purpose: test whether prepending a code-fence + function signature as
*prompt-tail* (not logit bias) lifts Gemma's NoCode branch on problems
where stock Gemma emits prose instead of code.

This is a distinct mechanism from R53.14's FirstTokenHook (ruled out —
forcing "def"/"class" via logit bias produces code-without-fence and
extractor fails). Here the fence AND the signature are in the prompt
context, so Gemma emits the body as an indented continuation — the
opening fence is guaranteed by construction.

Paired with the tier-2 AST walker: walker closed csv's SyntaxError
branch (R53.35, 0/0 → 8/8). This closes csv's NoCode branch if present.

Run via daemon (m, tok pre-bound):
  bin/gemma-run scripts/r53_38_force_fence.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path


# Daemon provides `m` and `tok` as globals; no re-load.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


FORCED_PROMPT = (
    "<start_of_turn>user\n"
    "{system}\n\n"
    "{prompt}\n\n"
    "Write ONLY the Python code — use ```python fencing. "
    "No explanation, no prose.\n"
    "<end_of_turn>\n"
    "<start_of_turn>model\n"
    "```python\n"
    "{signature}\n"
)


# Canonical signatures for R53.0 corpus problems that might need
# force-fence. Extend here as more NoCode-branch problems are
# identified. Signature syntax must match Python's `def` declaration
# exactly so Gemma's continuation is indented-body code.
SIGNATURES = {
    "csv_column_stats": "def csv_column_stats(text):",
    "log_level_counts": "def log_level_counts(lines):",
    "date_validation_chain": "def validate_date_chain(dates):",
}


def run_force_fence(target_name: str = "csv_column_stats",
                    max_tokens: int = 2048):
    from scripts.r53_eval_complex import (
        CORPUS, BASE_SYSTEM, score, extract_code,
    )
    from calm.sandbox import run_python

    # m, tok are daemon globals
    global m, tok

    prob = next(p for p in CORPUS if p.name == target_name)
    sig = SIGNATURES[target_name]

    prompt = FORCED_PROMPT.format(
        system=BASE_SYSTEM, prompt=prob.prompt, signature=sig)
    print(f"[r53.38] problem: {target_name}", flush=True)
    print(f"[r53.38] forced sig: {sig!r}", flush=True)
    print(f"[r53.38] prompt tail (last 120 chars):", flush=True)
    print(f"  ...{prompt[-120:]!r}", flush=True)

    t0 = time.time()
    out = m.generate(prompt, tok, max_tokens=max_tokens, device="cuda",
                     stop_on_eos=True)
    dt = time.time() - t0
    raw = out["text"]
    print(f"\n[r53.38] gen took {dt:.0f}s, {len(raw)} chars", flush=True)

    # Prepend the forced prefix back so the extractor sees the full
    # function (fence + sig + body). The prompt's tail `def csv_*(...):\n`
    # was part of Gemma's CONTEXT but not its output — glue them back.
    reconstructed = f"```python\n{sig}\n{raw}"

    # Trim post-fence turn markers
    for mark in ("<end_of_turn>", "<start_of_turn>"):
        i = reconstructed.find(mark)
        if i >= 0:
            reconstructed = reconstructed[:i]

    print(f"\n[r53.38] reconstructed (first 600 chars):", flush=True)
    print(reconstructed[:600], flush=True)
    print(f"\n[r53.38] ... (last 200 chars):", flush=True)
    print(reconstructed[-200:], flush=True)

    code = extract_code(reconstructed, list(prob.required))
    if not code:
        print(f"\n[r53.38] EXTRACT FAILED — NoCode even with force-fence",
              flush=True)
        print("[r53.38] DONE (null)", flush=True)
        return 0, 0

    print(f"\n[r53.38] extracted {len(code)} chars:", flush=True)
    print(code[:400], flush=True)
    if len(code) > 400:
        print("...", flush=True)

    passed, total, diag = score(reconstructed, prob)
    print(f"\n[r53.38] BASELINE SCORE: {passed}/{total}   diag={diag!r}",
          flush=True)

    # Apply walker chain on top — belt+suspenders. Try whenever we're
    # not clearly at a full pass: either tests failed, or sandbox
    # errored (total==0 with `err: ...` diag), or we have no score
    # signal at all. KeyError/TypeError/etc. during test exec show up
    # as `err: ...` and the walker may close the underlying bug.
    tried_walker = (passed > 0 and passed == total)
    if not tried_walker:
        print("\n[r53.38] trying AST walker on extracted code...", flush=True)
        # Pass the full diag text; walker regexes extract the key/type
        from calm.llm_computer.facades.ast_repair import repair
        rr = repair(code, diag)
        if rr.applied:
            print(f"[r53.38] walker applied {rr.kind}: {rr.notes}",
                  flush=True)
            script_ = rr.new_code + "\n\n" + prob.test_code + "\npass\n"
            r = run_python(script_, timeout=8.0)
            out_s = r.stdout or ""
            p2 = out_s.count("PASS")
            t2 = p2 + out_s.count("FAIL")
            err2 = r.error or ""
            if t2 == 0 and err2:
                # Still sandbox-erroring → try walker again with new error
                print(f"[r53.38] sandbox err after 1st walker: {err2[:80]!r}",
                      flush=True)
                rr2 = repair(rr.new_code, f"err: {err2}")
                if rr2.applied:
                    print(f"[r53.38] walker 2nd pass {rr2.kind}: {rr2.notes}",
                          flush=True)
                    script2 = (rr2.new_code + "\n\n" + prob.test_code
                               + "\npass\n")
                    r3 = run_python(script2, timeout=8.0)
                    out3 = r3.stdout or ""
                    p2 = out3.count("PASS")
                    t2 = p2 + out3.count("FAIL")
            print(f"[r53.38] AFTER WALKER: {p2}/{t2} PASS", flush=True)
            return p2, t2
        else:
            print(f"[r53.38] walker no-op: {rr.notes}", flush=True)

    return passed, total


# Daemon invokes as top-level exec — no __name__ gate.
passed, total = run_force_fence("csv_column_stats")
print(f"\n[r53.38] FINAL: {passed}/{total}", flush=True)
print("[r53.38] DONE", flush=True)
