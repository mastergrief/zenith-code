"""R53.38 — Force-fence prefix runner (generalized).

Purpose: prepend a fenced `def <name>(<args>):` to the prompt's
start-of-turn-model to lift Gemma's NoCode branch. Distinct from R53.14
first-token logit bias (ruled out) - here the fence AND signature are in
Gemma's CONTEXT, not injected via hook, so emission is indented-body by
construction.

R53.38v2 generalization (this commit): signature auto-derived from
the problem's 'required' list + prompt parse. Works on any
ComplexProblem without a hand-maintained SIGNATURES table.

Paired with the tier-2 AST walker cascade: force-fence closes NoCode
shape; walker cascade closes runtime bugs in the emitted code.

Run via daemon (m, tok pre-bound):
  bin/gemma-run scripts/r53_38_force_fence.py
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from typing import Optional


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# Which problem(s) to run. Override by editing or passing via RESET_GLOBALS.
# Names from r53_eval_complex.CORPUS.
TARGETS = [
    "csv_column_stats",      # known NoCode (R53.38v1 already: 0/0 → 8/8)
]


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


# Fallback hints when signature auto-derivation misses. Used ONLY
# when derive_signature() can't reconstruct a valid `def` line.
# Prefer letting derivation work; this is a belt+suspenders safety net.
MANUAL_OVERRIDES = {
    # name -> signature   (only fill when derivation is wrong)
}


def _derive_signature_from_prompt(prompt: str, fn_name: str) -> Optional[str]:
    """Find the function signature in the problem prompt. Looks for
    `def <fn_name>(<args>):` explicit pattern, or backtick-wrapped
    `<fn_name>(<args>)` shorthand. Returns the full `def ...:` line,
    or None when neither pattern matches.
    """
    # Pattern 1: explicit `def fn_name(args):` in prompt
    m = re.search(
        rf"def\s+{re.escape(fn_name)}\s*\([^)]*\)\s*(?:->\s*[^\s:]+\s*)?:",
        prompt)
    if m:
        return m.group(0)

    # Pattern 2: backtick-wrapped `fn_name(args)` — infer `:` + leading `def`
    m = re.search(
        rf"`{re.escape(fn_name)}\s*\(([^)]*)\)`", prompt)
    if m:
        args = m.group(1)
        return f"def {fn_name}({args}):"

    return None


def _signature_for(prob, fn_name: str) -> str:
    """Best-effort signature. Priority:
    1. MANUAL_OVERRIDES table
    2. Parse from prompt text
    3. Fallback: `def <fn_name>(text):` (csv-style default)
    """
    if fn_name in MANUAL_OVERRIDES:
        return MANUAL_OVERRIDES[fn_name]

    sig = _derive_signature_from_prompt(prob.prompt, fn_name)
    if sig:
        return sig

    # Generic fallback for string-input problems
    return f"def {fn_name}(text):"


def run_force_fence(target_name: str, max_tokens: int = 2048):
    from scripts.r53_eval_complex import (
        CORPUS, BASE_SYSTEM, score, extract_code,
    )
    from calm.sandbox import run_python
    from calm.llm_computer.facades.ast_repair import repair_cascade

    # m, tok are daemon globals
    global m, tok

    prob = next(p for p in CORPUS if p.name == target_name)

    # Derive signature: prefer problem.required[0] as the fn name
    if not prob.required:
        print(f"[r53.38] SKIP {target_name} — no required names", flush=True)
        return 0, 0
    fn_name = prob.required[0]
    sig = _signature_for(prob, fn_name)

    prompt = FORCED_PROMPT.format(
        system=BASE_SYSTEM, prompt=prob.prompt, signature=sig)
    print(f"[r53.38] problem: {target_name}", flush=True)
    print(f"[r53.38] derived sig: {sig!r}", flush=True)
    print(f"[r53.38] prompt tail (last 120 chars):", flush=True)
    print(f"  ...{prompt[-120:]!r}", flush=True)

    t0 = time.time()
    out = m.generate(prompt, tok, max_tokens=max_tokens, device="cuda",
                     stop_on_eos=True)
    dt = time.time() - t0
    raw = out["text"]
    print(f"\n[r53.38] gen took {dt:.0f}s, {len(raw)} chars", flush=True)

    # Prepend the forced prefix back so the extractor sees the full
    # function (fence + sig + body).
    reconstructed = f"```python\n{sig}\n{raw}"

    # Trim post-fence turn markers
    for mark in ("<end_of_turn>", "<start_of_turn>"):
        i = reconstructed.find(mark)
        if i >= 0:
            reconstructed = reconstructed[:i]

    print(f"\n[r53.38] reconstructed (first 400 chars):", flush=True)
    print(reconstructed[:400], flush=True)

    code = extract_code(reconstructed, list(prob.required))
    if not code:
        print(f"\n[r53.38] EXTRACT FAILED — NoCode even with force-fence",
              flush=True)
        return 0, 0

    print(f"\n[r53.38] extracted {len(code)} chars", flush=True)
    passed, total, diag = score(reconstructed, prob)
    print(f"[r53.38] BASELINE SCORE: {passed}/{total}   diag={diag!r}",
          flush=True)

    # Apply walker cascade on non-clean results
    if not (passed > 0 and passed == total):
        print("\n[r53.38] trying walker cascade on extracted code...",
              flush=True)
        rr = repair_cascade(code, diag, max_passes=4)
        if rr.applied:
            print(f"[r53.38] cascade applied {rr.kind}", flush=True)
            for note in rr.notes:
                print(f"  - {note}", flush=True)
            script_ = rr.new_code + "\n\n" + prob.test_code + "\npass\n"
            r = run_python(script_, timeout=8.0)
            out_s = r.stdout or ""
            p2 = out_s.count("PASS")
            t2 = p2 + out_s.count("FAIL")
            err2 = r.error or ""
            if t2 == 0 and err2:
                # Second cascade with fresh error — in case a rewrite
                # applied one fix and the re-run surfaced another error
                print(f"[r53.38] sandbox err after cascade: {err2[:80]!r}",
                      flush=True)
                rr2 = repair_cascade(rr.new_code, f"err: {err2}",
                                     max_passes=4)
                if rr2.applied:
                    print(f"[r53.38] 2nd cascade {rr2.kind}", flush=True)
                    script2 = (rr2.new_code + "\n\n" + prob.test_code
                               + "\npass\n")
                    r3 = run_python(script2, timeout=8.0)
                    out3 = r3.stdout or ""
                    p2 = out3.count("PASS")
                    t2 = p2 + out3.count("FAIL")
            print(f"[r53.38] AFTER CASCADE: {p2}/{t2} PASS", flush=True)
            return p2, t2
        else:
            print(f"[r53.38] cascade no-op: {rr.notes}", flush=True)

    return passed, total


# Daemon exec at top-level — skip when imported (no m/tok globals).
if "m" in globals() and "tok" in globals():
    results = []
    for target in TARGETS:
        print("\n" + "=" * 72, flush=True)
        p, t = run_force_fence(target)
        results.append((target, p, t))

    print("\n" + "=" * 72, flush=True)
    print("[r53.38] SUMMARY", flush=True)
    for name, p, t in results:
        verdict = "PASS" if (t > 0 and p == t) else "FAIL"
        print(f"  {name:30s}  {p:2d}/{t:2d}  {verdict}", flush=True)
    print("[r53.38] DONE", flush=True)
