"""R53.18a — minimal CALM-substrate intercept: sandbox-feedback retry loop.

The simplest possible CALM-style intervention. No substrate hook, no
PT, no per-token bias. Just:

  1. Gemma generates code (stock)
  2. Run tests via existing sandbox
  3. If pass → done
  4. If fail → build retry prompt with the failing-test output,
     regenerate, re-score
  5. Up to MAX_RETRIES (3)

Hypothesis: iterative refinement based on deterministic test feedback
lifts Gemma above its single-shot ceiling on the 6 R53.0 complex
problems. Specifically expect csv_column_stats and token_bucket
(which fail mid-output with NameError/TypeError) to recover with
ONE round of feedback.

If R53.18a wins (≥ stock + at least one previously-failing problem
recovered), validates the CALM-substrate architecture direction.
Then add per-token CALM hook (R53.18b) and PT spec extractor (R53.18c).

If R53.18a doesn't beat stock, sandbox feedback alone isn't enough —
need richer per-token intervention.

Daemon-only:
  bin/gemma-run scripts/r53_calm_substrate_retry.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List, Tuple


MAX_RETRIES = 2
MAX_TOKENS = 8192   # post-SWA-fix: no 512 cap, room for retrieval+reason+code


# Aggressively compact retry prompt — Gemma SWA caps total tokens at 512.
# Budget: prompt < 250 tokens, generation 250 tokens, total < 500.
RETRY_PROMPT_TEMPLATE = """\
Your code failed tests. Fix it.

Problem: {prompt}

Code:
```python
{prev_code}
```

Failures:
{test_output}

Output the corrected code as ```python``` block:
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
        CORPUS, gen_stock, score, extract_code,
        BASE_SYSTEM, _trim_markers,
    )
    from calm.llm_computer.gemma_substrate import KVCache
    from calm.sandbox import run_python

    # Detach any prior install state
    for layer in m.layers:
        if hasattr(layer, "card_slots"):
            layer.card_slots = []
    m.reserved_channels = []
    m.verification_hooks = []
    print("[r53.18a] cleared prior install state", flush=True)
    print(f"[r53.18a] MAX_RETRIES={MAX_RETRIES}, MAX_TOKENS={MAX_TOKENS}",
          flush=True)

    def get_test_output(code: str, test_code: str) -> str:
        """Run the bundled tests against the code, return stdout."""
        if not code:
            return "No extractable code"
        combined = code + "\n\n" + test_code + "\npass\n"
        result = run_python(combined, timeout=5.0)
        if result.error:
            return f"Runtime error: {result.error}\n{result.stdout or ''}"
        return result.stdout or "(no test output)"

    def gen_retry(p, prev_code: str, test_output: str) -> str:
        """Generate a corrected version after seeing test failures.
        Aggressively trims components to stay within Gemma's 512-token
        SWA window (prompt ≤ 250 tokens leaves ~250 for generation)."""
        # Trim each component
        problem_trim = p.prompt[:250]
        code_trim = prev_code[:300]
        out_trim = test_output[:200]
        retry_prompt = RETRY_PROMPT_TEMPLATE.format(
            prompt=problem_trim,
            prev_code=code_trim,
            test_output=out_trim,
        )
        out = m.generate(retry_prompt, tok, max_tokens=MAX_TOKENS,
                         device="cuda", stop_on_eos=True)
        return _trim_markers(out["text"])

    # Per-problem run with retry loop
    print(f"\n[r53.18a] running {len(CORPUS)} problems with retry...",
          flush=True)
    results: List[Tuple[str, int, int, int, int, int]] = []
    # (name, single-shot pass, total, after_retry pass, total, n_retries_used)

    for i, p in enumerate(CORPUS):
        print(f"\n[{i+1}/{len(CORPUS)}] {p.name}", flush=True)
        t0 = time.time()

        # SINGLE-SHOT (stock baseline this run)
        raw = gen_stock(m, tok, p, MAX_TOKENS)
        sp1, st1, _ = score(raw, p)
        print(f"  single-shot: {sp1}/{st1} ({time.time()-t0:.0f}s)",
              flush=True)

        # RETRY LOOP — pull code, run tests, feed back, regen
        best_pass, best_total = sp1, st1
        prev_raw = raw
        n_retries = 0
        for retry_idx in range(MAX_RETRIES):
            # Did we already pass everything? (when total > 0)
            if best_total > 0 and best_pass == best_total:
                break
            prev_code = extract_code(prev_raw, p.required)
            if not prev_code:
                # Can't retry without code to feed back — give up
                break
            test_output = get_test_output(prev_code, p.test_code)
            t1 = time.time()
            new_raw = gen_retry(p, prev_code, test_output)
            new_pass, new_total, _ = score(new_raw, p)
            n_retries += 1
            print(f"  retry {retry_idx+1}: {new_pass}/{new_total} "
                  f"({time.time()-t1:.0f}s)", flush=True)
            if new_total > best_total or (new_total == best_total
                                            and new_pass > best_pass):
                best_pass, best_total = new_pass, new_total
                prev_raw = new_raw
            else:
                # No improvement — stop retrying
                break

        results.append((p.name, sp1, st1, best_pass, best_total, n_retries))

    # ------------- AGGREGATE -------------
    print("\n" + "=" * 100, flush=True)
    print(f"  {'name':<28} {'single-shot':>13} {'after-retry':>13} "
          f"{'retries':>8} {'recovered':>10}", flush=True)
    print("-" * 100, flush=True)
    s_total = (0, 0)
    r_total = (0, 0)
    for name, sp, st, rp, rt, nr in results:
        recovered = "✓" if (rp/max(rt,1) > sp/max(st,1)) else (
            "=" if rp/max(rt,1) == sp/max(st,1) else "↓")
        print(f"  {name:<28} {sp:>4}/{st:<6}    {rp:>4}/{rt:<6}    "
              f"{nr:>4}     {recovered:>5}", flush=True)
        s_total = (s_total[0] + sp, s_total[1] + st)
        r_total = (r_total[0] + rp, r_total[1] + rt)
    print("-" * 100, flush=True)
    print(f"  {'TOTAL':<28} {s_total[0]:>4}/{s_total[1]:<6}    "
          f"{r_total[0]:>4}/{r_total[1]:<6}", flush=True)
    if s_total[1] and r_total[1]:
        delta = (r_total[0]/r_total[1] - s_total[0]/s_total[1]) * 100
        print(f"  Δ retry-vs-single-shot: {delta:+.1f}pp", flush=True)


if __name__ == "__main__":
    print("Daemon-only. Use:",
          "  bin/gemma-run scripts/r53_calm_substrate_retry.py",
          flush=True)
    sys.exit(1)
elif "m" in globals() and "tok" in globals():
    run_eval(m, tok)                                  # noqa: F821
