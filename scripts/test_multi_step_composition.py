"""Round 50.7: MultiStepCompositionFacade end-to-end on prod Gemma.

Hypothesis: promoting R46's safe_eval-only multiplication step to a
compiled multiplier card while keeping step-through digit bias for
delivery produces strictly-improving behavior vs R46 — same fix rate
on baseline failures, zero regressions, plus a `substrate_native`
flag on each result that records when the multiplication was served
by the compiled card (3390/3390 on a*b < 1000) instead of Python.

Scope: (a * b) + c and (a * b) - c with a*b < 1000, c ≥ 0. This is
the tier-3 narrowing of R46's general infix pipeline — we trade
breadth for a provably-verified multiplication step.

Two prompt cohorts:

  COMPOSITION_PROMPTS (20): (a * b) + c and (a * b) - c. Baselined
    on stock Gemma, the multiplication errors from R11 (17×23, 47×19,
    45×15) propagate into the composition step, so this is the
    facade's failure surface.

  REGRESSION_PROMPTS (10): additive-only chains Gemma gets right.
    Must NOT regress. Gate: zero regressions.

Gates:
  - parse coverage on composition set  ≥ 18/20
  - substrate-native coverage on parses ≥ 18/20  (multiplier in-range)
  - baseline-wrong fix rate             ≥ 80%
  - regression set: zero regressions

Prompt format matches test_multi_step_reasoning.py: trailing "Answer:"
or ":" so Gemma emits a short numeric continuation rather than a
verbose chain-of-thought.
"""

from __future__ import annotations

import os
import re
import sys


# (prompt, expected_integer_answer) — composition surface.
COMPOSITION_PROMPTS = [
    # --- (a * b) + c: R11 multiplication-failure operand pairs ---
    ("What is 17 * 23 + 5? Answer: ",          17 * 23 + 5),
    ("Compute 47 * 19 + 23: ",                 47 * 19 + 23),
    ("What is 45 * 15 + 8? Answer: ",          45 * 15 + 8),
    ("Compute 37 * 14 + 50: ",                 37 * 14 + 50),
    ("What is 29 * 13 + 7? ",                  29 * 13 + 7),
    ("Compute 23 * 17 + 10: ",                 23 * 17 + 10),
    ("What is 19 * 31 + 15? Answer: ",         19 * 31 + 15),
    ("Compute 41 * 13 + 20: ",                 41 * 13 + 20),
    # --- (a * b) - c ---
    ("What is 45 * 15 - 8? Answer: ",          45 * 15 - 8),
    ("Compute 29 * 13 - 7: ",                  29 * 13 - 7),
    ("What is 47 * 19 - 23? Answer: ",         47 * 19 - 23),
    ("Compute 37 * 14 - 50: ",                 37 * 14 - 50),
    # --- NL aliases: times/plus/minus ---
    ("What is 17 times 23 plus 5? Answer: ",   17 * 23 + 5),
    ("What is 47 times 19 plus 23? ",          47 * 19 + 23),
    ("Compute 45 times 15 minus 8: ",          45 * 15 - 8),
    ("What is 29 times 13 plus 7? Answer: ",   29 * 13 + 7),
    ("What is 17 multiplied by 23 plus 5? ",   17 * 23 + 5),
    # --- Unicode ×, explicit x ---
    ("What is 17×23+5? Answer: ",              17 * 23 + 5),
    ("Compute 47×19+23: ",                     47 * 19 + 23),
    ("What is 17x23+5? Answer: ",              17 * 23 + 5),
]


# Additive-only — Gemma typically correct. Zero regressions required.
REGRESSION_PROMPTS = [
    ("What is 100 + 50? Answer: ",             150),
    ("Compute 250 + 75: ",                     325),
    ("What is 1000 - 250? Answer: ",           750),
    ("Compute 500 - 125: ",                    375),
    ("What is 42 + 58? Answer: ",              100),
    ("Compute 200 - 50: ",                     150),
    ("What is 15 + 25? Answer: ",              40),
    ("Compute 300 - 100: ",                    200),
    ("What is 7 + 8? Answer: ",                15),
    ("Compute 99 - 33: ",                      66),
]


def first_int(text: str):
    m = re.search(r"-?\d+", text.replace(",", ""))
    return int(m.group(0)) if m else None


def run_cohort(facade, prompts, label, print_header=True):
    """Run a cohort through baseline (no bias) and facade paths, return
    per-prompt stats."""
    if print_header:
        print(f"\n--- {label} ({len(prompts)} prompts) ---")
        print(f"{'prompt':>45}  {'expected':>10}  {'base':>8}  "
              f"{'facade':>8}  base  facade  parse  native")
        print("-" * 115)

    rows = []
    for prompt, expected in prompts:
        base = facade.generate(prompt, use_bias=False)
        base_answer = first_int(base.generated_text)

        fac = facade.generate(prompt)
        fac_answer = first_int(fac.generated_text)

        parse_ok = fac.operands is not None
        native_ok = fac.substrate_native
        base_ok = base_answer == expected
        fac_ok = fac_answer == expected

        short = prompt if len(prompt) <= 43 else prompt[:40] + "..."
        print(f"{short!r:>45}  {expected:>10}  "
              f"{str(base_answer):>8}  {str(fac_answer):>8}  "
              f"  {'v' if base_ok else 'x'}      "
              f"{'v' if fac_ok else 'x'}      "
              f"{'v' if parse_ok else 'x'}      "
              f"{'v' if native_ok else 'x'}")

        rows.append({
            "prompt": prompt,
            "expected": expected,
            "base_answer": base_answer,
            "fac_answer": fac_answer,
            "base_ok": base_ok,
            "fac_ok": fac_ok,
            "parse_ok": parse_ok,
            "native_ok": native_ok,
        })
    return rows


def main():
    import torch  # noqa: F401
    from calm.llm_computer.gemma_substrate import (
        GemmaSubstrate, enable_triton_tq4)
    from calm.llm_computer.synth.gemma_tokenizer import GemmaTokenizer
    from calm.llm_computer.facades.multi_step_composition import (
        MultiStepCompositionFacade)

    gguf = os.path.expanduser("~/models/gemma-4-E4B-it-tq4-aligned.gguf")
    enable_triton_tq4(True)
    print("[r50.7] loading substrate...")
    m = GemmaSubstrate.from_gguf(gguf, max_len=256)
    m.preload_gpu("cuda")
    m.warmup(seq_lens=(1, 16))
    tok = GemmaTokenizer.from_gguf(gguf)

    facade = MultiStepCompositionFacade(max_tokens=40)
    facade.install(m, tok)
    print("[r50.7] facade installed")

    comp_rows = run_cohort(facade, COMPOSITION_PROMPTS, "COMPOSITION")
    regr_rows = run_cohort(facade, REGRESSION_PROMPTS, "REGRESSION")

    # --- Metrics ---
    n_comp = len(comp_rows)
    parse_ok = sum(r["parse_ok"] for r in comp_rows)
    native_ok = sum(r["native_ok"] for r in comp_rows)
    base_correct = sum(r["base_ok"] for r in comp_rows)
    fac_correct = sum(r["fac_ok"] for r in comp_rows)
    base_wrong = [r for r in comp_rows if not r["base_ok"]]
    fixes = sum(1 for r in base_wrong if r["fac_ok"])
    n_wrong = len(base_wrong)
    fix_rate = fixes / n_wrong if n_wrong else 1.0

    n_regr = len(regr_rows)
    regr_base_correct = sum(r["base_ok"] for r in regr_rows)
    regr_fac_correct = sum(r["fac_ok"] for r in regr_rows)
    regressions = sum(
        1 for r in regr_rows if r["base_ok"] and not r["fac_ok"])

    print(f"\n{'='*30} ROUND 50.7 SUMMARY {'='*30}")
    print(f"  COMPOSITION ({n_comp} prompts)")
    print(f"    parse coverage:       {parse_ok}/{n_comp}")
    print(f"    substrate-native:     {native_ok}/{n_comp}  "
          f"(multiplier card served the a*b step)")
    print(f"    baseline correct:     {base_correct}/{n_comp}  "
          f"({n_wrong} failure surface)")
    print(f"    facade correct:       {fac_correct}/{n_comp}")
    print(f"    fixes (base_x->fac_v): {fixes}/{n_wrong}  "
          f"({fix_rate*100:.0f}%)")
    print(f"  REGRESSION ({n_regr} prompts)")
    print(f"    baseline correct:     {regr_base_correct}/{n_regr}")
    print(f"    facade correct:       {regr_fac_correct}/{n_regr}")
    print(f"    regressions:          {regressions}/{regr_base_correct}")

    # --- Gates ---
    parse_gate = parse_ok >= 18
    native_gate = native_ok >= 18
    fix_gate = fix_rate >= 0.80 if n_wrong else True
    regr_gate = regressions == 0

    print(f"\n  Gates:")
    print(f"    parse >= 18/20:          "
          f"{'PASS' if parse_gate else 'FAIL'}")
    print(f"    native >= 18/20:         "
          f"{'PASS' if native_gate else 'FAIL'}  "
          f"(multiplier served a*b)")
    print(f"    fix rate >= 80%:         "
          f"{'PASS' if fix_gate else 'FAIL'}")
    print(f"    zero regressions:        "
          f"{'PASS' if regr_gate else 'FAIL'}")

    if parse_gate and native_gate and fix_gate and regr_gate:
        print(f"\n  v MULTI-STEP COMPOSITION FACADE VALIDATED.")
        print(f"    Compiled multiplier + safe_eval + step-through bias "
              f"fixes {fixes}/{n_wrong} Gemma failures with "
              f"{native_ok}/{n_comp} substrate-native coverage, zero "
              f"regressions.")
        return 0
    print(f"\n  ~ Gate failures - see per-prompt output above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
