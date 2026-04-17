"""Round 46.2: MultiStepReasoningFacade end-to-end on prod Gemma.

Hypothesis: the facade's parse → CALM-verify → step-through-digit-bias
pipeline can fix multi-step arithmetic prompts where Gemma's chain-of-
thought fails mid-computation (carrying forward a wrong intermediate
value). Extends R11's single-op multiplication fix to arbitrary N-op
chains.

Failure-surface gate (per capability_gain.md): measure Gemma baseline
FIRST. Only claim wins on prompts baseline actually gets wrong.

Gates:
  - parse coverage  ≥ 18/20    (parser robustness)
  - baseline wrong  ≥ 5/20     (confirms Tier-2 failure surface)
  - facade correct on wrong-baseline ≥ 80%
  - facade doesn't regress any prompt baseline got right

Expected format of a "parse win": the parsed integer extracted from
facade.generated matches the expected arithmetic value. We accept
answers with trailing text (Gemma emits e.g. "436.\n\nThe answer is
436") as long as the FIRST integer matches.
"""

from __future__ import annotations

import os
import sys


# (prompt, expected_integer_answer). Expected is computed in Python,
# the ground truth the facade must deliver into Gemma's output.
PROMPTS = [
    # --- Multi-digit × — Gemma's known failure surface (R11) ---
    ("What is 17 * 23 + 5? Answer with just the number: ",      17*23+5),
    ("Compute 47 * 19 + 23: ",                                  47*19+23),
    ("What is 45 * 15 - 8? Answer: ",                           45*15-8),
    ("Compute 37 * 14 + 50: ",                                  37*14+50),
    ("What is 29 * 13 - 7? ",                                   29*13-7),
    # --- 3-step chains involving multiplication ---
    ("What is 12 * 8 + 15 - 3? ",                               12*8+15-3),
    ("Compute 5 * 11 * 3: ",                                    5*11*3),
    # --- Precedence: Gemma sometimes gets precedence wrong ---
    ("What is 3 + 4 * 5? Answer: ",                             3+4*5),
    ("Compute 100 - 20 * 3: ",                                  100-20*3),
    # --- Parens ---
    ("What is (17 + 5) * 8? Answer: ",                          (17+5)*8),
    ("Compute (100 - 40) / 3: ",                                (100-40)//3),
    # --- Additive only — Gemma probably OK; regression guard ---
    ("What is 100 + 50 - 30? Answer: ",                         100+50-30),
    ("Compute 250 - 75 + 10: ",                                 250-75+10),
    ("What is 1000 - 250 - 125? ",                              1000-250-125),
    # --- NL word-ops ---
    ("What is 17 times 23 plus 5? Answer: ",                    17*23+5),
    ("Compute 100 divided by 4 minus 7: ",                      100//4-7),
    ("What is 8 times 7 minus 13? ",                            8*7-13),
    # --- Division exact ---
    ("What is 120 / 8 + 5? Answer: ",                           120//8+5),
    # --- Mixed large + precedence ---
    ("Compute 250 * 2 - 100 + 50: ",                            250*2-100+50),
    ("What is 47 * 19 - 13 * 5? Answer: ",                      47*19-13*5),
]


def first_int(text: str):
    import re
    m = re.search(r"-?\d+", text.replace(",", ""))
    return int(m.group(0)) if m else None


def main():
    import torch  # noqa: F401
    from calm.llm_computer.gemma_substrate import (
        GemmaSubstrate, enable_triton_tq4)
    from calm.llm_computer.synth.gemma_tokenizer import GemmaTokenizer
    from calm.llm_computer.facades.multi_step import MultiStepReasoningFacade

    gguf = os.path.expanduser("~/models/gemma-4-E4B-it-tq4-aligned.gguf")
    enable_triton_tq4(True)
    print("[r46.2] loading substrate...")
    m = GemmaSubstrate.from_gguf(gguf, max_len=256)
    m.preload_gpu("cuda")
    m.warmup(seq_lens=(1, 16))
    tok = GemmaTokenizer.from_gguf(gguf)

    facade = MultiStepReasoningFacade(max_tokens=40)
    facade.install(m, tok)
    print("[r46.2] facade installed\n")

    parse_ok = 0
    baseline_correct = 0
    facade_correct = 0
    baseline_wrong = []         # prompts baseline got wrong
    baseline_right = []         # prompts baseline got right
    fix_count = 0               # baseline wrong → facade right
    regress_count = 0           # baseline right → facade wrong

    print(f"{'prompt':>45}  {'expected':>10}  {'base':>8}  "
          f"{'facade':>8}  base  facade")
    print("-" * 105)

    for prompt, expected in PROMPTS:
        # Baseline Gemma (no facade, no bias)
        baseline = facade.solve(prompt, use_bias=False)
        base_answer = first_int(baseline.generated)

        # Facade (parse → eval → bias)
        fac = facade.solve(prompt)
        fac_answer = first_int(fac.generated)

        if fac.expression is not None:
            parse_ok += 1

        base_ok = base_answer == expected
        fac_ok = fac_answer == expected
        if base_ok:
            baseline_correct += 1
            baseline_right.append(prompt)
            if not fac_ok:
                regress_count += 1
        else:
            baseline_wrong.append(prompt)
            if fac_ok:
                fix_count += 1

        if fac_ok:
            facade_correct += 1

        short = prompt if len(prompt) <= 43 else prompt[:40] + "..."
        base_mark = "✓" if base_ok else "✗"
        fac_mark = "✓" if fac_ok else "✗"
        print(f"{short!r:>45}  {expected:>10}  "
              f"{str(base_answer):>8}  {str(fac_answer):>8}  "
              f"  {base_mark}      {fac_mark}")

    n = len(PROMPTS)
    n_wrong = len(baseline_wrong)
    fix_rate = fix_count / n_wrong if n_wrong else 1.0

    print(f"\n{'='*30} ROUND 46.2 SUMMARY {'='*30}")
    print(f"  parse coverage:        {parse_ok}/{n}")
    print(f"  baseline correct:      {baseline_correct}/{n}  "
          f"({n_wrong} failure surface)")
    print(f"  facade correct:        {facade_correct}/{n}")
    print(f"  fixes (base✗ → fac✓):  {fix_count}/{n_wrong}  "
          f"({fix_rate*100:.0f}%)")
    print(f"  regressions:           {regress_count}/{baseline_correct}")

    parse_gate = parse_ok >= 18
    fs_gate = n_wrong >= 5
    fix_gate = fix_rate >= 0.80 if n_wrong else True
    regr_gate = regress_count == 0

    print(f"\n  Gates:")
    print(f"    parse ≥ 18/20:           {'PASS' if parse_gate else 'FAIL'}")
    print(f"    failure surface ≥ 5/20:  {'PASS' if fs_gate else 'FAIL'}"
          f"  (Tier-2 requires Gemma to fail — see capability_gain.md)")
    print(f"    fix rate ≥ 80%:          "
          f"{'PASS' if fix_gate else 'FAIL'}")
    print(f"    zero regressions:        {'PASS' if regr_gate else 'FAIL'}")

    if parse_gate and fs_gate and fix_gate and regr_gate:
        print(f"\n  ✓ MULTI-STEP REASONING FACADE VALIDATED.")
        print(f"    Parse-verify-bias fixes {fix_count}/{n_wrong} Gemma "
              f"failures without regressing correct baselines.")
        return 0
    print(f"\n  ~ Gate failures — see per-prompt output above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
