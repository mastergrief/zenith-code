"""R85a — LLM-written Level-2 synthesizer demo.

Demonstrates `calm/llm_computer/llm_synthesizer.py`: given a free-text
LLM output (Gemma or otherwise) with a JSON-shaped FacadeSpec suggestion,
pass through 4 independent gates before writing any file to disk:

  1. JSON extraction (malformed → None)
  2. Field validation (fn_name in safe_eval? arity valid? regex compiles?)
  3. Spec synthesis (via MetaFacade.from_oracle)
  4. CALM oracle validation (safe_eval verifies expected outputs)

Only synthesized specs whose computed outputs match known test cases
survive to disk. This is the RLAIF-safe path: LLM proposes, CALM disposes.

Runs without Gemma — feeds curated mock LLM outputs through the pipeline.
Shows 6 cases: 3 pass, 3 fail (different failure modes).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from calm.llm_computer.llm_synthesizer import (
    parse_llm_suggestion, validate_suggestion,
    synthesize_and_validate, build_synth_prompt,
)


# Mock LLM outputs. Real Gemma would be noisier; these simulate different
# output qualities.
CASES = [
    # (name, mock_gemma_output, oracle_cases, expect_pass)
    (
        "good_factorial_clean",
        '{"fn_name":"factorial","arity":1,"nl_patterns":["factorial\\\\s+of\\\\s+(-?\\\\d+)"],"operand_type":"int","output_type":"int","max_operand":15}',
        [(5, 120), (7, 5040)],
        True,
    ),
    (
        "good_combinations_noisy_prose",
        '''Here is my JSON suggestion for the combinations facade:

```json
{
  "fn_name": "combinations",
  "arity": 2,
  "nl_patterns": ["(-?\\\\d+)\\\\s+choose\\\\s+(-?\\\\d+)"],
  "operand_type": "int",
  "output_type": "int",
  "max_operand": 50
}
```

I hope this works!''',
        [((10, 3), 120), ((8, 2), 28)],
        True,
    ),
    (
        "good_is_prime_bool_output",
        '{"fn_name":"is_prime","arity":1,"nl_patterns":["is\\\\s+(-?\\\\d+)\\\\s+prime"],"operand_type":"int","output_type":"bool","max_operand":1000}',
        [(7, True), (9, False), (17, True)],
        True,
    ),
    (
        "bad_hallucinated_fn",
        '{"fn_name":"carmichael_lambda","arity":1,"nl_patterns":["(-?\\\\d+)"],"operand_type":"int","output_type":"int","max_operand":100}',
        [(1, 1)],
        False,
    ),
    (
        "bad_not_json",
        "I think the factorial function takes one integer and returns an integer. But I don't know the syntax.",
        [(5, 120)],
        False,
    ),
    (
        "bad_regex_arity_mismatch",
        '{"fn_name":"gcd","arity":2,"nl_patterns":["(-?\\\\d+)"],"operand_type":"int","output_type":"int","max_operand":100}',
        [((12, 8), 4)],
        False,
    ),
    (
        "bad_oracle_mismatch",
        # Schema is valid but eval will not match expected
        '{"fn_name":"factorial","arity":1,"nl_patterns":["(-?\\\\d+)"],"operand_type":"int","output_type":"int","max_operand":20}',
        [(5, 999)],   # wrong expected
        False,
    ),
]


def main():
    print("=" * 60)
    print("LLM-written Level-2 synthesizer demo")
    print("=" * 60)
    print()
    print("Example synth prompt (what Gemma would see):")
    print(build_synth_prompt("I want a facade for Euler totient"))
    print("=" * 60)
    print()

    passes = 0
    expected_passes = 0
    for name, llm_out, cases, expect_pass in CASES:
        if expect_pass:
            expected_passes += 1
        print(f"[{name}] expect={'PASS' if expect_pass else 'REJECT'}")
        ok, details = synthesize_and_validate(llm_out, cases)
        stages_str = " → ".join(
            f"{stage}({'✓' if okp else '✗'})"
            for stage, okp, _ in details.get("stages", [])
        )
        print(f"  Result: {'PASS' if ok else 'REJECT'}")
        print(f"  Stages: {stages_str}")
        if details.get("suggestion"):
            print(f"  Parsed: {details['suggestion']}")
        failing_stage = next(
            (s for s in details.get("stages", []) if not s[1]), None
        )
        if failing_stage:
            print(f"  Failure at: {failing_stage[0]}: {failing_stage[2]!r}")
        if details.get("oracle_validation"):
            print(f"  Oracle: {details['oracle_validation']}")
        print()

        correct = (ok == expect_pass)
        if correct:
            passes += 1

    print("=" * 60)
    print(f"Test harness: {passes}/{len(CASES)} gate decisions correct")
    print(f"  (pipeline behaved as expected on {passes} of {len(CASES)} cases)")
    print()
    print("Shipped infrastructure:")
    print("  • calm/llm_computer/llm_synthesizer.py")
    print("  • build_synth_prompt(user_request) → prompt for Gemma")
    print("  • parse_llm_suggestion(text) → LlmSuggestion | None")
    print("  • validate_suggestion(sug) → issues (empty if pass)")
    print("  • synthesize_spec(sug) → FacadeSpec via MetaFacade")
    print("  • synthesize_and_validate(text, cases) → (ok, details)")

    recpath = (ROOT / ".claude" / "MEMORY" / "evals"
               / "2026-04-22_r85a_llm_synthesizer_demo.md")
    lines = [
        "# R85a — LLM-written Level-2 synthesizer demo",
        "",
        "Per `recursion.md` §'Level 3 MetaMetaFacade' + `augmentation_thesis.md`",
        "§'auto-generated facades'. Infrastructure for LLM-proposed /",
        "CALM-disposed spec synthesis.",
        "",
        "## Gate chain",
        "",
        "1. JSON extraction — malformed text → None (0% dangerous)",
        "2. Field validation — fn_name in safe_eval? arity ∈ {1,2}? regex compiles? group count = arity?",
        "3. Spec synthesis — MetaFacade.from_oracle builds FacadeSpec",
        "4. CALM oracle — validate_facade runs safe_eval on test cases",
        "",
        "Only specs passing all 4 gates ever touch disk. RLAIF-safe by",
        "construction: LLM proposes, deterministic verifier disposes.",
        "",
        "## Test harness results",
        "",
        "| test case | expected | got | correct |",
        "|---|---|---|---|",
    ]
    for name, llm_out, cases, expect_pass in CASES:
        ok, details = synthesize_and_validate(llm_out, cases)
        got = "PASS" if ok else "REJECT"
        want = "PASS" if expect_pass else "REJECT"
        correct = "✓" if (ok == expect_pass) else "✗"
        lines.append(f"| {name} | {want} | {got} | {correct} |")
    lines.extend([
        "",
        f"**{passes}/{len(CASES)} gate decisions correct.**",
        "",
    ])
    recpath.write_text("\n".join(lines) + "\n")
    print(f"\nreceipt → {recpath}")


main()
print("R85A_DONE")
