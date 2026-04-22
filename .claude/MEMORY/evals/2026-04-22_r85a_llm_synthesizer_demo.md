# R85a — LLM-written Level-2 synthesizer demo

Per `recursion.md` §'Level 3 MetaMetaFacade' + `augmentation_thesis.md`
§'auto-generated facades'. Infrastructure for LLM-proposed /
CALM-disposed spec synthesis.

## Gate chain

1. JSON extraction — malformed text → None (0% dangerous)
2. Field validation — fn_name in safe_eval? arity ∈ {1,2}? regex compiles? group count = arity?
3. Spec synthesis — MetaFacade.from_oracle builds FacadeSpec
4. CALM oracle — validate_facade runs safe_eval on test cases

Only specs passing all 4 gates ever touch disk. RLAIF-safe by
construction: LLM proposes, deterministic verifier disposes.

## Test harness results

| test case | expected | got | correct |
|---|---|---|---|
| good_factorial_clean | PASS | PASS | ✓ |
| good_combinations_noisy_prose | PASS | PASS | ✓ |
| good_is_prime_bool_output | PASS | PASS | ✓ |
| bad_hallucinated_fn | REJECT | REJECT | ✓ |
| bad_not_json | REJECT | REJECT | ✓ |
| bad_regex_arity_mismatch | REJECT | REJECT | ✓ |
| bad_oracle_mismatch | REJECT | REJECT | ✓ |

**7/7 gate decisions correct.**

