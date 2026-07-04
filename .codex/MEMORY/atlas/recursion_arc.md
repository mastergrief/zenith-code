# Recursion — Historical receipts

Shipped-facade dated inventory with commit SHAs, per-level demo
scripts + eval file cross-refs, code-DT self-distill roadmap R-numbers.
Current rules: `.claude/rules/recursion.md`.

## Shipping dates + commit SHAs

**Shipped as of 2026-04-22** (commits `3274659` F3, `5173745` M1+M2):

- **Fact-level recursion**: `auto_upgrade.py` +
  `gemma_learning_loop_demo.py`: 5/5 wrong → 5/5 correct.

- **Level 1**: generic decode-path facade auto-generator
  (`calm/llm_computer/recursion.py`). 6 shipped auto-facades
  (factorial, fibonacci, combinations, permutations, power,
  next_prime) lifting Gemma 17/30 → 30/30 across their domains.

- **Level 2**: `MetaFacade.from_oracle(fn_name, arity)` synthesizes
  the `FacadeSpec` itself. 5 meta-facades shipped (factorial,
  combinations, gcd, lcm, fibonacci) lifting 4/15 → 15/15.

## Demo scripts

- `scripts/r80a_recursion_demo.py` (F3 Level-1)
- `scripts/m1a_four_new_facades.py` (M1 4 new auto)
- `scripts/m2a_metafacade_demo.py` (M2 Level-2)

## Eval receipts

- `.claude/MEMORY/evals/2026-04-22_r80a_recursion_level1_demo.md`
- `.claude/MEMORY/evals/2026-04-22_m1a_four_new_facades.md`
- `.claude/MEMORY/evals/2026-04-22_m2a_level2_metafacade.md`

## Code-DT self-distill roadmap

Originally scoped as **R53.5 + R53.6** (inherited session-34 roadmap):
train `copy_code_best.pt` on 8970-example DB, install at L24 via
CardSlot, run CodeVerifierFacade-gated self-distillation loop.

Currently parked pending DT code-skeleton honest-val reaching the
≥ 0.40 install threshold (currently 0.193 — see
`MEMORY/atlas/delta_rule_arc.md` §"DT code-skeleton arc").

## Why CALM-gated recursion is safe

| Approach | Oracle | Failure mode |
|---|---|---|
| Self-Instruct (Wang 2022) | the generating model itself | amplifies biases, reinforces hallucinations |
| RLAIF / constitutional AI | judge LLM | judge bias leaks into student |
| Evol-Instruct | LLM scoring | same bias amplification |
| **Substrate card recursion** | **deterministic CALM + compiled verification** | **cannot amplify what's verified wrong** |

Every card in the recursion chain is gated by objective correctness
checks (safe_eval oracle, `ast.parse`, live A/B with 0 regressions).
Drift-free on compiled domains; open-ended creative tasks stay Tier 1.

## Cross-refs

- Current recursion spec (stub): `.claude/rules/recursion.md` — detail in this file
- DT code-skeleton progress: `MEMORY/atlas/delta_rule_arc.md`
- Capability-gain session receipts: `MEMORY/atlas/capability_gain_arc.md`
  §"2026-04-22 session receipts"
