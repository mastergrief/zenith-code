# Compute Facades — Historical receipts

Shipped-facade table with commits + dates, per-round provenance,
scope-expansion chronology. Current rules: `.claude/rules/compute_facades.md`.
This file exists for archaeology — "which facade shipped when",
"what commit introduced which discipline refinement".

## Shipped instances (as of 2026-04-22)

| Facade | File | Domain | Result |
|---|---|---|---|
| `MultiStepReasoningFacade` (R46.2) | `multi_step.py` | NL infix arithmetic | 17/17 Gemma fixes (`a385893`) |
| `BaseConversionFacade` (R22c) | `base_conversion.py` | Hex/binary → decimal | 10/10 vs 7/10 baseline (`7db6eb9`) |
| `NumberTheoryFacade` (R53a) | `number_theory.py` | mod / GCD / LCM | 15/15 vs 8/15 (`69279d4`) |
| `NumericEncodeFacade` (F2) | `numeric_encode.py` | int → hex/binary/octal | 12/12 on chain corpus (`5ee61a5`) |
| `Icd10RecallFacade` (R60a + F1) | `icd10_recall.py` | ICD-10 code → diagnosis TEXT, 72,748-code DB | 26/30 vs 8/30 baseline, first tier-3 (`afc0220`) |
| `PlannerFacade` (R70a + F2) | `planner.py` | orchestrates 4+ specialists + 2-step chains | 20/20 route single, 12/12 route chain |

### Auto-generated family via `recursion.py`

Commits `3274659` / `5173745` — see `recursion.md` for the Level-1 /
Level-2 pipeline spec.

**Level-1 hand-written `FacadeSpec`** (6 shipped):
- `factorial_auto.py`
- `fibonacci_auto.py`
- `combinations_auto.py`
- `permutations_auto.py`
- `power_auto.py`
- `next_prime_auto.py`

**Level-2 `MetaFacade.from_oracle(fn_name, arity)`** (5 shipped):
- `factorial_meta.py`
- `combinations_meta.py`
- `gcd_meta.py`
- `lcm_meta.py`
- `fibonacci_meta.py`

Total 17 facades share the identical R46.2 skeleton (verified 2026-04-22).

## Discipline scope expansions

### `▁`-strip + POST_BIAS_BUDGET discipline (R53a origin, 2026-04-22)

Exposed by `NumberTheoryFacade` R53a debugging (diagnostic
`scripts/r53a_debug_probe.py`, commit `69279d4`). Without the `▁`
strip, the first bias slot is wasted on a space — Gemma's natural
`0` token after `"Answer: "` has logit ~57-66 and +50 boost on `▁`
can't flip it.

**Scope at origin**: applied in `number_theory.py`, `numeric_encode.py`,
and all `recursion.py`-generated facades. NOT backported to
`multi_step.py` / `base_conversion.py` — those work because their
answer shapes don't trigger the `0`-run pattern (shipped tests still
10-17/17). Backport rule: only if a new facade shows the bug.

### Text-answer facade exception (Icd10 origin, 2026-04-22)

`Icd10RecallFacade` first text-answer facade — doesn't strip `▁`
because the diagnosis text starts with a capital letter (e.g. `▁Type`)
that IS a single merged BPE token including the leading space. This
generalized the step-through bias pattern to arbitrary Gemma BPE
sequences, opening tier-3 text-recall as a decode-path-addressable
target class.

### Boost-scaling for stubborn cases (Icd10 retry, 2026-04-22)

ICD-10 code-echo retry uses `boost * 3.0 = 150.0` and in-context
answer injection as last resort (commit `8ba151d`). Applied to the 4
stubborn ICD edges that resist the default boost.

## Session 2026-04-22 candidate-queue receipts

Shipped that session: mod/GCD/LCM (NumberTheory R53a), combinations,
permutations, power, next_prime (auto-generated), factorial, fibonacci
(auto + meta), int→hex/binary/octal (NumericEncode F2), ICD-10 text
recall (Icd10 R60a+F1).

**Session total per augmentation_thesis_arc**: 20/60 → 60/60 R22
retrieval, 12/30 → 26/30 tier-3 ICD-10, 0 → 15/15 NumberTheory,
0 → 12/12 Planner chain, 5 human-written + 11 auto/meta-generated
facades operational on prod Gemma. Measurement receipts and per-probe
JSONLs in `.cache/` for replay.

## R22b calibration reference (contrast case)

The "Decode-path vs CardSlot" decision rule in the current rules file
is informed by R22b's per-round calibration arc. Full receipt:
`MEMORY/atlas/delta_rule_arc.md` §"R22 install — full arc".

Key numbers (for context — decode-path facades sidestep ALL of these):
- R22f recalibration: `min_margin` 22.0 → 14.5 (commit `9691e06`)
- Per-N margin distribution: N=5 p50≈23.3, N=10 p5=15.21, N=15 p5=16.39
- 4 aligned gates: `write_margin`, `hook.min_margin`, `preserve`,
  N-range — all load-bearing

Decode-path compute facades have 1 gate (parse + evaluate succeed),
no VRAM, no training, no calibration, no channel conflicts. The
rule-of-thumb "CardSlot is for genuine trained-recall only" traces
to the cost asymmetry documented in R22b.

## Cross-refs

- Current rules: `.claude/rules/compute_facades.md`
- Retrieval-card install contrast: `MEMORY/atlas/delta_rule_arc.md`
  §"R22 install — full arc"
- Recursion pipeline: `.claude/rules/recursion.md`
- Capability-gain measurement: `MEMORY/atlas/capability_gain_arc.md`
- Session 2026-04-22 totals: `MEMORY/atlas/augmentation_thesis_arc.md`
  §"Shipped capability table"
