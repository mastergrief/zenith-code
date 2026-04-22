# R70c — PlannerFacade auto-facade dispatch A/B

Registers auto-generated Level-1/2 facades (factorial, fibonacci,
combinations, permutations, power, next_prime) with the Planner so
user queries reach them instead of falling through to multi_step
catch-all.

## Corpus

12 probes (2 per facade), covering canonical + variant
regex patterns from the auto-generated `_PARSE_RES` list.

## Results

| metric | value |
|---|---:|
| total probes | 12 |
| route accuracy | 12/12 |
| answer (with bias) | 12/12 |
| baseline (no bias) | 4/12 |
| Δ (bias − baseline) | +8 |
| wall Phase 2 | 12.3s |
| wall Phase 3 | 65.3s |

