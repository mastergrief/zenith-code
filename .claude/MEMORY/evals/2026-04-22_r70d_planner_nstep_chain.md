# R70d — PlannerFacade N-step chain dispatch

3+ step chain composition via 'then'/',' connectives:
primary facade → arith-op N → (arith-op N)* → optional numeric-encode.

## Corpus

9 chain probes: 2-step, 3-step, 4-step variants across
NumberTheory, auto-facades (factorial, power), MultiStep intermediate
arithmetic, NumericEncode terminal.

## Results

| metric | value |
|---|---:|
| chain probes | 9 |
| bias hits | 9/9 |
| baseline | 5/9 |
| Δ (bias − baseline) | +4 |
| wall Phase 2 | 109.6s |
| wall Phase 3 | 72.2s |

