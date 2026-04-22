# R53a — NumberTheoryFacade (second compute facade)

Decode-path tier-2 facade (parse → safe_eval → step-through
digit bias) for modular arithmetic + GCD + LCM. Generalizes
R46.2 (infix arithmetic) and R22c (base conversion).

## A/B (15-probe corpus, stock Gemma 4 E4B tq4)

| metric | value |
|---|---:|
| baseline | 8/15 |
| facade   | 15/15 |
| Δ        | +7 |
| regressions | 0 |
| wall time | 74.5s |

## Corpus

5 mod + 5 gcd + 5 lcm, mixed easy/medium/hard per op.

