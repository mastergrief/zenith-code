# R70a — PlannerFacade mixed-corpus A/B

Orchestration MVP per `tracing_roadmap.md` §'Planner card'.
Single NL entry point dispatches to 4 specialist facades
(multi_step / base_conv / number_theory / icd10) plus
pass-through. First-match-wins priority chain, no chaining yet.

## Mixed corpus

20 probes: 4 math / 4 base-conversion / 4 number-theory /
5 ICD-10 / 3 pass-through. Scoring per-domain (numeric
answer-check for facade outputs, bag-of-words for ICD-10,
non-empty for pass-through).

## Result

| metric | value |
|---|---:|
| total | 20 |
| route correct | 20/20 |
| answer correct | 18/20 |
| wall time | 113.9s |

## By facade

| facade | correct |
|---|---:|
| base_conv | 4/4 |
| icd10 | 5/5 |
| multi_step | 3/4 |
| number_theory | 4/4 |
| passthrough | 2/3 |

## Architecture

Each facade's `parse(prompt)` acts as a gate — the planner tries
them in priority order (icd10 > base_conv > number_theory >
multi_step > passthrough). Ambiguity is avoided because each
facade's gate requires domain-specific signal (ICD-10 phrase
+ code; hex/binary + 'in decimal'; specific mod/gcd/lcm
keywords; infix operators).

Next steps (Option C — compiled planner card): chain facades
so multi-step queries like 'GCD(48,180), convert to hex' route
through two facades in sequence, biasing Gemma to emit each
intermediate. Requires a new int→hex decode-path facade first.
