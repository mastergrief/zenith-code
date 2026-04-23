# M2a — Level-2 MetaFacade demo

Per `.claude/spec/recursion.md` §'Level 2'. MetaFacade
synthesizes the FacadeSpec itself from just
(oracle_fn_name, arity). All three Level-1 CALM gates still
apply (oracle validation → ast.parse → live A/B); only the
SPEC authorship moved from human to substrate.

## Descriptors → specs

- `factorial` arity=1 → `FactorialMeta` (module `factorial_meta`)
- `combinations` arity=2 → `CombinationsMeta` (module `combinations_meta`)
- `gcd` arity=2 → `GcdMeta` (module `gcd_meta`)
- `lcm` arity=2 → `LcmMeta` (module `lcm_meta`)
- `fibonacci` arity=1 → `FibonacciMeta` (module `fibonacci_meta`)

## Results

| spec | oracle | baseline | card | Δ | wall |
|---|:-:|:-:|:-:|:-:|---:|
| FactorialMeta | 4/4 | 0/3 | 3/3 | +3 | 19.2s |
| CombinationsMeta | 4/4 | 1/3 | 3/3 | +2 | 18.3s |
| GcdMeta | 4/4 | 1/3 | 3/3 | +2 | 17.7s |
| LcmMeta | 3/3 | 1/3 | 3/3 | +2 | 17.4s |
| FibonacciMeta | 4/4 | 1/3 | 3/3 | +2 | 17.8s |

**TOTAL**: baseline 4/15 → card 15/15 (Δ=+11)

## What MetaFacade replaced

A hand-written FacadeSpec requires the author to think about:
- Name conventions (PascalCase class, snake_case module)
- Canonical NL patterns (fn(args) / fn of args / a fn b)
- Integer capture groups with negative-number support
- safe_eval template formatting ({a}, {b})
- Arity-specific regex shapes
- max_tokens + max_operand guards

MetaFacade.from_oracle encodes all of these as a template
function over (fn_name, arity). The user supplies ONLY:
- safe_eval function name (must exist)
- Arity (1 or 2)
- Optional guard / extra-patterns overrides

## Generated files

- `calm/llm_computer/facades/factorial_meta.py` (FactorialMeta)
- `calm/llm_computer/facades/combinations_meta.py` (CombinationsMeta)
- `calm/llm_computer/facades/gcd_meta.py` (GcdMeta)
- `calm/llm_computer/facades/lcm_meta.py` (LcmMeta)
- `calm/llm_computer/facades/fibonacci_meta.py` (FibonacciMeta)
