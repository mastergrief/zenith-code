# M1a — 4 new auto-generated facades

Ships Combinations / Permutations / Power / NextPrime as
decode-path facades via `calm/llm_computer/recursion.py`.
Every facade is a `FacadeSpec` → oracle-validate →
generate_facade → import → install → live A/B. Zero human-
written Python per facade (the specs live in recursion.py
module-level constants; the implementations are auto-generated).

## Results

| spec | oracle | baseline | card | Δ | wall |
|---|:-:|:-:|:-:|:-:|---:|
| Combinations | 5/5 | 3/5 | 5/5 | +2 | 30.2s |
| Permutations | 5/5 | 1/5 | 5/5 | +4 | 26.1s |
| Power | 5/5 | 4/5 | 5/5 | +1 | 26.7s |
| NextPrime | 5/5 | 4/5 | 5/5 | +1 | 20.9s |

**TOTAL**: baseline 12/20 → card 20/20 (Δ=+8)

## Generated files

- `calm/llm_computer/facades/combinations_auto.py`
- `calm/llm_computer/facades/permutations_auto.py`
- `calm/llm_computer/facades/power_auto.py`
- `calm/llm_computer/facades/next_prime_auto.py`
