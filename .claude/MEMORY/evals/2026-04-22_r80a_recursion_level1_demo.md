# R80a — Recursion Level-1 demo

Phase B MVP per `.claude/spec/recursion.md`. Substrate
template-generates new decode-path facades, CALM oracle
validates the spec, generator writes the .py file, facade
installs on live Gemma, A/B runs against baseline.

## Specs shipped

- **Factorial** (`calm/llm_computer/facades/factorial_auto.py`): oracle 7/7, baseline 3/5 → card 5/5 (Δ=+2)
- **Fibonacci** (`calm/llm_computer/facades/fibonacci_auto.py`): oracle 7/7, baseline 2/5 → card 5/5 (Δ=+3)

## Aggregate

| metric | value |
|---|---:|
| specs generated | 2 |
| total probes | 10 |
| baseline total | 5/10 |
| with-facade total | 10/10 |
| Δ | +5 |

## What this proves

Level 1 of the recursion chain (`recursion.md`): the substrate
produces new capabilities without human-written Python. The
generator is deterministic (parameterized from `FacadeSpec`),
not LLM-written. CALM oracle gates the spec before any file
touches disk; ast.parse gates the generated source; live A/B
gates installation. Three CALM-anchored checkpoints in one
loop, no RLAIF-style bias amplification path.

Next step (Level 2 per recursion.md): replace the parameterized
template with a MetaFacade that, given a (failure_trace,
oracle_signature) pair, emits the FacadeSpec itself. That moves
code-spec authorship from human to substrate while keeping
CALM validation as the gate.
