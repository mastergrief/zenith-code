# CALM — Compute-Augmented Language Model Rules

## Core Principle

**LLM sequences, CPU computes, TMR verifies, injection feeds back.**
The model decides WHAT to compute. The engine decides HOW and verifies the result.
No fine-tuning required — the architecture handles everything.

## Engine Loop (`calm/engine.py`)

```
Planning turn (thinking ON, no CALM) → stop-mode execution loop:
  Model emits <calm> → STOP → parse → execute → TMR verify →
  inject [engine: stack=X] → model reads result → next block or answer
```

- **Thinking + stop incompatible**: `stop=["</calm>"]` fires during thinking,
  killing the response. The hybrid avoids this: think first, execute second.
- **Assistant prefill incompatible with thinking**: llama.cpp returns 400.
  Use multi-turn (assistant + user messages) instead.
- **Post-verify**: if model skips CALM, engine independently computes the
  answer and corrects if wrong. NL rewrites translate natural language
  prompts to computable expressions.
- **Bail-out**: 3 consecutive errored blocks OR 2 empty responses → stop.

## Four-Tier Parse Pipeline (`calm/interceptor.py`)

Each line inside `<calm>...</calm>` tries these tiers in order:

| Tier | Module | Handles | Falls through on |
|---|---|---|---|
| 1. NL parser | `nl_parser.py` | "multiply 17 by 23", "is 391 prime?" | No regex match |
| 2. Stack VM | `stack_vm.py` | `push 17\nmul`, builtins + aliases | Unknown word |
| 3. Expression | `expression.py` | `(17*23) + gcd(391,782)`, comprehensions | AST parse failure |
| 4. Sandbox | `sandbox.py` | Any Python: for loops, variables, classes | Timeout or error |

**Python compat in the interceptor** (not a separate tier):
- `x = expr` → strips assignment, evaluates RHS, stores in `self.variables`
- `print(expr)` → strips wrapper, evaluates inner
- `#` comments → treated as line comments
- `pop` → alias for `drop`, `print` → alias for `emit`

## Verification (`calm/verifier.py`)

4-lane TMR for every backend dispatch:

| Lane | Method | Example for GCD |
|---|---|---|
| Primary | Registered backend | Python `math.gcd` |
| Cross-check | Independent impl | Wasm Euclidean GCD |
| Algorithm | Different algorithm | Binary/Stein's GCD |
| Proof | Inverse/property check | `g\|a AND g\|b AND gcd(a/g, b/g)==1` |

- Float tolerance: `1e-12` relative, `1e-15` absolute (Newton vs hardware sqrt)
- DIVERGENCE = real failure (implementations disagree) — halts execution
- VERIFIED = all lanes agree — safe to proceed

## Expression Evaluator Safety (`calm/expression.py`)

- **AST-only**: `ast.parse(mode="eval")` + recursive node walker. Never `eval()`.
- **Whitelist**: only functions in `_FUNCTIONS` dict are callable
- **Comprehension support**: list comps execute via `_eval_comprehension` with
  per-variable scoping and 10,000 element limit
- **No attribute access**: `"hello".upper()` → blocked
- **No imports**: all functions are pre-registered

## Sandbox Safety (`calm/sandbox.py`)

- Subprocess isolation — model code can't affect parent process
- Import blocking: `os`, `subprocess`, `socket`, `http`, `pathlib`, etc.
- Timeout: 10s default, configurable
- Prelude injects all CALM functions (is_prime, fibonacci, etc.)
- **Not a security sandbox** — it's defense-in-depth, not a jail.
  Don't run untrusted code from external sources.

## Training Signal (`calm/training.py`)

Every `<calm>` block generates a labeled pair in `.calm_training/signal.jsonl`:
- `claimed=[401], actual=[391], correct=false` → model was wrong (training signal)
- `claimed=null, actual=[391], correct=true` → model deferred via `<pending>`

## CALM Block Detection

The interceptor detects two formats:
- Standard: `<calm>...</calm>`
- Gemma tool-call: `<|tool_call>call:calm` / `<channel|>`

Model-fabricated `[engine: ...]` lines inside blocks are stripped.

## Benchmark (`calm/benchmark.py`)

40 problems, 6 categories. Best: 39/40 (98%). Typical: 85-90% due to
nondeterminism in whether the model uses CALM vs answering from memory.

- **100% categories**: arithmetic, number theory, algebra, reasoning, multi-step
- **Weak spot**: sequences (model sometimes answers from memory, gets it wrong)
- **Keywords use `|` alternatives**: `"not prime|composite|no"` to reduce false negatives

## File Map

| File | LOC | Purpose |
|---|---|---|
| `engine.py` | 512 | Closed-loop execution engine |
| `interceptor.py` | 479 | 4-tier stream parser |
| `expression.py` | 559 | AST-safe expression evaluator |
| `verifier.py` | 557 | 4-lane TMR verification |
| `stack_vm.py` | 522 | Reference stack machine |
| `sandbox.py` | 234 | Subprocess Python isolation |
| `nl_parser.py` | 168 | NL → stack code translator |
| `grammar.py` | 110 | GBNF grammar generator |
| `training.py` | 94 | JSONL signal collector |
| `backends/math_ops.py` | 134 | 9 CPU math functions |
| `backends/string_ops.py` | 92 | 7 string functions |
| `backends/wasm_ops.py` | 174 | 17 wasm functions |
| `backends/calm_math.wat` | 60 | WebAssembly module |
| `reasoning.py` | 172 | Structured chain tracker |
| `benchmark.py` | 227 | 40-problem eval |
| Tests (8 files) | ~3000 | 214 tests |
