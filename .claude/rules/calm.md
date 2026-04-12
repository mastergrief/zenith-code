# CALM — Compute-Augmented Language Model Rules

## Core Principle

**Model reasons, backends compute, engine verifies.**
Intelligence comes from the system, not the weights. Adding a backend
module is equivalent to training — zero GPU cost, instant effect.

The model decides WHAT to compute. Modular backends decide HOW.
Auto-CALM makes this transparent — the model doesn't need to know
the engine exists.

## Two Modes

### Auto-CALM (default) — `calm/auto_calm.py`

Model writes naturally. Engine intercepts, verifies, corrects.

```
Prompt → precompute expressions → inject verified facts into system prompt
  ↓
Model generates response (with correct values available)
  ↓
Layer 1: extract claims from output → verify on CPU → correct if wrong
Layer 2: cross-check answer against precomputed value → retry if wrong
Layer 3: (intent-to-edit) diagnose bugs → template fix → test → self-heal
```

**40/40 (100%)** on benchmark. Model uses precomputed facts directly —
problems that took 30-165s drop to 1.7-2.3s.

### Explicit CALM (power user) — `calm/engine.py`, `calm/stream_engine.py`

Model emits `<calm>...</calm>` blocks. Engine stops, executes, injects.

```
Planning turn (thinking ON) → stop-mode execution loop:
  Model emits <calm> → STOP → 4-tier parse → execute → TMR verify →
  inject [engine: stack=X] → model reads result → next block or answer
```

- **85-98%** on benchmark (nondeterminism in block usage)
- Thinking + stop incompatible → hybrid plan-then-execute avoids this
- Assistant prefill incompatible with thinking → multi-turn instead

## Modular Backend Architecture

**Pattern**: write pure functions → export dict → register in `expression.py`.
Model gets smarter at that domain instantly.

### Current Backends (9 modules, 70+ functions)

| Backend | Functions | Domain | Verifiable? |
|---|---|---|---|
| `math_ops.py` | 9 | primes, GCD, factorize, fibonacci | 4-lane TMR |
| `string_ops.py` | 7 | len, case, contains, regex | exact match |
| `wasm_ops.py` | 17 | int/float via WebAssembly | cross-check |
| `code_ops.py` | 16 | read, write, test, lint, search | test pass/fail |
| `security_ops.py` | 8 | OWASP Top 10 detection | rule-based |
| `date_ops.py` | 6 | days_between, day_of_week, leap_year | deterministic |
| `convert_ops.py` | 5 | units (6 domains) + temperature | deterministic |
| `data_ops.py` | 11 | mean, median, stdev, regression | deterministic |
| `algo_ops.py` | 13 | sort, nCr, graph algos, LIS | deterministic |

### Adding a New Backend

1. Create `calm/backends/mydom_ops.py` with pure functions
2. Export: `MYDOM_FUNCTIONS = {"func_name": func, ...}`
3. Register in `calm/expression.py`:
   ```python
   try:
       from calm.backends.mydom_ops import MYDOM_FUNCTIONS
       _FUNCTIONS.update(MYDOM_FUNCTIONS)
   except ImportError:
       pass
   ```
4. (Optional) Add precompute patterns in `auto_calm.py:_precompute()`
5. (Optional) Add claim verification patterns in `auto_calm.py`

Each backend is optional — missing backends degrade gracefully via try/import.

## Auto-CALM Claim Verification

### Layer 1: Inline Claims

Extracts and verifies claims from model output:
- Arithmetic: `17 \times 23 = 391` (LaTeX + Unicode + plain)
- Functions: `factorial(10) = 3628800`
- GCD/LCM: `GCD of 391 and 782 is 391`
- Boolean: `391 is [not] prime`, `28 is a perfect number`, `X is divisible by Y`
- Filters conditional contexts: "if X is prime" → skip (question, not claim)
- Integer division awareness: "54 ÷ 7 = 7 remainder 5" → correct

### Layer 2: Precomputation

Extracts computations from the prompt BEFORE model responds:
- `"What is X?"` → evaluate X, inject as verified fact
- NL patterns: fibonacci(N), factorial(N), collatz_length(N), gcd(A,B), etc.
- Boolean: "Is X prime?", "Is X a leap year?"
- Conversions: "Convert 5 miles to km", "100 celsius to fahrenheit"
- Stats: "mean of [1,2,3]", "10 choose 3"
- Prompt-level answer verification with multi-turn retry

### Layer 3: Intent-to-Edit

3-step bug fixing: diagnose → template fix → verify.
- Model reads code + test failures, describes bugs in NL
- Engine applies deterministic templates:
  - `ZeroDivisionError` → zero-check guard
  - `ValueError` on float()/int() → try/except
  - `IndexError` → bounds-check guard
  - `AttributeError` on None → null guard
- Falls back to LLM full-rewrite if templates insufficient
- Self-healing: feeds remaining failures back (max 1 retry)
- Verified: 6/10 → 10/10 on calc.py, 8/13 → 13/13 on unseen store.py

## Auto-Training Data Collection

Every correction generates distillation-compatible JSONL:
- `MathCollector` → `.calm_training/auto/math.jsonl`
- `BoolCollector` → `.calm_training/auto/bool.jsonl`
- `CodeCollector` → `.calm_training/auto/code.jsonl`

Virtuous cycle: corrections → training data → (optional) fine-tune →
fewer corrections → higher-quality corrections. But backends are the
primary path — training is supplementary.

## Verification (`calm/verifier.py`)

4-lane TMR for math backend dispatches:

| Lane | Method | Example |
|---|---|---|
| Primary | Registered backend | Python `math.gcd` |
| Cross-check | Independent impl | Wasm Euclidean GCD |
| Algorithm | Different algorithm | Binary/Stein's GCD |
| Proof | Inverse/property | `g\|a AND g\|b AND gcd(a/g, b/g)==1` |

DIVERGENCE = real failure (lanes disagree) → halt.
VERIFIED = all lanes agree → safe.

## Expression Evaluator (`calm/expression.py`)

- **AST-only**: `ast.parse(mode="eval")` + recursive walker. Never `eval()`.
- **Whitelist**: only functions in `_FUNCTIONS` dict (70+ from all backends)
- **Comprehensions**: list/set/generator with per-variable scoping, 10K limit
- **No attribute access, no imports** — all functions pre-registered

## Benchmark

40 problems, 6 categories:

| Mode | arithmetic | number_theory | sequences | algebra | reasoning | multi_step | Total |
|---|---|---|---|---|---|---|---|
| Auto-CALM + precompute | 10/10 | 10/10 | 5/5 | 5/5 | 5/5 | 5/5 | **40/40** |
| Explicit CALM (best) | 10/10 | 10/10 | 3-5/5 | 5/5 | 5/5 | 5/5 | 85-98% |
| Auto-CALM (no precompute) | 9/10 | 10/10 | 2/5 | 4/5 | 5/5 | 5/5 | 88% |

## File Map

| File | LOC | Purpose |
|---|---|---|
| `auto_calm.py` | 1150 | Auto-CALM: claim verify + precompute + intent-to-edit |
| `auto_training.py` | 300 | Training data generation from corrections |
| `engine.py` | 527 | Explicit CALM v0.1: stop-mode |
| `stream_engine.py` | 240 | Explicit CALM v0.2: SSE streaming |
| `interceptor.py` | 479 | 4-tier parse + block detection |
| `expression.py` | 680 | AST-safe eval, 70+ functions from all backends |
| `verifier.py` | 560 | 4-lane TMR verification |
| `stack_vm.py` | 522 | Reference stack machine |
| `sandbox.py` | 250 | Subprocess Python isolation |
| `nl_parser.py` | 168 | NL → stack code translator |
| `backends/*.py` | ~850 | 9 modular compute backends |
| `tests/` | ~3,400 | 250 tests |
| `benchmark.py` | 227 | 40-problem eval |
