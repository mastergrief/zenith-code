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

**Pattern**: write a `*_ops.py` file in `calm/backends/`, export a `*_FUNCTIONS`
dict. Auto-discovery registers it — zero other files to edit.
Model gets smarter at that domain instantly.

Two types of backends coexist:
- **Compute backends** — deterministic functions (math, encoding, dates, etc.)
- **Knowledge backends** — factual lookup tables (countries, elements, constants, algorithms)

The engine doesn't care which type — same contract: pure function, deterministic
output, engine trusts it over the model.

### Current Backends (52 modules, 411 functions)

**Compute backends:**

| Backend | Funcs | Domain |
|---|---|---|
| `math_ops` | 9 | primes, GCD, factorize, fibonacci, collatz |
| `string_ops` | 7 | len, case, contains, regex |
| `wasm_ops` | 17 | int/float via WebAssembly cross-check |
| `code_ops` | 16 | read, write, test, lint, search |
| `security_ops` | 8 | OWASP Top 10 detection |
| `date_ops` | 6 | days_between, day_of_week, leap_year |
| `convert_ops` | 5 | units (6 domains) + temperature |
| `data_ops` | 11 | mean, median, stdev, regression |
| `algo_ops` | 13 | sort, nCr, graph algorithms, LIS |
| `quality_ops` | 7 | cyclomatic complexity, naming, dead code |
| `readability_ops` | 5 | Flesch-Kincaid, jargon, structure |
| `regex_ops` | 7 | pattern matching, validation |
| `json_ops` | 7 | validate, path, diff, format |
| `encoding_ops` | 12 | base64, hex, md5, sha256 |
| `git_ops` | 7 | log, blame, status, branches |
| `network_ops` | 9 | URL, IP, CIDR, HTTP status |
| `creative_ops` | 9 | brainstorm, combine, novelty |
| `impact_ops` | 7 | call graph, blast radius, coupling |
| `context_ops` | 7 | git archaeology, code age |
| `python_ops` | 9 | builtin/method verification |
| `math_extended_ops` | 15 | matrices, modular arithmetic, calculus |
| `perf_ops` | 6 | Big-O estimation, memory analysis |
| `deps_ops` | 6 | package versions, imports |
| `refactor_ops` | 4 | code smells, duplicates |
| `type_ops` | 4 | annotation coverage |
| `test_ops` | 4 | test summary, coverage |
| `doc_ops` | 4 | docstring coverage |
| `shell_ops` | 7 | exit codes, dangerous commands |
| `semver_ops` | 6 | version compare, satisfies |
| `config_ops` | 6 | YAML, TOML, INI, dotenv |
| `sql_ops` | 6 | parse, validate, risk, format |
| `cron_ops` | 6 | parse, explain, next runs, frequency |
| `bitwise_ops` | 18 | AND/OR/XOR/NOT, shifts, popcount, masks |
| `diff_ops` | 6 | unified diff parse, stats, apply |
| `package_ops` | 6 | pip/npm/cargo info |
| `ast_ops` | 7 | Python AST parse, functions, classes |
| `http_ops` | 7 | status codes (401 vs 403), methods, MIME |
| `uuid_ops` | 8 | generate, validate, parse, compare |
| `csv_ops` | 9 | parse, validate, column stats |
| `markdown_ops` | 7 | headers, TOC, code blocks, links |
| `unicode_ops` | 7 | codepoints, categories, confusables |
| `color_ops` | 9 | hex/RGB/HSL, WCAG contrast, complement |
| `jwt_ops` | 7 | decode header/payload, validate structure |
| `timezone_ops` | 7 | convert, UTC offset, DST awareness |
| `baseconv_ops` | 9 | binary/octal/hex/arbitrary base |
| `checksum_ops` | 8 | Luhn, ISBN-10/13, EAN, UPC |
| `bytesize_ops` | 7 | human-readable, IEC vs SI (MiB vs MB) |
| `duration_ops` | 7 | parse "2h30m", ISO 8601, convert |

**Knowledge backends** (`*_kb.py` — factual lookups, include `_DATA_VERSION`):

| Backend | Funcs | Domain |
|---|---|---|
| `country_kb` | 8 | capitals, ISO codes, currencies, calling codes (195 countries, 2025-01) |
| `elements_kb` | 9 | periodic table: symbols, weights, electron config (118 elements) |
| `constants_kb` | 5 | physical constants: speed of light, Planck, Avogadro (CODATA 2018) |
| `complexity_kb` | 5 | sort/DS/graph algorithm complexity (the most hallucinated CS topic) |

### Adding a New Backend

**Naming**: `*_ops.py` for compute (functions that DO something), `*_kb.py`
for knowledge (functions that LOOK UP something). Knowledge backends should
include a `_DATA_VERSION` date for staleness tracking.

1. Create `calm/backends/mydom_ops.py` (or `mydom_kb.py`) with pure functions
2. Export: `MYDOM_FUNCTIONS = {"func_name": func, ...}`
3. Done — auto-discovery in `calm/backends/__init__.py` registers it
4. (Optional) Add NL precompute patterns in `precompute.py`
5. (Optional) Add claim verification patterns in `verify.py`

**Defense in depth**: Layer 2 (precompute) injects correct answers before
generation. Layer 1 (verify) catches wrong claims after generation. Both
should cover the same domains — when precompute misses, verify is the safety net.

**Auto-learn guard**: `auto_learn.py` instantiates learned patterns with
numbers from the prompt. Large numbers (>10M) are skipped to prevent
combinatorial explosions (e.g. `factorial(4532015112830366)` from a credit
card number in the prompt).

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
- **Whitelist**: only functions in `_FUNCTIONS` dict (411+ from all backends)
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
| `auto_calm.py` | 324 | Facade: composes layers, CLI entry |
| `verify.py` | 323 | Layer 1: claim extraction + correction (incl. base conversion) |
| `precompute.py` | 410 | Layer 2: NL→expression precomputation + system prompt |
| `intent_edit.py` | 356 | Layer 3: NL diagnosis → template fix → verify |
| `stream_auto.py` | 437 | Streaming verification + tool-call handler |
| `auto_learn.py` | 220 | Self-learning from corrections (>10M guard) |
| `auto_training.py` | 337 | Training data generation |
| `engine.py` | 552 | Explicit CALM v0.1: stop-mode |
| `stream_engine.py` | 287 | Explicit CALM v0.2: SSE streaming |
| `interceptor.py` | 479 | 4-tier parse + block detection |
| `expression.py` | 657 | AST-safe eval, `_FUNCTIONS` dict (411 from registry) |
| `verifier.py` | 559 | 4-lane TMR verification |
| `stack_vm.py` | 522 | Reference stack machine |
| `sandbox.py` | 254 | Subprocess Python isolation |
| `nl_parser.py` | 168 | NL → stack code translator |
| `backends/__init__.py` | 63 | Auto-discovery registry: scans `*_ops.py` + `*_kb.py` |
| `backends/*_ops.py` | ~8,700 | 48 compute backends (functions that DO something) |
| `backends/*_kb.py` | ~800 | 4 knowledge backends (functions that LOOK UP something) |
| `learned_patterns.jsonl` | — | Self-learned error patterns (committed) |
| `tests/` | ~3,400 | 250 tests |
| `benchmark.py` | 227 | 40-problem eval (format-agnostic) |
