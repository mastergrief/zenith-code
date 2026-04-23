# CALM — Compute-Augmented Language Model Rules

**Part 1**

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

### Current Backends (120 modules, 1002 functions, 550 NL patterns)

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
| `geometry_ops` | 19 | circle, sphere, cone, trapezoid, distance, polygon angles |
| `probability_ops` | 11 | dice, coin, binomial, Bayes, permutations |
| `roman_ops` | 3 | Roman numeral ↔ decimal, validation |
| `financial_ops` | 10 | compound interest, loan payments, NPV, ROI, rule of 72 |
| `ratio_ops` | 9 | simplify fractions, percent change, decimal↔fraction |
| `cidr_ops` | 8 | subnet mask, host count, IP-in-subnet, overlap, private IP |

**Knowledge backends** (`*_kb.py` — factual lookups, include `_DATA_VERSION`):

| Backend | Funcs | Domain |
|---|---|---|
| `country_kb` | 8 | capitals, ISO codes, currencies, calling codes (195 countries) |
| `elements_kb` | 9 | periodic table: symbols, weights, electron config (118 elements) |
| `constants_kb` | 5 | physical constants: speed of light, Planck, Avogadro (CODATA 2018) |
| `complexity_kb` | 5 | sort/DS/graph algorithm complexity |
| `port_kb` | 5 | well-known ports: SSH=22, MySQL=3306, PostgreSQL=5432 (45 ports) |
| `ascii_kb` | 7 | control chars, escape sequences, CR vs LF, line endings |
| `license_kb` | 5 | SPDX licenses: MIT, GPL, Apache permissions/copyleft (12 licenses) |
| `regex_ref_kb` | 4 | common regex patterns (email, URL, IP, UUID) + syntax reference |
| `error_code_kb` | 4 | exit codes, POSIX errno, Unix signals |
| `design_pattern_kb` | 5 | 22 GoF + modern patterns: intent, participants, when to use |

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

## Feedback loops — closed, tested, measured (Vector 1, session 26)

Both self-learning subsystems have full test coverage and operator
visibility. The feedback loop is no longer "hoped to work" — it's
proven to work and quantifiable.

### AutoLearner (`calm/auto_learn.py`, 17 tests)

- `learn_from_correction(claim)` generalizes expressions (`17*23`
  → `"N * O"`, `is_prime(391)` → `"is_prime(N)"`). Frequency
  counter bumps on repeat, hit counter persists across reloads.
- `suggest_precomputes(prompt)` **shape-gates** pattern instantiation:
  function patterns require the function name (or alias: "prime",
  "fib", "factorial", "gcd", ...) in the prompt; arithmetic patterns
  require the operator OR a natural-language form ("plus", "times",
  "multiplied"). Before this (phase 1 defect), patterns fired on any
  prompt with a number — injecting irrelevant precomputes alongside
  intended ones.
- `prune_cold_patterns(min_hits, min_frequency)` culls never-fired
  patterns that were only seen once; high-frequency patterns survive.
- Tests at `calm/tests/test_auto_learn_loop.py` prove the loop
  closes: correct `17 * 23` → next `347 * 289` prompt precomputes
  100283 correctly.
- Effectiveness harness at `calm/closed_loop_eval.py` measured
  **90% → 100% hit rate over 3 rounds of 20 corrections each** with
  10× pattern compression via generalization.

### ModuleLearner (`calm/module_learning.py`, 11 tests)

- `record(module, issue_type, context)` tracks recurring cognitive-
  module issues with normalized keys.
- `suggest_prompt_additions(prompt)` returns prevention strings for
  context-matched issues seen ≥ 3 times. Context detection routes
  between comparison / debugging / explanation / design / operations
  / general.
- Tests at `calm/tests/test_module_learning_loop.py` prove the
  3-occurrence threshold works and preventions don't leak between
  contexts.

### End-to-end integration (`calm/tests/test_auto_calm_integration.py`)

Mocks `_generate` inside `AutoCalmEngine`, exercises the full
pipeline without live Gemma. Three tests:
- `test_loop_closes_in_auto_calm_engine` — round 1 LLM emits wrong
  answer → verifier + learner record pattern → round 2 prompt sees
  "Verified facts: ..." in system prompt BEFORE generation.
- `test_loop_shape_gate_prevents_noise` — factorial pattern does
  NOT leak into multiplication-prompt facts section. Phase-2 shape
  gate is load-bearing end-to-end.
- `test_verified_claim_does_not_learn` — correct first-time answer
  → zero patterns recorded. Guard against spurious accumulation.

### Operator visibility

```bash
PYTHONPATH=. python3 scripts/learning_dashboard.py
```

Prints both loops' current state — total patterns, hit counts, cold
patterns, recurring issues by module and context. Canonical ops tool
when diagnosing "why isn't the system precomputing my query?"
