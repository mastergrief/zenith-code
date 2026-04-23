# CALM — Compute-Augmented Language Model Rules

> Historical receipts (sandbox-fix origin + commit ref, AST-walker
> cross-refs, feedback-loop validation-arc session provenance,
> auto-upgrade session-of-origin): see `MEMORY/atlas/calm_part_1.md`
> + `MEMORY/atlas/calm_part_2.md`.

## Core Principle

**Model reasons, backends compute, engine verifies.** Intelligence
comes from the system, not the weights. Adding a backend module is
equivalent to training — zero GPU cost, instant effect.

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

**40/40 (100%)** on benchmark. Model uses precomputed facts directly
— problems that took 30-165s drop to 1.7-2.3s.

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

**Pattern**: write a `*_ops.py` file in `calm/backends/`, export a
`*_FUNCTIONS` dict. Auto-discovery registers it — zero other files to
edit. Model gets smarter at that domain instantly.

Two types of backends coexist:
- **Compute backends** — deterministic functions (math, encoding, dates, etc.)
- **Knowledge backends** — factual lookup tables (countries, elements, constants, algorithms)

The engine doesn't care which type — same contract: pure function,
deterministic output, engine trusts it over the model.

### Current Backends (120 modules, 1002 functions, 550 NL patterns)

**81 compute backends** (`*_ops.py`), function counts in parens, grouped by domain area:

- **Math/numeric** — `math` (9: primes, GCD, factorize, fibonacci, collatz), `math_extended` (15: matrices, modular arithmetic, calculus), `bitwise` (18), `baseconv` (9), `roman` (3), `ratio` (9), `geometry` (19), `probability` (11), `financial` (10), `data` (11: stats), `algo` (13: sort, nCr, graphs), `wasm` (17: cross-check), `checksum` (8: Luhn/ISBN/EAN)
- **Strings/parsing** — `string` (7), `regex` (7), `json` (7), `csv` (9), `markdown` (7), `unicode` (7), `ast` (7), `yaml`/`toml`/`ini` via `config` (6), `sql` (6), `diff` (6), `cron` (6)
- **Dates/time** — `date` (6), `timezone` (7), `duration` (7)
- **Network/identity** — `network` (9: URL/IP/CIDR/HTTP), `cidr` (8), `http` (7: status codes), `uuid` (8), `jwt` (7), `color` (9: WCAG), `bytesize` (7: IEC vs SI)
- **Encoding** — `encoding` (12: b64/hex/md5/sha256), `semver` (6), `convert` (5: units)
- **Code analysis** — `code` (16), `security` (8: OWASP), `quality` (7: cyclomatic), `readability` (5: Flesch-Kincaid), `impact` (7: blast radius), `context` (7: git archaeology), `python` (9: builtin verify), `perf` (6: Big-O), `deps` (6), `refactor` (4), `type` (4), `test` (4), `doc` (4), `package` (6: pip/npm/cargo), `git` (7), `shell` (7: exit codes/dangerous), `creative` (9: brainstorm)

**39 knowledge backends** (`*_kb.py`, factual lookups, include `_DATA_VERSION`):
`country` (195 countries), `elements` (118 periodic table), `constants` (CODATA physical), `complexity` (algorithm Big-O), `port` (45 well-known), `ascii` (control chars/escapes), `license` (12 SPDX), `regex_ref` (common patterns), `error_code` (exit/errno/signals), `design_pattern` (22 GoF+modern) — plus 29 more domain-specific KBs.

Count of every backend: `ls calm/backends/`. Function detail: each module's `*_FUNCTIONS` dict + optional `*_NL_PATTERNS`.

### Adding a New Backend

**Naming**: `*_ops.py` for compute (functions that DO something),
`*_kb.py` for knowledge (functions that LOOK UP something). Knowledge
backends should include a `_DATA_VERSION` date for staleness tracking.

1. Create `calm/backends/mydom_ops.py` (or `mydom_kb.py`) with pure
   functions
2. Export: `MYDOM_FUNCTIONS = {"func_name": func, ...}`
3. Done — auto-discovery in `calm/backends/__init__.py` registers it
4. (Optional) Add NL precompute patterns in `precompute.py`
5. (Optional) Add claim verification patterns in `verify.py`

**Defense in depth**: Layer 2 (precompute) injects correct answers
before generation. Layer 1 (verify) catches wrong claims after
generation. Both should cover the same domains — when precompute
misses, verify is the safety net.

**Auto-learn guard**: `auto_learn.py` instantiates learned patterns
with numbers from the prompt. Large numbers (>10M) are skipped to
prevent combinatorial explosions (e.g. `factorial(N)` shouldn't fire
when a prompt contains a 16-digit credit card number).

## Auto-CALM Claim Verification

### Layer 1: Inline Claims

Extracts and verifies claims from model output:
- Arithmetic: `17 \times 23 = 391` (LaTeX + Unicode + plain)
- Functions: `factorial(N) = <value>`, `fibonacci(N) = <value>`
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

## Auto-Training Data Collection

Every correction generates distillation-compatible JSONL:
- `MathCollector` → `.calm_training/auto/math.jsonl`
- `BoolCollector` → `.calm_training/auto/bool.jsonl`
- `CodeCollector` → `.calm_training/auto/code.jsonl`

Virtuous cycle: corrections → training data → (optional) fine-tune →
fewer corrections. But backends are the primary path — training is
supplementary.

## Feedback loops — closed, tested, measured

Both self-learning subsystems have full test coverage and operator
visibility. The feedback loop is not "hoped to work" — it's proven
to work and quantifiable.

### AutoLearner (`calm/auto_learn.py`)

- `learn_from_correction(claim)` generalizes expressions (`17*23`
  → `"N * O"`, `is_prime(391)` → `"is_prime(N)"`). Frequency counter
  bumps on repeat, hit counter persists across reloads.
- `suggest_precomputes(prompt)` **shape-gates** pattern instantiation:
  function patterns require the function name (or alias) in the
  prompt; arithmetic patterns require the operator OR a natural-
  language form ("plus", "times", "multiplied"). Without this shape
  gate, patterns fire on any prompt with a number — injecting
  irrelevant precomputes alongside intended ones.
- `prune_cold_patterns(min_hits, min_frequency)` culls never-fired
  patterns that were only seen once; high-frequency patterns survive.
- Loop-closes unit tests at `calm/tests/test_auto_learn_loop.py`:
  correct `17 * 23` → next `347 * 289` prompt precomputes 100283
  correctly.
- Effectiveness harness at `calm/closed_loop_eval.py`: hit rate
  improves monotonically across rounds; pattern compression via
  generalization.

### ModuleLearner (`calm/module_learning.py`)

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
  NOT leak into multiplication-prompt facts section. Shape gate is
  load-bearing end-to-end.
- `test_verified_claim_does_not_learn` — correct first-time answer
  → zero patterns recorded. Guard against spurious accumulation.

### Operator visibility

```bash
PYTHONPATH=. python3 scripts/learning_dashboard.py
```

Prints both loops' current state — total patterns, hit counts, cold
patterns, recurring issues by module and context. Canonical ops
tool when diagnosing "why isn't the system precomputing my query?"

### The rule

When adding any new pattern-database / self-tuning component:
1. Write the loop-closes unit test.
2. Write the effectiveness harness (before/after on a held-out set).
3. Write the end-to-end integration test with mocked upstream.
4. Add to the dashboard.

`workflow.md` §"Feedback-loop validation pattern" codifies this as
a project-wide rule.

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

- **AST-only**: `ast.parse(mode="eval")` + recursive walker.
  Never `eval()`.
- **Whitelist**: only functions in `_FUNCTIONS` dict (500+ from all
  backends)
- **Comprehensions**: list/set/generator with per-variable scoping,
  10K limit
- **No attribute access, no imports** — all functions pre-registered

## Sandbox stdlib pre-import (`calm/sandbox.py`)

`run_python()` wraps user code in a subprocess with `_safe_import`
replacing `builtins.__import__`. The hook blocks `os`, `subprocess`,
`pathlib`, etc. — but fires on every `__import__`, including transitive
loads from stdlib modules.

Symptom: `import statistics` triggers `statistics`'s own `import os`
(platform detection during first load) → hook blocks → user can't use
`statistics.mean`, `hashlib.sha256`, etc.

**Fix**: pre-import safe stdlib modules BEFORE installing the hook,
so `sys.modules` is warm. Pre-imported: `re`, `math`, `random`,
`time`, `datetime`, `hashlib`, `base64`, `collections`, `itertools`,
`functools`, `bisect`, `heapq`, `copy`, `csv`, `statistics`, `typing`,
`enum`, `dataclasses`, `abc`, `struct`, `decimal`, `fractions`,
`textwrap`. Then hook installs; `os`/`subprocess` still blocked at
user level.

**Rule**: any new sandbox-blocked module added to the hook's block
set must be checked against the pre-import list for transitive
collisions. If a commonly-used safe stdlib module loads it, the
user-facing module must be pre-imported.

## Benchmark

40 problems, 6 categories:

| Mode | arithmetic | number_theory | sequences | algebra | reasoning | multi_step | Total |
|---|---|---|---|---|---|---|---|
| Auto-CALM + precompute | 10/10 | 10/10 | 5/5 | 5/5 | 5/5 | 5/5 | **40/40** |
| Explicit CALM (best) | 10/10 | 10/10 | 3-5/5 | 5/5 | 5/5 | 5/5 | 85-98% |
| Auto-CALM (no precompute) | 9/10 | 10/10 | 2/5 | 4/5 | 5/5 | 5/5 | 88% |

## CALM + retrieval

CALM's Layer 2 precompute and the `CodeExampleDB` hybrid retrieval
(see `retrieval.md`) are complementary, not overlapping:

- **Layer 2 precompute** — exact-oracle injection. Computes verified
  answers from problem text via the 1002 backend functions. When it
  hits, the answer is deterministically correct. Format: `"Verified
  facts: 17 * 23 = 391"`.
- **Hybrid retrieval** — approximate-pattern injection. Surfaces
  similar (problem, solution) pairs from the DB via TF-IDF+BM25 +
  Gemma-dense + RRF. When it hits, it shows a pattern Gemma can
  imitate; when it misses, nothing is injected.

**Policy** (per Tier-1 preservation thesis — see
`augmentation_thesis.md` §"Automatic Tier-1 preservation"):

- If Layer 2 precompute returns a direct answer → inject verified
  fact, SUPPRESS retrieval injection (the answer is exact; similar
  patterns don't help).
- If precompute has nothing AND retrieval top-k are all above a
  threshold → inject retrieval (this is where it helps).
- Otherwise → pass-through, let Gemma native-solve.

This gating mimics what substrate RAG (`KnowledgeStore` at L30) does
automatically via hash-match. For prompt-level CALM+retrieval we
implement it explicitly in `CodeVerifierFacade.compute_hints`.

## File Map (key entry points)

- `auto_calm.py` — Facade composing all layers, CLI entry
- `verify.py` / `precompute.py` / `intent_edit.py` — Layers 1/2/3
- `stream_auto.py` — Streaming + tool-call handler
- `engine.py` / `stream_engine.py` / `interceptor.py` — Explicit CALM (stop-mode + 4-tier parse)
- `engine_v2.py` / `router.py` — 7-phase pipeline + cognitive routing
- `expression.py` / `verifier.py` / `stack_vm.py` — AST-safe eval + 4-lane TMR
- `sandbox.py` — Subprocess isolation + stdlib pre-import
- `auto_learn.py` / `module_learning.py` — Self-learning (see §"Feedback loops")
- `adaptive.py` / `conversation.py` — Adaptive budget + cross-turn state
- `factual_check.py` / `confidence_check.py` / `specificity.py` — Quality checks
- `backends/__init__.py` — Auto-discovery; `backends/*_ops.py`, `*_kb.py` — 120 backend modules
- `learned_patterns.jsonl` — Self-learned error patterns (committed)
- `tests/` — 70 files / 565 test functions; `benchmark.py` — 40-problem eval

## Cognitive Intelligence Layer (39 modules)

**The system that makes the LLM reliable.** Each module catches a
specific failure mode that raw model output exhibits. The router
auto-selects relevant modules per prompt (85-180ms overhead).
Weighted quality scoring: issue-finding modules weigh 2-3× more than
silent ones.

### Module Categories

| Layer | Modules | What they catch |
|-------|---------|-----------------|
| **Verification** | chain_verify, consistency, logic, scope | Multi-step errors, contradictions, invalid syllogisms, overgeneralization |
| **Reasoning** | decompose, causal, assumptions, analogy, temporal, counterfactual, hypothesis_gen | Missing decomposition, wrong causation, hidden assumptions, bad analogies, ordering errors |
| **Quality** | creativity, nuance, evidence, relevance, completeness, explanation, density, precision, compression, error_recovery, specificity | Redundant ideas, vague hedging, unsupported claims, tangents, incomplete answers, circular explanations, filler, vague language, generic platitudes |
| **Meta-cognitive** | calibration, judgment, metacognition, goal_tracking, abstraction, perspective, uncertainty, communication, prerequisites | Domain confidence, structured evaluation, quality reports, goal drift, abstraction mismatch, missing perspectives, uncertainty propagation, expertise adaptation, knowledge gaps |
| **Planning** | prioritize, constraints, risk, disambiguation, provenance, conflict_resolution | Priority ranking, requirement tracking, risk assessment, ambiguity detection, trust tracking, module disagreement |

### Engine V2 Pipeline (`calm/engine_v2.py`)

```
prompt → PRE-ANALYZE (expertise, ambiguity, decompose, risk)
       → ENRICH system prompt (beginner→detailed, risks→mention, learned patterns)
       → ADAPTIVE BUDGET (2K trivial → 32K deep)
       → PRECOMPUTE (1002 backend functions + 550 NL patterns)
       → GENERATE (Gemma with enriched context)
       → VERIFY (Auto-CALM claim checking + factual cross-check)
       → COGNITIVE ROUTE (39 modules, 85-180ms, weighted scoring)
       → MODULE LEARNING (record recurring issues, normalized keys)
       → CROSS-TURN STATE (consistency, goals, calibration, quality trend)
       → SELF-HEAL (if weighted quality < 75%: targeted correction → re-verify)
       → response + quality report
```

Overhead: ~150ms on top of model inference. Self-healing only fires
when quality drops below threshold AND the correction improves quality.

## Auto-Upgrade Loop (CALM-as-verifier)

CALM corrections feed the substrate's persistent knowledge layer: wrong
prompt → CALM verifies → correction logged → end-of-session compile
into substrate weights → save `.pt` → next session errors fixed.

CALM's role is the oracle: `calm/llm_computer/calm_verifier.py` wraps
`safe_eval` + the 1002-function registry so any domain CALM can
evaluate automatically becomes a domain the learning loop can correct.
`CalmVerifier.verify_nl(prompt)` returns `(expr, value)`; feed `value`
to `KnowledgeStore.add_correction(key, value)`. Existing correction
logs at `.calm_training/auto/` feed this pipeline — same corrections
generate training data AND compile directly to weights.

Compile/install mechanics + recall card structure (3 ReGLU per fact,
`CardSlot.attach(preserve=True)` until FFN migration ships):
`Substrate.md` §"Persistent Knowledge + Auto-Upgrade".

## Related rules

- `augmentation_thesis.md` — tier-1/2/3 framework, Tier-1 preservation
- `Substrate.md` — install modes for the CALM-produced recall cards
- `retrieval.md` — hybrid retrieval that complements Layer 2 precompute
- `workflow.md` §"Feedback-loop validation pattern" — the loop
  validation rule
- `MEMORY/atlas/calm_part_1.md` + `calm_part_2.md` — full historical
  part_1/part_2 receipts preserved
