**Part 2**

### The rule

When adding any new pattern-database / self-tuning component:
1. Write the loop-closes unit test.
2. Write the effectiveness harness (before/after on a held-out set).
3. Write the end-to-end integration test with mocked upstream.
4. Add to the dashboard.

`.claude/rules/workflow.md` §"Feedback-loop validation pattern"
codifies this as a project-wide rule.

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
- **Whitelist**: only functions in `_FUNCTIONS` dict (500+ from all backends)
- **Comprehensions**: list/set/generator with per-variable scoping, 10K limit
- **No attribute access, no imports** — all functions pre-registered

## Sandbox stdlib pre-import (`calm/sandbox.py`, R53.22 fix, commit `5dc2dc1`)

`run_python()` wraps user code in a subprocess with `_safe_import`
replacing `builtins.__import__`. The hook blocks a set including `os`,
`subprocess`, `pathlib`, etc. **But the hook fires on every
`__import__`, including transitive ones from stdlib modules.**

Symptom that forced the fix: `import statistics` inside user code
triggered `statistics`'s own `import os` (for platform detection during
first load) → hook blocked → `ImportError: blocked: os`. User couldn't
use `statistics.mean`, `hashlib.sha256`, etc. — csv_column_stats eval
stuck at 0/0 even with imports injected **(pre-R53.22 fix;
post-R53.35 reaudit csv is further unblocked by the `ast_repair`
walker's `syntax_repair` pass — 0/0 → 8/8 on live Gemma per
`MEMORY/atlas/capability_gain_arc.md` §"R53.35"; post-2026-04-21 walker has 7
rewrites — shadow_rename, dict-key synonym, syntax_repair (3
original), plus `fuzzy_rename_function` driven by `NameError`
extraction via Jaccard-similarity name matching (commit `805e539`))**.

**Fix**: pre-import safe stdlib modules BEFORE installing the hook, so
`sys.modules` is warm and user-level `import X` hits cache without
triggering new transitive loads:

```python
# Runs OUTSIDE the hook — pre-warms sys.modules
import re as _pre_re, math as _pre_math, random as _pre_random
import time as _pre_time, datetime as _pre_datetime
import hashlib as _pre_hashlib, base64 as _pre_base64
import collections as _pre_collections, itertools as _pre_itertools
import functools as _pre_functools, bisect as _pre_bisect
import heapq as _pre_heapq, copy as _pre_copy
import csv as _pre_csv, statistics as _pre_statistics
import typing as _pre_typing, enum as _pre_enum
import dataclasses as _pre_dataclasses, abc as _pre_abc
import struct as _pre_struct, decimal as _pre_decimal
import fractions as _pre_fractions, textwrap as _pre_textwrap

# Then hook installs; os/subprocess still blocked
```

**User `import os` remains blocked** — `os` is NOT in the pre-import
list. Verification: `import statistics; statistics.mean([1,2,3])` ✓;
`import hashlib; hashlib.sha256(...)` ✓; `import os; os.getcwd()` →
still `ImportError: blocked: os`.

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

## CALM + retrieval (R53 Phase 1)

CALM's Layer 2 precompute and the new `CodeExampleDB` hybrid retrieval
(see `retrieval.md`) are complementary, not overlapping:

- **Layer 2 precompute** — exact-oracle injection. Computes verified
  answers from problem text via the 1002 backend functions. When it
  hits, the answer is deterministically correct. Format:
  `"Verified facts: 17 * 23 = 391"`.
- **Hybrid retrieval** — approximate-pattern injection. Surfaces
  similar (problem, solution) pairs from the 8970-example DB via
  TF-IDF+BM25 + Gemma-dense + RRF. When it hits, it shows a pattern
  Gemma can imitate; when it misses, nothing is injected.

Policy (per R53.2b finding — see `augmentation_thesis.md` §"Tier-1
preservation"):

- If Layer 2 precompute returns a direct answer → inject verified
  fact, SUPPRESS retrieval injection (the answer is exact; similar
  patterns don't help).
- If precompute has nothing AND retrieval top-k are all above a
  threshold → inject retrieval (this is where it helps).
- Otherwise → pass-through, let Gemma native-solve.

This gating mimics what substrate RAG (`KnowledgeStore` at L30) does
automatically via hash-match. For prompt-level CALM+retrieval we have
to implement it explicitly in `CodeVerifierFacade.compute_hints`.

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
| `expression.py` | 657 | AST-safe eval, `_FUNCTIONS` dict (500 from registry) |
| `verifier.py` | 559 | 4-lane TMR verification |
| `stack_vm.py` | 522 | Reference stack machine |
| `sandbox.py` | ~280 | Subprocess Python isolation + stdlib pre-import |
| `nl_parser.py` | 168 | NL → stack code translator |
| `backends/__init__.py` | 77 | Auto-discovery registry: scans `*_ops.py` + `*_kb.py` + `*_NL_PATTERNS` |
| `backends/*_ops.py` | ~14,500 | 81 compute backends with NL patterns |
| `backends/*_kb.py` | ~4,600 | 39 knowledge backends with `_DATA_VERSION` |
| `engine_v2.py` | 414 | Full 7-phase cognitive pipeline with self-healing |
| `router.py` | ~850 | Cognitive router: 39 modules, weighted quality scoring |
| `adaptive.py` | 130 | Adaptive thinking budget (2K→32K based on complexity) |
| `conversation.py` | 130 | Cross-turn state: consistency, goals, calibration |
| `module_learning.py` | 176 | Learns recurring issues → prompt prevention (normalized keys, commit `054d477`) |
| `factual_check.py` | ~300 | 48 static + 10 dynamic cross-check patterns |
| `confidence_check.py` | ~130 | Overconfidence detection (absolutes, false certainty) |
| `specificity.py` | ~140 | Generic advice detection (platitudes, hand-waves) |
| `39 cognitive modules` | ~7,500 | See Cognitive Intelligence Layer below |
| `learned_patterns.jsonl` | — | Self-learned error patterns (committed) |

## Cognitive Intelligence Layer (39 modules)

**The system that makes the LLM reliable.** Each module catches a specific
failure mode that raw model output exhibits. The router auto-selects
relevant modules per prompt (85-180ms overhead). Weighted quality scoring
(commit `4fee43a`): issue-finding modules weigh 2-3× more than silent ones.

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

Overhead: ~150ms on top of model inference. Self-healing only fires when
quality drops below threshold AND the correction improves quality.

## Auto-Upgrade Loop (session 30)

CALM corrections feed the substrate's persistent knowledge layer:

```
User queries → CALM verifies → wrong? → correction logged
  → end of session: compile corrections into substrate weights
  → save .pt → next session: errors permanently fixed
```

**AutoUpgradeEngine** (`calm/llm_computer/auto_upgrade.py`):
- `query_with_verification(prompt)` — CALM verifies, logs if wrong
- `commit()` — compiles all corrections into knowledge card, installs
  into substrate via `install_compiled_card_hybrid`, saves .pt
- Each correction = 3 ReGLU neurons: `indicator(x == key)` step function
- Proven: 0/8 correct → 8/8 → 11/11 across 3 sessions, zero retraining

**KnowledgeStore** (`calm/llm_computer/persistent_knowledge.py`):
- `add_correction(key, value)` — deduplicates, latest wins
- `build_recall_model()` — compiles to Small2DTransformer
- `save_corrections() / load_corrections()` — JSON persistence
- Overrides work: key 7 changed 6→3, old fact replaced

**Install mode**: the recall card is FFN-only (ReGLU step functions +
LinearHead) so it installs via `CardSlot.attach(preserve=True)` today
— the `install_card_in_attention` path writes `attn_q/k/v/output` only
and has no FFN migration yet. See `MEMORY/atlas/Substrate_arc.md`
§"Card Installation" for the mode tradeoff table and known limits.

**CALM-as-verifier**: `calm/llm_computer/calm_verifier.py` (Round 5)
wraps `safe_eval` + 1002-function registry as the oracle for the
learning loop. Replaces per-domain hand-rolled verifiers — any domain
CALM can evaluate automatically becomes a domain the loop can
correct. `CalmVerifier.verify_nl(prompt)` returns `(expr, value)`;
feed `value` to `KnowledgeStore.add_correction(make_key(prompt), ...)`.

**Integration**: CALM's existing correction logs (`.calm_training/auto/`)
feed the auto-upgrade pipeline. The same corrections that generate
training data for optional fine-tuning ALSO compile directly into
substrate weights for instant, verified persistence.

| `tests/` | ~3,400 | 70 test files / 565 test functions across calm/ |
| `benchmark.py` | 227 | 40-problem eval (format-agnostic) |
