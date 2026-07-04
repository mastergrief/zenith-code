---
paths:
  - "calm/**"
  - "scripts/*calm*.py"
  - "scripts/learning_dashboard.py"
  - "scripts/minibench_calm_trained.py"
---

# CALM — Compute-Augmented Language Model Rules

> Historical receipts: `MEMORY/atlas/calm_part_1.md` +
> `MEMORY/atlas/calm_part_2.md`.

## Core Principle

**Model reasons, backends compute, engine verifies.** Intelligence
comes from the system, not the weights. Adding a backend module is
equivalent to training — zero GPU cost, instant effect. Auto-CALM hides
the engine from the model.

## Two Modes

### Auto-CALM (default) — `calm/auto_calm.py`

Precompute → generate → Layer 1 inline claim verify → Layer 2 answer
cross-check → Layer 3 intent-to-edit self-heal. **40/40** benchmark with
precompute. Full flow diagrams: `calm_part_1.md` §"Two Modes".

### Explicit CALM (power user) — `calm/engine.py`, `calm/stream_engine.py`

Model emits `<calm>...</calm>` blocks; engine stop-mode executes with
4-tier parse + TMR. **85-98%** on benchmark. Thinking + stop incompatible
→ hybrid plan-then-execute.

## Modular Backend Architecture

**Pattern**: `calm/backends/*_ops.py` or `*_kb.py` exporting a
`*_FUNCTIONS` dict — auto-discovery registers it. Compute backends DO;
knowledge backends LOOK UP (include `_DATA_VERSION`).

### Current Backends (120 modules, 1002 functions, 550 NL patterns)

Full domain inventory + counts: `calm_part_1.md` §"Current Backends".
Count live: `ls calm/backends/`.

### Adding a New Backend

1. Create `calm/backends/mydom_ops.py` (or `mydom_kb.py`) with pure
   functions
2. Export: `MYDOM_FUNCTIONS = {"func_name": func, ...}`
3. Done — auto-discovery in `calm/backends/__init__.py` registers it
4. (Optional) NL precompute patterns in `precompute.py`
5. (Optional) Claim verification patterns in `verify.py`

**Defense in depth**: Layer 2 precompute + Layer 1 verify cover the
same domains. **Auto-learn guard**: skip numbers >10M to prevent
combinatorial explosions on learned patterns.

## Auto-CALM Claim Verification

Three layers — detail + examples: `calm_part_1.md` §"Auto-CALM Claim
Verification".

- **Layer 1**: extract arithmetic/function/boolean claims from output;
  skip conditional contexts; verify on CPU.
- **Layer 2**: precompute from prompt before generation; NL patterns +
  multi-turn retry on answer mismatch.
- **Layer 3**: intent-to-edit — diagnose → template fix → verify;
  self-heal once if failures remain.

## Auto-Training Data Collection

Every correction → distillation JSONL (`.calm_training/auto/`).
Backends are primary; fine-tuning is supplementary.

## Feedback loops — closed, tested, measured

Self-learning must be proven, not hoped. Shape-gate pattern instantiation
(function name / operator / NL alias required). Dashboard:
`PYTHONPATH=. python3 scripts/learning_dashboard.py`.

### The rule

When adding any new pattern-database / self-tuning component:
1. Write the loop-closes unit test.
2. Write the effectiveness harness (before/after on a held-out set).
3. Write the end-to-end integration test with mocked upstream.
4. Add to the dashboard.

`workflow.md` §"Feedback-loop validation pattern" codifies this as
a project-wide rule.

Test receipts (AutoLearner, ModuleLearner, integration mocks):
`calm_part_1.md` §"Feedback loops".

## Verification (`calm/verifier.py`)

4-lane TMR: primary backend, cross-check, alternate algorithm, proof lane.
DIVERGENCE → halt; VERIFIED → safe.

## Expression Evaluator (`calm/expression.py`)

AST-only eval (`ast.parse(mode="eval")`); whitelist `_FUNCTIONS`; no
`eval()`, no attribute access, no imports. Comprehensions with scoping +
10K limit.

## Sandbox stdlib pre-import (`calm/sandbox.py`)

`_safe_import` hook blocks `os`/`subprocess`/etc. but fires on transitive
stdlib loads — pre-warm safe modules before hook install. **Rule**: new
blocked modules → check transitive collisions against pre-import list.
Full fix receipt: `calm_part_2.md` §"Sandbox stdlib pre-import".

## Benchmark

40-problem, 6-category eval — Auto-CALM + precompute **40/40**. Table:
`calm_part_2.md` §"Benchmark".

## CALM + retrieval

Layer 2 precompute (exact oracle) and `CodeExampleDB` hybrid retrieval
(see `retrieval.md`) are complementary:

- Precompute hit → inject verified fact, **suppress** retrieval.
- No precompute + retrieval above threshold → inject pattern.
- Otherwise → pass-through native solve.

Policy: `augmentation_thesis.md` §"Automatic Tier-1 preservation".
Gating in `CodeVerifierFacade.compute_hints`.

## File Map (key entry points)

`auto_calm.py`, `verify.py` / `precompute.py` / `intent_edit.py`,
`engine_v2.py` / `router.py`, `expression.py` / `verifier.py`,
`auto_learn.py` / `module_learning.py`, `backends/__init__.py`,
`tests/` (565 functions). Full LOC map: `calm_part_2.md` §"File Map".

## Cognitive Intelligence Layer (39 modules)

Router auto-selects failure-mode modules per prompt (~85-180ms).
Weighted quality scoring; self-heal below 75%. Module table + Engine V2
pipeline: `calm_part_2.md` §"Cognitive Intelligence Layer".

## Auto-Upgrade Loop (CALM-as-verifier)

CALM corrections → `KnowledgeStore` → compile into substrate weights →
save `.pt` → next session fixed. Oracle: `calm/llm_computer/calm_verifier.py`
(`verify_nl` → `add_correction`). Install mechanics:
`Substrate.md` §"Persistent Knowledge + Auto-Upgrade". Session receipts:
`calm_part_2.md` §"Auto-Upgrade Loop".

## Related rules

- `augmentation_thesis.md` — tier-1/2/3 framework, Tier-1 preservation
- `Substrate.md` — install modes for CALM-produced recall cards
- `retrieval.md` — hybrid retrieval complementing Layer 2 precompute
- `workflow.md` §"Feedback-loop validation pattern" — loop validation rule
- `MEMORY/atlas/calm_part_1.md` + `calm_part_2.md` — historical receipts
