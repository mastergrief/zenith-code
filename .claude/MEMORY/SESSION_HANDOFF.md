# Session Handoff — 2026-04-14 (Session 26)

## Goal

Build out the CRLM (compute-redistributed language model) architecture
across two research vectors:

1. **Vector 1 — close the feedback loop**: turn `AutoLearner` and
   `ModuleLearner` from "hoped to work" into "tested, measured, with
   operator visibility." Every production correction becomes permanent
   capability.

2. **Vector 2 — meta-structure HRM**: test the CRLM scaling law by
   pooling multiple domains into a single 48K HRM, measuring whether
   cross-domain exposure (a) holds in-distribution accuracy, (b) breaks
   single-domain ceilings, (c) generalizes to held-out formats.

Plus: the session-26 plan's 4 steps (3-digit math, LookUp/ReGLU IR,
adder, NL→math HRM) and follow-on compiler work (LookUpExact, semantic
KV, auto-scheduler, HullKVCache parity) that preceded the vectors.

Also: write forward-looking research roadmap and current-state SPEC to
`.claude/MEMORY/`.

User constraints: **no subagents** (direct `Edit`/`Write`/`Read`/
`Grep`/`Bash` only — per `feedback_no_agents`). **Hypothesis, build,
test, iterate.** R&D-at-small-scale only; Zenith harness integration
explicitly deferred.

---

## Completed — 28 commits from session-25 handoff (`91721f2..82055fa`)

All commits on `feature/multi-agent-qwen`. 313 tests collected and
passing across `calm/`, `calm/hrm/tests/`, `calm/llm_computer/tests/`.

### Session-26 plan (4 steps, from approved plan
`/home/gabe/.claude/plans/splendid-swinging-gosling.md`):

- **`4344f6c`** `hrm: 3-digit operand range + 500 epochs (smoke 5/5, held-out 100%)`
  - First step: bumped `_arithmetic_simple` range 99→999 in `calm/hrm/data.py`.
  - Initial `--epochs 100` failed at 26.7% full-expression (cosine LR
    to 0 too early, per documented pitfall). Fix: `--epochs 500`.
  - Decisively established the "ALWAYS `--epochs 500`" rule (observed
    4 times across session 26).
- **`c7c56c1`** `llm_computer: promote LookUp + ReGLU to first-class IR + declarative compiler`
  - `calm/llm_computer/gate_graph.py`: added `TokenEmbed`, `PosEmbed`,
    `LookUp`, `ReGLU`, `LinearHead` as first-class hardware nodes.
  - `calm/llm_computer/compile.py`: rewrote with `compile_program()`
    that walks hardware nodes, populates weights declaratively.
- **`fdc169f`** 1-digit adder — compositional test of LookUp+ReGLU.
- **`dbc5ef5`** 2-digit adder — 10,000/10,000 exhaustive in 0.38s,
  486K params, proof that IR scales via composition.
- **`292bfb0`** `hrm: integration #3 — NL → math expression HRM`
  - `calm/hrm/nl_data.py`, `scripts/train_hrm_nl.py`,
    `scripts/eval_hrm_nl.py`.
  - 48K params, 29/30 held-out, smoke 5/5.

### Session-26 follow-ons (word problems, compiler deferred work):

- **`fea22aa`** Word problems at 48K, 30/30 held-out.
- **`ae00b03`** Parabolic-key `LookUpExact` — data-dependent
  retrieval (`retrieve_by_index`, 256/256 exhaustive).
- **`15d4c9b`** `retrieve_threshold` — same-layer attn+FFN composition.
- **`d791121`** Semantic-key `LookUpExact` via ReGLU key-squaring
  (`read_by_key`, 96/96 across 4! × 4 perm/query combos).
- **`b5f27c8`** Greedy auto-scheduler (`calm/llm_computer/schedule.py`).
- **`071329a`** `HullKVCache` parity validated against batched
  hard-max attention on production programs.
- **`a43f0d2`** GSM-style word problems — first observed CRLM ceiling
  (93% at 48K due to digit transposition on 2-digit operands).

### Vector 1 — feedback loops (4 phases):

- **`661ef74`** Phase 1: first tests for `AutoLearner` (11 tests) +
  effectiveness harness (`calm/closed_loop_eval.py` showing
  90%→100% hit rate over 3 rounds).
- **`c5057d0`** Phase 2: hit-tracking per-pattern + shape-gated
  matching. Fixed real defect: patterns were firing on every numeric
  prompt (function patterns matching "what is 5*7?"), flooding system
  prompt with irrelevant precomputes.
- **`de9673a`** Phase 3: `ModuleLearner` parity tests (11 tests) +
  unified operator dashboard at `scripts/learning_dashboard.py`.
- **`b18845b`** Phase 4: end-to-end integration test at
  `calm/tests/test_auto_calm_integration.py` — mocks LLM generation
  inside `AutoCalmEngine`, proves learned patterns inject into
  system prompt on round 2. Three tests, all pass.

### Vector 2 phase 1 — multi-task HRM:

- **`bfd660f`** Multi-task 48K HRM across 4 pooled domains (math, NL,
  word, GSM). 500 epochs, best val_acc 100%.
- **`a3b33f1`** Per-domain eval: **30/30 on every domain**. GSM ceiling
  (93% per-domain) broken to 100% via cross-domain exposure.
  - Key finding: math-echo training sharpens digit-copy discipline
    that transfers to GSM's operand-localization failures.
  - "CRLM scaling ceiling was domain-isolation-bound, not
    architecture-bound."

### Vector 2 phase 2 — scaling experiments:

- **`08cca78`** Distribution-scaling curve — 17% → 28% → 50% → 100%
  held-out OOD.
  - Experiment 1 (capacity probe): h=64, 179K params. OOD 28%, +10pp
    from h=32. Sub-linear scaling.
  - Experiment 2a (distribution probe, 10 formats): h=32, 48K params,
    multi10 training set. OOD 50%, +33pp. Distribution wins.
  - Experiment 2b (distribution probe, 20 formats): h=32, 48K params,
    multi20 training set. OOD **100% on multi10 held-out** (clean
    comparison). 22% on multi20's harder held-out (reasoning-required
    cases — architecturally out of scope).
  - New artifacts: `calm/hrm/extended_data.py`,
    `calm/hrm/extended2_data.py`, `calm/hrm/multi10_data.py`,
    `calm/hrm/multi20_data.py`, `scripts/train_hrm_multi{10,20}.py`,
    `scripts/eval_hrm_{ood,multi10_ood,multi20_ood}.py`.

### Docs / SPEC / roadmap:

- **`3e2a771`** `docs: update rules + CLAUDE.md + write CRLM_SPEC`.
  Deleted stale `.claude/rules/vdd.md` and `orchestration.md` (both
  referenced agents removed in `bb7f13d`). Rewrote `CLAUDE.md` to
  reflect session-26 state. New `.claude/MEMORY/CRLM_SPEC.md` (387
  lines, 12 sections).
- **`82055fa`** `docs: RESEARCH_ROADMAP — 8-layer CRLM progression`.
  Forward-looking spec. Layers 0-8 from current state to formally-
  verified AI. Pace calibrated to this project's observed iteration
  speed (minutes-per-experiment, days-per-research-direction,
  10-100× faster than conventional research-team estimates).

### Key decisions + reasoning

- **"Always `--epochs 500`"** elevated from pitfall to rule. Observed
  4 times: math 3-digit, NL templates, word problems, GSM. Cosine LR
  to 0 at 100 epochs systematically under-fits NL-input HRMs.
- **Shape-gated pattern matching** (Vector 1 phase 2) was a defect
  fix, not a feature. Before the fix, function patterns like
  `factorial(N)` would fire on any prompt with a number. The shape
  gate (require function name / operator or NL alias in prompt)
  prevents pattern pollution.
- **Kill training early at val_acc saturation** — consistent with
  `best_val_acc` checkpointing. Applied to multi-task, multi10,
  multi20, h=64 capacity probe. Saves ~30-50% of training time with
  no impact on final checkpoint quality.
- **Distribution > capacity** — decisive empirical finding. 3.7×
  params gave +10pp OOD; 2.5× formats gave +33pp; 5× formats gave
  +83pp to perfect. Scaling story inverted from LLM-field defaults.
- **Multi-task broke GSM ceiling** — key finding for Vector 2. Same
  48K params, additional exposure to math-echo transferred to GSM's
  digit-localization, 93% → 100%.
- **Roadmap time estimates recalibrated** — user pushed back on
  conventional "months/years" estimates. Reality: sessions 24-26
  shipped ~3 months of traditional research-team output in days.
  TurboQuant tq4 custom CUDA kernels were 1 day. Roadmap now says
  Layer 6 (differentiable substrate) is 4-8 weeks, not 12-18 months.
- **No Zenith integration this session** — explicitly deferred by
  user. "R&D at small scale, not product integration." The HRM →
  `precompute.py` wiring stays in the CRLM_SPEC's "deferred" list.

---

## In Progress — nothing left hanging

All active work shipped. Multi20 training was killed early at epoch 50
(val_acc 99.9%) per the established pattern; checkpoint is at
`calm/hrm/checkpoints/multi20_best.pt`.

Two training monitors show "timed out" in the task list — those are
stale from multi10 / multi20 runs that were already killed for early
OOD eval. Ignore.

---

## Next Steps (ordered by priority)

### 1. Vector 2 Phase 3: Meta-learning (operation inference) — HIGH PRIORITY

The sharp next question. Distribution-scaling solved **format
invariance** (20 formats → 100% on held-out format variations). The
remaining 22% failure on multi20's own held-out test is all
**operation inference** — cases like "half of 80", "what is 50 percent
of 80", "how much does X exceed Y" that require the model to infer
a novel computational operation, not just extract structure.

**Experiment design:**
- Modify training pipeline: encoder input = `[3 example (input, output)
  pairs] + [query]` instead of just `[query]`.
- Target stays the same (math expression).
- Extend `max_enc` to ~512 to hold prefix + query.
- Train at h=32; if held-out operation-inference test climbs to ≥ 70%,
  meta-learning works at 48K. If not, scale to h=128 / h=256 per the
  `architecture.md` rule (1-10M params predicted for NL→structured
  with variation).

**Specific held-out operations to test**:
- `half of X` → `X / 2`
- `double of X` → `X * 2`
- `X percent of Y` → `X / 100 * Y`
- `how much more is X than Y` → `X - Y`
- `which is larger, X op1 Y or Z` → compute + compare

**Files to create**:
- `calm/hrm/meta_data.py` — few-shot-prefixed dataset generator
- `scripts/train_hrm_meta.py` — trainer with the meta-learning
  training regime
- `scripts/eval_hrm_meta_ood.py` — evaluates on operations never seen
  in training (only shown via in-context examples at inference)

**Time estimate at our pace**: 1-3 days of focused iteration. The data
pipeline change is mechanical; the research question is whether
48K has the capacity for meta-learning at all.

### 2. Vector 3 Phase 1: Compile 3 representative backends + dispatcher

The differentiable substrate direction. Layer 6 from `RESEARCH_ROADMAP.md`.
Picked as "most interesting" by the user mid-session.

**What it is**: compile gcd, factorial, is_prime as gate-graph programs.
Add an opcode-based dispatcher at position 0 (hard-max attention on
opcode channel routes to the right sub-graph). Result: one
`Small2DTransformer` that evaluates any of the 3 backends.

**Dependencies met**: we have `LookUpExact` (parabolic + semantic),
`ReGLU` composition, auto-scheduler. Everything the paper's §8 WASM
interpreter needs at small scale is in the IR already.

**Files to create**:
- `calm/llm_computer/programs/gcd.py` — gate-graph impl of Euclidean gcd
- `calm/llm_computer/programs/factorial.py` — iterative factorial
  via counter + product
- `calm/llm_computer/programs/is_prime.py` — trial division with step
  functions
- `calm/llm_computer/programs/dispatched.py` — opcode-routing wrapper
- `calm/llm_computer/tests/test_dispatched.py` — correctness suite

**Estimated time**: 1-2 weeks. Hardest part is the looping constructs
(factorial's iterative loop, is_prime's bounded trial division). Paper's
§4b (cumsum for instruction pointer) is the construction pattern.

### 3. Commit the old/legacy checkpoints OR gitignore them

Five HRM checkpoints are untracked but present on disk:
`math_hrm_best.pt`, `math_scratchpad_best.pt`, `math_seq2seq_best.pt`,
`math_structure_2digit.pt.bak`. These are legacy from earlier training
rounds (1a, 1c, 1d, pre-3-digit-bump). The CRLM_SPEC flagged them as
"could delete."

Decision for next session: either commit them for audit trail OR add
to `.gitignore`. Current state is untracked-and-present which is ugly.

### 4. Optional: run a 40-format scaling test

If the 20-format curve shows a cliff-to-100%, does 40 formats help the
"reasoning-required" cases? It probably doesn't (they're a different
kind of problem) but it would be a cheap data point — ~30 min training.

### 5. Update the old monitor/runtime state files

Untracked: `.claude/scheduled_tasks.lock`, `.port_sessions/`,
`calm/.module_learning.json`. Probably belong in `.gitignore`. The
`calm/.module_learning.json` file accumulates production state and
shouldn't be tracked.

---

## Key Context — things that save time if known upfront

### Hardware + environment

- RTX 4070 Laptop GPU, 8 GB VRAM. WSL2 Ubuntu 24.04.
- All HRM training runs fit easily. Largest checkpoint (h=64 multi-task)
  was 179K params.
- Training time patterns:
  - 48K HRM + 2K samples + 500 epochs = ~145s (math)
  - 48K HRM + 4K samples + 500 epochs = ~1371s (multi-task)
  - 48K HRM + 10K samples + 100 epochs = ~783s (multi10, killed early)
  - 48K HRM + 20K samples + 50 epochs = ~664s (multi20, killed early)

### CRLM scaling law as of this session

**Format invariance axis (Layer 2):**
- Distribution-scaling is decisively more effective than capacity-
  scaling. 20 formats + 48K params = 100% held-out OOD.
- Per-format training cost: ~1 engineer-hour design + ~15 min GPU.
- Production recipe is concrete: 20-40 baseline formats + nightly
  retrain from production failure clusters.

**Operation inference axis (Layer 3):**
- Distribution-scaling does NOT solve novel-operation inference.
- 22% on "half of X", "percent of Y", comparison cases even with 20
  format training.
- Architecture is a structure extractor, not a reasoner. Expected.

### Workflow rules reinforced this session

- **Always `--epochs 500`** — the cosine LR schedule needs the full
  sweep, rely on `best_val_acc` to pick the right checkpoint.
- **Monitor + grep-filtered tail** with `setsid` + `disown` + `<
  /dev/null` — the detached training pattern that survives WSL
  hiccups.
- **Kill early on val_acc saturation** — consistent with
  `best_val_acc` already saved; remaining epochs just decay LR
  against zero gradient.
- **Feedback-loop validation pattern** (new in `workflow.md`): unit
  tests for cycle-closes + effectiveness harness + end-to-end
  integration with mocked dependencies. Applied to both AutoLearner
  and ModuleLearner this session.

### Failed approaches (don't retry)

- **Capacity scaling as the primary OOD lever** — h=64 gave +10pp for
  3.7× params. Sub-linear and expensive. Don't reach for this first;
  distribution-scaling is ~5× better per effort.
- **Killing training at epoch 100 by default** — observed 4 times to
  under-fit. Always run the full 500 epochs schedule.
- **Assuming multi-task training would match per-domain** — it
  *exceeded* per-domain (100% vs 93% on GSM). Don't hedge expectations
  downward; pool aggressively when substrate is shared.
- **Big-model time estimates** — conventional research-team
  calibration is wrong for this project's pace. Drop estimates
  10-100× to reflect actual iteration speed.

### Memory + preferences

- User preference (from memory): no subagents in claw-code. Work
  directly with `Edit`/`Write`/`Read`/`Grep`/`Bash`.
- User preference: keep technical depth + tradeoff walk-throughs in
  explanations.
- User preference: inline harness demos via `printf "..." | zenith`
  pattern, not harness-tester subagent.
- User preference: check rendering before blaming the model (display
  double-print is more common than model looping).
- User focus this session: R&D at small scale. Zenith harness
  integration explicitly deferred.

---

## Files in Project — everything the next session needs to know

### Data modules (HRM training inputs)

- `calm/hrm/data.py` — math expression generator + tokenizer (updated
  session 26 with 3-digit operand range)
- `calm/hrm/nl_data.py` — NL template generator (13 templates)
- `calm/hrm/word_data.py` — word problem generator (14 templates,
  names, pronouns, multi-step)
- `calm/hrm/gsm_data.py` — GSM-style narrative generator (10
  templates, subordinate clauses)
- `calm/hrm/multi_data.py` — pools 4 domains (math, nl, word, gsm)
- `calm/hrm/extended_data.py` — 6 extended formats (code_var,
  prefix_op, distractor, units, let_bound, eq_complete)
- `calm/hrm/extended2_data.py` — 10 more extended formats (fn_call,
  phrasal, past_narr2, alt_let, eq_var, three_op, possessive,
  verb_by, question_first, when_then)
- `calm/hrm/multi10_data.py` — pools 10 formats
- `calm/hrm/multi20_data.py` — pools 20 formats

### Training scripts

- `scripts/train_hrm_nl.py` — NL domain trainer
- `scripts/train_hrm_word.py` — word problems trainer
- `scripts/train_hrm_gsm.py` — GSM-style trainer
- `scripts/train_hrm_multi.py` — multi-task (4 pooled)
- `scripts/train_hrm_multi10.py` — multi-10 distribution probe
- `scripts/train_hrm_multi20.py` — multi-20 distribution probe
- `calm/hrm/train.py` — base math trainer (structure-only + scratchpad
  + default modes)

### Eval scripts

- `scripts/eval_hrm_math.py` — math held-out via `--verified`
- `scripts/eval_hrm_nl.py` — NL held-out + smoke
- `scripts/eval_hrm_word.py` — word problems held-out + smoke
- `scripts/eval_hrm_gsm.py` — GSM held-out + smoke
- `scripts/eval_hrm_multi.py` — 4-domain multi-task per-domain
- `scripts/eval_hrm_ood.py` — 18-case OOD for the 4-domain baseline
- `scripts/eval_hrm_multi10_ood.py` — 18-case OOD for multi10
- `scripts/eval_hrm_multi20_ood.py` — 18-case OOD for multi20 (harder,
  includes reasoning cases)

### HRM checkpoints (all 48K params, all `--structure-only`)

| Checkpoint | Domain | Full-expression / held-out |
|---|---|---:|
| `math_structure_best.pt` | Math 3-digit | 30/30, smoke 5/5 |
| `nl_math_structure_best.pt` | NL templates | 29/30, smoke 5/5 |
| `word_problem_best.pt` | Word problems | 30/30, smoke 5/5 |
| `gsm_best.pt` | GSM-style | 28/30 (first observed ceiling) |
| `multi_task_best.pt` | 4-domain pooled | **30/30 all four domains** |
| `multi10_best.pt` | 10-format pooled | 50% on multi10 OOD |
| `multi20_best.pt` | 20-format pooled | **100% on multi10 OOD**, 22% on reasoning cases |

Legacy checkpoints (don't use): `math_hrm_best.pt`,
`math_scratchpad_best.pt`, `math_seq2seq_best.pt`,
`math_structure_2digit.pt.bak`.

### LLM-Computer IR (`calm/llm_computer/`)

- `model.py` — `Small2DTransformer`, `d_head=2`, hard-max attention
- `hull_cache.py` — `HullKVCache`, 108× speedup at N=2K, parity with
  batched attention validated
- `gate_graph.py` — 7 hardware nodes: TokenEmbed, PosEmbed, LookUp,
  LookUpExact, ReGLU, LinearHead, TokenInput/Output
- `compile.py` — declarative compiler (`compile_program`)
- `schedule.py` — greedy auto-scheduler (`auto_schedule`)
- `parse.py` — `parse_expression` via Python ast
- `interpret.py` — topo-walk interpreter, `Delegate` → safe_eval
- `programs/` — 9 compiled programs:
  - Primitives: add_one (1280p), copy_past (2560p), increment_counter
    (2176p), threshold (216p) — each with `*_ir.py` counterpart
  - Composition: adder_tiny (1020p, 1-digit, 16/16), **adder
    (486,012p, 2-digit, 10,000/10,000)**
  - Memory: retrieve_by_index (1164p), retrieve_threshold (590p),
    **read_by_key (1410p, semantic KV, 96/96)**

### Feedback loops (`calm/`)

- `auto_learn.py` — AutoLearner with hit-tracking, shape-gated
  matching, prune_cold_patterns. 17 tests.
- `module_learning.py` — ModuleLearner for cognitive module issue
  tracking. 11 tests.
- `tests/test_auto_learn_loop.py` — cycle closure validation
- `tests/test_module_learning_loop.py` — parity tests
- `tests/test_auto_calm_integration.py` — end-to-end through
  `AutoCalmEngine` with mocked LLM, 3 tests
- `closed_loop_eval.py` — effectiveness measurement (90%→100% over 3
  rounds, 10× compression via generalization)
- `scripts/learning_dashboard.py` — unified operator visibility tool

### Memory + documentation

- `.claude/MEMORY/CRLM_SPEC.md` — 387-line current architecture state
- `.claude/MEMORY/RESEARCH_ROADMAP.md` — 8-layer forward progression
- `.claude/MEMORY/MEMORY.md` — index
- `.claude/MEMORY/SESSION_HANDOFF.md` — THIS FILE
- `.claude/CLAUDE.md` — updated for session-26 state
- `.claude/rules/` — architecture.md, calm.md, commercial.md,
  training.md, workflow.md (no more vdd.md, orchestration.md)

### Research reference material

- `RESEARCH/` — Percepta March-2026 papers (LLM-Computer, 2D heads,
  compile-to-weights) + HRM papers
- `TQ/` — TurboQuant, QJL, PolarQuant papers

### External state (don't commit)

- `.port_sessions/` — runtime state
- `.claude/scheduled_tasks.lock` — lock file
- `calm/.module_learning.json` — production learning state
- `/tmp/hrm_*.log` — training logs from this session

### Canonical commands (copy-paste-ready)

```bash
# Full test suite (expect 313 passed)
PYTHONPATH=. python3 -m pytest calm/ -q

# Per-domain HRM evals
PYTHONPATH=. python3 scripts/eval_hrm_math.py \
    --ckpt calm/hrm/checkpoints/math_structure_best.pt \
    --n 30 --seed 9999 --verified
PYTHONPATH=. python3 scripts/eval_hrm_nl.py --n 30 --seed 9999
PYTHONPATH=. python3 scripts/eval_hrm_word.py --n 30 --seed 9999
PYTHONPATH=. python3 scripts/eval_hrm_gsm.py --n 30 --seed 9999
PYTHONPATH=. python3 scripts/eval_hrm_multi.py --n 30 --seed 9999

# Multi10 and multi20 held-out OOD
PYTHONPATH=. python3 scripts/eval_hrm_multi10_ood.py \
    --ckpt calm/hrm/checkpoints/multi10_best.pt
PYTHONPATH=. python3 scripts/eval_hrm_multi10_ood.py \
    --ckpt calm/hrm/checkpoints/multi20_best.pt  # expect 100%
PYTHONPATH=. python3 scripts/eval_hrm_multi20_ood.py \
    --ckpt calm/hrm/checkpoints/multi20_best.pt  # expect ~22%

# Closed-loop effectiveness
PYTHONPATH=. python3 -m calm.closed_loop_eval

# Learning dashboard
PYTHONPATH=. python3 scripts/learning_dashboard.py

# 2-digit adder demo
PYTHONPATH=. python3 -m calm.llm_computer.programs.adder

# Semantic KV demo
PYTHONPATH=. python3 -m calm.llm_computer.programs.read_by_key
```

---

## Verification checklist on resume

1. `git log --oneline 91721f2..HEAD | wc -l` → expect 28
2. `PYTHONPATH=. python3 -m pytest calm/ -q` → 313 passed
3. All 7 checkpoints in §"HRM checkpoints" above exist at their paths
4. `cat .claude/MEMORY/MEMORY.md` shows CRLM_SPEC + RESEARCH_ROADMAP
   + SESSION_HANDOFF + evals/
5. Nothing pushed to origin (branch is +135 ahead of origin per
   earlier output)

If those check out, the codebase is exactly where session-26 left it
and Vector 2 phase 3 (meta-learning) is the clean starting point for
session 27.
