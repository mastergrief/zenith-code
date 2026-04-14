---
name: CRLM Architecture SPEC
description: Post-discovery snapshot of the Compute-Redistributed Language Model architecture — HRMs + LLM-Computer compiler + CALM feedback loops — as of session 26
type: project
---

# CRLM Architecture SPEC — Session 26 Snapshot

Branch: `feature/multi-agent-qwen`
Date: 2026-04-14
Commits since `91721f2` (session-25 handoff): **23 committed**, 1 more pending (multi-task HRM training in progress)
Tests: **311 passing** across `calm/tests/`, `calm/hrm/tests/`, `calm/llm_computer/tests/`

---

## A. Executive summary

The CRLM thesis — *intelligence partitioned into structure (learned) + values (compiled, exact)* — is now empirically anchored. A 48K-parameter HRM translates natural language into structured math expressions at **93-100% accuracy across four input domains** (math-echo, NL templates, word problems, GSM-style narratives). A declarative gate-graph IR compiles hand-written programs into transformer weights — nine programs ship, including a 2-digit adder that scores 10,000/10,000 exhaustive cases at 486K params. The CALM feedback loops (AutoLearner + ModuleLearner) are now tested end-to-end, proven to close, and monitored via a unified dashboard. A three-vector research framework organizes next work: Vector 1 (feedback loops) is shipped; Vector 2 (meta-structure HRM) is phase-1 in progress; Vector 3 (unify the substrate) is scoped into four multi-quarter phases.

---

## B. Architecture inventory

| Component | Location | Size | Purpose |
|---|---|---|---|
| Python agent harness | `agents/` | ~4,400 LOC / 15 files | Terminal REPL with Gemma/llama.cpp, 20 tools, session mgmt |
| CALM engine (main) | `calm/` | ~37,400 LOC / 194+ files | Auto-CALM facade + 116 backends + 39 cognitive modules |
| CALM tests | `calm/tests/` | ~275 test files | Includes new `test_auto_learn_loop.py`, `test_module_learning_loop.py`, `test_auto_calm_integration.py` |
| HRM family | `calm/hrm/` | `data.py`, `model.py`, `train.py`, `inference.py`, `nl_data.py`, `word_data.py`, `gsm_data.py`, `multi_data.py` | Learned structure extractors |
| HRM tests | `calm/hrm/tests/` | `test_nl_data.py` | 4 pipeline tests |
| HRM checkpoints | `calm/hrm/checkpoints/` | 5 production + legacy | See §C |
| LLM-Computer | `calm/llm_computer/` | `model.py`, `hull_cache.py`, `gate_graph.py`, `compile.py`, `schedule.py`, `parse.py`, `interpret.py` + 9 programs + 6 test files | Declarative IR + compiler + primitives |
| Research source | `RESEARCH/` | 7 docs (Percepta + HRM) | Paper references |
| Rust port | `rust/` | upstream claw-code, 9 crates | Separate build, not CRLM |
| Training scripts | `scripts/train_hrm_*.py`, `scripts/eval_hrm_*.py` | 4 trainers + 4 evals | Per-domain HRM training |
| Feedback dashboard | `scripts/learning_dashboard.py` | ~80 lines | Operator visibility into both loops |

Wiring points into Zenith harness:
- `calm/precompute.py` — Layer 2 NL pattern bank (550 regex patterns). AutoLearner's learned patterns inject here.
- `calm/verify.py` — Layer 1 claim extraction/correction. Feeds `AutoLearner.learn_from_correction()`.
- `calm/factual_check.py` — Layer 3 misconception detection. 48 static + 10 dynamic patterns.
- `calm/engine_v2.py` — 7-phase pipeline orchestrator. Calls `ModuleLearner` via `module_learning.record_from_report()`.

---

## C. HRM checkpoints

All 5 production checkpoints are at `calm/hrm/checkpoints/`. All use the same sweet-spot architecture: `hidden=32, num_heads=4, L=H=dec=1, 48,864 params`.

| Checkpoint | Params | Domain | Train time | Per-token val | Full-expression / smoke | Seed |
|---|---|---|---|---|---|---|
| `math_structure_best.pt` | 48,864 | Math expression echo (3-digit ops) | ~732s @ 500ep | 100% | 30/30 held-out, 5/5 smoke | 42 |
| `nl_math_structure_best.pt` | 48,864 | NL templates ("what is X plus Y?") | ~794s @ 500ep | 99.8% | 29/30 held-out, 5/5 smoke | 42 |
| `word_problem_best.pt` | 48,864 | Word problems w/ names+pronouns | ~158s @ 100ep (killed early) | 99.7% | 30/30 held-out, 5/5 smoke | 42 |
| `gsm_best.pt` | 48,864 | GSM-style multi-sentence narratives | ~603s @ 500ep | 99.6% | 28/30 held-out, 5/5 smoke — **first observed ceiling** | 42 |
| `multi_task_best.pt` | 48,864 | All four domains pooled (Vector 2 phase 1) | 1371s @ 500ep | 100% | **30/30 all four domains** — GSM ceiling broken via cross-domain exposure | 42 |

Training data generators (one per domain):
- `calm/hrm/data.py:MathDataGenerator` (operand range 1-999)
- `calm/hrm/nl_data.py:NLMathDataGenerator` (13 templates, 1-999)
- `calm/hrm/word_data.py:WordProblemGenerator` (14 templates, 1-99, 104-char max)
- `calm/hrm/gsm_data.py:GSMDataGenerator` (10 templates, 1-99, 104-char max, multi-sentence)
- `calm/hrm/multi_data.py:MultiTaskGenerator` (pools all four, balanced)

Legacy checkpoints (not in production, kept for audit trail):
- `math_hrm_best.pt` (1.6MB, pre-Round-1e 244K-param seq2seq)
- `math_scratchpad_best.pt` (Round 1c/1d artifacts)
- `math_seq2seq_best.pt` (Round 1a ceiling, 51%)
- `math_structure_2digit.pt.bak` (pre-3-digit backup from session 26 step 1)

---

## D. LLM-Computer IR surface

### Node types (`calm/llm_computer/gate_graph.py`)

**Compute nodes** (interpreter walks these):
| Node | Purpose |
|---|---|
| `Const` | Integer literal |
| `BinOp` | `add`/`sub`/`mul`/`div` |
| `Delegate` | Route to `safe_eval` backend (1002 functions) |
| `Result` | Named graph output |

**Hardware nodes** (compiler walks these):
| Node | Purpose |
|---|---|
| `TokenEmbed` | Per-token entries into `tok.weight[k, channel]` |
| `PosEmbed` | Per-position entries into `pos.weight[p, channel]` |
| `LookUp` | Copy-from-pos-0 attention head (keys zero, first-tie argmax) |
| `LookUpExact` | Parabolic-key attention (`k_j = (2j, -j²)`). Per-channel coefficients enable both position-indexed and semantic-keyed retrieval |
| `ReGLU` | One FFN neuron: `out += coef · val · ReLU(gate)`, with `gate`/`val` as linear combos of residual channels |
| `LinearHead` | Final head entries into `head.weight[token, channel]` |
| `TokenInput` / `TokenOutput` | Legacy Layer-1 shorthand (kept for `add_one`-style demos) |

### Compiled programs (`calm/llm_computer/programs/`)

| Program | Params | Primitive exercised | Test coverage |
|---|---|---|---|
| `add_one` | 1,280 | Tok embed + linear head (identity) | Bit-match vs hand-wired |
| `copy_past` | 2,560 | `LookUp` (copy-from-pos-0) | Behavioral match (head packing differs) |
| `increment_counter` | 2,176 | `PosEmbed` + upper-half head | Bit-match |
| `threshold` | 216 | `ReGLU` step function | Bit-match |
| `adder_tiny` | 1,020 | 1-digit adder, LookUp + 14 ReGLUs | 16/16 exhaustive |
| `adder` | 486,012 | 2-digit adder, same pattern scaled | **10,000/10,000 in 0.38s** |
| `retrieve_by_index` | 1,164 | `LookUpExact` w/ parabolic keys | 256/256 exhaustive |
| `retrieve_threshold` | 590 | `LookUpExact` + `ReGLU` composition in 1 layer | 256/256 exhaustive |
| `read_by_key` | 1,410 | Semantic KV store (ReGLU key-squaring + LookUpExact coefs) | 96/96 (4! perms × 4 queries) |

### Compiler machinery

- `calm/llm_computer/compile.py` — `compile_program(graph, d_model, n_heads, n_layers, d_ffn, max_len, vocab_size)`. Zero-inits weights then walks hardware nodes, populating per-node contributions. Sequential head/neuron allocators.
- `calm/llm_computer/schedule.py` — `auto_schedule(graph)`. Topological one-pass placement: channels have availability phases; each node lands at the earliest compatible phase. Verified to produce same layering as hand-picked for all existing programs.
- `calm/llm_computer/parse.py` — `parse_expression(str) → GateGraph` via Python `ast.parse`.
- `calm/llm_computer/interpret.py` — `interpret(graph)` walks compute nodes via topological order; `Delegate` routes to `safe_eval`.
- `calm/llm_computer/hull_cache.py` — `HullKVCache` with Andrew's monotone chain. **108× speedup at N=2K** tested in isolation, **identical outputs to batched hard-max attention** tested via `test_hull_cache_attention.py`. Not wired into `Small2DTransformer.forward()` yet (perf path for long sequences).

---

## E. CALM feedback loops

Two independent learning loops now have full test coverage and a unified dashboard.

### AutoLearner (`calm/auto_learn.py`)

Purpose: correction → pattern → precompute injection → next similar prompt skips the error.

- `LearnedPattern`: fields `pattern_type` (arithmetic/function), `expression`, `frequency` (learn count), `hits` (fire count)
- `learn_from_correction(claim)` — generalizes the expression (17*23 → "N * O", is_prime(391) → "is_prime(N)")
- `suggest_precomputes(prompt)` — shape-gated matching (operator or function name must appear in prompt; prevents pattern pollution)
- `prune_cold_patterns(min_hits, min_frequency)` — removes patterns that never fired and weren't seen repeatedly
- `stats()` — reports total, total_hits, cold_patterns, top_patterns
- Persistence: `calm/learned_patterns.jsonl`
- Wired into: `calm/auto_calm.py` (lines 69-70, 78, 187) and `calm/stream_auto.py`

Tests: `calm/tests/test_auto_learn_loop.py` (17 tests, all pass):
- Generalization: multiplication, 1-arg function, 2-arg function
- Frequency increments on repeat
- Correct claims skip learning
- Persistence round-trip
- **THE LOOP TEST**: correct 17*23 → next 347*289 prompt gets 100283 precomputed
- Large-number guard (skip factorial on credit-card numbers)
- Hit counter: increments on fire, not on miss, persists
- Prune cold patterns: respects high-frequency retention
- Stats reports hits + cold counts

Effectiveness harness (`calm/closed_loop_eval.py`): 3 rounds of 20 corrections, 30 held-out each:
```
  round   patterns   seeded   hits  total     rate
  ------ --------- -------- ------ ------ --------
      1          6       20     27     30   90.0%
      2          7       20     30     30  100.0%
      3          7       20     30     30  100.0%
```
**20 errors → 6-7 patterns (10× compression via generalization), hit rate 90% → 100%.**

### ModuleLearner (`calm/module_learning.py`)

Purpose: recurring cognitive-module issues → prompt-adapting preventions.

- `IssueTrend`: fields `module`, `issue_type`, `context`, `frequency`, `last_seen_turn`, `prevention`
- `record(module, issue_type, context)` — increments or creates trend
- `record_from_report(CognitiveReport, prompt)` — classifies prompt context, records per-module issues
- `suggest_prompt_additions(prompt)` — returns list of preventions for trends with `frequency >= 3` matching the prompt's detected context
- Persistence: `calm/.module_learning.json`

Tests: `calm/tests/test_module_learning_loop.py` (11 tests, all pass):
- record creates/increments trends
- Prevention populated on first record (context-specific, from internal map)
- Suggest requires 3+ occurrences
- **THE LOOP TEST**: 3 similar issues → prevention fires on matching prompt
- Context detection: comparison, debugging, explanation, design, operations, general
- Context matching: preventions don't leak between contexts
- General-context patterns fire anywhere
- Persistence round-trip
- `record_from_report` routing
- `recurring_issues` filter

### End-to-end integration (`calm/tests/test_auto_calm_integration.py`)

Mocks `_generate` inside `AutoCalmEngine`, exercises the full pipeline without live Gemma:
- `test_loop_closes_in_auto_calm_engine` — round 1 LLM emits "17*23 = 400", learner records; round 2 prompt "347 * 289" → system prompt contains "Verified facts: 347 * 289 = 100283" before the mock LLM is called
- `test_loop_shape_gate_prevents_noise` — factorial pattern doesn't leak into multiplication-prompt system prompt (phase 2's shape gate is load-bearing end-to-end)
- `test_verified_claim_does_not_learn` — correct first-time answer → zero patterns recorded (guard against spurious accumulation)

### Dashboard (`scripts/learning_dashboard.py`)

Operator command that prints the state of both loops. Current production DB state (at SPEC write):
- AutoLearner: 10 patterns, 0 hits (hit tracking is new in this session), top pattern `N - O` with frequency 5
- ModuleLearner: 53 tracked issues, 0 recurring yet (3-strike threshold not crossed in real usage); top modules `scope` (15), `precision` (14); top contexts `comparison` (20), `explanation` (16), `general` (16)

---

## F. CRLM scaling law — empirical data

The claim in `.claude/rules/architecture.md`: **HRM size scales with problem-language complexity, NOT problem-difficulty**.

| Input language | Max sentence length | HRM size | Per-token val | Structural match | Note |
|---|---:|---:|---:|---:|---|
| Math expression echo (`347 * 289`) | ~20 chars | 48K | 100% | 100% (30/30) | Trivial echo, structure-only |
| NL templates ("what is X plus Y?") | ~30 chars | 48K | 99.8% | 97% (29/30) | Short templated NL |
| Word problems (names, pronouns, multi-step) | 78 chars | 48K | 99.7% | 100% (30/30) | Anaphora across 2-3 sentences |
| GSM-style (subordinate clauses, 3-4 terms) | 104 chars | 48K | 99.6% | 93% (28/30) | **First ceiling** |

Residual GSM failures are all digit-transposition (`21 → 12`, `15 → 51`) — encoder-side bottleneck localizing numeric spans inside longer filler text. Not a computation failure; a transcription failure.

### Consistent training lesson (observed 3 times this session)

**Cosine LR scheduled to 0 at 100 epochs under-fits on any non-trivial NL domain.** The fix is always `--epochs 500` with `best_val_acc` checkpoint selection:

| Domain | 100ep result | 500ep result | Delta |
|---|---:|---:|---:|
| Math 3-digit (session 26 step 1) | 84.3% per-token → 26.7% full-expr | 100% / 100% | +73pp |
| NL templates | 99.6% per-token → 96.7% full-expr (first pass with 2-digit cap was 83%) | 99.8% / 97% | baseline |
| Word problems | 99.7% per-token → 100% full-expr | same (killed early) | — |
| GSM-style | 99.0% per-token → 83.3% full-expr | 99.6% / 93.3% | +10pp |

Rule: always `--epochs 500`, always let `best_val_acc` pick the right moment. Observed best epoch has consistently landed between 100 and 300, never later.

---

## G. Three-vector research framework

Defined in this session to organize forward work after Vector 1 became the obvious target.

### Vector 1 — close the feedback loop ✓ SHIPPED

- Phase 1 (commit `661ef74`): first tests + effectiveness harness (11 tests + closed-loop eval)
- Phase 2 (commit `c5057d0`): hit tracking + shape-gated matching (6 new tests, defect fixed)
- Phase 3 (commit `de9673a`): ModuleLearner parity tests + unified dashboard (11 tests + dashboard)
- Phase 4 (commit `b18845b`): end-to-end integration through AutoCalmEngine (3 tests)

Total: 31 new tests, 2 new scripts, 1 defect fixed (pattern pollution), 0 regressions.

### Vector 2 — meta-structure HRM (PHASE 1 SHIPPED, STRONG RESULT)

Phase 1: multi-task HRM handling all four domains in one 48K-param model. **Per-domain eval: 30/30 full-expression on every domain** — including GSM where per-domain training plateaued at 93%. Cross-domain exposure (math-echo teaches precise digit copy; that discipline transfers to GSM's operand localization) breaks the ceiling without scaling parameters.

```
  domain            multi-task   per-domain
  math-echo            100.0%        100%
  nl-template          100.0%         97%
  word-problem         100.0%        100%
  gsm-style            100.0%         93%
```

Implication: the CRLM scaling ceiling observed on GSM was domain-isolation-bound, not architecture-bound. Pooling related domains extends reach without extra parameters. Open question: was the lift from cross-domain exposure or from 2× total training data (per-domain used 2000 samples @ 500ep vs multi-task 4000 @ 500ep)? Ablation needed.

**Hypothesis for phase 2 (after phase 1 completes):** the multi-task HRM, having seen multiple structurally-related domains, will transfer to a held-out 5th domain with few-shot examples in-context. Untested.

**Phase 3 (speculative):** in-context schema induction — show the HRM 5 examples of a new problem class during inference, have it induce the target schema. Research territory; not yet scoped to an implementation.

### Vector 3 — unified substrate (SCOPED, NOT STARTED)

Target: backends + HRM + orchestrator compile into a single `Small2DTransformer` whose weights are gradient-differentiable. Four phases:

1. Compile 3 representative backends + dispatcher (2-4 weeks) — gcd, factorial, is_prime with opcode routing
2. MILP scheduler + interval-coloring slot allocator (2 weeks) — needed at ≥30 gates
3. Fuse all 116 backends into one transformer, wire into `auto_calm.py` (4+ weeks)
4. Make weights learnable via gradient; fine-tune on production feedback (3-6 months)

Full design sketch in this session's conversation history (see section "Solving the substrate — four concrete phases").

---

## H. What's deferred

With concrete next steps for each.

### `HullKVCache` in `Small2DTransformer.forward()`
- **Status**: library works (108× speedup), parity with batched attention validated. Not wired into the forward pass.
- **Why deferred**: test programs use S ≤ 5, batched linear-scan is faster. Pays off only at long sequences.
- **Next step**: add a `forward_incremental()` method when a program with S > 256 ships.

### Latest-write perturbation for `LookUpExact`
- **Status**: semantic-key `read_by_key` works only when each key is stored once per sequence.
- **Why deferred**: no current program requires repeated keys.
- **Next step**: when building a program that writes the same key multiple times (e.g. mutable KV store), add `ε · (2j, -j²)` tie-breaker so the latest write wins per RESEARCH/02 §5.

### GSM scaling experiment (h=128)
- **Status**: 48K hits 93% on GSM. `training.md` prescribes h=128 (16×) for structural-relevant gate failure.
- **Why deferred**: 93% crosses the ≥ 90% ship gate; 7% residual on a templated benchmark isn't load-bearing.
- **Next step**: run only if a downstream use case cares about the last 7%.

### HRM → CALM precompute wiring in Zenith harness
- **Status**: HRMs are trained and validated. `precompute.py` is regex-based. They don't talk yet.
- **Why it matters**: this is the highest-leverage user-facing integration — turns the HRM from a bench artifact into a production component serving real Zenith queries.
- **Next step**: in `calm/precompute.py`, call NL-HRM before falling through to regex. ~2-3 days of work. See "option 1" in this session's conversation.

### Multi-task HRM transfer experiment
- **Status**: multi-task training completes during this session. Per-domain eval reveals whether one model handles all four.
- **Next step**: if multi-task hits ≥ 93% per domain, test transfer to held-out 5th domain (physics word problems? chemistry stoichiometry?) with little/no extra training.

---

## I. Canonical commands

```bash
# HRM smoke (fastest sanity check)
PYTHONPATH=. python3 scripts/eval_hrm_math.py \
  --ckpt calm/hrm/checkpoints/math_structure_best.pt \
  --n 30 --seed 9999 --verified

# Per-domain HRM evals
PYTHONPATH=. python3 scripts/eval_hrm_nl.py --n 30 --seed 9999
PYTHONPATH=. python3 scripts/eval_hrm_word.py --n 30 --seed 9999
PYTHONPATH=. python3 scripts/eval_hrm_gsm.py --n 30 --seed 9999

# Multi-task HRM (Vector 2 phase 1; requires multi_task_best.pt)
PYTHONPATH=. python3 scripts/eval_hrm_multi.py --n 30 --seed 9999

# Full test suite (expect 311 pass)
PYTHONPATH=. python3 -m pytest calm/ -q

# LLM-Computer only (30+ tests)
PYTHONPATH=. python3 -m pytest calm/llm_computer/tests/ -v

# Feedback-loop tests
PYTHONPATH=. python3 -m pytest calm/tests/test_auto_learn_loop.py \
  calm/tests/test_module_learning_loop.py \
  calm/tests/test_auto_calm_integration.py -v

# Closed-loop effectiveness measurement
PYTHONPATH=. python3 -m calm.closed_loop_eval

# Learning dashboard
PYTHONPATH=. python3 scripts/learning_dashboard.py

# 2-digit adder demo
PYTHONPATH=. python3 -m calm.llm_computer.programs.adder

# Semantic KV demo
PYTHONPATH=. python3 -m calm.llm_computer.programs.read_by_key
```

---

## J. Key commits (chronological, `feature/multi-agent-qwen`)

Session 25 handoff → SPEC write (23 commits):

### Session 26 main plan (steps 1-4)
- `4344f6c` hrm: 3-digit operand range + 500 epochs (smoke 5/5, held-out 100%)
- `c7c56c1` llm_computer: promote LookUp + ReGLU to first-class IR + declarative compiler
- `fdc169f` llm_computer: 1-digit adder via LookUp + ReGLU composition
- `dbc5ef5` llm_computer: 2-digit adder — 10,000/10,000 via compositional IR
- `292bfb0` hrm: integration #3 — NL → math expression HRM (48K params, 29/30, smoke 5/5)

### Session 26 follow-ons (word/GSM stress tests + compiler deferred work)
- `fea22aa` hrm: word problems at 48K — 30/30 held-out, smoke 5/5
- `ae00b03` llm_computer: parabolic-key LookUpExact — exact retrieval by data-dependent index
- `15d4c9b` llm_computer: same-layer LookUpExact + ReGLU composition (retrieve_threshold)
- `d791121` llm_computer: semantic-key LookUpExact (KV-store primitive, 96/96 cases)
- `b5f27c8` llm_computer: greedy auto-scheduler — no more hand-picked layer fields
- `071329a` llm_computer: validate HullKVCache as drop-in for Small2DTransformer attention
- `a43f0d2` hrm: GSM-style word problems at 48K — 28/30, smoke 5/5 (CRLM ceiling test)

### Vector 1 — close the feedback loop
- `661ef74` calm: close the feedback loop — first tests + effectiveness harness (Vector 1)
- `c5057d0` calm: hit tracking + shape-gated pattern matching (Vector 1 phase 2)
- `de9673a` calm: ModuleLearner tests + unified learning dashboard (Vector 1 phase 3)
- `b18845b` calm: end-to-end integration test — loop proven to close through AutoCalmEngine

### Session 25 cleanup (shipped at start of session 26)
- `94c1d50` hrm: Round 1e — structure-only loss + tiny model (48K params, 145s, 96.7%)
- `f4e8602` calm: llm_computer prototype — Small2DTransformer + HullKVCache + 4 primitive programs
- `08ec8c0` docs: RESEARCH/ (Percepta LLM-Computer) + TQ/ (TurboQuant) reference papers
- `bb7f13d` chore: remove VDD/subagent infrastructure
- `a398cae` docs: session 25 — HRM sweet spot + LLM-Computer prototype + rules updates
- `e272a66` rust: route Ollama provider through OpenAI-compat client
- `bf27d1f` calm: learned_patterns frequency bump from prior Gemma testing

Vector 2 phase 1 commit (multi-task HRM) pending after training completes.

---

## K. Verification checklist

On a fresh resume, run these to validate the SPEC is accurate:

1. **Branch + commits**: `git log --oneline 91721f2..HEAD | wc -l` → expect 23 (or 24 after multi-task commit lands)
2. **Full test suite**: `PYTHONPATH=. python3 -m pytest calm/ -q` → expect 311 passed
3. **HRM smoke**: run the 4 canonical `eval_hrm_*.py --verified` — expect 5/5 smoke each, held-out per §C
4. **LLM-Computer suite**: `PYTHONPATH=. python3 -m pytest calm/llm_computer/tests/ -v` → expect 30+ passed
5. **Feedback loop**: `PYTHONPATH=. python3 -m calm.closed_loop_eval` → expect 90% → 100% over 3 rounds
6. **Dashboard**: `PYTHONPATH=. python3 scripts/learning_dashboard.py` → shows both loops with current counts
7. **Artifacts exist**: all paths in §B and §D should `ls` cleanly
8. **Checkpoints exist**: all 5 production checkpoints in §C should be present at their paths (each ~200 KB except math_hrm_best.pt at 1.6 MB legacy)

---

## L. Open questions / research directions

- Does the multi-task HRM (Vector 2 phase 1, training as of SPEC) handle all four domains at ≥ 93%? Early signal is yes (100% val_acc at epoch 200), but per-token vs structural divergence still needs held-out eval.
- Does a multi-task HRM transfer to a held-out 5th domain with few-shot prompting? This is the Vector 2 phase 2 test, not yet run.
- Can the compiled substrate (Vector 3) actually match Python backend latency at runtime? Probably yes on GPU after MILP scheduling; needs phase 1 to prove.
- What's the right interface between HRM and `calm/precompute.py` for Zenith integration? The fast path: NL-HRM in front of regex; call HRM first, fall back to regex on parse failure. 2-3 days of work, highest-leverage user-facing integration remaining.
