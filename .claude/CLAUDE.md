# Zenith Code — Multi-Agent Harness + CALM Reasoning Engine

Fork of [ultraworkers/claw-code](https://github.com/ultraworkers/claw-code) with a Python agent harness, CALM reasoning engine, HRM + LLM-Computer (the CRLM stack), and a Rust port. **Working policy: no subagents — use `Edit`/`Write`/`Read`/`Grep`/`Bash` directly.**

## Default Workflow — Hypothesis, Test, Iterate
Full spec: `.claude/rules/workflow.md`

**Core principle: it works or it doesn't, it's better or it isn't.** Every
"done", "working", "fixed", "faster" claim must be backed by a measurement
taken *after* the change. No vibes, no "looks right", no "should be fine".

- **The loop:** state hypothesis → pick measurement first → minimal edit →
  build → measure → binary decision (ship or revert) → log what you ruled
  out → next hypothesis. Target: < 5 min per round.
- **Two measurements every round:** a raw/fast path (e.g. `llama-bench`,
  unit test, pytest) AND the user-facing path (e.g. chat API, full
  harness run, real inference). Only ship when both move together —
  raw-only wins are buried under overhead, user-only wins are noise.
- **Plateau = bug, not tuning.** 3 iterations in a row < 2% each → stop
  micro-tuning, go find the one wrong line. Session-16 example: ~6
  micro-opts stuck at 24 tok/s, then one line (cache 16-entry const-mem
  LUT in registers) moved +58%.
- **One round per commit, with a before/after table in the message.**
  `git log --oneline` becomes a readable perf changelog. Always
  checkpoint before risky swings (re-quantize, struct layout, training
  run) — your rollback is `git reset --hard HEAD`.
- **Correctness check every round.** Canonical smoke test: `17×23=391`
  via the chat API. Perf gains that break correctness are reverts.
## Working policy — no subagents

Work directly with `Edit`/`Write`/`Read`/`Grep`/`Bash`. Do not dispatch
subagents or create teams. Prior VDD/orchestration infrastructure was
removed in commit `bb7f13d`; the agent definitions and `/VDD`, `/DISCOVER`,
`/EVAL`, `/TRAIN-DATA` slash-commands no longer exist. Session 26 and
Vector 1 shipped 23+ commits + 311 tests directly; this is the proven
default for the project.

If a multi-step workflow is genuinely needed, structure it as sequential
hypothesis → build → test → commit rounds per the workflow rules above.

---

## Config `.claude/` Editing Directive

When editing `.claude/` configs (agents, CLAUDE.md, commands, rules etc):
- **Preserve structure** → Match existing formatting (bullets, sections, headers)
- **Match tone** → Imperative, terse, no fluff (e.g., "Do X" not "You should consider doing X")
- **Add value** → Every word must serve purpose (examples only if essential)
- **No verbosity** → 500 lines is hard limit, 250-500 is sweet spot. Be concise without losing context.
- **Maintain style & patterns** → Use existing conventions
- **No duplication** → Don't repeat information already present elsewhere
- **Verify integration** → New content must flow naturally with surrounding text

---

## Commercial Potential

Long-term commercial direction documented in `.claude/rules/commercial.md`. Currently R&D — focus on building the best system, not shipping a product. Commercial awareness is context, not a constraint.

## Substrate vs Cards vs CHRLM — vocabulary

Lock-in convention. Use these terms precisely in all new prose; don't
conflate.

- **Substrate** = architectural standard. `Small2DTransformer` +
  `d_head=2` invariant + channel allocation protocol + gate-graph IR +
  mode tokens + D2/D3/D5 + fast weights. **The spec, not a tensor.**
  Analogy: like x86 ISA.
- **Card** = an individual `.pt` weight tensor compliant with the spec.
  Compiled (gate-graph IR, exact) or trained (SGD, statistical).
  Analogy: x86 binaries.
- **Build** = a curated set of substrate-compliant cards orchestrated
  together for a domain. Examples: CHRLM (general), CHRLM-Coding
  (future), CHRLM-Math (future).
- **CHRLM** = the current general-knowledge build. Session 30:
  **unified single tensor** — Gemma + HRMs + compiled cards + knowledge
  DB ALL in ONE `.pt`, ONE forward pass, per-sub-head attention partition.
- **PT** (Pointer Transducer) = a `CopyAugmentedTransformer` card
  trained to transduce NL → formal expression via pointer-copy. Replaces
  HRM for structure extraction. One PT per **output-language family**
  (not per domain). ~185K params, ~32 sub-heads. Copies digits from
  input; generates operators from vocabulary. Session 31: 95-100%
  across 4 domains.
- **Output-language family** = a class of expression syntax. Function-call
  (`fn(args)`), infix arithmetic (`a + b`), boolean logic (`a > b and`).
  ~3-5 families cover 30+ domains. Adding a domain within an existing
  family is a data-only operation.
- **Domain** = a facade with imports/exports + PT + compiled ops +
  knowledge facts. ~32 sub-heads per domain, 30 domains on 8 GB VRAM.

**Session 30 validated through Level 5**: compiled programs live inside
Gemma's own attention layers. Per-sub-head partition: grouped-softmax
(Gemma), single-softmax (HRM), single-hard_max (compiled). Zero cross-
talk (0.00e+00). Full spec: `.claude/rules/Substrate.md`.

**Brain + Cards model**: Gemma (language + routing) dispatches to cards
(compiled programs, HRM specialists). Cards get installed INTO Gemma's
sub-heads, not alongside. Adding a card = weight edit, not retraining.
Auto-upgrade: CALM catches errors → compile into weights → persist.

## Architecture

**Model understands, transducers structure, cards compute, engine verifies.** Intelligence comes from the system architecture, not the weights. No single component reasons — the pipeline produces reasoned answers through composition. Gemma understands NL and routes; Pointer Transducers extract formal structure; compiled cards compute exactly; CALM verifies. Adding a backend module is equivalent to training — the model gets smarter at that domain instantly, with zero GPU cost.

Four active systems coexist:
1. **Python agent harness** (`agents/`, ~4,400 LOC across 15 files) — terminal coding assistant with dual backend (Ollama + llama.cpp), 3-level permissions, thinking mode, sessions, compaction, effort control, and llama.cpp hot-swap
2. **CALM engine** (`calm/`, ~37,400 LOC across 194 files) — modular compute + knowledge facade with cognitive intelligence layer. Auto-CALM (transparent verification + precomputation, 100% benchmark) + Engine V2 (7-phase pipeline: pre-analyze → enrich → precompute → generate → verify → cognitive route → self-heal) + 118 modular backends (1013 verified functions, 569 NL patterns, 100% coverage) + 39 cognitive modules (verification, reasoning, quality, meta, planning) + 48 factual check patterns + 10 dynamic cross-check patterns against backends + adaptive thinking budget + cross-turn conversation state + module self-learning with feedback loop. Full spec: `.claude/rules/calm.md`
3. **Rust claw-code port** (`rust/`) — upstream claw-code, 9 crates, separate build system
4. **Unified Single Tensor** (`calm/llm_computer/`) — the CHRLM architecture. **ONE `.pt` contains Gemma (tq4) + trained PTs (fp32) + compiled cards (fp32) + persistent knowledge DB.** Session 30 validated Level 5: all three types coexist in ONE attention layer via per-sub-head partition. **24 compiled programs** (exact): `adder` 10K/10K, `gcd` 256/256, `dispatched_v4` 791/791 (5 ops + cross-card gating), `reasoning_engine` 512/512. **Pointer Transducers** (session 31): `CopyAugmentedTransformer` with learned copy gate + pointer attention (1,089 extra params, 0.6%). Validated across 4 domains: NL math 100%, word problems 96%, GSM 95% (broke 93% ceiling), reasoning 86-88%. **One PT per output-language family** (funcall, logic, arithmetic) — ~3-5 PTs cover 30+ domains. **Auto-upgrade**: CALM → compile corrections into weights → persist (0/8 → 11/11 in 3 sessions). **Facade/import system** (`program_builder.py`): StdLib + CompiledOp + linker. `/domain` command for guided domain addition. Full spec: `.claude/rules/Substrate.md` + `.claude/rules/architecture.md`.

Serving: Gemma 4 E4B (primary) or Qwen 3.5 4B via llama.cpp at **512K context** (`ctx_size=524288`), **32K thinking budget**. Production: tq4+tq4 KV cache on Gemma E4B (`~/models/gemma-4-E4B-it-tq4-aligned.gguf`, 5.0 GB). CALM/Auto-CALM runs on the same llama-server instance. Harness auto-computes compaction threshold as `min(per-GGUF limit, int(ctx_size * 0.89))` — Gemma compacts at **227.5K tokens** (232960). Hot-swap between bases via `agents/model_swap.py`.

## Python Agent Harness (`agents/`, ~4,400 LOC across 15 core files)

### Core Files
- `agent.py` (563) — `Agent` class with dual backend, streaming, thinking mode, tool calling loop (max 10 rounds), connection retry with backoff, system prompt builder, effort mode, output dedup, `detect_llamacpp_model()`
- `tools.py` (1622) — 20 tools: `bash`, `read_file`, `write_file`, `edit_file`, `grep`, `list_files`, `list_directory`, `Agent`, `AgentCreate`, `AgentMessage`, `AgentGet`, `AgentList`, `AgentTerminate`, `Sleep`, `WebFetch`, `WebSearch`, `AskUserQuestion`, `TodoWrite`, `TodoRead`, `MultiEdit`
- `harness.py` (720) — Terminal REPL with colored output, streaming tokens, thinking display, readline history, 17+ slash commands including `/swap`
- `model_swap.py` (407) — `LlamaServerManager` for llama-server subprocess lifecycle
- `specialist_coordinator.py` (245) — auto-selects hot-swap vs Ollama multi-model mode
- `coordinator.py` (76), `swarm.py` (55), `example.py` (43) — coordination patterns

### Production Features
- `permissions.py` (133 lines) — 3 permission modes (READ_ONLY/WORKSPACE_WRITE/FULL_ACCESS), 4-level bash classification (SAFE/WRITE/DESTRUCTIVE/BLOCKED), git subcommand awareness, write redirect detection, system path blocking
- `compact.py` (244 lines) — Auto-compaction with per-GGUF context limits (Gemma 4 E4B 200K, Qwen 3.5 4B 130K, llama.cpp fallback 65K — NIAH-validated, see `.claude/MEMORY/evals/2026-04-07_summary_needle_comparison.md`), summary compression, env var override (`ZENITH_AUTO_COMPACT_TOKENS`)
- `config.py` (61 lines) — Config loader for `.zenithrc`/`zenith.json` with explicit `ZENITH_*` env var registry; `ctx_size` default is **524288** (512K)
- `history.py` (34 lines) — Timestamped audit log for tool calls, responses, errors, commands
- `session.py` (51 lines) — Save/load agent conversations to `.zenith_sessions/` as JSON, auto-save on exit

### Harness Commands
| Command | Action |
|---------|--------|
| `/help` | Show commands |
| `/agents` | List active agents |
| `/switch <name>` | Switch to specific agent |
| `/team` | Enable coordinator mode |
| `/solo` | Single agent mode |
| `/spawn <name> <role>` | Create new agent |
| `/reset` | Clear all histories |
| `/cd <path>` | Change working directory |
| `/model <name>` | Switch model for all agents |
| `/backend [ollama\|llamacpp]` | Show/switch backend (re-detects loaded GGUF + recomputes compaction threshold) |
| `/swap [name]` | Hot-swap the loaded GGUF via `LlamaServerManager` (substring match against `~/models/*.gguf`); no arg shows current + lists available |
| `/effort [low\|medium\|max]` | Show/set reasoning effort |
| `/history` | Show session event log |
| `/save` | Save active agent's session |
| `/sessions` | List saved sessions |
| `/load <id>` | Load a saved session |
| `/resume` | Load most recent session |
| `/distill status` | Show available specialist models |
| `/distill on/off` | Toggle specialist routing |
| `/exit` | Quit |

### Running the Harness
```bash
# Preferred: auto-starts llama.cpp with default ~/models/Qwen3.5-4B.Q5_K_M.gguf at 256K
zenith

# Pick a specific GGUF at launch time (new --gguf launcher flag)
zenith --gguf ~/models/gemma-4-E4B-it-Q5_K_M.gguf

# Or set the env var once in your shell rc
ZENITH_MODEL=~/models/gemma-4-E4B-it-Q5_K_M.gguf zenith

# Hot-swap from inside an active session (no restart needed)
> /swap gemma       # substring match in ~/models/*.gguf
> /swap qwen        # swap back

# Manual: specify backend and model (bypasses bin/zenith)
PYTHONUTF8=1 PYTHONPATH=. python3 agents/harness.py --backend llamacpp
PYTHONUTF8=1 PYTHONPATH=. python3 agents/harness.py --model qwen3.5:4b --backend ollama

# Programmatic / smoke-test invocation — pipe prompts via stdin, capture to log
printf "what is 2+2?\n/exit\n" | zenith --effort max > /tmp/zenith.log 2>&1

# Override context (smaller if VRAM constrained, or for faster cold start)
ZENITH_CTX=65536 zenith

# CLI flags: --model, --backend, --ctx-size, --effort, --resume, --permission-mode, --cd
```
`bin/zenith` launcher: auto-starts llama.cpp if not running, waits for health, passes `--backend llamacpp`. Default `ZENITH_CTX=262144` (256K). Configurable via `ZENITH_MODEL`, `ZENITH_PORT`, `ZENITH_CTX`, `ZENITH_LLAMA_SERVER` env vars, plus the `--gguf PATH` CLI flag (must be first arg). The stdin pipe form works in any environment (TTY or non-TTY) because the harness uses plain `input()`; redirect output to a file to keep model token spam out of your terminal/context. `bin/zenith` does NOT `cd` into the repo root before exec'ing the harness — this keeps `.zenithrc` lookup and CLAUDE.md auto-discovery honoring the user's actual cwd.

## CALM Engine (`calm/`, ~37,400 LOC, 250 tests, 100% benchmark)

Full spec: `.claude/rules/calm.md`

**"Deterministic brain on top of a probabilistic nervous system."** The LLM reasons, modular CPU backends compute, 4-lane TMR verifies, results feed back. No fine-tuning required — add a backend, model gets smarter instantly.

### Two Modes

**Auto-CALM (default)** — model writes naturally, engine verifies transparently:
- **Layer 1**: Extract `X = Y` claims from output, verify on CPU, correct if wrong
- **Layer 2**: Pre-compute answers from the prompt, inject as verified facts
- **Layer 3**: Model diagnoses bugs in NL, engine applies template fixes, verifies via tests
- Score: **40/40 (100%)** on 40-problem benchmark with precompute

**Explicit CALM (power user)** — model emits `<calm>...</calm>` blocks:
- Engine stops at `</calm>`, executes via 4-tier parse, injects results
- Score: 85-98% (nondeterminism in whether model uses blocks)

### Modular Backend Architecture (116 backends, 1002 functions, 550 NL patterns)

Two types: **compute backends** (79 `*_ops.py`, deterministic functions) and **knowledge backends** (37 `*_kb.py`, factual lookup tables). The engine doesn't care which — same contract. **100% NL pattern coverage** — every backend exports `*_NL_PATTERNS`.

Full backend table in `.claude/rules/calm.md`. 116 backends across: math (arithmetic, sequences, trig, number theory, combinatorics, calculus), strings (metrics, phonetics, text analysis), encoding, dates/time/calendar, statistics, coordinates/geospatial, HTTP/networking/CIDR, JWT, timezones, base conversion, checksums, byte sizes, durations, geometry, probability, roman numerals, financial, ratios/fractions, physics (kinematics, electricity, waves), music theory, chemistry (molecules, functional groups), logic/sets/boolean, graph theory, matrix/linear algebra, country data (195 countries), periodic table (118 elements), physical constants, algorithm complexity (sorting, search, DP, greedy, graph, NP), well-known ports, ASCII, licenses, design patterns, error codes, regex (common patterns + reference), currencies (155 ISO 4217), measurements (SI prefixes), data structures, SQL reference, git reference, Linux ops (chmod, signals, processes), encryption/security (hashing, key sizes, password strength, OWASP), databases (ACID, CAP, normal forms, indexes), testing patterns, API patterns (REST/GraphQL/gRPC), Docker, AWS services, DevOps/SRE, cloud patterns (circuit breaker, saga, 12-factor), compilers (stages, grammars, parsing), web (HTML/CSS/browser storage), type systems, concurrency, encoding reference, formatting, validation, color theory.

**Adding a backend**: write a `*_ops.py` or `*_kb.py` file in `calm/backends/`, export a `*_FUNCTIONS` dict + `*_NL_PATTERNS` list. Auto-discovery registers both — zero other files to edit.

**Defense in depth**: Layer 2 (precompute + 550 NL patterns) injects correct answers before generation. Layer 1 (verify) catches wrong claims after generation. Layer 3 (factual check, commit `04ae45a`) catches known misconceptions via 48 static patterns + 10 dynamic cross-check patterns that verify claims against backend functions at runtime. When precompute misses a phrasing, verify is the safety net; when verify misses a factual error, factual_check catches it.

**Adjacent track**: `calm/llm_computer/` ships 9 compiled programs built declaratively via gate-graph IR + auto-scheduler — primitives, 2-digit adder (10K/10K exhaustive), and semantic-keyed KV store. `HullKVCache` validated against production attention. Future direction (Vector 3): compile all 116 CALM backends into a single dispatched `Small2DTransformer`, making the compute substrate gradient-differentiable and fine-tunable on production feedback.

**Feedback loops tested + measured (Vector 1, session 26)**: `AutoLearner` and `ModuleLearner` both have end-to-end tests proving the cycle closes. Shape-gated pattern matching fixes the pattern-pollution defect (function patterns were firing on every numeric prompt). Effectiveness harness at `calm/closed_loop_eval.py` measures 90% → 100% over 3 rounds. End-to-end integration test at `calm/tests/test_auto_calm_integration.py` mocks LLM inference and proves learned patterns inject into the system prompt on round 2. Operator visibility: `scripts/learning_dashboard.py`.

### Cognitive Intelligence Layer (39 modules, Engine V2)

**Engine V2** (`calm/engine_v2.py`) — 7-phase pipeline with self-healing:
1. **PRE-ANALYZE**: profile user expertise, detect ambiguities, decompose, assess risks
2. **ENRICH**: inject pre-analysis + learned patterns into system prompt (beginner→detailed, expert→terse)
3. **ADAPTIVE BUDGET**: trivial=2K, easy=4K, medium=8K, hard=16K, deep=32K thinking tokens
4. **PRECOMPUTE**: inject verified backend facts (1002 functions, 550 NL patterns)
5. **GENERATE**: model responds with enriched context
6. **VERIFY + COGNITIVE ROUTE**: Auto-CALM claim verification + factual cross-check + 39 cognitive modules auto-selected by router (85-180ms)
7. **SELF-HEAL**: if quality < 75% (weighted scoring, commit `4fee43a`), generate targeted correction from module feedback. Confirmed working: bad responses trigger correction loop.

**Cognitive Router** (`calm/router.py`) — auto-selects relevant modules per prompt. Simple math → 6 modules. Architecture decision → 10-25 modules. Weighted quality scoring: issue-finding modules weigh 2-3× more than silent modules (verification=3×, quality/reasoning=2×, meta=1.5×).

**39 cognitive modules** across 5 layers (all error-free, commit `2116643`):

| Layer | Modules |
|-------|---------|
| Verification | chain_verify, consistency, logic, scope, factual_check, confidence_check |
| Reasoning | decompose, causal, assumptions, analogy, temporal, counterfactual, hypothesis_gen |
| Quality | creativity, nuance, evidence, relevance, completeness, explanation, density, precision, compression, error_recovery, specificity |
| Meta-cognitive | calibration, judgment, metacognition, goal_tracking, abstraction, perspective, uncertainty, communication, prerequisites |
| Planning | prioritize, constraints, risk, disambiguation, provenance, conflict_resolution |

**Factual verification** (`calm/factual_check.py`, commit `04ae45a`): 48 static misconception patterns (databases, security, performance, architecture, git, OS) + 10 dynamic cross-check patterns that verify claims against backend functions at runtime (hash lengths, OSI layers, currency decimals, molecular weights, note frequencies, country capitals).

**Cross-turn state** (`calm/conversation.py`): consistency tracking, calibration, goal tracking, provenance accumulation persist across turns. Catches contradictions and quality trend decline.

**Module learning** (`calm/module_learning.py`, fixed commit `054d477`): learns recurring quality issues and proactively injects prevention into system prompts. Keys normalized to accumulate across variable summaries. Feedback loop confirmed working: 3 similar prompts → prevention suggestion injected into next prompt's system prompt.

**Adaptive thinking** (`calm/adaptive.py`): dynamically estimates thinking budget. Precomputed answers → 2K (8x faster). Complex design → 16K (full budget).

### Auto-Training Data Collection

Every Auto-CALM correction generates distillation-compatible JSONL:
- `MathCollector` — wrong arithmetic → correct reasoning with `<think>`
- `BoolCollector` — wrong primality/divisibility → correct reasoning
- `CodeCollector` — bug diagnosis + fix → coding examples
- Output: `.calm_training/auto/{math,bool,code}.jsonl`

### Running CALM
```bash
# Auto-CALM (default, transparent, 100% benchmark)
python3 -m calm.auto_calm "What is 347 * 289? Is it prime?"

# Explicit CALM (power user, <calm> blocks)
python3 -m calm.engine "What is 17 * 23?"

# Intent-to-edit (fix bugs from NL description)
python3 -c "from calm.auto_calm import IntentToEdit; IntentToEdit().fix('app.py', 'test_app.py', verbose=True)"

# Run all 250 tests
python3 -m pytest calm/tests/ -v
```

## Pointer Transducers + LLM-Computer (`calm/hrm/` + `calm/llm_computer/`)

The CRLM split: **Pointer Transducers** (learned, ~185K params) handle NL → expression structure extraction via copy-augmented attention; **LLM-Computer** (analytically compiled) handles value computation. Full architecture spec: `.claude/rules/architecture.md`. Training rules: `.claude/rules/training.md`.

### Pointer Transducer (session 31, replaces HRM)

- **Architecture**: `CopyAugmentedTransformer` — decoder-only `Small2DTransformer` + learned copy gate + pointer attention. Subclasses base model; copy mechanism is additive (1,089 params, 0.6%). At each decode step, model chooses: generate from vocabulary OR copy from input position. Digits → copy, operators → generate.
- **File**: `calm/llm_computer/copy_augmented.py`
- **Forward returns log-probs** (not logits) — use `F.nll_loss`, not `F.cross_entropy`
- **Training**: scheduled sampling (tf_ratio 1.0→0.3), autoreg eval as gate metric, `--epochs 500`
- **Data**: all generators use `_sample_operand()` for balanced digit-length coverage (33/33/33 across [1-9]/[10-99]/[100+])

**Cross-domain PT results (session 31, all ~185K params):**

| Checkpoint | Domain | Val autoreg | Held-out | Max input |
|---|---|---:|---:|---:|
| `copy_augmented_hrm_best.pt` | NL math (13 templates) | 100% | 200/200 | 30 chars |
| `copy_word_best.pt` | Word problems (14 templates) | 98% | 96/100 | 78 chars |
| `copy_gsm_best.pt` | GSM-style (10 templates) | 100% | 95/100 | 104 chars |
| `copy_funcall_best.pt` | Funcall reasoning (percentage, ratio, etc.) | 86% | 171/200 | 88 chars |
| `copy_logic_best.pt` | Logic reasoning (compare, conditional, syllogism) | 86% | 88/100 | 121 chars |

**Output-language family principle**: one PT per output syntax family, not per domain. ~3-5 PTs cover 30+ domains. Adding a domain within a family = write templates + retrain (data-only).

**Old ceilings broken**: single-digit 0%→100% (balanced data), 3-digit 68%→100% (copy mechanism), GSM 93%→95% (copy), syllogism 36%→92% (family split).

**Remaining ceiling**: 3+ operand copy accuracy (68-83%). Known fix: two-stage decode via D5 recurrence (emit skeleton → fill slots independently).

### Legacy HRM (`HRMSeq2Seq`, session 26)

Still exists at `calm/hrm/model.py`. Encoder-decoder with nested L/H recurrence, 48K params, `--structure-only` mode. 5 checkpoints (`math/nl/word/gsm/multi_task_best.pt`). Superseded by PT for all new work, but still functional for eval comparison.

### LLM-Computer (`calm/llm_computer/`)

- **`Small2DTransformer`** — vanilla PyTorch, `d_head=2`, optional `use_hard_max=True`.
- **`CopyAugmentedTransformer`** — subclass with copy gate + pointer attention. The PT architecture.
- **`HullKVCache`** — online 2D convex hull. 108× speedup vs linear scan at N=2K.
- **Declarative IR + compiler** (`gate_graph.py` + `compile.py` + `schedule.py`): `TokenEmbed`, `PosEmbed`, `LookUp`, `LookUpExact`, `ReGLU`, `LinearHead`. Auto-scheduler assigns `(layer, phase)`.
- **Grammar-constrained decoding** (`grammar_decode.py`): inference-time mask for valid expressions + EOS boosting. Safe (0 regressions) but null on current models.
- **24 compiled programs** in `programs/`: `adder` (10K/10K), `gcd` (256/256), `dispatched_v4` (791/791), `reasoning_engine` (512/512), etc.
- **Parser + interpreter** (`parse.py`, `interpret.py`): `parse_expression()` via `ast.parse`; `interpret()` walks compute nodes; `Delegate` routes through `safe_eval` (1002-function registry).

### CRLM Pipeline (session 31)

```
NL input → Gemma (understands, routes) → PT (copies digits, generates structure)
                                            ↓
                                     expression string
                                            ↓
                                    safe_eval (1002 functions)
                                            ↓
                                    CALM verify (CPU cross-check)
                                            ↓
                                    verified answer
```

Add a domain: `/domain` command walks through scope → CALM backend → compiled card → templates → train PT → evaluate → install.

### HRM training journey (sessions 24, 25, 26)

| Round | Config | Params | Train time | Per-token | Full-expr |
|---|---|---|---|---|---|
| 1a | enc-dec + digit-reversal, h=64 | 244K | 8min | 51% | ~15-25% |
| 1c | scratchpad + `<call>` delegation, h=64 | 245K | 15min | 94% | 43% |
| 1d | 1c + place-value decomp | 245K | 16min | 94% | 37% |
| **1e** | **structure-only, h=32** (math 2-digit) | **48K** | **145s** | **99.7%** | **96.7%** |
| S26.1 | **3-digit operands + `--epochs 500`** | **48K** | **732s** | **100%** | **100% / 30** |
| S26.NL | NL templates, max_enc=48 | 48K | 794s | 99.8% | 29/30 |
| S26.WORD | Word problems (names, pronouns), max_enc=80 | 48K | 158s (killed early at 100 epochs; structural = 100%) | 99.7% | 30/30 |
| S26.GSM | GSM-style narratives, max_enc=128 | 48K | 603s | 99.6% | 28/30 — **first ceiling** |
| S26.MULTI | All four pooled (Vector 2 phase 1) | 48K | ~1000s | 100% | per-domain TBD |

Lesson: scratchpad with intermediate values forces memorization that small models can't deliver. **Stop asking the model to compute; let it emit structure and route values to the substrate.** The same 48K architecture carries across four input languages at 93-100% — validation of the CRLM scaling claim that HRM size scales with input-language complexity, not problem difficulty.

## Distillation Pipeline (`agents/distill/`, 10 Python files)

### Current Status
- **0.8B reasoning base**: trained (3 epochs, loss 1.106), eval: format learned but substance wrong — model too small
- **4B reasoning base (Qwen)**: trained on Colab A100 (1,339 examples after 2026-04-07 React/security expansion, 3 epochs), exported to GGUF Q5_K_M, serving via llama.cpp. Earlier eval: 3/5 PASS with thinking enabled (race condition, OOMKilled, architecture pass; React and security partial). **Subsequent 5-prompt A/B vs stock Gemma 4 E4B (2026-04-07) scored fine-tuned Qwen 0/5 — the React and security failures were correctness bugs (hallucinated Node.js `beforeOOM` API, broken Postgres `FOR UPDATE SKIP LOCKED` queue, regex-on-hostname SSRF check). See `.claude/MEMORY/evals/2026-04-07_qwen4b_vs_gemma4_e4b.md`.**
- **Gemma 4 E4B (stock)**: validated as alternative base 2026-04-07. Beats fine-tuned Qwen 5/0 on the same coding eval without any fine-tuning. NIAH effective context: 200K (vs Qwen's 130K). Multimodal (vision projector available). GGUF on disk at `~/models/gemma-4-E4B-it-Q5_K_M.gguf` (5.48 GB). Not yet fine-tuned on the distillation dataset.
- **Hot-swap infrastructure**: IMPLEMENTED (`agents/model_swap.py` + `SpecialistCoordinator` hot-swap mode). `LlamaServerManager` handles kill+restart swap cycles (~5–15s depending on model size). Swap cost on a warm page cache is mostly PCIe transfer time.
- **Specialists**: not yet trained — both base models are ready, specialist GGUFs don't exist on disk yet. When they do, `SpecialistCoordinator` auto-detects and switches to hot-swap mode.

### Pipeline Scripts
- `config.py` — Domains, model names, QLoRA params, paths
- `generate.py` — Teacher (9B) generates JSONL training data. CLI: `python -m agents.distill.generate --domain python`
- `train_base.py` — Stage 1: 0.8B reasoning base (local). Qwen 3.5 0.8B + QLoRA + `train_on_responses_only`, 3 epochs
- `train_4b_cloud.py` — Stage 1: 4B reasoning base (cloud). Qwen 3.5 4B + QLoRA, requires 40GB+ VRAM (A100)
- `train_4b_colab.ipynb` — Colab notebook for 4B training
- `train.py` — Stage 2: domain specialist training. Auto-uses reasoning base if available
- `export.py` — Convert merged model → GGUF → Ollama Modelfile → `ollama create`
- `validate.py` — A/B compare specialist vs base using 9B as judge
- `fetch_datasets.py` — Download Claude reasoning datasets from HuggingFace (nohurry, TeichAI, Crownelius)
- `filter_reasoning.py` — Tiered keyword filtering + dedup + merge hand-written with HuggingFace data

### Specialist Domains
| Domain | Ollama Name | Focus |
|--------|-------------|-------|
| orchestrator | specialist-orchestrator | Task routing/classification |
| typescript | specialist-ts | React, Node, TS, Next.js |
| python | specialist-py | FastAPI, Django, pytest |
| rust | specialist-rust | Ownership, tokio, serde |
| devops | specialist-devops | Docker, K8s, Terraform |
| reviewer | specialist-reviewer | Security, bugs, perf |

### Training Data (`agents/distill/data/`, gitignored except hand-written files)
- `claude_reasoning.jsonl` — 1,339 merged examples (832 filtered HuggingFace + 507 hand-written)
- `coding_reasoning_claude.jsonl` — 547 hand-written coding reasoning examples (committed). Includes +19 added 2026-04-07 (React + security, targeting Qwen eval gaps) and +21 added 2026-04-08
- `claude_reasoning_filtered.jsonl` — 832 filtered HuggingFace examples (intermediate)
- `claude_reasoning_prefilter.jsonl` — backup of pre-filter merged data
- `orchestrator.jsonl` — 252 routing examples (130 original + 121 Claude-authored)
- `orchestrator_claude.jsonl` — 121 Claude-authored routing examples (committed)
- `python.jsonl` — 25 examples (9B-generated)
- `typescript.jsonl` — 39 examples (9B-generated)
- `rust.jsonl` — 53 examples (9B-generated)

### Training Commands
```bash
# Stage 1: 0.8B reasoning base (local, 8GB VRAM)
PYTHONPATH=. python3 -m agents.distill.train_base

# Stage 1: 4B reasoning base (cloud, 40GB+ VRAM)
# Use train_4b_colab.ipynb on Google Colab with A100
# Or: python3 train_4b_cloud.py on RunPod/Lambda

# Stage 2: Specialist (on top of reasoning base)
PYTHONPATH=. python3 -m agents.distill.train --domain orchestrator

# Filter + merge reasoning data
PYTHONPATH=. python3 -m agents.distill.filter_reasoning        # filter only
PYTHONPATH=. python3 -m agents.distill.filter_reasoning --merge # filter + merge hand-written
```

## Training Philosophy

**Data quality > data quantity > model size > training tricks.**

1. Write high-quality examples — one good Claude-authored example teaches more than ten 9B-generated ones
2. Train on responses only — don't waste gradients learning to predict prompts
3. Match domain to task — coding reasoning data for coding models, routing data for routing models
4. Filter aggressively — removing bad data improves results more than adding mediocre data
5. 3 epochs on curated data — diverse enough (1,320 unique topics) to avoid memorization; 1 epoch underfits

## Serving Architecture

**llama.cpp (primary)** — Gemma 4 E4B via full GPU:
- **Production GGUF**: `~/models/gemma-4-E4B-it-tq4-aligned.gguf` (5.0 GB, TurboQuant tq4, 132-byte block alignment from session 16). **This is what CALM runs on.**
- **Alternative GGUFs**: `~/models/gemma-4-E4B-it-Q5_K_M.gguf` (5.48 GB, stock Q5), `~/models/Qwen3.5-4B.Q5_K_M.gguf` (2.9 GB, fine-tuned Q5)
- **TurboQuant tq4 KV cache**: `--cache-type-k tq4_k256 --cache-type-v tq4_k256`. 4.125 bpw, 16-level Lloyd-Max codebook, Pi rotation (seed=42, 256×256 orthogonal). 132-byte blocks (128 qs + 2 d + 2 pad for 4-byte aligned uint32 loads). **Old 130-byte GGUFs are incompatible — re-quantize.**
- Context: **512K** with tq4 KV (~5.0 GB weights + ~2.0 GB KV = ~7 GB VRAM). 32K thinking budget. Auto-CALM + harness share the same server.
- `--parallel 1` required — without it, llama-server splits `ctx_size` across 4 default slots
- Launch: `llama-server -m ~/models/gemma-4-E4B-it-tq4-aligned.gguf --ctx-size 524288 --parallel 1 --cache-type-k tq4_k256 --cache-type-v tq4_k256 -ngl 999 --port 8080`
- **~45-48 tok/s** on Gemma 4 E4B tq4 at 8K context (CALM benchmark runs)
- Hot-swap: `agents/model_swap.py:LlamaServerManager`. `/swap gemma` / `/swap qwen` in harness.

**Ollama (fallback)** — stock models, quick testing:
- Pulled models (verified via `curl -s localhost:11434/api/tags`): `qwen3.5:4b`, `qwen3.5:9b`, `qwen3:0.6b`, `qwen3:4b`, `qwen3:8b`, plus custom Modelfiles `qwen4b-fast:latest`, `qwen9b-fast:latest`, `reasoning-base:latest`
- Custom Modelfiles in `models/`: `Modelfile.qwen9b-fast` (2048 ctx), `Modelfile.qwen4b-fast` (8192 ctx), `Modelfile.reasoning-base` (32K ctx)
- Kill Windows Ollama to free VRAM: `taskkill /IM ollama.exe /F`

## Local Tools

- **llama.cpp**: built at `~/llama.cpp/build/bin/` with CUDA support (RTX 4070). **Branch `zenith` at `a6218df`** with 3 custom commits:
  - `7aae919` — `GGML_CUDA_OP_TIMING`: per-op/per-shape cudaEvent timing diagnostic. Enable: `cmake -DGGML_CUDA_OP_TIMING=ON`, `GGML_CUDA_DISABLE_GRAPHS=1 GGML_CUDA_OP_TIMING=1`
  - `29782ec` — Gemma gate+up ordering fix: upstream fusion check at `should_fuse_mul_mat` rejected Gemma's reversed ordering. **GLU fusion was never firing on any Gemma quant type upstream.** Worth upstreaming.
  - `a6218df` — fused gate+up+GLU tq4 kernel: `k_mmvq_tq4_k256_fused_preload_glu` in `mmvq-tq4.cu`. +0.68% avg (structural win, ships one Pi@x precompute + eliminates GLU kernel launch).
  - `llama-quantize`, `llama-server` — standard tools
  - **Local patch** at `tools/server/server-context.cpp:763-766` — comments out `n_ctx_slot = n_ctx_train` for >128K context on Gemma. Re-apply after `git pull`.
  - **5 mmvq-tq4 rounds reverted** (SHFL LUT, NB template, 4-row/block, PiX memoization, 2-way accumulator). Kernel is at a deep local optimum. See `SESSION_HANDOFF.md` ruled-out log.
- **Unsloth**: 2026.4.2 + PyTorch 2.10.0+cu128 (for local 0.8B training)
- **Serena MCP**: installed at `/home/gabe/serena-fork`, configured for this project

## Cloud Accounts

- **RunPod**: API key in `.env.local`, `runpodctl` installed, MCP server configured in `~/.claude.json`
- **Google Colab Pro**: 100 compute units, Colab MCP configured in `~/.claude.json`

## Hardware

- Laptop: Acer Nitro AN17-42
- GPU: NVIDIA RTX 4070 Laptop GPU (8 GB VRAM) — no Thunderbolt/eGPU support
- iGPU: AMD Radeon 780M (display only, no CUDA)
- RAM: 32 GB DDR5 5600MHz
- WSL2: Ubuntu 24.04 with GPU access
- Training (local): Unsloth 2026.4.2 + PyTorch 2.10.0+cu128
- Training (cloud): Google Colab Pro A100 (40GB)

## Key Constraints

- **8 GB VRAM**: both 4B-class bases fit at **256K context** with Q4 KV cache (llama.cpp) — Qwen 3.5 4B Q5 uses ~7.3 GB, Gemma 4 E4B Q5 uses ~6.7 GB (sliding-window attention makes Gemma's KV cache dramatically smaller at long context). 9B Q4 fits at 2K (Ollama). 0.8B FP16 fits at 32K.
- **Both 4B bases are trained at 256K context** (Qwen 3.5 4B and Gemma 4 E4B, verified via GGUF metadata `n_ctx_train=262144`). Earlier notes had Qwen at 32K — that was wrong.
- **Qwen 3.5 4B QLoRA**: 248K vocab CE loss OOMs on anything under 40GB VRAM. Must use cloud GPU (Colab A100).
- **Qwen 3.5 0.8B QLoRA**: fits locally at batch=1, seq_len=1024, packing=false
- **WSL2 + Windows Ollama**: both can run Ollama. Don't run both simultaneously — VRAM conflict. Prefer WSL-native.
- **No eGPU**: laptop has USB-C 3.2 but no Thunderbolt/USB4. Cloud GPUs for 4B+ training.

## Needle-in-Haystack Validation (2026-04-07)

Effective context for both base models was measured via single-needle, multi-needle, and distractor NIAH tests at 4K–220K haystack sizes. Full reports in `.claude/MEMORY/evals/2026-04-07_*_needle_256k_*.md`, summary in `2026-04-07_summary_needle_comparison.md`.

| Test type | Gemma 4 E4B | Qwen 3.5 4B |
|---|---:|---:|
| Single-needle (21 prompts, 4K–220K) | **21/21** | **21/21** |
| Multi-needle (7 prompts, 5 needles each) | **6/7** (fails 220K 4/5) | **5/7** (fails 180K 4/5, 220K 3/5 + hallucination) |
| Distractor (7 prompts, 4 decoys each) | **7/7** | **5/7** (U-shape dip at 64K/100K — picks wrong decoy) |
| **Total** | **34/35** | **31/35** |

Key findings:
- Both models handle single-needle cleanly through 220K (neither degrades on the easy test).
- Gemma's sliding-window attention protects against the "lost in the middle" failure mode Qwen exhibits at 64K/100K on the distractor test.
- Qwen's worst failure is **hallucination under pressure** (returns `14223-AZURE-MARTEN` when expected is `14223-CRIMSON-EAGLE` — correct number, invented suffix). Gemma's failures are silent omissions, which is a safer mode.
- `agents/compact.py:MODEL_CONTEXT_LIMITS` is the authoritative source of NIAH-validated values (Gemma 200K = 10% safety below the first failure point at 220K; Qwen 130K = safely below the first failure point at 180K).
- **The 256K tests required a local llama.cpp patch** to remove the per-slot training-context cap — see Local Tools above.

## Branch

`feature/multi-agent-qwen` on `mastergrief/zenith-code` (forked from `ultraworkers/claw-code`; renamed from `mastergrief/claw-code` 2026-04-07)

**IMPORTANT**: Assume nothing. Hypothesis, Build, Test & Iterate.