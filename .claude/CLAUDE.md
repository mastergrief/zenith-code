# Zenith Code — Multi-Agent Harness + CALM Reasoning Engine

**IMPORTANT**: Assume nothing. Hypothesis, Build, Test, Commit & Iterate. First Principles thinking. Do not discount anything until it's built and tested!

Fork of [ultraworkers/claw-code](https://github.com/ultraworkers/claw-code) with a Python agent harness, CALM reasoning engine, HRM + LLM-Computer (the CRLM stack), and a Rust port.

**Working policy: solo lead by default; triaged subagent use.** Work directly with Edit/Write/Read/Grep/Bash for fast-iteration R-round hypothesis-test loops, edits within the current session's working memory, and tasks under ~10 file changes. That's most of what we do here — the brief-writing + cold-start + round-trip overhead exceeds the work (R52.1 receipt: ~400 LOC delegated cost ~2000-word brief + 30min cold-read + 1hr/iteration vs 10min solo, plus missed 500× perf regressions from missing baselines).

**Spawn subagents when**:
- **(a) Semantic exploration across an unfamiliar subsystem** — when a question is semantic not literal (e.g. "find every tier-2 install pattern across `calm/llm_computer/facades/`"). Explore agent with `thoroughness: "very thorough"` parallelizes searches that would otherwise run sequentially.
- **(b) Independent second-opinion review on high-blast-radius changes** — code-reviewer or security-review agent AFTER a risky commit, BEFORE push. Fresh context catches what I rationalized. Candidates: Triton autograd / gradient-math commits (R52.1c-style cascade bugs), production-serve integrations, security-adjacent code.
- **(c) Large-session `/update` or `/handoff`** — when session scope exceeds ~10 commits OR ~8 touched doc files OR transcript > 50K tokens. The 3-agent split (transcript + code + docs) pays for itself; context stays clean.
- **(d) Context protection on high-volume searches** — when a grep would flood main context with >1000 expected matches. Agent scans in its own context, returns a summary.

**Never**: spawn agents "just in case", for work that fits in one direct tool call, or as a default orchestration pattern. User's explicit ask for teams/parallel workers overrides this — if asked, spawn.

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
- **Commit completed work before starting the next round.** One round
  per commit with a before/after table. Never leave measured,
  shippable work uncommitted while moving on — a crash, `git stash`,
  or `reset --hard` loses hours. `git log --oneline` becomes the perf
  changelog. Checkpoint before risky swings (re-quantize, struct
  layout, training run) — rollback is `git reset --hard HEAD`.
- **Correctness check every round.** Canonical smoke test: `17×23=391`
  via the chat API. Perf gains that break correctness are reverts.
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

**Session 30 validated through Level 5** (substrate-native demo):
compiled programs live inside Gemma's own attention layers. Per-sub-
head partition: grouped-softmax (Gemma), single-softmax (HRM),
single-hard_max (compiled). Zero cross-talk (0.00e+00).
**Session 32 ported the same pattern to prod Gemma 4 E4B**
(`GemmaSubstrate.install_card_in_attention` + `attention_partition` +
`convert_layer_to_fp32`) — three modes verified coexisting in one
real Gemma layer with distinct non-zero diffs. Full spec:
`.claude/rules/Substrate.md`.

**Brain + Cards model**: Gemma (language + routing) dispatches to cards
(compiled programs, HRM specialists, PTs). Two install paths on prod
Gemma: in-tensor (`install_card_in_attention`, weights live in
`attn_q/k/v/output`) and residual-additive (`CardSlot.attach(preserve=True)`,
card runs as separate Module — required for PTs). Adding a card =
weight edit, not retraining. Auto-upgrade: CALM catches errors →
compile into recall card → install via CardSlot → persist as JSON.

## Architecture

**Model understands, transducers structure, cards compute, engine verifies.** Intelligence comes from the system architecture, not the weights. No single component reasons — the pipeline produces reasoned answers through composition. Gemma understands NL and routes; Pointer Transducers extract formal structure; compiled cards compute exactly; CALM verifies. Adding a backend module is equivalent to training — the model gets smarter at that domain instantly, with zero GPU cost.

Four active systems coexist:
1. **Python agent harness** (`agents/`, ~4,423 LOC across 15 files) — terminal coding assistant with dual backend (Ollama + llama.cpp), 3-level permissions, thinking mode, sessions, compaction, effort control, and llama.cpp hot-swap
2. **CALM engine** (`calm/`, ~83,600 LOC across 413 .py files) — modular compute + knowledge facade with cognitive intelligence layer. Auto-CALM (transparent verification + precomputation, 100% benchmark) + Engine V2 (7-phase pipeline: pre-analyze → enrich → precompute → generate → verify → cognitive route → self-heal) + 120 modular backends (1002 verified functions, 550 NL patterns, 100% coverage) + 39 cognitive modules (verification, reasoning, quality, meta, planning) + 48 factual check patterns + 10 dynamic cross-check patterns against backends + adaptive thinking budget + cross-turn conversation state + module self-learning with feedback loop. Full spec: `.claude/rules/calm.md`
3. **Rust claw-code port** (`rust/`) — upstream claw-code, 9 crates, separate build system
4. **Unified Single Tensor** (`calm/llm_computer/`) — the CHRLM architecture. **ONE `.pt` contains Gemma (tq4) + trained PTs + compiled cards + persistent knowledge DB.** Session 30 validated Level 5 on the substrate-native demo (`HybridGroupedSmall2DTransformer`): three attention modes coexist in one layer via per-sub-head partition. **Session 32 ported the full pattern to prod Gemma 4 E4B** (`gemma_substrate.py`): coherent output at **42 tok/s decode** (160× over baseline, 90% of llama.cpp on the same GGUF) via Triton fused dequant kernels (`tq4_triton.py`, v2 default as of R53.29 — shared-mem LUT via `tl.gather`, -7% aggregate) + CUDA Graph capture + real tq4 KV storage (`KVCacheTq4`, ~3.6× memory, multi-token prefill + `trim_swa_storage` byte-copy shipped in R53.28). Fused flash-attention decode kernel (`tq4_flash_attn.py`, R53.34) wires tq4 K/V into a single-pass kernel (K-side reuses `tq4_matvec_triton`; V-side `_tq4_weighted_v_kernel` with grid=(n_heads_q,)). SWA layers fused; global layers (d_head=512) fall back to memoized dequant. Cards install two ways on prod Gemma: residual-additive (`CardSlot.attach(preserve=True)` for PTs) and in-tensor (`install_card_in_attention` + `convert_layer_to_fp32`, with per-sub-head dispatch via `attention_partition` for `mode='hard_max'|'softmax'|'grouped'`). Verification feedback closes the loop (`VerificationHook` biases Gemma logits with the card's argmax). Learning loop end-to-end: `KnowledgeStore` corrections compile into a recall card via `build_recall_model()`, install via `CardSlot`, persist as JSON — demo at `scripts/gemma_learning_loop_demo.py` (5/5 wrong → 5/5 correct). 29 compiled programs in `programs/` (`adder` 10K/10K, `multiplier` 3390/3390 on a·b<1000 — first compiled card to fix real Gemma arithmetic errors via step-through digit bias, `gcd` 256/256, `dispatched_v4` 791/791, `reasoning_engine` 512/512). **Pointer Transducers** (session 31, `CopyAugmentedTransformer`): one PT per output-language family (~3-5 PTs cover 30+ domains); cross-domain val acc 86-100%; checkpoints in `calm/hrm/checkpoints/copy_*_best.pt`. Domain registry: `.claude/MEMORY/substrate_registry.md`. `/domain` command for guided domain addition. Full spec: `.claude/rules/Substrate.md` + `.claude/rules/architecture.md`.

Serving: Gemma 4 E4B (primary) or Qwen 3.5 4B via llama.cpp at **512K context** (`ctx_size=524288`), **48K thinking budget** (`EFFORT["max"]["max_tokens"]=49152`). Production: tq4+tq4 KV cache on Gemma E4B (`~/models/gemma-4-E4B-it-tq4-aligned.gguf`, 5.0 GB). CALM/Auto-CALM runs on the same llama-server instance. Harness auto-computes compaction threshold as `min(per-GGUF limit, int(ctx_size * 0.89))` — Gemma compacts at **227.5K tokens** (232960). Hot-swap between bases via `agents/model_swap.py`.

**Tracing track (session 33-34, Rounds 13-52, 40-round arc)**: full mechanistic-interpretability arc shipped on prod Gemma. 7 capabilities mapped at sweep + per-head resolution (arithmetic, factual recall, induction, counting, comparison, SV agreement, multi-step composition). 3 causal validations: R28 (L30 H4/H6 on arithmetic, mean |Δ|=0.407, 9/10), R42 (L23 H1/H4 on SV agreement, |Δ|=0.467, 8/10), R43 (L23 on comparison + counting, 18/18 + 6/6). **Hub-sharing empirically proven**: L23 H1/H4 is a shared content-carrier head serving arithmetic + SV + comparison + counting + multi-step (via R46.2 `MultiStepReasoningFacade`, 17/17 real Gemma fixes). **5-for-1 compilation ROI validated.** Circuit typology: concentrated / cooperative / diffuse / hybrid-pipeline / deep-diffuse. **Session 34 (R51+R52) — tier-3 L24 distillation triple-null** (SAE features R50.5, MSE residuals R51.5, KL logits R52.3 — all fail identically: distillation-space loss improves but token preservation fails). **Reframing**: tier-2 stacking achieves tier-3-equivalent outcomes — every shipped augmentation is additive (VerificationHook, CardSlot+preserve, step-through bias). Tier-3 from-scratch distillation of deep-diffuse Gemma layers is closed as a pattern. See `.claude/rules/augmentation_thesis.md` §"Tier-2 stacking achieves tier-3-equivalent outcomes" for the reframing; `.claude/rules/tracing_roadmap.md` ruled-out log for per-round details. Full atlas: `.claude/MEMORY/atlas.md`. Rules: `.claude/rules/augmentation_thesis.md`, `.claude/rules/tracing_intelligence.md`, `.claude/rules/tracing_roadmap.md`.

## Python Agent Harness (`agents/`, ~4,423 LOC across 15 files)

Terminal coding assistant with dual backend (Ollama + llama.cpp), 20 tools (`tools.py`), 3-level permissions (`permissions.py`), auto-compaction with per-GGUF limits (`compact.py`: Gemma 4 E4B 200K, Qwen 3.5 4B 130K), sessions (`session.py`), history log (`history.py`), thinking mode, effort control (low/medium/max), llama.cpp hot-swap (`model_swap.py:LlamaServerManager`), `SpecialistCoordinator` auto-selecting hot-swap vs Ollama multi-model mode. Core class is `Agent` in `agent.py` with streaming + tool-calling loop (max 32 rounds, `max_tool_rounds` in `agent.py`). `ctx_size` default **524288** (512K). Config via `.zenithrc`/`zenith.json` + `ZENITH_*` env vars (`config.py`). Full file-by-file breakdown: `.claude/rules/architecture.md` §"Agent System" + §"File Organization".

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
# Preferred: auto-starts llama.cpp with default ~/models/gemma-4-E4B-it-tq4-aligned.gguf at 512K
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
`bin/zenith` launcher: auto-starts llama.cpp if not running, waits for health, passes `--backend llamacpp`. Default `ZENITH_CTX=524288` (512K). Configurable via `ZENITH_MODEL`, `ZENITH_PORT`, `ZENITH_CTX`, `ZENITH_LLAMA_SERVER` env vars, plus the `--gguf PATH` CLI flag (must be first arg). The stdin pipe form works in any environment (TTY or non-TTY) because the harness uses plain `input()`; redirect output to a file to keep model token spam out of your terminal/context. `bin/zenith` does NOT `cd` into the repo root before exec'ing the harness — this keeps `.zenithrc` lookup and CLAUDE.md auto-discovery honoring the user's actual cwd.

## CALM Engine (`calm/`, ~83,600 LOC, 72 test files / 565 test functions, 100% benchmark)

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

### Modular Backend Architecture (120 backends, 1002 functions, 550 NL patterns)

Two types: **compute backends** (81 `*_ops.py`) + **knowledge backends** (39 `*_kb.py`). Same contract: export `*_FUNCTIONS` dict + `*_NL_PATTERNS` list. Auto-discovery registers both — zero other files to edit.

**Defense in depth**: Layer 2 precompute (550 NL patterns) injects correct answers before generation. Layer 1 verify catches wrong claims after. Layer 3 factual_check catches known misconceptions (48 static + 10 dynamic cross-check patterns).

**Feedback loops** (Vector 1, session 26): `AutoLearner` + `ModuleLearner` end-to-end tested, 90% → 100% hit rate over 3 rounds on `calm/closed_loop_eval.py`. Operator visibility: `scripts/learning_dashboard.py`.

Full backend table + domain list + adding-a-backend walkthrough: `.claude/rules/calm.md`. `calm/llm_computer/` ships compiled programs (gate-graph IR + auto-scheduler) as an adjacent track.

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

# Run all 565 tests across 72 files
python3 -m pytest calm/ -v
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
- **29 compiled programs** in `programs/`: `adder` (10K/10K), `multiplier` (3390/3390, a·b < 1000, fixes Gemma's real arithmetic errors via step-through digit bias — see Round 11 in `tracing_roadmap.md`), `gcd` (256/256), `dispatched_v4` (791/791), `reasoning_engine` (512/512), etc.
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

## Distillation Pipeline (`agents/distill/`)

Two-stage QLoRA training pipeline for reasoning base + domain specialists. Current state: 4B Qwen base trained (serving via llama.cpp, eval 0/5 on coding A/B vs stock Gemma 4 E4B), stock Gemma 4 E4B validated as alternative base. Specialists not yet trained. Hot-swap infrastructure shipped (`agents/model_swap.py` + `SpecialistCoordinator`).

Full spec: `.claude/rules/distillation.md` — pipeline scripts, specialist domain table, training-data file list, training commands, training philosophy.

## Serving Architecture

**llama.cpp (primary)** — Gemma 4 E4B via full GPU:
- **Production GGUF**: `~/models/gemma-4-E4B-it-tq4-aligned.gguf` (5.0 GB, TurboQuant tq4, 132-byte block alignment from session 16). **This is what CALM runs on.**
- **Alternative GGUFs**: `~/models/gemma-4-E4B-it-Q5_K_M.gguf` (5.48 GB, stock Q5), `~/models/Qwen3.5-4B.Q5_K_M.gguf` (2.9 GB, fine-tuned Q5)
- **TurboQuant tq4 KV cache**: `--cache-type-k tq4_k256 --cache-type-v tq4_k256`. 4.125 bpw, 16-level Lloyd-Max codebook, Pi rotation (seed=42, 256×256 orthogonal). 132-byte blocks (128 qs + 2 d + 2 pad for 4-byte aligned uint32 loads). **Old 130-byte GGUFs are incompatible — re-quantize.**
- Context: **512K** with tq4 KV (~5.0 GB weights + ~2.0 GB KV = ~7 GB VRAM). 48K thinking budget (`EFFORT["max"]["max_tokens"]=49152`). Auto-CALM + harness share the same server.
- `--parallel 1` required — without it, llama-server splits `ctx_size` across 4 default slots
- Launch: `llama-server -m ~/models/gemma-4-E4B-it-tq4-aligned.gguf --ctx-size 524288 --parallel 1 --cache-type-k tq4_k256 --cache-type-v tq4_k256 -ngl 999 --port 8080`
- **42 tok/s steady decode** on Gemma 4 E4B tq4 (matches architecture.md; measured via Triton + CUDA Graphs, ~90% of llama.cpp on same GGUF)
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

## R53 — Verified Code-Reasoning Stack (Phase 1 shipped)

Phase 1 (retrieval + DB + generators) complete this session; Phase 2 (PT
training + L24/L30 install) pending. Full rules: `.claude/rules/retrieval.md`,
`.claude/rules/code_reasoning_db.md`, `.claude/rules/recursion.md`.

- **CodeExampleDB** — `calm/llm_computer/facades/code_example_db.py` —
  8970 unique examples across 10 corpora (MBPP, HumanEvalPlus, BigCodeBench,
  CodeContests Python3, generators, Claude-reasoning, etc.), dedup on
  problem hash, 4 retrieval modes (jaccard/tfidf/dense/hybrid).
- **Hybrid retrieval** — `retrieval.py` — TF-IDF+BM25 (68K vocab) +
  Gemma-dense (mean-pooled `token_embd`, fp16 + tq4 4× compression) +
  RRF fusion. Trie-backed fast tokenizer gives 13,000× speedup over
  naive `GemmaTokenizer.encode` (O(len × 262K vocab)). Indices cached
  at `.cache/r53_code_db/`.
- **CodeVerifierFacade** — `code_verifier.py` — intent classifier +
  suggested imports + security flags + `compute_hints()` returns
  prompt-prefix with retrieved examples.
- **9 data generators** at `calm/llm_computer/facades/data_generators/`
  (algorithm_problems, parameterized_math, stdlib_usage, bug_fix_pairs,
  security_patterns, regex_patterns, data_structures, datetime_utils,
  functional_patterns) — 222 sandbox-verified (problem, solution, tests)
  examples produced by `scripts/r53_run_data_generators.py`.
- **Rebuild pipeline**: `PYTHONPATH=. python3 scripts/r53_run_data_generators.py`
  (CPU: TF-IDF only) + `bin/gemma-run scripts/r53_build_dense.py`
  (daemon: dense + tq4 save).
- **R53.2b eval finding** — blanket prompt-RAG gives **+0.0pp retrieval-
  attributable gain** on complex multi-step coding (hinted = sanity-random).
  Prompt-length alone moves +7.4pp; retrieval content adds nothing on top.
- **R53.14/20a/20b (substrate-RAG on code, SWA-fix active) — NEGATIVE**.
  L41 CardSlot(preserve=True) + per-marker FirstTokenHook(boost=50)
  regresses **-9.3pp** on R53.0 corpus even with the SWA attention fix
  (ec8887f, 1a85b0c, b9512ec). Root cause is install-mechanism, not SWA:
  Gemma's first-token on code is confidently a fence/whitespace opener
  (logit margin 6.8-9.2), so forcing "def"/"class" produces
  code-without-fence → extractor fails. On HIT prompts only — miss
  prompts bit-identical. **First-token bias is wrong intervention for
  code**; AST-walker post-generation card is the correct tier-2 path.
  Reframes the "Automatic Tier-1 preservation" thesis: holds for
  hash-match at the output boundary (VerificationHook with `min_margin`),
  but NOT for residual-write CardSlot at arbitrary layers.

**R53 Phase 2 shipped (R53.25-R53.34)**:
- **R53.25** — MAX_TOKENS=900 alone lifts `log_level_counts` 0/0→6/6
  (+6 tests → 32/32, best R53 result). 4 prior null rounds were
  budget-starved, not substrate/sandbox/import failures.
- **R53.28** — `KVCacheTq4` multi-token prefill (S≥1) with per-layer
  position tracking; `trim_swa_storage` via direct tq4 byte-copy (no
  re-quant). Dense retrieval default `prefer_tq4=True`.
- **R53.29** — tq4 matvec v2 kernel: shared-mem centroid LUT via
  `tl.gather` from a program-local (16,) tile. **-7% matvec**
  aggregate, production default (cbb8073). v1 retained for A/B.
- **R53.30/R53.31/R53.32** — null ports from TurboQuant CUDA kernel:
  fp16 x_rot activation (+0.2/+8.7%), uint32 qs loads (+9.8/+16.4%),
  BLOCK_M sweep (heuristic holds). Lesson: Triton auto-coalesces on
  Ada L1; `tl.join`/reshape overhead can exceed BW savings.
- **R53.34** — fused flash-attention decode kernel
  (`calm/llm_computer/tq4_flash_attn.py`), cosine=1.0 parity vs fp32
  at N∈{16…1024}. Non-monotonic perf curve (2026-04-20 re-bench):
  -18% at N=64 (launch overhead dominates), **+14% at N=256 and +6%
  at N=1024** (mid-range sweet spot, captures 82-96% of fp16), -7% at
  N=4096 (cuBLAS-on-memo wins asymptotic). **Shipped with
  `_use_fused_flash_attn=True` default + runtime N-gate
  `128 < kv_cache.layer_pos[kv_src] < 2048`** in `_forward_layer`.
  Chat + short-eval decode runs fused; long R53 eval (AdaptiveBudget
  up to 16K) falls back to Phase 1 memo past N=2048 — no regression
  on long-decode workloads. Phase 1 memo remains the asymptotic
  winner and out-of-gate fallback. Full bench + policy:
  `.claude/rules/turboquant.md` §"Fused flash-attention decode".
  Adjacent null: TurboQuant Q_prod (3-bit Q_mse + 1-bit QJL) —
  unbiased inner-product but softmax-output cosine worse than Q_mse
  alone at every N tested (`tq4_qjl_torch.py` kept for NN/retrieval
  research). Full A/B: `tracing_roadmap.md` Round 53.34 row.
  `USE_TQ4_KV=True` ships in eval scripts via Phase 1 memo path.
- **AdaptiveBudget + 16K ceiling** is the new default in all R53 eval
  scripts (e7b4538); replaces fixed 200-400 tok defaults.
- **Sandbox stdlib pre-import fix** (5dc2dc1, R53.22 diagnosis): user
  `import statistics` triggered transitive `import os` → sandbox blocks
  → eval 0/0. Fix pre-loads ~23 safe stdlib modules before the
  `_safe_import` hook. User `import os` still blocked.

**R53.35 + R53.36 shipped (this session)**: tier-2 AST walker +
tier-3 install audit. Both load-bearing:

- **AST walker** (`calm/llm_computer/facades/ast_repair.py`) — 3
  deterministic rewrites driven by Python error text: shadow rename
  (TypeError callable), dict-key synonym (KeyError + curated
  synonym table), syntax repair (bracket mismatch via error offset
  + insert-before-`:` for `for/if/def` lines). Wired into
  `scripts/r53_21_import_inject.py`. R53.0 lifts: **token_bucket
  0/0 → 5/5 via shadow_rename, csv_column_stats 0/0 → 8/8 via
  syntax_repair (1 missing paren)**. Combined +13 tests on 2 of 6
  problems, mechanical, ~1s per fix, zero LLM retries. 36/36 unit
  tests. **Supersedes "Gemma ignores hints" framing on those two
  problems — Gemma's output was logically correct; extractor
  strictness hid the result.**
- **R51/R52 install audit** (`scripts/r53_36_audit_r51_install.py`)
  — verified install math zero-diff (`L24_installed == h_before +
  student(h_before)` bit-identical). R51-MSE student reproduces
  L24 at cos=0.89, scale=0.91 — 10% diffuse error cascades through
  L25..L41 into wrong argmax. R52-KL student is garbage (cos=-0.02,
  scale=94×) — KL-on-logits doesn't constrain residuals. **Tier-3
  remains closed at current loss space but not in principle;
  Jacobian-weighted loss is a credible reopen path.** Tier-2
  stacking (R46.2) stays the priority — already delivers 17/17
  user-facing wins without tier-3 cost. Full receipts in
  `.claude/rules/capability_gain.md` §R53.35-R53.36.

## Hardware

- Laptop: Acer Nitro AN17-42
- GPU: NVIDIA RTX 4070 Laptop GPU (8 GB VRAM) — no Thunderbolt/eGPU support
- iGPU: AMD Radeon 780M (display only, no CUDA)
- RAM: 32 GB DDR5 5600MHz
- WSL2: Ubuntu 24.04 with GPU access
- Training (local): Unsloth 2026.4.2 + PyTorch 2.10.0+cu128
- Training (cloud): Google Colab Pro A100 (40GB)

## Key Constraints

- **8 GB VRAM**: production default is Gemma 4 E4B tq4 + tq4 KV at **512K context** (~7 GB total, see Serving Architecture). Historical Q4-KV + Q5_K_M configs fit at **256K context** — Qwen 3.5 4B Q5 uses ~7.3 GB, Gemma 4 E4B Q5 uses ~6.7 GB (sliding-window attention makes Gemma's KV cache dramatically smaller at long context). 9B Q4 fits at 2K (Ollama). 0.8B FP16 fits at 32K.
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

**IMPORTANT**: Assume nothing. Hypothesis, Build, Test, Commit & Iterate. First Principles thinking. Do not discount anything until it's built and tested!