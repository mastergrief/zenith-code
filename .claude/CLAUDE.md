# Zenith Code — Multi-Agent Harness + CALM Reasoning Engine

**IMPORTANT**: Assume nothing. Hypothesis, Build, Test, Commit & Iterate. First Principles thinking. Do not discount anything until it's built and tested!

Fork of [ultraworkers/claw-code](https://github.com/ultraworkers/claw-code) with a Python agent harness, CALM reasoning engine, HRM + LLM-Computer (the CRLM stack), and a Rust port.

**Working policy: solo lead by default; agents in specific cases.**

**Direct tools by default** — Edit/Write/Read/Grep/Bash for fast-iteration R-round hypothesis-test loops, edits within the session's working memory, and tasks under ~10 file changes. That's most of what we do. Why: R52.1 receipt — ~400 LOC delegated cost a ~2000-word brief + 30 min cold-read + 1 hr/iteration vs ~10 min solo, plus missed 500× perf regressions from missing baselines.

**Slash commands with documented subagent use WIN over this default.** Specifically:
- **`/update`** — ALWAYS fires 3-agent split (transcript / code / docs) per `.claude/commands/update.md` Phase 1 whenever the session passes its own threshold (>1 subsystem, >3 commits, OR introduced a new mechanism). Don't subvert this by going inline — the command's docs are canonical.
- **`/handoff`** — ALWAYS fires 2-agent grounding (transcript + code/uncommitted) per `.claude/commands/handoff.md` when the session passes its threshold (>3 commits, >1 subsystem, new mechanism, OR session log exceeds ~30K tokens). Same rule: don't subvert.

**Discretionary spawn outside slash commands when**:
- **(a) Semantic exploration across an unfamiliar subsystem** — Explore agent with `thoroughness: "very thorough"` parallelizes searches that would otherwise run sequentially. Trigger: question is semantic, not a literal grep (e.g. "find every tier-2 install pattern across `calm/llm_computer/facades/`").
- **(b) Independent second-opinion review on high-blast-radius changes** — code-reviewer / security-review agent AFTER a risky commit, BEFORE push. Fresh context catches what the author rationalized. Candidates: Triton autograd / gradient-math (R52.1c cascade-bug class), production-serve integrations, security-adjacent code, refactors larger than one subsystem.
- **(c) Context protection on high-volume searches** — agent scans in its own context when a grep would flood main context with >1000 expected matches.

**Never**: spawn agents "just in case", for work that fits in one direct tool call, or as a default orchestration pattern when the above cases don't apply. User's explicit ask for teams/parallel workers overrides — if asked, spawn.

## Default Workflow — Hypothesis, Test, Iterate
Full spec: `.claude/rules/workflow.md` (historical receipts: `.claude/MEMORY/atlas/workflow_part_1.md` + `workflow_part_2.md`)

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

## Editing `.claude/` Configs

Rules for editing agents, CLAUDE.md, commands, rules, hooks: `.claude/rules/config_editing.md`.

## AI Room Collaboration

When the user directs direct collab with codex via the ai-room MCP —
two independent sessions coordinating through `ai_room_*` tools, NOT
subagent spawning. The "no subagents" working policy above is
unaffected. Full charter: `.claude/rules/AI_ROOM_COLLAB.md` (claude
side) + `.codex/rules/AI_ROOM_COLLAB.md` + `.codex/AGENTS.md` "AI
Room Collaboration" section (codex side).

Key rules (summary — see charter for full):
- **Role**: claude lead, codex peer. Lead swaps by subsystem. Voice preserved on split-owned files (peer reviews, doesn't rewrite).
- **Board-first**: `ai_room_task_create` + `_start` BEFORE writing code.
- **Cascade boundary**: pause + name one counter-case before dispatching >2 tasks or multi-subsystem edits.
- **Before idle**: `ai_room_resume_check` first; board is canonical.
- **Disagreement**: every non-trivial proposal names one risk or is marked "trivial, no counters." One cited correction beats three hedges; concede cited corrections first-round.
- **Round-closure signaling**: lead says "round closed unless one more hole" before synthesis; peer flags final hole or concurs. Predictable exits shrink dead-time.
- **Receipt discipline**: verbatim-lift one-liners from rounds into artifacts (commits, specs, handoffs); credit by message ID.
- **Parallel drafting**: on expertise-clean splits, both authors draft independently + cross-review + single commit. ~40% faster than sequential.
- **TDD by collab**: tests-for-desired-behavior; tests-later OK for crashes, NOT for silent-failure paths.
- **Validation discipline**: fresh-process seeded-log for landing-day code; real-product-path > unit tests for user-visible shape.

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
  (not per domain). ~185K params, ~32 sub-heads.
- **DT** (Delta-Transducer) = `CopyAugmentedDeltaNet` card — 2026-04-22
  canonical rename of PT+Delta. Underlying class unchanged. Default
  trained-card architecture for **retrieval/structure-extraction**
  regimes (MQAR, NL→math). Code-skeleton DT (NL → `def FN(<args>):`)
  is an open arc at 0.193 honest val (v13 ep16, 520 held-out) —
  not install-viable yet. See `delta_rule.md` §DT.
- **Output-language family** = a class of expression syntax. Function-call
  (`fn(args)`), infix arithmetic (`a + b`), boolean logic (`a > b and`).
  ~3-5 families cover 30+ domains. Adding a domain within an existing
  family is a data-only operation.
- **Domain** = a facade with imports/exports + PT + compiled ops +
  knowledge facts. ~32 sub-heads per domain, 30 domains on 8 GB VRAM.

**Brain + Cards model**: Gemma (language + routing) dispatches to cards
(compiled programs, HRM specialists, PTs). Three install paths —
decode-path facade (zero VRAM, cheapest), CardSlot residual-additive,
in-tensor. Full spec + tradeoffs: `.claude/rules/Substrate.md` §"Card
Installation", `.claude/rules/compute_facades.md`,
`.claude/rules/delta_rule.md` §"Retrieval card install". Auto-generation via
`calm/llm_computer/recursion.py` (`FacadeSpec` + `MetaFacade`) —
see `.claude/rules/recursion.md`.

## Architecture

**Model understands, transducers structure, cards compute, engine verifies.** Intelligence comes from the system architecture, not the weights. Adding a backend module is equivalent to training — the model gets smarter at that domain instantly, with zero GPU cost.

Four active systems coexist:
1. **Python agent harness** (`agents/`, ~4,423 LOC across 15 files) — terminal coding assistant with dual backend (Ollama + llama.cpp), 3-level permissions, thinking mode, sessions, compaction, effort control, llama.cpp hot-swap. Commands + launch: `.claude/rules/harness.md`. Internals: `.claude/rules/architecture.md` §"Agent System".
2. **CALM engine** (`calm/`, ~83,600 LOC across 413 .py files) — modular compute + knowledge facade with cognitive intelligence layer. Auto-CALM + Engine V2 (7-phase pipeline) + 120 modular backends + 39 cognitive modules + self-healing quality loop. Full spec: `.claude/rules/calm.md` (atlas: `MEMORY/atlas/calm_part_1.md` + `calm_part_2.md`).
3. **Rust claw-code port** (`rust/`) — upstream claw-code, 9 crates, separate build system.
4. **Unified Single Tensor** (`calm/llm_computer/`) — CHRLM architecture. ONE `.pt` contains Gemma (tq4) + trained PTs + compiled cards + persistent knowledge DB. Session 32 ported Level 5 to prod Gemma 4 E4B. Full spec: `.claude/rules/Substrate.md` + `architecture.md` + `delta_rule.md`.

Serving + VRAM + perf: `.claude/rules/environment.md` §"Serving Architecture".
Mechinterp tracing arc (sessions 33-34, full R-arc): `.claude/rules/augmentation_thesis.md` (current strategic positions) + `tracing_intelligence.md` (first-principles bound) + `.claude/MEMORY/atlas/tracing_roadmap_part_1.md` (per-round receipts) + `.claude/MEMORY/atlas/augmentation_thesis_arc.md` (capability map + R51/R52 distillation null detail).

## Python Agent Harness (`agents/`)

Terminal coding assistant, ~4,423 LOC across 15 files. Commands table
+ launch examples: `.claude/rules/harness.md`. Internals (streaming,
tool-call loop, permissions, compaction invariants, hot-swap):
`.claude/rules/architecture.md` §"Agent System" + §"File Organization".

## CALM Engine (`calm/`)

**"Deterministic brain on top of a probabilistic nervous system."** LLM
reasons, modular CPU backends compute, 4-lane TMR verifies, results feed
back. No fine-tuning required — add a backend, model gets smarter
instantly. 100% on 40-problem benchmark with precompute. Auto-CALM
(default, transparent) + Explicit CALM (`<calm>` blocks, power user).
120 backends / 1002 functions / 550 NL patterns. 39 cognitive modules
across 5 layers. Engine V2 7-phase pipeline with self-healing.

Full spec: `.claude/rules/calm.md` (atlas: `MEMORY/atlas/calm_part_1.md` + `calm_part_2.md`).

```bash
python3 -m calm.auto_calm "What is 347 * 289? Is it prime?"
python3 -m calm.engine "What is 17 * 23?"
python3 -m pytest calm/ -v
```

## Pointer Transducers + LLM-Computer (`calm/hrm/` + `calm/llm_computer/`)

The CRLM split: **Pointer Transducers** (learned, ~185K params) handle
NL → expression structure extraction via copy-augmented attention;
**LLM-Computer** (analytically compiled) handles value computation. PT
superseded HRM for all new work (session 31); PT+Delta
(`CopyAugmentedDeltaNet`, R-delta-20, 2026-04-21) supersedes plain PT
as the default trained-card architecture.

Architecture spec: `.claude/rules/architecture.md`. Training recipes +
checkpoint inventory: `.claude/rules/training.md` (atlas:
`MEMORY/atlas/training_part_1.md` + `training_part_2.md`). PT+Delta
mechanics + MQAR data-scaling curve + retrieval-card install:
`.claude/rules/delta_rule.md`. Domain registry:
`.claude/MEMORY/substrate_registry.md`. Add a domain: `/domain` command.

## Distillation Pipeline (`agents/distill/`)

Two-stage QLoRA training pipeline for reasoning base + domain specialists. Current state: 4B Qwen base trained (serving via llama.cpp, eval 0/5 on coding A/B vs stock Gemma 4 E4B), stock Gemma 4 E4B validated as alternative base. Specialists not yet trained. Hot-swap infrastructure shipped (`agents/model_swap.py` + `SpecialistCoordinator`).

Full spec: `.claude/rules/distillation.md` — pipeline scripts, specialist domain table, training-data file list, training commands, training philosophy.

## Serving, Hardware, Constraints

Consolidated in `.claude/rules/environment.md`: hardware (RTX 4070
Laptop / 8 GB VRAM / 32 GB RAM), serving architecture (llama.cpp
primary with tq4 + tq4 KV @ 512K, Ollama fallback), local tools
(custom llama.cpp `zenith` branch + patches), cloud accounts (RunPod,
Colab Pro), key VRAM / context constraints. Update there when
hardware, GGUF paths, accounts, or budgets change.

tq4 kernel internals + fused flash-attn decode + per-kernel bench
receipts: `.claude/rules/turboquant.md`.

## R53 — Verified Code-Reasoning Stack

Phase 1 (retrieval + DB + generators) shipped; Phase 2 (PT training + L24/L30 install) pending. Full receipts + per-round findings in: `.claude/rules/retrieval.md`, `.claude/rules/code_reasoning_db.md`, `.claude/rules/recursion.md`, `.claude/MEMORY/atlas/tracing_roadmap_part_1.md` ruled-out log, `.claude/rules/capability_gain.md` (receipts in `.claude/MEMORY/atlas/capability_gain_arc.md`).

## Needle-in-Haystack Validation

Effective context for both 4B base models (Gemma 4 E4B 200K, Qwen 3.5 4B 130K) validated against single / multi / distractor NIAH at 4K–220K. Full table + findings + `MODEL_CONTEXT_LIMITS` source of truth: `.claude/rules/niah_validation.md`.

## Branch

`feature/multi-agent-qwen` on `mastergrief/zenith-code` (forked from `ultraworkers/claw-code`; renamed from `mastergrief/claw-code` 2026-04-07)

**IMPORTANT**: Assume nothing. Hypothesis, Build, Test, Commit & Iterate. First Principles thinking. Do not discount anything until it's built and tested!
