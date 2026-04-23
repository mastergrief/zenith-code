# Zenith Code — Multi-Agent Harness + CALM Reasoning Engine

**IMPORTANT**: Assume nothing. Hypothesis, Build, Test, Commit & Iterate. First Principles thinking. Do not discount anything until it's built and tested!

Fork of [ultraworkers/claw-code](https://github.com/ultraworkers/claw-code) with a Python agent harness, CALM reasoning engine, HRM + LLM-Computer (the CRLM stack), and a Rust port.

**Working policy: no subagents.** Work directly with `Edit`/`Write`/`Read`/`Grep`/`Bash`. Do not dispatch subagents or create teams. Prior VDD/orchestration infrastructure was removed in commit `bb7f13d`; the agent definitions and `/VDD`, `/DISCOVER`, `/EVAL`, `/TRAIN-DATA` slash-commands no longer exist. Session 26 and Vector 1 shipped 23+ commits + 311 tests directly; this is the proven default for the project.

If a multi-step workflow is genuinely needed, structure it as sequential hypothesis → build → test → commit rounds per the workflow rules.

**ai-room collab is not a subagent pattern.** When the user directs claude+codex collaboration via ai-room MCP, it's two independent top-level sessions exchanging structured messages — not subagents inside one session. See "AI Room Collaboration" below. The no-subagents rule is unaffected.

## Default Workflow — Hypothesis, Test, Iterate
Full spec: `.codex/rules/workflow.md` (historical receipts: `.codex/MEMORY/atlas/workflow_part_1.md` + `workflow_part_2.md`)

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

## Editing `.codex/` Configs

Rules for editing AGENTS.md, commands, rules, hooks: `.codex/rules/config_editing.md`.

## AI Room Collaboration

When the user directs direct collaboration with Claude through the
ai-room MCP, work with Claude as the default operating mode rather
than treating the user as a relay. Claude is lead collaborator; don't
wait for the user between routine steps when claude and codex can
coordinate directly. Share concrete tasks, keep the split explicit,
actively cross-check each other's reasoning. Full charter:
`.codex/rules/AI_ROOM_COLLAB.md` + `.claude/rules/AI_ROOM_COLLAB.md`.

### Session Start — first action

When the ai-room MCP is registered (via `.codex/config.toml`
`[mcp_servers.ai_room]`), call `ai_room_resume_check` on your first
turn BEFORE replying to the user's prompt. If it returns `respond to
<id>` or `resume task <id>`, follow that directive. If `idle ok`,
proceed with the user prompt normally. First-action rule, not a
preference: silent-with-unread looks identical from outside to
not-connected. Cost ~200ms vs one user turn spent nudging.

### Grounded pushback

The highest-signal collab moves come from reading live code *before*
pushing back on a proposal. When claude proposes a design that looks
off, cite specific `file:line` evidence: "generic path X already
handles this at `foo.py:42`, the seam you want is actually at Y."
Prose-only disagreement is worth less than a grounded counter.

- Before pushing back, spend one read-pass on the relevant code.
  Cite files/functions checked.
- Don't silently agree with a proposal you have concerns about —
  surface them as specific counter-cases, not hedged prose.
- When a proposal assumes a primitive that doesn't exist as
  described, say so plainly and propose the narrower shape.

### Status cadence

Silent heads-down looks identical to "stalled" from outside. Claude
cannot tell the difference from the board or channel alone. Post at
these boundaries at minimum:

- **Task start**: one-line `task_start` note ("claiming X, first
  move is Y").
- **Design-turn landing**: when a substantive decision or code
  extraction lands, even if uncommitted — claude may be waiting on
  the contract shape to start its side.
- **Completion / blocker**: `task_complete` with a manifest, or an
  explicit "blocked on Z" post.

Meta-messages ("resume_check points me back to X", "approving direct
collab") without code delta look like ritual. If heads-down and
silent, a 30-word "working on Z next, ETA ~N min" clears the
ambiguity at near-zero cost.

### Concrete asks over open-ended scope

Open-ended ("implement X") stalls more than concrete ("extract
function Y returning struct Z with fields A/B/C"). When claude hands
over a slice with vague boundaries, push back once for sharpening
before starting. Symmetrically, when you hand claude a slice, give
the concrete contract (struct fields, file paths, schema shape) early
— claude can draft against a tentative contract without committing
but cannot TDD against nothing.

### Validation discipline

- **Fresh-process seeded-log for landing-day code.** Long-lived MCP
  subprocesses don't reload source. Testing landing-day behavior by
  live-posting through a stale subprocess tests stale code-in-memory.
  Seed the log with fixture entries BEFORE spawning a fresh
  subprocess.
- **Isolated scratch env for product-path proofs.** Run the product
  binary with a fresh `$CODEX_HOME=/tmp/...` (or equivalent) so shared
  state isn't polluted and cleanup is trivial.
- **Real-product-path > unit tests for user-visible shape.** Unit
  tests prove logic; they don't prove emitted-file shape, cleanup on
  exit, or real handshakes. Ship at least one "run the actual
  binary" smoke alongside the unit suite for anything crossing the
  fs/network boundary.

### Ack + signal discipline

- One reply per distinct signal. Do not ack an ack. Duplicate acks
  under context pressure multiply wake costs on the other side and
  fragment focus.
- If context is >90%, compact proactively. The token cost of compact
  is lower than the cost of repeated meta-only messages that then
  need re-context by the peer.
- When `resume_check` returns a directive, follow it. Do not send a
  new "standing by" message instead. Board is canonical.

### Commit hygiene for collab-scope work

- Bundle coherent session-work into one commit with a body that names
  each sub-feature (A / B / C). Attribute deferred semantics
  explicitly.
- Never cut a focused commit from a worktree that also contains
  unrelated prior-session drift and let the subject hide it. Name
  drift in the body or split.
- User-scope tooling changes (e.g. `~/.ai-room/*`) don't land in the
  repo commit; reference them in the body ("landed live; not
  repo-tracked").

---

## Commercial Potential

Long-term commercial direction documented in `.codex/rules/commercial.md`. Currently R&D — focus on building the best system, not shipping a product. Commercial awareness is context, not a constraint.

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
in-tensor. Full spec + tradeoffs: `.codex/rules/Substrate.md` §"Card
Installation", `.codex/rules/compute_facades.md`,
`.codex/rules/delta_rule.md` §"Retrieval card install". Auto-generation via
`calm/llm_computer/recursion.py` (`FacadeSpec` + `MetaFacade`) —
see `.codex/rules/recursion.md`.

## Architecture

**Model understands, transducers structure, cards compute, engine verifies.** Intelligence comes from the system architecture, not the weights. Adding a backend module is equivalent to training — the model gets smarter at that domain instantly, with zero GPU cost.

Four active systems coexist:
1. **Python agent harness** (`agents/`, ~4,423 LOC across 15 files) — terminal coding assistant with dual backend (Ollama + llama.cpp), 3-level permissions, thinking mode, sessions, compaction, effort control, llama.cpp hot-swap. Commands + launch: `.codex/rules/harness.md`. Internals: `.codex/rules/architecture.md` §"Agent System".
2. **CALM engine** (`calm/`, ~83,600 LOC across 413 .py files) — modular compute + knowledge facade with cognitive intelligence layer. Auto-CALM + Engine V2 (7-phase pipeline) + 120 modular backends + 39 cognitive modules + self-healing quality loop. Full spec: `.codex/rules/calm.md` (atlas: `MEMORY/atlas/calm_part_1.md` + `calm_part_2.md`).
3. **Rust claw-code port** (`rust/`) — upstream claw-code, 9 crates, separate build system.
4. **Unified Single Tensor** (`calm/llm_computer/`) — CHRLM architecture. ONE `.pt` contains Gemma (tq4) + trained PTs + compiled cards + persistent knowledge DB. Session 32 ported Level 5 to prod Gemma 4 E4B. Full spec: `.codex/rules/Substrate.md` + `architecture.md` + `delta_rule.md`.

Serving + VRAM + perf: `.codex/rules/environment.md` §"Serving Architecture".
Mechinterp tracing arc (sessions 33-34, full R-arc): `.codex/rules/augmentation_thesis.md` (current strategic positions) + `tracing_intelligence.md` (first-principles bound) + `.codex/MEMORY/atlas/tracing_roadmap_part_1.md` (per-round receipts) + `.codex/MEMORY/atlas/augmentation_thesis_arc.md` (capability map + R51/R52 distillation null detail).

## Python Agent Harness (`agents/`)

Terminal coding assistant, ~4,423 LOC across 15 files. Commands table
+ launch examples: `.codex/rules/harness.md`. Internals (streaming,
tool-call loop, permissions, compaction invariants, hot-swap):
`.codex/rules/architecture.md` §"Agent System" + §"File Organization".

## CALM Engine (`calm/`)

**"Deterministic brain on top of a probabilistic nervous system."** LLM
reasons, modular CPU backends compute, 4-lane TMR verifies, results feed
back. No fine-tuning required — add a backend, model gets smarter
instantly. 100% on 40-problem benchmark with precompute. Auto-CALM
(default, transparent) + Explicit CALM (`<calm>` blocks, power user).
120 backends / 1002 functions / 550 NL patterns. 39 cognitive modules
across 5 layers. Engine V2 7-phase pipeline with self-healing.

Full spec: `.codex/rules/calm.md` (atlas: `MEMORY/atlas/calm_part_1.md` + `calm_part_2.md`).

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

Architecture spec: `.codex/rules/architecture.md`. Training recipes +
checkpoint inventory: `.codex/rules/training.md` (atlas:
`MEMORY/atlas/training_part_1.md` + `training_part_2.md`). PT+Delta
mechanics + MQAR data-scaling curve + retrieval-card install:
`.codex/rules/delta_rule.md`. Domain registry:
`.codex/MEMORY/substrate_registry.md`. Add a domain: `/domain` command.

## Distillation Pipeline (`agents/distill/`)

Two-stage QLoRA training pipeline for reasoning base + domain specialists. Current state: 4B Qwen base trained (serving via llama.cpp, eval 0/5 on coding A/B vs stock Gemma 4 E4B), stock Gemma 4 E4B validated as alternative base. Specialists not yet trained. Hot-swap infrastructure shipped (`agents/model_swap.py` + `SpecialistCoordinator`).

Full spec: `.codex/rules/distillation.md` — pipeline scripts, specialist domain table, training-data file list, training commands, training philosophy.

## Serving, Hardware, Constraints

Consolidated in `.codex/rules/environment.md`: hardware (RTX 4070
Laptop / 8 GB VRAM / 32 GB RAM), serving architecture (llama.cpp
primary with tq4 + tq4 KV @ 512K, Ollama fallback), local tools
(custom llama.cpp `zenith` branch + patches), cloud accounts (RunPod,
Colab Pro), key VRAM / context constraints. Update there when
hardware, GGUF paths, accounts, or budgets change.

tq4 kernel internals + fused flash-attn decode + per-kernel bench
receipts: `.codex/rules/turboquant.md`.

## R53 — Verified Code-Reasoning Stack

Phase 1 (retrieval + DB + generators) shipped; Phase 2 (PT training + L24/L30 install) pending. Full receipts + per-round findings in: `.codex/rules/retrieval.md`, `.codex/rules/code_reasoning_db.md`, `.codex/rules/recursion.md`, `.codex/MEMORY/atlas/tracing_roadmap_part_1.md` ruled-out log, `.codex/rules/capability_gain.md` (receipts in `.codex/MEMORY/atlas/capability_gain_arc.md`).

## Needle-in-Haystack Validation

Effective context for both 4B base models (Gemma 4 E4B 200K, Qwen 3.5 4B 130K) validated against single / multi / distractor NIAH at 4K–220K. Full table + findings + `MODEL_CONTEXT_LIMITS` source of truth: `.codex/rules/niah_validation.md`.

## Branch

`feature/multi-agent-qwen` on `mastergrief/zenith-code` (forked from `ultraworkers/claw-code`; renamed from `mastergrief/claw-code` 2026-04-07)

**IMPORTANT**: Assume nothing. Hypothesis, Build, Test, Commit & Iterate. First Principles thinking. Do not discount anything until it's built and tested!
