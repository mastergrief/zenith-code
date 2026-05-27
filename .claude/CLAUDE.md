# Zenith Code — Multi-Agent Harness + CALM Reasoning Engine

**IMPORTANT**: Assume nothing. Hypothesis, Build, Test, Commit & Iterate. First Principles thinking. Do not discount anything until it's built and tested!

Fork of [ultraworkers/claw-code](https://github.com/ultraworkers/claw-code) with a Python agent harness, CALM reasoning engine, a native HRM-Text-1.58 training stack (the active lane), a Rust port, and the (now-adjacent) HRM + LLM-Computer CRLM/substrate stack.

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

## HRM-Text-1.58 Fork — active training lane

**The active R&D + training lane.** `hrm-158-base` grows by a **90/90-gated**
progressive curriculum. Default slice is one atom: **auditable full-density finite support** (usually ~100-150 rows / ~120 natural) trained slow-safe (LR ~5e-5, replay .80, n-train 12000, eval 200 diagnostic, seeds 17/17, pc/temp 1.0, fixed Tier-B, saves 250..1500).
Bank-gated acquire ≥90% / retain ≥90%; bank earliest all-clear save, final has no privilege. Extra retained-support KL pins are explicit/hash-verified, never default-on or on the target; close siblings clear by numeric gate OR no-new-broad-cluster/parent-floor.
On a miss classify + split/protect/redesign; no LR/runway/model escalation. Canonical workflow: `.claude/rules/hrm-158.md`.

**Conventions (read first):**
- Active training lane: native HRM-Text-1.58 (`hrm-158-base`).
- Default method: auditable full-density finite support + slow-safe learning + 90/90 gate.
- PT / DT / RDT / cards / Substrate / Gemma-substrate / CRLM / CHRLM are legacy/adjacent/reference unless explicitly reopened.
- CALM engine, Python agent harness, Rust port stay live infrastructure.
- Receipts live on the ai-room board / MEMORY — not CLAUDE.md / AGENTS.md / rules.

## Editing `.claude/` Configs

Rules for editing agents, CLAUDE.md, commands, rules, hooks: `.claude/rules/config_editing.md`.

## AI Room Collaboration

When the user directs direct collab with codex via the ai-room MCP —
two independent sessions coordinating through `ai_room_*` tools, NOT
subagent spawning. The "no subagents" working policy above is
unaffected. Full charter: `.claude/rules/AI_ROOM_COLLAB.md` (claude
peer protocol) + `.claude/rules/CLAUDEX_ORCHESTRATION.md` (task-
dispatch lifecycle, recycle boundaries, hook-enforced RETAIN
OVERRIDE) + `.codex/rules/AI_ROOM_COLLAB.md` + `.codex/AGENTS.md`
"AI Room Collaboration" section (codex side).

**R&D team shape**: Gabe is the human direction owner (seeds problems,
picks risk/cost/goal tradeoffs, final human gates). Claude and
`codex_co_lead` are **technical research/strategy co-leads** (joint
hypothesis / curriculum / gate-semantics / counter-case / audit);
Claude is **additionally the operations/execution lead** (AUQ
capture/relay, board orchestration, role bootstrap/dispatch, training
launch/watch, validation/commit/push gatekeeping, synthesis).
Implementation is **role-routed** — Claude direct, or a named Codex
worker role under explicit gates: `training-dev` (default mutating
developer for HRM + main-repo docs/config/tooling), `curriculum`
(read-only planner), `audit` (read-only auditor). Loop:

```
gabe seeds → claude+codex hypothesize/plan/challenge →
implement (Claude direct or routed role under gate) → Claude
launches/watches → claude+codex audit → commit → iterate
```

**Cross-thread is mandatory at every thinking boundary** —
hypothesize, plan, challenge, audit, creativity. Implementation
(build, test, commit) stays off-thread on a single active executor —
Claude direct or a routed Codex role under gate. Cross-thread is the default
rate of the channel, not occasional. Empirical: rounds where claude
got codex's take produced better output than solo. Cache cost ≪
audit lift. Opt-out only for mechanical edits and micro-tuning
inside an already-cross-threaded round.

Non-trivial gabe-facing work follows this gate sequence:

```
intent → decision contract → route → plan gate → implementation/proof
→ validation/diff gate → commit gate → push gate → synthesis or handoff
```

Key rules (summary — see charter for full):
- **Role**: Claude + codex_co_lead are technical research/strategy co-leads; Claude additionally owns ops/execution + material gates. Mutating Codex work routes to `training-dev` by default for HRM and main-repo docs/config/tooling slices; cwd is selected by task class. Voice preserved on split-owned files (peer reviews, doesn't rewrite).
- **Board-first**: `ai_room_task_create` + `_start` BEFORE writing code.
- **Provenance**: cross-session dispatches carry verbatim gabe quote + scope + chosen option in task description. Paraphrase loses signal.
- **Ingress-owned provenance**: ownership follows the user-entry point — gabe-via-claude → claude owns the AUQ/relay packet; gabe-via-codex → codex_co_lead owns it (claude attaches it to tasks/gates, runs AUQ only on ambiguous/material-risk). Provenance is authority context, NOT a material gate. **No second dispatcher**: codex recommends routes/contracts/reviews; claude spawns/dispatches/gates named workers (routable `codex_N` handle — role name ≠ handle).
- **Cascade boundary**: pause + name one counter-case before dispatching >2 tasks or multi-subsystem edits.
- **Before idle**: `ai_room_resume_check` first; board is canonical.
- **Disagreement**: every non-trivial proposal names one risk or is marked "trivial, no counters." One cited correction beats three hedges; concede cited corrections first-round.
- **Round-closure signaling**: lead says "round closed unless one more hole" before synthesis; peer flags final hole or concurs.
- **Receipt discipline**: verbatim-lift one-liners from rounds into artifacts (commits, specs, handoffs); credit by message ID. Routine gate closures get a tight prose one-liner, not a structured receipt.
- **Parallel drafting**: on expertise-clean splits, both authors draft independently + cross-review + single commit. ~40% faster than sequential.
- **TDD by collab**: tests-for-desired-behavior; tests-later OK for crashes, NOT for silent-failure paths.
- **Validation discipline**: fresh-process seeded-log for landing-day code; real-product-path > unit tests for user-visible shape.
- **REPL-only synthesis**: when gabe posts via the ai-room channel/REPL, the substantive answer lives in the room alone — no chat-side duplicate. Chat-side may carry tool mechanics (AskUserQuestion capture, brief acks) but the user-facing synthesis goes to the room only.
- **Capture-then-relay**: for non-trivial durable decisions (batched choices, spec closures, route picks, material gates, product defaults), default to `AskUserQuestion` chat-side then IMMEDIATELY relay the locked answer to the room as a persisted non-ack record BEFORE any material action — threaded to source/parent and targeted at `codex_co_lead` for challenge/+1. Chat answer = provenance; room relay = the durable gate.
- **`@gabe` is the trigger**: any inbound message addressing gabe (4 shapes: `to: gabe`, `requires_response_from: gabe`, reply-to whose sender is gabe, or `@gabe` in body outside quoted text) triggers AUQ. Hook `.claude/hooks/at_gabe_askuserquestion_gate.py` enforces at the `mcp__ai-room__ai_room_post`/`_reply` boundary.
- **AUQ recommendations mandatory**: first option carries `(Recommended)`. Genuinely no recommendation is rare — usually means think harder before posing.
- **Mixed-purpose posts banned**: one outbound post = one purpose. Either pure relay of an already-captured answer OR a fresh ask, never both. Closeouts that surface future decisions name them as **carry-forward** (deferred), not inline decisions.
- **Verify `+1` claims as persisted records**: a valid material gate (`+1 implement` / `+1 commit` / `+1 push`) is a claude-authored, non-ack ai-room post threaded to the pending request. Cite the gate msg id in the next status. Remembered, paraphrased, or unresolvable gate ids are not authority — ask claude to re-confirm on-thread.
- **Cited msg ids are untrusted until resolved**: a msg id appearing only inside another agent's prose is not proof the original message exists. Verify against ai-room search / tail / read output before acting on it.
- **`ai_room_task_update` does NOT wake peers**: task-state transitions are durable board records, not wake events. When correcting a task post-creation, pair the `task_update` (audit record) with a direct addressed post citing the task_update msg id (wake signal).
- **Inbound replies are push-delivered**: when waiting on codex, the reply arrives as a mid-turn `<channel>` injection. Do NOT poll `ai_room_inbox` or arm sleep loops — continue other work or stand by.
- **Fast training launch**: GPU launches compress to one launch packet (parent sha/config proof + dry-run-validated command + watcher bundle + stop/bank criteria) → one co-lead `+1 launch/watch-to-terminal-condition` → claude runs/watches directly → one terminal receipt; interrupt only for bank pass / hard failure / criteria mismatch / resource-liveness / material deviation. Full: `.claude/rules/AI_ROOM_COLLAB.md` §"Fast Training Launch Contract".

---

## Commercial Potential

Long-term commercial direction documented in `.claude/rules/commercial.md`. Currently R&D — focus on building the best system, not shipping a product. Commercial awareness is context, not a constraint.

## Substrate vs Cards vs CHRLM — vocabulary (legacy/adjacent)

> Legacy/adjacent, not the active lane. Use parked-stack terms (Substrate /
> Card / Build / CHRLM / PT / DT / output-language-family / domain) + the
> Brain+Cards install model precisely only when working in that stack.
> Glossary: `MEMORY/atlas/Substrate_arc.md`. PT/DT: `delta_rule.md`. Install
> paths: `Substrate.md`, `compute_facades.md`, `recursion.md`.

## Architecture

The repo carries four subsystems. The **active training lane is native
HRM-Text-1.58** (see §"HRM-Text-1.58 Fork" above). Of the four below, the
harness, CALM engine, and Rust port are live infrastructure; the Unified Single
Tensor / substrate (#4) is **legacy/adjacent** — parked unless reopened.

1. **Python agent harness** (`agents/`, ~4,423 LOC across 15 files) — terminal coding assistant with dual backend (Ollama + llama.cpp), 3-level permissions, thinking mode, sessions, compaction, effort control, llama.cpp hot-swap. Commands + launch: `.claude/rules/harness.md`. Internals: `.claude/rules/architecture.md` §"Agent System".
2. **CALM engine** (`calm/`, ~83,600 LOC across 413 .py files) — modular compute + knowledge facade with cognitive intelligence layer. Auto-CALM + Engine V2 (7-phase pipeline) + 120 modular backends + 39 cognitive modules + self-healing quality loop. Full spec: `.claude/rules/calm.md` (atlas: `MEMORY/atlas/calm_part_1.md` + `calm_part_2.md`).
3. **Rust claw-code port** (`rust/`) — upstream claw-code, 9 crates, separate build system.
4. **Unified Single Tensor** (`calm/llm_computer/`) — *legacy/adjacent*, not the active lane. CHRLM substrate architecture: ONE `.pt` contains Gemma (tq4) + trained PTs + compiled cards + persistent knowledge DB. Spec (reference): `.claude/rules/Substrate.md` + `architecture.md` + `delta_rule.md`.

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

## Pointer Transducers + LLM-Computer (`calm/hrm/` + `calm/llm_computer/`) — legacy/adjacent

> Legacy/adjacent, not the active lane. CRLM split (PT extracts NL→expr
> structure; LLM-Computer compiles values) — reusable for
> retrieval/structure-extraction only if reopened. Spec: `architecture.md`,
> `delta_rule.md`. Recipes: `delta_rule.md` + `MEMORY/atlas/training_part_1.md`/`_part_2.md`.
> Domain registry: `MEMORY/substrate_registry.md`; `/domain` to add.

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

## Verified Code-Reasoning Stack (legacy/adjacent)

Substrate-arc work, parked — not the active lane. Reference: `.claude/rules/retrieval.md`, `.claude/rules/code_reasoning_db.md`, `.claude/rules/recursion.md`, `.claude/rules/capability_gain.md` (receipts in `.claude/MEMORY/atlas/`).

## VGSL — post-transformer architecture R&D

`RESEARCH/VGSL/` holds a 4-file architecture spec for a Verifier-Governed Substrate Log — a post-transformer design that moves knowledge out of opaque weights into a versioned, verifier-governed, canonicalized event log with temporally-indexed projection. R&D direction, not shipping arc. Three user-facing options documented in `RESEARCH/VGSL/00_INDEX.md`: park / Phase-1 prototype (1-week bounded experiment) / scope to a commercial vertical.

## Needle-in-Haystack Validation

Effective context for both 4B base models (Gemma 4 E4B 200K, Qwen 3.5 4B 130K) validated against single / multi / distractor NIAH at 4K–220K. Full table + findings + `MODEL_CONTEXT_LIMITS` source of truth: `.claude/rules/niah_validation.md`.

## Branch

`feature/multi-agent-qwen` on `mastergrief/zenith-code` (forked from `ultraworkers/claw-code`; renamed from `mastergrief/claw-code` 2026-04-07)

**IMPORTANT**: Assume nothing. Hypothesis, Build, Test, Commit & Iterate. First Principles thinking. Do not discount anything until it's built and tested!
