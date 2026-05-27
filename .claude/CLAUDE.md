# Zenith Code — Multi-Agent Harness + CALM Reasoning Engine

**IMPORTANT**: Assume nothing. Hypothesis, Build, Test, Commit & Iterate. First Principles thinking. Do not discount anything until it's built and tested.

Fork of `ultraworkers/claw-code` with a Python agent harness, CALM reasoning engine, native HRM-Text-1.58 training stack (active lane), Rust port, and adjacent HRM + LLM-Computer CRLM/substrate stack.

## Start Here

- Default workflow: `.claude/rules/workflow.md`.
- Active HRM lane: `.claude/rules/hrm-158.md` and `.claude/rules/training.md`.
- Config/rules edits: `.claude/rules/config_editing.md`.
- ai-room collab: `.claude/rules/AI_ROOM_COLLAB.md` and `.claude/rules/CLAUDEX_ORCHESTRATION.md`.
- Long-running shell jobs: `.claude/rules/shell_monitor.md`.

## HRM-Text-1.58 Active Lane

Active fork target: **`hrm-158-base`**, a robust all-rounder native HRM-Text-1.58 checkpoint. Loss is response-only: prompt/instruction tokens are masked.

Default slice is one atom: **auditable full-density finite support** (usually ~100-150 rows / ~120 natural), trained slow-safe with LR ~5e-5, replay .80, n-train 12000, heldout/eval 200 diagnostic unless promoted, seeds 17/17, pc/temp 1.0, fixed Tier-B, saves 250..1500.

Bank gate is acquire ≥90% / retain ≥90% per slice. Bank earliest all-clear save; final has no privilege. Close siblings clear by numeric gate OR no-new-broad-cluster/parent-floor. Retention uses explicit replay + parent consistency + broad retained supports (L0b, math_a0) + direct close-sibling protection for shared template/emission surfaces; extra run-specific KL pins must be launch-command entries with ENABLED count/hash proof, never default-on and never on the target. Anchors are sentinels, not the primary retention mechanism.

On a miss: classify, then split smaller / protect / redesign. Do **not** stretch runway, bump LR, or add model capacity to force a fragile slice through. Cached/batched probe path is default; native ternary train is preferred when available. `.pt` artifacts are runtime/research outputs; commit code/tooling/docs/manifest receipts, not `.pt` by default.

Conventions:
- Active training lane: native HRM-Text-1.58 (`hrm-158-base`).
- Default method: auditable full-density finite support + slow-safe learning + 90/90 gate.
- PT / DT / RDT / cards / Substrate / Gemma-substrate / CRLM / CHRLM are legacy/adjacent/reference unless explicitly reopened.
- CALM engine, Python agent harness, and Rust port stay live infrastructure.
- Receipts live on the ai-room board / MEMORY, not this manifest or eager rules.

## Workflow

Every "done", "working", "fixed", or "faster" claim needs a measurement taken after the change. Loop: state hypothesis → pick measurement first → minimal edit → build → measure → binary decision → log what was ruled out → next hypothesis.

Use two measurements every round when applicable: a raw/fast path and a user-facing path. Perf gains that break correctness are reverts; canonical correctness smoke is `17×23=391` via the chat/API path.

Commit completed, measured work before starting the next risky round. Do not use `--no-verify`, force-push shared branches, or silently discard unrelated drift.

## Working Policy

Direct tools are default for fast iteration. Slash commands with documented agent use can override their own flow; otherwise avoid spawning agents "just in case." ai-room collaboration is not a subagent pattern: it is two independent top-level sessions exchanging structured messages.

## AI Room Collaboration

Gabe is the human direction owner / research sponsor / final risk-cost-goal authority. Claude and `codex_co_lead` are technical research/strategy co-leads. Claude is additionally operations/execution lead: AUQ capture/relay, board orchestration, role bootstrap/dispatch, training launch/watch, validation/commit/push gates, and final synthesis.

Implementation is role-routed: Claude direct, or a named Codex worker role under explicit gates. `training-dev` is the default mutating developer for HRM and main-repo docs/config/tooling; `curriculum` is read-only planning; `audit` is read-only gate/metric review.

Non-trivial cross-agent work follows:

```text
intent → decision contract → route → plan gate → implementation/proof
→ validation/diff gate → commit gate → push gate → synthesis or handoff
```

Key invariants:
- Board-first: create/start shared tasks before implementation work that outlives one exchange.
- Provenance: cross-session dispatches carry verbatim Gabe quote, scope, chosen option, and rejected alternatives when relevant.
- Material gates: persisted Claude-authored non-ack `+1 implement`, `+1 commit`, and `+1 push` records are authority; remembered/paraphrased gates are not.
- Capture then relay: non-trivial durable Gabe decisions are captured chat-side and immediately relayed to the room before material action.
- `ai_room_task_update` is durable board state but not a wake; pair required action with a direct addressed post.
- Before idle, run `ai_room_resume_check`; board state is canonical.
- Codex never directly asks Gabe for non-trivial durable decisions; route questions through Claude's AUQ flow.

## Architecture Index

The repo carries four subsystems. Full details are path-scoped references; open them when working in those paths.

- Python agent harness: `agents/`, `bin/zenith`, and `.claude/rules/harness.md`.
- CALM engine: `calm/` and `.claude/rules/calm.md`.
- Rust claw-code port: `rust/`.
- Unified Single Tensor / substrate: `calm/llm_computer/`, legacy/adjacent unless reopened; references include `.claude/rules/Substrate.md`, `architecture.md`, `delta_rule.md`, `compute_facades.md`, `recursion.md`, `retrieval.md`, and `capability_gain.md`.

Serving, hardware, and VRAM details live in `.claude/rules/environment.md`. TurboQuant internals live in `.claude/rules/turboquant.md`. NIAH/model context validation lives in `.claude/rules/niah_validation.md`. Long-term commercial direction lives in `.claude/rules/commercial.md`.

## Research Pointers

- VGSL post-transformer R&D: `RESEARCH/VGSL/00_INDEX.md`.
- Substrate / cards / CHRLM vocabulary: `.claude/MEMORY/atlas/Substrate_arc.md`, `.claude/rules/Substrate.md`, `.claude/rules/delta_rule.md`.
- Mechinterp tracing and augmentation thesis: `.claude/rules/tracing_intelligence.md`, `.claude/rules/augmentation_thesis.md`, and atlas receipts.
- Distillation pipeline: `agents/distill/` and `.claude/rules/distillation.md`.

## Branch

`feature/multi-agent-qwen` on `mastergrief/zenith-code` (forked from `ultraworkers/claw-code`).
