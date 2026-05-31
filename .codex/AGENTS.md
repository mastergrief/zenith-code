# Zenith Code — Multi-Agent Harness + CALM Reasoning Engine

**IMPORTANT**: Assume nothing. Hypothesis, Build, Test, Commit & Iterate. First Principles thinking. Do not discount anything until it's built and tested.

Use `apply_patch` for small/manual file edits when available. Use `python3` or Serena semantic edit tools for bulk, generated, or semantic edits. If the preferred edit path is blocked or impractical, state the blocker before using the fallback.

Fork of `ultraworkers/claw-code` with a Python agent harness, CALM reasoning engine, native HRM-Text-1.58 training stack (active lane), Rust port, and adjacent HRM + LLM-Computer CRLM/substrate stack.

## Start Here

- Default workflow: `.codex/rules/workflow.md`.
- Active HRM lane: `.codex/rules/hrm-158.md` and `.codex/rules/training.md`.
- Config/rules edits: `.codex/rules/config_editing.md`.
- ai-room collab: `.codex/rules/AI_ROOM_COLLAB.md` and `.codex/rules/CLAUDEX_ORCHESTRATION.md`.
- Long-running shell jobs: `.codex/rules/shell_monitor.md`.

## HRM-Text-1.58 Active Lane

Active fork target: **`hrm-158-base`**, a robust all-rounder native HRM-Text-1.58 checkpoint. Loss is response-only: prompt/instruction tokens are masked.

Default slice is one atom: **auditable full-density finite support** (usually ~100-150 rows / ~120 natural), trained slow-safe with LR ~5e-5, replay .80, n-train 12000, heldout/eval 200 diagnostic unless promoted, seeds 17/17, pc/temp 1.0, fixed Tier-B, saves 250..1500.

Bank gate is acquire ≥90% / retain ≥90% per slice. Bank earliest all-clear save; final has no privilege. Close siblings clear by numeric gate OR no-new-broad-cluster/parent-floor. Retention uses explicit replay + parent consistency + broad retained supports (L0b, math_a0) + direct close-sibling protection for shared template/emission surfaces; extra run-specific KL pins must be launch-command entries with ENABLED count/hash proof, never default-on and never on the target. Anchors are sentinels, not the primary retention mechanism.

On a miss: classify, then split smaller / protect / redesign. Do **not** stretch runway, bump LR, or add model capacity to force a fragile slice through. Cached/batched probe path is default; native ternary train is preferred when available. `.pt` artifacts are runtime/research outputs; commit code/tooling/docs/manifest receipts, not `.pt` by default.

Procedure targets use shaped computation supervision when answer-only rows memorize: `input/question → planning → reasoning → computing → answer/output`, with planning/reasoning/computing as temporary curriculum objects that fade only after held recombination and answer-only transfer clear.

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

No subagents by default: work directly with Edit/Write/Read/Grep/Bash for the active authorized role. In ai-room collaboration, mutating repo-file work defaults to gated `training-dev`; direct tools for co-lead/Claude stay scoped to orchestration, AUQ, board dispatch, training launch/watch, gates, synthesis, trivial non-mutating work, or explicit named exceptions. If a multi-step workflow is genuinely needed, structure it as sequential hypothesis → build → test → commit rounds. ai-room collaboration is not a subagent pattern; it is two independent top-level sessions exchanging structured messages.

## AI Room Collaboration

Gabe is the human direction owner / research sponsor / final risk-cost-goal authority. Claude and `codex_co_lead` are technical research/strategy co-leads. Claude is additionally operations/execution lead: AUQ capture/relay, board orchestration, role bootstrap/dispatch, training launch/run/watch, validation/commit/push gates, and final synthesis.

Named Codex roles do specialized slice work under the co-leads + gates. `training-dev` is the default always-on mutating lane for HRM and main-repo docs/config/tooling/scripts/tests/curriculum/probe repo-file work; `curriculum` is read-only planning; `audit` is read-only gate/metric review. Mutating work routes to `training-dev`, not the read-only co-lead handle, unless an explicit named exception says otherwise; always-on means lane/default route, not a permanently retained handle.

Non-trivial cross-agent work follows:

```text
intent → decision contract → route → plan gate → implementation/proof
→ validation/diff gate → commit gate → push gate → synthesis or handoff
```

Key invariants:
- Session start: when ai-room MCP is registered, call `ai_room_resume_check` before the first user reply and follow any directive.
- Board-first: create/start shared tasks before implementation work that outlives one exchange.
- Provenance: cross-session dispatches carry verbatim Gabe quote, scope, chosen option, and rejected alternatives when relevant.
- Material gates: persisted Claude-authored non-ack `+1 implement`, `+1 commit`, and `+1 push` records are authority; remembered/paraphrased gates are not.
- Ingress-owned provenance: Gabe-via-codex means this side owns the packet; Gabe-via-Claude means Claude owns it and Codex audits.
- Codex never directly asks Gabe for non-trivial durable decisions; route questions through Claude's AUQ flow.
- `ai_room_task_update` is durable board state but not a wake; pair required action with a direct addressed post.
- Before idle, run `ai_room_resume_check`; board state is canonical.
- Keep receipt bodies tight; summarize large logs/diffs and cite artifacts.
- Cross-thread refinement loop: non-trivial thinking boundaries iterate to convergence — anchor to the receipt, decompose proposed mechanisms, classify before building, converged design becomes pre-registered folds. See `.codex/rules/AI_ROOM_COLLAB.md` §"Refinement loop".

## Architecture Index

The repo carries four subsystems. Full details are path-scoped references; open them when working in those paths.

- Python agent harness: `agents/`, `bin/zenith`, and `.codex/rules/harness.md`.
- CALM engine: `calm/` and `.codex/rules/calm.md`.
- Rust claw-code port: `rust/`.
- Unified Single Tensor / substrate: `calm/llm_computer/`, legacy/adjacent unless reopened; references include `.codex/rules/Substrate.md`, `architecture.md`, `delta_rule.md`, `compute_facades.md`, `recursion.md`, `retrieval.md`, and `capability_gain.md`.

Serving, hardware, and VRAM details live in `.codex/rules/environment.md`. TurboQuant internals live in `.codex/rules/turboquant.md`. NIAH/model context validation lives in `.codex/rules/niah_validation.md`. Long-term commercial direction lives in `.codex/rules/commercial.md`.

## Research Pointers

- VGSL post-transformer R&D: `RESEARCH/VGSL/00_INDEX.md`.
- Substrate / cards / CHRLM vocabulary: `.codex/MEMORY/atlas/Substrate_arc.md`, `.codex/rules/Substrate.md`, `.codex/rules/delta_rule.md`.
- Mechinterp tracing and augmentation thesis: `.codex/rules/tracing_intelligence.md`, `.codex/rules/augmentation_thesis.md`, and atlas receipts.
- Distillation pipeline: `agents/distill/` and `.codex/rules/distillation.md`.

## Branch

`feature/multi-agent-qwen` on `mastergrief/zenith-code` (forked from `ultraworkers/claw-code`).
