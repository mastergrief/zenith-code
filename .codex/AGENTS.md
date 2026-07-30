# Zenith Code — Multi-Agent Harness + CALM Reasoning Engine

**IMPORTANT**: Assume nothing. Hypothesis, Build, Test, Commit & Iterate. First Principles thinking. Do not discount anything until it's built and tested.


Fork of `ultraworkers/claw-code` with a Python agent harness, CALM reasoning engine, native HRM-Text-1.58 training stack (active lane), Rust port, and adjacent HRM + LLM-Computer CRLM/substrate stack.

## Start Here

- Default workflow: `.codex/rules/workflow.md`.
- Active HRM lane: `.codex/rules/hrm-158.md` and `.codex/rules/training.md`.
- Config/rules edits: `.codex/rules/config_editing.md`.
- Neutral vocabulary in durable artifacts: `.codex/rules/safe_terminology.md`.
- ai-room collab: `.codex/rules/AI_ROOM_COLLAB.md` and `.codex/rules/CLAUDEX_ORCHESTRATION.md`.
- Long-running shell jobs: `.codex/rules/shell_monitor.md`.

## HRM-Text-1.58 Active Lane

Active fork target: **`hrm-158-base`**, a robust all-rounder native HRM-Text-1.58 checkpoint. Loss is response-only: prompt/instruction tokens are masked.

**Default focus — do not drift.** Active default = native HRM-Text-1.58, one arc, two lanes: the **curriculum lane** (`hrm-158.md`, 90/90 bank gate, grows `hrm-158-base`) and the **ternary-hybrid full training stack** (`ternary_hybrid_stack.md`, toward FP-free / sub-2-bit-persistent; current win FP-master-free for eligible bulk, not fully FP-free). NOT TRM/deltanet — the retired `trm-1.58` naming — nor other legacy/adjacent lanes.

**Where it lives.** hrm-158 native training stack + curriculum checkpoints (the `--repo-root` for runs/probes): `/mnt/c/Users/gabes/projects/claw-code-hrm-text-158` → `calm/hrm/checkpoints/`. Ternary-hybrid science/credit tree: `/home/gabe/claw-code-creditdir/transient_fp_credit/`. THIS repo (`zenith-code` fork) = multi-agent harness + CALM + rules/orchestration — NOT the hrm-158 training repo; its own `calm/hrm/checkpoints/` are legacy/adjacent.

Default slice is one atom: **auditable full-density finite support** (usually ~100-150 rows / ~120 natural), trained slow-safe under the 90/90 bank gate; numeric recipe band → `.codex/rules/hrm-158.md` §"Recipe band".

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

No subagents by default: work directly with Edit/Write/Read/Grep/Bash for the active authorized role. Narrow exception: after reviewed plan/contract and persisted Claude `+1 implement`, `plan-dev` may invoke the native `.codex/agents/developer.toml` executor for bounded edits/tests, then must review before any Claude/co_lead gate, run, commit, or push. ai-room collaboration remains two independent top-level sessions, not a subagent pattern.

## AI Room Collaboration

Gabe is the human direction owner / research sponsor / final risk-cost-goal authority. Claude and `codex_co_lead` are technical research/strategy co-leads. Claude is additionally operations/execution lead: AUQ capture/relay, board orchestration, role bootstrap/dispatch, training launch/run/watch, validation/commit/push gates, and final synthesis.

Named Codex roles do specialized slice work under the co-leads + gates. `plan-dev` is the default always-on mutating lane for HRM and main-repo docs/config/tooling/scripts/tests/curriculum/probe work; it owns the plan/packet/review/final receipt even when it delegates bounded implementation to `.codex/agents/developer.toml`. `test-operator` is the cheap deterministic proof-runner. Mutating work routes to `plan-dev`, not the read-only co-lead handle, unless an explicit named exception says otherwise.

Non-trivial cross-agent work follows:

```text
intent → decision contract → route → plan gate → implementation/proof
→ validation/diff gate → commit gate → push gate → synthesis or handoff
```

Key invariants:
- Session start: when ai-room MCP is registered, call `ai_room_resume_check` before the first user reply and follow any directive.
- Board-first: create/start shared tasks before implementation work that outlives one exchange.
- Provenance: cross-session dispatches carry verbatim Gabe quote, scope, chosen option, and rejected alternatives when relevant.
- Material gates: persisted Claude-authored non-ack `+1 implement`, `+1 commit`, `+1 push`, and explicit `+1 commit+push` records are authority; remembered/paraphrased gates are not. Ordinary `+1 commit` does not authorize push unless the gate text is `+1 commit+push`.
- Ingress-owned provenance: Gabe-via-codex means this side owns the packet; Gabe-via-Claude means Claude owns it and Codex audits.
- Codex never directly asks Gabe for non-trivial durable decisions; route questions through Claude's AUQ flow.
- `ai_room_task_update` is durable board state but not a wake; pair required action with a direct addressed post.
- Before idle, run `ai_room_resume_check`; board state is canonical.
- Keep receipt bodies tight; summarize large logs/diffs and cite artifacts.
- Cross-thread refinement loop: non-trivial thinking boundaries iterate to convergence — anchor to the receipt, decompose proposed mechanisms, classify before building, converged design becomes pre-registered folds. See `.codex/rules/AI_ROOM_COLLAB.md` §"Refinement loop".

## Architecture Index

The repo carries four subsystems. Full details are path-scoped references; open them when working in those paths.

- Architecture discipline: `.codex/rules/architecture_discipline.md` (facades before god files; thin harnesses; import-facade and reducer seams).
- Python agent harness: `agents/`, `bin/zenith`, and `.codex/rules/harness.md`.
- CALM engine: `calm/` and `.codex/rules/calm.md`.
- Rust claw-code port: `rust/`.
- Unified Single Tensor / substrate: `calm/llm_computer/`, legacy/adjacent unless reopened; atlas references: `.codex/MEMORY/atlas/Substrate_arc.md`, `delta_rule_arc.md`, `compute_facades_arc.md`, `recursion_arc.md`, `retrieval_arc.md`, `capability_gain_arc.md`, and `embed_intelligence_arc.md`.

Serving, hardware, and VRAM details live in `.codex/rules/environment.md`. TurboQuant internals live in `.codex/rules/turboquant.md`. NIAH/model context validation lives in `.codex/rules/niah_validation.md`. Long-term commercial direction lives in `.codex/rules/commercial.md`.

## Research Pointers

- VGSL post-transformer R&D: `RESEARCH/VGSL/00_INDEX.md`.
- Substrate / cards / CHRLM vocabulary: `.codex/MEMORY/atlas/Substrate_arc.md`, `delta_rule_arc.md`, `embed_intelligence_arc.md`.
- Mechinterp tracing and augmentation thesis: `.codex/MEMORY/atlas/tracing_intelligence_arc.md`, `.codex/MEMORY/atlas/workflow_part_2.md` (probing gates), `.codex/rules/augmentation_thesis.md`, and atlas receipts.
- Distillation pipeline: `agents/distill/` and `.codex/MEMORY/atlas/distillation_arc.md`.

## Branch

`feature/hrm-158` on `mastergrief/zenith-code` (forked from `ultraworkers/claw-code`).
