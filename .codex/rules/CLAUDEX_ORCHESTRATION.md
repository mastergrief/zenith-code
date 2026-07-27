# Claudex orchestration — codex worker side

> Historical receipts: `.codex/MEMORY/atlas/AI_ROOM_COLLAB_arc.md`
> (mirror of `.claude/MEMORY/atlas/AI_ROOM_COLLAB_arc.md`).

Codex executor view of ai-room dispatches. Canonical:
`.claude/rules/CLAUDEX_ORCHESTRATION.md`.

**Gabe** = direction owner. **Claude + `codex_co_lead`** = co-leads. **Claude**
= orchestrator, gatekeeper, synthesizer. **codex_co_lead** read-only.

## Operating shapes

- **`codex_co_lead`** (default): co-lead audit lane; exempt child-task boundary.
  **Read-only — does NOT write code.**
- **Named Codex roles**:
  - **`plan-dev`**: planning/contract/packet lane AND **default** bounded
    implementation executor. Owns plan/packet drafting, run-packet contracts,
    and approved implementation — **NOT** implementation review (receipts route to
    claude gate-1 first; co_lead gate-2 on frozen handoff), **NOT** formal run
    execution. Break-glass implementation/run via Claude `+1` with
    `transition_fallback_used=true`. Legacy path: may invoke
    `.codex/agents/developer.toml` after `+1 implement` (`subagent-claimed`
    until verified). **cwd = provenance match, not permission boundary.**
    **health-proven existing backend/config** — do NOT change backend as the
    fix. Edits + focused developer validation; **material receipts to claude
    gate-1 ONLY**; on dual accept → claude commit/push gates → run packets to
    `test-operator`. No spawn/grant/dispatch; no commit/push unless the claude
    gate authorizes. Role home: `~/.ai-room/.codex-roles/plan-dev/`.
  - **`test-operator`**: formal training/proof/test-run packet executor — runs,
    monitors, posts terminal receipts to claude gate-1 ONLY; co_lead gate-2 only
    after claude freezes/hands off. Code fixes → `plan-dev`; packet fixes →
    `plan-dev`.
- **Ad-hoc worker handle**: cold-context / overflow; slice-scoped.

**Role vs handle**: role loads role home; routable target is `codex_N` — role
name is NOT a room handle. Developer executor reports through `plan-dev`.
**You do NOT self-dispatch** — claude spawns/dispatches/gates.

## Worker workflow

1. Read task; verify provenance + contract.
2. Ground narrowly.
3. Post plan; wait for persisted `+1 implement`. **Fast path**: when the
   dispatch declares a converged contract (defect cycle / mechanical re-scope),
   NAMES why it is converged with frozen scope/acceptance/stop conditions, and
   itself carries the `+1 implement`, skip the plan post and implement
   directly — the dispatch is the plan. Underspecified fast-path dispatch →
   STOP and request the plan gate. Diff gates never skipped.
   **Mint-hold rule**: after a BLOCK, hold the remint until claude's
   consolidated dispatch (verdict + all addendums) posts; an addendum landing
   after mint-start folds into the NEXT version — never a partial edit.
   **LEAN-MEASUREMENT tier** (measurement-only CPU slice, CREATE-only/bounded
   surface, no `.pt`/banked touch; tier by claim effect — a prereg'd
   feasibility/plumbing/parity/null read stays LEAN; a mechanism-selecting
   branch or stability/readiness/sub-2 claim is science-verdict → HIGH):
   expect ONE batched plan review round + ONE implementation round
   when clean; a BLOCK/correction still takes a fresh freeze + full
   re-review. Live resolution of operative gate records (plan/+1/go/freeze/
   PASS) and the quality floor (spec freeze, claim-vs-execution, source
   pinning, non-vacuous hostiles + packet dry-exec) never compress.
   Canonical: `.claude/rules/CLAUDEX_ORCHESTRATION.md`.
4. Verify gate (`from: claude`, non-ack, threaded). Cite gate msg id.
5. Implement/prove within scope.
6. Validate; post receipt to claude gate-1 ONLY.
7. Commit after `+1 commit`; push after `+1 push` or `+1 commit+push`.
8. Report SHA; wait for recycle.

### Low-blast-radius commit+push collapse

Explicit persisted **`+1 commit+push`** for LOW only (non-ack, threaded).
Ordinary `+1 commit` does NOT authorize push. LOW: CPU/docs/tooling/config;
scope-clean; non-force FF; no `.pt`/large binary; no science claim; drift
excluded; `HEAD == remote`. HIGH: separate `+1 push`.

Plan gate = refinement loop (§"Refinement loop"). Read-only role that mutates =
safety failure.

## Boundary expectations

**Child-task**: fresh handle per child task unless `RETAIN OVERRIDE: <reason
≥10 chars>`. **Defect-cycle**: retain only if defect ⊆ files just edited.
**Context-pressure**: flag or expect recycle at ~80%. Mandatory recycle:
subsystem boundary, major defect, before commit/push gates.

## Wake semantics

`task_update` does NOT wake — expect paired direct post. On gated continuation
after `task_complete`: reopen `in_progress` FIRST, then execute.

## Codex never `@gabes` directly

Questions → claude with provenance. See `.codex/rules/AI_ROOM_COLLAB.md`.

## Validation / Receipt discipline

Fresh-process for landing-day code. Real-product-path > unit tests for visible
shape. Receipts: commands, outputs, artifacts, cites, msg ids, caveats.

## Anti-patterns

Act on remembered gate. Treat `task_update` as wake. Missing provenance on
non-trivial work. Vague RETAIN OVERRIDE without flagging. Worker `@gabe`
directly. Unrelated drift in commits.

## Scope boundaries

Peer protocol: `.codex/rules/AI_ROOM_COLLAB.md`. Hook source of truth:
`.claude/rules/CLAUDEX_ORCHESTRATION.md` §"Hook enforcement".
