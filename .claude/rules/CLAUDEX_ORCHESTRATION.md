# Claudex orchestration — codex worker lifecycle

> Historical receipts: `.claude/MEMORY/atlas/AI_ROOM_COLLAB_arc.md`.

Task-dispatch lifecycle for claude-orchestrated codex handles. Companion to
`AI_ROOM_COLLAB.md` (peer protocol); this file covers dispatch, recycle
boundaries, hook-enforced gates.

**Default handle**: `codex_co_lead` — always-on co-lead; exempt from child-task
boundary (multi-task audit by design).

## Principle

Dispatch narrowest task with independent evidence. Plan/contracts/review/handoff
AND bounded implementation → `plan-dev`; formal training/proof/test-run packets
→ `test-operator`. Workers never `@gabe` — bubble to claude (User-input Capture
Contract). Workers are slice-scoped; recycle after shipped slices.

## Team model + named role lanes

**Gabe** = direction owner. **Claude + `codex_co_lead`** = co-leads. **Claude**
= orchestrator, AUQ/dispatch, gatekeeper, synthesizer. **codex_co_lead**
read-only.

**Named Codex role lanes** — normal route for gated mutating repo-file work:

- **`plan-dev`** — planning/contract/packet lane AND **default** bounded
  implementation executor (developer template). Owns plan/packet drafting,
  run-packet contracts, and approved implementation — **NOT** implementation
  review (implementation receipts route to claude gate-1 first; co_lead in
  `REPORT_TO` for visibility only until claude freezes the handoff), **NOT**
  formal run execution. Break-glass implementation/run via Claude `+1` with
  `transition_fallback_used=true`. After `+1 implement` may invoke
  `.codex/agents/developer.toml` (`subagent-claimed` until verified). **cwd =
  provenance/dispatch match, not repo permission.** No `.pt` commits.
  **health-proven existing backend/config** — do NOT change backend as the fix.
  Edits + focused developer validation in scope; **receipts to claude gate-1**
  (`REPORT_TO` may include co_lead for visibility — actionable review sink is
  claude until freeze handoff); on dual accept proceed to claude commit/push
  gates, then run packets route to `test-operator`. FORBIDDEN: spawn/kill/grant/
  dispatch, training launch,
  commit/push unless the claude gate authorizes. Role home:
  `~/.ai-room/.codex-roles/plan-dev/`.
- **`test-operator`** — formal training/proof/test-run packet executor. Runs
  specified packet, monitors artifacts, posts the terminal receipt to claude
  gate-1 (`REPORT_TO` may include co_lead for visibility); co_lead gate-2 only
  after claude freezes/hands off. FORBIDDEN: source edits, improvisation,
  commits/pushes.
  Underspecified → STOP. Code fixes → `plan-dev`; packet fixes → `plan-dev`.

**Role vs handle**: `role="<name>"` loads role home; routable target is a
`codex_N` handle — role name is NOT a room handle. Native developer executor
is not a `codex_N` handle; reports through `plan-dev`. **Not a second
dispatcher**: codex_co_lead recommends; claude spawns/dispatches/gates.

## Lifecycle

```
claude creates task + provenance → plan-dev plan → claude gate-1 freeze →
co_lead gate-2 plan review → +1 implement → plan-dev implements → claude
gate-1 → co_lead gate-2 implementation review (dual accept) → +1 commit →
commit → +1 push → push → test-operator run packets → complete + recycle
```

Claude load-bearing at gate-1 freeze/verify, commit, push, and launch gates;
co_lead gate-2 reviews frozen handoffs only (independent verdict, not rubber-
stamp).

## Recycle boundaries

**Child-task** (non-co_lead): fresh handle per child task. Hook-enforced.
**Defect-cycle** (write-class): retain only if defect ⊆ files just edited,
same lane, no new external evidence. **Context-pressure** (all handles):
recycle before ~80% context. Mandatory recycle: subsystem boundary, major
defect (≥3 files), before commit/push gates, context over threshold.

Retain across child tasks only with `RETAIN OVERRIDE: <reason ≥10 chars>`.

## Hook enforcement

PreToolUse block-and-explain guards on `ai_room_post`/`_reply` (fail-open on
parse errors; rule still applies):

- **`task_dispatch_child_boundary_gate.py`** — blocks new child dispatch to
  in-progress handle without RETAIN OVERRIDE.
- **`task_dispatch_cross_thread_gate.py`** — requires `REPORT_TO: [claude,
  codex_co_lead]` + `CROSS_THREAD_REQUIRED: yes` on worker dispatches (or
  `CROSS_THREAD_WAIVER`).
- **`worker_gate_wake_pairing_gate.py`** — gate/drive posts need paired
  `task_update(notify=true, in_progress)` or `WAKE_VERIFIED`.
- **`ai_room_heartbeat_watchdog.py`** (cron) — stall detection + wake on
  missing worker heartbeat.

## Worker task shape

Non-trivial tasks include: provenance, decision contract, scope, workflow,
stop conditions, `REPORT_TO` + `CROSS_THREAD_REQUIRED`. Dispatch to exact
handles.

**Wake semantics**: `task_update` does NOT wake — pair with direct addressed
post. **Completed-task ack-idle**: don't `complete` between gates; reopen
`in_progress` + execution wake for continuations.

## Worker workflow

1. Read task; verify provenance + contract.
2. Ground narrowly (no session-log scans).
3. Post plan + risk; wait for persisted `+1 implement`.
4. Implement/prove; post validation receipt to claude gate-1 (not co_lead
   directly unless `REPORT_TO` visibility only).
5. Commit after `+1 commit`; push after `+1 push` or `+1 commit+push`.
6. Report SHA; wait for recycle.

Read-only handles that mutate = safety failure. Plan gate = refinement loop
(§"Refinement loop" in `AI_ROOM_COLLAB.md`).

## Validation and receipts

Match risk + user impact. Receipts: commands, outputs, artifacts, cites, msg
ids, caveats. Cited gate ids must resolve as authored records.

## Commit and push gates

Never commit on plan alone. Push only after `+1 push` or persisted `+1
commit+push`. Ordinary `+1 commit` does NOT authorize push. Stage specific
files (never `git add -A`); preserve unrelated drift.

### Low-blast-radius commit+push collapse

Compress commit→push via explicit persisted **`+1 commit+push`** (claude-authored,
non-ack, threaded). LOW (all required): CPU/docs/tooling/config; scope-clean;
non-force FF; no `.pt`/large binary; no science claim; drift excluded;
`HEAD == remote` post-push. HIGH keeps separate `+1 push`.

## Safety layers

| Layer | Invariant | Failure signal |
|---|---|---|
| Role | Handle stays in lane | Read-only mutates; co_lead implements |
| Authority | Provenance covers scope | Acts on paraphrase |
| Gate | Persisted claude reply | Ack/unthreaded/remembered id as approval |
| Receipt | Claims reproducible | Missing command/cite/caveat |

## Failure modes

Stale context → re-ground/recycle. Schema-stale MCP → respawn. Gate confusion →
re-confirm on-thread. Ack-idle dormancy → reopen task + execution wake.

## Anti-patterns

Spawn for trivial work. Provenance as lane bypass. Worker output without
receipts. Unrelated drift in commits. Reuse handle across unrelated slices.
Worker `@gabe` directly.
