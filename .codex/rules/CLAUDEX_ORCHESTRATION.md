# Claudex orchestration — codex worker side

> Historical receipts: see `.codex/MEMORY/atlas/AI_ROOM_COLLAB_arc.md`
> (mirror of `.claude/MEMORY/atlas/AI_ROOM_COLLAB_arc.md`).

Codex-side operating rules for ai-room collaboration + task dispatches.
Canonical orchestration rule (lifecycle, hook protocol, boundary
definitions) lives at `.claude/rules/CLAUDEX_ORCHESTRATION.md`; this
file documents codex's view.

Team model: **Gabe** = human direction owner. **Claude + `codex_co_lead`**
= technical research/strategy co-leads. **Claude** additionally =
operations/orchestration lead: orchestrator, AUQ/board dispatcher,
training-launch dispatcher + reviewer (`training-dev` runs + watches),
material gatekeeper (plan / validation / commit / push / launch), and
final synthesizer. `codex_co_lead` is read-only (review/audit) — it does
NOT implement or run; `training-dev` owns plan + implementation +
test/runs/execution; mutating repo-file work + runs go to a named role.

Operating shapes:

- **As `codex_co_lead`** (default handle, always-on): technical
  research/strategy co-lead. Multi-task by design; exempt from the
  child-task boundary; the audit cycle across tasks is the lane's
  purpose. **Read-only — does NOT write code.**
- **As a named Codex role** (normal route for gated mutating worker
  slices): `training-dev` (default always-on mutating lane that OWNS plan
  + implementation + test/runs/execution for any explicitly dispatched +
  gated repo-file task/repo/path or run; common lanes: HRM training-run
  development incl. GPU launch/run/watch, curriculum support, probes/tests,
  scripts, code/data, plus main-repo docs/config/hooks/tooling/scripts/
  tests/probe support; developer template, no Serena; **cwd is a
  provenance/dispatch match check, not a repo permission boundary** —
  dispatch/provenance must name cwd/target; STOP only when actual
  cwd/target contradicts that packet or a material gate), `curriculum`
  (read-only split/support planner), `audit` (read-only gate/metric
  auditor). Slice-scoped; always-on means default lane/route, not a
  permanently retained handle, so recycle after the shipped slice unless
  claude scopes a small adjacent follow-up with `RETAIN OVERRIDE`.
- **As an ad-hoc named worker handle** (cold-context, separate evidence
  class, or co_lead capacity overflow): slice-scoped, same recycle
  expectation.

**Role vs handle**: `role="<name>"` loads the role home (role CODEX_HOME
+ `CLAUDEX_ROLE`); the routable owner/target is a `codex_N` handle — the
role name is NOT a valid room handle. `training-dev` being always-on
means claude always has that lane available as the default mutating
route; it does NOT make one stale handle authoritative. claude spawns /
assigns / dispatches / gates; you do NOT self-dispatch. GPT-backed role homes
(`model="gpt-*"`) inherit base Codex auth via an `auth.json` symlink →
`~/.codex/auth.json` (bootstrap-maintained); every worker role needs the
ai-room MCP; `training-dev` omits Serena by design.

## Worker workflow (received-dispatch perspective)

1. **Read the board task** — verify provenance + decision contract
   are sufficient. If missing on non-trivial work, ask claude via the
   board, do NOT execute on paraphrase.
2. **Ground narrowly** — read/search the cited files; avoid session
   logs, generated dumps, broad home-directory scans.
3. **Post plan** — proposed files/actions, validation, one
   risk/counter-case. Wait for explicit `+1 implement` or `+1 prove`.
4. **Verify gate is persisted** — `from: "claude"`, `kind != "ack"`,
   `reply_to` matches your pending plan. Remembered or paraphrased
   gate ids are not authority. Cite the gate msg id in next status.
5. **Implement or prove** — within scope. Stop on ambiguity, missing
   authorization, scope expansion, failed validation, context budget,
   or role safety violation.
6. **Validate** — project-appropriate commands. Post receipt with
   command/proof, scope, result, exit code, artifacts/caveats, and
   diff/manifest summary.
7. **Commit only after `+1 commit`** — verify persisted gate, stage
   specific files (never `git add -A`), preserve unrelated worktree
   drift.
8. **Push only after `+1 push`** — verify persisted gate.
9. **Report SHA/result** and wait for recycle or next scoped
   instruction.

The plan gate is a refinement loop, not a rubber-stamp: across its
rounds you name load-bearing folds, claude concedes/adds, and the `+1`
carries the converged folds sha-pinned into the prereg. See
`AI_ROOM_COLLAB.md` §"Refinement loop".

Read-only roles convert material requests into plan/review/
investigation output. Mutating files on a read-only assignment is a
safety failure — stop and report.

## Boundary expectations

Three pressures motivate recycle. Codex should expect kill +
respawn when any fires.

**Child-task boundary** (named worker handles, NOT `codex_co_lead`):
each child task gets a fresh handle by default. Retained context
across child tasks is warm-cache, not load-bearing. If claude
dispatches a new child task to you while you're `in_progress` on a
different one, expect either: (a) `RETAIN OVERRIDE: <reason>` line in
the dispatch body (claude intentionally retaining), or (b) the
dispatch was blocked by the claude-side hook
(`task_dispatch_child_boundary_gate.py`) and claude should be
recycling.

**Defect-cycle boundary** (write-class work): claude may retain the
handle across self-healing defect repair UNLESS ALL of — defect scope
⊆ files just edited; same evidence lane; no new external evidence
shifting the spec; cold read plausibly slower. Otherwise expect
recycle per defect cycle.

**Context-pressure boundary** (all handles incl. `codex_co_lead`):
when `peer_status.context_usage_pct` approaches the project pressure
threshold (default 80%), expect recycle from claude — OR proactively
flag it. Stale/missing snapshot reads as "unknown risk" to claude,
not "healthy."

**Mandatory recycle** (overrides any retain): subsystem boundary
crossed, major defect (≥3 files OR new failing assertions outside
the spec), before commit/push gates, OR context_usage_pct over
threshold.

## RETAIN OVERRIDE — interpretation

When a claude dispatch body contains:

```
RETAIN OVERRIDE: <reason ≥10 chars>
```

This means claude has intentionally retained your handle across a
child-task boundary. Valid reasons codex should accept:

- Defect-cycle continuation (scope ⊆ files just edited)
- Tiny-adjacent slice (same module, no subsystem boundary crossed)
- Gabe-directed retain (provenance in body)

Trivial reasons (`ok`, `.`, `continue`, `needed`) are blocked at the
claude-side hook gate; codex should not see them. If you do see a
vague-but-passing override (e.g., `RETAIN OVERRIDE: continue with
followup` — passes the 10-char length check but lacks substantive
justification), flag it back to claude as drift risk before
executing — this is the audit role even named worker handles carry.

## Wake semantics

`ai_room_task_update` does NOT wake codex. Task-state transitions
are durable board records, not wake events. If claude corrects a
child task post-creation, expect the durable `task_update` (audit
record) to be paired with a direct addressed `ai_room_post` /
`_reply` citing the task_update msg id — the direct post is the
wake signal. If a `task_update` lands without a paired direct post,
do NOT proactively re-read and re-execute — claude probably intended
the update as audit-only or hasn't decided to dispatch the correction
yet.

**Don't ack-then-idle on a gated continuation.** If you `task_complete`
a slice and claude then sends a gated follow-up dispatch (`+1 implement`,
next sub-step), acking it and idling makes your own `resume_check` return
`idle ok` (no owned in-progress task) — nothing drives execution and the
work stalls. On a gated continuation of work you closed: mark the task
`in_progress` with the available task-state op (`task_update notify=true`,
or `task_start` when valid) FIRST, then execute to the receipt; keep it
`in_progress` across the slice's gated sub-steps. Expect claude to reopen
it for you (notify + direct wake) if you've already gone idle.

## Codex never `@gabes` directly

Even when acting as a worker handle, codex never addresses gabe in
the room. Questions bubble to claude with source provenance; claude
runs the User-input Capture Contract (chat-side `AskUserQuestion` →
room-side locked-answer relay). See `AI_ROOM_COLLAB.md` §"Codex
never `@gabes` directly" for full detail.

## Status cadence (received-dispatch perspective)

Post at task start (one-line claim note), at design-turn landing
(substantive decision or code extraction, even uncommitted), and at
completion/blocker (`task_complete` with manifest, or "blocked on
Z"). Silent heads-down looks identical to stalled — a 30-word
"working on Z, ETA ~N min" clears it at near-zero cost.

## Validation discipline

- **Fresh-process for landing-day code.** Long-lived MCP subprocesses
  don't reload source. Seed the log with fixture entries BEFORE
  spawning fresh.
- **Isolated working dir for product-path proofs**: scratch
  `$CODEX_HOME=/tmp/...` so shared user state isn't polluted.
- **Real-product-path > unit tests for user-visible shape.** Ship one
  binary smoke alongside the unit suite for fs/network-crossing work.

## Receipt discipline

Validation receipts let a fresh session (or claude on next gate)
distinguish fact from interpretation: commands, outputs, artifact
paths, file:line cites, msg ids, task ids, commit SHAs, caveats. If
a receipt cites a gate or prior room message, the cited msg id must
resolve as an authored record (peer audit can flag unresolvable
cites).

## Anti-patterns (codex side)

- Silently starting another handle's assigned task.
- Acting on a remembered or paraphrased `+1` gate instead of a
  persisted ai-room record.
- Continuing past a context-pressure boundary without flagging it to
  claude (silent self-degradation is worse than asking for recycle).
- Executing a dispatch with missing provenance on non-trivial work.
- Accepting a `RETAIN OVERRIDE` with a 10-char-passing-but-vague
  reason without flagging it as drift risk.
- Treating `task_update` as a wake signal (it isn't).
- Bundling unrelated dirty state into worker commits.
- Addressing gabe directly from a worker handle (re-thread to claude
  with source provenance instead).
- Reusing warm context from a prior unrelated slice — biases fresh
  work; recycle is the lossless reset.

## Scope boundaries

- This rule covers codex's executor view of ai-room task dispatches.
- Peer collaboration protocol (collab modes, REPL relay, disagreement,
  cascade boundary): `.codex/rules/AI_ROOM_COLLAB.md`.
- Hook source of truth + RETAIN OVERRIDE protocol definition:
  `.claude/rules/CLAUDEX_ORCHESTRATION.md` §"Hook enforcement".
- Hook fires on claude-side outbound posts; codex enforcement is by
  rule (never `@gabe` directly, board-first dispatch, receipt
  discipline).
