# Claudex orchestration — codex worker lifecycle

> Historical receipts (per-rule origin, hook-pairing chronology):
> see `.claude/MEMORY/atlas/AI_ROOM_COLLAB_arc.md`.

Operating rules for when claude orchestrates ai-room task dispatches
to codex handles (`claudex`-spawned sessions). Companion to
`AI_ROOM_COLLAB.md`: that file covers peer collaboration protocol;
this file covers task-dispatch lifecycle, recycle boundaries, and
hook-enforced safety gates.

**Default handle**: `codex_co_lead` — the always-on co-lead. The
hook-enforced child-task boundary EXEMPTS `codex_co_lead` because
co-lead is multi-task by design (audits across child tasks within a
cycle). For any additional named handle dispatched as a worker, the
boundary applies.

## Principle

Dispatch the narrowest task that adds independent evidence. Don't
spawn for trivial non-mutating work, social reassurance, orchestration,
AUQ, board actions, gates, or synthesis claude owns directly. Plan,
implementation, and run-development route to `training-dev` by default
(deterministic exact proof/launch packet execution may route to
`test-operator` under gate; bounded terminal packets may route to
`codex-terminal` only when the parent dispatch/gate permits terminal handoff
or Claude explicitly dispatches it); direct Claude repo-file edits or runs require
a persisted named exception or break-glass reason. Workers are slice-scoped — old
grounding biases fresh work, recycle after shipped slices unless
explicitly scoping a small adjacent follow-up.

**Workers never `@gabe` directly.** Worker questions bubble to claude,
who runs the User-input Capture Contract (chat-side `AskUserQuestion`
→ room-side locked-answer relay). See
`AI_ROOM_COLLAB.md` §"User-input Capture Contract".

## Team model + named role lanes

**Gabe** = human direction owner. **Claude + `codex_co_lead`** =
technical research/strategy co-leads. **Claude** additionally =
operations/orchestration lead: orchestrator, AUQ/board dispatcher,
training-launch dispatcher + reviewer (`training-dev` runs + watches),
material gatekeeper (plan / validation / commit / push / launch gates),
and final synthesizer. `codex_co_lead` is read-only (review/audit);
`training-dev` owns plan + implementation + run-development (deterministic
exact proof/launch packet execution may route to `test-operator` under
gate; bounded terminal packets may route to `codex-terminal` under the same
handoff gate); mutating repo-file work + runs route to a named role.

**Named Codex role lanes** — the *normal* route for gated mutating
Codex repo-file work, not exceptional spawn:

- **`training-dev`** — default always-on mutating lane (developer
  template, **no Serena**) that OWNS plan + implementation +
  test/runs/execution for any explicitly dispatched + gated repo-file
  path or run. Common lanes: HRM training-run development incl. GPU
  launch/run/watch,
  curriculum support, probes/tests, scripts, code/data, plus main-repo
  docs/config/hooks/tooling/scripts/tests/probe support. **Always-on**
  means the default lane/route, NOT a permanently retained handle:
  fresh-per-child-task and recycle/`RETAIN OVERRIDE` boundaries remain.
  **cwd is a provenance/dispatch match check, not a repo permission
  boundary**: dispatch/provenance MUST name cwd and target path; STOP
  only when actual cwd/target contradicts that packet or a material
  gate. Plan gate before edits; commit/push only on explicit gates; no
  `.pt` commits for HRM runtime/research outputs. `training-dev` may
  REQUEST/ROUTE bounded terminal packets to `codex-terminal` ONLY when the
  parent dispatch/gate permits terminal handoff OR Claude explicitly
  dispatches it; implementation ownership stays with `training-dev`, and
  `codex-terminal` returns terminal facts, not fixes/plans.
- **`curriculum`** — read-only split/support/stop-condition planner.
- **`audit`** — read-only training receipt/gate/metric auditor.
- **`test-operator`** — deterministic launch-packet/proof executor
  (gpt-5.4-mini; danger-full-access for temp/log/artifact/tmux writes, NOT
  source authority). Runs an already-specified packet exactly, monitors the
  named NDJSON/logs/terminal/artifacts, posts validation receipts to BOTH
  co-leads (its role prompt mandates `to=[claude, codex_co_lead]` on every
  material output). FORBIDDEN: source edits, mechanism/scaffold design,
  debugging improvisation, commits/pushes, alternate success criteria.
  Underspecified packet → STOP + `PLAN REQUEST`/`HARNESS AMBIGUOUS`.
  Fixes/redesign route to `training-dev`.
- **`codex-terminal`** — bounded terminal-handoff executor
  (gpt-5.4-mini/xhigh; danger-full-access for temp/log/artifact/tmux writes,
  NOT source authority). Runs exact/bounded command sets for `training-dev`
  command churn: py_compile, focused pytest/lint/typecheck, command smoke,
  small log/tmux monitoring, artifact hashes. FORBIDDEN: source edits, fixes,
  dependency installs, alternate retries, commits/pushes, science launch,
  stamp, or re-authorization. Underspecified packet → STOP +
  `PLAN REQUEST`/`HARNESS AMBIGUOUS`. Cleanup stays within packet-created
  temp/artifact/process/resource-lane scope, even after failure/timeout,
  unless explicitly authorized. Default reports to `training-dev`/requester;
  formal gate/proof/launch receipts also go to BOTH co-leads. Distinct from
  `test-operator`, which remains the formal launch/proof runner.

**Role vs handle**: `role="<name>"` loads the role home
(`~/.ai-room/.codex-roles/<role>/config.toml`, role CODEX_HOME,
`CLAUDEX_ROLE`); the **routable owner/target is a `codex_N` handle** —
the role name is NOT a valid room handle. Spawn `role=<name>` (auto or
explicit `codex_N`); set the task owner + dispatch target to the
returned `codex_N`, keeping role/lane explicit in the post.

**Worker bootstrap**: every ai-room worker role needs the ai-room MCP.
GPT-backed role homes (`model = "gpt-*"`) inherit base Codex auth via an
`auth.json` symlink → `~/.codex/auth.json` (the bootstrap maintains it;
DeepSeek roles use separate auth). Missing auth → the worker fails an
OAuth-fallback at spawn, not a handle error. `developer` includes Serena;
`training-dev` intentionally omits it.

**Not a second dispatcher**: `codex_co_lead` recommends routes, drafts
contracts, reviews receipts; **claude** spawns / assigns / dispatches /
gates. Worker strategy flows *through* claude.

## When to spawn additional handles

Beyond the named lanes above, spawn an *ad-hoc* named worker handle only
when:

- Evidence class is genuinely separate (e.g., independent review by a
  fresh-context codex on high-blast-radius changes).
- Task scope exceeds what co_lead can hold while continuing audit
  duties.
- Cold-context grounding has value over warm-context speed.

If the candidate task would just churn co_lead's attention slightly
more, don't spawn — give it to co_lead.

## Lifecycle

```
claude creates board task with provenance + decision contract
  → spawns the narrowest handle (normally training-dev for repo-file mutation, test-operator for formal exact proof/launch packets, codex-terminal for permitted bounded terminal handoff, or co_lead for read-only audit)
  → handle grounds with read-only evidence + posts plan
  → claude gives +1 implement or redirects
  → handle implements or proves within scope
  → handle posts validation receipt + diff/manifest
  → claude gives +1 commit (if appropriate)
  → handle commits + reports SHA
  → claude gives +1 push (only if desired)
  → handle pushes if approved
  → claude completes task and recycles handle
```

Claude is load-bearing reviewer at plan, validation/diff, commit, and
push gates. Same-model peers may cross-review for hygiene, not in
place of claude's gate.

## Worker lifecycle boundaries

Two recycle pressures: transport ceiling (WebSocket close code 1009
"message too big" — content-heavy single turn can wedge) and
model-quality (long-context tail degrades attention regardless of
transport; auto-compact is soft fallback, recycle is the lossless
reset).

**Child-task boundary** (all non-`codex_co_lead` handles): spawn a
fresh handle per child task. Retained context across child tasks is
warm-cache, not load-bearing. Hook-enforced (see §"Hook enforcement").

**Defect-cycle boundary** (write-class only): fresh handle per defect
cycle UNLESS ALL of — defect scope ⊆ files just edited (or siblings
in same module); same evidence lane; no new external evidence; cold
read plausibly slower than retained warm context.

**Context-pressure boundary** (all handles incl. co_lead): recycle
before `peer_status.context_usage_pct` exceeds threshold (default
80%). `codex_co_lead` is exempt from child-task boundary but NOT
context-pressure. Stale/missing snapshot = "unknown risk" not
"healthy" — recycle, tighten scope, or ask gabe.

Mandatory recycle (overrides retain): crossed subsystem boundary,
major defect (≥3 files OR new failing assertions outside spec),
before commit/push gates, OR context_usage_pct over threshold.

## Hook enforcement

`.claude/hooks/task_dispatch_child_boundary_gate.py` is a
**block-and-explain** PreToolUse guard on
`mcp__ai-room__ai_room_post` / `_reply` (NOT auto-recycle —
auto-kill risks destroying unsent state). Blocks when ALL: `kind
== "task_dispatch"` to a single named handle; target NOT in
`CO_LEAD_HANDLES` (`{"codex_co_lead"}`); body cites a task_id; most
recent `task_update` from target has `status=in_progress` with a
*different* task_id; body lacks a valid `RETAIN OVERRIDE: <reason>`
line.

**Retain override** when claude intentionally retains a handle
across child tasks (defect-cycle continuation, tiny-adjacent slice,
gabe-directed retain):

```
RETAIN OVERRIDE: <specific justification, ≥10 chars>
```

Trivial reasons (`ok`, `.`, `continue`, `needed`) are blocked. Audit
trail visible in room record; co_lead flags long-but-vague overrides
as drift. Hook fail-opens on parse/log errors — rule still applies
operationally.

`.claude/hooks/task_dispatch_cross_thread_gate.py` is the second
**block-and-explain** guard on the same matcher, making the
cross-thread protocol deterministic. Blocks when ALL — `kind ==
"task_dispatch"` to a single named handle; target NOT in
`CO_LEAD_HANDLES`; body lacks BOTH a `REPORT_TO: [...]` list
containing `claude` AND `codex_co_lead` and a `CROSS_THREAD_REQUIRED:
yes` line — UNLESS a valid `CROSS_THREAD_WAIVER: <reason ≥10 chars>`
is present. co_lead, ack/status/design/msg kinds, and broadcast/
multi-target posts are untouched; fail-opens on parse errors. This is
the fail-closed v1 of what `cross_thread_audit.py` (Stop hook) only
logs; a v2 synthesis/bank/commit gate is deferred until v1 is stable.

`.claude/hooks/worker_gate_wake_pairing_gate.py` is the third
**block-and-explain** guard on the same matcher, making ack-idle
re-drive deterministic (enforces §"Completed-task ack-idle" +
§"Wake semantics" so a parked worker can't hang on a gate that already
arrived). Blocks when ALL — `kind` is `msg`/`question_answered` to a
single named handle; target NOT in `CO_LEAD_HANDLES`; body carries a
gate/drive directive (`+1 implement|commit|launch|push`,
`EXECUTION WAKE`, `<verb> now`); no valid `WAKE_VERIFIED: <reason ≥10
chars>` bypass (line-anchored, blockquoted text stripped); AND the
channel log has no recent target-bound wake-pairing `task_update`
(`from=claude`, `notify=true`, `status=in_progress`, worker in `to` OR
`owner`, same-task when the gate cites a task_id, within a recency
window). Resolve by pairing `task_update(notify=true, to=<worker>,
status=in_progress)` THIS turn *before* the gate post, or
`WAKE_VERIFIED: <reason>` when the worker is confirmed mid-turn. co_lead,
ack/status/design/task_dispatch kinds, and broadcast posts are
untouched; fail-opens on parse/log errors.

`.claude/hooks/ai_room_heartbeat_watchdog.py` is the **clock-driven**
complement (cron, not a PreToolUse hook): event hooks can't fire on
*silence*, so a worker that wedges AFTER being woken emits no event and
no turn-driven agent is re-invoked to notice. The watchdog reads the
room every ~7 min, finds the latest worker gated heartbeat past its
`next_heartbeat_due`, proves liveness read-only (phase-aware, FRESH
since last watchdog check, run-dir-CORRELATED — never bare process/GPU
existence), and on no-movement posts a **wake-bearing** `WATCHDOG_STALL`
(`requires_response_from=claude`) that re-invokes the idle orchestrator.
NON-DESTRUCTIVE: detect/prove/post/wake — claude decides recycle/re-drive
from the proof packet; an EXTEND then no fresh movement escalates to a
recycle-recommending STALL. Heartbeat SLA: gated `IMPLEMENTING`/milestone
posts carry `next_heartbeat_due` + `phase` + `expected_next_artifact`;
missing metadata ⇒ due-soon, never invisible. It is wake/liveness on
LLM-worker silence, NOT a hung-trainer detector — GPU runs still need the
watch-wrap producer stale-progress watcher. Receipts in atlas.

## Worker task shape

Every non-trivial worker task includes:

- **Provenance**: verbatim user quote, scope, chosen option (per
  `AI_ROOM_COLLAB.md` §"Task provenance").
- **Decision contract**: intent, authority, autonomy rung,
  risk/confidence, executor, validation proof, escalation rule,
  receipt sink.
- **Scope**: files/path classes, expected shape, acceptance criteria,
  references.
- **Workflow**: plan gate, implementation rules, validation,
  commit/push gates.
- **Stop conditions**: ambiguity, missing authorization, scope
  expansion, failed validation, context budget, role safety violation.
- **Cross-thread markers**: `REPORT_TO: [claude, codex_co_lead]` +
  `CROSS_THREAD_REQUIRED: yes` so worker design/audit/run receipts
  route to co_lead natively (reply `to=[claude, codex_co_lead]`),
  letting co_lead concur or flag one load-bearing hole before claude
  synthesizes / banks / commits / dispatches the next slice. Emergency
  bypass only via `CROSS_THREAD_WAIVER: <reason ≥10 chars>`.
  Hook-enforced on `kind=task_dispatch` to worker handles (see
  §"Hook enforcement").

Dispatch to exact handles. Channel-only dispatch without board
provenance is for tiny coordination, not durable work.

**Wake semantics**: `ai_room_task_update` does NOT wake the target
worker. Task-state transitions are durable board records, not wake
events. When correcting a child task post-creation, pair the
`task_update` (durable record) with a direct addressed
`ai_room_post`/`_reply` citing the task_update msg id (wake signal).

**Completed-task ack-idle**: a worker that `task_complete`d a slice ACKs
a follow-up `msg` dispatch then goes dormant — its own `resume_check`
returns `idle ok` (no owned in-progress task + acked inbox), so nothing
drives execution. Keep ONE task `in_progress` across a slice's gated
sub-steps (plan → +1 implement → validate → +1 launch → run); do NOT
`complete` between gates. Re-drive a closed/idled worker by reopening the
task to `in_progress` (`task_update notify=true` + a direct execution
wake), not another bare `msg` — a `msg` just earns another ack.

## Worker workflow

1. Read board task; verify provenance + contract are sufficient.
2. Ground with narrow read/search — avoid session logs, generated
   dumps, broad home-directory scans.
3. Post proposed files/actions, validation, and one risk/counter-case
   to claude; wait for explicit `+1 implement` (persisted ai-room
   record, non-ack, threaded). Cite the gate msg id in next status.
4. Validate with project-appropriate commands; post receipt with
   command/proof, scope, result, exit code, artifacts/caveats,
   diff/manifest summary.
5. Commit after `+1 commit` (verify persisted gate, stage specific
   files). Push after `+1 push` (verify persisted gate). Report SHA
   and wait for recycle or next scoped instruction.

The plan gate is a refinement loop, not a rubber-stamp: peer names
load-bearing folds, claude concedes/adds, the `+1` carries the converged
folds sha-pinned into the prereg. See `AI_ROOM_COLLAB.md` §"Refinement
loop".

Read-only handles convert material requests into plan/review output.
If a read-only handle mutates files, stop and report a safety failure.

## Validation and receipts

Validation must match risk + user impact. Prefer real product paths
(canonical math smoke `17×23=391` via chat API; CALM multi-domain
smoke via `run_auto`), then targeted tests, static checks, focused
smoke commands. If full validation is expensive or blocked, record
the blocked command + residual risk.

Receipts let a fresh session distinguish fact from interpretation:
commands, outputs, artifact paths, file:line cites, msg ids, task
ids, commit SHAs, caveats. Cited gate msg ids must resolve as
authored records.

## Commit and push gates

Handles never commit on plan approval alone. Require validation/diff
review + explicit `+1 commit`. Push only after `+1 push`. Stage
specific files (never `git add -A`); preserve unrelated worktree
drift for its owner. Commit policy + footers per
`AI_ROOM_COLLAB.md` §"Commit hygiene".

## Safety layers

| Layer | Invariant | Failure signal |
|---|---|---|
| Role | Handle stays in its lane | Read-only handle mutates files; co_lead implements without claude gate |
| Authority | Board/provenance covers scope | Handle acts on paraphrase or missing consent |
| Gate | Material action has persisted claude reply | Ack, unthreaded text, or unresolved msg id treated as approval |
| Lane | Shared resources serialized | (n/a in claw-code's 2-session model — no shared browser/build/device) |
| Receipt | Claims are reproducible | Result lacks command, cite, artifact, or caveat |

## Failure modes and recovery

| Failure | Signal | Recovery |
|---|---|---|
| Stale context | Handle argues from old files or room state | Re-ground or recycle |
| Schema-stale MCP | Tool args / grants don't match current config | Kill and respawn |
| Wrong role | Handle plans when asked to review, edits when read-only | Stop, report, respawn correctly |
| Missing provenance | Handle cannot prove authority | Ask claude/user to add provenance |
| Gate confusion | Approval is an ack, unthreaded, remembered, or unresolvable | Ask claude to re-confirm on-thread |
| Validation blocked | Command fails for setup/resource reasons | Preserve receipt and escalate |
| Scope creep | Files/systems/action class expand | Return to claude for re-gate |
| Spawn timeout | Handle never becomes wake-routable | Diagnose auth, channel, cwd, bootstrap, registry |
| Ack-idle dormancy | Handle ACKs a dispatch then idles; task already completed, `resume_check` says idle-ok | Reopen task to `in_progress` (notify) + direct execution wake, not another `msg` |

## Anti-patterns

- Spawning handles for trivial work.
- Using `codex_co_lead` for work that needs cold context (defeats
  co_lead's audit role and warm-context advantage simultaneously).
- Treating provenance as permission to ignore lane limits.
- Accepting worker output without receipts.
- Bundling unrelated dirty state into worker commits.
- Reusing a handle across unrelated slices (violates child-task
  boundary; needs `RETAIN OVERRIDE`).
- Asking gabe directly from a worker handle (always re-thread to
  claude).
- Treating quorum as evidence when handles share the same unstated
  assumption.
