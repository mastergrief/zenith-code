# Agent Teams Rules

**Project default: lead-orchestrator + one builder-worker per task.** See
`../CLAUDE.md` §"Working policy". This rule documents the pattern in
detail. Full teams (N>1 workers) require explicit user authorization.

## Lead-orchestrator pattern (default)

The default loop for non-trivial work:

1. **Lead states hypothesis + deliverable spec.** Must include: file paths
   to read, exact deliverables (paths + line-count bounds), measurement gate,
   constraints (strictly additive, no commits, no Gemma run, etc.), and
   "what NOT to do" list.
2. **`TaskCreate` for the unit of work.** Mandatory — every agent spawn
   requires a corresponding task row. The task is what makes in-flight
   work visible to the user and gives the worker a handle to update.
   Skipping this is a silent violation of the lead-orchestrator contract.
3. **Spawn one worker** (`Agent(subagent_type="general-purpose",
   team_name=T, name="builder", run_in_background=True, prompt=<brief>)`).
   Immediately `TaskUpdate({taskId, owner: <worker-name>, status:
   "in_progress"})` so the task list reflects live ownership.
4. **Worker reads, builds, self-tests within its capability**, reports
   back via `SendMessage({to: "team-lead", ...})`.
5. **Lead reviews diff directly** (never trust worker's self-report —
   read the actual files changed). Commits if solid. Sends corrections
   via SendMessage if not. `TaskUpdate` to `completed` when shipped.
6. **Lead reports to user** + takes direction for next iteration.
7. **After 2-3 iterations** on the same task, spawn fresh worker
   with tightened spec. One worker carries a task through 2-3
   revision cycles; if it hasn't converged by then the spec is
   under-defined, not the worker incompetent.

**What the lead owns** (across all rotations):
- Conversational context with user
- Architectural decisions + trade-off rationale
- Commit messages with before/after tables
- Cross-round synthesis (what was ruled out, why)
- `.claude/` rules + MEMORY files

**What the worker owns** (per spawn, discarded on rotation):
- Reading project files (saves lead context-window)
- Writing implementation
- Self-testing within its capability (unit tests, parse checks, not GPU
  forwards unless lead explicitly assigns)
- Synthesis report via SendMessage

**What persists across rotations**:
- Git commits (lead commits between iterations → new worker reads commits
  to get state)
- TaskList entries (shared team file)
- Uncommitted files on disk (new worker can read + continue)
- Lead's in-session synthesis

## When teams ARE appropriate (N>1 workers)

- User explicitly asks for "a team", "parallel work", etc.
- Work partitions cleanly into disjoint file sets (two agents editing
  overlapping files will race at commit time).
- Research/analysis that needs multiple independent reads of large
  subtrees (protects main-context window).

## When teams are NOT appropriate

- Tightly sequential tasks with data dependencies between steps —
  serialize yourself; teammates can't share state mid-turn.
- Quick one-off edits, typo fixes, single-file changes — lead does
  directly, agent overhead isn't worth it.
- GPU-heavy probing rounds where `bin/gemma-run` daemon is running —
  lead orchestrates daemon scripts directly; worker can't hold daemon
  state.

## Tool Inventory

### Team management

| Tool | Purpose |
|---|---|
| `TeamCreate` | Creates team + shared task list at `~/.claude/teams/{name}/` and `~/.claude/tasks/{name}/` (1:1). Args: `team_name`, optional `description`, `agent_type` |
| `Agent` | Spawns a teammate. `team_name` + `name` params join them to the team. Named agents are resumable via SendMessage |
| `SendMessage` | ONLY way to talk to teammates. `{to: name, summary, message}`. Plain text output is invisible to other agents. `to: "*"` broadcasts (linear cost — use sparingly) |
| `TeamDelete` | Tears down; fails if members still active. Shut down teammates first |

### Task coordination (the shared surface)

| Tool | Purpose |
|---|---|
| `TaskCreate` | `{subject, description, activeForm?, metadata?}`. Always `pending`, no owner. Check TaskList first to avoid dupes |
| `TaskList` | Summary view. Claim `pending` tasks with empty owner + empty blockedBy, lowest ID first |
| `TaskGet` | Full detail including blocks/blockedBy. Always call before starting (tasks go stale) |
| `TaskUpdate` | Status (`pending → in_progress → completed`, or `deleted`), owner, addBlocks/addBlockedBy for DAGs. Only mark completed when **fully** done |
| `TaskOutput` / `TaskStop` | For **background task IDs** (bash `run_in_background`, async agents), NOT task-list IDs. Same word, different systems. For `local_agent` tasks NEVER Read the `.output` file — it's a JSONL transcript symlink that blows up context |

### Adjacent tools

- **`EnterWorktree` / `ExitWorktree`** — session-level worktree
  isolation. Only when user explicitly asks. `ExitWorktree` is a no-op
  if you didn't enter via the tool. `remove` action with uncommitted
  work requires `discard_changes: true`.
- **`AskUserQuestion`** — 1-4 questions, 2-4 options each. Optional
  `multiSelect`, optional markdown `preview` (single-select only).
  Users always get free-form "Other" fallback. In plan mode: clarify
  BEFORE finalizing; never for "is my plan ready?" (that's
  `ExitPlanMode`).
- **`EnterPlanMode` / `ExitPlanMode`** — plan → approval →
  implement flow. Required user consent. For non-trivial multi-file
  work.
- **`PushNotification`** — desktop + phone if Remote Control connected.
  ≤200 chars, one line, no markdown. Interruptive — reserve for "they
  walked away and should come back" moments. Don't use for routine
  progress.

## Agent spawning — key parameters

| Param | Effect |
|---|---|
| `subagent_type` | Gates tool access. **Explore/Plan/claude-code-guide are read-only** — never assign implementation work regardless of mode |
| `name` | Required for `SendMessage({to: name})` resumability. Without a name you can't address the agent again |
| `team_name` | Joins existing team (uses current team context if omitted) |
| `model` | `sonnet` / `opus` / `haiku`. Force specific model; else inherits from definition → parent |
| `mode` | `plan` / `acceptEdits` / `bypassPermissions` / `default` / `dontAsk` / `auto`. Overrides sub-agent permission mode |
| `isolation: "worktree"` | Per-agent temp git worktree. Auto-cleans on no-op; returns path + branch otherwise |
| `run_in_background: true` | Completion notification delivered automatically. **Do NOT poll or sleep** |

**Parallel spawn**: multiple `Agent` calls in one message = concurrent.
Single message, multiple tool uses. Critical for independent work.

**Reuse vs fresh**: `SendMessage({to: name})` resumes with full
conversation context. A new `Agent` call is cold — prompt must be
self-contained. Before spawning a duplicate, check for an idle
teammate of the right type.

## Workflow

**Invariant that applies to every pattern below**: **one `TaskCreate`
per agent, no exceptions.** The task row is the shared surface between
lead and worker and the only channel that surfaces in-flight work to
the user. No task → no spawn. Holds equally for single-worker and
multi-worker; the steps below are the same rule, scaled.

### Single-worker (lead + builder — default)

Every step below names the channel (tool) used. `SendMessage` is the
ONLY way to talk to the worker once spawned — plain text in the lead's
response is invisible to them. `TaskUpdate` is the ONLY way to express
state changes — don't send `{type: "task_completed"}` messages.

1. **Define hypothesis + spec.** [lead-only] One-sentence hypothesis,
   measurement gate, deliverable paths, constraints, "what NOT to do"
   list.
2. **`TeamCreate`** — single call, creates team + task list (1:1).
3. **`TaskCreate`** for the unit of work. **One task per agent, no
   exceptions.** Use `addBlockedBy` for DAG edges on prior rounds.
4. **Spawn one worker.** `Agent(subagent_type, team_name, name,
   run_in_background=True, prompt=<self-contained brief>)`. The
   brief is the ONLY channel the worker sees at spawn time — it
   must include file paths, line numbers, deliverables, measurement
   gate, "what NOT to do". Never "continue what we discussed."
5. **`TaskUpdate`** immediately after spawn: `{taskId, owner:
   <worker-name>, status: "in_progress"}`. Makes live ownership
   visible to the user.
6. **Wait for the worker's report** (`SendMessage` inbound arrives as
   a conversation turn; completion + idle notifications arrive
   automatically). Don't poll. Don't `Sleep`. Idle ≠ done —
   teammates go idle every turn as normal flow.
7. **Review diff directly.** Trust-but-verify — the worker's report
   describes intent, not reality. Read the actual files changed.
8. **Ship or revise:**
   - **Ship**: commit (if the task description authorized commits up
     front; otherwise confirm with user per CLAUDE.md's "only commit
     when explicitly asked"). `TaskUpdate({taskId, status:
     "completed"})` — but only once the downstream gate that depends
     on this work has passed. Marking completed earlier violates
     "Only mark completed when **fully** done" and hides work that
     may still need revision.
   - **Revise**: `SendMessage({to: <name>, summary, message})` with
     concrete file:line corrections. No hand-waving. The worker is
     idle after their last turn — the message wakes them.
9. **Rotation after 2-3 iterations.** One worker carries a task
   through 2-3 revision cycles; if it hasn't converged by then the
   spec is under-defined, not the worker incompetent. Shut down the
   current worker via `SendMessage({to: <name>, message: {type:
   "shutdown_request"}})`; spawn a fresh worker with a tightened
   spec (new `Agent` call, cold start, self-contained brief — don't
   replay chat history; pointer to recent commits + task list is
   enough).
10. **Teardown.** Once the task list is empty and all workers idle:
    `SendMessage({to: <name>, message: {type: "shutdown_request"}})`
    for each, then `TeamDelete`. Team = TaskList 1:1, so delete
    removes both.

### Multi-worker (N>1 — explicit authorization)

1. **Plan the partition.** Which teammates, which files, which tasks?
   Disjoint file sets only; no race at commit time.
2. **`TeamCreate`** — single call, creates both team and task list.
3. **`TaskCreate`** per unit of work. **One task per agent, no
   exceptions.** Use `addBlockedBy` for DAG edges.
4. **Spawn in parallel** — single message, multiple `Agent` calls,
   `run_in_background: true`. Brief each teammate like a cold
   colleague: file paths, line numbers, context, expected report
   shape. Self-contained prompts.
5. **`TaskUpdate`** to set `owner` + `status: "in_progress"` on each
   task at spawn time. Update to `completed` when the diff ships.
6. **Wait for notifications.** Messages + idle notifications arrive
   automatically. Don't poll.
7. **Verify.** Review diffs directly — an agent's summary describes
   intent, not reality.
8. **Shut down.** Per single-worker pattern.

## Nuances / footguns

- **Plain text doesn't reach teammates.** Your plain text output is
  NOT visible to other agents. Only `SendMessage` carries content to
  them — narration in your response is invisible.
- **Subagent prompts are conversation-less.** A spawned agent doesn't
  see your chat, your memory, or prior tool results. Brief it like a
  cold colleague: file paths, line numbers, context, expected report
  shape — never "continue what we discussed."
- **Background agents don't need babysitting.** Runtime sends a
  completion notification automatically. Don't `Sleep`, don't poll,
  don't call `TaskOutput` in a loop — continue other work and respond
  to the notification when it arrives.
- **Idle ≠ done.** Teammates go idle after every turn as normal flow.
  Don't complain about it; send them a message to wake.
- **Structured JSON status messages** (`{type: "idle"}`, `{type:
  "task_completed"}`) — don't send these. Use TaskUpdate for state,
  plain text for talk. Exception: the `shutdown_request/response`
  and `plan_approval_*` protocols.
- **Names, not UUIDs.** Reference teammates by name in `to`, owners.
  Discover via `~/.claude/teams/{team}/config.json`.
- **Two different "tasks".** TaskList entries (team coordination) vs
  background task IDs (process handles). Same word, different
  systems, different tools.
- **Team = TaskList 1:1.** TeamDelete removes both.
- **Parallel agents share no state.** If B needs A's output,
  sequence them — parallel invocation can't pass data between
  siblings.
- **Don't delegate understanding.** "Based on your findings, fix the
  bug" pushes synthesis onto the agent. Do synthesis yourself; hand
  off the change with explicit file:line targets.
- **Read-only agents can't write.** Explore/Plan/claude-code-guide
  fail on Edit/Write regardless of `mode` parameter.
- **`isolation: "worktree"` on Agent ≠ EnterWorktree.** First is
  per-agent sandbox; second switches YOUR session into a worktree.
- **Broadcast (`to: "*"`) is linear cost** — wakes every teammate.
  Use only when everyone genuinely needs it.
- **Trust but verify.** Read actual diffs after agents claim
  completion. Agent self-reports describe intent, not reality.
- **2-3 iteration cap per worker.** Budget a worker 2-3 revision
  cycles on the same task. If they need a 4th, your spec is
  under-defined. Rotate to fresh worker with a tightened brief —
  fresh cold-start + sharper spec is usually faster than continuing
  to iterate on a noisy conversation.
- **Don't replay chat history to a new worker.** The brief is
  self-contained: pointer at recent commits + task list + remaining
  work. The commits carry the state.
- **Lead never writes implementation on files the worker owns.**
  Lead is orchestrator; mid-flight implementation on the worker's
  claimed files creates race conditions with their edits. Commit
  worker output first, then if more needed, spawn next iteration.
  **Disjoint-files corollary**: lead MAY edit files outside the
  worker's claim (different module, different script) — declare the
  worker's claim explicitly in the brief ("you own X, Y, Z"), then
  lead is free elsewhere. This session: builder2 owned
  `calm/llm_computer/r51/student.py`; lead edited
  `scripts/r51_capture_broad.py` in parallel with zero collision.
- **Worker reports must surface design decisions, not just a diff.**
  Require every worker brief to mandate a "Design decisions worth
  flagging" + "Deferred / open" section in their report. The
  diff-only handoff loses the review's most valuable data —
  reviewer's choices, rejected alternatives, open questions.
  Builder2's S=1 dead-attention catch came from their prose report,
  not the code. Silent workers ship silent bugs.
- **Downstream revisions open a new task, not silent re-edit.**
  If worker N+1 flags an issue with worker N's shipped output,
  create a new task row (`addBlockedBy` → original) rather than
  reopening the completed one. Preserves audit trail + makes the
  revision visible to the user. Silent re-edits break "Only mark
  completed when fully done" retroactively.

## Protocol messages

Legacy JSON response types (respond with matching `_response`, echo
`request_id`):

- `shutdown_request` → `shutdown_response` with `approve: true/false`.
  Approving terminates the receiving agent.
- `plan_approval_request` → `plan_approval_response` with
  `approve: true/false`, optional `feedback`. Rejection sends the
  teammate back to revise.

Don't originate `shutdown_request` unless asked.

## Related rules

- `workflow.md` — hypothesis-test loop; the per-round discipline
  teammates should follow
- `../CLAUDE.md` §"Working policy" — the default no-subagents policy
  this rule is gated by
