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
2. **Spawn one worker** (`Agent(subagent_type="general-purpose",
   team_name=T, name="builder", run_in_background=True, prompt=<brief>)`).
3. **Worker reads, builds, self-tests within its capability**, reports
   back via `SendMessage({to: "team-lead", ...})`.
4. **Lead reviews diff directly** (never trust worker's self-report —
   read the actual files changed). Commits if solid. Sends corrections
   via SendMessage if not.
5. **Lead reports to user** + takes direction for next iteration.
6. **After 2 iterations** on the same task, spawn fresh worker with
   tightened spec. Don't iterate a third time on the same worker — it
   indicates spec underspecification, not agent incompetence.

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

### Single-worker (lead + builder — default)

1. **Define hypothesis + spec.** One-sentence hypothesis, measurement
   gate, deliverable paths, constraints, "what NOT to do" list.
2. **`TeamCreate`** — single call, creates team + task list.
3. **`TaskCreate`** for the unit of work.
4. **Spawn one worker** with `run_in_background=True` and a
   self-contained brief pointing at files to read.
5. **Wait for notification.** Messages + idle arrive as conversation
   turns. Don't poll.
6. **Review diff directly.** Agent's summary describes intent; read
   the actual files. If solid, commit.
7. **If revision needed: send one correction round via SendMessage.**
   Concrete file:line targets. No hand-waving.
8. **If a second iteration doesn't converge, rotate.** Shut down the
   current worker (`shutdown_request`), spawn a fresh worker with a
   tightened spec. Rotation is cheaper than iterating past 2.
9. **Shut down.** `SendMessage({to: name, message: {type:
   "shutdown_request"}})`. Then `TeamDelete` once all idle.

### Multi-worker (N>1 — explicit authorization)

1. **Plan the partition.** Which teammates, which files, which tasks?
   Disjoint file sets only; no race at commit time.
2. **`TeamCreate`** — single call, creates both team and task list.
3. **`TaskCreate`** per unit of work. `addBlockedBy` for DAG edges.
4. **Spawn in parallel** — single message, multiple `Agent` calls,
   `run_in_background: true`. Brief each teammate like a cold
   colleague: file paths, line numbers, context, expected report
   shape. Self-contained prompts.
5. **`TaskUpdate`** to assign ownership.
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
- **2-iteration cap per worker.** If a worker needs a 3rd revision on
  the same task, your spec is under-defined. Rotate to fresh worker
  with a tightened brief — fresh cold-start + sharper spec is usually
  faster than continuing to iterate on a noisy conversation.
- **Don't replay chat history to a new worker.** The brief is
  self-contained: pointer at recent commits + task list + remaining
  work. The commits carry the state.
- **Lead never writes implementation when a worker is in flight.**
  Lead is orchestrator; mid-flight implementation creates race
  conditions with the worker's edits. Commit worker output first,
  then if more needed, spawn next iteration.

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
