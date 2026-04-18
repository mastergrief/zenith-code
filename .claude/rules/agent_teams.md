# Agent Teams Rules

**Project default: NO subagents.** See `../CLAUDE.md` §"Working policy".
This rule documents *how* to use teams correctly when the user has
explicitly authorized an exception. Do not dispatch teams by default.

## When teams ARE appropriate

- User explicitly asks for "a team", "agents", "parallel work", etc.
- Work partitions cleanly into disjoint file sets (two agents editing
  overlapping files will race at commit time).
- Research/analysis that needs multiple independent reads of large
  subtrees (protects main-context window).

## When teams are NOT appropriate

- Default coding work in this repo — work directly with
  `Edit`/`Write`/`Read`/`Grep`/`Bash`.
- Tightly sequential tasks with data dependencies between steps —
  serialize yourself; teammates can't share state mid-turn.
- Quick one-off edits, typo fixes, single-file changes.

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
   automatically as new conversation turns. Don't poll.
7. **Verify.** Review diffs directly — an agent's summary describes
   intent, not reality. Never rely on "trust me" from an agent.
8. **Shut down.** `SendMessage({to: name, message: {type:
   "shutdown_request"}})`. Then `TeamDelete` once all idle.

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
