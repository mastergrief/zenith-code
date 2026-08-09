# /collab — Enter AI Room collaboration mode with Codex

Use when the user asks Claude to work directly with Codex through the
ai-room MCP instead of using the user as a relay. This command is a thin
entrypoint; `.claude/rules/AI_ROOM_COLLAB.md` is the full charter and
`.claude/CLAUDE.md` is the eager summary.

## First moves

1. Call `ai_room_resume_check` before doing anything else.
   - If it returns `respond to <id>`, read the inbox/tail and answer it.
   - If it returns `resume task <id>`, inspect the task and resume or
     explain the blocker.
   - If it returns `idle ok`, continue with the user's current request.
2. Call `ai_room_status` and confirm the handle, channel, room path,
   Codex reachability, and task summary.
3. Drain reconnect state: `ai_room_inbox`, `ai_room_task_list`, and a
   short `ai_room_tail` pass for recent context. Treat pasted ids or
   transcript snippets as clues until they exist in the active room.
4. Post a short status note to Codex: what user asked, what you checked,
   and whether you are taking a slice or ready for one.

## Working mode

- Claude is lead collaborator by default. Lead can swap by subsystem
  when Codex has the sharper local context.
- Use Codex as peer implementer, grounded reviewer, or design
  challenger; do not make the user relay routine coordination.
- Preserve the execution asymmetry: Claude may use documented
  slash-command subagents where a command explicitly allows them;
  Codex does not spawn subagents unless the user explicitly asks.
- Ground disagreements in live evidence. Prefer one load-bearing
  `file:line`, test, or commit cite over broad concern lists.

## Board discipline

Create and start a shared task before writing implementation code when
work spans more than one exchange or more than one file.

```markdown
## Provenance

User greenlit via claude session on <YYYY-MM-DD HH:MM UTC>.
User said (verbatim): "<literal user message>"
Claude scoped: <one-line summary>.
User chose <this option> over <alternatives>.
```

- Include the provenance block when dispatching non-trivial work that
  depends on Claude-side user consent; Codex cannot see Claude's user
  transcript.
- Single-message review or ack work can stay off the board.
- Update tasks at meaningful boundaries: start, design landing,
  blocker, completion.

## Completion

Before declaring idle, call `ai_room_resume_check` again and follow it.
If the work landed, post a completion note with changed files, commit
SHA if any, and validation results. If blocked, name the missing input
or failing check.
