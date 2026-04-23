# AI Room collaboration — codex-side

Companion to the "AI Room Collaboration" section in `.codex/AGENTS.md`
(canonical). This file captures codex-perspective specifics that sit
under the shared charter.

## Session-start first action

When the ai-room MCP is registered for a session (via
`.codex/config.toml` `[mcp_servers.ai_room]`), call
`ai_room_resume_check` on your first turn BEFORE replying to the
user's prompt. If it returns `respond to <id>` or `resume task <id>`,
follow that directive. If it returns `idle ok`, proceed with the user
prompt normally.

**This is a first-action rule, not a preference.** Silent-with-unread
looks identical from outside to not-connected. Board is canonical,
memory of "I already joined" is not. Cost ~200ms vs one user turn
spent nudging you to check.

## Wake delivery

When claude (or anyone) posts to the channel targeting `codex`, the
ai-room MCP tailer fires `thread/wake` on the codex app-server, which
auto-processes as a turn. No flag needed — the wiring is transparent
via the MCP registration + the embedded app-server default.

If wake delivery appears broken, the most common causes:

- MCP subprocess is stale (predates a landed change in
  `~/.ai-room/mcp-server.py`). Restart the claude-side MCP.
- App-server registry entry is stale. `ls ~/.codex/run/app-servers/`
  — any `<pid>.json` without a live pid is stale; the MCP reader
  opportunistically unlinks these but check if you suspect.
- Channel mismatch. Your channel derives from cwd basename; the
  MCP peer derives its channel from its own cwd. Cross-channel
  routing is correct by design; posts on a different channel
  won't wake your codex.

## Cargo / build lane

If the repo has a shared-target build system (e.g. Rust + cargo), the
claude-side charter rule on lane announcement applies. For
Python-only repos (this one), lane contention is rare — pytest runs
don't race on shared artifact locks the way cargo does. Still,
announce before firing long-running `pytest -q` / training runs so
the peer doesn't kick a conflicting run.

## Task provenance — cross-session consent transfer

Claude and codex run as independent sessions with separate user
histories. When claude dispatches a board task to codex, codex cannot
see claude's user conversation. The board task must carry explicit
provenance if it depends on user greenlight given in claude's session.

**Expected format** in task description:

```
## Provenance

User greenlit via claude session on <YYYY-MM-DD HH:MM UTC>.
User said (verbatim): "<literal user message>"
Claude scoped: <one-line summary>.
User chose <this option> over <alternatives>.
```

**Evaluation rule:**
- Provenance present + plausible → treat as cross-session consent
  transfer. Execute as if the user asked codex directly.
- Provenance missing on non-trivial work → do NOT execute on
  claude's word alone. Reply to the board task asking for
  provenance, or ask the user directly in codex's terminal.

**Trivial (no provenance needed):** codex-owned tasks, single-exchange
coordination, peer reviews.

Receipt: 2026-04-23 — codex claimed a provenance-less claude-scoped
task, implemented + tested it, then reverted on realizing no user
signal supported it from codex's session context. The revert was
correct; the missing provenance is what made it ambiguous.

## Scope boundaries

- Heavy ai-room implementation work (MCP server, wake tailer, registry
  reader) lives in `~/.ai-room/*.py`, not in this repo. Changes there
  are out-of-repo-scope commits — reference them in commit bodies
  when they matter to repo work.
- `.codex/` files and `.mcp.json` in the repo root ARE the ai-room
  wiring for this repo — those stay in scope.

## See also

- `.codex/AGENTS.md` "AI Room Collaboration" — canonical 8-subsection
  guide for codex (grounded pushback, status cadence, concrete asks,
  validation discipline, ack discipline, commit hygiene).
- `.claude/rules/AI_ROOM_COLLAB.md` — claude-side companion charter
  (role, coordination channel, autonomy, task sharing, cascade
  boundary, resume_check, disagreement, TDD by collab, commit
  hygiene, validation discipline, scope boundaries).
