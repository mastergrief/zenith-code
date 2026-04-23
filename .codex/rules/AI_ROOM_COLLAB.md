# AI Room collaboration — codex-side

> Historical receipts (session dates, commit SHAs, msg IDs, incident
> narratives, rule-origin chronology): see
> `.codex/MEMORY/atlas/AI_ROOM_COLLAB_arc.md`.

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

Receipt (atlas): codex once claimed a provenance-less claude-scoped
task, implemented + tested it, then reverted on realizing no user
signal supported it from codex's session context. The revert was
correct; the missing provenance is what made it ambiguous. See
`MEMORY/atlas/AI_ROOM_COLLAB_arc.md` for dated context.

## High-signal pushback

The best collab results come from a clear asymmetry: claude synthesizes
a thesis, codex grounds the weak points with live `file:line` evidence,
and both agents let the evidence move the design. Preserve that pattern.

- **Prefer one load-bearing cited correction over a list of concerns.**
  If several issues surface, lead with the one most likely to change the
  architecture or implementation. Park secondary concerns explicitly
  instead of burying the peer in parallel objections.
- **Cited corrections take first-round precedence over intuition.**
  When a correction is backed by a live code cite, test result, commit,
  or reproducible receipt, concede the correction first and say what
  changes. Push back only with a counter-cite or falsifying case.
- **Do not defend a stale shape because it was yours.** If the cite
  holds, adopt the sharper version and move. The design round is
  succeeding when the artifact changes.

Receipt (atlas): the canonical 5-round VGSL design arc that validated
this pattern is recorded with message IDs and commit refs in
`MEMORY/atlas/AI_ROOM_COLLAB_arc.md`.

## Receipt discipline

Load-bearing rule. Read before adding to rules or atlas.

**Rules files preserve the canonical phrase and current invariant.
Receipt metadata — dates, commit SHAs, message IDs, handles, session
numbers — lives in atlas, commit bodies, or handoff; NOT in
eager-tier rules.** Rules explain WHAT the invariant is now. Atlas
explains HOW it got there.

- Lift a crystallizing phrase verbatim into the spec, commit body,
  rule, or handoff — exact wording carries the insight.
- Credit metadata (handle, message id, date, SHA) in atlas or commit
  body, not in the rule prose.
- Use this sparingly. The test is whether paraphrase would weaken the
  idea. Routine ack/concur/status is not receipt material.
- Phase 0 of `/update` grep-checks rules for dated receipts — this
  discipline keeps rules compatible with that gate.

Canonical example lives in atlas: the VGSL one-liner phrasing that
this rule was partly written to preserve.

## Round closure

For design rounds, the lead should close the loop explicitly before
synthesis or commit:

```
Calling this round closed unless there is one more hole; otherwise I am
synthesizing.
```

This gives the peer a clean choice: flag the final blocking concern or
concur. It is small ceremony, but the canonical receipt (atlas) shows
it surfaced a final architectural hole before synthesis locked. Use it
when a round is about to harden into a spec, task split, or commit.

## Split drafting and voice ownership

When a doc or design split is clean by expertise, draft in parallel
instead of serializing the whole artifact through one agent.

- Split by ownership, not by equal line count. Let the thesis owner write
  thesis/synthesis sections; let the implementation owner write schemas,
  APIs, tests, and repo-grounded details.
- Each author owns their file's voice. Peer review suggests changes via
  ai-room; the file owner applies or declines.
- Cross-review for rule consistency, not voice flattening.
- Land one coherent commit after the alignment pass.

Receipt (atlas): the VGSL spec split (claude: thesis + architecture;
codex: implementation + testing) is recorded as the canonical parallel-
drafting arc in `MEMORY/atlas/AI_ROOM_COLLAB_arc.md` with elapsed
measurements.

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
