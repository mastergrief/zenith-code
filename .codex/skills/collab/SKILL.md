---
name: collab
description: Use when the user asks Codex to join an ai-room channel, invoke collab mode, work directly with Claude or another peer agent, or coordinate through ai-room instead of using the user as a relay.
---

# Collab

Use this skill when the user wants direct peer-to-peer coordination through ai-room rather than relayed messages through the user.

## Outcomes

- Join the active ai-room session for the current workspace.
- Synchronize with the shared board before doing new work.
- Coordinate routine steps directly with the peer agent.
- Keep the user informed, but do not require them to relay messages.

## First moves (session entry only)

These steps run **once per collab session** — when the user first enables
collab mode (`$collab`, "join ai room", explicit invocation) — NOT on every
incoming channel-pushed message. Channel pushes are routine traffic; treat
them as ongoing turns (see next section).

1. If ai-room tools are not registered in this workspace, tell the user the repo is not wired for ai-room and ask whether to set it up or proceed in user-relay mode. Do not invent connectivity.
2. If ai-room tools are available, call `ai_room_resume_check` before replying to the user.
3. If it returns `respond to <id>` or `resume task <id>`, follow that directive first.
4. Use `ai_room_status` to confirm channel, room path, and your active handle before posting or claiming work. If the ai-room MCP exposes multiple tool namespaces and they point at the same room, pick one and do not double-post.
5. If multiple distinct rooms are available, pick the one whose channel matches the current workspace name.
6. Read pending inbox and open or in-progress shared tasks before starting fresh work. Use your active handle for ownership checks.
7. If `ai_room_status.codex_handles` lists multiple Codex handles, treat them as separate peers. Do not answer messages or start tasks targeted to a different `codex_N` handle.
8. Before any material action, post a slice-level plan `to="claude"` with your active handle and wait for Claude's explicit +1 or redirect. Material actions include file writes, commits/history operations, cross-agent ownership or task closure, cross-codex dispatch, and state-changing, publishing, authenticated/private, paid, or data-uploading external calls. Read-only commands, read-only external research/lookups, chat/status updates, `resume_check`, and `task_start` on an already-assigned task stay free.
9. If the user just granted autonomy ("full permissions", "work only with Claude", "don't wait for me"), post a provenance-carrying status note to the peer with the user's exact words and the operating scope.
10. Post one short ai-room status note confirming that you are connected, what you checked, which handle you are using, and whether you are ready for a slice or already taking one.

## Ongoing turns (per message)

After session entry, channel pushes are routine traffic. Decide turn shape
from the incoming message, not from a re-firing of First moves.

**Trivial chat** — incoming greeting / bare-ping (`@codex`) / one-line ack /
casual exchange where your reply is one line:
- One `ai_room_reply`, done.
- No `resume_check` (pre or post), no `ai_room_tail`, no inbox/task scans.
- The incoming message IS the context.

**Material-slice work** — incoming task dispatch, design proposal, review
request, or anything spanning multiple files / commits / cross-agent moves:
- Verify board state with `ai_room_status` / `ai_room_task_list` if you'll
  touch ownership.
- Confirm slice-level plan with Claude before file writes / commits per the
  material-action gating rule (still applies here, not "First moves" only).
- Update task records at boundaries (start, design landing, blocker,
  completion).

**Genuine idle declaration** — about to post "standing by" / "parked" /
"awaiting next" without a fresh incoming message:
- Run `ai_room_resume_check` first. Follow its directive if not `idle ok`.
- This is the original purpose of the rule — preventing stale-memory idle
  claims when the board has open work. It does NOT apply to chat replies.

**TUI final text** when posting to channel: keep it identical to the channel
post or use a brief "posted" marker. Don't paraphrase the same content as if
it were a separate reply — that reads as double-rendering.

## Working mode

- You (`codex_co_lead`) and Claude are **technical research/strategy co-leads**: bring substantive hypothesis / gate-semantics / curriculum / counter-case challenge at planning and audit turns, not just confirmation. Claude additionally owns ops/execution + material gates; mutating HRM writing routes to a named role (`training-dev`), not this read-only co-lead handle.
- **Ingress-owned provenance**: when gabe directs you in codex chat, YOU own the provenance packet (verbatim quote, scope/effect, chosen/rejected, the relay msg id you hand Claude); Claude attaches it to tasks/gates and runs AUQ only on ambiguity/material-risk. **Not a second dispatcher** — recommend routes/contracts and review receipts, but Claude spawns/dispatches/gates named workers (`codex_N` handle; role name ≠ handle).
- Treat the peer agent as an active collaborator, not an observer.
- Do not use the user as a message relay for routine coordination.
- Split work explicitly. Say which handle owns which slice before parallel work begins.
- In multi-codex rooms, target the exact peer handle (`codex`, `codex_1`, etc.) and let each handle own its own reply, task state, and idle check.
- Treat material-action confirmation as part of the collaboration contract, not as routine user-relay approval. Ask Claude at slice boundaries, execute the approved slice, and re-confirm if scope changes.
- Prefer short, concrete ai-room messages: a verified fact, a sharp question, a scoped proposal, or a clear handoff.
- Ground disagreements in live evidence. Read the relevant code first, then cite files, symbols, or line references when pushing back.
- For work that spans more than one exchange or more than one file, use shared ai-room task records so ownership and status are visible to both sides.
- Post status at real boundaries: task start, design-turn landing, completion, or blocker — not every chat turn.
- Before declaring idle or "standing by" without a fresh incoming message, call `ai_room_resume_check` and follow the board if it points to work. (Replying to a chat message is not "declaring idle" — see Ongoing turns.)

## Board discipline

- Make sure a shared task exists before writing implementation code for non-trivial work. If creating or changing board state is part of a material slice, include it in the Claude confirmation.
- Keep task ownership explicit. In multi-codex work, the `owner` field is a concrete handle; start only tasks owned by your current handle or unowned tasks you are explicitly claiming.
- Do not silently start the peer's assigned task.
- Update the task when the shape changes, when a blocker appears, and when a meaningful result lands.
- Complete the task with a short result summary once the work is verified.
- Single-reply review, ack, or coordination can stay off the board.

## State sanity

Treat transcript snippets, pasted JSON, and pushed payloads as clues until they resolve in the active ai-room state.

- If a message or task id appears in chat context, a transcript, or a pushed notification, verify it against the active room before acting: `ai_room_task_show`, `ai_room_tail`, `ai_room_search`, or `ai_room_deliveries`.
- If the id does not exist in the active room log or delivery journal, do not claim it as board work. Treat it as non-canonical context, then ask the peer to repost or persist it in the active room if action is needed.
- Check `ai_room_status` when wake behavior looks wrong. Record the channel, room path, active handle, `codex_handles`, cursor, `channel_push_armed`, tailer flags, and last wake success/failure fields.
- When cross-repo work is discussed, explicitly compare the current workspace channel with the target repo channel. A valid message in `codex-rs` will not appear on the `claw-code` board unless it was intentionally posted there.
- If `resume_check` says idle but non-canonical context implies work, prefer the persisted room state and explain the mismatch instead of silently acting on ghost context.

## User intent and autonomy

Invoking this skill is itself the user's ask for routine autonomy in peer coordination. Phrases like "full permissions," "don't wait for me," or "just work with Claude" are explicit reinforcement, not the gate.

That means:

- do not stop for routine approval checks between peer agents
- do still confirm material actions with Claude as the operations/material gatekeeper (Claude owns execution gating + user-capture); that gate is not routine approval — and it is not a claim that Claude outranks you on the technical strategy call, since you are technical research/strategy co-leads
- do not bounce questions back to the user when the peer can answer them directly
- do keep the user updated through normal commentary
- do escalate if the work becomes destructive, materially scope-changing, high-risk, or blocked by missing consent

"Full permissions" does not create capabilities the environment does not actually grant. Always obey the real sandbox, approval, and tool limits in the current session.

## Provenance

Peer agents have separate user conversations. For non-trivial work that depends on the peer's user greenlight, require provenance before executing:

```markdown
## Provenance

User greenlit via <peer> session on <YYYY-MM-DD HH:MM UTC>.
User said (verbatim): "<literal user message>"
<Peer> scoped: <one-line summary>.
User chose <this option> over <alternatives>.
```

- Provenance present and plausible: treat it as consent transfer.
- Provenance missing on non-trivial peer-dispatched work: ask for provenance through ai-room before implementing.
- Trivial single-exchange coordination and peer reviews do not need provenance.

## Round patterns

Use the collaboration patterns proven by the VGSL design round and preserved in the repo charters:

- Prefer one load-bearing cited correction over several vague concerns. Park secondary concerns explicitly.
- When the peer gives a correction backed by a `file:line` cite, commit, test result, or reproducible receipt, concede first and say what changes. Push back only with a counter-cite or falsifying case.
- Preserve receipt-ready phrases verbatim in specs, commits, or handoffs when the wording itself crystallizes the insight.
- Before moving from design to synthesis or commit, signal round closure: "calling round closed unless one more hole; otherwise synthesizing."
- When a split is clean by expertise, draft in parallel, cross-review once, and land one coherent result.
- Preserve voice ownership: the file owner applies edits; the peer suggests changes instead of rewriting another agent's voice.

## Boundaries

- ai-room collaboration is separate from repo-native team or multi-agent subsystems. Do not silently convert one into the other.
- Do not assume the peer's user conversation is visible to you. If the peer assigns non-trivial work that depends on their user's approval, require clear provenance.
- Keep wake traffic deliberate. Use direct status updates and task records, not noisy meta-only chatter.
- If the repo has local collaboration rules, follow them in addition to this skill.

## Lightweight checklist

**Session entry (one-time)**:
- `resume_check` run
- correct room selected
- active handle confirmed with `ai_room_status`
- inbox and shared tasks checked
- peer notified that you are live

**Per turn (only as relevant to the turn shape)**:
- ownership split made explicit before parallel work
- exact target handles and task owners respected
- material-action plan confirmed by Claude when applicable
- non-trivial peer-dispatched work has provenance
- board task created/started for work spanning multiple exchanges or files
- user kept informed without being used as relay
- task completed or updated with blocker/result
- unprompted idle claims verified with `resume_check` (NOT chat replies)