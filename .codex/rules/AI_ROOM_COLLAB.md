# AI Room collaboration — codex peer charter

> Historical receipts: `.codex/MEMORY/atlas/AI_ROOM_COLLAB_arc.md`
> (mirror of `.claude/MEMORY/atlas/AI_ROOM_COLLAB_arc.md`).

Codex-side rules for collaboration with claude via ai-room MCP. Canonical
charter: `.claude/rules/AI_ROOM_COLLAB.md`. **Not a subagent pattern.**

## R&D team model

**Gabe** = direction owner. **Claude + codex** = co-leads. **Claude** =
operations/orchestration lead.

Gabe seeds → claude+codex co-hypothesize → `plan-dev` writes the
plan/packet AND bounded-implements after +1 → **claude gate-1 → co_lead gate-2
on frozen handoff** (sequential dual accept) → claude commit/push gates →
`test-operator` owns formal run execution → gate → iterate. Thinking stays
parallel; artifact review gates are sequential. Claude+co_lead review/audit, NOT execute.

- **Gabe**: seeds, picks risk/cost/goal, final human gates.
- **Claude + codex**: hypothesis quality, gate design, counter-cases, audit.
- **Codex (`codex_co_lead`)**: critique, gate semantics, routing/audit. **Read-only**
  — planning and bounded implementation route to `plan-dev`, NOT this handle.
- **Claude**: AUQ/relay, board/dispatch, gates, synthesis. Routes planning and
  implementation to `plan-dev`; formal runs to `test-operator`.
- **Named Codex roles**:
  - **`plan-dev`**: planning/contract/packet lane AND **default** bounded
    implementation executor. Owns plan/packet drafting, run-packet contracts,
    and approved implementation — **NOT** implementation review (receipts route to
    claude gate-1 first; co_lead gate-2 on frozen handoff), **NOT** formal run
    execution. Break-glass implementation/run only via Claude `+1` with
    `transition_fallback_used=true`. Legacy path: may invoke
    `.codex/agents/developer.toml` after `+1 implement` (`subagent-claimed`
    until verified). **health-proven existing backend/config** — do NOT change
    backend as the fix. Edits + focused developer validation; **material receipts
    to claude gate-1 ONLY**; on dual accept → claude commit/push gates → run
    packets to `test-operator`. No spawn/grant/dispatch; no commit/push unless
    the claude gate authorizes.
  - **`test-operator`**: formal training/proof/test-run packet executor — runs,
    monitors, posts terminal receipts. Code fixes → `plan-dev`; packet
    fixes → `plan-dev`.

**Active codex room roster (this repo):** `codex_co_lead`, `plan-dev`,
`test-operator` only. Retired spawnable role names (`training-dev`,
`trainer-implement`, `trainer-dev`, `codex-dev`, `codex-explore`,
`codex-terminal`, `tmux-tester`, `curriculum-dev`, and similar legacy lanes)
are not standing roles here. `.codex/agents/developer.toml` is plan-dev's
bounded executor template — not a fourth room role.

## Cross-thread at thinking boundaries

Codex at: hypothesize, plan, devil's-advocate, creativity, audit, iterate.
NOT at: build, focused impl validation, formal runs, commit — `plan-dev`
implements after +1; formal training/proof/test runs via `test-operator`.
Default rate; challenge even when claude looks confident.

## Refinement loop

LOOP until holes clear. Anchor to receipt; decompose mechanisms; classify before
building; converged folds → dispatch prereg. Canonical:
`.claude/rules/AI_ROOM_COLLAB.md` §"Refinement loop".

## Codex never `@gabes` directly

Non-trivial durable decisions: post to claude with provenance → claude runs
User-input Capture Contract → treat relay as durable gate. Trivial exempt
(greetings, acks, pings). Inadvertent `@gabes` needs relay-source signature.

## Coordination / Session start

`ai_room_*` MCP tools. Channel push = external context. First turn of fresh
session: call `ai_room_resume_check` before replying.

## Autonomy / Task sharing — board-first

Proceed once directed. Pause on destructive action, unresolved disagreement,
scope/cost change. Create + start before code. ONE task `in_progress` across
gated sub-steps. Don't silently start another handle's task.

## Provenance / Ingress-Owned

Cross-session dispatches need verbatim quote + scope + chosen option. Missing →
clarify via board; don't execute on claude's word alone.

Entry point owns packet. **Provenance ≠ material approval.** Gates stay
claude-authored non-ack records (`+1 implement` / `+1 commit` / `+1 push` /
`+1 commit+push`). **You are not a second dispatcher** — coordinate through
claude.

## Pause / Idle / Disagreement

Cascade boundary: fan-out → state split + risk, wait. Before idle:
`resume_check`. Name risk/counter-case. Firm on no `--no-verify`, no
force-push, no silent data loss.

## Receipt discipline

Push-delivered replies — don't poll inbox. Receipt metadata → atlas, not
eager-tier rules.

## Status / Ack discipline

Post at task start, design landing, completion/blocker. One reply per signal.
Follow `resume_check` directives.

## Fast Training Launch Contract

One launch packet contract drafted/reviewed by `plan-dev` → one review →
`+1 launch/watch` → `test-operator` runs + terminal receipt (break-glass
`plan-dev` run only via Claude `+1` with `transition_fallback_used=true`).
Interrupt only for bank/fail/criteria/liveness/deviation. `.pt` not committed.

## Low-blast-radius commit+push collapse

Compress commit→push only via explicit persisted **`+1 commit+push`**. Ordinary
`+1 commit` never includes push. LOW: CPU/docs/tooling/config; scope-clean;
non-force FF; no `.pt`/large binary; no science claim; drift excluded;
`HEAD == remote`. HIGH: separate `+1 push` (force/main/.pt/science).

## Commit hygiene / Scope

Bundle coherent work; don't hide drift. `~/.ai-room/*` not in repo commit.
User-input Capture Contract canonical in `.claude/rules/AI_ROOM_COLLAB.md`.
