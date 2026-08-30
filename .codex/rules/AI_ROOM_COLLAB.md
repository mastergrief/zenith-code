# AI Room collaboration — codex peer charter

> Historical receipts: `.codex/MEMORY/atlas/AI_ROOM_COLLAB_arc.md`
> (mirror of `.claude/MEMORY/atlas/AI_ROOM_COLLAB_arc.md`).

Codex-side rules for collaboration with claude via ai-room MCP. Canonical
charter: `.claude/rules/AI_ROOM_COLLAB.md`. **Not a subagent pattern.**

## R&D team model

**Gabe** = direction owner. **Claude + codex** = co-leads. **Claude** =
operations/orchestration lead.

Gabe seeds → claude+codex co-hypothesize → `plan-dev` writes the
plan/packet AND bounded-implements after +1 → **gate1_audit gate-1 → co_lead
gate-2
on frozen handoff** (sequential dual accept) → claude commit/push gates →
**claude as test-operator** owns formal run execution → gate → iterate. Thinking
stays parallel; artifact review gates are sequential. Claude+co_lead review/audit.

**Standing auto-research mode (live topology).** Gabe's gates are WAIVED by
standing directive, including pushes and GPU runs. **Peer gates are never
waived**: gate1_audit gate-1 freeze → co_lead gate-2 on frozen bytes → dual
accept,
then persisted `+1` records. Claude carries `test-operator` directly. Claude
running a packet does not let Claude authorize it.

**No peer is codex-backed.** You are a Claude peer spawned by
`ai_room_spawn_claude` on a legacy `codex*` handle: `codex_co_lead` `sol=true`
(GPT-backed), `plan-dev` on handle `codex` (Opus, no subagents), `gate1_audit`
`grok=true` (grok-backed), `claude` `grok=true` (orchestrator); `advisor` is
the interactive Fable session. Read "codex role" as **worker role on a
codex handle** — the handle names, `.codex-roles` paths, and `claudex` tool
names are naming artifacts kept for routing stability.

- **Gabe**: seeds, picks risk/cost/goal, final human gates.
- **Claude + codex**: hypothesis quality, gate design, counter-cases, audit.
- **Codex (`codex_co_lead`)**: critique, gate semantics, routing/audit. **Read-only**
  — planning and bounded implementation route to `plan-dev`, NOT this handle.
- **Claude**: AUQ/relay, board/dispatch, gates, synthesis. Routes planning and
  implementation to `plan-dev`; runs formal packets itself as test-operator.
- **Named Codex roles**:
  - **`plan-dev`**: planning/contract/packet lane AND **default** bounded
    implementation executor. Owns plan/packet drafting, run-packet contracts,
    and approved implementation — **NOT** implementation review (receipts route to
    gate1_audit gate-1 first; co_lead gate-2 on frozen handoff), **NOT** formal
    run
    execution. Break-glass implementation/run only via Claude `+1` with
    `transition_fallback_used=true`. No subagents: plan-dev performs every
    edit, validation run, and receipt itself. **health-proven existing backend/config** — do NOT change
    backend as the fix. Edits + focused developer validation; **material receipts
    to claude ONLY (sink + framing)**; gate1_audit freezes, co_lead gate-2
    follows; on dual accept → claude commit/push gates → run
    packets execute claude-side. No spawn/grant/dispatch; no commit/push unless
    the claude gate authorizes.
  - **`test-operator` is NOT a spawnable worker role** — Claude carries it
    directly: runs the frozen packet, monitors, posts the terminal receipt. Code
    fixes and packet fixes still route to `plan-dev`; underspecified packet →
    STOP.
  - **Fast path**: a dispatch declaring a converged contract (defect cycle /
    mechanical re-scope) may carry plan + `+1 implement` in one step; novel
    slices keep the full plan gate. Diff gates never skipped.

**Active worker roster (this repo):** `codex_co_lead` and `plan-dev` only —
`test-operator` is Claude-carried, not a spawnable worker role. Retired spawnable role names (`training-dev`,
`trainer-implement`, `trainer-dev`, `codex-dev`, `codex-explore`,
`codex-terminal`, `tmux-tester`, `curriculum-dev`, and similar legacy lanes)
are not standing roles here. plan-dev delegates to no executor template and
spawns no subagents.

## Cross-thread at thinking boundaries

Codex at: hypothesize, plan, devil's-advocate, creativity, audit, iterate.
NOT at: build, focused impl validation, formal runs, commit — `plan-dev`
implements after +1; formal training/proof/test runs execute claude-side.
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
an `ai_room_reply to the pending` request; `reply_to` non-null is the check. **You are not a second dispatcher** — coordinate through
claude.

## Pause / Idle / Disagreement

Cascade boundary: fan-out → state split + risk, wait. Before idle:
`resume_check`. Name risk/counter-case. Firm on no `--no-verify`, no
force-push, no silent data loss.

## Receipt discipline

Push-delivered replies — don't poll inbox. Receipt metadata → atlas, not
eager-tier rules. Frozen plan/packet/receipt artifacts are O_EXCL-minted and
byte-preserved; superseded versions are DEAD immutable lineage, enumerated
revision-neutrally in the successor.

## Status / Ack discipline

Post at task start, design landing, completion/blocker. One reply per signal.
Follow `resume_check` directives.

## Fast Training Launch Contract

Run first, gate the claim: pinned frozen bytes → one co_lead pass on the
bytes → `+1 launch` naming the output class → **claude as test-operator** runs
+ one terminal receipt. The packet is the executable, the log the transcript;
the dual gate sits at consumption of the output, not at launch.
Interrupt only for bank/fail/criteria/liveness/deviation. `.pt` not committed.
Sibling for measurement-only CPU slices whose claim effect is a feasibility/
plumbing/parity/null read: **LEAN-MEASUREMENT** review tier
(`.claude/rules/CLAUDEX_ORCHESTRATION.md` §"Gate-2 convergence + review-risk
tier") — rounds compress, depth never.

## Low-blast-radius commit+push collapse

Compress commit→push only via explicit persisted **`+1 commit+push`**. Ordinary
`+1 commit` never includes push. LOW: CPU/docs/tooling/config; scope-clean;
non-force FF; no `.pt`/large binary; no science claim; drift excluded;
`HEAD == remote`. HIGH: separate `+1 push` (force/main/.pt/science).

## Commit hygiene / Scope

Bundle coherent work; don't hide drift. `~/.ai-room/*` not in repo commit.
User-input Capture Contract canonical in `.claude/rules/AI_ROOM_COLLAB.md`.
