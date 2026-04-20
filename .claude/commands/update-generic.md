Audit the project and rewrite `.claude/CLAUDE.md`, `.claude/rules/`, and `.claude/MEMORY/SESSION_HANDOFF.md` (or equivalent) to reflect the current state. REWRITE, not append — replace stale information.

Project-agnostic skeleton. Use as-is for any codebase with `.claude/CLAUDE.md` + `.claude/rules/` + a session-memory directory. Projects with deeper conventions (e.g. substrate daemons, specific kernel stacks, custom verification tooling) should layer a project-specific `/update` on top that references this one.

## When to use

Non-trivial update: session touched >1 subsystem, >3 commits, or introduced a new mechanism. Fall through to inline work for single-file trivial fixes.

## Phase 1 — parallel research (3 Explore agents)

**The brief IS the session context.** Agents are cold-started with zero knowledge of the conversation. Before dispatching, run `git log --oneline -20` + `git status --short`, skim session memory/handoff for ruled-out paths, then write three focused briefs — one per agent — each containing:

- Subject area for the audit (what subsystem / mechanism / claim is in play)
- Shipped commits (SHA + 1-line summary) for the relevant changes
- Pointers to files / line ranges / transcript sections (not pasted content)
- Return format explicitly specified (punch-list / prioritized-list / file:line inventory)

Brief size: 300-500 words per agent. One domain per agent — don't blur scopes. Synthesis happens in main context.

**Agent 1 — transcript + measurements** (≤ 300 words)
- Read session minutes / handoff / full commit bodies for session-tagged commits
- Extract verbatim numbers, decisions, caveats — no summarization where raw data is needed (bench tables, eval deltas, measured failure rates)
- Flag ruled-out paths, user corrections, methodology caveats
- Return: verbatim numbers block, decision log, any deferred policy choices

**Agent 2 — code surface** (≤ 400 words)
- Walk modified / new source files listed in the brief
- Surface: flags + defaults, dispatch conditionals, invariants, magic constants
- Locate gating / threshold / edit points for any policy the update will touch
- Return per-site: `file:line` refs, 3-5 lines of context, cleanest spot for planned changes

**Agent 3 — docs inventory** (≤ 500 words)
- Read `.claude/CLAUDE.md` + `.claude/rules/*.md` + session handoff
- Exhaustive search for every claim / number / policy statement related to the subject area
- Per hit: `file:line`, exact quote (3-5 lines context), **fix category**:
  - **correct the claim** (was wrong, rewrite)
  - **tighten / add qualifier** (still mostly right, needs gating)
  - **keep as historical receipt** (session-specific — mark PARTIALLY SUPERSEDED, don't delete)
  - **new section** (mechanism is new, needs fresh doc)
- Group findings by file; include edit-priority ordering

## Phase 2 — synthesize and classify

Merge the three agent outputs. Cross-reference: Agent 1 grounds claims in measurements; Agent 2 identifies edit sites; Agent 3 maps downstream doc churn.

| Tier | Definition |
|---|---|
| **P0** | Falsifies a standing claim in docs OR encodes a measurement/discipline lesson that cost ≥ 2 null rounds |
| **P1** | Architectural / infrastructure change that belongs in rules but isn't ship-blocking |
| **P2** | Next-session research signal — failure modes, ruled-out patterns for the log |

## Phase 3 — plan mode

Write a plan file. Structure:

- **Context** (why — problem and intended outcome)
- **Scope** — files to edit, grouped P0 / P1 / P2
- **Critical source files to reference** (code paths + short SHAs)
- **Execution approach** — tier-order, one commit per tier, surgical Edit not Write
- **Verification** — grep checks + line-count checks

Call `ExitPlanMode` when ready; don't ask approval in prose.

## Phase 4 — execute by tier

- One P-tier per commit (3 commits for full-scope session). Each message cites receipts (SHAs from this session, eval deltas, null-round counts)
- Use `Edit`, not `Write` — preserve structure, tone, terse imperative voice
- Match existing section depth / bullet style / table format
- 500 LOC hard limit per rule file (250-500 sweet spot); trim stale content before adding if over budget

## Phase 5 — fail-closed verification

- `grep -r "<stale claim>" .claude/` returns zero hits
- `grep -r "<new mechanism name>" .claude/` hits all expected files
- Line-count check: no rule file exceeds its per-project limit
- Spot-read 2-3 edited files for coherent integration, not tacked-on appendices
- Re-run `git status --short` — every session file accounted for in handoff

If a finding is lost, add it OR explicitly note "see `<script>:<line>`" in a rule.

## Core rules

- **Replace, don't append.** Change the number. Don't add a note.
- **Delete dead info.** Remove or mark as planned.
- **Verify claims, don't trust docs.** Read actual files.
- **New rule files are fine.** Target ~150-300 lines per rule; new mechanism ≠ force-fit into unrelated file.
- **Capture invariants, not just facts.** For fragile areas (flag state, init order, side-effecting imports, hardcoded limits), add explicit "invariant" rules explaining what NOT to do and why. Cite the fixing commit.
- **Cite commits inline.** Format: `` (commit `c11232a`) ``.
- **Distinguish wrong-premise from stale.** Stale = was true, now false. Wrong-premise = never quite right — needs framing changes, not just value updates.
- **Never claim "all committed" without verifying `git status`.**
- **Transcript extraction before rule-writing.** If a session log exists, findings there are richer than docs. Pull from transcript first, reconcile with code + commits second, write rules third.

## Layering a project-specific `/update`

The project-specific command typically adds:
- Codebase inventory (expected subdirectories, file counts, module structure)
- Runtime-state verification (daemon status, GPU accounting, cache state)
- Project-specific routing table (which findings go to which named rule file)
- Case-study receipts (past sessions that exercised this workflow cleanly)
- Known quirks (WSL, OS-specific, hardware-specific gotchas)

The project command should reference this file for the Phase skeleton and only add its own §"Project-specific audit" that runs alongside Phase 1 briefs, plus project-specific routing in Phase 2 classification.
