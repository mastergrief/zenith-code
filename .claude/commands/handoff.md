Create a detailed session handoff document so a future Claude Code session can seamlessly continue.

## When to spawn agents

**Default for non-trivial sessions** (>3 commits, >1 subsystem, introduced a new mechanism, OR session log exceeds ~30K tokens): launch 2 Explore agents in parallel to ground the handoff against transcript + code state. The main context has the narrative but not the verbatim numbers, uncommitted state, and failure-mode details without hallucination risk.

**Inline-only** for short single-subsystem sessions (1-2 commits, no uncommitted state, narrative fresh): skip agents, write directly. Previous guidance was "never spawn agents" but that assumed context was always pristine — for long sessions it isn't.

## Phase 1 — parallel grounding (2 Explore agents)

Before dispatching, run `git log --oneline -10` + `git status --short` + `git diff --stat`. Brief each agent in 200-300 words with: subject area, relevant commits (SHA + 1-line), pointer to session transcript location, and return-format spec.

**Agent 1 — transcript + measurements** (≤ 300 words)
- Read session minutes at `.claude/MEMORY/minutes/*.md` for the current session (most recent by filename/mtime)
- Read commit bodies for every session-tagged commit (via `git show <sha>`)
- Extract verbatim: bench numbers, eval deltas, user corrections, decisions + their stated rationale, explicitly-deferred choices
- Flag: ruled-out paths, methodology caveats (e.g. single-run vs median), user pivots mid-session
- Return: verbatim numbers block, decision log, deferred-items list

**Agent 2 — code + uncommitted state** (≤ 400 words)
- Read `git status --short` output and list every modified/untracked file from this session
- For each non-trivial uncommitted file: read it, report purpose (1 sentence), flag risk level (session-critical / supporting / gitignored cache)
- For each session commit: report the files touched + one-sentence intent
- Verify runtime-state claims: daemon status, GPU residency (`nvidia-smi`), cache state, any build artifacts
- Return: committed-files map, uncommitted-files risk list, runtime-state snapshot

## Phase 2 — write the handoff

Merge the two agent outputs with the main-context narrative. The narrative gives you the story; the agents give you the ground truth. Cross-reference: every "we decided X" claim should map to a transcript line; every "file Y was changed" should appear in git state.

Write to `.claude/MEMORY/SESSION_HANDOFF.md` (overwrite previous). Do not declare done yet if Codex review is required; continue to Phase 3.

## Phase 3 — Codex cross-review when in collab mode

Run this phase when the user asks for Codex review, when the session used ai-room collaboration, or when the handoff records Codex-owned / dual-surface work. Skip only for Claude-only sessions with no Codex or ai-room state.

Post the drafted handoff to Codex via ai-room before final confirmation. Ask Codex to review:

- uncommitted-state classification: session-critical vs supporting vs runtime/cache vs parallel/upstream
- commit coverage, especially Codex-authored or dual-surface work
- pending ai-room tasks and ownership state
- next-step priority and blockers
- in-flight evals, daemons, logs, and process state

Apply factual corrections or explicitly record why they were declined. Codex reviews the final handoff; Codex does not spawn subagents unless the user explicitly asks. After review is resolved, confirm to the user what was captured and name any remaining uncommitted risk.

## Document Structure

```markdown
# Session Handoff — [YYYY-MM-DD] (session subject)

## Goal
What the user was trying to achieve. Be specific.

## Completed (N commits, <SHA_first> → <SHA_last>)
- Concrete deliverables with file paths and function names
- Commits organized by subsystem if the session touched multiple
- Eval deltas / bench numbers VERBATIM from transcript where relevant
- Key decisions + WHY (not just what)

## In Progress
- Partially complete work — what's done, what remains
- Current state (e.g., "training running but OOM on 0.8B")

## ⚠ Uncommitted
`git status --short` output, annotated:
- Files that ARE session work, unintentionally uncommitted (FIX on resume)
- Files modified by tooling / other agents / upstream (verify before touching)
- Gitignored caches / build artifacts (safe to ignore)

## Next Steps
1. Ordered by priority with rationale (not just "continue X")
2. Include specific file paths, function names, expected branch of behavior
3. Flag blockers and open policy questions explicitly

## Key Context
- Discoveries, gotchas, patterns that save time if known upfront
- Failed approaches (so future-you doesn't retry — cite SHAs)
- Methodology caveats (single-run, unverified assumption, etc.)
- Hardware / environment state at session end

## Files in Project (session-shipped)
- `path/to/file.py` — brief description of what it does
- Group by: new files / modified code / modified docs
```

## Rules

- **Ground truth first** — git diff + transcript before writing, never from memory alone
- **Verbatim numbers** — copy bench / eval numbers exactly from transcript; don't round or paraphrase
- **Always overwrites** — `SESSION_HANDOFF.md` is always replaced, not appended
- **Never claim "all committed"** without `git status --short` returning clean for session work. Flag uncommitted session files explicitly with risk level.
- **Capture reasoning, not just outcomes** — WHY decisions were made (usually in transcript), not just WHAT
- **Include failures** — failed approaches are as valuable as successes; cite the null-result commit SHAs
- **Cross-reference narrative against agents** — if the narrative says "X happened" but neither agent found it in transcript or code, flag as unverified
- **Codex review before done when in collab mode** — if the session used ai-room or the user asked for Codex review, complete Phase 3 before final user confirmation
- **No fluff** — every line should help future-you resume faster
- **Preserve historical receipts** — if a handoff claim is falsified later, a `/update` pass adds a postscript. Handoff rewrites are for session-end only.
