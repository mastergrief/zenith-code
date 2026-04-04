Create a detailed session handoff document so a future Claude Code session can seamlessly continue.

**Do NOT spawn subagents** — you already have the context, they don't.

## Steps

1. **Gather ground truth** (reduces hallucination):
   - `git diff --stat` — files changed this session
   - `git log --oneline -10` — recent commits
   - Read current `.claude/MEMORY/SESSION_HANDOFF.md` if it exists for prior context
2. **Reflect** on the full conversation: goal, progress, decisions, blockers, failures
3. **Write** to `.claude/MEMORY/SESSION_HANDOFF.md` (overwrite previous)
4. **Confirm** to the user what was captured

## Document Structure

```markdown
# Session Handoff — [YYYY-MM-DD]

## Goal
What the user was trying to achieve. Be specific.

## Completed
- Concrete deliverables with file paths and function names
- Key decisions made and WHY (not just what)
- Any commits pushed, branches created

## In Progress
- Partially complete work — what's done, what remains
- Current state (e.g., "training running but OOM on 0.8B")

## Next Steps
1. Ordered by priority
2. Include specific file paths and approach notes
3. Flag any blockers or open questions

## Key Context
- Discoveries, gotchas, or patterns that save time if known upfront
- Failed approaches (so future-you doesn't retry them)
- Hardware/environment state

## Files in Project
- `path/to/file.py` — brief description of what it does
```

## Rules
- **Ground truth first** — git diff before writing, never from memory alone
- **Be specific** — file paths, function names, line numbers where relevant
- **Capture reasoning** — WHY decisions were made, not just WHAT
- **Include failures** — failed approaches are as valuable as successes
- **Always overwrites** — SESSION_HANDOFF.md is always replaced, not appended
- **No fluff** — every line should help future-you resume faster
