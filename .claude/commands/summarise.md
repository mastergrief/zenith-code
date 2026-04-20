Read the latest session handoff + current repo state and produce a resume-brief for continuing work.

**Do NOT spawn subagents** — reading + a few bash commands is cheaper and keeps the user in control of what gets spot-checked.

## Steps

1. **Read** `.claude/MEMORY/SESSION_HANDOFF.md` in full (the path `/handoff` writes to). If missing, fall back to `git log -20 --oneline` + `git status -s` and note no handoff exists.
2. **Ground against current tree** — run in parallel:
   - `git rev-parse --abbrev-ref HEAD` — current branch
   - `git log @{upstream}..HEAD --oneline 2>/dev/null | head -20` — commits ahead of upstream (fail-quiet if no upstream set)
   - `git status -s` — uncommitted state
   - `git log -10 --oneline` — most recent commits (cross-check SHAs against handoff)
3. **Check background services** (non-blocking, fail-quiet) — only if the handoff or repo mentions them. Common patterns:
   - Local HTTP service on a port the handoff names: `curl -s -m 2 localhost:<port>/health 2>&1 || echo "<name> off"`
   - GPU state if the project uses one: `nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader 2>&1 | head -1`
   - Skip this step entirely if the handoff mentions no services.
4. **Synthesise** a brief with these sections:
   - **Last session goal** — one line from handoff.
   - **Shipped** — handoff's Completed bullets, with SHAs confirmed against `git log`. Flag any SHA in handoff not found in log.
   - **Paused / uncommitted** — handoff's In Progress cross-checked with `git status -s`. Note any new untracked files since handoff was written.
   - **Next steps** — verbatim from handoff's ranked list, preserving the prior session's priority order.
   - **Verify-against-tree checklist** — 2-5 load-bearing claims from the handoff the user should spot-check before acting (e.g. specific flag values, file states, env state). Surface them; don't auto-verify.
5. **Stop** — do not start any next-step work. Wait for the user to direct.

## Rules

- **Static read + git ground truth only** — no exhaustive grep or rule-file audit. The goal is orientation, not re-verification.
- **Project-agnostic** — don't assume branch names, service ports, GPU presence, or tooling. Let the handoff tell you what to check; fall back cleanly when signals are absent.
- **Flag drift** — any handoff claim contradicted by `git log` / `git status` gets called out explicitly, not quietly reconciled.
- **Preserve handoff ordering** — don't re-rank next steps; the prior session committed to a priority order.
- **Checklist over auto-action** — surface the 2-5 claims for the user to verify rather than grepping the tree yourself (cost control + user keeps judgment on which matter).
- **Under 400 words output** — this is a brief, not a rewrite of the handoff.
- **No proactive fixes** — if the handoff has gaps or stale claims, note them in the checklist; don't edit handoff or code until asked.
