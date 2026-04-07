Audit the project and rewrite `.claude/CLAUDE.md` and `.claude/rules/` files to reflect the current state of the codebase. This is a REWRITE, not an append — replace stale information with accurate information.

## Process

1. **Read recent commits FIRST** (before docs):
   - `git log --oneline -20` to see recent activity
   - For each non-trivial commit since the last `/update` or `/handoff`, read the message and `git show <hash>`
   - Extract: invariants exposed by fixes, sampling/config changes, environment gotchas, "wrong premise" claims that recent fixes have corrected
   - Note commit hashes — cite them in doc updates

2. **Read current docs**: Read `.claude/CLAUDE.md` and all files under `.claude/rules/`. Flag claims that might be stale or wrong-premise based on Phase 1.

3. **Audit the codebase**:
   - `agents/` — count files, check classes, verify tools list, harness commands
   - `agents/distill/` — pipeline status, training data counts, model configs
   - `models/` — list Modelfiles, verify parameters match reality
   - `rust/`, `src/`, `.gitignore` — verify against docs
   - **Hardcoded runtime constants** — grep `agents/agent.py` and friends for sampling params (`temperature`, `frequency_penalty`, `presence_penalty`, `max_tokens`), timeouts, retry counts, thresholds. Verify current values are documented.
   - **Load-bearing invariants** — for each recent bug fix, identify what the fix forbids ("don't reset X here", "don't call Y before Z") and check whether a doc rule captures it. Easy-to-reintroduce bugs NEED explicit invariant rules.

4. **Verify external runtime state** (don't trust docs — verify the running environment):
   - Ollama: `curl -s localhost:11434/api/tags` — actual pulled models
   - llama.cpp: `curl -s localhost:8080/health` — server up?
   - GPU: `nvidia-smi` if VRAM/GPU claims are doc-relevant
   - Any other external service the docs claim about (RunPod, Colab, etc.)

5. **Classify drift** in each doc section:
   - **Stale**: was true, now isn't (e.g., model count changed) — fix the number
   - **Wrong-premise**: based on a misunderstanding now corrected (e.g., "rare worst case" when it actually fired every time) — rewrite the framing, not just the value
   - **Missing**: invariant or constant exists in code but isn't documented — add a new rule

6. **Route findings to the right layer**:
   - Project facts, architecture, defaults, conventions → `CLAUDE.md` or `.claude/rules/*.md`
   - Environment quirks (WSL, OS, hardware, external service gotchas) → `.claude/rules/training.md` Known Issues
   - Personal debugging lessons or preferences → `~/.claude/projects/.../memory/` (NOT docs)
   - Heuristic: would another contributor cloning the repo benefit? → docs. Personal-only? → memory.

7. **Propose before editing**: Output the full list of proposed changes with `file:line` refs and one-line justifications. Wait for explicit user approval (`ok implement` / `do all` / `yes please`) before editing. This lets the user filter, reorder, or reject. Skip this gate only if invoked with explicit `--auto` intent.

8. **Rewrite in place**: Edit each section to match reality. Delete sections that no longer apply. Update numbers, file paths, patterns. Keep the same structure and voice — just make it true. **Cite commit hashes** for invariants and bug-fix rules.

9. **Report changes**: Summarize what was updated, in which file, and why. Include `git diff --stat .claude/`.

## Rules

- **Replace, don't append**. If a section says "5 tools" and there are now 7, change the number. Don't add a note.
- **Delete dead info**. If a feature doesn't exist yet, remove it or mark it as planned.
- **Keep it scannable**. Bullet-point style, same section headers where possible. Fast reference, not prose.
- **Verify claims**. Don't trust what the docs say — read the actual files. Line counts change, features evolve.
- **Update rules files too**. Check all rules under `.claude/rules/` against actual code state.
- **Capture invariants, not just facts**. For fragile areas (flag state, lock ordering, event sequencing, init order, side-effecting imports), add an explicit "invariant" rule explaining what NOT to do and why. Cite the fixing commit. Goal: a contributor reading the rule should be able to avoid reintroducing the bug.
- **Cite commits in doc updates**. When documenting a rule that came from a recent bug fix, include the commit hash inline (format: `` (commit `c11232a`) ``). Future readers can `git show <hash>` for full context.
- **Don't put project facts in memory**. Memory is for personal preferences and debugging lessons only. Project state, architecture, and conventions belong in `CLAUDE.md` or `.claude/rules/`.
- **Distinguish wrong-premise from stale**. A stale claim was true and is now false; a wrong-premise claim was never quite right because it was based on a misunderstanding. Wrong-premise claims need framing changes, not just value updates.
