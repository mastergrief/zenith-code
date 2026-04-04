Audit the project and rewrite `.claude/CLAUDE.md` and `.claude/rules/` files to reflect the current state of the codebase. This is a REWRITE, not an append — replace stale information with accurate information.

## Process

1. **Read current docs**: Read `.claude/CLAUDE.md` and all files under `.claude/rules/`
2. **Audit the codebase**: Check the actual state against what the docs claim:
   - `agents/` — count files, check classes, verify tools list, check harness commands
   - `agents/distill/` — check pipeline status, training data counts, model configs
   - `models/` — list Modelfiles, verify parameters match reality
   - `rust/` — check if Rust crate structure matches docs
   - `src/` — check Python source parity status
   - `.gitignore` — verify exclusions are correct
   - Run `ollama list` to verify available models
3. **Identify drift**: List what's wrong or outdated in each doc section
4. **Rewrite in place**: Edit each section to match reality. Delete sections that no longer apply. Update numbers, file paths, patterns. Keep the same structure and voice — just make it true.
5. **Report changes**: Summarize what was updated and why

## Rules

- **Replace, don't append**. If a section says "5 tools" and there are now 7, change the number. Don't add a note.
- **Delete dead info**. If a feature doesn't exist yet, remove it or mark it as planned.
- **Keep it scannable**. Bullet-point style, same section headers where possible. Fast reference, not prose.
- **Verify claims**. Don't trust what the docs say — read the actual files. Line counts change, features evolve.
- **Update rules files too**. Check all rules under `.claude/rules/` against actual code state.
