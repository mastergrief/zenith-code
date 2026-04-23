# Config `.claude/` Editing Directive

Rules for editing anything under `.claude/` (agents, CLAUDE.md, commands, rules, hooks):

- **Preserve structure** → Match existing formatting (bullets, sections, headers)
- **Match tone** → Imperative, terse, no fluff (e.g., "Do X" not "You should consider doing X")
- **Add value** → Every word must serve purpose (examples only if essential)
- **No verbosity** → 500 lines is hard limit, 250-500 is sweet spot. Be concise without losing context.
- **Maintain style & patterns** → Use existing conventions
- **No duplication** → Don't repeat information already present elsewhere
- **Verify integration** → New content must flow naturally with surrounding text

## Related rules

- `CLAUDE.md` — top-level index that points to every rule in this directory
- `workflow_part_1.md` §"Commit discipline" — commit-message conventions that apply to rule edits too
