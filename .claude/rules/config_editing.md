# Config `.claude/` Editing Directive

Rules for editing anything under `.claude/` (agents, CLAUDE.md, commands, rules, hooks).

## Style

- **Preserve structure** → Match existing formatting (bullets, sections, headers)
- **Match tone** → Imperative, terse, no fluff ("Do X" not "You should consider doing X")
- **Add value** → Every word must serve purpose (examples only if essential)
- **Maintain style & patterns** → Use existing conventions
- **No duplication** → Don't repeat information already present elsewhere
- **Verify integration** → New content must flow naturally with surrounding text

## Eager-tier line caps

- `.claude/rules/*.md`: **target ≤ 150 lines, hard cap 200**. Past cap →
  carve receipts to `MEMORY/atlas/<topic>_arc.md`. **DO NOT split into
  `_part_1/_part_2`** — splitting doesn't reduce eager-tier preload, it
  just hides the bloat.
- `.claude/CLAUDE.md`: target ≤ 100 lines (it's the manifest, not a doc).
- `.claude/MEMORY/atlas/*.md`: unbounded (query-triggered, not preloaded).

## Currency split — current invariants vs historical receipts

`.claude/rules/` files are auto-loaded into every session. They earn
that real-estate by being **current invariants** — how we do things
NOW. Receipts of how we got here belong in
`.claude/MEMORY/atlas/<topic>_arc.md` (query-triggered).

**Banned in `rules/*.md`** (these belong in atlas):
- R-numbers (`R\d+`, `R-delta-N`)
- Commit SHAs (7+ hex chars)
- Dated measurements (`YYYY-MM-DD` form, "session N", any specific calendar date)
- Bench tables (per-N speedup tables, per-round result tables)
- "Historical ships" / "Cancelled" / "Ruled out" / "Per-round arc" subsections
- "Why X not Y" forensics (the answer is current; the forensics is archive)

**Cross-ref pattern** — at the top of any rule with a companion atlas:

```markdown
> Historical receipts: see `MEMORY/atlas/<topic>_arc.md`.
```

**New rule files**: pair every new `rules/<name>.md` with a sibling
`MEMORY/atlas/<name>_arc.md` for receipts that justified it. Cite
commits in the atlas entry; reference the atlas from the rule with a
one-line pointer, not inline citations.

**Mid-session edits**: if you're tempted to add an R-number / commit
SHA / "we tried X and it didn't work" to a rules file — that's a
receipt. Append it to the matching atlas instead, and update the
rule only if a current invariant changed.

`/update` Phase 0 enforces this via grep against `rules/*.md`. Phase 5
fail-closed via `python3 scripts/measure_preload.py --max-tokens 15000`.
Don't subvert by going inline.

## Related rules

- `CLAUDE.md` — top-level index that points to every rule in this directory
- `.claude/commands/update.md` — `/update` workflow + Phase 0 contamination check + Phase 5 gates
- `workflow.md` §"Commit discipline" — commit-message conventions that apply to rule edits too
