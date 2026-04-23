Audit the project and rewrite `.claude/CLAUDE.md`, `.claude/rules/`, and `.claude/MEMORY/SESSION_HANDOFF.md` (or equivalent) to reflect the current state. REWRITE, not append — replace stale information.

Project-agnostic skeleton. Use as-is for any codebase with `.claude/CLAUDE.md` + `.claude/rules/` + a session-memory directory. Projects with deeper conventions (e.g. substrate daemons, specific kernel stacks, custom verification tooling) should layer a project-specific `/update` on top that references this one.

## When to use

Non-trivial update: session touched >1 subsystem, >3 commits, or introduced a new mechanism. Fall through to inline work for single-file trivial fixes.

## Phase 0 — contamination check (run FIRST)

Memory architecture splits content by *currency*: `.claude/rules/` =
current invariants, `.claude/MEMORY/atlas/<topic>_arc.md` = receipts,
ruled-out experiments, dated measurements, per-round arcs. If a rules
file already contains receipts, this `/update` will pile new ones on
top. Migrate first.

```bash
# Flag rules files contaminated with receipts.
# Uses PCRE (-P) for negative lookahead on dated file paths.
grep -nP '\b(R\d+(\.\d+)?|R-[a-z]+-\d+)\b|\b[a-f0-9]{7,}\b|\b20\d{2}-\d{2}-\d{2}(?![_A-Za-z0-9])|\b[Ss]ession \d+' \
  .claude/rules/*.md
```

Regex parts (keep in sync across Phase 0 + Phase 5):
- `R\d+(\.\d+)?` — round numbers: `R22`, `R47.2`
- `R-[a-z]+-\d+` — namespaced rounds: `R-delta-5`
- `[a-f0-9]{7,}` — commit SHAs (word-bounded)
- `20\d{2}-\d{2}-\d{2}(?![_A-Za-z0-9])` — bare dates; lookahead
  excludes eval-file paths like `2026-04-07_NAME.md`
- `[Ss]ession \d+` — session-numbering references

Benign hits excluded by design: capitalized model names (`Reasoning-*`),
dated eval file paths, hex-substring-in-word (`feedback`, `metafacade`).

For each hit: extract the contaminating block → move to
`.claude/MEMORY/atlas/<topic>_arc.md` → leave a single cross-ref at
the top of the rule:

```markdown
> Historical receipts: see `MEMORY/atlas/<topic>_arc.md`.
```

Optional preload gate (project-specific):

```bash
python3 scripts/measure_preload.py --max-tokens 15000
```

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
  - **misclassified by location** (content is a receipt — R-number, SHA, dated bench, ruled-out arc — but lives in `rules/`. Migrate to `MEMORY/atlas/<topic>_arc.md`, leave one-line cross-ref in rule)
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
- **Eager-tier line caps**: rules ≤ 150 lines target / 200 hard cap. Atlas unbounded. If a rule pushes past cap, carve receipts to atlas — DO NOT split into `_part_1/_part_2`.

## Phase 5 — fail-closed verification

- `grep -r "<stale claim>" .claude/` returns zero hits
- `grep -r "<new mechanism name>" .claude/` hits all expected files
- **Receipt contamination check** (must return zero hits in `rules/`):
  ```bash
  grep -nP '\b(R\d+(\.\d+)?|R-[a-z]+-\d+)\b|\b[a-f0-9]{7,}\b|\b20\d{2}-\d{2}-\d{2}(?![_A-Za-z0-9])|\b[Ss]ession \d+' \
    .claude/rules/*.md && echo "FAIL: rules/ contains receipts — migrate to MEMORY/atlas/"
  ```
- **Line-cap check**:
  ```bash
  for f in .claude/CLAUDE.md .claude/rules/*.md; do
    lines=$(wc -l <"$f")
    [ "$lines" -gt 200 ] && echo "OVER: $f: $lines lines (cap 200)"
  done
  ```
- Spot-read 2-3 edited files for coherent integration, not tacked-on appendices
- Re-run `git status --short` — every session file accounted for in handoff

If a finding is lost, add it OR explicitly note "see `<script>:<line>`" in a rule.

## Core rules

- **Rules files must not contain receipts.** A file in `.claude/rules/`
  MUST NOT contain: R-numbers (`R\d+`), commit SHAs (7+ hex chars),
  dated measurements (`YYYY-MM-DD`), bench tables, "Historical ships" /
  "Cancelled" / "Ruled-out" / "Per-round arc" subsections. These belong
  in `.claude/MEMORY/atlas/<topic>_arc.md`, linked from the top of the
  rule with one line: `> Historical receipts: see MEMORY/atlas/<topic>_arc.md`.
  This is the load-bearing discipline that keeps eager-tier preload
  bounded — Phase 0 grep-checks before any audit.
- **Eager-tier line caps**:
  - `.claude/rules/*.md`: target ≤ 150 lines, hard cap 200. Past cap →
    carve receipts to atlas. DO NOT split into `_part_1/_part_2`.
  - `.claude/MEMORY/atlas/*.md`: unbounded.
- **Replace, don't append.** Change the number. Don't add a note.
- **Delete dead info.** Remove or mark as planned.
- **Verify claims, don't trust docs.** Read actual files.
- **New rule files are fine.** Pair every new rule with a sibling `MEMORY/atlas/<name>_arc.md` for receipts.
- **Capture invariants, not just facts.** For fragile areas (flag state, init order, side-effecting imports, hardcoded limits), add explicit "invariant" rules explaining what NOT to do and why. **Cite the fixing commit in the atlas entry, not in the rule.**
- **Cite commits in atlas, not rules.** Format: `` (commit `c11232a`) `` belongs in `MEMORY/atlas/<topic>_arc.md`. The rule references the atlas with a one-line cross-ref.
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
