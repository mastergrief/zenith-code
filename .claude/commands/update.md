Audit the project and rewrite both eager doc surfaces to reflect the current state: `.claude/CLAUDE.md` + `.claude/rules/` for Claude, and `.codex/AGENTS.md` + `.codex/rules/` for Codex. REWRITE, not append — replace stale information. Includes a final audit pass to catch implicit/partial captures before declaring done.

**Handoff rewriting is owned by `.claude/commands/handoff.md`, not here.** After `/update` ships Claude + Codex eager-doc updates, run `/handoff` to rewrite `.claude/MEMORY/SESSION_HANDOFF.md` with its own 2-agent grounding. The two commands compose — they don't duplicate. `/update` validates handoff coherence during its audit phase (e.g. uncommitted files are listed); handoff-content editing happens under `/handoff`. If a repo later adds a Codex handoff file, `/update` may validate that it is coherent, but it still does not own handoff prose.

## Default workflow — 3-agent audit → P0/P1/P2 plan → execute by tier → Codex cross-review

The default for a non-trivial update (session touched >1 subsystem, >3 commits, or introduced a new mechanism): Claude launches **3 Explore agents in parallel** for the audit (transcript / code / docs), classifies findings by priority tier, drafts a plan file in plan mode, then executes one tier per commit. Fall through to inline work only for single-file trivial fixes.

Codex review gate: when `/update` changes `.codex/*` or otherwise touches both eager surfaces, Codex does **not** run subagents. Codex performs the final cross-review of the proposed diff before commit; Claude resolves blockers or records Codex sign-off.

Case-study receipt: the 2026-04-20 fused flash-attn flip (commit `ad1469e`) used this exact 3-agent split. Agent 1 extracted verbatim bench numbers from the minutes transcript; Agent 2 mapped the flag + dispatch + N-gate feasibility to 5 specific line numbers; Agent 3 enumerated 14 doc locations across 7 files with fix-category tags. Synthesis + plan + commit took ~30 min end-to-end.

### Phase 0 — contamination check (run FIRST, before any audit)

Memory architecture splits content by *currency*: `.claude/rules/` and
`.codex/rules/` = current invariants; `.claude/MEMORY/atlas/<topic>_arc.md`
and `.codex/MEMORY/atlas/<topic>_arc.md` = receipts / ruled-out / dated
measurements / per-round arcs. If a rules file already contains
receipts, this `/update` will pile new ones on top of old contamination.
Migrate first, audit second.

```bash
# Flag rules files contaminated with receipts.
# Uses PCRE (-P) for negative lookahead on dated file paths.
grep -nP '\b(R\d+(\.\d+)?|R-[a-z]+-\d+)\b|\b[a-f0-9]{7,}\b|\b20\d{2}-\d{2}-\d{2}(?![_A-Za-z0-9])|\b[Ss]ession \d+' \
  .claude/rules/*.md .codex/rules/*.md
```

Regex parts (keep in sync across Phase 0 + Phase 5):
- `R\d+(\.\d+)?` — round numbers: `R22`, `R47.2`, `R50.6`
- `R-[a-z]+-\d+` — namespaced rounds: `R-delta-5`, `R-delta-22`
- `[a-f0-9]{7,}` — commit SHAs (word-bounded)
- `20\d{2}-\d{2}-\d{2}(?![_A-Za-z0-9])` — bare dates; lookahead
  excludes eval-file paths like `2026-04-07_qwen4b.md`
- `[Ss]ession \d+` — "session N" / "Session N" session-numbering

Known benign hits that should NOT be flagged:
- Model names starting with a capital R (e.g. `Reasoning-3000x`) —
  tight `\b(R\d+|R-word-\d+)\b` excludes them
- Dated eval file paths (`.claude/MEMORY/evals/YYYY-MM-DD_*.md`,
  `.codex/MEMORY/evals/YYYY-MM-DD_*.md`)
- Hex-in-word (`feedback`, `metafacade`) — word-bounded `\b...\b`
  excludes them

For each hit: extract the contaminating block (R-number paragraph,
SHA citation, dated bench, "Ruled out" / "Cancelled" / "Historical
ships" subsection) → move to the same side's
`MEMORY/atlas/<topic>_arc.md` (`.claude/MEMORY/atlas/...` for Claude
rules, `.codex/MEMORY/atlas/...` for Codex rules; create if missing) →
leave a single cross-ref line at the top of the rule:

```markdown
> Historical receipts: see `MEMORY/atlas/<topic>_arc.md`.
```

Also run the preload measurement gate:

```bash
python3 scripts/measure_preload.py --surface both --max-tokens 150000
```

If eager-tier exceeds the cap, the contamination migration above is
mandatory before proceeding.

### Phase 1 — parallel research (3 Explore agents)

**The brief IS the session context.** Agents are cold-started with zero
knowledge of the conversation. Before dispatching, run `git log --oneline
-20` + `git status --short`, skim session memory / handoff for ruled-out
paths, then write three focused briefs — one per agent — each containing:

- The subject area for this audit (what subsystem / mechanism / claim is in play)
- The session's shipped commits (SHA + 1-line summary) for the relevant changes
- Pointers to files / line ranges / transcript sections the agent should read (not pasted content)
- Return format explicitly specified (punch-list / prioritized-list / file:line inventory)

Realistic brief size: 300-500 words per agent. Each agent owns ONE
domain — don't blur scopes. Synthesis happens in main context.

**Agent 1 — transcript + measurements**
- Read session minutes (`.claude/MEMORY/minutes/*.md`), handoff, Codex atlas receipts when relevant (`.codex/MEMORY/atlas/*.md`), and full commit bodies for session-tagged perf / fix / eval commits
- Extract verbatim numbers, decisions, caveats — no summarization where raw data is needed (bench tables, eval deltas, measured failure rates)
- Flag ruled-out paths, user corrections, methodology caveats (single-run vs median-of-5, etc.)
- Return: ≤ 300 words including a verbatim numbers block, decision log, and any explicitly-deferred policy choices

**Agent 2 — code surface**
- Walk modified / new source files listed in the brief
- Surface: flags + defaults, dispatch conditionals, invariants, shape-heuristic constants
- Locate gating / threshold points for any policy the update will touch
- Return per-site: `file:line` refs, 3-5 lines of context, and (if a change is planned) the cleanest spot to make it
- ≤ 400 words

**Agent 3 — docs inventory**
- Read `.claude/CLAUDE.md` + every `.claude/rules/*.md` + `.codex/AGENTS.md` + every `.codex/rules/*.md` + `.claude/MEMORY/SESSION_HANDOFF.md`
- Exhaustive search for every claim/number/policy statement related to the subject area
- Per hit: `file:line`, exact quote (3-5 lines context), **fix category** — one of:
  - **correct the claim** (was wrong, rewrite)
  - **tighten / add gate** (still mostly right, needs qualifier)
  - **keep as historical receipt** (session-specific, mark PARTIALLY SUPERSEDED not delete)
  - **misclassified by location** (content is a receipt — R-number, SHA, dated bench, ruled-out arc — but lives in `rules/`. Migrate to `MEMORY/atlas/<topic>_arc.md`, leave one-line cross-ref in rule)
  - **new section** (mechanism is new, needs fresh doc)
- Group findings by file; include edit-priority ordering
- ≤ 500 words

### Phase 2 — synthesize and classify

Merge the three agent outputs. Cross-reference: transcript findings (Agent 1) should ground claims in measurements; code-surface findings (Agent 2) should identify the edit sites; docs-inventory findings (Agent 3) should map every downstream update. Classify each finding:

| Tier | Definition | Examples |
|---|---|---|
| **P0** | Falsifies a standing claim in CLAUDE.md, AGENTS.md, or rules OR encodes a measurement/discipline lesson that cost ≥ 2 null rounds | Substrate thesis reframed, sandbox bug, MAX_TOKENS starvation |
| **P1** | Kernel / infrastructure / architectural change that belongs in rules but isn't ship-blocking | New kernel variant, new fused path, storage contract change |
| **P2** | Next-session research signal — failure modes observed, tier-3 targets, ruled-out patterns for the log | Gemma-ignores-hints, confidence-gate null, new PT ceiling |

### Phase 3 — plan mode

Write a plan file at the path given by the plan-mode system. Structure:

- **Context** (why — problem and intended outcome)
- **Scope** — files to edit, grouped by P0 / P1 / P2
- **Critical source files to reference** (code paths + short SHAs for traceability)
- **Execution approach** — tier-order, one commit per tier, surgical Edit not Write, 500 LOC rule per file
- **Verification** — grep checks + line-count checks that a post-edit run should pass

Call `ExitPlanMode` when ready; do not ask "is this ok?" in prose.

### Phase 4 — execute by tier

- One P-tier per commit (3 commits total for a full-scope session). Before committing any tier that edits `.codex/*` or both eager surfaces, post the proposed diff to Codex for cross-review and resolve blockers. Each commit message cites the receipts (commits from this session, eval deltas, null-round counts).
- Use `Edit`, not `Write` — preserve structure, tone, and terse imperative voice.
- Match existing section depth / bullet style / table format.
- **Eager-tier line cap**: rules files target ≤ 150 lines, hard cap 200. Atlas files unbounded (query-triggered). If a new section pushes a rule past cap, carve receipts to atlas/ — don't split into `_part_1/_part_2`.

### Phase 5 — verification (fail-closed)

Run every verification check listed in the plan file. Typical set:
- `grep -r "<stale claim>" .claude/ .codex/` returns zero hits
- `grep -r "<new mechanism name>" .claude/ .codex/` hits all expected files
- **Receipt contamination check** (must return zero hits in `rules/`):
  ```bash
  grep -nP '\b(R\d+(\.\d+)?|R-[a-z]+-\d+)\b|\b[a-f0-9]{7,}\b|\b20\d{2}-\d{2}-\d{2}(?![_A-Za-z0-9])|\b[Ss]ession \d+' \
    .claude/rules/*.md .codex/rules/*.md && echo "FAIL: rules/ contains receipts — migrate to MEMORY/atlas/"
  ```
- **Eager-tier line cap** (200 hard cap):
  ```bash
  for f in .claude/CLAUDE.md .claude/rules/*.md .codex/AGENTS.md .codex/rules/*.md; do
    lines=$(wc -l <"$f")
    [ "$lines" -gt 200 ] && echo "OVER: $f: $lines lines (cap 200)"
  done
  ```
- **Eager-tier token gate** (must pass):
  ```bash
  python3 scripts/measure_preload.py --surface both --max-tokens 150000
  ```
- Spot-read 2-3 edited files for coherent integration, not tacked-on appendices

If verification fails, fix before declaring done. If a finding is lost, add it OR explicitly note "see <script>:<line>" in a rule.

---

## Underlying discipline (feeds into Phase 1-2 briefs)

1. **Read recent commits + session log FIRST** (before docs):
   - `git log --oneline -20` — recent activity
   - For each non-trivial commit since the last `/update` or `/handoff`, read message + `git show <hash>`
   - **If a session transcript exists** (e.g., `.claude/MEMORY/Augment-notes.md`, `.claude/MEMORY/session_log.md`), read it. Session logs capture findings that never made it to commits. Extract: invariants exposed, findings, design pivots, ruled-out paths.
   - **Check `git status --short` for uncommitted session work** — these are at risk until committed; they MUST be flagged in the handoff even if you don't commit them.
   - Note commit hashes — cite them in doc updates.

2. **Read current docs**: `.claude/CLAUDE.md`, all `.claude/rules/*.md`, `.codex/AGENTS.md`, all `.codex/rules/*.md`, `.claude/MEMORY/SESSION_HANDOFF.md`. Flag claims that look stale or wrong-premise.

3. **Audit the codebase**:
   - `agents/`, `agents/distill/`, `calm/`, `scripts/`, `models/`, `rust/`, `src/` — verify counts/classes/tools against docs.
   - **Hardcoded runtime constants** — grep for sampling params (`temperature`, `frequency_penalty`, `presence_penalty`, `max_tokens`), timeouts, retry counts, thresholds, magic numbers.
   - **Load-bearing invariants** — for each recent bug fix, identify what the fix forbids ("don't reset X", "don't call Y before Z") and check for a doc rule capturing it. Easy-to-reintroduce bugs NEED explicit invariant rules.
   - **New code this session** — files in `git status` not-yet-committed with non-trivial content need a rule entry OR explicit handoff note.

4. **Verify runtime state** (don't trust docs — verify the running environment):
   - **Gemma substrate daemon**: `bin/gemma-run --status` — primary inference path, PyTorch + Triton stack via `bin/gemma_daemon.py` + `calm/llm_computer/gemma_substrate.py`. Check PID, max_len config, GPU residency (`nvidia-smi` for VRAM accounting).
   - **GPU**: `nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader` — verify VRAM claims match doc. Substrate baseline is ~5.07 GB for Gemma 4 E4B tq4 + Q6_K embd.
   - **Triton kernels**: `calm/llm_computer/tq4_triton.py` — verify shape heuristics + BLOCK_M selections haven't drifted from doc (re-run `test_tq4_triton` if spec'd).
   - **Indices + caches**: `ls .cache/r53_code_db/` — retrieval index state (tfidf.json, dense.pt, dense.tq4.pt).
   - **External services docs claim about** (RunPod, Colab, etc.) only if currently relevant; note: llama.cpp and Ollama are not used — do NOT add verify steps for them unless the project pivots back.

5. **Classify drift** in each doc section:
   - **Stale**: was true, now isn't (e.g. model count changed) — fix the number
   - **Wrong-premise**: based on a misunderstanding now corrected — rewrite the framing, not just the value
   - **Missing**: invariant / constant / mechanism exists in code but isn't documented — add a new rule
   - **New architectural mechanism** (distinct enough from existing rules) — create new side-appropriate rule files (`.claude/rules/*.md`, `.codex/rules/*.md`, or both). Don't force-fit it into an existing one that's off-topic.

6. **Route findings by *currency* AND *layer***:

   First decide *currency* — is this how we do it NOW, or a receipt of how we got here?

   | Finding type | Destination |
   |---|---|
   | **Current invariant / API / default** (the *rule*) | Side-appropriate eager rule: `.claude/rules/<topic>.md`, `.codex/rules/<topic>.md`, or both |
   | **Receipt: R-number, commit SHA, dated bench, ruled-out experiment, cancelled arc, "Historical ships"** | Same side's `MEMORY/atlas/<topic>_arc.md` (query-triggered). Cross-ref from rule with `> Historical receipts: see MEMORY/atlas/<topic>_arc.md`. |
   | **Per-round arc / null log** | Same side's `MEMORY/atlas/<topic>_arc.md` |
   | New architectural mechanism (current API + invariants only) | New side-appropriate `rules/<name>.md` + companion `MEMORY/atlas/<name>_arc.md` for the receipts that justified it |
   | Extension of existing concept (current state) | Edit existing rule; receipts of the change → atlas |
   | Strategic synthesis / thesis-level claims | `augmentation_thesis.md` or `commercial.md` (current framing); per-round derivation → atlas |
   | Environment quirks (WSL, OS, hardware, service gotchas) | `.claude/rules/training.md` §Known Issues and mirrored `.codex/rules/training.md` if Codex must follow it |
   | Session-specific state, mid-flight work, uncommitted files | `.claude/MEMORY/SESSION_HANDOFF.md` |
   | Personal debugging lessons / preferences | `~/.claude/projects/.../memory/` (NOT docs) |
   | Conversation transcript | `.claude/MEMORY/Augment-notes.md` or similar — preserve as-is, reference from handoff; add `.codex/MEMORY/atlas/*` only for distilled Codex-facing receipts |

7. **Handoff rewriting is delegated to `.claude/commands/handoff.md`.** Do NOT duplicate handoff structure or authorship discipline here. If this session was non-trivial (per the `/handoff` gate: >3 commits, >1 subsystem, new mechanism, or transcript >30K tokens), run `/handoff` after `/update` completes. The two commands compose — `/update` owns `.claude/CLAUDE.md`, `.claude/rules/`, `.codex/AGENTS.md`, and `.codex/rules/`; `/handoff` owns `SESSION_HANDOFF.md`.

8. **Propose before editing**: Output the full list of proposed changes with `file:line` refs + one-line justifications. Wait for explicit approval (`ok implement` / `do all` / `yes please`) before editing. Use plan mode if the session is substantial — writes a plan file the user can review. Skip this gate only with explicit `--auto` intent.

9. **Rewrite in place**: Edit each section to match reality. Delete sections that no longer apply. Cite commit hashes inline (format: `` (commit `c11232a`) ``).

10. **FINAL AUDIT — did we lose anything?** After writing all docs, do a second pass:
    - **Uncommitted-files check**: re-run `git status --short`. Verify every untracked + modified file from the session is reflected in the current `SESSION_HANDOFF.md`'s ⚠ UNCOMMITTED section. If missing, flag for the next `/handoff` run — do NOT hand-edit handoff content from `/update`.
    - **Transcript diff**: if a session log exists, scan it for topics/findings that don't appear in the new docs. Specifically check for:
      - User questions that led to architectural insights ("what is X?", "why doesn't Y?")
      - Debugging steps / fixes that exposed invariants
      - Performance discoveries (e.g., trie speedup, dequant batching)
      - Eval results partial vs final
      - User-suggested directions / decisions that weren't just "ok proceed"
    - **Implicit-capture flag**: findings that live ONLY in code comments or script internals aren't captured. Either document or explicitly note "see <script>:<line>" in a rule.
    - **Modified-file check**: verify every `M` file in `git status` has a reason to be modified — `bin/gemma_daemon.py` max_len change, `fetch_datasets.py` output renamed, etc. Flag unexpected modifications.
    - **Report findings gap**: if the audit finds something missed, add it OR explicitly note "session log §X has additional context on Y; rule intentionally omits for brevity".

11. **Report changes**: Summarize updated files + `git diff --stat .claude/ .codex/`. Include explicit sentence "final audit found [N] gaps, all addressed / documented as session-log references."

## Rules

- **Rules files must not contain receipts.** A file in `.claude/rules/`
  or `.codex/rules/` MUST NOT contain: R-numbers (`R\d+`), commit SHAs (7+ hex chars),
  dated measurements (`YYYY-MM-DD`), bench tables, "Historical ships" /
  "Cancelled" / "Ruled-out" / "Per-round arc" subsections. These belong
  in the same side's `MEMORY/atlas/<topic>_arc.md`, linked from the top
  of the rule with one line: `> Historical receipts: see MEMORY/atlas/<topic>_arc.md`.
  This is the load-bearing discipline that keeps eager-tier preload
  bounded — it's why `/update` Phase 0 grep-checks before any audit.
- **Eager-tier line caps**:
  - `.claude/rules/*.md` and `.codex/rules/*.md`: target ≤ 150 lines,
    hard cap 200. If past cap, carve receipts to atlas — DO NOT split
    into `_part_1/_part_2`.
  - `.claude/MEMORY/atlas/*.md` and `.codex/MEMORY/atlas/*.md`: unbounded
    (query-triggered, not preloaded).
- **Replace, don't append**. If a section says "5 tools" and there are now 7, change the number. Don't add a note.
- **Delete dead info**. If a feature doesn't exist yet, remove it or mark it as planned.
- **Keep it scannable**. Bullet-point style, same section headers where possible.
- **Verify claims**. Don't trust what the docs say — read the actual files. Line counts change, features evolve.
- **Update rules files too**. Check all rules under `.claude/rules/` and `.codex/rules/` against actual code state.
- **New files are fine**. Creating a new side-appropriate `rules/<name>.md` is preferred over cramming a new mechanism into an unrelated file. Pair every new rule with a sibling `MEMORY/atlas/<name>_arc.md` for receipts.
- **Capture invariants, not just facts**. For fragile areas (flag state, lock ordering, event sequencing, init order, side-effecting imports, hardcoded limits), add explicit "invariant" rules explaining what NOT to do and why. **Cite the fixing commit in the atlas entry, not in the rule.**
- **Cite commits in atlas, not rules**. Citation format: `` (commit `c11232a`) `` belongs in `MEMORY/atlas/<topic>_arc.md`. The rule references the atlas: `See MEMORY/atlas/<topic>_arc.md for the R22f recalibration receipt.`
- **Don't put project facts in memory**. Memory = personal preferences + debugging lessons. Project state / architecture / conventions → `.claude/CLAUDE.md`, `.codex/AGENTS.md`, `.claude/rules/`, or `.codex/rules/`.
- **Distinguish wrong-premise from stale**. Stale was true and is now false; wrong-premise was never quite right. Wrong-premise needs framing changes, not just value updates.
- **Session-log extraction before rule-writing**. If a transcript exists, findings there are richer than docs. Pull from transcript first, reconcile with code + commits second, write rules third.
