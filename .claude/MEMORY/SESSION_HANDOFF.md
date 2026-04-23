# Session Handoff — 2026-04-23 evening (RENAME N=50 ship, VGSL critique + Stage 1/2 reframe + park reco)

## Goal

Resume from morning handoff (N=50 MBPP A/B eval running on daemon PID 540958 mid-flight at 44/50, VGSL spec untouched since `c98a2a1` greenfield framing). Land the eval verdict + receipt; do the cleanup commit the prior handoff scoped; act on user direction to first-principles-review VGSL with codex; then on user reframe ("VGSL as in-tree drop-in upgrade") run a focused design discussion and land the spec update; then give a clear "what do you think" recommendation; then surface the next research-line of pursuit.

By session end: RENAME-canonical-for-MBPP confirmed at N=50 scale; VGSL spec rewritten with joint-critique findings + Stage 1/2 in-tree pursuit framing; claude's lean = **park VGSL with explicit revisit triggers**; next research line surfaced = **extend RENAME to HumanEvalPlus / BigCodeBench**.

## Completed (3 current-repo session commits, `1e0680e` → `94767db`)

Session-local commits on `feature/multi-agent-qwen` (chronological):
`1e0680e`, `362e5a5`, `94767db`. No sister-repo activity this session.

### N=50 RENAME-vs-DT eval ship

- **`1e0680e`** — `.claude/MEMORY/evals/2026-04-23_dt_rename_n50.md` (108 lines). Receipt for the N=50 scale-up of `dac50ed`. Verdict per matrix: **dac50ed stands**.

  | Method | Known (1-20) | Novel (21-50) | Total |
  |---|---|---|---|
  | stock | 9/60 = 15.00% | 0/90 = 0.00% | 9/150 = 6.00% |
  | dt-bias | 15/60 = 25.00% | 23/90 = 25.56% | 38/150 = 25.33% |
  | **rename** | **17/60 = 28.33%** | **36/90 = 40.00%** | **53/150 = 35.33%** |

  - Δ rename-vs-dt: **+15 cells (+9.99pp) total**, **+13 cells (+14.4pp) novel-only**
  - Δ rename-vs-stock: +44 cells (+29.33pp)
  - **RENAME 15 wins / 0 regressions** vs DT 12 wins / **1 regression** (`remove_kth_element` 3/3 → 0/3, arity hallucination — DT predicted `(list1, list2)`, actual `(list, K)`)
  - Overlap: 8 problems / **23 passing cells** (codex's correction: `first_repeated_char` is 2/3 both methods, not 3/3)
  - DT-only wins: `floor_Min` (+1), `get_median` (+3), `pair_OR_Sum` (+3), `are_Equal` (+2) = +9 cells (codex framing: "real but weak body-trajectory side signal; not globally null; carries one regression that RENAME does not. For MBPP shipping: DT remains obsolete vs deterministic RENAME. For future work: park as 'body-bias may matter in a different task class or stacked RENAME+DT experiment, must be re-tested with regression gate.'")
  - Codex cross-reviewed: 1 blocker (line 13 overclaim re: re-running 6.7h live eval) + 1 nit (`50× scale` → `N=50 scale`); both fixes landed verbatim before commit.

### Housekeeping cleanup

- **`362e5a5`** — `chore: housekeeping`. Three pre-existing low-risk items left uncommitted across the DT→RENAME arc, landed as one commit:
  - delete `.claude/MEMORY/SESSION_HANDOFF_1.md` (−173) — stale backup
  - delete `.claude/MEMORY/notesd.md` (−134) — stale notes
  - `scripts/dt_install_eval.py:42` (+1/−1) — `DT_EVAL_N` default 20 → 50 to match canonical eval scale

### VGSL spec joint-critique update

- **`94767db`** — `RESEARCH/VGSL/` four-file bundle, +638/−80. Joint critique → reframed in-tree pursuit; voice-preservation split (claude owns 00+01, codex owns 02+03). Cross-vocab consistent across all 4 files.

  - **`00_INDEX.md`** (claude, +191/−41): "What this is" softened to "non-weight knowledge substrate for a post-transformer stack"; new §"Pursuit path" with Stage 1/2 in-tree slice + 4 success gates; "What this subsumes" gets Stage swap target column; "Decision the user needs to make" restructured (Stage 1 in-tree primary; original 3 options as Alternatives 1/2/3); imitation-trained caveat ("VGSL attacks 2 of 3 constraints immediately; imitation-trained gated on Phase 4 with admitted training-signal-density risk").
  - **`01_ARCHITECTURE.md`** (claude, +147/−25): thesis preserved verbatim; joint-critique caveat added; Tier B claim softened ("research frontier, not solved common-sense merge"); §"Bootstrapping" Tier-B-seed bullet rewritten verbatim from codex's contradiction-fix; §"Why retrieval failed here" reframed as "anti-pattern receipt + dual-path discipline motivator, NOT whole-architecture origin story"; NEW §"Source-priority is projection policy, not merge semantics" codifies codex's correction.
  - **`02_IMPLEMENTATION.md`** (codex, +190/−0): TL;DR adds Stage 1 in-tree first / greenfield deferred framing; §"In-tree Stage 0/1"; **Invariant 8** ("Representative selection is projection policy. Source priority, latest-wins, and first-wins are never encoded as merge semantics"); §"Source priority encoding" with concrete schema (`source_tier`, `source_rank`, `corpus_order`, `line_no`, `event_id`, `stable_problem_key`); §"`CodeExampleDB`/`KnowledgeStore` assertions"; §"Stage-1/Stage-2 compatibility APIs"; §"Phase Boundaries" with Stage 0/1/2 + Phase 1-5.
  - **`03_TESTING.md`** (codex, +110/−14): "Prove parity before value" guiding principle; §"In-Tree Stage 1 Falsifier — `CodeExampleDB` Shadow Swap" with 4 success gates (parity / audit / performance / value); greenfield Phase 1 success gate adds event-sourced/temporal-table baseline (joint critique Finding 4); §"Tier B merge test" risk note acknowledges correlated-evidence + missing-contradiction risks; §"Phase Exit Criteria" + §"Recommended Baselines" updated.

  Codex's two decisive one-liners ("Merge is not fact movement..." / "Binding resolves references...") preserved verbatim per charter §F. Cross-review caught 1 blocking contradiction (`01:287-290` Tier-B-seed bullet still claimed first-occurrence-wins IS Tier-A merge) + 2 alignment nits (`00:3` post-transformer wording, `00:330` cross-ref to renamed `03` heading); all landed verbatim from codex's suggested replacements.

## In Progress

**Nothing.** Board idle both sides at handoff time.

- Slice A task `1776966574027-bb08a31d` (N=50 eval) — completed mid-session
- Spec-update parent task `1776979899534-1ac48036` — completed
- Sibling codex-owned VGSL critique + spec tasks — both completed
- VGSL first-principles review parent task `1776978553491-9cd7ee74` — completed

`ai_room_resume_check` → `idle ok: no owned tasks, no pending inbox`

## ⚠ Uncommitted

`git status --short` at handoff:

| Entry | Class | Action on resume |
|---|---|---|
| `M .claude/MEMORY/SESSION_HANDOFF.md` | session-critical | THIS handoff (being written now). `/handoff` convention is overwrite-in-place; do not commit unless explicitly asked. |
| `M .codex/AGENTS.md` | session-supporting drift | **+2 lines benign tooling policy** (Serena/python3 edit-tool guidance, "Treat `apply_patch` as last-resort fallback"). Likely codex-side parity nudge during this session. Low risk; safe to leave or land as separate parity commit (NOT bundled with VGSL). |
| `?? .cache/` | untracked cache (NOT gitignored — convention only) | Ignore by convention. Codex verified: `git check-ignore` returns nothing. |
| `?? .claude/MEMORY/memory_architecture.md` | session-supporting | Carry-over from prior session (mtime Apr 23 10:39, pre-session); not this session's work. |
| `?? .claude/MEMORY/minutes/` | untracked runtime state | Auto-generated session minutes (Stop-hook export). |
| `?? .claude/scheduled_tasks.lock` | untracked runtime state | Lock file — not gitignored, just not tracked. |
| `?? .codex/skills/` | parallel / upstream | Codex-side skills, NOT this session. Do not touch. |
| `?? .port_sessions/` | untracked cache (NOT gitignored — convention only) | Ignore by convention. Codex verified: `git check-ignore` returns nothing. |
| `?? calm/.module_learning.json` | untracked runtime state | Runtime state file — not gitignored. |
| `?? calm/hrm/checkpoints/dt_code_skel_v{4,5,9,13,14}*.pt` (6 files) | parallel / upstream | Prior-session DT checkpoints (mtimes Apr 22 + one Apr 23 10:14, pre-session). |
| `?? calm/hrm/checkpoints/stage1_argcount_best.pt`, `stage2_copyonly_best.pt` | parallel / upstream | Prior-session ruled-out training artifacts. |
| `?? calm/hrm/code_dt_stage2_data.py`, `pointer_supervision.py` | parallel / upstream | Prior-session ruled-out experimental modules. |
| `?? calm/llm_computer/facades/code_skeleton.py` | parallel / upstream | Prior-session regex-only CodeSkeletonFacade (ruled out). |
| `?? scripts/train_code_dt_stage2.py`, `train_code_dt_v17.py`, `train_stage1_argcount.py` | parallel / upstream | Prior-session ruled-out training scripts. |

**Coverage claim**: all 3 session commits landed. Only genuine session-critical uncommitted file is `SESSION_HANDOFF.md` (this file). `.codex/AGENTS.md` flagged for review but content-consistent with session's AI-Room activity.

## Next Steps

### 1. **Extend RENAME to HumanEvalPlus** — research-line of pursuit (~2h active + ~7h daemon)

User asked "how long would that take?" as the final session message; **claude's answer never landed in chat** (it was the last message before /handoff). Estimate from session context:

- **Active work**: 2-3 hours (harness extension to load HumanEvalPlus problem corpus + smoke test on N=5 + offline RENAME replay + receipt commit + codex cross-review). Function-name extraction is *easier* than MBPP (`entry_point` field is explicit). Test-format adaptation needed: HumanEvalPlus uses `def check(candidate): assert ...` wrapping vs MBPP's bare top-level asserts.
- **Daemon-bound wait**: ~6-7 hours unattended (HumanEvalPlus 164 problems; same per-problem cost as MBPP N=50 which took 6h43m).
- **Calendar**: 1 day if daemon runs in background; 2 sessions if context-switching.

**Hypothesis**: RENAME's MBPP win generalizes to other code benchmarks. Falsifiers: novel-zone +10pp confirms; flat/regress = MBPP-specific contract-name signal; DT suddenly viable on novel benchmark = architectural signal worth chasing.

**Files affected**: extend `scripts/dt_install_eval.py` to load HumanEvalPlus parallel to MBPP (probably new `load_humaneval_plus()` function + small sandbox-runner adaptation for `def check(candidate)` form). Receipt at `.claude/MEMORY/evals/2026-04-XX_dt_rename_humanevalplus.md`.

### 2. VGSL — claude's lean is **park** (with explicit revisit triggers)

Recommendation given to user via "what do you think?" exchange. **User has not formally said "park" or "Stage 1" — decision is open.** **No VGSL infrastructure work until user explicitly chooses Stage 1 / prototype** (codex's stricter framing, post-/handoff-review). If user picks Stage 1, the spec is ready (see `RESEARCH/VGSL/00_INDEX.md` §"Pursuit path" + `02_IMPLEMENTATION.md` §"In-tree Stage 0/1").

Revisit triggers (when one fires, re-read spec + execute):
- Correction conflict eats >2h debug session (audit-gate justifies itself)
- Source-priority causes measurable `CodeExampleDB` retrieval regression
- Regulated-vertical play emerges (legal/medical/financial)
- Temporal-query becomes real ask

### 3. R53 Phase 2 — PT training + L24/L30 install (substantive substrate work)

Per `CLAUDE.md` §"R53": "Phase 1 (retrieval + DB + generators) shipped; Phase 2 (PT training + L24/L30 install) pending." Bigger session budget; not queued for immediate next session unless user signals.

### 4. Optional — `.codex/AGENTS.md` parity commit

+2 lines benign tooling policy. Land as separate small commit if user wants clean tree, OR leave as drift. Not blocking.

### Deferred / separate slices (matches prior handoff; nothing changed)

- **Pre-existing contamination cleanup** — R52.1/R53/session-30/31/32/33-34/R-delta-20/`bb7f13d` in CLAUDE.md/AGENTS.md/other rules. Punted from `0757716` per pre-dispatch agreement. Mechanical, ~30-45 min.
- **Eager-tier line-cap overflow** — `retrieval.md` 270, `delta_rule.md` 203, `AI_ROOM_COLLAB.md` claude-side 304 (all over 200 hard cap). Carve to atlas. Mechanical, ~30-45 min.
- **Dense retrieval device-alignment fix** — parked from Slice B; `_cached_dequant` monkey-patch in `gemma_substrate.py:1458-1475` leaks GPU cache into `DenseIndex.load()`'s CPU dequant path. Unpark only if new workload demands hybrid RRF.
- **VGSL Stage 1 prototype** (only if user picks Stage 1) — ~1-2 weeks of careful refactor for in-tree `CodeExampleDB` shadow swap with `source_priority_v1` policy materializing current `examples` order bit-for-bit; 4 success gates per `RESEARCH/VGSL/03_TESTING.md` §"In-Tree Stage 1 Falsifier."

## Key Context

### Joint critique distilled findings (load-bearing for VGSL spec interpretation)

The 5 load-bearing primitives identified jointly:
1. **Non-destructive merge as projection-time aliasing** (codex's R4 insight, preserved verbatim at `01_ARCHITECTURE.md:88-101`)
2. **Verifier+canonicalizer versioning per event** (`01:103-115`)
3. **Binding/merge/projection separation** (`01:159-201`)
4. **Exact-default reads with explicit slow path** (`01:245-275`)
5. **Dependency-tracked derivation invalidation** (`01:117-131`)

Overstated framings (now softened in spec):
- "Post-transformer architecture" → "non-weight knowledge substrate for a post-transformer stack"
- "Versioned event log as novelty" → real prior art (Datomic, Kafka + materialized views, RDF/SPARQL with revision history); novel composition is verifier+canonicalizer versioning + non-destructive merge + dependency invalidation + exact-gated reads
- "Retrieval-null motivates whole thesis" → motivates dual-path discipline only; defensibility argument, not new-capability
- "Phase 1 falsifier proves novelty" → proves semantics only; distinctive value at Phase 3
- "Imitation-trained attacked at base" → attacked only when Phase 4 proposer/verifier generates beyond imitation

Real risks (3 stacked open-research dependencies):
1. Open-world canonicalization is "TODO" (`01:366-368`)
2. Tier B "common-sense merges" claim too strong; correlated evidence + missing contradictions
3. Phase 4 learned-proposer might fail; product-path VGSL survives without it (closed-world / Tier-A-heavy), research-path does not

### Codex's load-bearing correction (msg `1776979663929-802bd974`, verbatim)

> **CodeExampleDB source priority is projection policy, not merge semantics.**
>
> Source priority should live in the projection rule / selected-assertion derivation: `accepted_assertion.source_tier`, `source_rank`, stable tie-breaker `(corpus_order, line_no, event_id)`, and a projection output like `active_example_for_key = assertion_id`. Otherwise we conflate identity with representative selection and recreate the merge/binding mistake in a smaller room.

Codified in `02_IMPLEMENTATION.md` Invariant 8 + §"Source priority encoding"; cross-ref'd from `01_ARCHITECTURE.md` §"Source-priority is projection policy, not merge semantics."

### "Prove parity before value" — gate-design correction (codex msg `1776979663929-802bd974`)

Claude originally proposed "≥3 bugs surfaced in 1-week shadow mode" as success gate. Codex called this luck-shaped — would punish a clean system. Replaced with 4 explicit gates:
1. **Parity gate** (entry ticket — behavior-equivalent first)
2. **Audit gate** (forced auditability test)
3. **Performance gate** (≤10% overhead, O(1) active lookup)
4. **Value gate** (shadow-mode bugs are upside, not entry ticket)

Now in `03_TESTING.md` §"In-Tree Stage 1 Falsifier" §"Success gates."

### User pivots verbatim (this session)

| Moment | Verbatim | Effect |
|---|---|---|
| collab entry | invoked `/collab` | drained reconnect state, took over Slice A |
| codex async signal | "codex compiling" | proceeded solo on offline RENAME replay |
| forced grounding audit | "did you read this at session start'/mnt/c/Users/gabes/projects/claw-code/.claude/MEMORY/SESSION_HANDOFF.md'?" | claude conceded no, read in full, identified 5 missed gotchas |
| direction | "yes do cleanup commit then review'/mnt/c/Users/gabes/projects/claw-code/RESEARCH/VGSL' with codex with first principles thinking" | landed `362e5a5` cleanup + opened first-principles review |
| upside question | "what is the upside of vgsl ontop of substrate?" | claude framed as "purely a knowledge-substrate replacement for KnowledgeStore + CodeExampleDB + auto-upgrade loop"; commercial vertical = substantial; current R&D = marginal |
| **reframe** | "ok well if its a upgrade to VGSL is purely a knowledge-substrate replacement for KnowledgeStore + CodeExampleDB + auto-upgrade loop. then maybe its worth pursuit? discus with codex" | triggered design discussion → Stage 1/2 in-tree shadow-mode framing |
| spec-update direction | "ok update '/mnt/c/Users/gabes/projects/claw-code/RESEARCH/VGSL' with new ideas with codex" | landed `94767db` 4-file spec update |
| recommendation ask | "what do you think?" | claude's answer: park with revisit triggers |
| next-line ask | "so after last eval what was the next line of pursuit?" | claude surfaced HumanEvalPlus / BigCodeBench RENAME extension |
| timing ask | "how long would that take?" | claude estimated ~2h active + ~7h daemon |
| **handoff** | "/handoff lets hand off to pick up in new session" + "and include codex for review" | THIS handoff invocation |

### Subagent-policy reminder

Charter §C: claude-side documented exception preserved (slash-command subagents allowed where command explicitly allows them); codex-side no-subagents preserved. THIS /handoff fired 2-agent parallel grounding per command spec — within charter.

### Failed approaches this session (none new — all from prior sessions)

No new failed approaches this session. Prior session nulls remain valid: DT code-skeleton for MBPP (ruled out), retrieval-signature without rename (ruled out), batched autoreg eval (ruled out). DT-bias on MBPP confirmed null at N=50; RENAME canonical.

### Hardware / environment state at session end

- Gemma daemon PID 540958, running `bin/gemma_daemon.py`, etime ~7h46m, 75.7% CPU
- GPU: 7042/8188 MiB used by PID 540958, 5% util, 15W/99W, 37C — warm/idle-resident (model loaded, awaiting requests)
- Forensic dumps preserved: `/tmp/dt_install_eval_results.json` (258 KB, N=50 dump from 21:50). Other artifacts: `/tmp/dt_install_eval_n5_results.json` (25 KB), `/tmp/dt_install_eval_n20_results.json` (102 KB, prior-eval dump — preserved separately, NOT overwritten), `/tmp/dt_retrieval_offline_eval.json` (9.8 KB), `/tmp/dt_retrieval_offline_n20_results.json` (9.8 KB), train logs v14-v17 (prior session R&D).
- Branch `feature/multi-agent-qwen` on `mastergrief/zenith-code`
- ai-room cursor synced; both peers idle

## Files in Project (session-shipped)

### New files

- `.claude/MEMORY/evals/2026-04-23_dt_rename_n50.md` (108 lines, `1e0680e`) — N=50 RENAME-vs-DT receipt; canonical reference for `dac50ed`-stands-at-scale verdict.

### Modified files

- `RESEARCH/VGSL/00_INDEX.md` (+191/−41, claude-authored, `94767db`) — pursuit path, Stage 1/2 framing, decision restructure
- `RESEARCH/VGSL/01_ARCHITECTURE.md` (+147/−25, claude-authored, `94767db`) — caveats, Tier B softening, source-priority section
- `RESEARCH/VGSL/02_IMPLEMENTATION.md` (+190/−0, codex-authored, `94767db`) — Stage 0/1/2 implementation shape, Invariant 8, source priority encoding
- `RESEARCH/VGSL/03_TESTING.md` (+110/−14, codex-authored, `94767db`) — 4 in-tree gates, parity-first principle, event-sourced baseline
- `scripts/dt_install_eval.py` (+1/−1, `362e5a5`) — `DT_EVAL_N` default 20 → 50

### Deleted files

- `.claude/MEMORY/SESSION_HANDOFF_1.md` (−173, `362e5a5`) — stale handoff backup
- `.claude/MEMORY/notesd.md` (−134, `362e5a5`) — stale notes file

## Handoff verification

- ✅ 3 current-repo session commits verified individually: `1e0680e`, `362e5a5`, `94767db`. Verified via `git log --oneline -5` + per-commit `git show --stat`.
- ✅ Grounding via 2 parallel Explore agents (transcript+measurements / code+uncommitted state); this handoff uses their verbatim extractions for eval numbers + commit deltas + uncommitted-state classification.
- ✅ Daemon PID 540958 alive, etime 7h46m at handoff finalization, 7042/8188 MiB GPU resident.
- ✅ All ai-room tasks completed: Slice A (`1776966574027-bb08a31d`), VGSL critique parent (`1776978553491-9cd7ee74`), spec-update parent (`1776979899534-1ac48036`); both codex sibling tasks also completed.
- ✅ ai_room_resume_check → `idle ok: no owned tasks, no pending inbox`
- ✅ N=50 RENAME receipt commits verbatim numbers from `/tmp/dt_install_eval_results.json` + offline `scripts/dt_rename_offline_eval.py` rerun (codex re-verified the offline scorer; not the 6.7h live daemon eval).
- ✅ VGSL spec update preserves codex's two decisive one-liners verbatim ("Merge is not fact movement..." / "Binding resolves references...") per AI Room charter §F.
- ✅ Codex cross-review completed via ai-room (msg `1776981614275-30403a17`); 4 metadata corrections landed verbatim before user-facing confirmation: per-file numstats (`+191/-41`, `+147/-25`, `+190/-0`, `+110/-14`), `.cache/` + `.port_sessions/` reclassified from "gitignored cache" to "untracked cache (NOT gitignored — convention only)", `/tmp/dt_install_eval_n20_results.json` corrected from "overwritten" to "preserved separately", VGSL no-work framing strengthened.
