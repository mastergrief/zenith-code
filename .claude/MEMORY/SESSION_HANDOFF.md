# Session Handoff — 2026-04-20 (AST walker tier-2 shipped)

## Goal

Implement next-steps 1, 2, 3 from the prior 2026-04-20 handoff (fused flash-attn flip + watch-wrap + hook enforcement):
1. **AST walker tier-2 card** — projected 32/32 → ~43-45/46 via mechanical rewrite of R53.33 failure modes
2. **Verify `.claude/settings.json` hook** fires
3. **Commit optional captures** — `.claude/MEMORY/can_be_done.md` + `.claude/commands/summarise.md`

Workflow: hypothesis → build → test → commit → iterate.

## Completed (4 commits, `190fe55` → `b9aaedc`)

### Subsystem 1 — AST walker shipped (3 commits)

**`9db8319`** — `calm/llm_computer/facades/ast_repair.py` (269 LOC) + 21 unit tests.

Two narrow rewrites, error-text-driven, pure Python + `ast` stdlib:

1. **Shadow rename** (TypeError: 'X' object is not callable). Find
   `self.<name> = ...` assignments where `<name>` is also a method
   on the same class; rename attr to `_<name>`, rewrite all non-call
   read sites, preserve method body. Handles AugAssign. Scoped per
   class (doesn't collide across classes).

2. **Dict-key synonym** (KeyError: 'X'). Curated synonym table
   (`avg`→`mean`, `std`→`stdev`, `mu`→`mean`, etc). Rewrites Dict
   literals, Subscript access, `.get/.pop/.setdefault` args.
   Returns `none` when missing key has no known synonym.

Entry point `repair(code, error_output) -> RepairResult` dispatches
shadow first (runs unconditionally — static detection), then dict
synonym if KeyError in error text.

**`8cc2ff4`** — Wired into `scripts/r53_21_import_inject.py` via
`try_ast_repair()` — runs after import injection, before LLM
structured repair. MAX_AST_REPAIR_PASSES=4 (chain for csv's
potential `mean → stdev → min → max`). Reverts pre-pass state on
regression. Also fires on LLM-repaired code in attempt 2+ loop.
Results table grew `ast` column showing applied rewrite kinds.

**`b9aaedc`** (this commit) — doc updates: `tracing_roadmap.md`
adds row to shipped table + csv extractor ruled-out entry;
`capability_gain.md` adds R53.35 receipt under "Gemma ignores
targeted hints" confirming the tier-2 hypothesis.

Measurement (two paths, both moved, no regression):

  path                                            before   after  wall
  ---------------------------------------------   ------   ------ ----
  Raw:  pytest calm/llm_computer/tests/
        test_ast_repair.py                         n/a     21/21  8.6s
  User: token_bucket_rate_limiter (R53.0)          0/0      5/5   0.9s
  (Gemma attempt 1 was 0/0 in 79s; walker lifted in 0.9s post-gen)
  Regr: lru_cache_class (R53.0, no-regression)     9/9      9/9   123s

token_bucket specifics: Gemma emitted `self.tokens = capacity` in
`__init__` and a `def tokens(self)` method — canonical shadow.
Walker renamed attr to `_tokens`, method stayed, 5/5 sub-tests
passed: `tb.tokens() = 5.0`, drain-3, 4th-fails, refill-after-sleep,
cap-enforced.

### Subsystem 2 — Hook verified (no commit; just validation)

`.claude/hooks/enforce-watch-wrap.sh` smoke-tested on 4 inputs:
- block: `tail -f /tmp/foo | grep bar` ✓
- allow: `bin/watch-wrap --log ...` ✓
- allow: `while true; ... sleep 2; done` poll-loop ✓
- silent-allow: `ls -la` (no tail) ✓

File wiring is correct. Live activation in future sessions depends
on Claude Code's settings-watcher (handoff caveat from prior
session: new `.claude/settings.json` may need `/hooks` menu reload
once, or session restart).

### Subsystem 3 — Optional captures committed (1 commit)

**`190fe55`** — `.claude/MEMORY/can_be_done.md` (substrate
preserve/augment/plug thesis brain-dump, references
`gemma_substrate.py:install_card_in_attention`) +
`.claude/commands/summarise.md` (resume-brief reader pairing with
`/handoff` writer).

## In Progress

None. Round 1 of the AST walker is shipped with measurement. Open
iteration targets below.

## ⚠ Uncommitted

```
 M .claude/MEMORY/SESSION_HANDOFF.md              # this file
 M calm/hrm/checkpoints/meta_best.pt              # TEAMMATE — flag
 M scripts/r52_train_student_kl.py                # TEAMMATE — flag
 M scripts/r53_22_diagnose_csv.py                 # TEAMMATE — flag
?? .cache/                                         # gitignored cache
?? .claude/MEMORY/minutes.md                       # transcript — do NOT commit
?? .claude/MEMORY/minutes/                         # transcript — do NOT commit
?? .codex/, .port_sessions/                        # tooling — ignore
?? RESEARCH/{LLM-COMPUTER,NEURAL_COMPUTER,TQ,TRAINING}/  # teammate
?? calm/.module_learning.json                      # runtime
?? calm/hrm/checkpoints/copy_code_*.pt             # TEAMMATE — flag
?? calm/hrm/checkpoints/math_*.pt                  # TEAMMATE — flag
?? calm/llm_computer/checkpoints/substrate_hrmlm_v2*.pt       # TEAMMATE
?? calm/llm_computer/checkpoints/substrate_hrmlm_v2_tokenizer.json
?? calm/llm_computer/r51/checkpoints/              # TEAMMATE R51
?? calm/llm_computer/synth/chat_transcript.jsonl   # TEAMMATE
?? calm/llm_computer/synth/library.jsonl           # TEAMMATE
?? calm/llm_computer/tq4_autograd.py               # TEAMMATE — R52 work
```

**Session-critical unintentionally uncommitted**: **none**. All
this session's walker code, tests, wiring, and doc updates are
committed.

## Next Steps (ordered by commercial lift)

1. **Broader-corpus validation** (~1 day, HIGH lift) — the walker
   lifted one R53.0 problem from 0/0 → 5/5. Next: run on MBPP /
   HumanEvalPlus / BigCodeBench failures where Gemma produces code
   (unlike csv_column_stats). Projected: each problem Gemma fails
   via shadow or dict-synonym is a free +N tests. Existing pipeline
   — apply the R53 failure-surface gate (see
   `capability_gain.md` §"Failure-surface gate") to partition
   `fails_correctness` vs `solves_cleanly`, then run walker on the
   former.

2. **AST walker — new rewrites** (~2-3 days, MEDIUM lift) — widen
   the rewrite set beyond shadow + synonym:
   - **Off-by-one guard** (`IndexError: list index out of range`)
     — detect `for i in range(len(xs) + 1): xs[i]` patterns;
     rewrite `range(len(xs))`.
   - **Unused-var vs missing-var** (`NameError: 'X'` where X is
     an unused assignment two scopes up) — rewrite to use the
     assigned name.
   - **Missing `return`** (function runs but returns None, test
     expects int) — detect implicit None return + rewrite to
     return the last expression.
   Each rewrite needs: (a) test-case capturing the Gemma bug, (b)
   error-text detector, (c) AST transformer, (d) unit test, (e)
   integration in `ast_repair.repair()`.

3. **csv_column_stats extractor unblock** (~1-2 days, MEDIUM lift)
   — `test_end_to_end_csv_column_stats_passes_after_repair`
   confirms walker correctness on csv's bug pattern, but live
   Gemma at medium budget emits 0/0 NoCode. Two paths:
   - a) **Force code-fence prefix**: prepend `\n```python\ndef
     csv_column_stats(text):\n` to Gemma's context, making the
     first token always the fence-body. Mid-generation per-token
     hook — NOT first-token bias (ruled out on code R53.14/20a/20b).
   - b) **Wider extractor**: `extract_code()` already tries fence
     / def / class / import / whole-AST. Check what Gemma IS
     emitting (prose? `<think>` blocks that overflow? malformed
     fence?) and extend the extractor if a recoverable pattern
     exists. Probably faster than (a).

4. **Fused flash-attn long-context sweep** (~1 day, orthogonal) —
   re-bench `scripts/r53_phase2_bench.py` at N=8K/16K per prior
   handoff. Find or rule out an asymptotic crossover past the
   `N < 2048` gate's upper bound.

5. **Hook live-activation check** — next session with Claude
   Code: trigger a raw `tail -f | grep` via Monitor and verify
   the hook blocks. If not, open `/hooks` menu to reload.

## Key Context

**Decision rationale (WHY):**

- **Shadow detector runs unconditionally** (not only on TypeError):
  the detector is pure static analysis (parse + walk methods +
  walk assignments). Running it even when error text is missing
  lets the walker rescue code whose symptom surfaces as
  AttributeError, TypeError on a *later* call, or a missing test
  output. Unit test `test_repair_shadow_runs_unconditionally_without_error_text`
  locks in this behavior.
- **Dict synonym is KeyError-gated** (opposite of above): the
  table is short + curated; running it without an error target
  would rewrite any `avg` → `mean` in any code, potentially
  changing backend API keys that were legitimately `avg`. KeyError
  gives us the missing-key target to aim for.
- **Revert on regression**: walker rewrites are static but could
  plausibly regress if (1) the code legitimately uses a synonym
  name for a different purpose, or (2) shadow rename collides in
  an unusual way. `try_ast_repair` snapshots `(pre_code, pre_sp,
  pre_st)` before each pass and reverts cleanly on regression.
- **MAX_AST_REPAIR_PASSES=4**: csv has 4 expected sub-keys
  (mean/stdev/min/max). Each KeyError reveals the next. Cap at 4
  to avoid infinite loops on malformed code.
- **Walker before LLM repair in the pipeline**: it's 100-400× cheaper
  (0.9s vs 80-300s per Gemma round). Running first means we don't
  burn Gemma inference if a mechanical fix would work.

**Measurement discipline caveats**:

- **Full 6-problem corpus run was aborted** — daemon's
  AdaptiveBudget picked `hard (16384 tok)` for
  `linked_list_bugs`, which at memo-path decode (~7 tok/s past
  the fused gate N=2048) projects to ~35min per attempt ×
  3 attempts × 6 problems = 10+ hours. Killed after 45min on
  problem 1. Spot-checked with lru_cache_class instead (9/9
  preserved, 123s). Broader validation needs a cheaper loop —
  either reduce `hard` tier default, or use the MBPP corpus
  where per-problem time is shorter.
- **token_bucket measurement is single-run**, not median-of-5.
  Direction reliable (0/0 → 5/5 is binary). Magnitudes don't
  apply for this kind of result. No GPU-bench-discipline concern.
- **AdaptiveBudget calibration has drifted since R53.33**:
  linked_list_bugs went from `medium (8192)` in R53.33 to `hard
  (16384)` in this session. Unknown whether that lifts or
  regresses pass-rate — not measured. Flagged for future budget
  audit.

**Do NOT relitigate**:

- csv_column_stats 0/0 at 8K budget is not a walker regression —
  walker correctness on csv's bug pattern is locked in via unit
  test. The live Gemma failure is extraction-blocked (prompt-level),
  not logic-blocked. See `tracing_roadmap.md` ruled-out entry.
- AST walker for other error classes (off-by-one, missing-return,
  etc) is NEXT work (see §2 above), not part of R53.35's shipped
  scope.

**Failed approaches (cite SHAs)**:

- Full-corpus re-run at daemon default budgets: daemon killed at
  T+45min on `linked_list_bugs` after 0 completed problems.
  Aborted via `kill -9 705949`. Restarted daemon for spot-check.
  No commit.

**Runtime state at session end**:
- Branch: `feature/multi-agent-qwen`, **407 commits ahead of
  origin** (session 33-34's 13 + this session's 4 = 17 today, but
  prior 390 already pre-existing).
- Daemon: **PID 856719** running. Spot-check ran to completion
  (`[daemon] completed` marker, DONE). Loaded Gemma from
  `gemma-4-E4B-it-tq4-aligned.gguf`, ~3.5GB VRAM idle.
- GPU: ~511 MiB → ~5-6 GB depending on idle vs generation.
- No in-flight work. No stash.

## Files Changed (session-shipped)

### New files (3)
- `calm/llm_computer/facades/ast_repair.py` — walker, 269 LOC
- `calm/llm_computer/tests/test_ast_repair.py` — 21 unit tests, 300 LOC
- `.claude/MEMORY/can_be_done.md` (committed from prior session) —
  substrate thesis brain-dump
- `.claude/commands/summarise.md` (committed from prior session) —
  resume-brief reader

### Modified code
- `scripts/r53_21_import_inject.py` — `+try_ast_repair()`,
  `+MAX_AST_REPAIR_PASSES`, wired into main loop + LLM-repair
  inner loop, results tuple gained `ast_repairs` column, docstring
  rewrite, baseline annotations bumped to R53.33 32/32 +
  walker-projected 37+/37. +124 LOC net.

### Modified docs
- `.claude/rules/tracing_roadmap.md` — `ast_repair` row added to
  "Shipped and verified" + "Facades built" tables (+2 rows);
  ruled-out entry for csv_column_stats extractor bottleneck
  (R53.35) added.
- `.claude/rules/capability_gain.md` — R53.35 receipt added under
  §"Gemma ignores targeted hints" with measurement table + csv
  bottleneck flag.
- `.claude/MEMORY/SESSION_HANDOFF.md` — this file (overwrote
  prior 2026-04-20 fused-flash-attn handoff; previous content
  retained in git history at `b453a38`).

### Deleted
None.
