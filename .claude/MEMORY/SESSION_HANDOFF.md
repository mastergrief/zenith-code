# Session Handoff — 2026-04-20 (7-next-steps arc: refactor + walker +2 rewrites + force-fence win)

## Goal

Implement all 7 next-steps from the prior handoff (MBPP corpus walker, csv
force-fence prefix, 3 additional walker rewrites, Jacobian-weighted tier-3,
fused flash-attn long-context sweep, hook live-activation, LOC-cap trim).

Redirected mid-session: user asked to **refactor** the over-limit rule files
(preserve insights via migration, not trim-and-lose). Jacobian tier-3 (~2
weeks, speculative) stayed out of scope.

Workflow: hypothesis → build → test → commit → iterate. 8 commits,
`f65c376` → `51828f4`.

## Completed (8 commits)

### Round 1 — Refactor CLAUDE.md (commit `f65c376`, LOC 529→446)

R53 section duplicated canonical rule files (workflow.md §MAX_TOKENS,
Substrate.md §KVCacheTq4, turboquant.md §fused flash-attn, calm.md §sandbox,
capability_gain.md §R53.35-36). Compressed to Phase 1 deliverables +
headline findings + per-round pointer list. All 11 R53 anchors retained;
zero information loss. Under 500-LOC hard limit.

### Round 2 — Refactor augmentation_thesis.md (commit `9420e96`, LOC 509→469)

Empirical-basis R13-R52 per-round bullets (~83 lines) duplicated
tracing_roadmap.md's 51 per-round rows. Compressed to **7-capability
cluster table** (layer cluster, typology, key causal validation) + causal-
validation summary + facade summary + 3-null summary. Upgrade: table more
readable than bullets. All anchors preserved.

### Round 3 — settings.json hook verified (no commit)

Inspected `.claude/settings.json` + `.claude/hooks/enforce-watch-wrap.sh`
+ `bin/watch-wrap`. Smoke-tested with two inputs:

| Input | Expected | Actual |
|---|---|---|
| `tail -f /tmp/train.log \| grep epoch` (raw) | deny JSON | ✓ deny with actionable message |
| `bin/watch-wrap --log /tmp/x` (wrapped) | silent allow | ✓ exit 0 |
| `while true; … sleep` (poll loop) | silent allow | ✓ (pattern inspection) |

Hook wiring is correct; will fire in-session when Monitor invokes raw tail.
Pure verification — no code change.

### Round 4 — Walker rewrite: off-by-one range (commit `7d0222a`)

Fourth deterministic rewrite in `ast_repair.py`. Canonical Gemma fencepost
bug: `for i in range(len(xs) + 1): xs[i]`. Gated on IndexError in error
text AND body-subscript signal (filters intentional fencepost loops).
Handles `range(len(X)+1)`, `range(0, len(X)+1)`, `range(1+len(X))`.

| metric | before | after |
|---|---:|---:|
| walker rewrites | 3 | 4 |
| unit tests | 36 | 48 (+12) |
| tests passing | 36/36 | 48/48 |

### Round 5 — Walker rewrite: missing return (commit `b800183`)

Fifth rewrite. Canonical bug: function computes answer as bare `Expr` at
tail, forgets `return`. Gated on None-return error signal AND no existing
`return <value>` in function AND plausible-expr last statement. Bare
literals (`42`, `"str"`) excluded as dead code. Scope-aware walk respects
nested FunctionDef/Lambda boundaries.

| metric | before | after |
|---|---:|---:|
| walker rewrites | 4 | 5 |
| unit tests | 48 | 61 (+13) |
| tests passing | 48/48 | 61/61 |

### Round 6 — MBPP N=5 spot-check (commits `daefaed`, `51828f4`)

First wider-corpus walker test. Harness parses MBPP problems + bundled
asserts from `agents/distill/data/mbpp.jsonl`. Stock Gemma + extract + run
asserts + walker chain on failures.

**Result — diagnostic null**:

| outcome | count |
|---|---:|
| clean | 0/5 |
| walker_fixable | **0/5** |
| genuine_fail | 2/5 (IndentationError) |
| format_fail | 3/5 (MAX_TOKENS starvation) |

Per-problem: max_chain_length NO CODE (158s, truncated), first_repeated_char
IndentationError, get_ludic NO CODE (64s), reverse_words IndentationError,
prime_num (incomplete when stream ended).

**Two new findings (both outside current walker scope)**:

1. **MAX_TOKENS=1024 budget-starves MBPP.** Custom-class backtracking like
   `max_chain_length` needs ~1500-2000 tok. Per workflow.md §"MAX_TOKENS
   budget discipline": verify budget isn't clipping BEFORE diagnosing
   logic. Raised to 2048 in second commit.

2. **Empty `except:` blocks trigger harness-concatenation IndentationError.**
   Gemma emits `try: ... except Exception: # empty body`. Harness appends
   test try/except chain at col 0, empty except's expected indented body
   is unsatisfied. NEW WALKER CANDIDATE: detect empty `except:` / `try:`
   / `if:` blocks, insert `pass`.

Walker never fired on this sample (0/5) — dispatch gates never triggered
because no KeyError / TypeError-callable / IndexError / None-return
appeared. Sample of 5 MBPP problems happens to hit walker-outside-scope
patterns. Not a regression; the walker's +13 R53.0 test lifts stand.

### Round 7 — csv force-fence 0/0 → 8/8 (commit `21f5001`)

Closes csv_column_stats's NoCode branch. Measured end-to-end on live Gemma
through the daemon.

Mechanism: prepend `\`\`\`python\ndef csv_column_stats(text):\n` to the
prompt's `<start_of_turn>model\n` continuation point. Distinct from R53.14
first-token logit bias (ruled out — forces "def" without fence). Here the
fence AND signature are in Gemma's CONTEXT, so its continuation is
indented-body by construction.

Paired with walker chain: force-fence closes the NoCode shape; walker
closes runtime bugs in the emitted code.

| metric | before | after |
|---|---:|---:|
| csv_column_stats | 0/0 | **8/8** |
| mechanism | NoCode | force-fence + walker |
| Gemma native error | prose output | `KeyError: 'stdev'` |
| walker rewrite | — | dict_synonym (`stddev` → `stdev`) |
| wall time | — | ~4 min (decode) + ~1s (walker) |

First-pass script had a dispatch bug: walker gated on `passed < total` but
`total=0` when sandbox raises mid-test. Fixed to trigger walker whenever
tests don't cleanly pass; chained two walker passes for sequential errors.

**Combined R53.0 corpus lifts** (R53.35 walker + R53.38 force-fence):
- token_bucket_rate_limiter: 0/0 → 5/5 (shadow_rename)
- csv_column_stats: 0/0 → 8/8 (syntax_repair *first-bug branch* OR
  force-fence + dict_synonym *NoCode branch*)

**+13 tests mechanical, zero Gemma retries, ~1s per walker fix.**

### Round 8 — Long-N flash-attn bench script (commit `1d78fae`, run pending)

Ships ready-to-run `scripts/r53_37_long_n_bench.py` for N ∈ {8192, 16384}
extending the R53.34 curve past the runtime N-gate `128 < cached_kv_len <
2048`.

GPU discipline per workflow.md §"GPU bench discipline":
- `heavy_warmup(3.0s)` via dense fp16 matmuls
- `torch.cuda.Event(enable_timing=True)` (not `time.time()`)
- correctness sanity check vs fp16 argmax before timing
- paired same-process A/B per-N
- median-of-N_RUNS (configurable; default 1 for directional)

**Not executed this session** — full [8192, 16384] × N_RUNS=3 ≈ 9 hours,
exceeds session budget. Default config [8192] × 1 run ≈ 66 min for a
directional probe.

First-principles prediction (not measured): memo continues to dominate at
N ≥ 8K because fused has fixed 336 kernel launches/step vs memo's single
cuBLAS matmul per step scaling linearly with N. Crossover unlikely without
fused-kernel restructuring (all-heads-one-launch, TILE_N).

## In Progress

None. All 8 rounds closed with commits + measurements or shipped artifacts.

## Uncommitted (unchanged; teammate-owned)

```
 M calm/hrm/checkpoints/meta_best.pt              # TEAMMATE
 M scripts/r52_train_student_kl.py                # TEAMMATE
 M scripts/r53_22_diagnose_csv.py                 # TEAMMATE
?? .cache/, .codex/, .port_sessions/              # tooling/cache
?? .claude/MEMORY/minutes{,.md,/}                 # transcript — do NOT commit
?? RESEARCH/{LLM-COMPUTER,NEURAL_COMPUTER,TQ,TRAINING}/
?? calm/.module_learning.json                     # runtime
?? calm/hrm/checkpoints/copy_code*_best.pt        # TEAMMATE
?? calm/hrm/checkpoints/math_*_best.pt            # TEAMMATE
?? calm/llm_computer/checkpoints/substrate_hrmlm_v2*.pt
?? calm/llm_computer/r51/checkpoints/             # TEAMMATE R51
?? calm/llm_computer/synth/*.jsonl                # TEAMMATE
?? calm/llm_computer/tq4_autograd.py              # TEAMMATE R52
```

**Session-critical unintentionally uncommitted: none.**

## Key Findings

1. **Walker surface-area extension shipped**: 3 → 5 rewrites, +25 unit
   tests, zero regression. Off-by-one (IndexError) and missing-return
   (NoneType) cover two recurring Gemma failure modes. Both gated on
   error-text + AST-signal to avoid false positives.

2. **csv NoCode branch closed** (0/0 → 8/8 on live Gemma) via
   prompt-tail force-fence + walker dict_synonym. Distinct from R53.14
   first-token bias (which was ruled out); here the fence is in Gemma's
   context, not injected via logit hook.

3. **MBPP spot-check (N=5) produced 0 walker lifts** but surfaced two
   new actionable patterns: MAX_TOKENS starvation (per workflow.md
   budget discipline) and empty-block IndentationError. Both are
   walker-candidate patterns for follow-up, not walker regressions.

4. **LOC-cap violations eliminated** via refactor-to-canonical-rule, not
   trim. CLAUDE.md 529 → 446, augmentation_thesis.md 509 → 469. Every
   R-anchor preserved.

## Next Steps (ordered by lift)

1. **Empty-block walker rewrite** (~2 hours, MEDIUM lift) — R6 finding.
   Detect `except:` / `try:` / `if:` / `else:` / `while:` / `for:` blocks
   with no body (or comment-only body), insert `pass`. Tests via MBPP
   patterns + synthetic. Likely 2-3 R6 problems lift after this + larger
   MAX_TOKENS.

2. **MBPP N=20-50 re-run** (~40-100 min wall, HIGH lift) — now with
   MAX_TOKENS=2048 (committed) + empty-block rewrite (pending). Direction-
   reliable lift count for the "extractor-artifact" thesis at scale.

3. **Run R53.37 long-N bench** (~66 min for N=8K × 1 run) — confirm or
   deny the asymptotic memo-dominance prediction. Use when a long-run
   window opens.

4. **csv force-fence generalization** (~3 hours, MEDIUM) — currently
   hardcoded to csv_column_stats' signature. Generalize to auto-derive
   signature from problem's `required` field + prompt parse.
   Prerequisite for applying to MBPP/HumanEvalPlus NoCode branch.

5. **Walker cascading on sequential errors** (~1 hour) — R53.38 script
   chains two walker passes. Generalize this into `ast_repair.repair()`
   so it iterates up to N passes until no further applicable rewrite.
   Useful when e.g. syntax_repair then shadow_rename then dict_synonym
   are all needed.

6. **Jacobian-weighted tier-3 distillation** (~1-2 weeks, SPECULATIVE,
   ~30% prob). R53.36 audit showed R51-MSE student reaches cos=0.89 on
   L24 (close-miss cascade); a loss weighting by `J = d(head_logits) /
   d(h_L24)` might close the gap. Not priority — tier-2 stacking
   already delivers.

7. **LOC-cap maintenance** — pass through other rule files (retrieval.md,
   code_reasoning_db.md, capability_gain.md, tracing_roadmap.md) to
   check for drift above 500. Run `wc -l .claude/rules/*.md` periodically.

## Key Context

**Decision rationale (WHY)**:

- **Refactor over trim**: user explicitly redirected R1/R2 toward
  migration-to-canonical — preserves insights, improves readability of
  both source and destination. Commit `f65c376` demonstrates the
  pattern: audit duplication → confirm canonical coverage → replace
  with pointer.
- **Walker gates on BOTH error text AND AST signal**: belt+suspenders.
  Error text alone triggers on genuinely-different failures that
  happen to share a regex. AST signal alone fires on intentional
  code patterns. Both together = high-precision dispatch.
- **Force-fence ≠ first-token bias**: prompt-tail `def <name>(...):\n`
  puts the fence + signature in Gemma's CONTEXT before generation.
  Gemma's next tokens are indented-body by construction. Contrast
  R53.14 FirstTokenHook which biased logits AFTER first emission,
  producing fence-less code that failed the extractor.
- **R6 0/5 walker lift is NOT a regression**: the walker dispatch gates
  never triggered on this MBPP sample. Lesson: need to enlarge walker
  to cover MBPP's actual failure modes (empty blocks, MAX_TOKENS
  truncation side-effects) before measuring wider-corpus lift.

**Measurement discipline caveats**:

- R6 N=5 is a spot-check, not a proper failure-surface pass per
  capability_gain.md. Larger N (20-50) needed for actionable counts.
- R8 bench NOT RUN — script shipped as artifact. Honest call-out;
  execution awaits a long-run window.
- R7 walker chain used two-pass cascade; first pass might apply a
  rewrite that introduces a new error, second pass closes the chain.
  Worth generalizing into `ast_repair.repair()` (see next-step 5).

**Do NOT relitigate**:

- R1/R2 refactor is strictly additive (pointers to canonical, not
  deletion) — do not restore the duplicated sections.
- R7 csv force-fence + walker 8/8 stands; distinct from R53.35
  syntax_repair csv 8/8 (same problem, different Gemma output shape).
- R6 0/5 lift is spot-check only; conclusions about MBPP-wide walker
  coverage require N ≥ 20.

**Failed approaches (cite SHAs)**:

- First pass of `scripts/r53_38_force_fence.py` gated walker on
  `passed < total`; `total=0` when sandbox erred mid-test meant
  walker never attempted. Fixed by changing to
  `not (passed > 0 and passed == total)`, plus 2-pass chain. Receipt
  in commit `21f5001` body.
- First pass of `scripts/r53_39_mbpp_walker.py` used `@dataclass` for
  `MbppProblem` — Python 3.13 dataclasses machinery needs
  `sys.modules.get(cls.__module__).__dict__` which is None when
  daemon `exec`s a script in its own globals. Fixed by switching to
  plain class. Receipt in commit `daefaed`.

**Runtime state at session end**:
- Branch: `feature/multi-agent-qwen` at `51828f4`. 8 commits ahead of
  prior session start (`ad3fdae`).
- Daemon: PID 934814 still running, Gemma pre-loaded.
- GPU: ~5-7 GB depending on idle vs generation.
- Hook: `.claude/settings.json` + `.claude/hooks/enforce-watch-wrap.sh`
  verified active; raw `tail -f | grep` denied, wrapped allowed.

## Files Changed (session-shipped)

### New files (3)
- `scripts/r53_37_long_n_bench.py` — long-N flash-attn bench (235 LOC,
  artifact only)
- `scripts/r53_38_force_fence.py` — csv force-fence runner (161 LOC,
  run + measured)
- `scripts/r53_39_mbpp_walker.py` — MBPP corpus harness (307 LOC, run
  N=5 + measured null)

### Modified code
- `calm/llm_computer/facades/ast_repair.py` — added off-by-one (+~100
  LOC) and missing-return (+~110 LOC) rewriters; updated dispatch +
  module docstring. 3 → 5 rewrites.
- `calm/llm_computer/tests/test_ast_repair.py` — added 25 new tests
  across the two new rewrites. 36 → 61.

### Modified docs
- `.claude/CLAUDE.md` — R53 section compressed to pointer list
  (529 → 446 LOC). Commit `f65c376`.
- `.claude/rules/augmentation_thesis.md` — empirical basis compressed
  to cluster table + pointer (509 → 469 LOC). Commit `9420e96`.
- `.claude/MEMORY/SESSION_HANDOFF.md` — this file; overwrites prior
  2026-04-20 walker-only handoff.

### Optional artifacts
None new (can_be_done.md + summarise.md were in prior session).

### Deleted
None.
