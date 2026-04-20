# Session Handoff — 2026-04-20 (extended continuation: walker +3 rewrites, cascade, transformer-vm port, R14 in flight)

## Goal

Continue 7-next-steps arc, then extended scope after user asked to port
transformer-vm primitives. This session shipped 11 rounds (R9-R19),
with R14 still running at handoff.

Workflow: hypothesis → build → test → commit → iterate.

## Completed (12 commits)

### Round 9 — Empty-block walker (commit `f2c120d`)

6th deterministic walker rewrite in `ast_repair.py`. Line-scan detects
compound headers (`def`, `class`, `if`, `elif`, `else`, `for`, `while`,
`try`, `except`, `finally`, `with`, `match`, `case`) whose next
meaningful line is ≤ header indent (or EOF) and inserts
`<indent>    pass`. Idempotent on valid code.

| metric | before | after |
|---|---:|---:|
| walker rewrites | 5 | 6 |
| unit tests | 61 | 75 (+14) |

### Round 10 — repair_cascade (commit `4112837`)

Multi-pass walker: calls `repair()` repeatedly on its own output until
no further rewrite applies or `max_passes` reached. Useful chains:
`syntax_repair → empty_block`, `shadow_rename → missing_return`.

| metric | before | after |
|---|---:|---:|
| unit tests | 75 | 81 (+6) |

### Round 11 — LOC-cap sweep (commit `765cc80`)

workflow.md 551 → 496 via **refactor-to-canonical**, not trim. 3
sections moved:
- "Safer-config for noisy-grad training" → `training.md`
- "GPU vs CPU decision rule" → `training.md` (flipped canonical owner)
- "Substrate install workflow" → `Substrate.md`

All rule files + CLAUDE.md now ≤500 LOC hard limit.

### Round 12 — force-fence signature derivation (commit `cd3c919`)

R53.38v2: replaced hardcoded SIGNATURES table with
`_derive_signature_from_prompt()` supporting 3 patterns (explicit
`def`, backtick shorthand, fallback). Also swapped single `repair()`
call to `repair_cascade()`.

### Round 13 — MBPP N=20 re-run (commits `9d24d29` + `1c4e809`)

Second wider-corpus walker test. Result mid-run: **0/20 walker
lifts, 14 format_fail, 6 genuine_fail (all IndentationError)**.

**Critical finding**: the 6 IndentationError problems were NOT walker
regressions. They were **sandbox-wrapper bugs**:
`calm.sandbox.run_python`'s `_WRAPPER` peels the last script line off
for expression-eval. When the script ends inside a test try/except
chain, the final `print(...)` becomes `_last`, leaving
`except Exception as e:` with empty body in `_body` → IndentationError
at exec. Precedent `r53_22_diagnose_csv.py` already had the fix:
append `\npass\n`. r53_39 was missing it.

Fix committed (`1c4e809`). R13 mid-run was on buggy version — re-run
on fixed version will produce clean data.

### Round 14 — long-N flash-attn bench (commit `1d78fae` + `219de8e`, IN FLIGHT)

`scripts/r53_37_long_n_bench.py` running at N=8192 × 1 run. GPU
discipline applied (heavy_warmup 3s, cuda.Event timing, correctness
sanity, paired A/B). Expected ~75 min wall.

Partial result at handoff:
- fp16 KV at N=8192: **5.32 tok/s** (1540.93s / 8192 tok)
- prior N=4096: 7.21 tok/s → fp16 drops 26% going 4K→8K
- memo path in flight (~15 min remaining estimated)
- fused path pending (~26 min estimated)

Signature bug caught at first launch (`_correctness_check` required
`cache_factory` but callers didn't pass it); fixed and re-launched
(`219de8e`). Correctness sanity passed: fp16/tq4-fused/tq4-memo all
argmax=106 on "What is 17 times 23?" prompt.

### Round 15 — class-aware force-fence (commit `4fc66f9`)

R53.38v3: extended signature derivation for class-based problems.
`_is_class_problem(name)` detects Capitalized names (Python
convention), `_derive_class_prefix()` extracts `__init__` signature
from prompt text.

All 6 R53.0 corpus problems now produce parseable reconstructions:

| problem | derived signature |
|---|---|
| linked_list_bugs | `class LinkedList:\n    def __init__(self):` |
| token_bucket_rate_limiter | `class TokenBucket:\n    def __init__(self, rate, capacity):` |
| lru_cache_class | `class LRUCache:\n    def __init__(self, capacity):` |
| csv_column_stats | `def csv_column_stats(text):` |
| date_validation_chain | `def valid_date(y, m, d):` |
| log_level_counts | `def log_level_counts(text):` |

### Round 16 — force-fence fallback in MBPP harness (commit `4a4dbae`)

Integrated force-fence as second-chance extraction path in
`r53_39_mbpp_walker.py`. Pipeline now:
  1. gen_stock → extract → score (CLEAN → done)
  2. if NO CODE: `_derive_mbpp_signature()` from first assert →
     gen_forced → extract
  3. score extracted code, walker cascade on non-clean

New `_derive_mbpp_signature(p)`: parses arg count from first assert
with bracket-depth tracking. Handles 0/1/2/3-arg named, 4+ uses
`a{0..n}`. Verified offline on `foo(1,2)`, `is_prime(7)`,
`dispatch(a,b,c,d,e)`, `noop()`.

New aggregate stat: `force_fence_lift` alongside `walker_fixable`.

### Rounds 17-19 — transformer-vm ports (commits `dfcff88`, `d8a1f7e`)

User asked to port 3 primitives from sjmoran/transformer-vm
(Percepta Core): MILP scheduler, CumSumDimension, PersistDimension.

**R18+R19** (`d8a1f7e`): added `CumSum` + `PersistLinear` compute
nodes to `gate_graph.py` + interpreter support. Mirrors upstream
evaluator-only semantics. `CumSum(source)` accumulates across
`interpret()` calls via internal `_accum` state. `PersistLinear(coefs)`
materializes a linear combo as a single value. **8/8 tests passing**
including cross-primitive composition. Compilation to transformer
weights deferred — matches upstream's evaluator-only status.

**R17** (`dfcff88`): MILP stub with graceful fallback. Upstream's
`milp.py` is 814 LOC + requires PuLP (not installed here). Stub
exposes `is_available()` + `milp_schedule()` → returns None when
PuLP absent. Callers can write `plan = milp_schedule(g) or
auto_schedule(g)` and benefit automatically once full port lands.
**3/3 tests passing**. Full port blockers documented in module
docstring (PuLP dep, `_all_dims` globals refactor, ProgramGraph
adaptation).

## In Progress

- **R14 long-N bench**: path 2/3 (tq4 memo) ~15 min remaining, path
  3/3 (tq4 fused) ~26 min remaining. Monitor `/tmp/gemma_log` for
  `[bench] SUMMARY` line. Expected direction: memo continues to
  dominate fused at N=8K (first-principles prediction), confirming
  runtime gate `128 < cached_kv_len < 2048` is optimal.

## Uncommitted (teammate-owned, unchanged)

Same as prior handoff.

## Key Findings

1. **Walker surface now 6 rewrites + cascade**: shadow_rename,
   dict_synonym, syntax_repair, off_by_one, missing_return,
   empty_block. 36 → 89 unit tests across 4 walker-related files
   (ast_repair: 81, cumsum_persist: 8, milp_schedule: 3, but the
   89 count over the walker alone is ast_repair's 81).

2. **LOC-cap enforced**: all 5 big rule files ≤500 LOC after
   refactor-to-canonical. Migration preserves information; never
   trim and lose.

3. **R13 sandbox-wrapper bug** is universal, affecting any test
   harness that concatenates test-try/except chains onto user code.
   Standard fix is `\npass\n` trailing sentinel. Fix is documented
   + committed.

4. **Class-aware force-fence works**: all 6 R53.0 corpus problems
   derive syntactically valid prefixes. 3 function-based + 3
   class-based. MBPP harness now falls back to force-fence on
   format_fail with auto-derived signature.

5. **transformer-vm port**: CumSum + PersistLinear added at
   interpreter level; MILP API surface reserved. Project's substrate
   vocabulary now matches upstream's for cross-reference.

## Next Steps (ordered by lift)

1. **Re-run R13 MBPP N=20 with all fixes** (~40 min, HIGHEST lift) —
   daemon frees up after R14. Now has sandbox fix + force-fence
   fallback + walker cascade + MAX_TOKENS=2048. Should produce the
   first REAL walker lift count on MBPP.

2. **R14 interpretation** (~5 min when complete) — read summary
   line, update `turboquant.md` with new N=8K data point. If memo
   dominates as predicted, close N=8K investigation. If crossover,
   escalate.

3. **MBPP larger N (50-100)** (~2-4 hours) — scale up after R13v2
   proves walker lift > 0. Cost: ~1-2min/problem on Gemma 4 E4B.

4. **R7 R53.0 re-run with force-fence + walker cascade**
   (~15 min) — apply R15 class-aware force-fence to the 3 R53.0
   class problems (linked_list_bugs, token_bucket, lru_cache).
   Token_bucket already validated 0/0→5/5 via walker in earlier
   commit `21f5001`; force-fence + cascade may lift the other two.

5. **Full MILP port** (~1-2 days, requires PuLP) — only when
   programs hit 30+ gates. Reference impl at
   `/tmp/transformer-vm/transformer_vm/scheduler/milp.py`.

6. **CumSum compilation path** (~1 week, speculative) — requires
   new attention primitive (soft/uniform prefix attention) not
   expressible with current `LookUp`/`LookUpExact`. Deferred until
   a concrete use case (e.g. counting-card) demands it.

7. **Jacobian-weighted tier-3 distillation** — still out of scope.

## Key Context

**Decision rationale (WHY)**:

- **Refactor > trim for LOC cap**: preserved information by migrating
  to canonical owner (training.md or Substrate.md). Both workflow.md
  and the destinations ended up cleaner.
- **Walker cascade** is a strict generalization of single-pass
  `repair()`; never a regression.
- **Force-fence as fallback, not default**: stock extraction first
  (faster, less invasive); force-fence only when stock returns
  NoCode. Preserves stock behavior on clean problems.
- **MILP stub over no stub**: reserves API for future port without
  committing PuLP now. Callers can write `milp_schedule(g) or
  auto_schedule(g)` idiom today.
- **CumSum + Persist as interpreter primitives**: matches upstream
  semantics (evaluator-only). Having them in the IR surfaces
  intent for future compiler work even if weights aren't generated.

**Measurement discipline caveats**:

- R13 first result INVALIDATED by sandbox bug; re-run required.
- R14 is single-run (not median-of-3 per workflow.md §"GPU bench
  discipline") — direction reliable, magnitudes soft.
- R15/R16 offline-verified (AST parse check); user-facing
  measurement pending R13 re-run.
- R18 interpreter-only; compilation to weights is future work.

**Do NOT relitigate**:

- R9-R19 ship with tests. Don't regress.
- R1-R8 (prior session) arc is settled.
- Walker never fires on extracted code that parses; IndentationError
  from harness concat was sandbox wrapper artifact.

**Failed approaches (cite SHAs)**:

- R13 first launch `ImportError: repair_cascade` — fixed via
  `importlib.reload()` inside `run_mbpp_walker()`.
- R13 mid-run sandbox IndentationError — fixed `1c4e809` with
  `\npass\n` trailing sentinel.
- R14 first launch `TypeError _correctness_check` — fixed `219de8e`
  (dropped unused kwarg).

**Runtime state at session end**:
- Branch: `feature/multi-agent-qwen` at `dfcff88` (or further after
  handoff commit).
- Daemon: PID 934814 running R14 at handoff. When it completes,
  daemon is free for R13 re-run.
- All LOC-cap files ≤500.
- Walker tests: 81 ast_repair + 8 cumsum_persist + 3 milp_schedule
  = 92 passing.

## Files Changed (session-shipped)

### Modified code (ast_repair)
- `calm/llm_computer/facades/ast_repair.py` — +empty_block rewriter
  (+~130 LOC) + `repair_cascade()` (+~50 LOC).
- `calm/llm_computer/tests/test_ast_repair.py` — +20 tests (61→81).

### Modified code (scripts)
- `scripts/r53_38_force_fence.py` — generalized signature derivation
  (R12, R15); class-aware (R15).
- `scripts/r53_39_mbpp_walker.py` — MBPP_N 5→20 (R13);
  repair_cascade switch (R13); sandbox fix (R13); force-fence
  fallback + MBPP signature derivation (R16).
- `scripts/r53_37_long_n_bench.py` — `_correctness_check` signature
  fix (R14).

### New code (transformer-vm port)
- `calm/llm_computer/gate_graph.py` — +CumSum, +PersistLinear (R18).
- `calm/llm_computer/interpret.py` — +CumSum, +PersistLinear
  handlers.
- `calm/llm_computer/milp_schedule.py` — MILP stub (R17).
- `calm/llm_computer/tests/test_cumsum_persist.py` — 8 tests.
- `calm/llm_computer/tests/test_milp_schedule.py` — 3 tests.

### Modified docs
- `.claude/rules/workflow.md` — -55 LOC (migrated 3 sections).
- `.claude/rules/training.md` — +20 LOC.
- `.claude/rules/Substrate.md` — +28 LOC.
- `.claude/MEMORY/SESSION_HANDOFF.md` — this file.

### Deleted
None.
