# Session Handoff — 2026-04-20 (continuation arc: +2 walker rewrites, cascade, LOC-cap, MBPP-N20 receipt)

## Goal

Continue the 7-next-steps arc. This session: implement all remaining
next-steps from prior handoff (empty-block walker, walker cascading,
LOC-cap sweep, force-fence generalization, MBPP N=20 re-run,
R53.37 long-N bench). Jacobian tier-3 (~2 weeks, speculative) out of
scope.

Workflow: hypothesis → build → test → commit → iterate. 6 commits
shipped + R14 in flight at session end.

## Completed (6 commits, `f2c120d` → `1c4e809` plus R14 in flight)

### Round 9 — Empty-block walker (commit `f2c120d`)

Sixth deterministic rewrite in `ast_repair.py`. Addresses R53.39
finding: Gemma emits `try:`/`except:`/`if:`/etc. with no body →
IndentationError at parse time. Line-scan detects compound-statement
headers whose next meaningful line is ≤ header indent (or EOF) and
inserts `<indent>    pass`.

Runs in the syntax-repair tier (before AST-walking rewrites, which
need parseable code). Idempotent — noops on valid code.

| metric | before | after |
|---|---:|---:|
| walker rewrites | 5 | 6 |
| unit tests | 61 | 75 (+14) |
| tests passing | 61/61 | 75/75 |

Handles: def, class, if, elif, else, for, async for, while, try,
except, finally, with, async with, match, case.

### Round 10 — Walker cascading (commit `4112837`)

Added `repair_cascade(code, error_output, max_passes=4)` generalizing
R53.38's scripted 2-pass chain. Calls `repair()` repeatedly on its own
output until no further rewrite applies or max_passes reached.

Useful cascades: `syntax_repair → empty_block`, `shadow_rename →
missing_return`, `empty_block → shadow_rename`. `repair()` single-pass
preserved for callers needing atomic behavior.

Kind field semantics:
- 0 passes: `'none'`
- 1 pass: that rewrite's kind
- 2+ passes: `'cascade:<k1>+<k2>+...'`

| metric | before | after |
|---|---:|---:|
| unit tests | 75 | 81 (+6) |
| max rewrites/call | 1 | 4 (configurable) |

### Round 11 — LOC-cap sweep (commit `765cc80`)

Only workflow.md was over the 500-LOC hard limit (551). Refactored via
migration, not trim — three sections moved to canonical rule files:

- "Safer-config for noisy-grad training" → `training.md` §"Safer-
  config for noisy-grad training" (R52.2 receipt)
- "GPU vs CPU decision rule for substrate training" → `training.md`
  §"GPU vs CPU for substrate training" (flipped canonical direction:
  was a pointer-back from training.md, now training.md owns it)
- "Substrate install workflow" (6-step checklist) → `Substrate.md`
  §"Install Workflow (checklist)"

Also compressed training.md's "Export & Serving" (duplicated
CLAUDE.md) + tightened Triton-autograd pitfall + GPU prereq snippet.

| file | before | after | delta |
|---|---:|---:|---:|
| workflow.md | 551 | 496 | -55 |
| training.md | 474 | 494 | +20 |
| Substrate.md | 338 | 366 | +28 |
| **All rules** | **5738** | **5731** | **-7** |

All 5 big files now ≤ 500 LOC hard limit. Duplications eliminated.

### Round 12 — Force-fence generalization (commit `cd3c919`)

Replaces hardcoded SIGNATURES table with
`_derive_signature_from_prompt()`:
1. explicit `def <fn_name>(<args>):` in prompt
2. backtick-wrapped `<fn_name>(<args>)` shorthand
3. fallback `def <fn_name>(text):`

Also swaps walker invocation from single `repair()` to
`repair_cascade()`. Gates top-level exec on `m`/`tok` in globals so
signature-derivation tests can import offline.

Derivation verified on CORPUS (6 problems): 3 function-based cases
derive correctly; 3 class-based cases (LinkedList, TokenBucket,
LRUCache) fall back to wrong `def ClassName(text):` — class force-
fence is a separate workstream.

### Round 13 — MBPP N=20 re-run (commit `9d24d29` setup + `1c4e809` fix)

Second wider-corpus walker test with R9 empty-block + R10 cascade +
MAX_TOKENS=2048 active. Raised MBPP_N 5 → 20.

**Result** (invalidated mid-flight by sandbox bug):

| outcome | count |
|---|---:|
| clean | 0/20 |
| walker_fixable | 0/20 |
| genuine_fail | 6/20 (ALL IndentationError — sandbox bug) |
| format_fail | 14/20 |

**Critical diagnostic during R13**: MBPP#1 (first_repeated_char)
GENUINE-FAIL'd with IndentationError despite R9 walker being active.
Offline repro showed walker DOES lift the canonical empty-except case.
Root cause (identified mid-run): `calm.sandbox.run_python`'s `_WRAPPER`
treats the script as potentially-expression-evaluable — splits last
line off as `_last`, execs `_body` separately. When the script ends
inside a `try/except` test harness, the final `print(...)` gets peeled
off as `_last`, leaving `except Exception as e:` with empty body in
`_body` → IndentationError at exec time.

Precedent: `scripts/r53_22_diagnose_csv.py` already appends `\npass\n`
to avoid exactly this. My r53_39 harness was missing it.

**Fix committed** (`1c4e809`):
```python
script = code + "\n\n" + "\n".join(harness) + "\npass\n"
```

R13 mid-run was on the buggy version. All 6 IndentationError problems
are bogus — they'd score against Gemma's actual output after re-run.
The 14 format_fails are real (extractor genuinely rejected) and
potentially correctable by force-fence generalization (R12).

### Round 14 — R53.37 long-N bench (running at session end)

Daemon running `scripts/r53_37_long_n_bench.py` at N=8192, 1 run per
config (fp16 KV / tq4 memo / tq4 fused). Expected wall time: ~60-70
min. GPU-discipline compliant (heavy_warmup 3s, cuda.Event, paired
A/B, correctness sanity).

Result will be in `/tmp/gemma_log`; monitor runs through session end.
If memo dominates fused at N=8K (first-principles prediction), the
runtime N-gate `128 < cached_kv_len < 2048` is confirmed optimal and
no further bench work needed.

## In Progress

- **R14**: long-N bench at N=8192, single-run. Monitor via
  `tail /tmp/gemma_log` for `[bench] SUMMARY` line. If incomplete at
  handoff load, continue monitoring or wait for completion.

## Uncommitted (unchanged; teammate-owned)

Same as prior handoff — teammate checkpoints (R51/R52), research
directories, minutes logs. No session-critical work left uncommitted.

## Key Findings

1. **Walker surface-area doubled**: 3 → 6 rewrites (shadow_rename,
   dict_synonym, syntax_repair, off_by_one, missing_return, empty_block)
   + cascade runner. 36 → 81 unit tests. +14 rewrite types shipped in
   two sessions.

2. **LOC-cap discipline now enforced**: all .claude/rules/*.md ≤ 500,
   CLAUDE.md ≤ 500. Refactor pattern is migrate-to-canonical, preserving
   information — not trim. 3 sections moved workflow.md → training.md
   / Substrate.md.

3. **R13 null + sandbox bug**: the 6 MBPP IndentationErrors were
   test-harness artifacts (sandbox wrapper split last line), not Gemma
   capability gaps. Walker wasn't firing because extracted code parsed
   cleanly standalone — the bug was in harness-concat + sandbox line-
   split. Fix committed for next run.

4. **Force-fence generalizes**: signature auto-derivation works for
   function problems. Class problems (LinkedList, TokenBucket,
   LRUCache) need a separate class-fence mechanism (out of scope).

5. **R14 long-N bench artifact now measurable**: prior session shipped
   the bench as code only; this session actually runs it. Result
   pending at handoff.

## Next Steps (ordered by lift)

1. **Re-run R13 MBPP N=20 with fixed sandbox** (~40 min, HIGH lift) —
   commit `1c4e809` pending measurement. With 14 format_fails and 6
   previously-bogus genuine_fails, this should reveal the real
   walker lift count. If format_fails resolve with the generalized
   force-fence (R12), this could be the first >0 walker lift on MBPP.

2. **Force-fence + MBPP combined** (~1 hour, HIGH lift) — apply
   R53.38v2 force-fence to each MBPP problem's derived signature.
   Convert format_fails → extractable code → walker-eligible.

3. **Class-aware force-fence variant** (~3 hours, MEDIUM) — for
   problems like LinkedList: emit `class LinkedList:\n    def __init__(self, ...):\n`
   prefix instead of function `def`. Handle 3 R53.0 class problems +
   MBPP class problems.

4. **Empty-if/empty-for walker coverage MBPP** — investigate whether
   R13's 14 format_fails contain recoverable patterns the empty-block
   walker would catch after force-fence surfaces the code.

5. **R14 long-N bench interpretation** — pending completion. If memo
   > fused at N=8K as predicted, confirm runtime gate is optimal. If
   crossover observed, investigate kernel restructuring (one-Triton-
   kernel-per-layer).

6. **Jacobian-weighted tier-3 distillation** (~1-2 weeks, speculative)
   — still out of scope per prior handoff. Re-visit only if a card-
   specific workstream demands it.

## Key Context

**Decision rationale (WHY)**:

- **Cascade vs single-pass**: cascade defaults to max_passes=4 (enough
  for syntax_repair → empty_block → shadow_rename → dict_synonym
  chains). Single-pass `repair()` preserved for atomic callers.
- **Empty-block walker runs in syntax-repair tier**: fires on ANY
  unparseable code (not just explicit IndentationError text), because
  "code already parses" gate keeps it safe on valid input.
- **R13 sandbox bug is universal**: affects any test harness that
  wraps per-test try/except and concatenates. Moving to `run_python`
  with explicit `statements_only=True` flag (not implemented) would
  be a cleaner long-term fix than appending `pass`.
- **Force-fence derivation priorities**: prompt explicit `def` >
  backtick shorthand > fallback. The fallback is `def <name>(text):`
  which works for string-input problems but fails on class problems.

**Measurement discipline caveats**:

- R13 result is INVALIDATED (sandbox bug) — don't cite `0/20 walker
  lifts` as a capability gap; it's a harness artifact.
- R14 bench is single-run (direction only, magnitude soft per
  workflow.md §"GPU bench discipline"). Re-run median-of-3 for
  publishable comparison.
- Empty-block walker coverage is MBPP-specific untested — the 14
  format_fails may or may not be reachable after force-fence.

**Do NOT relitigate**:

- R9-R12 shipped with tests + refactor-to-canonical. Leave alone.
- R13's 14 format_fails are real, 6 genuine_fails are sandbox-bogus.
  Re-run clarifies which.
- Walker is NOT firing because extracted code always parses (by
  extractor invariant). The empty-block walker fires on unparseable
  code only; sandbox-boundary IndentationErrors are a separate class
  handled by the `\npass\n` trailing fix.

**Failed approaches (cite SHAs)**:

- R13 first launch hit `ImportError: repair_cascade not found` because
  daemon had stale `ast_repair` cached in `sys.modules`. Fixed by
  explicit `importlib.reload()` at top of `run_mbpp_walker()`.
  Receipt in commit `9d24d29` (subsequent reload fix not separately
  committed — rolled into R13 setup).
- R13 mid-run sandbox-boundary IndentationError appeared as 6 GENUINE
  FAIL; root cause was sandbox's `_WRAPPER` expression-eval line
  split, not Gemma's code. Fix committed `1c4e809`.

**Runtime state at session end**:
- Branch: `feature/multi-agent-qwen` at `1c4e809` (or further if R14
  commit fires).
- Daemon: PID 934814 running (refresh if needed), Gemma pre-loaded.
  R14 in flight at handoff time.
- GPU: ~5-7 GB.
- All LOC-cap files ≤ 500. `wc -l .claude/rules/*.md
  .claude/CLAUDE.md | sort -rn` confirms.
- 81 ast_repair unit tests passing.

## Files Changed (session-shipped)

### Modified code
- `calm/llm_computer/facades/ast_repair.py` — added empty_block
  rewriter (+~130 LOC) + repair_cascade() (+~50 LOC) + 6th rewrite
  docstring update.
- `calm/llm_computer/tests/test_ast_repair.py` — added 20 new tests
  (14 empty_block + 6 cascade). 61 → 81.
- `scripts/r53_38_force_fence.py` — generalized signature derivation
  + TARGETS loop + import-safety gate.
- `scripts/r53_39_mbpp_walker.py` — MBPP_N 5→20, repair_cascade
  switch, reload fix, `\npass\n` sandbox fix.

### Modified docs
- `.claude/rules/workflow.md` — -55 LOC (migrated 3 sections).
- `.claude/rules/training.md` — +20 LOC (received 2 migrated sections,
  compressed Export & Serving).
- `.claude/rules/Substrate.md` — +28 LOC (received 1 migrated section).
- `.claude/MEMORY/SESSION_HANDOFF.md` — this file; overwrites prior
  session handoff.

### New files
None (all session work landed in existing files).

### Deleted
None.
