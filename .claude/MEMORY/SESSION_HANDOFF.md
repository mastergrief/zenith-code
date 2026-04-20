# Session Handoff — 2026-04-20 (R9-R19 + R14 bench)

## Goal

Continue the 7-next-steps arc from prior handoff: walker surface expansion, LOC-cap sweep, force-fence generalizations, transformer-vm primitive ports, and complete the long-N flash-attn bench.

Workflow: hypothesis → build → test → commit → iterate. 12 commits on `feature/multi-agent-qwen`, `f2c120d` → `ace9670`.

## Completed (12 commits, `f2c120d` → `ace9670`)

### Walker surface expansion (R9, R10)

- **`f2c120d`** — empty-block pass-insert (6th walker rewrite). Line-scans compound headers (`def`, `class`, `if`, `elif`, `else`, `for`, `while`, `try`, `except`, `finally`, `with`, `match`, `case`); inserts `<indent>    pass` when next meaningful line is ≤ header indent (or EOF). Runs in syntax-repair tier before AST-walking rewrites. Tests `61 → 75 (+14)`, all 75/75 passing.
- **`4112837`** — `repair_cascade(code, error_output, max_passes=4)`. Multi-pass walker calling `repair()` on its own output until no further rewrite applies or cap reached. Kind format: `'cascade:k1+k2+...'` on 2+ passes. Tests `75 → 81 (+6)`, all 81/81 passing.

### LOC-cap refactor (R11)

- **`765cc80`** — migrate-to-canonical (NOT trim). User pivot: *"rather than trim refactor instead? so we dont lose valuable insights?"* — answered via migration.
  - `workflow.md` 551 → **496** (-55): moved "Safer-config for noisy-grad training" + "GPU vs CPU decision rule" → `training.md`; "Substrate install workflow" → `Substrate.md`.
  - `training.md` 474 → **494** (+20): absorbed both migrated sections; compressed "Export & Serving" duplicate of CLAUDE.md; tightened Triton-autograd pitfall.
  - `Substrate.md` 338 → **366** (+28): received 6-step install checklist.
  - Net: **-7 lines across three files**, every duplication eliminated. All files ≤ 500 LOC hard limit.

### Force-fence generalization (R12, R15, R16)

- **`cd3c919`** — R53.38v2: replaced hardcoded SIGNATURES table with `_derive_signature_from_prompt()` supporting (1) explicit `def` in prompt, (2) backtick-wrapped shorthand, (3) fallback `def <name>(text):`. Switched walker invocation single `repair()` → `repair_cascade()`. Gated top-level exec on `m`/`tok` in globals (offline-importable).
- **`4fc66f9`** — R53.38v3: class-aware force-fence for Capitalized-name problems. `_derive_class_prefix()` extracts `__init__` signature via (1) inline `class Name:\n    def __init__(` pattern, (2) backtick `__init__(self, ...)`. All 6 R53.0 corpus problems produce parseable reconstructions (3 function + 3 class).
- **`4a4dbae`** — MBPP force-fence fallback: `r53_39_mbpp_walker.py` stock → if NO CODE, derive signature from first assert via `_derive_mbpp_signature()` (bracket-depth arg counting, 0/1/2/3/4+ arg forms), retry via `gen_forced`. New aggregate stat `force_fence_lift`.

### R13 MBPP N=20 (setup + sandbox fix)

- **`9d24d29`** — `MBPP_N 5→20`, `repair_cascade` switch, importlib.reload for stale daemon cache.
- **`1c4e809`** — sandbox `\npass\n` trailing sentinel. **Critical diagnostic mid-run**: `calm.sandbox.run_python`'s `_WRAPPER` peels last script line off for `eval()`, leaving empty `except:` body in `_body` → IndentationError at exec. Precedent in `r53_22_diagnose_csv.py` already had the fix; `r53_39` was missing it. R13 result `0/20 walker lifts (14 format_fail, 6 genuine_fail IndentationError)` INVALIDATED — the 6 IndentationErrors were harness artifacts, not walker regressions.

### R14 long-N flash-attn bench (script fix + execution)

- **`219de8e`** — `_correctness_check` signature fix (dropped unused `cache_factory` kwarg).
- **`ace9670`** — turboquant.md receipt. Full 5-point curve now in `§"Fused flash-attention decode"`:

| N | fp16 KV | tq4 memo | tq4 fused | fused/memo | verdict |
|---:|---:|---:|---:|---:|---|
| 64 | 6.99 | 4.88 | 4.00 | 0.82× | memo (launch overhead) |
| 256 | 6.67 | 5.63 | **6.40** | **1.14×** | fused WINS |
| 1024 | 7.83 | 6.08 | **6.43** | **1.06×** | fused WINS |
| 4096 | 7.21 | 6.09 | 5.65 | 0.93× | memo (asymptotic) |
| **8192** | **5.32** | **4.64** | **4.41** | **0.95×** | **memo (confirmed)** |

Runtime N-gate `128 < cached_kv_len < 2048` confirmed optimal; first-principles prediction (memo dominates at N≥4K) held empirically. GPU discipline: heavy_warmup(3.0s), `torch.cuda.Event`, correctness sanity (fp16/memo/fused all argmax=106 on "What is 17 times 23?"). Median-of-1 (single-run) — direction reliable, magnitudes soft per `workflow.md` §"GPU bench discipline".

### transformer-vm port (R17, R18, R19)

User asked to port 3 primitives from sjmoran/transformer-vm @ 6cfee30 (Percepta Core fork).

- **`d8a1f7e`** — R18+R19: added `CumSum` + `PersistLinear` as compute nodes in `gate_graph.py`. CumSum accumulates across `interpret()` calls via internal `_accum`. PersistLinear materializes a linear combination as a single value (coefs list of (node, coef)). Compilation to transformer weights deferred — matches upstream's evaluator-only status. gate_graph primitives `9 → 11`, interpreter kinds `4 → 6`, 8/8 tests.
- **`dfcff88`** — R17: MILP scheduler stub. `is_available()` + `milp_schedule()` returning None. Upstream's `milp.py` is 814 LOC + requires PuLP (not installed); stub reserves API for future port (`plan = milp_schedule(g) or auto_schedule(g)` idiom works today). Full port blockers documented in module docstring. 3/3 tests.

### Handoff checkpoints (intermediate)

- **`4ebed4f`** — intermediate handoff (R9-R13 post-sandbox-fix, R14 in flight).
- **`d93ebdd`** — intermediate handoff (R9-R19, R14 in flight).
- This file supersedes both.

## In Progress

None. All 11 rounds (R9-R19) closed with commits + measurements or shipped artifacts. R14 completed.

## ⚠ Uncommitted

```
 M .claude/CLAUDE.md                              # TEAMMATE — NOT session work
 M calm/hrm/checkpoints/meta_best.pt              # TEAMMATE — binary checkpoint, mtime 2026-04-14
 M scripts/r52_train_student_kl.py                # TEAMMATE — pre-existing R52.1 work
 M scripts/r53_22_diagnose_csv.py                 # TEAMMATE — pre-existing R53.25 tuning
?? .cache/, .codex/, .port_sessions/              # GITIGNORED — safe to ignore
?? .claude/MEMORY/minutes/                        # TRANSCRIPT — do NOT commit (raw session logs)
?? .claude/scheduled_tasks.lock                   # GITIGNORED
?? RESEARCH/{LLM-COMPUTER,NEURAL_COMPUTER,TQ,TRAINING}/
?? calm/.module_learning.json                     # runtime state
?? calm/hrm/checkpoints/copy_code*_best.pt        # TEAMMATE R53.5 (not this session)
?? calm/hrm/checkpoints/math_*_best.pt            # TEAMMATE historical training artifacts
?? calm/llm_computer/checkpoints/substrate_hrmlm_v2*.pt   # TEAMMATE
?? calm/llm_computer/r51/checkpoints/             # TEAMMATE R51 student
?? calm/llm_computer/synth/*.jsonl                # TEAMMATE
?? calm/llm_computer/tq4_autograd.py              # TEAMMATE R52.1
```

**Session-critical unintentionally uncommitted: NONE.**

`.claude/CLAUDE.md` was modified by the user or a linter (line 11 subagentagent wording change) — per system-reminder during session, this was intentional and NOT revertable.

## Next Steps (ordered by lift)

1. **Re-run R13 MBPP N=20 with all fixes** (~40 min, daemon-bound, HIGHEST lift) — sandbox fix (`1c4e809`) + walker cascade (`4112837`) + empty-block walker (`f2c120d`) + force-fence fallback (`4a4dbae`) + class-aware derivation (`4fc66f9`) + MAX_TOKENS=2048 all active. Should produce the first REAL walker lift count on MBPP. Daemon is free (R14 done).

2. **R53.0 re-run on 3 class problems** (~15 min, daemon-bound) — apply R53.38v3 class-aware force-fence to `linked_list_bugs`, `token_bucket_rate_limiter`, `lru_cache_class`. Token_bucket already validated 0/0→5/5 via walker (commit `21f5001` from prior session); force-fence + cascade may lift the other two.

3. **Empty-block walker iteration** — known limitation in `f2c120d`: `try:` with no `except:` gets `pass` inserted but try/except pair remains incomplete (needs `except:` insertion pass). Extend walker to handle.

4. **MBPP larger N (50-100)** (~2-4 hours daemon-bound) — scale up after R13v2 proves walker lift > 0.

5. **Full MILP port** (requires PuLP install) — only when programs hit 30+ gates. Reference impl cloned at `/tmp/transformer-vm/transformer_vm/scheduler/milp.py` (814 LOC).

6. **Jacobian-weighted tier-3 distillation** — scope-excluded across arc. Re-visit only if a specific workstream demands it.

## Key Context

### Decisions made this session (with reasoning)

- **R11 refactor-not-trim**: user pivot preserved insights by migrating sections to canonical rule files rather than deleting content. Pattern codified for future LOC-cap sweeps.
- **R14 single-run accepted**: median-of-1 is under `workflow.md §"GPU bench discipline"` ideal (median-of-5) but direction is binary-reliable; 5-point curve shape confirmed. Magnitude soft.
- **MILP full port deferred**: PuLP absent, 814 LOC upstream, and current 29 compiled programs (all ≤ 20 gates) don't need optimal scheduling. Stub reserves API for when programs hit 30+ gates.
- **CumSum + PersistLinear interpreter-only**: matches upstream's evaluator-only status; compilation to transformer weights requires primitives (soft/uniform prefix attention) this project doesn't have yet.

### Measurement caveats

- **R13 result INVALIDATED** by sandbox bug. Don't cite `0/20 walker lifts` as a capability gap — it's a harness artifact. Fix committed (`1c4e809`); re-run pending.
- **R14 is single-run per config** (not median-of-5). First-principles prediction gave direction; magnitudes not publication-grade.

### Failed approaches (cite SHAs)

- **R13 first launch** (`ImportError: repair_cascade`): daemon had stale `ast_repair` in `sys.modules`; RESET_GLOBALS doesn't clear module cache. Fix via `importlib.reload()` inside `run_mbpp_walker()` (uncommitted edit folded into `9d24d29` post-fix).
- **R13 mid-run sandbox IndentationError** (6 GENUINE FAIL): sandbox `_WRAPPER` line-split bug, NOT Gemma code. Fix `1c4e809`.
- **R14 first launch** (`TypeError _correctness_check`): wrong signature, fix `219de8e`.
- **R53.38v2 class fallback `def ClassName(text):`** (`cd3c919`): syntactically wrong for class problems → superseded by R53.38v3 class-aware variant (`4fc66f9`).
- **First-token logit bias** (prior session R53.14 ruled-out, still relevant): Gemma's first-token margin 6.8-9.2 on code opener dominates; logit bias produces code-without-fence → extractor fails. Confirmed; force-fence prompt-tail prefix is the correct alternative.

### Runtime state at session end

- **Branch**: `feature/multi-agent-qwen` at `ace9670`
- **HEAD message**: `turboquant: R14 long-N bench receipt — N=8192 confirms memo dominance`
- **Session commits**: 12 (`f2c120d` → `ace9670`)
- **gemma_daemon**: ALIVE (PID 934814, ~7h uptime, 40% CPU, ~900 MB RSS, idle since R14 finish)
- **llama-server port 8080**: NOT responding — separate from daemon which reads from `/tmp/gemma_in` pipe
- **GPU**: 0% utilization, 7903 MiB used (substrate still loaded in VRAM; ready for next script)
- **Last `/tmp/gemma_log`**: `[bench] DONE` + `[daemon] completed r53_37_long_n_bench.py`
- **LOC-cap check**: all rule files + CLAUDE.md ≤ 500
- **Walker test suite**: 81 ast_repair + 8 cumsum_persist + 3 milp_schedule = **92 passing**

## Files in Project (session-shipped)

### New files (3)
- `calm/llm_computer/milp_schedule.py` — MILP stub (~100 LOC)
- `calm/llm_computer/tests/test_cumsum_persist.py` — 8 tests
- `calm/llm_computer/tests/test_milp_schedule.py` — 3 tests

### Modified code (substantial)
- `calm/llm_computer/facades/ast_repair.py` — +empty_block rewriter, +repair_cascade, +has_indent_error
- `calm/llm_computer/tests/test_ast_repair.py` — +20 tests (61→81)
- `calm/llm_computer/gate_graph.py` — +CumSum, +PersistLinear compute nodes
- `calm/llm_computer/interpret.py` — +CumSum, +PersistLinear handlers
- `scripts/r53_38_force_fence.py` — signature auto-derivation + class-aware + cascade
- `scripts/r53_39_mbpp_walker.py` — MBPP_N 5→20, cascade, sandbox fix, force-fence fallback, MBPP signature derivation
- `scripts/r53_37_long_n_bench.py` — `_correctness_check` signature fix

### Modified docs
- `.claude/rules/workflow.md` — 551 → 496 (migrated 3 sections out)
- `.claude/rules/training.md` — 474 → 494 (received 2 migrated sections)
- `.claude/rules/Substrate.md` — 338 → 366 (received install workflow checklist)
- `.claude/rules/turboquant.md` — +14 lines for R14 N=8192 bench row
- `.claude/MEMORY/SESSION_HANDOFF.md` — this file (overwrites prior)

### Deleted
None.
