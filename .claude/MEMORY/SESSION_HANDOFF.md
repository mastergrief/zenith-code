# Session Handoff — 2026-04-19 (R53.19 → R53.34 + doc sweep)

> **Postscript 2026-04-20** — the R53.34 "fused flash-attn is 8-10% slower
> at all N≤1024, shipped default OFF" disposition below is **partially
> superseded**. A clean re-run of `scripts/r53_phase2_bench.py` this session
> showed the curve is **non-monotonic**: fused wins +14% at N=256 and +6%
> at N=1024, loses -18% at N=64 and -7% at N=4096. Default flipped to
> `_use_fused_flash_attn=True` with runtime N-gate
> `128 < kv_cache.layer_pos[kv_src] < 2048` in
> `calm/llm_computer/gemma_substrate.py:_forward_layer`. Long R53 eval
> (AdaptiveBudget up to 16K) falls back to Phase 1 memo past N=2048 — no
> regression on long-decode workloads. Full bench table + rationale in
> `.claude/rules/turboquant.md` §"Fused flash-attention decode" and
> `tracing_roadmap.md` Round 53.34 revised row. Historical entries below
> retained as receipt; current-state truth is the rules.

## Goal

Drive the R53 arc past its null plateau. Specifically: diagnose why R53.19 v3 had
hit 26/26 but csv_column_stats + token_bucket sat at 0/0 despite shipped sandbox
fix, substrate install experiments, import-injection, mechanical repair. Then
port TurboQuant's fused flash-attn techniques to our Triton stack; rewrite
`.claude/rules/` + CLAUDE.md to reflect the session's shipped state.

## Completed (25 commits, 91e1e04 → 16817ad + teammate work)

### R53 kernel + eval stack

- **R53.19** (`a8729a0`): MAX_TOKENS 250 → 400 after SWA fix removed the 512-cap budget
- **R53.20a** (`ec8887f`): re-ran R53.14 substrate-RAG eval with SWA fix active.
  **NEGATIVE**: still -9.3pp regression (stock 25/27, prompt-RAG 25/27,
  substrate-RAG 10/12). Root cause is install-mechanism (L41 CardSlot + FirstTokenHook),
  NOT SWA bug. Gemma's first-token on code is confidently a fence/whitespace
  opener (logit margin 6.8-9.2); `min_margin=0.5` never gates; hook always fires
  on HIT → forces "def"/"class" → code-without-fence → extractor fails.
- **R53.20b** (`scripts/r53_20b_stacked.py`): substrate + prompt-RAG stacking.
  NEGATIVE: -7 tests vs R53.19 v3's 26/26 → 19/21. Substrate disruption pollutes
  attempt 1 output; repair retries can't recover.
- **R53.20 writeup** (`1a85b0c`): `.claude/MEMORY/evals/2026-04-19_r53_substrate_rag_null.md`
- **R53.21** (`b9512ec`): mechanical import injection (`COMMON_IMPORTS` table,
  50-entry). Fires correctly on csv (`StringIO + csv`) + log_level (`import re`)
  but null — injected imports still 0/0 because sandbox was blocking transitive os.
- **R53.22** diagnostic (`scripts/r53_22_diagnose_csv.py`): isolated the
  sandbox `ImportError: blocked: os` via `import statistics` transitive load.
- **R53.23** (`5dc2dc1`): sandbox fix — pre-import ~23 stdlib modules
  (re, math, random, time, datetime, hashlib, base64, collections, itertools,
  functools, bisect, heapq, copy, csv, statistics, typing, enum, dataclasses,
  abc, struct, decimal, fractions, textwrap) BEFORE `_safe_import` hook.
  User `import statistics` now hits sys.modules cache without triggering new
  transitive `os` load. User `import os` still blocked. Verified via 3-line test.
- **R53.24** (combined sandbox+import stack, minutes L969-1060): 26/26 null —
  sandbox fix + import injection together produced no lift over R53.19 v3's
  26/26 baseline. Diagnostic round; ruled out both mechanisms as the ceiling
  and forced the MAX_TOKENS experiment in R53.25.
- **R53.25** (`c6a6582`): MAX_TOKENS 400 → 900. **BEST R53 RESULT: 32/32.**
  Lifts `log_level_counts` 0/0 → 6/6 purely from budget. The prior 4 null rounds
  (R53.19 v3 through R53.24) were budget-starved the whole time.
- **R53.26 categorizer** (`scripts/r53_21_import_inject.py`): TypeError regex
  generalized from `'int'` to `r"TypeError: '(\w+)' object is not callable"` —
  matches int/float/str/list/dict.
- **R53.27** (fast diagnostic at 900 tok, minutes L1163-1201): interstitial
  re-run confirming R53.25's 32/32 was the genuine budget unlock, not a
  measurement artifact. Drove the escalation 900 → 2048 → 8192 → 16K default.
- **R53.28** (`069c614`): multi-token `KVCacheTq4.update(S≥1)` with per-layer
  `layer_pos[l]` tracking; `trim_swa_storage` via direct byte-copy (no re-quant).
  `DenseIndex.load(prefer_tq4=True)` now default. Adaptive budget + 16K ceiling
  integrated across eval scripts (`scripts/r53_21_import_inject.py`,
  `scripts/r53_eval_complex.py:_adaptive_budget`).
- **R53.28 eval bump** (`e86d787`, `e7b4538`): all R53 scripts from
  200-400 → 4096-16384 tokens.
- **R53.29 v2 kernel** (`cbb8073`): `_tq4_matvec_kernel_v2` uses `tl.gather`
  from a program-local `(16,)` centroid LUT tile. **-5 to -10% aggregate** vs
  v1 across 3 bench runs. Correctness: measured max abs diff 1.91e-6 / 1.43e-6
  per shape (4e-6 is a conservative upper ceiling).
- **R53.30 fp16 x_rot** (`cfa584f`): NULL. +0.2/+8.7% across two runs.
  Upcast-inside-dot eats BW savings on Ada.
- **R53.31 uint32 qs** (`cfa584f`): NULL. +9.8/+16.4%. `tl.join`/reshape
  overhead exceeds BW savings. Triton auto-coalesces already.
- **R53.32 BLOCK_M sweep** (`f251a7f`): current `_pick_block_m` heuristic holds.
- **R53.33** (paused at 5/6): fp16-KV + full stack got 5/5 linked_list, 12/12
  date_validation, 6/6 log_level, 0/0 csv (35 min gen, Gemma KeyError in own
  code), 0/0 token_bucket (39 min gen + 39 min repair, shadow bug persisted
  despite targeted rename hint with example). Running total 23/23. Daemon
  killed before lru_cache_class ran. **Projected final: 32/32** (matches
  R53.25 baseline).
- **USE_TQ4_KV revert** (`571c3ad`): disabled when diagnosing O(N²) dequant
  (linked_list 806s at tq4 KV vs 94s at fp16 KV — 8× slowdown per step).

### Teammate work (separate terminal, 2 of 3 tests completed)

Parallel terminal ran an independent test track. 2 of 3 tests finished
before session end — the third is not captured in minutes.md for this
session and its landed state should be confirmed by reading current tree
(`git log` on `calm/llm_computer/tq4_flash_attn.py` + `tq4_qjl_torch.py`
+ `scripts/r53_21_import_inject.py` for the `USE_TQ4_KV` flag state).

- **R53.34 fused flash-attn kernel** (`calm/llm_computer/tq4_flash_attn.py`):
  proper parallel V-kernel (`grid=(n_heads_q,)`), head-major storage contract,
  `tl.static_range` over BPR, Pi rotation at boundaries. K-side via existing
  `tq4_matvec_triton`; V-side new `_tq4_weighted_v_kernel`. Parity cosine=1.0
  vs fp32 at all tested N ∈ {16, 64, 128, 256, 1024}. Real-Gemma ablation
  Δmean=0, argmax=+0.
  - **PERF REGRESSION at short context**: 8-10% SLOWER than Phase 1 memoized
    dequant path at N ≤ 1024 (N=64: 5.60 vs 6.06 tok/s; N=1024: 5.11 vs 5.60;
    fp16 baseline 7.0-7.5). Root cause: 336 per-Q-head kernel launches per
    decode step dominate streaming-byte-load advantage at short ctx.
  - **Asymptotic crossover (N≫2K) NOT measured** (20+ min/run budget).
  - Shipped `_use_fused_flash_attn=False` default. Phase 1 memo path is the
    shipped winner (~77% of fp16 tok/s at ~50% KV memory).
  - `USE_TQ4_KV=True` re-enabled in `scripts/r53_21_import_inject.py` via
    Phase 1 memo path.
- **Q_prod (3-bit Q_mse + 1-bit QJL) null**: implemented Algorithm 2 from
  TurboQuant paper (`tq4_qjl_torch.py`, 132-byte block). Unbiased inner-product
  estimator works as proven (n=1000 mean = 1.15σ from truth). But empirical
  attention-output cosine WORSE than tq4 Q_mse alone at every tested N
  (Δ=-0.04 at N=16, Δ=-0.17 at N=1024). Softmax amplifies QJL variance more
  than it amplifies MSE-only structural bias. Kept in tree as research artifact
  for NN lookup / hash retrieval use cases.

### Doc sweep (R53 → rules alignment)

Ran parallel 2-Explore-agent audit against `.claude/rules/` and CLAUDE.md,
classified findings P0/P1/P2, shipped in 4 commits via plan mode:

- **P0** (`91e1e04`): CLAUDE.md substrate-RAG flip (advocacy → regression
  receipt + R53 Phase 2 status paragraph); `augmentation_thesis.md` new
  §"R53.14/20a/20b — substrate L41 install REGRESSES on code"; `calm.md`
  new §"Sandbox stdlib pre-import"; `workflow.md` new §§"MAX_TOKENS budget
  discipline" + "GPU bench discipline".
- **P1** (`c57b538`): `turboquant.md` v1/v2/v3/v4 variant table + new
  §"Fused flash-attention decode" with honest short-ctx perf story;
  `Substrate.md` + `architecture.md` KVCacheTq4 multi-token + tq4_flash_attn.py
  rows + 4.4× → ~3.6× memory reconcile; `tracing_roadmap.md` session header
  bumped to R53.34 + 8 new R53.x ruled-out rows.
- **P2** (`bc81dda`): `capability_gain.md` new §"Gemma ignores targeted hints
  (R53.19/R53.33 receipt)"; `retrieval.md` + `embed_intelligence.md` ruled-out
  sections for substrate-RAG-on-code + FirstTokenHook-on-code;
  `training.md` new §"Substrate eval defaults (R53.28 + R53.34)".
- **Trim** (`68ff533`): `augmentation_thesis.md` stale "Implication for R53.6"
  section (10 lines) — prediction was falsified by R53.14/20a/20b.

### Command update

- `/update` workflow (`99ccc59` + `16817ad`): default is now parallel-audit
  with 2 Explore agents → P0/P1/P2 classification → plan mode → one commit
  per tier → fail-closed verification. Brief size guidance corrected to
  300-500 words (the audit agents need complete context encoded in the brief
  because they're cold-started).

### Microbench infrastructure

- `scripts/bench_tq4_matvec.py` (`fdb7b60`): heavy_warmup(3s) + CUDA events
  + median of 5 × 2000 iters + same-process A/B. Before this, kernel bench
  variance was 20-30% run-to-run. After, v2 stabilized to -5 to -10%
  aggregate across 3 process runs.
- `scripts/test_tq4_matvec_v2_correctness.py`: max abs diff per shape vs v1.
- `scripts/test_kvcache_tq4_multitoken.py`: 5 tests for S≥1 / per-layer /
  prefill-then-decode / SWA-trim.
- `scripts/sweep_tq4_block_m.py` (`f251a7f`): 9-value sweep kept for future
  shape additions.

## In Progress

**Paused / not blocking**:

- **R53.33 at 5/6**: lru_cache_class un-run when daemon killed. Running total
  23/23. Projected final 32/32 (matches R53.25). 1 hour to close the loop.
  No new info expected — R53.25 already established the pattern.
- **AST walker tier-2 card**: identified as the clear next lever from R53.33
  partial receipts. Not yet started. ~2-3 days of pure Python work.

**Uncommitted state** (46 files in `git status --short`):

- Modified by teammate / linter during session: `calm/llm_computer/gemma_substrate.py`,
  `calm/llm_computer/tq4_flash_attn.py`, `scripts/r53_21_import_inject.py`,
  `scripts/r53_22_diagnose_csv.py`, `scripts/r52_train_student_kl.py`,
  `bin/gemma_daemon.py`. Changes appear intentional (system-reminders noted
  them); do NOT revert without reviewing.
- Deleted MEMORY files: `.claude/MEMORY/CRLM_SPEC.md`, `.claude/MEMORY/MEMORY.md`,
  `.claude/MEMORY/RESEARCH_ROADMAP.md`. Deletions pre-date this session —
  likely replaced by `atlas.md` + `substrate_registry.md`. Safe to commit deletion.
- Deleted RESEARCH files: `RESEARCH/00_INDEX.md`, `01_LLM_Computer_Overview.md`,
  `02_Fast_Attention_2D_Heads.md`, `03_Compiling_Programs_to_Weights.md`.
  Replaced by the `RESEARCH/LLM-COMPUTER/`, `RESEARCH/NEURAL_COMPUTER/`,
  `RESEARCH/TRAINING/` untracked directories.
- Untracked: `.cache/` (retrieval indices, gitignored), `.claude/MEMORY/can_be_done.md`,
  `.claude/scheduled_tasks.lock`, `.codex/`, `.port_sessions/`,
  `calm/.module_learning.json`, multiple `calm/hrm/checkpoints/*.pt`,
  `calm/llm_computer/checkpoints/substrate_hrmlm_v2*.pt`, `calm/llm_computer/r51/`,
  `calm/llm_computer/synth/`, `calm/llm_computer/tq4_autograd.py`,
  `scripts/r53_20b_stacked.py`, `scripts/r53_22_diagnose_csv.py`,
  `scripts/r53_substrate_rag_multitoken.py`, `scripts/test_kvcache_tq4_multitoken.py`,
  `scripts/test_tq4_matvec_v2_correctness.py` (these last five ARE from this
  session but already git-added inside the respective commits — verify via
  `git log --follow <path>`).

## Next Steps (ranked by commercial lift)

### 1. AST walker tier-2 card for code logic bugs (~2-3 days)

Biggest-lift target. R53.33 pinpointed two deterministic failure modes:

- **token_bucket shadow bug** (`self.consume = capacity` shadows method `consume`)
- **csv_column_stats KeyError** (dict access on key never constructed)

Both tier-2-addressable by a compiled post-generation walker. Pure Python +
existing `calm/sandbox.py` + `ast` stdlib. Pipeline:

1. Parse Gemma output with `ast.parse`
2. Detectors: method/attr shadow, undefined-ref, dict-key-never-set,
   off-by-one in `range(0, n-1)`, unused-var
3. Auto-rewrite OR return structured repair patch
4. Integrate as post-generation pass in `scripts/r53_21_import_inject.py`
   alongside existing import injection + R53.19 structured repair

R53.0 projected lift: 32/32 → **~43-45/46** (maximum achievable).

Entry point: add `scripts/r53_ast_walker.py` + wire into
`scripts/r53_21_import_inject.py`'s `inject_imports_if_possible` pipeline
(same contract — `code, test_code, run_fn, score_fn, problem → (code, pass, total, fixes_applied)`).

Commercial framing in `.claude/rules/capability_gain.md` §"Gemma ignores
targeted hints" — auditable rewrite, not probabilistic retry. Regulated
industries need this.

### 2. Broader-corpus validation (~1 day)

R53.0 is 6 problems — too small to claim generalization. Run R53.25's
winning stack (v2 kernel + sandbox fix + AdaptiveBudget + import injection)
+ AST walker (if shipped) against **MBPP or HumanEvalPlus** filtered via
the R53 failure-surface gate in `.claude/rules/capability_gain.md`. Target:
50-200 problems.

Start: `scripts/r53_run_data_generators.py` already fetches these corpora;
`calm/llm_computer/facades/code_example_db.py:DEFAULT_CORPORA` has them
loaded. Need a new eval driver that picks the failure-surface subset.

### 3. Fused kernel long-context perf validation (~1 day, orthogonal)

Run R53.34 fused kernel at N=4K, 8K, 16K to find (or rule out) the
asymptotic crossover. If fused beats Phase 1 memo somewhere, enable
`_use_fused_flash_attn=True` for long-ctx workloads only. Otherwise
retire.

Script: adapt `scripts/bench_tq4_matvec.py` pattern to drive `generate()`
with N ∈ {1024, 4096, 8192, 16384} and log `prefill_s` + `decode_s` from
the output dict.

### 4. Close R53.33 loop (~1 hour)

`bin/gemma-run --start` → dispatch `scripts/r53_21_import_inject.py` → wait
for lru_cache_class to complete → expect 32/32 total. Updates evals folder
if not matching baseline.

### 5. Speculative (only if bandwidth)

- **tq2 weight quantization**: port `quantize_tq4` to K=4 Lloyd-Max codebook.
  Expected ~10-15% PPL regression, enables Gemma 4 E4B at 1M+ context on 8 GB.
  Substrate absorbs quality drop via CALM verification. See CLAUDE.md
  discussion of tq1/tq2/tq3 tradeoffs (this session — not yet in rules).
- **Substrate-RAG on non-code domains**: L41-install regression is code-
  specific. Math / factual-recall may still benefit. Pick one non-code
  corpus, run R53.14-style A/B.

## Key Context

### Patterns that worked

- **Hypothesis-test-iterate, commit after every round**. Discipline caught
  two reversals (tq4 matvec v3, v4) before they rotted into the default.
- **Parallel 2-Explore-agent audits** with ~400-word briefs for doc updates.
  Agents do research + return structured punch lists; I synthesize into
  P0/P1/P2 plan. Saves 30-60 min of main-context bench/doc noise per session.
- **heavy_warmup + CUDA events + median of 5** for kernel benches. Before
  this, variance was 20-30% — false perf signals every round.
- **MAX_TOKENS first-order check** before diagnosing substrate / sandbox /
  import issues. R53.19-R53.24 burned four null rounds before R53.25 showed
  budget was the whole problem.

### Patterns that failed (don't retry)

- **Substrate-RAG at L41 with FirstTokenHook on code tasks**: -9.3pp
  regression pre- AND post-SWA-fix. Install-mechanism disrupts HIT prompts.
  Don't use first-token bias on code.
- **fp16 x_rot activation in Triton tq4 kernel**: +0.2/+8.7% null. Upcast
  cost matches BW savings on Ada.
- **uint32 vectorized qs loads in Triton**: +9.8/+16.4% slower. Triton
  auto-coalesces; `tl.join`/reshape overhead exceeds BW savings.
- **BLOCK_M sweep**: current `_pick_block_m` heuristic holds.
- **TurboQuant Q_prod (3-bit Q_mse + 1-bit QJL) for KV cache**: unbiased
  inner-product estimator IS correct, but softmax amplifies QJL variance.
  Kept as research artifact for NN/retrieval.
- **KVCacheTq4 with dequant-on-read at decode**: O(N²) dequant cost (linked_list
  5/5 took 806s vs 94s at fp16 KV). Reverted until Phase 1 memoized path
  landed (R53.28) + fused flash-attn (R53.34). Phase 1 memo is the shipped win.
- **Gemma ignoring targeted rename hints**: R53.19/R53.33 data shows Gemma
  retry emits the same shadow bug even with concrete example in hint. Prior
  dominance overwhelms in-context instruction. Tier-2 AST walker is the fix,
  not hint-tuning or bigger context.

### Environment state (session end)

- **Daemon**: NOT running (killed during session pause). Restart with
  `bin/gemma-run --start` (~3 min cold boot).
- **GPU**: clear. `nvidia-smi` should show no resident Python processes.
- **Working tree**: 46 modified/untracked entries. Session's commits are
  clean; the untracked files are either gitignored caches, session artifacts
  by other agents, or orphaned pre-session directories (RESEARCH/*.md
  deletions are replaced by RESEARCH/*/ subdirs).
- **Branch**: `feature/multi-agent-qwen`, 371+ commits ahead of
  `origin/feature/multi-agent-qwen`. Not pushed.

### Hardware / serving

Per CLAUDE.md — RTX 4070 Laptop 8 GB, Gemma 4 E4B tq4 via
`~/models/gemma-4-E4B-it-tq4-aligned.gguf` (5.0 GB). Daemon supports
max_len=32K. At 16K ctx: ~5.5 GB total VRAM.

## Files in Project (session-shipped)

### New files
- `calm/llm_computer/tq4_flash_attn.py` — R53.34 fused flash-attn decode kernel
- `scripts/bench_tq4_matvec.py` — stable Triton kernel A/B bench harness
- `scripts/sweep_tq4_block_m.py` — per-shape BLOCK_M sweep harness
- `scripts/test_tq4_matvec_v2_correctness.py` — max abs diff per shape vs v1
- `scripts/test_kvcache_tq4_multitoken.py` — 5 tests for multi-token KVCacheTq4
- `scripts/r53_20b_stacked.py` — substrate + prompt-RAG stacking eval (null)
- `scripts/r53_22_diagnose_csv.py` — sandbox blocking diagnostic (led to R53.23 fix)
- `scripts/r53_substrate_rag_multitoken.py` — multi-token step-through harness
- `.claude/MEMORY/evals/2026-04-19_r53_substrate_rag_null.md` — R53.20a/b writeup

### Modified — code
- `calm/sandbox.py` — pre-import ~23 stdlib modules before `_safe_import`
- `calm/llm_computer/gemma_substrate.py` — multi-token KVCacheTq4 + generate(use_tq4_kv=True) dispatch
- `calm/llm_computer/tq4_triton.py` — v1/v2/v3/v4 kernel variants, v2 default
- `calm/llm_computer/facades/retrieval.py` — DenseIndex.load(prefer_tq4=True) default
- `scripts/r53_21_import_inject.py` — USE_TQ4_KV toggle + AdaptiveBudget + generalized TypeError regex
- `scripts/r53_eval_complex.py` — `_adaptive_budget()` helper + shared 16K ceiling
- `scripts/r53_20b_stacked.py`, `r53_calm_substrate_full.py`, `r53_calm_substrate_retry.py`,
  `r53_substrate_rag_eval.py`, `r53_substrate_rag_confidence.py`,
  `r53_eval_complex_channel.py`, `r53_eval_phase1.py` — MAX_TOKENS → 4096-16384
- `bin/gemma_daemon.py` — max_len 1024 → 32768 (from prior session, still current)

### Modified — rules (P0/P1/P2 doc sweep)
- `.claude/CLAUDE.md` — R53 status flip, Phase 2 status, v2 kernel as default
- `.claude/rules/augmentation_thesis.md` — R53.14/20a/20b regression section
- `.claude/rules/calm.md` — sandbox stdlib pre-import section
- `.claude/rules/workflow.md` — MAX_TOKENS + GPU bench discipline sections
- `.claude/rules/turboquant.md` — kernel variant table + fused flash-attn spec
- `.claude/rules/Substrate.md` — KVCacheTq4 multi-token + tq4_flash_attn.py row
- `.claude/rules/architecture.md` — same updates, mirror
- `.claude/rules/tracing_roadmap.md` — 8 new R53.x ruled-out rows + header bump
- `.claude/rules/capability_gain.md` — Gemma-ignores-hints section
- `.claude/rules/retrieval.md` — substrate-RAG-on-code ruled out
- `.claude/rules/embed_intelligence.md` — FirstTokenHook-on-code ruled out
- `.claude/rules/training.md` — substrate eval defaults section
- `.claude/commands/update.md` — parallel-audit + P0/P1/P2 plan default

### Deletions (safe — pre-session orphans)
- `.claude/MEMORY/CRLM_SPEC.md`, `.claude/MEMORY/MEMORY.md`,
  `.claude/MEMORY/RESEARCH_ROADMAP.md`
- `RESEARCH/00_INDEX.md` through `03_Compiling_Programs_to_Weights.md`
  (replaced by `RESEARCH/LLM-COMPUTER/` + `NEURAL_COMPUTER/` + `TRAINING/` subdirs)
