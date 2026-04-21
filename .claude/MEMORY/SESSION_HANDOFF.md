# Session Handoff — 2026-04-21 (PT+Delta R13-R21 arc + /update docs pass)

## Goal

User opened with "so how does delta fit in now?" after reading the prior
SESSION_HANDOFF.md (PT+Delta R5-R11 arc). Session evolved into a 9-round
R13-R21 arc that solved MQAR N=5-20 via data scaling, shipped chunkwise
parallel DeltaNet + cached decode, consolidated PT+Delta as the default
trained-card architecture, and produced a deployable MQAR card
(`copy_augmented_delta_mqar_best.pt`). User then ran `/update` to absorb
both this session's Track B + an earlier session's Track A (decode
speedup) into `.claude/` docs.

Ended at a clean stopping point with R21 artifact shipped + /update 4
commits landed. Next round is R22 Gemma CardSlot install.

## Completed (18 commits, `7110990` → `65ced1f`)

### PT+Delta DeltaNet arc (14 commits, Track B)

| SHA | Round | Result |
|---|---|---|
| `7110990` | R13 | MQAR N=10 cracked at 2K/N → 100% (plain PT caps 34%). "+5 on N needs 2× data" rule discovered. Ruled out R10/R11a/R11b architectural nulls as undertrained. |
| `e61c2bc` | infra | `--task {mqar,reassign,scratchpad}` flag in `experiment_r10_mqar.py` |
| `c3c1569` | R14 | N=20 at 5K/N plateau 58% (plain PT 31%, +27pp). Data-bound signal: more data would lift, deferred R14-b |
| `bc52c87` | fix | Reassign generator positional-shortcut bug — `_gen_reassign` forced step n-1 to target_var, placing answer at -3 from query. Plain PT solved in 3 epochs, caught as diagnostic signal |
| `5abc2f5` | infra | `gen_reassign_hard_batch` + `_VARS_HARD = "abcdefghijklmnopqrst"` 20-char vocab |
| `2a62675` | R15/R15-b | Reassign moat scales with vocab size: 0pp at 5 vars (plain PT 100%), +12pp at 20 vars (PT+Δ 98% vs PT 86%), 10× convergence speedup. Refined commercial claim to "sparse-key retrieval" |
| `187203d` | R16 | Scratchpad null — both PT and PT+Δ at 0% full-expression match. State-carry ≠ arithmetic; 185K capacity can't internalize 243 single-digit facts. Brain+Cards thesis reinforced |
| `56fa281` | R17 | **Chunkwise parallel DeltaNet** (UT transform, paper §3-4). Bit-equivalent to per-position (max\|Δ\|=1.9e-6). Forward-only: 4.65-7.52× at L=32-256. e2e training: 322s→52s at 2K/N × N=[5,10] (~6×) |
| `49c13d7` | R14-b | N=20 at 10K/N solves 99% in 6 epochs (~7 min chunkwise). Closes MQAR scaling curve through N=20. Gap vs plain PT widens to +84pp |
| `78b5dfb` | R18 | Multi-head H=4 **null** — PT+Δ 21% vs plain PT 27% (-6pp). Per-head state (16,16) = 256 scalars, D/log(D) capacity ~6 << N=15. Aggregate drop 4096→1024. Reopen only at d_model ≥128 |
| `65fb148` | R19 | D5 refinement n_iters=2 **null** — 21% best/16% final. ARC Prize's "+13pp from refinement" is grid-reasoning-specific, MQAR has nothing to iteratively refine |
| `63a49fc` | R20 | **PT+Delta consolidated as default** — 200-ex held-out parity on NL math (99.5% both, +0.0pp). Functional superset + 3-10× epoch efficiency. Plain PT stays as ablation baseline |
| `e6f2d5c` | R20b | **Cached autoreg decode** — `decode_greedy_cached` on `CopyAugmentedDeltaNet`. 50/50 token-exact parity with uncached. Closes 5× inference gap to **1.18×** plain PT |
| `7bc13f1` | R21 | **Deployable MQAR card** — `copy_augmented_delta_mqar_best.pt` (748 KB, 183,877 params). 100% on N=5/10/15 held-out (fresh seed=777777+N), 0% on N=20 (OOD — not trained). `scripts/train_pt_delta_mqar.py` saves checkpoint |

### /update docs pass (4 commits)

| SHA | Tier | Files |
|---|---|---|
| `394716a` | P0 | Falsified "42 tok/s / 90% llama.cpp" in CLAUDE.md + architecture.md + turboquant.md (clean bench 25-33 tok/s). Added MQAR data-scaling rule to capability_gain.md as measurement-discipline receipt (4 null rounds → 1 flag change) |
| `63e9f86` | P1 | New `.claude/rules/delta_rule.md` (224 LOC). Track A sections in architecture.md + turboquant.md. Walker 7th-rewrite note in calm_part_2.md + augmentation_thesis.md. Eval_defaults extension in training_part_2.md. Cross-refs in Substrate.md + training.md |
| `90db4dd` | P2 | Tracing_roadmap.md gets R-delta ruled-out log (6 Track B nulls + 2 Track A nulls). R-number disambiguation header (Track B R5-R21 ≠ tracing-arc R13-R21) |
| `65ced1f` | P1-fixup | Phase-5 grep caught missing PT+Delta cross-refs in CLAUDE.md + architecture.md + augmentation_thesis.md. Added minimal pointers to delta_rule.md |

### Verbatim benchmark receipts

**MQAR data-scaling curve** (commits `7110990`, `49c13d7`):
```
N    per-N to saturate PT+Δ    PT+Δ vs plain PT gap
5     2K                         +21pp
10    2K                         +66pp
15    5K                         +75pp
20   10K                         +84pp
```

**R17 chunkwise speedup** (forward-only, B=16, fp32, RTX 4070):
```
L    per-pos   chunkwise(C=32)  speedup
32   46.2ms     9.9ms            4.65×
64   81.3ms    11.8ms            6.90×
128  147.9ms   23.8ms            6.22×
256  262.6ms   34.9ms            7.52×
```
End-to-end training: R13-med-2k 100% at ep10 t=52s chunkwise vs per-position ep10 t=322s → **~6× e2e**. Bit-equiv max\|Δ\|=1.907e-06.

**R20b inference gap** (200 NL math, max_gen=30):
```
Path                       acc    wall    vs plain PT
PT+Δ per-position          99.5%  45.6s   5.92×
PT+Δ chunkwise, uncached   99.5%  15.1s   1.96×
PT+Δ chunkwise, cached     99.5%   9.1s   1.18×
Plain PT baseline          99.5%   7.7s   1.00×
```
50/50 token-exact parity cached vs uncached.

**R21 deployable card held-out** (fresh seed=777777+N, 100 problems each):
```
N=5   100/100   2.0s (cached decode)
N=10  100/100   1.7s
N=15  100/100   1.9s
N=20    0/100   4.7s  (OOD — not in training)
```

## In Progress

**None.** Arc closed cleanly at R21 (deployable card) + /update docs pass. No orphan training runs, no uncommitted session work at risk.

## ⚠ Uncommitted

### Session-critical (HIGH — needs attention before next session)

- `.claude/MEMORY/SESSION_HANDOFF.md` [HIGH] — **THIS FILE** after I write it. Commit it before ending session.

### Pre-existing user/teammate work (MEDIUM — user owns, do not touch)

- `.claude/rules/calm.md` (D), `.claude/rules/workflow.md` (D) — user's ongoing doc-reorg from session start
- `.claude/rules/calm_part_1.md`, `.claude/rules/workflow_part_1.md`, `.claude/rules/workflow_part_2.md` [untracked] — user's doc-reorg drafts. User said "i've just fixed calm 1" mid-session (shortened 494→264 lines). Not this session's work.
- `.claude/MEMORY/SESSION_HANDOFF_1.md` — earlier session's handoff (decode-speedup track)
- `.claude/MEMORY/notesd.md` — teammate notes
- `scripts/r52_train_student_kl.py` (M) — pre-existing R52 refactor, teammate track
- `calm/hrm/checkpoints/meta_best.pt` (M) — binary re-save by teammate, same size
- `calm/llm_computer/tq4_autograd.py` [untracked] — R52.1 teammate work
- `calm/llm_computer/checkpoints/substrate_hrmlm_v2*`, `r51/checkpoints/` — teammate R51/R52 checkpoints
- `calm/llm_computer/synth/*.jsonl` — teammate synth corpus

### Research artifacts (MEDIUM — teammate-owned)

- `RESEARCH/LLM-COMPUTER/`, `RESEARCH/NEURAL_COMPUTER/`, `RESEARCH/TQ/`, `RESEARCH/TRAINING/` — teammate research dumps. LLM-COMPUTER was touched by `8df1f5f` (DeltaNet paper split from prior-to-this-session) but remaining untracked content is teammate work.

### Low-risk (safe to ignore)

- `.cache/`, `.codex/`, `.port_sessions/`, `.claude/scheduled_tasks.lock`, `calm/.module_learning.json` — runtime caches, gitignored-adjacent
- `calm/hrm/checkpoints/copy_code_*.pt, math_*.pt, unified_substrate_best.pt` — teammate checkpoint zoo
- `.claude/MEMORY/minutes/` — session transcripts (auto-rotated)

**Risk for this session's work: NONE.** All 18 commits landed cleanly.

## Next Steps

1. **R22 — Gemma CardSlot install** (the product step, ~half-day next session)
   - Card artifact ready: `calm/hrm/checkpoints/copy_augmented_delta_mqar_best.pt` (solves MQAR N=5-15 to 100% held-out, cached decode 1.18× plain PT)
   - Install via `CardSlot.attach(gemma, preserve=True)` at L30 (session 32 pattern, mechanics proven)
   - **Novel design piece: NL-context → 82-char-vocab adapter.** Three candidate adapters:
     - AST walker over Gemma's emitted Python code → extract `name = value` assignments → hash to PT+Delta's vocab
     - NER over NL → "Alice is 30" patterns → entity_hash + value_hash
     - Structured prompt format `<mem>k1=v1 k2=v2...</mem>` — the cheapest to prototype
   - `VerificationHook` with `vocab_mapping` + `min_margin=0.5` to close loop (prevents firing on unmatched prompts — R44 `HubInjectionCard` pattern)
   - Measure on multi-needle NIAH prompts (`.claude/MEMORY/evals/2026-04-07_*_needle_256k_*.md` baseline: Gemma 4/5 at 220K multi-needle, Qwen 3/5 + hallucination)
   - Regression guard on pure-Gemma prompts (hash-match gate should miss → Gemma output unchanged)
   - Decision rule: if card install preserves ≥80% on prompts Gemma hits <80% on, first real deployment win

2. **Re-bench "42 tok/s / 90% llama" claim in idle environment** (Track A's unfinished — user's open question post-/update)
   - Current clean bench: A 7.14 / B 5.56 / C 33.35 / D 25.02 tok/s
   - Architecture.md doc now says "unreproducible in current bench"; if idle rebench hits 42, restore the claim with date gate. If still 25-33, that's the new canonical number.
   - Pre-req: `pgrep -c rustc == 0 && pgrep -c cargo == 0 && ! pgrep codex_tui` (session log §"Bench-session variance" flagged rustc contamination)

3. **Track A unfinished work** (from SESSION_HANDOFF_1.md, still parked)
   - Clean e2e bench of Round 5 k+v fusion (microbench 1.65×, projected +4.4% e2e, never verified due to rustc contention)
   - R13 MBPP walker at ITERATION_N=5 baseline (rotation rotation via `bin/mbpp-rotate 0`, r53_39_mbpp_walker.py ready)
   - Round 6-9 kernel queue (q+k+v triple fusion, per-shape autotune, flash-attn TILE_N, fused mega-kernel)

4. **R-delta-16 scratchpad retry** (low priority, research-interest)
   - Pair PT+Delta state-carry with compiled arithmetic card (safe_eval) → should solve scratchpad via Brain+Cards composition
   - R20's `MultiStepReasoningFacade` already does this for Gemma; same pattern could drive standalone card

### Blockers / open policy questions

- None that block R22.
- User's doc-reorg (calm.md/workflow.md split) is pending commit but not blocking — user owns the cadence.

## Key Context

### Discoveries (save hours if known upfront)

1. **"Chunkwise must be set AFTER checkpoint load"** — existing `copy_augmented_delta_best.pt` config predates the `use_chunkwise` flag so defaults to per-position. `model.config.use_chunkwise = True` after load gives 3× inference speedup. Forgetting this was the source of R20's 5× misattribution (diagnosed in R20b). The new `copy_augmented_delta_mqar_best.pt` has `use_chunkwise=True` baked into its saved config.

2. **Loss-accuracy decoupling is the signal for capacity ceiling** — both R16 and R18 showed training loss dropping 3+ orders of magnitude while val accuracy stays flat/drops. Classic over-capacity memorization. Distinct from undertraining (both loss and acc climb).

3. **"Fast plain-PT solve + decisive loss drop = likely generator bug"** (R15 diagnostic rule). Reassign at 5-var vocab solved in 3 epochs because `_gen_reassign` forced step n-1 to target_var — positional shortcut, not mutation tracking. Caught by comparing against expected capability difficulty. Lesson codified in `workflow_part_1.md`.

4. **R-number collision** — this session's PT+Delta arc used R5-R21 which collides with the tracing-arc R13-R21 in `tracing_roadmap.md` / `atlas.md` (logit lens / activation patching / per-head / SAE / V probe). Resolution: PT+Delta numbering now uses "R-delta-NN" prefix in cross-file references; the arc stays scoped inside `.claude/rules/delta_rule.md`.

### Failed approaches (cite SHAs, don't retry)

- `dba270e` R-delta-5 pure DeltaNet at substrate scale — 19.7% @ n=5 (paper's d_head=128 regime doesn't transfer to d_head=2)
- `3b9087f` R-delta-8 sub-head partition — 44% plateau (convex-combo dilutes over heads)
- `1e9925e` R-delta-9 soft-gate dispatch — 46% plateau (convex-combo over mechanisms)
- `97fba23` R-delta-6b plain PT chain test — task too easy to distinguish mechanisms
- `6617a48` R-delta-11a/b capacity scaling (d_model 64→128, d_head 2→16) — both null at undertrained budget; data-scaling cracked it later
- `187203d` R-delta-16 scratchpad single card — state-carry ≠ arithmetic at 185K params
- `78b5dfb` R-delta-18 multi-head H=4 at d_model=64 — per-head state too small (-6pp vs plain PT)
- `65fb148` R-delta-19 D5 n_iters=2 on MQAR — ARC Prize finding is task-specific, doesn't transfer

### Methodology caveats

- **R14-b/R16/R18/R19 receipts bundled into R13's eval receipt** (`.claude/MEMORY/evals/2026-04-21_r13_pt_delta_mqar_data_scaling.md` appended rather than separate files). 9 receipt files in evals/, not 10 as my /update plan specified. Not a bug — deliberate scoping. R14 receipt exists within R13's arc receipt.
- **ARC Prize HRM analysis read mid-session** — confirmed refinement-loop finding is grid-specific; architecture contributes only ~5pp; memorization-not-generalization at small scale. Validates our substrate-composition bet (Brain+Cards > architecture lottery).
- **No GPU contention during this session** — R21 training + inference benches are reliable. SESSION_HANDOFF_1 flagged rustc contamination affecting Track A's end-of-session bench; that's parked, not active.

### Hardware / environment state at session end

- Branch: `feature/multi-agent-qwen` at `65ced1f` (P1-fixup)
- GPU: 1161 MiB used / 6788 MiB free / 18% util — clean idle (no resident card)
- Gemma daemon: NOT RUNNING (no `gemma_daemon` process)
- No orphan training processes
- `.cache/r53_code_db/`: intact, 180 MB (pre-session, untouched)
- Rotation state (`/tmp/substrate_eval_rotation.json`): absent (window unset)

## Files in Project (session-shipped)

### New files — code

- `calm/llm_computer/delta_rule.py` — `DeltaNetConfig`, `DeltaNetSmall2DTransformer`, `_delta_step`, `_delta_chunkwise`, `_delta_chunkwise_multihead`, `_delta_layer_stack`. Landed with R-delta-5 (`dba270e`), extended across R17/R18/R19.
- `calm/llm_computer/copy_augmented_delta.py` — `CopyAugmentedDeltaNet`, `CopyAugmentedDeltaConfig`, **`decode_greedy_cached`** (R20b, `e6f2d5c`), `_predict_next_token`, `build_copy_augmented_delta`.
- `scripts/train_pt_delta_mqar.py` — R21 checkpoint trainer (`7bc13f1`). Defaults: 5K/N × N=[5,10,15] × 20ep, chunkwise, scheduled sampling, batch=64.

### New files — checkpoints (tracked in git)

- `calm/hrm/checkpoints/copy_augmented_delta_best.pt` — R-delta-6a NL math card (pre-session, `31337f3`).
- **`calm/hrm/checkpoints/copy_augmented_delta_mqar_best.pt`** (748 KB, 183,877 params) — R21 deployable MQAR card (`7bc13f1`). 100% held-out on N=5/10/15. **This is the artifact R22 installs on Gemma.**

### New files — docs

- `.claude/rules/delta_rule.md` (224 LOC) — the PT+Delta arc canonical rule file. Architecture, chunkwise, cached decode, scaling curve, task-shape moat, R20 consolidation, R21 artifact, nulls, R22 install plan.
- `.claude/MEMORY/evals/2026-04-21_*.md` — 9 receipt files for R13/R15/R16/R17/R18/R19/R20/R20b/R21. R14 + R14-b bundled into R13.

### Modified code

- `scripts/experiment_r10_mqar.py` — added `--task`, `--chunkwise`, `--chunk-size`, `--n-delta-heads`, `--n-iterations` flags.
- `calm/hrm/memory_tasks.py` — fixed `_gen_reassign` positional shortcut (`bc52c87`) + added `_VARS_HARD` + `gen_reassign_hard_batch` (`5abc2f5`).

### Modified docs (via /update)

- `.claude/CLAUDE.md` — 42 tok/s falsified + PT+Delta Pointer Transducer section
- `.claude/rules/architecture.md` — 42 tok/s falsified + graph-captured tq4 decode section + PT+Delta bullet
- `.claude/rules/turboquant.md` — 42 tok/s falsified + bench table + Track A kernel sections (KVCacheTq4Static + k+v fusion)
- `.claude/rules/Substrate.md` — trained-cards default architecture section
- `.claude/rules/training.md` — PT+Delta production checkpoint list + recipe differences
- `.claude/rules/augmentation_thesis.md` — walker 7-rewrite update + R-delta-21 card row
- `.claude/rules/capability_gain.md` — MQAR data-scaling measurement-discipline receipt
- `.claude/rules/tracing_roadmap.md` — R-delta + Track A ruled-out rows + disambiguation header
- `.claude/rules/calm_part_2.md` — 7th walker rewrite (fuzzy_rename_function) note
- `.claude/rules/training_part_2.md` — eval_defaults extended with ITERATION_N/FINAL_N/resolve_problem_window

### Planning artifacts

- `/home/gabe/.claude/plans/first-lets-run-update-vectorized-ritchie.md` — the /update pre-execution plan (written in plan mode, approved)

## Handoff verification

- Main context claims vs git state: **match.** All 18 commits confirmed via `git log --oneline`.
- Main context claims vs transcript: **match** for all verbatim numbers (agent 1 extracted same tables).
- Uncommitted state: flagged with risk tiers; this HANDOFF is the only session-critical pending file.
- R14/R14-b receipt absent as separate files — **not a bug**, bundled into R13's receipt with appended R14-b section (verified in `2026-04-21_r13_pt_delta_mqar_data_scaling.md`).
