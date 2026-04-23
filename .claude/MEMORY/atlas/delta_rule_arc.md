# DT / DeltaNet — Historical arc (R-delta-5 → R-delta-22 + R22 install + DT code-skeleton)

Receipts and ruled-out log for the DT architecture. Current rules,
defaults, install pattern: `.claude/rules/delta_rule.md`. This file
exists for archaeology — "how we got here," "why these defaults," and
"what was tried that didn't work."

R5→R21 arc 2026-04-21, R22 install 2026-04-21 → 2026-04-22, DT
code-skeleton arc 2026-04-22.

R-numbers in the "R-delta" scope are distinct from tracing-arc R-numbers
in `tracing_roadmap.md` / `MEMORY/atlas/tracing_arc.md`. R20 consolidation
landed in commit `63a49fc`.

## Chunkwise UT transform — perf measurement (R17)

Training speedup (forward-only, B=16, fp32, RTX 4070):

| L | per-position | chunkwise | speedup |
|---:|---:|---:|---:|
| 32 | 46.2 ms | 9.9 ms | **4.65×** |
| 64 | 81.3 ms | 11.8 ms | **6.90×** |
| 128 | 147.9 ms | 23.8 ms | **6.22×** |
| 256 | 262.6 ms | 34.9 ms | **7.52×** |

End-to-end training: R13-med-2k (N=[5,10] × 2K/N × 15ep) hit 100% in
**52s chunkwise vs 322s per-position** — ~6× wall-clock.

Bit-equivalence verified at max |Δ| = 1.9e-6 at L=64, d=64 (R17).

## Cached decode — perf (R20b)

Inference on 200 NL math (max_gen=30, RTX 4070):

| path | accuracy | wall | vs plain PT |
|---|---:|---:|---:|
| per-position, uncached | 99.5% | 45.6 s | 5.92× |
| chunkwise, uncached | 99.5% | 15.1 s | 1.96× |
| **chunkwise, cached** | **99.5%** | **9.1 s** | **1.18×** |
| plain PT baseline | 99.5% | 7.7 s | 1.0× |

Parity: 50/50 token-exact match vs uncached on held-out NL math.
Commit `e6f2d5c`.

## MQAR data-scaling curve (R13 → R14-b)

Empirical rule for this architecture at d_model=64: **"+5 on N
needs 2× data."** Canonical receipt cross-ref: `capability_gain.md`.

| N | per-N training to saturate | best epoch | commit |
|---:|---:|---:|---|
| 5, 10 | 2000 | 10 | `7110990` R13 |
| 15 | 5000 | 14 | (R13-d within `7110990` arc) |
| 20 | 10000 | 6 | `49c13d7` R14-b |

Plain-PT gap (same training budget, best-epoch):

| N | plain PT | PT+Δ | gap |
|---:|---:|---:|---:|
| 5 | 79% | 100% | +21pp |
| 10 | 34% | 100% | +66pp |
| 15 | 24% | 99% | +75pp |
| 20 | 15% | 99% | +84pp |

## Task-shape moat (R15 / R15-b)

| Task | plain PT N=10 | PT+Δ N=10 | gap |
|---|---:|---:|---:|
| MQAR (each key unique) | 34% | 100% | **+66pp** |
| Hard-reassign (20-var vocab) | 86% | 98% | +12pp |
| Small-vocab reassign (5 vars) | 100% | 100% | 0pp |

DT also converges ~3-10× faster than plain PT even where both saturate
(R15-b: PT+Δ hits 94% at ep3 vs plain PT 86% final at ep30). Compounds
with chunkwise to ~20-50× faster training per card.

## R20 consolidation receipts (commit `63a49fc`)

Held-out test (`copy_augmented_hrm_best.pt` vs
`copy_augmented_delta_best.pt` on 200 NL math, seed=99999): **both
99.5%, delta 0.0 pp.** Established DT as strict functional superset on
copy-dominant structure tasks → defaults shipped.

`copy_augmented_delta_mqar_best.pt` (R21 deployable, 2026-04-21):
trained 5K/N × N=[5,10,15] × 20 ep, chunkwise, scheduled sampling
tf 1.0→0.3, ~2 min wall time.

## Nulls — ruled-out log (R-delta scope)

Each null tightened the product claim rather than breaking it.
R15/R15-b narrowed "mutation tracking" to "sparse-key retrieval";
R16 confirmed composition-per-card thesis (compute → compiled cards,
recall → DeltaNet).

| R | Commit | Approach | Reason ruled out |
|---|---|---|---|
| R-delta-5 | `dba270e` | Pure DeltaNet at substrate scale | 19.7% @ n=5 vs plain-PT baseline 65.1%. Paper's d_head=128 regime doesn't transfer to substrate d_head=2. Need PT copy-path shield (landed as R-delta-6a). |
| R-delta-6b | `97fba23` | Plain-PT multi-step chain test as capability wedge | Plain PT saturated 100% at all L=1-5; task too easy to distinguish mechanisms. PT+Delta wall-time prohibitive at max_len=128 (pre-chunkwise). |
| R-delta-8 | `3b9087f` | Sub-head partition (softmax + delta + copy) | 44% plateau @ ep25. Convex-combination over sub-heads dilutes each mechanism to a fraction of full state capacity at d_head=2. Compose at OUTPUT, not residual. |
| R-delta-9 | `1e9925e` | Soft-gate dispatch (3-way weighted sum) | 46% plateau; gate learns specialization but convex combination mathematically dilutes each mechanism below full-strength contribution. |
| R-delta-11a/b | `6617a48` | Capacity scaling (d_model 64→128 in 11a; d_head 2→16 in 11b) | Both null on MQAR ceiling at R10's 500/N × 40 ep budget. Misread as capacity limit; actually data-starved — cracked by R13 data-scaling. |
| R-delta-16 | `187203d` | Scratchpad single-card (DT solves ((a*b)+c)+d step-by-step) | Both plain PT and DT at 0% full-expression match; loss trajectories identical (1.4→0.3 over 12 ep). State-carry mechanism present but arithmetic lookup requires ~243 single-digit + multi-digit facts — beyond 185K-param capacity. Brain+Cards thesis: use compiled `safe_eval` card for compute. |
| R-delta-18 | `78b5dfb` | Multi-head delta state (H=4 at d_model=64) | DT 21% vs plain PT 27% (-6pp). Per-head state (16, 16) = 256 scalars has D/log(D) capacity ~6 keys, well below N=15. Aggregate storage 4096→1024 scalars. Reopen only at d_model ≥128. |
| R-delta-19 | `65fb148` | D5 refinement loop (n_iterations=2) on MQAR | 21% best / 16% final, same plateau as R-delta-11/18. ARC Prize's "+13pp from refinement" is grid-reasoning-specific — MQAR has nothing to iteratively refine (single-token retrieval). May still help scratchpad (R-delta-16 retry deferred). |

## R22 install — full arc (rounds 1-7 + R22e adapter fix + R22f recalibration)

7-round debug arc + R22e diagnostic shipped at `min_margin=22.0`
(+9/60 on 2026-04-21, `73df738`). R22f (2026-04-22, `9691e06`)
recalibrated to **14.5** after diagnosing flat N=10 cells as
gate-silence, not card failure.

### Why 14.5, not 22.0 (R22f)

R22f sweep showed N=5 card margins cluster at p50=23.3 (above 22.0
threshold); N=10 p50=20.83 p5=15.21; N=15 p50=18.63 p5=16.39.
Threshold=22.0 was N=5-calibrated and over-gated N≥10 despite
standalone card being 100% correct (20/20 each) on those Ns.
Threshold=14.5 sits below observed p5 across all Ns and preserves
zero-regression invariant.

### Historical ships

**2026-04-21 initial ship at min_margin=22.0** (`73df738`): +9/60
(21% relative), fired 19/60, N=10 cells flat. Per-cell: N=5/500 +5,
N=5/1500 +2, N=10 both 0, N=15 +1/+1. Supersedes the interim r22b
rounds 1-7 (2W 1R / Δ=+1) which were ADAPTER-REGEX bug, not card
calibration — `parse_mqar_prompt`'s `value of X` pattern matched
distractor prose before the real `Question:`. Fix in `c3eac18`:
anchor query-key search on LAST `"Question:"` marker.

**2026-04-22 R22f recalibration** (`9691e06` + receipt
`.claude/MEMORY/evals/2026-04-22_r22f_threshold_sweep.md`): sweep
over {22.0, 18.0, 14.5} produced 51 / 56 / **60**/60 respectively,
all zero-regression. 14.5 shipped as new default.

### Result at 14.5 (`9691e06`)

```
baseline:  42/60  (70.0%)
with card: 60/60  (100%)    Δ=+18 absolute, 43% relative
hook fired: 59/60
WINS: 18    REGR: 0
```

Per-cell at 14.5: all six cells 10/10. R22d rerun (`c3cc73f`,
all-keys-per-mem-block corpus) independently confirmed 42/60 → 60/60
at the same threshold.

### Aligned-gate forensics

Commits: `e169d6d` r6 + `7db6eb9` r7 + `c3eac18` R22e + `73df738`
initial ship + `9691e06` R22f recal. The `preserve=True` legacy
mode pinning channels even when card writes nothing was the source
of round 6's `q=v margin=0.00` regression — fixed by switching to
`preserve=False`.

### Per-round arc receipts

- `.claude/MEMORY/evals/2026-04-21_r22a_mqar_card_install.md`
- `.claude/MEMORY/evals/2026-04-21_r22b_round1_no_failure_surface.md`
- `.claude/MEMORY/evals/2026-04-21_r22b_round2_mixed_signal.md`
- `.claude/MEMORY/evals/2026-04-21_r22b_round3_margin_threshold.md`
- `.claude/MEMORY/evals/2026-04-21_r22b_round4_holdout.md`
- `.claude/MEMORY/evals/2026-04-21_r22b_round5_6_gate_fix.md`

## R-delta-22 — noise-augmented training (CANCELLED by R22e)

Cancelled — the R22 adapter bug (not distribution shift) was the source
of the ~67% fired precision seen in r22b rounds 5-7. Card is **100%
accurate on clean adapter outputs** (R22e standalone: 60/60). No
train/test distribution gap exists for the R21 MQAR card on the
adapter-extracted MQAR format.

Scaffolding stays in tree as an option if a FUTURE card genuinely
shows distribution shift:

- `calm/hrm/memory_tasks.py::_gen_mqar_noisy` — four noise types
  (clustered_keys, zipf_values, whitespace, separator_variants)
- `calm/hrm/memory_tasks.py::gen_mqar_batch_noisy(noisy_frac=0.5)`
- `scripts/train_pt_delta_mqar.py --noisy-frac 0.5` (default 0.0
  preserves R21 behavior)

Do NOT retrain R21 with `--noisy-frac > 0` unless a new diagnostic
finds a real gap AFTER verifying the adapter parses correctly.

## DT code-skeleton arc — R1→R27+R25b (2026-04-22)

First extended DT training session on a natural-language code corpus
(MBPP / HumanEvalPlus / BigCodeBench / Claude-reasoning + stdlib + pip
signatures → `def FN(<args>):` skeletons). 20 commits `d62335e` →
`7ba612f`.

### P0 methodology findings

**R27 — split-before-aug is mandatory** (`fa654bb`). Pre-R27 pipelines
called `split_pairs(pairs)` AFTER `_paraphrase_augment()`, which meant
val contained 8× paraphrase variants of train problems. Train/val
shared the underlying problems; only surface phrasing differed. Metric
was memorization, not generalization.

**R26 — aux copy-attention loss prevents gate collapse** (`d26c56a`).
v9 measurement: copy-gate decayed 0.5 → **0.018** over 66 epochs while
augmented-val autoreg climbed 0.30 → 0.75. The metric said "winning";
honest unaug val was 0.284 — model had become a 370-way classifier
with no copy.

Fix: `_copy_aux_loss(model, input_ids, target_ids, pad_id)` in
`scripts/train_code_dt.py`. Position-gated: applies at positions where
target_ids[b,t] appears anywhere in input_ids[b]. Loss formula:
`aux = -log(copy_logits[b, t, target].clamp(min=1e-10)) * copyable_mask`.
Added to main NLL as `total = main + copy_aux_weight * aux`. Default
weight 0.5. Self-gating (on copy_logits mass) was rejected — creates
chicken-and-egg where copy path starts at ~0 and never activates.

**v9's 0.750 is an inflated metric** — preserve as historical receipt
only. Honest measurements on 271-sample unaug val:
- greedy: 0.262
- +R4 skeleton repair: 0.262 (0 flips — R4 is null at full-val too)
- beam=4: 0.284
- avg copy-gate: 0.018 (collapsed)

### P1 training-infra levers

| R | SHA | Lever | Default |
|---|---|---|---|
| R20 | `70bb94b` | DataLoader `num_workers=2` + `pin_memory=True` + batch 64→256 + `non_blocking` H2D | `--num-workers 2` |
| R20b | (inline) | `--eval-cap 300` — subsample val during training (autoreg is greedy per-sample, O(N×200ms); 1628-val stalls) | 300 |
| R21 | `4f12a36` | `EMAWeights` class; apply_shadow/restore around eval + save. **Decay=0.995 not 0.999** — 0.999 too slow for 100-ep budget | `--ema-decay 0.995` |

R20 infra was the stealth biggest lever of the session: 67-min-per-run
→ 20-min-per-run enabled 13 training iterations. At decay=0.999 EMA
shadow is ~70% init at ep6 — signal blocked. Decay 0.995 stabilizes
by ep12.

### P1 data levers

| R | SHA | Lever | Default |
|---|---|---|---|
| R3 | `4ebcf74` | `WeightedRandomSampler` on per-class `sqrt_inverse` weights | `--balanced-sampler sqrt_inverse` |
| R5 | `12bbf84` | `--copy-gate-bias-init` (v4 -2.0 / v5 +1.0 failed / v9 0.0 collapsed / v11-v13 `-1.0` works with R26) | `-1.0` with R26 |
| R6 | `5b12d30` | `--normalize-skeletons` (strip ann + whitespace) + `--drop-rare-count N` | `3` |
| R8 | `74dc1e4` | `--extract-all-defs` + 20 new paraphrase templates | on |
| R9 | `df3cd73` | `--synth-rare N --synth-rare-max M`: programmatic rare-class prompt synthesis via `rare_class_synth.py` (semantic-typed templates) | `--synth-rare 60 --synth-rare-max 50` |
| R10-R15 | `f8c0769` | Family (arg_count) + domain (url/db/request/user/file/matrix) + 3-arg templates + type-annotation stripping + 16 new paraphrase families | auto |
| R18 | `66d7263` | Widened per-semantic template libraries (14→34 int, 13→33 string/list, added url/db/request/user domain templates) | auto |
| R19 | `3f051ab` | `--dedupe-ambiguous`: drop conceptual prompts with 3+ distinct skeletons, majority-vote on 2-skeleton ambiguity | on |
| R25 | `be19450` | `build_stdlib_corpus.py` scraper — inspect.signature + `__doc__` on stdlib (67 modules, +835 pairs) | — |
| R25b | `7ba612f` | Expanded to 124 modules (stdlib + top pip) → +3761 pairs (4.5× R25) | — |

**R28 auto-scan-all-installed** (UNCOMMITTED, dangerous). Using
`pkgutil.iter_modules()` with `PYTHONPATH=.` imports top-level scripts
as modules, which RUNS their top-level code at import. During one run
this loaded Gemma (2.06 GB) into CUDA and ran `r53_phase2_bench.py`'s
full ablation. Repo-exclusion set is necessary but insufficient — any
auto-scan risks pip packages with import-time side effects.

### P1 inference levers (compose with any checkpoint)

| R | SHA | Lever | File |
|---|---|---|---|
| R4 | `0a1322b` | Skeleton-repair regex rewrites (ruled-out for accuracy, kept for output validity) | `calm/hrm/dt_skeleton_repair.py` |
| R22 | `9987b8f` | Beam search w/ skeleton-validity bias — when multiple beams tie on logp, prefer the one that parses as valid skeleton | `scripts/eval_dt_beam.py` |
| R24 | `358cdf2` | Comprehensive post-train eval: greedy / +repair / beam / beam+repair + per-class + copy-gate | `scripts/eval_dt_final.py` |

### Full DT run trajectory (session 2026-04-22)

| Run | Config | Best val | On | Fate |
|---|---|---:|---|---|
| v4 | baseline (handoff) | 0.298 | aug val | kill ep34 |
| v5 | +R3+R5+R6 @ gate +1.0 | 0.067 | aug val | kill ep10 (gate overshoot) |
| v9 | +R5+R6+R8 @ gate 0.0 + R20 | **0.750** | **aug val (INFLATED)** | plateau ep76 |
| v9 honest | eval_dt_final on v9 | **0.284** | 271 unaug | — |
| v10 | +R9+R19 pre-R27 | 0.000 | flawed | kill ep4 |
| v11 | +R26+R27 @ gate -1.0 EMA 0.999 | 0.000 | 228 honest | kill (EMA too slow) |
| v12 | EMA 0.995 | 0.004 | 228 honest | kill (data upgrade) |
| v13 | +R25b (3761 extra pairs), 520 honest val | **0.193** | 520 honest | kill ep16 for R28 scale |

v13 ep16 0.193 on 520-sample honest val is the first legitimate DT
capability measurement on this corpus. Trajectory through ep16:
0.097 → 0.143 → 0.127 → 0.140 → 0.147 → 0.163 → 0.193 (non-decelerating).

### Ruled-out for DT code-skeleton (don't retry)

- **R2 retrieval-aug prompt** (`c0d58c5`): −0.108 at gate=0.19. Inject
  top-k retrieved skeletons into prompt → model attends to distractors,
  scrambles emission. Gate must be healthier (>0.3) for retrieval-aug
  to shine.
- **R4 skeleton-repair for accuracy** (`0a1322b`): 0 flips on v4_mid,
  0 flips on v9-final full val. Model's errors are wrong-content-
  cleanly-formed, not malformations. Repair still useful for output
  validity but doesn't lift autoreg.
- **Gate init +1.0** (v5 ep10=0.067): blocks gen path from emitting
  structure tokens (`def`, `FN`, `(`, `:`) that are NOT in the prompt.
  Fabricates like `FN(num1, n2223)`. Only viable with explicit
  structure-token fallback.
- **Gate init 0.0 alone** (v9): without R26 aux, copy-gate collapses.
  Must pair with R26 aux_weight ≥ 0.3.
- **EMA decay 0.999** (v11): too slow for 100-ep budget. Use 0.995.
- **Pre-R27 paraphrase-aug val split** (v4-v9): measures memorization
  not generalization. Split BEFORE aug, val stays raw.

### Install status

`calm/llm_computer/dt_install.py` scaffold is in place (R22 CardSlot
pattern, L30, `write_margin=min_margin=14.5`). NOT yet run against
live Gemma on code prompts. Install threshold: DT honest val ≥ 0.40
before wiring — below that, distractor misses hurt Gemma more than
accurate hits help (R2 precedent).

Full receipts for this arc: `.claude/MEMORY/SESSION_HANDOFF.md`
(2026-04-22) + commit bodies d62335e..7ba612f.

## Cross-refs

- Current rules: `.claude/rules/delta_rule.md`
- MQAR data-scaling rule canonical receipt: `capability_gain.md`
  §"MQAR data-scaling rule"
- R-delta nulls also indexed in: `tracing_roadmap_part_2.md`
  §"R-delta arc ruled-out log"

---

## RENAME-beats-DT on MBPP (2026-04-23)

DT-bias install for MBPP-style signature prediction is **ruled out**.
Replacement: `CodeRenameFacade` (post-gen AST rename, zero training,
zero decode bias, 40 LOC).

### A/B receipt (N=20 MBPP)

| Method | Pass | Delta vs stock | Regressions |
|---|---|---|---|
| stock Gemma | 9/60 = 15% | — | — |
| CodeRenameFacade (oracle-rename) | 17/60 = 28% | +8 / +13pp | **0** |
| DT-bias install (v14) | 15/60 = 25% | +6 / +10pp | 1 |

DT regression: `remove_kth_element` 3/3 → 0/3 — DT predicted
`args=[list1, list2]` when actual signature is `(list, k)` (arity
hallucination). RENAME is immune by construction (only renames, never
changes behavior).

### v14 DT honest val (code-skeleton)

- Checkpoint: `dt_code_skel_v14_ep18_0200.pt`
- Honest val: 0.200 greedy / 0.314 beam (563 held-out)
- Avg copy-gate: 0.040 (collapsed despite R26 aux loss)
- Threshold for install viability was ≥ 0.40 honest val; never reached.

### Retrieval-signature null (N=20 MBPP)

A pure retrieval approach to signature prediction WITHOUT the AST
rename escape hatch (i.e., renaming Gemma's output to a retrieved
neighbor's fn name) regresses the baseline by -9 tests:

| Method | Pass | Delta |
|---|---|---|
| retr-self-ok (DB contains MBPP verbatim) | 17/60 = 28% | +8 (trivial — top-1 = oracle) |
| retr-self-skip (novel prompt simulation) | **0/60 = 0%** | **−9, 3 regressions** |

Regressions: `reverse_words` 3/3→0/3 (renamed to `solution`);
`remove_kth_element` 3/3→0/3 (renamed to `remove_even`);
`tuple_modulo` 3/3→0/3 (renamed to `index_multiplication`).

Codex's receipt line: "nearest-neighbor naming is not a safe
substitute for caller-known contract names on MBPP-like prompts."

### Commit ledger

| Commit | Content |
|---|---|
| `6b90d13` | DT install facade + MBPP A/B harness (+33pp on N=5) |
| `dac50ed` | RENAME beats DT (+13pp vs +10pp, 0 regressions at N=20) |
| `ed795ef` | Retrieval-signature null (−9 tests, 3 regressions) |
| `eda1ca7` | Preserve v17 pointer-sup hook + document batched-autoreg null |

### DT's role going forward

- **MQAR retrieval card** (`copy_augmented_delta_mqar_best.pt`) — stays, shipped, 100% on N=5/10/15.
- **Canonical trained-card architecture for retrieval + structure-extraction** — stays per `delta_rule.md` task-shape rule.
- **Substrate primitive** in VGSL (see `RESEARCH/VGSL/`) — stays as `verified_program` node candidate.
- **MBPP code-skeleton signature prediction** — ruled out. Do not retarget DT here.
- **Arbitrary-convention output prediction in general** — ruled out. DT's ceiling is the task-shape rule's validity.
