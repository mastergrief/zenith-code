# Delta-Transducer (DT) / DeltaNet — Card architecture rules

**DT (delta-transducer)** is the canonical product name (adopted
2026-04-22) for the copy-augmented DeltaNet trained-card architecture.
Underlying implementation class `CopyAugmentedDeltaNet` stays; DT is
the product-level label used in new training scripts
(`scripts/train_code_dt.py`), checkpoints (`dt_*_best.pt`), and install
paths (`calm/llm_computer/dt_install.py`).

Older text below may still say "PT+Delta" or "CopyAugmentedDeltaNet" —
those refer to the same thing. Use **DT** in all new prose, commits,
and filenames.

R5→R21 arc (2026-04-21, "R-delta" scope — distinct from tracing-arc
R13-R21 in `MEMORY/atlas/tracing_arc_part_1.md`/`capabilities.md`). `CopyAugmentedDeltaNet`
is the **default trained-card architecture for retrieval + structure-
extraction regimes** (R20 consolidation, commit `63a49fc`), superseding
plain `CopyAugmentedTransformer` for new work. Code-skeleton DT is
a separate open arc with different training recipe — see §"DT code-
skeleton arc" at the bottom of this file before extrapolating these
defaults to code.

## Architecture

`calm/llm_computer/copy_augmented_delta.py:CopyAugmentedDeltaNet`
subclasses `calm/llm_computer/delta_rule.py:DeltaNetSmall2DTransformer`
which subclasses `Small2DTransformer`. Three mechanisms layered:

1. **DeltaNet backbone** (Yang 2024, arXiv:2406.06484) — Householder
   fast-weight recurrence at each layer:
   ```
   S_t = S_{t-1} - β_t (S_{t-1} k_t - v_t) k_t^T
   out_t = S_t @ q_t                (read-after-write)
   ```
   β_t ∈ (0, 1) learned per-position via `beta_head[layer]`. Keys/
   queries L2-normalized + optional SiLU feature map. State `S` is
   (d_model, d_model) per layer, reset each forward pass.
2. **Copy gate + pointer attention** (PT, session 31) — unchanged.
   `p_copy · copy_dist + (1 - p_copy) · gen_probs`. Copy bias init
   `-2.0` — model starts preferring generation, learns to copy.
3. **Output returns log-probs** (not logits) — use `F.nll_loss`,
   NOT `F.cross_entropy`.

**Substrate invariant preserved**: `d_head == 2`, asserted in
`build_copy_augmented_delta`. Total extra params over plain PT:
~260 (0.14%), just the per-layer β heads.

## Chunkwise parallel form (R17, UT transform)

Paper §3-4 algorithm (`02_Chunkwise_Parallel_Algorithm.md`).
Turns the O(L) per-position Householder loop into O(L/C) chunked
matmul-rich computation via pseudo-value `u_t = β_t(v_t - Σ u_i k_i·k_t)`
and the UT triangular solve `T = (I + tril(diag(β) KKᵀ, -1))⁻¹ diag(β)`.

- File: `calm/llm_computer/delta_rule.py:_delta_chunkwise` (single-head),
  `_delta_chunkwise_multihead` (H>1 with leading head dim)
- Enabled via `DeltaNetConfig.use_chunkwise=True`, default C=32 (sweet
  spot at L≤128 per paper)
- **Bit-equivalent** to per-position to fp32 epsilon (max |Δ| = 1.9e-6
  at L=64, d=64 — verified in R17)
- Training speedup (forward-only, B=16, fp32, RTX 4070):

  | L | per-position | chunkwise | speedup |
  |---:|---:|---:|---:|
  | 32 | 46.2 ms | 9.9 ms | **4.65×** |
  | 64 | 81.3 ms | 11.8 ms | **6.90×** |
  | 128 | 147.9 ms | 23.8 ms | **6.22×** |
  | 256 | 262.6 ms | 34.9 ms | **7.52×** |

- End-to-end training: R13-med-2k (N=[5,10] × 2K/N × 15ep) hit
  100% in **52s chunkwise vs 322s per-position** — ~6× wall-clock

Activation memory is O(L·d²) via autograd (no custom backward yet);
fits at d_model=64 but a larger d_model would need a custom backward
per paper's FlashLinearAttention pattern.

## Cached autoregressive decode (R20b)

`CopyAugmentedDeltaNet.decode_greedy_cached(prefix_ids, max_gen,
eos_token)` — KV-cache equivalent for DeltaNet state + copy-K over
prefix.

- **Prefill phase**: one full forward on prefix, captures per-layer
  `S_state` after prefix + `cached_copy_k = copy_k_proj(x)` over
  prefix positions. Honors `use_chunkwise` for prefill only.
- **Decode loop**: processes ONE new token per step:
  - Embed + per-layer `_delta_step` (one Householder update into
    cached S)
  - Single `Q_new @ K_cached` copy-attn, scatter_add to vocab
  - Blend + argmax
- Per-step cost drops from O(L) (uncached redoes full prefix) to O(1).

Measured inference on 200 NL math (max_gen=30, RTX 4070):

| path | accuracy | wall | vs plain PT |
|---|---:|---:|---:|
| per-position, uncached | 99.5% | 45.6 s | 5.92× |
| chunkwise, uncached | 99.5% | 15.1 s | 1.96× |
| **chunkwise, cached** | **99.5%** | **9.1 s** | **1.18×** |
| plain PT baseline | 99.5% | 7.7 s | 1.0× |

Parity: **50/50 token-exact match** vs uncached on held-out NL math.

Constraints: batch=1 only (cached state is per-sample list); prefix
must contain `<sep>` before decode. Commit `e6f2d5c`.

## MQAR data-scaling curve (R13 → R14-b)

Empirical rule for this architecture at d_model=64: **"+5 on N
needs 2× data."** Canonical receipt in `capability_gain.md`.

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

Plain PT is mechanism-capped at N≥10 regardless of data (softmax
at d_head=2 can't implement content-addressable lookup over
≥10 stored pairs). PT+Delta's fast-weight state IS explicit
(k→v) storage; retrieval cost doesn't depend on N.

## Task-shape-dependent moat (R15/R15-b)

Moat tracks how far the task is from softmax's natural biases
(recency + frequency):

| Task | plain PT N=10 | PT+Δ N=10 | gap |
|---|---:|---:|---:|
| MQAR (each key unique) | 34% | 100% | **+66pp** |
| Hard-reassign (20-var vocab) | 86% | 98% | +12pp |
| Small-vocab reassign (5 vars) | 100% | 100% | 0pp |

Commercial framing: PT+Delta wins when **key vocabulary is large
AND each key is sparse in the prefix**. Small-vocab reassign is
softmax-solvable (recency + frequency cues); unique-key retrieval
is where the mechanism advantage is load-bearing.

Also true even where both saturate: **PT+Delta converges ~3-10×
faster than plain PT** (R15-b: PT+Δ hits 94% at ep3, plain PT
hits 86% final at ep30). This compounds with chunkwise to
~20-50× faster training per card than plain-PT alternatives at
same final accuracy.

## R20 consolidation — defaults

Held-out test (`copy_augmented_hrm_best.pt` vs `copy_augmented_delta_best.pt`
on 200 NL math, seed=99999): **both 99.5%, delta 0.0 pp.**
PT+Delta is a strict functional superset on copy-dominant structure
tasks (copy gate approaches 1.0, Delta contribution ~0).

**`CopyAugmentedDeltaConfig` sweet-spot defaults** (commit `63a49fc`):
- `use_chunkwise=True` (always)
- `n_delta_heads=1` (R18 multi-head null at d_model=64)
- `n_iterations=1` (R19 D5 refinement null on MQAR)
- `chunk_size=32`

Plain PT (`calm/llm_computer/copy_augmented.py:CopyAugmentedTransformer`)
stays in tree as ablation baseline — every future architectural round
needs it as control. Existing PT checkpoints preserved (sunk cost),
NO benefit to retraining.

Checkpoints:
- `calm/hrm/checkpoints/copy_augmented_delta_best.pt` — R6a NL math
  (100% val autoreg at ep15, 2026-04-16)
- `calm/hrm/checkpoints/copy_augmented_delta_mqar_best.pt` — R21
  deployable MQAR card (2026-04-21), 100% on N=5/10/15 held-out,
  fresh seed=777777+N. 748 KB, 183,877 params. Trained by
  `scripts/train_pt_delta_mqar.py` (5K/N × N=[5,10,15] × 20 ep,
  chunkwise, scheduled sampling tf 1.0→0.3, ~2 min wall time).

## Nulls (for the ruled-out log)

Documented in `MEMORY/atlas/tracing_arc_part_2.md` §"R-delta ruled-out log" with
R-delta prefix to avoid collision with tracing-arc R-numbers:

- `dba270e` R-delta-5 pure DeltaNet at substrate scale (19.7% n=5)
- `3b9087f` R-delta-8 sub-head partition (44% plateau, capacity split)
- `1e9925e` R-delta-9 soft-gate dispatch (46% plateau, convex-combo dilutes)
- `97fba23` R-delta-6b plain PT chain test (task too easy to distinguish)
- `6617a48` R-delta-11a d_model 64→128 / R-delta-11b d_head 2→16
- `187203d` R-delta-16 scratchpad (state-carry ≠ arithmetic at 185K)
- `78b5dfb` R-delta-18 multi-head H=4 at d_model=64 (capacity wall)
- `65fb148` R-delta-19 D5 n_iters=2 on MQAR (ARC finding doesn't transfer)

Each null tightens the product claim rather than breaking it.
R15/R15-b narrowed "mutation tracking" to "sparse-key retrieval";
R16 confirmed composition-per-card thesis (compute → compiled cards,
recall → DeltaNet).

## R22 install — shipped (rounds 1-7 + R22e adapter fix + R22f
threshold recalibration, 2026-04-21 → 2026-04-22)

Card installed on prod Gemma via `CardSlot` + `VerificationHook` +
adapter. 7-round debug arc + R22e diagnostic shipped at `min_margin=22.0`
(+9/60 on 2026-04-21, `73df738`). R22f (2026-04-22, `9691e06`)
recalibrated the threshold to **14.5** after diagnosing the flat N=10
cells as gate-silence, not card failure:

```
install(m, card, layer_idx=30, ch_off=2480,
        write_margin=14.5, preserve=False)
hook.min_margin = 14.5
# CARD_N_RANGE = {5, 10, 15}
```

**Four aligned gates** (commits `e169d6d` r6 + `7db6eb9` r7 +
`c3eac18` R22e + `73df738` initial ship + `9691e06` R22f recal):
1. `write_margin=14.5` — skips residual write when card unconfident
2. `hook.min_margin=14.5` — skips logit bias when card unconfident
3. `preserve=False` — lets L31-L41 freely overwrite reserved channels.
   `preserve=True` pins channels even when card writes nothing,
   subtly shifts Gemma (round 6 `q=v margin=0.00` regression).
4. N-range gate `{5, 10, 15}` — skips card on N outside training dist

**Result at 14.5** (`9691e06`, same 60-prompt pooled corpus,
post-R22e adapter fix):

```
baseline:  42/60  (70.0%)
with card: 60/60  (100%)    Δ=+18 absolute, 43% relative
hook fired: 59/60
WINS: 18    REGR: 0
```

Per-cell at 14.5: all six cells 10/10. R22d rerun
(`c3cc73f`, all-keys-per-mem-block corpus) independently confirmed
42/60 → 60/60 at the same threshold.

**Why 14.5, not 22.0**: R22f sweep showed N=5 card margins cluster
at p50=23.3 (above 22.0 threshold); N=10 p50=20.83 p5=15.21; N=15
p50=18.63 p5=16.39. Threshold=22.0 was N=5-calibrated and over-gated
N≥10 despite standalone card being 100% correct (20/20 each) on
those Ns. Threshold=14.5 sits below observed p5 across all Ns and
preserves zero-regression invariant.

### Historical ships (preserved as receipts)

**2026-04-21 initial ship at min_margin=22.0** (`73df738`): +9/60
(21% relative), fired 19/60, N=10 cells flat. Per-cell: N=5/500
+5, N=5/1500 +2, N=10 both 0, N=15 +1/+1. Supersedes the interim
r22b rounds 1-7 (2W 1R / Δ=+1) which were ADAPTER-REGEX bug, not
card calibration — `parse_mqar_prompt`'s `value of X` pattern matched
distractor prose before the real `Question:`. Fix in `c3eac18`:
anchor query-key search on LAST `"Question:"` marker.

**2026-04-22 R22f recalibration** (`9691e06` + receipt
`.claude/MEMORY/evals/2026-04-22_r22f_threshold_sweep.md`): sweep
over {22.0, 18.0, 14.5} produced 51 / 56 / **60**/60 respectively,
all zero-regression. 14.5 shipped as new default.

## R-delta-22 — noise-augmented training (CANCELLED by R22e)

**Cancelled** — the R22 adapter bug (not a distribution shift) was
the source of the ~67% fired precision seen in r22b rounds 5-7. Card
is **100% accurate on clean adapter outputs** (R22e standalone:
60/60). No train/test distribution gap exists for the R21 MQAR card
on the adapter-extracted MQAR format.

Scaffolding stays in tree as an option if a FUTURE card genuinely
shows distribution shift:

- `calm/hrm/memory_tasks.py::_gen_mqar_noisy` — four noise types
  (clustered_keys, zipf_values, whitespace, separator_variants)
- `calm/hrm/memory_tasks.py::gen_mqar_batch_noisy(noisy_frac=0.5)`
- `scripts/train_pt_delta_mqar.py --noisy-frac 0.5` (default 0.0
  preserves R21 behavior)

Do NOT retrain R21 with `--noisy-frac > 0` unless a new diagnostic
finds a real gap AFTER verifying the adapter parses correctly.

Full per-round arc receipts:
- `.claude/MEMORY/evals/2026-04-21_r22a_mqar_card_install.md`
- `.claude/MEMORY/evals/2026-04-21_r22b_round1_no_failure_surface.md`
- `.claude/MEMORY/evals/2026-04-21_r22b_round2_mixed_signal.md`
- `.claude/MEMORY/evals/2026-04-21_r22b_round3_margin_threshold.md`
- `.claude/MEMORY/evals/2026-04-21_r22b_round4_holdout.md`
- `.claude/MEMORY/evals/2026-04-21_r22b_round5_6_gate_fix.md`

## DT code-skeleton arc — R1→R27+R25b (2026-04-22)

First extended DT training session on a natural-language code corpus
(MBPP / HumanEvalPlus / BigCodeBench / Claude-reasoning + stdlib + pip
signatures → `def FN(<args>):` skeletons). 20 commits `d62335e` → `7ba612f`.

### P0 methodology findings (read BEFORE next DT training)

**R27 — split-before-aug is mandatory** (`fa654bb`). Pre-R27
pipelines called `split_pairs(pairs)` AFTER `_paraphrase_augment()`,
which meant val contained 8× paraphrase variants of train problems.
Train/val shared the underlying problems; only surface phrasing
differed. Metric was memorization, not generalization.

Correct pipeline:
1. extract raw → normalize → R19 dedup
2. SPLIT raw → train_raw / val  (val never sees train problems)
3. synthesize rare (train only) → paraphrase-aug (train only) → drop_rare (train only)

**R26 — aux copy-attention loss prevents gate collapse** (`d26c56a`).
Without direct supervision, DT's copy mechanism collapses under
generation-path optimization pressure. v9 measurement: copy-gate
decayed 0.5 → **0.018** over 66 epochs while augmented-val autoreg
climbed 0.30 → 0.75. The metric said "winning"; honest unaug val
was 0.284 — the model had become a 370-way classifier with no copy.

Fix: `_copy_aux_loss(model, input_ids, target_ids, pad_id)` in
`scripts/train_code_dt.py`. Position-gated: applies at positions
where target_ids[b,t] appears anywhere in input_ids[b] (i.e. target
is in-principle copyable). Self-gating (on copy_logits mass) was
rejected — creates chicken-and-egg where copy path starts at ~0
and never activates.

Loss formula: `aux = -log(copy_logits[b, t, target].clamp(min=1e-10)) * copyable_mask`.
Added to main NLL as `total = main + copy_aux_weight * aux`.
Default weight 0.5. Expose `copy_logits` via
`CopyAugmentedDeltaNet._last_copy_logits_grad` (NOT detached — grad
flows back through copy_q_proj/copy_k_proj).

**v9's 0.750 is an inflated metric** — preserve as historical
receipt only. Honest measurements on 271-sample unaug val:
  - greedy: 0.262
  - +R4 skeleton repair: 0.262 (0 flips — R4 is null at full-val too)
  - beam=4: 0.284
  - avg copy-gate: 0.018 (collapsed)

### P1 training-infra levers

| R | SHA | Lever | Default |
|---|---|---|---|
| R20 | `70bb94b` | DataLoader `num_workers=2` + `pin_memory=True` + batch 64→256 + `non_blocking` H2D | `--num-workers 2` |
| R20b | (inline) | `--eval-cap 300` — subsample val during training (autoreg is greedy per-sample, O(N×200ms); 1628-val stalls) | 300 |
| R21 | `4f12a36` | `EMAWeights` class; apply_shadow/restore around eval + save. **Decay=0.995 not 0.999** — 0.999 is too slow for 100-ep budget | `--ema-decay 0.995` |

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
| R9 | `df3cd73` | `--synth-rare N --synth-rare-max M`: programmatic rare-class prompt synthesis via `rare_class_synth.py` (semantic-typed templates: int/string/list/url/db/etc.) | `--synth-rare 60 --synth-rare-max 50` |
| R10-R15 | `f8c0769` | Family (arg_count) + domain (url/db/request/user/file/matrix) + 3-arg templates + type-annotation stripping + 16 new paraphrase families | auto |
| R18 | `66d7263` | Widened per-semantic template libraries (14→34 int, 13→33 string/list, added url/db/request/user domain templates) | auto |
| R19 | `3f051ab` | `--dedupe-ambiguous`: drop conceptual prompts with 3+ distinct skeletons, majority-vote on 2-skeleton ambiguity | on |
| R25 | `be19450` | `build_stdlib_corpus.py` scraper — inspect.signature + `__doc__` on stdlib (67 modules, +835 pairs) | — |
| R25b | `7ba612f` | Expanded to 124 modules (stdlib + top pip) → +3761 pairs (4.5× R25) | — |

**R28 auto-scan-all-installed** (UNCOMMITTED, dangerous). Using
`pkgutil.iter_modules()` with `PYTHONPATH=.` imports top-level
scripts as modules, which RUNS their top-level code at import. During
one run this loaded Gemma (2.06 GB) into CUDA and ran
`r53_phase2_bench.py`'s full ablation. Repo-exclusion set is
necessary but insufficient — any auto-scan risks pip packages with
import-time side effects. Repo-exclusion edit is in the working tree
but not committed; consider committing with explicit docstring warning
or reverting to hand-curated module list.

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
0.097 → 0.143 → 0.127 → 0.140 → 0.147 → 0.163 → 0.193 (non-
decelerating).

### Ruled-out for DT code-skeleton (don't retry)

- **R2 retrieval-aug prompt** (`c0d58c5`): −0.108 at gate=0.19. Inject
  top-k retrieved skeletons into prompt → model attends to distractors,
  scrambles emission. Gate must be healthier (>0.3) for retrieval-aug
  to shine.
- **R4 skeleton-repair for accuracy** (`0a1322b`): 0 flips on v4_mid,
  0 flips on v9-final full val. Model's errors are wrong-content-
  cleanly-formed, not malformations. Repair still useful for output
  validity (cheap invariant-keeping) but doesn't lift autoreg.
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

## Related rules

- `.claude/spec/Substrate.md` — CardSlot / VerificationHook / in-attention install
- `augmentation_thesis.md` — tier-2 stacking framework (PT+Delta is
  a tier-2 card for retrieval failure modes)
- `capability_gain.md` — MQAR data-scaling receipt + the "plateau =
  bug, not tuning" canonical case
- `training.md` — PT vs PT+Delta training recipes
- `MEMORY/atlas/tracing_arc_part_2.md` §"R-delta ruled-out log" — null arc receipts
- `workflow_part_1.md` §"The loop" — hypothesis/test/commit discipline
  that produced the R-delta arc

## File map

| File | Role |
|---|---|
| `calm/llm_computer/delta_rule.py` | `DeltaNetConfig`, `DeltaNetSmall2DTransformer`, `_delta_step`, `_delta_chunkwise`, `_delta_chunkwise_multihead`, `_delta_layer_stack` |
| `calm/llm_computer/copy_augmented_delta.py` | `CopyAugmentedDeltaNet`, `decode_greedy_cached`, `_predict_next_token`, `build_copy_augmented_delta` |
| `calm/hrm/memory_tasks.py` | MQAR / reassign / reassign_hard / scratchpad generators (`gen_*_batch`) |
| `scripts/experiment_r10_mqar.py` | Ablation harness with `--task`, `--chunkwise`, `--n-delta-heads`, `--n-iterations` |
| `scripts/train_pt_delta_mqar.py` | Deployable card trainer, saves `copy_augmented_delta_mqar_best.pt` |
| `RESEARCH/DELTA-RULE/02_Chunkwise_Parallel_Algorithm.md` | Paper refactor (UT transform derivation) |
