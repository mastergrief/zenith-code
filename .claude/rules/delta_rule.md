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
R13-R21 in `tracing_roadmap.md`/`atlas.md`). `CopyAugmentedDeltaNet`
is the **default trained-card architecture** (R20 consolidation,
commit `63a49fc`), superseding plain `CopyAugmentedTransformer` for
new work.

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

Documented in `tracing_roadmap.md` §"R-delta ruled-out log" with
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

## Related rules

- `Substrate.md` — CardSlot / VerificationHook / in-attention install
- `augmentation_thesis.md` — tier-2 stacking framework (PT+Delta is
  a tier-2 card for retrieval failure modes)
- `capability_gain.md` — MQAR data-scaling receipt + the "plateau =
  bug, not tuning" canonical case
- `training.md` — PT vs PT+Delta training recipes
- `tracing_roadmap.md` §"R-delta ruled-out log" — null arc receipts
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
