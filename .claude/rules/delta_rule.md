# Delta-Transducer (DT) / DeltaNet — Card architecture rules

**DT (delta-transducer)** is the canonical product name (adopted
2026-04-22) for the copy-augmented DeltaNet trained-card architecture.
Underlying class `CopyAugmentedDeltaNet` unchanged. Use **DT** in new
prose, commits, training scripts (`scripts/train_code_dt.py`),
checkpoints (`dt_*_best.pt`), install paths
(`calm/llm_computer/dt_install.py`).

DT is the **default trained-card architecture for retrieval +
structure-extraction regimes**, superseding plain
`CopyAugmentedTransformer` for new work. Code-skeleton DT is a
separate open arc with a different recipe — see §"Code-skeleton
recipe" before extrapolating retrieval defaults to code.

> **Historical receipts** (R-delta-5 through R-delta-22, R22 install
> arc, full DT code-skeleton trajectory v4→v13): see
> `.claude/MEMORY/atlas/delta_rule_arc.md`.

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
   `p_copy · copy_dist + (1 - p_copy) · gen_probs`.
3. **Output returns log-probs** (not logits) — use `F.nll_loss`,
   NOT `F.cross_entropy`.

**Substrate invariant**: `d_head == 2`, asserted in
`build_copy_augmented_delta`. Total extra params over plain PT:
~260 (0.14%), just per-layer β heads.

## Chunkwise parallel form (UT transform)

Paper §3-4 algorithm (`RESEARCH/DELTA-RULE/02_Chunkwise_Parallel_Algorithm.md`).
Turns the O(L) per-position Householder loop into O(L/C) chunked
matmul-rich computation.

- Files: `delta_rule.py:_delta_chunkwise` (single-head),
  `_delta_chunkwise_multihead` (H>1)
- Enable via `DeltaNetConfig.use_chunkwise=True`, default `C=32`
- **Bit-equivalent** to per-position to fp32 epsilon
- Activation memory O(L·d²) via autograd — fits at d_model=64; larger
  d_model would need custom backward per FlashLinearAttention pattern

## Cached autoregressive decode

`CopyAugmentedDeltaNet.decode_greedy_cached(prefix_ids, max_gen, eos_token)`
— KV-cache equivalent for DeltaNet state + copy-K over prefix.

- **Prefill**: one full forward, captures per-layer `S_state` after
  prefix + `cached_copy_k = copy_k_proj(x)` over prefix positions.
  Honors `use_chunkwise` for prefill only.
- **Decode loop**: ONE new token per step — embed + per-layer
  `_delta_step` into cached S, single `Q_new @ K_cached` copy-attn,
  blend + argmax. Per-step cost O(L) → O(1).
- Constraints: batch=1 only (cached state is per-sample list); prefix
  must contain `<sep>` before decode.

## Default config (retrieval / NL math)

`CopyAugmentedDeltaConfig` sweet-spot defaults:
- `use_chunkwise=True` (always)
- `n_delta_heads=1`
- `n_iterations=1`
- `chunk_size=32`

DT is a strict functional superset of plain PT on copy-dominant
structure tasks (copy gate approaches 1.0, Delta contribution ~0).
Plain PT (`copy_augmented.py:CopyAugmentedTransformer`) stays in tree
as ablation baseline. Existing PT checkpoints preserved (sunk cost),
NO benefit to retraining.

**MQAR data-scaling rule**: at d_model=64, **"+5 on N needs 2× data."**
Plain PT is mechanism-capped at N≥10 regardless of data (softmax at
d_head=2 can't implement content-addressable lookup over ≥10 stored
pairs); DT's fast-weight state IS explicit (k→v) storage, so retrieval
cost doesn't depend on N.

**Task-shape rule**: DT wins when **key vocabulary is large AND each
key is sparse in the prefix**. Small-vocab reassign is softmax-solvable;
unique-key retrieval is where the mechanism advantage is load-bearing.

## Checkpoints

- `calm/hrm/checkpoints/copy_augmented_delta_best.pt` — NL math
  (100% val autoreg)
- `calm/hrm/checkpoints/copy_augmented_delta_mqar_best.pt` —
  deployable MQAR card, 100% on N=5/10/15 held-out. 748 KB, 183,877 params.
  Trained by `scripts/train_pt_delta_mqar.py`.
- `calm/hrm/checkpoints/dt_code_skel_v13_ep16_0193.pt` — code-skeleton
  DT (open arc), 0.193 honest val on 520 held-out. **Not install-viable**
  — threshold ≥ 0.40 honest val before wiring to Gemma.

## R22 install — current pattern

DT MQAR card installed on prod Gemma via `CardSlot` + `VerificationHook`
+ adapter:

```python
install(m, card, layer_idx=30, ch_off=2480,
        write_margin=14.5, preserve=False)
hook.min_margin = 14.5
# CARD_N_RANGE = {5, 10, 15}
```

**Four aligned gates** (load-bearing — change together):
1. `write_margin=14.5` — skips residual write when card unconfident
2. `hook.min_margin=14.5` — skips logit bias when card unconfident
3. `preserve=False` — lets L31-L41 freely overwrite reserved channels.
   `preserve=True` pins channels even when card writes nothing,
   subtly shifts Gemma. Use ONLY when channel isolation is
   load-bearing (chained cards reading earlier card's output channels).
4. N-range gate `{5, 10, 15}` — skips card on N outside training dist

**Threshold calibration rule**: for new retrieval cards, run standalone
on a representative corpus per input-distribution bucket, plot
(peak-median) margin distribution per bucket, set both gates below
the lowest observed p5 across all buckets. Single threshold calibrated
on one bucket will over-gate others.

Result: 42/60 → 60/60 (+18, 43% relative, 0 regressions) on the
distractor-confused MQAR corpus.

## Code-skeleton recipe (open arc, NOT install-viable)

Regime: NL problem description → `def FN(<args>):` skeleton. ~370-713
output classes, Zipf-distributed. Lower copyable-token density than
MQAR — most tokens in `def`/`FN`/`(`/`:` must be GENERATED, not copied.
**Retrieval defaults DO NOT transfer.**

Canonical flags for `scripts/train_code_dt.py`:

```
--balanced-sampler sqrt_inverse   # counter Zipf
--copy-gate-bias-init -1.0        # neutral; 0.0 collapses, +1.0 fabricates
--copy-aux-weight 0.5             # position-gated aux loss; prevents gate collapse
--ema-decay 0.995                 # 0.999 too slow for 100-ep budget
--normalize-skeletons             # strip type annotations + whitespace variants
--drop-rare-count 3
--extract-all-defs
--dedupe-ambiguous
--synth-rare 60 --synth-rare-max 50
--num-workers 2 --batch-size 256 --lr 3e-3 --eval-cap 300
```

**Mandatory pipeline order** (split-before-aug):
1. extract raw → normalize → dedup
2. SPLIT raw → train_raw / val (val never sees train problems)
3. synth rare (train only) → paraphrase aug (train only) → drop_rare (train only)

Splitting AFTER augmentation gives val 8× paraphrase variants of train
problems — measures memorization, not generalization. The aux copy-loss
(R26) prevents the copy gate from collapsing under generation-path
optimization pressure; without it, gate decays to ~0.018 and the model
becomes a 370-way classifier with no copy mechanism.

## Related rules

- `Substrate.md` — CardSlot / VerificationHook / in-attention install
- `augmentation_thesis.md` — tier-2 stacking framework
- `capability_gain.md` — measurement discipline (raw + user-facing)
- `training.md` — PT vs DT training recipes
- `MEMORY/atlas/delta_rule_arc.md` — full historical arc + receipts

## File map

| File | Role |
|---|---|
| `calm/llm_computer/delta_rule.py` | `DeltaNetConfig`, `DeltaNetSmall2DTransformer`, `_delta_step`, `_delta_chunkwise`, `_delta_chunkwise_multihead` |
| `calm/llm_computer/copy_augmented_delta.py` | `CopyAugmentedDeltaNet`, `decode_greedy_cached`, `build_copy_augmented_delta` |
| `calm/hrm/memory_tasks.py` | MQAR / reassign / scratchpad generators |
| `scripts/experiment_r10_mqar.py` | Ablation harness |
| `scripts/train_pt_delta_mqar.py` | Deployable MQAR card trainer |
| `scripts/train_code_dt.py` | Code-skeleton DT trainer |
| `calm/llm_computer/dt_install.py` | Install scaffold (R22 CardSlot pattern) |
| `RESEARCH/DELTA-RULE/02_Chunkwise_Parallel_Algorithm.md` | UT transform derivation |
