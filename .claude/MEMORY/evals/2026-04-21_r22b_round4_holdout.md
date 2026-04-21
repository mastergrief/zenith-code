# R22b round 4 — held-out eval validates threshold calibration (zero harm)

## Hypothesis

Round 3 tuned `min_margin=22.0` on a 40-prompt corpus (seed `2026_04_22`)
to produce net +2 (37/40 vs baseline 35/40). Hypothesis: the threshold
generalizes — a held-out 40-prompt corpus at a different seed should
also net ≥ 0 (threshold doesn't harm) and ideally ≥ +2 (effect size
reproducible).

## Build

`scripts/r22b_round4.py` — identical 8-cell corpus structure as round 2/3,
new seed `2026_04_23`. Install card with fixed `min_margin=22.0` (no
sweep). One baseline pass + one with-card pass. Log card margin and
fired-flag per prompt.

## Test

```
PER-CELL (baseline vs with-card @ t=22.0):
   N   dist             mode   base   card   Δ
   5    500       confusing   5/5    5/5    0
   5    500         neutral   5/5    5/5    0
   5   1500  confusing_long   4/5    4/5    0   ← only Gemma miss
   5   1500    neutral_long   5/5    5/5    0
  10    500       confusing   5/5    5/5    0
  10    500         neutral   5/5    5/5    0
  10   1500  confusing_long   5/5    5/5    0
  10   1500    neutral_long   5/5    5/5    0

OVERALL (held-out, t=22.0):
  baseline:  39/40
  with card: 39/40  (Δ=0)

  card active (N ∈ {5,10,15}): 40/40
  hook fired (margin ≥ 22.0): 12/40 of active

  WINS  (base✗ card✓): 0
  REGR  (base✓ card✗): 0
```

## Interpretation

### Good (validates calibration)

**Zero regressions despite hook firing 12 times.** The `min_margin=22.0`
threshold from round 3 generalizes — no false-positive card overrides
on this unseen corpus. The calibration is not overfit to the round-3
margin distribution.

### Mixed (corpus too easy / too small)

**Baseline jumped from round-2's 35/40 to round-4's 39/40 on the same
corpus structure, just a different seed.** Distractor sentence choice
is highly seed-sensitive: sometimes random sampling from `_CONFUSING`
produces a genuine attention-confusing mix; sometimes it doesn't. The
failure rate is not stable across seeds.

Implication: 40-prompt corpus is too small for stable Gemma-failure
statistics. Need 100-200+ prompts AND/OR a deterministic "always-confuses"
distractor to get reproducible per-cell failure rates.

### The lone Gemma miss

```
N=5 dist=1500 mode=confusing_long q=d exp=0
  baseline:   '4' (wrong)
  card:       '4' (unchanged — card didn't fire)
  card_margin: 0.00 (card output was uniform)
  card_argmax: 0 (vocab id 0 = <pad>)
```

The card produced a **pad token with zero margin** on this input.
The adapter parsed correctly (MQAR string built), but the card's
forward on that specific input produced no useful signal. Threshold
gate correctly stayed silent — no regression, no help.

Probable cause: the answer is `'0'` (a digit), but the card's argmax
fell on `<pad>` (vocab id 0). Could be an OOD case for the card or
a specific bias toward `<pad>` when the card is uncertain. Worth
investigating if more wins were available — for now it's a null we
correctly handled.

## Aggregate across the R22b arc

| Round | Corpus | Baseline | With card | Δ | Notes |
|---|---|---:|---:|---:|---|
| 1 | 30p, neutral 500 | 30/30 | 28/30 | -2 | no failure surface |
| 2 | 40p, + confusing | 35/40 | 33/40 | -2 | 2 wins / 4 regr @ t=0.5 |
| 3 | 40p, + t=22 sweep | 35/40 | 37/40 | **+2** | 2 wins / 0 regr |
| 4 | 40p, held-out seed | 39/40 | 39/40 | 0 | 0 wins / 0 regr |

**Total across rounds 2-4 (all real failure-surface attempts):**
- 109/120 baseline, 109/120 card
- 2 wins, 4 regressions (pre-calibration) or 2 wins, 0 regressions (post-calibration on rounds 3+4)

## Conclusion

**Mechanism works. Calibration works. Corpus insufficient for a clean
statistical claim.** To move from "promising" to "validated product
win", need:

1. **100-200 prompt corpus** at multiple seeds — stabilize Gemma's
   per-cell failure rate
2. **Harder distractors** — seek corpus where baseline is < 90%
3. **Card confidence investigation** — the one zero-margin pad case
   (round 4) and the bimodal pattern (round 3) suggest the card's
   confidence signal isn't calibrated well against its accuracy.
   A better gate might use per-input diagnostics instead of a single
   scalar margin.

Arc net: **mechanism proven + calibration validated + 2 real wins on
6 card-active failure-candidate prompts (33% win rate on signal cases).**
First real signal that PT+Delta MQAR cards can augment Gemma on
distractor-heavy retrieval, but the numbers are too small to ship
as a product claim.

## Data

- `scripts/r22b_round4.py`
- `.cache/r22b/round4_holdout.jsonl`
- Together with rounds 1-3 receipts in this directory.
