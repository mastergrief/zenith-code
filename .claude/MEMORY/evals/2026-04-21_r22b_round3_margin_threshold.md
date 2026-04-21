# R22b round 3 — margin-threshold calibration finds net-positive sweet spot

## Hypothesis

Round 2 showed 2 wins / 4 regressions at `min_margin=0.5`, net -2. The
card IS useful on the failure surface (Gemma's confusing-distractor
misses), but its OOD behavior produces "confident wrong" answers that
currently override correct Gemma baselines. Hypothesis: the card's
(peak - median) margin distribution differs between wins (high margin,
right) and regressions (moderate margin, wrong). A threshold `t` exists
such that card.margin ≥ t keeps most wins and cuts most regressions.

## Build

`scripts/r22b_round3.py` — run baseline + capture-with-card on the same
round-2 corpus (same seed). During capture, set hook `min_margin=0.0`
so it always captures `slot.last_output`. Record card's `(peak, median,
margin, argmax)` per prompt, then post-hoc sweep `min_margin` ∈
{0, 0.5, 2, 5, 10, 15, 18, 19, 19.5, 20, 20.5, 21, 21.5, 22, 22.5, 23, 25}.

For each threshold `t`: effective Gemma argmax per prompt is
`card_mapped_digit` if `card.margin >= t`, else `baseline_top`. Count
solves and compare to baseline.

## Test — first run (coarse sweep)

```
       t   solves   wins   regr    net
    base   35/40     —       —      —
     0.0   33/40      2      4   -2
     0.5   33/40      2      4   -2
     1.0   33/40      2      4   -2
     2.0   33/40      2      4   -2
     5.0   33/40      2      4   -2
    10.0   33/40      2      4   -2
    20.0   36/40      2      1   +1
```

Margin distribution (only card-active prompts where case was WIN or REGR):

```
 case   N  dist   mode             margin   base→card
  WIN   5   500  confusing            22.71   '3' → '2'
 REGR   5   500  confusing            21.82   '3' → '6'
  WIN   5  1500  neutral_long         22.95   '3' → '1'
 REGR  10   500  confusing            19.02   '3' → '6'
 REGR  10   500  confusing            19.58   '7' → '8'
 REGR  10  1500  confusing_long       19.66   '2' → '6'
```

**Clear bimodal split**: wins at 22.7+, 3 of 4 regressions at 19.0-19.7.
One regression (q=j at 21.82) falls between the modes.

## Test — second run (fine-grid sweep)

```
       t   solves   wins   regr    net
    base   35/40     —       —      —
     0.0   33/40      2      4   -2
     ...   (flat through t=19.0)
    19.0   33/40      2      4   -2
    19.5   34/40      2      3   -1
    20.0   36/40      2      1   +1
    20.5   36/40      2      1   +1
    21.0   36/40      2      1   +1
    21.5   36/40      2      1   +1
    22.0   37/40      2      0   +2   ← peak
    22.5   37/40      2      0   +2
    23.0   35/40      0      0    0   ← wins also cut
    25.0   35/40      0      0    0
```

**Clean transitions at the margin boundaries.** t=19.5 cuts the first
regression (the 19.02 one). t=20.0 cuts the 19.58 and 19.66. t=22.0
cuts the last regression (21.82) without losing any win (both wins
sit at 22.71/22.95). t=23.0 also cuts the wins → back to baseline.

**Best threshold: t=22.0**, net **+2** (37/40 vs baseline 35/40).

## Conclusion

**First net-positive capability gain from the PT+Delta MQAR card on
prod Gemma.** At `min_margin=22.0`:
- 2 WINS preserved (Gemma fixed on confusing-distractor failures)
- 0 REGRESSIONS
- Net **+2 over baseline 35/40 → 37/40**

This is a calibration win, not a mechanism win — the mechanism was
already proven in R22a. But it's the first time the R21 deployable
card as-installed produces a real product delta on a realistic
failure surface.

## Caveats

- Small sample: 2 wins / 6 card-active failure-candidate prompts = 33%
  win rate. 40-prompt corpus. Statistical significance weak.
- Threshold is empirically tuned on this corpus. Generalization to a
  held-out corpus (different seed, broader distractor variants) is
  round 4.
- The one "borderline" regression (q=j, margin 21.82) sits inside the
  bimodal gap. A threshold that catches it misses 1 win. A smart
  gate might need a second signal (Gemma's own confidence? card's
  confidence over Gemma's top-2 digit?) to handle borderlines.

## Round 4 scope

1. **Held-out eval**: new seed, same 8-cell corpus structure, apply
   `min_margin=22.0` fixed. Does net +2 hold on unseen prompts?
2. **Finer margin diagnostics**: log card's second-best + Gemma's top-2
   margin so a smarter gate can use relative confidence.
3. **Larger corpus**: scale from 40 to 100+ prompts for statistical
   significance. Now feasible with fast-tok patch.

## Data

- `scripts/r22b_round3.py` — baseline + capture + post-hoc sweep
- `.cache/r22b/round3_results.jsonl` — per-prompt margin logs
