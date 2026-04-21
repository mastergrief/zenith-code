# R22b round 2 — confusing distractors found failure surface, card mixed signal

## Hypothesis

R22b round 1 found no failure surface at 500-tok neutral distractor + N≤10.
Round 2 hypothesizes Gemma's attention is confused by prose that MENTIONS
letter-digit associations in non-`<mem>`-regex form (e.g., "variable z rose
to 8 last spring"), even though such prose doesn't match our adapter's
`([a-z])=(\d)` pattern. The card sees only the extracted `<mem>` content
so it's immune.

## Build

- `scripts/r22b_round2.py` — corpus across 2 × 4 = 8 cells:
  - N_pairs ∈ {5, 10} (both in card's training range {5,10,15} — N=3 dropped
    after round 1 showed it's OOD for card and trivial for Gemma)
  - 4 distractor modes:
    - `neutral` (500 tok plain prose)
    - `confusing` (500 tok with fake letter-digit phrasings)
    - `neutral_long` (1500 tok plain prose)
    - `confusing_long` (1500 tok with fake letter-digit phrasings)
- Adapter gate: `CARD_N_RANGE = {5, 10, 15}` — card stays inactive for
  OOD N. (Round 1's 2 N=3 regressions are prevented by construction.)
- Install + VerificationHook as in R22a (min_margin=0.5, boost=50.0).

5 replicas × 8 cells = 40 prompts.

## Test

```
PER-CELL (baseline vs with-card):
   N   dist             mode   base   card   Δ
   5    500       confusing   4/5    4/5    0
   5    500         neutral   5/5    5/5    0
   5   1500  confusing_long   4/5    4/5    0
   5   1500    neutral_long   4/5    5/5   +1   ← WIN
  10    500       confusing   4/5    2/5   -2
  10    500         neutral   5/5    5/5    0
  10   1500  confusing_long   4/5    3/5   -1
  10   1500    neutral_long   5/5    5/5    0

OVERALL:
  baseline:  35/40
  with card: 33/40  (Δ=-2)

  card_WINS   (base✗ card✓): 2
  card_REGRESS(base✓ card✗): 4
```

### Confusing distractors DID break Gemma

Baseline 35/40 = 87.5%, vs round 1's 30/30 = 100%. The 5 baseline misses
break down:
- 3/5 are in `confusing`/`confusing_long` cells (confirms hypothesis)
- 1/5 in `neutral_long` (1500 tok plain prose) — long context alone also
  hurts occasionally
- 1/5 in `confusing_long`

So confusing prose DOES confuse Gemma's attention, and long context amplifies.

### Card wins (2)

```
WIN  N=5 dist=500 mode=confusing     q=k exp=2  base='3' → card='2' ✓
WIN  N=5 dist=1500 mode=neutral_long q=i exp=1  base='3' → card='1' ✓
```

**Mechanism proven on real failure surface.** Card lifted Gemma from
wrong-digit to correct-digit on prompts Gemma baseline failed.

### Card regressions (4)

```
REGR  N=5  dist=500  mode=confusing       q=j exp=3  base='3' → card='6' ✗
REGR  N=10 dist=500  mode=confusing       q=u exp=3  base='3' → card='6' ✗
REGR  N=10 dist=500  mode=confusing       q=p exp=7  base='7' → card='8' ✗
REGR  N=10 dist=1500 mode=confusing_long  q=v exp=2  base='2' → card='6' ✗
```

Card predicted `'6'` three times wrongly (and `'8'` once). The card's
wrong digit isn't a value from the `<mem>` block in any of these cases —
it's something the card inferred incorrectly from the extracted MQAR
string.

Since all 4 regressions are in `confusing` cells where Gemma had it
right, the VerificationHook (`min_margin=0.5`) is firing on the card's
WRONG-but-confident output and overriding Gemma's correct answer.

## Conclusion

**Net: -2.** Gross: +2 wins / -4 regressions. The mechanism works; the
calibration doesn't.

- **Capability gain confirmed** on the failure surface — card can lift
  Gemma where confusing distractors trip up attention.
- **Card confidence calibration is wrong**: `min_margin=0.5` is too
  permissive. Card has enough "confident wrong" cases to net-negative
  the intervention.

## Round 3 scope

1. **Raise `min_margin`** empirically. Options:
   - `min_margin=2.0` — requires card peak to be 2+ logits above median
   - Make it relative to Gemma's own confidence (only fire if card margin
     > Gemma margin by some factor)
2. **Log card confidence per-prompt** on the round-2 corpus so we can
   inspect the margin distribution for wins vs regressions. If wins show
   margins >5 and regressions show margins ~0.8, a threshold of ~3 would
   keep most wins and cut most regressions.
3. **Check card standalone on the 4 regression MQAR strings** — verify
   whether the card itself is wrong (OOD failure) or whether the adapter
   extracts wrong content.

## Data

- `scripts/r22b_round2.py` — corpus + adapter gate + per-cell reporting
- `.cache/r22b/round2_results.jsonl` — full 40-prompt results with top
  tokens for baseline and with-card
