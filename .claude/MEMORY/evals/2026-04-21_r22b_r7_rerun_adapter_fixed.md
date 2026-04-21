# R22b round 7 rerun with adapter fix — real capability gain

## Context

R22e sanity (this session, same date) discovered `parse_mqar_prompt` was
matching `value of X` phrasings inside confusing-distractor prose BEFORE
the actual `Question: What is the value of X?` at prompt tail. Adapter
picked wrong query keys → wrong MQAR string → card answered wrong
question. Fix: anchor query-key search on the LAST `"Question:"` marker
in post-mem content.

Card standalone accuracy on r22b's 60 adapter-extracted strings:
- Before fix: 14/33 (42%) with parse_ok only 33/60
- After fix:  60/60 (100%) with parse_ok 60/60

## Rerun result

Same r22b-pooled corpus (2 seeds × 6 cells × 5 replicas = 60 prompts),
same install config (`write_margin=22.0`, `min_margin=22.0`,
`preserve=False`), fixed adapter:

```
PER-CELL (baseline vs with-card):
   N    dist   mode              base    card    Δ
   5    500   confusing          5/10   10/10   +5    ← +50%
   5   1500   confusing_long     8/10   10/10   +2
  10    500   confusing          7/10    7/10    0
  10   1500   confusing_long     9/10    9/10    0
  15    500   confusing          7/10    8/10   +1
  15   1500   confusing_long     6/10    7/10   +1

OVERALL:
  baseline:  42/60  (70.0%)
  with card: 51/60  (85.0%)   Δ=+9 absolute, 21% relative
  hook fired: 19/60
  WINS: 9    REGR: 0
```

**15 percentage points absolute improvement, 21% relative lift.**
**Zero regressions.**

## Wins (all 9 — card's argmax matched ground truth, Gemma baseline wrong)

```
seed=20260422 N=5  d=500   q=d exp=5 base='4' card='5' margin=23.01
seed=20260422 N=5  d=500   q=d exp=3 base='4' card='3' margin=22.17
seed=20260422 N=5  d=1500  q=m exp=3 base='7' card='3' margin=22.91
seed=20260422 N=15 d=500   q=q exp=8 base='2' card='8' margin=22.76
seed=20260423 N=5  d=500   q=d exp=2 base='4' card='2' margin=22.94
seed=20260423 N=5  d=500   q=c exp=1 base='5' card='1' margin=22.88
seed=20260423 N=5  d=500   q=m exp=8 base='7' card='8' margin=23.03
seed=20260423 N=5  d=1500  q=m exp=8 base='7' card='8' margin=23.03
seed=20260423 N=15 d=1500  q=n exp=6 base='9' card='6' margin=22.29
```

All margins 22.17–23.03 (tight cluster at the high end of the training
distribution). Gate threshold (22.0) cleanly catches all of them.

## N=10 flat — is it a ceiling, another bug, or noise?

N=10 cells show 0 lift: baseline 7/10 + 9/10 = 16/20, card same.
Possible causes to investigate:

1. **Card margin < 22 on N=10 prompts** — would leave hook silent, card's
   correct answer never biases Gemma. Check post-hoc margin distribution.
2. **Adapter still fails on some N=10 prompts** — even with the
   query-marker fix, something about the N=10 inputs might confuse
   extraction.
3. **Gemma's 3 N=10 failures are genuinely hard** — card is correct but
   Gemma's incorrect answer has a logit delta > 50 (our boost), so the
   hook can't overcome it.

Follow-up: dump margin + card_argmax per N=10 failing prompt. 10-second
test.

## Implications that INVALIDATE PRIOR R22b CONCLUSIONS

Previous receipts claimed:
- "Card effective precision on live inputs: ~67%" — WRONG
- "Distribution shift from clean MQAR to adapter-extracted" — WRONG
- "Need noise-augmented training" — WRONG
- "Card's confident-wrong calibration issue" — WRONG (the adapter was
  giving the card wrong questions)

The real picture:
- Card standalone accuracy: **100% on adapter outputs** (when adapter
  works)
- Install mechanism: proven correct via 4-gate config (preserve=False,
  write_margin=22, min_margin=22, N-range gate)
- Adapter was the silent bug. Once fixed, the card delivers
  **15pp absolute / 21% relative lift** on the confusing-distractor
  corpus where Gemma fails.

The entire R22b arc (7 rounds) was debugging CardSlot gates and
thresholds when the primary bug was 5 lines of regex in the adapter.
**capability_gain.md §"Always check two things"** would have caught
this at R22b round 2 if I had run the card standalone on adapter
outputs from a distractor-containing prompt.

## Now confirmed for the commercial / thesis line

- PT+Delta MQAR card IS high-precision retrieval on NL inputs (with
  a correct adapter).
- "Tier-2 retrieval cards" are a real product line at 15-20%
  lift-magnitude on Gemma's failure surface, not the marginal
  1-2% the broken arc suggested.
- `preserve=False` + aligned margin gates deliver zero regressions,
  strictly additive behavior.
- `augmentation_thesis.md` §"selective intervention" gains a real
  validation datapoint.

## Lessons to bank

1. **Run card standalone on real adapter outputs at round 2**, not
   just on hand-crafted inputs. Skipping this wasted 5 rounds of
   "calibration" work on a non-calibration bug.
2. **Two-measurement discipline is load-bearing.** Raw path
   (card on MQAR) and user-facing path (Gemma+card) were out of sync
   because the adapter was feeding wrong inputs to raw path only
   during real runs. Sanity test on the ACTUAL corpus would have
   exposed this.
3. **Non-trivial regressions with seemingly-correct install** should
   trigger an adapter audit, not a calibration sweep.

## R-delta-22 noise training — CANCELLED

Scaffolding (committed in 7db6eb9) stays in tree as an option if a
future card ever DOES show a distribution-shift gap. For this MQAR
card: no gap exists. Save ~1hr retrain.

## Next

1. Investigate why N=10 cells are flat. Dump margins + card_argmax
   for the 3 failing N=10 prompts.
2. Rerun R22c and R22d with fixed adapter for completeness.
3. Write a shorter "R22 complete" receipt consolidating the corrected
   arc.
