# R22b rounds 5-6 — CardSlot residual-write bug diagnosed + partially fixed

## Round 5 — pooled corpus exposes the bug

Hypothesis (round 5): 40 prompts insufficient to resolve effect magnitude;
pool two seeds into 60 prompts with N ∈ {5,10,15} × confusing distractor
modes for more signal.

Result:
```
OVERALL (pooled, t=22.0):
  baseline:  42/60
  with card: 38/60  (Δ=-4)
  hook fired: 4/60
  WINS: 3    REGR: 7
```

**Critical: 6 of 7 regressions had fired=False.** Hook silent yet Gemma's
top token changed from baseline.

Root cause: `CardSlot.card_output_fn` wrote card's log-probs to reserved
channels [2480:2560] on EVERY active prompt (regardless of margin). That
residual write flowed through to the head's vocab projection via
`preserve=True` masking, shifting Gemma's output independent of the
VerificationHook.

Round 3's claim of +2 at t=22 was based on a post-hoc sweep that assumed
"hook silent ⇔ Gemma baseline". The assumption was wrong — with CardSlot
installed but hook silent, Gemma is STILL affected by the residual write.

## Round 6 — margin-gate the residual write

Fix in `scripts/r22_install_mqar_card.py::install()`:
- New `write_margin: float = 0.0` parameter
- In `card_output_fn`, before writing card output to `h`, compute
  `(peak - median)` margin. If < write_margin, zero the logits AND skip
  the residual write.
- Mirrors the VerificationHook's margin gate so both intervention paths
  fire under the same confidence condition.

Rerun pooled corpus with `write_margin = hook.min_margin = 22.0`:

```
PER-CELL (pooled 2-seed):
   N   dist               mode   base     card    Δ
   5    500          confusing   5/10    6/10   +1
   5   1500     confusing_long   8/10    6/10   -2
  10    500          confusing   7/10    7/10    0
  10   1500     confusing_long   9/10    9/10    0
  15    500          confusing   7/10    8/10   +1
  15   1500     confusing_long   6/10    6/10    0

OVERALL (pooled, t=22.0, write_margin=22.0):
  baseline:  42/60
  with card: 42/60  (Δ=0)
  hook fired: 4/60
  WINS: 2    REGR: 2
```

**Regressions dropped 7 → 2** confirming the write-gate hypothesis.
Net is 0 because wins also dropped (3 → 2) — one of the round-5 wins
was via the residual-write path, not the hook.

## Remaining regressions

```
seed=20260423 N=5 dist=1500 mode=confusing_long q=y exp=1
  base='1' card='4' margin=22.61 fired=True
  — card confidently wrong, hook fired. Calibration issue.

seed=20260423 N=5 dist=1500 mode=confusing_long q=v exp=5
  base='5' card='7' margin=0.00 fired=False
  — margin 0 should have skipped hook AND write. Yet Gemma output
    differs from baseline. Likely cause: `preserve=True` masks
    subsequent layers' contributions to reserved channels even when
    card didn't write, so channels [2480:2560] remain pinned to L30's
    untouched output instead of being freely overwritten by L31-L41.
    Subtle but measurable effect on Gemma's head projection.
```

## Arc summary (R22b rounds 1-6)

| Round | Corpus | Δ | W/R | Note |
|---|---|---:|---|---|
| 1 | 30p, neutral 500 | -2 | 0/2 | no failure surface |
| 2 | 40p, confusing added | -2 | 2/4 | failure surface found |
| 3 | 40p, t=22 sweep | +2 (post-hoc) | 2/0 | **wrong assumption** |
| 4 | 40p, held-out seed | 0 | 0/0 | happened not to regress |
| 5 | 60p, pooled | -4 | 3/7 | **bug exposed** |
| 6 | 60p, write-gated | 0 | 2/2 | partial fix, calibration residue |

Arc takeaways:

1. **Post-hoc threshold sweeps are dangerous** when the ablation
   (baseline with NO card) is structurally different from the intervention
   (baseline with card installed but gate silent). The install mechanism
   has side effects beyond the gate.
2. **Residual-write gate must align with hook gate.** Otherwise the card
   affects output at any confidence level.
3. **`preserve=True` masking** in `GemmaSubstrate` may have subtle effects
   when card doesn't write — worth a dedicated round to test
   `preserve=False` at a cost of card output being overwritten by
   subsequent layers (would need VerificationHook reading from
   `last_output` which is already the case, so this might work).
4. **40-60 prompts is too small** for statistical significance on a
   ~5-10% effect size. Real product claim needs 200+ prompts.

## Next scope

R22c (multi-needle) was the original alternative to this distractor track.
With the mechanism understood (and two install bugs diagnosed), multi-needle
is worth trying:
- Gemma's 2026-04-07 NIAH multi-needle: 4/5 at 220K — documented failure.
- Adapter extension needed: parse two queries, compare to all k=v pairs,
  route to card for each answer slot.

OR: stop on R22 and return to the bigger-ticket items from
SESSION_HANDOFF.md (decode kernel queue, R13 MBPP walker baseline).

## Data

- `scripts/r22b_round5.py` — pooled corpus, exposed bug
- `scripts/r22b_round6.py` — write-gated, partial fix
- `scripts/r22_install_mqar_card.py` — install(), now with `write_margin`
- `.cache/r22b/round5_pooled.jsonl`, `.cache/r22b/round6_gated_write.jsonl`
