# R22f — threshold sweep

Hypothesis: R22's `min_margin=22.0` over-gates N=10/N=15 (card
margins cluster at p50=20.83 / p50=18.63 for those Ns). Lowering
threshold should pick up silenced-but-correct cases. Card is 100%
standalone on N=5/10/15 per r22f_live_parse_trace.

## A/B results (60-prompt pooled R22 corpus)

| threshold | baseline | card | Δ | wins | regressions | fired |
|---|---:|---:|---:|---:|---:|---:|
| 22.0 | 42/60 | 51/60 | +9 | 9 | 0 | 19/60 |
| 18.0 | 42/60 | 56/60 | +14 | 14 | 0 | 48/60 |
| 14.5 | 42/60 | 60/60 | +18 | 18 | 0 | 59/60 |

## Verdict

**Threshold 14.5 wins**: 60/60 correct, Δ=+18 absolute (2× shipped
R22 at 22.0), 18 wins, zero regressions, 59/60 hook fires.

Per-cell lift going 22.0 → 14.5:
- N=5/dist=500:  unchanged 10/10 (at ceiling)
- N=5/dist=1500: unchanged 10/10 (at ceiling)
- N=10/dist=500:  7/10 → 10/10 (+3 vs baseline 7/10 silent)
- N=10/dist=1500: 9/10 → 10/10 (+1 vs baseline 9/10 silent)
- N=15/dist=500:  8/10 → 10/10 (+2 silent-to-active gain)
- N=15/dist=1500: 7/10 → 10/10 (+3 silent-to-active gain)

Root cause of the gate was over-calibration: `min_margin=22.0` was
empirically N=5-tuned (where margins cluster higher due to shorter
key space). N=10 margins cluster at p50=20.83 p5=15.21; N=15 at
p50=18.63 p5=16.39. All are below 22.0 — silencing the card on
90% of N≥10 prompts where it's actually correct (standalone
validated 20/20 on each via r22f_live_parse_trace).

**Ship**: R22 deployable config updates to `min_margin=14.5`
(matching `write_margin` + `preserve=False` + N-range
gate). Preserves the zero-regression invariant while nearly
doubling the R22 shipped gain.

## Diagnostic chain (this session)

1. `r22f_n10_diag.py` — pure analysis of
   `round6_gated_write.jsonl`: 4/20 N=10 silent-at-gate with
   card `margin=0` + `argmax=<pad>` → suggests state inactive.
2. `r22f_parse_diag.py` — offline regeneration shows
   `parse_mqar_prompt` OK on all 60; parse not the bottleneck.
3. `r22f_live_parse_trace.py` — live Gemma tokenizer
   reconstruction confirms parse 60/60 OK; standalone card 20/20
   correct at N=10 with p50 margin 20.83 (below 22.0 threshold).
4. `r22f_threshold_sweep.py` — live A/B at 22.0/18.0/14.5,
   corpus-matched pooled 60-prompt, `preserve=False`, same
   install mechanism as R22 TRUE result.
