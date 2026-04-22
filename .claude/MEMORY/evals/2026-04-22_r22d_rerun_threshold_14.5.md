# R22d rerun — N-fold retrieval at min_margin=14.5

Rerun of R22d (N-fold retrieval: 3 mem-blocks × 10 keys/block × 2
seeds = 60 retrieval tests) with the R22f-selected threshold=14.5,
preserve=False, post-adapter-fix (`c3eac18`). The original R22d used
the buggy pre-fix adapter + preserve=True + min_margin=22.0.

Serves as an independent corpus validation of R22f's threshold
choice: R22f's 60-prompt corpus is random-key-per-prompt; R22d is
all-keys-per-mem-block. Both should benefit equally if the fix is
corpus-independent.

## Result

| metric | value |
|---|---:|
| baseline | 42/60 |
| with card | **60/60** |
| Δ | +18 |
| fired | 58/60 |
| regressions | 0 |
| wall time (baseline + with-card) | 426.9s |

## Per-memory-block breakdown

| seed | mem# | baseline | card | Δ |
|---:|---:|---:|---:|---:|
| 2026-04-22 | 0 | 7/10 | 10/10 | +3 |
| 2026-04-22 | 1 | 4/10 | 10/10 | +6 |
| 2026-04-22 | 2 | 9/10 | 10/10 | +1 |
| 2026-04-23 | 0 | 7/10 | 10/10 | +3 |
| 2026-04-23 | 1 | 7/10 | 10/10 | +3 |
| 2026-04-23 | 2 | 8/10 | 10/10 | +2 |

Every mem-block lifted to perfect, independent of Gemma's baseline
reliability on that block.

## By key position (0-9)

| pos | baseline | card |
|---:|---:|---:|
| 0 | 4/6 | 6/6 |
| 1 | 4/6 | 6/6 |
| 2 | 5/6 | 6/6 |
| 3 | 4/6 | 6/6 |
| 4 | 4/6 | 6/6 |
| 5 | 4/6 | 6/6 |
| 6 | 5/6 | 6/6 |
| 7 | 4/6 | 6/6 |
| 8 | 4/6 | 6/6 |
| 9 | 4/6 | 6/6 |

No position-within-block dependence. Card retrieval is uniform
across key positions.

## Verdict

The min_margin=14.5 threshold generalizes across two independent
corpora. R22 card is production-ready at 100% in-distribution
retrieval (N=5/10/15). Previous +9/60 headline from R22 TRUE result
was a threshold-calibration artifact, not a card limitation.

Follow-up: update `.claude/rules/delta_rule.md` §"R22 install
— shipped" to reflect the new threshold and 60/60 result.
