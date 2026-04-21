# R22b round 1 — no failure surface at 500-token distractor (hypothesis null)

Following the `capability_gain.md` §"Failure-surface gate" procedure.

## Hypothesis

Stock Gemma 4 E4B, which handled `<mem>` retrieval at 3/3 in R22a,
should break under longer context with distractor prefixes — specifically
at N_pairs ≥ 5 with ≥500 token distractor prefixes. The PT+Delta MQAR
card (distractor-invariant by construction — sees only the extracted
MQAR string) should lift the failing corpus.

## Build

- `scripts/r22b_gate.py` — combined gate+lift script. Generates 30
  candidates across 6 cells (N ∈ {3,5,10} × distractor ∈ {0, 500}),
  5 replicas per cell.
- Key optimizations to make iteration feasible:
  - Trie-backed fast tokenizer (`_monkey_patch_fast_encode` from
    `calm/llm_computer/facades/retrieval.py`) — tokenizing 30 prompts
    went from 4+ min stall → 0.0s.
  - Explicit Triton warmup: one forward per unique prefill length
    BEFORE the timing loop, so autotune (~3s per new S) doesn't
    pollute per-prompt measurements.
  - Cached token IDs per candidate (no re-tokenizing in loops).
- `make_prompt` embeds `<mem>k1=v1 ...</mem>` in the middle of a
  distractor-prose sandwich, then asks `Question: What is the value
  of X? Answer: `.

## Test

```
PER-CELL (baseline vs with-card):
  N   dist  base   card   Δ
  3      0  5/5    4/5    -1
  3    500  5/5    4/5    -1
  5      0  5/5    5/5    0
  5    500  5/5    5/5    0
 10      0  5/5    5/5    0
 10    500  5/5    5/5    0

OVERALL:
  baseline:  30/30
  with card: 28/30  (Δ=-2)
```

### Two card-induced regressions

Both in the N=3 cell (one with 0-distractor, one with 500-distractor):

```
N=3 dist=0   query=j expected=1  baseline_top='1' ✓  card_top='4' ✗
N=3 dist=500 query=e expected=9  baseline_top='9' ✓  card_top='7' ✗
```

Adapter parsed correctly in both (parse_ok=True) — so the card's
standalone forward produced the wrong digit, and `VerificationHook`
(min_margin=0.5) pushed Gemma toward it.

## Conclusion — hypothesis falsified

**500-token distractor + N≤10 is too easy** for Gemma. Its attention
handles the retrieval cleanly (30/30 baseline). The card adds small
regressions on N=3, which is OOD for R21's training distribution
(N ∈ {5,10,15}).

No failure surface found at this difficulty. Card-as-installed net
effect: **-2 prompts** (worse than stock).

## Performance receipts (iteration cost now manageable)

```
daemon load:           ~3 min (3-min first-load ceiling)
trie build:            0.3s
tokenize 30 prompts:   0.0s  (was 4+ min without patch)
warmup 16 shapes:      ~50s
BASELINE pass (30):    51.4s
install card:          ~1s
WITH-CARD pass (30):   52.4s
────────────────────────────
total script wall:     ~160s  (was 2+ hours before optimizations)
```

Per-prompt forward at S≈500: ~1.7s average.

## Round 2 scope

1. **Adapter gate**: only activate card when N_pairs ∈ {5..15} (the
   card's training distribution). N=3 prompts pass through with card
   inactive → no regression.
2. **Longer distractors**: 1500+ tokens, accepting the ~15-30s/prompt
   cost by reducing corpus size (maybe 9-15 prompts instead of 30).
3. **Confusing distractors**: include fake `x=y` pairs inside the
   distractor prose that could mislead Gemma's attention but won't
   mislead the card (which only sees the extracted `<mem>` content).
4. **Multi-needle**: ask about TWO keys, compound failure mode.

## Artifacts

- `scripts/r22b_gate.py` — the combined gate+lift script
- `scripts/r22b_measure_lift.py` — (now subsumed by gate)
- `.cache/r22b/candidates.jsonl` — 30 candidates
- `.cache/r22b/gate_lift_results.jsonl` — full results with top tokens
