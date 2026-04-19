# R53.20 — Substrate-RAG null result (post-SWA-fix)

2026-04-19, post 50f27e3 (post-forward SWA trim + max_len=32K).

Two rounds designed to validate substrate-RAG now that the SWA
attention bug is resolved. Both FALSIFY the substrate-RAG thesis as
implemented (CardSlot @ L41 + per-marker FirstTokenHook boost=50).

## R53.20a — Re-run R53.14 substrate-RAG vs stock vs prompt-RAG

Hypothesis: the -9.3pp regression from pre-fix R53.14 was SWA-bug
artifact. With proper attention, substrate-RAG ≥ stock.

Corpus: 6 R53.0 complex problems (linked_list, date_validation,
log_level, csv_column_stats, token_bucket, lru_cache).
Conditions: stock / prompt-RAG (channel-code-hybrid) / substrate-RAG
(KnowledgeStore@L41 + FirstTokenHook, per-marker "def"/"class").

| problem | stock | prompt | substrate |
|---|---|---|---|
| linked_list_bugs | 0/0 | 0/0 | 0/0 |
| date_validation | 10/12 | 10/12 | 10/12 |
| log_level_counts | 6/6 | 6/6 | **0/0** |
| csv_column_stats | 0/0 | 0/0 | 0/0 |
| token_bucket | 0/0 | 0/0 | 0/0 |
| lru_cache_class | 9/9 | 9/9 | **0/0** |
| **TOTAL** | **25/27** | **25/27** | **10/12** |

Δ substrate-vs-stock: **-9.3pp** (identical to pre-SWA-fix run).

**Hypothesis falsified.** SWA bug was not the cause. Substrate install
mechanism itself disrupts Gemma's output on HIT prompts. Matches
prior handoff observation: "even with hook silent, CardSlot's
residual write at ch[2480:2497] disrupts hits regardless."

## R53.20b — Stacked: substrate + prompt-RAG + structured repair

Hypothesis: repair layer (R53.19's categorizer + targeted retry)
recovers substrate's attempt-1 regressions, so the stack ≥ R53.19
v3's 26/26.

Stack: attempt 1 uses substrate (enrolled hash) + channel-code-hybrid
hints. Attempts 2-3 silence substrate (`current_query["key"]=0`,
non-enrolled hash → card output all-zero → Gemma unmodified) +
structured repair hint.

| problem | R53.19 v3 | R53.20b (att 1 → final) | Δ |
|---|---|---|---|
| linked_list_bugs | 5/5 | 0/0 → 0/0 | **-5** |
| date_validation | 12/12 | 10/12 → 10/12 | **-2** |
| log_level_counts | 0/0 | 0/0 → 0/0 | 0 |
| csv_column_stats | 0/0 | 0/0 → 0/0 | 0 |
| token_bucket | 0/0 | 0/0 → 0/0 | 0 |
| lru_cache_class | 9/9 | 9/9 → 9/9 | 0 |
| **TOTAL** | **26/26** | **19/21** | **-7** |

**Hypothesis falsified.** Substrate's attempt-1 disruption is
POLLUTIVE: when first-token bias forces Gemma off its natural output
format, attempt 1 produces unparseable or malformed code → repair
categorizer sees "NoCode" or "NameError" → Gemma can't reconstruct
clean output that attempt-1-without-substrate would have produced.

## Diagnosis

L41 + preserve=True + FirstTokenHook(boost=50) is architecturally
wrong for code tasks:

1. Gemma's first-token logit margin on code prompts is uniformly
   ~6.8-9.2 (very confident on whitespace/fence openers). The
   `min_margin=0.5` guard never fires silencing.
2. Per-marker targets ("def"/"class") are CONTENT choices, but
   Gemma's actual format variance is about FENCING (```python vs
   bare). Forcing "def" as first BPE when Gemma wanted to emit
   "```python\ndef..." produces code without fences → extractor
   fails.
3. CardSlot preserve=True at L41 has no subsequent layers to zero,
   so miss-case is bit-identical. But hit-case writes additively to
   ch[2480:2497], perturbing residual enough to change Gemma's
   downstream token-sequence habits.

## Viable paths forward

- **R53.21 CalmDecodeHook (per-token verifier)**: inspect top-K
  candidates mid-decode, suppress tokens that would produce
  NameError / SyntaxError / verifiable-wrong claims. Tier-1 by
  construction — no intervention when no verdict.
- **Pure PT / RAG without substrate**: R53.19 v3's 26/26 is the
  current ceiling via channel-code-hybrid + structured repair alone.
  Gains beyond 26/26 require decode-time intervention, not residual-
  level install.
- **Abandon first-token bias** on code tasks. The mechanism works
  for numeric answers (R11 multiplier, R46.2 multi-step) where the
  answer is a specific token sequence, but fails on code where the
  "right answer" is a family of equivalent outputs.

## Ruled out

- SWA bug as explanation for R53.14 regression (R53.20a confirms).
- Substrate install @ L41 + FirstTokenHook as standalone or stacked
  intervention on R53.0 corpus (R53.20a, R53.20b both regress).

## References

- Script: `scripts/r53_substrate_rag_eval.py` (R53.20a re-run)
- Script: `scripts/r53_20b_stacked.py` (R53.20b new)
- Commit: `ec8887f` (R53.20a null + R53.20b scaffold)
- Baseline: R53.19 v3 (26/26) from prior handoff
- Rules: `.claude/rules/augmentation_thesis.md` §"Automatic Tier-1
  preservation" — the property prompt-RAG lacks and substrate-RAG
  was supposed to provide; this round shows the provided mechanism
  *replaces* Tier-1 violation (blanket inject) with a *different*
  Tier-1 violation (first-token bias on HIT).
