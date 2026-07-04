# Embed Intelligence — Historical receipts

Delivery-mechanism validation rounds, threshold-calibration arc,
discipline-scope expansion commits, first-token-hook ruled-out
measurement. Current rules: `.claude/rules/embed_intelligence.md`.
This file exists for archaeology — "which round validated which
mechanism", "where the thresholds came from", "why first-token bias
was ruled out for code".

## Delivery-mechanism validation history

- **Round 6**: `VerificationHook` without `min_margin` guard corrupts
  unrelated prompts (Paris/Berlin/Rome) because recall card with no
  match still has a default argmax that fires. Led to the mandatory
  `min_margin` rule. See current spec § "Avoiding regressions".

- **Round 9**: Token-embedding projection at position -1, late layer,
  proven functionally equivalent to `VerificationHook` for single-token
  answers. Both are head-level biases via different mathematical routes.

- **Round 10a**: Projection at early layers ruled out — layer 1 gives
  0/7 with regressions; layer 33 gives 4/7 clean. Established the
  "late layers (33-41) only" rule.

- **Round 10b**: Strength sweep 0.1 → 50×. Binary behavior: α < 1
  silent, α ≥ 1 fires cleanly. No upper break point observed.
  Established default `strength=1.0` matching Gemma's native
  token-embedding scale.

- **Round 10c**: Continuation after injection is noisy — Gemma pushed
  off-distribution. Established that mechanism is a late-layer head
  bias, not deep integration ("reason with injected context").

- **Round 11**: Step-through digit bias enabled multi-token verified
  answers. Baseline 5/10 → facade 10/10 on 2-digit multiplication.
  Three genuine arithmetic fixes (17×23, 47×19, 45×15). Canonical
  mechanism demonstration.

## R46.2 — step-through extension to N-op chains

`MultiStepReasoningFacade` (commit `a385893`) extends step-through
digit bias to N-op infix compositions. Parses NL infix (e.g. "2 + 3
× 5 - 7") with parens and mixed precedence, evaluates via `safe_eval`
to the final answer, then emits one step-through digit bias per
intermediate AND final value.

Result: **17/17 real Gemma fixes, 0 regressions** on held-out prompts.
Confirms step-through biasing generalizes from single-op to N-op
composition — right embed mechanism for any verifier that produces
a multi-token numeric answer.

## R22 install threshold-calibration arc (MQAR card)

MQAR card shipped 2026-04-21 at `min_margin=22.0` (+9/60, commit
`73df738`). R22f recalibrated 2026-04-22 to `14.5` (+18/60, 60/60
total, commit `9691e06`) after diagnosing flat N=10 cells as
over-gating.

Per-N margin distribution measured during R22f sweep:
- N=5: p50≈23.3
- N=10: p50=20.83, p5=15.21
- N=15: p50=18.63, p5=16.39

The 22.0 threshold was N=5-calibrated and over-gated N≥10 despite
standalone card being 100% correct (20/20 each). 14.5 sits below
the lowest observed p5 across all Ns and preserves zero-regression.

**Generalized rule** (migrated to current spec): tune per-card AND
per-input-distribution bucket, not to a fixed 0.5.

Full install arc: `MEMORY/atlas/delta_rule_arc.md` §"R22 install".

## `write_margin` == `min_margin` alignment (commit `e169d6d`)

`card_output_fn` independently writes to the residual stream; without
a margin gate the write happens even when hook is silent, shifting
Gemma's head projection. The alignment rule was established after
measuring that silent-hook + writing-function-fn still shifts
downstream Gemma behavior.

## `▁`-strip + POST_BIAS_BUDGET discipline origin (R53a, 2026-04-22)

R53a `NumberTheoryFacade` debugging (diagnostic
`scripts/r53a_debug_probe.py`, commit `69279d4`): the naïve
`strip_bos` path left `▁` in the bias chain for prompts ending in a
space (e.g. `"Answer: "`). Step-0 then biased a SPACE — Gemma's
natural `0` token at that position had logit ~57-66, and +50 boost
on `▁` couldn't flip it. Result: bias started one step late, and
the answer got "0"-prefixed gibberish ("01000...").

**Scope at origin** (2026-04-22): applied in `number_theory.py`,
`numeric_encode.py`, and all `recursion.py`-generated facades (via
shared `_TEMPLATE`). NOT backported to `multi_step.py` /
`base_conversion.py` — those worked without the fix because their
answer shapes didn't trigger the "0"-run pattern (17/17 and 10/10
shipped tests still passed).

Text-answer facade exception (`Icd10RecallFacade` R60a, commit
`afc0220`): do NOT strip `▁` because the diagnosis text begins with
a capital letter and Gemma's BPE often merges the leading space
into the first-word token (e.g. `▁Type` as a single token).

## Boost-tuning for stubborn priors

ICD-10 code-echo retry (commit `8ba151d`): uses `boost * 3.0 = 150.0`
plus in-context answer injection as last resort for codes where
Gemma's code-analysis format prior overwhelms step-through bias.

Remaining resistant codes (T44.6X4D, T40.5X4D, V80.22XA, W10.0XXA):
genuine tier-3 edge that needs different mechanism (prompt reshape
or pure-DB bypass). 4 ICD codes out of 72,748-code DB.

## R53.14 / R53.20a / R53.20b — FirstTokenHook ruled out for code

POST-SWA-fix re-run produced same -9.3pp regression as pre-fix on
R53.0 6-problem code corpus.

Install: `VerificationHook(vocab_mapping=PER_MARKER_TARGETS,
boost=50, min_margin=0.5)` where targets are `"def"/"class"` per-problem.

| | stock | prompt-RAG | substrate @ L41 |
|---|---:|---:|---:|
| log_level_counts | 6/6 | 6/6 | **0/0** |
| lru_cache_class | 9/9 | 9/9 | **0/0** |
| **TOTAL** | 25/27 | 25/27 | **10/12** (-9.3pp) |

**Root cause — install-mechanism, not SWA**: Gemma's first-token on
code is confidently a fence/whitespace opener (margin 6.8-9.2), so
`min_margin=0.5` never gates, hook always fires on HIT, forces
`def`/`class` → code-without-fence → extractor fails.

**Confidence-gated hooks also failed at this site**: the measurable
margin doesn't correlate with "Gemma is uncertain about format" —
Gemma is uniformly confident on format openers.

**Ruled out**: first-token bias for code. Correct tier-2: post-generation
AST walker (see `MEMORY/atlas/compute_facades_arc.md` — rename-facade
/ AST-walker tier-2 pattern).

Full ruled-out entry: `MEMORY/atlas/tracing_roadmap_part_1.md`
§"Substrate L41 install REGRESSES on code".

## Operational spec — delivery mechanisms

Card computation → Gemma vocab logits → emitted tokens. Install modes
(where computation lives) are in `MEMORY/atlas/Substrate_arc.md`
§"Card Installation".

| Mechanism | Where it acts | Scope |
|---|---|---|
| **VerificationHook** | Head logits, after softcapping | Bias one Gemma token id by +boost |
| **Token-embedding projection** | Residual at position -1, late layer | Add `token_embd[answer_id]` to residual |
| **Step-through digit bias** | Head logits, once per generation step | Bias next-expected token each decode step |

(1) and (2) are equivalent for single-token answers at late layers.
(3) generalizes to multi-token answers.

### VerificationHook

```python
hook = VerificationHook(card_slot,
                        vocab_mapping={card_vocab_id: gemma_token_id},
                        boost=50.0, min_margin=0.5)
gemma.verification_hooks.append(hook)
```

- Fires after head + softcapping; reads `card_slot.last_output[0, -1]`.
- `min_margin` gates on `(peak - median)` — mandatory for recall cards.
- Tune per-card AND per-input-distribution bucket (not fixed 0.5).
- `write_margin == min_margin` on paired `card_output_fn`.

### Token-embedding projection

Late layers (33-41) only. Strength binary: α < 1 silent, α ≥ 1 fires.
Scale injected embedding by `sqrt(d_model)`. Late-layer head bias, not
deep integration.

### Step-through digit bias

Required for multi-token answers. Strip BOS + leading `▁` for integer
facades; `POST_BIAS_BUDGET=4` after bias chain. Text-answer facades:
do NOT strip `▁`.

### Which mechanism to use

| Situation | Mechanism |
|---|---|
| Single Gemma BPE token | VerificationHook or token-embd projection |
| Multi-digit arithmetic | Step-through digit bias |
| Multi-word / code | Step-through token bias |
| Recall card with unmatched key | `min_margin` gate always |

### Avoiding regressions

1. **Parse-state flag** in CardSlot writer — zero `card_out` in-place
   when parse fails.
2. **`min_margin` on VerificationHook** — silences low-confidence fires.

### Ruled out

First-token bias for code — Gemma's format openers are uniformly
confident; use post-generation AST walker instead. Receipt:
§"R53.14 / R53.20a / R53.20b" above.

## Cross-refs

- Current rules: `.claude/rules/embed_intelligence.md` (stub; detail in this file)
- Retrieval-card install: `.claude/rules/delta_rule.md` + `MEMORY/atlas/delta_rule_arc.md`
- Decode-path facades using step-through bias: `.claude/rules/compute_facades.md` + `MEMORY/atlas/compute_facades_arc.md`
- Tier-2 AST walker (ruled-out-FirstTokenHook replacement):
  `MEMORY/atlas/capability_gain_arc.md` §"R53.35"
