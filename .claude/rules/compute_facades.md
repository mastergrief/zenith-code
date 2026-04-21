# Compute Facades — the decode-path tier-2 card pattern

**What it is:** parser → `safe_eval` → step-through digit bias at
Gemma decode. Zero VRAM, zero training, zero channel budget, stacks
freely with any other install. Cheapest path from a new domain to a
measurable Gemma capability lift.

**When it applies:** deterministic compute domains where Gemma fails
on non-memorized cases (large-operand arithmetic, non-trivial
base conversion, GCD/LCM of multi-digit ints, days-between, compound
interest, unit conversion). Gemma's "capability" on these is
pattern-match on training data; exact compute moves it to 100%.

## Two proven instances (as of 2026-04-21)

| Facade | File | Domain | Result |
|---|---|---|---|
| `MultiStepReasoningFacade` (R46.2) | `calm/llm_computer/facades/multi_step.py` | NL infix arithmetic (+ - * / % **) | 17/17 real Gemma fixes, 0 regressions (commit `a385893`) |
| `BaseConversionFacade` (R22c) | `calm/llm_computer/facades/base_conversion.py` | Hex/binary → decimal | 10/10 vs baseline 7/10 (+3, 30% lift, 0 regressions, commit `7db6eb9`) |

Both share the identical skeleton (verified agent audit 2026-04-21):
- `_DIGIT_TO_GEMMA` map (byte-identical: digits 0-9 → Gemma BPE ids
  236771, 236770, 236778, 236800, 236812, 236810, 236825, 236832,
  236828, 236819)
- `DEFAULT_BOOST = 50.0`, `DEFAULT_MAX_TOKENS = 40-60`
- `install(gemma, tokenizer)` / `detach()` lifecycle
- `parse(prompt)` returns normalized expression / literal
- `evaluate(expression)` returns integer via `safe_eval` (R46.2) or
  `int(literal, base)` (R22c)
- `_generate(prompt, digit_token_ids, boost, max_tokens)` — prefill
  + autoregressive loop with per-step bias on next expected digit

**Differences are domain policy, not skeleton:**
- R22c's `"Answer: "` prompt-suffix trick when prompt ends with `?`
  (so first decode token is the leading digit, bias fires at step 0)
- R22c's `in|as|to decimal` gate (`parse()` early-return if the
  prompt doesn't signal "convert to decimal")

## Recipe — adding a new domain

Per-domain cost: **~2-4 hours once the ComputeFacade base class
lands** (2-4 hours for parser + evaluate + test corpus; zero
training, zero install engineering, zero channel budget).

1. Copy `multi_step.py` or `base_conversion.py` to a new file under
   `calm/llm_computer/facades/<domain>.py`.
2. Replace `parse()` with your domain's NL-to-expression extractor
   (regex or AST walker).
3. Replace `evaluate()` with the exact-compute function — prefer
   `safe_eval` (already registers 1002 CALM functions); fall back
   to a direct Python op if the answer is a literal conversion.
4. Test corpus: ~10 probes, half easy (Gemma likely right), half
   hard (non-memorized). Two-measurement A/B: baseline
   (`use_bias=False`) vs facade (`use_bias=True`). Accept ≥ 20%
   relative lift with 0 regressions on the hard half.
5. Commit with before/after table. No rules update needed unless
   the pattern evolves (new output encoding, non-digit answers).

Candidate queue (Gemma-failure surfaces, ~1 day each):
- **Modular arithmetic** — `127 mod 13 = 10` (Gemma often wrong)
- **GCD / LCM of multi-digit ints** — `gcd(48, 180) = 12`
- **Days-between dates** — `days_between(2024-03-15, 2024-07-22)`
- **Compound interest / loan payments** — financial_ops backends
- **Unit conversion with non-trivial ratios** — temperature,
  currency, imperial/metric

## Contrast with CardSlot installs (retrieval cards)

Compute facades sidestep almost every install-mechanism pitfall
that R22b encountered:

| Concern | CardSlot retrieval (R22) | Decode-path compute facade |
|---|---|---|
| VRAM | reserved channels + FP32 host possible | 0 |
| Install gates | 4 aligned (write_margin, min_margin, preserve, N-range) | 1 (bias fires if expression parses + evaluates) |
| Training data | needs noise augmentation if adapter format differs | N/A (no learned card) |
| Failure modes | adapter query-regex bug (R22e), margin calibration, preserve=True pinning | parser fails → fall back to Gemma natural (zero regression) |
| Per-domain cost | PT train (~30 min) + corpus + install tuning (days) | ~2-4 hours |
| Stacking | channel conflicts between cards | unlimited — facades don't share state |

**Rule (2026-04-21, session-level product call):** for deterministic
compute domains, ship a decode-path facade first. Reach for CardSlot
only when the domain is genuinely retrieval (key→value lookup with
no computable path) OR needs channel isolation for chained cards.

## Step-through digit-bias mechanics

The emit mechanism all compute facades share, lifted from R11 (real
Gemma multiplication fix, commit predates this file). See
`embed_intelligence.md §"Step-through digit bias (Round 11)"` for
first-principles derivation. Summary:

1. Tokenize the integer answer: `tok.encode(str(n))[1:]` (skip BOS).
   Gemma BPE tokenizes each digit as `▁d` or just `d` depending on
   position — multi-digit answers are 3-5 Gemma tokens.
2. Decode autoregressively. At each step where `digit_idx <
   len(digit_token_ids)`, add `boost` to the logit of the next
   expected digit BEFORE argmax. Gemma's natural prior on the digit
   has margin < 50, so the bias reliably flips the output.
3. After `digit_idx >= len(digit_token_ids)`, stop biasing.
   Gemma's natural decode continues (may emit units, trailing
   prose).

`boost=50.0` is the canonical value across all shipped facades —
overrides Gemma's strongest natural digit priors. Lower values
(10-20) don't reliably flip when Gemma is confidently wrong.

## Related rules

- `augmentation_thesis.md` §"Tier-2 stacking achieves tier-3-
  equivalent outcomes" — decode-path facades are the cleanest
  instantiation
- `capability_gain.md` — two-measurement discipline for each new
  facade
- `embed_intelligence.md` §"Step-through digit bias" — the emit
  mechanism in first-principles form
- `Substrate.md` §"Card Installation" — three-install typology
  (decode-path / CardSlot / in-tensor)
- `delta_rule.md` §R22 install — contrast case (retrieval card
  install with 4 gates, not a compute facade)
- `commercial.md` — decode-path facades as the cheapest domain-
  coverage lever for commercial verticals
