# Compute Facades — the decode-path tier-2 card pattern

**What it is:** parser → `safe_eval` → step-through bias at Gemma
decode. Zero VRAM, zero training, zero channel budget, stacks freely
with any other install. Cheapest path from a new domain to a measurable
Gemma capability lift. **Answer need not be integer** — Icd10RecallFacade
(2026-04-22) generalizes the pattern to arbitrary Gemma BPE token
sequences, including natural-language text.

**When it applies:** deterministic compute OR exact-lookup domains
where Gemma's training-data memorization is unreliable. Gemma's
"capability" on these is pattern-match; the facade turns it exact.

## Shipped instances (as of 2026-04-22)

| Facade | File | Domain | Result |
|---|---|---|---|
| `MultiStepReasoningFacade` (R46.2) | `multi_step.py` | NL infix arithmetic | 17/17 Gemma fixes (`a385893`) |
| `BaseConversionFacade` (R22c) | `base_conversion.py` | Hex/binary → decimal | 10/10 vs 7/10 baseline (`7db6eb9`) |
| `NumberTheoryFacade` (R53a) | `number_theory.py` | mod / GCD / LCM | 15/15 vs 8/15 (`69279d4`) |
| `NumericEncodeFacade` (F2) | `numeric_encode.py` | int → hex/binary/octal | 12/12 on chain corpus (`5ee61a5`) |
| `Icd10RecallFacade` (R60a + F1) | `icd10_recall.py` | ICD-10 code → diagnosis TEXT, 72,748-code DB | 26/30 vs 8/30 baseline, first tier-3 (`afc0220`) |
| `PlannerFacade` (R70a + F2) | `planner.py` | orchestrates 4+ specialists + 2-step chains | 20/20 route single, 12/12 route chain |

Plus the **auto-generated** family via `recursion.py` (commits
`3274659` / `5173745` — see `recursion.md`):
- Level-1 hand-written `FacadeSpec`: `factorial_auto.py`,
  `fibonacci_auto.py`, `combinations_auto.py`, `permutations_auto.py`,
  `power_auto.py`, `next_prime_auto.py`
- Level-2 `MetaFacade.from_oracle(fn_name, arity)`:
  `factorial_meta.py`, `combinations_meta.py`, `gcd_meta.py`,
  `lcm_meta.py`, `fibonacci_meta.py`

All 17 facades share the identical R46.2 skeleton (verified 2026-04-22):
- `_DIGIT_TO_GEMMA` map or arbitrary-text BPE tokens
- `DEFAULT_BOOST = 50.0`, `DEFAULT_MAX_TOKENS = 20-80`
- `install(gemma, tokenizer)` / `detach()` lifecycle
- `parse(prompt)` returns a domain spec (operands / literal / code)
- `evaluate(...)` returns the exact answer via `safe_eval` or direct op
- `_generate(prompt, bias_token_ids, boost, max_tokens)` — autoregressive
  decode with per-step bias

## Step-through bias mechanics + discipline

Canonical emit mechanism (R11 origin; refined 2026-04-22 R53a):

1. Tokenize the answer: `tok.encode(str(n))` returns
   `[BOS, ▁, d0, d1, ...]`. **Strip both BOS (id=2) AND leading `▁`
   (id=236743)**. Without the `▁` strip the first bias slot is wasted
   on a space — Gemma's natural `0` token after `"Answer: "` has
   logit ~57-66 and +50 boost on `▁` can't flip it (diagnostic
   `scripts/r53a_debug_probe.py`, commit `69279d4`).
2. Decode autoregressively. For each step with bias available, add
   `boost` to the expected token's logit before argmax.
3. **POST_BIAS_BUDGET=4**: after bias completes, Gemma often sticks in
   a same-digit loop (seen as "10000000..." artifact). Cap natural
   continuation at 4 more tokens then break. `_parse_int` also caps
   digit-runs at 12 chars to defeat residual loops.

**Scope**: the `▁`-strip + POST_BIAS_BUDGET discipline applies in
`number_theory.py`, `numeric_encode.py`, and all `recursion.py`-
generated facades. NOT yet backported to `multi_step.py` /
`base_conversion.py` — those work because their answer shapes don't
trigger the `0`-run pattern (shipped tests still 10-17/17). If a
future facade shows the bug, copy the fix.

For text-answer facades (`Icd10RecallFacade`): don't strip `▁` because
the diagnosis text starts with a capital letter (e.g. `▁Type`) that
IS a single merged BPE token including the leading space.

`boost=50.0` is the canonical value. Overrides overcome Gemma's
strongest natural priors on digits and capitalized-word starts. Lower
values (10-20) don't reliably flip when Gemma is confidently wrong.
ICD-10 code-echo retry uses `boost * 3.0 = 150.0` and in-context
answer injection as last resort (commit `8ba151d`).

## Adding a new domain — three paths in order of effort

### Level 0 (hours) — write a new facade by hand

Copy `multi_step.py` or `base_conversion.py`, replace `parse()` and
`evaluate()`. ~2-4 hours total including A/B corpus and ship.
(Historical path; rarely needed since Level-1 shipped.)

### Level 1 (minutes) — populate a `FacadeSpec` in `recursion.py`

```python
from calm.llm_computer.recursion import FacadeSpec, generate_facade

spec = FacadeSpec(
    name="Factorial",
    module_name="factorial_auto",
    description="Factorial (n!) via CALM safe_eval oracle.",
    parse_patterns=[
        r"factorial\s+of\s+(-?\d+)",
        r"(-?\d+)\s*!",
        r"what\s+is\s+(-?\d+)\s+factorial",
    ],
    eval_expr="factorial({a})",
    max_operand=20,
    operand_count=1,
    max_tokens=30,
)
validate_facade(spec, oracle_cases)   # CALM-gated
generate_facade(spec, overwrite=True)
```

See `.claude/spec/recursion.md` for the full Level-1 pipeline
(three CALM gates: oracle validate → ast.parse → live A/B).

### Level 2 (seconds) — `MetaFacade.from_oracle(fn_name, arity)`

```python
from calm.llm_computer.recursion import MetaFacade

spec = MetaFacade.from_oracle(
    fn_name="combinations", arity=2, max_operand=100,
    extra_patterns=[r"(-?\d+)\s+choose\s+(-?\d+)"],
)
# spec is a standard FacadeSpec; pipe through generate_facade as above
```

MetaFacade encodes the canonical 1-arg / 2-arg NL patterns
automatically. User supplies safe_eval function name + arity (+
optional overrides). 5 specs shipped this way on 2026-04-22 (commit
`5173745`, 4/15 → 15/15 lift).

## Candidate queue (Gemma-failure surfaces)

Shipped 2026-04-22: mod/GCD/LCM (NumberTheory), combinations,
permutations, power, next_prime (auto), factorial, fibonacci (auto),
int→hex/binary/octal (NumericEncode), ICD-10 text recall (Icd10Recall).

Remaining:
- **Days-between dates** — richer parser (ISO date extraction); uses
  `date_ops` backend
- **Compound interest / loan payments** — `financial_ops` backends
- **Unit conversion with non-trivial ratios** — temperature, currency,
  imperial/metric
- **Chemical formula / MW lookup** — text-answer, similar to ICD-10 shape
- **Drug interaction checker** — text-answer, hospital-vertical
  companion to ICD-10
- **Legal citation format** — text-answer, citation pattern validation

Per-domain cost at Level-2: **seconds** for spec synthesis + ~1 hour
for the CALM oracle test set + A/B corpus.

## Contrast with CardSlot installs (retrieval cards)

Compute facades sidestep almost every install-mechanism pitfall that
R22b encountered:

| Concern | CardSlot retrieval (R22) | Decode-path compute facade |
|---|---|---|
| VRAM | reserved channels + optional FP32 host | 0 |
| Install gates | 4 aligned (`write_margin`, `min_margin`, `preserve`, N-range) | 1 (bias fires iff parse + evaluate succeed) |
| Calibration | per-N margin sensitivity (R22f: 22.0→14.5 recal) | None |
| Training data | card must be trained | N/A |
| Failure modes | adapter regex bugs, margin calibration, preserve pinning | parser fails → fall back to natural Gemma (0 regression) |
| Per-domain cost | PT train (~30 min) + corpus + install tuning (days) | seconds (Level-2) to hours (Level-0) |
| Stacking | channel conflicts between cards | unlimited |

**Rule:** for any domain with a `safe_eval` oracle (or short
known-length text answer), ship a decode-path facade first. CardSlot
is for genuine **trained-recall** domains where no computable path
exists AND channel isolation is load-bearing.

**Exception refinement (2026-04-22):** tier-3 *text-recall* domains
(ICD-10 was the first instance) are decode-path-addressable when the
answer is short (≤80 Gemma BPE tokens) and looked up from a static
JSON DB. Full CardSlot with trained PT is only needed when the "key"
space is non-literal (e.g. NIAH-style retrieval under distractor prose,
like R22 MQAR).

## Related rules

- `recursion.md` — Level-1 `FacadeSpec` + Level-2 `MetaFacade`
- `augmentation_thesis.md` §"Tier-2 stacking" — strategic framing
- `capability_gain.md` — two-measurement A/B discipline
- `embed_intelligence.md` §"Step-through digit bias" — first-principles
  + `▁`-strip + POST_BIAS_BUDGET scope
- `Substrate.md` §"Card Installation" — three-install typology
- `delta_rule.md` §R22 install — contrast case (CardSlot with 4 gates)
- `commercial.md` — decode-path facades as the cheapest domain-coverage
  lever for commercial verticals
