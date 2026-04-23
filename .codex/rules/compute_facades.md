# Compute Facades — the decode-path tier-2 card pattern

**What it is**: parser → `safe_eval` → step-through bias at Gemma decode.
Zero VRAM, zero training, zero channel budget, stacks freely with any
other install. Cheapest path from a new domain to a measurable Gemma
capability lift. **Answer need not be integer** — pattern generalizes
to arbitrary Gemma BPE token sequences, including natural-language text.

**When it applies**: deterministic compute OR exact-lookup domains
where Gemma's training-data memorization is unreliable. Gemma's
"capability" on these is pattern-match; the facade turns it exact.

> Historical receipts (shipped-facade table with commits + dates,
> per-round provenance, scope-expansion chronology): see
> `MEMORY/atlas/compute_facades_arc.md`.

## Canonical skeleton

All decode-path facades share the same shape:

- `_DIGIT_TO_GEMMA` map (for integer answers) OR arbitrary-text BPE
  tokens (for text answers)
- `DEFAULT_BOOST = 50.0`, `DEFAULT_MAX_TOKENS = 20-80`
- `install(gemma, tokenizer)` / `detach()` lifecycle
- `parse(prompt)` returns a domain spec (operands / literal / code)
- `evaluate(...)` returns the exact answer via `safe_eval` or direct op
- `_generate(prompt, bias_token_ids, boost, max_tokens)` — autoregressive
  decode with per-step bias

## Step-through bias mechanics

Canonical emit mechanism:

1. Tokenize the answer: `tok.encode(str(n))` returns
   `[BOS, ▁, d0, d1, ...]`. **Strip both BOS (id=2) AND leading `▁`
   (id=236743)**. Without the `▁` strip the first bias slot is wasted
   on a space — Gemma's natural `0` token after `"Answer: "` has logit
   ~57-66 and +50 boost on `▁` can't flip it.
2. Decode autoregressively. For each step with bias available, add
   `boost` to the expected token's logit before argmax.
3. **POST_BIAS_BUDGET=4**: after bias completes, Gemma often sticks in
   a same-digit loop (long runs of the same digit pasted after the
   real answer). Cap natural continuation at 4 more tokens then break.
   `_parse_int` also caps digit-runs at 12 chars to defeat residual loops.

**Scope of the `▁`-strip + POST_BIAS_BUDGET discipline**: applies to
integer-answer facades (`number_theory`, `numeric_encode`, all
`recursion.py`-generated facades). NOT applied to facades whose answer
shapes don't trigger the `0`-run pattern (tests still pass) —
backport only if a new facade shows the bug.

**For text-answer facades**: do NOT strip `▁` — diagnosis text / names
starting with capitals (`▁Type`) merge into a single BPE token
including the leading space.

**`boost=50.0` is canonical.** Overrides overcome Gemma's strongest
natural priors on digits and capitalized-word starts. Lower values
(10-20) don't reliably flip when Gemma is confidently wrong. For
stubborn cases (code-echo retry etc.), `boost * 3.0 = 150.0` + in-context
answer injection as last resort.

## Adding a new domain — three paths

### Level 0 (hours) — hand-written facade

Copy an existing facade, replace `parse()` and `evaluate()`. ~2-4
hours including A/B corpus and ship. Rarely needed since Level-1.

### Level 1 (minutes) — populate a `FacadeSpec`

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

See `recursion.md` for the full pipeline (three CALM gates: oracle
validate → `ast.parse` → live A/B).

### Level 2 (seconds) — `MetaFacade.from_oracle(fn_name, arity)`

```python
from calm.llm_computer.recursion import MetaFacade

spec = MetaFacade.from_oracle(
    fn_name="combinations", arity=2, max_operand=100,
    extra_patterns=[r"(-?\d+)\s+choose\s+(-?\d+)"],
)
# spec is a standard FacadeSpec; pipe through generate_facade as above
```

MetaFacade encodes canonical 1-arg / 2-arg NL patterns automatically.
User supplies safe_eval function name + arity (+ optional overrides).

## Candidate queue (remaining Gemma-failure surfaces)

- **Days-between dates** — richer parser (ISO date extraction); uses
  `date_ops` backend
- **Compound interest / loan payments** — `financial_ops` backends
- **Unit conversion with non-trivial ratios** — temperature, currency,
  imperial/metric
- **Chemical formula / MW lookup** — text-answer, similar shape to ICD
- **Drug interaction checker** — text-answer, hospital-vertical
- **Legal citation format** — text-answer, citation pattern validation

Per-domain cost at Level-2: seconds for spec synthesis + ~1 hour for
CALM oracle test set + A/B corpus.

## Decode-path vs CardSlot — decision rule

| Concern | CardSlot retrieval | Decode-path compute facade |
|---|---|---|
| VRAM | reserved channels + optional FP32 host | 0 |
| Install gates | 4 aligned (write_margin, min_margin, preserve, N-range) | 1 (bias fires iff parse + evaluate succeed) |
| Calibration | per-distribution margin sensitivity | None |
| Training data | card must be trained | N/A |
| Failure modes | adapter regex bugs, margin calibration, preserve pinning | parser fails → fall back to natural Gemma (0 regression) |
| Per-domain cost | PT train + corpus + install tuning (days) | seconds (Level-2) to hours (Level-0) |
| Stacking | channel conflicts between cards | unlimited |

**Rule**: for any domain with a `safe_eval` oracle (or short
known-length text answer), ship a decode-path facade first. CardSlot
is for genuine **trained-recall** domains where no computable path
exists AND channel isolation is load-bearing.

**Tier-3 refinement**: tier-3 *text-recall* domains are decode-path-
addressable when the answer is short (≤80 Gemma BPE tokens) and
looked up from a static JSON DB. Full CardSlot with trained PT is
only needed when the "key" space is non-literal (NIAH-style retrieval
under distractor prose).

## Related rules

- `recursion.md` — Level-1 `FacadeSpec` + Level-2 `MetaFacade` pipelines
- `augmentation_thesis.md` §"Tier-2 stacking" — strategic framing
- `capability_gain.md` — two-measurement A/B discipline
- `embed_intelligence.md` — step-through bias first-principles
- `Substrate.md` §"Card Installation" — three-install typology
- `delta_rule.md` — retrieval-card install contrast (CardSlot with 4 gates)
- `commercial.md` — decode-path facades as cheapest domain-coverage lever
- `MEMORY/atlas/compute_facades_arc.md` — shipped-facade receipts
