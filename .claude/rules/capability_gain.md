---
paths:
  - "calm/llm_computer/**"
  - "scripts/*facade*.py"
  - "scripts/*projection*.py"
  - "scripts/*install*.py"
  - "scripts/*gate*.py"
  - "scripts/*measure*.py"
  - "scripts/*number_theory*.py"
---

# Capability Gain — telling real substrate wins from format coercion

The substrate thesis: compiled programs + DBs installed into Gemma
produce **capability Gemma cannot produce alone**. This file codifies
how to measure that — what counts, what doesn't, and what discipline
prevents claiming a win that's really a measurement artifact.

> Historical receipts (per-round measurements, R-arc detail, AST-walker
> case studies, CoT-depth survey, session receipts): see
> `MEMORY/atlas/capability_gain_arc.md`.

## The canonical failure mode — first-token argmax

**Bad metric**: argmax of `logits[0, -1]` immediately after the prompt.

This measures Gemma's **output format preferences**, not its
**knowledge**. A 5B LM has strong formatting habits — it likes to
emit `**` (markdown bold) or `\n` or "Step 1:" before getting to the
answer. The argmax-at-position-0 frequently doesn't contain the
answer even when Gemma knows it perfectly.

## The correct metric — continuation-parsed answer

**Generate N tokens** (N ≥ 30, often 60-180 for step-by-step models),
then parse the answer from the full continuation:

- Numeric: first integer in continuation matching expected (or any
  integer matching, depending on strictness).
- Boolean: case-insensitive substring match for "yes" / "no", priority
  to whichever appears first.
- Free-form: domain-specific extraction.

Budget the token count to cover Gemma's full preferred output format.
Truncation masquerades as capability failure.

## Two measurements every round

Every claimed capability gain needs **both** (cross-ref `workflow.md`):

| Measurement | Purpose |
|---|---|
| **Raw path** — compiled card standalone exhaustive test | Proves the card computes what we claim. |
| **User-facing path** — Gemma+card vs Gemma alone on continuation | Proves the card's output reaches the user's screen. |

Ship only when **both move**:
- Raw-only win means card works in isolation but isn't reaching
  Gemma's output. Not a substrate win.
- User-only "win" with no raw test is almost always format coercion,
  measurement artifact, or noise.

## Failure-surface gate (hard precondition)

Before building a facade for domain X, establish that Gemma actually
FAILS on X. A compiled card that "fixes" a task Gemma already solves
in continuation is format coercion, not capability.

**Procedure** (mandatory before any RAG / augmentation eval):

1. **Collect candidates** — 100-200 from a standard benchmark (MBPP,
   HumanEvalPlus, BigCodeBench for code; equivalent for other
   domains). Need bundled tests.
2. **Score stock Gemma** — run all candidates, score by running
   extracted code against bundled tests. Use a permissive extractor
   (code-fence OR bare `def` OR AST parse of whole output). The
   extractor must NOT be the limiting factor.
3. **Partition**:
   - `solves_cleanly` (≥80% tests pass) — ceiling, skip for eval
   - `fails_correctness` (tests run, fail) — target corpus ✓
   - `format_fails` (extract failed) — flag separately; extractor
     bugs, not capability failures
   - `partial` (20-80% pass) — interesting, include in target corpus
4. Result: a 30-50-problem corpus where Gemma is known to have
   headroom. THIS is the eval corpus.
5. ONLY then run the augmentation condition. Any delta is real signal.

If failure count on a domain < 3, skip — pursue harder targets.

## CoT-depth predicts failure

Gemma-4-E4B reliability drops as reasoning-chain length grows:

| Task type | CoT depth | Reliability |
|---|---|---|
| Single-op (`is_prime(17)`) | 1 | ~99% |
| 2-digit × | 2-3 | ~50-70% |
| `(a*b)+c` | 3-4 ops | fails often |
| Multi-step with intermediate state | N | drops fast with N |
| Competitive programming 5+ steps | large | mostly fails |

Bias the candidate pool toward multi-step (BigCodeBench multi-library
tasks, CodeContests, bug-fix chains). Single-function vanilla-algorithm
problems are at the ceiling and waste budget.

## Extractor asymmetry (gotcha)

If the extractor only handles fenced code blocks but Gemma sometimes
emits bare code, stock-vs-hinted comparison is confounded — hinted
prompts show fenced examples Gemma imitates (extracts cleanly); stock
emits bare (extract fails, scored 0/0). Apparent "hinted wins" is
format coercion.

**Fix**: extractor must be format-agnostic — try fence, try `def`, try
`class`, try `import`, try whole-output AST parse. Return first
AST-valid candidate.

## Gemma ignores targeted hints

The structured-repair pipeline that detects specific runtime errors
and emits a targeted repair hint with concrete examples does NOT lift
the ceiling on hint-blind failure modes. Gemma's retry rewrites the
code with the same bug.

**Mechanism**: prior over rate-limiter / csv-parser implementations is
learned from millions of training-data examples (including tutorials
where the bug IS the teaching example). Hint signal is ~200 tokens
in-context vs ~1M training instances baked into weights. Attention
cannot reliably amplify the hint enough to flip the prior at the
specific emission site.

**Detection and repair-signal are solved; the block is Gemma's
instruction-following on code-repair prompts.**

**Correct intervention**: compiled AST walker that parses Gemma's
output, detects shadow / missing-key / off-by-one / unused-var
patterns, mechanically rewrites — **no Gemma in the repair loop**.
Wall time ~0.9s (walker) vs 117-300s per Gemma retry round.
See `compute_facades.md` and `augmentation_thesis.md` §"Tier-2 stacking".

## Sanity checklist — when you think you have a gain but you don't

If any answer is yes, it's probably not real:

- [ ] Did you only measure "first token" argmax?
- [ ] Did Gemma's baseline continuation (60+ tokens) contain the correct answer?
- [ ] Was the token budget too small to cover Gemma's preferred format?
- [ ] Is the "failure" because Gemma emits `\n` / `**` / "Step 1:" first?
- [ ] Did the facade just force a shorter emission format rather than
      compute a different value?

If all answers are no, raw path is correct, and both measurements
moved → real. Commit with a before/after table per `workflow.md`.

## Causal-validation template (for new compilable circuits)

The forced-attention intervention (replace learned softmax with
one-hot at natural top position) is the established template for
"causally validated compilable circuit". When BOTH the raw path
(exhaustive card verification) AND the user-facing path
(forced-attention preservation OR Gemma-output fix rate) land
together, the circuit is ready to ship as a Tier-2 compiled card.

Workflow:
1. Per-head ablation localizes the load-bearing site.
2. Q/K/V decomposition isolates content-carrier (usually V).
3. Forced-attention at the localized head preserves capability →
   compilable.
4. Compile to facade → measure user-facing fix rate → ship.

See `tracing_intelligence.md` for the methodology framework and
`probing_methodology.md` for the per-tool gates.

## Commercial implication

- **Capability gain product**: "Gemma + our compiled cards gives you
  correct arithmetic/recall/composition that stock Gemma gets wrong."
  Verifiable, defensible, measurable.
- **Format coercion product**: "Our cards make Gemma emit digits
  faster." Marginal, hard to defend vs prompt engineering.

Only the first is a moat. Don't conflate.

## Related rules

- `workflow.md` — general hypothesis-test loop + two measurements
- `Substrate.md` — install modes (CardSlot / in-attention / decode-path)
- `compute_facades.md` — decode-path facade pattern (cheapest tier-2)
- `embed_intelligence.md` — card → Gemma token delivery
- `tracing_intelligence.md` — first-principles compilability
- `augmentation_thesis.md` — tier-1/2/3 framework
- `MEMORY/atlas/capability_gain_arc.md` — full per-round receipts
