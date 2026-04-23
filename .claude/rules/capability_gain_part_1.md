# Capability Gain — How to tell real substrate wins from format coercion

**Part 1**

The substrate thesis is that compiled programs + DBs installed into
Gemma produce **capability Gemma cannot produce alone.** Rounds 6-10
of this project's arc got this wrong — the "0/7 → 4/7 domain fix"
metric was measuring token FORMAT, not answer CORRECTNESS. Round 11
was the first round to demonstrate a real capability gain. This file
codifies the distinction so future rounds don't repeat the error.

## The canonical failure mode

**Bad metric**: argmax of `logits[0, -1]` immediately after the prompt.

This measures "would Gemma emit token X as the very next token?" —
which is a question about Gemma's **output format preferences**, not
its **knowledge**. A 5B LM like Gemma 4 E4B has strong formatting
habits: it likes to emit `**` (markdown bold) or `\n` or "Step 1:"
before getting to the answer. Its argmax-at-position-0 frequently
doesn't contain the final answer even when Gemma knows it perfectly.

Example from Round 10c:

    prompt: "what is 2 plus 3 equals"
    argmax immediately-next: "**"
    autoregressive continuation: "**\n\n**5**\n\nThe correct answer is 5."

Under the bad metric, Gemma "fails" — the immediate next token isn't
"5". Under the correct metric, Gemma gets it right; it just emits `**`
(markdown bold open) first, then the digit inside bold.

## The correct metric: continuation-parsed answer

**Generate N tokens** (N ≥ 30, often 60-180 for step-by-step models),
then parse the answer from the full continuation:

- Numeric answers: first integer in the continuation matching expected,
  OR any integer in the continuation matching expected (depending on
  how strictly you want to score)
- Boolean: "yes"/"no" case-insensitive substring match with priority
  to whichever appears first
- Free-form: domain-specific extraction

Budget the token count to cover Gemma's full "Step 1: Identify the
numbers..." pattern. Truncation masquerades as capability failure —
Round 10d's 40-token budget cut off 2-digit multiplication mid-answer;
180 tokens revealed Gemma actually solves most cases in continuation.

## Two measurements per round

Every round that claims a capability gain needs **both** (from
`workflow.md` §"Always check two things"):

| Measurement | Purpose |
|---|---|
| **Raw path** — compiled card standalone exhaustive test | Proves the card computes what we claim. `programs/multiplier.py` → 3390/3390 correct. |
| **User-facing path** — Gemma+card vs Gemma alone on continuation | Proves the card's output reaches the user's screen. Baseline parse vs facade parse. |

Only ship the round when **both** move:

- Raw-only win means the card works in isolation but isn't reaching
  Gemma's output. Not a substrate win.
- User-only "win" (no raw test) is almost always format coercion,
  measurement artifact, or noise.
- Both move → real capability gain.

## The failure-surface gate

Before building a facade for domain X, establish that Gemma actually
FAILS on X. A compiled card that "fixes" a task Gemma already solves
in continuation is format coercion, not capability.

Process:
1. Run Gemma baseline on 10-20 prompts in domain X with 180-token
   budget and continuation parsing.
2. Count real failures (not truncation, not parser issues).
3. If failure count < 3, skip domain X — pursue harder targets.

Observed Gemma 4 E4B failure surface (Round 10d, 180-token budget):

| Category | Pass | Substrate target? |
|---|---:|---|
| 1-digit + | 5/5 | No |
| 2-digit + | 5/5 | No |
| 3-digit + | 3/5 | Maybe (some failures look like truncation) |
| 1-digit × | 5/5 | No |
| 2-digit × | 2/4 | **Yes** — systematic off-by-10 errors |
| order-of-ops | 3/4 | No (failures are prompt format) |
| primality | 0/4 | Partial — prompt format issue |
| small factorial | 3/4 | No |
| word problems | 1/5 | Mostly prompt format |

Substrate's real value surface is narrower than "basic math." Don't
claim gains on other categories without a proper failure test.

## What a real capability gain looks like

Round 11 (the canonical demonstration, 2-digit multiplication):

    baseline Gemma:         5/10 (3 real arithmetic errors + 2 format)
    multiplier facade:      10/10
    improvement:            +5 (3 genuine + 2 format)
    multiplier standalone:  3390/3390 (raw path)

The three genuine fixes (17×23→401→391, 47×19→903→893, 45×15→705→675)
are the real wins. Gemma was arithmetically wrong; the compiled
multiplier was right; step-through digit bias delivered the correct
answer through Gemma's output. Both measurements moved.

**Independently confirmed by tracing (Rounds 17-19):** the 17×23
win is causally routed through specific weights — L23 H4's V-
projection (~2.6M params) is the load-bearing site for arithmetic
content in Gemma. Ablating H4's V alone drops correct-digit logit
by -9.51 on average across 10 arithmetic pairs. See
`.claude/MEMORY/atlas/tracing_arc_part_1.md` §"Gemma 4 E4B tracing findings."
This closes the loop: the capability gain isn't a prompt-engineering
trick, it's a measurable intervention on identifiable weights that
Gemma itself uses for the same task.

**Causal-validation chain (R28 → R42 → R43):** the forced-attention
intervention (replace learned softmax with one-hot at natural top
position) preserves behavior at three distinct (layer, capability)
pairs:

| Round | Layer / Facade | Capability | Measurement |
|---|---|---|---|
| R28 | L30 H4/H6 forced-attn | Arithmetic | mean \|Δ\|=0.407, argmax 9/10 |
| R42 | L23 H1/H4 forced-attn | SV agreement | mean \|Δ\|=0.467, argmax 8/10 |
| R43a | L23 H1/H4 forced-attn | Comparison | mean \|Δ\|=0.176, argmax 18/18 |
| R43b | L23 H1/H4 forced-attn | Counting | mean \|Δ\|=0.528, argmax 6/6 |
| R46.2 | `MultiStepReasoningFacade` (step-through digit bias, N-op) | Multi-step infix composition | **17/17 real Gemma fixes, 0 regressions** on held-out prompts (commit a385893) |

Same intervention template, spanning four layers, five capabilities,
~94% argmax preservation on the forced-attention rows + 17/17
user-facing delivery on R46.2. This is the template for "causally
validated compilable circuit" — when both the raw path (exhaustive
card verification) AND the user-facing path (forced-attention
preservation OR Gemma-output fix rate) land together, the circuit
is ready to ship as a Tier-2 compiled card.

## When you think you have a gain but you don't

Checklist — if any answer is yes, it's probably not real:

- [ ] Did you only measure "first token" argmax?
- [ ] Did Gemma's baseline continuation (60+ tokens) contain the correct answer?
- [ ] Was your token budget too small to cover Gemma's preferred format?
- [ ] Is the "failure" because Gemma emits `\n` / `**` / "Step 1:" first?
- [ ] Did the facade just force a shorter emission format rather than
      compute a different value?

If all answers are no, and the raw path is correct, and both
measurements moved, it's real. Commit with a before/after table per
workflow.md.

## The commercial implication

Capability gain = substrate adds something Gemma lacks.
Format coercion = substrate reformats what Gemma already does.

- **Capability gain product**: "Gemma + our compiled cards gives you
  correct arithmetic that stock Gemma gets wrong." Verifiable,
  defensible, measurable.
- **Format coercion product**: "Our cards make Gemma emit digits
  faster." Marginal, hard to defend vs prompt engineering.

Only the first is a moat. Don't overclaim by conflating them.

## Failure-surface gate as a hard precondition (R53.2 receipt)

The §"failure-surface gate" earlier in this file was a rule of thumb.
R53 made it load-bearing. Skipping it produces uninterpretable evals.

**Measured failure from violating the gate (R53.2, 12-problem simple
eval, `scripts/r53_eval_phase1.py`):**

- 6/12 problems: Gemma solves 100% stock and 100% hinted (ceiling)
- 2/12 problems: stock extraction failed but Gemma's actual code was
  correct (format coercion masked as capability gain)
- Only 4/12 had any usable signal
- Eval concluded "+2 facade wins" — all were extraction artifacts,
  not real gains

**Required procedure before building any RAG / augmentation eval:**

1. **Collect candidates** — aim for 100-200 from a standard benchmark
   (MBPP / HumanEvalPlus / BigCodeBench for code; equivalent for
   other domains). These come with test cases already.
2. **Score stock Gemma** — run all candidates through stock Gemma,
   score by running the extracted code against the bundled tests.
   Use a permissive extractor (code-fence OR bare `def ` OR AST parse
   of whole output). The extractor must NOT be the limiting factor.
3. **Partition** results:
   - `solves_cleanly` (>=80% tests pass) — ceiling, skip for eval
   - `fails_correctness` (tests run, fail) — target corpus ✓
   - `format_fails` (extract failed) — flag separately; these are
     extractor bugs, not capability failures
   - `partial` (20-80% pass) — interesting, include in target corpus
4. **Result**: a ~30-50-problem corpus where Gemma is known to have
   headroom. THIS is the Phase 1 eval corpus.
5. **Only then**: run the augmentation condition (hinted, substrate,
   etc). Any delta on this corpus is real signal.

### CoT-depth as a predictor of failure

Gemma-4-E4B reliability drops as reasoning-chain length grows:

| Task type | CoT depth | Measured Gemma reliability |
|---|---|---|
| Single-op (1+2, is_prime(17)) | 1 | ~99% |
| 2-digit × (17×23) | 2-3 | ~50-70% (R11 multiplier context) |
| (a*b)+c | 3-4 ops | fails often (R46 MultiStep fixed 17/17) |
| Multi-step with intermediate state (DP, simulation) | N | drops fast with N |
| Competitive programming 5+ steps | large | mostly fails (CodeContests data) |

When building the candidate pool, **bias toward multi-step problems**
(BigCodeBench's multi-library tasks, CodeContests, `bug_fix_pairs`-
style chains). Single-function vanilla-algorithm problems are at the
ceiling and waste budget.

### Extractor asymmetry (R53.2 specific gotcha)

If the extractor ONLY handles fenced code blocks and Gemma sometimes
emits bare code, stock-vs-hinted comparison is confounded — hinted
prompts show fenced examples that Gemma imitates (extracts cleanly);
stock emits bare code (extract fails, scored 0/0). Any apparent
"hinted wins" is format coercion, not capability gain.

**Fix**: the extractor must be format-agnostic — try fence, try `def`,
try `class`, try `import`, try whole-output AST parse. Return the
first AST-valid candidate. See `scripts/r53_eval_complex.py:extract_code`.

## Gemma ignores targeted hints (R53.19/R53.33 receipt)

The R53 structured-repair pipeline detects specific runtime errors,
classifies by regex, and emits a targeted repair hint with a concrete
rename example. Gemma's retry rewrites the code with the SAME bug.
This is not a prompt-engineering issue — it's a prior-dominance
failure mode that hint-tuning cannot fix.

**Concrete cases from R53.33** (historical framing — both cases
LIFTED by the R53.35 `ast_repair` walker, see §R53.35 below;
preserved here as the original receipt for "why hint-tuning
fails"):

