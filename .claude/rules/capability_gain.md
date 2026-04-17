# Capability Gain — How to tell real substrate wins from format coercion

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
`.claude/rules/tracing_roadmap.md` §"Gemma 4 E4B tracing findings."
This closes the loop: the capability gain isn't a prompt-engineering
trick, it's a measurable intervention on identifiable weights that
Gemma itself uses for the same task.

**Causal-validation chain (R28 → R42 → R43):** the forced-attention
intervention (replace learned softmax with one-hot at natural top
position) preserves behavior at three distinct (layer, capability)
pairs:

| Round | Layer | Capability | mean \|Δ\| | argmax match |
|---|---|---|---|---|
| R28 | L30 H4/H6 | Arithmetic | 0.407 | 9/10 |
| R42 | L23 H1/H4 | SV agreement | 0.467 | 8/10 |
| R43a | L23 H1/H4 | Comparison | 0.176 | 18/18 |
| R43b | L23 H1/H4 | Counting | 0.528 | 6/6 |

Same intervention, three layers, four capabilities, ~94% argmax
preservation. This is the template for "causally validated
compilable circuit" — when both the raw path (exhaustive card
verification) AND the user-facing path (forced-attention
preservation) land together, the circuit is ready to ship as a
Tier-2 compiled card.

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

## Related rules

- `workflow.md` — the general hypothesis-test loop
- `Substrate.md` — install modes
- `embed_intelligence.md` — delivery path from card to Gemma's tokens
- `tracing_intelligence.md` — first-principles bound on what's compilable
