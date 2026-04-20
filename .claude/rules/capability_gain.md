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

- **token_bucket_rate_limiter** (`'int' object is not callable`):
  categorizer emits `"You're calling an int value as if it were a
  function. A method/function name was overwritten by an int value
  (e.g. self.consume = capacity shadows method consume). Rename the
  int attribute (e.g. self.tokens = capacity) and use the new name
  everywhere you assigned the value."` Gemma retry emits the same
  `self.consume = capacity` shadow. 2344s wall time on retry, 0/0.
  **Post-R53.35**: `shadow_rename` walker rewrites the shadow in
  ~0.9s without retry, 0/0 → 5/5 (commit `8cc2ff4`).

- **csv_column_stats** (runtime `KeyError: 'score'`): Gemma writes
  code accessing a dict key it never constructed with that name.
  Categorizer emits targeted hint; retry emits the same KeyError
  pattern at a different access site. **Post-R53.35**: in practice
  Gemma's dominant failure on csv turned out to be a single missing
  `)` (SyntaxError), not a KeyError — the `syntax_repair` walker
  lifts csv 0/0 → 8/8 in 0.9s without retry (commit `c81feb6`). The
  dict-key-synonym rewrite remains available for the KeyError
  branch if it recurs.

**Mechanism**: the prior over rate-limiter / csv-parser implementations
is learned from millions of training-data examples (including
tutorials where the bug IS the teaching example). Hint signal is ~200
tokens in-context vs ~1M training instances baked into weights.
Attention cannot reliably amplify the hint enough to flip the prior
at the specific emission site. Token-sequence momentum from the
opening `class TokenBucket:` cascades deterministically into the
shadow.

**Generalized TypeError categorizer**: regex matches any shadowed
type, not just 'int':

```python
re.search(r"TypeError: '(\w+)' object is not callable", output)
```

Emits type-specific hint (float/str/list/dict/int). Correct detection
100% of the time; Gemma's retry success rate still ~0%. **Detection
and repair-signal are solved; the block is Gemma's
instruction-following on code-repair prompts.**

**Implication**: hint-tuning + retries will not lift this ceiling.
The correct tier-2 intervention is a compiled **AST walker card**
that parses Gemma's output, detects the shadow / missing-key /
off-by-one / unused-var patterns, and mechanically rewrites — no
Gemma in the repair loop. See `augmentation_thesis.md` §"R53.14/
20a/20b" for the tier-2-stacking framing.

### R53.35 — AST walker shipped, hypothesis confirmed on shadow rename

Built `calm/llm_computer/facades/ast_repair.py` — two deterministic
rewrites driven by runtime error text (not by spec or Gemma retry):

- **Shadow rename** (TypeError: 'X' object is not callable): find
  `self.<name> = ...` assignments where `<name>` is also a method
  on the same class; rename attribute to `_<name>`, rewrite all
  non-call read sites, preserve method body.
- **Dict-key synonym** (KeyError: 'X'): curated synonym table
  (`avg` → `mean`, `std` → `stdev`, etc); rewrites Dict literals,
  Subscript access, and `.get/.pop/.setdefault` args.

Wired into `scripts/r53_21_import_inject.py` — runs after import
injection, before LLM structured repair. Iterated up to 4 passes
(csv may chain `mean` → `stdev` → `min` → `max`). Reverts on
regression.

Measurement (two paths, both moved):

  path                                              before   after
  ---------------------------------------------     ------   ------
  Raw: pytest calm/llm_computer/tests/
       test_ast_repair.py                            n/a     21/21
  User-facing: token_bucket_rate_limiter (R53.0)     0/0     5/5
  No-regression: lru_cache_class (R53.0)             9/9     9/9

Wall time on the lift: 0.9s (AST walker) vs 117-300s per Gemma
retry round. Zero inference cost, strict improvement.

**Confirms the hypothesis** from R53.19/R53.33 receipt above: no
Gemma in the repair loop, mechanical rewrite, and the ceiling lifts.
Commercial framing ("auditable rewrite for regulated industries")
is load-bearing now, not aspirational.

**csv update — reaudit confirmed walker-fixable** (R53.35 reaudit,
`scripts/r53_35_reaudit.py`, `scripts/r53_diag_csv_raw.py`):

The initial R53.35 csv run reported `NoCode` because Gemma emitted
code inside a fenced block with a SyntaxError — a single unclosed
paren (`for i in range(min(num_cols, len(row)):` missing `)`
before the `:`). The format-agnostic extractor's final AST-validate
step correctly rejected 1742 tokens of otherwise-correct code.
Initial diagnosis as "extraction bottleneck / NoCode" conflated the
fence presence with parse validity.

Walker's third rewrite — **syntax_repair** — closes this:

  phase                                            result
  ----------------------------------------         ------
  Gemma raw (1742 tokens, parse)                   SyntaxError L42
  + syntax_repair (1 mismatch fix)                 OK
  exec + test_code                                 8/8 PASS

Measured end-to-end:

  pipeline step              csv_column_stats   token_bucket
  -------------              ----------------   ------------
  pre-walker                 0/0                0/0
  + ast_repair               **8/8** (syntax)   **5/5** (shadow)
  wall time                  ~0.9s              ~0.9s

Combined R53.0 lift: **+13 tests across 2 of 6 problems**,
mechanically, zero LLM retries. Receipt file:
`/tmp/r53_reaudit/csv_column_stats.txt`. Commit: `c81feb6`.

**The re-audit receipt redefines what "Gemma failed" means**: on
both token_bucket and csv, Gemma produced correct logic with a
single-character mechanical bug that the extractor's strict
AST-validate hid. Prior rules should be read with that
refinement — not all "Gemma failed on X" conclusions were
capability gaps; some were extractor-strictness artifacts.

### R53.36 — tier-3 install audit (R51/R52 revisit)

Question: are the R51.5 (MSE) and R52.3 (KL) tier-3 nulls the
same class of extractor-hidden artifact the csv reaudit revealed?

Audit (`scripts/r53_36_audit_r51_install.py`): three diagnostic
questions × 4 held-out prompts (multi/single/factual/code) × 2
student checkpoints.

**Q1 — training fidelity** (does the student reproduce L24?):

  prompt        R51-MSE cos  R51-MSE scale  R52-KL cos  R52-KL scale
  -----------   -----------  -------------  ----------  -------------
  multi-step    0.944        0.955          -0.020      91.68×
  single-op     0.962        0.962          -0.021      94.45×
  factual       0.954        0.944          -0.020      98.47×
  code          0.714        0.760          -0.030      91.12×
  -----------   -----------  -------------  ----------  -------------
  aggregate     **0.8935**   **0.9052**     **-0.0227** **93.93×**

**Q2 — install boundary** (`L24_installed == h_before +
student(h_before)`?): **max abs diff = mean abs diff = 0.00e+00**
on all 4 prompts × 2 students. **Install math is bit-identical.**
Not an install bug.

**Interpretation**:

- **R52-KL is a wrong-loss training failure.** Cos=-0.02 means
  the student's output is uncorrelated with L24's contribution;
  scale=94× means it's ~100× too big in magnitude. KL-on-final-
  logits never constrains residual reconstruction. The student
  learned to output something that makes L25..L41+head produce
  roughly-right logits via alternate pathways, without computing
  L24's function.
- **R51-MSE is NOT a csv-style artifact but is a close-miss.**
  Cos=0.89, scale=0.91 means the student DOES reproduce L24 on
  average. Yet R51.5 dual-gate reported 0.19 prefix match. The
  10% residual error is diffuse in channel basis but cascades
  through 17 downstream layers + head, amplifying into wrong
  argmax. MSE loss averages over 2560 channels — can't concentrate
  penalty on task-critical directions (digit-selectors, content-
  readers). That's why R51.5 noted arithmetic (sharp digits)
  preserves worst (0.11) and code (diffuse) preserves best (0.59).

**Implication**: tier-3 L24 distillation is closed **at current
loss space** but not in principle. A loss that weights by
downstream causal effect — e.g. `||J · (pred - contribution)||²`
where `J = d(head_logits) / d(h_L24)` — would concentrate
student training on task-critical directions. Speculative but
credible reopen path. Estimated ~1-2 weeks of work; commercial
lift is moderate since tier-2 stacking (R46.2 MultiStepReasoningFacade
17/17 real fixes) already augments L24's task at the output
level without tier-3 cost.

**Tier-3 is not reopened as an active workstream.** Tier-2
stacking remains the priority per `augmentation_thesis.md`
§"Tier-2 stacking achieves tier-3-equivalent outcomes". R53.36
refined *why* tier-3 was hard (sharp-direction miss + wrong-loss),
not *whether* tier-2 is correct.

## Related rules

- `workflow.md` — the general hypothesis-test loop
- `Substrate.md` — install modes
- `embed_intelligence.md` — delivery path from card to Gemma's tokens
- `tracing_intelligence.md` — first-principles bound on what's compilable
- `retrieval.md` — hybrid retrieval used by augmentation paths
- `augmentation_thesis.md` §"Automatic Tier-1 preservation" — why
  blanket augmentation fails even on passing extractor
