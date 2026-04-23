## **Part 2**

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

## MQAR data-scaling rule (R-delta arc, 2026-04-21)

Canonical receipt of "architecture changes don't substitute for
data" at substrate scale. PT+Delta MQAR benchmark — 4 architectural
null rounds before identifying the data lever:

- `6617a48` R11a d_model 64→128, R11b d_head 2→16 — null on MQAR
  ceiling at 500/N × 40 ep; misread as capacity limit.
- `78b5dfb` R18 multi-head H=4 — null (-6pp vs plain PT); per-head
  state (16, 16) below D/log(D) capacity for N=15.
- `65fb148` R19 D5 refinement n_iters=2 — null on MQAR; ARC's
  "refinement is the win" finding is grid-reasoning-specific,
  doesn't transfer to single-token retrieval.
- `7110990` + `49c13d7` R13/R14-b — **data scaling** at 2K/5K/10K
  per N solves N=5-20 cleanly. Plain PT gap: +21pp (N=5), +66pp
  (N=10), +75pp (N=15), +84pp (N=20).

**"+5 on N needs 2× data"** — clean empirical rule across N=5-20
at d_model=64. At this substrate scale, aggregate DeltaNet state
capacity (D² = 4096 scalars) is the binding constraint once N
exceeds per-N key-space-density threshold; inside that window,
only data-scaling moves the ceiling.

**Methodology receipt**: four architectural rounds nulled at
R10's 500/N undertraining; one flag change (`--per-N-train 2000/
5000/10000`) cracked each N-ceiling. Canonical "plateau = bug,
not tuning" case per `workflow_part_1.md`. See
`.claude/rules/delta_rule.md` for full arc + install path.

Generalization to Track A's architectural nulls: `aa46f2b`
batched pos_t (null, GPU variance dominates) and `6b27b90`
torch.compile on `_tq4_linear_kernel` (-1 to -7% across paths,
dynamic-shape recompile overhead > launch-savings already
captured by CUDA Graphs). Same principle: within fixed compute
budget, rearranging the dispatch doesn't add performance.

## 2026-04-22 session receipts (R22f + facade proliferation + recursion)

Full per-commit receipts landed as:

| Round | Commit | Receipt | Headline |
|---|---|---|---|
| R22f | `9691e06` | `evals/2026-04-22_r22f_threshold_sweep.md` | 51/60 → 60/60 via min_margin=14.5 recalibration. Per-N margin discipline: N=5 p50=23.3, N=10 p50=20.83 p5=15.21, N=15 p50=18.63 p5=16.39. Threshold-below-lowest-p5 rule. |
| R53a | `69279d4` | `evals/2026-04-22_r53a_number_theory_facade.md` | 8/15 → 15/15 mod/GCD/LCM. Exposed the `▁`-strip bug (token id 236743 consumes bias slot 0; Gemma natural `0` logit 57-66 dominates +50 boost on `▁`). |
| R22d rerun | `c3cc73f` | `evals/2026-04-22_r22d_rerun_threshold_14.5.md` | Independent corpus (all-keys-per-mem-block) confirms 42/60 → 60/60 at 14.5. 58/60 fired, 0 regressions. |
| R60a | `afc0220` | `evals/2026-04-22_r60a_icd10_tier3_demo.md` | 8/30 → 26/30 on 72,748-code ICD-10 DB. First tier-3 via decode-path. 4 edge codes resist; rule: text-recall decode-path works when answer is short known-length text. |
| R70a | `956a3ae` | `evals/2026-04-22_r70a_planner_mixed.md` | 20/20 route + 18/20 answer cross-domain Planner dispatch. |
| F1 | `8ba151d` | (r60a v2 receipt) | Code-echo detect+retry infrastructure; 4 stubborn edges confirmed structural. |
| F2 | `5ee61a5` | `evals/2026-04-22_r70b_planner_chain.md` | 12/12 route + 12/12 answer on 2-step chains ("X in hex"). Option C step-1 landed. |
| F3 | `3274659` | `evals/2026-04-22_r80a_recursion_level1_demo.md` | 5/10 → 10/10 via substrate-generated factorial + fibonacci facades. Level 1 shipped. Three-gate CALM discipline documented. |
| M1+M2 | `5173745` | `evals/2026-04-22_m1a_four_new_facades.md` + `m2a_level2_metafacade.md` | M1 12/20 → 20/20 (4 more Level-1 specs). M2 4/15 → 15/15 (5 Level-2 meta-synthesized specs). Level 2 shipped. |

**Session total**: 20/60 → 60/60 R22 retrieval, 12/30 → 26/30 tier-3
ICD-10, 0 → 15/15 NumberTheory, 0 → 12/12 Planner chain, 5 human-
written + 11 auto/meta-generated facades operational on prod Gemma.
Measurement receipts and per-probe JSONLs in `.cache/` for replay.

## Related rules

- `workflow.md` — the general hypothesis-test loop
- `Substrate.md` — install modes
- `delta_rule.md` — PT+Delta arc + MQAR scaling full receipt
- `embed_intelligence.md` — delivery path from card to Gemma's tokens
- `tracing_intelligence.md` — first-principles bound on what's compilable
- `retrieval.md` — hybrid retrieval used by augmentation paths
- `augmentation_thesis.md` §"Automatic Tier-1 preservation" — why
  blanket augmentation fails even on passing extractor
