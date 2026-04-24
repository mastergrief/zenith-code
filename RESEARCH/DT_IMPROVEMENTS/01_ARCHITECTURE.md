# Architecture — DT's meta role, three-regime framework, hypothesis framing

## Thesis

**DT is a decode-time structural prior whose usefulness is
regime-dependent.** The `CopyAugmentedDeltaNet` architecture combines
two mechanisms — copy gate + Householder fast-weight recurrence —
that together encode "shape-right even when content-wrong" at the
token level. Low val_acc on the content-prediction task (0.20 on
code-skeleton) is NOT the load-bearing metric when DT is deployed as
a decode-time bias: what matters is whether DT's greedy decode
reliably emits shape-canonical openings that force Gemma out of its
failure modes.

The MBPP "ruled out" verdict in `.claude/rules/delta_rule.md` is
CORRECT for DT as a **name-repair mechanism** on caller-contract
regimes (MBPP's `assert fn(...)` pins the name exactly; RENAME's
deterministic AST rewrite dominates DT's probabilistic prediction).
That verdict does NOT cover DT's **structural-scaffold** role on
prompt-copy regimes (HumanEvalPlus), where the prompt already
contains the signature and Gemma's failure mode is body-only
continuation with inconsistent indentation.

Fixing DT means either (a) scoping its deployment per regime so it
only fires where it helps, or (b) training it on regime-aware
supervision so it learns policy-conditional behavior. Both are
architectural changes, not hyperparameter tuning.

## Three-regime framework

The signature of a generated function can originate from three distinct
sources. Each implies different correct DT behavior:

### Regime 1 — Prompt-copy

**Where the signature lives**: inside the prompt, before Gemma decodes.

**Examples**: HumanEvalPlus (`def entry_point(args) -> T:\n    """docstring"""`), completion-style benchmarks, any problem with a pinned signature.

**Correct DT behavior**: recognize that the signature is already
determined and copy it verbatim into the decode bias. Do not predict.
Any prediction error is a regression vs the free ground truth.

**Current DT behavior**: predicts a skeleton from the prompt via its
trained mapping. Low val_acc means 80% of predictions differ from the
correct signature. When the prediction is wrong, the bias poisons
Gemma's decode (wrong arity → TypeError; wrong arg names → body-ref
NameError).

### Regime 2 — Caller-contract

**Where the signature lives**: in the test or caller (e.g. MBPP's
`assert fn(args) == expected`).

**Examples**: MBPP, any test-pinned benchmark, any production setting
where the caller's contract name is fixed.

**Correct DT behavior**: respect the caller's arity and name;
potentially predict arg NAMES (unused by caller) but never arg count.

**Current DT behavior**: predicts a full skeleton, often with wrong
arity (caller contract is not visible to DT at train time). MBPP's
verdict reflects this: DT gets name right ~20% of the time, wrong
name = always regresses vs RENAME (which takes deterministic input).

### Regime 3 — NL-inference

**Where the signature lives**: nowhere. Must be inferred from problem
description.

**Examples**: free NL problem descriptions, "write a function that
counts vowels in a string" with no test preview.

**Correct DT behavior**: actually predict. This is the regime DT was
trained for. val_acc=0.20 is the honest capability on this regime.

**Current DT behavior**: correct in shape (decode mostly emits
parseable `def FN(...):` structure). Content (arg names, arity) at
low accuracy. Gemma continues from DT's opening; sometimes the body
agrees with DT's biased signature, sometimes it doesn't.

## Current DT architecture (context)

The `CopyAugmentedDeltaNet` class (`calm/llm_computer/copy_augmented_delta.py`)
combines three mechanisms:

1. **DeltaNet backbone**: Householder fast-weight recurrence per
   layer. At each position, the per-layer state `S_t` is updated as
   `S_t = S_{t-1} - β_t (S_{t-1} k_t − v_t) k_t^T`, and the output
   is `S_t @ q_t`. This is the explicit (k→v) storage that makes DT
   a superior retrieval mechanism vs plain PT at large key cardinality.
2. **Copy gate + pointer attention** (inherited from PT): softmax
   mixture of generated distribution and copy-from-prefix
   distribution.
3. **Output returns log-probs** (not logits), trained with nll_loss.

At `d_head=2` substrate invariant. Default trained config:
`use_chunkwise=True, n_delta_heads=1, n_iterations=1, chunk_size=32`.
~185K params for typical deployments.

**Deployment as decode-time bias** (`CodeDtSkeletonFacade`,
`calm/llm_computer/facades/code_dt_skeleton.py`):

- `predict_skeleton(prompt)` runs DT greedy decode → raw
  `def FN(args):` string (untyped; FN is a placeholder, args is a
  comma-sep list).
- `parse_skeleton(raw)` extracts the arg list from the raw output.
- `solve(prompt, fn_name)` stitches `def <fn_name>(<args>):` using the
  DT's parsed args + the caller-supplied fn_name.
- `_generate()` tokenizes the full stitched skeleton, biases Gemma's
  decode step-through (one bias token per decode step) until
  bias_ids exhausted, then Gemma decodes freely.

**Key fact about the bias mechanism** (codex's cited correction during
the synthesis round): `bias_ids` is ONE whole-string tokenization,
`bias_idx` advances by exactly one per decode step. The bias is
position-advancing, NOT grammar-aware.

## Failure mode taxonomy

Combining the three regimes × the known failure vectors:

| Failure vector | Regime 1 (prompt-copy) | Regime 2 (caller-contract) | Regime 3 (NL-infer) |
|---|---|---|---|
| **Wrong arity** | Catastrophic: test calls with 2 args, fn takes 1 → TypeError every test | Catastrophic: caller contract fixed, DT arity ≠ → TypeError | Recoverable: body adapts to DT's arity, tests still check behavior |
| **Wrong arg names** | Soft fail: Gemma's body may reference natural-name → NameError | Neutral: caller doesn't reference arg names, body just needs to work | Neutral: same as R2 |
| **Wrong fn name** | Impossible: name is in the prompt | MBPP: this is the core failure; RENAME fixes | Neutral: name isn't constrained |
| **Shape-right content-wrong** | **Win**: Gemma's body completes correctly using DT's biased scaffold; test passes | Neutral: structure was already feasible via natural Gemma | Neutral: same |
| **Shape-right content-wrong (arity-OK, name-OK)** | **Win**: structural recovery of body-only emission failure mode | **Win**: RENAME fixes name mismatch after DT's scaffold | **Win**: best case |

The mismatch: **DT's unified training doesn't know which regime it's
operating on**. It fires bias uniformly whether or not the prompt has
a pinned signature. Wins and regressions are distributed across the
matrix above.

## HumanEvalPlus evidence (N=164 run complete)

Dump at `/tmp/he_install_eval_results.json` (46 MB). Daemon PID 157654 ran 9h10m clean exit.

**Final N=164 (live scorer)**:

| Metric | stock | dt | Δ |
|---|---|---|---|
| all_pass / 164 | 41 (25.00%) | 44 (26.83%) | **+3 (+1.83pp)** |
| any_pass / 164 | 54 (32.93%) | 59 (35.98%) | **+5 (+3.05pp)** |
| macro_mean | 0.2801 | 0.3030 | **+0.0229** |
| micro (FYI) | 31.34% | 33.69% | +2.35pp |
| per-problem wins (dt > stock) | — | 7 | — |
| per-problem regressions (dt < stock) | — | 2 | — |

**RENAME condition (codex offline replay)**: all_pass 37/164, macro 0.2618 — **EXACTLY equal to offline stock** (0 wins, 0 regressions, 150/164 no-op). Confirms RENAME is structurally a no-op on HE+ prompt-copy regime: prompt already carries correct signature; `rename_first_def` has no handle. See `.claude/MEMORY/evals/2026-04-24_dt_rename_humanevalplus.md` §"Offline-scorer numbers" for detail.

Falsifier hypothesis row 2 (`RENAME flat + DT flat = MBPP-specific contract-name coupling`) confirmed: RENAME IS flat on HE+; DT is small-positive on HE+. The "MBPP-specific" framing holds for RENAME; DT has a different scoped role on HE+ (structural scaffolding, not name repair).

Macro-delta trajectory as sample grew (small-sample high water then stabilization):

| N | macro delta |
|---|---|
| 40 | +0.068 |
| 60 | +0.046 |
| 80 | +0.034 |
| 100 | +0.038 |
| 120 | +0.031 |
| 140 | +0.027 |
| 164 | **+0.0229** |

Final settled within the predicted range (+0.020 to +0.030 based on trajectory). DT's structural-prior signal is small but holds at full scale.

**Per-row mechanism (all 7 wins + 2 regressions on N=164)**:

DT wins (stock=0 → dt=full or partial):

| task_id | fn | DT_args | mechanism |
|---|---|---|---|
| HumanEval/14 | all_prefixes | `['paixs']` | arity-right, name-fake, body self-consistent |
| HumanEval/23 | strlen | `['s']` | arity-right, name-right |
| HumanEval/24 | largest_divisor | `['val']` | arity-right, name-fake, partial (121/169) |
| HumanEval/28 | concatenate | `['coss', '**kwicas']` | arity-right (effective), name-fake |
| HumanEval/47 | median | `['s']` | arity-right, name-fake, body self-consistent |
| HumanEval/55 | fib | `['n']` | arity-right, name-right, minor (2/45) |
| HumanEval/85 | add | `['service_se_idex']` | arity-1 match, name-fake (1-arg `add`, distinct from HumanEval/53's 2-arg) |

DT regressions (stock=full → dt=0):

| task_id | fn | DT_args | failure mode |
|---|---|---|---|
| HumanEval/27 | flip_case | `['service']` | arity-right but body references natural `string` → NameError |
| HumanEval/53 | add | `['max_bokinexe']` | arity-wrong (1 vs 2) → TypeError every test |

Pattern: DT wins ~4.3% of corpus on structural failure rows;
regresses ~1.2% on arity-or-name mismatch rows; ties on 94.5%.
Net positive but small signal.

## Hypotheses in architectural framing

### H0 — Prompt-signature reconstruction facade (falsifier)

**Architectural framing**: DT's role on HE+ may be reducible to "bias
Gemma toward the known signature". If the signature is already in
the prompt, a regex-parse of the prompt yields the ground-truth
signature without needing DT's learned prediction. Building this as
a non-DT facade and A/B-ing vs DT on HE+ isolates DT's unique
contribution from the structural-scaffold effect.

**Mechanism** — two delivery variants:

1. **Prompt-prefix reuse** (HE+ primary). Keep the prompt-carried
   signature + docstring intact as the decode prefix. Let Gemma emit
   only the body continuation. Score via the existing `prompt + output`
   extraction path. Does NOT duplicate the signature after a prompt
   that already contains it. This is the natural HE+ shape, matching
   how the current scorer already handles body-only outputs.
2. **Deterministic known-signature bias** (control / ablation). Parse
   the signature from prompt or caller-contract; feed the parsed
   string through the same step-through bias mechanism DT uses (same
   as `_generate()` loop at `code_dt_skeleton.py:206-260`) but with
   ground-truth string instead of DT's prediction. This isolates
   whether DT's observed value is "force Gemma through a valid
   opening" rather than "learned signature inference". Useful on
   regimes where prompt-prefix reuse doesn't apply (NL-inference
   contexts where signature must be reconstructed rather than copied).

Zero training, zero DT dependency on either variant.

**What it falsifies**: if H0 matches DT's wins on HE+, DT has no
unique contribution on prompt-copy regimes. The `delta_rule.md`
verdict should extend to "DT is obsolete as both name-repair AND
structural-scaffold on prompt-copy regimes; use decode-path facade
built from prompt parse." NL-inference regime and retrieval-regime
DT are NOT addressed by H0 and require their own evaluations.

### H1b — Arity + name verifier-gated firing

**Architectural framing**: DT's regressions come from firing bias
when DT's prediction doesn't match the deterministically-knowable
truth (prompt-signature for R1, caller-contract for R2). Gating DT
firing on verifier-match reduces the firing surface to only
high-confidence correct predictions.

**Mechanism**: before `_generate` installs bias, compute
`expected_arity` from prompt/caller, compute `expected_name` from
prompt/caller, and compare against `dt_args` and `fn_name` (already
known to facade). If mismatch → `use_bias=False`. Preserves Tier-1
automatically (unmodified Gemma decode). Additive over existing DT.

**What it falsifies**: if H1b-gated DT loses wins (not just drops
regressions), the current "DT wins" were actually DT's wrong-but-useful
predictions that happened to route Gemma into correct bodies. Unlikely
but measurable.

### H3 — DT + function-RENAME stacking (scope-corrected)

**Architectural framing**: RENAME repairs function-name-contract
mismatch via deterministic AST rewrite. DT repairs structural
emission failures via decode-time bias. The two mechanisms operate
on disjoint failure modes. Composition is additive for regimes where
both fire.

**Mechanism**: after `DT.solve()` produces `r_dt.generated`, pipe
through `rename_first_def(r_dt.generated, caller_fn_name)` for
caller-contract regimes. Already-correct name → no-op.

**Scope** (from codex's cited correction): `rename_first_def` rewrites
function name + self-recursive calls ONLY. Parameter identifiers and
body references are untouched. Parameter-name poison (HumanEval/27
`flip_case` regression) requires a NEW `signature_rewrite` facade
that AST-rewrites def parameters + body references. That is separate
architectural work, not an extension of current RENAME.

**What it falsifies**: if DT+RENAME on MBPP is STRICTLY WORSE than
RENAME-alone, DT's structural-scaffold was actively anti-helpful on
MBPP (corrupting Gemma's natural body emission in ways RENAME can't
fix). Unlikely given MBPP tests show DT's body preservation holds.

### Training — Regime-aware signature-source policy

**Architectural framing**: DT currently trains on unified skeleton
prediction without distinguishing where the signature should come
from. A regime-conditional head (per-example label
`signature_source ∈ {copy, contract, infer}`) lets DT learn
policy-conditional behavior: copy when label=copy, respect when
label=contract, predict when label=infer.

**Mechanism**:
- Training data relabeling: each (prompt, skeleton) pair gains a
  `signature_source` tag derived from the source corpus.
- Training loss: base nll_loss (skeleton prediction) + auxiliary
  cross-entropy on a `source_head` predicting the regime tag.
- Inference: regime detector at inference time (prompt-shape heuristic
  — does prompt contain a `def` line?) dispatches DT accordingly.

**What it falsifies**: if regime-aware training doesn't reduce HE+
regressions, DT's failure mode isn't "doesn't know when to copy" —
it's "can't predict content reliably at any regime." The training
investment is only worth it if regime-conditioning actually routes
predictions.

### H2 — Stateful grammar-aware structural bias (demoted)

**Architectural framing**: the ideal decode-time mechanism would bias
STRUCTURAL tokens (delimiters, keywords) while letting Gemma emit
identifier content freely. This decouples shape-forcing (which works
at val_acc=0.20) from content-prediction (which doesn't). Current
`_generate` is position-advancing, not grammar-aware, so a naive
token mask desynchronizes.

**Mechanism**: decode-time state machine that tracks "what's the
next structural delimiter position in the expected skeleton" and
fires bias only when Gemma's next emission should be a delimiter.
Between delimiters, let Gemma emit freely. On delimiter positions,
bias toward the expected token. Requires:
- Small grammar parser for `def FN(args):` shape
- State tracking across decode steps (which delimiter we're expecting
  next)
- Integration with the existing step-through bias loop

**Scope**: 2-3× implementation cost of other hypotheses. Defer unless
Stages 1-4 all plateau.

**What it falsifies**: if grammar-aware bias doesn't beat H1b-gated
DT, fine-grained structural control adds no value beyond the
coarse-grained "fire-or-don't" gate. Likely only wins on complex
grammars (not simple `def FN(args):`).

## Relationship to broader substrate architecture

DT's decode-time structural prior role fits cleanly into the
`augmentation_thesis.md` tier-2 framework: it's an **emission-shape
controller** that sits between Gemma's native prior and the output.
Unlike VerificationHook (output-level bias for single-token answers)
or CardSlot (residual-level injection), DT operates at the DECODE
loop level across multiple tokens.

Three stack positions DT could occupy after these improvements:

1. **Gate 1** (H0 if it dominates): decode-path facade built from
   prompt-parse. Zero DT dependency. Covers HE+-shape regimes.
2. **Gate 2** (H1b-gated): current DT with arity+name verifier gate.
   Covers regime-specific wins while avoiding regressions.
3. **Gate 3** (H2 ideal): grammar-aware bias. Covers regimes where
   neither H0 nor H1b-gated DT suffices (long-horizon, speculative).

The regime-aware training (Stage 4) cuts across all three — makes DT
a better citizen regardless of deployment gate.

## What we are NOT claiming

- Not claiming DT should replace RENAME for MBPP. RENAME stays canonical.
- Not claiming DT should replace decode-path facades for recall
  domains (ICD-10, etc.). Those remain canonical.
- Not claiming the retrieval-regime DT (`CopyAugmentedDeltaNet` for
  MQAR / NL-math) needs any of these changes. Those work; this spec
  is scoped to code-skeleton DT specifically.
- Not claiming validation at val_acc=0.20 is acceptable in general.
  Low val_acc is a SYMPTOM; the structural-prior role works DESPITE
  that, not because of it. Regime-aware training (Stage 4) could
  improve val_acc and widen DT's useful deployment range.

## Open questions for `02_IMPLEMENTATION.md` and `03_TESTING.md`

- H0 implementation: where does the prompt-parse facade live?
  `compute_facades.md` decode-path facade directory? New file?
- H1b implementation: verifier-extraction interface — on
  `CodeDtSkeletonFacade` or on the scorer? Does it access
  `p.prompt`, `p.fn_name` through the existing NamedTuple?
- H3 implementation: where is the stacking invoked — inside `solve()`
  or at the eval-harness layer?
- Regime-aware training: corpus relabeling plan — which existing
  datasets are prompt-copy vs caller-contract vs NL-infer?
- H2 implementation: does the grammar parser live in the facade or
  a separate decode-time hook module?
- Testing: what's the smallest benchmark that distinguishes all
  three regimes? Does HE+ + MBPP + a free-NL corpus suffice?

Those are codex's files to answer.
