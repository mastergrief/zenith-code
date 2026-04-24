# DT Improvements - Testing

Falsifiers, measurement protocol, and success gates for the DT
architectural-improvement line. Implementation mechanics live in
[`02_IMPLEMENTATION.md`](02_IMPLEMENTATION.md); thesis and regime framing
live in [`01_ARCHITECTURE.md`](01_ARCHITECTURE.md). See
[`00_INDEX.md`](00_INDEX.md) for the full doc set.

---

## 1. Testing Goal

The first testing job is not to show that DT can be made larger or more
accurate in aggregate. The first job is to determine which mechanism is
actually carrying the win:

1. known signature reuse
2. caller-contract repair
3. DT structural bias
4. body-trajectory side effects
5. training-time regime separation

A result that improves one benchmark by mixing those mechanisms together
without attribution is not enough to guide the next build.

---

## 2. Required Baselines

Every report should include these rows when the benchmark supports them:

| Row | Meaning |
|---|---|
| `stock` | Gemma without DT bias or post-gen rename |
| `dt_current` | current `CodeDtSkeletonFacade` whole-skeleton bias |
| `rename` | current function-name-only `CodeRenameFacade` on stock output |
| `h0_signature` | deterministic prompt/caller signature reuse baseline |
| `h1b_dt_gated` | DT bias after arity/name verifier gate |
| `h3_dt_rename_ungated` | current DT followed by function-name RENAME; diagnostic attribution row |
| `h3_dt_rename_gated` | H1b-gated DT followed by function-name RENAME; product-candidate row |

Rows may be omitted only when the benchmark cannot provide the required
source evidence, and the report must say why. The H3 rows are deliberately
split: compare `h3_dt_rename_gated` against `h1b_dt_gated` to isolate the
function-RENAME contribution with the gate held constant, and compare
`h3_dt_rename_ungated` against `dt_current` only as a diagnostic for the
legacy ungated path.

---

## 3. Measurement Protocol

### Two paths per round

Each implementation round needs both a raw/fast path and a user-facing
path.

Raw/fast path:

- unit tests for parsers, gates, and AST rewrites
- offline replay against preserved forensic dumps
- small N smoke (`N=5`) before full benchmark runs

User-facing path:

- live daemon eval through `bin/gemma-run` or the current product path
- full scorer output with all rows and per-problem diagnostics
- receipt update with aggregate and failure tables

Only ship a mechanism when both paths agree. Offline-only wins may be
schema artifacts; live-only wins may be stochastic or prompt-shape noise.

### Metrics

For HumanEvalPlus, headline metrics remain problem-macro rather than
cell-micro:

| Metric | Definition |
|---|---|
| `all_pass` | problem has pass_count == total_count |
| `any_pass` | problem has pass_count > 0 |
| `macro_mean_fraction` | mean of pass_count / total_count over problems |
| `micro` | total passed cells / total cells, reported as FYI |

For MBPP, keep the existing cell counts and known/novel split where the
receipt already uses it.

### Dump discipline

Long-running eval dumps must keep:

- full untruncated outputs for every condition
- prompt, entry point, tests, inputs, and expected results needed for
  offline replay
- parsed signature evidence and gate reasons
- per-input pass/fail arrays for HE+
- raw DT skeleton and parsed DT args

No `[:2400]` truncation on new forensic surfaces.

---

## 4. H0 - Prompt-Signature Reconstruction Baseline

### Hypothesis

A deterministic signature-reuse baseline captures most DT wins in
prompt-copy regimes, and may capture some caller-contract regimes,
without trained DT or decode-time skeleton hallucination.

### Tests

1. **Parser coverage test.** On HumanEvalPlus raw rows, prompt-signature
   parser extracts the entry-point signature for 164/164 rows, or every
   miss is categorized.
2. **Delivery ablation.** Compare prompt-prefix reuse vs deterministic
   known-signature step-through bias. HE+ should not duplicate a signature
   already present in the prompt; MBPP/free-NL controls may need explicit
   scaffold bias.
3. **N=5 smoke replay.** `h0_signature` recovers the same body-only
   outputs as the prompt-prepend scorer on the preserved N=5 smoke dump.
4. **HE+ full replay.** When `/tmp/he_install_eval_results.json` lands,
   score `h0_signature` beside stock, DT, and RENAME.
5. **MBPP caller-contract control.** On MBPP N=50, report whether only
   function-name/arity evidence is enough to explain any DT wins.

### Success gate

H0 is successful if either condition holds:

- on prompt-copy HE+, `h0_signature` is within 1pp macro of `dt_current`
  with fewer signature-related regressions
- on any benchmark, H0 explains enough of DT's wins to change the next
  build decision from "retrain DT" to "ship parser/gate first"

### Falsifier

H0 is falsified if DT keeps a material advantage after prompt/caller
signature reuse is controlled:

- DT macro advantage over H0 >= 3pp on HE+ full
- DT wins not attributable to signature shape remain after failure
  classification
- H0 introduces regressions that stock and DT avoid

### Failure meaning

- parser misses prompt signatures: H0 parser is too brittle
- H0 equals stock and DT beats both: DT may have real body-trajectory value
- H0 equals DT: DT's value in that regime is mostly structure forcing

---

## 5. H1b - Arity+Name Verifier-Gated DT

### Hypothesis

Most harmful DT regressions are predictable before decode from mismatch
between DT's skeleton and a known prompt/caller signature.

### Tests

1. **Synthetic gate tests.** Given expected `(fn, args)` and DT skeletons,
   gate reasons are deterministic: `verified`, `arity_mismatch`,
   `arg_name_mismatch`, `fn_name_mismatch`, `dt_unparseable`, or
   `infer_mode`.
2. **Observed regression tests.** Encode the HE+ failure shapes from the
   synthesis round: arity mismatch and argument-name mismatch should both
   skip DT.
3. **N=5 replay.** Gate decisions match expected prompt-copy evidence.
4. **Full replay/live run.** Compare `dt_current` vs `h1b_dt_gated` on
   wins retained and regressions removed.

### Success gate

H1b is worth shipping if it removes signature-caused regressions while
retaining most DT wins:

- signature-mismatch regressions reduced to zero on inspected failures
- retained DT wins >= 80% of current DT wins
- macro score is no worse than stock by more than 0.5pp
- every skipped row has an explainable `gate_reason`

### Falsifier

H1b is falsified if the deterministic gate kills the signal:

- retained DT wins < 50% of current DT wins
- most DT wins occur in rows that the verifier would skip
- mismatch categories do not predict failures

### Failure meaning

- gate too strict: expected evidence parser is overclaiming
- gate too loose: name/arity checks miss real mismatch modes
- no effect: DT's regressions are body-trajectory, not signature mismatch

---

## 6. H3 - DT + Function-RENAME Stacking

### Hypothesis

DT structural bias and function-name RENAME are complementary only in
function-name-contract regimes. The stack should not be credited with
repairing argument-name poison.

### Tests

1. **Scope unit test.** `rename_first_def` rewrites function name and
   self-recursive calls, but does not rewrite parameters or body
   references.
2. **MBPP N=50 replay.** Compare stock, DT, RENAME,
   `h3_dt_rename_ungated`, and `h3_dt_rename_gated` on the preserved
   MBPP dump or a fresh equivalent run.
3. **Gate-held-constant attribution.** Attribute H3's product value by
   comparing `h3_dt_rename_gated` against `h1b_dt_gated`; use the ungated
   H3 row only to explain legacy DT+RENAME behavior.
4. **HE+ replay.** Show RENAME is near-no-op when outputs are body-only
   or already use the prompt function name.
5. **Argument-poison negative test.** A synthetic DT output with wrong
   parameter names remains unrepaired by H3 and is classified as needing
   `signature_rewrite`, not RENAME.

### Success gate

H3 is useful if:

- it preserves RENAME's zero-regression property on MBPP-style name
  mismatches
- `h3_dt_rename_gated` adds at least one win beyond both RENAME alone and
  `h1b_dt_gated` on a held-out slice
- every win is classified as function-name repair or DT body-trajectory,
  not vague "signature repair"

### Falsifier

H3 is falsified if:

- it introduces any regression not present in RENAME alone
- incremental wins appear only in `h3_dt_rename_ungated` and vanish in
  `h3_dt_rename_gated`
- gains depend on parameter-name rewrites that current RENAME does not do

---

## 7. Training - Regime-Aware Signature-Source Policy

### Hypothesis

DT training improves only after examples carry an explicit signature-source
policy: prompt-copy, caller-contract, or NL-inference.

### Dataset gates

Before training:

- every example has `signature_source`
- prompt-copy examples include the source signature span
- caller-contract examples include function name and arity evidence
- NL-inference examples are separated from source-known examples
- train/val splits are stratified by regime

### Standalone metrics

Report per regime:

| Metric | Meaning |
|---|---|
| exact skeleton | full `def FN(args):` exact match |
| arity match | argument count matches source or target |
| arg-name match | argument names match when names are known |
| hallucination rate | DT invents different args despite source-known mode |
| abstain/skip compatibility | output can be safely gated when untrusted |

### Success gate

A retrained DT is eligible for Gemma install only if:

- prompt-copy hallucination decreases versus current DT
- caller-contract arity/name mismatch decreases versus current DT
- NL-inference exact skeleton does not regress materially
- per-regime metrics improve, not just aggregate accuracy

### Falsifier

Regime-aware training is falsified if mode labels do not move the failure
modes:

- prompt-copy still hallucinates argument names
- caller-contract still violates arity/name evidence
- aggregate improves while either source-known regime worsens

---

## 8. H2 - Stateful Grammar-Aware Structural Bias

### Hypothesis

A parser-position-aware decode hook can provide structural help without
forcing content identifiers. A token-filter shortcut cannot.

### Required pre-tests

1. **Misalignment proof.** A fake-tokenizer test demonstrates that
   deleting arg-name tokens from `bias_ids` advances later structural
   tokens too early under the current `bias_idx += 1` loop.
2. **Controller alignment test.** `GrammarBiasController` observes
   emitted identifiers and biases only the next structural delimiter.
3. **Fallback test.** On unexpected emitted text, controller stops biasing
   rather than forcing an invalid structure.

### Success gate

H2 can enter live eval only after unit tests prove:

- structural tokens are biased at the correct decode positions
- naturally emitted identifiers do not desynchronize the controller
- malformed generations fail closed
- H2 can be disabled with byte-identical behavior to current generation

### Falsifier

H2 is falsified or deferred again if:

- controller state requires brittle token-specific hacks
- alignment depends on one tokenizer's merge behavior
- H2 produces syntax-valid but semantically mismatched signatures more
  often than H1b-gated whole-skeleton bias

---

## 9. Cross-Hypothesis Decision Matrix

| Result | Decision |
|---|---|
| H0 ~= DT on HE+ | build parser/gate path before DT retraining |
| H0 < DT and H1b removes regressions | ship verifier-gated DT experiment |
| H3 > RENAME with zero regressions | keep stack for function-name regimes |
| H3 only helps when args mismatch | do not ship H3; spec `signature_rewrite` separately |
| regime training improves source-known slices | consider new DT checkpoint |
| H2 unit tests fail alignment | keep H2 parked |

The user-facing decision should be framed as "what to build first," not
"which idea sounds best."

---

## 10. Receipt Requirements

Every landed receipt should include:

- exact command lines and env vars
- daemon PID and elapsed time for live runs
- input dump path and output dump path
- before/after table for all conditions
- wins/regressions table by problem
- at least three named failure examples
- whether the result changes the next implementation step

For this active HE+ arc, the full-run receipt should fill in:

```text
stock:  all_pass=X/164 any_pass=Y/164 macro=Z
DT:     all_pass=X/164 any_pass=Y/164 macro=Z
RENAME: all_pass=X/164 any_pass=Y/164 macro=Z
```

Then the improvement spec can reference those numbers without changing
its mechanism claims.

---

## 11. Minimal Test Inventory

Initial unit tests should cover:

- prompt signature parser for HE+ prompt shapes
- caller-contract parser for MBPP assertion shapes
- H1b gate reason matrix
- RENAME function-name-only scope
- signature-rewrite negative fixture, even before implementation
- H2 bias-index misalignment proof
- HE+ scorer prompt-prepend retry
- offline replay schema with full outputs and gate fields

Initial integration tests should cover:

- N=5 HE+ replay from preserved smoke dump
- MBPP N=50 offline replay from preserved dump
- full HE+ replay once `/tmp/he_install_eval_results.json` exists
- one fresh live smoke before any long daemon run

---

## 12. Non-Goals

- Do not treat an aggregate DT val score as sufficient evidence.
- Do not call RENAME a parameter-repair mechanism.
- Do not use micro cell-weighted HE+ as the headline metric.
- Do not accept a structural-bias implementation without alignment unit
  tests.
- Do not overwrite existing forensic dumps without a new path or explicit
  backup.
