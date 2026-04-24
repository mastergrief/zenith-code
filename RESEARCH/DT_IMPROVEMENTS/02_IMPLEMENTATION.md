# DT Improvements - Implementation

Implementation shape for the DT architectural-improvement line. Thesis and
regime framing live in [`01_ARCHITECTURE.md`](01_ARCHITECTURE.md); falsifiers
and success gates live in [`03_TESTING.md`](03_TESTING.md). See
[`00_INDEX.md`](00_INDEX.md) for the full doc set.

---

## 1. TL;DR

The first implementation move is not a better DT checkpoint. It is a
small set of facades and gates that separate three jobs the current DT
path mixes together:

1. **copy** a signature already present in the prompt
2. **respect** a signature contract exposed by tests or callers
3. **infer** a signature from natural language when no contract exists

The current `CodeDtSkeletonFacade` predicts one whole `def FN(args):`
string, tokenizes that whole string, and steps one forced token per
decode step. That is useful as a structural prior, but unsafe when the
signature is already known from the prompt or caller.

Implementation order:

1. **H0 - prompt-signature reconstruction baseline.** Parse the known
   signature and measure whether cheap signature reuse explains the DT
   wins before changing DT.
2. **H1b - arity+name verifier-gated DT.** Let DT fire only when it
   agrees with a deterministic signature source, or when no source is
   available.
3. **H3 - DT + function-RENAME stacking.** Stack only for function-name
   contract mismatch. Do not claim parameter repair.
4. **Training - regime-aware signature-source policy.** Add explicit
   `copy | contract | infer` labels so DT learns when not to hallucinate.
5. **H2 - stateful grammar-aware structural bias.** Defer. It is a new
   parser-position-aware decode hook, not a token filter.

---

## 2. Current Surfaces And Constraints

### `CodeDtSkeletonFacade`

Current code path:

- `predict_skeleton()` returns raw DT greedy text like `def FN(x):`
  (`calm/llm_computer/facades/code_dt_skeleton.py:109-138`).
- `parse_skeleton()` extracts only the argument list from that raw text
  (`code_dt_skeleton.py:140-149`).
- `solve()` substitutes the caller-provided function name, tokenizes the
  entire skeleton, and decides whether a bias can fire
  (`code_dt_skeleton.py:153-184`).
- `_generate()` appends a code fence and advances `bias_idx` exactly once
  per decode step (`code_dt_skeleton.py:206-260`).

That means H2 cannot be implemented by removing argument-name tokens
from `bias_ids`. If Gemma emits an argument name naturally while the hook
skips bias tokens, the next structural token fires at the wrong decode
position. Structural-only DT needs an explicit state machine over the
emitted token/text stream.

### `CodeRenameFacade`

Current RENAME path is intentionally narrow:

- `rename_first_def(source, new_name)` rewrites the first function name
  and self-recursive calls (`calm/llm_computer/facades/code_rename.py:147-164`).
- The AST/textual rewrite does not rename parameters or arbitrary body
  references (`code_rename.py:181-227`).
- The file docstring names the limitation directly: it only fixes the
  function-name failure mode, not wrong bodies or argument-name mismatches
  (`code_rename.py:36-42`).

So DT+RENAME is a function-name-contract stack, not a general identifier
repair stack.

### HumanEvalPlus prompt shape

HumanEvalPlus prompts already contain the target signature and docstring.
The current scorer had to retry `prompt + output` because Gemma often
emits body-only continuations after the code-fence decoration:

- `HumanEvalPlusProblem.prompt` is raw signature+docstring
  (`scripts/dt_install_eval.py:75-84`).
- `score_humaneval_plus()` retries with `p.prompt + "\n" + output` when
  the first extraction finds no code (`scripts/dt_install_eval.py:400-430`).
- The offline replay mirrors the same retry in
  `score_humaneval_output()` (`scripts/dt_rename_offline_eval.py:331-336`).

HE+ is therefore a prompt-copy regime first, and a DT-inference regime
only if the prompt parser fails.

---

## 3. Shared Data Model

The implementation should pass around explicit signature evidence rather
than burying it in facade-local strings.

```python
@dataclass(frozen=True)
class SignatureEvidence:
    source: Literal[
        "prompt",          # signature appears in prompt text
        "caller_contract", # tests/caller expose function name/arity
        "dt",              # DT predicted skeleton
        "generated",       # Gemma emitted a def naturally
    ]
    fn_name: str | None
    args: tuple[str, ...] | None
    arity: int | None
    raw: str
    confidence: float | None = None
```

```python
@dataclass(frozen=True)
class SignatureDecision:
    mode: Literal["copy", "contract", "infer", "unknown"]
    expected: SignatureEvidence | None
    dt: SignatureEvidence | None
    should_fire_dt: bool
    reason: str
```

The rule is simple:

- prompt evidence outranks caller-contract evidence when it contains a
  full parseable signature
- caller-contract evidence outranks DT when prompt evidence is absent
- DT is authoritative only in `infer` mode
- unknown mode must fail closed to natural Gemma or the H0 baseline

This can live in a new small module, for example
`calm/llm_computer/facades/code_signature.py`, so both live eval and
offline replay can use the same parse/verifier logic.

---

## 4. H0 - Prompt-Signature Reconstruction Baseline

### Purpose

H0 is a falsifier for DT improvement. If a deterministic prompt or
contract parser matches DT wins, the next product mechanism is a parser
facade, not a better trained card.

### Mechanism

Add a lightweight baseline facade:

```python
class CodeSignatureBaselineFacade:
    def inspect(self, prompt: str, tests: list[str] | None = None) -> SignatureDecision: ...
    def solve_with_signature(self, prompt: str, signature: SignatureEvidence) -> str: ...
```

The baseline should have two source modes and two delivery variants.

**Prompt-copy mode**:

1. Parse the first top-level `def <name>(<args>)` from the prompt.
2. Keep the prompt-carried signature and docstring as the decode prefix.
3. Let Gemma emit only the body continuation.
4. Score with the same `prompt + output` extraction used by HE+.

This is the HE+ primary path. It must not blindly duplicate the full
signature after a prompt that already contains that signature.

**Known-signature bias variant**:

1. Parse or reconstruct the known signature.
2. Start from an NL/task prompt that does not already end with the full
   target signature.
3. Apply the same step-through bias mechanism DT uses, but with the
   deterministic signature string instead of DT's predicted string.

This is an ablation, not the only H0 delivery path. It answers whether
DT's observed value is just "force Gemma through a valid opening" rather
than learned signature inference.

**Caller-contract mode**:

1. Parse function name from tests or caller metadata.
2. Parse arity when arguments are visible in assertion calls.
3. Construct only the minimal scaffold that is actually known.
4. Do not invent parameter names unless the test/caller exposes them.

This is a control path for MBPP-like benchmarks. It should never be
presented as full signature recovery if only arity is known.

### Integration targets

- Parser helpers: new `calm/llm_computer/facades/code_signature.py`
- Live A/B harness: `scripts/dt_install_eval.py`
- Offline replay: `scripts/dt_rename_offline_eval.py`
- Optional one-off runner: `scripts/dt_signature_baseline_eval.py`

### Required dump fields

Add fields to each eval row before running long jobs:

```json
{
  "signature_mode": "copy|contract|infer|unknown",
  "prompt_signature": "def f(x):",
  "contract_signature": null,
  "h0_output": "...",
  "h0_score": "1006/1006",
  "h0_diag": ""
}
```

The dump must preserve full outputs, as the HE+ forensic contract already
does.

---

## 5. H1b - Arity+Name Verifier-Gated DT

### Purpose

H1b makes DT a conditional structural prior instead of an unconditional
signature authority.

### Gate rule

Given `expected` evidence and `dt` evidence:

```python
def should_fire_dt(expected: SignatureEvidence | None,
                   dt: SignatureEvidence | None) -> tuple[bool, str]:
    if dt is None:
        return False, "dt_unparseable"
    if expected is None:
        return True, "infer_mode"
    if expected.fn_name and dt.fn_name and expected.fn_name != dt.fn_name:
        return False, "fn_name_mismatch"
    if expected.arity is not None and dt.arity is not None:
        if expected.arity != dt.arity:
            return False, "arity_mismatch"
    if expected.args is not None and dt.args is not None:
        if tuple(expected.args) != tuple(dt.args):
            return False, "arg_name_mismatch"
    return True, "verified"
```

For the current `CodeDtSkeletonFacade`, DT's raw skeleton uses `FN` as a
placeholder, so `fn_name` mismatch often means comparing caller/prompt
name against the substituted name rather than the raw `FN`. The important
checks are arity and argument names whenever a source can provide them.

### Result fields

Extend `CodeDtSkeletonResult` or wrap it with a richer result:

```python
@dataclass
class CodeDtVerifiedResult(CodeDtSkeletonResult):
    signature_mode: str = "unknown"
    expected_args: list[str] | None = None
    gate_reason: str = "not_checked"
    fire_bias_before_gate: bool = False
    fire_bias_after_gate: bool = False
```

The existing `used_bias` field should mean post-gate use, not raw
parseability.

### Integration shape

Minimal local change:

1. `CodeDtSkeletonFacade.solve(..., signature_decision=None)` accepts an
   optional precomputed decision.
2. If no decision is supplied, current behavior is preserved.
3. If a decision is supplied, `fire_bias = fire_bias and
   decision.should_fire_dt`.
4. Eval harnesses record `gate_reason` for every row.

Product-safe shape:

- Create `CodeDtVerifierFacade` that composes a signature inspector and
  `CodeDtSkeletonFacade` without changing the original class.
- Use the wrapper in new experiments first.
- Backport into `CodeDtSkeletonFacade` only after tests prove no behavior
  drift when the verifier is absent.

---

## 6. H3 - DT + Function-RENAME Stacking

### Purpose

H3 tests whether DT's structural prior and RENAME's caller-name repair
combine without inheriting either mechanism's unsafe claims.

### Stack order

Product-candidate stack:

```text
prompt -> H1b verifier gate -> Gemma decode -> optional function RENAME -> scorer
```

Diagnostic legacy stack, used only to attribute old behavior:

```text
prompt -> current ungated DT -> Gemma decode -> optional function RENAME -> scorer
```

Rules:

1. Product DT may bias only after H1b gate passes.
2. RENAME may rewrite only a function name to a caller-known target.
3. RENAME must not be credited with repairing parameter-name poison.
4. If prompt/caller signature gives exact argument names and DT disagrees,
   H1b skips DT before RENAME gets involved.

### Current RENAME remains scoped

Use the existing `rename_first_def()` for the H3 stack. Do not extend its
claim. Reports should keep two names distinct: `h3_dt_rename_ungated` for
legacy attribution and `h3_dt_rename_gated` for the product candidate. Do
not call either row `dt_signature_repair`.

### Separate future mechanism: `signature_rewrite`

Argument-name poison is a different mechanism. If the project wants to
repair DT outputs like `def flip_case(service):` when the prompt says
`def flip_case(string):`, that belongs in a new facade:

```python
class CodeSignatureRewriteFacade:
    def rewrite_signature_and_body(
        self,
        source: str,
        expected_args: tuple[str, ...],
        *,
        fn_name: str | None = None,
    ) -> RewriteResult: ...
```

Requirements for that future facade:

- parse the target function with AST
- map old parameter names to expected parameter names by position
- rewrite only function-local parameter definitions and references
- avoid rewriting globals, attributes, string literals, comments, helper
  functions, or nested scopes unless explicitly owned by the target
- reject on ambiguous arity or parse failure

It is not part of H3.

---

## 7. Training - Regime-Aware Signature-Source Policy

### Problem

A single DT trained to emit `def FN(args):` across prompt-copy,
caller-contract, and NL-inference examples learns the wrong invariant:
"always predict a signature." In prompt-copy and caller-contract regimes,
the correct action is often "copy or respect the source; do not invent."

### Data change

Extend `calm/hrm/code_dt_data.py` examples with an explicit regime label:

```json
{
  "problem": "...",
  "target": "def FN(x):",
  "signature_source": "prompt_copy|caller_contract|nl_infer",
  "source_signature": "def f(x):",
  "contract_fn_name": "f",
  "contract_arity": 1
}
```

### Model/input change

Add a compact mode token to the DT input, for example:

```text
<copy> prompt text ... <sep> def FN(args):
<contract> prompt text ... <sep> def FN(args):
<infer> prompt text ... <sep> def FN(args):
```

If the training code can support it cleanly, add an auxiliary
`source_head` predicting the same regime label. Treat that as a
diagnostic and regularizer; the required interface is still the explicit
mode token plus per-regime eval.

The first training target is policy separation, not a larger model:

- copy regime: exact prompt signature argument names should be preserved
- contract regime: function name and arity should respect caller/test
  evidence
- infer regime: DT may predict from NL when no source exists

### Acceptance condition before install

Do not install a retrained DT on Gemma until standalone evaluation is
reported by regime. Aggregate val accuracy can improve while the
prompt-copy slice worsens; that is a failed card for this use case.

---

## 8. H2 - Stateful Grammar-Aware Structural Bias

### Why H2 is deferred

The current bias API is one token list and one counter. A structural-only
bias wants to force punctuation and keywords while letting Gemma emit
content identifiers. That requires knowing whether the decoder is
currently at a structural position.

### Required controller

```python
class GrammarBiasController:
    def observe(self, token_id: int, text_delta: str) -> None: ...
    def next_bias_token_ids(self) -> dict[int, float]: ...
    def done(self) -> bool: ...
```

The controller owns a parser state such as:

```text
START -> DEF -> NAME -> LPAREN -> ARG_OR_RPAREN -> COMMA_OR_RPAREN -> COLON -> DONE
```

It may bias `def`, `(`, `,`, `)`, and `:` at state boundaries, but it
must observe naturally emitted identifiers and advance only after text
actually matches the grammar state.

### Integration target

H2 should not be patched directly into the current `_generate()` loop
until there is a fake-tokenizer unit test proving alignment. A safer path
is a new helper used by multiple facades:

```python
def generate_with_bias_controller(gemma, tok, prompt, controller, max_tokens): ...
```

Then `CodeDtSkeletonFacade` can choose between:

- whole-skeleton bias (current)
- verifier-gated whole-skeleton bias (H1b)
- grammar-aware structural bias (H2, later)

---

## 9. Rollout Plan

### Stage A - No-daemon mechanics

- Implement signature parser and decision types.
- Unit-test prompt-copy, caller-contract, and unknown cases.
- Unit-test H1b gate on synthetic skeletons.
- Unit-test RENAME scoping: function-name rewrite passes; parameter-name
  poison is not claimed as repaired.

### Stage B - Offline replay

- Add H0/H1b/H3 fields to offline replay where dumps contain enough
  metadata.
- Run against the N=5 HE+ smoke dump first.
- Run against MBPP N=50 forensic dump for RENAME and DT comparison.

### Stage C - Live eval

- Run live A/B only after the offline path proves shape and dump schema.
- Preserve the existing `stock` and `dt` columns so old receipts remain
  comparable.
- Add new columns instead of replacing old semantics.

### Stage D - Training

- Build regime-labeled data.
- Train only after H0 and H1b show there is remaining value not captured
  by deterministic copying/gating.
- Report per-regime held-out metrics before any Gemma install.

---

## 10. Non-Goals

- Do not change the active HumanEvalPlus daemon run while it is in
  progress.
- Do not claim current RENAME fixes argument-name mismatches.
- Do not retrain DT before H0 has had a chance to falsify the need.
- Do not ship H2 as a token-filter patch.
- Do not collapse prompt-copy, caller-contract, and NL-inference regimes
  into one aggregate metric.
