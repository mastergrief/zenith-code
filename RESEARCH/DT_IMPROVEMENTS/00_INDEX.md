# DT Architectural Improvements — rescoping DT's role across three regimes

High-level spec derived from a joint claude+codex design round on
2026-04-24 during the HumanEvalPlus RENAME extension eval, which
revealed DT behavior on a new benchmark shape that the MBPP
"DT-obsolete" verdict did NOT anticipate.

## What this is

A design spec for rescoping DT's role and architectural improvements
that match its actual strengths. Currently the `delta_rule.md`
verdict reads "DT ruled out for MBPP-style signature prediction."
That verdict was correct for **name-repair on caller-contract
regimes**. HumanEvalPlus N=164 evidence shows DT has an
unrelated positive role — **decode-time structural prior** —
that the ruled-out verdict does not cover.

This spec:
1. Formalizes the three regimes DT implicitly operates across
2. Diagnoses why DT's current unified training fails on two of them
3. Orders five hypotheses for architectural improvement from
   cheapest-falsifier to longest-horizon
4. Proposes a staged pursuit path — H0 first (does DT even need
   improving, or does a non-DT facade capture its wins?), then
   verifier-gating, then stacking, then training-regime awareness,
   then grammar-aware bias

Not a build plan. A durable record of the architecture thinking so
a future session can pick up without re-deriving the framework.

## Files

| File | Owner | Content |
|---|---|---|
| `00_INDEX.md` (this file) | claude | Overview, thesis, pursuit path, decision the user needs to make |
| `01_ARCHITECTURE.md` | claude | Three-regime framework, current DT failure modes, 5 hypotheses in architectural framing, receipts |
| `02_IMPLEMENTATION.md` | codex | Mechanism shape per hypothesis, file/class targets, integration paths |
| `03_TESTING.md` | codex | Falsifiers, success gates, measurement protocol, baseline requirements |

## Thesis (one line)

> **DT is a decode-time structural prior whose usefulness is
> regime-dependent. Scoping its role per regime (prompt-copy,
> caller-contract, NL-inference), gating it on deterministic
> verifiers, and stacking it with complementary post-gen repair
> gives more value than training a "better" unified DT.**

## The diagnosis — three regimes DT conflates

The current `CodeDtSkeletonFacade` treats every code-completion prompt
uniformly: predict a skeleton, bias Gemma toward it, hope the bias
opens a correct emission. Three distinct regimes exist where the
"correct skeleton" comes from different sources:

| Regime | Example | Where signature comes from | What DT SHOULD do |
|---|---|---|---|
| **Prompt-copy** | HumanEvalPlus | inside the prompt (signature + docstring given) | copy from prompt, don't predict |
| **Caller-contract** | MBPP (`assert fn(...)`) | pinned by test's assert | respect caller arity + name |
| **NL-inference** | free NL problem description | nowhere — must be inferred | actually predict |

Unified training on pooled data causes DT to **hallucinate uniformly**
— predicting arg names on prompt-copy regimes where it should copy,
predicting arities that don't match caller contracts on MBPP-style
regimes. Catastrophic when it hallucinates wrong arity; soft-loss
when it hallucinates wrong arg names that poison body references.

## HumanEvalPlus evidence

Live N=164 run complete (2026-04-24, T+9h10m, daemon PID 157654 → exited cleanly); codex offline RENAME replay via `scripts/dt_rename_offline_eval.py` followed.

| Metric | stock (live) | dt (live) | Δ dt vs stock |
|---|---|---|---|
| all_pass / 164 | 41 (25.00%) | 44 (26.83%) | **+3 (+1.83pp)** |
| any_pass / 164 | 54 (32.93%) | 59 (35.98%) | **+5 (+3.05pp)** |
| macro_mean | 0.2801 | 0.3030 | **+0.0229** |
| micro (cell-weighted, FYI) | 31.34% | 33.69% | +2.35pp |
| per-problem wins | — | 7 | — |
| per-problem regressions | — | 2 | — |

**Offline RENAME comparison** (same-scorer; offline scorer diverges from live on 4 rows — 3 full-pass → zero-pass plus HumanEval/20 1000/1000 → 998/1000. See receipt §"Live-vs-offline scorer drift"):

| Metric | stock (offline) | rename (offline) | Δ rename vs stock |
|---|---|---|---|
| all_pass / 164 | 37 (22.56%) | 37 (22.56%) | **+0** |
| macro_mean | 0.2618 | 0.2618 | **+0.0000** |
| per-problem wins | — | 0 | — |

**RENAME is EXACTLY a no-op on HE+** — 0 wins, 0 regressions, 150/164 rows unchanged by AST rewrite. Confirms the predicted structural finding: HE+ prompts carry the signature, so `rename_first_def` has no handle; Gemma's body-only outputs have no `def` to rename either.

**Trajectory (dt vs stock macro delta as N grew)**:
N=40 +0.068 → N=60 +0.046 → N=80 +0.034 → N=100 +0.038 → N=120 +0.031 → N=140 +0.027 → N=164 +0.023 macro.

DT edge is small (~+2pp macro) but persistent — 7 DT-wins stock=0 rows recovered via structural scaffold, 2 regressions on arity/name mismatch. Consistent with low val_acc (0.20) and the structural-prior-not-content-predictor role.

**Falsifier hypothesis row 2 confirmed** — "RENAME delta flat = MBPP-specific contract-name coupling." Cross-regime differentiation holds: RENAME canonical for MBPP (NL prompt regime); DT small-positive for HE+ (signature+docstring regime). See `.claude/MEMORY/evals/2026-04-24_dt_rename_humanevalplus.md` §"Cross-benchmark comparison" for the full matrix.

Mechanism observed (per-row analysis of first 60 problems):
- **DT wins** (5 rows stock=0 → dt=full): all had arity-plausible DT predictions even with bogus names (`['paixs']`, `['s']`, `['val']` on 1-arg functions). The fake name didn't hurt because Gemma's biased body used the biased name consistently.
- **DT regressions** (2 rows stock=full → dt=0): `add` regressed because DT predicted 1 arg for a 2-arg function → TypeError every test. `flip_case` regressed because DT's biased arg name `service` caused body to reference natural-name `string` → NameError.

The ratio (5 wins : 2 regressions) holds at N=140. DT's greedy-decode
output is shape-forcing even at low content val_acc; when the SHAPE
is right, Gemma recovers; when the shape forces wrong arity, nothing
recovers.

## Five hypotheses, ordered by ROI

1. **H0 — Prompt-signature reconstruction baseline.** Before investing in DT improvement, test whether a non-DT facade that reuses the prompt's known signature captures the same wins. Two delivery variants: (a) **prompt-prefix reuse** — keep the prompt-carried signature + docstring as the decode prefix and let Gemma emit body only (HE+ primary path); (b) **deterministic known-signature bias** — step-through-bias the parsed signature, same mechanism DT uses but with ground-truth string instead of DT's predicted string (ablation / control for regimes without prompt-signature). If (a) captures DT's HE+ wins, DT value is structure-forcing only; a cheap facade replaces the need for DT improvement.
2. **H1b — Arity + name verifier-gated DT firing.** Mirror `VerificationHook.min_margin` but deterministic: extract known arity + names from prompt or caller-contract; gate DT firing ON iff predicted matches known. Preserves Tier-1 on mismatches. Addresses both regression classes (arity + name).
3. **H3 — DT + function-RENAME stacking (scoped).** Chain DT (structural scaffold) → Gemma (body) → RENAME (fn-name repair). **Scoped**: RENAME fixes fn-name only, not parameter names or body references. Parameter-name poison requires a new `signature_rewrite` facade (separate work).
4. **Training: regime-aware signature-source policy.** Relabel training data with `signature_source ∈ {copy, contract, infer}`; DT learns policy-conditional behavior. Addresses the architectural root: DT doesn't know when it should defer.
5. **H2 — Stateful grammar-aware structural bias.** Bias structural tokens only (delimiters, `def`, `:`) not identifiers. Requires a decode-time state machine (parser interleaved with decode), not a token-mask. 2-3× implementation scope of other hypotheses.

Detail for each: `01_ARCHITECTURE.md` §"Hypotheses in architectural framing."

## Pursuit path

**Recommended staged pursuit** (matches workflow `it works or it doesn't, ship or revert` discipline):

### Stage 1 — H0 baseline (falsifier)

Build non-DT prompt-signature reconstruction facade with the two delivery variants (prompt-prefix reuse + deterministic known-signature bias). Run on HE+ N=164 same corpus. Gate results:

- **Captures ≥80% of DT's wins**: DT's value is structure-forcing only. Ship the facade; skip further DT improvements for HE+-shape regimes. `delta_rule.md` verdict stays for MBPP regime; new facade covers HE+ regime.
- **Captures <80% of DT's wins**: DT is doing something the facade can't (learned token distribution beyond structure, tail-case handling). H1b becomes load-bearing.
- **Captures MORE than DT**: DT is obsolete for **HE+-shape prompt-copy regimes specifically**. Retire DT for HE+-shape code tasks; decode-path facades dominate THAT regime. NL-inference regime DT and retrieval-regime DT (`CopyAugmentedDeltaNet` for MQAR / NL-math) are NOT falsified by this — they require their own evaluations.

Stage 1 is the unblocker. Until H0 runs, Stages 2-4 are speculative.

### Stage 2 — H1b gating (if H0 doesn't dominate)

Add arity + name verifier extraction. Gate DT's `use_bias` based on match. Measure: do regressions disappear? Do wins persist? Full HE+ corpus A/B.

Success gate: regressions drop to ≤1 from the N=164 ungated DT baseline of 2 regressions (HumanEval/27 flip_case name-poison + HumanEval/53 add arity-mismatch); wins stay within 20% of the 7 ungated DT wins.

### Stage 3 — H3 stacking (cheap + additive)

Add RENAME call after DT-biased decode for caller-contract regimes (MBPP). Measure: does stacking give net-positive on MBPP? Does it interfere on HE+ prompt-copy regimes (probably no-op there since prompt has correct name)?

Success gate: MBPP A/B shows net ≥ max(DT-alone, RENAME-alone). Zero regressions on HE+.

### Stage 4 — Regime-aware training (longer-horizon)

Requires re-labeling training corpus with `signature_source` tag + training a regime-conditional DT head. 1-2 week scope. Only pursue if Stages 1-3 show DT has unique value that conditioning could amplify.

### Stage 5 — H2 grammar bias (speculative)

Only if everything else plateaus. Decode-time state machine is the ideal mechanism but complex + novel.

## Decision the user needs to make

**Which stage to pursue first?** Default lean = Stage 1 (H0 baseline) because it's the falsifier — cheapest and potentially invalidates the need for Stages 2-4.

**Alternatives:**
- **Option A (park)**: accept the current `delta_rule.md` MBPP verdict + add a one-line clarifying scope note ("DT ruled out as name-repair for MBPP; may be useful as structural scaffold for other regimes; unvalidated"). No new work.
- **Option B (Stage 1 — recommended)**: 2-day scope. Build H0 facade, A/B vs DT on HE+, decide Stage 2 gating from result.
- **Option C (Stages 1+2 bundled)**: 1-week scope. Build H0 + H1b together, test all three conditions (stock, DT, H0-facade, H1b-gated DT) on HE+ full corpus.
- **Option D (full arc)**: 3-4 week scope. All 5 stages. Highest ROI only if HE+-like regimes are commercially important.

Default behavior if no pick: Option A (park + clarify verdict). Current MBPP story is still valid; rescoping is low-urgency since RENAME already covers the MBPP regime.

## What this subsumes

- `delta_rule.md` §"Status: ruled out for MBPP-style signature prediction" — remains correct; scope-clarifies that "ruled out" means "as name-repair mechanism"
- The synthesis round's meta-role framing ("DT is a decode-time structural prior") is promoted to the 01_ARCHITECTURE thesis

## What this does NOT subsume

- Retrieval-regime DT (`CopyAugmentedDeltaNet` for MQAR / NL-math structure-extraction) — orthogonal, not discussed here
- Plain PT architecture decisions — covered in `delta_rule.md`
- CodeExampleDB retrieval hits as an orthogonal tier-2 mechanism — covered in `retrieval.md`
- Substrate install paths (`CardSlot` / in-attention / decode-path facade) — covered in `Substrate.md`

## Cross-refs

- `.claude/rules/delta_rule.md` — current DT architecture + MBPP verdict
- `.claude/rules/augmentation_thesis.md` — tier-1/2/3 framework; structural-scaffold hypothesis fits tier-2
- `.claude/rules/compute_facades.md` — rename-facade scope, decode-path facade pattern
- `.claude/rules/capability_gain.md` — falsifier-first measurement discipline motivating H0
- `.claude/MEMORY/evals/2026-04-24_dt_rename_humanevalplus.md` — the eval receipt that motivated this spec
- `RESEARCH/VGSL/00_INDEX.md` — file-layout precedent for research specs

## Status

- **2026-04-24**: joint design round complete (msgs `1777043908015-3daef20f` / `1777044016567-cfa1c599` / `1777044064968-70319770` / `1777044094542-8f6185d0` / `1777044115381-081b85b1` / `1777044130291-3ffc5a4f`); spec drafted
- **Awaiting user decision** on pursuit stage
- No implementation work gated on this spec — pure architectural thinking
