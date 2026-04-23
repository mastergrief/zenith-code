# Gemma 4 E4B Circuit Atlas

Master reference for mapped circuits. Updated end of session 33
after R50.6. Every probe we run adds rows here. Future sessions
consult this BEFORE probing a capability to check whether it's
already mapped.

Sibling to `substrate_registry.md` (tracks what's INSTALLED in a
`GemmaSubstrate`). Atlas tracks what's MAPPED — empirical circuit
knowledge regardless of install state.

## Quick reference

- **7 capabilities** fully mapped at sweep + per-head resolution
- **3 causally validated** (R28 arithmetic L30, R42 SV L23, R43 comparison+counting L23)
- **3 circuit shapes** identified (concentrated, cooperative,
  diffuse) + **1 hybrid** (multi-stage pipeline)
- **~10 specialist heads + ~4 shared hub heads** identified so far
- **2 shipped facades**: `HubInjectionCard` (R44/R45, 5-for-1 via L23 hub) + `MultiStepReasoningFacade` (R46.2, 17/17 real Gemma fixes)

## Capabilities table

| # | Capability | Peak layer | Peak Δ | #hurts/N | Shape | Attn-compile? | Status | Rounds |
|---|---|---|---|---|---|---|---|---|
| 1 | **Arithmetic** (a·b) | L23 GLB | **-10.18** | 10/10 | Concentrated + compute pipeline | ✓ YES (R28 validated) | Done | R13-R28 |
| 2 | **Factual recall** (capital of X) | L11 GLB | -1.56 | 9/10 | Diffuse | ✗ FFN-locked | Done | R29-R30 |
| 3 | **Induction** (A B ... A → B) | L34 SWA | -1.17 | 20/20 | Concentrated (L37 H6 classic induction head) | ✓ YES | Done | R31-R33 |
| 4 | **Counting** (1 2 3 4 → 5) | L20 SWA | -3.93 | 6/6 | Cooperative (3-way L20) + H4 specialist across L31/33/37 | Partial | Done | R34-R35 |
| 5 | **Comparison** (which is larger) | L35 GLB | -1.77 | 18/18 | Diffuse | ✗ FFN-locked | Done | R36-R37 |
| 6 | **SV agreement** (subject-verb) | L23 GLB | -4.10 | 14/16 | Hybrid: 2 concentrated stages + 1 diffuse (pipeline L23→L29→L35) | Partial (R42 causal) | Done | R38-R40, R42 |
| 7 | **Multi-step composition** (N-op infix) | L24 SWA | **-17.23** | — | Diffuse at attn + FFN + per-neuron + SVD. SAE reconstructs rank-50 feature basis but features have zero causal effect (R50.5). | ✗ All install paths ruled out | Done (compilation open) | R47-R50.6 |

## Shipped facades

Facades backed by circuit atlas findings, not bolted on.

| Facade | Backing finding | Capabilities served | Measurement |
|---|---|---|---|
| `HubInjectionCard` (`calm/llm_computer/facades/hub_l23.py`) | R42/R43 L23 H1/H4 forced-attention causal validation | Arithmetic + SV agreement + comparison + counting + multi-step = **5-for-1** | Bit-identical to R43 inline intervention (R44); `generate()` verified 5×12 decode tokens (R45) |
| `MultiStepReasoningFacade` | R11 step-through digit bias extended to N operands via NL infix parser + `safe_eval` routing | Multi-op arithmetic expressions | **17/17 real Gemma fixes, 0 regressions on held-out** (R46.2) |

## Hub layers — multi-capability layers

These layers serve multiple capabilities. Compiling interventions
at hubs compounds across capabilities.

### L23 (GLOBAL, d_head=512) — primary hub

Serves: arithmetic (peak), SV agreement (peak), counting (secondary),
comparison (secondary).

| Head | Arithmetic | SV agreement | Comparison | Role |
|---|---|---|---|---|
| **H1** | -4.85 (R17) | -0.91 (R39) | -0.25 (R37 top) | "Second content item" reader — reads b-operand (R41, 3× b-bias), distractor noun (R40, 0.50 on distractor) |
| **H4** | -4.30 (R17) | -1.05 (R39) | — | Mixed content reader — subject complex on SV (R40, 0.76), diffuse across operands on arithmetic (R41, 1.3× b-bias) |
| H6 | moderate | -0.37 (R39) | — | Minor contributor across tasks |

**Key hub property**: H1 + H4 together form a "general cross-position
content-read" module. Downstream layers route their V outputs to
task-specific FFNs. Same attention mechanism, different Q patterns
per context.

### L29 (GLOBAL, d_head=512) — secondary hub

Serves: arithmetic (R16 top-10), SV agreement (L23→L29→L35 pipeline,
R38 16/16 hurts).

| Head | SV agreement | Role |
|---|---|---|
| **H7** | -0.93 (R39) | Number-feature extractor (reads singular/plural marking from subject content) |

### L35 (GLOBAL, d_head=512) — application hub

Serves: comparison (peak), SV agreement (final stage), arithmetic
(R16 secondary -1.50).

Per-head behavior is **diffuse across all capabilities** probed
(comparison R37 top H4=-0.37; SV R39 top H4=-0.19). Suggests L35
is primarily an FFN-based "apply" layer; attention contribution
is diffuse final cleanup.

### L37 (SWA, d_head=256) — pattern-completion hub

Serves: induction (peak), counting (secondary), comparison (minor),
arithmetic (secondary).

| Head | Induction | Counting | Role |
|---|---|---|---|
| **H6** | -0.52 (R32), 11/20 | — | **Classic induction head** (Olsson 2022): attends to position after prior occurrence of current token (R33 confirmed, 55% attention on answer-letter position) |
| **H4** | — | -1.02 (R35) | **Numeric successor specialist** — also shows up at L31, L33 for counting; weak at L35 for comparison |

## Specialist (single-capability) heads

Heads that fire strongly for one specific capability.

### Arithmetic specialists

| Layer | Head | Evidence | Role |
|---|---|---|---|
| L30 | **H6** | R25 Δ=-1.528, 10/10; R26 attends pos 3 (a_ones tok) at 0.61 | a_ones position selector |
| L30 | **H4** | R25 Δ=-0.33; R26 attends pos 7 (b_ones tok) at 0.42 | b_ones position selector |

### Induction specialists

| Layer | Head | Evidence | Role |
|---|---|---|---|
| L37 | **H6** | R32 Δ=-0.52, 11/20; R33 attends to answer-letter at 0.55 | Classic induction — reads position after prior occurrence |

### Counting specialists

| Layer | Head | Evidence | Role |
|---|---|---|---|
| L20 | **H2** | R35 Δ=-1.37, 4/6 | L20 cooperative triple (a) |
| L20 | **H5** | R35 Δ=-1.15, 4/6 | L20 cooperative triple (b) |
| L20 | **H6** | R35 Δ=-1.00, 4/6 | L20 cooperative triple (c) — sum ≈ full-layer -3.93 |
| L31 | **H4** | R35 Δ=-0.81, 4/6 | Numeric successor (first appearance) |
| L33 | **H4** | R35 Δ=-1.00, 4/6 | Numeric successor (second stage) |
| L37 | **H4** | R35 Δ=-1.02, 5/6 | Numeric successor (final, also see hub) |

### SV agreement specialists

| Layer | Head | Evidence | Role |
|---|---|---|---|
| L23 | **H1** (hub) | R39 Δ=-0.91; R40 attends distractor at 0.50 | Distractor reader (second noun) |
| L23 | **H4** (hub) | R39 Δ=-1.05; R40 attends subject complex at 0.76 | Subject reader (subject noun + modifier) |
| L29 | **H7** | R39 Δ=-0.93, 12/16 | Number-feature extractor |

### No attention-level specialists found for

- **Factual recall**: L5 and L11 diffuse (top head -0.08 vs full -1.56). Circuit is FFN-locked.
- **Comparison**: L23 and L35 diffuse at head level (top head -0.37 of full -1.77). Needs FFN or side-channel approach.

## Circuit shape classifier

Run this decision protocol on any new capability.

```
1. 42-layer attn ablation sweep
   → identifies peak layer(s)
2. Per-head ablation at peak layer
   → classify shape:

  If 1-2 heads >= 50% of full-layer Δ:
     → CONCENTRATED (compilable as 1-2 LookUpExact gates)
     Examples: arithmetic L23, induction L37 H6

  If 3-4 heads each -0.5 to -1.5 and sum ≈ full-layer Δ:
     → COOPERATIVE (compilable as 3-4 LookUpExact gates)
     Example: counting L20

  If no head >= -0.2 despite full-layer Δ <= -1.0 and sum of
     per-head ≈ 20% of full:
     → DIFFUSE (FFN-locked, not attention-compilable)
     Examples: factual recall L5/L11, comparison L35

  If multiple layers in a pipeline each hit all prompts with
     different head specialization per layer:
     → HYBRID PIPELINE (mixed compilation approach)
     Example: SV agreement L23 → L29 → L35
```

## Compilation priority ranking

By attention-compilability + hub-sharing potential:

1. **L23 H1/H4** (hub, **5 capabilities** — arithmetic + SV + comparison + counting + multi-step) — highest ROI, already shipped via `HubInjectionCard` (R44)
2. **L30 H4/H6** (arithmetic specialists, R28 validated) — proven path
3. **L37 H6** (induction head, classic) — single head, clean
4. **L37 H4** (numeric successor, 3 capabilities) — another hub candidate
5. **L20 H2/H5/H6** (counting cooperative) — 3 heads, moderate cost

## R42-R50.6 summary (post-R41 update)

### R42-R43 — Hub causal validation (+ atlas closure on 3 capabilities)
Forced-attention at L23 H1/H4 preserves SV agreement (R42, mean
|Δ|=0.467, 8/10), comparison (R43a, |Δ|=0.176, 18/18 — cleanest
result in the arc), counting (R43b, |Δ|=0.528, 6/6). L23 hub
empirically proven across 3 capabilities in addition to arithmetic
→ **32/34 argmax matches (94%)** cross-capability preservation.
**3 causal validations total** (R28, R42, R43).

### R44-R45 — `HubInjectionCard` facade
Wrapping R43 intervention as a runtime-dispatched facade. Detects
natural top-position via live Q/K rather than hand-coded per-task.
Bit-identical to R43 inline intervention. R45 extended to
`.generate()` prefill-only injection across 5×12 decode tokens —
works through autoregressive decode without per-step hook.

### R46 — `MultiStepReasoningFacade`
N-op extension of R11's step-through digit bias. Parses NL infix
("2+3×5-7"), routes values through `safe_eval`, step-through
digit-bias on each intermediate and final result. R46.2:
**17/17 real Gemma fixes, 0 regressions on held-out prompts**.
5th beneficiary of L23 hub (arithmetic + SV + comparison +
counting + multi-step = 5-for-1).

### R47 — Multi-step layer sweep
R47.1 initial sweep contaminated by a "copy c" prompt-format
shortcut (19 spurious layers with Δ<-1.0); R47.2 diagnosed the
contamination at L34 (top H4 only -0.20 of full Δ); R47.3 clean-
prompt sweep identified **L24 (SWA) as the multi-step composition
peak, Δ=-17.23 (69% > R16's L23 -10.18)**; R47.4 per-head at L24
diffuse (top H1 -0.635) — attention-level install ruled out.

### R48 — L24 FFN per-neuron ablation
R48.1: 10,240 neurons in 20 chunks ablated; no chunk carries the
signal. **ROME/MEMIT-style per-neuron install ruled out.** L24
composition is not a weight-localizable FFN circuit.

### R49 — L24 pathway + rank decomposition
R49.1 mean-centered SVD: composition signal is rank-34 @ 90%
variance (naive rank-1 is DC-dominated — don't skip centering).
R49.2: at pos(-1) only, K=1 preserves 100% (task-rank 1 at last
position). R49.3: FFN at all non-last positions is zero for
composition → per-layer-embd injection candidate. R49.4: pathway
decomposition (attn vs FFN vs proj) is distributed with significant
non-additive interactions. R49.5: joint-output task-rank 1 at
non-last positions (composition info spatially distributed across
prefix, not concentrated at final-token output).

### R50 — SAE arc (infra shipped, causal gap falsifies SAE-for-compilation)
R50.1: first Sparse Autoencoder in project; λ=5e-4 gave 98% recon
but L0=2823 (not sparse). R50.2: λ=5e-3 plateau at L0~1700, L1
can't break through. R50.3: **TopK SAE (K=50) reconstructs L24
composition at 99.1%, effective L0=50 → L24 composition is
rank-50 in a learned feature basis.** R50.4: K=100 SAE, 370
features with 70-1089× multi-op vs single-op activation ratio
(task-specific directions exist). R50.5: **ablating top-50
composition features has ZERO causal effect** (17/30 baseline →
17/30 ablated). R50.6: re-installing the trained SAE at L24 at
99.6% recon preserves arithmetic 100% (infra end-to-end, but
reconstructed basis does NOT recover a compilable composition
circuit).

**R50.5 falsifies R20's "SAE = next step" roadmap direction for
compilable-intervention work.** Reconstruction fidelity ≠ causal
effect under ablation on distributed circuits. Open problem:
find SAE architectures (transcoders, attention-SAE, cross-layer
SAE) where reconstructed components carry causal effect.

## Gaps in the atlas — unmapped capabilities

Priority candidates for future probing (each ~30-60 min):

- **Subtraction** — likely analog of arithmetic but with different compute circuit
- **Division** — likely more complex than multiplication
- **Anaphora resolution** ("Alice gave Bob a book. She...") — linguistic
- **Tense agreement** (past/present/future) — linguistic
- **Analogies** (A:B :: C:?) — abstract reasoning, known mechinterp target
- **Syllogism** (transitivity) — logical reasoning
- **Code syntax completion** — programming language circuits
- **Negation** (not X → ¬X semantics) — linguistic
- **Factual recall at scale** (not just country-capital) — different retrieval

## Protocol reference

Canonical mapping protocol (from R13-R50.6):

1. **Sanity baseline**: pick prompts where Gemma's argmax matches
   expected answer. Filter to >= 80% clean baseline rate.
2. **Layer sweep**: 42-layer attn ablation via `ZeroReturning`
   wrapper on `layer.attn_output`. Measure Δ(baseline-argmax logit).
3. **Per-head**: 8 heads × top 1-3 layers. Use `HeadAblatingWrapper`
   on `layer.attn_output` input.
4. **Classify shape** (concentrated / cooperative / diffuse / hybrid).
5. **Attention pattern** (for concentrated/hybrid): capture input
   to `attn_q`, recompute Q/K with NEOX RoPE, softmax weights.
6. **Causal validation** (optional, strong evidence): force the
   hypothesized attention pattern, measure effect preserved.

All scripts live at `scripts/test_*.py`. Look for named patterns:
- `test_{capability}_sweep.py` → layer sweep
- `test_{capability}_per_head.py` → per-head at top layers
- `test_{layer_head}_{task}_pattern.py` → attention-pattern inspection

## Related rules

- `augmentation_thesis.md` — strategic synthesis (thesis, tier
  framework, typology, compositional hypothesis)
- `tracing_intelligence.md` — first-principles bound on what's
  compilable
- `MEMORY/atlas/tracing_arc_part_1.md` + `tracing_arc_part_2.md` — concrete round-by-round progress log
- `capability_gain.md` — measurement discipline
- `Substrate.md` — install mechanics (how to compile into Gemma)
- `substrate_registry.md` — what's currently INSTALLED (vs this
  file: what's MAPPED)
