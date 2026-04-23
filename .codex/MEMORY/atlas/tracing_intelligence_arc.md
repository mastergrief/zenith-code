# Tracing Intelligence — Per-round validation receipts (Sessions 33-34, R13-R50.6)

The first mechanistic-interpretability arc on Gemma 4 E4B. Validated
that the tracing workflow (corpus + activation patching + per-head
ablation + Q/K/V decomposition + causal forced-attention) produces
clean compilable targets on capabilities whose information flow
aligns with Gemma's architecture. Current first-principles framing:
`.claude/rules/tracing_intelligence.md`. Companion full-arc narrative:
`MEMORY/atlas/tracing_roadmap_part_1.md`.

## R-by-R headline findings

- **R16 (activation patching)**: localizes arithmetic to **L22-L30
  cluster**, L23 peak (mean Δ=-10.18 on correct-digit logit, hurts
  10/10 arithmetic pairs). 5B params narrowed to 9 layers via a
  420-forward sweep in ~15 minutes.
- **R17 (per-head ablation)**: narrows L23 (8 Q-heads) to **H1 and
  H4** — other 6 heads have mean Δ ≈ 0.
- **R18 (Q/K/V decomposition)**: identifies **V (content) as the
  signal carrier**, not Q/K (attention pattern). H4's V ablation =
  -9.51, accounts for 93% of L23's total arithmetic contribution.
- **R19 (linear probe)**: V linearly encodes the product's first
  digit at 2x chance (0.22 vs 0.11, 270 samples). Real but indirect —
  V likely carries operand and intermediate representations, not the
  final digit. SAE work needed for clean features.
- **R20 (per-sub-head ablation)**: H4's 512-d output split into 256
  d_head=2 pairs; ablate each × 10 arithmetic pairs. **0 sub-heads
  with mean Δ < -1.0**; top sub-head = -0.583 (vs full H4 = -4.30).
  Top-8 sub-heads carry only 26% of damage; top-64 needed for 80%.
  Signal is distributed across H4's V subspace, not sparse in the
  d_head=2 basis.
- **R20-R40 (capabilities mapped)**: 6 capabilities — arithmetic,
  factual recall, induction, counting, comparison, SV agreement.
  Circuit typology: concentrated (arithmetic, induction), cooperative
  (counting L20), diffuse (factual, comparison), hybrid pipeline (SV
  agreement).
- **R42/R43 (hub-sharing causally proven)**: L23 H1/H4 forced-
  attention intervention (mirror of R28) preserves SV agreement
  (8/10), comparison (18/18, cleanest result), counting (6/6). Same
  heads + task-specific Q-routing proven cross-capability.
- **R44/R45 + R46.2 (facades shipped)**: `HubInjectionCard` wraps
  the R43 intervention (runtime Q/K dispatch, no per-task hand-coding);
  `MultiStepReasoningFacade` extends R11 step-through digit bias to
  N-op compositions → **17/17 real Gemma fixes, 0 regressions**.
  Becomes the 5th L23-hub beneficiary → **5-for-1 compilation ROI**.
- **R47-R50.6 (multi-step composition mapped)**: 7th capability on the
  atlas. L24 SWA Δ=-17.23. **Deep-diffuse** — ruled out at attention
  (R47.4), FFN per-neuron (R48.1), SVD rank (R49.1-5), AND top SAE
  features (R50.5 zero causal effect despite 99.1% reconstruction).
  Not currently compilable by any known substrate mechanism.

## SAE arc detail (R50.1-6)

The SAE infrastructure was built and exercised end-to-end on L24
multi-step composition (parallel track to the arithmetic L23 target).

- **R50.3**: TopK SAE (K=50) reconstructs L24 at 99.1%, effective L0=50
  → L24 composition IS low-rank in a learned feature basis.
- **R50.6**: re-installing the SAE at L24 (99.6% recon) preserves
  arithmetic 100% — end-to-end infra works.
- **R50.5 (the disconfirming result)**: ablating the top-50 SAE
  composition features has **ZERO causal effect** (17/30 baseline →
  17/30 ablated). Reconstruction fidelity does NOT imply causal
  localization on distributed circuits.

**Open problem**: the gap between SAE reconstruction and causal
intervention on distributed representations. Future SAE work for
compilable-intervention purposes needs architectures where reconstructed
components demonstrably carry causal effect (transcoders, attention-SAE,
cross-layer SAE, feature-circuit approaches).

The R20 "target SAE on H4's V output" recommendation for the
arithmetic circuit is NOT falsified — arithmetic is attention-
concentrated and R28-validated at the attention level, so SAE is
orthogonal. What R50.5 falsifies is "SAE = next compilable target
for distributed circuits."

## Why it worked — architectural prediction validated

Gemma's alternating SWA/global attention forces cross-operand
aggregation into global layers (L5, L11, L17, **L23**, **L29**, L35,
L41). Arithmetic requires seeing both operands → must happen at a
global layer. L23 and L29 were architecturally predicted; measurement
confirmed.

Capabilities whose information flow matches Gemma's structure should
localize similarly; capabilities that are inherently distributed
(open-ended semantics, long-range reasoning) will not.

## R-numbered methodology nulls

- **R13 (naive logit lens as sole tracing tool)**: top-5 tokens at
  middle layers = foreign-language / code noise. Only rank trajectories
  of tracked tokens give signal. Use ALONGSIDE activation patching,
  not alone.
- **R14→15 (single-prompt activation patching for localization)**:
  R14 claimed L35 is "THE arithmetic layer" from one prompt (17×23).
  R15 showed it doesn't generalize (only 2/10 flipped). Always
  aggregate across multiple inputs.
- **R14→16 correction (L35 as THE arithmetic circuit)**: L35 mean
  Δ=-1.50, 9/10 hurts. Minor contributor. Real cluster is L22-L30
  with L23 peak (-10.18, 10/10). R14's claim was premature.
- **R20 (per-sub-head d_head=2 ablation as further-localization
  tool)**: 0/256 sub-heads with mean Δ < -1.0; top-8 carries only
  26% of damage. Arithmetic signal in H4 is distributed across the
  512-d V subspace, not sparse in the d_head=2 basis. Don't re-run
  this probe on other heads hoping for sparsity in d_head=2 slots —
  target SAE on the full head/V output instead.
- **R50.5 (standard L1-SAE features as targets for causal ablation
  on distributed composition circuits)**: 99.1-99.6% reconstruction
  AND 370 task-specific feature directions (R50.4) AND zero causal
  effect under top-50 ablation. Reconstruction fidelity does not
  imply causal effect on deep-diffuse circuits. Before targeting SAE
  features for compilation, verify causal effect under ablation
  first. This is the canonical "interpretability-without-causality"
  null.

## Cross-refs

- Current first-principles framing: `.claude/rules/tracing_intelligence.md`
- Full R13-R50.6 narrative + further per-round detail:
  `MEMORY/atlas/tracing_roadmap_part_1.md`
- Augmentation thesis empirical basis: `MEMORY/atlas/augmentation_thesis_arc.md`
- Capability-gain measurement receipts: `MEMORY/atlas/capability_gain_arc.md`
