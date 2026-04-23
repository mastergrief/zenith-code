# Probing Methodology Rules

Mechinterp / activation-level probing discipline for this project. Extends
`workflow.md`'s general hypothesis-test-iterate loop with LLM-specific
methodology gates. Findings are empirical — every rule below was learned
by catching (or nearly missing) a bad interpretation in session 33's
R13-R50 arc.

## Empirical timeline — minutes to hours, NOT "weeks to months"

Published mechinterp literature routinely quotes "weeks to months per
capability." **This project's measured pace on this stack is
minutes-to-hours.** If a step looks like it'll take days, your
methodology or tooling is wrong; revisit before committing the time.

Measured on this project (RTX 4070 Laptop, 8 GB VRAM, WSL2):

| work | literature estimate | actual |
|---|---|---|
| Activation patching across 42 layers × 10 prompts | days | ~14 min |
| Per-head ablation at one layer × 10 prompts | days | ~3 min |
| Q/K/V decomposition on candidate head | days | ~30 min |
| Forced-attention causal validation | weeks | ~20 min |
| Full 6-capability atlas (R13-R43) | weeks/months | ~6 hours |
| SAE training (~8K samples, K=100, 3000 steps) | days | ~3 min on cached |
| Multi-step reverse-engineering arc (R47-R50) | months | ~3 hours |
| Facade ship (R44-R45 HubInjectionCard+generate) | weeks | ~1 hour |
| Multi-step product shipped (R46, 17/17 fixes) | weeks | ~1.5 hours |

What compounds the speed on this stack:
- Prod Gemma substrate with fused tq4 kernels (~100ms/forward)
- Triton + CUDA Graphs for tq4 matmul/dequant
- Hypothesis-test-iterate loop (one question per round)
- Committed artifacts (no context re-acquisition across rounds)
- `bin/gemma-run` daemon (eliminates ~3 min/round reload cost once
  started)
- Disk caching of expensive captures (`/tmp/r50_captures.pt` etc.)
- Monitor + filtered tail for streaming log (notification-driven,
  not polling)

Never let inherited literature estimates set your expectation for the
project's own pace.

## Prompt-format gate (from R47.2 copy-c discovery)

Before interpreting ablation results on an LLM, verify the prompt
actually activates the circuit you think it does — not a shortcut.

- On multi-step arithmetic prompts of the form `"{a} times {b} plus
  {c} equals "`, Gemma 4 E4B frequently just **echoes c** as the
  first answer token (R47.2 finding). The "ablation signal" then
  includes the copy-c shortcut circuit, not pure composition.
- Fix: before interpreting, check baseline-argmax-correct rate on
  the corpus. If < 50% of prompts have argmax = correct first
  token, the prompt activates shortcut circuits, not the target
  capability.
- R47.3 re-ran the sweep with `"What is ({a} * {b}) + {c}? Answer:
  "` (operand c mid-prompt, not trailing). Baseline argmax correct
  jumped from 0/10 to 8/10 — only THEN was the ablation
  interpretable.

Rule: audit the prompt format BEFORE committing to ablation data.
Near-miss shortcut circuits inflate Δ across unrelated layers.

## Task-rank vs PCA-rank dissociation (from R49.2, R50.5)

PCA/SVD rank at 90% explained variance ≠ task-relevant rank.

- **R49.2**: L24 FFN at position -1 has PCA rank 34 at 90% variance.
  But projection-test showed **task-relevant rank = 1** (just the
  mean constant suffices to preserve accuracy).
- **R50.5**: top-50 features ranked by multi/single activation
  ratio (1089× at top) had **zero causal effect** when ablated —
  correlation ≠ causation even when correlation is near-perfect.

Rule: always follow variance analysis with a projection-or-ablation
test that measures ACCURACY preservation, not just reconstruction
quality. Variance-rank is a lower bound on task-rank; ablation is
the actual filter.

## Superposition blinds ablation (from R48.1 → R50.3)

When per-neuron ablation says "diffuse" but full-layer ablation says
"strongly load-bearing", the circuit IS sharp — just not in the
neuron basis.

- **R48.1**: L24 FFN top 25% of neurons carry only 6% of the
  full-layer Δ. Sounds like genuine diffusion.
- **R50.3**: TopK SAE finds the same activations have task-rank 50
  in a learned feature basis, 99.1% reconstruction with L0=50.

Rule: "diffuse at neuron level + strong at layer level" =
superposition suspect. Reach for SAE before concluding the circuit
is uncompilable.

**L1 vs TopK sparsity**: L1-regularized SAE usually fails — R50.1
and R50.2 plateaued at L0~1700 despite 10× λ scaling. TopK SAE
succeeds because hard sparsity breaks through dense feature
correlations. Use TopK first, not L1.

## Tool-tier selection for probing

Pick the tool tier based on the superposition problem:

- **Ablation** (torchlight): points at a layer/head/neuron, asks "is
  anything happening here?" Binary, coarse. Use for initial
  localization.
- **Linear probes** (filtered torchlight): "is *this specific thing*
  readable?" Only finds what you guess to test for. Use to confirm
  a specific hypothesis.
- **PCA/SVD** (prism): shows the raw spectrum of variance but
  doesn't interpret the colors. Use to estimate rank, but don't
  trust the rank as task-relevant (see "Task-rank vs PCA-rank"
  above).
- **SAE** (microscope): overcomplete dictionary learns *which*
  features exist without being told. Unsupervised, high resolution,
  scales to thousands of features. Use when ablation + PCA say
  "diffuse" but full-layer says "load-bearing" — superposition is
  the suspect.

All four are complementary. Ablation narrows the target; SAE sees
the feature basis; causal validation closes the loop. Invest in
SAE AFTER ablation has given you a concrete target — not as a
first pass.

## Causal validation is non-negotiable

SAE reconstruction quality (variance-explained) and task-relevance
are different metrics. R50.6 showed installing a 99.6%-reconstruction
SAE preserves 100% task accuracy — good. R50.5 showed ablating the
top-50 correlation-ranked features has 0% effect on accuracy — bad.
Both are true simultaneously because reconstruction aggregates all
features, while individual-feature importance depends on their
specific causal role.

Rule: after identifying candidate features via correlation,
attribution, or activation difference, **always test them causally
via ablation**. Don't ship "composition-specific features" based on
ratio alone — they may be epiphenomenal.

## Related rules

- `workflow.md` — general hypothesis-test-iterate loop
- `augmentation_thesis.md` — circuit typology (concentrated /
  cooperative / diffuse / pathway-cooperative) and tier-1/2/3
  framework
- `tracing_intelligence.md` — first-principles bounds on what's
  compilable
- `.claude/MEMORY/atlas/tracing_arc_part_1.md` — round-by-round atlas progress
- `.claude/MEMORY/atlas/capabilities.md` — the capability/layer/head reference
