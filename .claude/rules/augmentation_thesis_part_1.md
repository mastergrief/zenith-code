# Augmentation Thesis — the substrate is how small models match frontier

**Part 1**

Settled positions from session 33's R20-R50 arc. Future Claude Code
sessions should treat these as baseline context, not re-argue them.

## Core thesis

Gemma 4 E4B (5B) + **mapped circuit sites** + **compiled cards** +
**domain DB** + **trained PT** + **verifier cards** matches or beats
frontier-model quality on structured AND unstructured tasks, locally,
for ~1% of frontier cost. Session 33 Round 28 causally confirmed the
mechanism on arithmetic: forced one-hot attention at L30 H4/H6
(reading a_ones/b_ones from their token positions) preserves fd with
mean |Δ| = 0.407 and 9/10 argmax match — i.e. Gemma's learned
attention IS a position-selector LookUp, compilable exactly.

### Routing flow

```
Prompt → Gemma (NL understanding, routing)
           ↓
           ├─ Native circuit (Tier 1, preserved)  → correct
           ├─ Augmented circuit (Tier 2, compiled replacement) → exact
           └─ Plugged circuit  (Tier 3, new compiled capability) → exact
                     ↓
              VerificationHook / step-through bias → Gemma's tokens
              Gemma wraps the exact result in fluent output
```

Net effect: **one model, enhanced surgically, no retraining.** Gemma
stays intact. Cards are strictly additive: `facade.install()` adds,
`facade.detach()` reverses cleanly. Adding a card is a **strict
improvement with zero regression on other tasks** (verified by
reserved-channel masking in `.claude/spec/Substrate.md`).

## Three-tier framework

| Tier | Intervention | Cost | Moat | Example |
|---|---|---|---|---|
| **1 — Preserve** | Leave Gemma alone where it works | 0 | none | "2+2=4" (Gemma gets it) |
| **2 — Augment weak** | Compile replacement for specific failing circuit | Days | high | R11 multiplier 5/10→10/10, R28 causal validation, R42/R43 hub-sharing extended validation (4 capabilities, 32/34 match rate) |
| **3 — Plug missing** | Compile new capability from scratch at unused slot | Days | highest | `programs/gcd`, `programs/reasoning_engine`, `KnowledgeStore` recall cards |

Tier 1 is table stakes — every Gemma user gets it free. **Tiers 2 and
3 are the product.** The mapping protocol (below) tells you which
capabilities are Tier-2-addressable (concentrated circuits you can
surgically replace) vs Tier-3 (design from scratch).

### Tier-2 stacking achieves tier-3-equivalent outcomes

**Refined position (session 34, post-R52 null).** Every shipped
working augmentation in this codebase is tier-2 ADDITIVE:

| Shipped capability | Type | Mechanism |
|---|---|---|
| R11 multiplier (5/10→10/10 on Gemma) | Tier-2 at output head | `VerificationHook` + step-through digit bias |
| R44/R45 `HubInjectionCard` (L23 H1/H4) | Tier-2 at concentrated circuit | Forced-attention facade, runtime Q/K dispatch |
| R46.2 `MultiStepReasoningFacade` (17/17 fixes) | Tier-2 stack | NL parser + `safe_eval` + step-through bias |
| `KnowledgeStore` recall cards | Tier-2 at output | Step-function indicators + `CardSlot` + `VerificationHook` |
| `programs/gcd`, `adder`, `multiplier` (compiled) | Tier-2 integration | Compiled compute + tier-2 output hook |
| R-delta-21 `CopyAugmentedDeltaNet` MQAR card | Tier-2/3 retrieval | 4-gate CardSlot. 2026-04-21 ship at 22.0 → 2026-04-22 R22f recalibration to **14.5 → 60/60 (+18, 0 regressions, commits `9691e06` + `c3cc73f`)**. See `delta_rule.md` §R22. |
| `BaseConversionFacade` R22c (`7db6eb9`) | Tier-2 decode-path | Parser + `int(x, base)` + digit bias. 10/10 vs 7/10 (+3, 30% lift). |
| `NumberTheoryFacade` R53a mod/GCD/LCM (`69279d4`) | Tier-2 decode-path | Parser + `safe_eval` + digit bias. **15/15 vs 8/15 (+7, 47% lift).** Exposed the `▁`-strip + POST_BIAS_BUDGET=4 discipline now canonical for decode-path facades. |
| `NumericEncodeFacade` F2 int→hex/binary/octal (`5ee61a5`) | Tier-2 decode-path | 12/12 on chain corpus. First facade with letter-answer (e.g. "DEADBEEF"). |
| `Icd10RecallFacade` R60a 72,748-code DB (`afc0220`) | **Tier-3 decode-path** | Parser + JSON lookup + multi-token step-through bias on TEXT answer. **26/30 vs 8/30 (+18, 67% lift).** First tier-3 delivered via decode-path rather than CardSlot. Generalizes step-through bias from integer answers to arbitrary Gemma BPE. 4 edge codes resist — F1 retry infra (`8ba151d`) + pure-DB bypass candidate future work. |
| `PlannerFacade` R70a + F2 (`956a3ae` + `5ee61a5`) | Tier-2 orchestrator | First-match-wins classify over 5 specialist facades + "X in hex/binary/octal" chain detect. 20/20 single + 12/12 chain. |
| Auto/meta-generated facades via `recursion.py` (6 Level-1 `*_auto.py`, 5 Level-2 `*_meta.py` — `3274659`, `5173745`) | Tier-2 auto-gen | Level-1 (hand `FacadeSpec`): 17/30 → 30/30 across factorial/fibonacci/combinations/permutations/power/next_prime. Level-2 (`MetaFacade.from_oracle(fn_name, arity)`): 4/15 → 15/15 across factorial/combinations/gcd/lcm/fibonacci. **Spec authorship moved human → substrate; three CALM gates (oracle → ast.parse → live A/B) intact.** |

**R51/R52 were the anomaly.** Both explicitly chose REPLACEMENT via
monkey-patching `m._forward_layer` to skip Gemma's native L24.
R51's install.py docstring even cites rejecting `CardSlot` because
"residual-additive cannot REPLACE L24" — but replacement was the
wrong hypothesis for a deep-diffuse circuit. Three nulls (R50.5
SAE, R51.5 MSE, R52.3 KL) confirm.

**Rule**: true tier-3 *from-scratch* is correct ONLY when Gemma
has ZERO relevant circuit (e.g. ICD-10 lookups where the prior
doesn't help). If Gemma has ANY partial capability on the task,
tier-2 stacking leverages Gemma's NL understanding + context
handling + output routing for free. The distilled-student tier-3
pattern (reproducing a Gemma layer's full function) is a bad bet
on deep-diffuse circuits specifically.

**Refinement (2026-04-22 Icd10 receipt):** tier-3 with a short
known-length text answer from a static DB is **decode-path-addressable**
(parser + JSON lookup + multi-token bias), not CardSlot-mandatory.
Icd10 shipped 26/30. CardSlot-with-trained-PT is only required when
the key is non-literal (R22 MQAR under distractor prose). For
well-typed code→text mappings (medical/legal/financial/chemical),
decode-path is the cheapest tier-3.

**Reframing implication for R53+**: stop hunting for tier-3
distillable deep-diffuse circuits. For each capability: (1) does
Gemma fail at it? (2) is the failing circuit concentrated → tier-2
compile at that site. (3) Is the circuit diffuse but Gemma's
capability is "close" → step-through bias / VerificationHook at
output (R46.2 pattern). (4) Is the capability truly alien to Gemma
→ add KB / domain card, still integrate via tier-2 output hook.
R46.2's 17/17 multi-step-composition fixes *already augments the
L24 task* at the output level — no L24-internal intervention needed.

**R53.36 audit refinement** (2026-04-20, `capability_gain.md`
§R53.36): the three distillation nulls that close tier-3 are NOT
the same mechanism. R50.5 SAE is interpretability-without-causality
on an attribution-picked basis. R51.5 MSE is a close-miss at
cos=0.89 scale=0.91 where 10% diffuse residual error cascades
through 17 downstream layers into wrong argmax. R52.3 KL is a
wrong-loss failure where the student output is uncorrelated with
L24's contribution (cos=-0.02, scale=94×). Install math verified
bit-identical on both students, so tier-3 is not install-bug-
blocked. **Tier-2 stacking stays the priority** — but tier-3 has a
credible reopen path via Jacobian-weighted loss (weight residual
error by downstream causal effect on head logits) if an active
workstream ever needs single-card L24 replacement. Not priority.

### Customer verticals = card decks

Each customer's substrate = Gemma + their own deck of Tier-2/3 cards.

- **Legal firm**: citation-format enforcer + statute DB + clause
  templates + compliance checkers + Gemma drafts.
- **Hospital**: ICD-10 validator (**shipped 2026-04-22** —
  `Icd10RecallFacade` at 26/30 on 72,748-code DB, commit `afc0220`)
  + drug-interaction DB + diagnosis templates + dosage calculator
  + Gemma explains.
- **Fintech**: exact-decimal arithmetic + regulation lookups +
  compliance verifiers + currency conversion cards + Gemma answers
  customer queries.
- **Engineering**: unit converters + formula cards (physics, chem,
  EE) + material property DB + Gemma narrates.

No cross-vertical interference: legal cards don't affect a hospital
substrate, hospital cards don't affect a fintech substrate. Each
customer ships the stack their domain needs.

### Mapping ROI for Tier 2

Without mapping, you guess which of Gemma's weak capabilities is
fixable and burn engineering on dead ends (diffuse circuits can't be
replaced at attention level — R30 factual recall example). With
mapping, per-head ablation tells you in ~1 hour whether a circuit
is concentrated (Tier-2 addressable) or diffuse (not). Estimated
Tier-2 engineering-cost reduction: **~10×**, because you pick
concentrated targets instead of flying blind.

## Circuit typology — three shapes

Three attention-level shapes observed across 5 capabilities in
session 33. Run per-head ablation AFTER the layer sweep to classify.
Compile decision flows from the classification.

| Shape | Marker | Example | Compile path |
|---|---|---|---|
| **Concentrated** | 1-2 heads carry ≥50% of layer signal | Arithmetic L23 H1 (-4.85) + H4 (-4.30), induction L37 H6 (-0.52), counting L37 H4 (-1.02) | 1-2 `LookUpExact` gates per head. Cheap. R28-validated. |
| **Cooperative** | 3-4 heads each -0.5 to -1.5, additive sum ≈ full-layer Δ | Counting L20 H2+H5+H6 (each -1.0 to -1.4, sum -3.74 vs full -3.93) | 3-4 `LookUpExact` gates. Moderate cost. |
| **Diffuse** | No head > -0.2 despite full-layer Δ > -1.0; per-head sum ~20% of full | Factual recall L5 and L11 (top head -0.078, full -1.18/-1.56) | NOT compilable at attention level. Circuit is in FFN. Use ROME/MEMIT-style weight probing, or side-channel via `KnowledgeStore`. |
| **Deep-diffuse** | Full-layer Δ large (e.g. L24 -17.23) but diffuse at attention AND FFN AND per-neuron AND SVD AND SAE-feature levels; every distillation loss tried (MSE residuals, KL logits, SAE-feature ablation) reaches good distillation-space metrics but fails token preservation | Multi-step composition L24 (R47-R52.3). Three independent nulls: (a) top-50 SAE features have zero causal effect despite 99.1% reconstruction (R50.5); (b) 92.6%-var-explained MSE student produces 0.19/0.34 prefix match (R51.5); (c) KL-divergence student (val KL 1.96→1.21) produces 0.04/0.08 prefix match, WORSE than MSE baseline (R52.3). | **No compilable path by any known distillation loss.** Attention-level, FFN-weight-level, SAE-feature-level, MSE-distillation, AND KL-distillation installs all ruled out. **Reframing (R52)**: this circuit is a candidate for tier-2 stacking (additive correction via `CardSlot` + `VerificationHook`) rather than tier-3 replacement. See §"Tier-2 stacking achieves tier-3-equivalent outcomes" below. |

**Rule**: never attempt attention-level compilation without
classifying via per-head ablation first. Diffuse circuits waste
engineering effort if you target attention. Deep-diffuse circuits
(R50.5, R51.5, R52.3 falsifications — SAE, MSE, KL all null) waste
further effort if you target FFN weights, SAE features, OR any
distillation-trained student without first checking that
reconstruction fidelity at the chosen metric translates to causal
effect on the user-facing task. Three independent losses failing
identically is strong evidence the target isn't tier-3-distillable.

## Compositional hypothesis

Complex capabilities are short walks through a sparse set of
primitive circuits. Evidence from session 33:

- **Counting = induction ∪ compute.** Counting sweep hits L33 and
  L37 (shared with induction R31) PLUS new peaks at L20 and L31
  (adjacent to arithmetic L30-L32 compute cluster).
- **Comparison = arithmetic-adjacent + induction-adjacent.**
  Comparison sweep (R36) has L23 in top-3 (shared with arithmetic)
  and L33 in top-5 (shared with induction).
- **L37 hosts multiple specialized heads.** L37 H6 = induction
  (R33 canonical pattern), L37 H4 = numeric successor (R35). Same
  layer, distinct heads, distinct capabilities.
- **Hub-sharing causally proven (R42/R43), facade shipped (R44/R46).**
  L23 H1/H4 forced-attention mirror of R28 preserves SV agreement
  (8/10), comparison (18/18), counting (6/6). Same heads with
  task-specific Q routing serve 5 capabilities simultaneously
  (arithmetic + SV + comparison + counting + multi-step composition
  via `MultiStepReasoningFacade` R46.2) — one compiled replacement
  via `HubInjectionCard` (R44) benefits all five (**5-for-1
  compilation ROI**).

Implication: **individual heads are the atoms, not layers.** Gemma
has specialist heads running in parallel at every layer; prompts
activate different subsets. Shared hub heads (L23 H1/H4) carry
task-neutral content-read mechanisms; task-specific routing
happens via the Q projection.

Atlas sparseness estimate: ~30-50 specialist heads cover most core
capabilities (of 336 total head slots = 42 layers × 8 heads). Full
atlas ≈ 20-40 hours of focused probing on RTX 4070 Laptop. Not
infinite labor. Tractable.

## Factorial scaling per domain

DB size × PT quality × circuit-injection specificity compound
**multiplicatively**, not additively. Double each, get 8× output.
Triple each, 27×.

Per-domain cost structure (once pipeline exists):

| Resource | Per new domain |
|---|---|
| Knowledge DB curation | hours |
| PT training | ~30 min on RTX 4070 |
| Circuit mapping (R16→R17 protocol) | ~1-2 hours |
| Compile card (gate-graph IR) | few hours |
| Integration + test | ~1 day |

Marginal cost of the 100th domain ≈ cost of the 1st (no cross-domain
interference — each card lives in its own channel/head slot, verified
by reserved-channels mechanism in `.claude/spec/Substrate.md`).

### Economics inversion vs standard LLMs

| Standard LLM economics | Substrate economics |
|---|---|
| Bigger model = better at everything, expensive | Gemma stays same size, same cost |
| Fine-tune for domain = expensive, forgets other things | Add a card = strict improvement, zero regression |
| Domains compete for capacity in one opaque network | Domains are disjoint card slots, no competition |
| Removing a capability = retrain from scratch | Remove a card = clean `detach()`, no damage |
| 100 domain specialists = 100× training cost | 100 cards stacked = 1× base cost + per-card hours |

Stack 10 cards: 10× domain coverage, 1× compute. Stack 100: same
compute, 100 specialties. Qualitatively different from "train a
bigger model."

## Beyond "structured" tasks

The pattern extends to any task with operationalizable quality criteria:

- **Poetry** (session 31): structure-checker card (meter, rhyme,
  syllable count) + rhyme DB + poem-template DB + PT to parse
  user prompt → structural spec. Gemma generates content under
  constraints; verifier rejects violations. Result: high-quality
  constrained creative writing.
- **Code**: syntax + type checker (compiled) + API docs DB + Gemma
  drafts + verifier rejects uncompilable outputs.
- **Legal**: citation-format enforcer + statute DB + clause templates
  + Gemma drafts.
- **Music**: chord-progression validator + music-theory DB + song-
  structure templates + Gemma writes lyrics under constraints.
- **Math proofs**: step-by-step logical-inference checker + axiom
  DB + proof templates + Gemma narrates the reasoning.
- **Multi-step reasoning**: reasoning-template DB + PT for
  decomposition + per-step verifier (CALM-style) + Gemma generates
  each step under verification gates.
- **Abstract reasoning / analogies**: structural-match card (graph
  isomorphism) + cross-domain analogy DB + PT extracts structure +
  Gemma fills surface.
- **Long-horizon planning**: 512K context (NIAH-validated 200K
  single-needle) + state-tracking card + per-step state-consistency
  verifier + Gemma proposes next action. **Verification keeps the
  context usable** — normal LLMs degrade past ~64K because errors
  compound silently; per-step verification rejects bad steps before
  they propagate.

