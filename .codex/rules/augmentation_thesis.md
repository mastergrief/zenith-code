# Augmentation Thesis — substrate is how small models match frontier

Settled positions. Future sessions should treat these as baseline
context, not re-argue them.

> Historical receipts (per-round empirical basis, shipped-capability
> table with commits + dates, distillation null detail,
> tier-1 preservation eval results, capability map): see
> `MEMORY/atlas/augmentation_thesis_arc.md`.

## Core thesis

Gemma 4 E4B (5B) + **mapped circuit sites** + **compiled cards** +
**domain DB** + **trained PT** + **verifier cards** matches or beats
frontier-model quality on structured AND unstructured tasks, locally,
for ~1% of frontier cost.

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

Net effect: **one model, enhanced surgically, no retraining.** Cards
are strictly additive: `facade.install()` adds, `facade.detach()`
reverses cleanly. Adding a card is a strict improvement with zero
regression on other tasks.

## Three-tier framework

| Tier | Intervention | Cost | Moat |
|---|---|---|---|
| **1 — Preserve** | Leave Gemma alone where it works | 0 | none |
| **2 — Augment weak** | Compile replacement for specific failing circuit | Days | high |
| **3 — Plug missing** | Compile new capability from scratch at unused slot | Days | highest |

Tier 1 is table stakes — every Gemma user gets it free. **Tiers 2
and 3 are the product.** The mapping protocol tells you which
capabilities are Tier-2-addressable (concentrated circuits to
surgically replace) vs Tier-3 (design from scratch).

## Tier-2 stacking achieves tier-3-equivalent outcomes

**Refined position**: every shipped working augmentation in this
codebase is tier-2 ADDITIVE — `VerificationHook` + step-through
digit bias on output, hub-injection facades at concentrated circuits,
NL parser + `safe_eval` + step-through bias for multi-step composition,
recall cards via `CardSlot` + `VerificationHook`, decode-path facades.

Default hypothesis for any new capability: tier-2 stacking. True
tier-3 *from-scratch* is correct ONLY when Gemma has ZERO relevant
circuit (e.g. ICD-10 lookups where the prior doesn't help). If Gemma
has ANY partial capability on the task, tier-2 stacking leverages
Gemma's NL understanding + context handling + output routing for free.

**Tier-3 with short known-length text answer from a static DB is
decode-path-addressable** (parser + JSON lookup + multi-token bias),
not CardSlot-mandatory. CardSlot-with-trained-PT is only required when
the key is non-literal (NIAH-style retrieval under distractor prose).
For well-typed code→text mappings (medical/legal/financial/chemical),
decode-path is the cheapest tier-3.

**Reframing for new capabilities**: for each:
1. Does Gemma fail at it?
2. Failing circuit concentrated → tier-2 compile at that site.
3. Circuit diffuse but Gemma's capability is "close" → step-through
   bias / `VerificationHook` at output.
4. Capability truly alien to Gemma → add KB / domain card, integrate
   via tier-2 output hook.

The distilled-student tier-3 pattern (reproducing a Gemma layer's
full function on deep-diffuse circuits) is a bad bet — three
distillation losses (SAE-feature ablation, MSE residuals, KL logits)
all failed identically.

## Circuit typology — three shapes (+ deep-diffuse)

Run per-head ablation AFTER the layer sweep to classify. Compile
decision flows from the classification.

| Shape | Marker | Compile path |
|---|---|---|
| **Concentrated** | 1-2 heads carry ≥50% of layer signal | 1-2 `LookUpExact` gates per head. Cheap. |
| **Cooperative** | 3-4 heads each -0.5 to -1.5, additive sum ≈ full-layer Δ | 3-4 `LookUpExact` gates. Moderate cost. |
| **Diffuse** | No head > -0.2 despite full-layer Δ > -1.0 | NOT compilable at attention level. Use ROME/MEMIT-style FFN weight probing, or side-channel via `KnowledgeStore`. |
| **Deep-diffuse** | Full-layer Δ large but diffuse at attention AND FFN AND per-neuron AND SAE-feature levels | Not tier-3-distillable at known loss spaces. Pivot to tier-2 stacking (additive correction at output). |

**Rule**: never attempt attention-level compilation without
classifying via per-head ablation first. Diffuse circuits waste
engineering effort if you target attention. Deep-diffuse circuits
waste further effort if you target FFN weights, SAE features, OR any
distillation-trained student without first checking that
reconstruction fidelity at the chosen metric translates to causal
effect on the user-facing task.

## Compositional hypothesis

Complex capabilities are short walks through a sparse set of
primitive circuits. **Individual heads are the atoms, not layers.**
Gemma has specialist heads running in parallel at every layer;
prompts activate different subsets. Shared hub heads carry
task-neutral content-read mechanisms; task-specific routing happens
via the Q projection.

One compiled hub-injection facade can serve N capabilities
simultaneously — when the same heads serve multiple tasks via
task-specific Q routing, replacing those heads is N-for-1
compilation ROI.

Atlas sparseness estimate: ~30-50 specialist heads cover most core
capabilities (of 336 total head slots = 42 layers × 8 heads). Full
atlas ~20-40 hours of focused probing on RTX 4070 Laptop. Tractable.

## Factorial scaling per domain

DB size × PT quality × circuit-injection specificity compound
**multiplicatively**, not additively. Double each, get 8× output.

Per-domain cost structure (once pipeline exists):

| Resource | Per new domain |
|---|---|
| Knowledge DB curation | hours |
| PT training | ~30 min on RTX 4070 |
| Circuit mapping (layer-sweep + per-head protocol) | ~1-2 hours |
| Compile card (gate-graph IR) | few hours |
| Integration + test | ~1 day |

Marginal cost of the 100th domain ≈ cost of the 1st (no cross-domain
interference — each card lives in its own channel/head slot).

### Economics inversion vs standard LLMs

| Standard LLM | Substrate |
|---|---|
| Bigger model = better at everything, expensive | Gemma stays same size, same cost |
| Fine-tune for domain = expensive, forgets other things | Add a card = strict improvement, zero regression |
| Domains compete for capacity in one opaque network | Domains are disjoint card slots |
| Removing a capability = retrain from scratch | Remove a card = clean `detach()`, no damage |
| 100 domain specialists = 100× training cost | 100 cards stacked = 1× base cost + per-card hours |

## Customer verticals = card decks

Each customer's substrate = Gemma + their own deck of Tier-2/3 cards.

- **Legal**: citation-format enforcer + statute DB + clause templates + compliance checkers + Gemma drafts
- **Medical**: ICD-10 validator + drug-interaction DB + diagnosis templates + dosage calculator + Gemma explains
- **Fintech**: exact-decimal arithmetic + regulation lookups + compliance verifiers + currency conversion + Gemma answers
- **Engineering**: unit converters + formula cards + material property DB + Gemma narrates

No cross-vertical interference: legal cards don't affect a hospital
substrate. Each customer ships the stack their domain needs.

## Beyond "structured" tasks

The pattern extends to any task with operationalizable quality criteria:
poetry (structure-checker + rhyme DB + PT), code (syntax + type
checker + API docs DB), legal (citation enforcer + statute DB), music
(chord-progression validator + theory DB), math proofs (step-by-step
inference checker + axiom DB), multi-step reasoning (template DB + PT
decomposition + per-step verifier), abstract reasoning (structural-
match card + analogy DB + PT extracts structure), long-horizon
planning (state-tracking card + per-step state-consistency verifier).

**Verification keeps long context usable** — normal LLMs degrade
past ~64K because errors compound silently; per-step verification
rejects bad steps before they propagate.

### Unstructured ≠ incompilable

The "unstructured" label is really "structure humans don't explicitly
articulate." Operational criteria emerge as soon as you compare a
good answer to a bad one. Each becomes a verifier card. Good
interpretive/creative work decomposes to: **retrieval** (DB) +
**transformation** (cards) + **verification** (cards or compute).

### Frontier models also interpolate

Every "creative" output from frontier models is a remix of training-
data patterns. The substrate's advantage is making that remix
*explicit and controlled* (DB + retrieval + verified composition)
rather than *opaque and sometimes wrong* (internal weights with no
auditing). What frontier models do better is interpolate over a
larger example set; the substrate recovers that by making relevant
examples explicit rather than weight-stored.

### Auditable / reversible / private = the *better* product

For compliance-adjacent industries (legal, medical, financial) the
substrate is the better product, not merely the cheaper one:
**auditable** (cards + facts, not opaque weights), **reversible**
(`detach()` removes any card), **private** (local, no API),
**correct** (verified by construction on compiled tasks). The business
question isn't "as smart as GPT-4 in general?" — it's "which product
do regulators and auditors trust?"

## Automatic Tier-1 preservation as substrate property

Substrate RAG via `KnowledgeStore` recall card at L30 has a structural
advantage over prompt-RAG that vanilla retrieval pipelines cannot
match — **automatic Tier-1 preservation via hash-gated injection**.

Hash-match lookup at L30: problem hash → stored key match → inject
verified solution into residual channels. Miss → zero output written
→ Gemma's L31..L41 proceeds with native residual (no intervention).

Automatic gating with zero policy logic. No probabilistic confusion
about when to trust retrieval. No prompt-length inflation. No
imitation-of-wrong-style risk.

| Aspect | Prompt RAG | Substrate RAG (L30 card) |
|---|---|---|
| Gate condition | always injects | hash-match only |
| Strong-prior preservation | disrupted | preserved by construction |
| Context budget | ~600 tokens eaten | zero tokens |
| Tier-1 adherence | violated | automatic |
| Content delivery | text through all 42 layers | direct residual write at L30 |
| Determinism | stochastic | compiled step-function exact |

**Install mechanism caveat**: hash-match Tier-1 holds at the OUTPUT
boundary (`VerificationHook` with small vocab_mapping + `min_margin`).
Does NOT hold for residual-write `CardSlot` at arbitrary layers.
First-token bias is the wrong intervention for code (Gemma's first
token on code is uniformly confident — confidence-gate doesn't fire).
Correct tier-2 for code: post-generation AST walker that mechanically
rewrites, no Gemma in the repair loop.

For prompt-RAG systems (not substrate): add explicit confidence
gating to retrieval — CALM-precompute-found → suppress retrieved
examples; intent classifier detects strong pattern → skip retrieval;
top-k below threshold → skip. Manually replicates what substrate
RAG gets for free.

### Commercial positioning sharpens

"RAG that knows when not to retrieve" is a different product from
"RAG with a bigger DB." Regulated industries specifically need
intervention-when-warranted, not intervention-always — spurious
injections drift output from user intent. **Selective-intervention
property is substrate-native** and hard to reproduce with vanilla
RAG pipelines.

## Anti-skepticism — what NOT to relitigate

| Objection | Settled counter |
|---|---|
| "Works only for structured/numeric tasks" | Poetry + analogies + long-horizon planning counter-examples; constraint gates generalize. |
| "Factual recall needs frontier models" | Diffuse-FFN circuit — compilable via FFN weight probing OR side-channel `KnowledgeStore` with verified retrieval. |
| "Retraining needed for each domain" | Tier-2 stacking takes hours/days per domain. PT is ~185K params, ~30 min. No base-model retraining. |
| "Frontier capabilities can't be matched" | Frontier advantages = scale-of-retrieval + verification + structure. All three compile. |
| "Doesn't scale — you'll cap on engineering" | Factorial scaling argument. Per-domain cost is flat. 100-domain substrate is weeks, not years. |
| "Verification is the bottleneck" | CALM verifies 100% on benchmark. Compiled verifiers are exact by construction. |

When probing a new capability, go straight to the protocol (layer
sweep → per-head classification → attention pattern → content probe
→ causal forced-intervention). Classify the circuit by shape. Compile
based on class. Do not re-derive the thesis mid-task.

## Related rules

- `Substrate.md` — install mechanics (CardSlot, in-attention, VerificationHook)
- `compute_facades.md` — decode-path tier-2 card pattern (zero-VRAM)
- `delta_rule.md` — DT (PT+Delta) trained-card architecture + retrieval-card install pattern
- `tracing_intelligence.md` — first-principles bound on what's compilable
- `capability_gain.md` — measurement discipline (raw + user-facing)
- `embed_intelligence.md` — card → Gemma token delivery
- `commercial.md` — product positioning (Tier 2/3 = the product)
- `workflow.md` — iteration discipline (hypothesis → test → commit)
- `calm.md` — verification layer (CPU oracle for compiled cards)
- `MEMORY/atlas/augmentation_thesis_arc.md` — empirical basis + per-round receipts
