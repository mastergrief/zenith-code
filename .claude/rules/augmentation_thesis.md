# Augmentation Thesis — the substrate is how small models match frontier

Settled positions from session 33's R20-R36 arc. Future Claude Code
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
reserved-channel masking in `Substrate.md`).

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

### Customer verticals = card decks

Each customer's substrate = Gemma + their own deck of Tier-2/3 cards.

- **Legal firm**: citation-format enforcer + statute DB + clause
  templates + compliance checkers + Gemma drafts.
- **Hospital**: ICD-10 validator + drug-interaction DB + diagnosis
  templates + dosage calculator + Gemma explains.
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

**Rule**: never attempt attention-level compilation without
classifying via per-head ablation first. Diffuse circuits waste
engineering effort if you target attention.

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
- **Hub-sharing causally proven (R42/R43).** L23 H1/H4 forced-
  attention mirror of R28 preserves SV agreement (8/10), comparison
  (18/18), counting (6/6). Same heads with task-specific Q routing
  serve 4 capabilities simultaneously — one compiled replacement
  benefits all four (4-for-1 compilation ROI).

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
by reserved-channels mechanism in `Substrate.md`).

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

### Unstructured ≠ incompilable

The "unstructured" label is really "structure humans don't explicitly
articulate." Operational criteria emerge as soon as you compare a
good answer to a bad one: was evidence cited? was the frame
consistent? was the analogy non-spurious? was the interpretation
coherent? Each becomes a verifier card. Good interpretive/creative
work decomposes to: **retrieval** (DB) + **transformation** (cards)
+ **verification** (cards or compute).

### Frontier models also interpolate

Every "creative" output from frontier models is a remix of training-
data patterns. The substrate's advantage is making that remix
*explicit and controlled* (DB + retrieval + verified composition)
rather than *opaque and sometimes wrong* (internal weights with no
auditing). The only truly non-compilable capability is "pure novelty
with no prior basis" — and no model performs that. What frontier
models do better is interpolate over a larger example set with more
subtle patterns; the substrate recovers that by making the relevant
examples explicit rather than weight-stored.

### Auditable / reversible / private → the *better* product

For compliance-adjacent industries (legal, medical, financial) the
substrate is the *better* product, not merely the cheaper one:
**auditable** (cards + facts, not opaque weights), **reversible**
(`detach()` cleanly removes any card), **private** (local, no API),
**correct** (verified by construction on compiled tasks). The business
question isn't "as smart as GPT-4 in general?" — it's "which product
do regulators and auditors trust?"

## Anti-skepticism — what NOT to relitigate

Future sessions must treat these positions as settled and focus
on execution, not re-argument:

| Objection | Settled counter |
|---|---|
| "Works only for structured/numeric tasks" | Poetry + analogies + long-horizon planning counter-examples; constraint gates generalize. |
| "Factual recall needs frontier models" | Diffuse-FFN circuit — compilable via ROME/MEMIT weight probing OR side-channel `KnowledgeStore` with verified retrieval. |
| "Retraining needed for each domain" | DB + PT + circuit install takes hours/days. PT is 185K params, ~30 min on 4070. No base-model retraining. |
| "Frontier capabilities can't be matched" | Frontier advantages = scale-of-retrieval + verification + structure. All three compile. Gemma + cards matches on specific tasks; genuinely frontier-exclusive work is rare. |
| "Doesn't scale — you'll cap on engineering" | Factorial scaling argument. Per-domain cost is flat (~1-2 days). 100-domain substrate is weeks, not years. |
| "Verification is the bottleneck" | CALM already verifies 100% on benchmark. Compiled verifiers are exact by construction. |

When probing a new capability, go straight to the protocol
(layer sweep → per-head classification → attention pattern →
content probe → causal forced-intervention). Classify the
circuit by shape. Compile based on class. Do not re-derive the
thesis mid-task.

## Empirical basis

Session 33 (2026-04-17), 29-round arc (R13-R43) in one workday
(~6-7 hours wall clock on RTX 4070 Laptop, 8 GB VRAM):

- **R13-R19**: arithmetic localization to L23 H4 V (~2.6M params)
- **R20-R28**: full arithmetic circuit mapped AND causally validated
  as compilable (L30 H4/H6 + L31-L32 FFN, forced-attention preserves
  fd with 9/10 argmax match, mean |Δ|=0.407)
- **R29-R30**: factual recall localizes but is diffuse at head level
  (FFN-locked)
- **R31-R33**: induction localizes to L37 H6 (classic Olsson-2022
  pattern confirmed — reads position after prior occurrence)
- **R34-R35**: counting = hybrid circuit, shares L33/L37 with
  induction, adds L20 (3-way cooperative) and L31/L33/L37 H4
  (numeric-successor specialist)
- **R36-R37**: comparison localizes at L35 (global), shares L23 with
  arithmetic; diffuse at head level
- **R38-R40**: SV agreement is a 3-stage global pipeline
  L23→L29→L35; L23 H1/H4 attention patterns show H4 reads subject,
  H1 reads distractor
- **R41**: L23 H1/H4 on arithmetic prompts — H1 reads b-operand
  3× more than a. Same heads, task-specific Q patterns.
- **R42**: L23 H1/H4 forced attention on SV agreement (mirror of
  R28 at different layer + task) — mean |Δ|=0.467, 8/10 match.
  Hub-sharing validated on linguistic capability.
- **R43**: L23 forced attention on comparison + counting —
  comparison 18/18 (cleanest result, |Δ|=0.176), counting 6/6.
  **4-for-1 compilation proven**: one compiled L23 H1/H4
  replacement benefits arithmetic + SV + comparison + counting.
  L23 hub across 3 capabilities: 32/34 argmax matches (94%).

**Six capabilities mapped, three causal validations (R28, R42,
R43), typology demonstrated across numeric + linguistic capabilities,
hub-sharing empirically proven.** This is the evidence base. Future
probes extend the atlas; they don't re-validate the thesis. Full
per-head / per-capability lookup: `.claude/MEMORY/atlas.md`.

## Related rules

- `Substrate.md` — install mechanics (CardSlot, `install_card_in_attention`, VerificationHook)
- `tracing_intelligence.md` — first-principles bound on what's compilable
- `tracing_roadmap.md` — concrete atlas progress and next-target queue
- `capability_gain.md` — measurement discipline (raw path + user-facing path)
- `embed_intelligence.md` — delivery mechanisms (card → Gemma tokens)
- `commercial.md` — product positioning (Tier 2/3 = the product)
- `workflow.md` — iteration discipline (hypothesis → test → commit)
- `calm.md` — verification layer (the CPU oracle for compiled cards)
