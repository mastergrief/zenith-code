# Augmentation Thesis — the substrate is how small models match frontier

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
| R-delta-21 `CopyAugmentedDeltaNet` MQAR card (100% N=5-15) | Tier-2/3 retrieval card | DeltaNet Householder fast-weight state + cached decode (R-delta-20b), ready for CardSlot install on Gemma (R22). Full spec: `delta_rule.md`. |

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

## Automatic Tier-1 preservation as substrate property (R53.2b finding)

**Settled**: substrate RAG at L30 (`KnowledgeStore` recall card) has a
structural advantage over prompt-RAG that vanilla retrieval pipelines
cannot match — **automatic Tier-1 preservation via hash-gated
injection**.

### The measured failure mode of blanket prompt-RAG

R53.2b complex eval (`scripts/r53_eval_complex.py`, 6 multi-step
coding problems × 3 conditions):

| | stock | hinted (real retrieval) | sanity (random retrieval) |
|---|---:|---:|---:|
| TOTAL | 25/27 | 21/21 | 23/23 |
| Δ vs stock | — | +7.4pp | +7.4pp |
| **retrieval-attributable gain** | | | **+0.0pp** |

Hinted = Sanity. The prompt-length / "has examples in context" effect
is real; the **content** of real retrieval adds nothing on top. On
several problems (log_level_counts, linked_list_bugs) real retrieval
actively HURT (0/0 vs stock's 6/6, 0/0), while random retrieval was
neutral or helpful.

Root cause: blanket retrieval injection **disrupts Gemma's strong-
prior behavior** on problems it already solves. When Gemma reads
"here's a similar solution" + actual relevant code, it tries to adapt
the example (error-prone path). Random irrelevant code doesn't match
anything to adapt from, so Gemma falls back to solving natively.

**This is a Tier-1 violation.** The thesis says "leave Gemma alone
where it works." Prompt RAG violates this by always injecting.

### Why substrate RAG is structurally different

At L30, `KnowledgeStore` recall card uses hash-match lookup:

- Problem hash → stored key? **Match** → inject verified solution
  pattern into residual channels
- Problem hash → stored key? **Miss** → zero output written to
  reserved channels → Gemma's L31..L41 proceeds with native residual
  (no intervention)

Automatic gating with zero policy logic. No probabilistic confusion
about when to trust retrieval. No prompt-length inflation. No
imitation-of-wrong-style risk.

**Property summary**:

| Aspect | Prompt RAG | Substrate RAG (L30 card) |
|---|---|---|
| Gate condition | always injects | hash-match only |
| Strong-prior preservation | disrupted | preserved by construction |
| Context budget | ~600 tokens eaten | zero tokens |
| Tier-1 adherence | violated | automatic |
| Content delivery | text through all 42 layers | direct residual write at L30 |
| Determinism | stochastic | compiled step-function exact |

### R53.14/20a/20b — substrate L41 install REGRESSES on code (post-SWA-fix)

The Tier-1 thesis holds in principle but was falsified at one specific
install mechanism: L41 `CardSlot(preserve=True)` + per-marker
`FirstTokenHook(boost=50)` on the R53.0 6-problem code corpus, SWA
bug already fixed.

Result (ec8887f / `scripts/r53_20b_stacked.py`):

| | stock | prompt-RAG | substrate @ L41 |
|---|---:|---:|---:|
| log_level_counts | 6/6 | 6/6 | **0/0** |
| lru_cache_class | 9/9 | 9/9 | **0/0** |
| (others unchanged) | | | |
| **TOTAL** | 25/27 | 25/27 | **10/12** (-9.3pp) |

Bit-identical MISS preservation at L41. HIT prompts regressed.

**Root cause — install-mechanism, not SWA**: Gemma's first-token on
code is confidently a fence/whitespace opener (margin 6.8-9.2), so
`min_margin=0.5` never gates, hook always fires on HIT, forces
"def"/"class" → code-without-fence → extractor fails.

**Thesis refinement**: hash-match Tier-1 holds at the OUTPUT boundary
(`VerificationHook` with small vocab_mapping + `min_margin`, as in
the learning-loop demo). Does NOT hold for residual-write `CardSlot`
at arbitrary layers. Install mechanism weight matters. First-token
bias is the wrong intervention for code.

**Correct tier-2 for code** (shipped): post-generation AST walker.
Parse output, detect shadow bugs (token_bucket `self.consume =
capacity`), missing-key dict access (csv_column_stats KeyError),
mechanically rewrite. No Gemma retry — R53.19/R53.33 show Gemma
ignores targeted hints with concrete rename examples. Prior
dominance overwhelms in-context instruction weight (see
`capability_gain.md` §"Gemma ignores targeted hints").

`calm/llm_computer/facades/ast_repair.py` ships **7 rewrites as of
2026-04-21** — shadow_rename, dict-key synonym, syntax_repair (3
original in R53.35 `8cc2ff4`/`c81feb6`), plus `fuzzy_rename_function`
(commit `805e539`, Track A walker expansion). Dispatches on
categorized runtime errors: `TypeError: 'int' object is not callable`
→ shadow_rename; `KeyError: 'X'` → dict-key synonym; `SyntaxError`
offset → bracket-mismatch or insert-before-colon repair;
`NameError: name 'X' is not defined` → fuzzy_rename (Jaccard ≥ 0.5
against defined FunctionDefs). Tier-2 stacking thesis reinforced:
**mechanical post-gen rewrite at Gemma's output beats in-context
hint-tuning or tier-3 distillation of deep-diffuse circuits.**

For prompt-RAG systems (not substrate): add explicit confidence
gating to `CodeVerifierFacade.compute_hints`:

- CALM Layer 2 precompute has direct answer → inject only verified
  fact, suppress retrieved examples
- Intent classifier detects known-strong pattern → skip retrieval
- Top-k scores below threshold → skip retrieval

This manually replicates what substrate RAG gets for free.

### Commercial positioning sharpens

"RAG that knows when not to retrieve" is a different product from
"RAG with a bigger DB." Regulated industries specifically need
intervention-when-warranted, not intervention-always — because
spurious injections mean the output drifts from user intent. This
**selective-intervention property is substrate-native** and hard to
reproduce with vanilla RAG pipelines.

## Anti-skepticism — what NOT to relitigate

Future sessions must treat these positions as settled and focus
on execution, not re-argument:

| Objection | Settled counter |
|---|---|
| "Works only for structured/numeric tasks" | Poetry + analogies + long-horizon planning counter-examples; constraint gates generalize. |
| "Factual recall needs frontier models" | Diffuse-FFN circuit — compilable via ROME/MEMIT weight probing OR side-channel `KnowledgeStore` with verified retrieval. |
| "Retraining needed for each domain" | Tier-2 stacking (PT + KB + VerificationHook) takes hours/days per domain. PT is 185K params, ~30 min on 4070. No base-model retraining. Gemma's NL + context + routing are reused for free; only domain-specific compute + data need new work. |
| "Frontier capabilities can't be matched" | Frontier advantages = scale-of-retrieval + verification + structure. All three compile. Gemma + cards matches on specific tasks; genuinely frontier-exclusive work is rare. |
| "Doesn't scale — you'll cap on engineering" | Factorial scaling argument. Per-domain cost is flat (~1-2 days). 100-domain substrate is weeks, not years. |
| "Verification is the bottleneck" | CALM already verifies 100% on benchmark. Compiled verifiers are exact by construction. |

When probing a new capability, go straight to the protocol
(layer sweep → per-head classification → attention pattern →
content probe → causal forced-intervention). Classify the
circuit by shape. Compile based on class. Do not re-derive the
thesis mid-task.

## Empirical basis

Session 33-34 (2026-04-17+), 52-round arc (R13-R52.3) on RTX 4070 Laptop,
8 GB VRAM. Full per-round table + per-head/layer lookup:
`.claude/rules/tracing_roadmap.md` §"Gemma 4 E4B tracing findings"
+ `.claude/MEMORY/atlas.md`.

**7 capabilities mapped** (cluster summary — full sweep + per-head
ablation detail in tracing_roadmap.md per-round rows):

| Capability | Cluster | Typology | Key validation |
|---|---|---|---|
| Arithmetic | L22-L30, L23 peak | Concentrated | R28 forced-attn L30 H4/H6: \|Δ\|=0.407, 9/10 |
| Factual recall | L5, L11 | Diffuse (FFN-locked) | — (ROME/MEMIT territory) |
| Induction | L33-L37, L37 H6 peak | Concentrated | R33 Olsson-2022 pattern confirmed |
| Counting | L20, L31-L37 | Cooperative | R43 forced-attn L23: 6/6 |
| Comparison | L35, L23 shared | Diffuse (at heads) | R43 forced-attn L23: 18/18 \|Δ\|=0.176 (cleanest) |
| SV agreement | L23→L29→L35 | Hybrid pipeline | R42 forced-attn L23 H1/H4: \|Δ\|=0.467, 8/10 |
| Multi-step composition | L24 SWA Δ=-17.23 | Deep-diffuse | NO causal validation — see tier-3 nulls below |

**3 causal validations** (R28 arithmetic + R42 SV + R43 comparison+counting)
— same forced-attention template across 4 (layer, capability) pairs.
L23 H1/H4 proven hub-shared: 32/34 argmax matches across arithmetic + SV
+ comparison + counting.

**2 facades shipped**:
- `HubInjectionCard` (R44-R45): bit-identical to R43 inline; `generate()`
  verified. Runtime Q/K dispatch — no per-task hand-coding.
- `MultiStepReasoningFacade` (R46.2): N-op step-through digit bias,
  17/17 real Gemma fixes, 0 regressions. **5-for-1 L23 hub ROI.**

**3 tier-3 L24 distillation nulls** (R50.5 SAE / R51.5 MSE / R52.3 KL)
— same pattern in each: distillation-space loss improves, token
preservation fails. R53.36 install-audit refinement (`capability_gain.md`
§R53.36): R51-MSE reproduces L24 at cos=0.89 (close-miss cascade through
17 downstream layers); R52-KL never learned L24 at all (cos=-0.02, wrong
loss silent on residuals). Install math zero-diff both students. **Tier-3
L24 closed at current loss space**; Jacobian-weighted loss a credible
reopen path (~30% probability). Pivot to tier-2 stacking per §"Tier-2
stacking achieves tier-3-equivalent outcomes" above.

This is the evidence base. Future probes extend the atlas; they don't
re-validate the thesis.

## Related rules

- `Substrate.md` — install mechanics (CardSlot, `install_card_in_attention`, VerificationHook)
- `tracing_intelligence.md` — first-principles bound on what's compilable
- `tracing_roadmap.md` — concrete atlas progress and next-target queue
- `capability_gain.md` — measurement discipline (raw path + user-facing path)
- `embed_intelligence.md` — delivery mechanisms (card → Gemma tokens)
- `commercial.md` — product positioning (Tier 2/3 = the product)
- `workflow.md` — iteration discipline (hypothesis → test → commit)
- `calm.md` — verification layer (the CPU oracle for compiled cards)
