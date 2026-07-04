# Augmentation Thesis — Empirical basis (Sessions 33-34, R20-R52.3, plus 2026-04-22 facade arc)

Per-round receipts, shipped-capability table with commits + dates,
R51/R52 distillation null detail, capability map, tier-1 preservation
eval. Current strategic positions: `rules/augmentation_thesis.md`.
This file exists for archaeology — "which receipt grounded which claim",
"what's been measured to support each tier-1/2/3 example".

## Empirical basis

Session 33-34 (2026-04-17+), 52-round arc (R13-R52.3) on RTX 4070
Laptop, 8 GB VRAM. Full per-round table + per-head/layer lookup:
`MEMORY/atlas/tracing_roadmap_part_1.md` §"Gemma 4 E4B tracing findings".

**7 capabilities mapped** (cluster summary — full sweep + per-head
ablation detail in `MEMORY/atlas/tracing_roadmap_part_1.md`):

| Capability | Cluster | Typology | Key validation |
|---|---|---|---|
| Arithmetic | L22-L30, L23 peak | Concentrated | R28 forced-attn L30 H4/H6: \|Δ\|=0.407, 9/10 |
| Factual recall | L5, L11 | Diffuse (FFN-locked) | — (FFN weight-probing territory) |
| Induction | L33-L37, L37 H6 peak | Concentrated | R33 Olsson-2022 pattern confirmed |
| Counting | L20, L31-L37 | Cooperative | R43 forced-attn L23: 6/6 |
| Comparison | L35, L23 shared | Diffuse (at heads) | R43 forced-attn L23: 18/18 \|Δ\|=0.176 (cleanest) |
| SV agreement | L23→L29→L35 | Hybrid pipeline | R42 forced-attn L23 H1/H4: \|Δ\|=0.467, 8/10 |
| Multi-step composition | L24 SWA Δ=-17.23 | Deep-diffuse | NO causal validation — see tier-3 nulls below |

**3 causal validations** (R28 arithmetic + R42 SV + R43
comparison+counting) — same forced-attention template across 4
(layer, capability) pairs. L23 H1/H4 proven hub-shared: 32/34
argmax matches across arithmetic + SV + comparison + counting.

**2 facades shipped from this arc**:
- `HubInjectionCard` (R44-R45): bit-identical to R43 inline; `generate()`
  verified. Runtime Q/K dispatch — no per-task hand-coding.
- `MultiStepReasoningFacade` (R46.2): N-op step-through digit bias,
  17/17 real Gemma fixes, 0 regressions. **5-for-1 L23 hub ROI.**

## Round 28 — causal confirmation of arithmetic mechanism

Session 33 R28: forced one-hot attention at L30 H4/H6 (reading
a_ones/b_ones from their token positions) preserves fd with mean
\|Δ\|=0.407 and 9/10 argmax match. Established that Gemma's learned
attention IS a position-selector LookUp, compilable exactly. Anchor
result for the "compilable circuit" framing.

## Shipped capability table (current state — for receipts only; current
text/thesis lives in `rules/augmentation_thesis.md`)

| Shipped capability | Type | Mechanism |
|---|---|---|
| R11 multiplier (5/10→10/10 on Gemma) | Tier-2 at output head | `VerificationHook` + step-through digit bias |
| R44/R45 `HubInjectionCard` (L23 H1/H4) | Tier-2 at concentrated circuit | Forced-attention facade, runtime Q/K dispatch |
| R46.2 `MultiStepReasoningFacade` (17/17 fixes) | Tier-2 stack | NL parser + `safe_eval` + step-through bias |
| `KnowledgeStore` recall cards | Tier-2 at output | Step-function indicators + `CardSlot` + `VerificationHook` |
| `programs/gcd`, `adder`, `multiplier` (compiled) | Tier-2 integration | Compiled compute + tier-2 output hook |
| R-delta-21 `CopyAugmentedDeltaNet` MQAR card | Tier-2/3 retrieval | 4-gate CardSlot. 2026-04-21 ship at 22.0 → 2026-04-22 R22f recalibration to **14.5 → 60/60 (+18, 0 regressions, commits `9691e06` + `c3cc73f`)**. See `delta_rule.md` §R22. |
| `BaseConversionFacade` R22c (`7db6eb9`) | Tier-2 decode-path | Parser + `int(x, base)` + digit bias. 10/10 vs 7/10 (+3, 30% lift). |
| `NumberTheoryFacade` R53a mod/GCD/LCM (`69279d4`) | Tier-2 decode-path | Parser + `safe_eval` + digit bias. **15/15 vs 8/15 (+7, 47% lift).** Exposed the `▁`-strip + POST_BIAS_BUDGET=4 discipline. |
| `NumericEncodeFacade` F2 int→hex/binary/octal (`5ee61a5`) | Tier-2 decode-path | 12/12 on chain corpus. First facade with letter-answer (e.g. "DEADBEEF"). |
| `Icd10RecallFacade` R60a 72,748-code DB (`afc0220`) | **Tier-3 decode-path** | Parser + JSON lookup + multi-token step-through bias on TEXT answer. **26/30 vs 8/30 (+18, 67% lift).** First tier-3 delivered via decode-path rather than CardSlot. 4 edge codes resist — F1 retry infra (`8ba151d`) + pure-DB bypass candidate future work. |
| `PlannerFacade` R70a + F2 (`956a3ae` + `5ee61a5`) | Tier-2 orchestrator | First-match-wins classify over 5 specialist facades + "X in hex/binary/octal" chain detect. 20/20 single + 12/12 chain. |
| Auto/meta-generated facades via `recursion.py` (6 Level-1 `*_auto.py`, 5 Level-2 `*_meta.py` — `3274659`, `5173745`) | Tier-2 auto-gen | Level-1 (hand `FacadeSpec`): 17/30 → 30/30 across factorial/fibonacci/combinations/permutations/power/next_prime. Level-2 (`MetaFacade.from_oracle(fn_name, arity)`): 4/15 → 15/15 across factorial/combinations/gcd/lcm/fibonacci. **Spec authorship moved human → substrate; three CALM gates (oracle → ast.parse → live A/B) intact.** |

## R51/R52 anomaly — distillation tier-3 nulls

Both R51 and R52 explicitly chose REPLACEMENT via monkey-patching
`m._forward_layer` to skip Gemma's native L24. R51's install.py
docstring even cites rejecting `CardSlot` because "residual-additive
cannot REPLACE L24" — but replacement was the wrong hypothesis for a
deep-diffuse circuit. Three independent nulls:

- **R50.5 SAE**: top-50 SAE features have zero causal effect despite
  99.1% reconstruction. Interpretability-without-causality on
  attribution-picked basis.
- **R51.5 MSE**: 92.6%-var-explained MSE student produces 0.19/0.34
  prefix match. Close-miss — R53.36 audit confirms cos=0.89 scale=0.91
  (student DOES reproduce L24 on average), but 10% diffuse residual
  error cascades through 17 downstream layers + head, amplifying into
  wrong argmax. MSE loss averages over 2560 channels — can't
  concentrate penalty on task-critical directions.
- **R52.3 KL**: KL-divergence student (val KL 1.96→1.21) produces
  0.04/0.08 prefix match, WORSE than MSE baseline. R53.36 audit
  reveals cos=-0.02 scale=94× — student never learned L24 at all.
  KL-on-final-logits never constrains residual reconstruction.

R53.36 install boundary check: `L24_installed == h_before +
student(h_before)` bit-identical (max abs diff = 0.00e+00) on both
students. **Tier-3 is NOT install-bug-blocked**; it's loss-space-
blocked at MSE and KL.

**Tier-3 reopen path** (not active): Jacobian-weighted loss —
weight residual error by downstream causal effect on head logits
(`||J · (pred - contribution)||²` where `J = d(head_logits)/d(h_L24)`).
Speculative; ~1-2 weeks of work; commercial lift moderate since
tier-2 stacking (R46.2 17/17 fixes) already augments L24's task at
output level without tier-3 cost.

## R53.2b — Tier-1 preservation finding (substrate RAG vs prompt RAG)

Complex eval (`scripts/r53_eval_complex.py`, 6 multi-step coding
problems × 3 conditions):

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

Root cause: blanket retrieval injection disrupts Gemma's strong-prior
behavior on problems it already solves. Established that prompt-RAG
violates Tier 1; substrate RAG (hash-gated at L30) preserves it
automatically.

## R53.14 / R53.20a / R53.20b — substrate L41 install REGRESSES on code

Tier-1 thesis holds in principle but was falsified at one specific
install mechanism: L41 `CardSlot(preserve=True)` + per-marker
`FirstTokenHook(boost=50)` on the R53.0 6-problem code corpus, SWA
bug already fixed.

Result (`ec8887f` / `scripts/r53_20b_stacked.py`):

| | stock | prompt-RAG | substrate @ L41 |
|---|---:|---:|---:|
| log_level_counts | 6/6 | 6/6 | **0/0** |
| lru_cache_class | 9/9 | 9/9 | **0/0** |
| (others unchanged) | | | |
| **TOTAL** | 25/27 | 25/27 | **10/12** (-9.3pp) |

Bit-identical MISS preservation at L41. HIT prompts regressed.

Root cause — install-mechanism, not SWA: Gemma's first-token on code
is confidently a fence/whitespace opener (margin 6.8-9.2), so
`min_margin=0.5` never gates, hook always fires on HIT, forces
"def"/"class" → code-without-fence → extractor fails.

Thesis refinement: hash-match Tier-1 holds at the OUTPUT boundary
(`VerificationHook` with small vocab_mapping + `min_margin`, as in
the learning-loop demo). Does NOT hold for residual-write `CardSlot`
at arbitrary layers. First-token bias is the wrong intervention for
code.

## AST walker — correct tier-2 for code (R53.35 + 2026-04-21)

`calm/llm_computer/facades/ast_repair.py` ships **7 rewrites as of
2026-04-21** — shadow_rename, dict-key synonym, syntax_repair (3
original in R53.35 `8cc2ff4`/`c81feb6`), plus `fuzzy_rename_function`
(commit `805e539`, Track A walker expansion).

Dispatches on categorized runtime errors:
- `TypeError: 'int' object is not callable` → shadow_rename
- `KeyError: 'X'` → dict-key synonym
- `SyntaxError` offset → bracket-mismatch or insert-before-colon repair
- `NameError: name 'X' is not defined` → fuzzy_rename (Jaccard ≥ 0.5
  against defined FunctionDefs)

Full receipts in `MEMORY/atlas/capability_gain_arc.md` §"R53.35".
Tier-2 stacking thesis reinforced: **mechanical post-gen rewrite at
Gemma's output beats in-context hint-tuning or tier-3 distillation
of deep-diffuse circuits.**

## R53.36 audit refinement (2026-04-20)

Three distillation nulls that close tier-3 are NOT the same mechanism:
- R50.5 SAE = interpretability-without-causality on attribution-picked basis
- R51.5 MSE = close-miss at cos=0.89 scale=0.91 where 10% diffuse residual error cascades through 17 downstream layers into wrong argmax
- R52.3 KL = wrong-loss failure where student output uncorrelated with L24's contribution (cos=-0.02, scale=94×)

Install math verified bit-identical on both students, so tier-3 is
not install-bug-blocked. **Tier-2 stacking stays the priority** —
but tier-3 has a credible reopen path via Jacobian-weighted loss
(weight residual error by downstream causal effect on head logits)
if an active workstream ever needs single-card L24 replacement.
Not priority. Full audit detail: `MEMORY/atlas/capability_gain_arc.md`
§"R53.36".

## Mapping ROI (Tier 2)

Without mapping, you guess which of Gemma's weak capabilities is
fixable and burn engineering on dead ends (diffuse circuits can't be
replaced at attention level — R30 factual recall example). With
mapping, per-head ablation tells you in ~1 hour whether a circuit is
concentrated (Tier-2 addressable) or diffuse (not). Estimated Tier-2
engineering-cost reduction: **~10×**, because you pick concentrated
targets instead of flying blind.

## Compositional hypothesis evidence

- **Counting = induction ∪ compute.** Counting sweep hits L33 and
  L37 (shared with induction R31) PLUS new peaks at L20 and L31
  (adjacent to arithmetic L30-L32 compute cluster).
- **Comparison = arithmetic-adjacent + induction-adjacent.**
  Comparison sweep (R36) has L23 in top-3 (shared with arithmetic)
  and L33 in top-5 (shared with induction).
- **L37 hosts multiple specialized heads.** L37 H6 = induction (R33
  canonical pattern), L37 H4 = numeric successor (R35). Same layer,
  distinct heads, distinct capabilities.
- **Hub-sharing causally proven (R42/R43), facade shipped (R44/R46).**
  L23 H1/H4 forced-attention mirror of R28 preserves SV agreement
  (8/10), comparison (18/18), counting (6/6). Same heads with
  task-specific Q routing serve 5 capabilities simultaneously
  (arithmetic + SV + comparison + counting + multi-step composition
  via `MultiStepReasoningFacade` R46.2) — one compiled replacement
  via `HubInjectionCard` (R44) benefits all five (**5-for-1
  compilation ROI**).

## Gemma-substrate instantiation (carved from rules 2026-07-04)

Current strategic positions live in `rules/augmentation_thesis.md`.
This section preserves Gemma-specific mechanics displaced by the
current-arc rewrite.

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

### Three-tier framework (Gemma substrate)

| Tier | Intervention | Cost | Moat |
|---|---|---|---|
| **1 — Preserve** | Leave Gemma alone where it works | 0 | none |
| **2 — Augment weak** | Compile replacement for specific failing circuit | Days | high |
| **3 — Plug missing** | Compile new capability from scratch at unused slot | Days | highest |

Tier 1 is table stakes — every Gemma user gets it free. **Tiers 2
and 3 are the product.** The mapping protocol tells you which
capabilities are Tier-2-addressable (concentrated circuits to
surgically replace) vs Tier-3 (design from scratch).

### Tier-2 stacking achieves tier-3-equivalent outcomes

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

### Circuit typology — three shapes (+ deep-diffuse)

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

### Atlas sparseness estimate

Atlas sparseness estimate: ~30-50 specialist heads cover most core
capabilities (of 336 total head slots = 42 layers × 8 heads). Full
atlas ~20-40 hours of focused probing on RTX 4070 Laptop. Tractable.

### Factorial scaling per domain

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

#### Economics inversion vs standard LLMs

| Standard LLM | Substrate |
|---|---|
| Bigger model = better at everything, expensive | Gemma stays same size, same cost |
| Fine-tune for domain = expensive, forgets other things | Add a card = strict improvement, zero regression |
| Domains compete for capacity in one opaque network | Domains are disjoint card slots |
| Removing a capability = retrain from scratch | Remove a card = clean `detach()`, no damage |
| 100 domain specialists = 100× training cost | 100 cards stacked = 1× base cost + per-card hours |

### Customer verticals = card decks

Each customer's substrate = Gemma + their own deck of Tier-2/3 cards.

- **Legal**: citation-format enforcer + statute DB + clause templates + compliance checkers + Gemma drafts
- **Medical**: ICD-10 validator + drug-interaction DB + diagnosis templates + dosage calculator + Gemma explains
- **Fintech**: exact-decimal arithmetic + regulation lookups + compliance verifiers + currency conversion + Gemma answers
- **Engineering**: unit converters + formula cards + material property DB + Gemma narrates

No cross-vertical interference: legal cards don't affect a hospital
substrate. Each customer ships the stack their domain needs.

### Frontier models also interpolate

Every "creative" output from frontier models is a remix of training-
data patterns. The substrate's advantage is making that remix
*explicit and controlled* (DB + retrieval + verified composition)
rather than *opaque and sometimes wrong* (internal weights with no
auditing). What frontier models do better is interpolate over a
larger example set; the substrate recovers that by making relevant
examples explicit rather than weight-stored.

### Automatic Tier-1 preservation as substrate property

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

#### Commercial positioning sharpens

"RAG that knows when not to retrieve" is a different product from
"RAG with a bigger DB." Regulated industries specifically need
intervention-when-warranted, not intervention-always — spurious
injections drift output from user intent. **Selective-intervention
property is substrate-native** and hard to reproduce with vanilla
RAG pipelines.

### Gemma-specific anti-skepticism counters

| Objection | Settled counter |
|---|---|
| "Factual recall needs frontier models" | Diffuse-FFN circuit — compilable via FFN weight probing OR side-channel `KnowledgeStore` with verified retrieval. |
| "Retraining needed for each domain" | Tier-2 stacking takes hours/days per domain. PT is ~185K params, ~30 min. No base-model retraining. |
| "Verification is the bottleneck" | CALM verifies 100% on benchmark. Compiled verifiers are exact by construction. |

## Cross-refs

- Current strategic positions: `rules/augmentation_thesis.md`
- Tracing arc receipts: `MEMORY/atlas/tracing_roadmap_part_1.md`
- Capability-gain measurement receipts: `MEMORY/atlas/capability_gain_arc.md`
- DT install receipts: `MEMORY/atlas/delta_rule_arc.md`
