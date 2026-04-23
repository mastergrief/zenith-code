**Part 2**

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
`.claude/MEMORY/atlas/tracing_arc_part_1.md` §"Gemma 4 E4B tracing findings"
+ `.claude/MEMORY/atlas/capabilities.md`.

**7 capabilities mapped** (cluster summary — full sweep + per-head
ablation detail in `MEMORY/atlas/tracing_arc_part_1.md` per-round rows):

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
- `compute_facades.md` — decode-path tier-2 card pattern (R46.2 + R22c), zero-VRAM compute facades
- `delta_rule.md` §R22 install — CardSlot retrieval card with 4-gate config (2026-04-21 shipped)
- `tracing_intelligence.md` — first-principles bound on what's compilable
- `MEMORY/atlas/tracing_arc_part_1.md` — concrete atlas progress and next-target queue
- `capability_gain.md` — measurement discipline (raw path + user-facing path)
- `embed_intelligence.md` — delivery mechanisms (card → Gemma tokens)
- `commercial.md` — product positioning (Tier 2/3 = the product)
- `workflow.md` — iteration discipline (hypothesis → test → commit)
- `calm.md` — verification layer (the CPU oracle for compiled cards)
