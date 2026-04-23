# The Molecule — Architectural Spec

Date: 2026-04-20
Origin: session conversation following R9-R19 + transformer-vm port + architectural push by user

## One-sentence thesis

**The project has been implicitly building ONE compound state-transition operator, where every named "component" (LLM, HRM, PT, Small2DTransformer, CALM, compiled cards, fast weights, hull cache, mixed geometries) is a slice / weight-bank / output-projection of the same underlying thing. The novel architecture is to design this from scratch, with no inherited pretrained LLM.**

---

## Why the naming misled us

We have named things separately because each was the solution to "the current transformer can't do X well":
- HRM/PT — because transformers can't extract structure reliably
- CALM — because transformers can't verify
- Compiled cards — because transformers aren't exact
- Fast weights — because transformers lack associative memory
- Mixed geometries — because dot-product attention is one similarity out of many
- Gate-graph IR — because weights are unauthorable

Each "component" was bolted on. But at the architectural level, they are all **state-transition operators or output projections on a shared residual state**. The separation is historical, not fundamental.

## The molecule — Ψ(Σ, token_in) → (Σ', y)

One operation. Compound state. Structured emission.

### Compound state Σ

```
Σ = (
    h:         (L, d),                # semantic residual (transformer-style)
    F:         (d, d),                # fast-weight matrix (associative memory)
    K, V:      (L, n_kv, d_head),     # KV cache (indexed memory)
    R:         (n_sub_heads, 8),      # per-sub-head routing tag
    C:         Dict[card_id, Tensor], # compiled-card working buffers
    τ:         int,                   # thinking-iteration counter
    π:         (L,),                  # per-position copy/generate prior
)
```

Each field interacts with every other per transition. None is separable without destroying function.

### One transition Ψ

Parameterized by weight banks:

```
Ψ_params = (
    W_core:     polymorphic core weights (trained from scratch or compiled)
    W_cards:    compiled weights installed from gate-graph IR
    W_verify:   oracle weights (frozen; CALM registry compiled into weight form)
    W_route:    routing projection
    W_fluency:  (optional plug-in) small trained fluency head
    W_judgment: (optional plug-in) small trained subjective discriminator
    W_research: (optional plug-in) experimental trainable head
    W_metacard: compiled program synthesizer (built-in for auto-upgrade)
    W_external: (optional plug-in) wrapper to external LLM fallback
)
```

Physically: ONE tensor operation on concatenated state, with block-sparse weights across disjoint channel ranges. Each block executes in parallel; outputs superpose in the shared residual.

### Multi-projection output y

```
y = (
    logits:    (vocab,),            # next-token prediction
    struct:    expr_str or None,    # structured-extraction output
    copy:      position or None,    # copy-decision output
    verify:    {ok, corrected_val}, # verification verdict
    confid:    [0, 1],              # confidence score
    card_out:  Dict[card_id, Any],  # compiled card results this step
    iter_next: int,                 # suggested thinking iterations
)
```

All projections evaluated from the same Σ' in one forward. No "run LLM first, then HRM, then CALM" — everything emits simultaneously.

## Polymorphic sub-head dispatch

Six attention reduction modes share the same score matmul. Only the reduction differs:

| Mode | Reduction | Role |
|---|---|---|
| GS (grouped-softmax) | group-softmax + mean | natural-language attention |
| SS (single-softmax) | vanilla softmax | HRM/PT-style structure |
| HM (hard-max) | argmax at tie-break | compiled card / exact lookup |
| UP (uniform-prefix) | 1/i over past | CumSum / counting / scanning |
| HL (hull-based) | O(log n) convex hull | indexed retrieval past 2K |
| FW (fast-weights) | outer-product + decay | short-term associative memory |

Plus mixed geometries per sub-head at d_head=2:
Euclidean / hyperbolic / spherical / toroidal / lattice.

## Why "one iterated core" and not "bands of layers"

Bands in a prior draft were **retrofit from Gemma's observed structure**, not first-principles. Stripped down:

- Layers exist because each layer = one matmul composition
- Depth gives serial composition
- But **D5 recurrence gives serial composition from ONE parameter set**
- Polymorphic sub-head modes provide per-iteration specialization
- Fast weights persist state across iterations

→ ONE polymorphic core, iterated N ∈ {1..16} times, router decides N per-token.

Parameter comparison:
- Standard 24-layer transformer at d_model=1024: ~300M params
- 1-layer iterated substrate at d_model=2048 (wider compensates): ~50M params (6× fewer)
- Compute scales with iterations chosen, not fixed depth

Adaptive thinking becomes an architectural primitive: trivial=1 iter, hard=16 iters. FLOPs scale with difficulty, not token count.

## Chassis = Small2DTransformer++

The molecule IS an extension of this project's existing `Small2DTransformer`:

```
Current Small2DTransformer:
  - d_head=2, single attention mode
  - Residual stream only
  - Single output projection (logits)

Molecule (Small2DTransformer++):
  + polymorphic sub-head dispatch (6 modes)
  + mixed geometries per sub-head
  + compound state (residual + fast-weights + card-slots + routing)
  + multiple output projections
  + reserved channel protocol native
  + D5 iteration as first-class
  + auto-upgrade wiring built-in
```

All extensions already exist as separate experimental modules in `calm/llm_computer/` (fast_weights.py, mixed_geometry.py, recurrent_substrate.py, combined_substrate.py, card_installer.py, hybrid_substrate.py, persistent_knowledge.py, auto_upgrade.py). The molecule = merging them into default structure, not optional flags.

## No Gemma

Previous drafts kept slipping into "Gemma + cards" framing. Correction: **Gemma is a historical accident** of this project having started from a pretrained model. The novel architecture does NOT include Gemma.

`W_core` is trained from scratch (or compiled from first principles — see "no-training path" below), designed for co-existence with compiled weight banks:
- Multi-objective loss covering all output projections from day 1
- Channel allocation respects reserved card slots from initialization
- Small (~300M-1.5B), not 5B-70B
- Training cost ~2 weeks on 1 A100, not months on clusters

## The no-training path (radical first principles)

**Gradient descent is ONE way to set weights — not the only way.** Training gives:
- Embeddings → can come from PMI/LSA factorization or structured feature encoding
- Semantic relations → compilable from knowledge graphs
- World knowledge → compilable from Wikidata/WordNet/ConceptNet (100M facts, each a recall card)
- Grammar/syntax → compilable CFG + morphological analyzer
- Style features → measurable decomposition into formality/warmth/hedging axes
- Decision composition → tiny lookup table over measurable axes

Only the residual — **fine-grained pragmatic judgment, truly implicit patterns** — genuinely requires training. Maybe 5-15% of observable LLM behavior.

The **fully-compiled molecule** is conceivable: `W_core` replaced with `W_grammar + W_lexicon + W_knowledge + W_discourse + W_reasoning + W_style + W_verify + W_persona`, no gradient descent anywhere. Knowledge-base expert systems failed in the 80s-90s because they lacked:
- Automated knowledge extraction (we have this now)
- Gate-graph IR + compilable attention (we have this now)
- Auto-upgrade loop (we have this now)
- Verification feedback (we have this now)

Expert systems didn't have these tools. We do. The no-training molecule is worth a serious attempt.

## How the molecule solves its weaknesses

Each weakness the molecule has relative to frontier LLMs becomes a **pluggable weight bank**:

| Weakness | Architectural fix |
|---|---|
| NL fluency | candidate generation via D5 iteration + multi-axis ranking head (W_fluency plug-in) |
| Unspecifiable subjective quality | small discriminator head as Ψ projection (W_judgment plug-in) |
| Rapid prototyping | W_metacard does programming-by-example card synthesis |
| Research flexibility | W_research for trainable experimental heads, distill to W_cards when stable |
| Frontier-level NL nuance | W_external optional wrapper to external LLM fallback |

**Plug-ins are optional.** A deployment for regulated industries (legal/medical/financial) might load ZERO plug-ins and run purely compiled — auditable, reversible, signed. A creative-writing deployment loads W_fluency + W_judgment. A research deployment loads W_research.

The molecule itself provides INTEGRATION — routing + verification + channel allocation + efficient forward. Plug-ins handle specialized capabilities.

## MetaCard and recursion

The molecule's built-in `W_metacard` takes **programming-by-example** input:

```
spec: fn_name : arg_types → return_type
examples: [(in1, out1), (in2, out2), ...]
constraints: monotonic, pure, bounded, etc.
```

And outputs a gate-graph IR specification, compiled + installed in W_cards. No NL understanding required — just structured search over IR space fitting the examples.

Auto-upgrade loop becomes this at runtime:
- Verification failure logged with (input, observed_output, expected_output)
- MetaCard synthesizes new card from examples
- New card installs, persists
- Molecule grows across sessions

This replaces gradient descent for specifiable capabilities. Training is REPLACED, not supplemented.

## Capabilities as emergent projections

What makes this a MOLECULE and not components:

- "LLM behavior" emerges from projecting Σ' on vocab axis with routing mostly GS
- "HRM behavior" emerges from projecting Σ' on structure axis with SS modes at structure sub-heads
- "CALM verification" emerges from the verify projection against W_verify oracles
- "Card compute" happens in card-slot channels with W_cards installed
- "Fast-weight recall" happens via the F field updated by FW sub-heads
- "Hull retrieval" happens via HL sub-head mode reading long context

**All from one forward. No "run component X first, then Y".**

Test: remove any field of Σ or any weight bank — remaining molecule still runs but loses the corresponding capability. Test: use any field in isolation — doesn't function without others. That's a molecule, not components.

## Concrete spec parameters (reference build)

For a production-target instance:

```
d_model       = 2048          # rich state, fits 8GB for training
d_head        = 2             # fundamental invariant
n_sub_heads   = 1024          # = d_model / d_head
core_layer    = 1             # single polymorphic core, iterated
n_iter_max    = 16            # thinking ceiling
vocab_size    = 65536         # 32K BPE + 32K reserved (modes, cards, sentinels)
max_context   = 8192          # extendable via HL mode + D5
core_params   ≈ 50M           # one layer
embed_params  ≈ 130M          # vocab × d_model (tied head)
router_params ≈ 1M
total_core    ≈ 180M          # without plug-ins
```

Channel allocation:
```
[0..1536)      shared residual (learned/compiled W_core domain)
[1536..1600)   routing control channels (per-sub-head mode tags)
[1600..1700)   dispatch metadata (card addresses, verification flags)
[1700..1900)   card I/O (200 channels / ~50 active cards per layer)
[1900..1980)   fast-weight read/write slots
[1980..2020)   hull-cache keys
[2020..2048)   verification sentinel channels
```

## Training/building curriculum

```
Phase 0 — scratch-build core:
  Option A (train): 2 weeks on 1 A100, multi-objective loss, 10-100B tokens
  Option B (compile): author W_core from structured rules (speculative, research-level)

Phase 1 — polymorphism calibration:
  Train routing on labeled task data (math → HM mode, count → UP mode, etc.)
  ~2 days. W_core frozen.

Phase 2 — card installation (ongoing, no training):
  Compile gate-graph IR cards per domain
  Install into reserved channel rectangles
  Auto-upgrade loop extends this indefinitely

Phase 3 — plug-in training (optional, per deployment):
  W_fluency: ~1 day on human-rated text
  W_judgment: ~1 day on pair-ranked subjective examples
  W_metacard: ~1 day on programming-by-example tasks (or compile from first principles)

Phase 4 — verification calibration:
  Tune verification bias strengths per card class
  ~1 day, one-shot

Total Phase 0-4: ~2 weeks on a single A100 vs months of cluster training for a 5B LLM.
```

## What shipping this would prove

Three testable architectural claims:

1. **Polymorphic sub-head dispatch at d_head=2 is more expressive than single-mode attention at equal parameter count.** Demonstrate a task solvable in mixed-geometry + HM mode that vanilla softmax transformers fail on at equal params.

2. **Compile-time capacity exceeds train-time capacity on specifiable ops.** Compiled multiplier 100% vs trained 95% at equal params (already shown at small scale, generalize).

3. **Compound-state single-forward execution outperforms pipeline orchestration on latency and error compounding.** Benchmark: molecule's verified emission vs LLM→extractor→verifier pipeline on the same task.

## What this is NOT

- Not a better transformer. A transformer is one mode of the molecule's dispatch.
- Not MoE (experts are whole FFNs; polymorphic sub-heads are finer-grained).
- Not Mamba / SSMs (different op entirely).
- Not a hybrid system (components don't cooperate; they coexist in one forward).
- Not an LLM+tools (tools are external processes; cards are weights inside Ψ).

## Immediate next-step concrete work (if this proceeds)

1. Document this spec (done — this file).
2. Name it in code: rename `Small2DTransformer` → `Substrate` to signal architectural status.
3. Promote D-extensions from experimental to default:
   - `fast_weights.py` default-on
   - `mixed_geometry.py` default per-sub-head
   - `recurrent_substrate.py` default iteration
   - `reserved_channels` default allocation
4. Add compound-state return signature to Substrate.forward().
5. Add multi-projection output dict to Substrate.forward().
6. Prototype W_metacard via programming-by-example.
7. Prototype W_fluency + W_judgment heads, train small-scale.
8. End-to-end small-scale demo: 10M-param Substrate + 5 compiled cards + 2 plug-ins, trained on a toy NL+structure+verify task.

## Rules references

This spec synthesizes and extends material from:
- `.claude/rules/augmentation_thesis.md` — substrate thesis, tier-2 stacking
- `.claude/rules/Substrate.md` — Small2DTransformer, d_head=2, per-sub-head partition
- `.claude/rules/tracing_intelligence.md` — what's compilable from first principles
- `.claude/rules/recursion.md` — MetaCard and card-that-builds-cards
- `.claude/rules/capability_gain.md` — measurement discipline
- `.claude/rules/calm.md` — oracle verification role
- `.claude/rules/training.md` — training vs compilation tradeoff
- `.claude/rules/commercial.md` — product positioning

## Relationship to the existing project

This spec is not a pivot. The project has been implicitly building this for 30+ sessions. The spec makes explicit what was already latent:

- 29 compiled programs → populate W_cards
- KnowledgeStore → populate W_cards with recall entries
- HRM / PT checkpoints → subsumed by polymorphic SS-mode sub-heads
- CALM 1002-function registry → compiles into W_verify
- Fast weights D1 experiments → become default F field update
- Mixed geometry D3 experiments → default per-sub-head geometry
- D5 recurrence → core iteration primitive
- Hull cache → HL sub-head mode
- Tracing atlas → informs initial routing protocol (where each capability cluster is)
- Tier-2 stacking → the design pattern for extending molecule additively

The current project has been building the pieces. The molecule is the ASSEMBLY OF THE PIECES INTO ONE OBJECT — which, crucially, DOES NOT require Gemma as the substrate. Gemma was convenient because it was available; the molecule is the project's own scratch-built chassis extended to its natural form.

## Decision points for the user

Three paths forward (ordered by ambition):

**A — Document the thesis only** (today)
- Ship this spec file
- No code changes
- Future sessions have shared vocabulary
- Zero implementation cost

**B — Incremental molecule-ward** (2-4 weeks)
- Rename Small2DTransformer → Substrate
- Promote D-extensions to defaults
- Compound-state + multi-projection return
- Ship small-scale demo (10M params, 5 cards)
- Validates the architectural thesis at proof-of-concept scale

**C — Full molecule from scratch** (2-3 months)
- 300M parameter Substrate trained scratch
- Full card library installed
- W_metacard operational
- Plug-ins optional
- Eval head-to-head vs Gemma + cards current-project baseline

The key bet being made: **an explicitly-compositional substrate outperforms monolithic training on production + regulated + specifiable workloads**, at 10-100× lower total cost over a multi-domain deployment.

If the bet fails, the molecule is still a better substrate for the current project's augmentation work — nothing is lost.

If the bet succeeds, this becomes a genuinely novel architecture class distinct from transformers, SSMs, and MoE — valued at whatever the market pays for auditable, verifiable, regulated-industry-deployable AI.
