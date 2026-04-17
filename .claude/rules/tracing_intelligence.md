# Tracing Intelligence — First-principles bound on what's compilable

**The claim**: a trained neural network is a fixed function — a
specific sequence of matmuls and activations. Any computation it
performs is deterministic and therefore traceable. Anything traceable
can be re-implemented as compiled weights. The substrate has **no
architectural ceiling at "deterministic compute"** — that's just
where the tracing is mathematically trivial. The ceiling moves as
interpretability advances.

This file is the first-principles companion to `Substrate.md`
(architectural invariants), `capability_gain.md` (measurement
discipline), and `tracing_roadmap.md` (what's actually been traced
and what's next).

## What "compilable" means here

A compiled card is a `Small2DTransformer` whose weights are set
deterministically, not learned. The gate-graph IR (`gate_graph.py`)
describes the computation as a set of gates:

| Gate | Encodes |
|---|---|
| `TokenEmbed` | Per-token scalar lookup table. Any function `f: vocab → ℝ^n`. |
| `PosEmbed` | Per-position scalar lookup. Any function `g: pos → ℝ^n`. |
| `LookUp` | Attention-head that copies values from position 0 to query positions. |
| `LookUpExact` | Parabolic-key attention — exact key-based retrieval from any past position. |
| `ReGLU` | `output += coef · val · ReLU(gate)`. Encodes: exact integer product (`a · ReLU(b)`), step functions (`ReLU(x-k+1) - ReLU(x-k)` = indicator), gated additions. |
| `LinearHead` | Final linear projection to vocab. |
| `Delegate` | Routes through Python's `safe_eval` (runtime Python). Makes the IR Turing-complete in principle. |

With these primitives you can encode:
- Integer arithmetic (adder, multiplier, GCD, factorial — done)
- Boolean logic / syllogism (reasoning_engine — done)
- Discrete lookup tables of any shape (KnowledgeStore — done)
- Step-function decomposition of any bounded integer function
- Bounded-depth sequential processing via multi-layer composition
- Tree / graph structures via LookUpExact with parent pointers

Turing-equivalence to fixed-depth transformers follows from:
attention = matrix multiplication with nonlinearity (encodable via
LookUp / LookUpExact), FFN = linear projection + ReLU + linear
projection (encodable via ReGLU combinations).

## The corrected list of "capabilities"

An earlier version of this project's rules said the substrate
"can't" handle:

- Semantic understanding of novel inputs
- Abstract reasoning beyond what the LM encodes
- Multi-step planning that isn't reducible to compilable primitives
- Anything requiring representational capacity the base LM lacks

**All four claims were wrong from first principles.** Correcting each:

### Semantic understanding of novel inputs

"Semantic understanding" = specific features activating in specific
patterns across Gemma's layers, which downstream layers use to predict
coherent continuations.

- Feature directions in Gemma's residual → vectors. **Encodable.**
- Attention patterns that combine features → LookUp / LookUpExact.
  **Encodable.**
- FFN transformations on feature activations → ReGLU.
  **Encodable.**

What blocks compilation: **nobody has traced Gemma 4 E4B's semantic
circuits yet.** This is interpretability labor, not architectural.
Sparse Autoencoder research on similar-size models (Claude 3 Haiku,
Gemma 2) has extracted feature inventories — the same work on Gemma
4 E4B + circuit-level probing would give us a compilable inventory.

### Abstract reasoning beyond what the LM encodes

Two cases:

1. **The abstraction exists in Gemma's weights** (many abstractions
   do, implicitly, from training): trace the circuit → compile.
   Same as semantic understanding above.

2. **The abstraction does NOT exist in Gemma's weights**: no circuit
   to trace, but you can **design** the circuit from scratch using
   the IR primitives. No different from how we compiled the
   multiplier — except labor is in circuit design rather than
   reverse-engineering.

What I got wrong earlier: conflating "the LM can't do it" with
"the substrate can't host it." Those are different claims. The
substrate can host any bounded computation; whether the LM has it
is orthogonal.

### Multi-step planning

Decompose to: parse goal → track subtask state → execute → handle
failure → update plan. Each is compilable:

- **Goal parse**: PT-style copy-augmented attention, or compiled
  template matching via LookUpExact against a catalog of known
  goal types.
- **State tracking**: channel-as-register pattern (already
  demonstrated in `depth_compound.py`, `dispatched_v4.py`). Each
  layer writes to specific channels, subsequent layers read them.
  42 layers × N channels = bounded planning depth.
- **Execution**: dispatch via opcode → compiled operations.
  `dispatched_v4` handles 5 ops + cross-card gating.
- **Failure handling**: conditional ReGLU (gate on error channel,
  val on recovery operation).

A general planner card is ~100-500 gate-graph nodes. Not harder than
reasoning_engine, just more state. Hasn't been built yet. Worth a
Round 20-ish target once the prerequisite facades are in place.

### Representational capacity the base LM lacks

"Capacity" = the set of feature directions (vectors in the residual
space) that represent concepts. Gemma's base weights encode whatever
features training produced.

- Adding a compiled card with **new feature directions** literally
  extends the substrate's representation. The card's TokenEmbed /
  residual channels hold vectors for concepts Gemma doesn't have.
- Downstream processing (card's LookUp / ReGLU / head) uses those
  new features deterministically.
- From the substrate-as-a-whole (Gemma + cards) perspective, the
  combined representation is larger than Gemma alone.

The confusion was between "base LM's frozen weights" (fixed) and
"substrate's combined capability" (extensible). The base LM is a
component; the substrate is the full system.

## Where the actual limits are

Not architectural. Engineering and research labor:

| Limit | Source |
|---|---|
| **Interpretability research pace** | Each traced circuit in a large model is weeks-to-months of human work with current tools. Automated circuit discovery (ACDC and successors) accelerates this but isn't turnkey. |
| **Circuit design labor** | For capabilities Gemma lacks, humans design the circuit from scratch. Novel circuit design is harder than reverse-engineering. |
| **Fixed depth of Gemma (42 layers)** | Compiled cards share Gemma's depth budget. Computations requiring more than 42 sequential steps need multi-forward orchestration (step-through decode, autoregressive refacade). |
| **Channel budget (d_model=2560)** | Each installed card consumes residual channels. Round 3 showed 30 small cards fit with zero regression; heavy domain facades → ~5-10 per 8 GB. |
| **FP32 host budget** | For in-attention install, each host layer costs ~330 MB SWA / ~600 MB global. Practical max: 5-7 hosting layers on 8 GB. CardSlot avoids this cost. |

## What "tracing" actually looks like in practice

For each target capability X, the workflow is roughly:

1. **Collect a corpus** where Gemma does X correctly (e.g., 500
   examples of "translate NL → expression").
2. **Identify candidate layers** via activation patching: zero out
   each layer's contribution on the corpus, see which zeroing most
   degrades performance. Those are the load-bearing layers.
   (Concrete example: Round 16 ran a 42-layer × 10-prompt sweep on
   Gemma 4 E4B arithmetic and localized to L22-L30 with L23 peak.)
3. **Identify candidate attention heads** via attention probing:
   extract each head's attention pattern on the corpus, look for
   structure (positional, topical, causal). (Rounds 17-18 ran a
   poor-man's version on L23: per-head ablation + Q/K/V decomposition
   localized 93% of L23's arithmetic contribution to H4's V
   projection. Proper attention-pattern probing still pending.)
4. **Identify candidate FFN neurons** via neuron probing: find
   neurons that fire on capability-relevant inputs and not on
   controls.
5. **Sparse Autoencoder on the residual**: extract 10K-100K
   interpretable feature directions. Each feature has a direction
   vector and a semantic label.
6. **Reverse-engineer the circuit**: which features are read by
   which attention heads, which features are written by which FFN
   neurons, how the composition produces the capability's output.
7. **Re-implement in gate-graph IR**: TokenEmbed / residual writes
   for each feature direction, LookUp / LookUpExact for each
   attention operation, ReGLU for each FFN transformation,
   LinearHead for output.
8. **Verify bit-equivalence** (or close-to) against the trained
   model's output on the capability.
9. **Install as a compiled card** into the substrate.

Steps 2-6 are weeks to months per capability with current tools.
Steps 7-9 are hours once the circuit is traced. The bottleneck is
clear: **interpretability**, not compilation.

## What to build when

Three tiers of capability, ordered by labor:

| Tier | Labor | Examples |
|---|---|---|
| **Trivial** — exact math or discrete lookups | Hours | adder, multiplier, is_prime, gcd, factorial, recall DB, country capitals, timezone conversions |
| **Designed** — novel circuits using IR primitives | Days to weeks | general planner, AST parser, simple type checker, analogy-by-structure-match, sequential reasoner |
| **Reverse-engineered** — tracing + compiling Gemma's own circuits | Weeks to months | semantic understanding circuits, induction heads, specific reasoning circuits |

Ship Trivial fast. Build Designed when a specific target is needed
and the circuit design is tractable. Reverse-engineered is the
long-horizon research bet — it's what makes the substrate
genuinely competitive with pure LM capacity, because it gives us
the LM's own capabilities as inspectable, reversible, auditable
compiled cards.

## Validated on Gemma 4 E4B (session 33, Rounds 13-43)

The first mechanistic-interpretability arc on this model. Validated
that steps 1-2 of the tracing workflow (corpus + activation patching)
produce clean results on capabilities whose information flow aligns
with Gemma's architecture. Full arc details in `tracing_roadmap.md`
§"Gemma 4 E4B tracing findings"; summary:

- **Activation patching (R16)** localizes arithmetic to **L22-L30
  cluster**, L23 peak (mean Δ=-10.18 on correct-digit logit, hurts
  10/10 arithmetic pairs). 5B params narrowed to 9 layers via a 420-
  forward sweep in ~15 minutes.
- **Per-head ablation (R17)** narrows L23 (8 Q-heads) to **H1 and
  H4** — other 6 heads have mean Δ ≈ 0.
- **Q/K/V decomposition (R18)** identifies **V (content) as the
  signal carrier**, not Q/K (attention pattern). H4's V ablation =
  -9.51, accounts for 93% of L23's total arithmetic contribution.
- **Linear probe (R19)**: V linearly encodes the product's first
  digit at 2x chance (0.22 vs 0.11, 270 samples). Real but indirect
  — V likely carries operand and intermediate representations, not
  the final digit. SAE work needed for clean features.
- **Per-sub-head ablation (R20)**: H4's 512-d output split into 256
  d_head=2 pairs; ablate each × 10 arithmetic pairs. **0 sub-heads
  with mean Δ < -1.0**; top sub-head = -0.583 (vs full H4 = -4.30).
  Top-8 sub-heads carry only 26% of damage; top-64 needed for 80%.
  Signal is distributed across H4's V subspace, not sparse in the
  d_head=2 basis.
- **6 capabilities mapped (R20-R40)**: arithmetic, factual recall,
  induction, counting, comparison, SV agreement. Circuit typology:
  concentrated (arithmetic, induction), cooperative (counting L20),
  diffuse (factual, comparison), hybrid pipeline (SV agreement).
- **Hub-sharing causally proven (R42/R43)**: L23 H1/H4 forced-
  attention intervention (mirror of R28) preserves SV agreement
  (8/10), comparison (18/18, cleanest result), counting (6/6).
  Same heads + task-specific Q-routing proven cross-capability.
  One compiled L23 H1/H4 replacement benefits 4 capabilities
  simultaneously — **4-for-1 compilation ROI**. Full per-capability
  detail: `.claude/MEMORY/atlas.md`.

Concrete target for the full Phase-2 SAE + ACDC pipeline:
`L23.attn_v (KV group 1)`, a 512-d projection with ~2.6M weights
(or H4's 512-d read slice). R20 confirmed this is the right
granularity — features live as distributed directions in V-space,
so the SAE needs the full head/V output as input rather than a
narrowed sub-head slice.

**Why it worked:** Gemma's alternating SWA/global attention forces
cross-operand aggregation into global layers (L5, L11, L17, **L23**,
**L29**, L35, L41). Arithmetic requires seeing both operands → must
happen at a global layer. L23 and L29 were architecturally predicted;
measurement confirmed. Capabilities whose information flow matches
Gemma's structure should localize similarly; capabilities that are
inherently distributed (open-ended semantics, long-range reasoning)
will not.

**Ruled out by this arc:**
- Naive logit lens as a sole tracing tool (R13): noisy at middle
  layers; use with other probes.
- Single-prompt activation patching as a localization tool (R14→15):
  overgeneralization risk; always aggregate.
- L35 as the arithmetic circuit (R14→16 correction): minor
  contributor; the real seat is L22-L30 with L23 peak.
- Per-sub-head d_head=2 ablation as a *further* localization tool
  inside a head (R20): H4's arithmetic signal is distributed across
  its 512-d V subspace, not sparse in d_head=2 slots. The head is
  the right granularity; go to SAE from here, not narrower ablation.

## The corrected upper bound

> The substrate can host any bounded computation expressible as a
> fixed sequence of matmuls + activations. Since any trained
> transformer of Gemma's size computes exactly that, the substrate
> is Turing-equivalent to one. Plus: the substrate can host any
> computation humans can DESIGN as a gate-graph, even if no model
> has been trained to do it.

Formally: `Capabilities(substrate) ⊇ Capabilities(trained_model_of_same_size) ∪ Human_Designable_Circuits`.

This is substantially larger than "deterministic arithmetic." The
distance between today's substrate (multiplier + a few DBs) and
the theoretical upper bound is labor, not architecture.

## Related rules

- `capability_gain.md` — how to verify that tracing + compilation
  actually delivered a capability, not just a format change
- `embed_intelligence.md` — delivery mechanisms (card → Gemma tokens)
- `Substrate.md` — install mechanisms (where cards live)
- `tracing_roadmap.md` — concrete plan of what to trace next
- `workflow.md` — hypothesis-test discipline
