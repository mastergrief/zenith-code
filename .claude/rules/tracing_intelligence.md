---
paths:
  - "scripts/*head*.py"
  - "scripts/*ablation*.py"
  - "scripts/*trace*.py"
  - "scripts/test_l*.py"
  - "scripts/*layer*.py"
  - "scripts/*logit*.py"
  - "calm/llm_computer/**"
---

# Tracing Intelligence — first-principles bound on what's compilable

**The claim**: a trained neural network is a fixed function — a
specific sequence of matmuls and activations. Any computation it
performs is deterministic and therefore traceable. Anything traceable
can be re-implemented as compiled weights. The substrate has **no
architectural ceiling at "deterministic compute"** — that's just
where the tracing is mathematically trivial. The ceiling moves as
interpretability advances.

This is the first-principles companion to `Substrate.md` (architectural
invariants), `capability_gain.md` (measurement discipline), and
`augmentation_thesis.md` (strategic framing).

> Historical receipts (per-round arc validation, SAE arc findings,
> methodology nulls, "why it worked" empirical breakdown): see
> `MEMORY/atlas/tracing_intelligence_arc.md` and
> `MEMORY/atlas/tracing_roadmap_part_1.md`.

## What "compilable" means

A compiled card is a `Small2DTransformer` whose weights are set
deterministically, not learned. The gate-graph IR (`gate_graph.py`)
describes computation as a set of gates:

| Gate | Encodes |
|---|---|
| `TokenEmbed` | Per-token scalar lookup table. Any function `f: vocab → ℝ^n`. |
| `PosEmbed` | Per-position scalar lookup. Any function `g: pos → ℝ^n`. |
| `LookUp` | Attention-head that copies values from position 0 to query positions. |
| `LookUpExact` | Parabolic-key attention — exact key-based retrieval from any past position. |
| `ReGLU` | `output += coef · val · ReLU(gate)`. Encodes: exact integer product, step functions (`ReLU(x-k+1) - ReLU(x-k)` = indicator), gated additions. |
| `LinearHead` | Final linear projection to vocab. |
| `Delegate` | Routes through Python's `safe_eval` (runtime Python). Makes the IR Turing-complete in principle. |

With these primitives you can encode:
- Integer arithmetic (adder, multiplier, GCD, factorial)
- Boolean logic / syllogism (reasoning_engine)
- Discrete lookup tables of any shape (`KnowledgeStore`)
- Step-function decomposition of any bounded integer function
- Bounded-depth sequential processing via multi-layer composition
- Tree / graph structures via `LookUpExact` with parent pointers

Turing-equivalence to fixed-depth transformers follows from: attention
= matrix multiplication with nonlinearity (encodable via
LookUp / LookUpExact), FFN = linear projection + ReLU + linear
projection (encodable via ReGLU combinations).

## What's compilable in principle

### Semantic understanding of novel inputs

"Semantic understanding" = specific features activating in specific
patterns across Gemma's layers, which downstream layers use to predict
coherent continuations.

- Feature directions in Gemma's residual → vectors. **Encodable.**
- Attention patterns combining features → LookUp / LookUpExact. **Encodable.**
- FFN transformations on feature activations → ReGLU. **Encodable.**

What blocks compilation: **nobody has traced Gemma 4 E4B's semantic
circuits yet.** Interpretability labor, not architectural.

### Abstract reasoning beyond what the LM encodes

Two cases:
1. **Abstraction exists in Gemma's weights** — trace the circuit →
   compile. Same as semantic understanding.
2. **Abstraction does NOT exist in Gemma's weights** — design the
   circuit from scratch using IR primitives. Labor is in circuit
   design rather than reverse-engineering.

Common confusion: conflating "the LM can't do it" with "the substrate
can't host it." Different claims. The substrate can host any bounded
computation; whether the LM has it is orthogonal.

### Multi-step planning

Decompose to: parse goal → track subtask state → execute → handle
failure → update plan. Each is compilable:

- **Goal parse**: PT-style copy-augmented attention, or compiled
  template matching via LookUpExact against a catalog of known goal types.
- **State tracking**: channel-as-register pattern (already demonstrated
  in `depth_compound.py`, `dispatched_v4.py`). Each layer writes to
  specific channels, subsequent layers read them. 42 layers × N
  channels = bounded planning depth.
- **Execution**: dispatch via opcode → compiled operations.
- **Failure handling**: conditional ReGLU (gate on error channel,
  val on recovery operation).

A general planner card is ~100-500 gate-graph nodes. Not harder than
reasoning_engine, just more state.

### Representational capacity the base LM lacks

"Capacity" = the set of feature directions in residual space that
represent concepts.

- Adding a compiled card with **new feature directions** literally
  extends the substrate's representation. The card's TokenEmbed /
  residual channels hold vectors for concepts Gemma doesn't have.
- Downstream processing (card's LookUp / ReGLU / head) uses those
  new features deterministically.

The confusion is between "base LM's frozen weights" (fixed) and
"substrate's combined capability" (extensible). The base LM is a
component; the substrate is the full system.

## Where the actual limits are

Not architectural. Engineering and research labor:

| Limit | Source |
|---|---|
| **Interpretability research pace** | Each traced circuit in a large model is weeks-to-months of human work with current tools. Automated circuit discovery (ACDC and successors) accelerates this but isn't turnkey. |
| **Circuit design labor** | For capabilities Gemma lacks, humans design the circuit from scratch. Novel circuit design is harder than reverse-engineering. |
| **Fixed depth of Gemma (42 layers)** | Compiled cards share Gemma's depth budget. Computations requiring more than 42 sequential steps need multi-forward orchestration (step-through decode, autoregressive refacade). |
| **Channel budget (d_model=2560)** | Each installed card consumes residual channels. Heavy domain facades → ~5-10 per 8 GB. |
| **FP32 host budget** | For in-attention install, each host layer costs ~330 MB SWA / ~600 MB global. Practical max: 5-7 hosting layers on 8 GB. CardSlot avoids this cost. |

## What tracing looks like in practice

For each target capability X, the workflow is roughly:

1. **Collect a corpus** where Gemma does X correctly.
2. **Identify candidate layers** via activation patching — zero each
   layer's contribution on the corpus, see which zeroing most degrades
   performance. Those are load-bearing layers.
3. **Identify candidate attention heads** via attention probing —
   extract each head's pattern on the corpus, look for structure
   (positional, topical, causal). Per-head ablation + Q/K/V
   decomposition isolates the content-carrier (usually V).
4. **Identify candidate FFN neurons** via neuron probing — find
   neurons that fire on capability-relevant inputs and not on controls.
5. **Sparse Autoencoder on the residual** — extract interpretable
   feature directions. Each feature has a direction vector and a
   semantic label.
6. **Reverse-engineer the circuit** — which features are read by which
   attention heads, which features are written by which FFN neurons,
   how the composition produces the capability's output.
7. **Re-implement in gate-graph IR** — TokenEmbed / residual writes for
   each feature direction, LookUp / LookUpExact for each attention
   operation, ReGLU for each FFN transformation, LinearHead for output.
8. **Verify bit-equivalence** (or close-to) against the trained
   model's output on the capability.
9. **Install as a compiled card** into the substrate.

Steps 2-6 are weeks to months per capability with current tools.
Steps 7-9 are hours once the circuit is traced. The bottleneck is
**interpretability**, not compilation.

## What to build when

Three tiers of capability, ordered by labor:

| Tier | Labor | Examples |
|---|---|---|
| **Trivial** — exact math or discrete lookups | Hours | adder, multiplier, is_prime, gcd, factorial, recall DB, country capitals, timezone conversions |
| **Designed** — novel circuits using IR primitives | Days to weeks | general planner, AST parser, simple type checker, analogy-by-structure-match, sequential reasoner |
| **Reverse-engineered** — tracing + compiling Gemma's own circuits | Weeks to months | semantic understanding circuits, induction heads, specific reasoning circuits |

Ship Trivial fast. Build Designed when a specific target is needed
and the circuit design is tractable. Reverse-engineered is the
long-horizon research bet — gives us the LM's own capabilities as
inspectable, reversible, auditable compiled cards.

## Validation summary (current state)

The first mechanistic-interpretability arc on Gemma 4 E4B validated
that steps 1-2 of the tracing workflow (corpus + activation patching)
produce clean results on capabilities whose information flow aligns
with Gemma's architecture. Atlas summary:

- Activation patching localizes capabilities to specific layer
  clusters (e.g. arithmetic to L22-L30 with L23 peak).
- Per-head ablation narrows further (e.g. L23 → H1+H4 carry the load).
- Q/K/V decomposition isolates V (content) as the signal carrier
  for arithmetic.
- Per-sub-head d_head=2 ablation is NOT a useful further-localization
  tool — signal is distributed across the head's full subspace.
- 7 capabilities mapped: arithmetic / factual recall / induction /
  counting / comparison / SV agreement / multi-step composition.
- Hub-sharing causally proven: L23 H1/H4 forced-attention
  intervention preserves SV agreement, comparison, counting.
- Multi-step composition (L24) is **deep-diffuse** — not currently
  compilable by any known substrate mechanism.

Full per-round detail: `MEMORY/atlas/tracing_intelligence_arc.md`
+ `MEMORY/atlas/tracing_roadmap_part_1.md`.

## Why it worked (architectural prediction)

Gemma's alternating SWA/global attention forces cross-operand
aggregation into global layers (L5, L11, L17, L23, L29, L35, L41).
Arithmetic requires seeing both operands → must happen at a global
layer. L23 and L29 were architecturally predicted; measurement
confirmed.

**Generalizable rule**: capabilities whose information flow matches
Gemma's structure should localize similarly. Capabilities that are
inherently distributed (open-ended semantics, long-range reasoning)
will not.

## Methodology nulls (what NOT to retry)

- Naive logit lens alone as tracing tool — top-5 at middle layers is
  noise; use only with rank trajectories of tracked tokens, alongside
  activation patching.
- Single-prompt activation patching for localization — overgeneralization
  risk; always aggregate across multiple inputs.
- Per-sub-head d_head=2 ablation as a further-localization tool inside
  a head — signal is distributed across the head's full subspace, not
  sparse in d_head=2 slots. The head is the right granularity.
- Standard L1-SAE features as targets for causal ablation on
  distributed composition circuits — reconstruction fidelity does NOT
  imply causal effect on deep-diffuse circuits. Verify causal effect
  under ablation BEFORE targeting SAE features for compilation.

## The corrected upper bound

> The substrate can host any bounded computation expressible as a
> fixed sequence of matmuls + activations. Since any trained
> transformer of Gemma's size computes exactly that, the substrate
> is Turing-equivalent to one. Plus: the substrate can host any
> computation humans can DESIGN as a gate-graph, even if no model
> has been trained to do it.

Formally: `Capabilities(substrate) ⊇ Capabilities(trained_model_of_same_size) ∪ Human_Designable_Circuits`.

The distance between today's substrate (multiplier + a few DBs) and
the theoretical upper bound is labor, not architecture.

## Related rules

- `capability_gain.md` — how to verify tracing + compilation actually delivered capability
- `embed_intelligence.md` — delivery mechanisms (card → Gemma tokens)
- `Substrate.md` — install mechanisms (where cards live)
- `augmentation_thesis.md` — strategic framing (tier-1/2/3)
- `probing_methodology.md` — per-tool methodology gates
- `workflow.md` — hypothesis-test discipline
- `MEMORY/atlas/tracing_intelligence_arc.md` — per-round validation receipts
- `MEMORY/atlas/tracing_roadmap_part_1.md` — full per-round narrative
