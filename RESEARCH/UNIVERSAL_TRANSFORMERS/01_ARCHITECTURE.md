# UT-Family — Architecture

The unified design spec. Primitives, invariants, the three-stage RDT
structure, ACT halting, loop-index differentiation, attention
variants, MoE FFN, NAMM as orthogonal memory layer, and what's
net-new vs the current substrate. Companion to `00_INDEX.md`
(overview + decision), `02_IMPLEMENTATION.md` (math + APIs +
schemas), `03_TESTING.md` (substrate-relevance gates).

## Thesis

> **Depth recurrence (UT/RDT) gives compositional reasoning at
> fixed parameter count; ACT and Parcae make it trainable; NAMM
> bolts on as an orthogonal attention-pruning layer.**

## §1 Three-stage RDT structure (the canonical shape)

```
Input
  ↓
[Prelude P]         — standard transformer layers, run once
  ↓
[Recurrent Block R] — looped T times
  ↑_______↓         (h_t updated each loop; input e re-injected)
  ↓
[Coda C]            — standard transformer layers, run once
  ↓
Output
```

Recurrent block update rule (with input injection):

```
h_{t+1} = A · h_t + B · e + Transformer(h_t, e)
```

Where:
- `h_t` — hidden state after loop iteration t
- `e` — encoded input (output of Prelude), **re-injected at every loop** to prevent drift
- `A`, `B` — learned injection parameters (load-bearing for stability — see `02_IMPLEMENTATION.md` §1)
- `Transformer(·)` — attention + MLP block, weights tied across all T iterations

**The injection of `e` at every step is what prevents drift.** Pure
weight-tied recurrence without re-injection loses the original input
signal across deep iterations.

## §2 The 2018 UT ancestor

Original Universal Transformer (Dehghani et al.) is the same shape
without explicit injection term: the per-position recurrent state
`H_t` is updated as

```
A_t = LayerNorm( (H_{t-1} + P_t) + MultiHeadSelfAttention(H_{t-1} + P_t) )
H_t = LayerNorm( A_t + Transition(A_t) )
```

with `P_t` a 2D (position, time) sinusoidal embedding summed
elementwise. Transition is either a separable convolution or a
position-wise FFN. Encoder-decoder; decoder additionally
cross-attends to final encoder `H_T`.

**Why this generalizes where vanilla Transformer doesn't:**
Vanilla Transformer foregoes RNN's iterative inductive bias.
Length-extrapolation tests (train 40, eval 400) on Copy / Reverse /
Addition: vanilla Transformer 0.53 / 0.13 / 0.07 char-acc; UT 0.91
/ 0.96 / 0.34. Same architecture family, recurrent inductive bias
recovers the algorithmic-task generalization.

**Turing-completeness.** Under reasonable assumptions, UT is
computationally universal. Reduces to a Neural GPU by
parameterizing self-attention as identity, transition as
convolution, and `T = input_length`. Vanilla Transformer cannot
scale depth with input size and is therefore strictly weaker.

## §3 Adaptive Computation Time (ACT)

Per-position learned halting probability (Graves 2016, applied
per-symbol in UT). Each position predicts a halting score per loop
iteration; cumulative score crosses threshold ⇒ position halts and
its state is copied forward to subsequent iterations. Final encoder
output is the per-position last-active representation.

**Empirical evidence ACT works:**
- bAbI tasks requiring 1 / 2 / 3 supporting facts ⇒ avg ponder time
  2.3 ± 0.8 / 3.1 ± 1.1 / 3.8 ± 2.2 (model learns to allocate
  computation to difficulty).
- LAMBADA: dynamic halting beats fixed 6 / 8 / 9-step UT — even
  though average dynamic depth is 8.2 ± 2.1, the learned variation
  outperforms fixed depth. ACT acts as **regularizer** by
  incentivizing fewer steps where possible.

**Modern framing (OpenMythos):** ACT addresses the **overthinking
failure mode** — beyond a certain depth, recurrence drifts past
the answer into noise. The model needs a learned signal for "answer
has converged, stop iterating."

## §4 Loop-index differentiation (open conjecture)

Without any positional signal across loops, the same weights must
handle both early-stage pattern matching and late-stage refinement.
A **RoPE-like embedding of the loop index** injected alongside the
input at each step would let the same parameters implement
functionally distinct operations across iterations.

If used, each loop is **not a repetition** — it is a distinct
computational phase, all sharing weights but operating in different
representational regimes. Substantially increases expressiveness
without parameter cost.

Concrete encoding (from `02_IMPLEMENTATION.md` §7): sinusoidal
embedding of `t ∈ [0, T)` summed with input injection `B·e`.

## §5 Attention variants

Switchable per `cfg.attn_type`:

| Option | Class | Description |
|---|---|---|
| `gqa` | GQA | Grouped Query Attention (Ainslie 2023). `n_kv_heads < n_heads` reduces KV-cache memory by `n_heads / n_kv_heads`. Native Flash Attention 2 path when `flash-attn>=2.8.3`. |
| `mla` | MLA | Multi-Latent Attention (DeepSeek-V2). Caches a compressed KV latent (`kv_lora_rank`) rather than full K/V, with split RoPE / no-RoPE head dims for position-aware compression. RoPE applied to Q and K **before** caching, so retrieval doesn't re-rotate. |

Both are KV-cache efficiency tricks orthogonal to RDT structure —
they reduce the bandwidth cost of long-context decode without
changing the recurrence.

## §6 Mixture of Experts (FFN replacement)

For large parameter counts, every FFN in the **Recurrent Block** is
replaced with a fine-grained MoE layer:

- Each FFN split into many small experts, each `1/m` the normal size
- Router selects top-`m·K` experts per token via learned affinity
- Small number of **always-on shared experts** absorb common
  cross-domain knowledge (syntax, basic reasoning, general context)
  — would otherwise be redundantly learned by every routed expert
- Routing-collapse prevention via **bias term on router logits**
  dynamically adjusted during training (keeps load balanced without
  distorting loss signal)

**Per-loop expert specialization.** As `h_t` evolves across loop
iterations, the router may select **different expert subsets at
each depth** — making every loop computationally distinct despite
shared weights. MoE provides breadth; looping provides depth.

If activation ratio is ~5%, total parameter count can be hundreds
of billions while activated-per-token compute stays small —
storage number, not compute number.

## §7 NAMM (Neural Attention Memory Models)

**Orthogonal layer**, not part of the recurrence. NAMMs are small
neural classifiers that decide per-token whether to **keep** or
**forget** each token in the transformer's working-memory context.

Three-step execution (full mechanics in `02_IMPLEMENTATION.md` §2):

1. **Spectrogram** — attention values per token converted to
   frequency-based representation (well-established across audio,
   medicine, seismology)
2. **EMA compression** — elementwise exponential moving average
   condenses to compact fixed-size feature summary
3. **Classifier score** — learned NN outputs keep/forget decision

**Universal transfer property.** Because NAMMs condition **only on
attention matrices** (universal across transformer layers and
architectures), one NAMM trained on Llama-3-8B language tasks
zero-shot transfers to:

- Llama-70B (cross-scale)
- Llava Next Video (cross-modality, vision)
- Decision Transformer (cross-domain, RL)

Beats hand-designed memory pruning (H₂O, L₂) on LongBench /
InfiniteBench / ChouBun while reducing context size — pruning
**without** the performance cost of prior methods.

**Layer-specialized behavior** (observed empirically): early layers
retain global / keyword tokens; later layers prune those (already
absorbed) and focus on local detail. Code tasks: prune contiguous
chunks of whitespace / comments / boilerplate. NL tasks: prune
mid-sentence grammatical filler.

## §8 What's invariant vs what each paper adds

| Mechanism | UT (2018) | OpenMythos (2026) | NAMM (2024) |
|---|---|---|---|
| Param-tied depth recurrence | ✓ (founding) | ✓ (RDT) | — (orthogonal) |
| Per-loop input injection | implicit (P_t sum) | explicit `B·e` | — |
| ACT halting | ✓ (founding) | ✓ (carries forward) | — |
| 2D (pos, time) embedding | ✓ (sinusoidal sum) | RoPE loop-index conjectured | — |
| Stability fix | — (not addressed) | **Parcae LTI: `A := Diag(-exp(log_A))`** | — |
| Attention variant | full softmax | GQA / MLA | reads any |
| FFN | dense | **Fine-grained MoE + shared experts** | reads any |
| Per-loop LoRA | — | optional (Bae 2024) | — |
| Continuous depth-wise batching | — | ✓ | — |
| Scaling laws | informal | **Parcae power laws** | — |
| Memory pruning | none | none | **NAMM (spectrogram + EMA + classifier)** |
| Universal cross-arch transfer | — | — | ✓ (reads attention only) |
| Training method | gradient | gradient | **evolutionary** |

## §9 Substrate framing

Where each primitive sits relative to existing substrate code:

- **D5 recurrent substrate** (`calm/llm_computer/recurrent_substrate.py`)
  — already implements param-tied depth recurrence via `n_iterations`
  kwarg. **No explicit injection term** (no `B·e` re-injection;
  pure layer-stack iteration). This is the substrate's RDT-family
  position; missing per-iter differentiation, learned halting, and
  injection-stability gating.
- **DT / CopyAugmentedDeltaNet** (`copy_augmented_delta.py`) — does
  Householder fast-weight recurrence on the **sequence axis**
  (per-position, not per-depth). Orthogonal to UT/RDT axis.
- **Compiled cards** (`programs/`) — exact, no recurrence; live
  inside Gemma's attention sub-heads via in-attention install
  (`install_card_in_attention`). Different family entirely.
- **`MultiStepReasoningFacade`** — decode-path multi-step CALC via
  parse → safe_eval → step-through bias. Solves multi-step
  *computable* composition; does NOT solve multi-step *reasoning*
  composition (no safe_eval reduction available).
- **L24 deep-diffuse multi-hop reasoning gap** — explicitly flagged
  in `augmentation_thesis.md` as not currently compilable by any
  known substrate mechanism. RDT is the architectural family
  designed for exactly this gap.

What this synthesis adds vs current substrate:

1. **Loop-index RoPE** — cheap addition to D5; gives same-weight
   iterations distinct functional roles
2. **ACT halting head** — small learned scalar per position decides
   `n_iterations` dynamically per input, replacing the fixed
   per-mode-token dispatch
3. **Parcae stability discipline** — only applies if D5 grows
   explicit injection. If/when, the `Diag(-exp(log_A))` parameterization
   is the load-bearing fix
4. **NAMM** — would require GemmaSubstrate Python serving path
   (llama.cpp has no Python attention hook) + evolutionary training
   infra. Wrong-layer fit; skip
5. **RDT Tier-3 card** — for L24 deep-diffuse gap. Speculative;
   needs failure-surface gate first

Ranked + costed in `03_TESTING.md` §3-7.

## §10 Memorization-reasoning tradeoff (caveat)

Looped models exhibit a documented dichotomy: **looping improves
reasoning but can hurt memorization**. Recurrent structure is
optimized for iterative composition (running a reasoning chain
forward); doesn't inherently improve storage of rote facts.

For this repo: factual recall is handled by `KnowledgeStore` recall
cards (`Substrate.md` §"Persistent Knowledge") and CALM's 39
knowledge backends (`*_kb.py`). RDT-shaped cards would target
*reasoning* gaps, not recall — the existing recall machinery covers
the memorization side of the dichotomy.

Looping-based regularization (Saunshi 2025) can balance the
tradeoff during training: stronger looping for reasoning tasks,
relaxed for retrieval. Relevant only if RDT cards are trained
multi-task; single-purpose cards inherit the bias of their training
distribution.
