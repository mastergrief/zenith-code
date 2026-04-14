# HRM — Training Procedure

All the training tricks that make HRM practical. BPTT's memory
problem, the one-step gradient solution, deep supervision, ACT
(Adaptive Computational Time), and the architectural choices that
tie it all together.

**Concept-owner for:** BPTT and its costs, the Implicit Function
Theorem + Neumann-series derivation of the one-step gradient, when
the approximation holds, deep supervision with state detachment,
ACT as Q-learning over segment count, why Q-learning is stable
without replay / target networks, inference-time scaling, Llama-
style architectural details (Post-Norm, RMSNorm, GLU, RoPE,
Adam-atan2).

See also: [`01_HRM_Overview.md`](01_HRM_Overview.md) for motivation
and results, [`02_Hierarchical_Convergence.md`](02_Hierarchical_Convergence.md)
for why the architecture needs any of this at all.

---

## 1. BPTT and why it doesn't fly

The obvious way to train a recurrent network is **Backpropagation
Through Time (BPTT)**: store every intermediate state during the
forward pass, chain derivatives backward through time.

Memory: `O(T)` for `T` steps. For HRM with `N = 4, T = 16` and deep
supervision (below), that's 64+ states per sample per segment —
prohibitive for large batches on modest GPUs. The memory pressure
forces small batches, which cripples GPU utilization.

Biological aside: the brain almost certainly doesn't do BPTT.
Cortical credit assignment appears to be short-range and online;
there is no evidence of activity history being replayed during
learning.

### The DEQ alternative

If a recurrent system converges to a fixed point, you can compute
gradients **at the fixed point** without unrolling. This is the
insight behind **Deep Equilibrium Models (DEQ)**.

HRM's architecture (see [`02`](02_Hierarchical_Convergence.md)) is
already built around fixed-point convergence within each cycle. So
DEQ-style training is natural here.

---

## 2. The one-step gradient — derivation

Suppose in cycle `k`, the L-module converges to `z_L*` satisfying

    z_L* = f_L(z_L*, z_H^(k−1), x̃; θ_L)

The H-module then takes a single step:

    z_H^k = f_H(z_H^(k−1), z_L*; θ_H)

Let

    F(z_H, θ) = f_H(z_H, z_L*(z_H); θ)

so the high-level fixed point is `z_H* = F(z_H*, θ)`. Let `J_F =
∂F/∂z_H` be its Jacobian.

### Implicit Function Theorem

By IFT, the gradient of `z_H*` w.r.t. `θ` — without unrolling — is:

    ∂z_H*/∂θ = (I − J_F|_{z_H*})^(−1) · ∂F/∂θ|_{z_H*}              (Eq. 1)

Exact, but that inverse is expensive (solve a linear system of size
`d × d`) and numerically delicate near `J_F ≈ I`.

### Neumann series expansion

Using the identity

    (I − J)^(−1) = I + J + J² + J³ + …

(valid for `‖J‖ < 1`), we can expand Eq. 1 as

    ∂z_H*/∂θ = (I + J_F + J_F² + …) · ∂F/∂θ

### The 1-step approximation

Keep only the first Neumann term: `(I − J_F)^(-1) ≈ I`. Then Eq. 1
simplifies to

    ∂z_H*/∂θ_H ≈ ∂f_H/∂θ_H
    ∂z_H*/∂θ_L ≈ (∂f_H/∂z_L*) · (∂z_L*/∂θ_L)
    ∂z_H*/∂θ_I ≈ (∂f_H/∂z_L*) · (∂z_L*/∂θ_I)                       (Eq. 2)

Applying the same trick to `z_L*`:

    ∂z_L*/∂θ_L ≈ ∂f_L/∂θ_L
    ∂z_L*/∂θ_I ≈ ∂f_L/∂θ_I                                         (Eq. 3)

Substitute Eq. 3 into Eq. 2: gradients involve only the **last**
state of each module. No unrolling. Memory: `O(1)`.

### In code

~~~python
def hrm(z, x, N=2, T=2):
    x = input_embedding(x)
    zH, zL = z
    with torch.no_grad():                     # discard all intermediate states
        for _i in range(N * T - 1):
            zL = L_net(zL, zH, x)
            if (_i + 1) % T == 0:
                zH = H_net(zH, zL)
    # 1-step grad: only the final L + H update are in the autograd graph
    zL = L_net(zL, zH, x)
    zH = H_net(zH, zL)
    return (zH, zL), output_head(zH)
~~~

The first `N·T − 1` steps run under `torch.no_grad()`. Only the
**last** L-update and H-update contribute to the autograd graph.
Backward pass is `O(1)` regardless of the total number of HRM steps.

---

## 3. When does the approximation hold?

The first-Neumann approximation `(I − J_F)^(-1) ≈ I` is:

- **Exact** when `J_F = 0` — the map is locally constant at the fixed
  point.
- **Good** when the spectral radius `ρ(J_F)` is small.
- **Poor** when `J_F` is close to `I` (near-marginal stability).

Stable fixed points of contractive dynamics satisfy `‖J_F‖ < 1`, so
the Neumann series converges and the first term is a reasonable
approximation. Empirically, HRM reaches near-perfect accuracy on the
headline benchmarks, so whatever slack the approximation introduces
is absorbed by the rest of the architecture.

### Theoretical cousins

- **Equilibrium propagation** (Scellier & Bengio, 2017). Replaces
  BPTT with local update rules at equilibrium. Biologically
  motivated.
- **Target propagation.** Similar biological motivation, different
  formal machinery.

The paper notes the method "aligns well with the perspective that
cortical credit assignment relies on short-range, temporally local
mechanisms rather than on a global replay of activity patterns."

### Unanswered questions

- The paper doesn't characterize the error of the 1-step
  approximation as a function of `‖J_F‖`.
- Whether specific architectural choices (Post-Norm, RMSNorm)
  implicitly keep `‖J_F‖` small during training is not analyzed.
- Does using a 2-term Neumann approximation
  `(I − J_F)^(-1) ≈ I + J_F` buy you anything? Unclear. At some
  point the memory savings evaporate.

---

## 4. Deep supervision — dense gradient signal without extra memory

Even with the one-step gradient, a single forward pass gives one
loss evaluation per sample. That's sparse. HRM fixes this with
**deep supervision**.

For each input `(x, y)`, HRM runs multiple forward passes called
**segments**. Each takes the previous segment's final hidden state
as its starting state — but **state is `detach()`-ed between
segments**. Gradients don't flow from segment `m+1` back through
segment `m`; each segment gets its own clean gradient update.

~~~python
for x, y_true in train_dataloader:
    z = z_init
    for step in range(N_supervision):
        z, y_hat = hrm(z, x)
        loss = softmax_cross_entropy(y_hat, y)
        z = z.detach()                # break gradient chain between segments
        loss.backward()
        opt.step()
        opt.zero_grad()
~~~

Three useful properties:

- **Dense supervision.** The model gets a loss signal every segment,
  not just at the end of the full trajectory. Analogous to auxiliary
  heads at intermediate layers in deep image classification networks.
- **Cheap.** Because state is detached, memory never grows with
  segment count.
- **Regularizing.** The H-module must produce useful answers at
  every segment boundary, which pushes it toward representations
  useful at multiple depths of thinking — not just at the final
  step.

The paper notes deep supervision empirically outperforms Jacobian-
based regularization techniques that other DEQ-style models use.
This is a small but important finding: a simpler training recipe
beats more sophisticated alternatives in practice.

---

## 5. Adaptive Computational Time (ACT) — thinking, fast and slow

### Cognitive-science motivation

Kahneman's System 1 / System 2 framing: the brain switches between
quick intuitive thinking and slow deliberate reasoning depending on
task difficulty. Importantly, the same neural circuits do both — it's
a **modulation of runtime**, not a switch between different systems.
Neuroscientific evidence places both modes in overlapping prefrontal-
cortex and default-mode-network regions.

HRM implements this as a **halting head** that decides per-example
how many segments to run.

### Q-learning formulation

A small **Q-head** reads the H-module's final state and predicts
Q-values for two actions:

    Q̂^m = σ(θ_Q^⊤ · z_H^(m·N·T))    →    (Q̂_halt^m, Q̂_continue^m)

The halt action is chosen when:

- Segment count hits `M_max` (hard cap), OR
- `Q̂_halt > Q̂_continue` AND segment count ≥ `M_min`.

`M_min` is stochastic: with probability `ε`, sampled uniformly from
`{2, …, M_max}` (exploration — encouraging longer thinking);
otherwise 1. Standard exploration/exploitation balance.

### Reward: did we get the answer right?

The MDP for Q-learning:

- **State:** HRM hidden state `z^m`.
- **Actions:** `{halt, continue}`.
- **Reward:** `1{ŷ^m = y}` if halt (correct → +1, wrong → 0); `0` if
  continue.
- **Targets:**

      Ĝ_halt^m     = 1{ŷ^m = y}
      Ĝ_continue^m = Q̂_halt^(m+1)                              if m ≥ N_max
                   = max(Q̂_halt^(m+1), Q̂_continue^(m+1))       otherwise

Per-segment loss combines prediction and Q-learning BCE:

    L^m_ACT = Loss(ŷ^m, y) + BinaryCrossEntropy(Q̂^m, Ĝ^m)

---

## 6. Why ACT is stable

Deep Q-learning is classically unstable — the standard fixes are
**replay buffers** and **target networks**, both absent here.

The paper's stability argument comes from recent theory (Gallici et
al. 2024) showing Q-learning converges if three conditions hold:

1. Network parameters are bounded.
2. Weight decay is used during training.
3. Post-normalization layers are present.

HRM satisfies all three:

- **Post-Norm** architecture (normalization after residual addition,
  not before). Paired with RMSNorm.
- **AdamW** optimizer — provably solves an `L∞`-constrained
  optimization problem, effectively bounding `‖θ‖_∞ ≤ 1/λ` for
  weight-decay coefficient `λ`.

So Q-learning is convergent **by construction of the architecture**,
not by add-on stabilization tricks.

This is a subtly important design decision. Most Q-learning systems
bolt on replay + target-network machinery as a fix. HRM's authors
picked architectural choices that make Q-learning inherently stable,
trading a little flexibility in architecture choice for a massive
simplification of the training loop.

---

## 7. Inference-time scaling — for free

Because ACT's halting decision is data-driven at inference time, a
model trained with `M_max = 8` can be **run** at `M_max = 16`
without retraining — and it continues to gain accuracy on harder
cases (Figure 5c).

This is a clean form of **inference-time scaling**, analogous in
spirit to "let o1 think for 30 more seconds" but implemented
architecturally rather than as a prompting trick. No separate
inference-scaling infrastructure, no RL-over-thought-traces
training, nothing. Just bump `M_max` at eval time.

### Task-dependent effect

- **Sudoku** (long-term planning): extra compute yields meaningful
  gains. Makes sense — harder puzzles genuinely need more search.
- **ARC-AGI** (few transformations per task): extra compute
  saturates quickly. The model doesn't burn steps artificially when
  they're not needed — it halts when the Q-head says it's done.

This is the version of inference-time scaling you'd actually want:
adaptive by example, not a global hyperparameter on the whole
inference run.

---

## 8. Architectural details

HRM's Transformer blocks are modern-standard, Llama-style:

- **Encoder-only** (no causal mask; both input and output are grids,
  not autoregressive sequences).
- **Rotary Positional Encoding (RoPE).**
- **Gated Linear Units (GLU)** in the FFN.
- **RMSNorm**, with scale and bias parameters **removed**.
- **No bias** on linear layers.
- **Post-Norm** (normalization after residual, not before) — critical
  for Q-learning stability (§6).
- **Truncated LeCun Normal** weight init.
- **Adam-atan2** optimizer — scale-invariant Adam variant — with
  constant LR + linear warm-up.

Both `f_L` and `f_H` are stacks of identical Transformer blocks
sharing the same hyperparameters. Multi-input fusion (L-module
taking `z_L^(i−1)`, `z_H^(i−1)`, `x̃`) is just **element-wise
addition**. More sophisticated gating could help — flagged as
future work.

For small-sample experiments, the output softmax is replaced with
**stablemax**, a numerically-stable softmax variant that improves
generalization on tiny datasets. The sequence-to-sequence loss is
averaged over all tokens.

Representation: inputs and outputs are tokenized grids (e.g., a
30×30 ARC grid = 900 tokens), flattened and padded to a maximum
sequence length.

---

## 9. Putting it all together

A full HRM training loop looks like this conceptually:

1. Sample a batch `(x, y)` from the training set.
2. Initialize hidden state `z = z_init`.
3. For each supervision segment `m = 1, …, M_max`:
   a. Forward pass through HRM (with no-grad for all but the last
      L/H step).
   b. Compute prediction loss + Q-learning BCE loss.
   c. Backward + optimizer step.
   d. Detach hidden state; carry to next segment.
   e. Check halting condition; if halt, break.
4. Next batch.

Four layered ideas:

- **Hierarchical convergence** (see
  [`02`](02_Hierarchical_Convergence.md)) gives you effective depth
  without instability.
- **One-step gradient** makes the training memory `O(1)` per
  segment.
- **Deep supervision** makes training memory `O(1)` across segments
  and gives dense gradient signal.
- **ACT** decides per-example how many segments to run, giving
  inference-time scaling.

Each idea is individually well-known (DEQ, auxiliary supervision,
adaptive halting, Q-learning). The contribution is composing them
into a stable recipe where each piece reinforces the others — in
particular, the Post-Norm + AdamW + RMSNorm architectural choices
serve double duty (enabling Q-learning stability *and* the one-step
gradient approximation).

---

## 10. Open questions about training

- **One-step gradient approximation error.** Uncharacterized. Would
  a two-step Neumann truncation help? What's the performance
  ceiling of the first-term approximation on different tasks?
- **Deep-supervision segment count `M`.** How many segments are
  optimal? How does accuracy respond to varying `M` at training
  time vs. inference time?
- **ACT stability in other architectures.** The Gallici-et-al.
  convergence argument depends on Post-Norm. Would HRM's Q-learning
  be stable with Pre-Norm? With standard LayerNorm instead of
  RMSNorm? With Adam instead of AdamW?
- **Can HRM be trained with PEFT / LoRA on top of a base model?**
  An open route to applying HRM-style reasoning to existing LLMs
  without the prohibitive cost of pretraining from scratch.
- **Scaling laws.** Does `N`, `T`, or model size need to grow with
  task complexity? How?
- **Curriculum learning.** HRM is trained on a fixed task
  distribution per run. Whether curriculum / task-difficulty
  scheduling helps is untested.
