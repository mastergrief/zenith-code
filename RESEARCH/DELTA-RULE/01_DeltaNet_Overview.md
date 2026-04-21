# DeltaNet — Overview

The *what* and *why*. The problem linear attention has with recall, the
delta-rule fix, and how it slots into a LLaMA-style transformer. Math
lives in [`02_Chunkwise_Parallel_Algorithm.md`](02_Chunkwise_Parallel_Algorithm.md);
numbers live in [`03_Empirics_and_Related_Work.md`](03_Empirics_and_Related_Work.md).
See [`00_INDEX.md`](00_INDEX.md) for the full doc set.

---

## 1. TL;DR

Linear-attention transformers replace softmax with a dot-product kernel
and can be rewritten as a linear RNN with a matrix-valued hidden state
`S_t ∈ ℝ^{d×d}`. The vanilla update `S_t = S_{t-1} + v_t k_t^T` is
cheap but additive — key-value pairs accumulate without any mechanism
for removal. Once the sequence length exceeds the head dimension, keys
collide, and the model underperforms softmax attention on in-context
retrieval (Schlag 2021; Arora 2024).

DeltaNet replaces the additive update with the **Widrow-Hoff delta
rule**: read the current key, compare against the target value, and
write the correction. This gives much better associative recall at the
cost of a recurrence whose state-to-state operator is no longer
elementwise — it's a rank-one update `I - β_t k_t k_t^T` applied to
`S_{t-1}`. The DeltaNet mechanism itself has been known since Schlag
2021; the contribution of this paper is the **hardware-efficient
training algorithm** that makes it practical at scale. With that
algorithm in hand, 1.3B DeltaNet (100B tokens) beats Mamba and GLA on
LM perplexity and zero-shot downstream tasks. Hybrid variants — interleaving
DeltaNet with sliding-window or two global attention layers — beat
Transformer++.

---

## 2. The problem with additive linear attention

Softmax attention is `o_t = Σ_i softmax(k_i^T q_t) v_i` — quadratic in
sequence length, requires a growing KV cache. Linear attention
(Katharopoulos 2020) replaces `exp(k^T q)` with `φ(k)^T φ(q)` and
rearranges:

```
S_t = S_{t-1} + v_t φ(k_t)^T ∈ ℝ^{d×n}
o_t = S_t φ(q_t)
```

Constant-memory inference, no KV cache. But **purely additive** — the
model cannot deallocate a written key-value pair to make room for a new
one. From the fast-weight-programmer perspective (Schlag 2021), this is
a Hessian-like update with limited memory capacity; once `L > d`, keys
collide.

On recall-heavy real-world tasks (SWDE key-value extraction, SQuAD,
FDA), gated linear transformers like Mamba and GLA underperform softmax
transformers — even though they match on perplexity. Recall is where
the gap lives.

---

## 3. The delta rule — two intuitions

### As online SGD

DeltaNet's update is one SGD step on a regression loss:

```
ℒ_t(S) = ½ ‖S k_t − v_t‖²
S_t = S_{t-1} − β_t ∇_{S_{t-1}} ℒ_t = S_{t-1} − β_t (S_{t-1} k_t − v_t) k_t^T
```

where `β_t ∈ (0, 1)` is a data-dependent "learning rate"
`β_t = σ(W_β x_t)`. Linear attention corresponds to the same picture
with the negative inner-product loss `ℒ_t = −⟨S k_t, v_t⟩`, whose
gradient is the additive outer-product term.

### As retrieve-then-interpolate

Equivalently: read the old value associated with `k_t`, interpolate
with the new target, and rewrite.

```
v_t^old = S_{t-1} k_t              (retrieve)
v_t^new = β_t v_t + (1 − β_t) v_t^old   (interpolate)
S_t = S_{t-1} − v_t^old k_t^T + v_t^new k_t^T   (remove + write)
```

`β_t = 1` → hard overwrite (`v_t^new = v_t`). `β_t = 0` → no-op
(`S_t = S_{t-1}`). The soft-writing-strength framing is what the original
DeltaNet paper uses; the SGD framing is what recent work (Longhorn,
TTT, Titans) picks up.

Collecting terms gives the compact form used throughout the paper:

```
S_t = S_{t-1}(I − β_t k_t k_t^T) + β_t v_t k_t^T    (generalized Householder)
```

`I − β_t k_t k_t^T` is a rank-one perturbation of the identity — a
**generalized Householder reflector**. When `β_t = 1` and `‖k_t‖ = 1`,
it's a projection that erases one direction while preserving the other
`d − 1`.

The output read-out is unchanged from linear attention: `o_t = S_t q_t`.

---

## 4. DeltaNet inside a transformer

The paper wires DeltaNet into a LLaMA-style architecture (Transformer++,
Touvron 2023) by dropping in the DeltaNet layer wherever self-attention
used to sit. Parameter allocation is roughly the same: `4 d²` for the
DeltaNet layer, `8 d²` for the SwiGLU FFN.

**Block diagram:**

```
  Input
    │
    ▼
 RMSNorm → DeltaNet ─┐
                    ⊕ residual
 RMSNorm → SwiGLU ──┘
    │
    ▼
  ...×N layers...
 RMSNorm → Linear → logits
```

**Inside the DeltaNet layer:**

```
  x_t
   ├─ Linear → Conv → SiLU → L2-norm → k_t
   ├─ Linear → Conv → SiLU → L2-norm → q_t
   ├─ Linear → Conv              → v_t
   └─ Linear → σ                 → β_t
    └─ RMSNorm → Linear (output projection)
```

Notable choices:

- **SiLU feature map** on Q and K (not `1 + ELU` as in Katharopoulos
  2020; SiLU ablated best, consistent with Qin 2022 / Dao-Gu 2024).
- **L2 normalization** on Q and K, not L1. L2-norm ensures the
  eigenvalues of `I − β_t k_t k_t^T` lie in `[0, 1]` (multiplicity `d−1`
  at eigenvalue 1, one eigenvalue at `1 − β_t ‖k_t‖²`). At `β_t = 1`,
  the transition matrix becomes a projection — clean "erase this
  subspace, preserve the rest" interpretation. Empirically L2 beats
  L1 by a large margin (see ablations in
  [`03`](03_Empirics_and_Related_Work.md)).
- **Short depthwise convolution** on Q, K, V after the linear
  projections. Generalizes the shift-SSM (H3, Fu 2023); standard in
  Mamba and xLSTM. Cheap in params and FLOPs.

---

## 5. Hybrid variants

Linear attention lacks positional information and struggles with
precise local comparisons. Two hybrid architectures fix this by
interleaving DeltaNet with softmax attention:

- **DeltaNet + Sliding Window Attention (SWA).** Every other layer is
  SWA. Follows Griffin (De 2024) and Samba (Ren 2024).
- **DeltaNet + two global attention layers.** Replace just two
  DeltaNet layers (the 2nd and the `(N/2 + 1)`-th) with full softmax
  attention. Follows H3 (Fu 2023) and YOCO (Sun 2024).

Both hybrids improve over pure DeltaNet on language modeling AND on
recall-intensive tasks, and both beat strong Transformer++ baselines.
The two-global-layer variant hits the best SWDE/SQuAD/FDA numbers in
the paper — see [`03`](03_Empirics_and_Related_Work.md).

---

## 6. Placement in the linear-RNN zoo

Many recent linear-RNN / SSM models fit the general form:

```
S_t = S_{t-1} • M_t + v_t k_t^T,    o_t = S_t q_t
```

where `•` is an associative operator. What varies is `M_t`:

| Model | `M_t` | Notes |
|---|---|---|
| Linear Attention | — (additive only) | Katharopoulos 2020 |
| RetNet | scalar decay `γ` | data-independent |
| GLA | `Diag(α_t)` | data-dependent, elementwise |
| Mamba / Mamba-2 | elementwise `exp(−(α_t 1^T) ⊙ exp(A))` | SSM-derived |
| RWKV-6 / HGRN-2 | `Diag(α_t)` | elementwise |
| **DeltaNet** | **`I − β_t k_t k_t^T`** | **structured, non-elementwise** |
| Gated DeltaNet | `α_t (I − β_t k_t k_t^T)` | delta rule + scalar decay |

The models that use Hadamard / diagonal `M_t` get cheap recurrences
(O(dn) per step) and trivially parallelize. DeltaNet's structured
rank-one-plus-identity `M_t` models **richer interactions** — state-to-
state, not just elementwise — but would cost O(dn²) per step naively.
The chunkwise algorithm in [`02`](02_Chunkwise_Parallel_Algorithm.md)
is what recovers practical training speed without giving up the
structured update.

The general class `M_t = D − a_t b_t^T` (Diagonal-Plus-Low-Rank, DPLR)
was explored in S4 (Gu 2021) with data-independent transitions.
DeltaNet is the data-dependent DPLR special case
`D = I, a_t = β_t k_t, b_t = k_t`.
