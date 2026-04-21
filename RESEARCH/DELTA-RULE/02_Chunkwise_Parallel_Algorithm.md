# DeltaNet — Chunkwise Parallel Algorithm

The core contribution: a training algorithm that turns DeltaNet's
sequential delta-rule recurrence into a sequence-parallel, matmul-rich,
tensor-core-friendly computation. Background and motivation in
[`01_DeltaNet_Overview.md`](01_DeltaNet_Overview.md); empirical speedup
numbers and full throughput comparison in
[`03_Empirics_and_Related_Work.md`](03_Empirics_and_Related_Work.md).
See [`00_INDEX.md`](00_INDEX.md) for the full doc set.

---

## 1. The bottleneck the algorithm solves

DeltaNet's recurrence is

```
S_t = S_{t-1}(I − β_t k_t k_t^T) + β_t v_t k_t^T       (Householder form)
    = S_{t-1} − β_t (S_{t-1} k_t − v_t) k_t^T         (SGD form)
```

Schlag 2021's original training algorithm was strictly sequential: one
step per token, full `d × d` state materialized. That's fine for small
models but badly hardware-inefficient on modern GPUs — no sequence-
level parallelism, no tensor cores, O(L) sequential kernel launches.
Linear attention doesn't have this problem because its recurrence
`S_t = S_{t-1} + v_t k_t^T` collapses into the parallel form
`O = (QK^T ⊙ M) V` with the causal mask `M`. DeltaNet's `I − β_t k_t k_t^T`
factor blocks that collapse.

The trick of this paper: find a reparameterization that **recovers the
linear-attention parallel form** while preserving delta-rule semantics,
so training can use FlashLinearAttention-style chunkwise kernels
(Yang 2024).

---

## 2. Pseudo-value reparameterization

**Claim:** `S_t` admits an additive representation
`S_t = Σ_{i=1}^{t} u_i k_i^T` for some `u_i ∈ ℝ^d`, where

```
u_i = β_i (v_i − v_i^old),     with   v_i^old = S_{i-1} k_i
```

Once the `u_i` vectors are known, DeltaNet's output computation becomes
**identical to linear attention with pseudo-values `u_i` substituted for
`v_i`:**

```
O = (Q K^T ⊙ M) U          where U ∈ ℝ^{L×d} stacks the u_i.
```

Every downstream step reuses the existing linear-attention infrastructure.

**Proof by induction.** Base case: `S_1 = β_1 v_1 k_1^T`, so `u_1 = β_1 v_1`.
Inductive step — rewrite the DeltaNet update using the hypothesis
`S_{t-1} = Σ_{i<t} u_i k_i^T`:

```
S_t = S_{t-1}(I − β_t k_t k_t^T) + β_t v_t k_t^T
    = Σ_{i<t} u_i k_i^T + β_t (v_t − Σ_{i<t} u_i (k_i^T k_t)) k_t^T
          ───────────────────────────────────────────
                      ≜ u_t
    = Σ_{i≤t} u_i k_i^T                                           (3)
```

`u_t` is computable in O(d) memory — no `d × d` state materialized. But
computing all `L` of the `u_i` values requires O(L² d) work and is
still sequential over `t`. The chunkwise form is what recovers
practical parallelism.

---

## 3. Chunkwise recurrence

Partition the sequence into `L/C` chunks of size `C` (typical `C = 64`
or `128`). Let `[t]` index the chunk, with `r ∈ [1, C]` indexing within
a chunk. Unrolling (4) gives:

```
S_t = Σ_{i=1}^{t} β_i (v_i k_i^T) · ∏_{j=i+1}^{t} (I − β_j k_j k_j^T)    (4)
```

Define per-chunk bookkeeping matrices:

- **Decay** `P_[t]^r = ∏_{i=1}^{r} (I − β_[t]^i k_[t]^i (k_[t]^i)^T)` —
  what must be applied to `S_[t]^0` to produce `S_[t]^r`.
- **Contribution** `H_[t]^r = Σ_{i=1}^{r} β_[t]^i v_[t]^i (k_[t]^i)^T
  P_{i+1}^r` — what the tokens inside chunk `[t]` add to `S_[t]^r`.

Then

```
S_[t]^r = S_[t]^0 P_[t]^r + H_[t]^r                              (5)
```

Both `P_[t]^r` and `H_[t]^r` are `d × d` — if materialized per position
they'd blow the memory budget. Same WY-representation trick saves it:

```
P_[t]^r = I − Σ_{i=1}^{r} w_[t]^i (k_[t]^i)^T                    (6)
H_[t]^r =     Σ_{i=1}^{r} u_[t]^i (k_[t]^i)^T

w_[t]^r = β_[t]^r (k_[t]^r − Σ_{i<r} w_[t]^i (k_[t]^i · k_[t]^r))  (7)
u_[t]^r = β_[t]^r (v_[t]^r − Σ_{i<r} u_[t]^i (k_[t]^i · k_[t]^r))
```

Both `w_[t]^i` and `u_[t]^i` are vectors in `ℝ^d` — O(Cd) storage per
chunk, not O(Cd²). Once computed, the chunk-to-chunk propagation
collapses to linear-attention-shaped matmuls:

```
S_[t+1] = S_[t] + (U_[t] − W_[t] S_[t]^T)^T K_[t]                (8)
O_[t]   = Q_[t] S_[t]^T + (Q_[t] K_[t]^T ⊙ M_C)(U_[t] − W_[t] S_[t]^T)
                                                                  (9)
```

where `Q_[t], K_[t], V_[t], W_[t], U_[t], O_[t] ∈ ℝ^{C×d}` stack the
chunk's per-position vectors and `M_C` is the chunk causal mask.

---

## 4. UT transform — making intra-chunk ops matmul-rich

Equation (7) is still a recurrence over `i` inside each chunk —
sequential, no tensor cores. The UT transform (Joffrain 2006, Dongarra
1989) converts it into a matrix inverse of a lower-triangular matrix:

```
T_[t] = (I + tril(diag(β_[t]) K_[t] K_[t]^T, −1))⁻¹ diag(β_[t])  (10)
W_[t] = T_[t] K_[t],     U_[t] = T_[t] V_[t]                     (11)
```

The triangular inverse is solved efficiently via forward substitution.
Everything else is matmuls — exactly what tensor cores want.

With this in hand, the full forward pass is:

```
for each chunk [t] (sequential over chunks):
    compute K_[t] K_[t]^T                           # matmul
    compute T_[t] via forward-substitution          # O(C²) work
    W_[t] = T_[t] K_[t], U_[t] = T_[t] V_[t]         # matmul
    intra-chunk output via (9)                      # matmul
    update S_[t+1] via (8)                          # matmul
```

Backward pass recomputes the hidden states instead of storing them, so
activation memory stays O(Ld) — the FlashLinearAttention pattern applied
to DeltaNet. PyTorch pseudocode in Appendix B of the paper; the Triton
implementation is adapted from FLA.

---

## 5. Complexity and throughput

- **Work:** `O(LCd + Ld²)` — strictly less than the naive parallel form
  `O(L²d + Ld²)` of linear attention when `C < L`.
- **Sequential steps:** `O(L/C)` chunk iterations (further reducible
  with chunk-level parallel scan, not used here).
- **Memory:** `O(Ld)` activations; never materialize per-token `S_t`.
- **C = L** recovers the fully parallel form; **C = 1** recovers the
  recurrent form. `C = 64` is the sweet spot for GPU occupancy.

Measured speedup of chunkwise form over the pure recurrent form on
sequences of length 512 → 16K at head dims 64/128/256 (Figure 1 of the
paper, single H100):

| Head dim | 512 | 2K | 8K | 16K |
|---:|---:|---:|---:|---:|
| 64 | ~2× | ~5× | ~12× | ~18× |
| 128 | ~2× | ~6× | ~18× | ~26× |
| 256 | ~2× | ~8× | ~22× | ~30× |

Speedup grows with both sequence length and head dimension — the two
axes where sequence-level parallelism and tensor cores matter most.

End-to-end 1.3B-model training throughput at different (training
length × batch size) pairs that multiply to 16K tokens:

- DeltaNet ≈ GLA throughput (both are FLA-based).
- Both significantly faster than Mamba.
- All three linear-time models beat Transformer++ at long sequences.

Full throughput comparison in [`03`](03_Empirics_and_Related_Work.md) §3.

---

## 6. Fully parallel form (for analysis, not training)

For completeness: the attention matrix of DeltaNet has closed form
`A_{ij} = k_j^T P_{j+1}^i q_i` for `j ≤ i`, factorable as
`A = (Q K^T ⊙ M) T` by combining (3) and (11). This is useful for
**interpretability analysis of DeltaNet as an "attention pattern"**
(Braun 2024) but not used for training — the matrix inverse in `T` is
cubic in sequence length without further tricks.

---

## 7. Generalizations the algorithm can absorb

The chunkwise algorithm extends to any `M_t` of the **Diagonal-Plus-Low-
Rank (DPLR)** form `M_t = D − a_t b_t^T`. DeltaNet uses `D = I,
a_t = β_t k_t, b_t = k_t`; S4 (Gu 2021) uses a data-independent DPLR.
**Gated DeltaNet** (Yang 2024b) uses `M_t = α_t (I − β_t k_t k_t^T)` —
adds a scalar decay on top. Same parallelism recipe applies; Gated
DeltaNet additionally fixes DeltaNet's length-generalization weakness
(see [`03`](03_Empirics_and_Related_Work.md) §Limitations).

**Block-diagonal generalized Householder** with block sizes that fit
GPU SRAM (e.g. 128) would let DeltaNet have a large overall head
dimension (and thus large recurrent state — key for recall-intensive
tasks) without paying the full `d²` intra-state-dependency cost. Called
out as future work; not implemented in this paper.
