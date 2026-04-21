# Lotus — Methodology

← back to [LOTUS.md](LOTUS.md)

**Fig. 1:** The comparison between previous method (e.g. GaLore) with fixed switching frequency and our greedy search strategy that updates the subspace adaptively. G∗ is the displacement of the unit gradient in a subspace. When the average displacement of unit gradient vector G_unit is lower than γ, the subspace will be switched.

## 3 Methodology

### 3.1 Adaptive Subspace Switching

Refreshing the orthogonal projector with the latest full-rank gradient realigns the low-rank basis with the its current dominant directions, so the top r singular vectors reclaim energy that had drifted outside the stale subspace; the Frobenius norm of the compressed gradient therefore "jumps back up." Yet because different spectral components of the gradient drift at different speeds, a fixed update frequency both wastes compute on already stable directions and allows fast-moving ones to leak energy before the next refresh. An adaptive switching schedule throttles and accelerates the process according to subspace drift to solve this imbalance.

To quantify how much displacement such a schedule can preserve, consider the ideal scenario in which every projected gradient step points in exactly the same direction; then the cumulative displacement after k steps is

    D_ideal = ‖ Σ_{i=0..k-1} −α · ĝ_{t−i} ‖_2
            = α · ‖ Σ_{i=0..k-1} ĝ_{t−i} ‖_2                      (1)

In the "best‑aligned" case where all ĝ are parallel and have unit norm, D_ideal ≈ k · α, where α is the learning rate. Then the actual displacement would be:

    D_actual = ‖ Σ_{i=0..k-1} Δw_{t−i} ‖_2
             = ‖ Σ_{i=0..k-1} −α · P_k · ĝ_{t−i} ‖_2               (2)

Then, we define the path‑efficiency ratio:

    ρ_t = D_actual / D_ideal
        = ‖ Σ_{i=0..k-1} P_k · ĝ_{t−i} ‖_2 / ‖ Σ_{i=0..k-1} ĝ_{t−i} ‖_2
        ∈ [0, 1]                                                    (3)

when ρ_t ≈ 1, the gradients remain confined within a narrow directional cone, indicating that the current subspace P_k is sufficiently representative for optimization. If ρ_t ≪ 1, significant cancellation occurs between successive steps, suggesting that the gradients exhibit substantial directional variation or extend beyond the span of the subspace P_k. Lotus adaptively switches the subspace when ρ_t < γ and t − t_last ≥ T_min, with threshold γ ∈ (0, 1). Noticing that we set a minimum interval condition, constraint t − t_last ≥ T_min is imposed to prevent excessive subspace switches during the initial noisy phase of optimization. Especially, k = 1 means that ρ_t reduces to the single-step displacement-gradient ratio.

**Lemma 3.1 (one‑step projected decrease).** If ρ_t ≥ ρ and the loss has standard L-smoothness. We can apply the standard upper bound for L-smooth functions to the subspace projected update rule, then:

    ℒ(w_{t+1}) ≤ ℒ(w_t) − α · ρ² · ‖g_t‖²₂ + (1/2) · α² · L · ‖g_t‖²₂   (4)

Where w_t ∈ ℝᵈ is the parameter vector in iteration t and g_t = ∇ℒ(w_t) is the gradient of the loss function at step t.

#### Algorithm 1  Lotus Algorithm

```
Input: Full-rank gradient G_F ∈ ℝ^{m×n};
       avg. unit gradient displacement threshold γ;
       verifying gap η
Initialize: Project count T ← 0

if Initialization or Subspace Switch then
    O_G      ← EfficientLowRankProject(G_F)
    G_init   ← O_G · G_F
    d_init   ← Normalize(G_init)
    T ← 1
end if

G_cur ← O_G · G_F
d_cur ← Normalize(G_cur)
T ← T + 1

if T mod η = 0 then
    Δd       ← d_cur − d_init
    ‖d̄‖      ← ‖Δd‖ / T          # avg. displacement
    if ‖d̄‖ < γ then
        Trigger Subspace Update
    end if
end if
```

**Theorem 3.2 (faster convergence with adaptive policy).** Let N_fix and N_ada denote the number of iterations required by the fixed and adaptive step size policies, respectively, to achieve the gradient tolerance condition Σ_{t=0..N-1} G_t ≤ ε, where G_t = ‖g_t‖²₂ and the step size constraint α < 2·ρ_fix² / L. Then the following inequality holds:

    N_ada ≤ (c_fix / c_ada) · (k / T) · N_fix < N_fix               (5)

This result demonstrates that Lotus's adaptive subspace switching achieves the same convergence criterion with strictly fewer iterations compared to the fixed policy, highlighting its efficiency.

### 3.2 Lotus Algorithm

In this section, we introduce Lotus, a training strategy designed to simultaneously accelerate computation and reduce memory usage. Lotus uses a power-iteration-based randomized SVD to markedly accelerate the gradient projection step. In addition, it incorporates a novel, more flexible path‑efficient switching policy: we define the path efficiency ρ_t of the accumulated gradient displacement, and whenever ρ_t drops below a preset threshold while the time since the previous switch exceeds T_min, the algorithm triggers a subspace recomputation. The details are in Algorithm 1. This mechanism guarantees that, compared to fixed-interval subspace switching, the adaptive strategy reaches the same gradient threshold in fewer iterations, thereby achieving faster convergence. The verifying gap η should be set within 25-100 steps and the threshold γ should be set within the 0.005-0.02 range to avoid too frequent or few updates.
