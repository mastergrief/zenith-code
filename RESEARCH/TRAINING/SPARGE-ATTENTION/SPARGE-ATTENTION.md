# SpargeAttention: Accurate and Training-free Sparse Attention Accelerating Any Model Inference

*Jintao Zhang, Chendong Xiang, Haofeng Huang, Jia Wei, Haocheng Xi, Jun Zhu, Jianfei Chen — ICML*
*Code: [github.com/thu-ml/SpargeAttn](https://github.com/thu-ml/SpargeAttn)*

## Abstract

An efficient attention implementation is essential for large models due to its quadratic time complexity. Fortunately, attention commonly exhibits sparsity, i.e., many values in the attention map are near zero, allowing for the omission of corresponding computations. Many studies have utilized the sparse pattern to accelerate attention. However, most existing works focus on optimizing attention within specific models by exploiting certain sparse patterns of the attention map. A universal sparse attention that guarantees both the speedup and end-to-end performance of diverse models remains elusive. In this paper, we propose **SpargeAttn**, a universal sparse and quantized attention for any model. Our method uses a two-stage online filter: in the first stage, we rapidly and accurately predict the attention map, enabling the skip of some matrix multiplications in attention. In the second stage, we design an online softmax-aware filter that incurs no extra overhead and further skips some matrix multiplications. Experiments show that our method significantly accelerates diverse models, including language, image, and video generation, without sacrificing end-to-end metrics.

> All experiments using SpargeAttn are based on SageAttention. An updated implementation based on SageAttention2 is available at the repo and offers an additional ~30% speedup over the attention in this paper.

## TL;DR

- **What:** training-free sparse + quantized attention with a two-stage online filter.
  - **Stage 1 (mask prediction):** Hilbert-curve permutation → block-level token compression → predicted attention map decides which `Q·Kᵀ` blocks to skip entirely.
  - **Stage 2 (softmax-aware warp filter):** during online softmax, detect blocks whose contribution is negligible post-`exp` and skip the corresponding `P·V` matmuls — **zero extra overhead**, fused with SageAttention kernel.
- **Gains:** 2.5×–5× faster than dense / existing sparse baselines; end-to-end quality preserved across language, T2I, and T2V models (Llama, CogVideoX, Flux, etc.) where prior sparse methods degrade.
- **Universal:** per-layer hyperparam auto-tuning (§3.6); plug-in replacement for FlashAttention/SageAttention in any model without retraining.

## Sections

- [01_background.md](01_background.md) — §1 Introduction + §2 Related Work (sparse attention patterns, training-free vs trained, quantized attention).
- [02_method.md](02_method.md) — §3 SpargeAttn: §3.1 Sparse FlashAttention, §3.2 Selective token compression for sparse prediction, §3.3 Masking of the first stage, §3.4 Sparse warp online softmax, §3.5 Combination with SageAttention, §3.6 Per-layer hyper-parameter determination, §3.7 Hilbert-curve permutation.
- [03_experiments.md](03_experiments.md) — §4 Experiment: §4.1 Setup, §4.2 Quality and efficiency evaluation, §4.3 Ablation study and key insights; §5 Conclusion + CogVideoX sparsity figures.
