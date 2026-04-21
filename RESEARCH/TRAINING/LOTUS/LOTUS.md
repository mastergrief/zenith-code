# Lotus: Efficient LLM Training by Randomized Low-Rank Gradient Projection with Adaptive Subspace Switching

*Index Terms — Efficient Training, Pre-training, Fine-tuning, Large Language Model*

## Abstract

Training efficiency in large-scale models is typically assessed through memory consumption, training time, and model performance. Current methods often exhibit trade-offs among these metrics, as optimizing one generally degrades at least one of the others. Addressing this trade-off remains a central challenge in algorithm design. While GaLore enables memory-efficient training by updating gradients in a low-rank subspace, it incurs a comparable extra training time cost due to the Singular Value Decomposition (SVD) process on gradients. In this paper, we propose Lotus, a method that resolves this trade-off by simply modifying the projection process. We propose a criterion that quantifies the displacement of the unit gradient to enable efficient transitions between low-rank gradient subspaces. Experimental results indicate that Lotus is the most efficient method, achieving a 30% reduction in training time and a 40% decrease in memory consumption for gradient and optimizer states. Additionally, it outperforms the baseline method in both pre-training and fine-tuning tasks.

## TL;DR

- **Two tricks on top of GaLore:** randomized SVD (rSVD) for the projection, plus Adaptive Subspace Switching (AdaSS) that switches subspaces when the path-efficiency ratio ρ_t drops below γ instead of on a fixed schedule.
- **Gains:** ~30% faster training, ~40% less gradient + optimizer memory; beats GaLore, LoRA, Apollo, AdaRankGrad on LLaMA C4 pre-training and on GLUE RoBERTa fine-tuning.
- **Hyperparams:** γ ∈ [0.005, 0.02] (default 0.01), verifying gap η ∈ [25, 100] (default 50).
- **Ablation:** rSVD alone ≈ GaLore; AdaSS provides nearly all the accuracy gain.

## Sections

- [01_background.md](01_background.md) — Introduction + Related Works (low-rank in weights / optimizer / gradient).
- [02_method.md](02_method.md) — Methodology: path-efficiency ratio ρ_t, Algorithm 1, Lemma 3.1 (one-step projected decrease), Theorem 3.2 (faster adaptive convergence), Lotus training strategy.
- [03_experiments.md](03_experiments.md) — Experiments: LLaMA C4 pre-training (Table 1), GLUE fine-tuning (Table 2), timing comparison (Fig. 2), switching-frequency (Table 3), rSVD/AdaSS ablation (Table 4), Conclusion.
