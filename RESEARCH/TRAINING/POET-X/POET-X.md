# POET-X: Memory-efficient LLM Training by Scaling Orthogonal Transformation

*Zeju Qiu, Lixin Liu, Adrian Weller, Han Shi, Weiyang Liu — [spherelab.ai/poetx](https://spherelab.ai/poetx)*

## Abstract

Efficient and stable training of large language models (LLMs) remains a core challenge in modern machine learning systems. To address this challenge, **Reparameterized Orthogonal Equivalence Training (POET)**, a spectrum-preserving framework that optimizes each weight matrix through orthogonal equivalence transformation, has been proposed. Although POET provides strong training stability, its original implementation incurs high memory consumption and computational overhead due to intensive matrix multiplications. To overcome these limitations, we introduce **POET-X**, a scalable and memory-efficient variant that performs orthogonal equivalence transformations with significantly reduced computational cost. POET-X maintains the generalization and stability benefits of POET while achieving substantial improvements in throughput and memory efficiency. In our experiments, POET-X enables the pretraining of billion-parameter LLMs on a single Nvidia H100 GPU, where standard optimizers such as AdamW run out of memory under the same settings.

*Keywords: Memory, Pretraining, Large Language Model*

## TL;DR

- **Motivation:** POET's spectrum-preserving training (orthogonal-equivalence reparam `W = R·W₀·S` with `R,S` orthogonal) is stable and generalizes well, but the dense R/S matmuls blow memory and throughput vs AdamW.
- **Trick stack (POET-X):** input-centric implementation + permutation-accelerated block-diagonal orthogonal factors + batch-parallel block-diag matmul + efficient Cayley-Neumann parameterization + activation checkpointing + quantized variant (POET-XQ).
- **Result:** billion-parameter pretraining on a single H100 (where AdamW OOMs); scales multi-node competitively with GaLore/APOLLO while beating them on validation perplexity, including under quantization.

## Sections

- [01_background.md](01_background.md) — §1 Introduction + §2 Preliminaries of POET (the orthogonal-equivalence reparameterization, Cayley form, merge-and-reinit scheme).
- [02_method.md](02_method.md) — §3 POET-X: §3.1 Input-centric implementation, §3.2 Permutation acceleration & reduction, §3.3 Batch parallel block-diagonal matmul, §3.4 Efficient Cayley-Neumann parameterization, §3.5 Memory-efficient checkpointing, §3.6 POET-XQ (quantized training).
- [03_experiments.md](03_experiments.md) — §4 Experiments: §4.1 Single-layer vs POET, §4.2 Multi-node LLM pretraining (Llama 3B/8B/13B), §4.3 In-depth efficiency study + §5 Related Work and Concluding Remarks.
