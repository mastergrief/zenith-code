# Litespark Technical Report: High-Throughput, Energy-Efficient LLM Training Framework

*Nii Osae Osae Dade, Moinul Hossain Rahat — Mindbeam AI (research@mindbeam.ai)*

## Abstract

Training Large Language Models (LLMs) is plagued by long training times and massive energy consumption, with modern models requiring months of computation and gigawatt-hours of electricity. In light of these challenges, we introduce **Litespark**, a novel pre-training framework that addresses these inefficiencies through targeted optimizations to transformer attention and MLP layers. Our approach combines architectural improvements with algorithmic enhancements to maximize Model FLOPs Utilization (MFU) while maintaining compatibility with standard transformer implementations. Comprehensive benchmarking on 3B and 30B parameter Llama models using the SlimPajama-627B dataset demonstrates substantial performance gains: **2×–6× training throughput improvement** and **55%–83% energy consumption reduction** across multi-node H200 GPU clusters. These optimizations are model- and hardware-agnostic, enabling broad applicability across transformer architectures and extending to post-training phases including supervised fine-tuning and direct preference optimization.

## TL;DR

- **What:** targeted kernel-level optimizations to the attention and MLP blocks in the standard transformer (Llama) stack — architectural + algorithmic — that stack on top of FlashAttention, quantization, pruning etc.
- **Gains (H200 clusters, SlimPajama-627B):**
  - 3B model: 2.00×–3.81× throughput, 55%–70% energy reduction
  - 30B model: 4.73×–6.36× throughput, 75%–83% energy reduction
  - MFU 3–8% → 17–40% at large scale; 44.7% → 89.35% MFU at 8 GPUs
- **Portable:** model- and hardware-agnostic; applies to SFT/DPO post-training, multimodal foundation models, and inference.

## Sections

- [01_setup.md](01_setup.md) — §1 Introduction + §2 Experimental Setup (H200 hardware, SlimPajama, Llama 3B/30B configs, AdamW/ZeRO-1, cosine LR).
- [02_results.md](02_results.md) — §3 Results: Table 3 (3B throughput), Table 4 (30B throughput), MFU analysis, Tables 5–6 (energy + CO₂).
- [03_conclusion.md](03_conclusion.md) — §4 Future directions (post-training, foundation models, inference) + §5 Conclusion.
