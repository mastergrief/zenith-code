# PolarQuant — Index

**Paper**: PolarQuant: Quantizing KV Caches with Polar Transformation
**Authors**: Insu Han (KAIST), Praneeth Kacham (Google Research), Amin Karbasi (Yale), Vahab Mirrokni (Google Research), Amir Zandieh (Google Research)

## Abstract

Large language models (LLMs) require significant memory to store Key-Value (KV) embeddings in their KV cache, especially when handling long-range contexts. Quantization of these KV embeddings is a common technique to reduce memory consumption. This work introduces PolarQuant, a novel quantization method employing random preconditioning and polar transformation. Our method transforms the KV embeddings into polar coordinates using an efficient recursive algorithm and then quantizes resulting angles. Our key insight is that, after random preconditioning, the angles in the polar representation exhibit a tightly bounded and highly concentrated distribution with an analytically computable form. This nice distribution eliminates the need for explicit normalization, a step required by traditional quantization methods which introduces significant memory overhead because quantization parameters (e.g., zero point and scale) must be stored in full precision per each data block. PolarQuant bypasses this normalization step, enabling substantial memory savings. The long-context evaluation demonstrates that PolarQuant compresses the KV cache by over ×4.2 while achieving the best quality scores compared to the state-of-the-art methods.

## Contents

### [Part 1 — Setup and Polar Transformation](01_Setup_and_Polar_Transformation.md)
Source §1 + §2 + §3.1. §1 **Introduction**: why KV-cache quantization matters, why traditional block-normalization incurs a ~1-bit memory overhead per number, and why random preconditioning eliminates the need to store per-block zero-points/scales. §1.1 **Contributions**: random preconditioning + recursive polar transform + long-context evaluation. §2 **Preliminaries**: notation, efficient token generation/KV caching background, random preconditioning definition (Lemma 1). §3.1 **Recursive Polar Transformation**: the Cartesian→polar conversion algorithm (Definition 1) that runs in place without materializing the full polar vector.

### [Part 2 — Distribution of Polar Angles](02_Distribution_of_Polar_Angles.md)
Source §3.2. The core analytical block: deriving the distribution of polar angles after random preconditioning. Lemma 2 (Distribution of a Gaussian Vector Under Polar Transformation) + its long proof showing that each angle concentrates around a tightly bounded form with analytically computable density. This concentration property is what lets PolarQuant quantize angles with small bit-widths without needing per-block normalization constants.

### [Part 3 — Algorithm and Experiments](03_Algorithm_and_Experiments.md)
Source §3.3 + §4 + §5 + §6. §3.3 **PolarQuant Algorithm and Main Theorem** (Algorithm 1, Theorem 1: asymptotically optimal worst-case error bound). §4 **KV Cache Quantization with PolarQuant** including §4.1 Practical Implementation (batched polar transformation, codebook construction, how angle distributions flatten after preconditioning — Figure 2). §5 **Experiments**: §5.1 random preconditioning on KV cache, §5.2 Needle-In-A-Haystack (Figure 3), §5.3 End-to-end LongBench-V1 generation (Table 1), §5.4 Runtime analysis (Table 2). §6 **Conclusion**.

## Key idea in one line

Preconditioning + polar coordinates → angle distribution concentrates analytically → no per-block normalization constants needed → >4.2× KV-cache compression with best-in-class long-context quality.
