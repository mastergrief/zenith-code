# TurboQuant — Index

**Paper**: TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate
**Authors**: Amir Zandieh (Google Research), Majid Daliri (NYU), Majid Hadian (Google DeepMind), Vahab Mirrokni (Google Research)

## Abstract

Vector quantization, a problem rooted in Shannon's source coding theory, aims to quantize high-dimensional Euclidean vectors while minimizing distortion in their geometric structure. We propose TurboQuant to address both mean-squared error (MSE) and inner product distortion, overcoming limitations of existing methods that fail to achieve optimal distortion rates. Our data-oblivious algorithms, suitable for online applications, achieve near-optimal distortion rates (within a small constant factor) across all bit-widths and dimensions. TurboQuant achieves this by randomly rotating input vectors, inducing a concentrated Beta distribution on coordinates, and leveraging the near-independence property of distinct coordinates in high dimensions to simply apply optimal scalar quantizers per each coordinate. Recognizing that MSE-optimal quantizers introduce bias in inner product estimation, we propose a two-stage approach: applying an MSE quantizer followed by a 1-bit Quantized JL (QJL) transform on the residual, resulting in an unbiased inner product quantizer. We also provide a formal proof of the information-theoretic lower bounds on best achievable distortion rate by any vector quantizer, demonstrating that TurboQuant closely matches these bounds, differing only by a small constant (≈ 2.7) factor. Experimental results validate our theoretical findings, showing that for KV cache quantization, we achieve absolute quality neutrality with 3.5 bits per channel and marginal quality degradation with 2.5 bits per channel. Furthermore, in nearest neighbor search tasks, our method outperforms existing product quantization techniques in recall while reducing indexing time to virtually zero.

## Contents

### [Part 1 — Introduction and Preliminaries](01_Introduction_and_Preliminaries.md)
Source §1 + §2. Problem setup (MSE and inner-product distortion objectives, unbiasedness requirement), related work (online vs offline quantization, KV cache compression, product quantization), overview of contributions with headline distortion bounds, and the mathematical preliminaries: Lemma 1 (Beta-distributed coordinates of rotated vectors), Shannon Lower Bound (Lemmas 2–3), and the QJL 1-bit inner-product quantizer (Definition 1, Lemma 4).

### [Part 2 — TurboQuant Algorithms](02_TurboQuant_Algorithms.md)
Source §3.1 + §3.2. The two core algorithms. §3.1 **MSE-Optimal TurboQuant**: random rotation → per-coordinate Lloyd-Max scalar quantizer with precomputed Beta-optimized codebooks, proved to achieve distortion ≤ (3π/2) · 4⁻ᵇ (Algorithm 1, Theorem 1 + proof). §3.2 **Inner-Product-Optimal TurboQuant**: two-stage cascade applying Q_mse at `b−1` bits then QJL on the residual, proved unbiased with distortion ≤ (3π/2) · ‖y‖²/d · 4⁻ᵇ (Algorithm 2, Theorem 2 + proof).

### [Part 3 — Lower Bounds and Experiments](03_Lower_Bounds_and_Experiments.md)
Source §3.3 + §4. §3.3 **Lower Bounds**: Shannon LB + Yao's minimax principle prove `D_mse ≥ 4⁻ᵇ` and `D_prod ≥ ‖y‖²/d · 4⁻ᵇ` for any randomized quantizer (Theorem 3 + proof), establishing TurboQuant's 2.7× optimality gap. §4 **Experiments**: empirical validation of distortion bounds, Needle-In-A-Haystack on Llama-3.1-8B-Instruct at 25% KV budget (perfect recall; beats PolarQuant / SnapKV / PyramidKV / KIVI), LongBench-V1 end-to-end generation (Table 1), and high-dimensional nearest-neighbor search beating PQ and RabitQ on recall@k with zero indexing time (Table 2).

## Key results at a glance

| Bound (unit-norm `x`) | b=1 | b=2 | b=3 | b=4 | General `b` |
|---|---:|---:|---:|---:|---|
| `D_mse(Q_mse)` upper | 0.36 | 0.117 | 0.03 | 0.009 | (3π/2) · 4⁻ᵇ |
| `D_prod(Q_prod)` upper (× d / ‖y‖²) | 1.57 | 0.56 | 0.18 | 0.047 | (3π/2) · 4⁻ᵇ |
| `D_mse` lower (information-theoretic) | — | — | — | — | 4⁻ᵇ |
| `D_prod` lower (× d / ‖y‖²) | — | — | — | — | 4⁻ᵇ |

Gap to optimal: ≤ 2.7× at all bit-widths; ≈ 1.45× at `b=1`.
