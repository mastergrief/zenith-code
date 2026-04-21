# QJL — Index

**Paper**: QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero Overhead
**Authors**: Amir Zandieh (Independent), Majid Daliri (NYU), Insu Han (Adobe Research)
**Date**: July 19, 2024
**Code**: https://github.com/amirzandieh/QJL

## Abstract

Serving LLMs requires substantial memory due to the storage requirements of Key-Value (KV) embeddings in the KV cache, which grows with sequence length. An effective approach to compress KV cache is quantization. However, traditional quantization methods face significant memory overhead due to the need to store quantization constants (at least a zero point and a scale) in full precision per data block. Depending on the block size, this overhead can add 1 or 2 bits per quantized number. We introduce QJL, a new quantization approach that consists of a Johnson-Lindenstrauss (JL) transform followed by sign-bit quantization. In contrast to existing methods, QJL eliminates memory overheads by removing the need for storing quantization constants. We propose an asymmetric estimator for the inner product of two vectors and demonstrate that applying QJL to one vector and a standard JL transform without quantization to the other provides an unbiased estimator with minimal distortion. We have developed an efficient implementation of the QJL sketch and its corresponding inner product estimator, incorporating a lightweight CUDA kernel for optimized computation. When applied across various LLMs and NLP tasks to quantize the KV cache to only 3 bits, QJL demonstrates a more than fivefold reduction in KV cache memory usage without compromising accuracy, all while achieving faster runtime.

## Contents

### [Part 1 — Introduction and Preliminaries](01_Introduction_and_Preliminaries.md)
Source §1 + §2. §1 **Introduction**: why KV-cache quantization matters for LLM serving, why traditional block-wise quantization incurs 1–2 bits/number overhead (zero-point + scale in fp16), positioning vs prior KV-cache quantization work. Includes the "KV Cache Quantization" subsection framing the problem. §2 **Preliminaries: Token Generation in Attention**: notation, attention mechanism, KV caching during autoregressive generation, and the memory cost that motivates quantization.

### [Part 2 — Quantized JL Transform](02_QJL_Transform.md)
Source §3. The core contribution. Definition 3.1 (QJL map + asymmetric inner-product estimator ProdQJL: QJL on one vector + full JL on the other). Lemma 3.2 proves ProdQJL is **unbiased**. Lemma 3.5 bounds its **distortion** (variance). Algorithm 1 (QJL Key Cache Quantizer) describes the end-to-end quantization procedure. Theorem 3.6 (the main distortion bound) applies Lemma 3.5 + union bound across the key sequence to give the per-query attention-score error guarantee at bit-width = 1 per coordinate.

### [Part 3 — Experiments and References](03_Experiments_and_References.md)
Source §4 + References. §4 **Experiments**: magnitude analysis of Llama-2 key-cache entries across layers (Figure 2 — motivates asymmetric quantization), long-context QA F1 scores (Table 1), regular-length benchmark accuracy (Table 2), and wall-clock runtime comparisons for encoding + generation (Figure 3). Demonstrates >5× KV-cache compression at 3-bit with no accuracy loss and faster runtime via the custom CUDA kernel. References section lists all cited works.

## Key idea in one line

QJL = `sign(S·x)` with `S` a standard Gaussian JL matrix → **1-bit per coordinate, zero per-block overhead** (no zero-point, no scale), **unbiased inner-product estimator** when paired asymmetrically with un-quantized JL on the query side.

## Relationship to TurboQuant

QJL is the 1-bit residual quantizer used as the second stage in TurboQuant's inner-product-optimal algorithm (see `../01_Introduction_and_Preliminaries.md` §2.2 and `../02_TurboQuant_Algorithms.md` §3.2). TurboQuant applies Q_mse at `b−1` bits and then QJL on the residual to get an unbiased inner-product estimate.
