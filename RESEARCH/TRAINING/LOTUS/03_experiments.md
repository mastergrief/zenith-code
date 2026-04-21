# Lotus — Experiments & Conclusion

← back to [LOTUS.md](LOTUS.md)

## 4 Experiments

**Fig. 2:** We visualize training time consumption comparison between GaLore, AdaRankGrad, Apollo and Lotus on both pre-training and fine-tuning tasks. (a) shows the ETA of pre-training LLaMA-type 3B models in C4. (b) shows the average time cost on 8 GLUE tasks. Lotus is the most effective method in terms of computational time efficiency. (Less is better.)

**Implementation Details.** We utilize GaLore [27] as our codebase for model training and evaluation. All model architectures involved in the experiment are consistent with GaLore. The data format in training and validation is BF16. We tune the hyper-parameters needed in the experiments to achieve optimal performance.

### 4.1 Pre-Training and Fine-tuning

To evaluate the effectiveness of Lotus, we pre-train LLaMA models of varying sizes on the C4 dataset—a widely used cleaned version of the Common Crawl corpus, following GaLore's experimental settings and using perplexity as the primary metric. The corresponding experimental results are reported in Table 1. All pre-training experiments here use NVIDIA H100 GPUs.

We fine-tune RoBERTa-Base model on 8 GLUE tasks to compare the results with full rank fine-tuning, Lora, GaLore and AdaRankGrad, showing the results in Table 2. We report the overall (matched and mismatched) accuracy for MNLI, Matthew's correlation for CoLA, Pearson correlation for STS-B, F1 score for MRPC, and accuracy for other tasks. Lotus surpasses previous methods on most tasks, achieving higher average scores while reducing memory costs. All fine-tuning experiments use NVIDIA RTX 4090 GPUs.

**Table 1:** Performance of several low-rank training algorithms compared with Lotus by pre-training LLaMA models of varying sizes on the C4 dataset. r denotes the rank of the low-rank factorization; d_model denotes the hidden states dimension of each model size. Values are perplexity (memory).

| Method      | 60M           | 130M          | 350M          | 1B            |
|-------------|---------------|---------------|---------------|---------------|
| Full Rank   | 34.06 (0.36G) | 25.08 (0.76G) | 18.80 (2.06G) | 15.56 (7.80G) |
| GaLore      | 34.88 (0.24G) | 25.36 (0.52G) | 18.95 (1.22G) | 15.64 (4.38G) |
| Low Rank    | 78.18 (0.26G) | 45.51 (0.54G) | 37.41 (1.11G) | 142.53 (3.62G)|
| LoRA        | 34.99 (0.36G) | 33.92 (0.80G) | 25.58 (1.76G) | 19.21 (6.17G) |
| ReLoRA      | 37.04 (0.36G) | 29.37 (0.80G) | 29.08 (1.76G) | 18.33 (6.17G) |
| AdaRankGrad | 34.24 (0.21G) | 25.22 (0.50G) | 18.91 (1.11G) | 14.71 (3.62G) |
| **Lotus**   | **33.75 (0.23G)** | **24.87 (0.51G)** | 18.91 (1.19G) | 15.33 (4.20G) |
| r / d_model | 128 / 256     | 256 / 768     | 256 / 1024    | 512 / 2048    |
| Training Tokens | 1.1B      | 2.2B          | 6.4B          | 13.1B         |

**Table 2:** Evaluating Lotus on the GLUE benchmark. We compare Lotus with various memory-efficient training methods and report the average metrics. Threshold γ = 0.01 and verifying gap η = 50 are used as the baseline throughout fine-tuning tasks.

| Method              | Memory | CoLA  | STS-B | MRPC  | RTE   | SST2  | MNLI  | QNLI  | QQP   | Avg   |
|---------------------|-------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|
| Full Fine-Tuning    | 747M   | 62.24 | 90.92 | 91.30 | 79.42 | 94.57 | 87.18 | 86.28 | 92.28 | 86.28 |
| LoRA (rank=4)       | 257M   | 61.38 | 90.57 | 91.07 | 78.70 | 92.89 | 86.82 | 92.18 | 91.29 | 85.61 |
| GaLore (rank=4)     | 253M   | 60.35 | 90.73 | 92.25 | 79.42 | 94.04 | 87.00 | 92.24 | 91.06 | 85.89 |
| Apollo (rank=4)     | 251M   | 59.75 | 89.89 | 90.74 | 74.00 | 93.11 | 85.62 | 91.70 | 89.38 | 83.87 |
| AdaRankGrad (rank=4)| 202M   | 61.40 | 90.97 | 92.60 | 81.23 | 94.80 | 86.60 | 92.50 | 90.40 | 86.31 |
| **Lotus (rank=4)**  | 251M   | **64.67** | 90.79 | **93.14** | **83.39** | 94.72 | **87.47** | **93.00** | 91.06 | **87.28** |
| LoRA (rank=8)       | 264M   | 61.83 | 90.80 | 91.90 | 79.06 | 93.46 | 86.94 | 92.25 | 91.22 | 85.93 |
| GaLore (rank=8)     | 257M   | 60.06 | 90.82 | 92.01 | 79.78 | 94.38 | 87.17 | 92.20 | 91.11 | 85.94 |
| Apollo (rank=8)     | 251M   | 60.63 | 90.08 | 90.51 | 74.36 | 93.34 | 85.90 | 92.20 | 88.98 | 84.50 |
| AdaRankGrad (rank=8)| 237M   | 62.00 | 90.89 | 93.20 | 81.23 | 94.80 | 86.50 | 92.60 | 89.70 | 86.36 |
| **Lotus (rank=8)**  | 254M   | **63.44** | **91.06** | **93.35** | **81.58** | **94.95** | **87.32** | **93.11** | **91.15** | **86.99** |

### 4.2 Training Time Efficiency and Performance Comparison

We compare the estimated time of arrivals (ETA) for pre-training LLaMA-type 3B models using an 8-bit optimizer with layer-wise weight updates on a single NVIDIA RTX 4090 GPU, following GaLore, and the average time cost on the fine-tuning tasks as shown in Figure 2. Our method demonstrates significant time savings compared to GaLore, AdaRankGrad, and Apollo. Additionally, Lotus shows faster subspace update frequency than GaLore in Table 3. These results indicate that Lotus is the simplest yet most effective method in terms of computational efficiency. We also quantify the contribution of randomized SVD (rSVD) and adaptive subspace switching (AdaSS) as shown in Table 4. This shows that rSVD closely matches the exact-SVD variant at the same rank, and most of the gain comes from our adaptive subspace update.

**Table 3:** Comparison of GaLore and Lotus on update frequency in GLUE benchmark.

| Method          | Subspace Account | Subspace Switching Frequency |
|-----------------|-----------------:|-----------------------------:|
| GaLore (rank=4) | 3536             | 1.6                          |
| Lotus (rank=4)  | 11614            | 6.5 ↑306%                    |
| GaLore (rank=8) | 3544             | 1.6                          |
| Lotus (rank=8)  | 11736            | 6.3 ↑320%                    |

**Table 4:** Component-wise evaluation of SVD, rSVD, and rSVD with AdaSS, illustrating how each component affects results.

| Rank | rSVD | AdaSS | Avg   |
|-----:|:----:|:-----:|------:|
| 4    |      |       | 85.89 |
| 4    | ✓    |       | 85.89 |
| 4    | ✓    | ✓     | 87.28 |
| 8    |      |       | 85.94 |
| 8    | ✓    |       | 86.07 |
| 8    | ✓    | ✓     | 86.99 |

## 5 Conclusion

We introduce Lotus, an adaptive low-rank update algorithm that delivers a 30% training speedup and a 40% reduction in memory, while achieving better convergence than full-rank pretraining, fine-tuning and recent memory-efficient methods. Guided by the hypothesis that alignment between unit-gradient displacement and subspace geometry governs update efficiency, Lotus adopts a physics-inspired view of gradient descent: it tracks the Euclidean distance between low-rank unit gradients and dynamically switches subspaces when directional consistency degrades. We provide theoretical analysis and extensive experiments showing that Lotus effectively balances memory, computation time, and performance for LLM training, offering a practical tool for efficient large-model optimization.
