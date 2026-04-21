# POET-X — Experiments, Related Work & Conclusion

← back to [POET-X.md](POET-X.md)

4Experiments and Results
Our experimental validation consists of (1) single-layer profiling to demonstrate the improvements over POET, (2) performance of POET-X on large-scale LLM pretraining, and (3) ablation studies to benchmark the runtime and memory usage by scaling the compute. Experimental settings and additional results are provided in the Appendices.

4.1Single-layer Benchmarking against POET
The original POET (Qiu et al., 2025a) cannot easily scale beyond 3B parameters, since both memory usage and training time become prohibitive. We quantitatively evaluate a single-layer forward and backward pass, showing improvements in both memory footprint and compute time.

Refer to caption
Figure 3:Latency breakdown of POET, POET-X, and PyTorch Linear Layers with sequence length 2048 and block size 
b
=
256
.
Refer to caption
Figure 4:Memory breakdown for training Llama-8B on a single GPU across different methods with batch size 1, sequence lengths 1024 and block size 
b
=
256
. Since POET (Qiu et al., 2025a) runs OOM under this setting, we estimate its memory footprint by profiling memory usage across different numbers of decoder layers (i.e., parameter sizes) and applying scaling.
We first compare the latency of a single forward and backward pass for POET-X, POET, and a standard linear layer. We set the layer’s input/output dimensions and sequence length to match typical settings in Llama-8B. Figure 3 reports the per-operation time breakdown. Compared with POET, the total forward and backward time drops from 10.59ms to 1.38ms for 
POET-X
fast
 and 1.89ms for 
POET-X
mem
. Relative to a highly optimized PyTorch linear layer (cuBLAS), POET-X incurs only modest overhead. Due to higher parameter efficiency, 
POET-X
fast
 achieves a backward-pass latency comparable to linear layers.

We profiled the memory consumption for training a Llama-8B model on a single GPU. To ensure a fair comparison, POET, 
POET-X
fast
, and 
POET-X
mem
 were configured with the exact same number of trainable parameters. A detailed breakdown of memory usage is presented in Figure 4. The visualization highlights distinct memory characteristics for each method. Compared to AdamW, POET should exhibit PEFT-like characteristics by significantly reducing memory consumption for gradients and optimizer states. However, as visualized in the diagram, POET is even worse in terms of memory than AdamW, because its original formulation requires a substantial amount of memory for activations, such as storing the transformed weight matrix (
𝑾
R
​
P
) for backpropagation. Consequently, this leads to an overall memory efficiency lower than that of AdamW. In contrast, 
POET-X
fast
 and 
POET-X
mem
 both exhibit a memory footprint typical of PEFT methods. This characteristics significantly enhance the scalability of the POET-X framework in large-scale pretraining of transformer models.

Refer to caption
Refer to caption
Figure 5:Validation perplexity dynamics with respect to GPU hours for training Llama-8B with 
𝑳
max
=
256
 (5B tokens) and 
𝑳
max
=
1024
 (10B tokens), respectively.
Method	Params (M)	Mem (G)	Val PPL
AdamW	2764.47	81.03	12.69
Muon (Kimi) (Liu et al., 2025a) 	2764.47	70.94	11.45
APOLLO (Zhu et al., 2025) 	2764.47	80.60	12.97
GaLore (Zhao et al., 2024) 	2764.47	74.50	14.88
POET-X
b
=
256
366.64	60.58	12.76
POET-X
b
=
512
570.06	68.52	12.05
Table 6:Validation Perplexity (PPL) comparison. The column labeled Params (M) reports the total number of trainable parameters for each optimizer, where 
b
 is the block size and 
𝑳
max
=
256
.
Method	Params (M)	Mem (G)	Val PPL
Quantized 8-bit APOLLO	2764.47	66.37	20.49
Quantized 8-bit GaLore	2764.47	66.28	17.74
POET-XQ
b
=
256
366.64	51.66	16.21
POET-XQ
b
=
512
570.06	60.65	14.78
Table 7:Validation Perplexity (PPL) comparison. The column labeled Params (M) reports the total number of trainable parameters for each optimizer, where 
b
 is the block size and 
𝑳
max
=
256
.
Method	Sequence Length 512	Sequence Length 1024
1 
×
 1 H100 	1 
×
 8 H100	1 
×
 1 H100	1 
×
 8 H100
8B	13B	8B	13B	8B	13B	8B	13B
8-bit Q-APOLLO	2.52	1.55	18.63	11.47	4.24	2.63	31.85	19.72
8-bit Q-GaLore	2.25	1.32	18.66	11.49	3.86	2.30	31.81	19.70
POET-XQ
b
=
256
2.77	1.82	21.96	14.22	4.12	2.72	32.41	21.67
POET-XQ
b
=
512
2.30	1.54	17.90	11.93	3.56	2.37	28.27	18.75
Table 8:Throughput (k tokens/s) with Llama-8B and Llama-13B.
Method	Llama-3B	Llama-8B	Llama-13B
𝑳
max
=
512
𝑳
max
=
1024
𝑳
max
=
2048
𝑳
max
=
512
𝑳
max
=
1024
𝑳
max
=
2048
𝑳
max
=
512
𝑳
max
=
1024
𝑳
max
=
2048
AdamW	28.74	28.78	31.43	78.89	76.34	78.69	OOM	OOM	OOM
Muon	18.62	19.27	21.82	50.30	53.46	54.98	76.32	77.02	OOM
APOLLO	19.74	20.40	21.32	51.34	53.49	56.94	OOM	OOM	OOM
GaLore	19.06	20.17	21.16	44.52	45.62	54.71	67.15	67.86	73.37
LoRA
r
=
160
13.60	16.79	22.68	27.90	33.63	43.78	42.48	49.78	63.50
LoRA
r
=
320
16.81	19.50	25.84	33.77	38.50	49.82	50.08	57.53	71.55
POET
b
=
256
33.45	34.79	38.02	OOM	OOM	OOM	OOM	OOM	OOM
POET
b
=
512
37.11	38.09	41.34	OOM	OOM	OOM	OOM	OOM	OOM
POET-X
fast
,
b
=
256
13.41	16.31	21.49	28.65	33.14	43.08	42.77	48.95	61.87
POET-X
fast
,
b
=
512
17.77	20.75	26.06	36.08	41.94	50.81	53.11	59.12	71.88
POET-X
mem
,
b
=
256
11.96	13.28	15.47	25.94	27.87	31.74	35.65	41.62	47.21
POET-X
mem
,
b
=
512
16.31	18.33	20.48	32.93	35.94	40.44	49.63	53.26	59.02
8-bit Q-GaLoRE	13.76	14.49	15.78	34.25	34.89	39.29	53.79	53.38	54.82
8-bit Q-APOLLO	14.02	14.82	16.42	33.14	34.29	38.25	51.04	51.97	55.76
POET-XQ
b
=
256
10.77	12.23	16.15	20.62	23.59	29.83	30.07	34.58	43.59
POET-XQ
b
=
512
14.48	16.71	20.48	27.07	30.26	36.91	39.12	43.98	53.17
Table 9:Peak memory (GB) for different models and sequence lengths (
𝑳
max
) on a single H100 (batch size 1, no gradient accumulation).
4.2Mulit-node LLM Pretraining
We evaluate POET-X by pretraining a Llama-3B transformer on the C4 dataset (Raffel et al., 2020), a widely-used, large-scale corpus derived from Common Crawl. We benchmark our method against AdamW, the prevailing optimizer for pretraining, and Muon (Liu et al., 2025a), a recent optimizer that utilizes graident orthogonalization to maximize training efficiency. We also compare POET against established memory-efficient pretraining methods, including GaLore (Zhao et al., 2024) and APOLLO (Zhu et al., 2025). Experimental settings, including hyperparameters, follows the settings established in (Qiu et al., 2025a; Jaiswal et al., 2024; Liu et al., 2025a). For POET-X, 
b
 denotes the block size of the block-diagonal orthogonal matrix.

We design our experiments according to the Chinchilla scaling law (Hoffmann et al., 2022), which suggests an approximate ratio of 20 training tokens per model parameter offers an ideal trade-off between model performance and compute budget. Accordingly, we train a Llama-3B model with a maximum sequence length 
L
max
 of 
256
 on 60B tokens to compare different training methods. In Table 6, we observe that POET-X achieves superior validation perplexity (PPL) compared to AdamW and other memory-efficient methods, while slightly underperforming Muon (with much lower GPU memory). Specifically, 
POET-X
b
=
512
 yields the second-best validation perplexity of 
12.05
, offering competitive performance with significantly lower memory.

Practical training efficiency. Table 6 shows that POET-X achieves superior iteration-wise convergence, which is consistent with findings on smaller-scale models (Qiu et al., 2025a). To evaluate practical efficiency beyond iteration, we evaluate the wall-clock time convergence in a controlled distributed setting. Experiments were conducted on a multi-node cluster comprising 32 Nvidia H100 GPUs (4 Nodes 
×
 8 GPUs) connected via InfiniBand. Figure 5 demonstrates that POET-X achieves better wall-clock efficiency than AdamW. The observed efficiency gains in the distributed training stem from the superior memory efficiency of POET-X, which permits the use of the Distributed Data Parallel (DDP), as the entire model, along with all the gradients and optimizer states can fit onto every single GPU, and only data is sharded, which in general provides higher throughput, stronger scalability, and more robustness. However, in the given setting, training AdamW with DDP will lead to OOM. Thus, we choose to use Fully Sharded Data Parallel (FSDP, with 8 shards and 4 replicates) for training AdamW. FSDP shards the model parameters, gradients, and optimizer states across GPUs, introducing significant collective communication overheads.

Refer to caption
Figure 6:Throughput (k tokens/s) across different numbers of GPUs. The solid line denotes the actual throughput, and the dashed line denotes the ideal linear scaling throughput. The ideal throughput of 
k
 GPUs is defined as 
T
k
,
i
​
d
​
e
​
a
​
l
=
T
8
,
r
​
e
​
a
​
l
×
k
/
8
).
Seq. Length 512 (Llama-8B)	Seq. Length 2048 (Llama-8B)	Seq. Length 512 (Llama-13B)	Seq. Length 2048 (Llama-13B)
1x1 H100	8x8 H100	Ratio	1x1 H100	8x8 H100	Ratio	1x1 H100	8x8 H100	Ratio	1x1 H100	8x8 H100	Ratio
AdamW	7.60	186.16	24.5	OOM	437.21	N/A	OOM	105.98	N/A	OOM	273.02	N/A
LoRA
r
=
160
5.28	310.59	58.8	6.42	396.97	61.8	3.68	217.52	59.1	4.42	OOM	N/A
LoRA
r
=
320
4.03	228.80	56.8	4.82	298.41	61.9	2.81	166.78	59.3	3.26	OOM	N/A
POET-X
mem
,
b
=
256
3.73	199.95	53.7	5.92	362.75	61.2	2.74	136.14	49.7	4.02	246.47	61.3
POET-X
mem
,
b
=
512
3.16	166.92	52.9	5.26	315.18	59.9	2.22	112.10	50.5	3.58	228.07	63.7
POET-X
fast
,
b
=
256
5.36	286.09	53.4	8.08	489.98	60.6	3.75	206.11	55.0	5.52	341.46	61.8
POET-X
fast
,
b
=
256
3.84	219.30	57.2	6.96	402.88	57.9	2.69	161.33	60.0	4.65	299.46	64.3
Table 10:Throughput (k tokens/s) comparison between baselines and POET-X. Ratio: throughput ratio between 1x1 H100 and 8x8 H100.
Experiments on POET-XQ. One of the major advantages of POET-X is its effortless application to quantized models. We conducted pretraining experiments on Llama-3B models trained on 10B tokens; the final validation perplexity is summarized in Table 7. We observe that POET-XQ outperforms both the GaLore and APOLLO baselines, with 
POET-XQ
b
=
512
 achieving the overall best performance of 
14.78
. Notably, POET-XQ achieves these results while maintaining a lower memory footprint. In Table 9, we highlight that 
POET-XQ
b
=
256
 requires the overall least training memory. Comparing the computational efficiency, POET-XQ also demonstrates higher throughput (Table 8). This efficiency gain stems largely from the fact that POET-XQ does not optimize low-precision weight matrices directly; this allows it to utilize many standard optimizers, making it easier for POET-X to be used in any quantized base models.

4.3In-depth Efficiency Study
Memory efficiency. We investigate whether POET-X’s memory efficiency holds when scaling key aspects of pretraining. We benchmark the peak GPU memory during training. This study systematically varies three factors: model size (3B, 8B, and 13B), input sequence length (512, 1024, and 2048), and POET-X’s block size 
b
 (256 and 512). Memory is measured on a single Nvidia H100 GPU with a batch size of 1. The experiments are performed under two precision settings: a standard setting where all tensors are stored in BF16, and a quantized setting (Zhang et al., 2024; Zhu et al., 2025), where attention and MLP weights are stored in INT8 while gradients and remaining parameters are kept in BF16. We compare POET-X against AdamW, Muon, GaLore, APOLLO, and LoRA. Although LoRA is suboptimal for pretraining (Lialin et al., 2023), it serves as a stringent baseline for memory usage; we therefore match LoRA’s trainable parameters to those of POET-X for a fair comparison. Table 9 shows that the original POET is prohibitively memory-intensive and fails to fit 8B and 13B models even at the smallest sequence length. 
POET-X
fast
 matches the memory efficiency of LoRA, and 
POET-X
mem
 outperforms all baselines across all scales. This memory advantage is most pronounced at scale. With the 13B model and a 2048 sequence length, 
POET-X
mem
,
b
=
256
 and 
POET-X
mem
,
b
=
512
 consume only 47.21G and 59.02G of memory, respectively.

Throughput efficiency. To study POET-X’s throughput scalability, we measure the throughput (k tokens/s) across diverse settings, varying the model size, input sequence length, POET-X block size, and the number of nodes. We scale the experiments from 1 to 8 nodes (i.e., from 1 to 64 GPUs). For distributed training, POET-X and LoRA adopt the DDP strategy, and AdamW employs an 8-shard, 
N
-replicate hybrid FSDP strategy, where 
N
 is the number of nodes. The throughput for single- and 64-GPU runs are presented in Table 10, and the node scaling experiments are shown in Figure 6. In Figure 6, we also compare the empirical throughput (solid line) against the theoretical, ideal linear scaling (dashed line). The ideal scaling of throughput with 
k
 GPUs is defined as 
T
k
,
i
​
d
​
e
​
a
​
l
=
T
8
,
r
​
e
​
a
​
l
×
k
/
8
.

As shown in Table 10 and Table 14 (Appendix), AdamW achieves better single-GPU performance when training Llama-8B with shorter sequence lengths (512 and 1024). However, when scaling either the sequence length or the model size, AdamW encounters OOM errors. Figure 6 also shows that while AdamW’s initial training throughput scales well, it soon deviates from the ideal linear scaling curve as the number of nodes increases. This bottleneck stems from the full-gradient all-reduce required across all nodes at every step, which leads to severe network congestion. FSDP’s intra-node all-gather and reduce-scatter operations further reduce throughput. In contrast, POET-X scales well with different model sizes and sequence lengths by minimizing communication overhead with only minimal collective operations.

5Related Work and Concluding Remarks
