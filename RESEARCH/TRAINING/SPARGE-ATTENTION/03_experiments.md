# SpargeAttention — Experiments & Conclusion

← back to [SPARGE-ATTENTION.md](SPARGE-ATTENTION.md)

4Experiment
4.1Setup
Models. We validate the effectiveness of SpargeAttn across diverse representative models from language, image, and video generation. Specifically, we conduct experiments on Llama3.1 (8B) (llama31model) for text-to-text, CogvideoX (2B), Mochi (genmo2024mochi), and Open-Sora-Plan (lin2024open) for text-to-video, Flux (.1-dev) (flux) and Stable-Diffusion3.5 (large) (stable_diffusion_3_5) for text-to-image.

Datasets. The Text-to-text model is evaluated on four zero-shot tasks: WikiText (merity2022pointer) to assess the model’s prediction confidence, Longbench (bai2023longbench) and En.MC of InfiniteBench (zhang-etal-2024-bench) for a comprehensive assessment of long context understanding capabilities, and the Needle-in-a-Haystack task (LLMTest_NeedleInAHaystack) to assess the model’s retrieval ability. Text-to-video models are evaluated using the open-sora (opensora) prompt sets. Text-to-image models are assessed on COCO annotations (lin2014microsoft).

End-to-end metrics. For Llama3.1, we use perplexity (ppl.) (jelinek1977perplexity) for WikiText, Longbench score (bai2023longbench), and retrival accuracy for the Needle-in-a-Haystack task (LLMTest_NeedleInAHaystack). For text-to-video models, following zhao2024viditq, we evaluate the quality of generated videos on five metrics: CLIPSIM and CLIP-Temp (CLIP-T) (liu2024evalcrafter) to measure the text-video alignment; VQA-a and VQA-t to assess the video aesthetic and technical quality, and Flow-score (FScore) for temporal consistency (wu2023exploring). For text-to-image models, generated images are compared with the images in the COCO dataset in three aspects: FID (heusel2017gans) for fidelity evaluation, Clipscore (CLIP) (hessel2021clipscore) for text-image alignment, and ImageReward (IR) (xu2024imagereward) for human preference.

Speed and sparsity metric. We use inverse latency 
1
/
t
 to evaluate the speed of sparse attention methods. Specifically, 
1
/
t
 = 
O
​
(
a
​
t
​
t
​
n
)
/
t
, where 
O
​
(
a
​
t
​
t
​
n
)
 represents the total number of operations in a standard attention computation, and 
t
 is the latency in seconds from a given 
(
Q
,
K
,
V
)
 to the output of attention. Note that this speed metric is completely fair. This is because the 
O
​
(
a
​
t
​
t
​
n
)
 is fixed for a set of inputs, and then the speed is determined by 
t
, which includes the time spent predicting the sparse region of the attention map. We define Sparsity as the proportion of the Matmul of 
Q
i
​
K
j
⊤
 plus 
P
~
i
​
j
​
V
j
 that are skipped relative to the total number of 
Q
i
​
K
j
⊤
 plus 
P
~
i
​
j
​
V
j
 in a full attention required.

Implementation and Hyper-parameters. We implement our method using CUDA. As discussed in Sec. 3.6, we need to determine 
l
1
,
l
2
 for models. We use (
l
1
=
0.08
,
l
2
=
0.09
) for Llama3.1, (
l
1
=
0.05
,
l
2
=
0.06
) for CogvideoX and Mochi, and (
l
1
=
0.07
,
l
2
=
0.08
) for Stable-Diffusion3.5 and Flux, (
l
1
=
0.03
,
l
2
=
0.035
) for Open-Sora-Plan.

Baselines. Currently, sparse attention methods applicable across different model types are limited. We choose block-sparse MInference (jiang2407minference) and FlexPrefill (FlexPrefill) as our baselines. To vary the sparsity of these baselines, we use 30% and 70% for MInference, and use 
γ
=
0.95
 and 
0.99
 for FlexPrefill according to their paper.

Refer to caption
Figure 6:Visible examples on CogvideoX using SpargeAttention.
Refer to caption
Figure 7:Comparison examples on Flux and Stable-Diffusion3.5. The sparsity of SpargeAttn, MInference and FlexPrefill is 0.38, 0.3, and 0.4 on Flux and 0.31, 0.3, and 0.35 on Stable-Diffusion3.5.
Refer to caption
Figure 8:Comparison examples on Mochi. The sparsity of SpargeAttn, MInference and FlexPrefill is 0.47, 0.3, and 0.4.
Refer to caption
Figure 9:A Needle-in-a-Haystack comparison example on Llama3.1. The sparsity of SpargeAttn, MInference, and FlexPrefill is 0.5, 0.5, and 0.54.
4.2Quality and Efficiency Evaluation
End-to-end metrics. We assess the end-to-end metrics of various models using SpargeAttn compared to using full attention and baselines. Table 1 shows the results. We can observe that our method incurs almost no end-to-end metric loss across various models compared to Full-Attention and surpasses baselines with various sparsity levels in terms of end-to-end accuracy. Fig. 6, 7, 8, and 12 show some visible comparison examples on CogvideoX, Flux, Stable-Diffusion3.5, Mochi, and Open-Sora-Plan, showing that SpargeAttn incurs no performance loss and outperforms baselines.

Refer to caption
Figure 10:Kernel speed comparison under varying sparsity. Input tensors have a sequence length of 22K and a head dimension of 128. SpargeAttn+FA2 means deploying our method on FlashAttention2.
Attention speed. Table 1 shows that our method achieves faster speeds compared to Full-Attention and surpasses baselines with various sparsity levels in terms of attention speed. Fig. 10 illustrates the kernel speeds of various methods across different sparsity, highlighting the efficiency of our approach and its significant advantage over other methods.

4.3Ablation Study and key Insights
Overhead of sparse block prediction. Table 3 compares the overhead of dynamic sparse block prediction in SpargeAttn compared with attention execution latency. The results indicate that the prediction overhead is minimal compared to attention, particularly for longer sequences.

Table 2:End-to-end generation latency using SpargeAttn.
Model	GPU	Original	 
SageAttn
SpargeAttn
CogvideoX	RTX4090	87 s	68 s	53 s
Mochi	L40	1897 s	1544 s	1037 s
Llama3.1 (24K) 	RTX4090	4.01 s	3.53 s	2.6 s
Llama3.1 (128K) 	L40	52 s	42s	29.98 s
End-to-end speedup. Table 2 shows the end-to-end latency on CogvideoX, Mochi, and Llama3.1 using SpargeAttn. Notably, SpargeAttn achieves 1.83x speedup on Mochi.

Table 3:Overhead of sparse block prediction in SpargeAttn.
Sequence Len	Prediction (ms)	Full Attention (ms)	Overhead
8k	0.251	6.649	3.78%
16k	0.487	26.83	1.82%
32k	0.972	106.68	0.911%
64k	2.599	424.24	0.612%
128k	8.764	1696.2	0.516%
Effect of Hilbert Curve permutation. We evaluate the impact of Hilbert Curve permutation on Mochi by comparing three metrics: average block similarity across blocks of query or key, L1 error defined in Sec. 3.6, and sparsity. Table 4 shows that the HilbertCurve permutation consistently achieves superior block self-similarity and sparsity, with only a marginal difference in accuracy. Please see Appendix A.1 for more analysis and details.

Table 4:Effect of permutation on sparsity and accuracy. Sim-q and Sim-k are the average block self-similarity of the query and key.
Method	Sim-q 
↑
Sim-k 
↑
L1 
↓
Sparsity 
↑
Random	0.321	0.019	0.0414	0.048
Rowmajor	0.551	0.390	0.0307	0.363
Timemajor	0.514	0.367	0.0342	0.338
HilbertCurve	0.572	0.479	0.0389	0.392
Table 5:Abalation of self-similarity judge.
Method	VQA-a 
↑
VQA-t 
↑
FScore 
↑
W/o. self-sim Judge	34.664	44.722	1.138
With self-sim Judge	54.179	67.219	1.807
Table 6:Analysis of sparsity from 
M
g
 and 
M
p
​
v
.
Strategy	only 
M
g
only 
M
p
​
v
M
g
 +
M
p
​
v
Sparsity	51.2%	27.7%	54%
Ablation of self-similarity judge We ablate the effect of the self-similarity judge on Mochi. As shown in Table 5, we find that self-similarity judge can guarantee end-to-end accuracy. Please see Appendix A.2 for more analysis.

Analysis of sparsity from 
M
g
 and 
M
p
​
v
. Table 6 shows the sparsity when only using 
M
g
, only using 
M
p
​
v
, and using 
M
g
+
M
p
​
v
 on Llama3.1 in Needle-in-a-Haystack task with 128K sequence length.

SpargeAttn enhance the LLM performance. From Table 1, Fig. 9 and  11, we observe that SpargeAttn enhances LLM performance in long-context tasks. This improvement may result from the fact that sparse attention helps the LLM focus on more relevant information.

Table 7:Sparsity increases with sequence length under a constant accuracy bound on Llama3.1.
Sequence Len	8K	16K	24K	48K	128K
Sparsity	6.8%	26.4%	35.7%	49.8%	54%
Sparsity increases with sequence length. As shown in Table 7, we find that on Llama3.1, sparsity increases with sequence length. This suggests that the longer contexts, the higher speedup of SpargeAttn can achieve.

Sparsity analysis over diffusion model. We conduct a detailed analysis of sparsity in CogvideoX across all layers, heads, timesteps, and samples using SpargeAttn to get more insights (See Appendix A.4 for detailed figures). We find that sparsity varied with layers and heads, indicating that setting different hyperparameters for each layer and head is necessary. We also find that for diffusion models, the sparsity increases with the sample timesteps.

5Conclusion
In this paper, we propose SpargeAttn, a universal sparse and quantized attention that executes attention efficiently and accurately for any input. Our method uses a two-stage online filter: in the first stage, we rapidly and accurately predict the attention map, enabling the skip of some matrix multiplications in attention. In the second stage, we design an online softmax-aware filter that incurs no extra overhead and further skips some matrix multiplications. Experiments show that SpargeAttn accelerates diverse models, including language, image, and video generation models, without sacrificing end-to-end metrics.

Acknowledgment
This work was supported by the NSFC Projects (Nos. 92270001, 62376131). J.Z is also supported by the XPlorer Prize.

Impact Statement
This paper presents work that aims to advance the field of Machine Learning. There are many potential societal consequences of our work, none of which we feel must be specifically highlighted here.

Appendix AAppendix
A.1Detailed Explain and results of permutation ablation
We use five distinct prompts and pre-searched hyperparameters with 
l
1
=
0.05
,
l
2
=
0.06
 on both CogvideoX and Mochi models. The permutation are performed separately in attention operation for 
Q
,
K
,
V
 after position embedding. To retain the original order of the input sequence, an inverse permutation is performed on the output of attention; for models using visual-language joint self-attention(e.g., CogvideoX), we only permute the visual tokens. When evaluating block self-similarity, we choose a block size of 
128
 for query and 
64
 for key, which aligns with our kernel implementation. The precision metric(L1) is evaluated using FlashAttention2 output as ground truth.

We choose different permutation methods to compare their impact on the performance of attention operations. Given a 3D visual token tensor with shape 
T
×
H
×
W
×
d
, the permutation finally results in a tensor with shape 
L
×
d
, where 
L
=
T
×
H
×
W
. The permutation methods and their detailed descriptions are shown in Table 8.

Table 8:The detailed description of different permutation methods.
Method	Detailed Description
Random	Random permutation of tokens, the order is recorded to perform inverse permutation.
Rowmajor	Permutation following row-major order. Tokens are continuous along the W dimension.
Columnmajor	Permutation following column-major order. Tokens are continuous along the H dimension.
Timemajor	Permutation following time-major order. Tokens are continuous along the T dimension.
HilbertCurve	Permutation following a Hilbert curve.
Detailed results of permutation ablation for the CogvideoX and Mochi models are presented in Table 9. The HilbertCurve permutation consistently achieves superior block self-similarity and sparsity, with only a marginal loss in precision. This suggests that the HilbertCurve permutation effectively enhances block self-similarity and sparsity. It is worth noting that the random permutation retains the precision metrics but sacrifices sparsity. This indicates that our algorithm has the property of dynamically adjusting and robust to complex token sequences.

Table 9:The impact of permutation on CogvideoX and Mochi models. Sim-q is the block self-similarity of the query, and Sim-k is the block self-similarity of the key.
Method	Sim-q
↑
Sim-k
↑
Precision(L1)
↓
Sparsity
↑
CogvideoX	Mochi	CogvideoX	Mochi	CogvideoX	Mochi	CogvideoX	Mochi
Random	0.502	0.321	0.025	0.019	0.0348	0.0414	0.027	0.048
Rowmajor	0.676	0.551	0.435	0.390	0.0265	0.0307	0.242	0.363
Columnmajor	0.633	0.547	0.335	0.394	0.0274	0.0342	0.198	0.366
Timemajor	0.692	0.514	0.479	0.367	0.0294	0.0342	0.238	0.338
HilbertCurve	0.709	0.572	0.523	0.479	0.0323	0.0389	0.265	0.392
A.2Ablation Study of Self-Similarity Judge
To investigate the impact of the self-similarity judge on attention performance, we follow the experimental setting outlined in Sec. A.1 and conduct an ablation study by removing the self-similarity judge. In most cases, the presence of highly localized patterns results in a minimal number of non-self-similar blocks, leading to only minor differences in precision and sparsity when averaging across all tensor cases. To obtain more meaningful and interpretable insights, we specifically analyze cases where the precision difference is statistically significant.

To this end, we apply a threshold-based selection criterion, retaining only those cases where the absolute difference between 
L
​
1
s
​
i
​
m
−
j
​
u
​
d
​
g
​
e
 (precision error with the self-similarity judge) and 
L
​
1
n
​
o
−
j
​
u
​
d
​
g
​
e
 (precision error without the self-similarity judge) exceeds 0.05. This criterion results in approximately 2% of the tensor cases being retained for further analysis. We employ precision (L1 error) and sparsity as evaluation metrics to assess the influence of the self-similarity judge on the attention output. The results are summarized in Table 10.

Table 10:Impact of the self-similarity judge on the accuracy and sparsity of attention.
Method	w/ judge	w/o judge	filter w/ judge	filter w/o judge
CogvideoX	Mochi	CogvideoX	Mochi	CogvideoX	Mochi	CogvideoX	Mochi
L1 error
↓
 	0.0316	0.0343	0.0325	0.0365	0.0843	0.0555	0.214	0.154
Sparsity 
↑
 	0.199	0.301	0.203	0.305	0.242	0.371	0.275	0.392
The findings demonstrate that the self-similarity judge effectively mitigates extreme precision loss while introducing only a marginal reduction in sparsity. Furthermore, we observe that a significant proportion of cases exhibiting notable differences originate from the Random permutation category in the CogvideoX model. This observation further highlights the role of the self-similarity judge in enhancing the model’s robustness to complex token sequences while maintaining high precision.

Refer to caption
Figure 11:A Needle-in-a-Haystack comparison example on Llama3.1. The sparsity of SpargeAttn, MInference, and FlexPrefill is 0.36, 0.3, and 0.3.
Refer to caption
Figure 12:Visible examples on Open-sora-Plan.
Refer to caption
Figure 13:Comparison examples on Mochi. The sparsity of SpargeAttn, MInference, and FlexPrefill is 0.47, 0.3, and 0.4.
Table 11:End-to-end metrics on Llama3.1 in the Needle-in-a-Haystack task with 16-28K sequence lengths.
Model (seq_len)
 	
Attention (Sparsity)
Speed (TOPS)
↑
NIAH 
↑
Llama3.1
(24K)
 	
Full-Attention
156.9	0.838
Minference (0.5)
 	122.5	0.635
FlexPrefill (0.6)
 	179.6	0.776
Minference (0.3)
 	102.3	0.652
FlexPrefill (0.3)
 	117.6	0.797
SpargeAttn (0.36)
 	443.6	0.863
A.3Additional Experiments
In this section, we present additional experimental results further to evaluate the performance of SpargeAttn compared to baselines. Fig. 11 and 11 show the results on Llama3.1 in the Needle-in-a-Haystack task with 16-28K sequence length. Fig 13 shows a visible comparison example on Mochi.

A.4Sparsity analysis over diffusion model
In this section, we analyze the sparsity patterns in CogvideoX across different dimensions: model layers, denoising timesteps, input samples, and attention heads. Figure 14 illustrates the layer-wise sparsity. Figure 15 demonstrates timestep-wise sparsity. Figure 16 highlights sample-wise sparsity. Figure 17 presents head-wise sparsity, illustrating the diversity in attention behavior across different heads. These analyses are helpful for the design of some diffusion algorithms (zheng2023dpm; zheng2024diffusion; zheng2024masked; zheng2025direct; zhao2024identifying; zhao2025riflex; wang2024framebridge).

Refer to caption
Figure 14:Layer-wise sparsity of CogvideoX.
Refer to caption
Figure 15:Timestep-wise sparsity of CogvideoX.
Refer to caption
Figure 16:Sample-wise sparsity of CogvideoX.
