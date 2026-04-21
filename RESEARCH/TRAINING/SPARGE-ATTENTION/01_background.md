# SpargeAttention — Background

← back to [SPARGE-ATTENTION.md](SPARGE-ATTENTION.md)

1Introduction
As sequence lengths in large models become longer, such as 45K-128K in video generation and language models (yang2024cogvideox; bao2024vidu; llama31model; zhang2025sage), the time consuming of attention occupies a significant portion of inference latency in large models (zhangsurvey; 2024sageattention). Fortunately, the attention map 
P
=
Softmax
​
(
Q
​
K
⊤
/
d
)
 exhibits inherent sparsity, as the softmax operation often creates many values approaching zero (zhangsurvey). Sparse attention exploit such sparsity to accelerate attention by (1) constructing a “sparse mask”, which indicates the important non-zero entries of the attention map 
P
 that should be computed, and (2) computing attention only for the parts corresponding to the sparse mask. There are three distinct categories of sparse attention methods based on how the sparse mask is generated. pattern-based method (zhang2023h2o; xiao2024infllm; moaattention; zhu2024sampleattention; xiao2024duoattention; xiao2023efficient; zhang2025fast) relies on specific sparsity patterns based on empirical observations, dynamic sparse attention (ribar2023sparq; singhania2024loki; jiang2407minference; FlexPrefill; gao2024seerattention; xi2025sparse; yang2025sparse; zhang2025spargeattn_wksp) computes the mask on-the-fly based on the inputs, and training-based method (kitaev2020reformer; pagliardini2023fast; zhang2025faster) directly train models with native sparse attention.

Refer to caption
Figure 1:SpargeAttn can achieve 1.83x speedup on Mochi on L40 GPU, with no video quality loss.
Limitation. (L1. Universality) Though existing sparse attention methods already demonstrate promising speedup on some tasks, their universality is still limited. Existing works are typically developed for specific tasks, such as language modeling, utilizing task-specific patterns such as sliding windows or attention sinks. However, the attention pattern varies significantly across tasks (see examples in Fig. 2), making these patterns hard to generalize. (L2. Usability) Moreover, it is difficult to implement both accurate and efficient sparse attention for any input. This is because accuracy demands precise prediction of the sparse regions in the attention map, while efficiency requires the overhead of this prediction to be minimal. However, current methods are difficult to effectively satisfy both of the requirements simultaneously. For example, MInference (jiang2407minference) requires a large sequence length, such as 100K, to achieve a noticeable speedup.

Goal. We aim to design a training-free sparse attention operator that accelerates all models without metrics loss.

Our approach. In this work, we develop SpargeAttn, a training-free sparse attention that can be adopted universally on various tasks, including language modeling and text-to-image/video, and various sequence lengths. We propose three main techniques to improve the universality, accuracy, and efficiency. First, we propose a universal sparse mask prediction algorithm, which constructs the sparse mask by compressing each block of 
Q
, 
K
 to a single token. Importantly, we compress selectively based on the similarity of tokens within the block, so the algorithm can accurately predict sparse masks universally across tasks. Second, we propose a sparse online softmax algorithm at the GPU warp level, which further omits some 
P
​
V
 products by leveraging the difference between global maximum values and local maximum values in online softmax. Third, we integrate this sparse approach into the 8-bit quantized SageAttention framework for further acceleration. To the best of our knowledge, SpargeAttn is the first sparse attention method that can actually accelerate across language, image, and video models without compromising accuracy.

Result. We evaluate SpargeAttn on a variety of generative tasks, including language modeling and text-to-image/video, with comprehensive performance metrics on the model quality. SpargeAttn can robustly retain model end-to-end performance while existing sparse attention baselines incur degradation. Moreover, SpargeAttn is 2.5x to 5x faster than existing dense and sparse attention models.

Refer to caption
Figure 2:Some sampled patterns of attention map 
P
 in video, image, and language generation models.
2Related Work
Depending on how the sparsity mask is constructed, sparse attention methods can be divided into three types (zhangsurvey): (1) Pattern required methods rely on some fixed patterns of the attention map, such as sliding windows or attention sinks (xiao2023efficient). H2O (zhang2023h2o), InfLLM (xiao2024infllm), and DUOAttention (xiao2024duoattention) rely on sliding window pattern. SampleAttention (zhu2024sampleattention), MOA (moaattention), and StreamingLLM (xiao2023efficient) rely on sliding window and attention sink pattern. DitFastAttn (yuan2024ditfastattn) relies on sliding window patterns and similarities between different attention maps. Moreover, DitFastAttn is restricted to simple diffusion transformers, showing incompatibility with language models and MMDiT models like Flux (flux), Stable Diffusion3 and 3.5 (stable_diffusion_3_5), and CogVideoX (yang2024cogvideox). As the pattern varies across models, these methods may not universally work for different models. (2) Dynamic sparse methods dynamically construct the sparse mask based on the input without the need of preset patterns, and are thus potentially more universal. Existing works can be further categorized into channel compression and token compression. Channel compression methods include SparQAttn (ribar2023sparq) and LokiAttn (singhania2024loki). They construct the mask by carrying full attention with reduced dimensionality. However, as the dimension is already small, e.g., 64, 128, in commonly used attention, the speedup potential might be limited. Token compression methods include MInference (jiang2407minference) and FlexPrefill (FlexPrefill). They construct the mask by compressing each block of tokens to a single token and compute attention on this shorter sequence. However, this approximation is too aggressive: missing important blocks of 
P
 is possible if they do not have a large attention score on the compressed sequence. SeerAttention (gao2024seerattention) requires training of additional parameters for attention, which is expensive to use. Moreover, they are all designed for language models, and their applicability to other model types, such as diffusion models, remains uncertain. (3) Training-based methods modify the attention computation logic, requiring retraining the entire model, such as Reformer (kitaev2020reformer) and FastAttention (pagliardini2023fast). These methods are much more expensive to use than training-free methods.

There are other ways to accelerate attention (zhangsurvey), such as optimizing the kernel implementation (dao2022flashattention; dao2023flashattention; shah2024flashattention), quantization (2024sageattention; zhang2024sageattention2; zhang2025sageattention2++; zhang2025sageattention3), distributing the workload (liu2023ringattentionblockwisetransformers), and designing linear time attention (wang2020linformer; choromanski2020rethinking; yu2022metaformer; katharopoulos2020transformers). They are orthogonal to our approach.

Refer to caption
Figure 3:Workflow of SpargeAttn.
