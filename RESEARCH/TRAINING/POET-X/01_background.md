# POET-X — Background & Preliminaries

← back to [POET-X.md](POET-X.md)

1Introduction
Recent years have witnessed the remarkable progress of large language models (LLMs). However, training these models typically demands an enormous amount of computational resources, and the process often remains unstable. The reParameterized Orthogonal Equivalence Training (POET) algorithm (Qiu et al., 2025a) has recently demonstrated strong training stability, owing to its spectrum-preserving property. Nonetheless, POET suffers from poor memory efficiency and also runs significantly slower than Adam (Kingma & Ba, 2014) due to the cost of intensive large-scale matrix multiplications.

To address this, we propose POET-X, a fast, scalable and memory-efficient training algorithm that significantly enhances POET’s GPU memory and runtime efficiency while preserving its spectrum-preserving property. At the core of POET lies the orthogonal equivalence transformation, and our key contribution is to make this transformation scalable. This is achieved through a comprehensive analysis and optimization of the GPU memory usage and runtime cost of every computation involved in POET. By fully exploiting POET’s inherent sparse-training nature, POET-X achieves extremely efficient memory utilization comparable to parameter-efficient finetuning methods such as LoRA (Hu et al., 2022), while attaining a runtime comparable to Adam (Kingma & Ba, 2014). These results are particularly significant, as POET-X effectively enables the pretraining of billion-parameter LLMs (e.g., Llama-8B) on a single NVIDIA H100 GPU while achieving consistently better performance than the de facto AdamW optimizer.

Our underlying motivation for scaling POET arises from the great potential of sparse training. POET optimizes orthogonal matrices that transform neurons, and these matrices are generally constrained to be sparse. Such sparsity leads to strong parameter efficiency. However, in the original implementation of POET, this efficiency was not reflected in GPU memory usage, preventing POET from being practically applicable. Our work aims to bridge the gap between parameter and memory efficiency, thereby unlocking the full potential of POET’s sparse training.

Specifically, we introduce the following strategies to improve POET’s memory efficiency:

• Drawing inspiration from matrix-free methods (Chen, 2005) used for solving large-scale linear systems, we reformulate POET’s original weight-centric computation into an input-centric form. This reformulation turns POET into a sequence of linear maps, eliminating unnecessary memory consumption by avoiding the storage of intermediate activations associated with weight matrices.
• Leveraging the block-sparse structure of the orthogonal matrices in POET, we introduce a parallel batch-wise computation strategy for their matrix multiplications.
• We greatly improve the memory efficiency of the Cayley-Neumann parameterization (CNP) (Qiu et al., 2025a) for representing orthogonal matrices. This is done by storing only half of all skew-symmetric matrices in CNP.
After carefully benchmarking the runtime of POET, we identify the most time-consuming operations and take the following steps to improve them:

• After revisiting the matrix multiplication in CNP, we find that, with proper matrix rearrangement, the number of total matrix multiplications can be reduced. The same computation reduction can be performed in both forward and backward passes of CNP.
• For the permutation operations in POET-X, we exploit memory-efficient ways to implement them and also effectively get rid of multiple permutations by merging permutations to the weight matrices in advance.
• For all the computations in POET-X, we develop specialized CUDA kernels to ensure that both forward and backward passes can be performed efficiently.
The central contribution of this paper lies in developing effective means to scale up orthogonal equivalence transformations. While this scalability is crucial for enabling the memory efficiency of POET-X, the proposed techniques are actually of independent interest for optimizing orthogonal matrices in large-scale settings. We briefly summarize our contributions as follows:

• We carefully examine both forward and backward computations in original POET and identify multiple dimensions to improve memory and runtime efficiency.
• Compared to original POET, the proposed POET-X achieves 3x GPU memory reduction and 8x runtime speed-up without sacrificing original POET’s strong training stability. POET-X’s strong memory-efficiency makes it possible for a single Nvidia H100 GPU to pretrain a LLM up to 13B parameters. Our experiments demonstrate that POET-X consistently offers better-than-AdamW performance and LoRA-level GPU memory efficiency.
2Preliminaries of POET
POET reparameterizes each neuron as 
𝑾
R
​
P
=
𝑹
​
𝑾
0
​
𝑷
, where 
𝑾
0
∈
ℝ
m
×
n
 is a fixed random weight matrix, and 
𝑹
∈
ℝ
m
×
m
, 
𝑷
∈
ℝ
n
×
n
 are trainable orthogonal matrices. This formulation performs an orthogonal equivalence transformation (OET) on 
𝑾
0
, defined as 
OET
​
(
𝑾
;
𝑹
,
𝑷
)
=
𝑹
​
𝑾
​
𝑷
 which multiplies 
𝑾
 by orthogonal matrices from both sides. The forward pass of POET is thus

𝒚
=
𝑾
R
​
P
⊤
​
𝒙
=
(
𝑹
​
𝑾
0
​
𝑷
)
⊤
​
𝒙
,
(1)
s.t.
{
𝑹
⊤
𝑹
=
𝑹
𝑹
⊤
=
𝑰
,
𝑷
⊤
𝑷
=
𝑷
𝑷
⊤
=
𝑰
}
.
After training, 
𝑹
 and 
𝑷
 can be merged into 
𝑾
R
​
P
, ensuring that POET-trained networks have the no inference overhead.

Spectrum preservation. POET can be viewed as learning weight matrices by jointly transforming their left and right singular vectors while keeping the singular values fixed. Given the singular value decomposition (SVD) 
𝑾
0
=
𝑼
​
𝚺
0
​
𝑽
⊤
, the reparameterized weight matrix is expressed as 
𝑾
R
​
P
=
𝑹
​
𝑼
​
𝚺
0
​
𝑽
⊤
​
𝑷
, where both 
𝑹
​
𝑼
 and 
𝑽
⊤
​
𝑷
 are orthogonal. This construction effectively forms another SVD of 
𝑾
R
​
P
, ensuring that its spectral properties remain identical to those of the original matrix 
𝑾
0
. More interestingly, if we use zero-mean isotropic Gaussian to initialize 
𝑾
0
, then the hyperspherical energy (Liu et al., 2018) is also provably small due to its invariance under orthogonal transformation (Liu et al., 2021a, b; Qiu et al., 2025a). The properties of spectrum preservation and provably small hyperspherical energy guarantee POET’s training stability.

Refer to caption
Figure 1:Fully-stochastic POET (with 
b
=
1
/
8
) vs. block-stochastic POET (with 
b
=
8
) for the weight matrix update coverage. In the toy experiment, we use a 
64
×
64
 weight matrix and run both POET variants for 100-step update, so 200 are the largest possible update steps (multiplication by 
𝑹
 and 
𝑷
 counts as two updates). Block-stochastic POET ensures balanced update for the weight matrix while fully-stocahstic POET does not.
