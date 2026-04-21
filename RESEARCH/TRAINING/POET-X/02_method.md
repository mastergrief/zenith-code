# POET-X — Method: Scaling Orthogonal Transformation

← back to [POET-X.md](POET-X.md)

## 3 POET-X: Fast, Memory-efficient Training by Scaling Orthogonal Transformation

We build POET-X on top of block-stochastic POET due to its uniform coverage of all dimensions of the weight matrix. This property is important for memory-efficiency, because it ensures the balanced weight update even with a very small number of parameters unlike the fully-stochastic POET (see Figure 1). In block-stochastic POET, for the 
i
-th iteration, the orthogonal matrix 
𝑹
i
 is parameterized by

𝑹
i
=
𝚿
i
⊤
⏟
Column-permute
⋅
Diag
​
(
𝑮
~
i
1
,
𝑮
~
i
2
,
⋯
,
𝑮
~
i
⌈
m
b
⌉
)
⏟
Orthogonal matrix 
​
𝑮
i
⋅
𝚿
i
⏟
Row-permute
(2)
in which 
𝑮
~
i
j
∈
ℝ
b
×
b
 is the 
j
-th block of the block-diagonal orthogonal matrix 
𝑮
i
, and 
𝚿
i
,
∀
i
 are all random permutation matrices. 
𝑷
i
 is also parameterized the same way. Similarly to POET, POET-X optimizes weight matrices by multiplying 
𝑹
i
 and 
𝑷
i
 into 
𝑾
i
−
1
 after every certain number of iterations (i.e., performing the weight update 
𝑾
i
=
𝑹
i
​
𝑾
i
−
1
​
𝑷
i
). Therefore, how to perform these multiplications with the orthogonality constraint on 
𝑹
i
,
𝑷
i
 in a fast and memory-efficient way is our central challenge.

3.1Input-centric Implementation
The original implementation of POET directly operates on the weight matrix 
𝑾
 (i.e., 
𝑾
←
𝑹
i
​
𝑾
​
𝑷
i
), which is a weight-centric formulation. Despite simplicity, the weight-centric implementation incurs 
𝒪
​
(
n
​
m
2
)
 complexity (assuming an input vector 
𝒙
∈
ℝ
m
). Moreover, when computing the gradient w.r.t. 
𝑹
i
 and 
𝑷
i
, both gradients require accessing the weight matrix 
𝑾
, thus increasing the memory.

Inspired by matrix-free computation in solving large-scale linear systems, we use an input-centric formulation to implement the update 
𝑾
←
𝑹
i
​
𝑾
​
𝑷
i
, as shown below

𝑷
i
⊤
​
𝑾
⊤
⏟
① matrix-matrix mult.
​
𝑹
i
⊤
⏞
② matrix-matrix mult.
​
𝒙
⏟
③ matrix-vector mult.
⇔
𝑷
i
⊤
​
𝑾
⊤
​
𝑹
i
⊤
​
𝒙
⏟
① matrix-vector mult.
⏞
② matrix-vector mult.
⏟
③ matrix-vector mult.
(3)
where the left is the weight-centric formulation which requires two matrix-matrix multiplications and one matrix-vector multiplication, and the right is the input-centric formulation which requires three matrix-vector multiplications. The input-centric formulation has also been shown effective in orthogonal finetuning (Qiu et al., 2025b). However, unlike orthogonal finetuning which only has one orthogonal matrix 
𝑹
i
 to learn (without the need to access 
𝑾
), POET-X has an additional orthogonal matrix 
𝑷
i
 on the left of 
𝑾
 in the above formula. It is highly nontrivial to achieve memory-efficiency in this case, because computing the gradient w.r.t. 
𝑷
i
 still requires accessing 
𝑾
, incurring large memory consumption and computational overhead.

3.2Permutation Acceleration and Reduction
To address this challenge, we start by writing out the complete inference formula for one weight matrix:

𝒛
=
𝚽
n
​
𝑮
P
⊤
​
𝚽
n
⊤
​
𝑾
​
𝚽
m
​
𝑮
R
⊤
​
𝚽
m
⊤
​
𝒙
(4)
where 
𝑹
=
𝚽
m
⊤
​
𝑮
R
​
𝚽
m
 and 
𝑷
=
𝚽
n
⊤
​
𝑮
P
​
𝚽
n
. We simplify the notation by dropping the iteration index 
i
. Because the inference involves the multiplication of four permutation matrices, we focus on reducing their memory cost.

Permutation acceleration. To avoid explicitly construct the permutation matrices, we implement our customized CUDA operator to perform the permutation. The key idea is to implement an index mapping. Let 
𝑾
∈
ℝ
m
×
n
 be the original matrix, 
𝑾
i
,
:
 be the 
i
-th row of 
W
, and 
𝑾
:
,
j
 denote the 
j
-th column. We define two base index sets 
ℐ
m
=
{
1
,
2
,
⋯
,
m
}
 and 
ℐ
n
=
{
1
,
2
,
⋯
,
n
}
. Then a permutation of indices 
π
 is defined as 
π
​
(
ℐ
m
)
=
{
π
​
(
1
)
,
π
​
(
2
)
,
⋯
,
π
​
(
m
)
}
 where 
π
:
ℐ
→
ℐ
 is a bijection. 
π
−
1
 defines the inverse mapping of 
π
, and therefore we have 
π
​
(
π
−
1
​
(
i
)
)
=
i
. Let 
𝚿
m
∈
{
0
,
1
}
m
×
m
 be the permutation matrix defined by the permutation 
π
p
, and 
𝚿
n
∈
{
0
,
1
}
n
×
n
 be the permutation matrix defined by the permutation 
π
q
. Then we have the following equations that always hold true:

𝚿
m
​
𝑾
≡
𝑾
′
⇔
(
𝑾
′
)
i
,
:
=
𝑾
π
p
​
(
i
)
,
:
(5)
𝚿
m
T
​
𝑾
≡
𝑾
′
⇔
(
𝑾
′
)
i
,
:
=
𝑾
π
p
−
1
​
(
i
)
,
:
𝑾
​
𝚿
n
≡
𝑾
′
⇔
(
𝑾
)
:
,
j
=
𝑾
:
,
π
q
−
1
​
(
j
)
𝑾
​
𝚿
n
T
≡
𝑾
′
⇔
(
𝑾
′
)
:
,
j
=
𝑾
:
,
π
q
​
(
j
)
.
Therefore, to perform the multiplication between the permutation matrix and the weight matrix, we only need to store this permutation index set and access the weight matrix in a prescribed order. Such bijection mapping can be directly used for both forward and backward computation. We conduct an experiment in Table 1 (setup given in Appendix A) to show the acceleration performance of our customized CUDA operator. The results show that the customized CUDA operator is effective with up to 20
×
 speedup.

Hidden Dim.	PyTorch (ms)	Ours (ms)	Speedup
2048	1.621	0.086	18.75
×
4096	3.269	0.167	19.57
×
8192	6.584	0.393	16.76
×
16384	13.246	0.946	14.00
×
Table 1:Comparison between PyTorch-native and our customized permutation under different hidden dimensions.
Hidden Dim.	4 permute (ms)	2 permute (ms)	Speedup
Block size = 256, Compiled = False
4096	15.858	12.017	1.32
×
8192	38.583	31.028	1.24
×
16384	120.869	94.280	1.28
×
Block size = 256, Compiled = True
4096	8.705	7.496	1.16
×
8192	26.272	22.980	1.14
×
16384	90.802	78.863	1.15
×
Block size = 512, Compiled = False
4096	23.592	13.421	1.76
×
8192	42.062	34.398	1.22
×
16384	126.400	110.007	1.15
×
Block size = 512, Compiled = True
4096	10.235	8.986	1.14
×
8192	29.321	26.012	1.13
×
16384	96.798	86.380	1.12
×
Table 2:Runtime improvement of permutation reduction.
Permutation reduction. In the input-centric formulation of POET, the forward pass requires 4 permutations in total: 
π
p
, 
π
q
, 
π
p
−
1
 and 
π
q
−
1
. We find that 2 permutations can be merged to the weight matrix 
𝑾
 in advance:

𝒛
=
𝚽
n
​
𝑮
P
⊤
​
𝚽
n
⊤
​
𝑾
​
𝚽
m
⏟
Pre-computed by permuting 
𝑾
​
𝑮
R
⊤
​
𝚽
m
⊤
​
𝒙
.
(6)
In the inner loop of optimizing orthogonal matrices 
𝑮
P
 and 
𝑮
R
, we can pre-compute the permuted 
𝑾
 at the beginning since 
𝑾
 stays fixed in the inner loop. The pre-computation can avoid repeated permutation when learning 
𝑮
P
 and 
𝑮
R
. We empirically compare the runtime of performing the full 4 permutations all the time and the proposed permutation reduction strategy in Table 2 (setup given in Appendix A). The results well validate its effectiveness.

3.3Batch Parallel Computation for Block-diagonal Matrix Multiplication
Seq. Length	PyTorch (ms)	Ours (ms)	Speedup
2048	0.790	0.334	2.37
×
4096	1.525	0.642	2.38
×
8192	2.918	1.266	2.30
×
16384	6.120	2.661	2.30
×
Table 3:Runtime comparison between PyTorch-native block-diagonal matrix construction/multiplication and our batch-wise matrix multiplication strategy under different sequence length.
Seq. Length	PyTorch (MB)	Ours (MB)	Reduction
2048	280	192	31.43%
4096	360	272	24.44%
8192	520	432	16.92%
16384	840	752	9.55%
Table 4:GPU memory comparison between PyTorch-native block-diagonal matrix construction/multiplication and our batch-wise matrix multiplication strategy under different sequence length.
In the original implementation of block-stochastic POET, all orthogonal matrices adopt a block-diagonal sparse structure:

𝑮
P
=
Diag
​
(
𝑮
~
P
1
,
⋯
,
𝑮
~
P
⌈
n
b
⌉
)
,
𝑮
R
=
Diag
​
(
𝑮
~
R
1
,
⋯
,
𝑮
~
R
⌈
m
b
⌉
)
.
(7)
Consequently, the algorithm must construct numerous large yet sparse orthogonal matrices before performing matrix multiplications. However, we observe that, for block-diagonal matrices, multiplications occur only within each block, making it unnecessary to construct the full block-diagonal matrices in the first place. Motivated by this observation, we propose a batch-parallel strategy, in which we skip the explicit construction of block-diagonal matrices and instead treat each block as an independent matrix, performing batch-wise matrix multiplications accordingly. As shown in Table 3 and Table 4, our batch-parallel strategy not only saves GPU memory but also improves runtime.

3.4Efficient Cayley-Neumann Parameterization
Efficiently ensuring that each diagonal block (e.g., 
𝑮
~
P
i
,
∀
i
, 
𝑮
~
R
j
,
∀
j
) in Eq. 7 remains orthogonal during training poses a major challenge. The original POET addresses this by introducing the Cayley-Neumann Parameterization (CNP), which approximates the matrix inverse in the Cayley transform using a Neumann series. CNP improves numerical efficiency at the cost of a slight loss of orthogonality; however, empirical results show that this minor deviation does not affect performance (Qiu et al., 2025a).

Refer to caption
Figure 2:Illustration of efficient Cayley-Neumann parameterization (batch-wise implementation).
The orthogonalization procedure in POET consists of two steps: skew_symmetric and cayley_neumann. In skew_symmetric, it will construct a skew-symmetric matrix 
𝑸
∈
ℝ
b
×
b
 which satisfies 
𝑸
=
−
𝑸
⊤
. Then in cayley_neumann, it will turn a skew-symmetric matrix to an (approximately) orthogonal matrix. Based on empirical results in Qiu et al. (2025a), 
k
=
3
 achieves a good accuracy-efficiency trade-off. Therefore, we consider the CNP with 
k
=
3
 to construct an orthogonal matrix 
𝑮
:

𝑮
≈
(
𝑰
+
𝑸
)
​
(
𝑰
+
∑
i
=
1
3
𝑸
i
)
=
𝑰
+
2
​
𝑸
+
2
​
𝑸
2
+
2
​
𝑸
3
+
𝑸
4
,
(8)
which is used to replace orthogonal matrices during training and effectively converts a constrained optimization problem into an unconstrained one.

In POET-X, we propose to store skew-symmetric matrices only with their upper-triangular part. The parameter count of 
𝑸
 is 
b
​
(
b
−
1
)
/
2
 compared to the original 
b
2
. Consequently, the optimizer states and gradients are computed based on this compact representation rather than the full matrix, effectively reducing the POET-related memory footprint by half. This corresponds to the new skew_symmetric operation in POET-X, which efficiently constructs the skew-symmetric matrix 
𝑸
 from the actual trainable parameters. Then we consider the cayley_neumann operation. We re-arrange the terms in CNP (
k
=
3
) to the following form:

𝑮
≈
2
​
(
𝑸
+
𝑸
2
+
𝑸
2
⋅
𝑸
)
+
𝑸
2
⋅
𝑸
2
+
𝑰
,
(9)
from which we observe that all downstream computations depend solely on 
𝑸
 and 
𝑸
2
. This observation reveals a key opportunity for better efficiency. We leverage this dependency through kernel fusion, a simple yet powerful strategy: the two tensors (
𝑸
 and 
𝑸
2
) are loaded only once into the GPU’s low-latency shared memory. From this local cache, we compute higher-order terms (
𝑸
3
, 
𝑸
4
) and perform the final summation within a single Triton (Tillet et al., 2019) kernel. Compared to a naive PyTorch implementation that repeatedly reads 
𝑸
 and 
𝑸
2
 from the slower global GPU memory for each computation in 
𝑮
, our approach drastically reduces data transfer overhead. Besides, the fusion of multiple tensor operations into one custom kernel reduces the number of PyTorch operator calls, consequently improving the kernel launching time on the CPU.

Beside the forward pass, we find that the same strategy can also be applied in the backward pass. Given a function 
f
​
(
𝑮
​
(
𝑸
)
)
, the gradient w.r.t. 
𝑸
 is given by:

∇
1
=
∂
f
∂
𝑮
,
∇
2
=
∇
1
𝑸
⊤
+
𝑸
⊤
​
∇
1
,
∇
3
=
∇
1
(
𝑸
2
)
⊤
+
𝑸
⊤
∇
2
,
∇
4
=
∇
2
(
𝑸
2
)
⊤
+
(
𝑸
2
)
⊤
∇
2
,
∂
f
∂
𝑸
=
2
​
∇
1
+
2
​
∇
2
+
2
​
∇
3
+
∇
4
=
2
​
(
∇
1
+
∇
2
)
+
(
2
​
𝑸
⊤
+
(
𝑸
2
)
⊤
)
​
∇
2
+
(
2
​
∇
1
+
∇
2
)
​
(
𝑸
2
)
⊤
from which we observe that the gradient of 
f
 w.r.t. 
𝑸
 depends on 
𝑸
, 
𝑸
2
, as well as the gradient tensors 
∇
1
 and 
∇
2
. This dependency enables a similar strategy for shared-memory reuse and computation fusion. In our implementation, both the forward and backward kernels (including the batch-based optimizations) are implemented as customized Triton operators. This design enables fine-grained control over GPU memory access and computation, resulting in a substantial runtime speedup. To evaluate the new efficient CNP, we empirically compare our Triton implementation with the PyTorch-native implementation in Table 5. The results show 2-3
×
 speed up, validating its effectiveness.

Batch-wise CNP implementation. In practice, we implement both skew_symmetric and cayley_neumann operations in a batch-wise manner. This can further improve the runtime. An illustration is given in Figure 2.

N	PyTorch (ms)	Triton (ms)	Speedup
Block Size = 256
64	0.316	0.107	2.96
×
128	0.610	0.204	2.99
×
192	0.881	0.297	2.97
×
256	1.156	0.388	2.98
×
320	1.404	0.479	2.93
×
Block Size = 512
64	1.308	0.660	1.98
×
128	2.477	1.296	1.91
×
192	3.659	1.937	1.89
×
256	4.825	2.569	1.88
×
320	6.069	3.249	1.87
×
Table 5:Runtime comparison between PyTorch-native implementation of the Cayley-Neumann parameterization and our optimized Triton kernel under different number of blocks in POET-X.
3.5Boosting Memory-efficiency with Checkpointing
The efficiency of POET-X is substantially enhanced by its input-centric implementation, which incorporates various optimizations and custom kernels. To simplify the subsequent analysis, we omit the permutation matrices from Equation 4. This simplification is justified because our custom permutation kernel incurs no additional memory overhead. This leads to the following simplified forward pass: 
𝒛
=
𝑮
P
⊤
​
𝑾
​
𝑮
R
⊤
​
𝒙
. We know that for POET training, the central weight matrix 
𝑾
 does not require gradients, while the structured matrices 
𝑮
P
 and 
𝑮
R
 are the ones to be optimized. The input is 
𝒙
 and the output is 
𝒛
. The forward pass is executed as a sequence of three matrix multiplications:

mm1
: 
​
𝒂
=
𝑮
R
⊤
​
𝒙
,
mm2
: 
​
𝒃
=
𝑾
​
𝒂
,
mm3
: 
​
𝒛
=
𝑮
P
⊤
​
𝒃
.
(10)
PyTorch Autograd Engine requires saving intermediate activations during the forward pass to enable the backward pass. These saved tensors are one of the major contributors to the peak memory consumption. We thus begin by analyzing which additional activations have to be saved:

• mm3 backward computes 
∇
𝑮
P
=
𝒃
​
∇
𝒛
⊤
 and 
∇
𝒃
=
𝑮
P
​
∇
𝒛
. To compute the gradient for the parameters (i.e., 
∇
𝑮
P
), the Autograd engine must have saved the activation 
𝒃
 from the forward pass. This will result in saving an addtional tensor of shape 
ℝ
N
×
m
.
• mm2 backward computes 
∇
𝒂
=
𝑾
⊤
​
∇
𝒃
. Since 
𝑾
 has no gradient, the activation 
𝒂
 is not required to compute the gradient for 
𝑾
. Therefore, there is no additional activation needed to be saved.
• mm1 backward computes 
∇
𝑮
R
=
𝒙
​
∇
𝒂
⊤
 and 
∇
𝒙
=
𝑮
R
​
∇
𝒂
. To compute 
∇
𝑮
R
, the Autograd engine needs 
𝒙
, which is the original input and already available. For 
∇
𝒙
, the tensor 
𝑮
R
 is already in the memory.
We introduce two variants for POET-X that has different memory efficiency. The first, 
POET-X
fast
, follows standard Autograd logic, which necessitates saving an additional activation tensor during the forward pass. The second variant, 
POET-X
mem
, employs gradient checkpointing to circumvent this memory cost by recomputing the required tensor on-the-fly during the backward pass, making it our most memory-efficient version. 
POET-X
fast
 is faster and can be used when the memory is not a critical limitation. We provide an extensive ablation study to the compute-memory trade-off between these two variants in Section 4.

3.6POET-XQ: Quantized POET-X Training
With custom CUDA kernels for both POET-X forward and backward passes, POET-X can readily support quantized training. The core idea is to store only the base model’s low-bit quantized weight matrices and dequantize them on the fly whenever needed, such that activation involving high-precision weights are never stored in memory. For this reason, POET-XQ can be implemented only on 
POET-X
mem
, where intermediate activations are recomputed on the fly. In contrast, 
POET-X
fast
 would require storing an extra activation tensor, which in turn requires storing high-precision weight matrices. Without custom CUDA kernels, POET-X will need to store high-precision weights and thus cannot support memory-efficient quantized training.

