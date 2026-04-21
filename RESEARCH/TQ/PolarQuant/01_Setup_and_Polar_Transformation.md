# PolarQuant — Part 1: Setup and Polar Transformation
_Part 1 of 3. See [`00_INDEX.md`](00_INDEX.md) for full paper TOC._

PolarQuant: Quantizing KV Caches with Polar Transformation
Insu Han
KAIST insu.han@kaist.ac.kr
Praneeth Kacham
Google Research pkacham@google.com
Amin Karbasi
Yale University amin.karbasi@yale.edu
Vahab Mirrokni
Google Research mirrokni@google.com
Amir Zandieh
Google Research zandieh@google.com
Abstract
Large language models (LLMs) require significant memory to store Key-Value (KV) embeddings in their KV cache, especially when handling long-range contexts. Quantization of these KV embeddings is a common technique to reduce memory consumption. This work introduces PolarQuant, a novel quantization method employing random preconditioning and polar transformation. Our method transforms the KV embeddings into polar coordinates using an efficient recursive algorithm and then quantizes resulting angles. Our key insight is that, after random preconditioning, the angles in the polar representation exhibit a tightly bounded and highly concentrated distribution with an analytically computable form. This nice distribution eliminates the need for explicit normalization, a step required by traditional quantization methods which introduces significant memory overhead because quantization parameters (e.g., zero point and scale) must be stored in full precision per each data block. PolarQuant bypasses this normalization step, enabling substantial memory savings. The long-context evaluation demonstrates that PolarQuant compresses the KV cache by over 
×
4.2
 while achieving the best quality scores compared to the state-of-the-art methods.

1Introduction
Transformer-based models form the backbone of modern artificial intelligence systems and have been instrumental in driving the ongoing AI revolution. Their applications span various domains, including frontier language models (LLM) [1, 3, 15] to text-to-image [32, 12, 28], text-to-video synthesis [16, 30], coding assistants [27] and even multimodal models that ingest text, audio, image, and video data [29, 15]. The self-attention mechanism [37] is at the heart of these models as it enables capturing the direct dependencies of all tokens in the input sequence. The ability of these models grows along with their size and context length [21], which leads to computational challenges in terms of huge memory consumption to support fast inference.

Most large language models, as well as multimodal and video models, adopt an autoregressive, decoder-only architecture that generates tokens sequentially. To avoid redundant attention score computations during the generation phase, these models employ a KV caching scheme, which stores the key and value embeddings of previously generated tokens in each attention layer. However, a significant challenge in deploying autoregressive Transformers lies in the substantial memory demands, as the KV cache size scales with both the model size (i.e., the number of layers and attention heads) and the context length. Furthermore, serving each model session typically necessitates its own dedicated KV cache, further compounding memory demands. This has become a significant bottleneck in terms of memory usage and computational speed, particularly for models with long context lengths. Thus, reducing the KV cache size while preserving accuracy is critical to addressing these limitations.

Several approaches have been proposed to address the KV caching challenge. Architectural solutions, such as multi-query attention [34], grouped-query attention [2], and multi-head latent attention [9], modify the transformer architecture to reduce the memory demands during inference by decreasing the number of key-value pairs that are to be stored.

Another orthogonal line of research focuses on reducing the KV cache size by pruning or evicting redundant or unimportant tokens [6, 44, 25, 38, 42, 24]. However, eviction strategies face limitations in long-context tasks that require precise knowledge extraction, such as needle-in-haystack scenarios. Additionally, some recent works tackle the issue from a systems perspective, such as offloading [35, 36] or integrating virtual memory and paging strategies into the attention mechanism [23].

A simple yet effective approach to reducing KV cache size is quantizing the floating-point numbers (FPN) in the KV cache by storing their approximations using fewer number of bits. Several quantization methods have been proposed specifically for the KV cache [40, 39, 11, 20, 43, 26, 17]. Recently, a new KV cache quantization method called QJL [41] introduced an efficient, data-oblivious 1-bit quantization approach based on sketching techniques. This method does not require tuning or adaptation to the input data, incurs significantly lower memory overhead compared to prior works, and achieves superior performance. A very recent work, Lexico [22], applies techniques from sparse representation learning to compress the KV cache by learning a universal dictionary such that all key and value embeddings are represented as extremely sparse vectors within the learned dictionary. Unfortunately, this approach requires solving a computationally expensive matching pursuit algorithm for each key and value embedding, making Lexico relatively slow.

Traditional KV cache quantization methods face significant “memory overhead” due to the need for data normalization before quantization. Most methods group data into blocks–either channel-wise or token-wise–and independently normalize each block which requires computing and storing quantization constants (e.g., zero points and scales) in full precision. This process can add over 1 additional bit per quantized number, resulting in considerable memory overhead. We show that applying a random preconditioning matrix on the embedding vectors eliminates the need for data normalization. This approach aligns with the recent use of random Hadamard matrices as preconditioners before quantizing embedding vectors in attention layers to improve quality [33, 4].

1.1Contributions
We propose quantizing KV vectors in polar coordinates instead of the usual Cartesian coordinates. This shift enables more efficient representation and compression of KV embeddings.

Random Preconditioning. We apply a random rotation to the vectors before quantization, which preserves inner products while randomizing the distribution of each vector. This preconditioning causes the angles in polar coordinates to concentrate, allowing us to quantize them with high precision using small bit-widths. We derive the analytical distribution of angles after preconditioning and leverage this insight to construct an optimized quantization codebook, minimizing quantization error.

Recursive Polar Transformation. We introduce a computationally efficient recursive polar transformation that converts vectors into polar coordinates, enabling practical deployment of our approach. We are able to prove an error bound in ?? showing our algorithm is asymptotically optimal for worst-case KV embedding vectors.

Performance on Long-Context Tasks. We evaluate PolarQuant on long-context tasks and demonstrate that it achieves the best quality scores compared to competing methods while compressing the KV cache memory by over 
×
4.2
.

2Preliminaries
We use boldface lowercase letters, such as 
𝒙
 and 
𝒚
, to denote vectors, and boldface uppercase letters, like 
𝑴
, to denote matrices. To denote a slice of a vector 
𝒙
 between the coordinate indices 
i
 and 
j
 inclusive of the endpoints, we use the notation 
𝒙
i
:
j
. For a matrix 
𝑴
, we write 
𝑴
i
,
:
 to denote its 
i
-th row vector, which we will simply refer to as 
𝑴
i
.

2.1Efficient Token Generation and KV Caching
Autoregressive Transformers often utilize cache storage for faster token generation. Given an input prompt, models encode the prompt information into two types of embeddings, called Key and Value. To generate subsequence tokens efficiently, the Key-Value (KV) embeddings are cached to avoid recomputing them.

The Key-Value (KV) caching method leverages the architecture of transformer decodcers, where a causal mask in applied in the attention mechanism. Once the keys and values are computed for a given token, they remain unchanged for subsequent token generation. By caching these key-value pairs, the model avoids redundant computations, as it only needs to compute the query for the current token and reuse the cached keys and values for attention.

This approach significantly reduces computation time during token generation. Instead of processing the entire sequence repeatedly, the KV cache enables the model to efficiently focus on the incremental computation of new tokens. This makes the method particularly useful in real-time applications, such as conversational AI and text generation, where fast and resource-efficient inference is critical.

2.2Random Preconditioning
A critical step in the PolarQuant algorithm is random preconditioning of the KV vectors prior to quantization. This involves applying a random projection matrix to the embedding vectors before quantizing them. To analyze the algorithm effectively, we rely on specific facts and properties of multivariate normal random variables, which are outlined below.

Refer to caption
Figure 1:Overview of recursive polar transformation procedure in ??
Fact 1.
For any positive integer 
d
, if 
𝐱
∈
ℝ
d
 is a zero mean unit variance isotropic Gaussian random variable in dimension 
d
, i.e., 
𝐱
∼
𝒩
⁢
(
0
,
𝐈
d
)
, then its 
2
-norm, denoted by 
r
:=
‖
𝐱
‖
2
, follows a generalized gamma distribution with the following probability density for any 
r
≥
0
:

f
R
⁢
(
r
)
=
2
2
d
/
2
⋅
Γ
⁢
(
d
/
2
)
⁢
r
d
−
1
⁢
exp
⁡
(
−
r
2
/
2
)
The proof of Fact 1 is provided in ??. We also use the following facts about the moments of the univariate normal distribution.

Fact 2 (Moments of Normal Random Variable).
If 
x
 is a normal random variable with zero mean and unit variance 
x
∼
𝒩
⁢
(
0
,
1
)
, then for any integer 
ℓ
, 
𝔼
x
∼
𝒩
⁢
(
0
,
1
)
[
|
x
|
ℓ
]
=
2
ℓ
/
2
⁢
Γ
⁢
(
(
ℓ
+
1
)
/
2
)
/
π
.

PolarQuant algorithm applies a random preconditioning prior to quantization. This preconditioning involves multiplying each embedding vector by a shared random sketch matrix 
𝑺
 with i.i.d. normal entries. By the Johnson-Lindenstrauss (JL) lemma [10], this preconditioning preserves the norms and inner products of the embedding vectors with minimal distortion. A key property of this preconditioning, which we will leverage in our later analysis, is that the embedding vectors after preconditioning follow a multivariate normal distribution. This is formalized in the following fact.

Fact 3.
For any vector 
𝐱
∈
ℝ
d
 if 
𝐒
∈
ℝ
m
×
d
 is a random matrix with i.i.d. normal entries 
𝐒
i
,
j
∼
𝒩
⁢
(
0
,
1
)
, then the vector 
𝐒
⋅
𝐱
 has multivariate normal distribution 
𝐒
⋅
𝐱
∼
𝒩
⁢
(
0
,
‖
𝐱
‖
2
⋅
𝐈
m
)
.

The following lemma establishes the distribution of the polar angle of a point 
(
x
,
y
)
 in dimension 2, where the 
x
 and 
y
 coordinates are independent samples from the Euclidean norm of multivariate normal random variables.

Lemma 1.
For any positive integer 
d
, if 
x
,
y
≥
0
 are two i.i.d. random variables with generalized gamma distribution with probability density function 
f
Z
⁢
(
z
)
=
2
2
d
/
2
⋅
Γ
⁢
(
d
/
2
)
⁢
z
d
−
1
⁢
exp
⁡
(
−
z
2
/
2
)
, then the angle variable 
θ
:=
tan
−
1
⁡
(
y
/
x
)
 follows the probability density function:

f
Θ
⁢
(
θ
)
=
Γ
⁢
(
d
)
2
d
−
2
⋅
Γ
⁢
(
d
/
2
)
2
⋅
sin
d
−
1
⁡
(
2
⁢
θ
)
.
Additionally, 
𝔼
[
Θ
]
=
π
/
4
 and 
Var
⁢
(
Θ
)
=
O
⁢
(
1
/
d
)
.

See ?? for a proof.

3PolarQuant
We now describe our approach of quantizing angles in polar coordinates and using it to the KV cache problem. In ??, we introduce how to recursively transform Cartesian vector to polar coordinates. In ??, we provide an analysis of polar angle distributions with preconditioning. In ??, we explain details of quantization polar transformed embeddings and practical implementation.

3.1Recursive Polar Transformation
There are various methods to derive the polar representation of 
ℝ
d
. Here we propose a polar transformation that can be recursively computed from the Cartesian coordinates of points in 
ℝ
d
. Throughout this work, we assume that 
d
 is an integer power of 
2
.

At a high level, our approach begins by grouping pairs of coordinates of a 
d
-dimensional vector 
𝒙
 and transforming each pair into 2D polar coordinates. This produces 
d
/
2
 radius and angle pairs. Next, we gather 
d
/
2
 of radii and apply the polar transform to them. This procedure is recursively repeated 
log
2
⁡
d
 times and the final output consists of a single final radius and a collection of 
1
,
2
,
4
,
…
,
d
/
2
-dimensional angle vectors. A formal definition is provided in ??.

Definition 1 (Cartesian to Polar Transformation).
For any integer power of two 
d
, the polar representation of any vector 
𝐱
∈
ℝ
d
 includes 
d
−
1
 angles and a radius. Angles are organized into a collection of 
log
2
⁡
d
 vector of angles 
ψ
(
1
)
,
ψ
(
2
)
,
…
⁢
ψ
(
log
2
⁡
d
)
 such that 
ψ
(
1
)
∈
[
0
,
2
⁢
π
)
d
/
2
 and 
ψ
(
ℓ
)
∈
[
0
,
π
/
2
]
d
/
2
ℓ
 for any 
ℓ
≥
2
. In other words, the angles are computed in 
log
2
⁡
d
 levels and there are 
d
/
2
ℓ
 angles in level 
l
. These angles are defined by the following relation for 
ℓ
∈
{
2
,
3
,
…
⁢
log
2
⁡
d
}
:

ψ
j
(
1
)
:=
tan
−
1
⁡
(
𝒙
2
⁢
j
/
𝒙
2
⁢
j
−
1
)
⁢
 for 
⁢
j
∈
[
d
/
2
]
,
ψ
j
(
ℓ
)
:=
tan
−
1
⁡
(
‖
𝒙
(
j
−
1
/
2
)
⁢
2
ℓ
+
1
:
j
⁢
2
ℓ
‖
2
‖
𝒙
(
j
−
1
)
⁢
2
ℓ
+
1
:
(
j
−
1
/
2
)
⁢
2
ℓ
‖
2
)
⁢
 for 
⁢
j
∈
[
d
/
2
ℓ
]
.
The reverse of this transformation maps the angles and the radius of any point to its Cartesian vector representation using the following equation:

𝒙
i
=
‖
𝒙
‖
2
⋅
∏
ℓ
=
1
log
2
⁡
d
(
cos
⁡
ψ
⌊
i
2
ℓ
⌋
(
ℓ
)
)
𝟏
{
(
i
⁢
mod
⁢
2
ℓ
)
≤
2
ℓ
−
1
}
⋅
∏
ℓ
=
1
log
2
⁡
d
(
sin
⁡
ψ
⌊
i
2
ℓ
⌋
(
ℓ
)
)
𝟏
{
(
i
⁢
mod
⁢
2
ℓ
)
>
2
ℓ
−
1
}
A visual diagram of the algorithm is shown in ?? and the pseudocode is provided in ?? (see Polar procedure). In what follows, we analyze the distribution of angles generated in each quantization level.

