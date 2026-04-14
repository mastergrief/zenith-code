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

3.2Distribution of Polar Angles Under Random Preconditioning
One of our primary objectives is to eliminate the need for explicit normalization (e.g., minimum/maximum values) of the KV cache data prior to quantization, thereby reducing quantization overhead. To achieve this, our algorithm applies random preconditioning to the embedding vectors. This preconditioning involves multiplying each embedding vector by a shared random sketch matrix 
𝑺
 with i.i.d. normal entries. By the Johnson-Lindenstrauss (JL) lemma [10], this preconditioning preserves the norms and inner products* of the embedding vectors with minimal distortion. A key property of this preconditioning, which we will leverage in our later analysis, is that the embedding vectors after preconditioning follow a multivariate normal distribution. This has been formalized in Fact 3.

During the preconditioning stage, the sketch is applied to all embedding vectors in the KV cache, allowing the analysis of PolarQuant to effectively treat the vectors being quantized as samples from a multivariate normal distribution. So for the analysis and design of PolarQuant we can assume without loss of generality that our goal is to quantize a random vector with multivariate Gaussian distribution. A critical insight is that the distribution of angles after random preconditioning becomes predictable and can be analytically derived, which enables the design of optimal quantization schemes.

The polar distribution of a Gaussian vector is derived in the following lemma.

Lemma 2 (Distribution of a Gaussian Vector Under Polar Transformation).
For an integer power of two 
d
, suppose that 
𝐱
∼
𝒩
⁢
(
0
,
I
d
)
 is a random zero mean isotropic Gaussian random variable in dimension 
d
. Let 
ψ
d
⁢
(
𝐱
)
:=
(
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
)
 denote the set of polar angles obtained by applying the polar transformation defined in ?? on 
𝐱
. Denote the radius of 
𝐱
 by 
r
=
‖
𝐱
‖
2
. The joint probability density function for 
(
r
,
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
)
 is the following:

f
R
,
Ψ
d
⁢
(
r
,
ψ
d
⁢
(
𝒙
)
)
=
f
R
⁢
(
r
)
⋅
∏
ℓ
=
1
log
2
⁡
d
f
Ψ
(
ℓ
)
⁢
(
ψ
(
ℓ
)
)
,
(1)
where 
f
R
⁢
(
r
)
 is the p.d.f. defined in Fact 1, 
f
Ψ
(
1
)
 is p.d.f. of the uniform distribution over 
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
:

f
Ψ
(
1
)
:
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
→
(
2
⁢
π
)
−
d
/
2
,
and for every 
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
 the p.d.f. 
f
Ψ
(
ℓ
)
 is the following:

f
Ψ
(
ℓ
)
:
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
→
ℝ
+
f
Ψ
(
ℓ
)
⁢
(
𝝍
)
=
∏
i
=
1
d
/
2
ℓ
Γ
⁢
(
2
ℓ
−
1
)
2
2
ℓ
−
1
−
2
⋅
Γ
⁢
(
2
ℓ
−
2
)
2
⁢
sin
(
2
ℓ
−
1
−
1
)
⁡
(
2
⁢
ψ
i
)
.
Proof.
The proof is by induction on 
d
. First for the base of induction we prove the result in dimension 
d
=
2
. So we prove that for a 2-dimensional random Gaussian vector 
𝒚
=
(
y
1
,
y
2
)
∈
ℝ
2
 if 
(
r
,
θ
)
 is the polar representation of this vector then following holds:

f
R
,
Θ
⁢
(
r
,
θ
)
=
1
2
⁢
π
⋅
r
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
,
To prove this, let 
f
Y
⁢
(
𝒚
)
 be the probability density function of the vector random variable 
𝒚
. We know 
𝒚
 has a normal distribution so we have:

f
R
,
Θ
⁢
(
r
,
θ
)
=
r
⋅
f
Y
⁢
(
𝒚
)
=
r
⋅
1
2
⁢
π
⁢
e
−
y
1
2
+
y
2
2
2
=
1
2
⁢
π
⋅
r
⁢
e
−
r
2
2
,
where the first equality above follows from the change of variable from 
(
y
1
,
y
2
)
 to 
r
=
y
1
2
+
y
2
2
 and 
θ
=
tan
−
1
⁡
(
y
2
/
y
1
)
. This proves the base of induction for 
d
=
2
.

Now we prove the inductive step. Suppose that the lemma holds for dimension 
d
/
2
 and we want to prove it for dimension 
d
. Denote 
θ
:=
ψ
(
log
2
⁡
d
)
, 
ϕ
1
:=
(
ψ
1
:
d
/
2
l
+
1
(
l
)
)
ℓ
=
1
log
2
⁡
d
−
1
, 
ϕ
2
:=
(
ψ
d
/
2
ℓ
+
1
+
1
:
d
/
2
ℓ
(
l
)
)
ℓ
=
1
log
2
⁡
d
−
1
, 
r
1
:=
‖
𝒙
1
:
d
/
2
‖
, and 
r
2
:=
‖
𝒙
d
/
2
+
1
:
d
‖
. Essentially we sliced all the angle vectors 
ψ
(
ℓ
)
 in half and named the collection of first half vectors 
ϕ
1
 and the collection of second halves 
ϕ
2
. Using the definition of 
ψ
(
ℓ
)
’s in ??, 
ϕ
1
 is exactly the polar transformation of 
𝒙
1
:
d
/
2
, and 
ϕ
2
 is the polar transformation of 
𝒙
d
/
2
+
1
:
d
, so by the definition of 
ψ
d
⁢
(
𝒙
)
 in the lemma statement we have 
ϕ
1
=
ψ
d
/
2
⁢
(
𝒙
1
:
d
/
2
)
 and 
ϕ
2
=
ψ
d
/
2
⁢
(
𝒙
d
/
2
+
1
:
d
)
. Thus, we can write:

f
R
,
Ψ
d
⁢
(
r
,
ψ
d
⁢
(
𝒙
)
)
=
f
R
,
Θ
,
Φ
1
,
Φ
2
⁢
(
r
,
θ
,
ϕ
1
,
ϕ
2
)
=
r
⋅
f
R
1
,
R
2
,
Φ
1
,
Φ
2
⁢
(
r
⁢
cos
⁡
θ
,
r
⁢
sin
⁡
θ
,
ϕ
1
,
ϕ
2
)
=
r
⋅
f
R
1
,
Φ
1
⁢
(
r
⁢
cos
⁡
θ
,
ϕ
1
)
⋅
f
R
2
,
Φ
2
⁢
(
r
⁢
sin
⁡
θ
,
ϕ
2
)
=
r
⋅
f
R
,
Ψ
d
/
2
⁢
(
r
1
,
ϕ
1
)
⋅
f
R
,
Ψ
d
/
2
⁢
(
r
2
,
ϕ
2
)
,
(2)
where the third line above follows from the change of variable from 
(
r
1
,
r
2
)
=
(
r
⁢
cos
⁡
θ
,
r
⁢
sin
⁡
θ
)
 to 
r
=
r
1
2
+
r
2
2
 and 
θ
=
tan
−
1
⁡
(
r
2
/
r
1
)
. In the fourth line above we used the definition of 
θ
=
ψ
(
log
2
⁡
d
)
=
tan
−
1
⁡
(
‖
𝒙
d
/
2
+
1
:
d
‖
2
‖
𝒙
1
:
d
/
2
‖
2
)
 from ??.

Now if we let 
f
Ψ
d
/
2
⁢
(
ϕ
1
)
:=
∏
ℓ
=
1
log
2
⁡
d
−
1
f
Ψ
(
ℓ
)
⁢
(
ψ
1
:
d
/
2
ℓ
+
1
(
ℓ
)
)
 and 
f
Ψ
d
/
2
⁢
(
ϕ
2
)
:=
∏
ℓ
=
1
log
2
⁡
d
−
1
f
Ψ
(
ℓ
)
⁢
(
ψ
d
/
2
ℓ
+
1
+
1
:
d
/
2
ℓ
(
ℓ
)
)
, by the inductive hypothesis we have 
f
R
,
Ψ
d
/
2
⁢
(
r
1
,
ϕ
1
)
=
2
2
d
/
4
⋅
Γ
⁢
(
d
/
4
)
⁢
r
1
d
/
2
−
1
⁢
exp
⁡
(
−
r
1
2
/
2
)
⋅
f
Ψ
d
/
2
⁢
(
ϕ
1
)
 and 
f
R
,
Ψ
d
/
2
⁢
(
r
2
,
ϕ
2
)
=
2
2
d
/
4
⋅
Γ
⁢
(
d
/
4
)
⁢
r
2
d
/
2
−
1
⁢
exp
⁡
(
−
r
2
2
/
2
)
⋅
f
Ψ
d
/
2
⁢
(
ϕ
2
)
. Plugging these values into ?? gives:

f
R
,
Ψ
d
⁢
(
r
,
ψ
d
⁢
(
𝒙
)
)
=
4
⋅
(
r
1
⁢
r
2
)
d
−
1
2
d
/
2
⁢
Γ
⁢
(
d
/
4
)
2
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
⋅
f
Ψ
d
/
2
⁢
(
ϕ
1
)
⋅
f
Ψ
d
/
2
⁢
(
ϕ
2
)
=
2
⁢
r
d
−
1
⋅
sin
d
/
2
−
1
⁡
(
2
⁢
θ
)
2
3
⁢
d
/
2
−
2
⋅
Γ
⁢
(
d
/
4
)
2
⁢
e
−
r
2
/
2
⋅
f
Ψ
d
/
2
⁢
(
ϕ
1
)
⋅
f
Ψ
d
/
2
⁢
(
ϕ
2
)
=
f
R
⁢
(
r
)
⋅
Γ
⁢
(
d
/
2
)
⋅
sin
d
/
2
−
1
⁡
(
2
⁢
θ
)
2
d
/
2
−
2
⋅
Γ
⁢
(
d
/
4
)
2
⁢
f
Ψ
d
/
2
⁢
(
ϕ
1
)
⋅
f
Ψ
d
/
2
⁢
(
ϕ
2
)
=
f
R
⁢
(
r
)
⋅
f
Ψ
d
⁢
(
ψ
d
⁢
(
𝒙
)
)
,
(3)
which completes the inductive proof of this lemma. ∎

?? demonstrates that the angles of Gaussian vectors in polar coordinates have independent distributions, as the probability density function is separable. Moreover, all angles within the same level share identical distributions. Specifically, at level 
ℓ
 all angles follow the distribution 
ψ
i
(
ℓ
)
∼
∏
i
=
1
d
/
2
ℓ
Γ
⁢
(
2
ℓ
−
1
)
2
2
ℓ
−
1
−
2
⋅
Γ
⁢
(
2
ℓ
−
2
)
2
⁢
sin
2
ℓ
−
1
−
1
⁡
(
2
⁢
ψ
i
(
ℓ
)
)
. This density becomes increasingly concentrated around 
π
/
4
, particularly at higher levels 
ℓ
. This property is highly beneficial for reducing quantization error for the angles at higher levels.

Algorithm 1 PolarQuant
1:  input: embedding 
𝑿
∈
ℝ
n
×
d
, precondition matrix 
𝑺
∈
ℝ
d
×
d
, bit width 
b
 // Cartesian to Polar transform
2:  
𝐑
i
,
𝚿
i
(
1
)
,
…
,
𝚿
i
(
log
2
⁡
d
)
←
 Polar
(
𝑿
i
⋅
𝑺
)
 for 
i
∈
[
n
]
 // Codebook Construction
3:  Find partition intervals and centroids 
(
I
k
(
ℓ
)
,
θ
k
(
ℓ
)
)
k
∈
[
2
b
]
 of 
𝚿
(
ℓ
)
∈
ℝ
n
×
(
d
/
2
ℓ
)
 that minimize the cost in ?? for 
ℓ
∈
[
log
2
⁡
d
]
 (See ?? for details) // Angles Quantization
4:  
𝑱
i
(
ℓ
)
←
Quant
⁢
(
𝚿
i
(
ℓ
)
,
(
I
k
(
ℓ
)
,
θ
k
(
ℓ
)
)
k
∈
[
2
b
]
)
 for 
i
∈
[
n
]
 and 
ℓ
∈
[
log
2
⁡
d
]
5:  output: 
𝐑
∈
ℝ
n
×
1
,
𝑱
(
1
)
∈
[
2
b
]
n
×
d
/
2
,
…
,
𝑱
(
log
2
⁡
d
)
∈
[
2
b
]
n
×
1
,
(
I
k
(
ℓ
)
,
θ
k
(
ℓ
)
)
k
∈
[
2
b
]
6:  Procedure Polar (
𝒚
)
7:  
𝒓
(
0
)
←
𝒚
∈
ℝ
d
8:  for 
ℓ
=
1
,
…
,
log
2
⁡
d
 do
9:     for 
j
=
1
,
…
,
d
/
2
ℓ
 do
10:        
𝝍
j
(
ℓ
)
←
tan
−
1
⁡
(
𝒓
2
⁢
j
(
ℓ
−
1
)
/
𝒓
2
⁢
j
−
1
(
ℓ
−
1
)
)
11:        
𝒓
j
(
ℓ
)
←
‖
𝒓
2
⁢
j
−
1
:
2
⁢
j
(
ℓ
−
1
)
‖
2
12:     end for
13:  end for
14:  output: 
𝒓
(
log
2
⁡
d
)
,
𝝍
(
1
)
,
…
,
𝝍
(
log
2
⁡
d
)
15:  Procedure Quant 
(
𝝍
,
(
I
k
,
θ
k
)
k
∈
[
2
b
]
)
16:  
𝒋
i
←
argmin
k
∈
[
2
b
]
|
θ
k
−
𝝍
i
|
 for 
i
∈
[
d
′
]
 s.t. 
𝝍
∈
ℝ
d
′
17:  output: 
𝒋
18:  Procedure DeQuant
(
𝒓
,
(
𝒋
(
ℓ
)
)
ℓ
∈
[
log
2
⁡
d
]
,
(
θ
k
(
ℓ
)
)
k
∈
[
2
b
]
,
𝑺
)
19:  for 
ℓ
=
log
2
⁡
d
,
…
,
1
 do
20:     for 
j
=
1
,
…
,
d
/
2
ℓ
 do
21:        
i
←
𝒋
j
(
ℓ
)
22:        
𝒓
2
⁢
j
−
1
(
ℓ
−
1
)
←
𝒓
j
(
ℓ
)
⋅
cos
⁡
θ
i
(
ℓ
)
23:        
𝒓
2
⁢
j
(
ℓ
−
1
)
←
𝒓
j
(
ℓ
)
⋅
sin
⁡
θ
i
(
ℓ
)
24:     end for
25:  end for
26:  output: 
r
(
0
)
⋅
S
⊤
3.3PolarQuant Algorithm and Main Theorem
PolarQuant starts by first applying random preconditioning, then transforming the vectors into polar coordinates, and finally quantizing each angle. Since ?? shows that the angles in polar coordinates are independent random variables, each angle can be quantized independently to minimize the total mean squared error. Jointly quantizing multiple angle coordinates offers no additional benefit due to their independence, making our approach both computationally efficient and effective. Therefore, we can focus on one angle at level 
l
 and design optimal quantization scheme for it so as to minimize the mean squared error.

Consider an angle 
ψ
i
(
ℓ
)
 at some level 
ℓ
. According to ??, its values lie within the range 
[
0
,
π
/
2
]
 for 
ℓ
≥
2
 and for 
ℓ
=
1
 it takes values in the range 
[
0
,
2
⁢
π
)
 with a probability density function given by 
f
ℓ
⁢
(
ψ
i
(
ℓ
)
)
:=
Γ
⁢
(
2
ℓ
−
1
)
2
2
ℓ
−
1
−
2
⋅
Γ
⁢
(
2
ℓ
−
2
)
2
⁢
sin
2
ℓ
−
1
−
1
⁡
(
2
⁢
ψ
i
(
ℓ
)
)
. The goal of quantization to 
b
-bits is to partition the range 
[
0
,
π
/
2
]
 (or 
[
0
,
2
⁢
π
)
 in case of 
ℓ
=
1
) into 
2
b
 intervals 
I
1
(
ℓ
)
,
I
2
(
ℓ
)
,
⋯
⁢
I
2
b
(
ℓ
)
 and find corresponding centroids 
θ
1
(
ℓ
)
,
θ
2
(
ℓ
)
,
…
⁢
θ
2
b
(
ℓ
)
 such that the following is mean squared error is minimized:

𝔼
ψ
i
(
ℓ
)
∼
f
ℓ
⁢
(
ψ
i
(
ℓ
)
)
[
∑
j
∈
[
2
b
]
:
ψ
i
(
ℓ
)
∈
I
j
(
ℓ
)
|
ψ
i
(
ℓ
)
−
θ
j
(
ℓ
)
|
2
]
.
(4)
This problem is a continuous analog of the k-means clustering problem in dimension 1. Since we have an explicit formula for the p.d.f. of angle 
ψ
i
(
ℓ
)
∼
f
ℓ
⁢
(
ψ
i
(
ℓ
)
)
=
Γ
⁢
(
2
ℓ
−
1
)
2
2
ℓ
−
1
−
2
⋅
Γ
⁢
(
2
ℓ
−
2
)
2
⁢
sin
2
ℓ
−
1
−
1
⁡
(
2
⁢
ψ
i
(
ℓ
)
)
 the optimal interval partitions and centroids for ?? can be efficiently computed using numerical methods. For example, one can run k-means clustering on the gathered angle values which can be considered samples from the distribution. This approach ensures minimal quantization error for each angle independently and the overall reconstruction error as well.

We provide a pseudocode of PolarQuant in ??. Our main result and error bound are proved in the following.

Theorem 1.
For a 
d
-dimensional vector 
𝐱
∼
N
⁢
(
0
,
I
d
)
, the polar quantization scheme in ?? uses 
O
⁢
(
log
⁡
1
/
ε
)
 bits per coordinate + the space necessary to store 
‖
𝐱
‖
2
, while reconstructing a vector 
𝐱
′
 from such a representation satisfying

𝔼
[
‖
𝒙
−
𝒙
′
‖
2
2
]
=
ε
⋅
‖
𝒙
‖
2
2
.
The proof of ?? can be found in ??. We note that a scheme which uses a deterministic 
ε
-net 
𝒩
 of the unit sphere 
𝕊
d
−
1
, with 
|
𝒩
|
=
O
⁢
(
1
/
ε
)
d
 and rounds the vector 
𝒙
^
=
𝒙
/
‖
𝒙
‖
2
 also uses 
O
⁢
(
log
⁡
1
/
ε
)
 bits per coordinate while achieving the above bounds in the worst case instead of in expectation over the Gaussian distribution. But our construction (i) gives the flexibility to vary the size of the codebook used per each level depending on the resource constraints and as the above theorem shows, can approach the same quality as pinning to an 
ε
-net on average, (ii) does not need to store a 
|
𝒩
|
-size codebook which is impractical even for modest sizes of 
d
 and (iii) has a fast decoding/encoding implementation.

4KV Cache Quantization with PolarQuant
In this section, we describe how PolarQuant can be applied to the KV cache problem and our practical implementation. Formally, given a stream of 
(
𝒒
1
,
𝒌
1
,
𝒗
1
)
,
…
,
(
𝒒
n
,
𝒌
n
,
𝒗
n
)
, where 
𝒒
i
,
𝒌
i
,
𝒗
i
∈
ℝ
d
 are query, key and value embeddings at 
i
-th generation step for all 
i
∈
[
n
]
. Let 
𝑲
:
i
,
𝑽
:
i
∈
ℝ
i
×
d
 be matrices defined by stacking 
𝒌
1
,
…
,
𝒌
i
 and 
𝒗
1
,
…
,
𝒗
j
 in their rows, respectively. The goal is to compute:

softmax
⁢
(
𝑲
:
i
⋅
𝒒
i
d
)
T
⋅
𝑽
:
i
.
(5)
For an efficient token generation, the KV cache at 
i
-th generation step 
(
𝑲
:
i
,
𝑽
:
i
)
 are stored in the memory. To reduce the memory space, we invoke PolarQuant (??) on these embeddings. Let 
*
⁢
m
⁢
𝑲
^
:
i
,
*
⁢
m
⁢
𝑽
^
:
i
∈
ℝ
i
×
d
 be their dequantizations using DeQuant procedure in ??. Then, we estimate ?? by computing

softmax
⁢
(
*
⁢
m
⁢
𝑲
^
:
i
⋅
𝒒
i
d
)
⋅
*
⁢
m
⁢
𝑽
^
:
i
.
(6)
Note that the naïve cache requires 
d
⋅
b
FPN
 memory space to store each 
d
-dimensional embedding where 
b
FPN
 is the number of bits to represent a single floating-point number. If we quantize 
log
2
⁡
d
 level angles with 
b
 bits each and keep centroids in 
b
FPN
 bits, the memory space becomes 
(
b
FPN
+
(
d
−
1
)
⁢
b
)
. For example, 
𝙻𝚕𝚊𝚖𝚊
-
3.1
-
𝟾
⁢
𝙱
-
𝙸𝚗𝚜𝚝𝚛𝚞𝚌𝚝
 is represented by 
b
FPN
=
16
 bits and has 
d
=
128
. For 
b
=
3
, we can save the memory space 
4.008
 times. In ??, the PolarQuant with KV cache marginally degrades the performance of LLMs on various tasks.

Refer to caption
(a)without random preconditioning
Refer to caption
(b)with random preconditioning
Figure 2:Distributions of angles of polar transformed key embeddings (a) with and (b) without random preconditioning. Preconditioning flattens the angle distribution and removes outliers which allows angle quantization more accurately.
4.1Practical Implementation
The PolarQuant algorithm recursively reduces the dimension of radii by half until the input has dimension 1. We recurse on the polar transformation for a constant 
L
=
4
 levels. Thus, for an embedding of dimension 
d
, we obtain 
d
/
16
-dimensional radii and 
15
⁢
d
/
16
 angle values. We also define different numbers of bits for each quantization level: 
b
=
4
 bits for the first level, and 
b
=
2
 bits for the remaining levels. This is because the range of angle at the first level 
[
0
,
2
⁢
π
)
 is 
4
 times wider than the others 
[
0
,
π
/
2
]
. Consequently, the representation of a block of 16 coordinates uses 
b
FPN
+
32
+
8
+
4
+
2
=
b
FPN
+
46
 bits that translates to 
62
/
16
=
3.875
 bits per coordinate when 
b
FPN
=
16
 bits.

We implement PolarQuant using the Pytorch [31] framework. Since the smallest data type is represented in 8 bits  (
𝚝𝚘𝚛𝚌𝚑
.
𝚞𝚒𝚗𝚝𝟾
), we pack quantized angle indices into 8-bit unit. To accelerate computation on GPU clusters, we implement CUDA kernels for two key operations: (1) the product of query vectors with the dequantized key cache, i.e., 
*
⁢
m
⁢
𝑲
^
:
i
⋅
𝒒
i
, and (2) the product of attention scores with the dequantized value cache as per ??. For the preconditioning matrix 
𝑺
, we generate a random rotational matrix. The matrix 
𝑺
 is shared across key and value embeddings, as well as all layers and attention heads in the Transformer architecture.

For angle codebook construction (line 3 in ??), we use the 1-D k-means++ clustering on either online angles obtained from polar-transformed inputs or offline precomputed angles. Both approaches approximate the solution to ?? by discretizing with samples from angle distributions. While online approach requires additional clustering computation during every prefill stage, this one-time cost is offset by improved performance compared to the offline approach. We present detailed runtime and performance comparisons in ??.

Refer to caption
(a)Exact (16 bits), Score: 
0.995
Refer to caption
(b)SnapKV, Score: 
0.858
Refer to caption
(c)PyramidKV, Score: 
0.891
Refer to caption
(d)KIVI, Score: 
0.984
Refer to caption
(e)PolarQuant, Score: 
0.991
Refer to caption
(f)PolarQuant-R, Score: 
0.990
Figure 3:Needle-In-A-Haystack test using 
𝙻𝚕𝚊𝚖𝚊
-
3.1
-
𝟾
⁢
𝙱
-
𝙸𝚗𝚜𝚝𝚛𝚞𝚌𝚝
. The test spans different depths and context lengths ranging from 4K to 104K. Green/red colors indicate high/low recall scores (higher is better). PolarQuant shows the best performance.
5Experiments
All experiments are performed with a single NVIDIA RTX A6000 GPU with 48GB VRAM.

5.1Random Precondition on KV Cache
We first explore the effectiveness of preconditioning. In particular, we choose a single prompt from Qasper dataset in LongBench [5] and extract the corresponding KV cache. To observe how preconditioning improves, we transform the KV cache into 4-level polar coordinates and plot their angle distributions of the key cache. Note that the first level angles are range in 
[
0
,
2
⁢
π
)
 and the rest are in 
[
0
,
π
/
2
]
. The results are illustrated in ??. As shown in ??, the distribution of angles get predictably sharper around 
π
/
4
 as the level increases. Moreover, we observe that at the first level the preconditioning flattens the angle distribution and removes outliers. This allows us to quantize angles in the KV cache more accurately.

5.2Needle-In-A-Haystack
Next we evaluate our method for the “Needle-In-A-Haystack” test [19]. It asks the model to retrieve the information in a given sentence where the sentence (the “needle”) is placed in an arbitrary location of a long document (the “haystack”). We follow the same setting from Fu et al. [14] and use the 
𝙻𝚕𝚊𝚖𝚊
-
3.1
-
𝟾
⁢
𝙱
-
𝙸𝚗𝚜𝚝𝚛𝚞𝚌𝚝
 to run the test. We vary the input sequence lengths from 4K to 104K. The evaluation is based on the recall score by comparing the hidden sentence. We compare PolarQuant to SnapKV [24], PyramidKV [8] and KIVI [26], where we use their implementations from [18]. All methods are set to a compression ratio of 
0.25
, i.e., required memory is 
×
0.25
 the full KV cache. Specifically, we run our algorithm with and without the preconditioning and refer to them as PolarQuant-R and PolarQuant, respectively. In ??, we observe that quantization methods (e.g., KIVI, PolarQuant) outperform token-level compression methods (e.g., SnapKV, PyramidKV). PolarQuant shows better scores than KIVI. Additionally, PolarQuant shows a marginally better score than PolarQuant-R.

5.3End-to-end Generation on LongBench
We run various KV cache compression algorithms for LongBench datasets [5], which encompasses diverse long-text scenarios including single/multi-document question-answering (SQA/MQA), summarization (Sum), few-shot learning (Few), synthetic tasks (Syn), and code completion (Code). Since the number of generated tokens is small compared to the input sequence length across all datasets, we preserve all new streamed query, key, and value pairs from the generation stage in full precision (16 bits) for all methods. We evaluate PolarQuant against the baseline methods using in ?? as well as StreamingLLM [38] and HeadKV [13] on 
𝙻𝚕𝚊𝚖𝚊
-
3.1
-
𝟾
⁢
𝙱
-
𝙸𝚗𝚜𝚝𝚛𝚞𝚌𝚝
.

We investigate two variants of PolarQuant-R: one using online codebook construction and another using offline one discussed in ??. The online variant performs clustering for each individual input prompt and layer, while the offline one employs a single precomputed codebook that is shared across all input prompts, layers, and attention heads. This offline approach is supported by our findings that the angle distribution, when preconditioned, remains consistent regardless of the input.

As reported in ??, our methods achieve superior performance compared to other methods, i.e., the average performance scores are higher by a large margin. This justifies the performance benefits of the quantization of polar coordinates. Moreover, the preconditioned variants (PolarQuant-R) generally demonstrates better performance than the non-preconditioned version. Among them, the online variant performs slightly better than the offline one.

Table 1:LongBench-V1 [5] results of various KV cache compression methods on 
𝙻𝚕𝚊𝚖𝚊
-
3.1
-
𝟾
⁢
𝙱
-
𝙸𝚗𝚜𝚝𝚛𝚞𝚌𝚝
. The best values among compression methods are indicated in bold.
Method	Task	Average
SQA	MQA	Sum	Few	Syn	Code
Exact (16 bits)	45.71	45.32	26.69	68.62	59.25	46.17	48.63
Snapkv	38.23	42.61	19.07	64.65	59.60	43.28	44.57
HeadKV	39.45	42.69	19.77	68.07	59.48	42.60	45.34
PyramidKV	36.80	41.54	18.91	64.88	59.68	42.38	44.03
StreamingLLM	25.68	35.79	20.90	56.91	58.81	32.07	38.36
KIVI	43.38	37.81	27.44	68.60	58.67	44.29	46.70
PolarQuant	44.03	44.34	27.32	68.68	59.82	44.46	48.11
PolarQuant-R (offline)
44.71	44.72	26.43	68.58	60.08	45.20	48.29
PolarQuant-R (online)
45.45	45.13	26.42	68.54	59.57	45.13	48.37
 
Table 2:Wall-clock runtime comparisons of various KV cache compression methods. The input sequence length is 
n
=
16
,
384
 and the number of generated tokens is 
1
,
024
.
Method	Prefill Time (sec)	Generation Time (sec)
Exact (16 bits)	2.934	38.374
SnapKV	3.438	34.053
PyramidKV	3.428	32.732
HeadKV	3.300	34.401
KIVI	3.590	49.564
PolarQuant	11.623	43.652
PolarQuant-R (online)
11.633	44.448
PolarQuant-R (offline)
3.364	44.097
 
5.4Runtime Analysis
We evaluate wall-clock runtimes of both prefill and token generation stages. Using the Llama model with an input prompt length of 
16
,
384
, we measure the time to generate 
1
,
024
 tokens for each method. Table 2 summaries the result. Token eviction approaches (SnapKV, PyramidKV, and HeadKV) demonstrate faster generation times compared to exact and quantization methods, though at the cost of lower quality. Among quantization approaches, our PolarQuant algorithms achieve 14% faster generation time than the KIVI while maintaining superior performance. These results demonstrate that PolarQuant offers advantages in both computational efficiency and model performance. To achieve faster prefill times, we recommend using offline codebook construction, as it significantly reduces runtime by eliminating the need for clustering, though this results in a modest performance trade-off. We leave even better codebook construction approaches for future research.

6Conclusion
We propose PolarQuant, a novel quantization method applied to angles in polar coordinates. We connect it to the random preconditioning which allows us to formalize angle distribution to be quantized. We provide rigorous theoretical bounds on quantization error. When applied to the KV cache compression problem, PolarQuant significantly reduces memory requirements during LLM inference while maintaining model performance. The principles underlying our method extend beyond KV cache compression, offering potential applications in LLM weight quantization and general vector similarity search problems.