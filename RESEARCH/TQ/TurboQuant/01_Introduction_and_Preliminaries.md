# TurboQuant — Part 1: Introduction and Preliminaries
_Part 1 of 3. See [`00_INDEX.md`](00_INDEX.md) for full paper TOC._

TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate
Amir Zandieh
Google Research zandieh@google.com
Majid Daliri
New York University daliri.majid@nyu.edu
Majid Hadian
Google DeepMind majidh@google.com
Vahab Mirrokni
Google Research mirrokni@google.com
Abstract
Vector quantization, a problem rooted in Shannon’s source coding theory, aims to quantize high-dimensional Euclidean vectors while minimizing distortion in their geometric structure. We propose TurboQuant to address both mean-squared error (MSE) and inner product distortion, overcoming limitations of existing methods that fail to achieve optimal distortion rates. Our data-oblivious algorithms, suitable for online applications, achieve near-optimal distortion rates (within a small constant factor) across all bit-widths and dimensions. TurboQuant achieves this by randomly rotating input vectors, inducing a concentrated Beta distribution on coordinates, and leveraging the near-independence property of distinct coordinates in high dimensions to simply apply optimal scalar quantizers per each coordinate. Recognizing that MSE-optimal quantizers introduce bias in inner product estimation, we propose a two-stage approach: applying an MSE quantizer followed by a 1-bit Quantized JL (QJL) transform on the residual, resulting in an unbiased inner product quantizer. We also provide a formal proof of the information-theoretic lower bounds on best achievable distortion rate by any vector quantizer, demonstrating that TurboQuant closely matches these bounds, differing only by a small constant (
≈
2.7
) factor. Experimental results validate our theoretical findings, showing that for KV cache quantization, we achieve absolute quality neutrality with 3.5 bits per channel and marginal quality degradation with 2.5 bits per channel. Furthermore, in nearest neighbor search tasks, our method outperforms existing product quantization techniques in recall while reducing indexing time to virtually zero.

1Introduction
Vector quantization (VQ) in Euclidean space is crucial for efficiently handling high-dimensional vectors across a spectrum of computational domains, from training and deploying large-scale AI and deep learning models to powering vector databases for search/retrieval systems. The core objective is to compress high dimensional vectors by quantizing them–converting floating-point coordinate values to low-bitwidth integers–while minimizing distortion, quantified by metrics such as mean-squared error (MSE) or inner product errors. By preserving these properties, inner product queries can be answered rapidly, with minimal latency, and using reduced computational and communication resources.

This problem’s roots trace back to Shannon’s seminal work on Source Coding theory [48, 49], which established that the least distortion achievable by block source codes, now known as vector quantizers, is defined by the Shannon distortion-rate function, determined by the statistical properties of the source and the chosen distortion measure, such as MSE. Today, VQ plays a critical role in fundamental computational domains, including AI, deep learning, and search systems.

A key application of VQ is in the deployment of AI models, including large language models (LLMs) [5, 18, 7, 52]. As LLM capabilities depend heavily on their model size and context length [34], serving them requires substantial memory demands and increased inference latency. This latency is primarily attributed to communication bottlenecks between HBM and SRAM on accelerators, or across distributed clusters. By compressing or quantizing model weights and activations, we can effectively mitigate these bottlenecks, resulting in significant reductions in inference costs. Inner product operations between activations and weights is at the core of deep learning models. Thus, model quantization schemes strive to compress weights and/or activation vectors while accurately preserving these inner products.

Decoder based transformer models [54] present another compelling use case. These models must store key/value (KV) embeddings from previously generated tokens in the KV cache, the size of which scales with both model size (number of layers and attention heads) and context length. This scaling is a significant bottleneck in terms of memory usage and computational speed, especially for long context models. Therefore, reducing the KV cache size without compromising accuracy is essential. In this context, the preservation of the Euclidean structure of these embedding vectors–their inner products and distances–is crucial for maintaining model performance. VQ emerges as the most suitable framework for addressing this challenge, offering a robust approach to compressing high-dimensional embeddings while preserving their essential geometric properties.

Additionally, nearest neighbor (NN) search in high-dimensional spaces with inner product or cosine similarity [1, 27] is a cornerstone of vector databases [4, 2, 3]. These databases are fundamental for retrieval-augmented generation [23, 19] and information retrieval [35, 46]. VQ, a.k.a. product quantization (PQ), plays a critical role in these applications. It enables efficient compression of database vectors, optimizes memory usage, and facilitates low-latency, accurate estimations of inner products with query vectors, thereby enabling fast and precise nearest neighbor searches.

Existing VQ algorithms present a trade-off: either they lack accelerator (vectorization) compatibility and exhibit slow computation, making them unsuitable for real-time AI applications like KV cache quantization, or they suffer from suboptimal distortion bounds relative to bit-width. Our objective is to introduce an algorithm that addresses these limitations. Specifically, we design TurboQuant: a lightweight, capable of online application (crucial for scenarios like KV cache quantization), and highly accelerator-friendly—a critical attribute for modern AI workloads.

The core of TurboQuant is a two-stage process. First, we develop a vector quantizer with optimal distortion rate in terms of mean-squared error (MSE). Subsequently, we apply a 1-bit quantizer to the residual, resulting in an unbiased and low-distortion inner product quantizer. We demonstrate that quantizers optimized for MSE do not produce unbiased estimators for inner products, and our two-stage solution effectively bridges this gap. Our MSE-optimal quantizer starts by randomly rotating 
d
-dimensional input vectors. Observing the key fact that each coordinate in the rotated vectors follows a Beta distribution, we design optimal Lloyd-Max quantizer [42, 43] for each coordinate by solving a continuous k-means problem. This method gives optimal MSE distortion bound and minimizes the L2 norm of the residual. To obtain an unbiased and low-distortion quantizer for inner products, we compose our quantizer with the recently developed Quantized Johnson-Lindenstrauss (QJL) transform [62], which quantizes each coordinate of the residual vector to a single bit. Our algorithm offers provably optimal distortion bounds for both MSE and inner products, achieving an exponential improvement over existing methods in terms of bit-width dependence.

1.1Problem Definition
Formally, our goal is to design a quantization map, denoted as 
Q
:
ℝ
d
→
{
0
,
1
}
B
, that transforms 
d
-dimensional vectors to a binary string of 
B
 bits. If we set 
B
=
b
⋅
d
 for some 
b
≥
0
, this quantizer will have a bit-width of 
b
, representing the average number of bits used to encode each real-valued coordinate of 
ℝ
d
. Crucially, we require an inverse map, 
Q
−
1
:
{
0
,
1
}
B
→
ℝ
d
 that performs dequantization, approximately reconstructing original vectors from their quantized representations. Of course, this transformation is inherently lossy, as 
Q
 is not a bijection. So, our primary objective is to minimize distortion, with a specific focus on mean-squared error (MSE) and inner product distortion.

We make no assumptions about the input vector dataset, considering the worst-case scenario. We let the quantizer 
Q
​
(
⋅
)
 to be randomized, leading to stochastic outputs. Considering randomized quantizers, it is more appropriate to define the expected distortion over the randomness of the quantizer’s output. Thus, we aim to design quantizers that for any desired bit-width 
b
 minimize the following expected distortion measures for any (worst-case) vectors 
𝒙
,
𝒚
∈
ℝ
d
:

(MSE)	
D
𝚖𝚜𝚎
:=
𝔼
Q
[
‖
𝒙
−
Q
−
1
​
(
Q
​
(
𝒙
)
)
‖
2
2
]
(1)
(inner-prod error)	
D
𝚙𝚛𝚘𝚍
:=
𝔼
Q
[
|
⟨
𝒚
,
𝒙
⟩
−
⟨
𝒚
,
Q
−
1
​
(
Q
​
(
𝒙
)
)
⟩
|
2
]
.
(2)
The expectations above are takes with respect to the randomness of the quantizer 
Q
​
(
⋅
)
. Furthermore, for inner-product quantizers, we require unbiasedness of the inner product estimator, a desirable property for numerous applications. More precisely, we require:

(unbiased inner-prod)	
𝔼
Q
[
⟨
𝒚
,
Q
−
1
​
(
Q
​
(
𝒙
)
)
⟩
]
=
⟨
𝒚
,
𝒙
⟩
.
We aim to design computationally efficient quantizers 
Q
𝚖𝚜𝚎
 and 
Q
𝚙𝚛𝚘𝚍
, that achieve optimal bounds for the distortion measures defined above, for any given bit-width 
b
. Additionally, we aim for 
Q
𝚙𝚛𝚘𝚍
 to provide unbiased inner product estimates. In particular, assume that we are given 
n
 real-valued vectors 
x
1
,
x
2
,
…
​
x
n
∈
ℝ
d
. We design the following primitives:

• Quant: efficiently quantizes the dataset and computes 
Q
​
(
𝒙
1
)
,
Q
​
(
𝒙
2
)
,
…
​
Q
​
(
𝒙
n
)
.
• DeQuant: given a quantized dataset, can efficiently reconstruct original vectors by computing 
Q
−
1
​
(
Q
​
(
𝒙
i
)
)
 for any 
i
∈
[
n
]
.
1.2Related Work
Beginnings of VQ.
The vector quantization theory started by Shannon’s seminal work [48, 49] on achievable distortion-rate functions. In 1963, Zador [61] made significant advances by employing high-resolution methods to derive the limiting operational distortion-rate function for fixed-rate quantization at high rates that closely matches Shannon’s distortion-rate function. However, Zador did not specifically consider implementable algorithms. Gersho’s influential paper [25], further advanced the vector quantization by popularizing high-resolution theory, simplifying Zador’s results, introducing lattice vector quantization, and proposing a key conjecture that shaped the field. Despite these theoretical advancements, the practical applicability of vector quantization remained unclear in early years. The most straightforward encoding method, brute-force nearest neighbor search, was computationally expensive, hindering the adoption of VQ in practice.

Online vs Offline Quantization.
Online (data-oblivious) quantization methods apply instantly without needing data-specific tuning or calibrations [16, 8, 41, 47, 28]. In contrast, offline (data-dependent) methods require heavy preprocessing and learning to adapt the quantization map to the data, making them unsuitable for dynamic data scenarios [37]. For instance, methods such as those presented in [20, 39, 57, 13] use second-order (Hessian) information to tune the quantization map which requires heavy preprocessing and even in some cases post processing as well.

Online KV Cache Compression.
Several approaches have been proposed to compress the KV cache. These include architectural modifications [50, 6, 15] which restructure the transformer to minimize the number of stored key-value pairs. Additionally, pruning or evicting redundant or less critical tokens has emerged as another approach [11, 66, 40, 58, 64, 38, 29].

A simple yet effective approach to reducing KV cache size is quantizing the KV cache. Several quantization techniques have been developed specifically for this purpose [60, 59, 17, 33, 65, 41, 30, 36, 28]. Recently, a new quantization called QJL [62] introduced an efficient, data-oblivious 1-bit quantization approach based on sketching techniques, which provides unbiased estimates for inner product queries. This method does not require tuning or adaptation to the input data and we make use of this technology in our quantizer optimized for inner product distortion.

Product Quantization (PQ).
In Near Neighbor (NN) search problem with Euclidean datasets, the index size poses a significant memory bottleneck, often mitigated by quantization techniques, commonly referred to as Product Quantization (PQ) in the NN literature. Many of these algorithms rely on constructing a quantization codebook using variations of k-means during the indexing phase [31, 9, 24, 56, 27]. Therefore, these methods are ill-suited for online settings due to their requirement for extensive preprocessing.

Recently, a grid-based PQ method was introduced in [22], eliminating the need for preprocessing. This approach operates by projecting a uniform grid onto the unit sphere and conducting a search to identify the nearest projection to the data points. While the paper’s theoretical guarantees are suboptimal, likely due to loose analysis—as practical performance surpasses theoretical bounds—the grid projection and binary search algorithm is also computationally slow and particularly inefficient on accelerators like GPU because of their algorithm’s inherent lack of vectorization, which prevents parallel processing.

1.3Overview of Techniques and Contributions
MSE Optimzied TurboQuant.
Our first VQ algorithm is designed to minimize MSE distortion deinfed in ??. To achieve this, we apply a random rotation to the input vectors, thereby inducing a Beta distribution on each coordinate, irrespective of the input vectors themselves. In high dimensions 
d
, the distribution of each coordinate converges to a Gaussian distribution 
𝒩
​
(
1
,
1
/
d
)
 due to concentration of measure and the central limit theorem. Furthermore, any two distinct coordinates become nearly uncorrelated and, more importantly, almost independent (a deeper result that goes beyond just correlation). This near-independence is a crucial aspect that simplifies our quantization design. It allows us to quantize each coordinate using optimal scalar quantization, disregarding interactions or correlations between different coordinates, while still achieving near-optimal distortion.

We find optimal scalar quantizers for random variables with Beta distributions by solving a continuous 
1
-dimensional k-means problem using the Max-Lloyd algorithm. We precompute and store these optimal codebooks for a range of practically useful bit-widths, to enable efficient subsequent invocations of our TurboQuant algorithm.

In ?? we prove that the 
b
-bit MSE optimized TurboQuant 
Q
𝚖𝚜𝚎
:
ℝ
d
→
{
0
,
1
}
b
⋅
d
 achieves the following distortion for any worst-case vector 
𝒙
∈
ℝ
d
 with 
‖
𝒙
‖
=
1
:

• 
D
𝚖𝚜𝚎
​
(
Q
𝚖𝚜𝚎
)
:=
𝔼
[
‖
𝒙
−
Q
𝚖𝚜𝚎
−
1
​
(
Q
𝚖𝚜𝚎
​
(
𝒙
)
)
‖
2
2
]
≤
3
​
π
2
⋅
1
4
b
 for any 
b
≥
0
.
• For small bit-widths the above distortion upper bound can be further refined. Specifically, for 
b
=
1
,
2
,
3
,
4
 we have 
D
𝚖𝚜𝚎
​
(
Q
𝚖𝚜𝚎
)
≈
0.36
,
0.117
,
0.03
,
0.009
, respectively.
Note that the unit norm assumption, 
‖
x
‖
2
=
1
, is standard and not restrictive. For datasets that do not satisfy this assumption we can compute and store the 
L
​
2
 norms in floating-point precision and rescale the dequantized points using these stored norms.

Inner Product TurboQuant.
We show that the MSE optimized quantizers are biased for inner product estimation and thus a different VQ scheme is needed to get an unbiased inner product quantizer. Our solution is a two stage algorithm that first applies the abovementioned 
Q
𝚖𝚜𝚎
 with a bit-width one less than our target budget and then apply a QJL [62] on the residual error. This is proved to be unbiased and also has nearly optimal inner product error rate.

In ?? we prove that the 
b
-bit inner product optimized TurboQuant 
Q
𝚙𝚛𝚘𝚍
:
ℝ
d
→
{
0
,
1
}
b
⋅
d
 achieves the following distortion for any worst-case vectors 
𝒙
,
𝒚
∈
ℝ
d
 with 
‖
𝒙
‖
=
1
:

• 
𝔼
[
⟨
𝒚
,
Q
𝚙𝚛𝚘𝚍
−
1
​
(
Q
𝚙𝚛𝚘𝚍
​
(
𝒙
)
)
⟩
]
=
⟨
𝒚
,
𝒙
⟩
• 
D
𝚙𝚛𝚘𝚍
​
(
Q
𝚙𝚛𝚘𝚍
)
:=
𝔼
[
|
⟨
𝒚
,
𝒙
⟩
−
⟨
𝒚
,
Q
𝚙𝚛𝚘𝚍
−
1
​
(
Q
𝚙𝚛𝚘𝚍
​
(
𝒙
)
)
⟩
|
2
]
≤
3
​
π
2
⋅
‖
𝒚
‖
2
2
d
⋅
1
4
b
 for any 
b
≥
0
.
• For small bit-widths the above distortion upper bound can be further refined. Specifically, for 
b
=
1
,
2
,
3
,
4
 we have 
D
𝚙𝚛𝚘𝚍
​
(
Q
𝚙𝚛𝚘𝚍
)
≈
1.57
d
,
0.56
d
,
0.18
d
,
0.047
d
, respectively.
Lower Bound.
In ??, we leverage Shannon’s lower bound and Yao’s minimax principle to prove that for any randomized quantization algorithm 
Q
:
ℝ
d
→
{
0
,
1
}
b
⋅
d
 with bit-width 
b
, there exist hard input instances 
𝒙
,
𝒚
∈
ℝ
d
 with 
‖
𝒙
‖
=
1
 such that the following lower bounds hold:

• 
D
𝚖𝚜𝚎
​
(
Q
)
:=
𝔼
[
‖
𝒙
−
Q
−
1
​
(
Q
​
(
𝒙
)
)
‖
2
2
]
≥
1
4
b
• 
D
𝚙𝚛𝚘𝚍
​
(
Q
)
=
𝔼
[
|
⟨
𝒚
,
𝒙
⟩
−
⟨
𝒚
,
Q
−
1
​
(
Q
​
(
𝒙
)
)
⟩
|
2
]
≥
‖
𝒚
‖
2
2
d
⋅
1
4
b
As demonstrated by our lower bounds, TurboQuant’s MSE distortion is provably within a factor of at most 
3
​
π
2
≈
2.7
 of the information-theoretical lower bound. Notably, for smaller bit-widths, this factor significantly decreases. For instance, at a bit-width of 
b
=
1
 TurboQuant achieves a distortion that is only a factor of approximately 
1.45
 away from the optimal which is also confirmed by our experimental results, indicating its efficiency in low-bit-width scenarios.

Experimental Results.
In ??, we empirically validate our theoretical distortion bounds, demonstrating that TurboQuant’s observed distortions closely align with our predictions across various real-world datasets, approaching the established lower bounds.

Furthermore, in ?? and ??, we showcase TurboQuant’s efficacy in online KV cache quantization. Specifically, we achieve perfect long-context retrieval in needle-in-a-haystack tasks and maintain high performance on other long-context downstream tasks, all while compressing the KV cache by a factor exceeding 
5
×
.

Finally in ?? we apply TurboQuant to various high-dimensional near neighbor search tasks. TurboQuant consistently outperforms data-dependent product quantization (PQ), while reducing the indexing time to essentially zero.

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

We use the notation 
SS
d
−
1
 to denote the hypersphere in 
ℝ
d
 of radius 
1
. For a random variable 
x
 we denote its differential entropy as 
h
​
(
x
)
. For random variables 
x
 and 
y
, the mutual information between them is denoted as 
I
​
(
x
;
y
)
=
h
​
(
x
)
−
h
​
(
x
|
y
)
.

Given that TurboQuant employs random rotation to mitigate worst-case input scenarios, understanding the statistical properties of random points on a hypersphere is essential. The following lemma outlines one such property that we will need for analysis and design purposes:

Lemma 1 (coordinate distribution of random point on hypersphere). For any positive integer 
d
 if 
𝐱
∈
SS
d
−
1
 is a random variable uniformly distributed over the unit hypersphere, then for any 
j
∈
[
d
]
 the coordinate 
𝐱
j
 follows the following (scaled/shifted) Beta distribution:
𝒙
j
∼
f
X
​
(
x
)
:=
Γ
​
(
d
/
2
)
π
⋅
Γ
​
(
(
d
−
1
)
/
2
)
​
(
1
−
x
2
)
(
d
−
3
)
/
2
.
In high dimensions this beta distribtion converges to the normal distribution 
f
X
​
(
⋅
)
→
𝒩
​
(
0
,
1
/
d
)
.

Proof.
f
X
​
(
x
)
 equals the ratio of the area of a sphere with radius 
1
−
x
2
 in dimension 
d
−
1
 to the volume of a unit sphere in dimension 
d
 scaled down by 
1
/
1
−
x
2
 (by Pythagorean theorem). Therefore,
f
X
​
(
x
)
=
2
​
π
(
d
−
1
)
/
2
Γ
​
(
(
d
−
1
)
/
2
)
⋅
(
1
−
x
2
)
(
d
−
2
)
/
2
2
​
π
d
/
2
Γ
​
(
d
/
2
)
⋅
1
/
1
−
x
2
=
Γ
​
(
d
/
2
)
π
⋅
Γ
​
(
(
d
−
1
)
/
2
)
​
(
1
−
x
2
)
(
d
−
3
)
/
2
.
∎

2.1Shannon Lower Bound on Distortion
The Shannon Lower Bound (SLB) is a powerful tool, derived from Shannon’s lossy source coding theorem [49], that provides a universal lower bound on the optimal achievable distortion rate for any lossy compression scheme. Specifically, we use a version of SLB tailored for the mean-squared error (MSE) distortion measure applied to general 
d
-dimensional sources.

Lemma 2 (SLB). Let 
𝐱
∈
ℝ
d
 be a random vector with an arbitrary probability distribution 
p
X
 and finite differential entropy 
h
​
(
𝐱
)
. Define the MSE distortion-rate function 
D
​
(
B
)
 for total bit complexity 
B
≥
0
 as:
D
(
p
X
,
B
)
:=
inf
{
𝔼
[
∥
𝒙
−
𝒚
∥
2
2
]
:
I
(
𝒙
;
𝒚
)
≤
B
}
,
where the infimum is taken over all joint distributions of 
𝐱
 and a reconstruction random vector 
𝐲
∈
ℝ
d
 such that the mutual information 
I
​
(
𝐱
;
𝐲
)
 is at most 
B
 and 
𝔼
[
‖
𝐱
−
𝐲
‖
2
2
]
 is the expected MSE distortion, calculated with respect to the joint distribution of 
𝐱
 and 
𝐲
. Then, for any bit complexity 
B
≥
0
, the following Shannon Lower Bound holds:

D
​
(
p
X
,
B
)
≥
d
2
​
π
​
e
⋅
2
(
2
/
d
)
​
(
h
​
(
𝒙
)
−
B
)
.
This is a classic result proved using backward Gaussian test channel (for a proof see [14]). Our lower bound result uses a corollary of SLB that corresponds to the uniformly distributed random points on the unit hyeprsphere. We present this in the following lemma:

Lemma 3 (SLB for random point on hypersphere). Let 
𝐱
∈
SS
d
−
1
 be a random variable uniformly distributed over the unit hypersphere and define the MSE distortion-rate function 
D
​
(
B
)
 for total bit complexity 
B
 as per ??. Then, for any bit complexity 
B
≥
0
, the following distortion lower bound holds:
D
​
(
B
)
≥
2
−
2
​
B
/
d
.
Proof.If we let 
A
d
 denote the area of the hypersphere 
SS
d
−
1
, the entropy of uniform distribution over hypersphere is 
h
​
(
𝒙
)
=
log
2
⁡
A
d
. Plugging this into the SLB from ?? we get 
D
​
(
B
)
≥
d
2
​
π
​
e
⋅
A
d
2
/
d
⋅
2
−
2
​
B
/
d
. Using Stirling’s approximation formula for Gamma function we have 
A
d
=
2
​
π
d
/
2
Γ
​
(
d
/
2
)
≥
(
2
​
π
​
e
d
)
d
/
2
⋅
2
​
d
π
⋅
(
1
−
O
​
(
1
/
d
)
)
. By substituting this into the inequality obtained from ?? we get the desired lower bound. ∎
2.2QJL: 1-bit inner product quantization
As previously stated, we design two VQ algorithms: one optimized for minimizing MSE and the other for minimizing inner product error. We show that MSE-optimal quantizers do not necessarily provide unbiased inner product estimates, particularly exhibiting significant bias at lower bit-widths. Our solution for inner product quantization is a two-stage algorithm. First, we apply the MSE-optimal quantizer using one less bit than the desired bit-width budget, thus minimizing the L2 norm of the residuals. Next we apply an unbiased and optimal single-bit quantizer to the residual. For the single-bit inner product quantizer, we utilize the recently proposed Quantized Johnson-Lindenstrauss (QJL) algorithm [62], which is an optimal inner product quantizer with a bit-width of one. Here, we present the QJL algorithm and its essential theoretical guarantees.

Definition 1 (QJL). For any positive integer 
d
 the QJL map 
Q
𝚚𝚓𝚕
:
ℝ
d
→
{
−
1
,
+
1
}
d
 is defined as:
Q
𝚚𝚓𝚕
​
(
𝒙
)
:=
𝚜𝚒𝚐𝚗
​
(
𝑺
⋅
𝒙
)
 for any 
​
𝒙
∈
ℝ
d
,
where 
𝐒
∈
ℝ
d
×
d
 is a random matrix with i.i.d. entries sampled from the normal distribution 
𝒩
​
(
0
,
1
)
 and the 
𝚜𝚒𝚐𝚗
 function is applied entry-wise to its vector input. The inverse/dequantization map 
Q
𝚚𝚓𝚕
−
1
:
{
−
1
,
+
1
}
d
→
ℝ
d
 is defined as:

Q
𝚚𝚓𝚕
−
1
​
(
𝒛
)
:=
π
/
2
d
⋅
𝑺
⊤
⋅
𝒛
 for any 
​
𝒛
∈
{
−
1
,
+
1
}
d
.
In the next lemma we restate the results from [62] that show the QJL is unbiased and also has small inner product distortion:

Lemma 4 (performance guarantee: QJL). Let 
Q
𝚚𝚓𝚕
 and 
Q
𝚚𝚓𝚕
−
1
 be defined as per ??. For any vector 
𝐱
∈
SS
d
−
1
 and any 
𝐲
∈
ℝ
d
 we have the following:
• Unbiased: 
𝔼
[
⟨
𝒚
,
Q
𝚚𝚓𝚕
−
1
​
(
Q
𝚚𝚓𝚕
​
(
𝒙
)
)
⟩
]
=
⟨
𝒚
,
𝒙
⟩
.
• Variance Bound: 
𝚅𝚊𝚛
​
(
⟨
𝒚
,
Q
𝚚𝚓𝚕
−
1
​
(
Q
𝚚𝚓𝚕
​
(
𝒙
)
)
⟩
)
≤
π
2
​
d
⋅
‖
𝒚
‖
2
2
Proof.The unbiasedness immediately follows from Lemma 3.2 of [62]. To show the variance bound let 
𝒔
1
,
𝒔
2
,
…
​
𝒔
m
 denote the rows of the random matrix 
𝑺
 in ??. We have:
⟨
𝒚
,
Q
𝚚𝚓𝚕
−
1
​
(
Q
𝚚𝚓𝚕
​
(
𝒙
)
)
⟩
=
1
d
​
∑
i
∈
[
d
]
π
/
2
⋅
𝒔
i
⊤
​
𝒚
⋅
𝚜𝚒𝚐𝚗
​
(
𝒔
i
⊤
​
𝒙
)
.
Since 
𝒔
i
’s are i.i.d. the above is indeed the average of 
d
 i.i.d. random samples defined as 
z
i
:=
π
/
2
⋅
𝒔
i
⊤
​
𝒚
⋅
𝚜𝚒𝚐𝚗
​
(
𝒔
i
⊤
​
𝒙
)
 for 
i
∈
[
d
]
. Let us now upper bound the variance of a single 
z
i
 using Fact 3.4 from [62]:

𝚅𝚊𝚛
​
(
z
i
)
=
π
/
2
⋅
𝚅𝚊𝚛
​
(
𝒔
i
⊤
​
𝒚
⋅
𝚜𝚒𝚐𝚗
​
(
𝒔
i
⊤
​
𝒙
)
)
≤
π
/
2
⋅
𝔼
[
(
𝒔
i
⊤
​
𝒚
)
2
]
=
π
/
2
⋅
‖
y
‖
2
2
,
(3)
where the last equality above follows because 
𝒔
i
⊤
​
𝒚
 is a Gaussian random variable with mean zero and variance 
‖
𝒚
‖
2
2
. Now the variance of the average of 
d
 i.i.d. random samples 
z
1
,
z
2
,
…
​
z
d
 is:

𝚅𝚊𝚛
​
(
⟨
𝒚
,
Q
𝚚𝚓𝚕
−
1
​
(
Q
𝚚𝚓𝚕
​
(
𝒙
)
)
⟩
)
=
1
d
2
​
∑
i
∈
[
d
]
𝚅𝚊𝚛
​
(
z
i
)
≤
π
2
​
d
⋅
‖
𝒚
‖
2
2
.
∎

