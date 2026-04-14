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

3TurboQuant: High Performance Quantization
We developed two VQ algorithms, each tailored to a specific objective. The first algorithm is designed to minimize the MSE between the original and reconstructed vectors after quantization. The second algorithm is optimized for unbiased inner product estimation, addressing the bias inherent in MSE-optimal quantizers. These algorithms are detailed in the following subsections.

Furthermore, in ??, we establish information-theoretic lower bounds on the best achievable distortion rates for any vector quantizer. This analysis demonstrates that TurboQuant achieve near-optimality, differing from the lower bound by only a small constant factor across all bit-widths.

3.1MSE Optimal TurboQuant
Let 
𝒙
∈
SS
d
−
1
 be a (worst-case) vector on the unit sphere in dimension 
d
. We aim to quantize 
𝒙
 to 
b
 bits per coordinate while minimizing the reconstruction MSE defined in ??. We start by randomizing this vector by multiplying it with a random rotation matrix 
𝚷
∈
ℝ
d
×
d
. We can generate 
𝚷
 by applying QR decomposition on a random matrix with i.i.d Normal entries.

The resulting rotated vector, 
𝚷
⋅
𝒙
, is uniformly distributed on the unit sphere 
SS
d
−
1
. As shown in ??, each coordinate of 
𝚷
⋅
𝒙
 follows a Beta distribution, which converges to a normal distribution in high dimensions. Furthermore, in high dimensions, distinct coordinates of 
𝚷
⋅
𝒙
 become nearly independent [55], allowing us to apply optimal scalar quantizers to each coordinate independently. Therefore, by ??, our task reduces to designing a scalar quantizer for random variables with the distribution 
f
X
​
(
x
)
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
 for 
x
∈
[
−
1
,
1
]
.

The optimal scalar quantization problem, given a known probability distribution, can be framed as a continuous k-means problem in dimension one. Specifically, we aim to partition the interval 
[
−
1
,
1
]
 into 
2
b
 clusters/buckets. The optimal solution adheres to a Voronoi tessellation [42], meaning interval boundaries are the midpoints between consecutive centroids, when arranged in sorted order. Therefore, with 
c
i
’s denoting the centroids in ascending order, we can formulate the scalar quantization as the following k-means optimization problem:

𝒞
​
(
f
X
,
b
)
:=
min
−
1
≤
c
1
≤
c
2
≤
…
≤
c
2
b
≤
1
​
∑
i
=
1
2
b
∫
c
i
−
1
+
c
i
2
c
i
+
c
i
+
1
2
|
x
−
c
i
|
2
⋅
f
X
​
(
x
)
​
𝑑
x
.
(4)
Note that 
𝒞
​
(
f
X
,
b
)
 in ?? denotes the optimal MSE cost function for bit-width 
b
, a quantity we will bound to prove the upper bound on the end-to-end MSE of TurboQuant. The problem in ?? can be solved using iterative numerical methods to achieve any desired precision. We solve ?? for a range of practically relevant bit-widths 
b
 once, and store the results for future uses by the quantizer.

For example, in moderately high dimensions 
d
, where the distribution 
f
X
​
(
x
)
 closely approximates a normal distribution, the optimal quantization centroids for bit-widths 
b
=
1
,
2
 are 
{
±
2
/
π
d
}
 and 
{
±
0.453
d
,
±
1.51
d
}
, respectively.

Therefore the quantizer 
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
 first computes 
𝚷
⋅
𝒙
 and then computes and stores the indices of the nearest centroids to each coordinate of this vector. The dequantization map 
Q
𝚖𝚜𝚎
−
1
:
{
0
,
1
}
b
⋅
d
→
ℝ
d
 reconstructs the vector by retrieving the centroids corresponding to the stored indices and then rotating the result back to the original basis through multiplication with 
𝚷
⊤
. A pseudocode for these procedures is given in ??.

Algorithm 1 
TurboQuant
𝚖𝚜𝚎
: optimized for MSE
1: input: dimension 
d
 and bit-width 
b
 // Global Parameters for Setting up 
TurboQuant
𝚖𝚜𝚎
2: Generate a random rotation matrix 
𝚷
∈
ℝ
d
×
d
3: Construct codebook by finding centroids 
c
1
,
c
2
,
…
​
c
2
b
∈
[
−
1
,
1
]
 that minimize MSE cost in ??  
4: Procedure 
Quant
𝚖𝚜𝚎
​
(
𝒙
)
5: 
𝒚
←
𝚷
⋅
𝒙
6: 
𝚒𝚍𝚡
j
←
arg
⁡
min
k
∈
[
2
b
]
⁡
|
𝒚
j
−
c
k
|
 for every 
j
∈
[
d
]
{ 
𝚒𝚍𝚡
j
’s are 
b
-bit integers}
7: output: 
𝚒𝚍𝚡
  
8: Procedure 
DeQuant
𝚖𝚜𝚎
​
(
𝚒𝚍𝚡
)
9: 
𝒚
~
j
←
c
𝚒𝚍𝚡
j
 for every 
j
∈
[
d
]
10: 
𝒙
~
←
𝚷
⊤
⋅
𝒚
~
11: output: 
x
~
We are now ready to prove our main theorem for 
TurboQuant
𝚖𝚜𝚎
.

Theorem 1 (performance guarantee: 
TurboQuant
𝚖𝚜𝚎
). For any bit-width 
b
≥
1
 and any vector 
𝐱
∈
SS
d
−
1
, the procedure 
Quant
𝚖𝚜𝚎
​
(
𝐱
)
 in ?? outputs an index vector 
𝚒𝚍𝚡
∈
[
2
b
]
d
. When this index vector is passed to the primitive 
DeQuant
𝚖𝚜𝚎
​
(
𝚒𝚍𝚡
)
, it produces a reconstructed vector 
𝐱
~
∈
ℝ
d
 that satisfies the following distortion bounds:
• MSE defined as 
D
𝚖𝚜𝚎
:=
𝔼
𝒙
~
[
‖
𝒙
−
𝒙
~
‖
2
2
]
 is bounded by 
D
𝚖𝚜𝚎
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
• For small bit-widths, specifically 
b
=
1
,
2
,
3
,
4
 the MSE exhibits finer-grained distortion values: 
D
𝚖𝚜𝚎
≈
0.36
,
0.117
,
0.03
,
0.009
, respectively.
Proof.We start the proof by showing that 
D
𝚖𝚜𝚎
=
d
⋅
𝒞
​
(
f
X
,
b
)
, where 
𝒞
​
(
f
X
,
b
)
 is the optimal MSE cost for scalar quantizer defined in ??. Let 
𝒚
~
 be defined as per line 9 of ??. Since 
𝚷
 is a rotation matrix we can write: 
‖
𝒙
−
𝒙
~
‖
2
=
‖
𝚷
⋅
𝒙
−
𝒚
~
‖
2
. Using the notation 
𝒚
=
𝚷
⋅
𝒙
 as per line 5 of ?? and plugging this into the definition of 
D
𝚖𝚜𝚎
 we can write:
D
𝚖𝚜𝚎
=
𝔼
[
‖
𝒚
−
𝒚
~
‖
2
2
]
=
∑
j
∈
[
d
]
𝔼
[
|
𝒚
j
−
𝒚
~
j
|
2
]
=
∑
j
∈
[
d
]
𝔼
[
|
𝒚
j
−
c
𝚒𝚍𝚡
j
|
2
]
=
d
⋅
𝔼
[
|
𝒚
1
−
c
𝚒𝚍𝚡
1
|
2
]
=
d
⋅
min
−
1
≤
c
1
≤
c
2
≤
…
≤
c
2
b
≤
1
​
∑
i
=
1
2
b
∫
c
i
−
1
+
c
i
2
c
i
+
c
i
+
1
2
|
x
−
c
i
|
2
⋅
f
X
​
(
x
)
​
𝑑
x
=
d
⋅
𝒞
​
(
f
X
,
b
)
.
The third equality above follows from the definition of 
𝒚
~
 in line 9 of ?? and the fourth line above follows because all 
𝒚
j
’s have identical distribution of 
𝒚
j
∼
f
X
​
(
⋅
)
 as shown in ??. The last two lines above follows because 
c
𝚒𝚍𝚡
j
 is chosen to be the nearest centroid to each coordinate 
𝒚
j
 in line 6.

Now we must bound the optimal k-means cost 
𝒞
​
(
f
X
,
b
)
. For moderate values of 
d
, 
f
X
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
. By numerically solving the optimization problem in ?? for values 
b
=
1
,
2
,
3
,
4
 we get that 
𝒞
​
(
f
X
,
b
)
≈
0.36
d
,
0.117
d
,
0.03
d
,
0.009
d
, respectively. For larger bit-widths 
b
>
4
, we can apply the Panter-Dite [44] high-resolution formula for the distortion of a fixed-rate scalar quantizer, yielding the following bound:

𝒞
​
(
f
X
,
b
)
≤
1
12
⋅
(
∫
f
X
​
(
x
)
1
/
3
​
𝑑
x
)
3
⋅
1
4
b
=
3
​
π
2
​
d
⋅
1
4
b
.
This completes the proof. ∎

Entropy Encoding Codebook Pointers.
TurboQuant’s efficiency can be further increased by applying entropy encoding to the indices that point to the closest codebook elements. Specifically, the probability of each codeword index appearing in the quantized vectors can be computed as 
p
ℓ
:=
∫
c
ℓ
−
1
+
c
ℓ
2
c
ℓ
+
c
ℓ
+
1
2
f
X
​
(
x
)
​
𝑑
x
. Optimally coding the indices, reduces the average bit-width to nearly the entropy of the distribution 
{
p
i
}
i
∈
[
2
b
]
. This lossless compression does not affect the distortion and provides a bit-width reduction at no cost. The most significant reduction occurs for 
b
=
4
, where the entropy of 
{
p
i
}
i
∈
[
2
b
]
 is approximately 
3.8
. Detailed calculations for optimal prefix codes reveal that the average bit-width can be reduced by 
5
%
. However, given the limited gain, we have chosen not to incorporate this technique into TurboQuant to maintain simplicity and speed.

3.2Inner-product Optimal TurboQuant
For important applications like nearest neighbor search, having an unbiased inner product estimator is essential. However, 
TurboQuant
𝚖𝚜𝚎
 presented in ?? does not provide unbiased inner product estimates with query vectors. To illustrate this, consider the case with a bit-width of 
b
=
1
. In this scenario, the optimal codebooks that solve the optimization problem in ??, for sufficiently large 
d
, are 
{
±
2
π
​
d
}
. This implies that the quantization map for 
TurboQuant
𝚖𝚜𝚎
 is 
Q
𝚖𝚜𝚎
​
(
𝒙
)
=
𝚜𝚒𝚐𝚗
​
(
𝚷
⋅
𝒙
)
 for any 
𝒙
∈
ℝ
d
, and the dequantization map is 
Q
𝚖𝚜𝚎
−
1
​
(
𝒛
)
=
2
π
​
d
⋅
𝚷
⊤
⋅
𝒛
 for any 
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
. Therefore, for large enough 
d
, according to ??, we have 
𝔼
[
⟨
𝒚
,
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
⟩
]
=
2
π
⋅
⟨
𝒚
,
𝒙
⟩
, which has a multiplicative bias of 
2
/
π
. This bias diminishes with increasing bit-widths 
b
, as we empirically demonstrate in ??.

To address this bias, we propose a solution that combines 
TurboQuant
𝚖𝚜𝚎
 with an instance of QJL [62]. Specifically, let 
Q
𝚖𝚜𝚎
 be the quantization map corresponding to 
TurboQuant
𝚖𝚜𝚎
 with a bit-width of 
b
−
1
. For any 
𝒙
∈
SS
d
−
1
 the residual vector, defined as 
𝒓
:=
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
, has a small L2 norm, i.e., on expectation 
𝔼
[
‖
𝒓
‖
]
=
𝒞
​
(
f
X
,
b
−
1
)
 (per ??). We can then apply the QJL quantization map 
Q
𝚚𝚓𝚕
 on this residual vector, resulting in an overall bit-width of 
b
 and providing the following unbiased inner product estimator:

⟨
𝒚
,
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
⟩
+
‖
𝒓
‖
2
⋅
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
𝒓
)
)
⟩
.
More formally, the quantization map 
Q
𝚙𝚛𝚘𝚍
:
SS
d
−
1
→
[
2
b
−
1
]
d
×
{
−
1
,
1
}
d
×
ℝ
 is defined as:

Q
𝚙𝚛𝚘𝚍
​
(
𝒙
)
=
[
Q
𝚖𝚜𝚎
​
(
𝒙
)
,
Q
𝚚𝚓𝚕
​
(
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
)
,
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
]
.
A pseudocode for this procedure is given in ??.

Algorithm 2 
TurboQuant
𝚙𝚛𝚘𝚍
: optimized for inner product
1: input: dimension 
d
 and bit-width 
b
 // Global Parameters for Setting up 
TurboQuant
𝚙𝚛𝚘𝚍
2: Instantiate a 
TurboQuant
𝚖𝚜𝚎
 with bit-width 
b
−
1
 as per ??
3: Generate a random projection matrix 
𝑺
∈
ℝ
d
×
d
 with i.i.d. entries 
𝑺
i
,
j
∼
𝒩
​
(
0
,
1
)
  
4: Procedure 
Quant
𝚙𝚛𝚘𝚍
​
(
𝒙
)
5: 
𝚒𝚍𝚡
←
Quant
𝚖𝚜𝚎
​
(
𝒙
)
6: 
𝒓
←
𝒙
−
DeQuant
𝚖𝚜𝚎
​
(
𝚒𝚍𝚡
)
{ residual vector}
7: 
𝚚𝚓𝚕
←
𝚜𝚒𝚐𝚗
​
(
𝑺
⋅
𝒓
)
{ QJL on residual vector}
8: output: 
(
𝚒𝚍𝚡
,
𝚚𝚓𝚕
,
‖
r
‖
2
)
  
9: Procedure 
DeQuant
𝚙𝚛𝚘𝚍
​
(
𝚒𝚍𝚡
,
𝚚𝚓𝚕
,
γ
)
10: 
𝒙
~
𝚖𝚜𝚎
←
DeQuant
𝚖𝚜𝚎
​
(
𝚒𝚍𝚡
)
11: 
𝒙
~
𝚚𝚓𝚕
←
π
/
2
d
⋅
γ
⋅
𝑺
⊤
⋅
𝚚𝚓𝚕
12: output: 
x
~
𝚖𝚜𝚎
+
x
~
𝚚𝚓𝚕
We prove the main result for 
TurboQuant
𝚙𝚛𝚘𝚍
 in the following theorem.

Theorem 2 (performance guarantee: 
TurboQuant
𝚙𝚛𝚘𝚍
). For any bit-width 
b
≥
1
 and any vector 
𝐱
∈
SS
d
−
1
, the procedure 
Quant
𝚙𝚛𝚘𝚍
​
(
𝐱
)
 in ?? outputs an index vector 
𝚒𝚍𝚡
∈
[
2
b
−
1
]
d
 along with a sign vector 
𝚚𝚓𝚕
∈
{
−
1
,
1
}
d
 and a positive number 
γ
≥
0
. When these vectors and the scalar value are passed to the primitive 
DeQuant
𝚙𝚛𝚘𝚍
​
(
𝚒𝚍𝚡
,
𝚚𝚓𝚕
,
γ
)
, it produces a reconstructed vector 
𝐱
~
∈
ℝ
d
 that for any vector 
𝐲
∈
ℝ
d
 satisfies the following properties:
• Expected inner-product 
𝔼
𝒙
~
[
⟨
𝒚
,
𝒙
~
⟩
]
=
⟨
𝒚
,
𝒙
⟩
• Inner-product distortion defined as 
D
𝚙𝚛𝚘𝚍
:=
𝔼
𝒙
~
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
𝒙
~
⟩
|
2
]
 is bounded by 
D
𝚙𝚛𝚘𝚍
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
• For small bit-widths, specifically 
b
=
1
,
2
,
3
,
4
, 
D
𝚙𝚛𝚘𝚍
 exhibits finer-grained distortion values: 
D
𝚙𝚛𝚘𝚍
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
Proof.First we compute the conditional expectation of the inner product estimate 
⟨
𝒚
,
𝒙
~
⟩
 conditioned on 
𝒙
~
𝚖𝚜𝚎
 as follows:
𝔼
[
⟨
𝒚
,
𝒙
~
⟩
|
𝒙
~
𝚖𝚜𝚎
]
=
𝔼
𝒙
~
𝚚𝚓𝚕
[
⟨
𝒚
,
𝒙
~
𝚖𝚜𝚎
+
𝒙
~
𝚚𝚓𝚕
⟩
|
𝒙
~
𝚖𝚜𝚎
]
=
⟨
𝒚
,
𝒙
~
𝚖𝚜𝚎
⟩
+
𝔼
𝒙
~
𝚚𝚓𝚕
[
⟨
𝒚
,
𝒙
~
𝚚𝚓𝚕
⟩
|
𝒙
~
𝚖𝚜𝚎
]
=
⟨
𝒚
,
𝒙
~
𝚖𝚜𝚎
⟩
+
⟨
𝒚
,
𝒓
⟩
=
⟨
𝒚
,
𝒙
⟩
,
where the first equality follows from the definition of 
𝒙
~
 in line 12 of the algorithm. The third equality above follows from ?? and last line follows from definition of the residual vector 
𝒓
=
𝒙
−
𝒙
~
𝚖𝚜𝚎
 in line 6. Now we can computed the unconditional expectation using the law of total expectation: 
𝔼
𝒙
~
[
⟨
𝒚
,
𝒙
~
⟩
]
=
𝔼
𝒙
~
𝚖𝚜𝚎
[
𝔼
[
⟨
𝒚
,
𝒙
~
⟩
|
𝒙
~
𝚖𝚜𝚎
]
]
=
𝔼
[
⟨
𝒚
,
𝒙
⟩
]
=
⟨
𝒚
,
𝒙
⟩
, which proves the first claim of the theorem.

We apply the same conditioning on 
𝒙
~
𝚖𝚜𝚎
, when computing the distortion, and then compute the resulting conditional distortion:

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
𝒙
~
⟩
|
2
|
𝒙
~
𝚖𝚜𝚎
]
=
𝔼
𝒙
~
𝚚𝚓𝚕
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
𝒙
~
𝚖𝚜𝚎
+
𝒙
~
𝚚𝚓𝚕
⟩
|
2
|
𝒙
~
𝚖𝚜𝚎
]
=
𝔼
𝒙
~
𝚚𝚓𝚕
[
|
⟨
𝒚
,
𝒓
⟩
−
⟨
𝒚
,
𝒙
~
𝚚𝚓𝚕
⟩
|
2
|
𝒙
~
𝚖𝚜𝚎
]
=
𝚅𝚊𝚛
​
(
⟨
𝒚
,
𝒙
~
𝚚𝚓𝚕
⟩
|
𝒙
~
𝚖𝚜𝚎
)
≤
π
2
​
d
⋅
‖
𝒓
‖
2
2
​
‖
𝒚
‖
2
2
,
where the second equality above follows from the definitions of 
𝒓
 and 
𝒙
~
𝚖𝚜𝚎
 in lines 6 and 10 of ??. The third line above follows because 
𝔼
[
⟨
𝒚
,
𝒙
~
𝚚𝚓𝚕
⟩
]
=
⟨
𝒚
,
𝒓
⟩
, by ??. The last line follows from the variance bound of QJL estimator shown in ?? and using the fact that 
𝒙
~
𝚚𝚓𝚕
 in line 11 is re-scaled by 
γ
=
‖
𝒓
‖
.

Now by law of total expectation along with the fact that 
𝒓
=
𝒙
−
𝒙
~
𝚖𝚜𝚎
 we can bound the inner product distortion as follows:

D
𝚙𝚛𝚘𝚍
=
𝔼
𝒙
~
𝚖𝚜𝚎
[
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
𝒙
~
⟩
|
2
|
𝒙
~
𝚖𝚜𝚎
]
]
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
⋅
𝔼
[
‖
𝒙
−
𝒙
~
𝚖𝚜𝚎
‖
2
2
]
=
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
⋅
D
𝚖𝚜𝚎
.
The theorem follows by invoking the MSE bounds from ?? with bit-width 
b
−
1
. ∎

3.3Lower Bounds
We show that TurboQuant achieves an optimal distortion rate, up to a small constant factor, for any bit-width by proving lower bounds on the best achievable distortion for any compression algorithm. Our lower bound proof leverages Yao’s minimax principle. This principle allows us to relate the lower bound for randomized algorithms with worst-case deterministic input vectors to the lower bound for deterministic algorithms with randomized input vectors. Subsequently, we derive a lower bound on the achievable distortion rate for the latter using Shannon’s lower bound (SLB) presented in ??. Formally, we prove the following theorem.

Theorem 3 (lower bound on best achievable compression distortion). For any randomized quantization algorithm 
Q
:
SS
d
−
1
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
 and any reconstruction map 
Q
−
1
:
{
0
,
1
}
b
⋅
d
→
ℝ
d
, there exist a hard input instance 
𝐱
∈
SS
d
−
1
 such that:
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
.
Furthermore, there exists a 
𝐲
∈
SS
d
−
1
 such that:

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
1
d
⋅
1
4
b
Proof.By Yao’s minimax principle the expected MSE of the optimal randomized compression algorithm for worst-case inputs (
D
𝚖𝚜𝚎
) is equal to the expected MSE of the optimal deterministic compression algorithm when applied to inputs drawn from a maximally difficult randomized distribution. By definition, the MSE of the latter scenario is lower-bounded by the best achievable MSE for inputs uniformly distributed on the unit hypersphere.
The best achievable MSE for a compression algorithm with bit-width 
b
, operating on uniformly distributed inputs from the sphere 
SS
d
−
1
, is lower bounded in ??. Therefore, by invoking ?? we conclude that 
D
𝚖𝚜𝚎
≥
1
4
b
.

Furthermore, from 
D
𝚖𝚜𝚎
≥
1
4
b
 and using the definition of 
D
𝚖𝚜𝚎
 we conclude that:

D
𝚖𝚜𝚎
=
∑
j
=
1
d
𝔼
[
|
𝒙
j
−
[
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
]
j
|
2
]
=
∑
j
=
1
d
𝔼
[
|
⟨
𝒆
j
,
𝒙
⟩
−
⟨
𝒆
j
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
1
4
b
.
By pigeonhole principle there exist an index 
j
∈
[
d
]
 such that 
𝔼
[
|
⟨
𝒆
j
,
𝒙
⟩
−
⟨
𝒆
j
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
1
d
⋅
1
4
b
, which completes the proof. ∎

We note that a comparable lower bound for the worst-case distortion in vector quantization can be derived using “sphere packing” arguments (indeed, with larger constants as this is a harder problem) [26]. However, ?? offers a more robust and relevant lower bound for our analysis. This is because it establishes a lower bound on the expected distortion, rather than the worst-case error, and aligns seamlessly with our upper bounds presented in ?? and ??.

4Experiments
All experiments are performed using a single NVIDIA A100 GPU. The experimental section is divided into two parts: one to empirically validate the theoretical results, and another to evaluate the performance of our methods on downstream tasks, specifically KV cache quantization and nearest neighbor vector search.

4.1Empirical Validation
(a)
TurboQuant
prod
Refer to caption
(b)
TurboQuant
mse
Refer to caption
Figure 1:Error distribution of 
TurboQuant
prod
 and 
TurboQuant
mse
 for Inner Product Estimation.
In this section, we verify the theoretical results established in previous sections. We conduct our experiments using the DBpedia Entities dataset, which has been encoded into a 1536-dimensional space using OpenAI3 embeddings. To perform our experiments, we randomly sample 100,000 data points from the dataset, denoted as training set, which serves as our primary dataset. Additionally, we extract 1,000 distinct entries, denoted as query set, to be used as query points.

We evaluate two quantization methods: 
TurboQuant
prod
 and 
TurboQuant
mse
. The method 
TurboQuant
mse
 is designed to be optimzed for estimating the mean squared error (MSE) between the quantized and original vectors. In contrast, 
TurboQuant
prod
 is unbiased for estimating the inner product between the quantized and original vectors.

(a)
TurboQuant
𝚙𝚛𝚘𝚍
Refer to caption
(b)
TurboQuant
𝚖𝚜𝚎
Refer to caption
Figure 2:The variance of Inner-product error remains constant for 
TurboQuant
𝚙𝚛𝚘𝚍
, while in 
TurboQuant
𝚖𝚜𝚎
 increases with the average inner product. Bit-width is 
b
=
2
.
Both methods are applied to the task of inner product estimation by quantizing training set and analyzing the distortion in inner product calculations across different bit widths. As shown in ??, increasing the bit width reduces variance in both methods. However, when used for inner product estimation, 
TurboQuant
mse
 introduces bias. This bias diminishes as the bit width increases and eventually converges to zero.

The experimental results, illustrated in ??, confirm that 
TurboQuant
prod
 remains unbiased for inner product estimation across all bit widths, while 
TurboQuant
mse
 gradually improves with increasing bit width.

As observed in ??, when quantizing to 2 bits, the variance remains constant regardless of the inner product of the original vector in the TurboQuantprod approach. However, the same plot indicates that the bias in the TurboQuantmse approach is dependent on the average inner product. As the average inner product increases, the bias also increases.

(a)inner-prod error
Refer to caption
(b)MSE
Refer to caption
Figure 3:Comparison of inner-product error and MSE against theoretical bounds across different bit ratios.
Along with the histograms, we also plot ?? the average inner product error and MSE between the original and quantized vectors across different bit ratios. These plots are drawn alongside the upper and lower bounds established in our theoretical analysis. Our observations confirm that the results align with the theoretical predictions. Specifically, for inner product estimation, the TurboQuantprod approach performs better at lower bit ratios. However, as the bit count increases, TurboQuantmse reduces bias and ultimately achieves superior performance in inner product estimation.

4.2Needle-In-A-Haystack
SnapKV
Score: 0.858
Refer to caption 	
PyramidKV
Score: 0.895
Refer to caption	
KIVI
Score: 0.981
Refer to caption
PolarQuant
Score: 0.995
Refer to caption 	
Full-Precision
Score: 0.997
Refer to caption	
TurboQuant
Score: 0.997
Refer to caption
Figure 4:Evaluation of Llama-3.1-8B-Instruct on the “Needle-In-A-Haystack” test, where a model must retrieve a hidden sentence from long-context sequences. While some methods struggle with recall, TurboQuant, despite being more than 
4
×
 quantized, achieves the same exact performance as the uncompressed baseline.
The “Needle-In-A-Haystack Test”” [32] is a benchmark designed to evaluate a model’s ability to retrieve specific information embedded within a long document. The test involves placing a unique sentence (the ”needle”) at an arbitrary location within a much larger text (the ”haystack”) and assessing whether the model can successfully extract it.

Following the experimental setup of Fu et al. [21], we conduct evaluations using the 
𝙻𝚕𝚊𝚖𝚊
-
3.1
-
𝟾
​
𝙱
-
𝙸𝚗𝚜𝚝𝚛𝚞𝚌𝚝
 model. To analyze performance across different input sequence lengths, we vary the document size from 4k to 104k tokens. The primary metric used for evaluation is the recall score, which measures how accurately the model retrieves the hidden sentence.

For comparison, we benchmark our approach against several state-of-the-art memory-efficient methods, including PolarQuant [28], SnapKV [38], PyramidKV [12], and KIVI [41]. Each method is tested under a memory compression ratio of 0.25, meaning that only 25% of the full KV cache is utilized.

The results, illustrated in ??, reveal that quantization methods with theoretical guarantees, such as PolarQuant and TurboQuant, outperform token-level compression techniques like SnapKV and PyramidKV, as well as scalar quantization approaches like KIVI, which lack formal theoretical guarantees. Notably, TurboQuant achieves identical performance to the full-precision model, even at 
4
×
 compression, making it a robust solution for long-context processing.

4.3End-to-end Generation on LongBench
We experiment with various KV cache compression algorithms on the LongBench dataset [10], which encompasses a broad range of long-text scenarios, including single- and multi-document question-answering, summarization, few-shot learning, synthetic tasks, and code completion. To ensure a balanced evaluation across different context lengths, we employ LongBench-E, a subset designed with a more uniform length distribution. This enables a fair assessment of each model’s performance across varying context sizes, making it a more reliable benchmark for evaluating compression techniques.

We compare TurboQuant against the leading baseline methods introduced in ??, using both 
𝙻𝚕𝚊𝚖𝚊
-
3.1
-
𝟾
​
𝙱
-
𝙸𝚗𝚜𝚝𝚛𝚞𝚌𝚝
 and 
𝙼𝚒𝚗𝚒𝚜𝚝𝚛𝚊𝚕
-
𝟽
​
𝙱
-
𝙸𝚗𝚜𝚝𝚛𝚞𝚌𝚝
. Unlike existing approaches such as KIVI and PolarQuant, which leave generated tokens unquantized, our method applies quantization even during the streaming generation process.

As shown in ??, our approach outperforms other methods for both 
𝙻𝚕𝚊𝚖𝚊
-
3.1
-
𝟾
​
𝙱
-
𝙸𝚗𝚜𝚝𝚛𝚞𝚌𝚝
 and 
𝙼𝚒𝚗𝚒𝚜𝚝𝚛𝚊𝚕
-
𝟽
​
𝙱
-
𝙸𝚗𝚜𝚝𝚛𝚞𝚌𝚝
, achieving significantly higher average scores. We evaluate our method using 2.5-bit and 3.5-bit quantization during text generation. These non-integer bit precisions result from our strategy of splitting channels into outlier and non-outlier sets, and applying two independent instances of TurboQuant to each, allocating higher bit precision to outliers. This outlier treatment strategy is consistent with prior work [63, 51] . For example, in our 2.5-bit setup, 32 outlier channels are quantized at 3 bits, while the remaining 96 channels use 2 bits, leading to an effective bit precision of 
(
32
×
3
+
96
×
2
)
/
128
=
2.5
. For 3.5-bit quantization, a different ratio of outliers and regular channels leads to a higher effective bit precision. Despite using fewer bits than competing techniques, TurboQuant maintains performance comparable to unquantized models. Remarkably, we achieve this while compressing quantized vectors by at least a factor of 
4.5
×
.

Method	KV Size	SingleQA	MultiQA	Summarization	Few shot	Synthetic	Code	Average
𝙻𝚕𝚊𝚖𝚊
-
3.1
-
𝟾
​
𝙱
-
𝙸𝚗𝚜𝚝𝚛𝚞𝚌𝚝
 
Full Cache	
16
45.29
45.16
26.55
68.38
59.54
46.28
50.06
KIVI	
3
43.38
37.99
27.16
68.38
59.50
44.68
48.50
KIVI	
5
45.04
45.70
26.47
68.57
59.55
46.41
50.16
PolarQuant	
3.9
45.18
44.48
26.23
68.25
60.07
45.24
49.78
TurboQuant (ours) 	
2.5
44.16
44.96
24.80
68.01
59.65
45.76
49.44
TurboQuant (ours) 	
3.5
45.01
45.31
26.00
68.63
59.95
46.17
50.06
𝙼𝚒𝚗𝚒𝚜𝚝𝚛𝚊𝚕
-
𝟽
​
𝙱
-
𝙸𝚗𝚜𝚝𝚛𝚞𝚌𝚝
 
Full Cache	
16
47.53
49.06
26.09
66.83
53.50
47.90
49.89
TurboQuant (ours) 	
2.5
48.38
49.22
24.91
66.69
53.17
46.83
49.62
 
Table 1:LongBench-V1 [10] results of various KV cache compression methods on 
𝙻𝚕𝚊𝚖𝚊
-
3.1
-
𝟾
​
𝙱
-
𝙸𝚗𝚜𝚝𝚛𝚞𝚌𝚝
.
4.4Near Neighbour Search Experiments
In this section, we establish the strength of our proposed method, even in the context of near-neighbor search. We conduct our experiments using the DBpedia [53] Entities dataset, which has been encoded into 1536-dimensional1
1https://huggingface.co/datasets/Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M
 and 3072-dimensional 2
2https://huggingface.co/datasets/Qdrant/dbpedia-entities-openai3-text-embedding-3-large-3072-1M
 spaces using OpenAI3 embeddings. Additionally, we evaluate performance on a lower-dimensional dataset, utilizing the standard GloVe [45] embeddings. To construct our experimental setup, we randomly sample 100,000 data points from the dataset, denoted as training set, which serves as our primary training and evaluation set. Furthermore, we extract 1,000 distinct entries, denoted as query set, to be used as query points for datasets that do not explicitly provide a query set. For the GloVe dataset, we use a pre-existing query set consisting of 10,000 points.

We compare our method, TurboQuant, against two baseline quantization approaches: Product Quantization (PQ) and RabitQ [22]. To ensure a fair comparison, we quantize the dataset training set using all three methods and evaluate their performance based on recall ratio at top-k, denoted as 1@k. Specifically, this metric assesses how often the true top inner product result is captured within the top-k approximated results returned by each algorithm.

Approach	d=200	d=1536	d=3072
Product Quantization	37.04	239.75	494.42
RabitQ	597.25	2267.59	3957.19
TurboQuant	0.0007	0.0013	0.0021
Table 2:Quantization time (in seconds) for different approaches across various dimensions using 4-bit quantization.
Product Quantization (PQ) relies on the k-means algorithm to construct codebooks, which require separate storage. As the number of bits increases, the size of the codebook grows exponentially, leading to additional storage overhead. In our experiments, we carefully tuned the parameters to match the bit allocation of other methods. The most efficient implementation, designed for rapid querying, employs AVX2 In-Register Lookup Tables (LUTs). Specifically, it uses LUT16 with (l = 16) codewords. However, we observed substantial quality degradation at this configuration. To achieve a balance between speed and accuracy, we opted for a version of PQ that uses LUT256, which contains 256 codewords. For 2-bit quantization, it groups 4 coordinates per lookup, while for 4-bit quantization, it groups 2 coordinates per lookup. Notably, since we use the same dataset for both training and evaluation, PQ benefits from an inherent advantage in this setup.

RabitQ. Unlike PQ, RabitQ lacks a fully vectorized implementation, making it impossible to leverage GPU acceleration. As a result, it runs significantly slower on CPU. Additionally, the method incurs extra computational overheads that we do not explicitly account for in the bit ratio comparisons. While RabitQ claims a certain bit ratio, in practice, it utilizes more bits than reported due to these inefficiencies.

Despite the advantages granted to the baseline methods, TurboQuant consistently outperforms both Product Quantization and RabitQ in terms of recall ratio across all experiments. This demonstrates the robustness and efficiency of our approach, making it a compelling alternative for high-dimensional quantization-based search tasks.

(a)GloVe - d=200
Refer to caption
(b)OpenAI3 - d=1536
Refer to caption
(c)OpenAI3 - d=3072
Refer to caption
Figure 5:Recall comparison on different datasets with different embedding dimensions.