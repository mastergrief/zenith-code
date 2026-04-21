# PolarQuant — Part 3: Algorithm and Experiments
_Part 3 of 3. See [`00_INDEX.md`](00_INDEX.md) for full paper TOC._

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