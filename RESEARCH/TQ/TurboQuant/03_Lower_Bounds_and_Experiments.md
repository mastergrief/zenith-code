# TurboQuant — Part 3: Lower Bounds and Experiments
_Part 3 of 3. See [`00_INDEX.md`](00_INDEX.md) for full paper TOC._

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