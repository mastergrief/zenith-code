# QJL — Part 1: Introduction and Preliminaries
_Part 1 of 3. See [`00_INDEX.md`](00_INDEX.md) for full paper TOC._

QJL: 1-Bit Quantized JL Transform for KV Cache Quantization
with Zero Overhead
Amir Zandieh
Independent Researcher
amir.zed512@gmail.com
Majid Daliri
New York University
daliri.majid@nyu.edu
Insu Han∗
Adobe Research
insuh@adobe.com
July 19, 2024
Abstract
Serving LLMs requires substantial memory due to the storage requirements of Key-Value
(KV) embeddings in the KV cache, which grows with sequence length. An effective approach to
compress KV cache is quantization. However, traditional quantization methods face significant
memory overhead due to the need to store quantization constants (at least a zero point and a
scale) in full precision per data block. Depending on the block size, this overhead can add 1 or 2
bits per quantized number. We introduce QJL, a new quantization approach that consists of a
Johnson-Lindenstrauss (JL) transform followed by sign-bit quantization. In contrast to existing
methods, QJL eliminates memory overheads by removing the need for storing quantization
constants. We propose an asymmetric estimator for the inner product of two vectors and
demonstrate that applying QJL to one vector and a standard JL transform without quantization
to the other provides an unbiased estimator with minimal distortion. We have developed
an efficient implementation of the QJL sketch and its corresponding inner product estimator,
incorporating a lightweight CUDA kernel for optimized computation. When applied across
various LLMs and NLP tasks to quantize the KV cache to only 3 bits, QJL demonstrates a more
than fivefold reduction in KV cache memory usage without compromising accuracy, all while
achieving faster runtime. Codes are available at https://github.com/amirzandieh/QJL.
1 Introduction
Large language models (LLMs) have garnered significant attention and demonstrated remarkable
success in recent years. Their applications span various domains, including chatbot systems [1, 3]
to text-to-image [28, 11, 24], text-to-video synthesis [26], coding assistant [7] and even multimodal
domain across text, audio, image, and video [25]. The Transformer architecture with self-attention
mechanism [32] is at the heart of these LLMs as it enables capturing intrinsic pairwise correlations
across tokens in the input sequence. The ability of LLMs grows along with their model size [17],
which leads to computational challenges in terms of huge memory consumption.
Deploying auto-regressive transformers during the generation phase is costly because commercial
AI models must simultaneously serve millions of end users while meeting strict latency requirements.
One significant challenge is the substantial memory needed to store all previously generated keyvalue (KV) embeddings in cache to avoid recomputations. This has become a major memory and
speed bottleneck, especially for long context lengths. Additionally, the GPU must load the entire
∗Work done while at Yale University.
1
arXiv:2406.03482v2 [cs.LG] 18 Jul 2024
LLM
what is 20+24?
Prompt
encoding vector
cache
K V 44
Answer
LLM
(ATTN-MLP-LAYERNORM)x𝐿 (ATTN-MLP-LAYERNORM)x𝐿 Attention
softmax(q⊤K⊤)V
q
cache
Prompt Encoding Decoding (Token Generation)
q ∈ Rd
S ∈ Rm×d
S · q ∈ Rm
Sij ∼ N (0, 1)
k ∈ Rd sign(Sk) ∈ {±1}m S · k ∈ Rm
Query Embed.
JL
transform
Key Embed.
sign(·) ∥k∥2 /m
⟨Sq, QJL(S, k)⟩≈ε ⟨q, k⟩
Lemma 3.2 & 3.5
K
Cache
V
KV Cache Quantization
Per-token
Quantization
Per-token
Quantization QJL(·)
QJL(S, k)
×
Figure 1: Overview of the KV cache quantization via Quantized JL (QJL) transform
KV cache from its main memory to shared memory for each token generated, resulting in low
arithmetic intensity and leaving most GPU threads idle. Therefore, reducing the KV cache size
while maintaining accuracy is crucial.
There are several approaches to address this challenge. One method involves reducing the number
of heads in the KV cache using multi-query attention [29] and multi-group attention [2], but these
require fine-tuning the pre-trained models or training from scratch. Another line of work tries to
reduce the KV cache size by pruning or evicting unimportant tokens [39, 21, 33, 37]. Additionally,
some recent works tackle the issue from a system perspective, such as offloading [30] or using virtual
memory and paging techniques in the attention mechanism [18].
A simple yet effective approach is to quantize the floating-point numbers (FPN) in the KV
cache using fewer bits. Several quantization methods have been proposed specifically for the
KV cache [36, 34, 10, 16, 38]. Most recently, KIVI [22] and KVQuant [13] proposed per-channel
quantization for the key cache to achieve better performance. However, all existing quantization
methods for the KV cache face significant “memory overhead” issues. Specifically, all these methods
group the data into blocks, either channel-wise or token-wise, and calculate and store quantization
constants (at least a zero point and a scale) for each group. Depending on the group size, this
overhead can add approximately 1 or 2 additional bits per quantized number, which results in
significant computational overhead. In this work, our goal is to develop an efficient, data-oblivious
quantization method, referred to as a sketching technique. This method, which we call QJL, does
not need to be tuned by or adapted to the input data with significantly less overhead than prior
works, without any loss in performance.
2
1.1 Overview of Contributions
The decoding phase in the attention mechanism involves the following computations: (1) computing
attention scores by applying the softmax function to the inner product between the current query
embedding and all previously generated keys, and (2) multiplying the attention scores with all
previously generated values. To make the attention score calculations in step (1) more memory
efficient, we quantize the keys in the cache. We introduce a quantization scheme for key embeddings,
named QJL, leveraging randomized sketching techniques. Alongside, we develop a high-accuracy
estimator for the inner product of query/key pairs, crucial for mitigating errors amplified by the
softmax operation in attention score calculations.
Firstly, we revisit a fundamental concept in numerical linear algebra: applying a JohnsonLindenstrauss (JL) transform, i.e., a random Gaussian projection, to a pair of vectors and then
computing the inner product of the projected vectors provides an unbiased and low-distortion
estimator for their original inner product [8]. To address the key cache quantization problem, our
aim is to quantize the result after applying the JL transform to a key embedding, ideally to just a
single bit. Surprisingly, we prove that by applying the JL transform to a key embedding and then
quantizing the result to a single bit (the sign bit), while applying the same JL transform to the
query embedding without quantization, we still obtain an unbiased estimator of their inner product
(see Lemma 3.2). Moreover, the distortion of this estimator is small and comparable to that of
the standard JL transform (see Lemma 3.5). In Theorem 3.6, we demonstrate that the proposed
inner product estimator based on QJL achieves a relative distortion of 1 ± ε on the final attention
scores. Notably, the number of required bits for representing quantized keys is independent of the
embedding dimension and scales logarithmically with the context length, using a fixed number of
bits per token.
Thus the QJL sketch combines a JL transform—a random Gaussian projection—with quantization
to the sign bit. An overview of this approach is illustrated in Figure 1. Unlike previous methods, the
QJL sketch can quantize vectors with zero overhead because it does not require grouping the data
and storing quantization constants (zeros and scales) per group. Furthermore, this is a data-oblivious
algorithm that does not rely on specific input, requires no tuning, and can be easily parallelized and
applied in real-time.
The value cache quantization used to make step (2) memory efficient is known to be a straightforward task, and a standard token-wise quantization is very effective and efficient in practice, as
observed in prior work [22, 13]. Hence, we follow the same approach for the value therein.
Furthermore, we analyzed the distribution of outliers in large language models (LLMs). We
observed that while there are no significant outliers in the initial layers, certain fixed key embedding
channels (coordinates) in the deeper layers exhibit considerably larger magnitudes (see Figure 2).
To address this, we identify these outlier channels during the prompt phase and simply apply two
independent copies of our quantizer to the outliers and inliers separately.
The QJL transform and its accompanying inner product estimator are highly efficient and
GPU-friendly algorithms. In particular, we provide a lightweight CUDA kernel for their efficient
computation. We apply QJL and our inner product estimator to compress the KV cache in several
LLMs, including Llama-2 [31] and its fine-tuned models by long sequence [19], under various NLP
tasks. Our results show that quantizing the KV cache to only 3 bits per FPN results in no accuracy
drop compared to the exact model with 16 bits per FPN while reducing cache memory usage by over
fivefold and increasing the generation speed significantly for long contexts. For example, our proposed
quantization shows better F1 scores on long-range question-answering tasks from LongBench [4] (a
collection of long-context datasets) compared to the recent KV cache quantization methods, while
minimizing memory overheads.
3
2 Preliminaries: Token Generation in Attention
Deploying auto-regressive language models for inference involves performing attention decoding in an
online setting, where key and value embeddings from each transformer layer are cached in memory
to remove redundant computations. The model sequentially uses and updates the KV cache to
generate the next token, one at a time.
More precisely, in every phase of token generation, the stream of tokens is represented by a triplet
of vectors called by the query, key, and value embeddings, respectively. Let qi
, ki
, vi ∈ R
d be the
triplet at i-th generation phase and n be the total number of tokens in the stream so far either in
the prompt encoding (prefill) or the generation (decoding) phase. Then, the attention output in
n-th generation phase can be written as
on =
X
i∈[n]
Score(i) · vi
, (1)
where Score ∈ R
n
is the vector of attention scores defined as:
Score := softmax ([⟨qn, k1⟩,⟨qn, k2⟩, . . .⟨qn, kn⟩]). (2)
The output embedding on will be used for computing the next tokens in the stream qn+1, kn+1, vn+1
unless the generation phase terminates. Observe that to compute output on, one needs to store all
previous key and value embeddings {ki
, vi}i∈[n] and keeping them in full precision requires significant
memory for long-context inputs. The time complexity to compete Equation (2) is O(nd) due to
the computation of n inner products. Additionally, the inference speed is also impacted by the KV
cache size, as the KV cache must be loaded from GPU main memory for every token generated,
resulting in low arithmetic intensity and underutilization of GPU cores [27]. In this work, we focus
on compressing the KV cache by quantizing tokens, thereby reducing the memory required to store
each key or value embedding in the cache.
