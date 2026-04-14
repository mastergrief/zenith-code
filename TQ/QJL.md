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
speed bottleneck, especially for long context lengths. Additionally, the GPU must load the entire KV cache from its main memory to shared memory for each token generated, resulting in low
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
works, without any loss in performance

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
3 Quantized Johnson-Lindenstrauss (QJL) Transform
Our goal is to save memory space for storing the KV cache while the inner product between query
and key remains undistorted. To achieve this, we first transform the embedding vectors using a
random projection that preserves the inner products, acting as a preconditioning step, and then
quantize the result. Specifically, we project the input vectors onto a random subspace by applying
the Johnson-Lindenstrauss (JL) transform [15], which amounts to multiplying by a random Gaussian
matrix. The inner product of the resulting vectors after applying this projection provides an unbiased
and low-distortion estimator for the inner product of the original vectors [8]. We introduce a 1-bit
Johnson-Lindenstrauss transform, comprising a JL transformation followed by quantization to a single
sign bit, and demonstrate its ability to offer an unbiased and low-distortion inner product estimator.
We complement our binary quantizer by developing an unbiased estimator for the inner product of
the quantized vector with any arbitrary vector. This inner product estimator is asymmetric, as one
of the vectors is quantized to a single bit while the other remains unquantized, making it well-suited
for the KV cache mechanism. The Quantized Johnson-Lindenstrauss (QJL) transformation, acting
as a 1-bit quantizer, alongside our proposed estimator, is formally defined in the following definition:
Definition 3.1 (QJL and inner product estimator). For any positive integers d, m, let S ∈ R
m×d
be a JL transform matrix, i.e., entries of S are i.i.d. samples from the zero mean and unit variance
Normal distribution. The QJL is a mapping function HS : R
d → {−1, +1}
m defined as:
HS(k) := sign(Sk) for any k ∈ R
d
. (3)
Furthermore, for any pair of vectors k, q ∈ R
d
the estimator for their inner product ⟨q, k⟩ based on
the aforementioned quantizer is defined as:
ProdQJL(q, k) :=
p
π/2
m
· ∥k∥2 · ⟨Sq, HS(k)⟩. (4)
Now, we show that the inner product estimator ProdQJL(q, k), exactly like the inner product
of JL-transformed vectors without quantization to sign bit, is an unbiased estimator. The crucial
point to note is that if we applied QJL to both vectors q and k in Equation (4), we would obtain an
unbiased estimator for the angle between these vectors, as shown in [6]. However, to estimate the
inner product one needs to apply the cosine function on top of the angle estimator, which results
in a biased estimation. Thus, to achieve an unbiased inner product estimator, it is necessary to
asymmetrically apply quantization to the JL transform of only one of the vectors q and k.
Lemma 3.2 (Inner product estimator ProdQJL is unbiased). For any vectors q, k ∈ R
d
the expected
value of the estimator ProdQJL(q, k) defined in Equation (4) is:
E
S
[ProdQJL(q, k)] = ⟨q, k⟩,
where the expectation is over the randomness of the JL matrix S in Definition 3.1.
Proof. Let s1, s2, . . . sm denote the rows of the JL matrix S. Additionally, let us decompose q to its
projection onto the vector k and its orthogonal component, i.e., q
⊥k
:= q −
⟨q,k⟩
∥k∥
2
2
· k. We can write,
ProdQJL(q, k) =
p
π/2
m
X
i∈[m]
∥k∥2 · s
⊤
i q · sign(s
⊤
i k)
=
p
π/2
m
X
i∈[m]
⟨q, k⟩
∥k∥2
· s
⊤
i k · sign(s
⊤
i k) + ∥k∥2 · s
⊤
i q
⊥k
· sign(s
⊤
i k)
=
p
π/2
m
X
i∈[m]
⟨q, k⟩
∥k∥2
· |s
⊤
i k| + ∥k∥2 · s
⊤
i q
⊥k
· sign(s
⊤
i k).
Since si
’s have identical distributions, we have:
E
S
[ProdQJL(q, k)] = p
π/2

⟨q, k⟩
∥k∥2
· E
h
|s
⊤
1 k|
i
+ ∥k∥2 · E
h
s
⊤
1 q
⊥k
· sign(s
⊤
1 k)
i
.
To calculate the above expectation let us define variables x := s
⊤
1 k and y := s
⊤
1 q
⊥k
. Note that x
and y are both zero-mean Gaussian random variables and because ⟨q
⊥k
, k⟩ = 0. By the following
Fact 3.3, x and y are independent.
Fact 3.3. If x ∈ R
d
is a vector of i.i.d. zero-mean normal entries with variance σ
2 and A ∈ R
m×d
is
a matrix, then A · x is a normal random variable with mean zero and covariance matrix σ
2
· AA⊤.
This implies that the second expectation term above is zero because E

s
⊤
1 q
⊥k
· sign(s
⊤
1 k)

=
E[y · sign(x)] = E[y] · E[sign(x)] = 0. Furthermore, x is a Gaussian random variable with mean
zero and variance ∥k∥
2
2
. Therefore, we have
E
S
[ProdQJL(q, k)] = p
π/2 ·
⟨q, k⟩
∥k∥2
· E
x
[|x|] = ⟨q, k⟩.
where the equality comes from the following Fact 3.4:
Fact 3.4 (Moments of Normal Random Variable). If x is a normal random variable with zero mean
and variance σ
2
, then for any integer ℓ, the ℓ-th moment of x is E

|x|
ℓ

= σ
ℓ
· 2
ℓ/2Γ((ℓ + 1)/2)/
√
π.
This completes the proof of Lemma 3.2.
Now we show that the inner product estimator ProdQJL in Definition 3.1, just like the estimators
based on the standard JL transform, has a bounded distortion with high probability.
Lemma 3.5 (Distortion of inner product estimator ProdQJL). For any vectors q, k ∈ R
d
if the
estimator ProdQJL(q, k) is defined as in Equation (4) for QJL with dimension m ≥
4
3
·
1+ε
ε
2 log 2
δ
, then:
Pr
S
[|ProdQJL(q, k) − ⟨q, k⟩| > ε∥q∥2∥k∥2] ≤ δ,
where the probability is over the randomness of the JL matrix S in Definition 3.1.
Proof. First note that, letting s1, s2, . . . sm denote the rows of the JL transform matrix S, we have:
ProdQJL(q, k) = 1
m
X
i∈[m]
p
π/2 · ∥k∥2 · s
⊤
i q · sign(s
⊤
i k).
Since si
’s are i.i.d. the above is indeed the average of m i.i.d. estimators defined as zi
p
:=
π/2 · ∥k∥2 · s
⊤
i
q · sign(s
⊤
i k) for i ∈ [m]. Let us now calculate the ℓ-th moment of zi using Fact 3.4:
E
h
|zi
|
ℓ
i
=
p
π/2 · ∥k∥2
ℓ
· E
h
|s
⊤
i q|
ℓ
i
=
√
π · ∥k∥2∥q∥2
ℓ
·
Γ((ℓ + 1)/2)
√
π
, (5)
where the second equality above follows because s
⊤
i
q is a Gaussian random variable with mean zero
and variance ∥q∥
2
2
along with Fact 3.4. Now we can prove the result by invoking the unbiasedness of
the estimator, Lemma 3.2, along with an appropriate version of Bernstein inequality and using the
moment bounds in Equation (5). More specifically, our moment calculation in Equation (5) implies:
E
h
|zi
|
ℓ
i
= E

|zi
|
2

·
√
π∥k∥2∥q∥2
ℓ−2
·
Γ((ℓ + 1)/2)
Γ(3/2) ≤ E

|zi
|
2

·

2
3
· ∥k∥2∥q∥2
ℓ−2
·
ℓ!
2
Therefore, by invoking a proper version of the Bernstein inequality, for instance Corollary 2.11 from
[5], we have the following:
Pr
S
[|ProdQJL(q, k) − ⟨q, k⟩| > t] ≤ 2 exp 
3
4
·
mt2
∥k∥
2
2
∥q∥
2
2 + ∥k∥2∥q∥2 · t

.
If we set t = ε∥q∥2∥k∥2 the above simplifies to:
Pr
S
[|ProdQJL(q, k) − ⟨q, k⟩| > ε∥q∥2∥k∥2] ≤ 2 exp 
3
4
·
mε2
1 + ε

.
Therefore if m ≥
4
3
·
1+ε
ε
2 log 2
δ
the error bound follows. This completes the proof of Lemma 3.5.
Note that the distortion bound in Lemma 3.5 has remarkably small constants, even smaller than
those of the original unquintized JL transform. This indicates that quantizing one of the vectors to
just a single sign bit does not result in any loss of accuracy. We use these properties of QJL and our
inner product estimator to prove the final approximation bound on our KV cache quantizer.

Algorithm 1 QJL Key Cache Quantizer
Input: Stream of key tokens k1, k2, . . . ∈ R
d
, integer m
1: Draw a random sketch S ∈ R
m×d with i.i.d. entries Si,j ∼ N (0, 1) as per Definition 3.1
2: repeat
3: Compute k˜
i ← sign (Ski) and νi ← ∥ki∥2
4: store the quantized vector ˜ki and the key norm νi
in the cache
5: until token stream ends
Procedure EstimateScores(qn)
6: Compute inner product estimators qKg(j) ←
√
π/2
m
· νi
· ⟨Sqn, k˜
j ⟩ for every j ∈ [n]
7: Score ^ ← softmax 
qKg

return Score ^
3.1 Key Cache Quantization via QJL
The key cache is used in the computation of attention scores as shown in Equation (2). To calculate
these scores, we need to compute the inner products of the current query embedding with all key
embeddings in the cache. We design a quantization scheme that allows for a low-distortion estimate
of the inner products between an arbitrary query and all keys in the cache. In this section, we develop
a practical algorithm with provable guarantees based on QJL and the inner product estimator defined
in Definition 3.1.
The quantization scheme presented in Algorithm 1 applies QJL, defined in Definition 3.1, to each
key embedding, mapping them to binary vectors and storing the results in the key cache. We show
in the following theorem that the attention scores calculated by Algorithm 1 have very small (1 ± ε)
relative distortion with high probability:
Theorem 3.6 (Distortion bound on QJL key cache quantizer). For any sequence of key tokens
k1, . . . kn ∈ Rd and any integer m, Algorithm 1 stores binary vectors k˜
1, . . . k˜
n ∈ {−1, +1}
m along
with scalar values ν1, . . . νn in the cache. If the key embeddings have bounded norm maxi∈[n] ∥ki∥2 ≤ r
and m ≥ 2r
2
ε
−2
log n, then for any query embedding qn ∈ Rd with bounded norm ∥qn∥2 ≤ r the
output of the procedure EstimateScores(qn) satisfies the following with probability 1 −
1
poly(n)
sinultaneously for all i ∈ [n]:



Score ^(i) − Score(i)


 ≤ 3ε · Score(i),
where Score is the vector of attention scores defined in Equation (2).
Proof. The proof is by invoking Lemma 3.5 and a union bound. For every j ∈ [n] the estimator
qKg(j) computed in line 6 of Algorithm 1 is in fact equal to the inner product estimator qKg(j) =
ProdQJL(qn, kj ) as defined in Equation (4). Thus by Lemma 3.5 we have the following with probability
at least 1 −
1
n3/(2+2ε)
:


qKg(j) − ⟨qn, kj ⟩


 ≤
ε
r
2
· ∥qn∥2∥kj∥2 ≤ ε,
where the second inequality follows from the preconditions of the theorem regarding the boundedness
of the norms of the query and key embeddings. By union bound, the above inequality holds
simultaneously for all j ∈ [n] with high probability in n. Thus after applying the softmax function
in line 7 of Algorithm 1 we get that with high probability in n:
Score ^(i) ∈ e
±2ε
· Score(i) ∈ (1 ± 3ε) · Score(i).

This completes the proof of Theorem 3.6.
This theorem shows that if the query and key embeddings have constant norms, as is common
in practical scenarios, we can quantize each key embedding such that only m ≈ ε
−2
log n bits are
needed to store each key token. This is independent of the embedding dimension of the tokens and
scales only logarithmically with the sequence length.
3.2 Value Cache Quantization
We quantize the value cache using a standard quantization method, i.e., normalizing each token’s
entries and then rounding each entry to a few-bit integer representation. This approach aligns with
prior work, which has shown that standard token-wise quantization is highly effective for the value
cache and results in a minimal accuracy drop [22, 13].
4 Experiments
In this section, we validate the empirical performance of our algorithm. All experiments are conducted
under a single A100 GPU with 80GB memory. We implement two main CUDA kernels for our core
primitives: one for quantizing embedding vectors using various floating point data types such as
bfloat16, FP16, and FP32, and the other for computing the inner product of an arbitrary embedding
vector with all quantized vectors in the cache. The algorithm’s wrapper is implemented in PyTorch,
handling all the housekeeping tasks. We plan to complete implementation in the CUDA for future
work, which will further accelerate our algorithm.
4.1 Practical Consideration
Outliers. As reported in recent works e.g., KIVI [22], KVQuant [13], key embeddings typically
contain outliers exhibiting a distinct pattern. Specifically, certain coordinates of key embeddings
display relatively large magnitudes. To further investigate these observations, we analyze the
distribution of the magnitudes of key embedding coordinates across different layers. Firstly, we
observe that there are no significant outliers in the initial attention layers. However, in the deeper
layers, certain fixed coordinates of key embeddings consistently exhibit large magnitudes, and this
pattern persists within these channels across all tokens. The distribution of outliers across different
layers for the Llama-2 model is plotted in Figure 2. It is evident that in the initial layers, outliers are
rare, but as we approach the final layers, their frequency and impact increase significantly. Secondly,
the outliers show a persistent pattern in specific fixed coordinates of the key embeddings. This
observation aligns with previous findings that certain fixed embedding coordinates exhibit larger
outliers [9, 20, 22, 13].
As demonstrated in Theorem 3.6, the distortion on the attention scores is directly proportional
to the norms of the embeddings. Therefore, capturing these outlier coordinates is essential, as
their large magnitudes contribute significantly to the norms of key embeddings. By identifying and
isolating these outlier channels, we can reduce the norm of the key embeddings and, consequently,
significantly decrease the final distortion. Next, we quantize the outliers using an independent
instance of our QJL quantizer but with a lower compression rate, utilizing more bits to accurately
represent each outlier coordinate.
Orthogonalized JL transform. We observed that orthogonalizing the rows of the JL matrix S
in Definition 3.1 almost always improves the performance of our QJL quantizer. This finding aligns
with previous work on various applications of the JL transform, such as random Fourier features [35]
and locality sensitive hashing [14]. Consequently, in our implementation and all experiments, we
first generate a random JL matrix S with i.i.d. Gaussian entries and then orthogonalize its rows
using QR decomposition. We then use this orthogonalized matrix in our QJL quantizer, as described
in Algorithm 1.
4.2 End-to-end text generation
Next we benchmark our method on LongBench [4], a benchmark of long-range context on various
tasks. We choose the base model as longchat-7b-v1.5-32k [19] (fine-tuned Llama-2 with 7B parameter
with 16,384 context length) and apply following quantization methods to this model; KIVI [22],
KVQuant [36] and our proposed quantization via QJL. Each floating-point number (FPN) in the
base model is represented by 16 bits, and we choose proper hyper-parameters of KIVI and QJL
so that their bits per FPN become 3. For KVQuant, we follow the default setting which holds its
bits per FPN as 4.3. To validate the quality of those quantized models, we benchmark them on 6
question-answer datasets from LongBench [4], and we set the maximum sequence length to 31,500.
We follow the same approach of prompting and evaluating to evaluate the prediction of the model
from the original repository. Table 1 summarizes the results. Our proposed QJL achieves the highest
F1 score within the quantization methods for NarrativeQA, Qasper and 2WikiMultiQA.
Although KVQuant performs better than other methods for MultiQA-en dataset, it requires a
huge amount of preprocessing which leads to slow runtime. To validate this, we additionally report
runtime of prompt encoding, KV cache quantization, and decoding (token generation) in a single
attention layer. Figure 3 shows the wall-clock time to encode a prompt and quantize the KV cache,
generate 128 tokens for llama2 model, and generate 64 tokens for llama3 model using different
quantization methods in a single attention layer of these models. Note that QJL is the only method
that can quantize Llama3, as our kernels support grouped query attention and BF16 data type. we
observe the same speed for Llama3 as the exact method for generation. The input sequence lengths
vary between 1k to 128k. As shown in Figure 3, KVQuant runs slower than other methods during
both prompt encoding and decoding phases. On the other hand, both KIVI and our QJL with 3
9
Methods Bits
Datasets from LongBench [4]
NarrativeQA Qasper MultiQA-en MultifQA-zh HotpotQA 2WikiMultiQA
FP16 (baseline) 16 20.79 29.42 42.83 34.33 33.05 24.14
KIVI [22] 3 20.96 29.01 40.93 34.75 32.79 23.01
KVQuant [13] 4.3 20.14 28.77 44.22 34.44 34.06 23.05
QJL (ours) 3 21.83 29.44 41.52 34.42 35.62 23.60
Table 1: Evaluation (F1 scores) of various quantization methods on long-context question-answering
datasets from LongBench [4]. We set bits per floating-point number (FPN) to 3. Bold indicates the
highest scores within quantization methods.
Models Methods Bits
Datasets from LM-eval [12]
Lambada-OpenAI HellaSwag PIQA MathQA MMLU
Llama-2-7B
FP16 (baseline) 16 73.90 57.18 78.07 28.11 41.85
KIVI [22] 3 73.88 57.13 78.07 28.11 41.81
QJL (ours) 3 73.88 57.14 78.07 28.17 41.78
Llama-3-8B
BF16 (baseline) 16 75.59 60.17 79.65 40.64 62.09
QJL (ours) 3 75.61 60.13 79.87 40.60 62.12
Table 2: Evaluation (accuracy) of various quantization methods on regular length datasets from
LM-eval [12]. These comparisons are not typically based on long-context length; however, as evident,
even in these cases, our QJL with 3 bits per FPN performs comparably to the baseline with 16 bits
per FPN.
bits per FPN show marginal runtime overhead compared to the exact baseline during prompting but
reduce KV cache memory usage by at least a factor of 5.
We additionally test our method on datasets Lambada-OpenAI, HellaSwag, PIQA, MathQA, and
MMLU, which have shorter sequence lengths. We benchmark our method using LM-eval [12] framework
to ensure a thorough evaluation across various metrics. We evaluate quantization methods with
accuracy across Llama-2-7B [31] and Llama-3-8B [23] models. Note that KIVI only supports a
half-precision floating point, whereas our method can be used for any precision format type. This
makes it unable to run KIVI on the Llama-3 model.
As a results, QJL can significantly reduce memory usage by utilizing only 3 bits per FPN,
compared to the 16 bits per FPN in the baseline, achieving around an 81% reduction in memory.
We observe that this efficiency does not compromise performance significantly. Across all datasets,
our method’s accuracy is generally comparable to the baseline, with slight variations. In Table 2,
our QJL on the Llama-3-8B performs on average about slightly better than the baseline across all
datasets.
