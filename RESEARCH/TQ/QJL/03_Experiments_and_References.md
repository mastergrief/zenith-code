# QJL — Part 3: Experiments and References
_Part 3 of 3. See [`00_INDEX.md`](00_INDEX.md) for full paper TOC._

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
8
0
20
40
60
80
100
120 Channels (Sorted)
0
200
400
600
800
1000
1200
1400
Tokens
0.0
0.5
1.0
1.5
2.0
2.5
Magnitude
0.2
0.4
0.6
0.8
1.0
(a) Layer 0, Head 0
0
20
40
60
80
100
120 Channels (Sorted)
0
200
400
600
800
1000
1200
1400
Tokens
0
2
4
6
8
10
12
14
Magnitude
1
2
3
4
5
6
7
(b) Layer 15, Head 0
0
20
40
60
80
100
120 Channels (Sorted)
0
200
400
600
800
1000
1200
1400
Tokens
0
2
4
6
8
10
12
14
Magnitude
1
2
3
4
5
6
7
8
(c) Layer 31, Head 0
Figure 2: The magnitude of key cache entries for different layers of the Llama-2 model, based
on an example prompt, reveals notable patterns. The coordinates of embeddings (channels) are
sorted by their average magnitude over tokens. In the initial layers, no significant outlier patterns
are observed. However, in the deeper layers, a few channels (approximately four) exhibit visibly
larger magnitudes, indicating the presence of significant outliers. This observation highlights the
importance of addressing these outliers to improve quantization accuracy and reduce distortion in
the key cache.
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
References
[1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni
Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4
technical report. arXiv preprint arXiv:2303.08774, 2023.
10
2k 8k 32k 64k
token length
0.0
0.5
1.0
1.5
encode + quantize (ms)
FP16
QJL (ours)
KVQuant
KIVI
(a) Prompt encoding (Llama2)
2k 8k 32k 64k
token length
0
2
4
6
decode (ms)
FP16
QJL (ours)
KVQuant
KIVI
(b) Token generation (Llama2)
1k 2k 8k 32k 64k
token length
0.0
0.5
1.0
1.5
total (ms)
BF16
QJL (ours)
(c) Encode and generate (Llama3)
Figure 3: Wall-clock time (ms) to encode a prompt and quantize the KV cache (left), generate 128
tokens for llama2 model (middle), and generate 64 tokens for llama3 model (right) using different
quantization methods in a single attention layer model. The input sequence length varies from 1k to
64k. Both KIVI and QJL (ours) with 3 bits per FPN show faster decoding time than the baseline.
However, KVQuant is significantly slower during both quantizing and decoding phases. QJL is the
only method that can quantize Llama3, as our kernels support grouped query attention and BF16
data type. We observe the same speed for Llama3 as the exact method for generation. Note that
our memory usage is at least 5-fold less than the exact method and can support all data types.