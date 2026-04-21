# DeltaNet — Empirics and Related Work

Numbers and context. Synthetic benchmarks, language modeling at 340M /
1.3B / 3B scale, ablations, throughput, limitations, and how DeltaNet
sits in the recent linear-RNN / fast-weight / online-learning
literature. Algorithm in
[`02_Chunkwise_Parallel_Algorithm.md`](02_Chunkwise_Parallel_Algorithm.md);
conceptual overview in [`01_DeltaNet_Overview.md`](01_DeltaNet_Overview.md).
See [`00_INDEX.md`](00_INDEX.md) for the full doc set.

---

## 1. Synthetic benchmarks

### MQAR (Multi-Query Associative Recall, Arora 2024)

Tests in-context recall of multiple key-value pairs under repeated
querying. 64 key-value pairs, sequence length 512.

| Model | d=64 | d=128 | d=256 | d=512 |
|---|---:|---:|---:|---:|
| Transformer | high | high | 100% | 100% |
| Mamba (w. conv) | ~20% | ~60% | ~95% | 100% |
| GLA | low | mid | high | ~95% |
| RetNet | low | mid | mid | high |
| RWKV-4 | low | low | mid | high |
| Hyena | low | low | mid | high |
| **DeltaNet (no conv, 2 heads)** | **high** | **~100%** | **100%** | **100%** |

DeltaNet hits perfect recall at the hardest setting, and uniquely
outperforms Mamba in the low-`d` regime **without using convolutions**.

### MAD (Mechanistic Architecture Design, Poli 2024)

Six subtasks probing token-manipulation capabilities:

| Model | Compress | Fuzzy Recall | In-Context | Memorize | Noisy Recall | Selective Copy | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|
| Transformer | 51.6 | 29.8 | 94.1 | 85.2 | 86.8 | 99.6 | 74.5 |
| Hyena | 45.2 | 7.9 | 81.7 | 89.5 | 78.8 | 93.1 | 66.0 |
| Multihead Hyena | 44.8 | 14.4 | 99.0 | 89.4 | 98.6 | 93.0 | 73.2 |
| Mamba | 52.7 | 6.7 | 90.4 | 89.5 | 90.1 | 86.3 | 69.3 |
| GLA | 38.8 | 6.9 | 80.8 | 63.3 | 81.6 | 88.6 | 60.0 |
| **DeltaNet** | 42.2 | **35.7** | **100** | 52.8 | **100** | **100** | **71.8** |

DeltaNet dominates on Fuzzy Recall (35.7 vs next-best 29.8), In-Context,
Noisy Recall, and Selective Copy — every recall-shaped task. It
underperforms on Memorize (52.8 — verbatim sequence memorization),
which is the one MAD subtask that's really about capacity rather than
retrieval. RegBench results follow the same pattern (appendix).

---

## 2. Language modeling

All models trained on the same SlimPajama subset with the Mistral
tokenizer. 340M trained for 15B tokens, 1.3B for 100B tokens. Evaluated
on Wikitext perplexity, LAMBADA, six zero-shot common-sense tasks
(PIQA, HellaSwag, WinoGrande, ARC-e, ARC-c), and three recall-heavy
real-world tasks (SWDE semi-structured extraction, SQuAD reading
comprehension, FDA key-value lookup).

### 340M / 15B

| Model | Wiki↓ | LMB ppl↓ | LMB acc↑ | PIQA | Hella | Wino | ARC-e | ARC-c | Avg | SWDE | SQuAD | FDA | State |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Transformer++ | 28.39 | 42.69 | 31.0 | 63.3 | 34.0 | 50.4 | 44.5 | 24.2 | 41.2 | 42.2 | 22.1 | 21.4 | N/A |
| RetNet (no conv) | 32.33 | 49.19 | 28.6 | 63.5 | 33.5 | 52.5 | 44.5 | 23.4 | 41.0 | 13.3 | 27.6 | 2.9 | 512× |
| Mamba (w. conv) | 28.39 | 39.66 | 30.6 | 65.0 | 35.4 | 50.1 | 46.3 | 23.6 | 41.8 | 12.4 | 23.0 | 2.1 | 64× |
| GLA (no conv) | 28.65 | 43.35 | 30.3 | 64.8 | 34.5 | 51.4 | 45.1 | 22.7 | 41.5 | 18.6 | 27.2 | 8.1 | 128× |
| GLA (w. conv) | 29.47 | 45.53 | 31.3 | 65.1 | 33.8 | 51.6 | 44.4 | 24.6 | 41.8 | 24.0 | 24.7 | 7.3 | 128× |
| DeltaNet (no conv) | 29.08 | 50.87 | 30.0 | 63.6 | 33.6 | 51.7 | 46.0 | 23.0 | 41.3 | 24.6 | 26.9 | 4.5 | 128× |
| **DeltaNet (w. conv)** | **28.24** | **37.37** | **32.1** | 64.8 | 34.3 | 52.2 | 45.8 | 23.5 | **42.1** | 26.4 | 28.9 | 12.8 | 128× |
| + Sliding Attn | 27.06 | 38.17 | 33.4 | 64.0 | 35.3 | 50.9 | 45.9 | 23.2 | 42.1 | 39.3 | 32.5 | 18.8 | N/A |
| + Global Attn (2 layers) | 27.51 | 35.04 | 33.5 | 64.0 | 34.5 | 51.7 | 46.0 | 23.3 | 42.1 | **42.9** | **32.1** | **23.1** | N/A |

DeltaNet with conv beats all pure-recurrent baselines on every metric.
Hybrid variants beat Transformer++ on the LM metrics and push
recall-intensive numbers further up (DeltaNet + Global Attn hits SWDE
42.9 vs Transformer++ 42.2 — parity on recall with faster inference).

### 1.3B / 100B

| Model | Wiki↓ | LMB ppl↓ | LMB acc↑ | PIQA | Hella | Wino | ARC-e | ARC-c | Avg | SWDE | SQuAD | FDA | State |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Transformer++ | 16.85 | 13.44 | 48.9 | 70.8 | 49.6 | 53.6 | 56.0 | 26.5 | 50.9 | 66.6 | 31.5 | 27.4 | N/A |
| RetNet (no conv) | 18.64 | 17.27 | 43.3 | 70.0 | 47.3 | 52.5 | 54.8 | 25.6 | 48.9 | 42.8 | 34.7 | 14.3 | 512× |
| Mamba (w. conv) | 17.06 | 13.89 | 46.2 | 72.2 | 40.1 | 54.1 | 59.0 | 28.2 | 50.0 | 41.4 | 35.2 | 6.2 | 64× |
| GLA (no conv) | 17.22 | 14.47 | 46.9 | 71.8 | 49.8 | 53.9 | 57.2 | 26.6 | 51.0 | 50.6 | 42.6 | 19.9 | 256× |
| GLA (w. conv) | 17.25 | 14.92 | 46.2 | 70.6 | 49.9 | 53.0 | 55.3 | 27.0 | 50.4 | 52.4 | 37.4 | 22.3 | 256× |
| **DeltaNet (w. conv)** | **16.87** | **12.21** | **48.9** | 71.2 | 50.2 | 53.6 | 57.2 | 28.3 | **51.6** | 49.5 | 37.4 | 17.2 | 128× |
| **+ Sliding Attn** | **16.56** | **11.74** | 49.2 | 71.8 | **51.1** | 52.8 | 58.9 | **28.8** | **52.1** | 53.3 | **43.3** | 22.3 | N/A |
| + Global Attn (2 layers) | 16.55 | 12.40 | 48.8 | 70.8 | 50.7 | 54.2 | 58.4 | 28.1 | 51.8 | **71.0** | 43.0 | **29.8** | N/A |

Pure DeltaNet + conv matches Transformer++ on wiki ppl (16.87 vs
16.85) and beats it on zero-shot average. Hybrids dominate. **DeltaNet
+ Global Attn (2 layers)** hits SWDE **71.0** — better than the full
Transformer++ baseline (66.6) — at `N/2` the softmax-attention layer
count.

One caveat: at 1.3B, pure DeltaNet **underperforms GLA** on
recall-intensive tasks. Reason: GLA's elementwise recurrence scales
state size more easily (128× → 256× expansion), and recall is
state-size bound. The hybrid variants close this gap.

### 3B / 1T tokens (compared across model families)

| Model | ARC | HellaSwag | OBQA | PIQA | WinoGrande | MMLU | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|
| Llama-3.2-3B | 59.1 | 73.6 | 43.4 | 77.5 | 69.2 | 54.1 | 62.8 |
| PowerLM-3B | 60.5 | 74.6 | 43.6 | 79.9 | 70.0 | 45.0 | 62.3 |
| **DeltaNet-3B** | 60.4 | 72.8 | 41.0 | 78.5 | 65.7 | 40.7 | 59.8 |
| RecurrentGemma-2B | 57.0 | 71.1 | 42.0 | 78.2 | 67.6 | 31.8 | 57.9 |
| RWKV-6-3B | 49.5 | 68.6 | 40.6 | 76.8 | 65.4 | 28.4 | 54.9 |
| Mamba-2.7B | 50.3 | 65.3 | 39.4 | 75.8 | 63.1 | 26.1 | 53.3 |

DeltaNet-3B slightly underperforms the Transformer-architecture baselines
(Llama-3.2, PowerLM) but beats every pure-recurrent competitor at
comparable scale. Token counts aren't perfectly matched, so the
comparison isn't apples-to-apples.

---

## 3. Throughput

Training throughput of 1.3B models on a single H100, under various
(length × batch) pairs that multiply to 16K tokens:

- **Transformer++**: fastest at short sequences, degrades quickly past
  2K × 8.
- **GLA**: fastest linear-time model. Flat-ish curve.
- **DeltaNet**: close to GLA throughout, slightly behind. Overtakes
  Transformer++ around 4K sequence length.
- **Mamba**: clearly behind GLA and DeltaNet, especially at long
  sequences — the SSM-style kernel is less hardware-efficient than
  FLA-based models at this scale.

All linear-time models beat Transformer++ for training lengths ≥ 8K.

---

## 4. Ablations (340M)

Feature map × normalization:

| Config | Wiki↓ | LMB ppl↓ | LMB acc↑ | Avg zero-shot | SWDE | SQuAD | FDA |
|---|---:|---:|---:|---:|---:|---:|---:|
| L1-norm & 1+ELU (Schlag 2021) | 31.12 | 55.96 | 26.3 | 40.1 | 14.5 | 23.9 | 6.2 |
| L2-norm & 1+ELU | 28.03 | 37.62 | 32.2 | 42.1 | 23.8 | 28.6 | 13.1 |
| L2-norm & ReLU | 28.75 | 43.53 | 30.2 | 40.9 | 27.2 | 26.7 | 9.0 |
| **L2-norm & SiLU (final)** | **28.24** | **37.37** | **32.1** | **42.1** | **26.4** | **28.9** | **12.8** |

Switching L1 → L2 alone is worth +2pp zero-shot and ~3× on FDA — the
biggest single ablation move. SiLU vs 1+ELU is neutral on LM metrics
but cleaner on recall.

---

## 5. Limitations

### Training speed behind GLA

DeltaNet's `I − β_t k_t k_t^T` transition models state-to-state
interactions; GLA's `Diag(α_t)` is purely elementwise. GLA can tile the
head dimension arbitrarily because intra-state operations are
independent across channels; DeltaNet has to marginalize across head
dimension inside the kernel (like softmax attention does), which caps
the practical head dimension and therefore the recurrent state size.
**Open:** block-diagonal Householder with GPU-SRAM-sized blocks (~128)
would restore per-block independence while keeping a large overall head
dim — called out as future work.

### Length generalization

DeltaNet lacks explicit decay and does not extrapolate beyond the
training length as cleanly as GLA, RetNet, or Mamba. **Fixed in Gated
DeltaNet** (Yang 2024b) by adding a scalar decay `α_t` on top of the
Householder update: `M_t = α_t (I − β_t k_t k_t^T)`.

### Expressiveness upper bound

Irie 2022 proved theoretical limits on delta-rule expressiveness.
Recurrent DeltaNet (Irie 2021), Modern Self-Referential Weight Matrix
(Irie 2022), and the mesa-layer (Van Oswald 2023) lift those limits but
are not parallelizable across sequence length — illustrating a
fundamental parallelism-vs-expressiveness trade-off (Merrill 2024). TTT
(Sun 2024) and Titans (Behrouz 2024) offer a middle ground:
hybrid cross-chunk nonlinear, intra-chunk linear.

---

## 6. Related work

### Linear attention as iterated Hopfield networks

Linear transformers are a type of iterated Hopfield network (Millidge
2022). Classical Hopfield nets have limited memory capacity under a
purely Hebbian rule (McEliece 1987). Recent modern Hopfield work uses
higher-order polynomials (Demircigil 2017) or exponential kernels
(Ramsauer 2020) to enhance capacity — related to linear attention with
polynomial kernels (Keller 2021, Arora 2024). **The delta rule gets a
better recall-memory tradeoff frontier than any purely Hebbian
alternative** (Arora 2024) and has seen adoption in real-world retrieval
(Mao 2024, Schmidhuber 2024).

### Online learning / meta-learning

The delta rule = one SGD step on `½‖Sk − v‖²`. Connects to meta-learning
via gradient descent (Irie 2022) and to Test-Time Training (TTT, Sun
2024), Longhorn (Liu 2024), Titans (Behrouz 2024). Titans explicitly
add momentum and weight decay to the delta-rule update; this can be
seen as replacing the one-step SGD with one step of AdamW.

### Structured-matrix viewpoint

Dao-Gu 2024's state-space duality (SSD) casts SSMs as masked structured
matrix multiplications. DPLR transition `M_t = D − a_t b_t^T`
generalizes DeltaNet's special case. Gu 2021's S4 explored
data-**in**dependent DPLR; DeltaNet is the data-**dependent** `D = I`
subcase. Further generalizations (e.g. multiple low-rank terms) are
future work.

### Cross-ref to this project's substrate

Our `calm/llm_computer/fast_weights.py` implements asymmetric Hebbian
fast weights (Schlag 1991 style), `W_fast_t = λ W_fast_{t-1}
+ η (v_t ⊗ k_t) / d_model`. Session-30 Round 1 got 99.1% on n=3
associative recall at `d_head=2` (novel result); Rounds 3 and 4
diagnosed an n=10 structural ceiling from cross-key leakage. Round 4's
attempted delta-rule + write-gate variant gave a clean null
(`.claude/rules/architecture.md` §"Substrate Extensions"). DeltaNet is
the known-working delta-rule mechanism — with the parallelism recipe
our ad-hoc Round 4 implementation lacked. The chunkwise algorithm in
[`02`](02_Chunkwise_Parallel_Algorithm.md) is the direct port target
for Round 5 if we reopen the associative-recall line.

---

## 7. Takeaway

DeltaNet was always the theoretically cleaner linear transformer —
better memory-capacity frontier than the additive variant, naturally
interpretable as "retrieve old value, interpolate, rewrite." It wasn't
usable at scale because the recurrence resisted parallelization. This
paper's contribution is the **WY-representation / pseudo-value
reparameterization that recovers linear-attention-style chunkwise
training**, and the empirical demonstration that DeltaNet + short
convolution + hybrid attention layers is a competitive architecture
at 1.3B / 100B. The remaining scale gaps (head dimension in the
kernel, length generalization) have known fixes (block-diagonal
Householder, Gated DeltaNet); the broader expressiveness ceiling is the
open research question.
