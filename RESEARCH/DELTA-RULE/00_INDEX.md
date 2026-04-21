# DeltaNet — Index

"Parallelizing Linear Transformers with the Delta Rule over Sequence
Length." Yang, Wang, Zhang, Shen, Kim (MIT / Soochow / MIT-IBM Watson AI
Lab). NeurIPS 2024. arXiv:2406.06484.

A hardware-efficient chunkwise training algorithm for DeltaNet — a linear
transformer that replaces the additive memory update `S_t = S_{t-1} + v_t
k_t^T` with the Widrow-Hoff delta rule, enabling better associative
recall at the cost of a structured (non-elementwise) recurrence. The
algorithm turns the sequential update into a sequence-parallel one via
the Householder / WY representation trick, which is what made it possible
to scale DeltaNet from synthetic-only experiments to 1.3B params / 100B
tokens, where it beats Mamba and GLA on language modeling and — with
sliding-window or global attention hybrids — beats Transformer++.

---

## Concept map

- **[`01_DeltaNet_Overview.md`](01_DeltaNet_Overview.md)** — What & why.
  Additive-update collision problem, delta-rule fix, retrieve-then-
  interpolate and online-SGD intuitions, LLaMA++ architecture, hybrid
  variants, placement in the linear-RNN zoo. No heavy math.
- **[`02_Chunkwise_Parallel_Algorithm.md`](02_Chunkwise_Parallel_Algorithm.md)** —
  The core contribution. Generalized-Householder recurrence, pseudo-value
  reparameterization `u_i = β_i(v_i - v_i^old)`, chunkwise form via WY
  representation, UT transform to make intra-chunk ops matmul-rich,
  complexity, throughput numbers.
- **[`03_Empirics_and_Related_Work.md`](03_Empirics_and_Related_Work.md)** —
  Results + context. MQAR / MAD / RegBench synthetics, 340M/15B and
  1.3B/100B language-modeling tables, recall-intensive benchmarks
  (SWDE/SQuAD/FDA), 3B scaling, ablations, limitations, related work
  (Hopfield networks, fast weights, TTT, Longhorn, Titans).

---

## Why this is in `RESEARCH/`

This project's substrate already has a runtime Hebbian fast-weight
mechanism (`calm/llm_computer/fast_weights.py`, Schlag-style asymmetric
outer-product writes). Session-30 Round 1 got 99.1% on 3-pair associative
recall at `d_head=2` — a novel empirical result — but Rounds 3 and 4
diagnosed a **structural n=10 ceiling**: cross-key leakage, not capacity.
Round 4 tried delta-rule + write-gate as a fix and got a clean null
(10.5-12.2% across variants). DeltaNet is the delta-rule successor with
the parallelism recipe our Round 4 variants lacked, and its
chunkwise-Householder algorithm is the natural next thing to try. Read
[`02`](02_Chunkwise_Parallel_Algorithm.md) first if that's the reason
you're here. The full substrate-extension context lives in
`.claude/rules/architecture.md` §"Substrate Extensions".

---

Source: Yang et al., *Parallelizing Linear Transformers with the Delta
Rule over Sequence Length*, NeurIPS 2024.
