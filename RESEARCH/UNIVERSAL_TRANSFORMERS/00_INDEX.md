# UT-Family — Unified Synthesis

A unified synthesis of three Universal-Transformer-family papers
(`UT.md` / `OPEN_MYTHOS.md` / `EUTM.md`) refactored into a single
4-file spec, framed against this repo's substrate work (D5 recurrent
substrate, DT, CALM, the L24 deep-diffuse gap).

## What this is

A **substrate-relevance-first synthesis** of three otherwise-loose
external research artifacts that all sit in the same family
(parameter-tied depth recurrence + adaptive halting + learned
memory). The three primary sources stay in this directory as raw
citations; the four unified files (this one + `01` / `02` / `03`)
are the framing artifact for the next experimental round on the
substrate side.

Not a build plan. Not a paper for external publication. A durable
internal synthesis so future sessions can act on the common
mechanisms (loop-index embedding, ACT halting, Parcae stability,
NAMM pruning, RDT-as-Tier-3) without re-reading 65KB of
disconnected source.

## Files

| File | Role | Content |
|---|---|---|
| `00_INDEX.md` (this file) | Overview | Thesis, files map, the three insight families, decision the user needs |
| `01_ARCHITECTURE.md` | Design spec | Unified architecture (Prelude/Recurrent/Coda + ACT + loop-index + MoE + MLA + NAMM), invariants, what's net-new vs current substrate |
| `02_IMPLEMENTATION.md` | Mechanics | Parcae LTI stability math, NAMM 3-step pipeline, ACT computation, scaling laws, evolutionary training, OpenMythos config table |
| `03_TESTING.md` | Substrate-relevance gates | 5 ranked candidates (A loop-index / B ACT / C Parcae-on-injection / D NAMM-skip / E RDT card for L24), failure-surface gates, falsifier table |
| `EUTM.md` | Source (preserved) | Sakana NAMM blog, Dec 2024 |
| `OPEN_MYTHOS.md` | Source (preserved) | OpenMythos RDT reconstruction, 2026 |
| `UT.md` | Source (preserved) | Original UT paper, Dehghani et al. 2018 |

## Thesis (one line)

> **Depth recurrence (UT/RDT) gives compositional reasoning at
> fixed parameter count; ACT and Parcae make it trainable; NAMM
> bolts on as an orthogonal attention-pruning layer. The substrate
> already implements the core mechanism (D5); net-new is per-iter
> differentiation, learned halting, injection-stability, and a
> Tier-3 card that targets the documented L24 deep-diffuse gap.**

## The three insight families

**UT (2018, foundation).** Depth recurrence as alternative to width
— same self-attention + transition function applied T times with
shared weights. Per-position **Adaptive Computation Time (ACT)**
halting: harder symbols get more refinement steps. Under reasonable
assumptions, **Turing-complete** (vs vanilla Transformer's fixed
depth). Empirical wins: bAbI SOTA, LAMBADA SOTA, Copy/Reverse/Add
length-extrapolation 40→400, +0.9 BLEU on WMT14 En-De.

**OpenMythos / Parcae (2026, modern scaling).** Same RDT core,
scaled with **MoE FFN** (fine-grained routed + always-on shared
experts), **MLA** (DeepSeek-V2 compressed-KV) or GQA attention,
and the load-bearing **Parcae stability fix**: parameterize
injection matrix `A := Diag(-exp(log_A))` so ρ(A) < 1 by
construction. Establishes scaling laws — more loops + fewer tokens
beats more tokens + fewer loops at fixed FLOP. 770M looped ≈ 1.3B
fixed-depth quality. Open conjectures: **RoPE-style loop-index
embedding** to differentiate same-weight iterations, **per-loop
LoRA** (Bae 2024), **continuous depth-wise batching** for 2-3×
inference throughput.

**NAMM (2024, orthogonal memory).** **Neural Attention Memory
Models** — small classifiers trained via **evolution** (binary
keep/forget is non-differentiable) to decide per-token retention.
Pipeline: attention values → spectrogram → EMA compression →
classifier score. Critical property: conditioning **only on
attention matrices** ⇒ **universal transfer** across architectures
and modalities (Llama-3 → Llama-70B → Llava-Video → Decision
Transformer with zero retraining). Beats hand-designed memory
pruning (H₂O, L₂) on LongBench / InfiniteBench / ChouBun while
also reducing context size.

## What this means for the substrate

Decomposition table — what's already in the repo vs net-new:

| Mechanism | Substrate has | Net-new from these papers |
|---|---|---|
| Param-tied depth recurrence | **Yes** — D5 (`recurrent_substrate.py`, `n_iterations` kwarg) | — |
| Sequence-axis fast-weight recurrence | **Yes** — DT (`copy_augmented_delta.py`, Householder rule) | — (different axis from RDT) |
| Multi-step CALC composition | **Yes** — `MultiStepReasoningFacade` (parse → safe_eval → step-through bias) | — |
| Per-iteration differentiation | No | Loop-index RoPE embedding (cheap, additive) |
| Learned halting | No (fixed `n_iterations` per mode token) | ACT (UT, OpenMythos) |
| Injection stability constraint | N/A — D5 has no explicit injection term | Parcae LTI fix `A := Diag(-exp(log_A))`, ρ(A) < 1 |
| Attention-matrix learned pruning | No (current: tq4 KV @ 512K + auto-compaction summarizer) | NAMM (wrong-layer fit — see `03_TESTING.md` §6) |
| Multi-step REASONING (not calc) at L24 | No — documented deep-diffuse gap (`augmentation_thesis.md`) | RDT-style Tier-3 card (speculative — gated on failure-surface corpus) |

Full per-mechanism mapping: `01_ARCHITECTURE.md` §"What's invariant
vs what each paper adds" + §"Substrate framing".

## Decision the user needs to make

Six candidates, refined in Round 2 collab with codex (see
"Discussion receipts" below). Ranked by **what unlocks the next
decision**, not by raw cost. Full spec: `03_TESTING.md` §3-7.

| # | Candidate | Cost | Verdict |
|---|---|---|---|
| **E0** | Failure-surface scout — collect 30-50 stock-Gemma BigCodeBench failures, partition by capability vs format vs library-recall vs **iteratively-refineable structure** | low (~1-2 days) | **Recommended first** — without this, every other candidate is engineering tuning on a card class that has nowhere to live |
| **E1** | RDT Tier-3 card targeting the L24 deep-diffuse gap | high (~2-4 weeks) | Conditional on E0 surfacing ≥10 surviving prompts |
| **A** | Loop-index RoPE embedding on D5 | ~1 day | **Demoted** from "recommended first" — vacuous without (a) E0 corpus OR (b) a separate D5 card showing multi-iter headroom; documented null at R19 (see receipts) |
| **B** | ACT halting on D5 cards | medium | Ship if A becomes non-vacuous AND a corpus shows fixed-iters underperforms |
| **C** | Parcae stability fix | medium | Park until D5 grows explicit `B·e` injection |
| **D** | Build NAMM pipeline | high | **Skip** — wrong layer (no llama.cpp attn hook, no evo infra) |

**Recommended next step**: Candidate **E0** — stock-Gemma-only
corpus scout on BigCodeBench multi-library, reusing the
`dt_install_eval.py` extractor/sandbox style at N=30 smoke.
Two-stage falsifier (Round-3 cross-review fix):
- **Smoke (N=30)**: <5 survivors → park or try one alternate corpus source (e.g. CodeContests stdin runner) before full E0
- **Full E0 (100-200, post-expansion)**: <10 survivors → UT-family synthesis is dead-end for this substrate, park the entire arc

## Discussion receipts (Round 1-2, claude+codex, 2026-04-25)

Two-round collab refined the candidate ranking before any
implementation. Verbatim lifts that earn their place in the spec:

> **"no known positive D5 headroom; one known null"** — codex Round 1
> (msg `1777120318613-0cf85283`), citing
> `.claude/MEMORY/evals/2026-04-21_r19_d5_refinement_null.md:23-28`
> (PT+Δ at n_iters=2 regressed −6pp vs plain PT, not "stayed in
> plateau" as the original synthesis softened it).

> **"Refinement-loop benefit requires structured output it can
> iteratively improve"** — claude Round 2 (msg
> `1777120398328-fd67f769`), lifted from `r19_d5_refinement_null.md:57-65`
> with codex's wording guard ("structured output it can iteratively
> improve" — NOT "multi-line program", which would let multi-line
> static library recall sneak through).

Examples that satisfy the iteratively-refineable filter:
BigCodeBench multi-library programs, CodeContests algorithmic
solutions, scratchpad multi-step arithmetic, grid-reasoning.
Counter-example: API-trivia recall, library-name lookup, single-
token retrieval (R19 MQAR null came from this class).

Round-2 close phrase from codex (msg `1777120468569-94c180db`):
*"invert the ranking: failure-surface gate before loop-index."*

## Related rules in this repo

- `architecture.md` §"Substrate extensions" — D2/D3/D5/fast-weights spec
- `delta_rule.md` — DT (CopyAugmentedDeltaNet) Householder fast-weight backbone
- `augmentation_thesis.md` §"Circuit typology" + §"deep-diffuse" — Tier 1/2/3 framing + L24 gap
- `capability_gain.md` §"Failure-surface gate" — pre-Candidate-E discipline
- `Substrate.md` §"Substrate Extensions" — D5 + combined substrate
- `workflow.md` §"two measurements" — discipline for any of A/B/E

## Source citations

- `EUTM.md` — Sakana AI, "An Evolved Universal Transformer Memory" (Dec 2024). Paper + ChouBun benchmark + GitHub training code.
- `OPEN_MYTHOS.md` — Kye Gomez, OpenMythos open-source theoretical reconstruction (2026). References Parcae (Prairie 2026), Saunshi 2025, Relaxed Recursive Transformers (Bae 2024), DeepSeek-V2 MLA, Mixture-of-Depths.
- `UT.md` — Dehghani, Gouws, Vinyals, Uszkoreit, Kaiser, "Universal Transformers" (DeepMind / Google, 2018). Original ACT framing from Graves 2016.
