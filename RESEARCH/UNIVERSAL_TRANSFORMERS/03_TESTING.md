# UT-Family — Substrate-Relevance Gates

Five candidate experiments for applying UT/RDT/NAMM mechanisms to
this repo's substrate, ranked by fit + cost, each with hypothesis +
measurement + falsifier. Companion to `00_INDEX.md` (overview +
decision), `01_ARCHITECTURE.md` (architecture spec),
`02_IMPLEMENTATION.md` (mechanics + math).

This file follows the project's `_TESTING.md` convention but holds
**substrate-relevance experiment design**, not unit-test plans for
an implementation we're shipping. The "tests" here are
hypothesis-test rounds per `workflow.md` §"The loop".

## §1 What we already have (no-op vs the 3 papers)

| Mechanism in source papers | Substrate equivalent already shipped |
|---|---|
| Param-tied depth recurrence (UT/RDT) | **D5** — `calm/llm_computer/recurrent_substrate.py:69-84`, `n_iterations` kwarg iterates same layers HRM-style. Pure layer-stack iteration: no input injection, no loop-index, no ACT |
| Sequence-axis fast-weight recurrence | **DT** — `calm/llm_computer/copy_augmented_delta.py`, Householder rule per-position |
| Multi-step CALC composition | **`MultiStepReasoningFacade`** — parse → safe_eval → step-through bias for N-op chains |
| Long-context KV efficiency | **tq4 KV @ 512K** in llama.cpp serving + 200K Gemma / 130K Qwen NIAH-validated effective context (`niah_validation.md`) |
| Adaptive context management | **Auto-compaction** — summarizer at 89% safe-ctx threshold (`compact.py`) |

**D5 empirical state**: one known null
(`.claude/MEMORY/evals/2026-04-21_r19_d5_refinement_null.md`),
zero known positive multi-iter results. R19 ran D5 n_iters=2 on
MQAR retrieval; result was a −6pp regression vs plain PT
(21% best-epoch vs 27%). The receipt's mechanism diagnosis
(`:57-65`) is load-bearing for the §7 failure-surface gate:
*"Refinement-loop benefit appears to require tasks where the model
emits a structured output it can iteratively improve"* — single-
token retrieval has nothing to refine.

The substrate is already in the RDT family by mechanism. Net-new
candidates extracted in §2; ranking inverted in §8 per Round-2
receipts.

## §2 Net-new mechanisms (decomposition vs source papers)

| Net-new | Source paper | Cost to ship | Repo fit |
|---|---|---|---|
| Loop-index RoPE differentiation | OpenMythos §"Loop Index Embedding Hypothesis" | low (~1 day) | direct — drops into D5 |
| ACT halting on D5 | UT §2.2 + OpenMythos §"Overthinking" | medium | direct — replaces fixed `n_iterations` dispatch |
| Parcae LTI stability constraint | OpenMythos §"Stability Problem" | medium (gated on D5 redesign) | conditional — only if D5 grows `B·e` injection |
| NAMM (evolved memory) | EUTM | high | poor — wrong-layer fit (see §6) |
| RDT-shaped Tier-3 card for L24 deep-diffuse | OpenMythos thesis applied to `augmentation_thesis.md` gap | high | speculative — gated on failure-surface corpus |

## §3 Candidate A — Loop-index RoPE on D5

**Status: demoted from "recommended first" in Round-2 collab
(2026-04-25).** See `00_INDEX.md` §"Discussion receipts" + §8
below. Vacuous as a first round because (a) D5 has no known
positive multi-iter headroom — R19
(`.claude/MEMORY/evals/2026-04-21_r19_d5_refinement_null.md:23-28`)
documents a −6pp regression at n_iters=2 — and (b) loop-index can
only differentiate iterations on tasks where iterative refinement
helps, which requires a corpus that satisfies the §7 filter 4
("iteratively-refineable structured output"). E0 (§7a) must surface
that corpus first.

**Hypothesis (deferred until pre-conditions met)**: D5 same-weight
iterations currently lack any signal distinguishing iter 1 from
iter T. A RoPE-style sinusoidal encoding of `t ∈ [0, T)` added to
the residual at each iteration will let the same weights specialize
per-iter (early iters do pattern matching, late iters do refinement)
— measurable as a quality lift on tasks where `n_iterations > 1`
currently underperforms what the thesis predicts.

**Measurement** (raw + user-facing per `workflow.md`):

- **Raw**: existing D5 cards trained at `n_iterations = 4` with vs
  without loop-index — measure val acc on the same heldout split
- **User-facing**: Gemma + D5-card-with-loop-index A/B vs
  Gemma + D5-card-without on the card's domain corpus
- **Diagnostic**: probe per-iteration hidden state norm and
  attention pattern; loop-index should produce iter-distinguishable
  patterns if the mechanism differentiates

**Cost**: ~1 day (one constant encoding function + one add in the
iteration loop + retrain on existing data)

**Falsifier**: no measurable delta (≤2%) between with vs without
loop-index on D5 multi-iter evals → rules out the cheap version of
"iterations specialize". Forces choice between (a) ACT halting next
or (b) abandoning per-iter differentiation entirely.

**Prerequisite**: existing D5 baseline at `n_iterations > 1` that
currently produces measurable headroom (multi-iter not at ceiling
on its eval). If D5 cards at `n_iterations = 4` already saturate,
loop-index has no headroom to lift; falsifier triggers vacuously.
Verify before committing.

## §4 Candidate B — ACT halting on D5

**Hypothesis**: Fixed `n_iterations` per mode token wastes compute
on inputs that converge early AND under-allocates compute on
inputs that need more iterations. Per-position learned halting
(ACT) recovers both ends — concretely: a corpus exists where
`n_iterations = 4` fixed underperforms `n_iterations = 2` on a
subset of inputs (overthinking) AND underperforms `n_iterations = 8`
on a different subset (underthinking).

**Measurement**:

- **Raw**: D5 card with ACT vs D5 card with fixed `n_iterations`
  swept across {1, 2, 4, 8} — measure val acc and average iteration
  count
- **User-facing**: end-to-end Gemma + ACT-D5-card on the card's
  domain corpus, A/B vs best-fixed baseline
- **Diagnostic**: histogram of halt-iteration per input; ACT should
  show non-trivial spread, not a delta at one value

**Cost**: medium (~3-5 days). Add `halting_head` parameter, modify
iteration loop to gate per-position, add ponder-cost regularizer to
loss, retrain.

**Falsifier**: ACT halting picks effectively-uniform iteration
count across the corpus (e.g. all positions halt within ±1 of the
mean) → adaptive computation isn't being learned. Implies either
(a) the corpus has uniform difficulty (try harder corpus) or (b)
ponder-cost regularizer is mistuned (try smaller weight) or
(c) the mechanism isn't load-bearing for D5's domains (rule out).

**Prerequisite**: Candidate A wins (or doesn't matter). If
loop-index alone fixes the multi-iter problem, ACT may be
redundant overhead.

## §5 Candidate C — Parcae stability for D5 + injection

**Status**: **parked, conditional trigger.**

D5 currently iterates layers with no explicit `B·e` injection term.
The Parcae fix `A := Diag(-exp(log_A))` with `ρ(A) < 1` constrains
**injection parameters**, which D5 doesn't have. So Parcae's fix
doesn't directly apply to current D5.

**Trigger condition**: if D5 is redesigned to add explicit input
injection — `h_{t+1} = A · h_t + B · e + LayerStack(h_t)` — then
Parcae's stability constraint becomes load-bearing. Without it,
high-`n_iterations` training will diverge.

**Diagnostic if D5 ever adds injection**:

```python
A = d5_card.injection.A
rho = torch.linalg.eigvals(A).abs().max().item()
assert rho < 1.0, f"D5 spectral radius {rho:.4f} → unstable"
```

Run before each training round; track in the run log.

**Falsifier (conditional)**: if D5+injection trains stably at
high `n_iterations` (e.g. 16) without the Parcae parameterization,
the stability fix isn't load-bearing for our scale. Unlikely given
the Parcae empirical evidence (every divergent run had `ρ(A) ≥ 1`),
but worth measuring.

## §6 Candidate D — NAMM (skip)

**Decision: skip** unless a public weight ships.

**Why wrong-layer fit**:

1. **Production serving is llama.cpp** (C++, `localhost:8080`).
   Has no Python attention hook; NAMMs need attention matrices per
   token per layer. Adding NAMMs forces all serving onto the
   `GemmaSubstrate` Python path, which is ~5-10× slower.
2. **No evolutionary training infrastructure** in the repo. Building
   evolutionary trainer + population management + per-generation
   eval orchestration is several weeks of work before the first
   experiment runs.
3. **Effective context already covered**. NIAH-validated 200K Gemma
   / 130K Qwen (`niah_validation.md`) covers all current workloads.
   Auto-compaction at 89% safe-ctx (`compact.py`) handles the
   over-context case. Marginal value of NAMM-style pruning at
   current scale is small.
4. **Substrate-RAG already has automatic Tier-1 preservation** via
   hash-gated injection (`augmentation_thesis.md` §"Automatic Tier-1
   preservation"). Different mechanism, similar property
   (selective intervention without global context modification).

**Trigger to revisit**: Sakana or a third party publishes a NAMM
weight trained on Llama-3-8B. Universal-transfer property would let
us apply it to Gemma 4 E4B with **zero training cost**. Cost
becomes (1) wiring a Python attention hook into `GemmaSubstrate`
forward path + (2) load NAMM weight + (3) measure. Days, not weeks.

**Falsifier (if revisited)**: applied NAMM doesn't lift Gemma's
LongBench / InfiniteBench score AND doesn't reduce KV-cache memory
without hurting NIAH ⇒ universal-transfer doesn't generalize to
Gemma's attention pattern. Park.

## §7 Candidate E — RDT Tier-3 card for L24 deep-diffuse multi-hop

**Status**: **speculative — gated on failure-surface corpus.**

`augmentation_thesis.md` documents L24 multi-step composition as
**deep-diffuse** — full-layer ablation Δ is large but signal is
diffuse at attention AND FFN AND per-neuron AND SAE-feature levels.
Currently flagged as "not tier-3-distillable at known loss spaces"
and "pivot to tier-2 stacking" as the official position.

RDT is the architectural family **specifically designed** for this
gap: depth recurrence with input injection and ACT halting handles
multi-step composition that static depth can't.

### Failure-surface gate (mandatory, per `capability_gain.md` + R19 mechanism filter)

Before designing the card, partition candidates through **four
filters in order**. The first three follow `capability_gain.md`
§"Failure-surface gate"; the fourth was added in Round 2 collab
(2026-04-25) lifting from `r19_d5_refinement_null.md:57-65`:

1. Collect **100-200 multi-hop reasoning prompts** from a standard
   benchmark with bundled tests (BigCodeBench multi-library
   preferred; see §7a below)
2. Filter to `fails_correctness` + `partial` partitions per
   `capability_gain.md:60-66` — drop `solves_cleanly` (ceiling)
   and `format_fails` (extractor bugs, not capability failures)
3. Filter out prompts that are **decode-path-addressable** —
   anything reducible to `safe_eval` belongs to
   `MultiStepReasoningFacade`, not RDT
4. **(NEW, R19-derived)** Filter out prompts whose failure isn't
   addressable by iterative refinement. Rule: keep iff the task
   produces **structured output the model can iteratively improve**.
   - **Satisfies**: multi-line program body, multi-step algorithmic
     solution, scratchpad arithmetic, grid-reasoning layout.
     Refinement = revise body, reorder calls, fix imports, correct
     intermediate values
   - **Doesn't satisfy** (drop): API-trivia recall, library-name
     lookup, single-token retrieval, "what's the function for X"
     factual queries. R19's MQAR null came from this class — single-
     token output had nothing to refine, so D5 n_iters=2 regressed
     −6pp vs plain PT
5. The remaining corpus IS the eval target. If < 10 prompts
   survive **at the full E0 (100-200) scale, post-expansion**,
   pivot — RDT card has insufficient headroom to justify the cost;
   UT-family synthesis is dead-end for this substrate. (Smoke-stage
   gate is separate — see §7a.)

### §7a Recommended E0 scout configuration

Cheapest concrete operationalization of the gate above (codex
Round 1 cite `code_example_db.py:48-51` + `r53_fetch_corpora.py:273-285`):

- **Corpus**: BigCodeBench (local at `bigcodebench.jsonl`, includes
  unit tests, multi-library coordination by design ⇒ inherently
  multi-step structured output)
- **Stock-only baseline**: stock Gemma 4 E4B via existing serving;
  no DT, no RENAME, no RDT yet
- **Extractor + sandbox**: reuse `dt_install_eval.py:44-60` style
- **Smoke size**: N=30 first. Two-stage expand-or-park gate:
  - **≥5 survivors** → expand to 100-200 (full E0)
  - **<5 survivors** → insufficient signal from BigCodeBench smoke;
    park OR try one alternate corpus source (e.g. CodeContests with
    stdin runner work) before full E0
- **Skip for E0**: HE+ (single-function, weak for RDT thesis),
  CodeContests (stdin/sample-I/O runner work not yet built)

### If the corpus exists

**Hypothesis**: A small RDT-shaped card (e.g. 2 prelude layers +
recurrent block × T iterations + 1 coda, weights tied, total
≤500K params) trained on multi-hop reasoning will solve a
non-trivial fraction of the deep-diffuse corpus where compiled
cards (no recurrence) and trained PT/DT (no depth recurrence) both
fail.

**Measurement**:

- **Raw**: card standalone on the failure-surface corpus
- **User-facing**: Gemma + card via `CardSlot.attach` at L24 (the
  documented gap location), A/B vs Gemma alone on same corpus
- Verify no regression where Gemma was already correct

**Cost**: high (~2-4 weeks). Card design + training data curation
+ training + install + threshold calibration.

**Falsifier**: card standalone solves the corpus but Gemma + card
shows no lift → install-mechanism (`CardSlot` margin gates,
`VerificationHook`) fails to deliver the card's output to Gemma's
token stream. Different problem from card design; might be
addressable with `embed_intelligence.md` mechanisms (step-through
bias for multi-token answers).

**Falsifier 2**: card standalone fails the corpus → recurrent
inductive bias alone doesn't solve multi-hop reasoning at this
scale. Either (a) corpus is too hard for ≤500K params (try
larger), (b) RDT thesis doesn't generalize from algorithmic tasks
to multi-hop NL reasoning, or (c) failure-surface gate let
non-RDT-addressable prompts through.

## §8 Recommended next experiment (inverted Round 2, 2026-04-25)

**Candidate E0** (failure-surface scout, §7a above) is the right
first round. Inversion grounded by Round-2 collab receipts (see
`00_INDEX.md` §"Discussion receipts"):

- **D5 multi-iter headroom is empirically absent.** R19 documents
  PT+Δ at n_iters=2 regressing −6pp vs plain PT
  (`r19_d5_refinement_null.md:23-28`). No countervailing positive
  D5 receipt exists in the repo.
- **Refinement-loop benefit requires iteratively-refineable
  structured output** (`r19_d5_refinement_null.md:57-65`). MQAR's
  single-token retrieval is the wrong shape — that explains R19,
  not a tunable bug.
- **A (loop-index) is therefore vacuous as a first round.** It can
  only become non-vacuous if E0 surfaces a corpus where iteratively-
  refineable failures exist, OR a separate D5 card is trained on a
  refineable task and shows multi-iter headroom. Currently neither
  exists.

**E0 unlocks the next decision tree** (full-E0 outcomes,
post-smoke-expansion per §7a):
- Full E0 surfaces ≥10 surviving prompts → E1 (RDT Tier-3 card)
  becomes the priority; A/B/C become its in-design choices
- Full E0 surfaces <10 → UT-family synthesis is dead-end for this
  substrate; park the entire arc
- (Smoke-stage <5 → see §7a's park-or-alternate clause; never
  reaches this tree)

Cost asymmetry: E0 is ~1-2 days (corpus assembly + sandbox runs),
**lower than A's 1-day implementation** but with strictly more
information value. There's no "ship A as cheap diagnostic" argument
that survives the R19 receipt.

## §9 Falsifier table (one row per candidate, Round-2 ordering)

| Candidate | Pre-condition | Measurement | Falsifier (rules out) |
|---|---|---|---|
| **E0** Failure-surface scout | None — runs against existing local corpus | 4-filter partition (§7 above) on N=30 BigCodeBench smoke; expand to 100-200 on ≥5 survivors | **Two-stage**: smoke <5 survivors → park or try one alternate source (per §7a); full E0 (100-200) <10 survivors post-expansion → UT-family arc dead-end for this substrate; park |
| **E1** RDT card | E0 surfaces ≥10 surviving prompts | Card standalone + Gemma+card A/B on E0 corpus | Card standalone fails ⇒ thesis null at this scale; OR card succeeds standalone but Gemma+card flat ⇒ install-mechanism null |
| **A** Loop-index (demoted) | EITHER E0 found a refineable corpus AND a D5 card is trained on it OR a separate D5 card already shows multi-iter headroom | val acc with vs without loop-index on that card | ≤2% delta → loop-index doesn't differentiate iterations cheaply; abandon per-iter differentiation, ACT (B) becomes the only remaining iter-differentiation bet |
| **B** ACT | A becomes non-vacuous AND loop-index alone insufficient | halt-iter histogram + val acc | Effectively-uniform halting (no spread) → adaptive computation isn't being learned |
| **C** Parcae | D5 grows explicit `B·e` injection | spectral radius diagnostic + train stability | Stable training at `T = 16` without `Diag(-exp(log_A))` parameterization → Parcae fix not load-bearing at our scale |
| **D** NAMM (skip) | Public Llama-3-NAMM weight published | LongBench / InfiniteBench / NIAH on Gemma+NAMM | No lift OR NIAH degradation → universal-transfer doesn't generalize to Gemma's attention pattern |

## §10 What this does NOT validate

- **Whether RDT-as-architecture is the right foundation for a
  ground-up frontier-scale base model.** Out of scope; this repo
  augments Gemma, doesn't train new bases.
- **Whether the OpenMythos reconstruction matches actual Anthropic
  Mythos design.** Speculation in the source paper; irrelevant to
  substrate work.
- **Whether evolutionary training generalizes to other
  non-differentiable substrate problems.** Possibly interesting
  long-term but not on the current critical path.

These are out-of-scope per the substrate-relevance lens. Anyone
returning to them should re-scope the synthesis from a different
lens (e.g. "training a new base from scratch" — would justify
revisiting OpenMythos as more than a citation).
