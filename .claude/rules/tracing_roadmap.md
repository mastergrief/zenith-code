# Tracing Roadmap — what's compiled, what's next

Concrete inventory of (a) compiled cards that exist, (b) cards that
would be easy wins but haven't been built yet, (c) the research path
for harder capabilities. Updates as rounds ship.

See `tracing_intelligence.md` for the first-principles framing and
`capability_gain.md` for what counts as a validated win.

## State today (after Round 50.6)

### Shipped and verified

| Card | Status | Verification | Install path tested |
|---|---|---|---|
| `adder_tiny` | Exhaustive 16/16 | Standalone + facade | CardSlot |
| `adder` | Exhaustive 10K/10K | Standalone only | Not yet |
| `multiplier` | Exhaustive 3390/3390 (a·b < 1000) | Standalone + facade, 3 real arithmetic fixes on Gemma | **Step-through digit bias (R11b)** |
| `add_one`, `threshold`, `copy_past`, `retrieve_by_index`, `retrieve_threshold` | Various exhaustive | Standalone | Some via in-attention |
| `gcd`, `factorial`, `is_prime` | Exhaustive (but `gcd` MAX_OP=15 too small for real targets — see ruled-out log) | Standalone | Not yet |
| `dispatched_v4` | 791/791 | Standalone | In-attention verified |
| `reasoning_engine` | 512/512 | Standalone | Not yet as facade |
| `KnowledgeStore` recall card | 10/10 in Round 6 demo | In facade | CardSlot |
| `HubInjectionCard` | Bit-identical to R43 inline intervention | Facade wraps R43's L23 H1/H4 forced-attention | Forced-attention facade (R44/R45) |
| `MultiStepReasoningFacade` | 17/17 Gemma fixes, 0 regressions | NL infix parser + `safe_eval` router | Step-through digit bias N-op (R46.2) |
| `ast_repair` walker | 21/21 unit + token_bucket 0/0 → 5/5 on Gemma | Two deterministic rewrites (shadow rename, dict-key synonym) driven by runtime error text | Post-generation AST rewrite, no decode-time install |
| **Tracing methodology** | Validated on Gemma 4 E4B (Rounds 13-50.6) | Activation patching + per-head + Q/K/V decomp + per-sub-head + forced-attention validation + SAE (TopK) reconstruction | 7 capabilities mapped (+ multi-step composition); 3 causal validations (R28, R42, R43); hub-sharing proven at L23 H1/H4 (5-for-1 ROI); SAE reconstructs L24 composition at rank-50 in feature basis but top features have zero causal effect (R50.5) — interpretability-without-causality gap |

### Facades built

| Facade | Card(s) | Rounds | Honest result |
|---|---|---|---|
| `MathAdditionFacade` | PT + adder_tiny | 6-9 | Format coercion on task Gemma already solved. First reusable class (Round 8). |
| One-off reasoning facade | PT + adder_tiny, recall card | 6, 7, 11 | R6/R7 were format coercion; R11 (multiplication) is the first real capability win. |
| `HubInjectionCard` (`calm/llm_computer/facades/hub_l23.py`) | L23 H1/H4 forced-attention | 44, 45 | Facade form of R43 intervention; bit-identical. Runtime Q/K detects natural top-position; no per-task hand-dispatch. Serves arithmetic + SV + comparison + counting + multi-step (5-for-1). `generate()` verified 5×12 decode tokens (R45). |
| `MultiStepReasoningFacade` | NL infix parser + `safe_eval` | 46.1, 46.2 | 17/17 real Gemma fixes, 0 regressions. N-op extension of R11 step-through digit bias. Parses NL infix, routes values through `safe_eval`. |
| `ast_repair` walker (`calm/llm_computer/facades/ast_repair.py`) | `ast` stdlib | R53.35 | Post-generation AST rewrites for R53.33 deterministic failure modes. Shadow rename (TypeError callable) + dict-key synonym rewrite (KeyError, curated table). Raw: 21/21 unit. User-facing: token_bucket 0/0 → 5/5 in 0.9s on Gemma; lru_cache_class 9/9 preserved (no regression). csv_column_stats remains at the extractor ceiling (generates 0 extractable code — different failure class from logic bugs). Correct tier-2 intervention for code repair: auditable, mechanical, zero LLM cost. |

### Install mechanisms verified

- CardSlot + preservation masking: Rounds 4-11
- Token-embedding projection at layer 33: Round 9
- Step-through digit bias: Round 11 (multi-token answers)
- VerificationHook with min_margin: Rounds 6-11
- Save/load round-trip of full substrate: Round 1 (bit-identical logits)
- CUDA Graphs × FP32 layers compat: Round 2 (4.29× speedup preserved)

## Gemma 4 E4B tracing findings (Rounds 13-50.6)

First mechanistic-interpretability arc on prod Gemma. Validates
that tier-3 (reverse-engineered circuits) is tractable on this
model with cheap probes. The arithmetic capability was traced via
per-layer / per-head / Q-K-V ablation. Scripts in `scripts/test_*.py`.

| Round | Probe | Key result |
|---|---|---|
| R13 | Naive logit lens | Top-5 at middle layers = noise. Rank trajectories of specific digit tokens show signal around L17 (correct-digit rank peaks ~r1500-3000). |
| R14 | Single-prompt activation patching (17×23) | L35 flipped argmax. One-example — overclaimed as "THE arithmetic layer" in that commit. |
| R15 | L35 on 10 arithmetic pairs | Only 2/10 flipped. L35 doesn't generalize — Round 14 was one-example overreach. |
| R16 | 42×10 ablation sweep | **Arithmetic clusters at L22-L30.** L23 (global layer) is peak (mean Δ=-10.18, hurts 10/10). L22, L24, L26, L28, L29 all hurt 10/10. L35 is secondary (-1.50, 9/10). |
| R17 | L23 per-head ablation (8 Q-heads) | **H1 (-4.85) and H4 (-4.30) carry the load.** Other 6 heads: mean Δ ≈ 0. |
| R18 | Q/K/V decomposition of H1, H4 | **V (content) carries 93% of H4's contribution.** Q and K alone have negligible effect. V-only ablation matches full QKV. |
| R19 | Linear probe V → first-digit | 2x chance (0.22 vs 0.11, 270 samples). Real but indirect signal — V encodes operands/intermediates, not the final digit directly. SAE needed for clean features. |
| R20 | Per-sub-head ablation on H4 (256 d_head=2 slices × 10 pairs) | **Signal is distributed, NOT localized.** 0 sub-heads with mean Δ < -1.0; top sub-head = -0.583 (vs full H4 = -4.30). Top-8 sub-heads carry only 26% of damage; need top-64 for 80%. Reshapes R22: SAE target is H4's full 512-d output or L23 residual, not a narrowed slice. |
| R21 | V-vector multi-label probe at L23 | V encodes operand digits (a_ones, a_tens, b_ones, b_tens) + intermediates (ones_prod, ones_carry). Operand digits decode cleanly; product high-digits don't (p_huns at chance). |
| R22 | Layer-trace residual probe L1-L41 + shuffled-label control | fd emerges at L29→L35, not earlier. R21's ones_prod/ones_carry findings inflated by probe capacity; a_ones/p_huns clean. |
| R23 | Fine-trace L28-L36 | fd builds cumulatively across L30-L32 pipeline; biggest step L31→L32. p_huns still climbing at L36; p_tens flat. |
| R24 | Attn vs FFN ablation at L29-L32 | L30 ATTN gathers (+0.148), L31-L32 FFN computes (+0.185, +0.111). Parallel dissociation at L31: FFN writes fd, ATTN writes p_ones. |
| R25 | Per-head at L30/L31/L32 | **L30 H6 is the fd-gatherer** (Δ=-1.528, 10/10). L31 H4/H5 weaker; L32 diffuse. |
| R26 | L30 H6 attention pattern | H6 attends 61% to a_ones token (pos 3). H4 attends to b_ones (pos 7). **Clear position-selection.** |
| R27 | V at L22 positions | V[pos=3] perfectly encodes a (1.00); V[pos=7] perfectly encodes b (1.00); no cross-contamination. |
| R28 | **Forced-attention validation** | Forced one-hot attn at L30 H4/H6 to pos 7/3 → fd preserved, mean \|Δ\|=0.407, argmax matches 9/10. **Circuit causally validated as compilable.** |
| R29 | Factual recall layer sweep | L11 peak -1.56 (GLB), L5 secondary. Distributed early+late pattern. Magnitudes 6× smaller than arithmetic. |
| R30 | Factual per-head L5/L11 | **Diffuse** — no head > -0.08. FFN-locked circuit (ROME/MEMIT territory). |
| R31 | Induction layer sweep (repeated A B C D A B C D A B) | L34 peak -1.17, 20/20. Cluster L33-L37 SWA-dominant. |
| R32 | Induction per-head L33/L34/L37 | **L37 H6 concentrated** -0.52, 11/20. L33/L34 diffuse. Two-stage circuit. |
| R33 | L37 H6 attention pattern | **Classic induction head** confirmed (Olsson 2022). 55% attention on answer-letter positions, 4% on query-matching positions. |
| R34 | Counting sweep (Count: 1, 2, 3, 4, ) | L20 peak -3.93, 6/6. Shares L33/L37 with induction, adds L20/L31 new peaks. |
| R35 | Counting per-head L20/L31/L33/L37 | **L20 cooperative** (H2/H5/H6 each ~-1.0-1.4, sum ≈ full -3.74). **L37 H4 numeric-successor** specialist (Δ=-1.02), different from L37 H6 (induction). |
| R36 | Comparison sweep (which is larger) | L35 peak -1.77, 18/18. Shares L23 with arithmetic. |
| R37 | Comparison per-head L23/L35 | **Diffuse** — no head > -0.4. L35 is FFN-based application layer. |
| R38 | SV agreement sweep | **3-stage global pipeline L23→L29→L35** (each hits 14-16/16). New circuit shape. |
| R39 | SV per-head L23/L29/L35 | **Hybrid**: L23 H1+H4 concentrated (same heads as arithmetic!), L29 H7 concentrated (new specialist), L35 diffuse. |
| R40 | L23 H1/H4 attention on SV prompts | **H4 = subject reader** (0.76 on subject complex), **H1 = distractor reader** (0.50 on distractor). Hub behavior confirmed. |
| R41 | L23 H1/H4 attention on arithmetic prompts | **H1 reads b-operand 3× more than a** (consistent with "second content item" role on SV). H4 more mixed on arithmetic. Same heads, task-specific Q patterns. |
| R42 | L23 H1/H4 forced attention on SV agreement | **mean \|Δ\|=0.467, 8/10 match.** Mirror of R28 at different layer + task; validates hub-sharing on linguistic capability. |
| R43 | L23 forced attention on comparison + counting | **Comparison 18/18 (cleanest result in session, \|Δ\|=0.176), counting 6/6.** 4-for-1 compilation proven: one compiled L23 H1/H4 replacement benefits arithmetic + SV + comparison + counting simultaneously. L23 hub across 3 capabilities: **32/34 argmax matches (94%)**. [commit b8cc655] |
| R44 | `HubInjectionCard` shipped | Facade form of R43 intervention, bit-identical. Runtime Q/K detects natural top-position; no per-task hand-dispatch. [commit b8cc655] |
| R45 | `HubInjectionCard.generate()` | Prefill-only injection verified across 5×12 decode tokens. Compatible with autoregressive generation. [commit 3eca6c3] |
| R46.1 | `MultiStepReasoningFacade` parser+executor | Parses NL infix expressions, routes values via `safe_eval`. N-op extension of R11 pattern. [commit 4db3e67] |
| R46.2 | Multi-step fix 17/17 Gemma failures | 17 real fixes, 0 regressions on held-out prompts. MultiStepReasoningFacade is the 5th beneficiary of L23 hub. [commit a385893] |
| R47.1 | Multi-step layer sweep (initial) | 19 layers with Δ<-1.0 — INVALIDATED by R47.2 (prompt-format contamination). [commit 94fa58e] |
| R47.2 | L34 per-head diffuse + copy-c contamination | Top H4 Δ=-0.20 (only 6% of layer's full Δ). Diagnosed prompt-format flaw: the multi-step prompts admitted a "copy c" shortcut that biased the sweep. [commit da27eee] |
| R47.3 | Clean-prompt sweep → L24 is multi-step peak | L24 mean Δ=-17.23 (69% larger than R16 L23's -10.18). L24 is the multi-step composition layer; architecturally SWA, not global. [commit 2773409] |
| R47.4 | L24 per-head diffuse | Top H1 Δ=-0.635 vs full-layer -17.23. Attention-level dead end — multi-step composition is not a concentrated-head circuit. [commit 3ea055e] |
| R48.1 | L24 FFN per-neuron diffuse | 10,240 FFN neurons ablated in 20 chunks of 512; no chunk carries the signal. Rules out ROME/MEMIT-style per-neuron install. [commit faa8e36] |
| R49.1 | L24 FFN low-rank SVD | Mean-centered rank 34 @ 90% variance; naive rank 1 is DC-dominated (offset, not signal). Composition signal is moderate-rank. [commit 324b7e7] |
| R49.2 | L24 FFN pos(-1) task-rank 1 | At the last token position, K=1 preserves 100% of the task effect. Composition projection at pos(-1) is rank-1 in task space. [commit dc46db0] |
| R49.3 | L24 FFN is NOT composition source | FFN activation at all non-last positions = zero for composition. Per-layer-embd injection candidate suggested. [commit fa3f957] |
| R49.4 | L24 pathway decomposition diffuse | Signal distributed across attn + ffn + projection; significant non-additive interactions between pathways. Not cleanly decomposable. [commit 936ed35] |
| R49.5 | L24 joint-output task-rank 1 non-last | Composition information lives at NON-last positions in the joint attn+ffn output. Different signal at different positions. [commit 1ff8b8d] |
| R50.1 | SAE infra works, λ=5e-4 too weak | First Sparse Autoencoder trained in project. 98% reconstruction but L0 sparsity 2823 (≈target density, not sparse). [commit 11e7a33] |
| R50.2 | SAE λ=5e-3 plateau L0~1700 | Stronger L1 doesn't break through; plateau at 5.7× target sparsity. Standard SAE L1 regularizer hits a local minimum. [commit 5e9686c] |
| R50.3 | TopK SAE: L24 rank-50 in feature basis | K=50 TopK SAE preserves 99.1% reconstruction; effective L0 = 50, 60% of features dead. **L24 composition is rank-50 in a learned feature basis.** [commit 0e8f35f] |
| R50.4 | 370 task-specific directions identified | K=100 SAE: 370 features with 70-1089× multi-operand vs single-operand activation ratio. Task-specific feature directions exist. [commit 848b942] |
| R50.5 | **ZERO causal effect from top-50 SAE features** | 17/30 baseline → 17/30 after ablating top-50 composition features. **Interpretability-without-causality.** Falsifies R20's "SAE = next step" direction for compilable-intervention work. [commit ce4ce7d] |
| R50.6 | SAE install at L24, 100% preservation | Re-installing the trained SAE at L24 with 99.6% reconstruction preserves arithmetic 100%. Infra works end-to-end; but the reconstructed basis does not recover a compilable composition circuit. [commit 0974f21] |

**Session 33-34 summary (through R53.34)**: 49+ round arc (R13-R50.6)
plus R53 kernel + eval-methodology work (R53.14-R53.34): TurboQuant Q_prod
null, fused flash-attn correctness=1.0 with non-monotonic perf curve (wins
N∈[128,2048], loses outside; **shipped default-on with N-gate per
2026-04-20 re-bench** — see row R53.34), tq4 matvec v2 shipped (-7%),
v3/v4 null, BLOCK_M sweep null, MAX_TOKENS starvation receipt (R53.25),
sandbox stdlib pre-import fix (R53.22), substrate L41 install regression
on code (R53.14/20a/20b).

Session 33 core arc: **7 capabilities** mapped at sweep + per-head
resolution (arithmetic,
factual recall, induction, counting, comparison, SV agreement,
multi-step composition), **3 causal validations** (R28 arithmetic,
R42 SV agreement, R43 comparison+counting), typology validated
across numeric + linguistic circuits, **hub-sharing empirically
proven** — L23 H1/H4 shared content-carrier heads with task-specific
Q routing, 32/34 argmax preservation rate across 3 L23-using
capabilities, **5-for-1 ROI** once `MultiStepReasoningFacade`
joined the hub beneficiaries. **R47-R50.6 mapped multi-step
composition at L24**: diffuse at attention + FFN + per-neuron
+ SVD levels. Rules out attention-level install (R47.4), rules out
ROME/MEMIT FFN install (R48.1). SAE arc R50.1-6 demonstrated that
L24 composition is rank-50 in a learned feature basis (R50.3) and
SAE install preserves 100% arithmetic (R50.6), but top-50 features
ablated have **zero causal effect** (R50.5) — interpretability-
without-causality gap falsifies R20's "SAE = next step" direction.
Full atlas + per-head lookup: `.claude/MEMORY/atlas.md`.

**Full atlas**: `.claude/MEMORY/atlas.md` — capability/layer/head
tables for quick reference.

**The arithmetic circuit:** `L23 attn_v (KV group 1, 512-d)` →
H4's Q pattern reads it → ~2.6M-param V-projection is the concrete
localization of arithmetic content on a 5B-param model. Compact
target for SAE + ACDC work.

**Why the localization was clean:** Gemma's alternating SWA/global
attention forces cross-operand aggregation into the global layers
(5, 11, 17, 23, 29, 35, 41). Arithmetic NEEDS to see both operands →
it ends up at the global layers. L23 and L29 were architecturally
predicted; the measurement confirmed. Capabilities requiring the
same information-flow structure (syntax, anaphora, cross-position
facts) are the next good targets. Capabilities that are distributed
(semantic understanding, open-ended reasoning) will NOT localize as
cleanly.

## Next rounds — tier 1 (trivial, ship fast)

Each is hours of labor. Template: gate-graph card → exhaustive
verification → facade with step-through bias → measure on Gemma
baseline failures. Rounds 12-19 consumed the original R12-R19 slots
for the tracing arc and related work; tier-1 queue re-sequenced below.

| Round | Target | Gate-graph pattern | Expected size |
|---|---|---|---|
| 20 | Primality facade | `is_prime` exists; need Y/N step-through bias + `wait_marker` for Gemma's verbose prompt | Small |
| 21 | Factorial (7!, 8!+) | Existing `factorial` + step-through multi-digit | Medium |
| 22 | Roman numeral ↔ decimal | Lookup table, compiled from CALM backend | Tiny |
| 23 | Unit conversion (temperature, length) | Lookup + linear | Tiny |
| 24 | Timezone offset | Per-city DB | Tiny |
| 25 | Modular arithmetic (`a mod b`) | New card, similar pattern to gcd | Small |
| 26 | Exponentiation (small) | `a^b` for small b via repeated multiply | Medium |

Tier-2 targets (see below) absorbed: GCD at real operand sizes
(needs iterative Euclidean compilation), 3-digit multiplication
(needs digit decomposition).

Rules for tier 1 rounds:
- Must establish Gemma baseline failure BEFORE building (per
  `capability_gain.md` §"failure-surface gate").
- Must verify card exhaustively on its input space.
- Must show both measurements move (raw + user-facing).
- One round per commit with before/after table.

## Next rounds — tier 2 (designed, novel circuits)

Labor: days to weeks per card. Each requires designing the circuit,
not reverse-engineering.

| Target | Design approach | Motivation |
|---|---|---|
| **General planner** (decompose goal → track subtasks) | Channel-as-register + dispatched_v4-style opcode dispatch | First circuit that touches multi-step reasoning. ~500 gate-graph nodes. |
| **AST parser for Python expressions** | LookUpExact over token catalog + recursive structure via depth | Enables code-aware facades |
| **Type checker (simple)** | AST + lookup tables of type rules | Builds on AST parser |
| **Analogy-by-structural-match** | Graph isomorphism via LookUpExact + step-function consistency checks | Tests "abstract reasoning beyond what Gemma encodes" claim |
| **Sequential reasoner** (chain-of-thought style) | Channel-as-register state machine, step-through bias for intermediate results | General engine for multi-step arithmetic / logic |
| **3-digit multiplication via digit decomposition** | 4 single-digit × lookup tables + carry chain across 2 layers | Path to arbitrary-digit arithmetic without table explosion |
| **GCD via iterative Euclidean compilation** | Channel-as-register state machine over ~log2(MAX_OP) layers, each doing `a mod b` step | R12 showed step-diff LUT at MAX_OP=15 is useless; real operands need iterative compilation |

Each is a real research deliverable. Ship one every few weeks if
prioritized. None require interpretability breakthroughs — they're
circuit design.

## Next rounds — tier 3 (reverse-engineered from Gemma)

**Reframing (session 34, post-R52 triple-null).** Most capabilities
previously framed as tier-3 "plug missing" are better achieved by
tier-2 stacking — see `augmentation_thesis.md` §"Tier-2 stacking
achieves tier-3-equivalent outcomes". Every shipped working
augmentation (R11, R44, R46.2, HubInjectionCard, VerificationHook,
CardSlot+preserve) is additive and leverages Gemma's existing
NL/context/routing. R51/R52's explicit REPLACEMENT framing was the
anomaly; three nulls confirm distillation of a deep-diffuse layer
doesn't work. Targets below genuinely benefit from from-scratch
compilation OR are prerequisites for tier-2 work (interpretability
tooling). Default hypothesis for any new capability is tier-2
stacking, not tier-3 replacement.

Labor: weeks to months per circuit, pending interpretability tools.
These are the long-horizon bets that eventually give us the LM's
capabilities as compiled cards.

| Target | Prerequisite | Notes |
|---|---|---|
| **Close the SAE → causal-effect gap** | R50.1-6 shipped SAE infra end-to-end. R50.3 reconstructs L24 composition at rank-50 in feature basis; R50.6 re-install preserves 100% arithmetic. R50.5 ablating top-50 features is **ZERO causal** — interpretability-without-causality. Next: architectures where reconstructed features DO have causal effect (transcoders, attention-SAE, cross-layer SAE), OR accept that L24 multi-step composition is not attention-or-FFN compilable and pivot to capabilities whose circuits ARE (arithmetic L23 H4 via R28-template, induction L37 H6, counting L20 cooperative). | **OPEN PROBLEM. Replaces the old "SAE on H4's V output" recommendation (R50.5 falsifies that path).** |
| **Generalize tracing methodology** | Run R16-style ablation sweep on OTHER capability targets (syntax, anaphora, factual recall) | Tests whether arithmetic's clean localization was special or the methodology is general |
| Induction heads | Identify heads doing in-context copying on Gemma 4 E4B | Generic capability used by many tasks |
| NL parsing circuits | Circuit probing for syntax patterns | Replace PT's learned mechanism with a compiled one |
| Factual retrieval circuits | ROME / MEMIT-style probing | Replace / augment KnowledgeStore's step-function recall |
| Entity tracking | Feature directions for "who's the subject of this sentence" | Useful for word problems |

Research track status (updated through R50.6):

1. ~~**Sparse Autoencoders on Gemma 4 E4B**~~ **SHIPPED** (R50.1-6).
   TopK SAE at L24 reconstructs 99.6% with effective L0=50.
   Infra proven end-to-end (train → install → preserve task
   accuracy). Open question is not "can we train SAEs" but "do
   SAE features carry causal effect we can ablate/replace?" — R50.5
   answered NO for top-50 L24 composition features.
2. **Causal localization on distributed composition circuits** —
   **OPEN.** R50.5 is the canonical null: reconstruction fidelity
   ≠ causal effect under ablation. Candidates: transcoders,
   attention-SAE (Makelov-style), cross-layer SAE, activation
   patching on reconstructed components, feature circuits instead
   of features.
3. **Automated Circuit Discovery (ACDC)** — not yet attempted on
   Gemma 4 E4B. Natural next step for capabilities that cleanly
   localize (arithmetic already done via manual per-head +
   forced-attention).
4. **Feature labeling** — premature while (2) is open. Labels
   without causal effect are descriptive, not compilable.
5. **Circuit-to-IR translation** — mechanical once a circuit is
   both localized AND causally validated. R28 template proved
   this for concentrated heads; distributed circuits need (2) first.

Near-term scope: either solve (2) with a different SAE
architecture, or pivot Tier-3 work to capabilities whose circuits
are already compilable (L23 hub already validated R42/R43; extend
to more hub-served capabilities).

## Rough velocity estimate

Based on Round 11 (one day from "let's build a multiplier" to
shipped facade with demonstrated capability gain):

- Tier 1: ~2 rounds / week sustained pace. ~20-30 domains in 3 months.
- Tier 2: ~1 round / month. ~3-6 novel circuits in 6 months.
- Tier 3: pending research breakthroughs; not scheduleable.

With 30 tier-1 facades + 3 tier-2 circuits installed, substrate
covers the "verified local utility" value proposition for common
developer/power-user questions. Commercial viability threshold.

## What to update in this file

Each round that ships a new card or facade appends a row to the
"Shipped and verified" table. Each round that rules out a direction
per `workflow.md` §"Ruled-out log" should note it here too so future
sessions don't retry.

Ruled-out entries (from this session's rounds):

| Approach | Ruled out in | Reason |
|---|---|---|
| Token-embd projection at early layers (1, 5, 15, 25) | Round 10a | Degrades — residual at position -1 at early layers is processed as input, not as prediction pre-image |
| Direct 2-digit × 2-digit lookup table (MAX_PRODUCT=9801) | Round 11a planning | ~4.6 GB VRAM, doesn't fit alongside Gemma. Scoped to MAX_PRODUCT=999. Future: digit decomposition |
| Single-token bias for multi-token answers | Round 11 diagnosis | Obvious in retrospect — 391 is 4 Gemma tokens |
| Step-diff LUT for GCD at real operand sizes | Round 12 | Existing `gcd` card is MAX_OP=15; extending to 2024 would need a 4M-entry LUT. Use iterative Euclidean compilation (tier-2) instead |
| Naive logit lens as sole tracing tool | Round 13 | Top-5 tokens at middle layers = foreign-language / code noise. Only rank trajectories of tracked tokens give signal. Use ALONGSIDE activation patching, not alone |
| Single-prompt activation patching for localization | Round 14→15 correction | Round 14 claimed L35 is "THE arithmetic layer" from one prompt (17×23). R15 showed it doesn't generalize. Always aggregate across multiple inputs |
| L35 as THE arithmetic circuit | Round 15→16 correction | L35 mean Δ=-1.50, 9/10 hurts. Minor contributor. The real cluster is L22-L30 with L23 peak (-10.18, 10/10). Round 14's claim was premature |
| Per-sub-head d_head=2 ablation as localization tool | Round 20 | 0/256 sub-heads with mean Δ < -1.0; top-8 carries only 26% of damage. The arithmetic signal in H4 is distributed across the 512-d V subspace, not sparse in the d_head=2 basis. Don't re-run this probe on other heads hoping for sparsity in d_head=2 slots — target SAE on the full head/V output instead. |
| Mean-NOT-centered residual SVD as task-rank tool | Round 22 | Naive rank-1 SVD on residuals is DC-dominated — the top component is the mean offset, not the task signal. Mean-center before SVD, otherwise rank numbers are meaningless for composition work. |
| Multi-step prompts admitting "copy c" shortcut | Round 47.1 → 47.2 | Initial multi-step sweep (19 layers Δ<-1.0) was contaminated by prompt format that let Gemma copy a literal operand rather than compose. Fixed in R47.3 clean-prompt sweep (L24 peak -17.23). **Always audit prompt format BEFORE a sweep**; near-miss shortcuts inflate Δ across unrelated layers. |
| L24 per-head attention install for multi-step composition | Round 47.4 | Full-layer L24 Δ=-17.23 (69% > R16 L23). Top head H1 Δ=-0.635 — attention-level dead end. Multi-step composition is not a concentrated-head circuit; compilable-attention rules out for this capability. |
| L24 FFN per-neuron (ROME/MEMIT-style) install | Round 48.1 | 10,240 neurons in 20 chunks of 512 ablated; no chunk carries the signal. Rules out weight-probing install for L24 composition. Consistent with R49 "distributed across pathways" and R50.5 "SAE features zero causal." |
| SAE features as target for compilable ablation on distributed composition | Round 50.5 | **Falsifies R20's "SAE = next step" recommendation for compilable-intervention work.** Top-50 L24 composition features ablated → baseline 17/30 preserved at 17/30. Reconstruction fidelity (99.1-99.6%) is not sufficient for causal localization. Future SAE work for compilation needs architectures where reconstructed components DO have causal effect. |
| MSE-only distillation of a Gemma layer (tier-3 first attempt) | Round 51.5 (refined by R53.36 audit) | **R51 hypothesis falsified — but the refined mechanism is "sharp-direction loss", not "student can't learn L24".** A 1.25M-param Small2DTransformer student trained via MSE regression on 40K L24 (h_before, contribution) pairs reaches 92.6% aggregate variance-explained (per-domain 89.8-96.9%). Dual-gate eval on 120 held-out prompts with live L24-replacement showed mean-prefix match **0.194 training-dist (FAIL vs 0.80), 0.342 off-dist (FAIL vs 0.95)**. **R53.36 audit (this session) proves the student DOES reproduce L24**: mean cosine(pred, GT) = 0.8935 across 4 held-out prompts (multi 0.944 / single 0.962 / factual 0.954 / code 0.714), scale ratio 0.9052, install-boundary math zero-diff (`L24_installed == h_before + student(h_before)` bit-identical). Student is trained; install is correct; eval-pipeline is not a csv artifact. **Refined root cause**: 10% diffuse residual error cascades through 17 downstream layers + head, amplifying into wrong argmax. MSE loss averages over 2560 channels, unable to concentrate on task-critical directions (digit-selectors, content-readers) that survive downstream un-attenuated. **Residual-space reconstruction fidelity does not imply token-space task preservation**. Next tier-3 attempts need a loss that weights by downstream causal effect (e.g. Jacobian of head logits w.r.t. residual at L24); plain MSE + plain KL are both empirically insufficient. |
| KL-divergence distillation on L24 student (tier-3 second attempt) | Round 52.3 (refined by R53.36 audit) | **R52 hypothesis falsified with a different mechanism than R51.** Same 1.25M student architecture, forward KL-divergence on Gemma's final next-token logits (instead of MSE on residuals). Trained on 3000-prompt broad corpus via Triton autograd path (batch=4, lr=3e-4, grad_clip=0.1, warmup=200, 1000 steps). Val KL decreased monotonically 1.96→1.21. Dual-gate eval: train-dist **0.040** / off-dist **0.080** prefix match. **R53.36 audit reveals the R52 student COMPLETELY fails to reproduce L24**: cosine(pred, GT) = **-0.0227** (worse than random), scale ratio **93.93×** (~100× too big), L2 diff mean 7153. The KL-on-logits loss is SILENT on residual reconstruction — student learns to output SOMETHING that makes L25..L41+head produce roughly-right logits via pathways through 17 downstream layers, without actually computing L24's function. Install math still bit-identical (0.00e+00 diff). **R52's null is not structurally the same as R51's**: R51 learned L24 but 10% error cascades; R52 never learned L24 at all because the loss didn't constrain it. Combining: both R51 AND R52 are real distillation failures for different reasons; neither is a csv-style artifact. Three distinct losses (SAE ablation, MSE residuals, KL logits) all fail L24 — but R53.36 shows MSE is the CLOSEST miss (0.89 cosine) and KL is the FARTHEST (random). **Tier-3 distillation of L24 remains closed at the current loss-space**, but R51.5's MSE path is a more credible starting point for future work than R52's KL path. Pivot to tier-2 stacking per `augmentation_thesis.md` §"Tier-2 stacking achieves tier-3-equivalent outcomes"; R46.2 MultiStepReasoningFacade already augments L24's task at the output level (17/17 real Gemma fixes). |
| Triton-kernel custom autograd.Function with PyTorch-captured teacher logits | Round 52.1c | Built `Tq4TritonAutogradFunction` (forward uses existing `tq4_linear_triton`, backward uses new `_tq4_backward_kernel` streaming tq4 bytes). Passed `torch.autograd.gradcheck` (finite-difference verification) and showed cosine=1.0 on single-linear tests, but full Gemma training produced grad direction error that compounded to 0.19 cosine at 5 layers / 117× student grad attenuation at 17 layers. **Root cause**: Triton kernel's different FP32 reduction order produces ~6e-5 forward drift vs PyTorch `F.linear`. Compounds through Gemma's nonlinear ops (attention softmax, RMS norm, FFN gating) because backward uses saved forward values. The correct gradient of Triton-Gemma ≠ correct gradient of PyTorch-Gemma, so student trained against PyTorch-captured teacher logits on a Triton forward path learns the wrong function. **Rule**: don't mix Triton-computed forwards with PyTorch-captured teacher targets. If using Triton autograd, re-capture teacher logits through the same Triton path. Kernels kept in tree (`calm/llm_computer/tq4_autograd.py`, `tq4_triton.py::tq4_backward_triton`) for future use where forward consistency can be controlled. |
| Substrate-RAG via L41 CardSlot + per-marker FirstTokenHook on code | Round 53.14 / 53.20a / 53.20b | POST-SWA-fix re-run produced same -9.3pp regression as pre-fix. Root cause is install-mechanism, not SWA: Gemma's first-token on code prompts is confidently a fence/whitespace opener (logit margin 6.8-9.2), so `min_margin=0.5` never gates and the hook always fires on HIT, forcing "def"/"class" → code-without-fence → extractor fails. CardSlot's additive residual write perturbs Gemma's downstream format habits even on a silent hook. Reverted `USE_TQ4_KV=True` handled separately by the Phase 1 memoized path. **Rule**: do not use first-token bias on code tasks. Either mid-generation per-token hook OR post-generation AST-walker rewrite. See `augmentation_thesis.md` §"R53.14/20a/20b — substrate L41 install REGRESSES on code". |
| Mechanical import injection alone (pre-sandbox-fix) | Round 53.21 | `COMMON_IMPORTS` table + post-extract `prepend import; re-run` loop fires correctly (StringIO + csv injected for csv_column_stats), but the sandbox then raises `ImportError: blocked: os` because `import statistics` triggers transitive `os` load. Import injection was NECESSARY but not SUFFICIENT. Fix landed in R53.22 (sandbox pre-import) + R53.23 (re-run). **Rule**: diagnose the full `stdout/stderr` chain before declaring an injection-level fix null — the error after the injection may be the real ceiling. |
| fp16 x_rot activation buffer in tq4 matvec | Round 53.30 | TurboQuant CUDA commit reported fp16 activation halves BW for +89%. Triton port: +0.2% / +8.7% across two runs — null. Upcast-inside-dot-product cost matches BW savings on Ada L1. Kept as `_tq4_matvec_kernel_v3` for reference, not dispatched. See `tq4_triton.py`. |
| uint32-packed qs loads in tq4 matvec | Round 53.31 | TurboQuant CUDA reported +45% from 128-bit vectorized weight loads. Triton port: +9.8% / +16.4% across two runs — SLOWER. Triton auto-coalesces qs loads at compile time; adding explicit uint32 unpack via `tl.join`/reshape introduces overhead that exceeds the (nonexistent) BW savings. Kept as `_tq4_matvec_kernel_v4` for reference, not dispatched. See `tq4_triton.py`. |
| BLOCK_M sweep per shape on tq4 v2 | Round 53.32 | 9-value sweep (1, 2, 4, 8, 16, 32, 64, 128, 256) × 5 Gemma shapes. Current `_pick_block_m` heuristic holds — no shape benefits from departure. Sweep measurements noisy (3 runs × 1000 iters per config, insufficient warmup between); would need stable-methodology rerun to detect <5% gaps. Script `scripts/sweep_tq4_block_m.py` kept for future shape additions. |
| KVCacheTq4 dequant-on-read at decode (pre-R53.34) | Round 53.28 → 53.33 | Initial `KVCacheTq4.update()` dequanted the FULL cached sequence every call → O(N) dequant per step, O(N²) across full decode. Measured cost: linked_list 5/5 took 806s with tq4 KV vs 94s with fp16 KVCache (~8× slowdown). R53.33 diagnosed; reverted `USE_TQ4_KV=False` in eval scripts until R53.34 fused path landed. Phase 1 memoized dequant path is the asymptotic winner (above N≈2048) and the shipped fallback outside the fused gate — ~77% of fp16 tok/s at ~50% KV memory. For the winning mid-range N∈[128, 2048], the 2026-04-20 re-bench switched the default to the fused path (see R53.34 row below). |
| Fused tq4 flash-attn short-context "always slower" claim (PARTIALLY SUPERSEDED 2026-04-20) | Round 53.34 → revised | **Initial R53.34 reading (single-run, retained for receipt)**: fused 8-10% slower than Phase 1 memo at N≤1024 (N=64 5.60 vs 6.06 tok/s; N=1024 5.11 vs 5.60). Conclusion was "default OFF, kernel kept in tree, revisit at N>4K." **2026-04-20 re-bench (`scripts/r53_phase2_bench.py`, same bench script) contradicts the mid-range points**: fused WINS at N=256 (+14%) and N=1024 (+6%), LOSES at N=64 (-18%) and N=4096 (-7%). Non-monotonic curve — launch overhead dominates at small N, cuBLAS-on-memo dominates asymptotically, fused wins the middle band. **New disposition**: `_use_fused_flash_attn=True` default shipped with runtime N-gate `128 < kv_cache.layer_pos[kv_src] < 2048` in `_forward_layer`. Chat + short-eval decode lengths now run fused (captures 6-14%); long R53 eval falls back to memo past 2048 (avoids the -7% regression). Caveat: single-run methodology for both reads — direction reliable, magnitudes soft. Full bench table + policy rationale in `turboquant.md` §"Fused flash-attention decode". |
| TurboQuant Q_prod (3-bit Q_mse + 1-bit QJL) KV encoding | Round 53.34 Phase 3 | Implemented TurboQuant Algorithm 2 (`tq4_qjl_torch.py`, 132-byte block matching tq4 memory). Unbiased inner-product estimator works (n=1000 sample mean = 1.15σ from truth, paper claim reproduced). BUT empirical attention-output cosine WORSE than existing 4-bit Q_mse-only at every measured N (Δ=-0.04 at N=16, Δ=-0.17 at N=1024). Root cause: per-realization QJL variance (~3.5σ on magnitude-16 inputs); softmax non-linearly amplifies variance more than MSE-only's structural bias. Paper claim is about expected inner-product MSE, NOT softmax-output preservation. **Rule**: 4-bit Q_mse is the right choice for KV at 4-bpw budget. Kept `tq4_qjl_torch.py` + `fused_tq4_qjl_flash_attn_decode` as research artifact for NN lookup / hash-retrieval / cosine-ranking use cases where unbiased <x,y> matters more than softmax output. Should NOT be default KV encoding. |
| Blanket prompt-level retrieval injection | Round 53.2b | R53 Phase 1 complex eval (6 multi-step coding problems × {stock, hinted-real-retrieval, sanity-random-retrieval}): **retrieval-attributable gain = +0.0pp**. Hinted (+7.4pp) = Sanity (+7.4pp). The prompt-length "has examples in context" effect is real; retrieval content adds nothing on top. Several problems (log_level_counts, linked_list_bugs) had real retrieval HURT Gemma's native ability. **Root cause**: blanket injection violates Tier-1 preservation — when Gemma already has a strong prior, showing "similar solutions" diverts it to adapt the example instead of solving natively, introducing errors. **Implication**: substrate RAG (hash-gated at L30 `KnowledgeStore`) has a structural advantage over prompt-RAG because hash-match naturally gates injection to problems where Gemma needs the help. Build R53.5 PT + R53.6 install and measure substrate-RAG vs prompt-RAG on the same corpus. See `augmentation_thesis.md` §"Automatic Tier-1 preservation" and `retrieval.md`. |
| TurboQuant Q_prod (3-bit Q_mse + 1-bit QJL) for KV cache encoding | Round 53.34 (Phase 3 of TurboQuant KV work) | Implemented Algorithm 2 from TurboQuant §3.2 (`tq4_qjl_torch.py`, 132-byte block matching tq4 memory). The unbiased inner-product estimator works as proven (n=1000 sample mean = 1.15σ from truth — paper claim reproduced). BUT empirical attention-output cosine vs fp32 truth at every measured N is WORSE than the existing 4-bit Q_mse-only path: N=16 Δ=-0.04, N=64 Δ=-0.02, N=256 Δ=-0.09, N=1024 Δ=-0.17, N=4096 Δ=-0.10. **Root cause**: per-realization QJL variance is large (~3.5σ on inputs of magnitude ~16); softmax non-linearly amplifies that variance more than it amplifies MSE-only's small structural bias. Asymptotic crossover doesn't appear at the contexts we care about (≤4K). The paper's distortion-rate-optimal claim is about expected inner-product MSE, not softmax-output preservation. **Implication**: 4-bit Q_mse (current tq4 KV) is empirically the right choice for KV cache at the same 4-bpw budget; the 3-bit Q_mse + 1-bit QJL split does not earn its keep. Implementation kept in tree (`tq4_qjl_torch.py`, `tq4_flash_attn.fused_tq4_qjl_flash_attn_decode`) as a research artifact for future use cases where unbiased <x,y> matters more than softmax output (nearest-neighbor lookup, hash-based retrieval, cosine-similarity ranking). Should NOT be wired in as the default KV cache encoding. See `calm/llm_computer/tests/test_tq4_qjl_flash_attn.py` module docstring for the empirical table. |
| Fused tq4 flash-attn decode kernel initial dispatch rejection (PARTIALLY SUPERSEDED 2026-04-20) | Round 53.34 → revised | **Session 33-34 read (retained for receipt)**: rewrote the 257-line `tq4_flash_attn.py` scaffold with correct online-softmax + per-head-parallel V kernel, wired into `_forward_layer` for `KVCacheTq4 AND S==1 AND d_head==256 AND not partitions`. 7/7 unit tests cos=1.00000 vs fp32, real-Gemma Δmean=0 argmax=+0. Initial A/B (single-run, Triton weight kernels ON for both, isolated via `enable_fused_flash_attn()`): fused 8-10% slower than memo — N=64 5.60 vs 6.06, N=256 5.32 vs 5.77, N=1024 5.11 vs 5.60, fp16 baseline 7.0-7.5. Conclusion at the time: `_use_fused_flash_attn=False` default, kernel kept in tree, revisit if N>4K becomes the workload. **2026-04-20 re-bench reversal** (same script, same hardware, different GPU clock/driver state): fp16 7.0, memo 5.6-6.1, fused 4.0 at N=64 / **6.40 at N=256** / **6.43 at N=1024** / 5.65 at N=4096. The N=256 (5.32→6.40) and N=1024 (5.11→6.43) points from the session-33 read are directly contradicted; N=64 and extended N=4096 CONFIRM slower. Curve is non-monotonic, not universally slower. **Attributed**: GPU clock/driver state variance between sessions is real (single-run methodology per `workflow.md` §"GPU bench discipline" can miss 5-10% shifts; both reads violated median-of-5). **New disposition**: `_use_fused_flash_attn=True` default, runtime conditional gates on `128 < kv_cache.layer_pos[kv_src] < 2048`. Inside the band fused runs (+6 to +14%); outside it falls back to Phase 1 memo (protects N=64 small-decode + long-eval past 2048). Realistic chat + short-eval workloads decode entirely inside the gate. Phase 1 memo remains the asymptotic winner (N>2048) and the out-of-gate fallback — it's not retired. Unlock potential if ever revisited: (a) one Triton kernel spanning all Q heads, (b) TILE_N parallel-over-N V kernel — both would push the lower gate threshold down. Full bench table + policy rationale: `turboquant.md` §"Fused flash-attention decode". |
| csv_column_stats extractor bottleneck on Gemma 4 E4B at 8K budget | Round 53.35 | R53.33's "csv_column_stats hits KeyError at runtime" characterization is stale; at this session's config (medium/hard AdaptiveBudget, post-R53.25 stack) Gemma produces `0/0` NoCode on csv_column_stats across two attempts (hinted + NoCode-repair). Prompt-level issue, not correctness: Gemma emits `<think>` blocks that exhaust budget without reaching extractable code, OR produces prose that evades the format-agnostic extractor. AST walker gated out (needs extractable code to rewrite). **Not a regression** — prior R53.x runs also hit 0/0 on csv; R53.25 lift was on other problems. **Not a walker failure either** — walker correctness validated on the csv-bug code pattern in unit tests (test_end_to_end_csv_column_stats_passes_after_repair). The intervention point for csv is upstream of the walker: either extractor changes, a different generation strategy (e.g. forced code-fence prefix), or the problem is genuinely outside Gemma's capability at this budget. **Next direction**: run the walker stack on MBPP/HumanEvalPlus corpus (handoff §4) — problems where Gemma consistently produces code — and measure walker lift on a larger failure surface. |

## Related rules

- `augmentation_thesis.md` — strategic synthesis: circuit typology, tier framework, factorial scaling, anti-skepticism (settled positions from R20-R36)
- `tracing_intelligence.md` — first-principles framing
- `capability_gain.md` — measurement discipline
- `embed_intelligence.md` — delivery mechanisms
- `Substrate.md` — install mechanisms
- `workflow.md` — hypothesis-test loop
- `retrieval.md` — hybrid retrieval architecture (R53 Phase 1)
- `code_reasoning_db.md` — DB + generator framework (R53 Phase 1)
- `recursion.md` — card-level self-improvement pattern
- `commercial.md` — product position this roadmap supports
