# Tracing Roadmap — what's compiled, what's next

**Part 1**

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
| `ast_repair` walker | **36/36 unit** + token_bucket 0/0 → 5/5 + csv_column_stats 0/0 → 8/8 on Gemma | **Three** deterministic rewrites (shadow rename, dict-key synonym, syntax repair with bracket-mismatch + insert-before-colon) driven by runtime error text | Post-generation AST rewrite, no decode-time install |
| **Tracing methodology** | Validated on Gemma 4 E4B (Rounds 13-50.6) | Activation patching + per-head + Q/K/V decomp + per-sub-head + forced-attention validation + SAE (TopK) reconstruction | 7 capabilities mapped (+ multi-step composition); 3 causal validations (R28, R42, R43); hub-sharing proven at L23 H1/H4 (5-for-1 ROI); SAE reconstructs L24 composition at rank-50 in feature basis but top features have zero causal effect (R50.5) — interpretability-without-causality gap |

### Facades built

| Facade | Card(s) | Rounds | Honest result |
|---|---|---|---|
| `MathAdditionFacade` | PT + adder_tiny | 6-9 | Format coercion on task Gemma already solved. First reusable class (Round 8). |
| One-off reasoning facade | PT + adder_tiny, recall card | 6, 7, 11 | R6/R7 were format coercion; R11 (multiplication) is the first real capability win. |
| `HubInjectionCard` (`calm/llm_computer/facades/hub_l23.py`) | L23 H1/H4 forced-attention | 44, 45 | Facade form of R43 intervention; bit-identical. Runtime Q/K detects natural top-position; no per-task hand-dispatch. Serves arithmetic + SV + comparison + counting + multi-step (5-for-1). `generate()` verified 5×12 decode tokens (R45). |
| `MultiStepReasoningFacade` | NL infix parser + `safe_eval` | 46.1, 46.2 | 17/17 real Gemma fixes, 0 regressions. N-op extension of R11 step-through digit bias. Parses NL infix, routes values through `safe_eval`. |
| `ast_repair` walker (`calm/llm_computer/facades/ast_repair.py`) | `ast` stdlib | R53.35 / R53.35v2 | Post-generation AST rewrites for R53.33 deterministic failure modes + Gemma-output bracket bugs. **Three rewrites**: shadow rename (TypeError callable), dict-key synonym (KeyError, curated table), syntax repair (bracket-mismatch via Python error-offset + insert-before-colon for `for/if/def` lines). Raw: 36/36 unit. User-facing: **token_bucket 0/0 → 5/5** via shadow_rename in 0.9s; **csv_column_stats 0/0 → 8/8** via syntax_repair (one missing `)` before `:` on line 42, commit `c81feb6`). lru_cache_class 9/9 preserved (no regression). date_validation_chain 10/12, log_level_counts 6/6 — walker no-op on clean Gemma output. Correct tier-2 intervention for code repair: auditable, mechanical, zero LLM cost. |
| `NumberTheoryFacade` (`calm/llm_computer/facades/number_theory.py`) | decode-path compute | R53a 2026-04-22 | 3rd shipped decode-path facade (mod/GCD/LCM). 15/15 vs baseline 8/15 (Δ=+7, 47% lift, 0 regressions, commit `69279d4`). Caught + fixed leading `▁`-strip (id=236743) + POST_BIAS_BUDGET=4 discipline (scope: number_theory + numeric_encode + all recursion-generated, NOT multi_step/base_conversion). |
| `NumericEncodeFacade` (`calm/llm_computer/facades/numeric_encode.py`) | decode-path compute | F2 2026-04-22 | int→hex/binary/octal. 12/12 on chain corpus (commit `5ee61a5`). First facade with LETTER-or-digit answer (e.g. "DEADBEEF"). |
| `Icd10RecallFacade` (`calm/llm_computer/facades/icd10_recall.py`) | decode-path tier-3 text recall | R60a + F1 2026-04-22 | 72,748-code CMS DB. **8/30 → 26/30 (+18, 67% lift, 0 regressions, commit `afc0220`).** First shipped tier-3 capability via decode-path (not CardSlot). Generalizes step-through bias from integer-answer to arbitrary Gemma BPE text sequences. 4 edge codes (T44.6X4D, T40.5X4D, V80.22XA, W10.0XXA) with unusual internal tokens resist all intervention — F1 code-echo detect+retry infrastructure added (`8ba151d`) but final fix needs prompt reshape or pure-DB bypass. |
| `PlannerFacade` (`calm/llm_computer/facades/planner.py`) | decode-path orchestrator | R70a + F2 2026-04-22 | First-match-wins classify over 5 specialist facades (icd10 → base_conv → numeric_encode → number_theory → multi_step) + chain detect for "X in hex/binary/octal". 20/20 route + 18/20 answer single corpus (`956a3ae`). 12/12 route + 12/12 answer chain corpus (`5ee61a5`). Option A (runtime-glue) shipped; Option C (compiled planner card) deferred. |
| Auto-generated facades via `calm/llm_computer/recursion.py` (`factorial_auto`, `fibonacci_auto`, `combinations_auto`, `permutations_auto`, `power_auto`, `next_prime_auto`) | Level-1 auto | F3 + M1 2026-04-22 | Substrate generates its own facades from `FacadeSpec` via template+ast.parse+CALM oracle validation. **F3 demo**: 5/10 → 10/10 on factorial + fibonacci (commit `3274659`). **M1**: 12/20 → 20/20 on combinations/permutations/power/next_prime (commit `5173745`). Three-gate CALM discipline (oracle validate → ast.parse → live A/B) keeps loop drift-free. |
| Meta-synthesized facades via `MetaFacade.from_oracle(fn_name, arity)` (`factorial_meta`, `combinations_meta`, `gcd_meta`, `lcm_meta`, `fibonacci_meta`) | Level-2 meta-synthesized | M2 2026-04-22 | Spec synthesized from just (fn_name, arity) — canonical NL pattern library per arity. 4/15 → 15/15 (+11, commit `5173745`). Spec authorship moved from human to substrate while CALM gates remain intact. |

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
