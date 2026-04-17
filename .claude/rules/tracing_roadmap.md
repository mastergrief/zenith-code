# Tracing Roadmap — what's compiled, what's next

Concrete inventory of (a) compiled cards that exist, (b) cards that
would be easy wins but haven't been built yet, (c) the research path
for harder capabilities. Updates as rounds ship.

See `tracing_intelligence.md` for the first-principles framing and
`capability_gain.md` for what counts as a validated win.

## State today (after Round 20)

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
| **Tracing methodology** | Validated on Gemma 4 E4B (Rounds 13-20) | Activation patching + per-head + Q/K/V decomp + per-sub-head | Arithmetic circuit localized to L22-L30; R20 proved signal is distributed across H4's sub-heads, not localized to a few |

### Facades built

| Facade | Card(s) | Rounds | Honest result |
|---|---|---|---|
| `MathAdditionFacade` | PT + adder_tiny | 6-9 | Format coercion on task Gemma already solved. First reusable class (Round 8). |
| One-off reasoning facade | PT + adder_tiny, recall card | 6, 7, 11 | R6/R7 were format coercion; R11 (multiplication) is the first real capability win. |

### Install mechanisms verified

- CardSlot + preservation masking: Rounds 4-11
- Token-embedding projection at layer 33: Round 9
- Step-through digit bias: Round 11 (multi-token answers)
- VerificationHook with min_margin: Rounds 6-11
- Save/load round-trip of full substrate: Round 1 (bit-identical logits)
- CUDA Graphs × FP32 layers compat: Round 2 (4.29× speedup preserved)

## Gemma 4 E4B tracing findings (Rounds 13-20)

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

**The circuit:** `L23 attn_v (KV group 1, 512-d)` → H4's Q pattern
reads it → ~2.6M-param V-projection is the concrete localization of
arithmetic content on a 5B-param model. Compact target for SAE +
ACDC work.

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

Labor: weeks to months per circuit, pending interpretability tools.
These are the long-horizon bets that eventually give us the LM's
capabilities as compiled cards.

| Target | Prerequisite | Notes |
|---|---|---|
| **Deepen L22-L30 arithmetic circuit** | R13-20 localized to layer + head + V projection; R20 showed signal is distributed across H4's sub-heads. Next: SAE on H4's 512-d V output (or full L23 residual) across an arithmetic corpus; ACDC for cross-layer connections. | IMMEDIATE next step. Target is the 1024-d V of KV group 1 (or H4's 512-d slice); features are distributed directions in V-space, not sparse in sub-head basis. |
| **Generalize tracing methodology** | Run R16-style ablation sweep on OTHER capability targets (syntax, anaphora, factual recall) | Tests whether arithmetic's clean localization was special or the methodology is general |
| Induction heads | Identify heads doing in-context copying on Gemma 4 E4B | Generic capability used by many tasks |
| NL parsing circuits | Circuit probing for syntax patterns | Replace PT's learned mechanism with a compiled one |
| Factual retrieval circuits | ROME / MEMIT-style probing | Replace / augment KnowledgeStore's step-function recall |
| Entity tracking | Feature directions for "who's the subject of this sentence" | Useful for word problems |

Prerequisite research track:
1. **Sparse Autoencoders on Gemma 4 E4B** — train SAEs on residual
   activations across layers. Extract 10K-100K interpretable
   features.
2. **Automated Circuit Discovery (ACDC)** — for each target
   capability, run ACDC on a corpus of examples to find load-bearing
   components.
3. **Feature labeling** — human or LM-assisted naming of SAE features
   based on activating examples.
4. **Circuit-to-IR translation** — convert identified circuits into
   `gate_graph.py` nodes. Largely mechanical once the circuit is
   understood.

None of these are in scope for near-term rounds. Flag them as the
multi-quarter R&D direction.

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

## Related rules

- `tracing_intelligence.md` — first-principles framing
- `capability_gain.md` — measurement discipline
- `embed_intelligence.md` — delivery mechanisms
- `Substrate.md` — install mechanisms
- `workflow.md` — hypothesis-test loop
- `commercial.md` — product position this roadmap supports
