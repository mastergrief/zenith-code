# Recursion — card-level self-improvement via CALM oracle

The substrate's killer property is that **cards can build cards**.
Each level of recursion is guarded by CALM's deterministic verifier,
so improvements compound without the bias-amplification failure mode
of self-training on a learned judge (Self-Instruct, RLAIF, etc.).

Already shipped end-to-end for facts (`calm/llm_computer/auto_upgrade.py`
+ `scripts/gemma_learning_loop_demo.py`): **5/5 wrong → 5/5 correct**
after one self-improvement cycle. Generalization to all card types is
mechanical.

## Level 1 — card self-distills its own domain

```
CardV1 (trained on seed corpus + DB)
   ↓
install on Gemma (CardSlot or in-attention)
   ↓
card generates candidate outputs on unseen prompts
   ↓
CALM + sandbox verify each intermediate value + final answer
   ↓
keep verified (input, verified_output) pairs
   ↓
CardV2 trained on seed + verified-new
   ↓
repeat: each iteration absorbs cases the previous version solved
```

This is what `auto_upgrade.py` does for recall cards. Extended to the
code PT (R53.5) or future MultiStepCoding card, the loop becomes:

- PT generates reasoning trace for a new problem
- CodeVerifierFacade runs extracted code against tests
- Passing → add to corpus
- Failing → CALM generates corrected trace → add to corpus
- Retrain PT on expanded corpus

Seed cost: human-curated 222 examples (generators). After N iterations,
the card owns tens of thousands of verified traces with zero manual
labeling.

## Level 2 — cards that build cards

A **MetaCard** receives: "here's a capability Gemma fails on" and
emits: a full card spec (probing target, IR layout, training recipe).

```
MetaCard inputs:
  - failure traces from CALM (what capability is broken)
  - probing results (where the circuit lives)
  - circuit classification (concentrated / cooperative / diffuse)
    ↓
MetaCard outputs:
  - IR template: which primitives (TokenEmbed / LookUp / ReGLU)
  - install pattern: in-attention / CardSlot / HubInjection
  - training recipe: data generator + verification oracle
    ↓
card compiled → installed → verified → deployed
```

Humans design the meta-process once; the system generates specific
cards on demand when new capability gaps appear.

The pieces for this already exist independently — probing
methodology, circuit classifier, IR compiler, install patterns. Meta-
card is the glue that automates their composition.

## Level 3 — meta-cards that build meta-cards

Same pattern one level up. The MetaCard itself has failure modes
(e.g. misses certain circuit typologies). A MetaMetaCard watches for
these failures, designs a new MetaCard variant that handles them.

Same CALM-discipline keeps the recursion from diverging — at every
level, the "was this better?" oracle is deterministic (CALM tests,
circuit mapping is measurable, card compilation is exact).

## Why this is safe where Self-Instruct / RLAIF fails

| Approach | Oracle | Failure mode |
|---|---|---|
| Self-Instruct (Wang 2022) | the generating model itself | amplifies biases, reinforces hallucinations (student ≈ teacher) |
| RLAIF / constitutional AI | judge LLM | judge bias leaks into student |
| Evol-Instruct | LLM scoring | same bias amplification |
| **Substrate card recursion** | **deterministic CALM tests + compiled verification** | **cannot amplify what's verified wrong** |

Every card in the recursion chain is gated by running its output
against tests (or against compiled verifiers for specific domains).
Whatever survives has **passed objective correctness checks**, not
"looked good to another LLM". Drift-free on compiled domains.

For open-ended creative tasks with no verifiable correctness — no
card is trained, Gemma's probabilistic output is preserved (Tier 1).
The system never claims more than it can prove.

## Capability completeness as a fixed point

As recursion continues at all three levels:

- Card library grows (more domains covered)
- Each card covers more of its domain (self-distill fills gaps)
- MetaCard gets better at spotting which domains need cards
- MetaMetaCard gets better at designing MetaCard variants

Asymptotically: **for every task with a verifiable success criterion,
the substrate has a card that solves it exactly.** Tasks without
verifiable criteria fall to Gemma's native output with CALM claim-
verification on anything factual.

This is a different performance profile from scaling a monolithic
model:

- Monolithic scaling: statistical average improves, tail failures
  persist (including arithmetic errors at 100B+)
- Substrate recursion: compiled domains become **provably correct**;
  domains without oracles stay probabilistic but with per-claim
  verification overlay

The commercial positioning follows: regulated industries want the
"provably correct on compiled domains" half. General-purpose gets
the probabilistic half with claim verification. Same substrate,
different card stack per customer.

## Concrete state

### What's shipped

- `calm/llm_computer/auto_upgrade.py` — fact-level recursion
  (`AutoUpgradeEngine.commit()` compiles corrections into recall card)
- `calm/llm_computer/persistent_knowledge.py` — `KnowledgeStore` with
  `add_correction(key, value)` + `build_recall_model()` + save/load
- `scripts/gemma_learning_loop_demo.py` — Level 1 end-to-end on Gemma
  substrate (2+3, 4+1, 3+2, 5+1, 2+4 mod 8 → 5/5 wrong → 5/5 correct
  after compile + install + persist)

### What's next (R53.5 + R53.6)

- Train `copy_code_best.pt` PT on 8970-example DB + generator traces
  (Level 1 seed)
- Install at L24 via CardSlot; Install KnowledgeStore at L30
- Run self-distillation loop: PT generates plans → CodeVerifierFacade
  validates → passing plans absorbed into next PT training round
- After N rounds, measure: does the PT cover multi-step compositions
  the initial version missed? (expected: yes, monotonic growth)

### What's further out (Level 2+)

- Wire probing tools (`tracing_roadmap.md`) to auto-classify new
  failure modes on prod queries
- Auto-generate card specs from the classification (MetaCard v0)
- Closed loop: user prompts → failures logged → MetaCard builds new
  card → installed → user next prompt works

## Related rules

- `augmentation_thesis.md` §"Auto-upgrade loop" — the factorial
  scaling property that makes recursion economically viable
- `Substrate.md` §"Auto-Upgrade Loop" — technical install path for
  compiled recall cards
- `calm.md` §"Auto-Upgrade Loop" — CALM's role as oracle
- `capability_gain.md` — measurement discipline per recursion step
- `probing_methodology.md` — circuit mapping (Level 2 input)
- `commercial.md` — verifiable-augmentation-as-product positioning
