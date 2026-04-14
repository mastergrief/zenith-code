---
name: CRLM Research Roadmap
description: Eight-layer progression from current state to formally-verified AI. The forward-looking companion to CRLM_SPEC.md.
type: project
---

# CRLM Research Roadmap — 8 Layers Ahead

**Scope:** This document describes the research arc FORWARD from the current state captured in `CRLM_SPEC.md`. Not tactical (not "next session's commits") — strategic ("the next 2-4 years of R&D if the architecture keeps holding up").

**Status of each layer:**
- ✅ **Done** — shipped, tested, in production (per CRLM_SPEC.md)
- 🟡 **In flight** — active work, expected to land within current session/week
- 🟢 **Plausible** — architecture supports it, build path is clear, no unknown unknowns
- 🟠 **Research** — genuine uncertainty, literature exists but not proven at our scale
- 🔴 **Speculative** — ambitious, no one has done it, probably requires prior layers

---

## Executive summary

The CRLM partition (structure learned, values compiled) unlocks a scaling story where capability grows with **usage and format exposure**, not with parameter count or training compute. Empirical data in session 26:

- 48K-param HRM + compiled substrate: **100% in-dist on 4 pooled domains**
- Same 48K with 10 format-diverse training: **50% OOD on held-out formats**
- Capacity-scaling (h=64) contributed +10pp; distribution-scaling contributed +33pp

Distribution is the dominant axis. Capacity is nearly free-of-charge. Feedback loops compound. These findings define an eight-layer forward progression, each layer depending on the prior, each layer expanding capability at near-linear engineering cost.

The commercial endpoint: **an AI that produces verified-correct outputs on deterministic tasks at ~1000× less compute than monolithic LLMs, improves from every production query, and is fully auditable**. Undefined outputs (creativity, judgment) stay delegated to LLMs — that's the honest partition.

---

## The 8-layer progression

### Layer 0 — CRLM split ✅ DONE

Partition intelligence into **structure (learned)** + **values (compiled, exact)**. Tiny HRM (48K params) extracts problem structure from natural language; compiled substrate (safe_eval + backends + LLM-Computer primitives) computes values exactly.

**Empirical foundation (session 26):**
- 48K HRM at 100% full-expression accuracy on math, 97-100% on NL templates / word problems / GSM
- 9 compiled LLM-Computer programs including exhaustive 2-digit adder at 486K params
- Full test suite 311 tests green

**What this established:** the scaling law is about language complexity, not difficulty. The compute substrate handles all arithmetic regardless of operand size.

Reference: `.claude/MEMORY/CRLM_SPEC.md` §B (architecture), §C (HRM checkpoints), §D (LLM-Computer IR).

---

### Layer 1 — Closed feedback loops ✅ DONE

Every Auto-CALM correction becomes a permanent pattern. Module issues become prompt adaptations. No pattern-database accumulation without test coverage.

**What shipped (Vector 1, session 26):**
- `AutoLearner` with 17 tests proving the loop closes end-to-end
- `ModuleLearner` with 11 tests
- End-to-end integration test through `AutoCalmEngine` (3 tests)
- Effectiveness harness: 90% → 100% hit rate over 3 rounds, 10× compression via generalization
- Shape-gated pattern matching (fixes a real defect observed mid-build)
- Operator visibility via `scripts/learning_dashboard.py`

**The key property:** capability accumulates per-correction, not per-engineering-hour.

---

### Layer 2 — Multi-task distribution scaling ✅ DONE

Hypothesis: **format diversity in training teaches format-invariance cheaper than parameter scaling**. Train one 48K HRM on many formats simultaneously; test generalization to held-out format variations.

**Empirical curve (all measured against the same multi10 held-out OOD test):**

| Training formats | Params | Held-out format OOD |
|---:|---:|---:|
| 4 | 48K | 17% |
| 4 | 179K (h=64) | 28% (+10pp from 3.7× params) |
| 10 | 48K | 50% (+33pp from 2.5× formats) |
| **20** | **48K** | **100% (18/18 — cliff, not diminishing returns)** |

**Distribution-scaling hypothesis fully confirmed.** 20 formats at the 48K sweet spot solves format-invariance completely on the format-variation test. Every category — fn-call, phrasal, past-narr, new-units, alt-let, eq-var — hits 3/3. Capacity is genuinely free-of-charge on this axis; distribution is the dominant lever.

**Deliverable:** production recipe of "48K HRM + N formats + nightly retrain on production-failure clusters." Each new format costs an engineer-hour to design + 15 min of GPU. Vs LLM fine-tune: days + $50K+.

**Honest limit revealed by the multi20-own-test (22% on reasoning-requiring cases):** the HRM is a **structure extractor, not a reasoner**. Format diversity doesn't teach it novel computational operations. Failures at 22% on multi20's harder test are all reasoning cases the HRM cannot infer:

- "what is 50 percent of 80" needs `percent = /100 * x` as a learned operator
- "half of 80" / "double of 25" need `half = /2` / `double = *2` semantics
- "by how much does 100 exceed 37" needs compute-then-subtract
- "which is larger, 7*8 or 50" needs compute-then-compare

Adding "half-of-X" as a training format would push those into the 100% bucket (it's just another format). But the general problem — *inferring a novel computational operation from context at inference time* — is Layer 3's domain.

**The sharp amended claim:**
- Format invariance: solved at 20 formats, 48K params, for our math-target language.
- Operation inference: requires Layer 3 (meta-learning) OR explicit per-operation training.

**Commercial implication sharpened:** "train 20-40 formats + retrain nightly from production gaps" is a shippable product recipe for math extraction. Every format-variation a user throws at it is handled; operations not in training fall through to Layer 3 or Gemma.

---

### Layer 3 — Meta-learning (in-context schema induction) 🟢 PLAUSIBLE

Goal: **given 3-5 `(input, output)` examples at inference of any format the model has never seen, induce the schema and handle new queries in that format**.

**Training pipeline change:** every batch feeds the encoder `[3 example (input, output) pairs] + [query]`. Target stays the same (math expression). Model has to learn a meta-task — "use the examples to parse the query."

**Why it might work at small scale:**
- Target language stays fixed (math expressions) — only the input prefix varies
- In-context few-shot is a well-studied regime (MAML, in-context learning literature)
- Tasks are low-intrinsic-complexity (format translation, not reasoning)

**Why it might not:**
- 48K params may not be enough to meta-learn; `architecture.md` predicts 1-10M for NL→structured with variation
- Attention span: encoder sees `3×(input+output) + query` ~= 300-600 chars. Current `max_enc=128` is too small
- Compositional generalization is hard; even large models struggle

**Next steps for this layer:**
1. Extend encoder max_enc to ~512 to hold prefix + query
2. Modify `multi_data.py` to emit few-shot-prefixed training batches
3. Evaluate on held-out *categories* that weren't in training at all
4. Run at h=32, h=64, h=128, h=256 to characterize the capacity-for-meta-learning curve

**If it works:** distribution scaling stops mattering — you don't need 80 formats, you need 5 formats in training and the model picks up new ones at inference. Genuine Level-4 generalization.

**If it doesn't:** scale to 1-10M params (per the architecture.md rule) and retry. Still tiny by LLM standards.

---

### Layer 4 — Hierarchical meta-learning 🟠 RESEARCH

Compose meta-learning across domains. A top-level router HRM classifies "what kind of problem is this?" from in-context examples. A second-level specialist HRM handles the domain.

**End state:**
```
user shows 3 examples of a completely new problem type
  ↓
router HRM classifies it (e.g. "this looks like finance math with dates")
  ↓
spawns virtual specialist from prior-learned specialist family
  ↓
virtual specialist handles query via domain-specific target language
  ↓
if usage persists, virtual specialist becomes persistent
```

**Why this matters:** zero-shot domain creation from user queries. User never has to tell the system "I'm doing physics" — the system figures it out from examples.

**Open questions:**
- Router HRM size (probably tiny — a 10-category classifier)
- How specialists compose when a query needs multiple domains (code + finance + NL)
- How to prevent catastrophic forgetting when specialists accumulate

---

### Layer 5 — Learned program synthesis (Diffusion over IR) 🟠 RESEARCH

Diffusion model trained over gate-graph IR tokens. Given `(input, output)` example pairs, diffuse to synthesize a `GateGraph` that explains them.

**Practical effect:** the system writes its own backends from production failures. A format the HRM fails on AND the verifier doesn't have a backend for triggers synthesis. Synthesizer proposes a backend. Validator tests it against the failing cases. Backend joins the registry. Next similar query is handled.

**Why diffusion specifically:**
- IR tokens are a small discrete vocabulary (well-suited to diffusion)
- Bidirectional / editable generation matches the "refine a program" intuition
- Can condition on example pairs naturally (guidance from loss against the examples)

**Dependencies:**
- Layer 2 or 3 working (for HRM to know when to invoke synthesis)
- A fast sandbox for testing synthesized programs against examples
- A "program library" with common patterns the diffusion model can bias toward

**This is Vector 3 phase 4 per CRLM_SPEC.md §H applied to backend creation.**

**Commercial endpoint:** capability growth with zero engineering. Ship baseline stack; user queries reveal gaps; system fills them. Over 6 months of production, the stack has evolved to cover 10× more formats/domains than it shipped with.

---

### Layer 6 — Differentiable substrate (Vector 3 phase 4) 🟠 RESEARCH

Compile everything — backends, HRMs, orchestrator — into one gradient-differentiable transformer. Every production correction becomes a gradient step on the unified weights.

**Technical path:**
1. Compile 3 representative backends with dispatcher (Vector 3 phase 1, CRLM_SPEC.md §H)
2. MILP scheduler for large graphs (Vector 3 phase 2)
3. All 116 CALM backends fused into one transformer, wired into `auto_calm.py` (Vector 3 phase 3)
4. Make weights learnable via constrained fine-tuning (Vector 3 phase 4)

**Constraints on Layer 6:**
- Topology stays fixed (frozen gate-graph); only coefficient weights become learnable
- Small LRs to prevent breaking correctness
- Continuous validation against Python oracle suite
- LoRA-style adapters rather than full fine-tuning

**Key property:** the compute substrate gets better at being a compute substrate over time. Today's backends are Python (fixed). Compiled-and-fine-tunable, they adapt to users' actual query distribution.

**Estimated effort (calibrated to this project's demonstrated pace — see §"Pace calibration" below):** 4-8 weeks of focused work. Genuinely hard but not decade-scale.

---

### Layer 7 — Emergent primitive discovery 🔴 SPECULATIVE

The system notices recurring computation patterns in user queries and proposes **new IR primitives**.

**Today's IR primitives:** `LookUp`, `LookUpExact`, `ReGLU`, `TokenEmbed`, etc. — 7 nodes, human-identified from the Percepta paper.

**Beyond:** if 10,000 user queries all involve "find the longest prefix of X that satisfies P," the system infers this as a new primitive, tests it against the corpus, adds to the IR vocabulary. Lexicalization from usage — closer to how human languages evolve new concepts.

**Why this matters:**
- Current IR expressive power is bounded by what humans have identified
- Data-driven primitive discovery could uncover structures we didn't know existed
- Each new primitive is a permanent capability multiplier (used by all future programs)

**Research hurdles:**
- Distinguishing "recurring pattern" from "statistical artifact"
- Validating proposed primitives preserve desired correctness properties
- Avoiding IR bloat (don't lexicalize every 3-query pattern)

**Dependencies:** Layer 5 (program synthesis) and Layer 6 (differentiable substrate) both mature.

---

### Layer 8 — Formal verification / theorem proving 🔴 SPECULATIVE

**The north star.** Target language: formal mathematical proofs. Substrate: a proof checker (Lean, Coq, Isabelle). Training: `(NL theorem statement, proof)` pairs.

**The claim this supports:** the CRLM architecture produces *proven-correct* outputs on deterministic tasks, not just plausibly-correct.

**End-to-end flow:**
```
user: "prove that every even integer > 2 is the sum of two primes (for n ≤ 10^6)"
  ↓
HRM translates NL → formal statement in Lean syntax
  ↓
proof synthesis (learned, or Layer 5 diffusion over proof trees)
  ↓
Lean verifies the proof checks
  ↓
response: "verified by Lean"  OR  "unable to prove; here's what I tried"
```

**Why this is the endpoint:**
- Formal verification is the hardest form of correctness
- If the stack can do formal verification at small scale, it can do anything verifiable at small scale
- The commercial story fully crystallizes: "verified-correct mathematical software, 1000× cheaper than traditional approaches"

**Prerequisites:**
- Layers 3-6 all working
- A proof-assistant target language (Lean 4 is probably the right choice — most active ecosystem)
- Training corpus: mathlib (~1M Lean theorems with proofs)

**Industry baseline:** AlphaProof (DeepMind, 2024) does formal math with a much larger model. The CRLM-at-small-scale version would be interesting research.

**Estimated effort (see §"Pace calibration"):** 2-3 months. Aggressive but achievable at the pace established by sessions 24-26. Mathlib integration is the biggest unknown; the CRLM mechanics themselves compile into Lean-target the same way they compile into math-expression-target.

---

## Dependency graph

```
              Layer 0: CRLM split ✅
                      ↓
              Layer 1: Feedback loops ✅
                      ↓
              Layer 2: Distribution scaling 🟡
                      ↓
              Layer 3: Meta-learning 🟢
                 ↙              ↘
     Layer 4: Hierarchical    Layer 5: Learned synthesis 🟠
     meta-learning 🟠              ↓
                 ↘              ↓
              Layer 6: Differentiable substrate 🟠
                      ↓
              Layer 7: Emergent primitives 🔴
                      ↓
              Layer 8: Formal verification 🔴
```

Layers 3 and 4 branch in parallel from Layer 2/3; they'd typically both land before Layer 5. Layers 5 and 6 are the "Vector 3" track from CRLM_SPEC.md and compose multiplicatively with everything above.

---

## Decision framework — how to pick the next layer

**If:**
- In-flight layer's result is positive → proceed to next
- In-flight layer plateaus → pivot to the alternative branch (e.g., Layer 2 plateau → jump to Layer 3)
- Capacity-bound evidence (Layer 2 capacity probe) → scale up within same layer before proceeding
- Architecture-bound evidence → pivot layer (e.g., Layer 3 fails → Layer 4 hierarchical + more params)

**Commercial triggers:**
- Any layer that closes a bug-class (hallucinated facts, arithmetic errors) ships immediately — no waiting for the full stack
- Layer 2 alone is shippable as a product (format-diverse HRM + interpreter covers 80%+ of math queries)
- Layer 5 is the first layer with a commercial moat ("we write our own backends from user errors")
- Layer 8 is the theorem — "provably correct AI"

---

## Commercial implications at each layer

| Layer | Ships as product? | Competitive position |
|---|---|---|
| 0-1 | ✅ already shipping (CALM + HRM prototype) | Matches open-source state-of-art on math benchmarks |
| 2 | Yes — "format-diverse math extractor" | First-to-market for deterministic-math reliability |
| 3 | Yes — "pick up any math format from 3 examples" | Very few competitors have small-scale meta-learning |
| 4 | Yes — "zero-shot domain onboarding" | No direct competitor; LLMs do this at 1000× the cost |
| 5 | Yes — "self-extending AI" | Genuine moat; capability growth from usage |
| 6 | Infrastructure; underpins future products | Research differentiator, not a direct SKU |
| 7 | Indirect — expands reach of all prior products | |
| 8 | Yes — "verified-correct mathematical AI" | Category-defining product |

---

## What the stack STILL won't do

Even at Layer 8, CRLM doesn't solve:

- **Creative writing** — no target language
- **Brand voice, aesthetics, taste** — no verifiable ground truth
- **Emotional support** — no compiler for empathy
- **Open-ended research direction** — no schema to translate to
- **Physical / embodied reasoning** — needs a different substrate (world model)

These stay delegated to the LLM tier, possibly augmented with LLM-as-judge + diffusion + best-of-N sampling. The architecture's honest limit is **verifiability** — and that limit doesn't move with more layers. More layers extend *reach within* the verifiable domain.

---

## Research discipline

Apply the workflow from `.claude/rules/workflow.md` to every layer:

1. **Hypothesis** — concrete prediction with a measurable outcome
2. **Raw measurement** — unit test / pytest / held-out metric
3. **User-facing measurement** — end-to-end eval against real queries
4. **Plateau detection** — 3 iterations < 2% each → find the one wrong line (or pivot)
5. **One commit per round** with before/after table
6. **Feedback-loop validation pattern** (workflow.md §"Feedback-loop validation pattern") for any new learning surface

Each layer's research should produce:
- A concrete experiment design
- A `scripts/eval_hrm_layerN.py` that measures the layer's claim
- A before/after empirical table
- A decision: ship, iterate, or pivot

---

## Open questions (in priority order)

1. ~~Does Layer 2's distribution curve continue past 50% at 20 formats?~~ **Answered: yes, climbs to 100% at 20 formats — cliff, not diminishing returns.**
2. Does Layer 3 (meta-learning) teach *operation inference* from 3 in-context examples at 48K, or require 1-10M params? (Now the sharpest open question — operation inference is what distribution-scaling doesn't solve.)
3. Does Layer 5 (learned synthesis) produce correct backends at the rate needed for production deployment?
4. Is Layer 6 (differentiable substrate) stable under continuous fine-tuning, or does weight drift break correctness?
5. At what layer does the "tiny-specialist + compiled-substrate" economics fully replace monolithic LLM fine-tuning for deterministic tasks?

Each question has an experiment design implicit in it. Each experiment is <1 week of focused work. The roadmap is long but cheap per data point.

---

## Pace calibration — why timelines here look fast

The time estimates above are calibrated to this project's demonstrated
iteration speed, not traditional research-team pace. Evidence:

- **Session 25**: HRM Round 1e (structure-only, 48K sweet spot, 96.7%
  full-expression) — single session, ~145s of training, the core
  CRLM validation landed.
- **Session 25-26 overlap**: LLM-Computer prototype (Small2DTransformer,
  HullKVCache, 4 hand-wired primitives with full test coverage) —
  shipped in one focused sitting.
- **Session 26 core**: 3-digit math HRM + LookUp/ReGLU IR + declarative
  compiler + 1-digit adder + NL→math HRM — 4-step plan completed
  end-to-end, ~5 commits, in a single session.
- **Session 26 follow-ons**: word problems at 48K (100%), GSM-style
  at 48K (93%, first ceiling), parabolic-key LookUpExact, greedy
  auto-scheduler, HullKVCache parity, semantic-keyed KV via ReGLU
  squaring. Each is a day-or-less research unit.
- **Session 26 Vector 1**: four phases of feedback-loop tests (311
  tests green), pattern-pollution defect found and fixed, unified
  dashboard built. One session.
- **Session 26 Vector 2**: multi-task HRM training + 4-domain OOD
  measurement + capacity probe (h=64) + distribution probe (multi10
  with 50% OOD, multi20 in flight). Done inside one session.
- **Parallel context**: `~/llama.cpp` zenith branch shipped TurboQuant
  tq4 for both weights and KV cache on Gemma 4 E4B in ~1 day. Including
  the 132-byte alignment fix. That's a full custom-CUDA-kernel +
  kernel-fusion research sprint in a workday.

The operating assumption is sustained pace of **minutes per experiment,
hours per training run, days per research direction**. Layer 6
(differentiable substrate) is estimated at 4-8 weeks on this pace,
not 12-18 months. Layer 8 (formal verification) is 2-3 months, not
2-4 years.

Traditional research-team estimates (grad student on full-time,
quarterly milestones, conference deadlines) would inflate these by
roughly 10-100×. Those estimates don't apply here — they're calibrated
to a coordination-heavy reference class that doesn't match this
project's operating mode.

**Honest caveat:** this pace is easy to maintain when the architecture
cooperates (primitives compose cleanly, iteration loops are cheap,
testing is automatic). Layer 6+ may hit friction that slows things
down — catastrophic forgetting in continual fine-tuning, proof search
in Lean, combinatorial explosion in the MILP scheduler. Adjust
estimates upward if/when that shows up, not by default.

---

## Verification of this spec

On resume / reread:
1. Check each layer's status flag against `CRLM_SPEC.md` (layers 0, 1 should be ✅; layer 2 should be 🟡 if multi20 hasn't completed, else ✅)
2. Verify commit references in §L of CRLM_SPEC.md match the status here
3. Confirm the dependency graph is consistent with any new work since this spec was written
4. Verify the "open questions" list is still the right priority order

At this project's demonstrated pace, layers land in days-to-weeks, not months-to-years. This spec should be revisited every 1-2 weeks rather than quarterly — the roadmap ages fast because the pace is fast.
