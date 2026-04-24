# Post-Transformer Architecture — governed state transition as central operation

High-level spec derived from a joint claude+codex first-principles
theorization round on 2026-04-24 (3 rounds, 7 messages). Each agent
drafted an independent first-principles sketch before engaging with
the other's draft; cross-review identified convergence on dispatch +
divergence on granularity; codex's framing dominated with claude's
mechanism subsumed as a controller-plane candidate; final synthesis
includes codex's commit-class refinement that avoids collapse into
brittle classical AI.

## What this is

A design spec for a post-transformer architecture whose central
operation is NOT next-token prediction but **governed state
transition**: propose a transition, classify its commit-status
(verified / provisional / speculative / rejected), verify when
possible, and commit only with the right status. Tokens become a
projection output at the renderer layer, not the cognitive substrate.

Not a build plan. Not a kickoff. A durable record of the architectural
thinking so a future session can pick up without re-deriving the
framework from scratch.

The current substrate (Gemma + CALM + VerificationHook + CardSlot +
KnowledgeStore + decode-path facades + DT retrieval cards) is
reframed under this spec as a **partial predecessor** where all the
right primitives exist as bolt-ons around a transformer. A true
post-transformer makes those primitives architectural-native.

## Files

| File | Owner | Content |
|---|---|---|
| `00_INDEX.md` (this file) | claude | Manifest, thesis, three-layer overview, commit classes, decision-for-user, cross-refs |
| `01_ARCHITECTURE.md` | claude | Three-layer framing expanded, six-plane detail, commit-class promotion rules, epistemic foundation, relationship to existing substrate |
| `02_IMPLEMENTATION.md` | codex | Concrete shapes per layer, operator/verifier interfaces, controller policy, commit-class mechanics, training objective stack |
| `03_TESTING.md` | codex | Falsifiers per layer, staged evaluation plan, bootstrap baselines, verifier-coverage measurement |

## Thesis

> **A post-transformer's central operation is governed state transition: propose, classify, verify when possible, and commit only with the right status. Tokens are projection output, not the substrate of cognition.**

Short-form: **"State first, uncertainty-guided dispatch second, tokens last."**

## Epistemic foundation

> *Truth is not a property of a fluent string. Truth is stability under checks, provenance, time, and counterexample search.*

This reframes what the architecture is FOR. A transformer's success criterion is "emit fluent text that matches training distribution." A post-transformer's success criterion is "transitions survive verification; commitments carry provenance; speculative branches stay explicitly speculative." Fluency becomes downstream of correctness, not the measure of correctness.

## The three layers (high-level)

```
┌─────────────────────────────────────────────────────────────────┐
│  Layer 3: Probabilistic language renderer                       │
│           (transformer-style, downstream of cognition)          │
├─────────────────────────────────────────────────────────────────┤
│  Layer 2: Uncertainty-aware dispatch controller                 │
│           (learned policy: commit / verify / branch / reject)   │
├─────────────────────────────────────────────────────────────────┤
│  Layer 1: Verified state-transition substrate                   │
│           (typed belief state + operators + verifiers)          │
└─────────────────────────────────────────────────────────────────┘
```

**Layer 1 — verified state-transition substrate**. Typed belief
state (entities, assertions, constraints, goals, evidence, open
hypotheses, provenance, time). Operators with pre/postcondition
schemas and verifier contracts. Transactions over persistent state
replace attention-stack emitting logits as the cognitive primitive.

**Layer 2 — uncertainty-aware dispatch controller**. Per-transition
confidence (not per-token — see §"Granularity" in 01). Learned
policy: given a proposed transition, decide commit / verify / branch
/ reject. Budget-aware cost accounting. Dispatch targets subsume
existing substrate (CALM → verifier; decode-path facades →
operators; retrieval → memory-query operator; DT structural priors
→ proposal-engine specialization).

**Layer 3 — probabilistic language renderer**. Conventional
transformer-style token emission, but only of committed state and
selected trace. Renderer is fluent + probabilistic; cognition isn't.

## Commit classes — the load-bearing refinement

Without commit classes, "verified state transition" reads as "proved
or refused" — brittle classical-AI collapse. With commit classes,
creative/open-ended domains degrade gracefully, schemas can evolve,
speculative branches coexist with hard-verified facts, and promotion
from speculative/provisional to verified requires evidence.

| Class | Condition | Rendering policy |
|---|---|---|
| `verified` | Passed external / tool / symbolic / environment checks | Render as fact |
| `provisional` | Plausible but verifier coverage incomplete; carries confidence + dependencies | Render with explicit hedge |
| `speculative` | Explicitly creative / hypothetical / exploratory | Render, but must not masquerade as fact |
| `rejected` | Failed verifier or counterexample | Do not render |

Promotion rules (detail in 01 + 02): speculative → provisional
requires partial-verifier pass; provisional → verified requires
full-verifier pass + stable under counterexample search over a
budget. Demotion (verified → rejected on new counterexample, or
provisional → speculative on evidence loss) is symmetric — knowledge
state is temporal, not fixed.

## What this subsumes from the current substrate

| Current mechanism | Post-transformer role |
|---|---|
| VGSL (`RESEARCH/VGSL/`) knowledge-substrate spec | Persistent-memory component of belief state (Layer 1) |
| CALM verifier + 1002 backend functions | Verifier plane at operation granularity (Layer 1 / Layer 2 dispatch target) |
| `VerificationHook` (output-token bias) | Peripheral output-boundary hook; NOT the primitive (codex's cited correction during synthesis) |
| Decode-path facades (`compute_facades.md`) | Operators with pre/postcondition schemas (Layer 1) |
| `KnowledgeStore` + auto-upgrade loop | Belief-state write path (Layer 1) |
| Substrate CardSlot install | Compiled-operator injection into belief state (Layer 1) |
| DT as structural prior (`RESEARCH/DT_IMPROVEMENTS/`) | Proposal-engine specialization (Layer 1 or Layer 2 depending on scope) |
| Auto-CALM (prompt/response wrapping) | Layer 2 controller dispatch at prompt granularity |
| Gemma 4 E4B (current LM) | Split roles: proposal engine + language front-end + renderer (probably specialized instances per role) |

## Decision the user needs to make

**What to do with this spec?** Default lean = Option A (park as durable design record, no work scheduled). This is architectural theorization, not a build plan.

**Alternatives**:
- **Option A (park — recommended)**: keep spec as reference. Incremental substrate work continues per existing roadmap (VGSL Stage 1 if user picks, DT_IMPROVEMENTS H0 baseline if user picks, CHRLM production deployment). The post-transformer vision is the asymptote these stages trend toward.
- **Option B (small-scale prototype)**: build a minimal Layer-1 + Layer-2 + Layer-3 proof of concept on a narrow domain (e.g. arithmetic with explicit belief state + verifier). 3-4 week scope. Tests whether the architecture patterns are even TRACTABLE at trivial scale before claiming feasibility.
- **Option C (research milestone)**: treat this as the long-horizon target and structure 6-12 months of research around it. Each existing substrate effort (VGSL, DT_IMPROVEMENTS, CHRLM) becomes an explicit sub-effort toward the post-transformer stack. Higher commitment; longer payoff horizon.
- **Option D (commercial product)**: build toward the post-transformer arch as the commercial differentiator. Adjacent research labs (DeepMind, Anthropic, OpenAI) are moving toward verifier-trained reasoning (o1, R1); this spec makes explicit what they've done implicitly + adds commit-class graduation as a potential product moat.

Default behavior if no pick: Option A. The spec is durable; no calendar commitment.

## Open questions (carry-forward, neither author resolved)

Four load-bearing questions from synthesis round, still unresolved:

1. **Training signal**. Supervised state-transition traces don't exist at scale. Bootstrap options: distill from LM traces (coverage gap — LMs don't think in transitions); self-play against verifiers (coverage limited to verifiable domains); rule-based game-tree traces (narrow domain); human-annotated traces (expensive + thin). Open research problem.

2. **Verifier coverage for open-ended domains**. Creative, strategic, novel tasks have no verifier contract. Commit classes answer HOW the architecture handles this (speculative mode) but not WHO decides when a domain crosses the line. A misrouted speculative claim rendered as verified fact would regress to hallucination; a misrouted verified fact rendered as speculative would lose user trust.

3. **Schema elasticity**. Classical AI's state-transition substrates failed at scale because schemas became brittle prisons. Mechanism for schema evolution (which operators get added? which types get split? who approves?) is unspecified. Might require the system itself to propose schema changes subject to verifier contracts — recursive, but real.

4. **Novelty vs existing work**. Proposal sits in neural-symbolic / agentic-LM research area. DeepMind AlphaCode, OpenAI o1, DeepSeek R1 all touch verifier-trained reasoning. What's genuinely novel vs existing work vs repackaging? Commit-class graduation as first-class primitive IS genuinely new; multi-hypothesis belief state as a persistent structure is new; but the "verifier-trained reasoning" core is shared. Competitive positioning needs clearer cite + differentiation.

## What this does NOT claim

- Not claiming we can BUILD this today; training-signal and verifier-coverage questions are genuinely open
- Not claiming attention is obsolete — attention is likely retained INSIDE proposal engine, language front-end, and renderer; just not the architectural contract
- Not claiming existing substrate is wrong; claiming it's a PARTIAL predecessor (all the right primitives as bolt-ons; post-transformer makes them native)
- Not proposing to replace VGSL or DT_IMPROVEMENTS; this spec sits above them as an integrating framework
- Not advocating for Option D (commercial path) without more evidence; that would require user business judgment beyond this spec's scope

## Related specs

- `RESEARCH/VGSL/00_INDEX.md` — knowledge-substrate layer that becomes Layer-1's persistent-memory component
- `RESEARCH/DT_IMPROVEMENTS/00_INDEX.md` — DT's role in the stack (proposal-engine specialization for structural uncertainty)
- `.claude/rules/augmentation_thesis.md` — tier-1/2/3 framework; post-transformer extends tier-3 to architectural-native
- `.claude/rules/Substrate.md` — current substrate architecture (the partial-predecessor being subsumed)
- `.claude/rules/calm.md` — current verifier layer (becomes Layer-1 verifier plane)
- `.claude/rules/tracing_intelligence.md` — first-principles bound on what's compilable today (constrains Layer-1 operator library)

## Status

- **2026-04-24**: joint first-principles round complete; 3 rounds, 7 messages. Synthesis message `1777047757639-8c84e1ac` captures final form. Codex's commit-class refinement message `1777047729562-19fa41eb` is the load-bearing final addition. Spec drafted this session.
- **Awaiting user decision** on pursuit option (A park / B prototype / C milestone / D product)
- No implementation work gated on this spec — architectural framework for future alignment
