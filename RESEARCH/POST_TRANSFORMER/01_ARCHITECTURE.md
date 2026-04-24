# Architecture — governed state transition, three layers, commit classes

## Thesis expansion

A transformer's central operation is `p(next_token | prefix)`. Every
fact, plan, skill, uncertainty, and intermediate computation is
compressed into transient vectors over a token stream. That was the
breakthrough for language modeling — but it is also the ceiling.

A post-transformer's central operation is **governed state
transition**:

```
controller(S_t, goal) → proposed_transition
classify(proposed_transition) → commit_class ∈ {verified, provisional, speculative, rejected}
verify(proposed_transition) if classification warrants
commit | branch | reject
render tokens only when state is licensed for the task
```

The architectural contract exposed to the rest of the system is no
longer "a stack of attention layers emits logits." It is **"a
controller proposes and governs state transitions under verifier
pressure, with explicit commit-class graduation."**

Attention is likely retained INSIDE the proposal engine, language
front-end, and renderer — but those are implementation details of
layers, not the architecture's exposed operation.

## Epistemic foundation

> *Truth is not a property of a fluent string. Truth is stability
> under checks, provenance, time, and counterexample search.*

This claim justifies every other architectural choice:

- **Stability under checks** → verifier plane must be first-class
- **Provenance** → every commit must carry its derivation / operator / dependencies
- **Time** → belief state is temporal, not fixed; past commits remain queryable
- **Counterexample search** → verified class requires passing not just one check but surviving bounded search

A transformer's success metric ("output fluent tokens matching
training distribution") is orthogonal to this definition. Most
hallucinations are fluent. Most training-distribution tokens are
also plausible but unverified. The post-transformer rearranges what
the architecture optimizes FOR — from fluency to stability.

## The three layers

### Layer 1 — Verified state-transition substrate

The cognitive primitive. Typed belief state + operator library +
verifier plane.

**Belief state `S_t`** contains:

```
entities       : typed objects, unique-id'd
assertions     : claims over entities, with commit_class + dependencies
constraints    : invariants on entities / assertions
goals          : target states the system is trying to reach
evidence       : external observations, tool outputs, verifier results
open_hypotheses: speculative branches with provisional provenance
action_trace   : chronological record of transitions
provenance_idx : source/derivation map (who wrote what, when, via what operator)
time_idx       : temporal index; enables "what was true at t?" queries
```

**Operator `O`** signature:

```
input_schema    : typed arguments
preconditions   : what must hold in S_t for O to be applicable
proposed_effects: what O would write to S_t
verifier(s)     : checks the proposed_effects against external/symbolic/test reality
cost_model      : compute + latency + external-call budget
renderer_hooks  : how committed effects should surface at Layer 3
```

**Transaction `Step`**:

```
propose(O, args, S_t) → Step.proposed
simulate(Step.proposed) → hypothetical S_t+1
verify(Step.proposed) → commit_class
commit(Step) → S_t → S_t+1   (if class ∈ {verified, provisional})
branch(Step) → {S_t, S_t+1}  (if class = speculative, multiple hypotheses coexist)
reject(Step) → S_t unchanged (if class = rejected)
```

### Layer 2 — Uncertainty-aware dispatch controller

The dispatch primitive. Learned policy deciding which transitions
to verify, commit, branch, or reject based on confidence + budget.

**Controller signature** (from synthesis):

```
controller(S_t, goal) → (
    proposed_transition : Step,
    transition_confidence : float,    // NOT token confidence
    verifier_need : dict[verifier_name → expected_coverage_gain],
    dispatch_target : OperatorId,
    budget_value : float,             // cost-adjusted expected benefit
)
```

**Dispatch policy** (confidence alone NEVER grants `verified` — only verifier pass does; see §"Commit classes — promotion rules"):

| Confidence | Budget low? | Action |
|---|---|---|
| High | — | run free verifier if available; commit `verified` only on pass, otherwise `provisional` if useful |
| Medium | No | invoke explicit verifier; commit based on result |
| Medium | Yes | commit as `provisional` with confidence recorded; verify later if a dependency arises |
| Low | No | branch (`speculative` class); search + verifier in parallel |
| Low | Yes | reject or defer; do not commit |

The controller is a LEARNED policy trained on:
- Verifier outcomes (which dispatches led to verified commits)
- Budget realization (which dispatches were worth their cost)
- Downstream failure recovery (which rejections saved rework)

**Granularity correction (load-bearing)**: confidence attaches to
**transitions / assertions / operators**, NOT to tokens. Token-level
confidence (current transformer log-probs, `VerificationHook` output
bias) is peripheral — it's a rendering-time hook. The primitive
decision is at the transaction level. Cited precedent:
`calm/verifier.py` works at operation/state level;
`compute_facades.md` evidence that decode-path token-level bias can
corrupt correct trajectories; `calm/uncertainty.py` already has
fact-level confidence via `UncertainFact`.

### Layer 3 — Probabilistic language renderer

The I/O primitive. Transformer-style token emission, but downstream
of Layer 1's committed state + Layer 2's dispatch decisions.

**Renderer responsibilities**:
- Turn committed state + selected trace into text / code / UI actions / API calls
- Emit commit-class markers alongside content (verified claims get plain text; provisional claims get hedges; speculative claims get explicit flags)
- Preserve entailment: rendered output must be derivable from committed state
- Probabilistic fluency IS allowed at this layer — renderer learns humanlike phrasing without jeopardizing the cognitive substrate's correctness

**What the renderer is NOT**:
- Not the source of facts (facts come from committed state in Layer 1)
- Not where reasoning happens (reasoning is governed transitions in Layer 1, dispatched by Layer 2)
- Not a verifier (it may render unverified content IF commit_class is preserved in output)

Attention is likely the right primitive inside the renderer — it's
doing exactly what transformers do well (sequence generation from a
context). The architectural shift is that cognition happens BEFORE
the renderer, not WITHIN it.

## Six-plane architecture (Codex's expansion)

Layer 1 decomposes into six operational planes:

### 1. Language / perception front-end

Converts text or environmental observations into candidate
entities, assertions, goals, and operator calls. A transformer-like
model is a good fit for this role — its strength is exactly parsing
natural language into semantic tokens. But it's a PARSER/PROPOSER,
not the whole mind.

### 2. Working belief state

Typed, inspectable, temporal graph/log. Supports multiple live
hypotheses (via branch commits) instead of collapsing immediately
to one "answer." This directly subsumes:
- VGSL's versioned event log (`RESEARCH/VGSL/00_INDEX.md`)
- CALM's precompute fact injection (`calm/auto_calm.py`)
- KnowledgeStore's corrections (`calm/llm_computer/persistent_knowledge.py`)

### 3. Proposal engine

Neural policy that proposes the next useful transition. Trained on
trace data (verified transaction histories). The LEARNED part is
**search guidance + analogy**, NOT the fact store. Proposal engine
suggests `Step.proposed`; it does not assert the step.

### 4. Verifier layer

Type checks + invariant checks + executable tests + symbolic
solvers + retrieval / provenance checks + environment observations
+ secondary critics. Verifiers DO NOT need to solve everything —
they need to reject enough invalid transitions that committed state
stays cleaner than raw model output.

Subsumes:
- CALM 4-lane TMR (`calm/verifier.py`)
- CALM backend functions (1002 deterministic compute oracles)
- VerificationHook (output-token gate; peripheral, see granularity correction above)
- Sandbox execution (`calm/sandbox.py`)
- External tools (Python exec, web API calls, retrieval DBs)

### 5. Controller / budgeter

Described in full as Layer 2 above. Dispatches among commit / verify
/ branch / reject. Tracks budget; manages exploration vs exploitation.

### 6. Renderer

Described as Layer 3 above.

## Commit classes — promotion rules and rendering policy

Four classes; graduation rules make the architecture non-brittle.

### Class definitions

| Class | Entry condition | Rendering policy | Can become |
|---|---|---|---|
| **`speculative`** | No verifier available OR verifier explicitly declines (open-ended domain) OR user-marked "what-if" | Render with explicit flag: "hypothetically", "could be", "one possibility" | → `provisional` on partial-verifier pass |
| **`provisional`** | Plausible per proposal-engine confidence AND partial-verifier pass AND verifier coverage incomplete | Render with hedge: "likely", "typically", confidence-marker | → `verified` on full-verifier pass + counterexample-survival; or → `rejected` on counterexample |
| **`verified`** | Full-verifier pass + stable under counterexample search within budget | Render plainly as fact | → `rejected` on new counterexample (knowledge is temporal) |
| **`rejected`** | Failed any verifier or counterexample | Do not render (except as diagnostic for debugging) | Terminal unless new evidence reopens |

### Promotion mechanics

**speculative → provisional** (weakest promotion):
- Some verifier partially validates (e.g. passes type check but not full symbolic proof)
- Proposal-engine confidence above threshold
- No immediate counterexample found in budget

**provisional → verified** (strongest promotion):
- Full verifier applicable and passes
- Counterexample search exhausts budget without finding a contradiction
- Dependencies (other provisional/verified claims this rests on) all hold

**any class → rejected** (demotion):
- New counterexample surfaces (symmetric — verified state is not immutable)
- A dependency gets rejected (cascades through)
- Verifier contract itself is updated and the claim no longer survives

**verified → provisional** (rare demotion):
- Verifier coverage weakens (e.g. verifier was retired; new coverage is partial)
- This is the "knowledge-state-is-temporal" case; past truth stays queryable via time_idx

**rejected → speculative** (reopen-on-new-evidence):
- Rejected class is NOT terminal. A new transition carrying new evidence
  (updated verifier contract; previously-unknown dependency now satisfied;
  counterexample withdrawn) re-opens the claim as speculative, subject to
  the same graduation rules as any new speculative claim.
- No class is ever overwritten in-place — every class change is itself a
  transition in the action trace.

### Why commit classes avoid classical AI brittleness

Classical state-transition substrates failed at scale because they
required every fact to be provable. Open-world domains (common
sense, creative tasks, strategic reasoning, novel situations) don't
admit proofs. Commit classes let the architecture:

- **Operate in open domains** without refusing (speculative mode)
- **Hedge appropriately** when partial evidence is available (provisional mode)
- **Commit strongly** where verifiers can reach (verified mode)
- **Avoid contamination** between modes (rendering markers prevent speculative claims masquerading as verified)

The crucial property: **promotion requires evidence, never just
time or repetition**. This prevents the "LLM sounds confident →
user believes → claim 'verifies' via social reinforcement" failure
mode that pure-transformer stacks exhibit.

## Relationship to existing substrate (the partial predecessor)

Every primitive in this architecture exists TODAY in the claw-code
substrate — as a bolt-on around Gemma. The post-transformer
proposal makes them native.

| Plane | Current bolt-on | Post-transformer native |
|---|---|---|
| Language front-end | Gemma's forward pass (natural language → hidden states) | Gemma-like model specialized for parsing, not generation |
| Belief state | `KnowledgeStore` + `CardSlot` + persistent `.pt` state | First-class typed graph; temporal-by-default |
| Proposal engine | Gemma's next-token distribution (implicit transition proposer) | Explicit transition proposer trained on trace data |
| Verifier layer | CALM 4-lane TMR + 1002 backend functions + sandbox | First-class verifier protocol; operators declare verifier contracts |
| Controller | Auto-CALM wraps prompt/response; Auto-upgrade feeds corrections | Learned dispatch policy; confidence + budget aware |
| Renderer | Gemma's same forward pass (ambiguous with cognition) | Separate model; trained on committed-trace → text |

The substrate has proven these pieces work individually. The
post-transformer claim is that **architecting them as first-class
primitives with an explicit contract** unlocks capabilities that
bolt-ons can't: clean separation of concerns, trainable dispatch,
principled open-world handling via commit classes, and temporal
knowledge queries.

## What this architecture does NOT claim

- **Not claiming attention is wrong**. Attention is likely retained inside 3 of the 6 planes (language front-end, proposal engine, renderer). What's replaced is the USE of attention as the architectural contract.
- **Not claiming we can train this today**. The four open questions in `00_INDEX.md` §"Open questions" are genuinely unresolved — training signal, verifier coverage, schema elasticity, novelty vs existing work.
- **Not claiming to replace neural learning with symbolic logic**. Layers 2 + 3 are LEARNED; Layer 1's operators can be compiled OR learned OR mixed. The framework accommodates both poles and their mixture.
- **Not claiming existing substrate is wrong**. Substrate (Gemma + CALM + cards + facades + KnowledgeStore + DT) is a correct partial predecessor. Post-transformer extends the trajectory, doesn't reject it.
- **Not claiming commercial readiness**. This is theorization + framework; productization requires answering all four open questions.

## Relationship to related research

**DeepMind AlphaCode** — competitive programming via generate-and-filter. Shares verifier-centric philosophy; doesn't have explicit belief state or commit classes.

**OpenAI o1 / DeepSeek R1** — reasoning-trace-trained models. Share the "verified-trace training" insight; don't separate cognition from rendering (still token-output throughout).

**Neural-symbolic hybrid research** — broad area including NeSy, Logic Tensor Networks, DeepProbLog. Share the state-transition + verifier idea; typically rigid schemas (classical-AI risk). Commit classes + learned dispatch policy are our differentiators.

**Active inference (Friston et al.)** — predict + act to confirm. Similar philosophical framing but operates at perception-action level; doesn't address knowledge representation + verifier contracts.

**Ought's Factored Cognition / Anthropic's constitutional AI** — decompose reasoning into auditable steps. Shares auditability; less explicit about commit classes + promotion rules.

**What's load-bearingly novel in this proposal**:
1. Commit-class graduation (4 classes, explicit promotion rules) as a first-class architectural primitive
2. Multi-hypothesis belief state as persistent structure (not just chain-of-thought branching that collapses at each step)
3. Learned dispatch policy over a unified operator/verifier library (vs hand-coded tool-use or content-routing MoE)
4. Budget-aware controller with explicit cost accounting for verifier-invocation decisions

Items 1 + 2 are strongly novel. Items 3 + 4 have precedents in other research areas but the integration into this layered architecture is the contribution.

## Cross-refs

- `00_INDEX.md` (this spec) — manifest + decision-for-user
- `02_IMPLEMENTATION.md` (this spec) — concrete shapes + operator/verifier interfaces
- `03_TESTING.md` (this spec) — falsifiers + staged evaluation plan
- `RESEARCH/VGSL/01_ARCHITECTURE.md` — knowledge-substrate detail (becomes Layer 1 belief-state component)
- `RESEARCH/DT_IMPROVEMENTS/01_ARCHITECTURE.md` — DT's role (proposal-engine specialization for structural uncertainty)
- `.claude/rules/Substrate.md` — current substrate architecture (partial predecessor)
- `.claude/rules/calm.md` — current verifier layer (becomes verifier plane)
- `.claude/rules/augmentation_thesis.md` — tier-1/2/3 framework (this spec extends tier-3 to architectural-native)
- `.claude/rules/tracing_intelligence.md` — first-principles bounds on what's compilable today (constrains Layer-1 operator library)
