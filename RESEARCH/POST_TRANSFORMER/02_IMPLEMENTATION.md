# Post-Transformer Architecture - Implementation

Implementation shape for the proposed post-transformer architecture. Thesis
and architectural framing live in [`01_ARCHITECTURE.md`](01_ARCHITECTURE.md);
falsifiers and success gates live in [`03_TESTING.md`](03_TESTING.md). See
[`00_INDEX.md`](00_INDEX.md) for the full doc set.

---

## 1. TL;DR

A post-transformer should not be implemented as a bigger decoder with a
confidence head attached to logits. The implementation target is a governed
state-transition runtime.

> A post-transformer's central operation is governed state transition: propose,
> classify, verify when possible, and commit only with the right status. Tokens
> are projection output, not the substrate of cognition.

Short form:

> State first, uncertainty-guided dispatch second, tokens last.

The first build should be a thin runtime around existing primitives, not a new
foundation model. It should make the boundary explicit:

1. parse language or observations into candidate state changes
2. propose transitions against a typed working state
3. classify each transition as `verified`, `provisional`, `speculative`, or
   `rejected`
4. dispatch verifiers or tools only when they can change the commit class
5. render text from committed state and selected trace

That is enough to test the architectural claim without pretending to solve all
training and representation problems at once.

---

## 2. Layer Model

The implementation has three layers and six planes.

### Three layers

1. **Verified state-transition substrate.** Typed belief state, operators,
   verifier contracts, provenance, dependency edges, and commit classes.
2. **Uncertainty-aware dispatch controller.** Learned or rule-backed policy that
   chooses whether to commit, verify, branch, ask, retrieve, compute, or reject
   a proposed transition.
3. **Probabilistic language renderer.** Transformer-style token emission, but
   downstream of state and trace rather than the owner of truth.

### Six planes

1. **Language/perception front-end** converts prompts, files, tests, retrieved
   records, and observations into candidate state deltas.
2. **Working belief state** stores entities, assertions, goals, constraints,
   evidence, open hypotheses, and action/observation traces.
3. **Proposal engine** suggests the next useful transition.
4. **Verifier layer** checks typed preconditions, invariants, tests, tools,
   retrieval provenance, symbolic programs, and environment observations.
5. **Controller/budgeter** chooses action under confidence and cost.
6. **Renderer** projects the selected state/trace into prose, code, commands, or
   API calls.

The layers are logical, not necessarily separate models. Early prototypes can
share one transformer-backed proposer/renderer and implement the state,
verifier, and controller in Python.

---

## 3. Current Repo Precedents

This architecture should be grounded in mechanisms that already exist in narrow
forms.

### Output-boundary token hooks are useful but not primary

`VerificationHook` maps a card's verified output to a Gemma token and boosts the
logit when a margin clears `min_margin` (`calm/llm_computer/gemma_substrate.py:1153-1194`).
That is a useful but peripheral output-boundary hook. It proves the
card/operator did the real verification before the renderer token changed; the
hook is not the post-transformer primitive.

### Operation-level verification is closer to the primitive

`VerifiedDispatcher.execute()` snapshots state, runs primary and shadow
implementations, and records unanimity over resulting stack/error state
(`calm/verifier.py:86-165`). That is already state-transition verification:
the accepted object is not a fluent string, but the result of an operation over a
state.

### Fact-level uncertainty already exists

`UncertainFact` carries statement confidence and dependencies, and confidence is
bounded by weakest premises (`calm/uncertainty.py:28-44`, `:96-128`). This is the
right granularity for the post-transformer controller: confidence attaches to
facts, transitions, and operators, not isolated tokens.

### Decode-path facades show the need for operator choice

`CodeDtSkeletonFacade` and `CodeRenameFacade` show why token bias is not a
universal interface. DT step-through bias can help structural emission, but the
RENAME result showed that caller-known contract repair is sometimes safer as a
post-generation AST rewrite. The controller must choose the operator, not merely
boost the next token.

### VGSL supplies the persistent-memory precedent

`RESEARCH/VGSL/` already frames immutable assertions, projections, provenance,
and temporal replay. In this spec, VGSL is not the whole architecture; it is a
candidate persistent belief-state and memory layer under the governed
transition runtime.

---

## 4. Core Types

The minimum prototype should define typed records before any model training.

```python
@dataclass(frozen=True)
class BeliefState:
    entities: Mapping[EntityId, EntityRecord]
    assertions: Mapping[AssertionId, AssertionRecord]
    goals: tuple[GoalRecord, ...]
    constraints: tuple[ConstraintRecord, ...]
    hypotheses: Mapping[HypothesisId, HypothesisRecord]
    trace: tuple[TraceEvent, ...]
    budget: BudgetState
```

```python
@dataclass(frozen=True)
class Transition:
    transition_id: str
    kind: Literal[
        "assert_fact",
        "derive",
        "retrieve",
        "compute",
        "test",
        "decompose_goal",
        "ask_user",
        "observe_environment",
        "render",
    ]
    inputs: tuple[StateRef, ...]
    proposed_effects: tuple[StateDelta, ...]
    preconditions: tuple[PredicateRef, ...]
    provenance: tuple[ProvenanceRef, ...]
    confidence: float | None
    operator_id: str
```

```python
@dataclass(frozen=True)
class OperatorSpec:
    operator_id: str
    input_schema: SchemaRef
    output_schema: SchemaRef
    preconditions: tuple[PredicateRef, ...]
    verifier_ids: tuple[str, ...]
    cost_model: CostModelRef
    renderer_hooks: tuple[str, ...]
```

```python
@dataclass(frozen=True)
class VerificationResult:
    verifier_id: str
    status: Literal["pass", "fail", "inconclusive", "not_applicable"]
    confidence_delta: float
    evidence: tuple[EvidenceRef, ...]
    counterexample: EvidenceRef | None = None
    cost: float = 0.0
```

The important constraint is that `Transition` is the unit of control. Tokens may
appear in provenance, evidence, or rendered output, but tokens are not the
state-governance primitive.

---

## 5. Commit Classes

The runtime must preserve four commit classes. They are not decoration; they are
how the system avoids confusing fluent speculation with truth.

| Commit class | Meaning | Render policy |
|---|---|---|
| `verified` | Passed domain-appropriate verifier(s) or exact tool checks | May render as fact/result |
| `provisional` | Plausible, supported by partial evidence, verifier coverage incomplete | Render with uncertainty and dependencies |
| `speculative` | Creative, hypothetical, or unverifiable branch | Render only as speculation or option |
| `rejected` | Failed verifier, contradicted by evidence, or dominated by counterexample | Do not render as candidate answer except in audit/debug |

### Promotion and demotion

Allowed transitions:

```text
speculative -> provisional   when evidence/dependencies become plausible
provisional -> verified      when verifier coverage is sufficient
verified    -> provisional   when verifier version changes or dependency weakens
any         -> rejected      when a hard counterexample lands
rejected    -> speculative   only through a new transition with new evidence
```

No class should be overwritten in place. The state log records a transition that
changes the projected class.

### Why this matters

"Verified" must not mean "proved or refused." Open-ended domains need
speculative and provisional branches. The hard rule is only that a speculative
branch cannot masquerade as committed fact.

---

## 6. Controller Interface

The load-bearing controller shape is:

```python
controller(S_t, goal) -> (
    proposed_transition,
    transition_confidence,
    verifier_need,
    dispatch_target,
    budget_value,
)
```

A concrete first version can use a dataclass:

```python
@dataclass(frozen=True)
class ControllerDecision:
    transition: Transition
    confidence: float
    verifier_need: Literal["none", "cheap", "required", "unavailable"]
    dispatch_target: str | None
    budget_value: float
    action: Literal["commit", "verify", "branch", "ask", "retrieve", "reject"]
    rationale: str
```

### Dispatch rules for an MVP

Start rule-backed, then learn the policy once traces exist.

1. Commit as `verified` only when required verifiers pass.
2. Commit as `provisional` when evidence is useful but verifier coverage is
   incomplete.
3. Commit as `speculative` when the user asks for ideation or the domain lacks
   verifier coverage.
4. Reject on hard verifier failure or contradiction.
5. Dispatch a verifier only when the expected class change justifies the cost.
6. Ask or retrieve when the missing precondition is cheaper to obtain than to
   guess.

### Learned controller later

A learned controller should predict:

- probability a transition survives verification
- expected utility of each verifier/tool
- cost to obtain evidence
- expected value of branching
- risk of rendering a provisional/speculative result as fact

The policy target is not next-token likelihood. It is trace quality under
budget: did the chosen transition sequence reach a useful, correctly classified
state?

---

## 7. Operator And Verifier Registry

Operators should be first-class registry records.

```python
@dataclass(frozen=True)
class OperatorRegistryEntry:
    spec: OperatorSpec
    implementation_ref: str
    version: str
    safe_to_autorun: bool
    side_effect_class: Literal["pure", "filesystem", "network", "external"]
    verifier_requirements: tuple[str, ...]
```

Verifier entries should be similarly explicit:

```python
@dataclass(frozen=True)
class VerifierRegistryEntry:
    verifier_id: str
    accepted_transition_kinds: tuple[str, ...]
    evidence_schema: SchemaRef
    version: str
    max_cost: float
    failure_is_hard: bool
```

### Initial operator set

A useful prototype can start with these operators:

| Operator | Purpose | Existing precedent |
|---|---|---|
| `parse_claims` | text -> candidate assertions | CALM extraction / uncertainty helpers |
| `run_compute` | exact math/program backend | `calm/verifier.py` dispatcher |
| `retrieve_memory` | pull candidate evidence | VGSL / CodeExampleDB / retrieval docs |
| `run_tests` | execute code or benchmark checks | HE+ / MBPP eval harnesses |
| `rewrite_code_contract` | mechanical code transform | `CodeRenameFacade` |
| `render_answer` | state/trace -> prose | current LLM generation |

The first registry should be boring and auditable. The novelty is not fancy
operators; it is that operator use is governed by state and commit class.

---

## 8. Runtime Loop

A minimal runtime loop:

```python
def step(state: BeliefState, goal: GoalRecord) -> BeliefState:
    candidates = proposal_engine.propose(state, goal)
    ranked = controller.rank(candidates, state, goal)

    for decision in ranked:
        if not budget.can_pay(decision):
            continue

        if decision.action == "verify":
            result = verifier_layer.run(decision.dispatch_target, decision.transition, state)
            classified = classify(decision.transition, [result], state)
            return commit_transition(state, classified)

        if decision.action in {"commit", "branch", "reject"}:
            classified = classify(decision.transition, [], state)
            return commit_transition(state, classified)

        if decision.action in {"ask", "retrieve"}:
            obs = dispatch(decision.dispatch_target, decision.transition)
            return record_observation(state, obs)

    return mark_blocked(state, goal, reason="no affordable transition")
```

Rendering is a transition too, but a special one: it should read from committed
state and trace, not silently invent state.

---

## 9. Training Objective Stack

The architecture needs a staged training stack. Do not start with end-to-end
training.

### Objective 1 - language grounding

Train or prompt the front-end to map text into candidate `Transition` and
`AssertionRecord` objects. This can bootstrap from LLM extraction, synthetic
data, and existing benchmark traces.

### Objective 2 - transition prediction

Given state and goal, predict useful next transitions. Initial supervision can
come from:

- program traces
- proof/search traces
- tool-use logs
- benchmark solution traces
- human-authored task decompositions
- existing CALM/facade/eval logs

### Objective 3 - verifier survival

Reward proposals that survive verifiers and reduce unresolved uncertainty. This
is where exact domains should dominate early training because the signal is
cheap and unambiguous.

### Objective 4 - search efficiency

Reward shorter, cheaper traces that reach the same or better commit class.
Budget is part of the target, not an afterthought.

### Objective 5 - renderer faithfulness

Renderer output must be entailed by committed state and selected trace. Penalize
unsupported statements even if the prose is fluent.

### Objective 6 - active uncertainty handling

Train the system to ask, retrieve, branch, or mark unresolved when verifier
coverage is missing. This is the practical answer to open-ended domains: do not
force false certainty.

---

## 10. Build Phases

### Stage 0 - Paper runtime

Define schemas and replay a few hand-authored traces without models. The only
goal is to prove that state, commit classes, and trace rendering are coherent.

### Stage 1 - Rule-backed controller over existing tools

Wrap existing deterministic surfaces:

- CALM compute dispatch
- `VerificationHook`-style verified outputs
- Code RENAME-style mechanical rewrites
- VGSL-like append/projection memory
- benchmark test runners

No learning yet. Measure whether governed state beats ad hoc tool wiring on
auditability and failure classification.

### Stage 2 - LLM proposal engine

Use a current transformer to propose transitions in JSON or a small typed DSL.
The runtime classifies and verifies; the model is not allowed to commit facts by
itself.

### Stage 3 - Learned controller

Train the controller on Stage-1/2 traces. Objective: choose cheaper and more
successful verify/branch/commit actions than the rule controller.

### Stage 4 - Integrated renderer

Train the renderer to generate final answers from state and trace. This is where
user-visible language quality returns, but it remains downstream of governed
state.

### Stage 5 - Schema evolution loop

Add operator/schema proposal as a transition kind. New schemas start
`speculative`, become `provisional` after local tests, and become `verified`
only after replay compatibility and downstream evals.

---

## 11. Non-Goals

This spec does not claim:

- attention disappears internally
- all knowledge becomes symbolic
- all outputs must be fully proven
- creative work should be suppressed
- current transformers become useless
- verifier coverage is solved
- training data for state transitions already exists at scale

The claim is narrower and sharper: the system-level primitive should be governed
state transition, and token generation should be a renderer over state rather
than the state itself.
