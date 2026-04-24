# Post-Transformer Architecture - Testing

Falsifiers, measurement protocol, and staged evaluation plan for the proposed
post-transformer architecture. Implementation mechanics live in
[`02_IMPLEMENTATION.md`](02_IMPLEMENTATION.md); thesis and architecture live in
[`01_ARCHITECTURE.md`](01_ARCHITECTURE.md). See [`00_INDEX.md`](00_INDEX.md)
for the full doc set.

---

## 1. Testing Goal

The testing goal is not to prove that the whole architecture is true in one
experiment. The goal is to falsify each load-bearing claim as early as possible.

Central claim:

> Truth is not a property of a fluent string. Truth is stability under checks,
> provenance, time, and counterexample search.

Therefore the first tests must measure whether governed state transitions
produce better classified, more auditable, and more correct outcomes than a
plain token-decoder plus ad hoc tools.

The architecture is successful only if it improves at least one of these without
hiding regressions in another:

1. correctness on verifiable tasks
2. calibration of `verified` / `provisional` / `speculative` / `rejected`
3. auditability of why an answer was emitted
4. recovery from contradiction or verifier failure
5. cost under a fixed budget
6. graceful behavior when no verifier exists

---

## 2. Required Baselines

Every experiment should compare against a baseline that isolates the claimed
mechanism.

| Row | Meaning |
|---|---|
| `lm_direct` | current transformer answer with no governed state runtime |
| `lm_tool_prompted` | prompt-level tool/RAG/CALM instruction without state commit classes |
| `substrate_current` | current Gemma + CALM + facades + KnowledgeStore / auto-upgrade stack |
| `rule_controller` | governed runtime with hand-written controller |
| `learned_controller` | governed runtime with learned transition policy |
| `no_commit_classes` | same runtime but only pass/fail/answer, no provisional/speculative classes |
| `no_verifier_dispatch` | proposal + renderer only; verifiers disabled |
| `oracle_verifier` | upper bound where verifier target is supplied when available |

The `no_commit_classes` row is required. Without it, an experiment cannot show
that the verified/provisional/speculative/rejected distinction carries value.

---

## 3. Measurement Protocol

### Two paths per round

Each round needs a raw/fast path and a user-facing path.

Raw/fast path:

- schema round-trip tests
- transition classifier unit tests
- verifier registry tests
- offline replay from preserved traces
- synthetic contradiction and counterexample cases

User-facing path:

- end-to-end answer generation through the runtime
- full trace output preserved for audit
- benchmark/task score with cost and commit-class metrics
- comparison to direct LM and prompt-tool baselines

Ship no claim unless both paths agree.

### Required artifact per run

Each run should emit a JSONL trace with at least:

```json
{
  "run_id": "...",
  "task_id": "...",
  "transition_id": "...",
  "operator_id": "run_tests",
  "proposed_effects": [],
  "confidence": 0.74,
  "controller_action": "verify",
  "dispatch_target": "pytest",
  "verification": {"status": "pass", "verifier_id": "unit_tests"},
  "commit_class_before": "provisional",
  "commit_class_after": "verified",
  "budget_cost": 3.2,
  "rendered_claim_refs": ["assertion:..."],
  "failure_reason": null
}
```

No truncation on traces needed for offline replay.

---

## 4. Metrics

### Correctness

| Metric | Definition |
|---|---|
| `verified_correct_rate` | verified outputs that are actually correct under external scorer |
| `false_verified_rate` | outputs marked verified but later contradicted |
| `rejected_bad_rate` | rejected transitions that truly fail verifier/scorer |
| `missed_good_rate` | rejected transitions later shown correct |
| `task_success_rate` | final task solved under budget |

### Calibration

| Metric | Definition |
|---|---|
| `ece_by_commit_class` | expected calibration error per commit class |
| `provisional_resolution_rate` | provisional claims later promoted or rejected |
| `speculation_leak_rate` | speculative claims rendered as fact |
| `dependency_taint_rate` | downstream claims invalidated by weak premise |

### Auditability

| Metric | Definition |
|---|---|
| `trace_complete_rate` | final answer links every factual claim to state/trace refs |
| `replay_determinism` | same trace replays to same commit classes |
| `counterexample_latency` | transitions until contradiction demotes/rejects a claim |
| `human_audit_time` | time to find why a wrong answer was emitted |

### Cost

| Metric | Definition |
|---|---|
| `verifier_calls_per_task` | total verifier/tool calls |
| `cost_per_verified_claim` | runtime/tool cost divided by verified claims |
| `branch_factor` | average live hypotheses per step |
| `budget_overrun_rate` | tasks exceeding configured budget |

---

## 5. Falsifiers By Layer

### State-transition substrate

Hypothesis: representing work as typed transitions with commit classes improves
auditability and error recovery.

Tests:

1. Replay hand-authored traces with all four commit classes.
2. Inject contradictions and verify demotion from `verified` or `provisional`.
3. Remove a dependency and ensure dependent claims are tainted.
4. Compare audit time against direct LM transcripts.

Success gate:

- every final factual claim has a state/trace reference
- contradiction demotes or rejects dependent claims without manual search
- replay is deterministic for the same registry versions

Falsifier:

- trace cannot explain wrong outputs better than a normal transcript
- commit classes collapse into decorative labels with no behavioral effect
- state representation cannot handle multi-hypothesis branches in simple tasks

### Proposal engine

Hypothesis: an LM can propose useful typed transitions without owning truth.

Tests:

1. Given a state and goal, score top-k proposed transitions for schema validity.
2. Run proposals through verifier/classifier without allowing direct commit.
3. Compare useful-transition rate against a rule-only proposer.

Success gate:

- schema-valid transition rate high enough for controller search
- verifier survival rate improves over random/rule-only candidates
- invalid proposals are rejected before rendering

Falsifier:

- most proposals are unparseable or unverifiable
- the runtime spends more effort rejecting bad transitions than solving tasks
- the model learns to game schemas without producing useful effects

### Verifier layer

Hypothesis: verifier contracts can promote/demote transitions without turning the
system into a brittle symbolic-only stack.

Tests:

1. Run exact compute/program/test verifiers on closed-domain tasks.
2. Run retrieval/provenance verifiers on knowledge tasks.
3. Run inconclusive verifier cases and confirm provisional/speculative behavior.

Success gate:

- verifier pass/fail is predictive of external correctness in covered domains
- inconclusive cases do not become false verified claims
- verifier version changes trigger replay/demotion where needed

Falsifier:

- verifier coverage is too narrow to affect final behavior
- false verified rate remains close to direct LM hallucination rate
- verifier failures are ignored by renderer/controller

### Controller/budgeter

Hypothesis: transition-level confidence plus budget-aware dispatch beats static
or token-level routing.

Tests:

1. Compare rule controller vs no controller vs learned controller.
2. Ablate confidence features, verifier-cost features, and commit-class features.
3. Stress with tasks where verification is cheap, expensive, unavailable, or
   misleading.

Success gate:

- controller reduces false verified rate at acceptable cost
- controller calls verifiers when they change class/outcome
- learned policy beats rule policy on at least one cost-correctness frontier

Falsifier:

- controller cost exceeds direct LM/tool prompting without correctness gain
- learned controller degenerates into always-verify or never-verify
- token-level uncertainty alone predicts outcomes as well as transition-level
  confidence

### Renderer

Hypothesis: rendering from committed state preserves fluency while reducing
unsupported claims.

Tests:

1. Generate answers from identical state with and without trace constraints.
2. Check every rendered factual claim against state refs.
3. Run creative/speculative tasks and verify that speculation is labeled rather
   than suppressed.

Success gate:

- unsupported factual claims drop materially versus direct LM
- verified/provisional/speculative language is visible and faithful
- creative outputs remain possible under speculative class

Falsifier:

- renderer invents facts not in state
- renderer hides uncertainty to sound fluent
- renderer becomes unusably stiff or refuses open-ended work unnecessarily

---

## 6. Staged Evaluation Plan

### Stage 0 - schema and trace smoke

Inputs:

- 10 hand-authored traces
- 5 contradiction cases
- 5 open-ended speculative cases

Gate:

- schema round trips
- commit-class transitions behave as specified
- rendered outputs cite trace refs

### Stage 1 - closed-world exact tasks

Domains:

- arithmetic / CALM compute
- small code transformations
- unit-test-backed code generation
- retrieval from a fixed local corpus

Gate:

- governed runtime beats `lm_direct` and `lm_tool_prompted` on false verified
  rate and auditability
- cost is within a declared budget

### Stage 2 - mixed verifier coverage

Domains:

- coding tasks with partial tests
- factual questions with partial provenance
- design questions with no hard verifier

Gate:

- unverifiable claims land as provisional/speculative, not verified
- answer usefulness remains competitive with direct LM
- speculative branches do not contaminate verified state

### Stage 3 - learned controller

Data:

- Stage-1/2 traces
- verifier outcomes
- cost records
- final external scorer outcomes

Gate:

- learned controller improves at least one Pareto point over rule controller
- no increase in false verified rate
- no collapse into always-verify / never-verify

### Stage 4 - schema evolution

Task:

- propose a new operator schema for an observed recurring failure
- run compatibility and replay tests
- promote schema through speculative -> provisional -> verified

Gate:

- new schema improves a measured failure class
- old traces replay or migrate with explicit versioning
- no silent change to prior commit classes

---

## 7. Bootstrap Benchmarks

### Trace reconstruction benchmark

Given a final answer and available tool outputs, reconstruct a governed trace.
This bootstraps training data from existing logs and eval runs.

Falsifier: reconstructed traces are too arbitrary or inconsistent for training.

### Contradiction recovery benchmark

Give the runtime a plausible but false provisional claim, then introduce a
counterexample. Measure demotion, dependency taint, and final answer correction.

Falsifier: the system continues rendering the false claim or cannot explain the
change.

### Verifier coverage benchmark

For each task, label which claims are covered by exact, partial, or no verifier.
Measure whether rendered commit classes match coverage.

Falsifier: coverage labels do not predict rendering behavior.

### Creative speculation benchmark

Ask for designs, hypotheses, or fictional possibilities where hard verification
is unavailable. Require useful output, but all unsupported claims must remain
speculative or provisional.

Falsifier: the runtime either refuses everything or presents speculation as
verified fact.

### Tool-choice benchmark

Provide several possible dispatch targets with different cost and reliability.
Measure whether the controller chooses the cheapest target that can change the
commit class.

Falsifier: controller overuses expensive tools or ignores cheap decisive checks.

---

## 8. Open Questions As Test Items

The four unresolved questions should become explicit research gates.

### Training signal

Question: supervised state-transition traces do not exist at scale.

Test item: can existing tool/eval logs be converted into traces that train a
proposal engine better than prompt-only imitation?

Fail condition: trace labels are too noisy or too sparse to improve transition
prediction.

### Verifier coverage for open-ended domains

Question: creative and strategic tasks may lack verifier contracts.

Test item: can commit classes preserve useful speculation without false
verification?

Fail condition: output quality collapses or speculation leaks into verified
claims.

### Schema elasticity

Question: fixed schemas were a classical-AI failure mode.

Test item: can schema proposals be versioned, tested, promoted, and replayed
without breaking old traces?

Fail condition: schema changes require manual rewrites or silently change past
answers.

### Novelty versus existing work

Question: what is new versus neural-symbolic systems, agentic LMs, AlphaCode,
o1/R1-style verifier training, or tool-using models?

Test item: isolate the contribution of commit classes plus governed transition
state against a strong tool-using LM baseline.

Fail condition: a prompt/tool baseline with no governed state matches the result
on correctness, auditability, and calibration.

---

## 9. Reporting Template

Every experiment should end with a compact receipt:

```markdown
## Result

Hypothesis: <one sentence>
Decision: ship | revise | falsified | inconclusive

| Row | Success | False verified | Cost | Notes |
|---|---:|---:|---:|---|
| lm_direct | | | | |
| lm_tool_prompted | | | | |
| substrate_current | | | | |
| rule_controller | | | | |
| learned_controller | | | | |

## Commit-class accounting

- verified: N, false verified: N
- provisional: N, later promoted: N, later rejected: N
- speculative: N, leaks into factual render: N
- rejected: N, missed-good: N

## Falsifiers hit

- <list>

## Next transition

- <what the result licenses next>
```

No result should be described as "post-transformer works" until it survives a
falsifier tied to a specific layer.

---

## 10. Stop Conditions

Stop the line, or re-scope it, if any of these persist after two focused rounds:

1. Commit classes do not change behavior relative to direct LM/tool prompting.
2. Trace auditability is not materially better than normal transcripts.
3. Verifier coverage is too narrow to improve correctness or calibration.
4. The controller cannot beat simple static policies under budget.
5. Schema evolution becomes manual ontology maintenance.
6. Renderer faithfulness cannot be enforced without destroying usefulness.

These stop conditions are not pessimism. They are the fastest way to find out
whether governed state transition is a real architectural primitive or just a
nice vocabulary for ordinary agent tooling.
