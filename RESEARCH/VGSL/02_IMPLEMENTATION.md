# VGSL — Implementation

How the proposed **Verifier-Governed Substrate Log (VGSL)** should be
realized as data structures, event schemas, projection rules, and
read/write APIs. Design thesis and invariants live in
[`01_ARCHITECTURE.md`](01_ARCHITECTURE.md); falsifiers and success gates
live in [`03_TESTING.md`](03_TESTING.md). See
[`00_INDEX.md`](00_INDEX.md) for the full doc set.

---

## 1. TL;DR

VGSL should not be implemented as a mutable knowledge graph. It should
be implemented as:

1. an **append-only event log**
2. a **versioned verifier/canonicalizer registry**
3. a **projection builder** that folds events into one or more current or
   historical views
4. a **dual-path read surface**

The key implementation move is:

> **Merge is not fact movement. Merge is projection-time aliasing over
> immutable assertions.**

That is the difference between a coherent retractable system and a
destructive one that needs policy guesses on split.

This inverts two shortcut patterns that are acceptable in today's narrow
repo subsystems but do not generalize to open-world state:

- `KnowledgeStore.add_correction()` is explicitly `latest correction wins`
  at `calm/llm_computer/persistent_knowledge.py:71-76`
- `CodeExampleDB` dedup is explicitly `first occurrence wins` at
  `calm/llm_computer/facades/code_example_db.py:135-137`

VGSL replaces hidden precedence with **explicit log semantics + explicit
projection semantics**.

The first pursuit path is not greenfield. Stage 1 is an in-tree,
shadow-mode replacement for `CodeExampleDB` that preserves the existing
public surface and materializes the current `examples` list bit-for-bit.
Stage 2 applies the same substrate to `KnowledgeStore` +
`auto_upgrade`. The broader open-world VGSL phases remain research
follow-ons after those closed-world swaps prove parity, audit value, and
acceptable performance.

---

## 2. Layer Model

### In-tree Stage 0/1

Before the greenfield closed-world prototype, VGSL should land as a
shadow-mode substrate under existing repo APIs:

1. **Stage 0 - event log + projection materializer.** Build the minimal
   append-only log, verifier/canonicalizer registry, and materialized
   projection cache needed by current callers.
2. **Stage 1 - `CodeExampleDB` swap.** Emit one `accepted_assertion` per
   source-tagged example and materialize the exact current ordered
   representative list under `source_priority_v1`.
3. **Stage 2 - `KnowledgeStore` + `auto_upgrade`.** Represent
   corrections as assertions and compile recall-card weights from the
   materialized current projection.

These stages are closed-world. They do not require open-world bindings,
entity merges, Tier-B evidence merges, or a learned proposer.

### Closed-world Phase 1

For the first falsifier, VGSL only needs two layers:

1. **Assertion log** — immutable accepted assertions over local ids
2. **Projection** — current or historical view built by folding the log

This is enough for the API-Contract-Evolution benchmark in
[`03_TESTING.md`](03_TESTING.md).

### Open-world extension

Once the design leaves a closed world, VGSL needs four layers:

1. **Assertion log** — immutable accepted assertions over local mention
   ids / local entity placeholders
2. **Binding log** — reference-resolution events from mentions/placeholders
   to entity ids or entity clusters
3. **Merge log** — identity/equivalence events over entity ids
4. **Projection** — fold of assertions + bindings + merges under a
   chosen time/version regime

The decisive separation is:

> **Binding resolves references; merge resolves identity; projection
> composes both.**

Without that split, open-world VGSL still conflates disambiguation with
entity identity.

---

## 3. Core Invariants

The implementation must preserve these invariants:

1. **Append-only log.** No accepted event is edited in place.
2. **Versioned replay.** Every accept/retract/projection-sensitive event
   records the verifier and canonicalizer version that justified it.
3. **Immutable raw assertions.** Accepted assertions stay attached to
   their original local ids forever.
4. **Non-destructive merge.** Merge events alias ids in projection; they
   never rewrite stored assertions.
5. **Dependency-tracked derivation.** Derived assertions record which
   source assertions, bindings, merges, and verifier versions supported
   them.
6. **Tier-C isolation.** Weak candidates never enter the exact fast path.
7. **Projection plurality.** Conservative and pragmatic projections may
   coexist over the same log.
8. **Representative selection is projection policy.** Source priority,
   latest-wins, and first-wins are never encoded as merge semantics.
   They are explicit projection rules over immutable assertions.

If any of these invariants fail, replay reproducibility and retraction
coherence fail with them.

---

## 4. Event Model

Every event should have a common envelope:

```json
{
  "event_id": "uuid-or-monotone-id",
  "ts": "2026-04-23T18:00:00Z",
  "kind": "accepted_assertion",
  "record_class": "verified_program",
  "proposer": "policy|rule|import",
  "provenance_chain": ["source_a", "source_b"],
  "verifier_id": "merge_verifier",
  "verifier_version": "v1.2+sha256:...",
  "canonicalizer_id": "entity_merge",
  "canonicalizer_version": "v1.0+sha256:..."
}
```

### Phase-1 required kinds

- `accepted_assertion`
- `accepted_derivation`
- `retracted`
- `verifier_upgraded`
- `canonicalizer_upgraded`
- `projection_rule_changed`

### Open-world extension kinds

- `binding`
- `binding_retracted`
- `merge`
- `merge_retracted`
- `hypothesis`
- `hypothesis_promoted`
- `invalidated_pending_reverify`

### Source priority encoding

The Stage-1 `CodeExampleDB` shadow swap needs deterministic assertion
metadata, not new event kinds:

```json
{
  "record_class": "code_example",
  "source_path": "agents/distill/data/mbpp.jsonl",
  "source_tier": "benchmark",
  "source_rank": 2,
  "corpus_order": 3,
  "line_no": 417,
  "stable_problem_key": 1234567890
}
```

`source_tier` / `source_rank` describe quality. `(corpus_order,
line_no, event_id)` is the stable tie-breaker. The active
representative is derived by the `source_priority_v1` projection rule;
it is not stored as a fact and not encoded on a merge.

### [OPEN] Projection-rule events

`projection_rule_changed` should be rare. The spec must treat it as a
versioned event because replay under a new projection rule is otherwise
not deterministic.

---

## 5. Record Classes

VGSL only needs four record classes in the first serious spec:

- `exact_fact`
- `verified_program`
- `hypothesis`
- `claim`

Their implementation burden is different.

### `exact_fact`

Smallest unit. Should canonicalize to a typed subject/predicate/object
form or other domain-native exact form.

### `verified_program`

Code-like artifact with deterministic canonicalization, e.g. AST
normalization + alpha-renaming. This follows the same spirit as
`program_builder.py:102-170`, where imports and schedule are explicit
rather than implicit.

### `hypothesis`

Non-durable by default. Should not enter the exact fast path. See
[`01_ARCHITECTURE.md`](01_ARCHITECTURE.md) for why hypothesis logging is
not a Phase-1 primitive.

### `claim`

Open-world natural-language-shaped assertion. This is where binding and
merge semantics become necessary.

---

## 6. Accepted Assertions

Accepted assertions should be stored against original ids, not
canonicalized merged ids:

```json
{
  "kind": "accepted_assertion",
  "assertion_id": "a123",
  "record_class": "claim",
  "subject_id": "entity_local_17",
  "predicate": "capital_of",
  "object_id": "entity_local_42",
  "payload": {
    "text": "Paris is the capital of France."
  },
  "support": ["wikidata:Q90", "wikidata:Q142"]
}
```

That looks redundant until a merge is retracted. Then it is the whole
system's coherence story.

### `CodeExampleDB` assertions

Stage 1 maps each ingested example to an assertion:

```json
{
  "kind": "accepted_assertion",
  "assertion_id": "code_ex_000417",
  "record_class": "code_example",
  "subject_id": "problem_hash:8f41...",
  "payload": {
    "problem": "...",
    "solution": "...",
    "code_fragment": "...",
    "reasoning_trace": "..."
  },
  "source_path": "agents/distill/data/mbpp.jsonl",
  "source_tier": "benchmark",
  "source_rank": 2,
  "corpus_order": 3,
  "line_no": 417
}
```

All duplicate problem hashes remain in the log. The projection
materializes exactly one active representative per problem hash under
`source_priority_v1`, preserving today's
`calm/llm_computer/facades/code_example_db.py` behavior:
`DEFAULT_CORPORA` priority order, first unique hash wins, and
`self.examples[doc_idx]` remains the retrieval-index ABI.

### `KnowledgeStore` assertions

Stage 2 maps each correction to an assertion:

```json
{
  "kind": "accepted_assertion",
  "assertion_id": "corr_000123",
  "record_class": "knowledge_correction",
  "subject_id": "query_key:17",
  "payload": {
    "correct_value": 391
  },
  "source_tier": "verified_calm",
  "source_rank": 0
}
```

The current projection collapses these by query key under a
`latest_verified_correction_v1` policy. `KnowledgeStore.add_correction`
stays as a compatibility wrapper; recall-card compilation consumes the
projection output instead of the raw correction list.

---

## 7. Derived Assertions

Derived assertions are accepted outputs that depended on earlier
projection assumptions.

They must record their dependency set:

```json
{
  "kind": "accepted_derivation",
  "assertion_id": "d991",
  "depends_on": {
    "source_assertions": ["a123", "a124"],
    "bindings": ["b17"],
    "merges": ["m88"],
    "verifier": "derivation_verifier@v3"
  },
  "payload": {
    "subject_id": "cluster_paris",
    "predicate": "population_bucket",
    "object_literal": "2M_to_3M"
  }
}
```

On retraction of any required dependency, these should become
`invalidated_pending_reverify`, not arbitrarily reassigned.

---

## 8. Binding Events

Binding is a separate primitive from merge.

It maps local references to entity ids or clusters:

```json
{
  "kind": "binding",
  "binding_id": "b17",
  "mention_id": "mention_44",
  "entity_id": "entity_paris_fr",
  "tier": "a|b|c",
  "evidence": {
    "canonical_id": "wikidata:Q90"
  }
}
```

Retraction of a bad binding is `binding_retracted`. It does not imply a
merge retraction.

---

## 9. Merge Events

Merge events define projection-time aliasing, not assertion movement.

```json
{
  "kind": "merge",
  "merge_id": "m88",
  "members": ["entity_paris_fr_1", "entity_paris_fr_2"],
  "into_cluster": "cluster_paris_fr",
  "tier": "a|b|c",
  "evidence": {
    "proof_chain": ["wikidata:Q90"],
    "shared_predicates": [],
    "contradictions": [],
    "similarity_score": null
  }
}
```

Tier semantics:

- **Tier A** — proof-based, durable unless the external authority changes
- **Tier B** — evidence-threshold, retractable on contradiction or
  verifier upgrade
- **Tier C** — candidate only, never consulted by exact reads

Do not use merge fields to select the "best" representative for a
closed-world duplicate set. In `CodeExampleDB`, duplicate problem hashes
share identity, but source priority is representative selection. A merge
may say "these assertions are about the same problem"; the
`source_priority_v1` projection rule decides which assertion appears in
the materialized `examples` list.

### [OPEN] Cluster ids

The spec should decide whether `into_cluster` is a fresh synthetic id or
the oldest surviving member id. Fresh synthetic ids are cleaner for
replay and split, at the cost of one more indirection.

---

## 10. Projection Builder

Projection is a fold, not a mutable store.

Inputs:

- event log
- projection time `t`
- verifier version filter
- canonicalizer version filter
- projection mode (`conservative`, `pragmatic`, `exploratory`)

Outputs:

- active assertions
- active bindings
- active merge alias sets
- invalidated-but-not-reverified derivations
- materialized compatibility views (for Stage 1/2 wrappers)

### Fold order

1. load events up to `t`
2. filter by projection/version policy
3. materialize active bindings
4. materialize active merge alias sets
5. resolve active assertions through bindings + merges
6. mark derived assertions valid or `invalidated_pending_reverify`

### Implementation note

Alias sets are a projection concern, so a union-find-like structure is
appropriate inside the projection builder, not in the durable log.

### Multiple projections

VGSL should support at least:

- **Conservative** — Tier A only
- **Pragmatic** — Tier A + Tier B
- **Exploratory** — Tier A + Tier B + explicit Tier C read path

This is the ontology-level analog of the repo's exact-gated recall path
in `augmentation_thesis.md:206-216`.

Stage 1 deliberately does not need multiple user-facing projections.
It needs one explicit policy:

```python
source_priority_v1(assertions) -> list[CodeExample]
```

The policy groups assertions by `stable_problem_key`, sorts each group
by `(source_rank, corpus_order, line_no, event_id)`, selects the first
representative, and emits the ordered `examples` list. That list is the
compatibility boundary: TF-IDF doc ids, dense vectors, channel maps, and
callers that index `db.examples` must see the same order as today's
first-occurrence-wins implementation.

---

## 11. Read APIs

Minimum descriptive API:

```python
append_event(event) -> event_id
build_projection(*, at_time=None, verifier_version=None,
                 canonicalizer_version=None, mode="conservative")
read_current(key, *, mode="conservative")
read_at(key, t, *, mode="conservative")
read_all(key)
read_canonical(key, *, verifier_version=None, canonicalizer_version=None)
```

Stage-1 compatibility APIs:

```python
CodeExampleDB.load_default()
CodeExampleDB.load_paths(paths)
db.examples
db.retrieve(query, ...)
db.retrieve_channel(query, ...)
db.build_tfidf()
db.build_dense(m, tok)
db.save_indices(path)
db.load_indices(path)
db.export_pt_signature_jsonl(path)
```

The wrapper may be backed by a log, but the list order and index mapping
are part of the API. `load_indices()` remains caller-synchronized with
the materialized projection; if the projection policy changes, caches
must rebuild or carry a projection-policy hash.

Stage-2 compatibility APIs:

```python
KnowledgeStore.add_correction(query_key, correct_value)
KnowledgeStore.save_corrections(path)
KnowledgeStore.load_corrections(path)
KnowledgeStore.build_recall_model(...)
AutoUpgradeEngine.commit()
```

`AutoUpgradeEngine.commit()` should still save a `.pt`; the difference
is that the recall-card weights are compiled from the current projection
of `knowledge_correction` assertions rather than directly from the
mutable `corrections` list.

Open-world extension:

```python
propose_binding(...)
propose_merge(...)
reverify(assertion_id)
```

Fast path discipline:

- exact-gated reads must never consult Tier C
- exploratory reads may consult Tier C, but only explicitly and with a
  verifier-bounded path

This mirrors `retrieval.md:198-209`, where code hits and reasoning
traces are retrieved separately to avoid contamination.

---

## 12. Relation to Current Repo Mechanisms

VGSL should reuse, not ignore, the repo's existing receipts:

- `VerifiedDispatcher` already models versioned, multi-lane structural
  verification (`calm/verifier.py:86-160`)
- `VerificationHook` already models gated intervention at the output
  boundary (`calm/llm_computer/gemma_substrate.py:1153-1188`)
- `KnowledgeStore` already proves cross-session persistence, but with a
  narrow `latest wins` policy (`persistent_knowledge.py:71-76`)
- `CodeExampleDB` already proves multi-source ingest and projection-like
  dedup, but with a narrow `first wins` policy
  (`code_example_db.py:135-137`)

VGSL generalizes these into an explicit event/projection substrate
instead of a set of local shortcuts.

For the in-tree pursuit, `CodeExampleDB` comes first. It has a crisp
closed-world invariant: problem-hash identity plus source-priority
representative selection. It also has broad enough call-site coverage to
make parity meaningful without touching model serving. `KnowledgeStore`
and `auto_upgrade` come second because they cross into runtime
weight-compilation and saved `.pt` artifacts.

---

## 13. Phase Boundaries

### Stage 0 - In-tree substrate skeleton

- append-only event store
- version registry
- materialized projection cache
- no public behavior change

### Stage 1 - `CodeExampleDB` shadow swap

- one explicit `source_priority_v1` projection policy
- materialized ordered `examples` list matches current implementation
- retrieval APIs and index artifacts remain compatible
- no open-world bindings, Tier-B merges, or learned proposer

### Stage 2 - `KnowledgeStore` + `auto_upgrade`

- `add_correction` emits `knowledge_correction` assertions
- current projection preserves latest-verified correction behavior
- recall-card compilation consumes projection output
- `.pt` save path remains the serving artifact

### Phase 1

- no open-world bindings
- no entity merges
- no learned proposer
- closed-world temporal replay only

### Phase 2

- Tier-A bindings and merges from trusted external ids

### Phase 3

- Tier-B retractable merges
- dependency-tracked derivation invalidation

### Phase 4

- learned exploration/policy

### Phase 5

- optional scratchpad/hypothesis audit layer

---

## 14. Open Questions

- [OPEN] exact event schema for `projection_rule_changed`
- [OPEN] cluster-id policy for merges
- [OPEN] eager vs lazy re-verification scheduling
- [OPEN] external snapshot hashing standard
- [OPEN] whether `hypothesis` remains a record class in closed-world
  deployments or is deferred entirely to open-world/debug builds
