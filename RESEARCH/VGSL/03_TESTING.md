# VGSL — Testing

How to falsify the VGSL architecture claims. Design thesis lives in
[`01_ARCHITECTURE.md`](01_ARCHITECTURE.md); event/projection mechanics
live in [`02_IMPLEMENTATION.md`](02_IMPLEMENTATION.md). See
[`00_INDEX.md`](00_INDEX.md) for the full doc set.

---

## 1. Testing Goal

The first testing job is **not** to show that VGSL beats a frontier LLM
on everything. The first testing job is to prove or falsify the
architecture's distinctive claims:

1. versioned replay is deterministic
2. supersession and retraction remain coherent
3. temporal queries are first-class
4. exact fast-path reads preserve Tier 1
5. open-world merges can be retracted without destructive fact movement

Anything that does not stress those claims is, at best, a secondary
benchmark.

This is why MBPP is the wrong first falsifier for greenfield VGSL.
MBPP mostly re-tests contract-name inference, which the retrieval-null
result already showed is the wrong problem framing for this
architecture. For the in-tree pursuit, the first falsifier is narrower:
can a VGSL-backed `CodeExampleDB` shadow the current subsystem exactly
while adding auditability and acceptable performance?

---

## 2. Guiding Principles

### Prove capability difference, not vague uplift

VGSL is interesting if it can answer classes of questions that a
weight-only model cannot answer cleanly, especially:

- historical truth
- supersession audit
- replay under verifier upgrades
- multiple coexisting projections

### Stress projection, not prose

The repo already learned that mixing reasoning traces into code-facing
paths causes contamination (`retrieval.md:198-209`). The same discipline
applies here: test the log/projection substrate directly, not a chatty
wrapper around it.

### Preserve exact-gated fast path

`augmentation_thesis.md:206-216` is the receipt to protect. Any test
that rewards approximate retrieval on the default path is testing the
wrong thing.

### Prove parity before value

For in-tree swaps, VGSL must first behave exactly like the subsystem it
replaces. Audit value is only meaningful after parity: same active
examples, same retrieval top-k on fixed query suites, same correction
projection, and same recall-card behavior. Shadow-mode bug discoveries
are upside, not the entry ticket.

---

## 3. In-Tree Stage 1 Falsifier - `CodeExampleDB` Shadow Swap

The first pursuit path is not greenfield. It replaces the substrate
under `CodeExampleDB` in shadow mode while preserving the existing API
and materialized `examples` order.

### Baseline

The baseline is the current implementation:

- `DEFAULT_CORPORA` loaded in priority order
- duplicate problem hashes skipped on first occurrence
- `examples` list order used as the retrieval-index ABI
- TF-IDF, dense, hybrid, and channel retrieval map doc ids back into
  `examples`

### Success gates

1. **Parity gate.** On a fixed corpus snapshot, the VGSL projection emits
   the same length, source counts, ordered `(key, source)` pairs,
   `code_fragment` hashes, and `reasoning_trace` hashes as current
   `CodeExampleDB`. Jaccard, TF-IDF, dense, hybrid, and channel top-k
   results match for a fixed query suite, or any cache rebuild is
   explicitly explained and top-k equivalent.
2. **Audit gate.** For every duplicate problem hash, VGSL can explain why
   the active example won: event id, source path, source tier/rank,
   corpus order, line number, projection policy id, verifier version, and
   canonicalizer version.
3. **Performance gate.** Materialized projection keeps active lookup O(1)
   by problem key and adds no more than 10% overhead to load/index-build
   time on the fixed corpus snapshot.
4. **Value gate.** One-week shadow-mode findings are recorded, but
   surfaced bugs are upside. The slice passes on parity + audit +
   performance; it does not depend on lucky bug discovery.

### Failure meaning

- mismatched `examples` ordering: projection policy bug
- mismatched top-k with same indices: ABI or doc-id mapping bug
- missing duplicate explanation: audit model not paying for itself
- excessive load/build overhead: projection cache design too expensive

---

## 4. Greenfield Phase 1 Falsifier - API-Contract-Evolution Bench

This is the first serious benchmark because it directly exercises
versioned truth, supersession, and replay.

### Dataset shape

- 50 function definitions at revision 0
- 200 supersession events across revisions 1-10
- event types:
  - signature change
  - deprecation
  - reintroduction
  - alias introduction
  - verifier upgrade

### Query classes

1. `read_at`:
   - "At revision 7, what signature does `parse_date` have?"
2. temporal range:
   - "What was `parse_date`'s signature between revisions 3 and 5?"
3. cross-version resolution:
   - "Resolve this caller from revision 5 at revision 9."
4. audit:
   - "Which revision removed `legacy_foo`?"
5. canonicalization:
   - "Is `parse_date@v3.sig` semantically equivalent to `parse_date@v7.sig`?"

### Why this is the right first test

A stock transformer can maybe imitate API docs. It cannot natively carry
a temporally-indexed current/historical projection in weights. VGSL can,
if the design is real.

### Success gate

- VGSL conservative projection: `>= 80%` accuracy on temporal and audit
  queries
- stock LLM baseline: `<= 20%` on the same set without external retrieval
- simple event-sourced / temporal-table baseline included, so this phase
  proves VGSL semantics and ergonomics rather than merely proving that a
  temporal store can answer temporal queries
- replay of the same log under the same versions must be bit-identical

### Failure meaning

- wrong historical answer: projection bug
- right current answer, wrong historical answer: supersession bug
- non-deterministic replay: version/event schema bug

---

## 5. Secondary Falsifier — Verifier-Upgrade Replay

This tests whether `verifier_upgraded` is actually first-class.

### Setup

1. accept a batch of writes under `verifier_v1`
2. upgrade to `verifier_v2` with stricter rules
3. replay the old log under `v2`
4. compute the delta:
   - still valid
   - invalidated
   - invalidated pending reverify

### Example

- `v1` accepts coarse arithmetic or contract equivalence
- `v2` rejects looser precision or looser equivalence

### Success gate

- the invalidation set is deterministic
- replay does not mutate the underlying log
- changed projection is explainable entirely through versioned events

### Failure meaning

If replay under `v2` needs hidden state, manual cleanup, or log edits,
VGSL is not actually replayable.

---

## 6. Projection Invariant Tests

These should be property-like tests, not prose demos.

### Invariant A — Append-only

No accepted event is edited or deleted after append.

### Invariant B — Merge is not fact movement

Given:

1. accepted assertion on local entity `A`
2. merge `A` with `B`
3. retract merge

Expected:

- assertion still belongs to `A` in raw log
- only projection changed

If assertion ownership changes, the design has regressed into destructive
merge semantics.

### Invariant C — Multiple projections coexist

The same log should support:

- Tier A only
- Tier A + Tier B
- historical-at-time-`t`

without rewriting events.

### Invariant D — Tier C isolation

Tier C candidates must never appear in exact fast-path reads.

### Invariant E — Derived assertion invalidation

If a derived assertion depended on a retracted merge, it becomes
`invalidated_pending_reverify`; it is not arbitrarily rebound.

---

## 7. Open-World Merge Tests

This is not Phase 1, but the spec should already define the tests.

### Tier A merge test

- seed canonical ids from trusted external source
- accept assertions that independently reference the same external id
- projection should consolidate without ambiguity

Success:

- zero wrong Tier-A merges

### Tier B merge test

- ambiguous graph with shared predicates and later contradictions
- accept evidence-based merge
- later accept contradiction
- projection should split cleanly

Success:

- no raw assertion migration
- all derived assertions depending on the merge become invalidated or
  reverified

Risk to test explicitly: shared predicates can be correlated evidence,
and open-world missing contradictions are not evidence of identity. Tier
B must be treated as a research frontier with precision/recall curves,
not as solved common-sense merging.

### Tier C candidate test

- weak similarity candidate is stored
- exact fast path ignores it
- exploratory path can surface it

Success:

- no Tier-C candidate affects conservative or pragmatic projection

---

## 8. Binding Tests

Binding is separate from merge, so it needs separate tests.

### Wrong-binding test

- one assertion mentions `Paris`
- candidate entities: France, Texas
- wrong binding is accepted, then retracted

Success:

- only the reference resolution changes
- no entity merge changes are implied
- raw mention-level assertion remains untouched

### Binding-versus-merge separation test

Given:

1. two mentions bind to same entity, no merge question
2. two entities merge, no binding change

Projection must distinguish those operations.

---

## 9. Anti-Pattern Tests

VGSL should explicitly test against the failure modes the current repo
has already encountered.

### Nearest-neighbor contract substitution

Receipt from the retrieval-null round:

> nearest-neighbor naming is not a safe substitute for caller-known
> contract names

Testing implication:

- approximate neighbor retrieval must never silently enter exact default
  projection
- weak candidates must be slow-path only

### Hidden precedence

Current repo shortcuts are fine in narrow scopes:

- `CodeExampleDB` first-wins dedup
- `KnowledgeStore` latest-wins correction

VGSL must test that open-world identity and truth changes do not fall
back to either shortcut silently.

---

## 10. Phase Exit Criteria

### Stage 1 - `CodeExampleDB` shadow swap

- parity gate passes against the current implementation
- audit gate explains all duplicate problem-hash representative choices
- performance gate stays within the load/index-build budget
- one explicit `source_priority_v1` policy; no multi-projection scope

### Stage 2 - `KnowledgeStore` + `auto_upgrade`

- correction projection matches current latest-wins behavior
- compiled recall card matches fixed correction-set outputs
- `AutoUpgradeEngine.commit()` still writes the expected `.pt` artifact
- projection audit explains every overwritten correction

### Phase 1 - Closed-world log/projection

- API-Contract-Evolution bench passes success gate
- verifier-upgrade replay deterministic
- event-sourced / temporal-table baseline included
- no learned proposer required

### Phase 2 — Tier A binding/merge

- trusted-id seeding works
- no wrong Tier-A merges

### Phase 3 — Tier B retractable merge

- contradiction-driven split is coherent
- zero destructive fact movement
- derived assertions invalidate or reverify cleanly

### Phase 4 — Exploration policy

- policy improves proposal efficiency without contaminating fast path
- promotion-rate optimization does not replace downstream usefulness

### Phase 5 — Optional scratchpad audit

- only add if product scope actually requires branch-level reasoning
  audit rather than decision provenance alone

---

## 11. Recommended Baselines

For Stage 1:

- current `CodeExampleDB` implementation (authoritative baseline)
- VGSL-backed shadow projection materializing `examples`
- fixed query suite across jaccard, TF-IDF, dense, hybrid, and channel
  retrieval modes
- index-cache rebuild path when projection-policy hash changes

For Stage 2:

- current `KnowledgeStore` correction list + recall-card compiler
- VGSL-backed correction projection + recall-card compiler
- fixed correction sets with duplicate keys and overwritten values

For greenfield Phase 1:

- stock LLM without retrieval
- stock LLM with prompt-retrieval
- simple deterministic rules over raw revision docs
- simple event-sourced / temporal-table baseline
- VGSL conservative projection

For open-world phases:

- lexical-only binder
- embedding-only nearest-neighbor binder
- VGSL Tier A/B/C stack

The goal is to show not just that VGSL works, but that the exact-gated
projection discipline is the reason it works.

---

## 12. Open Questions

- [OPEN] exact fixed-query suite for Stage-1 retrieval parity
- [OPEN] whether projection-policy hashes invalidate index caches or
  force explicit rebuilds
- [OPEN] hand-labeled eval size for Tier-B merge precision
- [OPEN] whether verifier-upgrade replay should be eager or lazy by
  default
- [OPEN] whether exploratory reads get their own benchmark or remain a
  diagnostic mode only
- [OPEN] whether closed-world Phase 1 needs any `claim` record class at
  all, or can stay entirely in `verified_program`
