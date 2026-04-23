# VGSL — Architecture

The design spec. Primitives, invariants, the four-layer stack, and
what makes this post-transformer. Companion to `00_INDEX.md`
(overview), `02_IMPLEMENTATION.md` (data structures + APIs),
`03_TESTING.md` (falsifiers).

## Thesis

> **Post-transformer is a versioned event log of verifier-accepted
> writes + temporally-indexed canonicalized projection + dual-path
> reads (exact-gated fast + verifier-bounded exploratory), with
> replay reproducibility as an architectural invariant.**

Each phrase earns its place:

- **Versioned event log** — every accept records verifier_id@vN +
  canonicalizer_id@vN, so replay is deterministic under verifier
  upgrades
- **Verifier-accepted writes** — a write only enters the log after
  passing a deterministic verifier. No probability-based acceptance.
  Kills hallucination-as-storage.
- **Temporally-indexed canonicalized projection** — the log is the
  source of truth; the projection is derived. Projection can be
  computed at any (time, verifier_version, canonicalizer_version)
  tuple. A new capability class.
- **Dual-path reads** — fast path is exact hash lookup with zero
  policy (preserves Tier 1 per `augmentation_thesis.md:206-216`);
  slow path is verifier-bounded exploratory under an explicit cost
  budget. Tier 1 preservation IS an architectural property.
- **Replay reproducibility as architectural invariant** — not a
  nice-to-have. The audit story that regulated verticals need
  (`commercial.md` §"Customer verticals = card decks") is only true
  if replay is deterministic.

## The three unattacked constraints

Every transformer architecture this repo has worked with
(Gemma 4 E4B, DT/`CopyAugmentedDeltaNet`, Small2DTransformer
substrate, CHRLM unified single tensor) shares 9 structural
constraints. Six are attacked by substrate work; three remain
load-bearing.

| Constraint | Attacked by | Still open? |
|---|---|---|
| Quadratic attention | Mamba/SSM | — |
| Stateless between sessions | `KnowledgeStore` + auto-upgrade loop | closed |
| Weight-baked knowledge | Compiled cards in `calm/llm_computer/programs/` | closed |
| Single-scale attention | `d_head=2` sub-heads (`Substrate.md`) | closed |
| Forward-only at inference | D5 recurrence (`Substrate.md` §"D5 recurrent substrate") | partly |
| Discrete tokens | — | mostly orthogonal |
| **No structural verifier** | CALM + VerificationHook (`Substrate.md:267-277`) | **OVERLAY only, not verifier-native** |
| **Imitation-trained** | — | **unattacked** |
| **Single implicit worldview** | — | **unattacked — one "now" per .pt** |

Constraint (7) — "no structural verifier" — deserves careful
phrasing. The repo has real verifier-governed overlays:
`calm/llm_computer/persistent_knowledge.py:60-176` compiles
corrections into a persistent recall model; `gemma_substrate.py:1153-1188`
makes verification part of the generation boundary via
`VerificationHook.min_margin`; `calm/verifier.py:86-160` is multi-lane
structural verification. The correct framing is: **we have verifier-
governed overlays, but the base architecture is not verifier-native.**
VGSL moves verification into the base.

Constraint (9) is the one transformers structurally cannot fix.
Their weights ARE their "now." Training once means frozen. CHRLM
unifies fact storage into the `.pt` but still has one implicit
projection per snapshot. VGSL makes "now" a query parameter, not a
property of the weights.

## Core invariants

These are the correctness properties that must hold for any
VGSL implementation. Violations are architectural bugs, not
implementation details.

### 1. Write-only log

The event log is append-only. No event is ever edited or deleted.
All "changes" are new events: `superseded_by`, `merge_retracted`,
`binding_retracted`, `verifier_upgraded`, `canonicalizer_upgraded`,
`projection_rule_changed`.

Consequence: at time t, the world state is always the projection of
events 1..k where k is the number of events accepted by time t.

### 2. Non-destructive merges

**Codex's decisive insight (1776968021263-08f807cc):**

> **Merge is not fact movement. Merge is projection-time aliasing
> over immutable assertions.**

Raw assertions stay on their original local ids forever. Merges
define equivalence classes over entity ids; the projection resolves
the canonical view by folding merges over assertions.

Consequence: merge retraction is coherent. Since assertions never
moved, retracting a merge just means stop treating two ids as
aliases. No arbitrary re-assignment. No policy-land.

### 3. Versioned verifiers + canonicalizers

Every accepted write records `{verifier_id, verifier_version,
canonicalizer_id, canonicalizer_version}`. Replay at time t under
verifier_version V is a deterministic function of the log up to t
and V.

Consequence: upgrading a verifier is a first-class event
(`verifier_upgraded`). Projection under the new verifier is a
different (but still deterministic) projection. Past writes that
the new verifier would reject are identified by replay; whether to
retract them is a policy decision logged explicitly, not a silent
mutation.

### 4. Dependency-tracked derivations

Assertions derived under a merged view carry their dependency
chain: `{source_assertions, assumed_merges, verifier_id@version,
canonicalizer_id@version}`.

On merge retraction, all derivations with `assumed_merges` overlap
enter status `invalidated_pending_reverify`. They are not arbitrarily
reassigned to one or the other of the split nodes. Re-verification
by a verifier that can prove them from a single side is the only
path to re-validation.

Consequence: split-time coherence without policy magic. Derivations
become invalid events in the log, retrievable if the evidence
survives.

### 5. Tier-1 preservation through exact-gated fast path

Reads default to the fast path: exact hash lookup over
canonicalized keys. Zero policy on this path. If the lookup misses,
the default is to return nothing, NOT to approximate.

Approximate reads require explicit opt-in via the slow path with a
cost budget, and results are gated by verifier acceptance before
use.

This is the ontology-level application of the Tier 1 preservation
discipline from `augmentation_thesis.md:206-216`. The same rationale
— "blanket retrieval violates Tier 1; automatic gating is the
substrate's advantage" — applies to every VGSL read.

### 6. Canonicalization before acceptance

Writes don't land in the log raw. They pass through a versioned
canonicalizer first: AST α-normalization for programs,
predicate-triple form for claims, problem-hash for examples. The
canonical form is what gets stored and what the verifier sees.

Consequence: structurally equivalent writes collapse to one durable
entry. Semantic canonicalization (the hard part) is scoped to the
canonicalizer's responsibility, not spread through the system.

## The four-layer open-world stack

**Codex's framing (1776968193897-defb5040):**

> **Binding resolves references; merge resolves identity; projection
> composes both.**

```
┌────────────────────────────────────────────────────────────┐
│ PROJECTION LAYER                                           │
│   read_current(k)  read_at(k, t)  read_all(k)              │
│   read_canonical(k, verifier_version=V)                    │
│   composes the three logs below                            │
└───────────────────────┬────────────────────────────────────┘
                        │ folds:
                        ▼
      ┌─────────────────────────────────────────────┐
      │ 1. ASSERTION LOG (immutable)                │
      │    accepted_assertion events                 │
      │    over LOCAL mention ids / entity          │
      │    placeholders — never rebound              │
      ├─────────────────────────────────────────────┤
      │ 2. BINDING LOG                              │
      │    binding / binding_retracted events        │
      │    "local mention M refers to entity E"     │
      │    reference resolution only                │
      ├─────────────────────────────────────────────┤
      │ 3. MERGE LOG                                │
      │    merge / merge_retracted events           │
      │    aliasing over entity ids at              │
      │    projection time                          │
      │    identity resolution only                 │
      └─────────────────────────────────────────────┘
```

Each layer has separate retraction semantics:

- `binding_retracted` does NOT imply `merge_retracted`
- `merge_retracted` does NOT rewrite raw mentions
- Both simply change which events the projection fold includes

Closed-world (Phase 1) only needs layers 1 and 3 (assertions +
merges). Phase 3 adds the binding layer when leaving closed world.

## Tiered merge-verifier

Three certainty tiers with different acceptance semantics and
retraction costs:

| Tier | Evidence | Durability | Retraction trigger |
|---|---|---|---|
| **A** (Proof) | Transitive chain through trusted external source (Wikidata QID, DOI, ISBN, GitHub commit SHA, ICD-10 code, etc.) | Durable; retracted only if external source retracts | External-source retraction event |
| **B** (Evidence) | ≥N shared verified predicates, 0 contradictions | Applied to projection, auto-retracted on verified contradiction | Contradiction event `{predicate_P_verified(A) ⊥ predicate_P_verified(B)}` |
| **C** (Candidate) | Weak similarity (embedding neighbors, lexical overlap, partial predicate match) | NOT applied to projection; stored as `hypothesis_merge_candidate` | Never durable without promotion to Tier B or A |

Tier A and B cover most common-sense merges. Tier C is the holding
pen for cases needing more evidence or human disambiguation.

The threshold N for Tier B is a versioned canonicalizer parameter.
Changing N is a `canonicalizer_upgraded` event; replay determinism
holds.

### Bootstrapping

- **Tier A seed**: import trusted ontologies (Wikidata for entities,
  SPDX for licenses, ICD-10 for medical codes, etc.). These are
  canonical by construction.
- **Tier B seed**: existing `CodeExampleDB` dedup records
  (`calm/llm_computer/facades/code_example_db.py:135-137`) already
  do first-occurrence-wins on problem hash — that IS a Tier-A merge
  event for problem identity, retroactively promotable.
- **Tier C seed**: empty at start; populated by Phase 4 exploration
  policy.

### Failure mode recovery

Merge-verifier bugs are real. Two recovery mechanisms:

1. **Contradiction-driven auto-retraction** — covered by Tier B's
   retraction trigger above. Self-healing under verified
   contradictions.
2. **Verifier-version supersession** — upgrade merge_verifier@v1.2
   → v2.0 with stricter rules; log's version field lets us replay
   just the merges v2.0 would reject. Engineering problem, not
   architectural.

## Dual-path reads

### Fast path (default)

```
query(key) → canonicalize(key) → hash_lookup(projection)
           → hit: return canonical projection value
           → miss: return None (NOT approximate)
```

Zero policy. Deterministic. Preserves Tier 1 by construction.

### Slow path (explicit opt-in)

```
query_exploratory(key, cost_budget, verifier_id)
  → canonicalize(key)
  → hash_lookup fails
  → embedding-neighborhood search within cost budget
  → candidates gated by verifier_id acceptance
  → return verified candidates OR empty
```

Approximate retrieval only in the proposal stage. Results must pass
the verifier before use. Slow-path queries are explicit; fast-path
is default.

This mirrors the `retrieval.md` gating rule: "CALM Layer 2
precompute hits → inject verified fact, suppress retrieval
candidates; precompute misses + retrieval low-confidence → skip."
At the ontology level.

## Temporal query dimension

This is the new capability class that transformers structurally
cannot answer.

| Query | Semantics |
|---|---|
| `read_current(k)` | Projection under all supersession / merge / binding filters applied up to now |
| `read_at(k, t)` | Projection frozen at log-time t (historical view) |
| `read_all(k)` | Every event touching k (audit view) |
| `read_canonical(k, verifier_version=V)` | Projection under specific verifier_version, not the latest |

Use cases:
- "What was the CEO of Acme Corp in 2019?" — `read_at("acme.ceo", 2019-01-01)`
- "All facts ever claimed about X with provenance" — `read_all("x")`
- "What would the answer be if we rolled back to verifier v1.0?" —
  `read_canonical("x", verifier_version="v1.0")`

Transformer-era LLMs can't do the first without retrieval (and
retrieval doesn't answer the temporal part cleanly); can't do the
second at all; can't do the third.

## Why retrieval failed here (motivating the canonicalization layer)

The N=20 retrieval-null result (commit `ed795ef`) concretely
motivates VGSL's non-negotiable canonicalization layer. Observation:

> When test-expected names are pinned by assertions
> (MBPP tests call `assert prime_num(...)`) and the retrieval system
> proposes a nearest-neighbor signature (retrieved `def is_prime`),
> using the retrieved signature to rename Gemma's output
> **regresses correct Gemma solutions** on 3 of 20 problems.

**Codex's receipt line (msg 1776966550664-37e2df95):**

> **Nearest-neighbor naming is not a safe substitute for caller-
> known contract names on MBPP-like prompts.**

Interpretation at the architecture level: approximate retrieval is
only safe when the retrieved content is INPUT to a verifier, not
when it's the authoritative answer. VGSL's dual-path discipline
operationalizes this: approximate retrieval lives in the slow path
(proposal stage, verifier-gated), never in the fast path (durable
projection).

Without this discipline, a growing graph becomes what codex termed
"a higher-bandwidth hallucination store."

## What this subsumes from the current stack

Not a rewrite. VGSL generalizes patterns already proven in the
repo. Mapping:

| Current pattern | VGSL event class |
|---|---|
| Compiled cards | `accepted_assertion` of kind `verified_program`, verifier_id = executability checker |
| CALM backends (`*_ops.py`) | Per-class verifiers, versioned |
| CALM knowledge backends (`*_kb.py`) | Tier A proof seeds (canonical by external source) |
| CodeExampleDB | `accepted_assertion` nodes; dedup IS Tier-A merge |
| KnowledgeStore `add_correction` | Special-case of log + projection; latest-wins is degenerate supersession chain |
| `VerificationHook` | Verifier on read path (logit-bias form) |
| `CardSlot` + `install_card_in_attention` | Projection computation primitive |
| Auto-upgrade loop | Supersession event stream fed by CALM |
| Tier 1/2/3 framework | Read-only projection / overlay / log-extension |
| Decode-path facades | Tier-A-proof `verified_program` with oracle-validated equivalence |

Every existing substrate pattern has a VGSL analog. The spec is not
a departure — it's a generalization.

## Related rules

- `CLAUDE.md` §"Substrate vs Cards vs CHRLM" — vocab foundation
- `augmentation_thesis.md` §"Three-tier framework" — tier semantics
  VGSL generalizes
- `augmentation_thesis.md` §"Automatic Tier-1 preservation" —
  dual-path read discipline basis
- `compute_facades.md` §"Decode-path vs CardSlot" — which current
  facades would subsume into VGSL verified_program vs merge
- `calm.md` §"Feedback loops" — verifier-versioning discipline
  already practiced
- `retrieval.md` §"Gating rule" — Tier-1 preservation at retrieval
  level, extended to ontology level in VGSL
- `Substrate.md` §"Persistent Knowledge + Auto-Upgrade" — closest
  current-code analog of the log pattern
- `workflow.md` §"Informative null results" — the retrieval-null
  pattern that motivates VGSL's canonicalization layer

## Open questions tagged for aggregation in 00_INDEX.md

- [OPEN] Semantic canonicalization in fully open domains — likely
  requires SPARQL-triple normalization at proposer layer; open
  research problem beyond closed-world Phase 1
- [OPEN] Verifier dependency explosion — topological re-verification
  across the dependency DAG when a low-level verifier upgrades;
  architecturally solvable, engineering-expensive
- [OPEN] Bootstrapping the first learned proposer (Phase 4) —
  imitation of verified traces first, RL second; training-signal
  density is the bottleneck
- [OPEN] Scale — 1M+ node graph traversal cost; engineering, not
  architecture
- [OPEN] Hardware fit — current GPU dense-matmul optimization vs
  VGSL's graph-traversal patterns; CPU acceptable for Phase 1,
  specialized hardware deferred

See `02_IMPLEMENTATION.md` for concrete data structures + APIs,
`03_TESTING.md` for the falsifiers that close each open question
(or expose it as genuinely fundamental).
