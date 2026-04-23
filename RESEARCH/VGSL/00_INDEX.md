# VGSL — Verifier-Governed Substrate Log

A non-weight knowledge-substrate hypothesis for a post-transformer
stack, refined over a 4-round collaboration between claude and codex
on 2026-04-23, then re-scoped post-joint-critique on the same date.

## What this is

A design spec for a **non-weight knowledge substrate for a
post-transformer stack** — a versioned, verifier-governed,
canonicalized event log with temporally-indexed projection that
externalizes the knowledge a transformer queries. Not a replacement
for transformers; a replacement for the destructive shortcuts
(`KnowledgeStore` latest-wins, `CodeExampleDB` first-wins) the
current substrate uses for persistent state.

Not a build plan. Not a kickoff. A durable record of the
architecture so a future session (or a future claude+codex round)
can pick up without losing the design.

**2026-04-23 update**: post-joint-critique (claude+codex
first-principles review), the spec's framing has been softened on
"post-transformer" and the recommended primary pursuit path has
been rescoped to a **staged in-tree drop-in upgrade** (see
§"Pursuit path"). The original greenfield Phase 1+ remains as the
long-term arc.

## Files

| File | Owner | Content |
|---|---|---|
| `00_INDEX.md` (this file) | claude | Overview, thesis, decision the user needs to make |
| `01_ARCHITECTURE.md` | claude | Primitives, invariants, four-layer open-world stack, what this adds vs current substrate |
| `02_IMPLEMENTATION.md` | codex | Data structures, event schemas, APIs, projection fold pseudocode |
| `03_TESTING.md` | codex | Falsifiers, benchmarks, success gates, eval rubrics |

## Thesis (one line)

> **Post-transformer is a versioned event log of verifier-accepted
> writes + temporally-indexed canonicalized projection + dual-path
> reads (exact-gated fast + verifier-bounded exploratory), with
> replay reproducibility as an architectural invariant.**

## The diagnosis

Every architecture this repo has worked with (Gemma, DT, Small2DTransformer,
CHRLM unified substrate) shares 9 structural constraints. Substrate work
has attacked 6; three remain load-bearing unfixed:

1. **Imitation-trained from human text** — ceiling is human expertise
2. **No structural verifier** — CALM + VerificationHook + KnowledgeStore
   (`Substrate.md:267-277`) are verifier-governed OVERLAYS, not a
   verifier-native base
3. **Single implicit worldview** — no temporal queries, no coexisting
   projections, no audit trail. Transformer has one "now" baked into
   weights; can't answer "what was true at time t?"

VGSL attacks (2) and (3) **immediately** via the log + projection
substrate. It only attacks (1) — imitation-trained — once a working
proposer/verifier loop generates beyond human text imitation
(Phase 4 in the original arc; not in scope for the in-tree Stage
1/2 pursuit). Phase 4 carries open training-signal-density risk
(see `01_ARCHITECTURE.md` §"Open questions" + `00_INDEX.md`
§"Risks"). Honest framing: VGSL closes 2 of 3 constraints
deterministically; the third is gated on a separate research bet.

## Design round receipts

Four rounds of codex pushback shaped the final design. Full transcript
is in the ai-room log (`/home/gabe/.ai-room/channels/claw-code`);
message IDs for each round:

| # | Codex's push | Outcome |
|---|---|---|
| 1 | `1776967036951-2b6a5404` — "graph itself isn't the novelty; ontology drift is fatal risk" | Sharpened claim to versioning + canonicalization + projection discipline |
| 2 | `1776967183018-6f967a7b` — "supersession must be first-class; MBPP is wrong falsifier" | Added supersession events; swapped benchmark to API-Contract-Evolution |
| 3 | `1776967881548-f94b60d5` — "Problem 2 premature; audit is decision-provenance not branch-enum" | Dropped scratchpad audit from v1; pivoted depth to merge-verifier |
| 4 | `1776968021263-08f807cc` — "merge is not fact movement; it's projection-time aliasing" | **Decisive insight.** Non-destructive merge semantics adopted. |
| 5 | `1776968193897-defb5040` — "binding ≠ merge; reference resolution is a separate primitive" | Four-layer stack: assertion / binding / merge / projection |

Claude's role was synthesis + thesis shape. Codex's role was
grounded correction with file:line cites. Both one-liners credited
to codex are verbatim in `01_ARCHITECTURE.md`:

- "Merge is not fact movement. Merge is projection-time aliasing over immutable assertions."
- "Binding resolves references; merge resolves identity; projection composes both."

## Core primitives (summary — full spec in 01_ARCHITECTURE.md)

- **Append-only event log** — versioned (verifier_id@vN + canonicalizer_id@vN per write)
- **Four event classes**: `accepted_assertion`, `binding`, `merge`, `hypothesis` + their retraction/upgrade counterparts
- **Tiered merge-verifier** — Tier A proof / Tier B evidence-retractable / Tier C candidate-only
- **Non-destructive merges** — assertions stay on original local ids; merges are projection-time aliases
- **Dependency-tracked derivations** — retraction → `invalidated_pending_reverify`, no arbitrary reassignment
- **Dual-path reads** — exact-hash fast (zero policy, preserves Tier 1 per `augmentation_thesis.md:206-216`) + verifier-bounded exploratory
- **Temporal queries** — `read_current`, `read_at(t)`, `read_all`, `read_canonical(verifier_version)`

## What this subsumes from the current stack

Not a rewrite. VGSL is the generalization of patterns the repo has
already proven; the current stack becomes special-cases of the log.
The **Stage swap target** column shows the staged pursuit path
(see §"Pursuit path" below):

| Current pattern | VGSL equivalent | Stage swap target |
|---|---|---|
| Compiled cards (`calm/llm_computer/programs/*.py`) | `accepted_assertion` of kind `verified_program` | Stage 3+ generalization |
| CALM backends (`calm/backends/*_ops.py`, `*_kb.py`) | Per-class verifiers (versioned) | Stage 3+ generalization |
| **CodeExampleDB** (`calm/llm_computer/facades/code_example_db.py`) | `accepted_assertion` nodes per source-tagged example; source-priority is **projection policy** (`source_priority_v1`), not merge field | **Stage 1 — shadow-mode swap** |
| **KnowledgeStore** (`calm/llm_computer/persistent_knowledge.py:71-76`) | Special-case of log + projection; latest-wins is a degenerate supersession | **Stage 2 — paired with `auto_upgrade`** |
| **Auto-upgrade loop** (`calm/llm_computer/auto_upgrade.py`) | Supersession event stream; recall-card-weight compilation moves to projection-builder output | **Stage 2 — paired with `KnowledgeStore`** |
| Tier 1/2/3 framework (`augmentation_thesis.md`) | Read-only projection / overlay / log-extension at base layer | Stage 3+ generalization |

Source-priority encoding correction (joint critique, codex msg
`1776979663929-802bd974`): the original draft framed
`CodeExampleDB`'s first-occurrence-wins as "Tier-A merge by problem
hash." That conflates identity-resolution with representative-
selection. **Merge says "these assertions share problem identity";
projection says "of these N aliased records, this one represents
the cluster under policy P."** Source-priority lives in
`accepted_assertion.source_tier` + `source_rank` + tie-breaker
`(corpus_order, line_no, event_id)` as assertion metadata; the
projection rule (`source_priority_v1` for closed-world Stage 1)
selects the representative. Full spec: `02_IMPLEMENTATION.md`
§"Source priority encoding."

## Pursuit path (post-joint-critique)

The original 5-Phase greenfield rollout (below) remains the long-term
arc. Post-joint-critique (claude+codex first-principles review,
2026-04-23), the recommended **primary pursuit path** is a staged
in-tree drop-in upgrade that delivers audit/replay value without
claiming architectural novelty.

### Stage 1 — `CodeExampleDB` shadow-mode swap (~1-2 weeks if pursued)

Replace `CodeExampleDB`'s implicit load-order-priority with an
explicit VGSL-backed event log + projection. One policy:
`source_priority_v1`, which **materializes the current `examples`
ordering bit-for-bit**. No new behavior; same retrieval, same
indices, same ABI. Underneath, every example is an
`accepted_assertion` event with `source_tier` + `source_rank` +
tie-breaker as metadata.

**Four success gates** (per `03_TESTING.md`):

1. **Parity gate** — VGSL-backed wrappers must be behavior-equivalent
   to current `CodeExampleDB` first. Same retrieval results across
   jaccard/tfidf/channel modes, same `examples` ordering, same cache
   compatibility. **Entry ticket.**
2. **Audit gate** — for every duplicate problem hash, VGSL must
   explain *why* the active example won (event ids + source priority
   + verifier/canonicalizer version). Forced auditability.
3. **Performance gate** — ≤10% overhead on load/index-build, O(1)
   active lookup. Concrete budget.
4. **Value gate** — one-week shadow-mode operation; surfaced
   coherence bugs are *upside*, not entry ticket.

### Stage 2 — `KnowledgeStore` + `auto_upgrade` swap

Once Stage 1 ships and projection-cache + parity-harness
infrastructure exist, replace `KnowledgeStore.add_correction`
(latest-wins) with `accepted_assertion` events; recall-card-weight
compilation moves to projection-builder output (run when `.pt` is
saved). Stage 2 uses the `latest_verified_correction_v1` projection
policy (parallel to Stage 1's `source_priority_v1`). Same four
gates apply. Full implementation shape:
`02_IMPLEMENTATION.md` §"`KnowledgeStore` assertions" + §"Stage-2
compatibility APIs."

### Stage 3+ deferred

Multi-projection modes (Conservative/Pragmatic/Exploratory),
open-world bindings/merges, learned proposers, the original
greenfield Phase 1+ — all explicitly out of scope until Stages 1-2
prove out, OR a regulated vertical demands the audit story directly.

### Why this is sharper than the original 3-option decision

| Original Option 2 (greenfield Phase 1) | Reframed Stage 1 |
|---|---|
| Build API-Contract-Evolution Bench (synthetic) | Replace existing in-tree subsystem with measurable failure modes |
| Success gate = ≥80% on bench | Parity-first; audit utility second |
| Doesn't move existing eval surface | Improves audit/debugging on existing workflow immediately |
| Doesn't claim novelty (per joint critique) | Doesn't claim novelty — claims *utility* on real failure modes |
| ~2-3 sessions | Stage 1 estimated ~1-2 weeks (real work, not vibes) |

Two of the three real risks the joint critique flagged also shrink:

- **Open-world canonicalization TODO** — doesn't apply; both subsystems are closed-world
- **Phase 4 learned proposer might fail** — doesn't apply; no learned proposer in Stage 1/2

The remaining risk — **R53 scripts touch `db.examples` directly**, requiring wrapper to preserve list-ordering as ABI — is mitigated by `source_priority_v1` materializing `examples` order bit-for-bit.

## What this adds vs current substrate

The honest framing post-joint-critique: VGSL is a **knowledge
substrate that complements transformers**, not a replacement for
them. Generation, language understanding, and attention all still
live in the transformer. What VGSL adds:

1. Knowledge is explicit, typed, persistent — not in weights
2. Truth can change monotonically via supersession — not fixed at training time
3. Audit is durable and replayable — not opaque
4. Correctness is structural — writes gated by verifier, not sampled from probability distribution
5. New capabilities are verified compositions — not fine-tunes or RLHF
6. Temporal queries are a new capability class — transformers structurally cannot answer "what was true at t?"
7. Multiple projections coexist — conservative / pragmatic / exploratory views on the same log
8. Replay reproducibility is an architectural invariant — not a nice-to-have

Joint-critique caveat: items (1)-(5) describe **post-implicit-storage**
patterns shared with existing event-sourcing systems (Datomic,
Kafka + materialized views, RDF/SPARQL with revision history). The
**load-bearing novelty** for an LLM-knowledge substrate is the
specific composition: per-event verifier+canonicalizer versioning,
non-destructive merge as projection-time aliasing, dependency-tracked
derivation invalidation, and exact-default reads with verifier-gated
slow path. Item (6) is genuinely new for transformer stacks but is
a niche capability class (audit verticals + version-controlled
artifacts), not a general LLM workload need.

## Open questions (aggregated from all 4 files)

All `[OPEN]` tags across `01_ARCHITECTURE.md`, `02_IMPLEMENTATION.md`,
and `03_TESTING.md` consolidated here. Each is architecturally or
empirically genuine — not TODOs, not blockers for Phase 1.

### Architecture-level (from `01_ARCHITECTURE.md`)

- **Semantic canonicalization in fully open domains** (beyond
  predicate-hash) — probably requires adopting a canonical form like
  SPARQL triples at the proposer layer. Phase 3+.
- **Verifier dependency explosion** — when a low-level verifier
  upgrades, topological re-verification across the dependency DAG.
  Architecturally solvable (all deps in log); expensive at scale.
- **Bootstrapping the first learned proposer** — hand-written in
  Phase 1; learned policy in Phase 4 via imitation of verified
  traces first, RL second. Training-signal density is the
  bottleneck.
- **Scale** — 1M+ node graph traversal cost. Engineering, not
  architecture.
- **Hardware fit** — current GPUs are dense-matmul-optimized; VGSL
  may benefit from event-driven / neuromorphic hardware eventually.
  CPU is fine for Phase 1.

### Implementation-level (from `02_IMPLEMENTATION.md`)

- **Exact event schema for `projection_rule_changed`** — rare event
  but replay under a new projection rule is otherwise not
  deterministic; schema must be specified before Phase 1 ship.
- **Cluster-id policy for merges** — fresh synthetic id vs oldest
  surviving member id. Fresh synthetic ids cleaner for replay +
  split at cost of one more indirection.
- **Eager vs lazy re-verification scheduling** — when a merge
  retracts and derivations need reverify. Engineering choice with
  different latency/cost tradeoffs; architecturally neutral.
- **External snapshot hashing standard** — how verifier records
  external corpus dependencies for reproducibility. Needs to
  standardize hash format + retention policy.
- **Whether `hypothesis` remains a record class in closed-world
  deployments** or is deferred entirely to open-world / debug
  builds — cross-file alignment between my 01 and codex's 02 leaves
  this genuinely open.

### Testing-level (from `03_TESTING.md`)

- **Hand-labeled eval size for Tier-B merge precision** — how many
  labeled examples needed to trust the merge-verifier's rules.
  Empirical, driven by domain.
- **Eager vs lazy verifier-upgrade replay** — whether replay runs at
  upgrade time or on first read. Related to implementation's
  re-verification scheduling open question.
- **Whether exploratory reads get their own benchmark** — or remain
  a diagnostic mode only. Depends on Phase 4 scope.
- **Whether closed-world Phase 1 needs any `claim` record class at
  all** — or can stay entirely in `verified_program`. Affects
  whether Phase 1 is pure code-contract bench or allows typed
  claims.

## Phased rollout (summary — details in 03_TESTING.md)

**Stage 0 — In-tree shadow swap (recommended primary path post-joint-critique)**

| Stage | Scope | Falsifier |
|---|---|---|
| 1 | `CodeExampleDB` shadow-mode swap with `source_priority_v1` policy | 4 success gates: parity / audit / performance / value (existing subsystem IS the baseline) |
| 2 | `KnowledgeStore` + `auto_upgrade` shadow-mode swap | Same 4 gates; recall-card-weight compilation moves to projection-builder output |

**Greenfield arc (long-term, deferred unless vertical demands it)**

| Phase | Scope | Falsifier |
|---|---|---|
| 1 | Log semantics + versioned replay + projection + hand-written proposer | API-Contract-Evolution Bench (closed world) **with event-sourced/temporal-table baseline** (per joint critique Finding 4) |
| 2 | Verifier-upgrade replay semantics | 1000-write v1→v2 upgrade, confirm correct rejections |
| 3 | Open-world: binding log + Tier-A merge-verifier | Entity-resolution eval, 0 Tier-B wrong merges |
| 4 | Tier-B evidence-threshold merges + exploration policy | Task-specific (Tier B is **research frontier**, not solved common-sense merge — see `01_ARCHITECTURE.md` §"Tiered merge-verifier" for correlated-evidence + missing-contradiction risks) |
| 5 | Optional: scratchpad/hypothesis audit if a use-case emerges | N/A |

**Not in v1**:
- Learned policy — storage semantics are the frontier, not routing
- Working-memory primitives — hand-written proposer doesn't need them
- MBPP as benchmark — retests the already-ruled-out "contract-name inference from nearest neighbor" failure mode (see `01_ARCHITECTURE.md` §"Why retrieval failed here" — credited to the N=20 retrieval-null result, commit `ed795ef`)
- Multi-projection (Conservative/Pragmatic/Exploratory) for Stage 1 — **only one explicit policy** (`source_priority_v1`) materializing current `examples` order bit-for-bit. Multi-projection deferred to Stage 3+.

## Decision the user needs to make

VGSL is an R&D direction, not a shipping arc. Post-joint-critique,
the recommended primary pursuit path is the **in-tree Stage 1 slice**.
Original 3 options remain as alternatives.

**Primary recommendation — Stage 1 in-tree `CodeExampleDB` shadow-mode swap.**
~1-2 weeks if pursued. Deliverable is the shadow-mode wrapper
passing the 4 success gates (parity / audit / performance / value).
Stage 2 (`KnowledgeStore` + `auto_upgrade`) conditional on Stage 1
success. Doesn't claim novelty; claims utility on real failure
modes (latest-wins + first-wins overwrites that fail compliance
audits at scale). See §"Pursuit path" above.

**Alternative 1 — Park.** File this spec as a receipt. Continue
shipping substrate cards on the current stack. Revisit VGSL when
the current stack hits an audit-shaped ceiling (regulated vertical,
compliance ask, debug-coherence pain).

**Alternative 2 — Greenfield Phase 1 prototype.** 2-3 sessions to
build log + projection + API-Contract-Evolution bench. Original
Option 2 from pre-critique synthesis. Per joint critique: only
worth it if framed as **semantics proof, not novelty proof**, with
event-sourced/temporal-table baseline included and explicit "Phase 1
validates semantics only; Phase 3 validates distinctive
merge/retraction value" scoping. Testable within a week. See
`03_TESTING.md` §"Greenfield Phase 1 Falsifier - API-Contract-Evolution
Bench" for the bench definition.

**Alternative 3 — Scope bound to a commercial vertical.** Pick one
regulated domain (legal citations, medical codes per `commercial.md`'s
vertical deck thesis) where the closed-world discipline + audit
story IS the product. Sell as "auditable knowledge substrate for
regulated verticals," NOT as "post-transformer." Narrower scope,
stronger product hypothesis.

**Joint claude+codex lean** (post-critique synthesis): Stage 1
in-tree slice if there's appetite for the architectural payoff
without committing to greenfield. Alternative 3 (vertical) if a
concrete regulated-vertical play is in view now. Alternative 1
(park) if neither vertical pressure nor curiosity budget is
present. Alternative 2 (greenfield Phase 1) only as bounded
preflight for Alternative 3.

## Related documents

- `CLAUDE.md` §"Substrate vs Cards vs CHRLM" — vocab this spec
  extends
- `augmentation_thesis.md` — Tier 1/2/3 framework this spec
  generalizes
- `compute_facades.md` — decode-path facade pattern that VGSL would
  subsume
- `calm.md` §"Feedback loops" — verifier discipline VGSL adopts
  wholesale
- `Substrate.md` §"Persistent Knowledge + Auto-Upgrade" — the closest
  current-code analog
- `retrieval.md` §"Gating rule" — Tier-1-preservation discipline
  VGSL's dual-path reads apply at ontology level
- `workflow.md` — hypothesis-test-iterate discipline this spec was
  developed under
