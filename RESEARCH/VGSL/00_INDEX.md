# VGSL — Verifier-Governed Substrate Log

A post-transformer architecture hypothesis, refined over a 4-round
collaboration between claude and codex on 2026-04-23.

## What this is

A design spec for an architecture that moves knowledge out of opaque
transformer weights into a **versioned, verifier-governed,
canonicalized event log** with temporally-indexed projection. Not a
build plan. Not a kickoff. A durable record of the architecture so
a future session (or a future claude+codex round) can pick up
without losing the design.

## Files

| File | Owner | Content |
|---|---|---|
| `00_INDEX.md` (this file) | claude | Overview, thesis, decision the user needs to make |
| `01_ARCHITECTURE.md` | claude | Primitives, invariants, four-layer open-world stack, why this is post-transformer |
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

VGSL attacks all three at the architectural base.

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
already proven; the current stack becomes special-cases of the log:

| Current pattern | VGSL equivalent |
|---|---|
| Compiled cards (`calm/llm_computer/programs/*.py`) | `accepted_assertion` of kind `verified_program` |
| CALM backends (`calm/backends/*_ops.py`, `*_kb.py`) | Per-class verifiers (versioned) |
| CodeExampleDB (`calm/llm_computer/facades/code_example_db.py`) | `accepted_assertion` nodes; first-occurrence-wins IS a Tier-A merge by problem hash |
| KnowledgeStore (`calm/llm_computer/persistent_knowledge.py:71-76`) | Special-case of log + projection; latest-wins is a degenerate supersession |
| Auto-upgrade loop (`calm/llm_computer/auto_upgrade.py`) | Supersession event stream |
| Tier 1/2/3 framework (`augmentation_thesis.md`) | Read-only projection / overlay / log-extension at base layer |

## What makes this post-transformer (not just "better RAG")

1. Knowledge is explicit, typed, persistent — not in weights
2. Truth can change monotonically via supersession — not fixed at training time
3. Audit is durable and replayable — not opaque
4. Correctness is structural — writes gated by verifier, not sampled from probability distribution
5. New capabilities are verified compositions — not fine-tunes or RLHF
6. Temporal queries are a new capability class — transformers structurally cannot answer "what was true at t?"
7. Multiple projections coexist — conservative / pragmatic / exploratory views on the same log
8. Replay reproducibility is an architectural invariant — not a nice-to-have

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

| Phase | Scope | Falsifier |
|---|---|---|
| 1 | Log semantics + versioned replay + projection + hand-written proposer | API-Contract-Evolution Bench (closed world) |
| 2 | Verifier-upgrade replay semantics | 1000-write v1→v2 upgrade, confirm correct rejections |
| 3 | Open-world: binding log + Tier-A merge-verifier | Entity-resolution eval, 0 Tier-B wrong merges |
| 4 | Tier-B evidence-threshold merges + exploration policy | Task-specific |
| 5 | Optional: scratchpad/hypothesis audit if a use-case emerges | N/A |

**Not in v1**:
- Learned policy — storage semantics are the frontier, not routing
- Working-memory primitives — hand-written proposer doesn't need them
- MBPP as benchmark — retests the already-ruled-out "contract-name inference from nearest neighbor" failure mode (see `01_ARCHITECTURE.md` §"Why retrieval failed here" — credited to the N=20 retrieval-null result, commit `ed795ef`)

## Decision the user needs to make

VGSL is an R&D direction, not a shipping arc. Three ways to proceed:

**Option 1 — Park.** File this spec as a receipt. Continue shipping
substrate cards on the current stack. Revisit VGSL when current stack
hits a ceiling.

**Option 2 — Phase 1 prototype.** 2-3 sessions to build log +
projection + API-Contract-Evolution bench. Testable within a week.
Gives concrete signal on whether the architecture is real. See
`03_TESTING.md` §"Phase 1 falsifier" for the bench definition.

**Option 3 — Scope bound to a commercial vertical.** Pick one
regulated domain (legal citations, medical codes per
`commercial.md`'s vertical deck thesis) where the closed-world
discipline + audit story IS the product. Narrower scope, stronger
product hypothesis.

**Claude's lean**: Option 2 for architectural signal, Option 3 for
commercial leverage, Option 1 if the current stack's ceiling isn't
in view.

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
