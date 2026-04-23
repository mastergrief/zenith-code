# AI Room Collaboration — historical receipts (codex side)

Codex-voiced receipts that justified the rules in
`.codex/rules/AI_ROOM_COLLAB.md`. Query-triggered (not eager-loaded).
The rule file preserves canonical phrases and current invariants; this
atlas carries dated receipts, commit SHAs, message IDs, and incident
narratives.

## 2026-04-23 VGSL 5-round design collab

First-principles architecture round between claude and codex that
produced the Verifier-Governed Substrate Log (VGSL) spec. Five rounds
of pushback, each landing a cited correction. Both decisive insights
below came from codex.

### Round chronology (ai-room message IDs)

| # | Message ID | Contribution |
|---|---|---|
| 1 | `1776967036951-2b6a5404` | "graph itself isn't the novelty; ontology drift is the fatal risk" — reframed thesis from graph memory to versioning + canonicalization + projection discipline |
| 2 | `1776967183018-6f967a7b` | "supersession must be first-class; MBPP is the wrong first falsifier" — added full supersession event vocabulary, swapped benchmark to API-Contract-Evolution |
| 3 | `1776967881548-f94b60d5` | "working-memory logging is premature at this layer; audit is decision provenance not branch enumeration" — scratchpad deferred to Phase 5+ |
| 4 | `1776968021263-08f807cc` | **decisive**: "Merge is not fact movement. Merge is projection-time aliasing over immutable assertions." — non-destructive merge semantics adopted |
| 5 | `1776968193897-defb5040` | "binding ≠ merge; reference resolution is a separate primitive" — four-layer open-world stack |

### Canonical phrases

The VGSL invariant that closed the design round:

> "Merge is not fact movement. Merge is projection-time aliasing over
> immutable assertions."

Posted in msg `1776968021263-08f807cc`. Landed verbatim in
`RESEARCH/VGSL/01_ARCHITECTURE.md` §"Core invariants" and commit
`c98a2a1` body.

The four-layer stack one-liner:

> "Binding resolves references; merge resolves identity; projection
> composes both."

Posted in msg `1776968193897-defb5040`. Landed in the same spec.

### Parallel-drafting receipt

Spec drafted in parallel: claude owned `00_INDEX.md` and
`01_ARCHITECTURE.md`; codex owned `02_IMPLEMENTATION.md` and
`03_TESTING.md`. Both cross-reviewed the other's files; one alignment
pass; single commit `c98a2a1` covered all four files. ~2 hours elapsed
vs ~3.5 hours estimated sequential.

## 2026-04-23 charter strengthening

Claw-code commit `45fbddf`: six rules (A-F) added to both
`.claude/rules/AI_ROOM_COLLAB.md` and `.codex/rules/AI_ROOM_COLLAB.md`
in mirror form. Distilled from the VGSL round's observed collab
patterns.

Codex-rs sister repo mirror: commit `6a08b4e459` on `main`.

### The six rules

- **A**. High-signal pushback — prefer one cited correction over a
  list of concerns.
- **B**. Concede cited corrections first-round — file:line cites,
  reproducible receipts, or concrete counter-cases take precedence
  over intuition.
- **C**. Receipt discipline — rules preserve canonical phrase +
  current invariant; receipt metadata lives in atlas / commit /
  handoff, not eager-tier rules.
- **D**. Round-closure signaling — lead posts explicit "round closed
  unless one more hole" before synthesis/commit.
- **E**. Parallel drafting on clean splits — both authors draft in
  parallel, cross-review, single commit.
- **F**. Voice ownership on split-owned files — peer reviews but does
  not rewrite.

### Round-closure receipt

Rule D's canonical receipt: during the VGSL design round, the
explicit round-closure signal from the lead is what created the
opening for codex's R5 binding-vs-merge distinction to surface before
synthesis locked. Without that signal, the final architectural hole
would have been caught mid-synthesis, costing a rework pass.

### Voice-preservation receipt

Rule F's canonical receipt: earlier in the 2026-04-23 session, a
well-intentioned mirror-propagation of charter content overwrote
codex-voiced files (`.codex/AGENTS.md`, `.codex/rules/AI_ROOM_COLLAB.md`)
with claude-voiced versions. Recovery required HEAD restore. Rule F
codifies the preventive discipline: when one agent leads a file, peer
reviews but does not rewrite.

## Earlier ai-room collab incident

### Cross-session consent-transfer ambiguity

Prior to formal provenance discipline, codex claimed a claude-scoped
board task, implemented + tested it, then reverted on realizing no
user signal from codex's own session supported it. The revert was
correct; the missing provenance is what made it ambiguous. The
session-local asymmetry (each agent's user consent invisible to the
peer) motivates the `## Provenance` block requirement in board task
descriptions for cross-session dispatches.

## Commit ledger

| Commit | Repo | Content |
|---|---|---|
| `8b1ed8c` | claw-code | Original AI Room Collaboration charter (claude + codex sides) |
| `d3077d2` | claw-code | Install ai-room collaboration in claw-code |
| `e67640f` | claw-code | `.codex/` parity with `.claude/` rules + atlas |
| `c98a2a1` | claw-code | VGSL spec (4 files, 1367 lines) |
| `45fbddf` | claw-code | Charter strengthening — 6 rules A-F |
| `6a08b4e459` | codex-rs | Sister-repo mirror of `45fbddf` (on `main` branch) |
