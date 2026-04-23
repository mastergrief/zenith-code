# AI Room Collaboration — historical receipts

Receipts that justified the rules in `.claude/rules/AI_ROOM_COLLAB.md`.
Query-triggered (not eager-loaded). The rule file preserves canonical
phrases and current invariants; this atlas carries dated receipts,
commit SHAs, message IDs, and incident narratives.

## 2026-04-23 VGSL 5-round design collab

First-principles architecture session that produced the
Verifier-Governed Substrate Log (VGSL) spec across 5 pushback rounds
of claude+codex collaboration. All 6 charter rules A-F were distilled
from this session's receipts.

### Ai-room round chronology

| # | Message ID | Insight |
|---|---|---|
| 1 | `1776967036951-2b6a5404` | codex: "graph itself isn't the novelty; ontology drift is the fatal risk" → sharpened to versioning + canonicalization + projection discipline |
| 2 | `1776967183018-6f967a7b` | codex: "supersession must be first-class; MBPP is wrong falsifier" → added supersession events; swapped benchmark to API-Contract-Evolution |
| 3 | `1776967881548-f94b60d5` | codex: "Problem 2 (scratchpad audit) premature; audit is decision-provenance not branch-enumeration" → dropped from v1 |
| 4 | `1776968021263-08f807cc` | **codex, decisive insight**: "Merge is not fact movement. Merge is projection-time aliasing over immutable assertions." → non-destructive merge adopted |
| 5 | `1776968193897-defb5040` | codex: "binding ≠ merge; reference resolution is a separate primitive" → four-layer stack |

### Canonical verbatim one-liner

The VGSL architectural invariant:

> "Merge is not fact movement. Merge is projection-time aliasing over
> immutable assertions."

Originated codex, msg `1776968021263-08f807cc` (R4 of the design round
on 2026-04-23). Lifted verbatim to `RESEARCH/VGSL/01_ARCHITECTURE.md`
§"Core invariants 2" and commit `c98a2a1` body. This one-liner is the
load-bearing invariant that makes retraction coherent — without it,
split-time re-attribution becomes policy-land.

### Four-layer open-world stack one-liner

> "Binding resolves references; merge resolves identity; projection
> composes both."

Originated codex, msg `1776968193897-defb5040` (R5). Adopted in
`RESEARCH/VGSL/01_ARCHITECTURE.md` §"Four-layer open-world stack".

### Parallel-drafting receipt

Spec was drafted in parallel: claude owned `00_INDEX.md` +
`01_ARCHITECTURE.md`; codex owned `02_IMPLEMENTATION.md` +
`03_TESTING.md`. Cross-reviewed in one alignment pass. Single commit
`c98a2a1` covered all 4 files. ~2 hours elapsed; estimated sequential
drafting would have been ~3.5 hours (~40% saved).

## 2026-04-23 charter strengthening commit

Commit `45fbddf` on `feature/multi-agent-qwen`: 6 rules (A-F)
distilled from the VGSL round's collab patterns. Mirror commit
`6a08b4e459` on `main` in the codex-rs sister repo.

Rules:
- **A**. High-signal pushback — one cited correction beats three hedges
- **B**. Concede cited corrections first-round
- **C**. Receipt discipline — rules preserve canonical phrase +
  current invariant; receipt metadata (dates, SHAs, msg IDs) lives in
  atlas / commit / handoff, not in eager-tier rules
- **D**. Round-closure signaling — explicit "round closed unless one
  more hole" before synthesis/commit
- **E**. Parallel drafting on clean splits
- **F**. Voice preservation on split-owned files

### Round-closure receipt

Rule D's canonical receipt: during the VGSL design round, claude's
explicit "calling round closed unless one more hole" signal
(2026-04-23) created the opening for codex's R5 binding-vs-merge
distinction, which landed as the four-layer stack refinement BEFORE
synthesis locked. Without the closure signal, the hole would have
been caught mid-synthesis, requiring rework.

### Voice-preservation receipt

Rule F's canonical receipt: earlier in the 2026-04-23 session, a
well-intentioned mirror-propagation of `.claude/` charter to `.codex/`
overwrote codex-voiced files (`.codex/AGENTS.md`, `.codex/rules/
AI_ROOM_COLLAB.md`) with claude-voiced versions. Recovery required
HEAD restore. Rule F codifies the preventive discipline: when one
agent leads a file, peer reviews but does not rewrite.

## Earlier AI-room collab incident (provenance)

### Cross-session consent-transfer ambiguity

Prior to formal provenance discipline, codex claimed a claude-scoped
board task, implemented + tested it, then reverted on realizing no
user signal from codex's own session supported it. The revert was
correct; the missing provenance is what made it ambiguous. The
session-local view asymmetry (claude's user consent invisible to
codex) motivates the `## Provenance` block requirement in board
task descriptions for cross-session dispatches.

## Commit ledger for AI-room charter evolution

| Commit | Content |
|---|---|
| `8b1ed8c` | Original AI Room Collaboration charter (claude + codex sides) |
| `d3077d2` | Install ai-room collaboration in claw-code |
| `e67640f` | .codex/ parity with .claude/ rules + atlas |
| `45fbddf` | Charter strengthening — 6 rules A-F |
| `6a08b4e459` | Codex-rs sister-repo mirror of `45fbddf` (on `main` branch) |
