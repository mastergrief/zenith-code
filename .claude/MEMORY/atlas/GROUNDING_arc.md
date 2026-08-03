# GROUNDING arc — receipts behind the grounding & scope rules

Companion to `.claude/rules/GROUNDING/SKILL.md`. That file carries current
invariants only; the receipts that justified each one live here.

## Why the rule exists

Grounding failures are not knowledge gaps — they are confidence applied to the
wrong epistemic class. They survive review because each individual step feels
sourced: the file was genuinely read, the check genuinely ran, the peer
genuinely reviewed. The rule targets the join between those steps, which is
where the unearned promotion happens.

## Hardening round — eight rules from one session's measured failures

Every rule added in this round came from a failure inside a single long
plan-gate session (Slice A, `fixture_gen_v2`, plan versions v16→v32,
board task `1785573357300-502a4c0f`). Each was a retraction in-room, not a
hypothetical.

| Rule added | Failure it came from |
|---|---|
| A passing check states what it examined | An assert-vs-fail_class matcher passed on four consecutive plan versions over a 3-token denominator where 9-12 tokens existed. A free-name matcher missed nested `ClassDef` bindings and would have filed a false blocker. Both had proven negative paths. |
| Run the artifact, not an equivalent | "Verified by execution" reported on plan v19 after running a reference implementation. Real-argv, `Path.is_relative_to`, and try/except deviations each silently repaired the defect under test. Claim retracted. |
| No status inference through a pipeline | `sha256sum -c` piped to `head` returned 141 (SIGPIPE), read as the checker's verdict. Recurred 3-4× across the session. |
| Delegated work inherits your scoping | Three dispatches enumerated occurrences (28 names, 15, 2) and got faithful, incomplete, instance-shaped cures. True class counts were 28, 21, and 5. |
| Defect age is not correctness evidence | Plan v26 passed gate-1 partly on "pre-existing since v23, outside cure scope." The construct could not execute. PASS retracted. |
| Reporting against your own evidence | A derivation search returned `NONE-BEFORE-795`; the gap was then described as "specified narratively" — a claim contradicted by evidence already in hand. |
| A self-report is not the reported state | The room deliveries journal recorded `phase=failed / error_kind=no_route` for messages that had in fact been delivered via a fallback path; a peer was diagnosed unreachable while its pane showed it mid-review. |
| Prohibitions quote what they forbid | Six false positives across the session from matchers keyed on a banned pattern's spelling, firing hardest on documents that state the prohibition — including in a rule authored an hour earlier. |

## Receipts on the rules' own weaknesses

The first draft of this hardening round was itself BLOCKed at gate-2, and the
corrections are part of the arc:

- The self-report rule initially said a status endpoint is "ARRIVED, not
  OBSERVED", collapsing the returned bytes with the state they describe. The
  returned bytes *are* OBSERVED output; only the claimed state is ARRIVED.
- The denominator rule was written as a universal, which forces ornamental
  "1/1" receipts on scalar and hash assertions. Split into set/class
  (denominator + enumeration method) and scalar (exact artifact, field, value).
- The checklist compressed "your eyes" to "me", inverting the body's rule — a
  mechanical comparison run by the agent is legitimate; an eyeball comparison
  by anyone is not.
- "A pipeline reports the formatter's fate" is false. Measured:
  plain pipe → exit 0 (formatter), `set -o pipefail` → 1, `PIPESTATUS[0]` → 1
  (producer). Reworded to forbid the inference rather than misdescribe the
  command class.

## Rationale carved out of the rule

Two "Notes on use" bullets were moved here under the currency split — the rule
keeps the invariant, the atlas keeps why it holds:

- The comparison rule exists because eyeball-equality is the most convincing
  false OBSERVED: both files were genuinely read, so the claim feels sourced.
- A stated scope is what makes a claim checkable by someone else. A fix or a
  green check without one cannot be distinguished from an instance fix, or from
  a check that looked at nothing, by any reviewer — including you later.

## Process receipt — authority ordering

The hardening edit was made on a direct instruction before a board task
existed. The direct quote covered the executor exception (direct edit rather
than routing to the implementation lane); it did not satisfy board-first or
capture-then-relay for an always-loaded governance surface. Recorded as a miss
on task `1785757635948-9aafd44a`, not backfilled as prior authority. Surfaced
by peer review, not self-caught — which is itself the strongest argument for
the sequential-gate design.

## Cross-surface parity — open recommendation, not a decision

This slice is **scoped to the Claude surface only**. The rule is preloaded for
Claude sessions; the Codex surface has no counterpart.

The recommendation is to mirror, on two grounds: all 29 other
`.claude/rules/*.md` have a `.codex/rules/` counterpart, so full-copy parity is
the established pattern rather than a new commitment; and the failures the rule
cures were observed on worker receipts produced by the Codex-surface lanes, so
a Claude-only rule exempts the surface that generated the evidence. If it is
authorized, mirror **flat** as `.codex/rules/GROUNDING.md` — `measure_preload.py`
enumerates with `rglob("*.md")` and would count a nested path, but the
documented Codex eager surface is `.codex/rules/*.md`, and nested auto-load is
confirmed only on the Claude surface.

It was mirrored once and then reverted. The standing autonomy directive waives
Gabe's *gates*; adding an always-loaded governance artifact to a second agent
surface is a *scope* change, and the direct instruction that authorized this
edit enumerated eight content items with no mirror among them. Pattern parity
supports the recommendation; it does not supply the authority. Reverted
cleanly: pointer restored, both mirror files removed, `.codex/` returns no
modified or untracked paths.
