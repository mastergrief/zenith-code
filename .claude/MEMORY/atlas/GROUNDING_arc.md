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

**SUPERSEDED — do not act on the mirror option above.** The Codex doc-mirror was
retired on Gabe's direction; see §"Five adopted invariants" below. The conditional
"if authorized, mirror flat" is kept as the record of what was considered, not as
a live option.

It was mirrored once and then reverted. The standing autonomy directive waives
Gabe's *gates*; adding an always-loaded governance artifact to a second agent
surface is a *scope* change, and the direct instruction that authorized this
edit enumerated eight content items with no mirror among them. Pattern parity
supports the recommendation; it does not supply the authority. Reverted
cleanly: pointer restored, both mirror files removed, `.codex/` returns no
modified or untracked paths.

## v2 additions — receipts from the 2026-08-03 session (the rule's first full day in force)

Five additions landed in v2, each pinned to a failure that occurred WHILE v1
was loaded — the honest test of what the rule was missing.

Sources for the numbered claims below. Audit headline posted in room record
`1785830156565-b7b2d61a` (≥13 retracted gate-1 PASSes, ~87 plan versions =
29 predecessor + 58 Slice A, ≥52 gate cycles, last GPU artifact Jul 23).
Day ratio: journal sweep over
`~/.ai-room/channels/claw-code/messages.jsonl`, window
2026-08-02T18:00:00Z–2026-08-03T22:14:59Z — window total **800** re-verified
by replayable count this session; the 778/8 ceremony/science split is the
audit-session keyword classification (session transcript
`97ee46eb-3bb4-4d0d-ae4b-5556e3198f55.jsonl`; no standalone room record).
Version counts replayable (directories only — a bare `v[0-9]*` glob also
matches two sibling `.json` files and returns 60, the property-vs-spelling
defect this very rule targets; caught at gate-2):
`find /home/gabe/plan-dev-scratch/repin -maxdepth 1 -type d -regextype
posix-extended -regex '.*/fixture_gen_v2_plan_v[0-9]+$' | wc -l` → 29;
same with `fixture_gen_v2_slice_a_plan_v[0-9]+$` → 58 (both re-run
2026-08-04).

1. **Denominator source** (Verification discipline). Root cause of the
   withdrawn-PASS class — ≥13 per audit record `1785830156565-b7b2d61a`,
   enumerated to 15 in the session-transcript sweep (v19, v26, v32, v34, v36,
   v38, v42, v44, v47, v48, v50-v53, plus the v58 implementation PASS):
   obligations enumerated from the dispatch's restatement, the receipt's
   claims, or the prior round's blocker list — never the frozen spec.
   Recurred at three successively finer granularities (v51→v52→v53) before
   being named.
2. **Executes, not mutates** (Cure the class). The "structurally closed"
   residual retraction (integration v5 gate-1 `1785794370407`, retracted in
   remint dispatch `1785794730905`): effect surface enumerated over mutation
   sections only; §8 validation also ran and wrote (2069 ignored entries
   invisible to the preserve equation; stale `.pyc` for the exact land
   modules).
3. **Proportionality** (new section). fixture-gen-v2 consumed ~87 plan
   versions (29 predecessor + 58 Slice A — replayable dir counts above) and
   ≥52 gate cycles (audit record `1785830156565-b7b2d61a`) for a CPU fixture
   migration; the loop measured its own pathology (Aug 2 on-record, "29
   versions / 34h / zero implementation") and continued anyway. The
   3-cure-rounds trigger is workflow.md's plateau rule ported to verification.
4. **Retraction rate** (Report faithfully). ≥13 individually-correct
   gate-1 PASS retractions in one day (audit record `1785830156565-b7b2d61a`;
   enumerated to 15 in the session-transcript sweep), none treated as data
   about the producing process.
5. **Symmetric calibration** (Verification discipline; replaces the
   negative-path-only bullet). During the 2026-08-03 session an 18-case
   battery (case count per room record `1785439207486-ecfe92a8`, the
   post-cure 18/18 set-equality PASS in the same lineage) produced false
   failures on a known-correct artifact across its full case set before
   recalibration — session-transcript observation; no standalone room record
   of the inverted run, so the exact-count causal claim is stated softly
   here. Two integration-battery hostiles likewise nearly blocked correct
   work in the v5–v7 rounds. These are positive-path failures the v1 text
   covered only for prohibition matchers.

Paid for by merging the ARRIVED/reviewer bullets, compressing the
system-self-report bullet, and condensing Notes on use. 149 → 161 lines
(target 150, hard cap 200; overage disclosed at review).

Queued follow-up, NOT this slice: mechanical enforcement of the two verdict
fields (gate-1 checklist / PreToolUse hook on room PASS posts) — prose decays,
hooks don't; same lesson as the staged-digest gate hook.

Related same-day evidence: R2→R3 of the integration plan (an "exhaustive"
git-metadata allowlist that omitted porcelain-implied writes — cured by
authorizing effects BY CAUSE, not by list; co_lead `1785831090174`), and 8
substring over-fires of the staged-digest gate classifier in one day (task
`1785437094843-e23dd080`), two of them triggered by prose *describing* this
very work — the prohibition-quotes-what-it-forbids bullet, live.

## Five adopted invariants (2026-08-08) — Gabe-directed, direct-implementation exception

Five rules adopted in-room across one session and living only in the channel
journal until now. Absence re-derived before the edit, not adopted from the
relay: five property-keyed patterns over `.claude/rules/` returned 0 hits each,
and the same patterns fired 4-100x against the room journal — matcher calibrated
in both directions, per the rule it was checking for.

1. **Empty-denominator = FAIL** (`1786213635220-49f8589b` thread). A sweep whose
   denominator resolves empty exits nonzero. Zero occurrences and zero scope are
   indistinguishable from outside, and the second reports success.
2. **Inventory-denominator + totality question** (same thread). Every cure to
   that point had derived its scope from the *bounced artifact's structure*
   (`section 2`, `### F` headers, `clean_conjuncts` vocabulary) rather than from
   its claims. The advisor recorded its own fingerprints on the instance-6
   opening: it prescribed fixture-vocabulary scoping and named the guarantee
   list for the stub but not for the rules.
3. **Two-implementation differential** (`1786214563446-0b1f53a2`). Sharp property
   is *post-choice binding*: a check bound to an object produced downstream of
   the deciding event cannot see the choice. Calibrate by running the conforming
   AND the declared-unsafe implementation through the fixtures.
4. **Deny-by-default attribution** (`1786219135553-2346de8b`). Measured by
   binding position rather than time: the class was migrating outward from
   operative positions to annotation positions, because prereg'd classifiers and
   pinned witnesses already occupied the operative layer. The cure is not a
   fifth guard against writing the sentence — it is making the sentence inert.
5. **Generate-don't-transcribe** (`1786225189316-0b955181`). Prior cures all
   verified a hand-made transfer after the fact; this one removes the transfer.
   Held on its own surface the same night: typed checkpoint basenames in the
   PHASE4D packet went 3 -> 0.

Placed into existing sections rather than appended as a block. 169 -> 193 lines
(hard cap 250). Gate EXIT=0 under the surface it now enforces: `measure_preload.py
--max-tokens 150000` (bare, default now `claude`) reports 13 eager files / 1887
lines / ~27169 tokens. `--surface both` remains available and reports 24 files /
~47049 tokens — the figure the pre-retirement runs cited.

Item 4 landed on its second attempt. The first draft asserted that an attribution
claim "binds only from a schema-owned field" and that "the same claim in prose is
void". co_lead blocked it: that is the half of the advisor's framing I had
**rejected** on the record in three durable places, and it was being restated
inside the file whose own §"Report faithfully" calls restating a retraction a
distinct failure. The live counter-example is ours — `traces_truncate_or_absent`
is schema-owned *and* was defective (a bare `else` on a four-conjunct AND, atlas
§7 defect #4, which is how arm A got a truncation label at `too_long == 0`). So
"schema-owned ⇒ binds" was already falsified in this repo. The landed text binds
on **basis + locator regardless of layer** and keeps the field layer as a
*survival* claim, not a truth claim.

**Codex mirror not written, and the mirror instruction is being retired** on
Gabe's direction ("you can delete wherever it says to do codex mirrors"). There
was no codex-side grounding analogue to mirror into: no `GROUNDING` file under
`.codex/rules/`, and 0 hits for the observed/inferred/arrived vocabulary across
that surface.

**Same-session evidence for items 3 and 5, from the gate that adopted them.**
Gate-1 on the PHASE4D packet raised six blockers across four passes and
*produced two of them itself*: a reachability derivation whose fixture sampled
six rows that all shared the same value of the deciding variable — shipped with
a true-but-irrelevant execution receipt attached, which is what let it survive
two reviews — and a resolver prefilter that excluded the files it was resolving.
Filed as **discrimination** (evidence structurally cannot decide the
proposition), not content-provenance, per the sorting rule at
`1786226920534-236ef49c`: source exists and claim diverges -> content-provenance;
evidence cannot decide -> discrimination. The reviewing instrument generating the
defects the review exists to catch is why the same audit proposed gate-symmetry:
blockers resting on a derivation owe the same pre-registration fields a packet
check owes. Adopted in-room, NOT in this change's scope.
