# Safe Terminology — arc receipts

Receipts justifying `.claude/rules/safe_terminology.md`. Query-triggered; not preloaded.

## Origin — LANDS-AB packet-validator hardening (Stage B.1)

The validator-hardening slice ran seven review rounds of exemption-logic defects.
Each round's findings were written up in offensive-security framing in room posts,
receipts, and new negative-case identifiers.

Gabe (chat-side, verbatim, in order):

1. "dont use words like launder etc please as it can trigger safeguards"
2. "attacker, launder etc"

Rationale as stated: the wording risks tripping safety classifiers. The artifacts in
question are durable — committed test identifiers, tool comments, and O_EXCL-minted
receipts that later sessions and other models read — so the exposure is ongoing, not
confined to one conversation.

Relayed to the worker lane as two dispatch amendments (terminology directive +
amendment 1) covering substitutions, scope, the entrenched-convention boundary, and
the no-extra-run constraint. Room msg ids live on the board.

## Concrete renames ordered in that slice

- one new negative-case identifier: `..._CANNOT_LAUNDER_OUT_OF_REPO_REF_FAIL`
  → `..._CANNOT_INJECT_OUT_OF_REPO_REF_FAIL`
- two fixture field names: `launder_path` / `launder_sha256`
  → `injected_path` / `injected_sha256`

Left untouched **at that time** by explicit decision: the `_FAIL` suffix convention
and the then-permitted charged phrasing for negative-case tests, both already spanning
committed Phase A lineage and prior gate records. Renaming those would have inflated
the slice diff and broken traceability to the records authorizing the work. **That
carve-out was later withdrawn — see "Full pivot" below.**

## Full pivot — the entrenched-convention exemption is withdrawn

Gabe (chat-side, verbatim): **"well we need to completely pivot away from using this
terminology 100%"**, following an assessment that the rule's own denylist table was
itself an exposure surface because eager-tier rules preload into every session.

Captured via `AskUserQuestion`; two locked answers:

1. **Scope/timing** — stop immediately in all writing; the existing corpus is renamed
   in a **dedicated slice after the in-flight validator work commits**.
   *Rejected*: folding a corpus rename into the live slice (extra suite run, diff
   inflation past scope, breaks the citation chain from six rounds of gate records);
   *rejected*: going-forward-only with no corpus rename ever.
2. **Entrenched carve-out** — the charged phrasing for negative-case tests is retired
   too; **`_FAIL` is kept** (pure outcome marker, most-cited suffix in frozen records).
   *Rejected*: retiring `_FAIL` as well (largest possible identifier churn);
   *rejected*: keeping the carve-out.

Rule restructured to **positive-only** form as a result: the eager-tier file states
concept → approved term and carries none of the retired vocabulary, so the rule stops
being its own false-positive surface. Enumeration for mechanical checking moved here.

Two gaps found and closed in the same edit: the rule had **no `.codex/` mirror** at
all (so codex-side workers were never governed by it), and `.codex/AGENTS.md` carried
no pointer to it.

## Correctness frame — the second restructure

The positive-only form still failed its own goal in one way: an eager-tier file is
loaded on every request, and that version still (a) enumerated three charged domains
in its principle sentence and (b) drew its approved vocabulary from the register of a
third party acting on the system rather than from the behaviour of the check.

Gabe's challenge, in substance: does discussing these concepts at all, in a file
preloaded into every session, defeat the point? Assessed as correct. Resolution —
state a defect as a property of **our own code**: which check, which input class,
what it does or fails to do. The domain enumeration was deleted outright; the
positive instruction replaces it and is more actionable. Approved vocabulary moved to
the correctness register (`under-fires`, `unverified admission`, `coverage gap`,
`admitting path`).

Two boundaries held deliberately:

- **Established precise technical names are kept** where they are the exact name for
  a mechanism. Precision-first outranks register, and churning an approved term
  desynchronizes prose from receipts already minted under it.
- **The invariant stayed in the rule.** An earlier proposal to shrink the eager file
  to a pointer with the table in the atlas was withdrawn: `config_editing.md`'s
  currency split puts *current invariants* in `rules/` and only *receipts* here.

Epistemic status: **hypothesis, not a measured finding.** One observed fallback
event, no controlled comparison, no visibility into the classifier. By `workflow.md`
§"Core principle" this does not clear the bar for a "fixed" claim; the only available
observable is fallback frequency across many sessions. The fallback is a model
switch, not a refusal, so the objective is reducing frequency — and past some point
further neutralising language costs precision for unmeasurable benefit, which
precision-first forbids.

## Retired-term enumeration — for mechanical checking only

The rename slice greps for these; they are deliberately kept out of the preloaded
rule. The financial-concealment verb is the one already appearing in §"Concrete
renames" above, so it is not restated here.

`attacker` · `attacker-chosen` · `adversarial` · `hostile` · `exploit` (verb) ·
`bypass` (noun) · `attack vector` · `malicious` · `payload` · `victim`

Boundary: `_FAIL` is **not** in scope. Neither are ordinary uses that are not
describing a defect (e.g. `payload` as an HTTP/JSON body in third-party API prose) —
the principle governs, and precision-first forbids renaming a term whose neutral
substitute would misdescribe the thing.

## Traceability design for the rename slice

No new old → new mapping document. The rename commit's own diff is the mapping
(`git log -S '<identifier>'` resolves any cited name), and that commit sha is
recorded here when the slice lands. Rationale: prior gate records and O_EXCL receipts
citing old identifiers are immutable, and a rename that orphans those citations costs
more than the vocabulary it removes.

Known lagging occurrences at pivot time, recorded so the rename slice need not
rediscover them — 2 tool comments in the LANDS-AB validator (`:599`, `:1104`) plus 7
in the pre-existing test-module corpus. `:1104` is a cross-reference to specific
tests, so the rename must keep that pointer resolvable.

## v4 — remove the remaining self-referential eager rationale

Gabe (chat-side, in substance): how can the rule file itself be improved so it does not
trigger the fallbacks it exists to reduce?

Assessment: after the correctness-frame rewrite, the body vocabulary was clean, but the
**rationale paragraph** still explained the rule as managing how automated classifiers
read our artifacts. That self-description is a **plausible remaining eager residue** —
no ranking is claimed, because the ranking would be unmeasured: one observed fallback
event, no controlled comparison, no classifier visibility. The edit needs no ranking to
justify it: the paragraph was self-referential motivation rather than an operative
instruction, and the currency split already assigns motivation to the atlas.

Resolution: the rule now motivates itself purely as engineering style — durable artifacts
are read out of context, and check-describing wording stays exact without the
surrounding conversation. That claim is true and sufficient on its own merits. The
classifier/fallback motivation lives HERE, in the query-triggered atlas, per the
currency split. One stray descriptor ("charged reading" on the `_FAIL` line) removed in
the same pass. Epistemic status unchanged: hypothesis; whether ANY eager residue matters,
or which mattered most, remains unmeasured.

## Why the "precision first" clause exists

The findings being described were genuine fail-open defects with demonstrated rc=0
escapes and rc=2 controls. A vocabulary rule that encouraged vaguer language would
have degraded exactly the artifacts whose precision the gates depend on — so the rule
fixes wording while explicitly forbidding any softening of claims, severities, or
failure classes. Record corrections and retractions remain mandatory regardless of
vocabulary.
