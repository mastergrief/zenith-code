# Safe Terminology — neutral vocabulary in durable artifacts

> Retired-term enumeration (mechanical-check seed) + historical receipts:
> `MEMORY/atlas/safe_terminology_arc.md`.

Hardening work produces **long-lived artifacts** — committed tests, tool comments,
frozen receipts, commit messages, room posts — that are read later, out of their
original context. Wording that describes our own checks travels well: it stays exact
without the surrounding conversation and names the defect where the fix goes.
Rationale receipts: the atlas.

**Precision first.** Neutral wording must be *equally exact*. Never soften a finding,
downgrade a severity, or blur a failure class to sound calmer. If a neutral phrasing
would misdescribe the defect, name the defect plainly and keep the technical term.
This rule changes vocabulary, never claims.

## The principle — describe the check, not a third party

State a defect as a property of **our own code**: which check, which input class, and
what it does or fails to do. Do not narrate it as an action performed by an outside
party.

This reframing is the whole rule, and it is not a euphemism — it is better
engineering language. *"The guard under-fires for short-form head claims"* carries
exactly the information a third-party narrative carries, is shorter, and names the
defect where the fix goes. Where a mechanism has an established precise technical
name, keep it (precision-first) and prefer the correctness frame around it.

## Approved vocabulary — concept → term

| Concept | Write |
|---|---|
| a value reaches a trusted set without verification | **unverified admission**, **unbound reference admitted** |
| a check does not fire on inputs it covers | **under-fires**, **unchecked path**, **coverage gap** |
| the code path that admits it | **admitting path** |
| origin of an unvalidated input | **caller-supplied**, **packet-author-supplied**, **unvalidated input**, **arbitrary key/value** |
| a test asserting a defect fails closed | **negative-case test**, **robustness test** |
| a set of such tests | **negative-case battery**, **robustness battery** |
| making a condition hold in a test | **satisfy the condition**, **exercise the path**, **reach** |
| the value or field involved | **supplied value**, **target field**, **non-conforming** |

Positive-only by design: this table names what to write, not what to avoid, so the
file carries none of the retired vocabulary while remaining sufficient to comply.
The enumeration a mechanical check needs lives in the atlas — query-triggered, not
preloaded into every session.

## Scope — two different scopes, do not conflate them

**Authoring scope** (what you write from now on): all prose, room posts, receipts,
plans, commit messages, code comments, and **new** identifiers.

**Mechanical-correction scope** (what gets rewritten): the **mutable repo corpus
only** — tracked source, tests, tooling, comments, and docs. Immutable records
(already-posted room messages, O_EXCL-minted receipts, commit messages in history)
are **never rewritten**; they are cited and neutrally restated per §"Referring to
prior history".

## The governed set

`_FAIL` is **kept**: a pure outcome marker, and the most-cited identifier suffix
across frozen gate records.

Governed: terms the principle catches in **defect context**. Not governed: `_FAIL`,
and ordinary uses that are not describing a defect — precision-first forbids
substituting a term whose replacement would misdescribe the thing.

The governed set is bounded by **defect context and the principle**, not
"everything except `_FAIL`". The atlas list is the current **mechanical-check seed,
not an exhaustive semantic boundary** — a term the list omits is still governed if
the principle catches it, and a checker must not treat that list as complete.

## Correcting the existing corpus — a dedicated slice, never a live one

**Never fold a corpus rename into an unrelated live slice.** It gets its own:

1. **rule edit first**, so it governs the rename;
2. mechanical rename — **rename-only, behavior-identical**: no assertion,
   class-binding, or cure-semantics edits in the same pass;
3. **one preregistered suite run** — that run is the slice's proof and is expected,
   not an "extra" run;
4. the receipt states rename-only explicitly;
5. traceability preserved per the next section.

## Traceability — the rename commit IS the mapping

Prior gate records, frozen receipts, and room posts cite identifiers by their old
names, and those records are immutable. **Do not author a new document enumerating
them.** `git log -S '<identifier>'` plus the rename commit's own diff resolve
old → new line-for-line; record that commit sha in the atlas. A rename that orphans
the citation chain does more damage than the vocabulary it removes.

## Cost discipline

Vocabulary never outranks **claim-vs-execution match**, and never justifies a
validation run **beyond** the rename slice's own preregistered one. Inside a live
slice: wording fixes to text you are already writing ride a run already planned —
apply before it, not after. If correcting wording would desynchronize a receipt from
what was actually measured, **keep the validated wording and disclose it in one
line.**

Renames change tool/test bytes → any manifest or source-set pin regenerates as usual.

## Referring to prior history

Cite the msg id / artifact and **restate the finding neutrally** rather than quoting
retired wording. Retractions and record corrections stay mandatory
(`AI_ROOM_COLLAB.md` §"Receipt discipline") — vocabulary is never a reason to leave
a false claim standing.

## Related rules

- `config_editing.md` — eager-tier caps + currency split this file follows
- `AI_ROOM_COLLAB.md` — receipt discipline, record corrections
- `CLAUDEX_ORCHESTRATION.md` — receipt/identifier expectations at review gates
