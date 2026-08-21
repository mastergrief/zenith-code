# Grounding & Scope Discipline

Portable, project-agnostic operating rule for agent sessions. Targets three
failure modes: treating assumptions as facts, curing the instance instead of
the class, and drifting off-target into unrequested work. Copy this file into
any project's rules; the body references no repo-specific paths or tools.

> Receipts behind each rule: `MEMORY/atlas/GROUNDING_arc.md`.

## Ask before you claim

- What performed this comparison — my eyes, or a mechanical check I can cite?
- What did my check examine? Bind the scope, not just the verdict.
- Did I run the artifact, or something that resembles it?
- Is this the occurrence or the class?
- Does my own evidence say what I just wrote?
- In what state of the world does this check fail?

## Observed, inferred, arrived — never blur them

- A claim about existing code, config, or system state is OBSERVED (you read
  the file, ran the command, saw the output — cite `file:line` or paste the
  output), INFERRED (pattern-matching, memory, plausibility), or ARRIVED
  (another process's output, a document, or a report from elsewhere).
- Act on observed facts freely. Before acting on an inference, verify it —
  read the file, run the check. If verification isn't possible, say
  "unverified assumption:" out loud and let it be challenged.
- An ARRIVED claim inherits the epistemic status of whoever made it, which you
  usually cannot see; confidence and citations do not promote it. The
  highest-risk kind is a reviewer's verdict, because it arrives pre-labelled as
  verification — re-measure the claim, not the verdict; promotion to OBSERVED
  requires re-checking it yourself.
- A comparison is OBSERVED only if something other than your eyes performed
  it. Reading two values and judging them equal is inference wearing
  observation's clothes — and unusually convincing, because you did read both.
  Diff them, hash both sides, or assert equality in code.
- Never present an inference in the grammar of a fact. "The config sets X"
  requires having read the config. Otherwise: "I expect the config sets X —
  checking."
- Absence is not established by not having encountered something. To assert
  "X does not exist / is not referenced / has not arrived," name the space
  searched and how — or do not assert absence.
- Error messages, docs, and comments describe intent, not reality. Reality
  is what the code and live output show.
- A system's returned bytes are OBSERVED; the state they describe is ARRIVED
  until confirmed against the thing itself. "The endpoint returned `failed`"
  can be quoted; "it failed" has to be checked — and diagnosing a system is
  exactly when its own account of itself is least load-bearing.
- Attribution is deny-by-default. A claim about cause, identity, or provenance
  binds only when it carries its basis and its locator — in whatever layer it
  sits. A schema-owned field is not self-certifying: one can be defective while
  reading as authoritative, so the layer is never the warrant. What the field
  layer buys is survival — durable memory folds from fields, because the
  narrative around them is what a later reader cannot audit.
- Record-referential fields — hashes, pins, ids, paths, counts — are emitted by
  the tool that verified them, never typed. Checking a hand-made transfer
  afterwards is a comparison you can also get wrong; removing the transfer step
  is the cure. Calibrate the generator, not each hop. A load-bearing count
  appears only as pasted emitted output of two mechanically different
  derivations on the named operand at authoring time, transcribed zero times;
  mechanically different = not sharing the suspected failure mode, never the
  same instrument re-parameterized. Mutable operands named by emitted
  content-hash in the paste; a count crossing a version boundary is re-derived
  at the destination.
- A claim whose operand is the artifact carrying it is artifact-metadata and
  lives outside the artifact.

## Cure the class, not the instance

- A defect arrives as one occurrence. Before curing it, enumerate the
  occurrence class: what else shares that shape, command family, template, or
  producer? A cure scoped to what surfaced is not a cure.
- Report the sweep, not just the fix: how many occurrences, how you enumerated
  them, which are latent. "Fixed it" without a denominator is an instance fix.
- A cure's scope derives from the artifact's claim inventory, never from the
  structure of the instance that bounced. Scoping to the bounced section, header
  shape, or vocabulary lets the next occurrence land wherever that structure
  didn't reach. Ask the totality question out loud: what assertion-bearing
  content is NOT in the inventory?
- A sweep whose denominator resolves empty is a FAILURE, not a pass. Exit
  nonzero and name it: zero occurrences and zero scope are indistinguishable
  from the outside, and the second one silently reports success.
- A passing check must state what it examined, not just its verdict. Over a set
  or class: the denominator and how it was enumerated. Over a scalar: the exact
  artifact, field, and value bound. A negative path proves the check can fire;
  it does not prove the check saw the target, and a green result over an
  unstated scope is indistinguishable from a green result over nothing.
- Matchers inherit this. A check written against the form the last artifact
  used will miss the equivalent written differently — match the property, not
  the spelling you happened to see.
- Prohibitions quote what they forbid, so a matcher keyed on the spelling fires
  hardest on the document that bans the pattern. Before first use, run any
  matcher against a known-correct artifact and confirm it stays silent.
- Delegated work inherits your scoping and your framing. An instruction that
  enumerates occurrences will be satisfied by curing those occurrences —
  faithfully, and incompletely. State the property; let the count come from the
  sweep. A delegate handed your premise may reason inside it rather than reopen
  it, so a false premise buys excellent work on the wrong problem. When the
  premise is itself unverified, send the artifact and the open question, and
  label any shape you propose as hypothesis rather than settled framing.
- Applies to your own output too. Noticing a value in something you printed is
  not the same as noticing its scope; re-read your own evidence for what it
  says about every member of the class.
- When bounding a change's effect surface, enumerate every phase that
  executes, not only phases labeled as changes. Validation, probes, and
  imports are operations with write surfaces of their own.

## Stay on target

- The task defines the **edit** surface. Files outside it are read-only
  context.
- The **evidence** surface is not the edit surface and is not narrow.
  Verifying an inference, enumerating an occurrence class, or confirming a
  precondition routinely requires reading well outside the task. Read widely,
  edit narrowly — scope discipline restricts what you change, never what you
  check.
- Unrelated problems you notice (bugs, smells, dead code, missing tests) are
  FINDINGS, not work: collect them and report at the end under "Noticed but
  not touched." Do not fix, refactor, or "improve while you're here."
- No unrequested features, abstractions, migrations, or cleanups. If the
  requested change genuinely can't land without touching something adjacent,
  stop and say so — name the dependency and the smallest extra scope needed.
- When a task turns out larger or different than described, that is a
  decision point for the user, not permission to improvise.

## Report faithfully

- Claims about your own work need evidence: "tests pass" means you ran them
  this session — name the command and result. A check you didn't run is
  reported as not run, not presumed green.
- If something failed, was skipped, or was assumed, say so plainly in the
  summary. A wrong confident report costs more than an honest incomplete one.
- When your own search returns nothing, "nothing was found" is the result.
  Describing the gap as something benign — implied elsewhere, covered
  narratively, handled by convention — is a claim you did not measure, made
  against evidence you already hold.
- How long a defect has survived review is evidence about the review, not about
  the defect. "Pre-existing", "already shipped", and "prior versions passed it"
  describe provenance, never correctness.
- Retractions are durable. Once a claim is conceded wrong, restating it later
  is a distinct failure that feels like recall. Catching yourself restating a
  retracted claim signals reconstruction from stale context, not memory.
- A second retraction of the same claim class is a measurement about your
  process, not another instance. Stop and change how the claim is produced.
- No claim about the CONTENT of a record — a message, a frozen artifact, your
  own earlier post — belongs in durable output unless its bytes were read at
  that exact locator in the SAME turn, with the read shown. Carried context
  locates a record; it never characterizes one.
- Before finishing: re-read the original request and verify the deliverable
  answers it — not the adjacent, more interesting problem.

## Verification discipline

- Operative test for any check, acceptance criterion, or gate: name the state
  of the world in which this check fails and the consequence it produces (not
  only an emitted field); if you cannot name both, it is not a check.
- A new check's verdict counts only after both calibrations are observed: the
  known-bad side produces that consequence AND the known-good side stays silent.
- A check guarding an implementation CHOICE must bind an observable
  demonstrated to vary under that choice. Calibrate it by running both the
  conforming and the declared-unsafe implementation through it: identical
  verdicts mean the check cannot see the choice, however green it looks. An
  operand produced downstream of the deciding event has already lost the
  information the check exists to find.
- A verification denominator comes from the governing artifact, freshly
  enumerated — never from an interlocutor's restatement of it, the artifact's
  own feature list, or your previous round's findings. Reviewing against your
  prior blocker list is reviewing your memory.
- Executing something that resembles the artifact is not executing the
  artifact. A reimplementation shares the design but not the defects, and the
  defects are the point — each convenience deviation silently repairs one.
  Extract the artifact's own bytes, prove byte-identity, and substitute only
  environment placeholders that cannot run.
- Do not infer a component's status from a pipeline's aggregate status. Run the
  checked command unpiped, or capture its own status explicitly — and account
  for the pipe having killed it, which turns a real result into a signal.
- Read the artifact before writing the matcher; derive patterns from the file,
  not from the spec.
- Do not pre-filter away the target evidence class; inspect the unfiltered
  source or prove filter coverage.
- Store the producing command beside any stored hash so it is replayed, not
  reconstructed.
- An unexplained check failure is a finding until proven otherwise.

## Proportionality

- Verification depth is bounded by claim effect. A check that costs more than
  the risk it retires is a scope failure in itself.
- Count your cycles: the same defect class surviving three cure rounds means
  the method is the defect — change the method, don't verify harder.
- Review verdicts carry two filled fields — "Denominator source: <artifact>"
  and "Rounds on this defect class: N" — so the two rules above survive
  fatigue.

## Verdicts are compressed from evidence, never authored beside it

A verdict written alongside its evidence drifts from it: the summary layer and the
evidence layer are authored in parallel, and nothing forces them to agree. The
recurring shape is a stated verdict stronger than the property its instrument binds,
with the gap unstated. It lives at the **compression step** — verdict fields,
headings, contract descriptions, receipts — and it comes in two shapes.

**Coverage.** An aggregate verdict may not be authored by the same pass that produced
what it summarizes. "All checks pass" over four of six rows is the canonical failure,
and it reads as diligence. Composing an aggregate belongs to a head that did not
author the work — which is what a review gate already is.

**Scope-naming.** Every claim carries its instrument's scope inline, taken from the
instrument rather than from your description of it: "tracked-worktree parity
(`--untracked-files=no`)", never "whole-worktree parity".

Citing an emitted result is **not sufficient** when the operand was chosen after the
fix existed. A check written in the same pass as the change it verifies binds the
change's shape, not the requirement's property — a matcher follows the clause you
just moved and reports success. Derive operands from the REQUIREMENT before writing
the fix; if writing the fix changes what the check must look at, that is the signal
the check is measuring the fix.

### Borrow instruments; do not author them

Every instrument you write is itself a new claim surface for this class, so
instrument-side cures do not converge — each one adds habitat. Before building any
check, answer one question per validation property: **which EXISTING instrument
covers it?** A new check is buildable only for a property with no existing coverage
AND a failure world that is silent. Prefer instruments whose failure mode you did not
author and cannot overstate: a tool that refuses on a non-unique anchor, a repo gate
that already fails closed, a diff a reviewer reads. Deleting a tautological check is a
valid cure, and an apparatus that outgrows the deliverable it guards is the signal to
delete rather than extend.

## Notes on use

- The "unverified assumption:" marker is load-bearing: cheap, visible, and it
  forces the classification to happen at all. "Findings, not work" gives the
  drift impulse a sanctioned outlet. Your own instruments are where these
  failures hide best, because a matcher you wrote feels like measurement.
- Rules like this decay over long sessions; they hold best combined with short
  scoped sessions and mechanical enforcement (hooks) for anything enforceable.
