# Architecture Discipline - historical receipts

Receipts that justified `.claude/rules/architecture_discipline.md` and
`.codex/rules/architecture_discipline.md`. Query-triggered, not eager-loaded:
the rule files preserve current invariants; this atlas carries dated receipts,
message IDs, SHAs, and incident narrative.

## 2026-06-02 god-file refactor incident

Gabe asked for a prevention rule after the transient-FP-credit science harness
became a god file:

> "can we create an architectural discipline rule so this never happens again?
> i.e modularity/facade with reimports etc"

Gabe then delegated the architecture choice to Claude + codex co-lead:

> "whatever you both believe the best. most maintainable architecture would
> be."

The triggering file was the live creditdir harness
`transient_fp_credit_science_train.py`, discovered in task
`1780393262176-f97ed879` as a task-local/outside-repo script with sha256
`ea6d44ba34ec5ab1f52198558078f01e9875fe4560fa04f17dee1bab3cd2ca8d` and
10,330 lines. It mixed authoritative train state, prior-log evidence,
current-run telemetry, validation, launcher logic, and artifact contracts. It
had taken repeated semantic patches during the C arc and then required a
dedicated behavior-preserving refactor rather than another science patch.

Concrete room anchors:
- Architecture-rule task: `1780419617856-198e7fca`.
- Refactor task: `1780393262176-f97ed879`.
- Phase-1 discovery receipt: `1780418883961-b5bd30ce`.
- Phase-2A implement gate: `1780419062704-da1a4819`.
- Architecture-rule plan: `1780419834222-ef175453`.
- Architecture-rule implement gate: `1780419902401-721cae39`.

Target-confirm from discovery:
- Live creditdir harness sha256:
  `ea6d44ba34ec5ab1f52198558078f01e9875fe4560fa04f17dee1bab3cd2ca8d`.
- Active repo trainer imported by the harness:
  `scripts/train_hrm_text_158.py`, sha256
  `ae6cd766a19f461b50b3090a6d62593484d0533a017589ecfc3ec1a26db101cb`.
- Read-only bridge file:
  `scripts/hrm_text_158_credit_bridge.py`, sha256
  `3846a0754083daa92e195a087bc53ec90ea2868d8685e1af714d9adf86b235ac`.

The rule was codified during the C-negative -> Phase-2A refactor pause, when
the science lane was clean enough to extract the lesson without touching the
active creditdir harness in this repo-rule slice.

## Distilled invariant

The prevention lesson is not only "stop at file-size thresholds." The durable
end-state is repo-tracked facades, reducers, and contracts owning reusable
logic; task-local harnesses remain thin orchestrators; one named import facade
owns dynamic repo path/hash/contract checks; and the q/acc mutation loop stays
isolated and equivalence-protected until characterization is strong enough to
extract it safely.

## Deferred enforcement idea

This slice codifies discipline but does not mechanically enforce it. A future
lint or PreToolUse hook could flag file-size stop conditions and repeated
semantic patches before the next mutating gate. That enforcement is deferred and
was not part of task `1780419617856-198e7fca`.
