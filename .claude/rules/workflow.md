**IMPORTANT**: Assume nothing. Hypothesis, Build, Test, Commit & Iterate. First Principles thinking. Do not discount anything until it's built and tested!

# Workflow — hypothesis, test, iterate

**Default working loop for all measurable work** (kernels, harness, training,
config). Historical receipts: `MEMORY/atlas/workflow_part_1.md` +
`MEMORY/atlas/workflow_part_2.md`.

## Core principle — it works or it doesn't

**No vibes.** "Done", "working", "better", "fixed" require a post-change
measurement — number, test, output, or artifact diff. Intermediate states
(half-applied edits, unrun tests) are liabilities. Pure UI/brand judgment:
say so and ask the user.

## The loop

1. **State the hypothesis** (one sentence with predicted metric movement).
2. **Pick the measurement first** — if you can't specify it, sharpen the hypothesis.
3. **Minimal edit** — don't bundle unrelated changes.
4. **Build / run / test.**
5. **Measure** (twice if noisy).
6. **Binary decision** — ship, or revert with a one-line ruled-out log.
7. **Next hypothesis.**

At **route birth** (and death / escalation): `advisor` issues the route license before contract/gates, and it **binds** — Claude executes, escalating disagreement to Gabe rather than overriding (`AI_ROOM_COLLAB.md` §advisor). Direction lead, not an artifact reviewer or gate.

Target: **< 5 minutes per round**; use a lighter proxy if slower.

## Always check two things

Raw-path AND user-facing measurement on the same change. Raw-only win →
wrong path. User-only win → usually noise. Both move → real.

| Work type | Raw path | User-facing path |
|---|---|---|
| CUDA kernel opt | `llama-bench -n 64 -p 0 -r 3` | chat completion w/ fixed prompt |
| Python harness | unit test / pytest | `printf "prompt\n/exit\n" \| zenith` |
| Training filter | schema + dedup count | loss curve on few hundred steps |

Card installs with adapters: **(raw on REAL adapter outputs) + (A/B)**.
Full table + adapter-robustness: `workflow_part_1.md` §"Always check two things".

## Plateau detection

**3 iterations < 2% each → bug, not tuning.** Stop micro-tuning; find the
wrong line (suspicious ratio to reference, unused headroom, compute/bandwidth
math mismatch).

## Route license

Every lineage carries a **route license**: the terminal measurement it heads for
plus its named branches. `advisor` issues, renews, and kills it; issuance at
lineage birth IS the route-check, not a separate mechanism. No license means no
named terminal — the work has already drifted and nobody can say from what. A
route whose terminal measurement is unnamed or unobservable is not licensable.

An expired license means **no further work on that lineage** until a lightweight
route-check renews or kills it. Renewal at expiry is the one unwaivable step.
Expiry fires on the first of: **never issued** (work began with no named
terminal); **defect** (second substantiated gate-2 BLOCK on the lineage, or a task
supersession whose deliverable shas are unchanged); **time** (90 min active work on
one lineage with neither a branch-selecting measurement produced nor a renewal).

A running measurement with a live prereg **is** a terminal ahead — the clock does
not fire while it runs.

There is ONE expiry set with one owner. Expiry sends the lineage back to
`advisor` for renewal or kill — there is no delegated variant that expires on
its own rows, and no waiver.

A renewal is an explicit record naming itself a renewal and the BLOCKs it consumes; the counter re-arms at zero; a disposition citing a license never renews it.
The advisor may **law-freeze** a lineage as a route term: governing law is fixed at that record; later adoptions bind from the next measurement boundary.

**Every disposition names a delta or names its absence.** A consultation MATTERED
iff its disposition names a route changed or killed, a premise corrected, a branch
added or removed, or a prescription refuted — refuting the advisor counts. It was
PERFORMED iff the disposition reads "no delta: plan confirmed intact", which is a
legal outcome, because forcing deltas manufactures them. A delta counts only if the
frozen artifact's change list cites the consult id: a delta that never reaches bytes
is ceremony wearing a delta.

**Enforcement is one check, not three.** Field presence on a frozen record is
greppable. The windowed no-delta rate and the lineage clock have **no emitter**, so
both are declared obligations, self-assessed, and no mechanical count is claimed
until an emitter lands — a rate asserted without one is a number nobody computed.

**Daily terminal.** The day ends with ≥1 branch-selecting measurement — prereg
written BEFORE the run, branches diverging in next-action, ≥1 branch terminal for a
route — OR a one-paragraph post naming the blocking seam and tomorrow's first slice.
Dressing non-measurement work up as the terminal is the failure this clause exists
to catch.
If the day's work freezes any record, **Disposition on EVERY frozen record** applies — `ADVISOR_ROUTE: <id>` (`AI_ROOM_COLLAB.md` §advisor disposition) — not optional on freeze days.

## Empirical pace

Minutes-to-hours on this stack, not mechinterp weeks-to-months. If a step
looks like days, revisit methodology. Detail:
`MEMORY/atlas/workflow_part_2.md` §"Empirical timeline — full reference".

## MAX_TOKENS budget discipline

Before logic/substrate/sandbox diagnosis, verify output budget isn't clipping
(≥ 4K eval default, not ≤ 400). Import `EVAL_CTX_SIZE`, `EVAL_MAX_TOKENS`,
`ITERATION_N`, `FINAL_N` from `calm/llm_computer/eval_defaults.py`.
Iterate at `ITERATION_N`; `FINAL_N` only for commit receipts. "No output /
NoCode" → check `max_tokens` first. Receipt: `workflow_part_1.md` §"MAX_TOKENS
budget discipline".

## GPU bench discipline

Warmup, GPU events, median-of-N, paired A/B, correctness before timing.
Protocol: `workflow_part_1.md` §"GPU bench discipline".

## Daemon state invariants

Daemon scripts: `clear_card_state()` at startup (hooks, `card_slots`,
`reserved_channels` persist). Pattern: `workflow_part_1.md` §"Daemon state
invariants".

## Commit discipline — git log as progress changelog

Commit completed measured work before the next round; one round per commit;
before/after table in perf/correctness messages; checkpoint before risky
swings. Template: `workflow_part_1.md` §"Commit discipline".

**Gate cost is tiered, so the round cadence survives.** LOW-tier rounds (docs,
tests — non-control-plane, reversible) commit under claude's commit gate alone;
HIGH
and every control-plane change keep the co_lead `DIFF_DIGEST` PASS. Tier by
claim effect, per `CLAUDEX_ORCHESTRATION.md` §"Gate-2 convergence + review-risk
tier". `commit_precondition_colead_gate.py` enforces this from the STAGED PATH
SET and fails closed: a mixed or empty set, or any unreadable listing, is HIGH.

Commit shell shape: prefer `git -C <literal-path> commit -F <file>`.
Do not pipe a commit through another command — that defeats the
co_lead-gate shape allowlist and masks the commit's exit status.
Authoritative accepted forms live in the live recognizer
`.claude/hooks/commit_precondition_colead_gate.py` (its denial text);
this line is one preferred form + pointer, not a second allowlist.

## Informative null results

A null that diagnoses the failure mode IS shippable (same before/after
discipline). Pattern: `workflow_part_2.md` §"Informative null results".

## Curriculum nulls: change supervision shape

Train-perfect + held-recombination failure → change supervision shape
(`input/question → planning → reasoning → computing → answer/output`) before
range/runway/LR/model size. Fade scaffolds only after held recombination
transfers.

## Long-running training supervision

**Review routing (ai-room):** thinking parallel; artifact gates sequential
(gate1_audit gate-1 → co_lead gate-2 on frozen handoff; claude frames the
handoff and authors all `+1` records). **Passive-wait-don't-poll**
at gates. Tiered ceremony (HIGH / LEAN-MEASUREMENT / LOW):
`CLAUDEX_ORCHESTRATION.md` §"Gate-2 convergence + review-risk tier".

**Foreground only — no detach:** forbid `setsid`, `nohup`, `disown`,
`run_in_background`, trailing `&`. Run training foreground in a dedicated
shell; log to file; arm `bin/watch-wrap` Monitor with failure signatures
(`Traceback|Error|Killed|OOM|FAILED|assert`) plus progress/success/stop-on.
`-u` for unbuffered stdout. Each notification = plateau checkpoint; kill
after 2-3 flat evals. Full pattern: `workflow_part_2.md` §"Long-running
training supervision".

### HRM-Text-1.58 curriculum extension

Generic <5-min loop ≠ finish every training run. `hrm-158-base` runs in
**gated slices**: auditable full-density finite support (~100-150 rows),
slow-safe, **90/90 bank gate** (acquire ≥90% / retain ≥90%). On misses:
classify + split/protect/redesign — **no LR/runway/model escalation**.
Recipe knobs: `hrm-158.md` §"Recipe band". Full workflow: `hrm-158.md`.

## Sweet-spot search / tool priority

Search downward for max capability/parameter; measure user-facing gate.
Diagnostic escalation ladder (bench → static → print → remove-it → profilers
last). `workflow_part_2.md` §"Sweet-spot search" + §"Tool priority".

## Full-GPU for trainer-loop work; CPU only for non-loop checks

**Dividing line = loop entry.** Forward/backward, probes, q/acc update,
checkpoint-load-and-step → **GPU** (even 1-step smoke). CPU only for
non-loop checks: `py_compile`, import/argparse, fixture schema parse,
dry-run flag gates, hash/preflight/git-state. Checkpoint 1-step smoke is
GPU. Heavy CPU pre-gates → `cpu_guardrail_too_heavy`, hard-timeout, route
to GPU.

**GPU-hot-loop**: hot path GPU-resident + kernelized, not just `device=cuda`.
Launch packets declare per-phase **phase budgets** (forward/backward, update,
emission, flush). Watch-wrap heartbeat ≠ hot-loop progress. Past phase budget
→ liveness failure: stack-sample, kill/release, classify. Materialization
failures → **class audit** (all callers + hot-path `.tolist()`/dict builds)
before relaunch. New observers need scale-smoke or cost model. Receipts
separate DEVICE vs HOT-LOOP residency.

**Gate weights**: (1) **GPU correctness smoke** — hashes, scratch path,
pipefail, re-hash, hard step bound, artifacts, duration. (2) **GPU dynamics
run** — smoke set + watcher + stop conditions. Minutes-to-write tradeoff
only when native GPU path exists + parity-validated AND includes launch
contract. Fleet: 4070 = GPU; 1070 = audit/probe lane.

## Probing-specific methodology gates

Three mechinterp gates — full spec:
`MEMORY/atlas/workflow_part_2.md` §"Probing-specific methodology gates".
Prompt-format (>50% baseline before ablation); task-rank vs PCA-rank
(projection test); superposition suspect → TopK SAE not L1.

## Pitfalls

No bundling; "didn't crash" ≠ pass; drift ≠ improvement; perf needs
correctness smoke (`17×23=391`); trust ≥3 bench reps; re-bench baseline
after env change. Extended: `workflow_part_2.md` §"Pitfalls to avoid".

## Feedback-loop validation pattern

Learn+apply systems need: (1) loop-closes unit test, (2) effectiveness
harness on held-out inputs, (3) E2E integration with mocked upstream.
Shape-gate apply phase; visibility via `scripts/learning_dashboard.py`.
Full spec: `workflow_part_2.md` §"Feedback-loop validation pattern".

## CALM iteration pattern

Hypothesis-test-iterate for CALM backends/modules/quality-gap. Ops: `calm.md`.
Pattern: `workflow_part_2.md` §"CALM iteration pattern".

## Substrate install workflow

Checklist: `MEMORY/atlas/Substrate_arc.md` §"Install Workflow (checklist)".
Arc summary: 6-step Allocate → Convert → Install → Verify → Register → Commit.

## When this workflow doesn't apply

UI/design judgment; exploratory research without a target (qualitative reads
OK); pure discovery reading (metric required once you edit).

**IMPORTANT**: Assume nothing. Hypothesis, Build, Test, Commit & Iterate. First Principles thinking. Do not discount anything until it's built and tested!
