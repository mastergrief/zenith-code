# Claudex orchestration — codex worker lifecycle

> Historical receipts: `.claude/MEMORY/atlas/AI_ROOM_COLLAB_arc.md`.

Task-dispatch lifecycle for claude-orchestrated codex handles; companion to
`AI_ROOM_COLLAB.md` — dispatch, recycle boundaries, hook-enforced gates.

**Default handle**: `codex_co_lead` — always-on co-lead; exempt from child-task
boundary (multi-task audit by design).

## Principle

Dispatch narrowest task with independent evidence. Plans/contracts AND bounded
implementation → `plan-dev`; formal run packets → `test-operator`. Workers
never `@gabe` — bubble to claude; slice-scoped, recycle after shipped slices.

## Team model + named role lanes

**Gabe** = direction owner. **Claude + `codex_co_lead`** = co-leads. **Claude**
= orchestrator, AUQ/dispatch, gatekeeper, synthesizer. **codex_co_lead**
read-only.

**Named Codex role lanes** — normal route for gated mutating repo-file work:

- **`plan-dev`** — planning/contract/packet lane AND **default** bounded
  implementation executor (developer template). Owns plan/packet drafting,
  run-packet contracts, and approved implementation — **NOT** implementation
  review (implementation receipts route to claude gate-1 ONLY; co_lead gate-2
  only after claude freezes the handoff), **NOT** formal run execution.
  Break-glass implementation/run via Claude `+1` with
  `transition_fallback_used=true`. After `+1 implement` may invoke
  `.codex/agents/developer.toml` (`subagent-claimed` until verified). **cwd =
  provenance/dispatch match, not repo permission.** No `.pt` commits.
  **health-proven existing backend/config** — do NOT change backend as the fix.
  Edits + focused developer validation in scope; **material receipts to claude
  gate-1 ONLY** (`REPORT_TO: [claude]` — naming co_lead in `REPORT_TO` does NOT
  wake or route to co_lead); on dual accept proceed to claude commit/push
  gates, then run packets route to `test-operator`. FORBIDDEN: spawn/kill/grant/
  dispatch, training launch, commit/push unless the claude gate authorizes.
  Role home: `~/.ai-room/.codex-roles/plan-dev/`.
- **`test-operator`** — formal training/proof/test-run packet executor. Runs
  specified packet, monitors artifacts, posts the terminal receipt to claude
  gate-1 ONLY; co_lead gate-2 only after claude freezes/hands off. FORBIDDEN:
  source edits, improvisation, commits/pushes.
  Underspecified → STOP. Code fixes → `plan-dev`; packet fixes → `plan-dev`.

**Role vs handle**: `role="<name>"` loads role home; routable target is a
`codex_N` handle — role name is NOT a room handle; developer executor reports
through `plan-dev`. **Not a second dispatcher**: co_lead recommends only.

## Lifecycle

```
claude creates task + provenance → plan-dev plan → claude gate-1 freeze →
co_lead gate-2 plan review → +1 implement → plan-dev implements → claude
gate-1 → co_lead gate-2 implementation review (dual accept) → +1 commit →
commit → +1 push → push → test-operator run packets → complete + recycle
```

Claude load-bearing at gate-1 freeze/verify, commit, push, and launch gates;
co_lead gate-2 reviews frozen handoffs only (independent, not rubber-stamp).

**Converged-contract fast path ("plan-dev = dev" mode).** When the slice
contract is already converged — measured defect cycle, mechanical re-scope,
co_lead-prereg'd branch/acceptance details — the dispatch IS the plan: claude
may fold plan gate + `+1 implement` into the dispatch and plan-dev implements
directly. Novel slices (new mechanism, measurement design, anything minting
science semantics) keep the full plan gate — refutation is cheapest there;
skipping it just moves the bounce to the costlier diff review. A fast-path
dispatch must NAME why the contract is converged and enumerate frozen
scope/acceptance/stop conditions — never a generic plan-gate waiver. Diff
gates (gate-1 freeze → co_lead gate-2) are NEVER skipped in either mode.

**Mint-hold rule**: after a BLOCK, hold the remint until claude's consolidated
dispatch (verdict + all addendums) posts; a late addendum folds into the NEXT
version, never a partial edit.

## Recycle boundaries

**Child-task** (non-co_lead): fresh handle per child task. Hook-enforced.
**Defect-cycle** (write-class): retain only if defect ⊆ files just edited,
same lane, no new external evidence. **Context-pressure** (all handles):
recycle before ~80% context. Mandatory recycle: subsystem boundary, major
defect (≥3 files), before commit/push gates, context over threshold.

Retain across child tasks only with `RETAIN OVERRIDE: <reason ≥10 chars>`.

## Hook enforcement

PreToolUse block-and-explain guards on `ai_room_post`/`_reply` (fail-open on parse errors; rule still applies):

- **`task_dispatch_child_boundary_gate.py`** — blocks new child dispatch to
  in-progress handle without RETAIN OVERRIDE.
- **`task_dispatch_cross_thread_gate.py`** — blocks worker dispatches that route
  material receipts to both co-leads in parallel; requires `REPORT_TO: [claude]`
  + `CROSS_THREAD_REQUIRED: yes` (or `CROSS_THREAD_WAIVER`). co_lead handoff
  dispatches exempt.
- **`commit_precondition_colead_gate.py`** (Bash matcher) — once `git commit` is
  recognized, blocks unless a fresh co_lead validation/diff PASS echoes the
  staged `DIFF_DIGEST`. `git push` is not co_lead-gated (force-push blocked).
  No auto-match of a room-posted PASS → executor flags first; claude authorizes
  `CO_LEAD_GATE_OVERRIDE` bound to target-repo path + 64-hex DIFF_DIGEST +
  co_lead PASS msg id — never unilateral.
- **`worker_gate_wake_pairing_gate.py`** — gate/drive posts need paired
  `task_update(notify=true, in_progress)` or `WAKE_VERIFIED`.
- **`ai_room_heartbeat_watchdog.py`** (cron) — stall detection + missing-heartbeat wake.

## Worker task shape

Non-trivial tasks include: provenance, decision contract, scope, workflow,
stop conditions, `REPORT_TO` + `CROSS_THREAD_REQUIRED`. Dispatch to exact handles.

**Wake semantics**: `task_update` does NOT wake — pair with direct addressed
post. **Completed-task ack-idle**: don't `complete` between gates; reopen
`in_progress` + execution wake for continuations.

## Worker workflow

1. Read task; verify provenance + contract.
2. Ground narrowly (no session-log scans).
3. Post plan + risk; wait for persisted `+1 implement`.
4. Implement/prove; receipt to claude gate-1 ONLY (co_lead gate-2 after freeze).
5. Commit after `+1 commit`; push after `+1 push` or `+1 commit+push`.
6. Report SHA; wait for recycle.

Read-only handles that mutate = safety failure. Plan gate = refinement loop
(§"Refinement loop" in `AI_ROOM_COLLAB.md`).

## Validation and receipts

Match risk + user impact. Receipts: commands, outputs, artifacts, cites, msg
ids, caveats. Cited gate ids must resolve as authored records. Receipt commands
are exact replayable argv (env vars verbatim, no ellipsis) — else receipt defect.

## Gate-2 convergence + review-risk tier

**Exhaustive-first review.** The FIRST gate-2 on an implementation runs a
plan-derived conformance checklist in ONE pass — provenance/authority,
input-contract validation, fixed geometry, phase/step order, state-compat,
fail-closed defaults, route/credit integrity. Every re-review still rechecks
prior blockers AND re-runs the full hazard sweep — **substantiated evidence is
never suppressed and PASS is never forced**. Convergence comes from batching:
surface ALL substantiated blockers in one verdict, not one axis per bounce.
When a plan-derived axis surfaces late, the verdict names why it was missed
(process retrospective), and the checklist gains that axis.

**Review-risk tier — by control-plane blast radius, not file extension.**
HIGH — mints a science verdict, touches a banked/`.pt` artifact, makes an
acquisition/sub-2 claim, **or alters cross-session control-plane behavior**
(gates, authorization hooks, staging/index semantics, review rules) → full
dual-gate + per-round freeze/DIFF_DIGEST. LOW — local, reversible,
non-control-plane surfaces (docs, local tooling, tests) → claude gate-1 + one
co_lead pass expected. **LEAN-MEASUREMENT** — measurement-only CPU
slice (CREATE-only / bounded-correction surface; no `.pt`/banked touch) whose
terminal output is a preregistered feasibility/plumbing/parity/null
classification answering only whether the bounded carrier/path works. Tier by
CLAIM EFFECT, not branch presence: HIGH begins when the output mints or
selects a science/mechanism verdict, an acquisition/stability/bank/readiness/
sub-2 claim, or authorizes a science run. Expect ONE
batched full-depth plan gate-2 pass + ONE implementation pass when clean; a
BLOCK/correction still takes a fresh freeze + full re-review — PASS never
forced. Compresses: artifact-internal prior-review-id backreferences /
multi-receipt self-binding ceremony. Never compresses: live in-room
resolution of the operative gate records (plan/+1/go/freeze/PASS —
author/thread/kind/verdict, fail-closed) and the quality floor —
governing-spec freeze, claim-vs-execution match, source-set pinning (incl.
dirty imported bytes), non-vacuous hostiles + frozen-packet dry-execution.
In ALL tiers a post-review correction changes the diff → fresh frozen
DIFF_DIGEST + matching PASS (hook-enforced); tiers lower expected review
rounds, never the digest/PASS requirement. Bank-gate discipline
(`hrm-158.md`, `ternary_hybrid_stack.md`), science-verdict minting, and
control-plane edits are never LOW or LEAN.

## Commit and push gates

Never commit on plan alone. Push only after `+1 push` or persisted `+1
commit+push`. Ordinary `+1 commit` does NOT authorize push. Stage specific
files (never `git add -A`); preserve unrelated drift.

### Low-blast-radius commit+push collapse

Compress commit→push via explicit persisted **`+1 commit+push`** (claude-authored,
non-ack, threaded). LOW (all required): CPU/docs/tooling/config; scope-clean;
non-force FF; no `.pt`/large binary; no science claim; drift excluded;
`HEAD == remote` post-push. HIGH keeps separate `+1 push`.

## Safety layers

| Layer | Invariant | Failure signal |
|---|---|---|
| Role | Handle stays in lane | Read-only mutates; co_lead implements |
| Authority | Provenance covers scope | Acts on paraphrase |
| Gate | Persisted claude reply | Ack/unthreaded/remembered id as approval |
| Receipt | Claims reproducible | Missing command/cite/caveat |

## Failure modes / Anti-patterns

Stale context → re-ground/recycle. Schema-stale MCP → respawn. Gate confusion →
re-confirm on-thread. Ack-idle → reopen + execution wake. Spawn for trivial work.
Provenance as lane bypass. Output without receipts. Unrelated drift in commits. Unaudited handle reuse. Worker `@gabe`.
