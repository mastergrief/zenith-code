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

## 2026-06-14 Sequential artifact review gates (Gabe directive)

**Motivating incident:** W6 S3c GPU launch packet provenance churn across
v3 → v4 → v4.1 → v5 (`w6_narrow_carrier_gpu_dynamics_s3c_launch_packet_*`).
Parallel dual-review while plan-dev edited in-place let claude, co_lead, and
plan-dev react to a moving target — placeholder SHAs, mutual content-sha
binding cycles, and contradictory "frozen" room claims. v5 immutable filenames
(main `5d8c619cdbee7c723a2dd609a7bc1a0f9c71609821d2a7f50db8aad9fcd0c828`,
replay `0ef2800773a666f7927bbeccc74d5c33a3da3d9d2d677acc8804009f68695dc1`)
cleared both legs under the prior parallel model before this rule change.

**Gabe directive (verbatim intent):** parallel review causes issues — claude
is gate-1, co_lead is gate-2. **New semantics:** keep dual visibility;
sequence artifact gates. `REPORT_TO` = audit/provenance visibility (not
parallel review). `REVIEW_ORDER` = gate sequencing.

**Authority chain:** claude status `1781474774491-49d0cace`; co-design
claude `1781474972562` + co_lead `1781475023717`; plan-dev encode dispatch
`1781475088207-ba0e40a5`.

**Exception preserved:** design/refinement threads may still address BOTH
co-leads before artifact freeze (thinking parallel); artifact review +
material gates go sequential.

**Rules landed:** `.claude/rules/AI_ROOM_COLLAB.md` (team model, plan-dev
lane, cross-thread table, review gate glossary, Fast Training Launch
Contract), `.claude/rules/CLAUDEX_ORCHESTRATION.md` (lifecycle + worker
receipt discipline), `.claude/CLAUDE.md` (terse sequential-gate line).

## Commit ledger for AI-room charter evolution

| Commit | Content |
|---|---|
| `8b1ed8c` | Original AI Room Collaboration charter (claude + codex sides) |
| `d3077d2` | Install ai-room collaboration in claw-code |
| `e67640f` | .codex/ parity with .claude/ rules + atlas |
| `45fbddf` | Charter strengthening — 6 rules A-F |
| `6a08b4e459` | Codex-rs sister-repo mirror of `45fbddf` (on `main` branch) |

## 2026-05-25 Fast Training Launch Contract

Gabe directed (via `codex_co_lead` chat) after asking codex directly how
to speed up training runs ("dont forward to claude"): "ok implement those
changes" + "and how do we retain this contract on new session starts etc?
do docs need updating?". Ingress = codex, so codex_co_lead owned the
packet. Protocol relay: msg `1779738566231-79b7cef3`; docs-retention
relay: msg `1779738678606-3c9d9958`.

Compresses the GPU-launch gate sequence to cut micro-ack overhead: one
launch packet (parent sha/config proof + dry-run-validated command +
watcher bundle + stop/bank criteria + artifact/log paths) → one co-lead
`+1 launch/watch-to-terminal-condition` → claude runs/watches directly →
one terminal receipt; interrupt only for bank pass / hard failure /
criteria mismatch / resource-liveness failure / material parent-recipe
deviation. Does NOT skip safety (the packet still requires full proof).

Landed in both charters (`AI_ROOM_COLLAB.md` §"Fast Training Launch
Contract") + startup pointers (`.claude/CLAUDE.md` key-rules,
`.codex/AGENTS.md` AI-Room section). Origin: the micro-ack overhead
observed during the L0c2-K1 identity arc (a cosmetic flag-spelling
oscillation + repeated holding-acks). First applied on the
L0c2-K1-identity-2digit STEP-2 launch — packet `1779738822112-348c6b76`,
co-lead +1 `1779738910204-a4907f77`.

## 2026-06-02 Ack-idle wake-pairing hook

Second occurrence of the **ack-idle worker hang** in one session
motivated a deterministic PreToolUse guard. Failure mode: a worker
finishes its turn "holding for Claude's gate" (no self-driving work);
the `+1` / drive arrives as a plain `kind=msg`; a plain msg does NOT
re-wake a worker whose turn already ended (authority != wake — same as
`task_update` being durable state, not a wake event); the worker's
`resume_check` returns "idle ok" and it sits idle for minutes/hours.

Two incidents: (1) the first "nothing is happening" ack-idle earlier in
the session; (2) the D2a re-drive where claude's `+1 implement`
(msg `1780431337021`) landed 12 seconds after codex's "holding for
Claude implement gate" (msg `1780431325655`) — a near-miss just past
the turn boundary. Recovered both times via `task_update notify=true` +
direct execution wake (the §"Completed-task ack-idle" invariant).

Root cause: **authority and wake are decoupled** — a `+1` is a durable
authority record but not a wake event. Gabe (verbatim, via claude
chat): "ok create a hook for it so we dont accidently hang for
minutes/hours not being productive" + "make hook yourself and have co
lead review" (direct-author named exception after the training-dev
spawn failed twice on a stale `lease_in_wrong_channel` — codex_1 then
codex_3 lease parked in the `ai-room` channel).

Hook: `.claude/hooks/worker_gate_wake_pairing_gate.py` (3rd PreToolUse
guard on the `ai_room_post` matcher). **STATEFUL** design — upgraded
from claude's initial honor-system marker after verifying `notify` IS
persisted in the task_update record (log keys
from/to/owner/status/notify/reply_to/ts). Blocks a gate/drive to a
single parked non-co_lead worker unless a recent claude-issued,
target-bound, same-task, `notify=true`/`in_progress` wake-pairing
`task_update` exists in the channel log (recency window 1800s), or a
`WAKE_VERIFIED: <reason ≥10 chars>` bypass is present. co_lead folds
adopted: same-target + same-task binding (defeats "an unrelated notify
allows a bare gate on a different task"), `from=claude` requirement,
recency window, line-anchored + blockquote-stripped + non-trivial
`WAKE_VERIFIED`. Validation: 17/17 fixture cases
(`.claude/hooks/test_worker_gate_wake_pairing_gate.py`); py_compile OK;
preload ~142k < 150k gate. Task `1780432224760-2b7dfecc`; route-change
relay `1780432542504`; hook-review request `1780433198650`.

## 2026-06-03 Heartbeat watchdog (clock-driven wake) — the 8h-wedge fix

During the overnight ternary-hybrid arc, a `training-dev` worker **wedged
silently for ~8h** mid-implementation: it ACKed a gate, then emitted no
further event. Because channel pushes + PreToolUse hooks are event-driven,
nothing re-invoked claude OR co_lead to notice — the stall sat until co_lead's
manual `peer_status` progress audit (~8h later). Filesystem check confirmed
no work was lost (trainer sha unchanged, no hung process, gpu clean) — the
wedge cost only wall-time. The wake-pairing hook guarantees claude *pairs* the
wake when it gates; it cannot see a worker that wedges *after* being woken.

Gabe (verbatim, via codex chat): "how do we make this deterministic so it
doesnt happen again? i.e a real heartbeat" → "the heartbeat should be a wake
mechanism right?" → "can heartbeat be driven by a hook?". Converged
architecture (claude + co_lead): **hook ARMS/ENFORCES the SLA metadata on
events; an external CLOCK (cron) DETECTS expiry + WAKES** — an event hook
can't fire on silence. The `ai-room` CLI `post` supports
`--requires-response-from` + `--response-deadline-secs`, so a cron script can
post a wake-bearing record that re-invokes an idle orchestrator.

`.claude/hooks/ai_room_heartbeat_watchdog.py` (v1, cron `*/7`): reads the room,
finds the latest worker gated heartbeat past `next_heartbeat_due` (+600s
grace; missing metadata ⇒ +1800s due-soon, never invisible), proves liveness
read-only, posts wake-bearing `WATCHDOG_STALL` (`requires_response_from=claude`)
on no-movement, `WATCHDOG_HEARTBEAT_EXTEND` on movement. Non-destructive
(claude decides recycle/re-drive). co_lead review caught TWO silent-suppression
blockers, both conceded + fixed: (1) **sticky movement** — compared mtime to
the original hb ts, so a moved-once-then-wedged worker extended forever; fixed
to FRESH movement since the last watchdog check + EXTEND→escalate-to-recycle;
(2) **uncorrelated process/GPU** — bare process-match/GPU≥2000 MiB counted as
moved regardless of phase/run_dir; fixed to phase-aware (code phases = file
freshness only; gpu phases also accept a run-dir-CORRELATED process) — GPU MiB
is reported, never a movement signal. Per-worker selection (worker B's terminal
can't clean worker A). Validation: 11/11 fixtures (incl. the 3 hardening cases
that fail on pre-hardening code); py_compile OK; live dry-run vs real room =
CLEAN (no false-stall during healthy heartbeating). Caveat (co_lead): a
correlated live process is liveness, not progress — GPU launch packets still
need the watch-wrap producer stale-progress watcher as the hung-trainer
detector. v2 (an event hook that ENFORCES the SLA metadata) deferred. The
watchdog cron receipt is a Stage-B launch precondition (no unattended GPU run
without it or a named `MANUAL WATCH EXCEPTION`).

## Retired spawnable codex role names

Carved out of `rules/AI_ROOM_COLLAB.md` §"R&D team model" to hold the eager
tier under its cap. The **current invariant** stays in the rule — the standing
roster is `codex_co_lead`, `plan-dev`, `test-operator` and nothing else. This
is the enumeration a mechanical check would need, kept query-triggered rather
than preloaded.

Not standing roles in this repo: `training-dev`, `trainer-implement`,
`trainer-dev`, `codex-dev`, `codex-explore`, `codex-terminal`, `tmux-tester`,
`curriculum-dev`, and similar legacy lanes.

Two things are frequently mistaken for a fourth room role and are not:
`.codex/agents/developer.toml` is plan-dev's bounded executor template, and
Claude-side `Explore`-agent fan-out is orchestration.

## Advisor hardening — instrument pre-check + defect-class audit (2026-08-07)

Gabe, verbatim: "how could we harden you effectively using advisor?" then
"ok implement all direct with co_lead review". Claude-direct under that named
exception; co_lead review. Relay `1786112157860-e10e70e2`, task
`1786112135681-ee33b0ef`.

**What the measurement showed.** Claude's defect ledger for the session was
entirely **apparatus**, not design: a member loop that iterated a dict as a
list and reported green over zero members; `RUNNER_PASS` counted 2 where both
hits were the substring inside `S5_NO_RUNNER_PASS_YET`; an absence asserted
against the watch-wrap monitor stream instead of the attested
`external_runner_log_path`; a packet gate that bound member bytes but never
enumerated the claims the run would emit; an unsorted `rg` hash frozen as
replay authority; then a hash frozen with no producing argv. Advisor's charter
barred it from every one of those — pre-artifact only, never fed plans, diffs,
or receipts. More design consultation would have caught none of them.

**Routing measurement.** 14 frozen records minted in the session: 4 carried
`ADVISOR_WAIVER`, **10 carried no advisor field at all**, against a charter
requiring every consultation *or* waiver be recorded. Claude-initiated
solicitations: zero; the single contact was advisor-initiated, relaying Gabe.
So the dominant failure was unrecorded disposition, not bad advice — and the
`ADVISOR_WAIVER: artifact review (plan gate)` label was correct by the letter
while routing around trigger 4 on rounds whose defect was an unnamed
measurement.

**Three cures.** (1) Trigger 4 keys on the instrument, not the artifact stage.
(2) Instrument pre-check — a check may be shown before it runs, for two answers
only, false-fire state and false-silence state. (3) Defect-class audit —
advisor reads the room journal and reports the recurring class, its trend, and
whether the last cure held; the self-measurement Claude cannot make, since a
class is visible only across occurrences and the agent producing them
reconstructs its own history and experiences that as memory. Plus: a frozen
record with no advisor field is a gate defect.

**Mode router — added at gate-2 round 3.** The first two frozen versions kept
describing advisor as pre-artifact and consulted before the decision contract,
then added two modes that are neither. co_lead blocked twice on the resulting
collision. The third pass swept the whole surface instead of the cited lines
and found **six** occurrences of the same universal stage claim — the rule
bullet, the agent body, the H1, and the frontmatter description — each earlier
round having cured a subset, which is the instance fix the occurrence-class
rule names. Cure: mode is selected by **what the solicitation carries** (a
check shown pre-run → pre-check; a journal request → audit; neither → design),
and only design triggers 1-3 are stage-bounded. Trigger 4 and both non-design
modes are stage-independent — the fact the collision had been hiding.

**Round 4 — the class stopped being cured in prose.** The round-3 router was
still an instance fix: it defined precedence for `check + design` and left the
other mixed payloads unrouted, and it protected pre-check and audit from the
category waiver while leaving stage-independent trigger 4 exposed to it. Seven
occurrences of one class over four rounds is a measurement about the method, so
the cure became enumeration: a **7-row truth table** over every non-empty
subset of `{check, journal request, design question}` under total precedence
`check > journal > design`, answer-the-highest / return-the-rest / never blend;
plus a **waiver matrix** naming the three things no stage or category waiver
reaches. The table is machine-checked by
`scratchpad/verify_mode_truth_table.py`, which parses it out of the live agent
file and asserts totality, precedence consistency, and exact return sets —
calibrated on one positive and three negative arms. Its first run failed on the
live file, and the defect was in the checker: `set('') <= {':','-'}` is True, so
a separator guard testing only the first cell silently dropped every row whose
check column was blank. The same matcher-form-over-property class, now inside
the instrument built to close it. A committed test was out of scope this slice;
the verifier stays a scratchpad artifact and the truth table is the normative
surface.

**Rejected:** advisor as reviewer, gate, or approver. Three-way review dilutes
responsibility, and advisor output is ARRIVED — Claude re-derives it anyway, so
a gate role buys a round and no evidence.

**Line-cap note.** `rules/AI_ROOM_COLLAB.md` sat at 199/200 hard cap. Substance
went to `.claude/agents/fable-advisor.md` (unbounded); the rule kept invariant
pointers and landed at exactly 200. No new tool grant was needed — advisor
already held `ai_room_read` / `_tail` / `_search`.

**Deferred, not implemented:** advisor's own relayed proposals — a commit gate
refusing staged `.claude/hooks/*` without a both-arm calibration receipt, and a
dead-letter journal so a fail-closed guard's silence is observable rather than
inferred. Disposition `1786110087799-d92d0a65`; Claude reversed the proposed
ranking (dead-letter first) and amended the calibration predicate, since a
paired `test_<hook>.py` measures filename adjacency, not calibration. Measured
then: 14 non-test hooks, 8 paired, 6 unpaired including the control-plane
`task_dispatch_child_boundary_gate.py`.
