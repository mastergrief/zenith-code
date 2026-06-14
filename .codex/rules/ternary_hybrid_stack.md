# Ternary-Hybrid Training Stack — FP-free / sub-2-bit-persistent research lane

> Canonical: `.claude/rules/ternary_hybrid_stack.md`. This is codex's lane
> copy — plan-dev owns these trainer internals, so the rule lives in
> `.codex/rules/` too; keep both in sync. Receipts live on the ai-room board +
> commit log + `MEMORY/atlas/ternary_hybrid_stack_arc.md`, NOT this rule.

The research lane **toward** FP-free / sub-2-bit *persistent* ternary
training: can the ternary genuinely keep training (not freeze) with no FP
trainable masters/moments? The **current win is FP-master-free for the
eligible bulk** (no FP master weights, no Adam moments) — NOT yet fully
FP-free: frozen FP32 scales and FP `lm_head`/`embd`/norms remain. Distinct
from `hrm-158` — that is the **curriculum** lane (grows `hrm-158-base` via
gated finite-support slices). Same model substrate + byte tokenizer;
different question.

## What it is

- Native HRM-Text-1.58, ternary bulk linears (BitLinear on q/k/v/o/gate/up/
  down; `lm_head`/`embd`/norms stay FP). **Effective forward weights are
  ternary**; the research target is the *persistent training state*, not the
  forward path.
- "Ternary-hybrid" = ternary effective weights + an **integer-dominated q/acc
  state plus frozen FP scale metadata** (no FP masters, no Adam moments for the
  eligible bulk; eligible optimizer state entries = 0). The Adam moment is
  replaced, for the eligible bulk, by an integer **vote accumulator**.

## The 3-ledger (the honest accounting — load-bearing)

Always report FP-free claims as three separate ledgers; never collapse them.

| Ledger | bits/eligible weight | carrier |
|---|---|---|
| Effective **forward** (logical ternary) | **1.585** (`log2(3)`) | `q_int8.float32 × frozen_scale` |
| Physical **persistent** train-state | **~24** | int8 q (8) + **int16 vote-acc (16)** + FP32 scale (~0) |
| Eval / **export** | n/a | non-authoritative probe export; regeneration recipe only |

- **The win is real and specific**: no FP master weights, no Adam moments for
  the eligible bulk (eligible optimizer state entries = 0). State it that
  precisely — "FP-master-free for eligible bulk", not bare "FP-free".
- **It is NOT sub-2-bit persistent.** The **int16 vote accumulator** (the
  moment replacement) dominates persistent bits and is the next reduction
  target (int16 → narrower / sparser vote rep). Never upgrade "native ternary
  effective-weight training" into "fully FP-free persistent state."

## Training dynamics + the stability problem

- Attribution verdict: **ternary DIRECTION flips are the erosion cause**;
  freeze-scales → ternary breaches, freeze-ternary → scale-only preserves.
  **Scale was not the repair lever for this failure** (scale support may still
  belong to the forward/hybrid ledger). Stability work targets direction-flip
  dynamics; magnitude/groupwise-scale and rotation were ruled out as the
  *repair for this erosion* / wrong granularity — NOT as separately scoped lanes.
- Candidate flips are generated when vote accumulators cross threshold; the
  global cap ranks priority rows by `abs_new_acc`. Flip demand can vastly
  exceed the applied-rate cap → a saturated deferred backlog.

## Stability mechanisms (the levers under study)

- **functional-window-veto**: veto a flip window by its *measured* effect on
  protected rows (CE-worsening / correct→incorrect), cheap batched surrogate.
- **global-applied-rate-cap**: bound applied flips/step; anneal schedule.
- **preservation-trust-region**: trust-mass soft/hard floors + floor-trajectory.
- **replay + parent-consistency (pc)**: retention of acquired priors.

## Verdict taxonomy

- **`functional_veto_stable_ternary`** (THE WIN): floor held + flat post-
  coverage trajectory + accepted ≥ 10× steps + sustained post-coverage
  accepted + pressure dissipating + attribution proves ternary still trains.
- **`functional_veto_freeze`**: stabilises by vetoing (near-)everything into a
  frozen ternary — NOT a win (rate-limited class).
- **`functional_veto_insufficient`**: floor breaches OR pressure not dissipated
  despite the gate. Sub-reasons: `surrogate_inadequate` (proxy too sparse →
  richer probe set, not abandon mechanism), `incomplete_or_pressure_hidden_in_backlog`.

## Research invariants

- **Single-variable**: one mechanism per run; candidate-gen / parity path
  unchanged unless that IS the variable. No new FP trainable state, no
  q-teacher, no *unregistered* scale/residual/rotation rescue inside the run
  (those may be separately scoped lanes — just not silent additions).
- **Classify before changing mechanism**: name the failure class (and the
  cheapest read-only measurement that proves it) before building. "Pick the
  measurement first" applies to mechanism choice, not just runs.
- **Measurement-shape ≠ stability-verdict**: a shape-only margin/distribution
  run makes NO stability or backlog-dissipation claim. Keep the two schemas
  separate.
- **Resume framing**: the deferred-backlog object is NOT serialized — it
  restarts empty on resume. Deep pressure is carried by the persisted q/acc
  state plus current-step votes/credit (ranked via `abs_new_acc`), so a resumed
  window samples **q/acc pressure**, never backlog age/drainage. No
  pre-resume-backlog claims from a resumed run; ruling out a resume artifact
  needs a from-clean-parent contiguous run.
- **Banked sha read-only**: re-hash before + after; no repo/banked `.pt`
  mutation. Runtime/research `.pt` train-state artifacts stay in the
  credit/science tree and are NOT committed.
- **Compact instrumentation only**: histograms / quantiles / survival counts /
  small hashes — never raw per-proposal index/score/threshold arrays. Margins
  use the **per-row effective threshold** and the **exact live cap-priority
  rows**; static threshold is metadata only.

## Fastest-science loop

Optimize **information per GPU-minute**, not training volume. The control law:

- **Pre-register the branch classifier before launch** — every run decides
  between NAMED branches (which next mechanism it selects), not just
  better/worse. No verdict the prereg didn't define.
- **CPU for schema/parity/safety only; GPU for science** (`workflow.md`
  §"Full-GPU for trainer-loop work").
- **N=20 only as a preterminal screen** (obvious null / bug / liveness);
  **N=50 or the prereg equivalent for a branch verdict.**
- **One variable per run** unless prereg'd as a factorial (§"Research
  invariants").
- **Tiny route patches, not architecture rewrites** — earn the rewrite with a
  measurement first.
- **Role routing**: command churn / thinking / context → `plan-dev`; exact-packet
  terminal receipts → `test-operator` (`CLAUDEX_ORCHESTRATION.md`).
- **Commit every useful null** — a clean negative shrinks the search space
  (`workflow.md` §"Informative null results").

Exemplar (a pattern, NOT a privileged mechanism): the lane3 shadow-prefix
curve picks the next mechanism from one run — current low-K beats
all/random/inverted → rate-cap/trust-region lever; random matches/beats
current → ranking/update-law problem; inverted wins → sign/direction problem;
no arm improves → representation-not-viable / insufficient-separation.

**Sub-2-first launch gate**: no main mechanism-science launch until the
executable `full_sub2_runtime_ready_for_science` checker passes — OR a named
`pre_full_stack_diagnostic` exception is justified BEFORE launch (diagnostic
reason + why cheaper than completing the full stack first). Readiness is
fail-closed over a stable surface enum; classes are `sub2` /
`explicit_exception` / `transient_fp_debt` / `pre_full_stack_diagnostic` /
`missing`. `transient_fp_debt` (dense transient credit / FP captures) and
`pre_full_stack_diagnostic` block main science and never count as sub-2;
main-ready requires missing=0, diagnostic=0, transient_fp_debt=0, all non-sub2
rows justified `explicit_exception`s. The executable checker + its receipt are
authoritative — this rule is the invariant pointer, not a second semantics body.

## Validation

- Producer/consumer watcher for live runs; CPU smoke validates **schema only,
  not science**. Assert: addendum schema present, zero raw-array keys, split
  counts sum, required sections present, banked head unchanged.
- A run that diagnoses a failure mode is a **shippable null** — same
  before/after discipline, receipt on the board.

## Relationship to the other lanes

- `hrm-158` — curriculum lane (90/90 bank gate, full-density slices). Grows
  `hrm-158-base` via BitLinear / native ternary **effective**-weight training
  with its **normal** checkpoint/training state — UNLESS this integer-state
  research trainer (q + int16 vote-acc, no FP masters) is explicitly selected.
  Don't conflate the two lanes.
- `training` rules — model/tokenizer specifics + native-ternary-train flags.
- post-hoc quantization (tq4/tq3) is NOT this lane (training-time ternary, not
  export quantization).
- This lane feeds the curriculum lane only once FP-free persistent training is
  stable; specialists/MoE branch from robust base checkpoints, not weak ones.
