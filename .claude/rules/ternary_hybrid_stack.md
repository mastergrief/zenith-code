# Ternary-Hybrid Training Stack — FP-free / sub-2-bit-persistent research lane

The research lane **toward** FP-free / sub-2-bit *persistent* ternary
training: can the ternary genuinely keep training (not freeze) with no FP
trainable masters/moments? The **current win is FP-master-free for the
eligible bulk** (no FP master weights, no Adam moments) — NOT yet fully
FP-free: frozen FP32 scales and FP `lm_head`/`embd`/norms remain. Distinct
from `hrm-158.md` — that is the **curriculum** lane (grows `hrm-158-base`
via gated finite-support slices). Same model substrate + byte tokenizer;
different question.

> Historical receipts (per-run ledgers, attribution measurements, mechanism
> nulls, chain-head shas, msg IDs) live on the **ai-room board** + commit log
> + `MEMORY/atlas/ternary_hybrid_stack_arc.md`, NOT this eager-tier rule.

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
| Physical **persistent** train-state **(dense LIVE)** | **~24** | int8 q (8) + **int16 exact_accumulator_shadow (16)** + FP32 scale (~0) |
| Eval / **export** | n/a | non-authoritative probe export; regeneration recipe only |

- **The win is real and specific**: no FP master weights, no Adam moments for
  the eligible bulk (eligible optimizer state entries = 0). State it that
  precisely — "FP-master-free for eligible bulk", not bare "FP-free".
- **Two-tier pass/fail bar:** working **≤2.5 bpw scale-inclusive**; north star
  **<2.0**; **"sub-2"**=actually <2.0; ledgers exact; acc headroom vs q 1.6 ≈
  **0.4** / **0.9** (north-star / working-bar).
- **Working-bar non-override:** ≤2.5 does **not** satisfy/override
  `full_sub2_runtime_ready_for_science` (incl. `transient_fp_debt` /
  diagnostic blocks) until a separately gated checker re-tier lands.
- **NOT (yet) sub-2-bit persistent TOTAL; the LIVE row ≠ the saved-byte q
  ledger** — keep them separate. **Saved-byte checkpoint q** is banked below
  2.0: base-3 `packed_base3_5ternary_uint8` = **8/5 = 1.6 bpw** physical
  (q+scale-inclusive north-star **< 2.0**; the 2-bit pack sits at the 2.0
  boundary). Dense LIVE vote-acc is **int16 exact_accumulator_shadow**
  (hard-required) → dense-LIVE ≈ **24 bpw** (8+16+~0), not ~16. **W8 (±127)**
  = default-off range faithfulness (clip→int8→int16; in-vivo range/parity
  only) — **not** a realized container shrink (no W8 pack; ckpt sparse /
  W5-W6). **16-bit acc remains the dominator**; fixed-width W-series cannot
  reach acc ceilings (~0.4=2.0−1.6; ~0.9=2.5−1.6) vs saved-q 1.6 —
  **strengthened** by larger dense-LIVE dominator; sub-2 route is
  event-coded/sparse/forgettable, NOT narrower fixed width. Full persistent
  sub-2 **unbanked** until acc term + live-carrier authority + base-3
  checkpoint-wiring clear. Never "fully FP-free persistent state."
- **Event-coded live carrier = closed Phase-A accumulator-drain experiment**
  (sparse `hot_exact` + cold default): **bounded negative as a sub-2 mechanism
  at terminal Phase-A geometry** (V4-LIVE); receipts in atlas. Does **not** close
  all event-coded/acc alternatives; no readiness/full-sub2 claim (activations/KV
  are a separate `full_sub2_runtime` surface — see below).
- **Sparse acc route: measurement-backed AND parity-proven (feasibility only)** — acc info content far below both ceilings;
  dense-transient credit (BDL core + TSA) re-carries as sparse integer attribution at bit parity under the frozen A+B CPU fixture (no density/byte-savings/GPU/production/readiness claim);
  remaining `transient_fp_debt` closure = production landing of the sparse byte-level live carrier, incl. TSA B-site integration.
- **3-ledger = weight-persistent train-state accounting; activations/KV are full-sub2-runtime target surfaces with separate levers.** Activations/residuals, attention-KV buffers, and backward-saved tensors are FP today under the D2.1 BitLinear contract (weights ternarized, activations not), and remain required `full_sub2_runtime` surfaces currently blocking main science. Their path is forward/runtime activation-KV quantization or recompute/compression (separately scoped), NOT the weight vote-accumulator: activations are transient, KV has no trainable optimizer state, no persistent votes to accumulate. Do not conflate the persistent-weight drain (the dense vote-acc, int16 container under selected/default dense LIVE) with total-runtime memory (activations/KV scale with batch×seqlen, distinct levers).
- **Ternary-rotor lane** (separately-scoped; plan + screen receipts:
  `.claude/MEMORY/ternary-rotor.md`) covers those runtime surfaces via rotated
  scalar quantization + remat. Two standing invariants from it: (1) SDPA-saved
  q/k/v behind the fused attention kernel are SCREEN-EXCLUDED from saved-tensor
  quantization — the fused backward recomputes scores against the exact forward
  logsumexp, so perturbed saves blow up as exp(Δ); attention precision goes
  through a quantize-then-compute kernel (the KV surface), never a saved-tensor
  swap (dim4-bisect receipt pending). (2) 4-level (2-bit) codes are 2.0 bpw before scales
  under this lane's additive scale-inclusive flat-code ledger — scale packing
  lands ~2.1–2.2 bpw, which **clears the ≤2.5 working bar** but still **fails
  strict <2.0** (accounting stays true; only its disqualifying force is
  tiered). A north-star / "sub-2" KV/runtime claim still requires 3-level
  (ternary) codes + base-3 packing, scale-inclusive. (Ledger-contract
  invariant, not a universal info-theory law.)
- **Two bit-width axes stay separate.** Persistent train-state WIDTH (q / dense-LIVE vote-acc: int8 + int16; W8 is range evidence only, not a realized container shrink → the sub-2 weight target) is distinct from decision/eligibility QUANTIZATION (ranking discrimination). A decision/receipt-family collapse or null is a representation limit, NOT evidence against persistent-width reduction; decision-family discrimination is a separate axis and B5b/H1 nulls do not close the persistent-width lane.

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
  window samples **q/acc pressure**, never backlog age/drainage. No pre-resume-backlog claims
  from a resumed run; ruling out a resume artifact needs a from-clean-parent
  contiguous run.
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
- **Tiny route patches** — earn architecture rewrites with a measurement first.
- **Role routing + ceremony**: churn/thinking → `plan-dev`; exact-packet
  receipts → `test-operator`. Tier by claim effect: a prereg'd feasibility/
  plumbing/parity/null read (does the carrier/path work) = LEAN-MEASUREMENT;
  mechanism-selecting branch / stability/readiness/sub-2 claims =
  science-verdict → HIGH (`CLAUDEX_ORCHESTRATION.md` §review-risk tier).
- **Commit every useful null** — a clean negative shrinks the search space.

Exemplar (a pattern, NOT privileged): lane3 shadow-prefix curve picks the next
mechanism from one run — low-K beats all/random/inverted → rate-cap/trust
lever; random ≥ current → ranking/update-law problem; inverted wins →
sign/direction problem; no arm improves → representation-not-viable.

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
- A run that diagnoses a failure mode is a **shippable null** (workflow.md
  §"Informative null results") — same before/after discipline, receipt on the
  board.

## Relationship to the other lanes

- `hrm-158.md` — curriculum lane (90/90 bank gate, full-density slices). Grows
  `hrm-158-base` via BitLinear / native ternary **effective**-weight training
  with **normal** checkpoint/training state — UNLESS integer-state research
  trainer (q + int16 dense vote-acc under selected/default dense LIVE; no FP
  masters) is selected. Don't conflate the two lanes.
- `training.md` — model/tokenizer specifics + native-ternary-train flags.
- `turboquant.md` — post-hoc quantization (tq4/tq3); NOT this lane (this is
  training-time ternary, not export quantization).
- This lane feeds the curriculum lane only once FP-free persistent training is
  stable; specialists/MoE branch from robust base checkpoints, not weak ones.
