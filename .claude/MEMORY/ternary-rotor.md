# Ternary-Rotor Path — rotated low-bit quantization for the HRM-158 `full_sub2_runtime` surfaces

Status: **Phases 0–2 CLEARED, Phase 3a screened (informative null + repair
design), 2026-07-22.** Plan drafted after the Bonsai/rotorquant session
(empirical base: `/mnt/c/Users/gabes/projects/prism-box/local-driver.md`,
moved out of this repo 2026-08-06). This is a **separately-scoped
lane** per `ternary_hybrid_stack.md` — it does NOT modify the curriculum lane
or the vote-acc/event-coded lane, and no claim here is banked until the named
gates clear.

## 0. Status log (receipts in `claw-code-hrm-text-158/artifacts/rotor/`)

All screens ran on the banked parent `L0c2K2add50s...step00750` (the chain
head the active add60* runs train from), RTX 4070, deterministic eval set =
6 rows/rung seed-17 from the A0 exhaustive supports + canonical 17×23 (67
rows), thresholds prereg'd before launch (strict-exact drop ≤2 rows, CE
delta ≤0.10 nats; grad screens: median cos ≥0.99, min cos ≥0.95).

- **Phase 0 DONE**: `calm/hrm_text_158/native_full_stack/rotor_runtime_quant.py`
  (line-faithful torch port of the validated turbo2/turbo3 C path: signed
  seed-42 FWHT, polar group norm, Lloyd-Max centroids, fp16 corrected norm) +
  scale-inclusive bits ledger. 14/14 tests
  (`calm/llm_computer/tests/test_hrm_text_158_rotor_runtime_quant.py`).
  Ledger hand-audit: turbo2 = 2.125 bpw, turbo3 = **3.125** bpw (the sign
  plane is stored at full width in the block layout), int8-scale variant
  2.0625 — none sub-2.
- **Phase 1 VERDICT — prereg branch (b) `3bit_lane_packing_mandatory_for_sub2`**:
  residual seams (`residual.post_attn`+`post_mlp` via the existing
  `activation_codec_seam` — zero model edits needed): 3-bit all-seams fully
  clean (d_strict=+0, d_CE −0.07); 2-bit all-seams FAILS (−5/67 strict), and
  all five failures are carry-adjacent arithmetic rows — the low-redundancy
  compute path the §2 caveat predicted. **Residual stream needs 3-bit at
  29.6M scale; the 27B 2-bit evidence does not transfer to compute surfaces.**
- **Phase 2 VERDICT — prereg branch (a) `kv_at_2bit`**: K(post-rope)+V at
  head_dim=128 (one rotation group): BOTH widths clean (d_strict=+0, d_CE
  within ±0.015). **KV is a retrieval surface — 2-bit holds at HRM scale**,
  matching Bonsai.
- **Phase 3a (gradient-fidelity screen, GACT-style saved-tensors hook,
  forward exact — loss bit-identical in every condition):
  prereg branch (c) park-null for BLANKET saved-tensor quantization**, with
  the null fully classified by bisect:
  - 4-D SDPA-saved q/k/v must NEVER be quantized behind the fused attention
    kernel (backward recomputes scores against the exact forward logsumexp →
    exp(Δ) blowup; measured med_cos 0.65, rel 7×10³). Attention-internal
    precision is the Phase 2 KV lever via a quantize-then-compute kernel.
  - dim3 **512-wide** saved activations at 3-bit: **CLEAN under prereg bars**
    (med 0.997, min 0.992).
  - dim3 **1536-wide SwiGLU-saved** tensors: the entire remaining corruption
    (min 0.748; same-surface cluster on every `mlp.gate_up_proj` grad —
    multiplicative silu backward amplifies quant error).
  - Harness validated: control determinism exact, identity-hook cos=1.0,
    healthy control grad norms (0.38–134).
- **Phase 3b VERDICT — prereg branch (b) `phase3b_run_at_3bit`**
  (quantize-narrow + remat-wide screen, `--mode 3b`): every SwiGLU
  checkpoint-wrapped so the 1536-wide intermediates are RECOMPUTED in
  backward from the saved block input; pack hook rotor-quantizes only
  512-wide dim-3 saves; SDPA stays kernel-exact.
  - `remat_only` control: **exactly lossless** (cos 1.00000 on all 33
    params) — the remat mechanism itself is bit-clean.
  - **3-bit narrow: med 0.995, min 0.979 — clears both prereg bars**
    (vs blanket-mode min 0.750: the SwiGLU cluster is fully repaired).
  - 2-bit narrow: med 0.973 — fails the median bar. 3-bit is the
    activation width, consistent with Phase 1.
  - Structural bytes ledger: the wide family (largest per-token
    activation) is no longer stored at all; the narrow family stores at
    3.125/32 ≈ 10% of fp32.
  - Receipt: `artifacts/rotor/phase3b_narrow_remat_screen_receipt.json`.
  Trainer-production integration point for the same semantics:
  `native_full_stack/activation_residuals_m1_remat.py` (Tier-1 seam remat
  codec). **Next: the full Phase 3 training run** (slice + replay + 90/90
  vs FP-saved control, 3-bit narrow + remat-wide) — remains gated on a
  dispatched launch packet.
- **Phase 4a VERDICT — `kv_sub2_at_ternary_1p75_q175_geometry`**: the sharp
  form of the packing question — 4-level codes are 2.0 bits flat, so NO
  scale packing clears <2.0; the only sub-2 route is **3-level (ternary)
  codes**. Screened rotated ternary KV (Lloyd-Max 3-level for N(0,1/128),
  a=1.224σ=0.108188) on the banked parent: **fully clean** (d_strict=+0,
  d_CE +0.006; also clean on the step-1512 child). Ledger: base-3 pack in
  the proven Q1_75 geometry (26B codes + 2B fp16 scale per 128) =
  **1.75 bpw scale-inclusive — genuinely sub-2** (1.6875 with int8 scale).
  First `full_sub2_runtime` surface with a measured sub-2 path.
  Receipt: `artifacts/rotor/phase4a_kv_ternary_screen_receipt.json`.
- **Phase 3 formal run LAUNCHED** (in flight): trainer integration landed —
  `--backward-saved-codec rotor3b` in `scripts/train_hrm_text_158.py`
  (SwiGLU checkpoint-wrap + 512-wide saves rotor-3bit via facade imports;
  codec/wrap promoted to `rotor_runtime_quant.py` on second use). GPU smoke
  passed (43 steps, finite, learning, peak 3.7GB). Formal arm: exact
  replica of the FP control `L0c2K2add60to89Trace` recipe (rung
  `L0c2-K2-addition-60to89-trace`, seed 17, replay .80, n12k, lr 5e-5,
  pc 1.0, 8 retained supports w/ ENABLED count+hash receipts, ceL0c1x3,
  anchors v1 r3, from `add50s_step00750`) + `--backward-saved-codec
  rotor3b`, saves 250–1500, ~7.5 s/step ≈ 3.2 h. Read: per-save probe A/B
  vs the existing FP control checkpoints under the 90/90 semantics.
- Screen tooling: `scripts/hrm_text_158_rotor_forward_activation_screen.py`
  (`--surface residual|kv|kv_ternary`),
  `scripts/hrm_text_158_rotor_backward_saved_screen.py`
  (`--mode blanket|3b`).
- **Formal-run launch-attempt classification chain** (WSL / 8 GB 4070; all
  attempts bit-identical replicas through their last common step — losses
  match exactly, so kills lose nothing but wall time):
  - **A1 `fragmentation`**: default allocator; 2.9 GB resv-alloc gap, pace
    collapsed 5→45.6 s/step by step ~75. Lever: `max_split_size_mb:128`.
  - **A2 `vmm_crash`**: `expandable_segments:True` → hard crash T+880s
    "CUDA driver error: device not ready" inside pack hook (CUDA VMM
    unstable on WSL). Ruled out; back to `max_split_size_mb:128`.
  - **A3 `h2d_churn`**: healthy 5.2 s/step to step 150 (resv 7.62 GB),
    then permanent stall in the 150→175 window (≥14 min, GPU 100%,
    VRAM pegged). Interrupt stack: `_signed_fwht` rebuilding S1/S2 sign
    tensors from Python lists per pack call (H2D per save). Fix: per
    (device,dtype) constant cache. Result: ~30% faster (3.4 s/step) but
    same stall → churn was real, not root cause.
  - **A4 `pack_transient_spike`**: hypothesis = whole-save fake-quant
    allocates ~10× fp32 transients (~1.3 GB/call). Fix: chunked pack
    (`_CHUNK_VALUES` = 2M, bit-identical via `torch.equal`). Same stall at
    the same window, resv again exactly 7.62 GB → size not the poison.
  - **A5 `alloc_count_cliff`** (current classification): py-spy dump mid-
    stall shows the process ACTIVE inside the FWHT butterfly during a
    checkpointed-forward pack — stuck-slow, not deadlocked. Mechanism:
    FWHT allocates ~14 intermediates/call × thousands of calls/step;
    fine below ~7.5 GB resv, but once reservation crosses the WDDM
    commit region every tiny alloc triggers eviction (~ms each) →
    minutes/step. Allocation COUNT under near-full reservation is the
    poison. Lever A6/A7: `garbage_collection_threshold:0.8` (keep resv
    below the cliff). Fallback if insufficient: allocation-free FWHT
    (preallocated per-shape workspace + `out=` ops).
  - A6 died untested with an overnight host reboot (/tmp wiped —
    lesson: launch scripts + run logs now live in
    `claw-code-hrm-text-158/artifacts/rotor/runs/`, not /tmp). A7 =
    relaunch of the A6 config → same stall (gc_threshold null).
  - A8 `alloc-free workspace pack` (bit-exact, +1 unit test): stalled one
    window EARLIER; peak_alloc unchanged (5.16 vs 5.11) → alloc-count null.
    py-spy mid-stall: grinding in PLAIN forward, not pack.
  - A9 `--empty-cache-every 10`: same stall → resv-gap null. All
    allocator-side levers exhausted (5 nulls).
  - **RECLASSIFICATION (the actual failure): per-step CUDA memory LEAK in
    the codec bundle, ~15-25 MB/step monotonic.** Decisive smokes:
    (a) forced `--bp-min-steps 5 --bp-max-steps 5` WITH codec: 4.9 s/step,
    peak_alloc only 2.9 GB — bp_steps was never the driver; (b) base recipe
    codec-off: FLAT 2.21 GB steps 50→100 — no leak without codec. The
    "stall window" is just where the cumulative leak crosses the 8 GB wall
    (explains why it moved between attempts). The A3-A7 "cliff" story was
    the leak's SYMPTOM (thrash once near-full), not the cause. Leak bisect
    (trainer diag values `rotor3b_remat_only` / `rotor3b_quant_only` /
    `rotor3b_hooks_noop`, non-science arms): wrap clean; quant-only leaks;
    hooks-NOOP leaks identically → mechanism, not content. gc.collect()
    null → reachable, not Python cycles. Referrer forensics
    (`HRM158_LEAK_DIAG`): ~8.7 retained (3064,260) LOGITS roots/step, each
    still carrying grad_fn.
  - **ROOT CAUSE (fixed, unit-tested): pack passthrough returned saved
    tensors AS-IS.** A packed PyObject still carrying grad_fn forms a
    C++<->Python cycle (graph Node -> packed tensor -> grad_fn -> Node)
    invisible to refcounting AND gc — the documented saved_tensors_hooks
    footgun; the torch docs' own pack example detaches. FIX: passthrough
    returns `t.detach()` (storage-shared view; saved-tensor semantics
    unchanged for single-backward). Validation (co_lead-corrected receipt):
    the (3064,260) logits-cycle is GONE; the honest flatness receipt is
    ATTEMPT-10's own cur_alloc 0.67->0.68 GB flat through step 50 at
    3.4 s/step (vs +26 MB/step before). The hooks_noop validation arm
    retains a RESIDUAL slow growth of a different shape ((8,383,260),
    ~6.4 MB/10 steps, grad_fn-roots creep) — named separately, watch under
    real rotor3b; do NOT cite that arm as "flat". FP control never affected
    (no hooks). A10 = formal relaunch, fixed codec. Ops note: /mnt/c drvfs
    READ-CACHE staleness produced two false stall alarms — run logs go on
    the Linux FS next time; drop_caches before pace reads.
- **Gate-2 audit (co_lead, post-hoc, msg `1784793102933-64c5698f`): REVISE
  on receipt hygiene; science cleared.** Ledger math independently replayed
  (turbo2 2.125 non-sub2; ternary 1.75/1.6875 sub2); Phase-4a prereg
  confirmed; refute attempts on both rules invariants failed on mechanism.
  Actions taken: (a) phase4a receipt `bits_ledger` keys were mislabeled
  turbo2/turbo3 for ternary values → screen emitter fixed
  (`ternary_fp16_scale`/`ternary_int8_scale`) + ERRATA sidecar next to the
  frozen receipt; (b) SDPA invariant softened to "screen-excluded, bisect
  receipt pending" in both rules mirrors — **OPEN ITEM: mint a frozen
  dim4-vs-dim3 bisect receipt (GPU, after attempt-10 frees the card)**;
  (c) 4-level invariant scoped to the additive scale-inclusive flat-code
  ledger contract (not a universal info-theory claim).
- **Acc-entropy claim boundaries (audit-sharpened, keep load-bearing):**
  frozen-law projection only (coupled live-q retest before locking mechanism
  design); proven bound is marginal/iid H >> 0.4 (joint/spatial coding open
  but needs ~10x — implausible, not impossible); armC topk=1024 is the cap
  MAX while the live formula starts at 512, so early live can sit above
  armC — armB bounds the favorable side, bracket does not prove the exact
  live schedule.
- **A/B read discipline (attempt 7+):** score as `rotor3b` BUNDLE
  (remat-wide + 3-bit-narrow) vs `none` — not "quant alone"; fake-quant
  pack materializes no packed storage, so VRAM deltas are remat, not codec;
  wall-clock compares vs FP control require identical
  PYTORCH_CUDA_ALLOC_CONF (FP control predates the gc_threshold env —
  numeric A/B unaffected, wall-clock NOT comparable as-is).
- **ATTEMPT-10 TERMINAL (committed HRM `729ac94`; co_lead combined gate-2
  PASS; dual-accepted):** first rotor3b formal run to reach terminal (1513
  steps), `cur_alloc` flat 0.67->0.68 GB end-to-end — detach fix held. A/B
  vs FP control (frozen manifest `rotor3b_attempt10_ab_manifest.json` sha
  `7f4cd62e`, 12 ckpt + 108 audit shas, deterministic aggregator + 2 hashed
  box runners; 1070 audit lane). **BEHAVIORAL finding only (bounded, no
  equivalence margin):** retention NEAR-MATCHED on all audited priors
  (`|count delta|<=2` every save; both clear numeric retained gates at saves
  500-1500) EXCEPT the named single-save exception — step250 `l0c1`
  rotor3b 108/121=0.8926 (<0.90) vs FP 109/121=0.9008; acquisition NOT
  equivalent — rotor3b trails FP 30/120 on the trace target at step1500
  (47 vs 77); NEITHER arm banks (both <90% acquire; procedure slice,
  `trace_held`~0 both — supervision-shape gap, not rotor). **Do NOT restate
  as "retention-equivalent" or "rotor3b==FP".** Gradient-magnitude mechanism
  ("transparent at minima / noise where large") is HYPOTHESIS-only — needs a
  direct per-arm gradient comparison. Observed `pc_kl` max 0.164825.
- **Vote-lifetime screen (persistent q/acc pivot classify-before-build;
  committed `729ac94`; co_lead PASS):** F4_underspecified via censor_guard —
  `lifetime_censored_frac` 0.9951 (>=0.50) + p50 lifetime 79 (>=32); the
  rider-bound guard refused a forgetting-family pick on 99.5%-censored data.
  Raw directional signal (NOT a verdict): `never_convert_frac` 0.9947
  (99.5% of vote mass never flips) — consistent with F3 sparse-hot/
  forgettable-cold, withheld until censoring clears. F4 NEXT (prereg): longer
  window and/or coupled-q motion — a SEPARATE future plan, not authorized by
  this null.

## 1. What this path is

Apply rotorquant-style **block-diagonal-rotation + scalar quantization**
(TurboQuant WHT / PlanarQuant Givens-2D / IsoQuant quaternion-4D families;
working reference now in-tree — §3) to the HRM-158 **runtime/transient
surfaces** that carry FP debt today and block
`full_sub2_runtime_ready_for_science`:

| Surface (checker row) | Today | Rotor target | Prize |
|---|---|---|---|
| Attention-KV buffers | FP (D2.1: weights ternary, activations not) | rotated 2–3-bit K/V | small in bytes (max_len 384) but flips the row off `transient_fp_debt` |
| Forward activations / residuals | FP | rotated 2–4-bit hidden states | medium; cleanest transfer of the technique |
| Backward-saved tensors | FP (the big transient debt in training) | rotated 2–4-bit saved activations (GACT-style) | **largest lever by bytes**; highest risk (perturbs gradients) |

**Explicitly out of scope / untouched by this lane:**
- The **dense-LIVE vote accumulator (int16 container under selected/default
  dense LIVE; W8 = range evidence only)** — the persistent-state dominator
  (W8 branch-2 `1785833373077-316a0309` / PASS `1785833670092-646c6665`).
  Integer vote tallies are not a correlated geometric vector field; rotation
  does not apply. Its sub-2 route remains **event-coded/sparse/forgettable**
  per the lane rule. Nothing here changes the full-sub2-persistent gate.
- `lm_head`/`embd`/norms and frozen FP32 scales — standing `explicit_exception`
  class, unchanged.
- q itself — already 1.6 bpw base-3 saved-byte; done.

## 2. Honest accounting (do not overclaim)

- **"Sub-2" is scale-inclusive (<2.0 bpw) per the lane's own rule.** Rotorquant
  as shipped is 2-bit codes + fp16 group-128 scales ≈ **2.125 bpw — NOT
  sub-2**. Claims from this lane are "near-2-bit runtime surface" until a
  named packing step clears 2.0: wider groups, int8/shared scales, or a
  base-3-style pack on the code stream (the Q1_75 trick, proven for weights).
- **Evidence regime caveat:** all 2026-07-22 quality evidence (needle-exact
  retrieval at 40K depth with 2-bit and 3-bit rotated KV) is from a **27B**
  model (Bonsai). A 29.6M HRM is far less redundant; tolerance must be
  re-measured at our scale, not assumed.
- **Gradient-path caveat:** backward-saved quantization perturbs gradients and
  can interact with the ternary **direction-flip erosion** dynamics the
  stability lane studies. Any backward-surface run is single-variable,
  pre-registered, and read against the 90/90 retention machinery.

## 3. Assets in hand (from the Bonsai session)

- **Merged working tree**: scratchpad `prismml-llamacpp`, branch
  `q1_75-planarquant` (commits `e003240` Q1_75, `b050761` planarquant merge);
  built + validated on the box at `~/prismml-llamacpp-pq/`.
- Type inventory: `turbo2` (2-bit polar, no QJL), `turbo3` (2-bit + 1-bit
  QJL), `turbo4`, `planar3/4` (Givens, deferred-K), `iso3/4` (quaternion,
  deferred-K). Enum slots 44–50 (Q1_75 keeps 43 — baked into on-disk GGUFs).
- CUDA kernels validated on **sm_61 Pascal** (oldest plausible target; runs
  everywhere newer). CPU reference paths: `ggml-turbo-quant.c`,
  `ggml-planar-quant.c`. Python/Triton research stack: scratchpad
  `rotorquant/turboquant/`.
- Rotation matrices are **fixed, model-independent constants**
  (`src/turbo-rotation-data*.h`) — no per-model calibration. Rotation block =
  128 → matches HRM-158 `head_dim=128` natively; hidden 512 = 4×128 blocks.
- Known defects: planar/iso K **double-buffers a full f16 K** during prefill
  (deferred quantization) — fine in RAM, fatal for VRAM capacity; planar3 +
  host-KV (`-nkvo`) **segfaults** (deferred-convert assumes device buffers);
  turbo2 has **no published PPL** upstream (our needle pass is the only
  quality evidence).

## 4. Phased plan (each phase = one atom, prereg'd, shippable null)

Per `ternary_hybrid_stack.md` §"Fastest-science loop": pre-register the branch
classifier before launch; one variable per run; N=20 preterminal screen, N=50
equivalent for branch verdicts; commit every useful null.

### Phase 0 — measurement harness + bits ledger (CPU-safe, no science claim)
- Wrap the rotate+quantize reference into a small importable facade for the
  HRM probe stack (architecture_discipline: facade, not a god-file bolt-on).
- Deliverable: a **bits-ledger function** for any tensor — effective bpw,
  scale-inclusive, per the 3-ledger discipline. This is the accounting
  authority for every later claim.
- Gate: round-trip unit tests (rotate→quant→dequant→unrotate) match reference
  within tolerance; ledger audited by hand for one tensor.

### Phase 1 — forward-activation tolerance screen (GPU; cheapest decisive read)
- Hypothesis: HRM-158 forward passes tolerate rotated 3-bit (then 2-bit)
  activation quantization at ≤ε CE degradation on banked-support probes.
- Design: quantize→dequantize the hidden state at ONE seam (post-block
  residual), inference only, on the banked `hrm-158-base` chain head. Sweep
  {3-bit, 2-bit} × {1 seam, all seams}. NO training.
- Prereg branches: (a) both widths clean → Phase 2 at 2-bit; (b) 3-bit clean,
  2-bit degrades → 3-bit lane; packing step becomes mandatory for any sub-2
  talk; (c) 3-bit degrades → representation-not-viable at HRM scale → park
  with a committed null.
- Measurement: strict-exact + parsed-correct on A0 exhaustive supports + probe
  CE deltas. Small-model redundancy is THE question.

### Phase 2 — attention-KV surface (inference)
- Rotated K/V at head_dim=128 in the HRM attention path; rerun Phase 1 gates +
  retention probes. Prize is the checker row, not bytes.
- Gate: A0 audit parity — no new broad cluster vs unquantized inference.

### Phase 3 — backward-saved tensors (training; the big lever, highest risk)
- Enter only after Phases 1–2 clear at the chosen width.
- Design: GACT-style — save rotated+quantized activations, dequantize for
  backward. ONE run, one slice, from a clean banked parent, standard slow-safe
  recipe, replay+pc unchanged. This is a TRAINING-DYNAMICS experiment: read is
  acquisition/retention vs an FP-saved control run + the erosion/collision
  diagnostics from the stability lane.
- Prereg branches: (a) 90/90 parity with control → surface banked at that
  width; (b) acquisition parity but retention collision → classify (gradient
  noise mimicking rewarm perturbation? cluster-swap?), split smaller or raise
  width — do NOT knob-escalate; (c) acquisition damaged → bounded negative for
  that width; drop to next width or park.
- Hard rules: no `.pt` commits; watcher/OVERLAP discipline for long runs;
  receipts to the board.

### Phase 4 — packing step to clear <2.0 (only if a 2-bit surface banks)
- Kill the scale overhead: wider groups / int8 scales / base-3 pack of the
  2-bit code stream (Q1_75 machinery). Deliverable: checker rows move to
  `sub2` class under the scale-inclusive ledger, with the executable checker —
  not this file — as authority.

## 5. Sequencing, routing, claims

- **Routing**: plan/contract + implementation → `plan-dev`; formal runs →
  `test-operator`; claude gate-1 → co_lead gate-2 sequential review;
  board-first task before Phase 0 code.
- **Priority**: must not displace the active curriculum slice or the vote-acc
  event-coded investigation. Phases 0–1 are cheap and parallel; Phase 3
  competes for 4070 time and queues behind curriculum launches.
- **Claim discipline**: nothing is "sub-2 runtime achieved" until the
  `full_sub2_runtime_ready_for_science` checker passes with these rows in
  `sub2`/`explicit_exception` class and the bits-ledger receipt is on the
  board. "Rotor gives everything but the vote-acc" is the *candidate-path*
  sentence, not a banked result.

## 6. One-line summary

Rotorquant provides a working, in-tree, Pascal-proven mechanism for every
`full_sub2_runtime` transient surface (KV, forward activations,
backward-saved), landing at ~2.1 bpw as shipped with a known packing step to
get under 2.0; it does not touch the dense-LIVE int16 vote-acc dominator
(W8 range-only; `1785833373077-316a0309` / PASS `1785833670092-646c6665`),
which remains the single genuine sub-2-persistent blocker on its
event-coded route.

## 7. Phase-0 carve null — attempt-10 rotor3b trace feasibility (2026-08-08)

> Receipt only. No science over-claim. No readiness / bank / codec attribution.
> Atlas/receipt home for this rotor-lane surface; not `ternary_hybrid_stack_arc`
> (vote-acc / FP-free persistent lane).

**The run.** Arm-A rotor3b
`fa0209851a0b559c6ab4f3f78dce2fd50275ef2cb2af7bad2041a373ef85a10e`,
ckpt step **1513**,
`--surface kv_ternary --max-gen 160 --phase0-n-rows 10`,
`EXIT=0`, `.pt` pre == post.
Authority: `+1 launch` `1786216405035-ea739d54`; terminal
`1786216640432-0e210cf0`; co_lead `1786216791437-2965244c` /
`1786216973217-4f4fe972`; atlas dispatch `1786217111547-a138913e`.
Frozen plan v4 `7396b8b4…` (consumer = this completed run).

**Training provenance (arm A) — facts and dates only; no attribution.**
- arm A save ladder trained **2026-07-23** (st_mtime 10:32:57 → 12:17:46, seven
  saves, monotonic ~20 min spacing).
- trainer introducing the rotor3b codec: `52de311c`, **2026-07-22 22:31** —
  twelve hours before that first save (ARRIVED from gate locators).
- rung generator: `313e88fd`, **2026-05-27 20:50**, unchanged since (ARRIVED
  from gate locators).
- sibling step-saves under "not taken" item 2 share arm A's save ladder
  (monotonic mtimes, OBSERVED); trainer identity for those bytes is
  INFERRED from the ladder plus the `52de311c` date — not read from a
  launch record (same binding as not-taken item 2).

| | baseline | tt_kv |
|---|---|---|
| exact trace | 3/10 | 0/10 |
| parsed answer | 4/10 | 0/10 |
| too_long / nonfinite | 0 / 0 | 0 / 0 |
| median emitted "tokens" (whitespace words) | 10.0 | 2.0 |
| mean per-row elapsed | 4.77782912140101 s | 7.389483192700572 s |

`phase0_class = traces_truncate_or_absent`, sole failing conjunct
`exact>=5` (3 < 5). `m_b = 4.77782912140101`, `m_t = 7.389483192700572`,
`m_t/m_b = 1.5466`. `per_row_elapsed_s` **10/10 distinct** on both conditions.

**The null (headline).** Two independently-authored gates refuse this
checkpoint: Phase-0's `exact>=5` at **3/10**, and Phase-1's
`baseline_floor = 0.50` (frozen v11 `:306`) at an observed **0.30** — below
even F5's known-bad fixture value of **0.40** (v11 `:152`). A codec-tolerance
screen measures degradation from a baseline; with 3 of 10 correct there are
three rows a codec could break. Shippable null per `workflow.md`
§"Informative null results".

### Train membership — structural, with the weakest link marked

(addendum `1786217208549-677312e1` to atlas dispatch `1786217111547-a138913e`)

`generators.py:3495-3515` is the train-side sampler for this rung: when
`split=="train"` it takes `pool = _l0c2k2_addition_60to89_trace_train_enumerate()`,
then `while len(out) < n: cycle = list(pool); rng.shuffle(cycle)` — the
**entire pool**, order only; no subsampling and no holdout from this pool.
Pool size measured: **120** train / **40** held
(`PYTHONPATH=. python3 -c` over the two enumerate helpers). At `n = 12000`
that is exactly **100×** every train row. Phase-0's ten rows are `pool[:10]`,
so their train membership is structural for any `n >= 120`.

| link | status |
|---|---|
| sampler cycles the whole pool, no holdout | **OBSERVED** — read at `generators.py:3495-3515` |
| pool = 120 rows; `n=12000` → 100× each | **OBSERVED** — computed (`len(train_enumerate)=120`) |
| this checkpoint used this rung at `n >= 120` | **recipe-naming evidence** — filename encodes `L0c2K2add60to89TraceRotor3b…_n12k_…`; **not** the launch-command record, and the `.pt` was not opened |

### Classification consequence — SUPERSEDED reading (kept visible)

> **SUPERSEDED** by arm-B control terminal `1786219418393-102409a0` (BRANCH 2).
> Falsifier: B baseline exact **6/10 (0.60)** on the **identical** ten-row list
> (field-compared: `row_source` / `phase0_n_rows` / `sample_seed_used` /
> `max_gen` / `surface_conditions` / `schema` / `n_trace_golds` SAME; recorded
> `rows` lists identical 10/10). Surface is answerable today; the surface-level
> reading dies. Retraction is durable — do not silent-overwrite.

**Original reading (held, then falsified):** By `hrm-158.md:137` arm-A 3/10 is
a **train-set miss** (row IS in train, decodes wrong). Repair column
*"targeted singleton repair"* presumes a singleton — **seven of ten train rows
decode wrong** on arm A. Operative clause was `hrm-158.md:54`: record as an
**acquisition diagnostic on the trace surface**, upstream of every codec
question (and removing the framing in which `exact>=5` could look mis-set).

**Current reading:** re-scope the diagnostic to **arm A**, not the surface.
Arm-A failure is unchanged and still real. What dies is the claim that the
surface is **unanswerable**. The surface remains **unacquired** — under
`hrm-158.md`'s 90/90 both arms fail acquisition. **Currency (two instruments):**
Phase-0 screen n=10 → A **0.30** / B **0.60**; frozen attempt-10 manifest
n=120 `trace_train`@1500 → A **0.3917** (47/120) / B **0.6417** (77/120).
Both resolutions fail 90/90; only Phase-1's 0.50 tolerance precondition moved
(B clears it at either resolution; A does not). See arm-B + reconciliation
subsections.

### Two available measurements — SUPERSEDED absences (search space named)

1. **Held-set read.** Sampler names a **40-row** trace audit support never
   sampled into train (`_l0c2k2_addition_60to89_trace_held_enumerate`);
   **disjoint** from the 120-row train pool on question text (unique 120/40;
   train∩held = 0 — OBSERVED).
   > **SUPERSEDED absence claim.** Prior text said this held read "is currently
   > unmeasured." That asserted absence **without naming the space searched**.
   > The space is frozen attempt-10 A/B manifest
   > `artifacts/rotor/runs/rotor3b_attempt10_ab_manifest.json` sha
   > `7f4cd62eaa589df18ca046526af3e141af90fd917a040778f6366213f9b16d37` /
   > 48448 B: `per_surface_table.trace_held` is measured at **every save on
   > both arms** and is **0/40** at all saves except `fp_control@1500 = 1/40`.
   > What remains unmeasured under the **Phase-0 screen** at `_final` (1513)
   > is a different instrument/timepoint — not "held unknown."

2. **Sibling step-saves** on disk (`…_final_step00250`…`01500` + `_final`).
   Same save ladder (~20 min spacing across ~1h45m, 10:32:57 → 12:17:46,
   monotonic, OBSERVED); trainer identity INFERRED from ladder + `52de311c`
   date — not a launch record.
   > **SUPERSEDED for train ladder at n=120 (steps 250–1500).** Manifest
   > `trace_train` already holds those six saves both arms (see reconciliation
   > ladder). Arm A max **0.3917** — never clears 0.50. Manifest does **not**
   > cover `_final` (1513). Sibling-sweep GPU cancelled (v1–v3 DEAD).

**Four contract defects against frozen v4** — recorded, deliberately **not**
cured in v4 (consumer is the completed run; cure travels to the next Phase-1
packet):
1. `median_emitted_tokens` counts **whitespace words** (screen `:589-591`),
   while its threshold and `max_gen` are in **byte-tokens** — three quantities,
   two units, one comparison.
2. `median <= 80` is a **dead conjunct** for trace-shaped output: golds measure
   ~5.03 bytes/word, so `max_gen 160` caps trace-shaped output near ~31 words
   (reachable only by pathological whitespace-dense output; dead for this
   surface, not absolutely).
3. **F3d guards an unreachable world absolutely**: firing input `median = 120`
   needs ≥239 bytes against a 160-byte budget.
4. `traces_truncate_or_absent` is a **bare `else` on a four-conjunct AND**
   (`screen.py:299-309`): positive class
   `traces_emit_and_terminate` verifies every conjunct it names (sound);
   negative class names truncation whenever *any* conjunct fails — how arm A
   got a truncation label at `too_long == 0` and median 10.0 (accuracy miss).
   Cure is a **negative class that names which conjunct failed**, not a
   relabel. Also: `phase0_class` is computed from **baseline only**
   (`:627-634`) and says nothing about `tt_kv`.

**`m_t` is a bimodal mean, not an estimate.** Content-matched, directly
observed, no extrapolation (arm A):

```
baseline >=10 words  n=8  mean= 5.124s
tt_kv    >=10 words  n=4  mean=17.050s
tt_kv    <=2 words   n=6  mean= 0.949s
content-bearing ratio 3.33x  vs  m_t/m_b 1.5466x  ->  m_t understates by 2.31x
```

Six of ten `tt_kv` rows emitted 1–2 words. **Any Phase-1 budget must use
content-bearing figures, never the bimodal mean** — arm A 17.05 s at n=4
(not 7.39 s); arm B same defect shape (see arm-B subsection).

**Constraints for the NEXT Phase-1 packet** (binding; do not edit v11/v12 here):
Phase-1 keys on **conjunct values, never the class label**; the **negative
class names which conjunct failed** (not a bare else / relabel of
`traces_truncate_or_absent`); `phase0_class` is baseline-only and must not be
read as a `tt_kv` verdict; the `median` conjunct arrives in reconciled units
or is deleted; no unreachable-world fixtures (F3d's shape); `m_t` enters as
the content-bearing figure (arm A 17.05 s / n=4; arm B 19.518 s / n=4 class),
never the bimodal mean.

**Named unverified — remaining after addendum closed train membership:**
- launch-command record that this arm trained this rung at `n >= 120` (filename
  encodes it; `.pt` not opened);
- whether this arm ever **banked** the trace surface;
- whether correct trace emission costs what similar-length incorrect emission
  costs at the content-bearing n=4 class;
- whether any sibling checkpoint / step-save clears 0.50 here (candidates on
  disk; not measured).

**Explicitly NOT claimed (arm A / pre–arm-B):**
- `tt_kv` 0/10 at median 2 words is **not codec evidence**. Phase-0 mints no
  codec attribution.
- `traces_truncate_or_absent` on arm A is an accuracy-miss under the exact≥5
  conjunct routed through a bare else — not a truncation proof, and not a
  surface-level unanswerability claim (see SUPERSEDED classification + arm B).
- R7 applies at n=10; arm-A observed rate 0.30.
- Nothing here touches `ready_for_main_science` (stays false) or the parked
  interlock `1786191801421-4f35bb78` under HARD HOLD `1786131978901-9738999d`.

### Arm-B control — non-rotor twin (BRANCH 2; 2026-08-08)

Authority: freeze `PHASE0_ARMB_LAUNCH_DELTA_v2.md` `0a558771…` → gate-2 PASS
`1786218944721-7e6ab491` → `+1 launch` `1786219095412-a92294c7` → terminal
`1786219418393-102409a0`. Numbers from
`artifacts/rotor/phase0_armb_control_nonconsumable/phase0_armb_control.json`
fields (not room prose).

| | baseline | tt_kv |
|---|---|---|
| exact trace | 6/10 | 0/10 |
| parsed answer | 6/10 | 0/10 |
| too_long / nonfinite | 0 / 0 | 0 / 0 |
| median emitted "tokens" (whitespace words) | 10.0 | 1.0 |
| mean per-row elapsed | 5.154594410899881 s | 8.129835036598525 s |

`phase0_class = traces_emit_and_terminate` (earned: positive class verifies
every conjunct; baseline-only). `phase0_exact_trace_rate_on_10 = 0.6`.
`ckpt_step = 1512`. `device = cuda`. `.pt` pre == post ==
`10d0e60fcf8f744026600e8771e6b13b2dba4a0da5cc7fcd835cae717b39ab3b`.
Write surface: only under
`artifacts/rotor/phase0_armb_control_nonconsumable/`
(`phase0_armb_control.json` + log).

**Comparability (fields, not "same rows" prose):** A receipt
`phase0_trace_feasibility_armA/phase0_baseline_feasibility.json` vs B receipt
above — `row_source` / `phase0_n_rows` / `sample_seed_used` / `max_gen` /
`surface_conditions` / `schema` / `n_trace_golds` all SAME; recorded `rows`
lists **IDENTICAL** 10/10 (first question: `63 plus 7 equals what?`).

**Branch 2 (pre-registered classifier):** A 3/10 (0.30) → B 6/10 (0.60) on
identical rows = a measured A/B gap on the trace surface; **cause not
attributed** — arms differ in the rotor arm AND `layers.py` / `hrm.py` /
`transformer.py` / the trainer across 2026-05-27 → 2026-07-23. Surviving
negative form (v2 `:98` positive half retracted): **a gap between A and B is
never evidence about ternary-KV tolerance.**

**Reading consequence:** atlas prior "acquisition diagnostic on the **trace
surface**" is **SUPERSEDED** (see classification subsection). Diagnostic
re-scopes to **arm A**. Surface is answerable at 0.60 on a checkpoint that
exists today (B). Arm-A 3/10 failure unchanged.

**B `tt_kv` NON-CONSUMABLE** (receipt prose is primary; path is name
obfuscation / defense-in-depth only — position-keyed globs under
`artifacts/rotor/**/*.json` still catch it): emitted, not budget/timeout/
tolerance authority. Same bimodal shape as A (computed from B receipt
`token_counts` + `per_row_elapsed_s`):

```
tt_kv content-bearing (token_count >= 8)  n=4  mean=19.51797105924561 s
tt_kv near-empty     (token_count <= 2)   n=6  mean= 0.5377443548338002 s
content-bearing / m_t  19.518 / 8.130  -> understates by ~2.40x
```

`m_b` needs no correction (all baseline rows content-bearing on B).

**Not taken by this control:** Phase-1 consumption of B baseline (clears
frozen v11 `baseline_floor=0.50` at 0.60, but **exceeds** delta v2 declared
consumability = classifier only) — successor HIGH contract + gate.
Arm-A sibling step-saves remain arm-A-scoped (B clearing 0.50 does not
answer them).

**Explicitly NOT claimed (arm B / gap):**
- no rotor3b-only (or any single-factor) attribution of the A/B gap;
- no ternary-KV tolerance claim from the gap or from B `tt_kv`;
- no readiness / bank / `ready_for_main_science` (stays false);
- no Phase-1 authorization from this fold.

### Reconciliation — frozen attempt-10 A/B manifest (zero GPU)

Authority: dual-accepted plan `PHASE0_S7_RECONCILIATION_PLAN_v7.md`
`6c8a5f60fb5c57ce37b6d09f04987846efa25e3101e8ba648118a772b77c997e` /
9322 B / 0444; gate-1 freeze `7452e713…`; co_lead gate-2 PASS
`1786223095427-6cd54aa7`; `+1 implement` `1786223126009-7580d365`.
Implement from frozen plan bytes. Advisor disposition on freeze lineage:
`ADVISOR: consulted 1786219135553-2346de8b` + `CLAUDE_REDERIVATION: partially
adopted` (field-layer-only-outlives-session adopted; prose-vs-field framing
rejected as cure mechanism; named-unverified-list join kept as bounded
detector).

#### Manifest identity + `trace_train` ladder (n=120)

Source: `artifacts/rotor/runs/rotor3b_attempt10_ab_manifest.json` sha
`7f4cd62eaa589df18ca046526af3e141af90fd917a040778f6366213f9b16d37` / 48448 B.
**12/12** checkpoint `sha256` values re-hashed byte-identical to on-disk arm-A
(rotor3b) and arm-B (fp_control) ladder files — **same bytes**, not basename
match. Manifest does **not** cover `_final` (step 1513).

`per_surface_table.trace_train` (box audit lane; not Phase-0 screen):

| save | rotor3b (A) | frac | fp_control (B) | frac | Δ |
|---|---|---|---|---|---|
| 250 | 0/120 | 0.0000 | 0/120 | 0.0000 | 0 |
| 500 | 0/120 | 0.0000 | 0/120 | 0.0000 | 0 |
| 750 | 1/120 | 0.0083 | 0/120 | 0.0000 | +1 |
| 1000 | 4/120 | 0.0333 | 13/120 | 0.1083 | −9 |
| 1250 | 31/120 | 0.2583 | 44/120 | 0.3667 | −13 |
| 1500 | 47/120 | 0.3917 | 77/120 | 0.6417 | −30 |

Arm A **monotonic rise, never clears 0.50** (max 0.3917). Earliest B clear of
0.50 on this ladder: **step1500 only** (1250 at 0.3667).

**Margins vs Phase-1 floor 0.50** (floor untouched; SE = sqrt(p(1−p)/120)):

| point | rate | (rate−0.50)/SE |
|---|---|---|
| A@1500 | 0.3917 | **−2.43σ** |
| B@1500 | 0.6417 | **+3.24σ** |
| B@1250 | 0.3667 | **−3.03σ** |

A/B gap @1500: **0.2500**, z = **4.00σ** (vs ~1.41σ at n=10). Gap is real;
**stays unattributed** (rotor arm + `layers.py` / `hrm.py` / `transformer.py` /
trainer; eight-week separation).

#### Sibling-sweep cancelled

PLAN v1/v2/v3 DEAD. Three of four stage-1 n=120 points already in manifest;
only `_final`@n=120 under Phase-0 unmeasured. **No `_final` n=120 run:**
clearing 0.50 from 0.3917 in 13 steps needs +13/120 rows; n=10 put `_final`
at 0.30 (~1 SE of 0.3917) — foregone. Branch-set asserted "mutually exclusive
and jointly exhaustive" was **false** (ARRIVED retraction): no class for
*monotonic rise that never clears the floor* — the observed A ladder.

#### Held finding + structure (categorical null)

`trace_held` n=40: **0/40 both arms every save** except `fp_control@1500 =
1/40`. Train rises to 0.3917 / 0.6417.

**Measured structure:** op1 held∖train = ∅; op2 held∖train = ∅; op2 sets
identical; **PAIRS in held also in train: 0 of 40** (genuine held; no pair
leakage). Expected held exact under train rate 0.3917: **15.7**; observed
**0**; **P(0 of 40 | p=0.3917) = 2.32e-9** — categorical, not a sampling floor.
Stated-and-rejected: "both low ⇒ undertrained" does not survive the asymmetry.

**Gold already shaped** (current state, not a next step):
`ones 3+7=10; write 0 carry 1; tens 6+1=7; answer 70` =
planning/reasoning/computing/answer. Train 0.39 ≠ train-perfect →
`hrm-158.md` curriculum-null clause does not fire as written.

**Classification:** surface **neither acquired nor generalising**; shaped
trace emitted as memorised text rather than computed. A codec-tolerance screen
on **trace** would measure tolerance of **memorised retrieval**, not
computation — upstream of Phase-1 / ladder / codec. No bank/acquire claim; no
supervision-change prescription in this fold.

#### Three-way surface reachability (two visible retractions)

| surface | reachable by screen today? | status |
|---|---|---|
| **`mathA0`** (R0 / R1 / R1b1–9) | **yes, no code change** (non-phase0 + `--surface kv_ternary`) | **screened on a 67-row sample = 5.34% of the 1255-row support**, at a checkpoint that is **not** an attempt-10 child; prereg passed |
| **`idfull` / `add120` / `add50s` / `l0c1` / `language`** | **no** — separate probe audit flags, not `build_exhaustive_supports()` | **row-builder work required** |
| **trace** (`trace_train`) | Phase-0 path only, hardcoded | today's null (memorised emission) |

Mapping: manifest `surface_flag_expansion` (`mathA0` →
`--exhaustive-finite-supports`); screen `:160` `_build_eval_rows` →
`build_exhaustive_supports()`. **Shared-producer identity** (not row-set
diff): probe and screen both call that builder; aggregate **1255** matches
manifest `mathA0` total; no interposed filter on the measured call path
(cardinality + shared producer — not byte-identical row enumeration).

**Retraction 1 (visible):** original "screening idfull/add120/mathA0/add50s
requires a code change" / whole-screen "not flag-reachable" — **FALSE in
scope** (over-generalised from Phase-0 only).

**Retraction 2 (visible):** first retraction over-corrected to "acquired
surfaces = that builder" — **FALSE**; builder covers **`mathA0` only**.
"Requires a code change" was false for mathA0 and **true** for the other four.

**Still OBSERVED:** Phase-0 `build_phase0_rows` hardcodes trace prefix;
`--surface` / `SURFACE_CONDITIONS` select **codec** conditions, not row sets.

#### phase4a on mathA0 sample (already measured)

Path: `artifacts/rotor/phase4a_kv_ternary_screen_receipt.json` sha
`6a0d15a5b9a5345561e802c00c2615813dd1d3ff17d0feee4eba81ae96c5d402` / 31102 B.

| field | value |
|---|---|
| row set | mathA0 sample, n=67 / 1255 = **5.34%** |
| ckpt | `…L0c2K2add50s…_step01000_final_step00750.pt` step **750** — not attempt-10 |
| baseline / tt_kv | strict_exact **66/67** both; CE 0.26293009170245973 → 0.269338747389899 |
| observed | exact drop **0**; ce delta **0.00641** nats |
| prereg | drop_max_rows 2; ce_delta_max_nats 0.1 |
| verdict | `kv_sub2_at_ternary_1p75_q175_geometry` |
| ledger | turbo2 **1.75 bpw** scale-inclusive, `sub2_scale_inclusive true` |

**mathA0 acquisition ladder** (manifest; replace weakest single-save cite):

```
save   rotor3b            fp_control
 250   1254/1255 0.9992   1252/1255 0.9976   delta 2
 500   1254/1255 0.9992   1254/1255 0.9992   delta 0
 750–1500  1254/1255 0.9992 both
```

Manifest reports mathA0 **flat at 1254/1255 (0.9992) across all six saves for
rotor3b**; **fp_control is 1252/1255 (0.9976) at 250 and 1254/1255 from 500
onward**. Acquired and stable — not a single-save read. **Guard:** "flat
across six saves" is **exact for rotor3b** and **false for fp_control@250** —
both arms written explicitly; do not collapse to "both arms flat."

**Bounds:**
- Exact channel at ceiling (66/67) → ≤2-row prereg low power; **CE carries
  sensitivity**.
- Different checkpoint than attempt-10 children.
- **Same `tt_kv` codec on Phase-0 trace collapsed decode length** — 6/10 rows
  to a **single token**, exact **0/10** vs baseline **6/10** on identical rows
  (`phase0_armb_control.json` tt_kv `token_counts` `[16,1,17,1,8,9,1,1,1,1]`,
  baseline `[10,22,10,15,10,8,10,17,10,10]`). phase4a prereg carries **no
  emission-length or collapse class** → collapse on mathA0 would report as
  generic exact drop / codec intolerance, not collapse. **Same absence as
  v11's missing `codec_emission_collapse` terminal class.**

#### v11 four symbols still absent

`find` over both repo trees → **1819 + 946 = 2765** `.py` files; **zero**
`def` hits for each of:

| symbol | hits |
|---|---|
| `build_active_rows` | 0 |
| `assert_phase1_preflight_ok` | 0 |
| `select_phase1_timeout_key` | 0 |
| `assert_timeout_key_valid_for_arm` | 0 |

Implementing only `build_active_rows` leaves v11 unexecutable. Blocks **v11
Phase-1 as specified**, **not** non-phase0 mathA0 codec measure (phase4a
proof). Secondary: gitignore-respecting `rg` enumerates fewer (2727 / 2709
observed) and also zero. Bare name `build_active_rows` appears only in three
claw-code **minutes** transcripts (end-time component drifts while sessions
run) — not code; symbol referenced nowhere except records of its absence.

#### Settlement order (split) + successor

| target | implement for codec measure? |
|---|---|
| **mathA0** | **no** — phase4a already ran on sample; attempt-10 transfer = cheap successor |
| **idfull / add120 / add50s / l0c1 / language** | **yes** — row-builder work |
| **trace** | N/A for tolerance-of-computation — today's null |

**Cheap open question (successor; not this write):** does phase4a's **mathA0**
result hold on **attempt-10 children**? Same non-phase0 path, `--surface
kv_ternary`, ~2 min GPU/ckpt, no code. Successor prereg must not rest on
exact-drop and CE alone if emission collapse is in scope — name
`codec_emission_collapse` as its own terminal class (v11 gap).

#### Named unverified (remaining)

- Cross-instrument parity: box-aggregator vs Phase-0 screen (n=10≈n=120 within
  ~1 SE is evidence, not proof).
- `_final` (1513) at n=120 under Phase-0 screen — unmeasured, not scheduled.
- Weights A vs B never compared; multi-factor drift benignity undischarged.
- phase4a → attempt-10 mathA0 transfer unmeasured.

#### Session pattern (finding, not work)

Four narrowings-after-drafting where artifacts already held answers: sibling
ladder (manifest); held (manifest); phase4a (rotor artifact); rung-name mapping
(screen + `surface_flag_expansion`). "Has this already been measured / is it
already reachable?" is on no gate checklist.

#### Explicitly NOT claimed (reconciliation fold)

- no GPU in this fold; no Phase-1 authorization; no acquire/bank/readiness;
- no supervision-shape prescription; no implement of the four absent symbols
  or multi-surface row builders; no attempt-10 re-run;
- no rotor-only A/B cause; no ternary-KV tolerance claim from trace tt_kv or
  from the A/B gap;
- `ready_for_main_science` stays false; HARD HOLD `1786131978901-9738999d`
  untouched; interlock parked.
