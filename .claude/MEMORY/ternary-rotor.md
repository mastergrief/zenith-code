# Ternary-Rotor Path — rotated low-bit quantization for the HRM-158 `full_sub2_runtime` surfaces

Status: **Phases 0–2 CLEARED, Phase 3a screened (informative null + repair
design), 2026-07-22.** Plan drafted after the Bonsai/rotorquant session
(empirical base: `MEMORY/local-driver.md`). This is a **separately-scoped
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
