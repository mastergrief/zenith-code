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
- The **int8 W8 vote accumulator** — the persistent-state dominator. Integer
  vote tallies are not a correlated geometric vector field; rotation does not
  apply. Its sub-2 route remains **event-coded/sparse/forgettable** per the
  lane rule. Nothing here changes the full-sub2-persistent gate.
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
get under 2.0; it does not touch the int8 vote-acc dominator, which remains
the single genuine sub-2-persistent blocker on its event-coded route.
