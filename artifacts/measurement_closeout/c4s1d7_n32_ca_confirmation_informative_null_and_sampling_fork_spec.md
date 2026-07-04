# C4.S1d.7 n=32 CA-confirmation informative null + structural-vs-sampling fork spec

**Schema:** `hrm_text_158_measurement_closeout_design_spec/v1`  
**Task:** `1783074444338-d3ef40af` (parent Arc#2b `1782633464140-b85ec12a`)  
**Frozen plan source:** design_proposal `1783201482322` (+1 implement `1783201548146`)  
**Terminal read basis:** recovered primary n=32 receipt + n=16 fallback receipt (on-disk)

---

## 1. What ran

n=32 CA-confirmation using the fold-2a runner (`run_callsite_band_counter_ca_confirmation`) at fold-2b packet revision `v1_n32_ca_confirmation_4c355f7` / git baseline `4c355f7` (packet committed `a34d508`).

- `eligible_module_limit=32` threaded (all-bitlinear eligible scope)
- Default 4-point sampler: `sampled_states=[0, 10, 21, 31]` (`{0, n//3, 2n//3, n-1}`)
- `mark_count=4` (one band-counter mark per sampled state)
- Band-counter-only B-arm (`TRACEMALLOC=0`, `PROFILE_S1D7_BAND_COUNTER_ONLY=1`), bootstrap `-B` guard-ordering
- Static-pre-append measurement contract (`measurement_contract=static_pre_append_v1`)

**Primary confirmation receipt:**

`/home/gabe/hrm158_c4s1_phase3_gpu_gate/C4S1_PHASE3_N32_CA_CONFIRMATION_V1/prelaunch/callsite_band_counter_ca_confirmation/callsite_band_counter_ca_confirmation_receipt.json`

**Raw profiler marks (primary B-arm):**

`.../callsite_band_counter_b/host_rss_profile.jsonl:16-19`

---

## 2. Infrastructure — HEALTHY (instrument gate false-negative only)

The recovered receipt shows `infra_ok=false` and `ok=false`. This is **NOT** a real infra/liveness failure. It is an **instrument-gate false-negative** from the pre-fold-2d-a mark-count mis-spec:

- Stale check: `checks.s1d7_band_counter_mark_count_eq_n_states=false` (required `mark_count==n_states==32`)
- Actual delivery: `mark_count=4` matching `len(sampled_states)=4` under the 4-point sampler
- **Correction (fold-2d-a, commit `4c355f7`):** split axes → `eligible_module_limit_eq_n_states` + `s1d7_band_counter_mark_count_eq_sampled_state_count`

**Genuine infra signals (all clean):**

| Field | Value |
|-------|-------|
| `checks.observer_guard_clear` | `true` |
| `checks.tracemalloc_mark_count_eq_0` | `true` |
| `checks.tracemalloc_perturbed_false` | `true` |
| `checks.infra_not_null` | `true` |
| `checks.no_profile_env_mutual_exclusion_abort` | `true` |
| `checks.b_profile_mark_count_gt_0` | `true` |
| `s1d7_tracemalloc_mark_count` | `0` |
| `runs.A.exit_code` / `runs.B.exit_code` | `0` (no timeout) |
| `runs.A/B.subprocess_timeout_expired` | `false` |

**Withdrawn misread:** guard-ENTER telemetry was previously misclassified as a liveness kill. Runs completed with full per-state band rows captured.

**Total wall (primary):** `total_wall_seconds≈350.0` (`runs.A≈205.0s`, `runs.B≈145.0s`).

---

## 3. Science result — INSUFFICIENT_CB_STATES (genuine, not instrument breakage)

| Field | Value |
|-------|-------|
| `terminal_branch` | `INSUFFICIENT_CB_STATES` |
| `classifier.terminal_branch` | `INSUFFICIENT_CB_STATES` |
| `cb_state_count` | `1` |
| `crossing_weighted_ca_share` | `null` (W uncomputable — need ≥2 CB states) |
| `crossing_weighted_ca_share_ok` | `null` |
| `per_cb_ca_share_ok` | `null` |

W/P classifier branches require `cb_state_count >= 2`. With only one crossing-bearing state, the terminal branch is correctly `INSUFFICIENT_CB_STATES` — a **genuine science result**, not infra failure.

### 3.1 Per-state bands (primary n=32)

| State | `is_crossing_bearing` | Bands (A/C/E) | Crossings | `per_cb_ca_share` |
|-------|----------------------|---------------|-----------|-------------------|
| 0 | `true` | 22640 / 48640 / 5408 | 512 | **0.929** (~93.0%) |
| 10 | `false` | 112 / 0 / 0 | 0 | `null` |
| 21 | `false` | 112 / 0 / 0 | 0 | `null` |
| 31 | `false` | 112 / 0 / 0 | 0 | `null` |

State-0 within-state split matches the reduced_n4 observation (fold-1): C=63.4%, (C+A)=93.0% at the crossing-bearing state.

### 3.2 RSS + B5 fallback

- `peak_rss_gib=10.204` (> 6.5 GiB threshold) → B5 `FEASIBILITY_SUBSAMPLE` fallback fired correctly
- n=16 fallback receipt: `peak_rss_gib=5.699`, `sampled_states=[0,5,10,15]`, same `INSUFFICIENT_CB_STATES` / `cb_state_count=1`
- Fallback total wall ≈155s; infra signals likewise clean (exit 0, observer clear, tracemalloc=0)

**n=16 fallback receipt:**

`.../feasibility_subsample_n16/prelaunch/callsite_band_counter_ca_confirmation/callsite_band_counter_ca_confirmation_receipt.json`

---

## 4. Pattern — state0-only crossing under 4-point sampler

Consistent across reduced scales under the default `{0, n//3, 2n//3, n-1}` sampler:

| Scale | `sampled_states` | CB states | Zero-crossing sampled |
|-------|------------------|-----------|----------------------|
| n=4 | `[0,1,2,3]` | state 0 only | states 1–3 |
| n=16 (fallback) | `[0,5,10,15]` | state 0 only | states 5/10/15 |
| n=32 (primary) | `[0,10,21,31]` | state 0 only | states 10/21/31 |

The 4-point sampler always includes state 0 (which is crossing-bearing) but skips contiguous early states 1–9 at n=32.

---

## 5. OPEN fork — structural vs sampling artifact

Two competing hypotheses remain **unresolved**:

1. **Structural:** only state 0 has S1d.7 crossings at this bank/scale; later states are genuinely zero-crossing.
2. **Sampling artifact:** crossing-bearing states exist in `[1..9]` but the 4-point sampler `{0,10,21,31}` never samples them.

**Decider:** dense contiguous early sample `states [0..9]` at n=32 module scope.

**Prerequisite — FIX-C-PROPER (separate source slice):** `probe.py` must actually **read and thread** a dense `sampled_states` set to the sampler. The fold-2d-a-descoped `HRM_TEXT_158_OBMALLOC_EXPANDED_SAMPLED_STATES` env plumbing was dead (set but never consumed). FIX-C-PROPER advances the `probe.py` baseline and requires its own guard/packet currency refresh.

**After FIX-C-PROPER:** dense-[0..9] GPU run is a **separate +1 launch gate** (test-operator). FIX B (primary-receipt preservation on RSS-only fallback) de-risks that run.

---

## 6. Pre-registered next slices

1. **FIX-C-PROPER** — source slice: wire `probe.py` to consume dense `sampled_states`; regression proving `range(10)` → 10 distinct marks
2. **dense-0-9 launch** — GPU confirmation run after FIX-C-PROPER + packet currency refresh
3. **fold-3** (unchanged) — crossing-bearing-only dominance gate revision (separate from this null)

---

## ANTI-OVERCLAIM (verbatim)

> informative null; NO CA verdict; NO candidate-C resolution; NO reduction eligibility; NOT the ~430MB C4.S1d bank pin; NOT a sub-2 proof.
