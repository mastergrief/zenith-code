# C4.S1d.7 dense-[0..9] structural-vs-sampling fork resolution spec

**Schema:** `hrm_text_158_measurement_closeout_design_spec/v1`  
**Task:** `1782633464140-b85ec12a`  
**Frozen plan source:** design_proposal `1783245916589` (+1 implement `1783245945994`)  
**Terminal read basis:** on-disk wrapper + primary n=32 + n=16 fallback receipts (dense decider run)

---

## 1. What ran

Dense-[0..9] CA decider at n=32 bank-scale using `orchestrate_ca_confirmation_with_fallback` (packet family `v1_n32_dense_09_ca_decider`, git baseline `58568e2` / committed packet chain ending `257437d`).

- `RUN_ID=C4S1_PHASE3_N32_DENSE_09_CA_DECIDER_V1`
- `eligible_module_limit=32` (primary) / `16` (fallback)
- Dense contiguous sampler: `sampled_states=[0,1,2,3,4,5,6,7,8,9]`
- `mark_count=10` (one band-counter mark per sampled state)
- Env override: `HRM_TEXT_158_OBMALLOC_EXPANDED_SAMPLED_STATES=0,1,2,3,4,5,6,7,8,9`
- Band-counter-only B-arm (`TRACEMALLOC=0`, `PROFILE_S1D7_BAND_COUNTER_ONLY=1`), bootstrap `-B` guard-ordering

**Run root:**

`/home/gabe/hrm158_c4s1_phase3_gpu_gate/C4S1_PHASE3_N32_DENSE_09_CA_DECIDER_V1`

**Wrapper receipt:**

`/home/gabe/hrm158_c4s1_phase3_gpu_gate/C4S1_PHASE3_N32_DENSE_09_CA_DECIDER_V1/prelaunch/ca_confirmation_wrapper_receipt.json`

**Primary n=32 dense confirmation receipt:**

`/home/gabe/hrm158_c4s1_phase3_gpu_gate/C4S1_PHASE3_N32_DENSE_09_CA_DECIDER_V1/prelaunch/callsite_band_counter_ca_confirmation/callsite_band_counter_ca_confirmation_receipt.json`

**Fallback n=16 dense confirmation receipt:**

`/home/gabe/hrm158_c4s1_phase3_gpu_gate/C4S1_PHASE3_N32_DENSE_09_CA_DECIDER_V1/feasibility_subsample_n16/prelaunch/callsite_band_counter_ca_confirmation/callsite_band_counter_ca_confirmation_receipt.json`

**Fold-2d-c anchor (prior open fork):**

`artifacts/measurement_closeout/c4s1d7_n32_ca_confirmation_informative_null_and_sampling_fork_spec.md` §5

---

## 2. Infrastructure — HEALTHY

| Field | Primary n=32 | Fallback n=16 |
|-------|--------------|---------------|
| `infra_ok` | `true` | `true` |
| `ok` | `true` | `true` |
| `checks.observer_guard_clear` | `true` | `true` |
| `checks.tracemalloc_mark_count_eq_0` | `true` | `true` |
| `checks.s1d7_band_counter_mark_count_eq_sampled_state_count` | `true` | `true` |
| `checks.eligible_module_limit_eq_n_states` | `true` | `true` |
| `runs.B.exit_code` | `0` | `0` |
| `runs.B.subprocess_timeout_expired` | `false` | `false` |
| `s1d7_tracemalloc_mark_count` | `0` | `0` |

**Total wall:** primary `≈275.0s`; fallback `≈145.0s` (on-disk `total_wall_seconds`).

---

## 3. Q4 salvage — completed primary fork-readable despite RSS fallback

From wrapper receipt (`ca_confirmation_wrapper_receipt.json`):

| Field | Value |
|-------|-------|
| `primary_dense_fork_readable` | `true` |
| `science_verdict_source` | `fallback` |
| `fallback_trigger.rss_breach` | `true` |
| `fallback_trigger.timeout_breach` | `false` |
| `wrapper.terminal_branch` | `FEASIBILITY_SUBSAMPLE` |

Primary n=32 completed cleanly (`infra_ok=true`, `mark_count=10`, full `per_state` coverage for states 0–9, `runs.B.exit_code=0`) but breached RSS (`peak_rss_gib=10.254` > 6.5 GiB threshold). Post-primary fallback fired per packet contract (slice5:6634–6652; **not** an in-run interrupt). Because `primary_dense_fork_readable=true`, the structural-vs-sampling fork read is taken from the **completed n=32 primary `per_state`**, not discarded as mere `FEASIBILITY_SUBSAMPLE` telemetry. Q4 contract worked as designed.

---

## 4. Fork resolution — REGISTERED structural-vs-sampling fork

**CONCLUSION:** dense [0..9] measurement, together with the existing fold-2d-c anchors, resolves the **REGISTERED** structural-vs-sampling fork as **state0-only / STRUCTURAL** for **THIS** S1d.7 bank-scale decider. The state0-only crossing concentration is genuine structure at the measured scope, not a 4-point-sampler artifact: dense contiguous [0..9] finds states 1–9 zero-crossing in **both** the n=32 primary and the n=16 fallback.

**Resolution status (self-limiting):** `RESOLVED_STRUCTURAL_STATE0_ONLY_DENSE_09_BANKSCALE`

### 4.1 Primary n=32 dense science fields

| Field | Value |
|-------|-------|
| `terminal_branch` | `INSUFFICIENT_CB_STATES` |
| `cb_state_count` | `1` |
| `s1d7_band_counter_mark_count` | `10` |
| `sampled_states` | `[0,1,2,3,4,5,6,7,8,9]` |
| `eligible_module_limit` | `32` |
| `peak_rss_gib` | `10.254` |
| `crossing_weighted_ca_share` | `null` |

### 4.2 Per-state bands — primary n=32 dense [0..9]

| State | `is_crossing_bearing` | Bands (A/C/E) | Crossings | `per_cb_ca_share` |
|-------|----------------------|---------------|-----------|-------------------|
| 0 | `true` | 22640 / 48640 / 5408 | 512 | **0.929** (~92.9%) |
| 1 | `false` | 112 / 0 / 0 | 0 | `null` |
| 2 | `false` | 112 / 0 / 0 | 0 | `null` |
| 3 | `false` | 112 / 0 / 0 | 0 | `null` |
| 4 | `false` | 112 / 0 / 0 | 0 | `null` |
| 5 | `false` | 112 / 0 / 0 | 0 | `null` |
| 6 | `false` | 112 / 0 / 0 | 0 | `null` |
| 7 | `false` | 112 / 0 / 0 | 0 | `null` |
| 8 | `false` | 112 / 0 / 0 | 0 | `null` |
| 9 | `false` | 112 / 0 / 0 | 0 | `null` |

### 4.3 Fallback n=16 dense confirmation

| Field | Value |
|-------|-------|
| `terminal_branch` | `INSUFFICIENT_CB_STATES` |
| `cb_state_count` | `1` |
| `sampled_states` | `[0,1,2,3,4,5,6,7,8,9]` |
| `eligible_module_limit` | `16` |
| `peak_rss_gib` | `5.769` |
| State pattern | state 0 only crossing-bearing; states 1–9 zero-crossing |

Same state0-only pattern at reduced eligible scope confirms the dense early-state read is not an n=32-only artifact.

---

## 5. HARD LIMITS (load-bearing — read before interpreting §4)

These limits are co-equal with the fork-resolution conclusion. Do **not** read §4 broader than this section permits.

### 5.1 W/P remains uncomputable

- `cb_state_count=1` → terminal branch `INSUFFICIENT_CB_STATES` on both primary and fallback.
- `crossing_weighted_ca_share=null`; W/P classifier branches (`CA_PERSISTS`, `CA_MIXED`, `CA_DILUTES`) are **not** reachable.
- This run provides **no** CA-share persistence verdict.

### 5.2 Scope is the PRE-REGISTERED fork only — NOT a universal census

- This resolves the **pre-registered** fold-2d-c structural-vs-sampling fork at the **dense-[0..9] S1d.7 bank-scale decider** scope **only**.
- It is **NOT** an exhaustive all-state proof.
- States **outside** dense `[0..9]` are **NOT** newly ruled out.
- This is **NOT** a universal crossing census across all module states.

### 5.3 NOT implementation readiness

- Fork resolution is a measurement/decider outcome, not authorization to implement reduction, candidate-C resolution, or bank pinning.

---

## 6. Pre-registered next

**fold-3** — crossing-bearing-only dominance-gate design, informed by this structural result.

---

## ANTI-OVERCLAIM (verbatim)

> decider precondition; NO CA verdict; NO candidate-C resolution; NO reduction eligibility; NOT the ~430MB C4.S1d bank pin; NOT a sub-2 proof.
