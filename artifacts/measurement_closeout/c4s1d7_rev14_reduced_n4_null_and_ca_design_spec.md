# C4.S1d.7 rev14 reduced_n4 informative null + (C+A) reduction-design spec

**Schema:** `hrm_text_158_measurement_closeout_design_spec/v1`  
**Task:** `1783074444338-d3ef40af` (parent Arc#2b `1782633464140-b85ec12a`)  
**Frozen plan source:** design_proposal `1783184921285` (dual-accept gate-1 `1783184974588` + co_lead gate-2 `1783185252506`)  
**Terminal read basis:** gate-1 `1783184087626` + co_lead gate-2 `1783184281293`

---

## 1. Informative null (shippable)

### 1.1 Infrastructure — feasible

rev14 guard-ordering **CONFIRMED**:

- `s1d7_band_counter_mark_count=4`
- `observer_guard_clear`
- `tracemalloc_mark_count=0`
- `guard_ran_before_pinned_imports=true` (rev14 bootstrap guard-ordering)
- lane self-managed
- ~35s wall (reduced n=4 band-counter-only B-arm class post static-pre-append)

**Primary smoke receipt:**

`/home/gabe/hrm158_c4s1_phase3_gpu_gate/C4S1_PHASE3_GPU_CALLSITE_V1/prelaunch/callsite_band_counter_scale_smoke/callsite_band_counter_scale_smoke_receipt.json`

### 1.2 Science — state-0 within-state split (crossing-bearing)

At the mass-bearing crossing state (state 0):

| Band | Bytes | Share |
|------|------:|------:|
| C (journal/encode) | 48640 | **63.4%** |
| A (crossing list-comp) | 22640 | 29.5% |
| E (numpy remove/upd) | 5408 | 7.1% |
| **Total** | **76688** | — |

Derived metrics (state 0 only):

- `(C+A)/(A+C+E) = 93.0%`
- `C/max(A,E) = 48640/22640 ≈ 2.15` (< 3×)
- **C-only dominance REFUTED** vs `DOMINANCE_C_SHARE_MIN=0.80` (`calm/hrm_text_158/native_full_stack/s1d7_band_counter.py:26`)
- `call_site_status=UNRESOLVED`, `s1d7_call_site_candidate=null`

**Physical seam** (within-state-0, crossing-bearing):

`calm/hrm_text_158/native_full_stack/event_coded_acc_live_carrier.py:909-952`

- Band A: line 910 (crossing_indices list-comp)
- Band C: lines 941-952 (`_append_event` journal)
- Band E: lines 914-917 (numpy remove/upd arrays)
- Logical marker: `:895`; candidate metadata: `:896`

**Crossing count (state 0):** `crossing_indices_len=append_event_count=512`

**Source:** raw profiler mark state 0 @  
`/home/gabe/hrm158_c4s1_phase3_gpu_gate/C4S1_PHASE3_GPU_CALLSITE_V1/prelaunch/callsite_band_counter_scale_smoke/callsite_band_counter_b/host_rss_profile.jsonl:16`  
(`s1d7_band_counters.counts.crossing_indices_len` and `append_event_count`)

### 1.3 Cross-state picture (profiler-grounded)

**Raw band-counter marks:** `host_rss_profile.jsonl:16-19` (4 marks → `s1d7_band_counter_mark_count=4`)

| State | Profile line | Bands | Crossings | Role |
|-------|-------------|-------|-----------|------|
| 0 | L16 | C=48640 / A=22640 / E=5408 | 512 | Crossing-bearing |
| 1 | L17 | A=112 / C=0 / E=0 | 0 | Measured zero-crossing |
| 2 | L18 | A=112 / C=0 / E=0 | 0 | Measured zero-crossing |
| 3 | L19 | A=112 / C=0 / E=0 | 0 | Measured zero-crossing |

**Dominance table truncation:** receipt `s1d7_band_counter_dominance.per_state` contains **2 rows** (states 0 and 1 only). Evaluation fail-closed at state 1 with `BAND_COUNTER_C_NOT_TOP_IN_STATE` (`s1d7_band_counter.py:279-288`). States 2–3 raw marks exist in the profiler but never enter the dominance table.

**Interpretation:** n=4 sampled all 4 states — **1 crossing-bearing + 3 measured zero-crossing**. Design direction rests on the one crossing-bearing state's within-state `(C+A)=93.0%`.

---

## 2. (C+A) reduction-design spec — PRE-REGISTER ONLY

**Not implemented.** Design target for a future reduction fold:

- **Primary bracket:** S1d.7 crossing+commit composite **bands A+C** (physical A910 + C941-952), with E tracked as residual/nuisance
- **Acceptance sketch (design only):** `(C+A)/(A+C+E) >= 0.80` on **crossing-bearing states** (`crossing_indices_len > 0`); E bounded separately or `C >= 3×E`
- **Rejected:** another GPU round mass-gating the same C-only `>=0.80` single-band acceptance bar

Rationale: alloc at the crossing-bearing state is an **A+C split** (+ small E), not C-monolithic. No valid single-band C gate rescues state 0 (63.4% < 80%).

---

## 3. n=32 / 430MB-scale confirmation — implementation precondition

Any reduction **implementation** is blocked until full eligible scope (n=32 states) band-counter confirmation at 430MB-scale bank.

- reduced_n4 does **NOT** pin the 430MB bank
- reduced_n4 does **NOT** grant reduction eligibility
- n=32 is required to sample **more crossing-bearing states** beyond the single state-0 observation at n=4

---

## 4. Smoke-gate confound caveat + future fix

**Observation:** Per-state C-top gate (`s1d7_band_counter.py:279-288`) fails on non-crossing states. At n=4, **3 measured zero-crossing states** (states 1–3: C=0, 112B A-residual, 0 crossings) cause `BAND_COUNTER_C_NOT_TOP_IN_STATE` before aggregate dominance math.

**Future fix (separate post-n=32 slice):** dominance evaluation scoped to `crossing_indices_len > 0` states only. Evidence-backed by the 3 measured zero-crossing states at n=4 — not hypothetical.

**Explicitly NOT proposed:** relaxing `DOMINANCE_C_SHARE_MIN=0.80` or skipping per-state checks.

---

## 5. Dilution risk (named)

If n=32 is mostly zero-crossing (as 3/4 states at n=4), crossing mass concentrates in few states. Aggregate cross-state share metrics need **crossing-weighting** in the n=32 confirmation design.

---

## 6. Sequencing (3-fold)

1. **Fold-1 (this artifact):** persist committed informative null + (C+A) design spec — no reduction claim
2. **Fold-2:** n=32 confirmation packet (test-operator GPU; crossing-bearing-state metrics)
3. **Fold-3:** packet-family gate revision (crossing-bearing-only dominance — separate slice post n=32)

---

## ANTI-OVERCLAIM (verbatim)

> reduced n=4 mechanism-local only; NOT a candidate-C resolution; NOT reduction eligibility; NOT the ~430MB C4.S1d bank pin; NOT a sub-2 proof.
