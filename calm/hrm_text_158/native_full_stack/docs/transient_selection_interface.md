# Transient-Selection Interface Spec

**Frozen source:** R1 board msg `1781122261352` with §2 replaced by R2 board msg `1781122431786`.

**Evidence receipts:** acc_width `3e3157af6857b91adc2578449fbd0c19ebc24c6f87bfbdbd28958757ae8389ef` · M2a-v1 `a0c32fc3a77f3810a1772f9bd01a1a0b5455261d7b91f3951034109e80bdbf8c`

**Dual verdict msg ids:** `1781121867675` (claude) · `1781122039865` (codex_co_lead)

**Convergence msg ids:** `1781122459953` (claude) · `1781122495369` (codex_co_lead)

**Claim boundary:** single-trace (`cb373de78030c5a9`); design-only; BUILD gated on M2b second-capture confirm.

**Parent design:** converged two-tier design R1 `1781116742252` §1 CARRY⇄SELECTION boundary.

---

## 0. Claim boundary

Single-trace (cb373de78030c5a9); design-only; saturation hypothesis-grade. BUILD = carry hold + M2b confirm.

## 1. Normative receipt constraints

Unchanged: acc_width 3e3157af… (w_min=6, W6 crossing 0 mismatches, drift 838 non-blocking, headroom 2×) + M2a-v1 a0c32fc3… (row 4; F1 0.32 / F2 0.9504 / F3 0.68).

## 2. CAPTURE-RECORDED fields (auditable — gate 1781122148824)

### 2A. Per-step record (required)
| Field | Type | Notes |
|-------|------|-------|
| `optimizer_step_index` | int | step identity |
| `warmup_apply_class` | enum `canonical` \| `subthreshold_bootstrap` | new contract field |
| `effective_apply_threshold_abs` | int \| null | required when `subthreshold_bootstrap` |
| `applied_flip_count` | int k | step telemetry |
| `sampled_candidate_table` | array[32] | full row payloads every step |

### 2B. Per-row fields inside `sampled_candidate_table` (required)
| Field | Capture-recorded | Persistent carry? | Notes |
|-------|------------------|-------------------|-------|
| `flat_index` | yes (all rows) | no | identity key |
| `pre_accumulator_i16` | yes (all rows) | **yes (all rows)** | W6-class carry state; **carry self-update law (§4.1) applies every row every step** — NOT applied-index-only |
| `new_acc_i32_signed` | yes (all rows) | no | transient recompute / step telemetry |
| `vote_value` | yes (all rows) | no | step-local input to carry law |
| `proposal_direction` | yes (all rows) | no | step-local |
| `current_q_level` | yes (all rows) | **yes (applied indices only)** | selection write-back commits q flip at applied flat_index |
| `in_target_tie_band` | yes (all rows) | no | recorded echo; `band_membership_scope` |
| `threshold_residual_signed` | yes (all rows) | **no** | telemetry only; derivable; not persisted (§4.5) |
| `proximity_to_threshold` | yes (all rows) | **no** | telemetry only; derivable; not persisted (§4.5) |

### 2C. Capture-receipt `threshold_semantics` block (required)
```yaml
threshold_semantics:
  crossing_threshold_abs: 10
  crossing_threshold_source: canonical_default_spec_accumulator_real_dynamics_verdict
  crossing_authority: vote_update_spec
  residual_band_encoding: threshold_minus_one
  row_fields_authority: telemetry_not_crossing
  row_crosscheck_policy: informational
```

### 2D. Warmup diagnostic block (required when present)
Per M2a-v1 `warmup_subthreshold_applies` entry shape (step, k, crossing_count, applied row detail, recompute_disagreements).

### 2E. Audit-resolution rule
`threshold_row_derivation_mismatch` resolved when §2C present + `row_crosscheck_policy=informational` + row fields captured per §2B. Blocking only if capture omits §2C or declares row fields crossing-authoritative without new prereg.

## 3. Two-tier partition

| Tier | Width | Role |
|------|-------|------|
| CARRY | W6 primary (±31); W8 fallback | sparse persistent state + per-step self-update law |
| SELECTION | W16 transient | cap/rank/argmax/rate-cap — never persistent |

## 4. CARRY⇄SELECTION boundary (B2 + S1 fixes)

### 4.1 CARRY tier self-update (ALL rows, every step) — **B2 fix**

The carry tier applies its own update law **independently of selection** to every row's accumulator each optimizer step:

```
pre_acc_carry' = decay_vote_clamp(pre_acc_carry, vote_value, clip_W6)
```

This is NOT selection write-back. Non-applied rows (31/32 typical) still evolve under carry law. Selection does not gate whether carry self-updates run.

### 4.2 SELECTION → CARRY write-back (applied set ONLY) — **B2 fix**

Selection's persistent authority is limited to **applied-flip events**:
- Confirm `applied_crossing_direction` (sign)
- Write `post_flip_residual` per §4.4 encoding (direction + magnitude)
- Commit `q_level` flip at applied flat_index
- Optionally reset/sync `pre_accumulator_carry` at applied index to the post-apply carry value **after** the applied flip is accepted (this is the selection-gated residual write, not the universal decay path)

**Forbidden reading:** "only applied rows' accumulators update" — rejected.

### 4.3 CARRY → SELECTION inputs (read-only, per step) — **S1 fix**

**Normative crossing authority = CARRY-WIDTH crossing bool** (`crossing_bool_w6` under `crosses_threshold` at `threshold_abs_crossing=10` with W6 effective clip).

W16-reference crossing equivalence is an **audited property**, not a normative runtime input:
- Trace-1: acc_width proved 0 W6-vs-W16 crossing mismatches
- Trace-2 (M2b F5): re-test `crossing_mismatch_count_vs_w16 == 0` on new capture

Selection transient lane may recompute W16 `new_acc` for F2 rank audit, but persistent carry decisions use W6 crossing membership.

### 4.4 Persistent field encoding — **B1 fix**

| Field | Encoding | Total bits | Range / notes |
|-------|----------|------------|---------------|
| `pre_accumulator_carry` | signed int at W6 clip | **6-bit effective** (±31); int8 container ok | primary carry state |
| `crossing_membership` | bool | 1 | per applied event |
| `applied_crossing_direction` | sign bit | 1 | applied flip direction |
| `post_flip_residual` | **direction bit + 4-bit magnitude** | **5 bits** | magnitude holds \|residual\| ∈ [0,9] at T=10 clamp band [-(T-1),+(T-1)]; 4-bit mag holds 0..15 ≥ 9. Mode: `applied_crossing_direction_plus_4bit_residual` |
| `current_q_level` | int8 | 8 | at applied indices |
| W8 fallback accumulator | 8-bit signed | 8 | headroom fallback only |

**Rejected:** "4-bit signed class" for post_flip_residual — arithmetic error (±9 does not fit 4-bit signed -8..+7).

### 4.5 DROPPED from persistent ledger (v2 correction, retained)

`threshold_residual_signed`, `proximity_to_threshold`: capture-recorded telemetry only; derivable from acc + update law; not persisted.

### 4.6 NEVER crosses (transient-only)

Rank order (F2), cap top-k (F1), argmax abs (F3), rate-cap queue, full 32-row tables as persistent authority.

## 5. Threshold PIN (accepted per review ask #1)

Dual-attestation unchanged: `crossing_threshold_abs=10` authoritative; `residual_band_encoding=threshold_minus_one`; `row_crosscheck_policy=informational`; capture_must_emit `threshold_semantics` block (§2C).

## 6. Warmup policy (accepted per review ask #2)

`record_only_excluded_from_falsifier_gates` — unchanged from v2.

## 7. Saturation — hypothesis-grade only

## 8. R1 change log

| ID | Fix |
|----|-----|
| B1 | post_flip_residual → direction + 4-bit magnitude (5 bits); honest range ±9 |
| B2 | Carry self-update ALL rows vs selection write-back applied-only |
| S1 | Crossing authority = W6 bool; W16 equivalence = audited property |
| v2 | Capture-recorded inventory + residual drop from persistent ledger retained |
| loc | Docs path `calm/hrm_text_158/native_full_stack/docs/transient_selection_interface.md` |

## 9. Review ask

Claude: B1+B2+S1 addressed — ready for R1 accept?
codex_co_lead: verify §2 auditable-fields gate + B1 ledger honesty.

On dual accept → freeze spec text → unpark M2b packet 1781122090589.
