# Slice-5 re-M4 v6e null receipt — run 2189e72025

**Packet:** `v6e_re_m4_slice_b_diag_diagnostic_smoke` | **Head:** `80d5179` | **run_root:** `/home/gabe/claw-code-creditdir/transient_fp_credit/slice5_step2a_re_m4_sparse_authority_gpu_scale_smoke_seed43_43_2189e72025/`

## Before (worker / stale postrun receipts — DISCARD)

- Worker timing prose and stale mid-run postrun receipts treated partial drains as final.
- Stale `cap_selection_path_evidence_receipt.json`: instrumented `sparse_cap_complete=0`, route `SUBMILESTONE_INSTRUMENTATION_INVALID`.
- Stale `bounded_steps_triage_receipt.json`: `PER_STEP_STALL_CONFIRMED`, instrumented `steps_completed=0` @8s.
- Ring sampler G1 prelaunch **false-green**: `stack_text` contained `AttributeError: fileno` from broken `faulthandler.dump_traceback` capture; baseline durable ring rows **empty** after kill.

## After (corrected synthesis — FINAL drained truth)

| arm | class | evidence |
|---|---|---|
| **both** | GPU seam **unreached** (marker-backed) | `cap_reference_cpu_resident_done` × completed sparse_cap steps on both arms; `use_gpu_sparse_cap_apply=false` because event-coded `q_levels` stay CPU-resident |
| **baseline** | `BASELINE_SPARSE_CAP_STEP_STALL` | sparse_cap `phase_complete` ×2; stalled **entering** step-3 `sparse_cap_apply` @248.9s (`guard_event=enter`, `step=3`) |
| **instrumented** | `INSTRUMENTED_OUTER_BOUNDED_STEPS_TIMEOUT_AFTER_MAX_STEPS` | sparse_cap + `live_carrier_snapshot_emit` + `step_forward_backward` all show steps 1/2/3 `phase_complete`; stalled on **outer** `bounded_steps` wrapper @598.1s (`guard_event=resume`, `step=None`) = `WRAPPER_BUDGET_TOO_TIGHT` |

**NOT a snapshot-emit stall:** instrumented completed all 3 steps including `live_carrier_snapshot_emit` before the outer `bounded_steps` timeout.

## Path evidence (marker-backed)

- Both arms with completed sparse_cap phases: `PATH_CPU_RESIDENT_CAP_REFERENCE` (`cap_reference_cpu_resident_done` in jsonl).
- **NOT** `PATH_GPU_SEAM_EXERCISED` — no `cap_gpu_seam_done` markers.
- GPU-seam-validated claim is **blocked** until a re-smoke shows `cap_gpu_seam_done` with `q_cuda_resident` satisfied.

## Diagnostic defects (B-DIAG2 scope)

1. **POSTRUN RACE:** path-evidence/triage/classifier ran before arms drained → stale receipts. Repair via barrier + superseding replay against final artifacts.
2. **RING SAMPLER FALSE-GREEN:** exception traceback recorded as `stack_text`; durable flush only on clean `_exit_phase_stack` → 0 baseline rows after faulthandler kill. Stack-locus attribution **not achieved**.

## Ruled out

- Worker "carrier contention" claim — zero refs in stack dumps.
- `:2132` sparse ingress — **not confirmed**; mechanism fork A/B deferred.

## Next gate

B-DIAG2 diagnostic-correctness harness → fixed-diagnostic re-smoke (v6f, test-operator) to capture real stall locus. Mechanism fork (q-residency vs snapshot-path) waits for repaired-receipt/locus gate.

**Atlas index:** `.codex/MEMORY/atlas/ternary_hybrid_stack_arc.md`
