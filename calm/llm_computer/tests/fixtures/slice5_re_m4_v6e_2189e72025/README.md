# Fixture: v6e run 2189e72025 (FINAL drained truth)

Copied from run_root before any cleanup. Used for B-DIAG2 replay-repair tests.

## True-final classification (NOT stale mid-run receipts)

- **Both arms:** `PATH_CPU_RESIDENT_CAP_REFERENCE` (`cap_reference_cpu_resident_done` markers)
- **Baseline:** sparse_cap complete ×2; stalled entering step-3 `sparse_cap_apply` @248.9s
- **Instrumented:** all 3 steps complete (sparse_cap + snapshot_emit + forward_backward); outer `bounded_steps` @598.1s
- **Triage:** `WRAPPER_BUDGET_TOO_TIGHT` + split-arm guards
- **Classifier:** `INSTRUMENTED_OUTER_BOUNDED_STEPS_TIMEOUT_AFTER_MAX_STEPS`

## Stale receipts in `prelaunch/` (repair targets, NOT authority)

- `cap_selection_path_evidence_receipt.json` — mid-run race (instrumented sparse_cap_complete=0)
- `bounded_steps_triage_receipt.json` — mid-run race (PER_STEP_STALL_CONFIRMED)
