# Ternary-Hybrid Training Stack — receipts arc (HRM-158 fork)

Query-triggered receipts for the native HRM-Text-1.58 / Slice-5 sparse-authority
GPU liveness lane. Canonical training code and checkpoints live on
`feature/hrm-text-1.58` in this repo; cross-repo science tree:
`/home/gabe/claw-code-creditdir/transient_fp_credit/`.

Main-repo mirror (broader ternary-hybrid arc): `zenith-code` →
`.codex/MEMORY/atlas/ternary_hybrid_stack_arc.md`.

## Per-run / per-mechanism receipts

### 2026-06-30 — Slice-5 re-M4 v6d null (run 2189e72024)

- **Receipt:** `.codex/MEMORY/atlas/slice5_re_m4_v6d_null_receipt_2189e72024.md`
- **Packet:** `v6d_re_m4_slice_b_gpu_cap_seam_smoke` | **Head:** `0feb7bc`
- **Verdict:** `BASELINE_SPARSE_CAP_STEP_STALL` + `INSTRUMENTED_OUTER_BOUNDED_STEPS_TIMEOUT_AFTER_MAX_STEPS` + historical `SUBMILESTONE_INSTRUMENTATION_INVALID`
- **GPU cap-seam:** path-unconfirmed on v6d (`cap_selection_cpu_copy.jsonl` absent); instrumented 3-step completion is necessary-and-partially-sufficient vs baseline step-2 wall
- **Next:** B-DIAG diagnostic harness → re-smoke with ring sampler (G1 non-perturbation gate required before locus attribution)
