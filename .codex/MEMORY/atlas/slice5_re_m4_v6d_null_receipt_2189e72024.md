# Slice-5 re-M4 v6d null receipt — run 2189e72024

**Packet:** `v6d_re_m4_slice_b_gpu_cap_seam_smoke` | **Head:** `0feb7bc` | **run_root:** `/home/gabe/claw-code-creditdir/transient_fp_credit/slice5_step2a_re_m4_sparse_authority_gpu_scale_smoke_seed43_43_2189e72024/`

## Before (worker / coarse classifier — DISCARD)

- Worker claimed `event_coded_acc_live_carrier.py` contention — **overclaim** (zero refs in `liveness_stack_dump.txt` on either arm).
- Coarse wording implied neither arm reached 3 steps — **false** for instrumented.
- Terminal `SUBMILESTONE_INSTRUMENTATION_INVALID` masked instrumented outer-timeout story.

## After (corrected synthesis)

| arm | class | evidence |
|---|---|---|
| baseline | `BASELINE_SPARSE_CAP_STEP_STALL` | sparse_cap `phase_complete` ×1; step-2 stall @226s (`sparse_cap_apply`) |
| instrumented | `INSTRUMENTED_OUTER_BOUNDED_STEPS_TIMEOUT_AFTER_MAX_STEPS` | sparse_cap `phase_complete` ×3 (~198/102/92s); outer `bounded_steps` kill @510.9s (`guard_event=resume`) |
| both | `SUBMILESTONE_INSTRUMENTATION_INVALID` (historical) | `cap_selection_cpu_copy.jsonl` absent despite completed sparse_cap phases |

## GPU cap-seam status (PROVISIONAL)

Instrumented cleared 3 full sparse_cap cycles — **necessary-and-partially-sufficient** vs baseline step-2 wall. **Path-unconfirmed:** `cap_selection_cpu_copy.jsonl` absent on v6d; cannot assert `cap_gpu_seam_done` until B-DIAG harness + re-smoke.

**Root cause (B-DIAG):** cap_selection **marker emission** was gated inside the CUDA-only branch (`q_levels.device.type == "cuda"`), but event-coded live states keep `q_levels` CPU-resident (`make_event_coded_live_tensor_state`). B-DIAG moved the emit to a sibling of post_cap_sync/boundary_normalize and path-labels it (`gpu_seam` vs `cpu_resident_reference` / `cpu_reference`). This does **NOT** establish the GPU seam ran in v6d and does **NOT** remove `q_cuda_resident` from the GPU apply condition (`use_gpu_sparse_cap_apply` still requires env gates + CUDA availability + q on device).

## Triage (bounded_steps)

- **(a) too-tight wrapper budget:** PRIMARY for instrumented (3 steps ~504s vs 300s resumed-parent silent guard).
- **(b) classifier ordering:** SECONDARY — baseline-driven `phase_guard_locus=sparse_cap_apply` masked instrumented story.
- **(c) per-step stall (instrumented):** REJECTED — 3 sparse_cap completes.

## Ruled out

- Carrier contention as dominant locus (stack dumps).
- `:2132` sparse ingress — **not confirmed**; no mechanism patch in B-DIAG.

## Next gate

B-DIAG harness landed → diagnostic re-smoke (test-operator, separate `+1 launch`) with ring sampler (G1 non-perturbation pass required before locus attribution).

**Atlas index:** `.codex/MEMORY/atlas/ternary_hybrid_stack_arc.md` (HRM-158 fork; mirrors `.claude/MEMORY/atlas/ternary_hybrid_stack_arc.md`).
