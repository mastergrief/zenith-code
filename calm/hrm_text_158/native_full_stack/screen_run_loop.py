"""Thin re-export / orchestration shim for forgetting-mechanism screen (r6c).

Keeps public names stable for the CLI + tests. Owns only the glue that wires:
  screen_model_runtime -> screen_execution_loop -> screen_receipt_output

Lower seams own the real concerns. Bound by PLAN_v9 sha 07a02aff….
"""
from __future__ import annotations

import argparse

import torch

from calm.hrm_text_158.native_full_stack.phase_probe_sets import (
    build_phase1_probe_sets,
)
from calm.hrm_text_158.native_full_stack.screen_execution_loop import (  # noqa: F401
    CLIP,
    TOPK_PER_STEP,
    run_train_loop,
)
from calm.hrm_text_158.native_full_stack.screen_model_runtime import (  # noqa: F401
    EXPECTED_PARENT_SHA256,
    _exact_match_count,
    load_and_patch_runtime,
)
from calm.hrm_text_158.native_full_stack.screen_receipt_output import (  # noqa: F401
    AUTHORITY_DISPATCH,
    COMMIT_SURFACE_FILES,
    PLAN_SHA256,
    assemble_arm_receipt,
    emit_receipt_json,
    run_schema_only,
)


def run_arm_screen(args: argparse.Namespace) -> int:
    """Single-arm / Phase-0 live screen — thin glue over the three owners."""
    if not args.ckpt_path:
        raise SystemExit("--ckpt-path is required unless --schema-only/--aggregate-phase1")

    if args.correctness_smoke:
        args.steps = 1
        args.skip_probes = True

    device = str(args.device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("cuda requested but unavailable")

    rt = load_and_patch_runtime(ckpt_path=str(args.ckpt_path), device=device)
    m = rt["m"]
    tok = rt["tok"]
    eligible = rt["eligible"]
    q_levels = rt["q_levels"]
    frozen_scales = rt["frozen_scales"]
    max_seq_len = rt["max_seq_len"]

    # Probe sets (PLAN_v9) — always construct + pin even if eval skipped.
    probe_sets = build_phase1_probe_sets()
    acq_set = set(probe_sets["acquisition"])
    print(
        f"[forget-mech] probes ready acq={probe_sets['acquisition_n']} "
        f"ret={probe_sets['retention_n']} "
        f"acq_sha={probe_sets['acquisition_selection_sha256'][:12]}…",
        flush=True,
    )

    acq_step0 = ret_step0 = None
    if not args.skip_probes:
        print("[forget-mech] step0 exact-match probes…", flush=True)
        acq_step0 = _exact_match_count(
            m, tok, probe_sets["acquisition"], max_seq_len=max_seq_len, device=device
        )
        ret_step0 = _exact_match_count(
            m, tok, probe_sets["retention"], max_seq_len=max_seq_len, device=device
        )
        print(
            f"[forget-mech] step0 acq={acq_step0}/{probe_sets['acquisition_n']} "
            f"ret={ret_step0}/{probe_sets['retention_n']}",
            flush=True,
        )

    from calm.hrm_text_158.curriculum.exhaustive_supports import (
        build_exhaustive_supports,
    )

    pool = [
        (q, int(e), rung)
        for rung, rows in build_exhaustive_supports().items()
        for q, e in rows
    ]

    loop_out = run_train_loop(
        m=m,
        tok=tok,
        eligible=eligible,
        q_levels=q_levels,
        pool=pool,
        acq_set=acq_set,
        arm=str(args.arm),
        steps=int(args.steps),
        batch=int(args.batch),
        topk=int(args.topk),
        max_seq_len=max_seq_len,
        device=device,
        correctness_smoke=bool(args.correctness_smoke),
    )

    acq_final = ret_final = None
    if not args.skip_probes:
        print("[forget-mech] final exact-match probes…", flush=True)
        acq_final = _exact_match_count(
            m, tok, probe_sets["acquisition"], max_seq_len=max_seq_len, device=device
        )
        ret_final = _exact_match_count(
            m, tok, probe_sets["retention"], max_seq_len=max_seq_len, device=device
        )
        print(
            f"[forget-mech] final acq={acq_final}/{probe_sets['acquisition_n']} "
            f"ret={ret_final}/{probe_sets['retention_n']}",
            flush=True,
        )

    receipt = assemble_arm_receipt(
        args=args,
        device=device,
        sha_before=rt["sha_before"],
        scale_sha_before=rt["scale_sha_before"],
        q_sha_before=rt["q_sha_before"],
        frozen_scales=frozen_scales,
        q_levels=loop_out["q_levels"],
        ckpt_path=str(args.ckpt_path),
        probe_sets=probe_sets,
        acq_step0=acq_step0,
        ret_step0=ret_step0,
        acq_final=acq_final,
        ret_final=ret_final,
        loop_out=loop_out,
    )

    if args.correctness_smoke:
        fails = [k for k, v in receipt["asserts"].items() if not v]
        if fails:
            raise SystemExit(f"correctness-smoke FAILED: {fails}")

    emit_receipt_json(receipt, args.output_json)
    return 0


__all__ = [
    "AUTHORITY_DISPATCH",
    "CLIP",
    "COMMIT_SURFACE_FILES",
    "EXPECTED_PARENT_SHA256",
    "PLAN_SHA256",
    "TOPK_PER_STEP",
    "run_arm_screen",
    "run_schema_only",
]
