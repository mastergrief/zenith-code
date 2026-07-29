#!/usr/bin/env python3
"""Thin LANDS-AB evaluation runner (PLAN_v6 / IMPLEMENT_v2).

CPU characterization + diagnostic reducer smoke. Formal GPU matrix is packet-gated later.
Science path is evidence-bound (raw observations); reducer-smoke is synthetic_only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="LANDS-AB eval thin runner")
    ap.add_argument(
        "--mode",
        choices=("cpu-static-ab", "cpu-s3-char", "reducer-smoke"),
        default="cpu-static-ab",
    )
    ap.add_argument("--seed", type=int, default=158)
    ap.add_argument("--dim", type=int, default=4)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args(argv)

    if args.mode == "reducer-smoke":
        # Diagnostic stdout-only; never silently ignore an explicit --out.
        if args.out:
            print(
                "error: reducer-smoke does not write --out "
                "(stdout-only synthetic diagnostic; use cpu-static-ab/cpu-s3-char for O_EXCL writes)",
                file=sys.stderr,
            )
            return 2
        from calm.hrm_text_158.native_full_stack.lands_ab_eval_branch_reducer import (
            all_true_matrix,
            reduce_lands_ab_branch_strict,
        )
        from calm.hrm_text_158.native_full_stack.lands_ab_eval_schema import (
            CLAIM_CEILING,
            DIAGNOSTIC_RECEIPT_SCHEMA,
        )

        out = reduce_lands_ab_branch_strict(
            {
                "scope_creep": False,
                "fixture_contract_raw_fail": False,
                "surface_pass_by_row": all_true_matrix(),
            }
        )
        payload = {
            "schema": DIAGNOSTIC_RECEIPT_SCHEMA,
            "synthetic_only": True,
            "science_claim": False,
            "claim_ceiling": dict(CLAIM_CEILING),
            "branch_id": out["branch_id"],
            "ok": out["ok"],
            "reason_codes": out["reason_codes"],
            "caveat": "synthetic all_true_matrix diagnostic; NOT evidence-bound; cannot mint LANDS-AB",
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if args.mode == "cpu-s3-char":
        from calm.hrm_text_158.native_full_stack.lands_ab_eval_measurement import (
            run_s3_apply_equivalence_cpu_tiny_diagnostic,
        )
        from calm.hrm_text_158.native_full_stack.lands_ab_eval_runtime_io import (
            o_excl_write_text,
        )

        result = run_s3_apply_equivalence_cpu_tiny_diagnostic(seed=args.seed, dim=args.dim)
        result = dict(result)
        result["synthetic_only"] = True
        result["science_claim"] = False
        result["diagnostic_only"] = True
        text = json.dumps(result, indent=2, sort_keys=True, default=str)
        print(text)
        if args.out:
            o_excl_write_text(Path(args.out), text)
        return 0

    # default: evidence-bound G_CPU_STATIC_AB
    from calm.hrm_text_158.native_full_stack.lands_ab_eval_measurement import (
        measure_g_cpu_static_ab,
    )
    from calm.hrm_text_158.native_full_stack.lands_ab_eval_runtime_io import (
        o_excl_write_text,
        resolve_run_scratch_dir,
        runtime_scratch_raw_path,
    )
    import uuid

    result = measure_g_cpu_static_ab()
    text = json.dumps(result, indent=2, sort_keys=True, default=str)
    print(text)
    if args.out:
        o_excl_write_text(Path(args.out), text)
    else:
        # formal default: O_EXCL raw obs under run root (LANDS_AB_RUN_ROOT or unique scratch)
        root = resolve_run_scratch_dir(create=True)
        path = runtime_scratch_raw_path(
            scratch_dir=root, gating_row="G_CPU_STATIC_AB", run_nonce=uuid.uuid4().hex[:12]
        )
        o_excl_write_text(path, text)
        print(json.dumps({"raw_obs_path": str(path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
