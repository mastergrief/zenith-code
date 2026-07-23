#!/usr/bin/env python3
"""Thin CLI — censor_null_pressure_metric_diagnostic (PLAN_v6 rev4).

Argparse + wiring only.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calm.hrm_text_158.native_full_stack.pressure_metric_benchmark import (  # noqa: E402
    run_paired_benchmark,
)
from calm.hrm_text_158.native_full_stack.pressure_metric_proof import (  # noqa: E402
    emit_json,
    run_formal_diagnostic,
)
from calm.hrm_text_158.native_full_stack.pressure_metric_telemetry import (  # noqa: E402
    AUTHORITY_DISPATCH,
    FORMAL_BATCH,
    FORMAL_STEPS,
    PAIRED_N,
    PARENT_SHA256,
    PLAN_SHA256,
)
from calm.hrm_text_158.native_full_stack.screen_execution_loop import (  # noqa: E402
    TOPK_PER_STEP,
)


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_schema_only(args: argparse.Namespace) -> int:
    receipt = {
        "screen": "censor_null_pressure_metric_diagnostic/v1",
        "plan_sha256": PLAN_SHA256,
        "authority_dispatch": AUTHORITY_DISPATCH,
        "schema_only": True,
        "cli": {
            "steps": int(args.steps),
            "batch": int(args.batch),
            "topk": int(args.topk),
            "device": str(args.device),
            "telemetry": bool(args.telemetry),
        },
        "expected_parent_sha256": PARENT_SHA256,
        "limits": [
            "schema-only — no science",
            "formal GPU paired-25/150 = claude/test-operator",
        ],
        "seam_map": {
            "telemetry": "pressure_metric_telemetry",
            "lifecycle": "pressure_metric_lifecycle",
            "classifier": "pressure_metric_classifier",
            "readiness": "pressure_metric_readiness",
            "receipt": "pressure_metric_receipt",
            "warmup_runtime": "pressure_metric_warmup_runtime",
            "proof": "pressure_metric_proof",
            "benchmark": "pressure_metric_benchmark",
            "cli": "hrm_text_158_censor_null_pressure_metric_diagnostic.py",
        },
    }
    emit_json(receipt, args.output_json)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="PLAN_v6 censor_null_pressure_metric_diagnostic thin CLI"
    )
    ap.add_argument("--ckpt-path", default=None)
    ap.add_argument("--steps", type=int, default=FORMAL_STEPS)
    ap.add_argument("--batch", type=int, default=FORMAL_BATCH)
    ap.add_argument("--topk", type=int, default=TOPK_PER_STEP)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--output-json", default=None)
    ap.add_argument("--schema-only", action="store_true")
    ap.add_argument("--skip-probes", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--telemetry",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    ap.add_argument("--paired-benchmark", action="store_true")
    ap.add_argument("--paired-n", type=int, default=PAIRED_N)
    ap.add_argument("--uninstrumented-25-json", default=None)
    ap.add_argument("--instrumented-25-json", default=None)
    ap.add_argument("--paired-timing-json", default=None)
    ap.add_argument("--paired-proof-json", default=None)
    ap.add_argument("--paired-proof-sha256", default=None)
    ap.add_argument(
        "--diagnostic-override",
        action="store_true",
        help="Allow CPU / non-prereg geometry as NON-PROOF only.",
    )
    args = ap.parse_args()
    if args.schema_only:
        return run_schema_only(args)
    if args.paired_benchmark:
        if not args.ckpt_path:
            raise SystemExit("--ckpt-path required for --paired-benchmark")
        return run_paired_benchmark(
            ckpt_path=str(args.ckpt_path),
            device=str(args.device),
            batch=int(args.batch),
            topk=int(args.topk),
            seed=int(args.seed),
            paired_n=int(args.paired_n),
            repo_root=_repo_root(),
            uninstrumented_25_json=args.uninstrumented_25_json,
            instrumented_25_json=args.instrumented_25_json,
            paired_timing_json=args.paired_timing_json,
            output_json=args.output_json,
            diagnostic_override=bool(args.diagnostic_override),
        )
    if not args.ckpt_path:
        raise SystemExit("--ckpt-path required unless --schema-only")
    return run_formal_diagnostic(
        ckpt_path=str(args.ckpt_path),
        device=str(args.device),
        steps=int(args.steps),
        batch=int(args.batch),
        topk=int(args.topk),
        seed=int(args.seed),
        telemetry=bool(args.telemetry),
        skip_probes=bool(args.skip_probes),
        paired_proof_json=str(args.paired_proof_json or ""),
        paired_proof_sha256=str(args.paired_proof_sha256 or ""),
        repo_root=_repo_root(),
        output_json=args.output_json,
    )


if __name__ == "__main__":
    raise SystemExit(main())
