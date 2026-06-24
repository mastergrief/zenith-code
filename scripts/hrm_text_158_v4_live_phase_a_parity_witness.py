#!/usr/bin/env python3
"""CLI harness for V4-LIVE Phase-A GPU-vs-CPU parity witness."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from calm.hrm_text_158.native_full_stack.v4_live_phase_a_parity_witness import (
    run_phase_a_parity_witness,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run deterministic V4-LIVE Phase-A parity witness (CPU, read-only)."
    )
    parser.add_argument(
        "--phase-root",
        required=True,
        help="Phase-A scratch root containing receipt.json and votes_emit/v1/.",
    )
    parser.add_argument(
        "--json-out",
        required=True,
        help="Output path for binary verdict JSON.",
    )
    parser.add_argument(
        "--demotion-band",
        type=int,
        default=1,
        help="Demotion band used by the Phase-A GPU run (default 1).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional exclusive upper bound on compared step indices.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    verdict = run_phase_a_parity_witness(
        Path(args.phase_root),
        demotion_band=int(args.demotion_band),
        max_steps=args.max_steps,
    )
    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(verdict, indent=2, sort_keys=True), flush=True)
    return 0 if bool(verdict.get("phase_a_parity_pass")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
