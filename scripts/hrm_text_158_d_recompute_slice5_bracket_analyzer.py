#!/usr/bin/env python3
"""Emit Slice-5 offline density bracket receipt from preserved run artifacts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from calm.hrm_text_158.native_full_stack.d_recompute_window_slice5_bracket_analyzer import (
    analyze_slice5_density_bracket_from_run_root,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Offline Slice-5 bracket analyzer over preserved recompute-window logs "
            "(CPU read-only; never enters trainer loop)."
        )
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--classifier-receipt",
        type=Path,
        default=None,
        help="Defaults to {run_root}/classifier_receipt.json",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=None,
        help="Defaults to {run_root}/d_recompute_window_diagnostic/recompute_window_log.jsonl",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Defaults to {run_root}/prelaunch/postrun_input_manifest.json when present",
    )
    parser.add_argument("--json-out", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    receipt = analyze_slice5_density_bracket_from_run_root(
        args.run_root,
        classifier_receipt_path=args.classifier_receipt,
        log_path=args.log_path,
        manifest_path=args.manifest_path,
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"bracket_decision": receipt["bracket_decision"], "json_out": str(args.json_out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
