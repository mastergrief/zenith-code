#!/usr/bin/env python3
"""Thin CLI for the transient selection information audit harness."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from calm.hrm_text_158.native_full_stack.transient_selection_information_audit import (
    build_transient_selection_information_audit,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Emit the CPU read-only transient_selection_information_audit_v0 "
            "receipt over a B2b stable-copy trace with B2c integrity pins."
        )
    )
    parser.add_argument("--stable-trace", type=Path, required=True)
    parser.add_argument("--original-trace", type=Path, required=True)
    parser.add_argument("--capture-receipt", type=Path, required=True)
    parser.add_argument("--b2c-receipt", type=Path, required=True)
    parser.add_argument("--stable-copy-dir", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--expected-stable-trace-sha256")
    parser.add_argument("--expected-original-trace-sha256")
    parser.add_argument("--expected-capture-receipt-sha256")
    parser.add_argument("--expected-b2c-receipt-sha256")
    parser.add_argument("--rate-cap", type=int, default=1)
    parser.add_argument("--indent", type=int, default=2)
    args = parser.parse_args(argv)

    expected_shas = {
        key: value
        for key, value in {
            "stable_trace": args.expected_stable_trace_sha256,
            "original_trace": args.expected_original_trace_sha256,
            "capture_receipt": args.expected_capture_receipt_sha256,
            "b2c_receipt": args.expected_b2c_receipt_sha256,
        }.items()
        if value
    }

    try:
        receipt = build_transient_selection_information_audit(
            stable_trace_path=args.stable_trace,
            original_trace_path=args.original_trace,
            capture_receipt_path=args.capture_receipt,
            b2c_receipt_path=args.b2c_receipt,
            expected_shas=expected_shas or None,
            stable_copy_dir=args.stable_copy_dir,
            rate_cap=args.rate_cap,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        sys.stderr.write(f"hard failure: {exc}\n")
        return 2

    encoded = json.dumps(receipt, indent=args.indent, sort_keys=True) + "\n"
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(encoded, encoding="utf-8")
    sys.stdout.write(encoded)
    summary = {
        "primary_label": receipt.get("primary_label"),
        "failure_reasons": receipt.get("failure_reasons", []),
    }
    sys.stderr.write(json.dumps(summary, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
