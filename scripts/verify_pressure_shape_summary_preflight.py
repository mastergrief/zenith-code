#!/usr/bin/env python3
"""Preflight check that probe receipts contain compact pressure_shape_summary fields."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from calm.hrm_text_158.native_full_stack.pressure_shape_agreement import (
    load_receipt,
    verify_pressure_shape_summary_preflight,
)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Verify pressure_shape_summary availability in a probe receipt.",
    )
    ap.add_argument(
        "--receipt",
        type=Path,
        required=True,
        help="Absolute path to on/receipt.json or off/receipt.json.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to write preflight JSON.",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    receipt_path = args.receipt.resolve()
    if not receipt_path.is_file():
        print(f"error: receipt not found: {receipt_path}", file=sys.stderr)
        return 2
    receipt = load_receipt(receipt_path)
    payload = verify_pressure_shape_summary_preflight(receipt, receipt_path=receipt_path)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if bool(payload.get("pass")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
