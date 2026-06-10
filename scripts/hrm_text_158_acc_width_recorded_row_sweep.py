#!/usr/bin/env python3
"""Thin CLI for the acc_width_recorded_row_sweep harness."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from calm.hrm_text_158.native_full_stack.acc_width_recorded_row_sweep import (
    DEFAULT_HEADROOM_FACTOR,
    DEFAULT_WIDTH_GRID,
    build_acc_width_recorded_row_sweep,
)


def _parse_width_grid(raw: str) -> tuple[int, ...]:
    widths = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not widths:
        raise ValueError("width grid must contain at least one integer width")
    return widths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Emit the CPU read-only acc_width_recorded_row_sweep_v0 receipt "
            "over a stable B2b trace with capture/b2c/audit integrity pins."
        )
    )
    parser.add_argument("--stable-trace", type=Path, required=True)
    parser.add_argument("--capture-receipt", type=Path, required=True)
    parser.add_argument("--b2c-receipt", type=Path, required=True)
    parser.add_argument("--audit-receipt", type=Path, required=True)
    parser.add_argument(
        "--chain-manifest",
        type=Path,
        help="Optional chain manifest for composed production vote-spec fallback.",
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument(
        "--width-grid",
        default=",".join(str(width) for width in DEFAULT_WIDTH_GRID),
        help="Comma-separated signed accumulator widths to sweep.",
    )
    parser.add_argument(
        "--headroom-factor",
        type=float,
        default=DEFAULT_HEADROOM_FACTOR,
    )
    parser.add_argument("--expected-stable-trace-sha256")
    parser.add_argument("--expected-capture-receipt-sha256")
    parser.add_argument("--expected-b2c-receipt-sha256")
    parser.add_argument("--expected-audit-receipt-sha256")
    parser.add_argument("--indent", type=int, default=2)
    args = parser.parse_args(argv)

    expected_shas = {
        key: value
        for key, value in {
            "stable_trace": args.expected_stable_trace_sha256,
            "capture_receipt": args.expected_capture_receipt_sha256,
            "b2c_receipt": args.expected_b2c_receipt_sha256,
            "audit_receipt": args.expected_audit_receipt_sha256,
        }.items()
        if value
    }

    try:
        receipt = build_acc_width_recorded_row_sweep(
            stable_trace_path=args.stable_trace,
            capture_receipt_path=args.capture_receipt,
            b2c_receipt_path=args.b2c_receipt,
            audit_receipt_path=args.audit_receipt,
            expected_shas=expected_shas or None,
            chain_manifest_path=args.chain_manifest,
            width_grid=_parse_width_grid(args.width_grid),
            headroom_factor=float(args.headroom_factor),
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
        "w_min": receipt.get("w_min"),
        "failure_reasons": receipt.get("failure_reasons", []),
    }
    sys.stderr.write(json.dumps(summary, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
