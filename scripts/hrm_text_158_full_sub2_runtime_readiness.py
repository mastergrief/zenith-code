#!/usr/bin/env python3
"""Emit the HRM-Text-1.58 full-sub2 runtime readiness receipt."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from calm.hrm_text_158.native_full_stack.full_sub2_runtime_readiness import (
    FIXTURE_CURRENT_REPO,
    FIXTURE_LIVE_P1_AUTHORITY_CONVERSION,
    FULL_SUB2_RUNTIME_FIXTURE_NAMES,
    fixture_full_sub2_runtime_ready_for_science,
    live_p1_authority_conversion_surfaces,
)
from calm.hrm_text_158.native_full_stack.trainer_sub2_authority import (
    live_conversion_receipt_from_dict,
    validate_trainer_sub2_authority_live_conversion_receipt,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Emit a fail-closed full_sub2_runtime_ready_for_science JSON receipt."
    )
    parser.add_argument(
        "--fixture",
        choices=FULL_SUB2_RUNTIME_FIXTURE_NAMES,
        default=FIXTURE_CURRENT_REPO,
        help="read-only static fixture to emit",
    )
    parser.add_argument(
        "--live-p1-receipt-json",
        type=Path,
        help=(
            "required when --fixture live_p1_authority_conversion; "
            "path to a validated P1b live conversion receipt JSON"
        ),
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        help="optional path for the JSON receipt; stdout is always emitted",
    )
    parser.add_argument(
        "--expect-ready",
        action="store_true",
        help="return nonzero unless ready_for_main_science is true",
    )
    parser.add_argument("--indent", type=int, default=2)
    args = parser.parse_args(argv)

    if args.fixture == FIXTURE_LIVE_P1_AUTHORITY_CONVERSION:
        if args.live_p1_receipt_json is None:
            parser.error(
                "--fixture live_p1_authority_conversion requires "
                "--live-p1-receipt-json PATH"
            )
        if not args.live_p1_receipt_json.is_file():
            raise SystemExit(
                f"live P1 receipt JSON not found: {args.live_p1_receipt_json}"
            )
        try:
            receipt_payload = json.loads(
                args.live_p1_receipt_json.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError as exc:
            raise SystemExit(
                f"invalid live P1 receipt JSON: {args.live_p1_receipt_json}: {exc}"
            ) from exc
        live_receipt = live_conversion_receipt_from_dict(receipt_payload)
        validate_trainer_sub2_authority_live_conversion_receipt(live_receipt)
        receipt = live_p1_authority_conversion_surfaces(live_receipt)
    else:
        receipt = fixture_full_sub2_runtime_ready_for_science(args.fixture)

    payload = receipt.to_dict()
    encoded = json.dumps(payload, indent=args.indent, sort_keys=True) + "\n"
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(encoded, encoding="utf-8")
    sys.stdout.write(encoded)
    if args.expect_ready and not receipt.ready_for_main_science:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
