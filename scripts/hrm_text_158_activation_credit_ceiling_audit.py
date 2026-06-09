#!/usr/bin/env python3
"""Emit the tracked activation-credit ceiling audit from fixed receipt artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from calm.hrm_text_158.native_full_stack.activation_credit_ceiling_audit import (
    build_activation_credit_ceiling_audit,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Emit the tracked activation-credit raw-vs-compressed ceiling audit, "
            "including the offline sub-2 ordinal and sidecar cost-ledger sweep payloads."
        )
    )
    parser.add_argument("--seed43-receipt", type=Path, required=True)
    parser.add_argument("--seed29-receipt", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument(
        "--expect-known-anchors",
        action="store_true",
        help="return nonzero unless the known branch-4 raw/family anchors reproduce exactly",
    )
    parser.add_argument("--indent", type=int, default=2)
    args = parser.parse_args(argv)

    payload = build_activation_credit_ceiling_audit(
        seed43_receipt_path=args.seed43_receipt,
        seed29_receipt_path=args.seed29_receipt,
    )
    encoded = json.dumps(payload, indent=args.indent, sort_keys=True) + "\n"
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(encoded, encoding="utf-8")
    sys.stdout.write(encoded)
    anchors = payload["known_branch4_anchor_reproduction"]
    if args.expect_known_anchors and bool(anchors["strict_stop_triggered"]):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
