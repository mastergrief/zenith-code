#!/usr/bin/env python3
"""CPU classify driver for optimizer persistent carrier width narrowability."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from calm.hrm_text_158.native_full_stack.optimizer_persistent_carrier_width import (
    PROOF_LAW_HISTORICAL_NO_DRAIN_CONTROL,
    PROOF_LAW_LOCKED_ARM_A,
    classify_from_parent_receipt_file,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-receipt", type=Path, required=True)
    parser.add_argument(
        "--proof-law-id",
        choices=(PROOF_LAW_LOCKED_ARM_A, PROOF_LAW_HISTORICAL_NO_DRAIN_CONTROL),
        required=True,
    )
    parser.add_argument("--control-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    receipt = classify_from_parent_receipt_file(
        args.parent_receipt,
        proof_law_id=args.proof_law_id,
        control_only=bool(args.control_only),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt.to_dict(), indent=2) + "\n", encoding="utf-8")
    print(
        "RACC_OPTIMIZER_PERSISTENT_CARRIER_WIDTH_CLASSIFY="
        f"{receipt.branch_id} parent={receipt.parent_receipt_sha256[:8]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
