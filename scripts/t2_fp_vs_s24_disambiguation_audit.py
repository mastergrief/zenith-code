#!/usr/bin/env python3
"""CPU audit driver for T2 FP-vs-S24 disambiguation diagnostic."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from calm.hrm_text_158.native_full_stack.t2_fp_vs_s24_disambiguation import (
    run_t2_fp_vs_s24_disambiguation,
    validate_t2_disambiguation_receipt,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    receipt = run_t2_fp_vs_s24_disambiguation(
        checkpoint_path=str(args.checkpoint) if args.checkpoint is not None else None,
    )
    validate_t2_disambiguation_receipt(receipt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt.to_dict(), indent=2) + "\n", encoding="utf-8")
    print(
        "T2_FP_VS_S24_DISAMBIGUATION="
        f"{receipt.branch_id} recommended={receipt.recommended_next_slice} "
        f"anchor_pass={receipt.anchor_precondition_pass} "
        f"self_consistency={receipt.self_consistency_pass}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
