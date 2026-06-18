#!/usr/bin/env python3
"""Read-only anchored T2 FP credit degeneracy grounding diagnostic CLI."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from calm.hrm_text_158.native_full_stack.fp_credit_degeneracy_diagnostic import (
    run_anchored_fp_credit_degeneracy_diagnostic,
)
from calm.hrm_text_158.native_full_stack.realistic_gradient_parity_probe import (
    DEFAULT_T2_CHECKPOINT_REL,
    discover_t2_checkpoint,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.checkpoint is not None:
        checkpoint_path = str(args.checkpoint)
        checkpoint_sha256 = None
    else:
        discovery = discover_t2_checkpoint()
        if not discovery.checkpoint_present:
            raise FileNotFoundError(DEFAULT_T2_CHECKPOINT_REL)
        checkpoint_path = str(discovery.checkpoint_path)
        checkpoint_sha256 = discovery.checkpoint_sha256
    payload = run_anchored_fp_credit_degeneracy_diagnostic(
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
    )
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    enrichment = payload["enrichment"]
    print(
        "FP_CREDIT_DEGENERACY "
        f"branch_id={payload['branch_id']} "
        f"anchor_in_band={payload['full_pool_fp4_anchor_in_band']} "
        f"full_pool_fp4_fraction={enrichment['full_pool_fp4_fraction']:.6f} "
        f"parity_fp4_balance={enrichment['parity_fp4_balance']:.6f} "
        f"enrichment={enrichment['parity_index_enrichment_over_full_pool']:.6f} "
        f"coverage={enrichment['integer_move_coverage_vs_full_fp']:.6f} "
        f"full_keys={payload['key_filter_audit']['full_pool_key_count']} "
        f"parity_valid={payload['key_filter_audit']['parity_valid_key_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
