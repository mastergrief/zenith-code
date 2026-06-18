#!/usr/bin/env python3
"""Read-only 3C-A rescale-law sweep (classify-before-build step 1).

Thin CLI wrapper over calm.hrm_text_158.native_full_stack.rescale_law_readonly_sweep.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from calm.hrm_text_158.native_full_stack.rescale_law_readonly_sweep import run_readonly_sweep


def _print_compact_table(payload: dict) -> None:
    print("candidate_id\tshift\tmove_n\trank_rate\tevent_rate\tfrac_coll\tverdict")
    for row in payload["candidate_results"]:
        print(
            f"{row['candidate_id']}\t{row['rescale_shift']}\t{row['move_candidate_count']}\t"
            f"{row['rank_positions_match_rate']:.4f}\t{row['events_match_rate']:.4f}\t"
            f"{row['fractional_collision_share_of_mismatches']:.4f}\t{row['parity_verdict']}"
        )
    print("--- selector ---")
    print(json.dumps(payload["selector"], indent=2, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit full JSON payload on stdout (default prints compact table).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    payload = run_readonly_sweep(repo_root=REPO_ROOT)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _print_compact_table(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
