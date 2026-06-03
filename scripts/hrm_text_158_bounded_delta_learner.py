#!/usr/bin/env python3
"""Default-off C2.0 bounded-delta learner CPU dry-run entrypoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    RUN_BOUNDED_DELTA_LEARNER_ENV,
    run_c2_bounded_delta_cpu_dry_run,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="run the CPU dry-run smoke")
    parser.add_argument(
        "--enable-bounded-delta-learner",
        action="store_true",
        help=f"explicitly arm this default-off runner (alternative: {RUN_BOUNDED_DELTA_LEARNER_ENV}=1)",
    )
    parser.add_argument("--device", default="cpu", choices=("cpu",), help="C2.0 gate is CPU-only")
    args = parser.parse_args()

    if not args.dry_run:
        raise RuntimeError("Only --dry-run is implemented in the C2.0 default-off gate")

    receipt = run_c2_bounded_delta_cpu_dry_run(
        enabled=args.enable_bounded_delta_learner,
        device=args.device,
    )
    print(json.dumps(receipt.to_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
