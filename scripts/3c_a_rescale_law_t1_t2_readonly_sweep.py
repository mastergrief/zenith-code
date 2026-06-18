#!/usr/bin/env python3
"""Read-only dual-tier 3C-A rescale-law sweep (T1 + anchored T2)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from calm.hrm_text_158.native_full_stack.realistic_gradient_parity_probe import (
    DEFAULT_T2_CHECKPOINT_REL,
    discover_t2_checkpoint,
)
from calm.hrm_text_158.native_full_stack.rescale_law_readonly_sweep import (
    run_dual_tier_readonly_sweep,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.checkpoint is not None:
        checkpoint_path = str(args.checkpoint)
    else:
        discovery = discover_t2_checkpoint()
        if not discovery.checkpoint_present:
            raise FileNotFoundError(DEFAULT_T2_CHECKPOINT_REL)
        checkpoint_path = str(discovery.checkpoint_path)
    payload = run_dual_tier_readonly_sweep(
        checkpoint_path=checkpoint_path,
        repo_root=REPO_ROOT,
    )
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    print(
        "DUAL_TIER_SWEEP selector="
        f"{payload['selector']['selector_outcome']} "
        f"routing={payload['selector'].get('routing_strength')} "
        f"shift={payload['selector'].get('selected_shift')}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
