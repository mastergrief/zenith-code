#!/usr/bin/env python3
"""Fork B resume-parity SCIENCE CLI — refuses without --allow-gpu-launch.

Thin argv wrapper around ``run_fork_b_resume_parity_certificate``. Does NOT
mint a science label under developer validation. Formal launch is a later gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fork B 5-arm resume-parity science run")
    p.add_argument("--allow-gpu-launch", action="store_true", default=False)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--parent-sha256", required=True)
    p.add_argument("--scratch-root", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--cuts", default="4,16,28")
    p.add_argument("--k-steps", type=int, default=4)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--batch-seed", type=int, default=44)
    p.add_argument("--support-order-seed", type=int, default=43)
    p.add_argument("--ordering-seed", type=int, default=17)
    p.add_argument("--developer-validation", action="store_true", default=True)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.allow_gpu_launch:
        print(
            "REFUSED: Fork B science CLI requires --allow-gpu-launch "
            "(no implicit GPU; formal launch is a separate gate)",
            file=sys.stderr,
        )
        return 2
    # Formal parent load + runner wiring lands with the launch packet.
    # This CLI refuses silent science; developer smoke uses pytest entrypoints.
    receipt = {
        "schema": "fork_b_resume_parity_science_cli/v1",
        "status": "CLI_WIRED_AWAITING_LAUNCH_PACKET",
        "allow_gpu_launch": True,
        "parent": str(args.parent),
        "parent_sha256": str(args.parent_sha256),
        "cuts": [int(x) for x in str(args.cuts).split(",") if x.strip()],
        "k_steps": int(args.k_steps),
        "steps": int(args.steps),
        "device": str(args.device),
        "developer_validation": bool(args.developer_validation),
        "science_label": None,
        "note": (
            "CLI accepts --allow-gpu-launch but does not auto-run the formal "
            "certificate; use the gated run-packet / pytest GPU smoke."
        ),
    }
    out = Path(args.scratch_root)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "fork_b_science_cli_receipt.json"
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"receipt_path": str(path), "science_label": None}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
