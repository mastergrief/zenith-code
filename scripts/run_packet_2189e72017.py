#!/usr/bin/env python3
"""Thin entrypoint for H=200 B1 de-censor relaunch run_id 2189e72017."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKET = (
    "artifacts/consensus_prep/"
    "d_recompute_window_feasibility_gpu_launch_packet_v2_h200_relaunch_draft.json"
)
REQUIRED_ANCESTOR = "5cc8fb95fc093d1897ddc128d850e3df9a1ff5d3"  # Slice-1 science-fix commit


def main() -> int:
    driver = REPO_ROOT / "scripts/hrm_text_158_d_recompute_window_run_packet.py"
    return int(
        subprocess.run(
            [
                sys.executable,
                str(driver),
                "--packet",
                str(REPO_ROOT / PACKET),
                "--require-ancestor",
                REQUIRED_ANCESTOR,
                "--repo-root",
                str(REPO_ROOT),
            ],
            cwd=REPO_ROOT,
            check=False,
        ).returncode
    )


if __name__ == "__main__":
    raise SystemExit(main())
