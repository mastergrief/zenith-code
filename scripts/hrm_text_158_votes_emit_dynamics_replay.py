#!/usr/bin/env python3
"""CLI harness for offline votes-emit dynamics replay scoring."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from calm.hrm_text_158.native_full_stack.votes_emit_dynamics_replay import (
    ARM_MODE_LIVE_ALTERED_DYNAMICS,
    ARM_MODE_REPLAY_ONLY,
    REPLAY_MODE_R_DYNAMICS,
    REPLAY_MODE_R_STATIC,
    score_votes_emit_replay_run,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score votes-emit dynamics-proof replay artifacts (CPU-only)."
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--replay-mode",
        choices=(REPLAY_MODE_R_STATIC, REPLAY_MODE_R_DYNAMICS),
        default=REPLAY_MODE_R_STATIC,
    )
    parser.add_argument("--arm-id", default="V0")
    parser.add_argument(
        "--arm-mode",
        choices=(ARM_MODE_REPLAY_ONLY, ARM_MODE_LIVE_ALTERED_DYNAMICS),
        default=ARM_MODE_REPLAY_ONLY,
    )
    parser.add_argument("--from-clean-contiguous", action="store_true")
    parser.add_argument("--live-evidence", action="store_true")
    parser.add_argument("--run-health-ok", action="store_true", default=True)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()

    receipt = score_votes_emit_replay_run(
        Path(args.run_root),
        replay_mode=str(args.replay_mode),
        arm_id=str(args.arm_id),
        arm_mode=str(args.arm_mode),
        from_clean_contiguous=bool(args.from_clean_contiguous),
        live_evidence=bool(args.live_evidence),
        run_health_ok=bool(args.run_health_ok),
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.json_out)


if __name__ == "__main__":
    main()
