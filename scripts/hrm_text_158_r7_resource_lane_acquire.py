#!/usr/bin/env python3
"""Acquire GPU resource lane for R7 from-clean diagnostic."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def acquire_resource_lane(run_root: Path, *, mock: bool = False) -> dict:
    run_root.mkdir(parents=True, exist_ok=True)
    if mock:
        holding = {
            "schema": "hrm_text_158_r7_resource_lane_holding/v1",
            "request_alias": "gpu:hrm-text-158",
            "mock": True,
            "acquire_result": {
                "acquired": True,
                "token": "mock-token-r7",
                "canonical_name": "gpu:TheRig:uuid:GPU-mock-r7",
            },
        }
    else:
        ai_room_root = Path.home() / ".ai-room"
        if str(ai_room_root) not in sys.path:
            sys.path.insert(0, str(ai_room_root))
        from mcp_server_lib.main import init_room

        init_room(channel="claw-code")
        from mcp_server_lib import compat, facade_exports

        compat.init(facade_exports)
        from mcp_server_lib.tools.resource_lanes import tool_resource_lane_acquire

        handle = os.environ.get("R7_OPERATOR_HANDLE", "test-operator")
        raw = tool_resource_lane_acquire(
            handle,
            {
                "name": "gpu:hrm-text-158",
                "ttl_seconds": 1800,
                "note": "R7-from-clean-cap-defer-diagnostic",
            },
        )
        result = json.loads(raw)
        holding = {
            "schema": "hrm_text_158_r7_resource_lane_holding/v1",
            "request_alias": "gpu:hrm-text-158",
            "acquire_result": result,
        }
        if not result.get("acquired"):
            out = run_root / "prelaunch" / "resource_lane_holding.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(holding, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            raise SystemExit("resource_lane_acquire_failed")
    out = run_root / "prelaunch" / "resource_lane_holding.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(holding, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return holding


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="R7 resource lane acquire.")
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--mock", action="store_true", help="Write synthetic holding (CPU smoke).")
    args = parser.parse_args(argv)
    holding = acquire_resource_lane(args.run_root, mock=args.mock)
    print(json.dumps(holding))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
