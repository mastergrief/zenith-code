#!/usr/bin/env python3
"""Release GPU resource lane for R7 from-clean diagnostic."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def release_resource_lane(run_root: Path, *, mock: bool = False) -> dict:
    holding_path = run_root / "prelaunch" / "resource_lane_holding.json"
    if not holding_path.is_file():
        witness = {"released": False, "reason": "no_holding_file"}
        print(json.dumps(witness))
        return witness
    holding = json.loads(holding_path.read_text(encoding="utf-8"))
    acquire = holding.get("acquire_result") or {}
    token = acquire.get("token")
    if not token:
        witness = {"released": False, "reason": "no_token_in_holding"}
        print(json.dumps(witness))
        return witness
    lane_name = (
        acquire.get("canonical_name")
        or acquire.get("physical_key")
        or holding.get("request_alias", "gpu:hrm-text-158")
    )
    if mock or holding.get("mock"):
        result = {"released": True, "mock": True}
    else:
        ai_room_root = Path.home() / ".ai-room"
        if str(ai_room_root) not in sys.path:
            sys.path.insert(0, str(ai_room_root))
        from mcp_server_lib.main import init_room

        init_room(channel="claw-code")
        from mcp_server_lib import compat, facade_exports

        compat.init(facade_exports)
        from mcp_server_lib.tools.resource_lanes import tool_resource_lane_release

        handle = os.environ.get("R7_OPERATOR_HANDLE", "test-operator")
        raw = tool_resource_lane_release(handle, {"name": lane_name, "token": token})
        result = json.loads(raw)
    witness = {
        "schema": "hrm_text_158_r7_resource_lane_release_witness/v1",
        "release_request": {
            "name": lane_name,
            "token_present": True,
            "canonical_name": acquire.get("canonical_name"),
            "request_alias": holding.get("request_alias"),
        },
        "release_result": result,
    }
    out = run_root / "post_gpu" / "resource_lane_release_witness.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(witness, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(witness))
    return witness


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="R7 resource lane release.")
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--mock", action="store_true", help="Skip live lane release (CPU smoke).")
    args = parser.parse_args(argv)
    release_resource_lane(args.run_root, mock=args.mock)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
