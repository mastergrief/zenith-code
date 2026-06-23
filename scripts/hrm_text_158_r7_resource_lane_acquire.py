#!/usr/bin/env python3
"""Acquire GPU resource lane for R7 from-clean diagnostic."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable

# Bounded wait-before-acquire on lane contention. Each attempt is a fresh
# acquire_once (the lane is NEVER held while sleeping); on a non-acquired
# result we sleep the next backoff entry and re-query. After the final
# attempt the caller fail-closes (SystemExit), never spawns, never
# force-releases. Total bounded wait = sum of the first (attempts-1)
# backoff entries = 130s, under the 240s cap.
DEFAULT_ACQUIRE_MAX_RETRIES = 5
DEFAULT_ACQUIRE_BACKOFF_SECONDS: tuple[float, ...] = (10.0, 20.0, 40.0, 60.0, 60.0)


def acquire_with_backoff(
    acquire_once: Callable[[], dict],
    *,
    max_retries: int = DEFAULT_ACQUIRE_MAX_RETRIES,
    backoff_schedule: tuple[float, ...] = DEFAULT_ACQUIRE_BACKOFF_SECONDS,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict:
    """Bounded retry/backoff around a single-shot acquire callable.

    Returns the first acquired result, or the last non-acquired result after
    exhausting the bounded attempts. Wait-before-acquire only: never holds the
    lane while sleeping, never force-releases, never spawns.
    """
    if max_retries < 1:
        raise ValueError("max_retries must be >= 1")
    last_result: dict = {}
    for attempt in range(max_retries):
        last_result = acquire_once()
        if last_result.get("acquired"):
            return last_result
        if attempt == max_retries - 1:
            break
        backoff = (
            backoff_schedule[attempt]
            if attempt < len(backoff_schedule)
            else backoff_schedule[-1]
        )
        sleep_fn(float(backoff))
    return last_result


def acquire_resource_lane(
    run_root: Path,
    *,
    mock: bool = False,
    max_retries: int = DEFAULT_ACQUIRE_MAX_RETRIES,
    backoff_schedule: tuple[float, ...] = DEFAULT_ACQUIRE_BACKOFF_SECONDS,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict:
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

        def acquire_once() -> dict:
            # Single-shot acquire (wait_seconds omitted → server returns
            # immediately on contention); bounded waiting is owned by
            # acquire_with_backoff, which never holds the lane while sleeping.
            raw = tool_resource_lane_acquire(
                handle,
                {
                    "name": "gpu:hrm-text-158",
                    "ttl_seconds": 1800,
                    "note": "R7-from-clean-cap-defer-diagnostic",
                },
            )
            return json.loads(raw)

        result = acquire_with_backoff(
            acquire_once,
            max_retries=max_retries,
            backoff_schedule=backoff_schedule,
            sleep_fn=sleep_fn,
        )
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
    parser.add_argument(
        "--acquire-max-retries",
        type=int,
        default=DEFAULT_ACQUIRE_MAX_RETRIES,
        help="Bounded acquire attempts on lane contention (wait-before-acquire; default 5).",
    )
    parser.add_argument(
        "--acquire-backoff-secs",
        type=float,
        nargs="+",
        default=list(DEFAULT_ACQUIRE_BACKOFF_SECONDS),
        help="Backoff seconds between attempts (default 10 20 40 60 60; total <=240s).",
    )
    args = parser.parse_args(argv)
    holding = acquire_resource_lane(
        args.run_root,
        mock=args.mock,
        max_retries=args.acquire_max_retries,
        backoff_schedule=tuple(args.acquire_backoff_secs),
    )
    print(json.dumps(holding))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
