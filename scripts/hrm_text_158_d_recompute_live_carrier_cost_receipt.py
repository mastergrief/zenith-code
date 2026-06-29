#!/usr/bin/env python3
"""CPU cost receipt for Slice-5 live carrier snapshot reads (Step-2a)."""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from calm.hrm_text_158.native_full_stack.event_coded_acc_checkpoint_codec import (
    EventCodedAccEvent,
)
from calm.hrm_text_158.native_full_stack.event_coded_acc_live_carrier import (
    EventCodedAccLiveState,
    VARINT_BOUNDARY_FLAT_INDICES,
)


def _build_representative_carrier(
    *,
    logical_numel: int,
    hot_rows: int,
    backlog_rows: int,
    event_count: int,
) -> EventCodedAccLiveState:
    hot_exact = {
        int(index): (127 if index % 2 == 0 else -127)
        for index in range(int(hot_rows))
    }
    carrier = EventCodedAccLiveState.with_hot_exact(
        logical_numel=int(logical_numel),
        demotion_band=1,
        hot_exact=hot_exact,
    )
    for index in range(int(backlog_rows)):
        carrier.backlog.add(int(index))
    for index in range(int(event_count)):
        carrier._append_event(
            EventCodedAccEvent(
                flat_index=int(index * 17 + 3),
                direction=int(index % 2),
                residual_mag=int(index % 16),
                event_type=1,
            )
        )
    carrier.assert_live_carrier_byte_counters_exact()
    return carrier


def _snapshot_read_timings_seconds(
    carrier: EventCodedAccLiveState,
    *,
    iterations: int,
) -> list[float]:
    timings: list[float] = []
    for _ in range(int(iterations)):
        start = time.perf_counter()
        carrier.live_carrier_byte_snapshot()
        timings.append(time.perf_counter() - start)
    return timings


def _hot_churn_timing_seconds(
    carrier: EventCodedAccLiveState,
    *,
    hot_rows: int,
) -> float:
    start = time.perf_counter()
    for index in range(int(hot_rows)):
        carrier.hot_exact[int(index)] = int((index % 254) - 127)
        carrier.live_carrier_byte_snapshot()
    return float(time.perf_counter() - start)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--logical-numel", type=int, default=268435456)
    parser.add_argument("--hot-rows", type=int, default=4096)
    parser.add_argument("--backlog-rows", type=int, default=130816)
    parser.add_argument("--event-count", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument(
        "--snapshot-read-p95-max-seconds",
        type=float,
        default=0.001,
        help="CPU gate: snapshot-read p95 must be below this threshold.",
    )
    args = parser.parse_args(argv)

    carrier = _build_representative_carrier(
        logical_numel=int(args.logical_numel),
        hot_rows=int(args.hot_rows),
        backlog_rows=int(args.backlog_rows),
        event_count=int(args.event_count),
    )
    snapshot = carrier.live_carrier_byte_snapshot()
    timings = _snapshot_read_timings_seconds(carrier, iterations=int(args.iterations))
    p50 = float(statistics.median(timings))
    p95 = float(statistics.quantiles(timings, n=20)[18]) if len(timings) >= 20 else max(timings)
    pmax = float(max(timings))

    hot_churn_rows = (0, 64, 576, 4096)
    hot_churn: dict[str, float] = {}
    for row_count in hot_churn_rows:
        churn_carrier = EventCodedAccLiveState.with_hot_exact(
            logical_numel=int(args.logical_numel),
            demotion_band=1,
            hot_exact={0: 1},
        )
        hot_churn[f"hot_rows_{row_count}"] = _hot_churn_timing_seconds(
            churn_carrier,
            hot_rows=int(row_count),
        )

    pass_gate = bool(p95 < float(args.snapshot_read_p95_max_seconds))
    receipt = {
        "logical_numel": int(args.logical_numel),
        "hot_rows": int(args.hot_rows),
        "backlog_rows": int(args.backlog_rows),
        "event_count": int(args.event_count),
        "iterations": int(args.iterations),
        "varint_boundary_indices": list(VARINT_BOUNDARY_FLAT_INDICES),
        "live_carrier_byte_snapshot": snapshot,
        "snapshot_read_seconds": {
            "p50": p50,
            "p95": p95,
            "max": pmax,
        },
        "snapshot_read_p95_max_seconds": float(args.snapshot_read_p95_max_seconds),
        "snapshot_read_p95_gate_pass": pass_gate,
        "hot_churn_seconds": hot_churn,
        "live_carrier_bytes_exact": True,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if pass_gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
