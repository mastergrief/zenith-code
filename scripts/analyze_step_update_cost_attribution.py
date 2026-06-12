#!/usr/bin/env python3
"""Offline step_update cost attribution CLI for F1 discriminator derivation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from calm.hrm_text_158.native_full_stack.step_update_cost_attribution import (
    DEFAULT_THRESHOLD_LINEAGE_PACKET_MSG_ID,
    DEFAULT_THRESHOLD_S,
    ThresholdConfig,
    analyze_run_log,
    build_derivation_receipt,
)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Analyze step_update phase telemetry from durable run.log JSONL."
    )
    ap.add_argument(
        "--run-log",
        type=Path,
        action="append",
        default=[],
        help="Absolute path to run.log (repeatable).",
    )
    ap.add_argument(
        "--source-label",
        action="append",
        default=[],
        help="Label paired with each --run-log (repeatable).",
    )
    ap.add_argument(
        "--attribution-out",
        type=Path,
        default=None,
        help="Write single-run attribution JSON (requires exactly one --run-log).",
    )
    ap.add_argument(
        "--derivation-receipt-out",
        type=Path,
        default=None,
        help="Write multi-run derivation receipt JSON.",
    )
    ap.add_argument(
        "--threshold-s",
        type=float,
        default=DEFAULT_THRESHOLD_S,
        help="step_update liveness threshold in seconds (default: 95.0).",
    )
    ap.add_argument(
        "--threshold-lineage-packet-msg-id",
        default=DEFAULT_THRESHOLD_LINEAGE_PACKET_MSG_ID,
        help="Lineage packet msg id for threshold_s.",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run_logs = list(args.run_log)
    labels = list(args.source_label)

    if not run_logs:
        print("error: at least one --run-log is required", file=sys.stderr)
        return 2

    if labels and len(labels) != len(run_logs):
        print("error: --source-label count must match --run-log count", file=sys.stderr)
        return 2

    if not labels:
        labels = [path.parent.name if path.parent.name else path.stem for path in run_logs]

    threshold = ThresholdConfig(
        threshold_s=float(args.threshold_s),
        lineage_packet_msg_id=str(args.threshold_lineage_packet_msg_id),
    )

    if args.attribution_out is not None:
        if len(run_logs) != 1:
            print("error: --attribution-out requires exactly one --run-log", file=sys.stderr)
            return 2
        attribution = analyze_run_log(run_logs[0].resolve(), threshold=threshold)
        args.attribution_out.parent.mkdir(parents=True, exist_ok=True)
        args.attribution_out.write_text(
            json.dumps(attribution, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    if args.derivation_receipt_out is not None:
        sources = list(zip(labels, run_logs, strict=True))
        receipt = build_derivation_receipt(
            [(label, path.resolve()) for label, path in sources],
            threshold=threshold,
            output_dir=args.derivation_receipt_out.parent,
        )
        args.derivation_receipt_out.parent.mkdir(parents=True, exist_ok=True)
        args.derivation_receipt_out.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    if args.attribution_out is None and args.derivation_receipt_out is None:
        attribution = analyze_run_log(run_logs[0].resolve(), threshold=threshold)
        json.dump(attribution, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
