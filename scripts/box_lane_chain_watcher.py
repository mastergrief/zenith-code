#!/usr/bin/env python3
"""Science-chain producer/consumer overlap watcher (§4B)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from calm.hrm_text_158.native_full_stack.box_lane import (
    EXIT_OK,
    EXIT_OVERLAP_FAILURE,
    classify_overlap,
    process_science_chain_log,
)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Box-lane science-chain overlap watcher (§4B).")
    ap.add_argument("producer_log", type=Path, help="Producer log with capture_complete events.")
    ap.add_argument("--manifest", type=Path, required=True, help="Overlap manifest output path.")
    ap.add_argument("--waive", action="store_true", help="Do not exit nonzero on overlap failures.")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    lines = args.producer_log.read_text(encoding="utf-8").splitlines()
    states = process_science_chain_log(lines)
    entries: list[dict[str, object]] = []
    flagged: list[str] = []
    for chain_id, state in sorted(states.items()):
        verdict = classify_overlap(state)
        if verdict.status in {
            "SERIAL_FALLBACK",
            "INELIGIBLE",
            "QUARANTINED_AFTER_CONSUMER_FAIL",
        }:
            flagged.append(chain_id)
        entries.append(
            {
                "chain_id": chain_id,
                "seed": state.seed,
                "status": verdict.status,
                "issues": list(verdict.issues),
                "consumer_terminal_status": state.consumer_terminal_status,
                "quarantined": state.quarantined,
                "pipeline_eligible": verdict.pipeline_eligible,
                "verdict_eligible": verdict.verdict_eligible,
                "overlap_seconds": verdict.overlap_seconds,
                "producer_capture_complete_ts": state.producer_capture_complete_ts,
                "producer_next_start_ts": state.producer_next_start_ts,
                "consumer_audit_start_ts": state.consumer_audit_start_ts,
            }
        )
    manifest = {
        "schema": "hrm158_box_lane_overlap_manifest/v1",
        "producer_log": str(args.producer_log),
        "waived": bool(args.waive),
        "n_overlap": sum(1 for e in entries if e["status"] == "OVERLAP"),
        "n_flagged": len(flagged),
        "entries": entries,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "manifest": str(args.manifest),
                "n_overlap": manifest["n_overlap"],
                "n_flagged": manifest["n_flagged"],
            }
        )
    )
    if flagged and not args.waive:
        return EXIT_OVERLAP_FAILURE
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
