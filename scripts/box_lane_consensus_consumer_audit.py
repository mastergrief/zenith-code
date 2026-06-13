#!/usr/bin/env python3
"""Box-side consensus bounded-delta consumer audit (§D tooling)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from calm.hrm_text_158.native_full_stack.box_lane import (
    EXIT_ARTIFACT_RSYNC_MISMATCH,
    EXIT_OK,
    audit_consensus_bounded_delta_consumer,
    format_consumer_audit_start_line,
    format_consumer_terminal_line,
)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Consensus bounded-delta consumer audit (§D).")
    ap.add_argument("--chain-root", type=Path, required=True)
    ap.add_argument("--primary-label", default="S44")
    ap.add_argument("--isolation-label", default="S44_iso43")
    ap.add_argument(
        "--transport-manifest",
        type=Path,
        default=None,
        help="Optional box_artifact_transport.json for producer/consumer sha checks.",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Audit receipt path (default: <chain-root>/box_consensus_consumer_audit.json).",
    )
    ap.add_argument(
        "--chain-log",
        type=Path,
        default=None,
        help="Append watcher lines (consumer_audit_start / consumer_terminal).",
    )
    ap.add_argument("--chain-id", default=None, help="Watcher chain_id (default: chain-root basename).")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    chain_root = args.chain_root.resolve()
    chain_id = args.chain_id or chain_root.name
    transport_rows = None
    if args.transport_manifest is not None and args.transport_manifest.exists():
        transport_payload = json.loads(args.transport_manifest.read_text(encoding="utf-8"))
        transport_rows = transport_payload.get("artifacts")

    now = time.time()
    if args.chain_log is not None:
        args.chain_log.parent.mkdir(parents=True, exist_ok=True)
        with args.chain_log.open("a", encoding="utf-8") as handle:
            handle.write(format_consumer_audit_start_line(chain_id=chain_id, ts=now) + "\n")

    audit = audit_consensus_bounded_delta_consumer(
        chain_root,
        primary_label=args.primary_label,
        isolation_label=args.isolation_label,
        transport_artifacts=transport_rows,
    )
    output_path = args.output or (chain_root / "box_consensus_consumer_audit.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    status = "pass" if audit["pass"] else "fail"
    if args.chain_log is not None:
        with args.chain_log.open("a", encoding="utf-8") as handle:
            handle.write(
                format_consumer_terminal_line(chain_id=chain_id, status=status, ts=now + 0.01) + "\n",
            )

    print(json.dumps({"output": str(output_path), "pass": audit["pass"], "status": status}))
    if not audit["pass"]:
        return EXIT_ARTIFACT_RSYNC_MISMATCH
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
