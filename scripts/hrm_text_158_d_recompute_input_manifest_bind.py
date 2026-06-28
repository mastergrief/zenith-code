#!/usr/bin/env python3
"""Thin CLI: write the bound postrun input manifest from a packet spec.

Runs AFTER the producing run (two-phase first-classification bind). Reads ONLY
the artifacts named in the packet's ``expected_native_input_manifest_spec``
allowlist, records their live hashes, and fails closed if any are missing. The
emitted manifest is the pinned reproduction input consumed by the postrun
classifier via ``--input-manifest``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from calm.hrm_text_158.native_full_stack.d_recompute_input_manifest_bind import (
    build_input_manifest,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="D recompute-window postrun input-manifest bind")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--packet",
        type=Path,
        required=True,
        help="Launch packet JSON with expected_native_input_manifest_spec",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output path for the bound postrun input manifest JSON",
    )
    args = parser.parse_args(argv)

    packet = json.loads(args.packet.read_text(encoding="utf-8"))
    try:
        manifest = build_input_manifest(args.run_root, packet)
    except ValueError as exc:
        print(json.dumps({"pass": False, "error": str(exc)}, indent=2))
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "pass": True,
                "out": str(args.out),
                "run_id": manifest["run_id"],
                "spec_sha256": manifest["spec_sha256"],
                "artifact_count": len(manifest["artifacts"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
