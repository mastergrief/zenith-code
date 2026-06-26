#!/usr/bin/env python3
"""Build a bankable joint-drain envelope verdict from a Phase A evidence manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from calm.hrm_text_158.native_full_stack.carrier_envelope_projector import (
    build_verdict_from_manifest_path,
    canonical_json,
)


def _git_head(repo_root: Path) -> str:
    return (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root)
        .decode("utf-8")
        .strip()
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="Path to v4 Phase A evidence manifest JSON.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path to write the envelope verdict JSON.",
    )
    parser.add_argument(
        "--rollup",
        type=Path,
        default=None,
        help="Optional carrier_growth compact rollup JSON for oracle transforms.",
    )
    parser.add_argument(
        "--packet-sha",
        default=None,
        help="Optional packet sha256 for provenance.",
    )
    parser.add_argument(
        "--head-commit",
        default=None,
        help="Optional HEAD commit override (defaults to git rev-parse HEAD).",
    )
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    rollup = None
    if args.rollup is not None:
        rollup_payload = json.loads(args.rollup.read_text(encoding="utf-8"))
        rollup = rollup_payload.get("rollup", rollup_payload)

    verdict = build_verdict_from_manifest_path(
        manifest_path=str(args.manifest),
        head_commit=str(args.head_commit or _git_head(repo_root)),
        packet_sha=args.packet_sha,
        rollup=rollup,
    )
    text = canonical_json(verdict)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    print(json.dumps({"output": str(args.output), "sha256": digest}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
