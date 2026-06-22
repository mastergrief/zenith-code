#!/usr/bin/env python3
"""Thin CLI for the R7 mechanism classifier probe."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from calm.hrm_text_158.native_full_stack.r7_mechanism_classifier_probe import (
    build_classifier_probe_receipt,
)


def _git_head_sha(repo_root: Path) -> str:
    output = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
    )
    return output.strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Emit the read-only R7 mechanism classifier receipt.",
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--head-sha256")
    parser.add_argument("--sidecar-name", default="r7_cap_defer_pressure_sidecar.jsonl")
    parser.add_argument("--indent", type=int, default=2)
    args = parser.parse_args(argv)

    head_sha = args.head_sha256 or _git_head_sha(REPO_ROOT)
    receipt = build_classifier_probe_receipt(
        run_root=args.run_root,
        head_sha256=head_sha,
        sidecar_name=str(args.sidecar_name),
    )
    payload = json.dumps(receipt, indent=args.indent, sort_keys=True)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(payload + "\n", encoding="utf-8")
    else:
        sys.stdout.write(payload + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
