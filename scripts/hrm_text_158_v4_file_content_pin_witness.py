#!/usr/bin/env python3
"""V4-LIVE file-content pin witness — authoritative launch preflight."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "v4_live_file_content_pin_witness/v1"
PREFLIGHT_AUTHORITY = "file_content_sha256_at_head"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_packet_pins(packet_path: Path) -> dict[str, str]:
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    pins = packet.get("code_pins", {}).get("file_content_sha256_at_head")
    if not isinstance(pins, dict) or not pins:
        raise ValueError("packet missing code_pins.file_content_sha256_at_head")
    return {str(rel): str(expected) for rel, expected in pins.items()}


def run_file_content_pin_witness(
    *,
    packet_path: Path,
    json_out: Path,
    pins_override: dict[str, str] | None = None,
) -> dict[str, Any]:
    pins = pins_override if pins_override is not None else load_packet_pins(packet_path)
    per_file: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    missing_files: list[str] = []

    for rel_path, expected_sha in sorted(pins.items()):
        file_path = Path(rel_path)
        if not file_path.is_file():
            missing_files.append(rel_path)
            row = {
                "path": rel_path,
                "expected": expected_sha,
                "actual": None,
                "match": False,
                "error": "missing_file",
            }
            per_file.append(row)
            mismatches.append(row)
            continue
        actual_sha = sha256_file(file_path)
        match = actual_sha == expected_sha
        row = {
            "path": rel_path,
            "expected": expected_sha,
            "actual": actual_sha,
            "match": match,
        }
        per_file.append(row)
        if not match:
            mismatches.append(row)

    all_match = len(mismatches) == 0
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "preflight_authority": PREFLIGHT_AUTHORITY,
        "packet_path": str(packet_path),
        "pinned_file_count": len(pins),
        "mismatch_count": len(mismatches),
        "missing_file_count": len(missing_files),
        "per_file": per_file,
        "mismatches": mismatches,
        "all_match": all_match,
        "file_content_pin_pass": all_match,
    }
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify launch packet file_content_sha256_at_head pins against on-disk shas."
    )
    parser.add_argument(
        "--packet",
        required=True,
        type=Path,
        help="Path to v4_live_trainer_integration_gpu_launch_packet_v1.json",
    )
    parser.add_argument(
        "--json-out",
        required=True,
        type=Path,
        help="Output path for witness JSON (e.g. {run_root}/prelaunch/file_content_pin_witness.json)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        payload = run_file_content_pin_witness(
            packet_path=args.packet,
            json_out=args.json_out,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"file_content_pin_witness failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload["file_content_pin_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
