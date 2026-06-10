#!/usr/bin/env python3
"""Thin CLI for the two_tier_carry_falsifier_battery harness."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from calm.hrm_text_158.native_full_stack.two_tier_carry_falsifier_battery import (
    build_two_tier_carry_falsifier_battery,
    verify_manifest_preflight,
)


def _required_bound_path(
    bound_paths: dict[str, str | None],
    role: str,
) -> Path:
    raw = bound_paths.get(role)
    if not raw:
        raise ValueError(f"manifest missing required role: {role}")
    return Path(raw)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Emit the CPU read-only two_tier_carry_falsifier_battery_v0 receipt "
            "over a manifest-bound B2b trace with capture/b2c/audit integrity pins."
        )
    )
    parser.add_argument("--chain-manifest", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    parser.add_argument(
        "--fals-root",
        type=Path,
        help="Optional falsifier battery root recorded in manifest preflight.",
    )
    parser.add_argument("--indent", type=int, default=2)
    args = parser.parse_args(argv)

    try:
        manifest_payload = json.loads(
            args.chain_manifest.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        sys.stderr.write(f"hard failure: {exc}\n")
        return 2

    preflight = verify_manifest_preflight(
        manifest_payload,
        fals_root=args.fals_root,
    )
    bound_paths = dict(preflight.get("bound_paths") or {})

    try:
        receipt = build_two_tier_carry_falsifier_battery(
            stable_trace_path=_required_bound_path(bound_paths, "stable_copy_00"),
            b2b_trace_path=_required_bound_path(bound_paths, "b2b_trace"),
            capture_receipt_path=_required_bound_path(bound_paths, "capture_receipt"),
            b2c_receipt_path=_required_bound_path(bound_paths, "b2c_receipt"),
            audit_receipt_path=_required_bound_path(bound_paths, "audit_receipt"),
            acc_width_receipt_path=_required_bound_path(
                bound_paths, "acc_width_receipt"
            ),
            chain_manifest_path=args.chain_manifest,
            fals_root=args.fals_root,
            trace_hash=preflight.get("trace_hash"),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        sys.stderr.write(f"hard failure: {exc}\n")
        return 2

    encoded = json.dumps(receipt, indent=args.indent, sort_keys=True) + "\n"
    args.output_receipt.parent.mkdir(parents=True, exist_ok=True)
    args.output_receipt.write_text(encoded, encoding="utf-8")
    sys.stdout.write(encoded)
    summary = {
        "primary_label": receipt.get("primary_label"),
        "failure_reasons": receipt.get("failure_reasons", []),
        "classifier_row": (
            receipt.get("classifier", {}).get("matched_row")
            if isinstance(receipt.get("classifier"), dict)
            else None
        ),
    }
    sys.stderr.write(json.dumps(summary, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
