#!/usr/bin/env python3
"""Validate test-operator launch-injected dispatch receipt before GPU arms."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

WITNESS_SCHEMA = "hrm_text_158_slice5_launch_injected_dispatch_witness_receipt/v1"
OP_RECEIPT_SCHEMA = "hrm_text_158_slice5_launch_injected_dispatch_receipt/v1"
STALE_DISPATCH_MSG_ID = "1782816449484-bb5f2b80"
ROOM_MSG_ID_RE = re.compile(r"^\d+-[0-9a-f]+$")
TERMINAL_STATUSES = frozenset({"completed", "blocked", "aborted"})
ALLOWED_NON_TERMINAL_STATUSES = frozenset({"claimed", "started"})


def _normalize_run_root(path: Path | str) -> str:
    return str(path).rstrip("/") + "/"


def _derive_run_id_from_run_root(run_root: Path | str) -> str:
    return str(run_root).rstrip("/").rsplit("_", 1)[-1]


def validate_launch_injected_dispatch_receipt(
    *,
    run_root: Path,
    op_receipt: dict[str, Any],
) -> dict[str, Any]:
    failures: list[str] = []
    expected_run_id = _derive_run_id_from_run_root(run_root)

    dispatch_msg_id = str(op_receipt.get("dispatch_msg_id") or "").strip()
    if not dispatch_msg_id:
        failures.append("missing_dispatch_msg_id")
    elif dispatch_msg_id == STALE_DISPATCH_MSG_ID:
        failures.append(f"stale_dispatch_msg_id:{STALE_DISPATCH_MSG_ID}")
    elif not ROOM_MSG_ID_RE.match(dispatch_msg_id):
        failures.append("invalid_dispatch_msg_id_format")

    issuer = op_receipt.get("issuer")
    if not issuer:
        failures.append("missing_issuer")

    if op_receipt.get("claimed") is not True:
        failures.append(f"claimed_not_true:{op_receipt.get('claimed')!r}")

    status = str(op_receipt.get("dispatch_run_status") or "")
    if not status:
        failures.append("missing_dispatch_run_status")
    elif status in TERMINAL_STATUSES:
        failures.append(f"dispatch_already_terminal:{status}")
    elif status not in ALLOWED_NON_TERMINAL_STATUSES:
        failures.append(f"unexpected_dispatch_run_status:{status}")

    intended_run_id = str(op_receipt.get("intended_run_id") or "")
    if intended_run_id != expected_run_id:
        failures.append(f"intended_run_id_mismatch:{intended_run_id!r}")

    intended_run_root = op_receipt.get("intended_run_root")
    marker_run_root = op_receipt.get("marker_run_root")
    if not intended_run_root:
        failures.append("missing_intended_run_root")
    if not marker_run_root:
        failures.append("missing_marker_run_root")
    if intended_run_root and marker_run_root:
        if _normalize_run_root(marker_run_root) != _normalize_run_root(intended_run_root):
            failures.append(
                f"marker_run_root_mismatch:{marker_run_root}!={intended_run_root}"
            )
    if intended_run_root and not str(intended_run_root).rstrip("/").endswith(expected_run_id):
        failures.append(f"intended_run_root_suffix_mismatch:{intended_run_root!r}")
    if not str(run_root).rstrip("/").endswith(expected_run_id):
        failures.append(f"run_root_suffix_mismatch:{run_root}")

    return {
        "schema": WITNESS_SCHEMA,
        "op_receipt_schema": op_receipt.get("schema", OP_RECEIPT_SCHEMA),
        "run_id": expected_run_id,
        "run_root": str(run_root),
        "dispatch_msg_id": dispatch_msg_id,
        "issuer": issuer,
        "claimed": op_receipt.get("claimed"),
        "dispatch_run_status": status or None,
        "marker_run_root": marker_run_root,
        "intended_run_root": intended_run_root,
        "intended_run_id": intended_run_id or None,
        "pass": not failures,
        "failures": failures,
    }


def emit_launch_injected_dispatch_witness_receipt(
    *,
    run_root: Path,
    op_receipt_path: Path,
) -> dict[str, Any]:
    expected_run_id = _derive_run_id_from_run_root(run_root)
    if not op_receipt_path.is_file():
        witness = {
            "schema": WITNESS_SCHEMA,
            "run_id": expected_run_id,
            "run_root": str(run_root),
            "pass": False,
            "failures": ["missing_launch_injected_dispatch_receipt"],
        }
        return witness

    try:
        op_receipt = json.loads(op_receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "schema": WITNESS_SCHEMA,
            "run_id": expected_run_id,
            "run_root": str(run_root),
            "pass": False,
            "failures": [f"op_receipt_read_error:{exc}"],
        }

    return validate_launch_injected_dispatch_receipt(
        run_root=run_root,
        op_receipt=op_receipt,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--in",
        dest="op_receipt_path",
        type=Path,
        required=True,
        help="Test-operator written prelaunch/launch_injected_dispatch_receipt.json",
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    receipt = emit_launch_injected_dispatch_witness_receipt(
        run_root=args.run_root,
        op_receipt_path=args.op_receipt_path,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0 if receipt.get("pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
