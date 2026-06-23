#!/usr/bin/env python3
"""Post R8 global_cap_relax_512 terminal validation receipt to ai-room."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def _init_ai_room() -> None:
    ai_room_root = Path.home() / ".ai-room"
    if str(ai_room_root) not in sys.path:
        sys.path.insert(0, str(ai_room_root))
    from mcp_server_lib.main import init_room

    init_room(channel="claw-code")
    from mcp_server_lib import compat, facade_exports

    compat.init(facade_exports)


def _r8_validation_receipt_body(terminal: dict, *, run_root: Path, terminal_sha: str) -> dict:
    return {
        "schema": "hrm_text_158_r8_global_cap_relax_terminal_validation_receipt/v1",
        "run_root": str(run_root),
        "primary_branch": terminal.get("primary_branch"),
        "next_action": terminal.get("next_action"),
        "steps_observed": terminal.get("steps_observed"),
        "run_metrics": terminal.get("run_metrics"),
        "branch_selection": terminal.get("branch_selection"),
        "audit_summary": terminal.get("audit_summary"),
        "baseline_comparison": terminal.get("baseline_comparison"),
        "r7_baseline_provenance": terminal.get("r7_baseline_provenance"),
        "classifier_receipt_sha256": terminal.get("classifier_receipt_sha256"),
        "terminal_receipt_sha256": terminal_sha,
        "explicit_non_claims": terminal.get("explicit_non_claims", []),
    }


def post_terminal_to_ai_room(
    run_root: Path,
    *,
    skip_post: bool = False,
    operator_handle: str | None = None,
) -> dict:
    terminal_path = run_root / "terminal_receipt.json"
    if not terminal_path.is_file():
        raise SystemExit("missing_terminal_receipt_for_ai_room_post")
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    terminal_sha = hashlib.sha256(terminal_path.read_bytes()).hexdigest()
    post_body = {
        "schema": "hrm_text_158_r8_global_cap_relax_ai_room_terminal_post/v1",
        "run_root": str(run_root),
        **_r8_validation_receipt_body(terminal, run_root=run_root, terminal_sha=terminal_sha),
    }
    payload_path = run_root / "post_gpu" / "ai_room_terminal_post_payload.json"
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    payload_path.write_text(json.dumps(post_body, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    witness: dict = {
        "schema": "hrm_text_158_r8_ai_room_post_witness/v1",
        "terminal_receipt_sha256": terminal_sha,
        "payload_path": str(payload_path),
        "post_skipped": skip_post,
        "msg_id": None,
        "post_result": None,
    }

    if skip_post:
        witness_path = run_root / "post_gpu" / "ai_room_post_witness.json"
        witness_path.write_text(json.dumps(witness, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return witness

    _init_ai_room()
    from mcp_server_lib.tools.messages import tool_post

    handle = operator_handle or os.environ.get("R8_OPERATOR_HANDLE", "test-operator")
    receipt_text = json.dumps(
        _r8_validation_receipt_body(terminal, run_root=run_root, terminal_sha=terminal_sha),
        indent=2,
        sort_keys=True,
    )
    post_command = (
        f"R8 global_cap_relax_512 terminal validation_receipt "
        f"run_root={run_root} branch={terminal.get('primary_branch')}"
    )
    raw = tool_post(
        handle,
        {
            "body": receipt_text,
            "to": "claude",
            "kind": "validation_receipt",
            "scope": "R8 global_cap_relax_512 from-clean diagnostic terminal",
            "command": post_command,
            "result": f"primary_branch={terminal.get('primary_branch')}",
            "artifact_paths": [str(terminal_path), str(payload_path)],
        },
    )
    msg_id = None
    if raw.startswith("posted id="):
        msg_id = raw.split("posted id=", 1)[1].split(" ", 1)[0]
    witness["msg_id"] = msg_id
    witness["post_result"] = raw
    witness_path = run_root / "post_gpu" / "ai_room_post_witness.json"
    witness_path.write_text(json.dumps(witness, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return witness


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="R8 ai-room terminal post.")
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--skip-post", action="store_true", help="Dry-run: write witness only.")
    args = parser.parse_args(argv)
    witness = post_terminal_to_ai_room(args.run_root, skip_post=args.skip_post)
    print(json.dumps(witness))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
