"""Write the Step-1 optimizer/update-law diagnostic science launch packet.

This is CPU-only packet authoring. It does not launch GPU work, mutate `.pt`
artifacts, or claim readiness; Step-2 launch remains separately gated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from calm.hrm_text_158.native_full_stack.bounded_delta_learner import file_sha256
from calm.hrm_text_158.native_full_stack.optimizer_update_law_science import (
    SCIENCE_MODE_BRANCH_VERDICT,
    SCIENCE_MODE_PRETERMINAL_SCREEN,
    build_optimizer_update_law_science_packet,
    validate_optimizer_update_law_science_packet,
)
from scripts.hrm_text_158_bounded_delta_acquisition_probe import DEFAULT_PARENT, DEFAULT_PARENT_SHA256


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Write the CPU-only Step-1 optimizer/update-law science packet.",
    )
    ap.add_argument("--parent", type=Path, default=Path(DEFAULT_PARENT))
    ap.add_argument("--parent-sha256", default=DEFAULT_PARENT_SHA256)
    ap.add_argument(
        "--mode",
        choices=(SCIENCE_MODE_PRETERMINAL_SCREEN, SCIENCE_MODE_BRANCH_VERDICT),
        default=SCIENCE_MODE_PRETERMINAL_SCREEN,
    )
    ap.add_argument(
        "--scratch-root",
        type=Path,
        default=Path("/tmp/hrm158_optimizer_update_law_science_packet"),
    )
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--without-inverted-falsifier", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    parent = Path(args.parent)
    parent_sha = str(args.parent_sha256)
    parent_hash_basis = "provided_parent_sha256"
    if parent.exists():
        observed_sha = file_sha256(parent)
        parent_hash_basis = "read_only_parent_file_sha256"
        if parent_sha and observed_sha != parent_sha:
            raise RuntimeError(
                f"parent sha mismatch: observed {observed_sha} != expected {parent_sha}",
            )
        parent_sha = observed_sha
    packet = build_optimizer_update_law_science_packet(
        parent_path=parent,
        parent_sha256=parent_sha,
        mode=str(args.mode),
        launch_gate_id=None,
        include_inverted=not bool(args.without_inverted_falsifier),
    )
    packet["parent_hash_basis"] = parent_hash_basis
    packet["dry_run_packet_written"] = True
    packet["gpu_launch_command_authorized"] = False
    packet["step2_launch_gate_required"] = True
    validate_optimizer_update_law_science_packet(packet)
    out_path = args.json_out or (Path(args.scratch_root) / "optimizer_update_law_science_packet.json")
    _write_json_atomic(out_path, packet)
    packet["packet_path"] = str(out_path)
    validate_optimizer_update_law_science_packet(packet)
    print(json.dumps(packet, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
