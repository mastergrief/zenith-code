"""Write optimizer/update-law diagnostic science authoring packets.

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
    STEP1_DRY_RUN_PACKET_KIND,
    STEP2_LAUNCH_BUNDLE_PACKET_KIND,
    build_optimizer_update_law_launch_bundle,
    build_optimizer_update_law_science_packet,
    validate_optimizer_update_law_launch_bundle,
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
        description="Write CPU-only optimizer/update-law science authoring packets.",
    )
    ap.add_argument("--parent", type=Path, default=Path(DEFAULT_PARENT))
    ap.add_argument("--parent-sha256", default=DEFAULT_PARENT_SHA256)
    ap.add_argument(
        "--packet-kind",
        choices=(STEP1_DRY_RUN_PACKET_KIND, STEP2_LAUNCH_BUNDLE_PACKET_KIND),
        default=STEP1_DRY_RUN_PACKET_KIND,
    )
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
    ap.add_argument(
        "--run-root",
        type=Path,
        default=Path("/tmp/hrm158_optimizer_update_law_step2_launch_bundle"),
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--symbolic-resource-lane", default="gpu:0")
    ap.add_argument("--phase-timeout-seconds", type=float, default=1800.0)
    ap.add_argument("--total-timeout-seconds", type=float, default=14400.0)
    ap.add_argument("--max-silent-phase-seconds", type=float, default=300.0)
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
    if args.packet_kind == STEP1_DRY_RUN_PACKET_KIND:
        packet = build_optimizer_update_law_science_packet(
            parent_path=parent,
            parent_sha256=parent_sha,
            mode=str(args.mode),
            launch_gate_id=None,
            include_inverted=not bool(args.without_inverted_falsifier),
        )
        packet["packet_kind"] = STEP1_DRY_RUN_PACKET_KIND
        packet["parent_hash_basis"] = parent_hash_basis
        packet["dry_run_packet_written"] = True
        packet["gpu_launch_command_authorized"] = False
        packet["step2_launch_gate_required"] = True
        validator = validate_optimizer_update_law_science_packet
        default_name = "optimizer_update_law_science_packet.json"
    else:
        packet = build_optimizer_update_law_launch_bundle(
            parent_path=parent,
            parent_sha256=parent_sha,
            repo_root=REPO_ROOT,
            run_root=args.run_root,
            device=str(args.device),
            launch_gate_id=None,
            symbolic_resource_lane=str(args.symbolic_resource_lane),
            phase_timeout_seconds=args.phase_timeout_seconds,
            total_timeout_seconds=args.total_timeout_seconds,
            max_silent_phase_seconds=args.max_silent_phase_seconds,
        )
        packet["parent_hash_basis"] = parent_hash_basis
        packet["dry_run_packet_written"] = True
        packet["gpu_launch_command_authorized"] = False
        packet["step2_launch_gate_required"] = True
        validator = validate_optimizer_update_law_launch_bundle
        default_name = "optimizer_update_law_step2_launch_bundle.json"
    validator(packet)
    out_path = args.json_out or (Path(args.scratch_root) / default_name)
    _write_json_atomic(out_path, packet)
    packet["packet_path"] = str(out_path)
    validator(packet)
    print(json.dumps(packet, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
