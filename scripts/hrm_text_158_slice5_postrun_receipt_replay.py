#!/usr/bin/env python3
"""Replay postrun receipts from FINAL drained run_root artifacts (Slice B-DIAG2)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from scripts.hrm_text_158_slice5_cap_selection_path_evidence import (
    evaluate_cap_selection_path_evidence,
)
from scripts.hrm_text_158_slice5_bounded_steps_triage import triage_bounded_steps
from scripts.hrm_text_158_slice5_launch_arm_barrier import assert_postrun_barrier_ready
from scripts.hrm_text_158_slice5_live_carrier_scale_smoke_receipt import (
    emit_live_carrier_scale_smoke_receipt,
)
from scripts.hrm_text_158_slice5_milestone_stall_classifier import classify_milestone_stall

REPLAY_SCHEMA = "hrm_text_158_slice5_postrun_receipt_replay/v1"
STALE_RECEIPT_NAMES = (
    "cap_selection_path_evidence_receipt.json",
    "bounded_steps_triage_receipt.json",
    "milestone_stall_classifier_receipt.json",
    "live_carrier_scale_smoke_receipt.json",
)
SUPERSEDING_RECEIPT_NAMES = tuple(
    name.replace(".json", "_superseding.json") for name in STALE_RECEIPT_NAMES
)
REPLAY_OUTPUT_NAMES = STALE_RECEIPT_NAMES + SUPERSEDING_RECEIPT_NAMES + (
    "postrun_receipt_replay_receipt.json",
)


def _load_packet(packet_path: Path) -> dict[str, Any]:
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    if "packet_revision" not in packet or "run_id" not in packet:
        raise ValueError("packet missing packet_revision or run_id")
    return packet


def _max_steps_hard(packet: dict[str, Any]) -> int:
    scale = packet.get("scale_smoke") or {}
    if scale.get("max_steps_hard") is not None:
        return int(scale["max_steps_hard"])
    if scale.get("steps") is not None:
        return int(scale["steps"])
    return 3


def _write_superseding(path: Path, *, receipt_kind: str, run_root: Path, payload: dict[str, Any]) -> None:
    wrapped = {
        "schema": REPLAY_SCHEMA,
        "supersedes_stale_mid_run_receipts": True,
        "derived_from_final_drained_artifacts": True,
        "source_run_root": str(run_root),
        "receipt_kind": receipt_kind,
        "payload": payload,
    }
    path.write_text(json.dumps(wrapped, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _mark_stale_receipts(*, prelaunch: Path) -> list[dict[str, Any]]:
    stale: list[dict[str, Any]] = []
    for name in STALE_RECEIPT_NAMES:
        path = prelaunch / name
        if not path.is_file():
            continue
        stale.append(
            {
                "path": str(path),
                "superseded": True,
                "authority": False,
                "reason": "stale_mid_run_receipt",
            }
        )
    return stale


def replay_postrun_receipts(
    *,
    run_root: Path,
    packet_path: Path,
    require_barrier: bool = True,
) -> dict[str, Any]:
    failures: list[str] = []
    prelaunch = run_root / "prelaunch"
    prelaunch.mkdir(parents=True, exist_ok=True)
    packet = _load_packet(packet_path)

    barrier = assert_postrun_barrier_ready(run_root=run_root)
    if require_barrier and not barrier.get("pass"):
        return {
            "schema": REPLAY_SCHEMA,
            "run_root": str(run_root),
            "pass": False,
            "failures": list(barrier.get("failures") or []),
            "barrier": barrier,
            "receipts_written": False,
        }

    max_steps = _max_steps_hard(packet)
    path_receipt = evaluate_cap_selection_path_evidence(run_root=run_root)
    triage_receipt = triage_bounded_steps(
        run_root=run_root,
        max_steps_hard=max_steps,
    )
    classifier_receipt = classify_milestone_stall(run_root=run_root, packet=packet)

    (prelaunch / "milestone_stall_classifier_receipt.json").write_text(
        json.dumps(classifier_receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    terminal_receipt = emit_live_carrier_scale_smoke_receipt(
        run_root=run_root,
        packet_path=packet_path,
    )

    superseding_paths = {
        "cap_selection_path_evidence_receipt": prelaunch
        / "cap_selection_path_evidence_receipt_superseding.json",
        "bounded_steps_triage_receipt": prelaunch
        / "bounded_steps_triage_receipt_superseding.json",
        "milestone_stall_classifier_receipt": prelaunch
        / "milestone_stall_classifier_receipt_superseding.json",
        "live_carrier_scale_smoke_receipt": prelaunch
        / "live_carrier_scale_smoke_receipt_superseding.json",
    }
    _write_superseding(
        superseding_paths["cap_selection_path_evidence_receipt"],
        receipt_kind="cap_selection_path_evidence_receipt",
        run_root=run_root,
        payload=path_receipt,
    )
    _write_superseding(
        superseding_paths["bounded_steps_triage_receipt"],
        receipt_kind="bounded_steps_triage_receipt",
        run_root=run_root,
        payload=triage_receipt,
    )
    _write_superseding(
        superseding_paths["milestone_stall_classifier_receipt"],
        receipt_kind="milestone_stall_classifier_receipt",
        run_root=run_root,
        payload=classifier_receipt,
    )
    _write_superseding(
        superseding_paths["live_carrier_scale_smoke_receipt"],
        receipt_kind="live_carrier_scale_smoke_receipt",
        run_root=run_root,
        payload=terminal_receipt,
    )

    stale_marked = _mark_stale_receipts(prelaunch=prelaunch)

    return {
        "schema": REPLAY_SCHEMA,
        "run_root": str(run_root),
        "pass": True,
        "failures": failures,
        "barrier": barrier,
        "stale_receipts_marked": stale_marked,
        "receipts_written": True,
        "superseding_paths": {k: str(v) for k, v in superseding_paths.items()},
        "path_evidence": path_receipt,
        "bounded_steps_triage": triage_receipt,
        "classifier": classifier_receipt,
        "terminal_receipt": terminal_receipt,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument(
        "--skip-barrier",
        action="store_true",
        help="Fixture/test only: derive receipts without live PID barrier.",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    receipt = replay_postrun_receipts(
        run_root=args.run_root,
        packet_path=args.packet,
        require_barrier=not args.skip_barrier,
    )
    text = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0 if receipt.get("pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
