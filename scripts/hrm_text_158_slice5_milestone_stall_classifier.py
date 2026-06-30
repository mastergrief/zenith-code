#!/usr/bin/env python3
"""Slice-5 milestone stall classifier (extracted from v6 replay heredoc)."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

CLASSIFIER_RECEIPT_SCHEMA = "hrm_text_158_slice5_milestone_stall_classifier_receipt/v3"
PHASES = (
    "step_forward_backward",
    "sparse_vote_construction",
    "sparse_cap_apply",
    "live_carrier_snapshot_emit",
    "artifact_flush",
)
BUDGETS = {
    "step_forward_backward": 90,
    "sparse_vote_construction": 120,
    "sparse_cap_apply": 180,
    "live_carrier_snapshot_emit": 60,
    "artifact_flush": 60,
}
SUB_PHASES = (
    ("cap_selection_cpu_copy", "sparse_cap_apply_cap_selection_cpu_copy.jsonl", 45),
    ("post_cap_apply_sync", "sparse_cap_apply_post_cap_apply_sync.jsonl", 120),
    ("boundary_normalize", "sparse_cap_apply_boundary_normalize.jsonl", 45),
)
ARMS = ("baseline_snapshot_off", "instrumented_snapshot_on")
FLAT_COUNTER_REASONS = frozenset({"parent_phase_flat_counter", "sub_phase_flat_counter"})


def helper_script_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_packet(packet_path: Path) -> dict[str, Any]:
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    if not packet.get("packet_revision"):
        raise ValueError(f"packet missing packet_revision: {packet_path}")
    if not packet.get("run_id"):
        raise ValueError(f"packet missing run_id: {packet_path}")
    return packet


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def infer_sparse_authority(run_root: Path, receipt: dict[str, Any]) -> bool:
    summary = receipt.get("bounded_delta_global_summary") or {}
    if "event_coded_sparse_vote_authority" in summary:
        return bool(summary.get("event_coded_sparse_vote_authority"))
    prelaunch = run_root / "prelaunch"
    for witness_name in (
        "probe_cli_sparse_authority_flag_witness.json",
        "probe_sparse_authority_wiring_witness.json",
    ):
        witness_path = prelaunch / witness_name
        if witness_path.is_file():
            witness = json.loads(witness_path.read_text(encoding="utf-8"))
            if witness.get("pass"):
                return True
    return False


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _milestone_counters(rows: list[dict[str, Any]]) -> list[int]:
    counters: list[int] = []
    for row in rows:
        if row.get("schema") != "hrm_text_158_phase_milestone_counter/v1":
            continue
        counters.append(int(row.get("milestone_counter", -1)))
    return counters


def _counters_monotonic(counters: list[int]) -> bool:
    return all(counters[i] >= counters[i - 1] for i in range(1, len(counters)))


def _has_monotonic_progress_after_sub(scratch: Path, sub_id: str) -> bool:
    order = [entry[0] for entry in SUB_PHASES]
    if sub_id not in order:
        return False
    later_names = {
        entry[0]: entry[1] for entry in SUB_PHASES[order.index(sub_id) + 1 :]
    }
    for jsonl_name in later_names.values():
        counters = _milestone_counters(_read_jsonl(scratch / "liveness_milestones" / jsonl_name))
        if len(counters) >= 2 and _counters_monotonic(counters) and counters[-1] > counters[0]:
            return True
    return False


def _detect_phase_guard_kill(scratch: Path) -> str | None:
    lap_path = scratch / "last_active_phase.json"
    if not lap_path.is_file():
        return None
    lap = json.loads(lap_path.read_text(encoding="utf-8"))
    if lap.get("failure_class") != "LIVENESS_FAILURE":
        return None
    if lap.get("event") != "active_phase_guard":
        return None
    phase = lap.get("phase")
    return str(phase) if phase else None


def _analyze_parent_phase(
    *,
    arm: str,
    phase_id: str,
    scratch: Path,
    sparse_auth: bool,
    cuda: bool,
    failures: list[str],
    stall_hits: list[dict[str, Any]],
    instrumentation_invalid: list[dict[str, Any]],
    waive_artifact_flush: bool,
) -> None:
    if phase_id == "live_carrier_snapshot_emit" and arm == "baseline_snapshot_off":
        return
    if phase_id == "artifact_flush" and waive_artifact_flush:
        return

    path = scratch / "liveness_milestones" / f"{phase_id}.jsonl"
    if not path.is_file():
        if phase_id == "sparse_cap_apply" and sparse_auth and cuda:
            stall_hits.append(
                {
                    "arm": arm,
                    "phase_id": phase_id,
                    "reason": "missing_parent_sparse_cap_jsonl",
                }
            )
        else:
            failures.append(f"missing_milestone_jsonl:{arm}:{phase_id}")
        return

    rows = _read_jsonl(path)
    if not rows:
        failures.append(f"empty_milestone_jsonl:{arm}:{phase_id}")
        return

    counters = _milestone_counters(rows)
    for row in rows:
        if row.get("schema") != "hrm_text_158_phase_milestone_counter/v1":
            instrumentation_invalid.append({"arm": arm, "phase_id": phase_id, "reason": "bad_schema"})
    if counters and not _counters_monotonic(counters):
        instrumentation_invalid.append({"arm": arm, "phase_id": phase_id, "reason": "non_monotonic"})

    elapsed = [
        float(row["elapsed_since_phase_enter_seconds"])
        for row in rows
        if "elapsed_since_phase_enter_seconds" in row
    ]
    if (
        elapsed
        and counters
        and (max(elapsed) - min(elapsed)) >= BUDGETS[phase_id]
        and max(counters) == min(counters)
    ):
        stall_hits.append(
            {
                "arm": arm,
                "phase_id": phase_id,
                "budget": BUDGETS[phase_id],
                "reason": "parent_phase_flat_counter",
            }
        )


def _analyze_sub_phases(
    *,
    arm: str,
    scratch: Path,
    sparse_auth: bool,
    cuda: bool,
    parent_sparse_cap_complete: bool,
    stall_hits: list[dict[str, Any]],
    instrumentation_invalid: list[dict[str, Any]],
) -> None:
    if not (sparse_auth and cuda):
        return

    for sub_id, jsonl_name, sub_budget in SUB_PHASES:
        sub_path = scratch / "liveness_milestones" / jsonl_name
        if not sub_path.is_file():
            if _has_monotonic_progress_after_sub(scratch, sub_id):
                continue
            if not parent_sparse_cap_complete:
                continue
            instrumentation_invalid.append(
                {"arm": arm, "sub_phase_id": sub_id, "reason": "missing_after_parent_complete"}
            )
            continue

        sub_rows = _read_jsonl(sub_path)
        if not sub_rows:
            if not parent_sparse_cap_complete:
                continue
            instrumentation_invalid.append(
                {"arm": arm, "sub_phase_id": sub_id, "reason": "empty_after_parent_complete"}
            )
            continue

        sub_counters = _milestone_counters(sub_rows)
        for row in sub_rows:
            if row.get("schema") != "hrm_text_158_phase_milestone_counter/v1":
                instrumentation_invalid.append(
                    {"arm": arm, "sub_phase_id": sub_id, "reason": "bad_schema"}
                )
        sub_elapsed = [
            float(row["elapsed_since_phase_enter_seconds"])
            for row in sub_rows
            if "elapsed_since_phase_enter_seconds" in row
        ]
        if (
            sub_elapsed
            and sub_counters
            and (max(sub_elapsed) - min(sub_elapsed)) >= sub_budget
            and max(sub_counters) == min(sub_counters)
        ):
            stall_hits.append(
                {
                    "arm": arm,
                    "sub_phase_id": sub_id,
                    "reason": "sub_phase_flat_counter",
                    "budget": sub_budget,
                }
            )


def _derive_stall_ids(stall_hits: list[dict[str, Any]]) -> tuple[str | None, str | None]:
    for hit in stall_hits:
        if hit.get("reason") not in FLAT_COUNTER_REASONS:
            continue
        if hit.get("sub_phase_id"):
            return str(hit["sub_phase_id"]), "sparse_cap_apply"
        if hit.get("phase_id"):
            return None, str(hit["phase_id"])
    return None, None


def _derive_milestone_locus(stall_hits: list[dict[str, Any]]) -> str | None:
    for hit in stall_hits:
        if hit.get("reason") not in FLAT_COUNTER_REASONS:
            continue
        if hit.get("sub_phase_id"):
            return str(hit["sub_phase_id"])
        if hit.get("phase_id"):
            return str(hit["phase_id"])
    return None


def classify_milestone_stall(
    *,
    run_root: Path,
    packet: dict[str, Any],
) -> dict[str, Any]:
    failures: list[str] = []
    stall_hits: list[dict[str, Any]] = []
    instrumentation_invalid: list[dict[str, Any]] = []
    phase_guard_loci: list[str] = []

    cuda = cuda_available()
    for arm in ARMS:
        scratch = run_root / arm
        guard_locus = _detect_phase_guard_kill(scratch)
        if guard_locus:
            phase_guard_loci.append(guard_locus)

    probe_phase_guard_kill = bool(phase_guard_loci)
    phase_guard_locus = phase_guard_loci[0] if phase_guard_loci else None
    waive_artifact_flush = probe_phase_guard_kill

    for arm in ARMS:
        scratch = run_root / arm
        receipt_path = scratch / "receipt.json"
        receipt = (
            json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt_path.is_file()
            else {}
        )
        sparse_auth = infer_sparse_authority(run_root, receipt)
        parent_sparse_cap_rows = _read_jsonl(
            scratch / "liveness_milestones" / "sparse_cap_apply.jsonl"
        )
        parent_sparse_cap_complete = bool(parent_sparse_cap_rows)

        for phase_id in PHASES:
            _analyze_parent_phase(
                arm=arm,
                phase_id=phase_id,
                scratch=scratch,
                sparse_auth=sparse_auth,
                cuda=cuda,
                failures=failures,
                stall_hits=stall_hits,
                instrumentation_invalid=instrumentation_invalid,
                waive_artifact_flush=waive_artifact_flush,
            )

        _analyze_sub_phases(
            arm=arm,
            scratch=scratch,
            sparse_auth=sparse_auth,
            cuda=cuda,
            parent_sparse_cap_complete=parent_sparse_cap_complete,
            stall_hits=stall_hits,
            instrumentation_invalid=instrumentation_invalid,
        )

    stalled_sub_phase_id, stalled_parent_phase_id = _derive_stall_ids(stall_hits)
    milestone_locus = _derive_milestone_locus(stall_hits)

    classification = "PASS_MILESTONE_REPLAY"
    if instrumentation_invalid and not stall_hits:
        classification = "SUBMILESTONE_INSTRUMENTATION_INVALID"
    elif stall_hits:
        classification = "LIVENESS_FAIL_KERNELIZED_BUT_STALLED"
    elif probe_phase_guard_kill:
        classification = "LIVENESS_FAIL"
    elif failures:
        classification = "MILESTONE_ARTIFACT_INCOMPLETE"

    return {
        "schema": CLASSIFIER_RECEIPT_SCHEMA,
        "packet_revision": packet["packet_revision"],
        "run_id": packet["run_id"],
        "helper_script_sha256": helper_script_sha256(),
        "classification": classification,
        "stall_hits": stall_hits,
        "instrumentation_invalid": instrumentation_invalid,
        "stalled_sub_phase_id": stalled_sub_phase_id,
        "stalled_parent_phase_id": stalled_parent_phase_id,
        "phase_guard_locus": phase_guard_locus,
        "milestone_locus": milestone_locus,
        "probe_phase_guard_kill": probe_phase_guard_kill,
        "cuda_available": cuda,
        "pass": classification == "PASS_MILESTONE_REPLAY",
        "failures": failures,
    }


def emit_classifier_receipt(
    *,
    run_root: Path,
    packet_path: Path,
) -> dict[str, Any]:
    packet = _load_packet(packet_path)
    receipt = classify_milestone_stall(run_root=run_root, packet=packet)
    prelaunch = run_root / "prelaunch"
    prelaunch.mkdir(parents=True, exist_ok=True)
    out = prelaunch / "milestone_stall_classifier_receipt.json"
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--packet",
        type=Path,
        default=Path(
            "artifacts/consensus_prep/slice5_step2a_live_carrier_gpu_scale_smoke_launch_packet_v6_draft.json"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        receipt = emit_classifier_receipt(run_root=args.run_root, packet_path=args.packet)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"classifier error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
