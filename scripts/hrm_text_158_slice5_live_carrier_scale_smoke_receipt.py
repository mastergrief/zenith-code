#!/usr/bin/env python3
"""Slice-5 live-carrier GPU scale-smoke terminal receipt (extracted from v6 replay heredoc)."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from scripts.hrm_text_158_slice5_milestone_stall_classifier import infer_sparse_authority

RECEIPT_SCHEMA = "hrm_text_158_slice5_live_carrier_gpu_scale_smoke_receipt/v1"
FEASIBLE_PEAK = 5.0
LIVENESS_TERMINALS = frozenset(
    {
        "LIVENESS_FAIL_KERNELIZED_BUT_STALLED",
        "SUBMILESTONE_INSTRUMENTATION_INVALID",
        "LIVENESS_FAIL",
        "LIVENESS_FAIL_TOTAL_TIMEOUT",
        "INSTRUMENTED_OUTER_BOUNDED_STEPS_TIMEOUT_AFTER_MAX_STEPS",
        "BASELINE_SPARSE_CAP_STEP_STALL",
    }
)


def _max_steps_hard(packet: dict[str, Any]) -> int:
    scale = packet.get("scale_smoke") or {}
    if scale.get("max_steps_hard") is not None:
        return int(scale["max_steps_hard"])
    if scale.get("steps") is not None:
        return int(scale["steps"])
    return 10


def helper_script_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _load_packet(packet_path: Path) -> dict[str, Any]:
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    if not packet.get("packet_revision"):
        raise ValueError(f"packet missing packet_revision: {packet_path}")
    if not packet.get("run_id"):
        raise ValueError(f"packet missing run_id: {packet_path}")
    return packet


def _m4_mode_from_packet(packet: dict[str, Any]) -> str:
    decision = packet.get("decision_contract") or {}
    if decision.get("m4_mode"):
        return str(decision["m4_mode"])
    revision = str(packet.get("packet_revision", ""))
    if revision.startswith("v6c_"):
        return "re_M4_phase_guard_classifier_extract"
    return "re_M4_post_submilestone_emit"


def emit_live_carrier_scale_smoke_receipt(
    *,
    run_root: Path,
    packet_path: Path,
) -> dict[str, Any]:
    packet = _load_packet(packet_path)
    prelaunch = run_root / "prelaunch"
    prelaunch.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []

    classifier_path = prelaunch / "milestone_stall_classifier_receipt.json"
    if not classifier_path.is_file():
        failures.append("missing_milestone_stall_classifier_receipt")
        classifier: dict[str, Any] = {}
    else:
        classifier = json.loads(classifier_path.read_text(encoding="utf-8"))

    injected_dispatch: dict[str, Any] = {}
    injected_path = prelaunch / "launch_injected_dispatch_receipt.json"
    witness_path = prelaunch / "launch_injected_dispatch_witness_receipt.json"
    if not injected_path.is_file():
        failures.append("missing_launch_injected_dispatch_receipt")
    else:
        injected_dispatch = json.loads(injected_path.read_text(encoding="utf-8"))
        if not injected_dispatch.get("dispatch_msg_id"):
            failures.append("launch_injected_dispatch_receipt_missing_id")
    if not witness_path.is_file():
        failures.append("missing_launch_injected_dispatch_witness_receipt")
    elif not json.loads(witness_path.read_text(encoding="utf-8")).get("pass"):
        failures.append("launch_injected_dispatch_witness_not_pass")

    classifier_classification = classifier.get("classification", "MISSING")
    stalled_sub_phase_id = classifier.get("stalled_sub_phase_id")
    stalled_parent_phase_id = classifier.get("stalled_parent_phase_id")
    phase_guard_locus = classifier.get("phase_guard_locus")
    milestone_locus = classifier.get("milestone_locus")

    baseline_rc = (
        int((prelaunch / "baseline_launch_rc.txt").read_text().strip())
        if (prelaunch / "baseline_launch_rc.txt").is_file()
        else None
    )
    instr_rc = (
        int((prelaunch / "instrumented_launch_rc.txt").read_text().strip())
        if (prelaunch / "instrumented_launch_rc.txt").is_file()
        else None
    )

    base_path = run_root / "baseline_snapshot_off" / "receipt.json"
    instr_path = run_root / "instrumented_snapshot_on" / "receipt.json"
    if not base_path.is_file():
        failures.append("missing_baseline_receipt")
    if not instr_path.is_file():
        failures.append("missing_instrumented_receipt")
    base = json.loads(base_path.read_text(encoding="utf-8")) if base_path.is_file() else {}
    instr = json.loads(instr_path.read_text(encoding="utf-8")) if instr_path.is_file() else {}

    for label, receipt in (("baseline", base), ("instrumented", instr)):
        if receipt and not receipt.get("persistent_accumulator_event_coded_live"):
            failures.append(f"{label}_missing_persistent_accumulator_event_coded_live")
    if instr.get("d_live_carrier_snapshot_enabled") is not True:
        failures.append("instrumented_missing_d_live_carrier_snapshot_enabled")
    if base.get("d_live_carrier_snapshot_enabled"):
        failures.append("baseline_had_snapshot_enabled")

    base_steps = base.get("step_reports") or {}
    instr_steps = instr.get("step_reports") or {}
    deltas: dict[str, float] = {}
    for key in sorted(set(base_steps) | set(instr_steps), key=lambda x: int(x)):
        b_step = base_steps.get(key) or {}
        i_step = instr_steps.get(key) or {}
        if "duration_seconds" in b_step and "duration_seconds" in i_step:
            deltas[key] = float(i_step["duration_seconds"]) - float(b_step["duration_seconds"])

    jsonl_path = (
        run_root
        / "instrumented_snapshot_on"
        / "d_recompute_window_diagnostic"
        / "live_carrier_snapshot.jsonl"
    )
    rows_by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
    if jsonl_path.is_file():
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows_by_step[int(row["step"])].append(row)
    else:
        failures.append("missing_live_carrier_snapshot_jsonl")

    total_rows = sum(len(v) for v in rows_by_step.values())
    classification = "PASS_COST_CHARACTERIZED"

    c8_baseline = int(
        (base.get("bounded_delta_global_summary") or {}).get("C8_TRANSIENT_DENSE_COMPUTE_NUMEL")
        or (base.get("global_summary") or {}).get("C8_TRANSIENT_DENSE_COMPUTE_NUMEL")
        or 0
    )
    sparse_auth_baseline = (base.get("bounded_delta_global_summary") or {}).get(
        "event_coded_sparse_vote_authority"
    )
    if base_path.is_file():
        if c8_baseline > 0:
            failures.append(f"baseline_c8_transient_dense_{c8_baseline}")
        if sparse_auth_baseline is not True:
            failures.append(f"baseline_sparse_authority_not_true:{sparse_auth_baseline!r}")
    elif not infer_sparse_authority(run_root, base):
        failures.append(f"baseline_sparse_authority_not_true:{sparse_auth_baseline!r}")

    peak_emit_delta = max(deltas.values()) if deltas else None
    emit_too_expensive = peak_emit_delta is not None and peak_emit_delta > FEASIBLE_PEAK

    max_steps_hard = _max_steps_hard(packet)
    steps_completed_base = int(base.get("steps_completed") or 0)
    steps_completed_instr = int(instr.get("steps_completed") or 0)
    baseline_clean = (baseline_rc in (0, None)) and steps_completed_base >= max_steps_hard
    instrumented_clean = (instr_rc in (0, None)) and steps_completed_instr >= max_steps_hard
    if baseline_rc not in (0, None):
        failures.append(f"baseline_launch_rc_{baseline_rc}")
    if instr_rc not in (0, None):
        failures.append(f"instrumented_launch_rc_{instr_rc}")

    if classifier_classification == "LIVENESS_FAIL_KERNELIZED_BUT_STALLED":
        classification = "LIVENESS_FAIL_KERNELIZED_BUT_STALLED"
    elif classifier_classification == "SUBMILESTONE_INSTRUMENTATION_INVALID":
        classification = "SUBMILESTONE_INSTRUMENTATION_INVALID"
    elif classifier_classification == "LIVENESS_FAIL_TOTAL_TIMEOUT":
        classification = "LIVENESS_FAIL_TOTAL_TIMEOUT"
    elif classifier_classification == "INSTRUMENTED_OUTER_BOUNDED_STEPS_TIMEOUT_AFTER_MAX_STEPS":
        classification = "INSTRUMENTED_OUTER_BOUNDED_STEPS_TIMEOUT_AFTER_MAX_STEPS"
    elif classifier_classification == "BASELINE_SPARSE_CAP_STEP_STALL":
        classification = "BASELINE_SPARSE_CAP_STEP_STALL"
    elif classifier_classification == "LIVENESS_FAIL":
        classification = "LIVENESS_FAIL"
    elif not baseline_clean or c8_baseline > 0 or (
        base_path.is_file() and sparse_auth_baseline is not True
    ):
        classification = "LIVENESS_FAIL"
    elif total_rows == 0:
        failures.append("SMOKE_INVALID_NO_EVENT_CODED_CARRIER")
        classification = "SMOKE_INVALID_NO_EVENT_CODED_CARRIER"
    else:
        steps_completed = int(instr.get("steps_completed") or 0)
        for step in range(1, steps_completed + 1):
            if len(rows_by_step.get(step, [])) < 1:
                failures.append(f"zero_rows_step_{step}")
                classification = "SMOKE_INVALID_NO_EVENT_CODED_CARRIER"
        if classification != "SMOKE_INVALID_NO_EVENT_CODED_CARRIER":
            if not instrumented_clean:
                classification = "SMOKE_INSTRUMENTED_TIMEOUT_POSSIBLE_EMIT_OVERHEAD"
            elif emit_too_expensive:
                classification = "EMIT_PATH_TOO_EXPENSIVE"
            elif peak_emit_delta is not None and peak_emit_delta <= FEASIBLE_PEAK and total_rows > 0:
                classification = "EMIT_FEASIBLE_FOR_STEP2B_SCALE_SMOKE"
            else:
                classification = "PASS_COST_CHARACTERIZED"

    receipt = {
        "schema": RECEIPT_SCHEMA,
        "packet_revision": packet["packet_revision"],
        "m4_mode": _m4_mode_from_packet(packet),
        "run_id": packet["run_id"],
        "helper_script_sha256": helper_script_sha256(),
        "classifier_classification": classifier_classification,
        "stalled_sub_phase_id": stalled_sub_phase_id,
        "stalled_parent_phase_id": stalled_parent_phase_id,
        "phase_guard_locus": phase_guard_locus,
        "milestone_locus": milestone_locus,
        "baseline_launch_rc": baseline_rc,
        "instrumented_launch_rc": instr_rc,
        "persistent_accumulator_event_coded_live_baseline": base.get(
            "persistent_accumulator_event_coded_live"
        ),
        "persistent_accumulator_event_coded_live_instrumented": instr.get(
            "persistent_accumulator_event_coded_live"
        ),
        "total_live_carrier_snapshot_rows": total_rows,
        "emit_rows_per_step": {str(k): len(v) for k, v in sorted(rows_by_step.items())},
        "per_step_duration_delta_seconds": deltas,
        "peak_per_step_emit_delta_seconds": peak_emit_delta,
        "classification": classification,
        "emit_too_expensive": bool(emit_too_expensive),
        "max_steps_hard": int(max_steps_hard),
        "sampler_non_perturbation_pass": classifier.get("sampler_non_perturbation_pass"),
        "ring_attribution_eligible": classifier.get("ring_attribution_eligible"),
        "dispatch_msg_id_authority": packet.get("dispatch_msg_id_authority"),
        "packet_dispatch_msg_id_sentinel": packet.get("dispatch_msg_id"),
        "injected_dispatch_msg_id": injected_dispatch.get("dispatch_msg_id"),
        "dispatch_run_status_at_preflight": injected_dispatch.get("dispatch_run_status"),
        "dispatch_claim_run_root": injected_dispatch.get("marker_run_root"),
        "dispatch_intended_run_root": injected_dispatch.get("intended_run_root"),
        "dispatch_intended_run_id": injected_dispatch.get("intended_run_id"),
        "dispatch_issuer": injected_dispatch.get("issuer"),
        "pass": not failures
        and classification
        in (
            "EMIT_FEASIBLE_FOR_STEP2B_SCALE_SMOKE",
            "EMIT_PATH_TOO_EXPENSIVE",
            "PASS_COST_CHARACTERIZED",
        ),
        "failures": failures,
    }
    out = prelaunch / "live_carrier_scale_smoke_receipt.json"
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
        receipt = emit_live_carrier_scale_smoke_receipt(
            run_root=args.run_root,
            packet_path=args.packet,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"receipt error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
