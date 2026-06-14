#!/usr/bin/env python3
"""CPU-only scale smoke for S3c slim receipt emit + chunked wiring sidecars."""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path
from typing import Any

from calm.hrm_text_158.native_full_stack.s3bb_headroom_telemetry import (
    HEADROOM_TELEMETRY_SCHEMA_VERSION,
    HEADROOM_WIRING_SIDECAR_SCHEMA_VERSION,
    RECEIPT_EMIT_PROFILE_SLIM,
    WARMUP_STEPS,
    append_headroom_wiring_sidecar_chunk,
    compare_arm_wiring_guards,
    compare_arm_wiring_guards_streaming,
)

DEFAULT_MODULES = 32
DEFAULT_LANES_PER_MODULE = 1_048_576
DEFAULT_MEASURED_STEPS = 18
DEFAULT_WARMUP_STEPS = 2

GATES = {
    "receipt_json_dumps_wall_seconds_lte": 60.0,
    "receipt_emit_peak_rss_mib_lte": 4096.0,
    "receipt_file_size_mib_lte": 50.0,
    "sidecar_emit_wall_seconds_lte": 180.0,
    "sidecar_emit_peak_rss_mib_lte": 512.0,
    "sidecar_file_size_bytes_lte": 5 * 1024 * 1024 * 1024,
    "sidecar_compare_wall_seconds_lte": 300.0,
    "sidecar_compare_peak_rss_mib_lte": 512.0,
}


def _read_rss_mib() -> float:
    status_path = Path("/proc/self/status")
    if status_path.is_file():
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return float(usage.ru_maxrss) / 1024.0


class _PhaseRssTracker:
    def __init__(self) -> None:
        self._baseline_mib = _read_rss_mib()
        self.peak_delta_mib = 0.0

    def sample(self) -> None:
        current_mib = _read_rss_mib()
        self.peak_delta_mib = max(self.peak_delta_mib, current_mib - self._baseline_mib)


def _synthetic_lane_values(
    *,
    module_index: int,
    step: int,
    lanes_per_module: int,
    arm_offset: int,
) -> tuple[list[int], list[int]]:
  acc = [((module_index + step + lane + arm_offset) % 31) - 15 for lane in range(lanes_per_module)]
  q = [((lane + module_index) % 3) - 1 for lane in range(lanes_per_module)]
  return acc, q


def _build_slim_receipt(
    *,
    modules: int,
    measured_steps: int,
    warmup_steps: int,
    lanes_per_module: int,
    sidecar_path: Path,
) -> dict[str, Any]:
    step_reports: dict[str, Any] = {}
    for step in range(1, warmup_steps + measured_steps + 1):
        step_reports[str(step)] = {
            "headroom_telemetry": {
                "schema_version": HEADROOM_TELEMETRY_SCHEMA_VERSION,
                "global_max_abs_accumulator": 21,
                "margin_to_w6_boundary_min": 10,
                "lanes_within_K_of_boundary_fraction": 0.0,
                "out_of_domain_lane_count": 0,
                "would_strict_raise_step": False,
                "strict_raise_count": 0,
                "boundary_value_error_caught": False,
                "eligible_module_count": modules,
                "total_lane_count": modules * lanes_per_module,
            }
        }
    return {
        "schema": "hrm_text_158_c2p1_real_model_bounded_delta_probe/v0",
        "phase": "s3bb-w6-headroom-diagnostic",
        "receipt_emit_profile": RECEIPT_EMIT_PROFILE_SLIM,
        "steps_completed": warmup_steps + measured_steps,
        "stop_reason": "max_steps_completed",
        "step_reports": step_reports,
        "headroom_wiring_sidecar_path": str(sidecar_path),
        "headroom_wiring_sidecar_schema": HEADROOM_WIRING_SIDECAR_SCHEMA_VERSION,
        "checkpoint_payload": {
            "checkpoint_payload_omitted": True,
            "reason": RECEIPT_EMIT_PROFILE_SLIM,
            "tensor_count": modules,
            "dry_run": True,
        },
    }


def _emit_sidecar(
    *,
    sidecar_path: Path,
    modules: int,
    measured_steps: int,
    warmup_steps: int,
    lanes_per_module: int,
    arm_offset: int,
    rss_tracker: _PhaseRssTracker | None = None,
) -> None:
    if sidecar_path.is_file():
        sidecar_path.unlink()
    for step in range(warmup_steps + 1, warmup_steps + measured_steps + 1):
        for module_index in range(modules):
            state_key = f"model.module_{module_index}"
            acc, q = _synthetic_lane_values(
                module_index=module_index,
                step=step,
                lanes_per_module=lanes_per_module,
                arm_offset=arm_offset,
            )
            append_headroom_wiring_sidecar_chunk(
                sidecar_path,
                step=int(step),
                state_key=state_key,
                accumulator_lanes=acc,
                q_lanes=q,
            )
            if rss_tracker is not None:
                rss_tracker.sample()


def run_scale_smoke(
    *,
    output_dir: Path,
    modules: int,
    lanes_per_module: int,
    measured_steps: int,
    warmup_steps: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    oracle_sidecar = output_dir / "oracle_headroom_wiring_sidecar.jsonl"
    treatment_sidecar = output_dir / "treatment_headroom_wiring_sidecar.jsonl"
    receipt_path = output_dir / "receipt.json"

    sidecar_emit_start = time.perf_counter()
    sidecar_rss = _PhaseRssTracker()
    _emit_sidecar(
        sidecar_path=oracle_sidecar,
        modules=modules,
        measured_steps=measured_steps,
        warmup_steps=warmup_steps,
        lanes_per_module=lanes_per_module,
        arm_offset=0,
        rss_tracker=sidecar_rss,
    )
    _emit_sidecar(
        sidecar_path=treatment_sidecar,
        modules=modules,
        measured_steps=measured_steps,
        warmup_steps=warmup_steps,
        lanes_per_module=lanes_per_module,
        arm_offset=0,
        rss_tracker=sidecar_rss,
    )
    sidecar_emit_seconds = time.perf_counter() - sidecar_emit_start
    sidecar_emit_peak_rss_mib = sidecar_rss.peak_delta_mib
    sidecar_file_size_bytes = oracle_sidecar.stat().st_size + treatment_sidecar.stat().st_size

    oracle_receipt = _build_slim_receipt(
        modules=modules,
        measured_steps=measured_steps,
        warmup_steps=warmup_steps,
        lanes_per_module=lanes_per_module,
        sidecar_path=oracle_sidecar,
    )
    treatment_receipt = _build_slim_receipt(
        modules=modules,
        measured_steps=measured_steps,
        warmup_steps=warmup_steps,
        lanes_per_module=lanes_per_module,
        sidecar_path=treatment_sidecar,
    )

    receipt_emit_start = time.perf_counter()
    receipt_rss = _PhaseRssTracker()
    receipt_text = json.dumps(oracle_receipt, separators=(",", ":"), sort_keys=True)
    receipt_rss.sample()
    receipt_path.write_text(receipt_text, encoding="utf-8")
    receipt_rss.sample()
    receipt_emit_seconds = time.perf_counter() - receipt_emit_start
    receipt_emit_peak_rss_mib = receipt_rss.peak_delta_mib
    receipt_file_size_mib = receipt_path.stat().st_size / (1024.0 * 1024.0)

    compare_start = time.perf_counter()
    compare_rss = _PhaseRssTracker()
    guards = compare_arm_wiring_guards_streaming(
        oracle_receipt,
        treatment_receipt,
        oracle_sidecar_path=oracle_sidecar,
        treatment_sidecar_path=treatment_sidecar,
    )
    compare_rss.sample()
    compare_seconds = time.perf_counter() - compare_start
    compare_peak_rss_mib = compare_rss.peak_delta_mib

    metrics = {
        "modules": modules,
        "lanes_per_module": lanes_per_module,
        "measured_steps": measured_steps,
        "warmup_steps": warmup_steps,
        "receipt_json_dumps_wall_seconds": receipt_emit_seconds,
        "receipt_emit_peak_rss_mib": receipt_emit_peak_rss_mib,
        "receipt_file_size_mib": receipt_file_size_mib,
        "sidecar_emit_wall_seconds": sidecar_emit_seconds,
        "sidecar_emit_peak_rss_mib": sidecar_emit_peak_rss_mib,
        "sidecar_file_size_bytes": sidecar_file_size_bytes,
        "sidecar_compare_wall_seconds": compare_seconds,
        "sidecar_compare_peak_rss_mib": compare_peak_rss_mib,
        "wiring_guards": guards,
        "gates": GATES,
    }
    failures: list[str] = []
    if receipt_emit_seconds > GATES["receipt_json_dumps_wall_seconds_lte"]:
        failures.append("receipt_json_dumps_wall_seconds")
    if receipt_emit_peak_rss_mib > GATES["receipt_emit_peak_rss_mib_lte"]:
        failures.append("receipt_emit_peak_rss_mib")
    if receipt_file_size_mib > GATES["receipt_file_size_mib_lte"]:
        failures.append("receipt_file_size_mib")
    if sidecar_emit_seconds > GATES["sidecar_emit_wall_seconds_lte"]:
        failures.append("sidecar_emit_wall_seconds")
    if sidecar_emit_peak_rss_mib > GATES["sidecar_emit_peak_rss_mib_lte"]:
        failures.append("sidecar_emit_peak_rss_mib")
    if sidecar_file_size_bytes > GATES["sidecar_file_size_bytes_lte"]:
        failures.append("sidecar_file_size_bytes")
    if compare_seconds > GATES["sidecar_compare_wall_seconds_lte"]:
        failures.append("sidecar_compare_wall_seconds")
    if compare_peak_rss_mib > GATES["sidecar_compare_peak_rss_mib_lte"]:
        failures.append("sidecar_compare_peak_rss_mib")
    metrics["failures"] = failures
    metrics["passed"] = not failures
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--modules", type=int, default=DEFAULT_MODULES)
    parser.add_argument("--lanes-per-module", type=int, default=DEFAULT_LANES_PER_MODULE)
    parser.add_argument("--measured-steps", type=int, default=DEFAULT_MEASURED_STEPS)
    parser.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)

    metrics = run_scale_smoke(
        output_dir=args.output_dir,
        modules=int(args.modules),
        lanes_per_module=int(args.lanes_per_module),
        measured_steps=int(args.measured_steps),
        warmup_steps=int(args.warmup_steps),
    )
    payload = json.dumps(metrics, indent=2, sort_keys=True)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(payload, encoding="utf-8")
    print(payload)
    return 0 if metrics["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
