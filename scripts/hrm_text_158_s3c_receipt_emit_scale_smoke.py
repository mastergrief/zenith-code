#!/usr/bin/env python3
"""CPU-only scale smoke for S3c slim receipt emit + chunked wiring sidecars."""
from __future__ import annotations

import argparse
import json
import resource
import shutil
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from calm.hrm_text_158.native_full_stack.s3bb_headroom_telemetry import (
    HEADROOM_TELEMETRY_SCHEMA_VERSION,
    HEADROOM_WIRING_SIDECAR_SCHEMA_VERSION,
    RECEIPT_EMIT_PROFILE_SLIM,
    WARMUP_STEPS,
    append_headroom_wiring_sidecar_chunk,
    compare_arm_wiring_guards_streaming,
    compute_headroom_telemetry_from_accumulators,
)

METRICS_SCHEMA_VERSION = "hrm_text_158_s3c_receipt_emit_scale_smoke/v2.1"
CREDITDIR_MARKERS = ("claw-code-creditdir", "transient_fp_credit")

DEFAULT_MODULES = 32
DEFAULT_LANES_PER_MODULE = 1_048_576
DEFAULT_MEASURED_STEPS = 18
DEFAULT_WARMUP_STEPS = 2

# phase_c 575s: dual-review budget-accept after full-default x18 measured ~523s at v2
# (json.loads-per-line streaming compare; postrun-only, not training hot-loop).
GATES = {
    "receipt_json_dumps_wall_seconds_lte": 60.0,
    "receipt_emit_peak_rss_mib_lte": 4096.0,
    "receipt_file_size_mib_lte": 50.0,
    "phase_b_sidecar_emit_wall_seconds_per_arm_max_lte": 180.0,
    "phase_b_sidecar_emit_peak_rss_mib_lte": 512.0,
    "phase_b_prime_aggregate_peak_rss_mib_lte": 512.0,
    "phase_c_compare_wall_seconds_lte": 575.0,
    "phase_c_compare_peak_rss_mib_lte": 512.0,
    "sidecar_file_size_bytes_lte": 7_000_000_000,
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


def _validate_output_dir(output_dir: Path, *, require_creditdir: bool) -> None:
    resolved = str(output_dir.resolve())
    if require_creditdir and not any(marker in resolved for marker in CREDITDIR_MARKERS):
        raise ValueError(
            f"output_dir must live under credit/science tree ({CREDITDIR_MARKERS}), got {resolved!r}"
        )


def _disk_precheck(output_dir: Path, *, sidecar_file_size_budget_bytes: int) -> None:
    parent = output_dir if output_dir.exists() else output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(parent)
    required = int(1.2 * sidecar_file_size_budget_bytes)
    if usage.free < required:
        raise OSError(
            f"insufficient disk free bytes: need>={required}, have={usage.free} at {parent}"
        )


def _synthetic_lane_arrays(
    *,
    module_index: int,
    step: int,
    lanes_per_module: int,
    arm_offset: int,
) -> tuple[np.ndarray, np.ndarray]:
    lanes = np.arange(lanes_per_module, dtype=np.int64)
    acc = ((module_index + step + lanes + arm_offset) % 31) - 15
    q = ((lanes + module_index) % 3) - 1
    return acc.astype(np.int16), q.astype(np.int16)


def _mirror_probe_aggregate_rss_sample(
    *,
    modules: int,
    measured_steps: int,
    warmup_steps: int,
    lanes_per_module: int,
    arm_offset: int,
    rss_tracker: _PhaseRssTracker,
) -> float:
    wall_start = time.perf_counter()
    for step in range(warmup_steps + 1, warmup_steps + measured_steps + 1):
        flat_pieces = []
        for module_index in range(modules):
            acc, _q = _synthetic_lane_arrays(
                module_index=module_index,
                step=step,
                lanes_per_module=lanes_per_module,
                arm_offset=arm_offset,
            )
            flat_pieces.append(torch.from_numpy(acc))
        concatenated = torch.cat(flat_pieces)
        compute_headroom_telemetry_from_accumulators(concatenated)
        rss_tracker.sample()
    return time.perf_counter() - wall_start


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
    phase_a_seconds: list[float],
    rss_tracker: _PhaseRssTracker | None = None,
    lane_factory: Callable[..., tuple[np.ndarray, np.ndarray]] | None = None,
) -> float:
    factory = lane_factory or _synthetic_lane_arrays
    if sidecar_path.is_file():
        sidecar_path.unlink()
    phase_b_seconds = 0.0
    for step in range(warmup_steps + 1, warmup_steps + measured_steps + 1):
        for module_index in range(modules):
            gen_start = time.perf_counter()
            acc_arr, q_arr = factory(
                module_index=module_index,
                step=step,
                lanes_per_module=lanes_per_module,
                arm_offset=arm_offset,
            )
            phase_a_seconds.append(time.perf_counter() - gen_start)
            emit_start = time.perf_counter()
            acc = acc_arr.tolist()
            q = q_arr.tolist()
            append_headroom_wiring_sidecar_chunk(
                sidecar_path,
                step=int(step),
                state_key=f"model.module_{module_index}",
                accumulator_lanes=acc,
                q_lanes=q,
            )
            phase_b_seconds += time.perf_counter() - emit_start
            if rss_tracker is not None:
                rss_tracker.sample()
    return phase_b_seconds


def _cleanup_sidecars(*paths: Path) -> None:
    for path in paths:
        if path.is_file():
            path.unlink()


def run_scale_smoke(
    *,
    output_dir: Path,
    modules: int,
    lanes_per_module: int,
    measured_steps: int,
    warmup_steps: int,
    require_creditdir: bool = False,
    cleanup_sidecars: bool = True,
    gate_overrides: dict[str, float] | None = None,
    lane_factory: Callable[..., tuple[np.ndarray, np.ndarray]] | None = None,
) -> dict[str, Any]:
    _validate_output_dir(output_dir, require_creditdir=require_creditdir)
    effective_gates = dict(GATES)
    if gate_overrides:
        effective_gates.update(gate_overrides)
    _disk_precheck(
        output_dir,
        sidecar_file_size_budget_bytes=int(effective_gates["sidecar_file_size_bytes_lte"]),
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    oracle_sidecar = output_dir / "oracle_headroom_wiring_sidecar.jsonl"
    treatment_sidecar = output_dir / "treatment_headroom_wiring_sidecar.jsonl"
    receipt_path = output_dir / "receipt.json"

    phase_a_seconds: list[float] = []
    aggregate_rss = _PhaseRssTracker()
    phase_b_prime_wall_seconds = _mirror_probe_aggregate_rss_sample(
        modules=modules,
        measured_steps=measured_steps,
        warmup_steps=warmup_steps,
        lanes_per_module=lanes_per_module,
        arm_offset=0,
        rss_tracker=aggregate_rss,
    )

    sidecar_rss = _PhaseRssTracker()
    oracle_phase_b_seconds = _emit_sidecar(
        sidecar_path=oracle_sidecar,
        modules=modules,
        measured_steps=measured_steps,
        warmup_steps=warmup_steps,
        lanes_per_module=lanes_per_module,
        arm_offset=0,
        phase_a_seconds=phase_a_seconds,
        rss_tracker=sidecar_rss,
        lane_factory=lane_factory,
    )
    treatment_phase_b_seconds = _emit_sidecar(
        sidecar_path=treatment_sidecar,
        modules=modules,
        measured_steps=measured_steps,
        warmup_steps=warmup_steps,
        lanes_per_module=lanes_per_module,
        arm_offset=0,
        phase_a_seconds=phase_a_seconds,
        rss_tracker=sidecar_rss,
        lane_factory=lane_factory,
    )
    phase_b_by_arm = {
        "oracle": float(oracle_phase_b_seconds),
        "treatment": float(treatment_phase_b_seconds),
    }
    phase_b_per_arm_max_seconds = max(phase_b_by_arm.values())
    phase_b_total_seconds = float(oracle_phase_b_seconds + treatment_phase_b_seconds)
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

    compare_start = time.perf_counter()
    compare_rss = _PhaseRssTracker()
    guards = compare_arm_wiring_guards_streaming(
        oracle_receipt,
        treatment_receipt,
        oracle_sidecar_path=oracle_sidecar,
        treatment_sidecar_path=treatment_sidecar,
    )
    compare_rss.sample()
    phase_c_compare_wall_seconds = time.perf_counter() - compare_start

    metrics: dict[str, Any] = {
        "metrics_schema_version": METRICS_SCHEMA_VERSION,
        "modules": modules,
        "lanes_per_module": lanes_per_module,
        "measured_steps": measured_steps,
        "warmup_steps": warmup_steps,
        "phase_a_data_gen_wall_seconds": float(sum(phase_a_seconds)),
        "phase_b_prime_aggregate_wall_seconds": float(phase_b_prime_wall_seconds),
        "phase_b_prime_aggregate_peak_rss_mib": float(aggregate_rss.peak_delta_mib),
        "phase_b_sidecar_emit_wall_seconds_by_arm": phase_b_by_arm,
        "phase_b_sidecar_emit_wall_seconds_per_arm_max": float(phase_b_per_arm_max_seconds),
        "phase_b_sidecar_emit_wall_seconds_total": float(phase_b_total_seconds),
        "phase_b_sidecar_emit_peak_rss_mib": float(sidecar_rss.peak_delta_mib),
        "phase_c_compare_wall_seconds": float(phase_c_compare_wall_seconds),
        "phase_c_compare_peak_rss_mib": float(compare_rss.peak_delta_mib),
        "receipt_json_dumps_wall_seconds": float(receipt_emit_seconds),
        "receipt_emit_peak_rss_mib": float(receipt_rss.peak_delta_mib),
        "receipt_file_size_mib": float(receipt_path.stat().st_size / (1024.0 * 1024.0)),
        "sidecar_file_size_bytes": int(sidecar_file_size_bytes),
        "per_phase_record_counts": {
            "sidecar_records_per_arm": modules * measured_steps,
            "total_lanes_compared": modules * lanes_per_module * measured_steps,
        },
        "wiring_guards": guards,
        "gates": effective_gates,
    }
    if gate_overrides:
        metrics["gate_overrides_applied"] = dict(gate_overrides)

    failures: list[str] = []
    for metric_key, gate_key in (
        ("receipt_json_dumps_wall_seconds", "receipt_json_dumps_wall_seconds_lte"),
        ("receipt_emit_peak_rss_mib", "receipt_emit_peak_rss_mib_lte"),
        ("receipt_file_size_mib", "receipt_file_size_mib_lte"),
        (
            "phase_b_sidecar_emit_wall_seconds_per_arm_max",
            "phase_b_sidecar_emit_wall_seconds_per_arm_max_lte",
        ),
        ("phase_b_sidecar_emit_peak_rss_mib", "phase_b_sidecar_emit_peak_rss_mib_lte"),
        ("phase_b_prime_aggregate_peak_rss_mib", "phase_b_prime_aggregate_peak_rss_mib_lte"),
        ("phase_c_compare_wall_seconds", "phase_c_compare_wall_seconds_lte"),
        ("phase_c_compare_peak_rss_mib", "phase_c_compare_peak_rss_mib_lte"),
        ("sidecar_file_size_bytes", "sidecar_file_size_bytes_lte"),
    ):
        if float(metrics[metric_key]) > float(effective_gates[gate_key]):
            failures.append(metric_key)

    metrics["failures"] = failures
    metrics["passed"] = not failures

    if cleanup_sidecars:
        _cleanup_sidecars(oracle_sidecar, treatment_sidecar)

    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--modules", type=int, default=DEFAULT_MODULES)
    parser.add_argument("--lanes-per-module", type=int, default=DEFAULT_LANES_PER_MODULE)
    parser.add_argument("--measured-steps", type=int, default=DEFAULT_MEASURED_STEPS)
    parser.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument(
        "--retain-artifacts",
        action="store_true",
        help="Keep sidecar jsonl files after run (default: cleanup).",
    )
    args = parser.parse_args(argv)

    metrics = run_scale_smoke(
        output_dir=args.output_dir,
        modules=int(args.modules),
        lanes_per_module=int(args.lanes_per_module),
        measured_steps=int(args.measured_steps),
        warmup_steps=int(args.warmup_steps),
        require_creditdir=True,
        cleanup_sidecars=not bool(args.retain_artifacts),
    )
    payload = json.dumps(metrics, indent=2, sort_keys=True)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(payload, encoding="utf-8")
    print(payload)
    return 0 if metrics["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
