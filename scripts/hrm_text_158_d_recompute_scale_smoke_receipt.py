#!/usr/bin/env python3
"""Emit compact representative D-ON scale-smoke receipt with H=100 byte extrapolation."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

from calm.hrm_text_158.native_full_stack.d_recompute_window_emit import (
    D_RECOMPUTE_WINDOW_LOG_FILENAME,
)
from calm.hrm_text_158.native_full_stack.d_recompute_window_receipt_compact import (
    DEFAULT_EXTRAPOLATED_H100_RECEIPT_BYTES_MAX,
    DEFAULT_EXTRAPOLATED_H100_RECOMPUTE_LOG_BYTES_MAX,
    DEFAULT_H100_CONFIRMATION_STEPS,
    DEFAULT_RECEIPT_BYTES_PER_STEP_MAX,
    DEFAULT_RECOMPUTE_LOG_BYTES_PER_STEP_MAX,
    extrapolate_h100_byte_projections,
)

RECEIPT_SCHEMA = "hrm_text_158_d_recompute_scale_smoke_receipt/v1"
DIAGNOSTIC_SUBDIR = "d_recompute_window_diagnostic"
MIB_TO_BYTES = 1048576
DEFAULT_MIN_FREE_MEMORY_BYTES = 1536 * MIB_TO_BYTES
DEFAULT_MIN_FREE_MEMORY_MIB = 1536
MIN_FREE_MEMORY_BYTES_PROVISIONAL_LABEL = "PROVISIONAL_UNTIL_D_ON_SMOKE_CALIBRATES"

NvidiaSmiRunner = Callable[[list[str]], subprocess.CompletedProcess[str]]


def _default_nvidia_smi_runner(argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        check=False,
        capture_output=True,
        text=True,
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _gpu_process_dead() -> dict[str, Any]:
    pattern = "hrm_text_158_bounded_delta_acquisition_probe"
    result = subprocess.run(
        ["pgrep", "-af", pattern],
        check=False,
        capture_output=True,
        text=True,
    )
    matches = [
        line.strip()
        for line in (result.stdout or "").splitlines()
        if line.strip() and "pgrep -af" not in line
    ]
    return {
        "pgrep_exit_code": int(result.returncode),
        "matches": matches,
        "process_dead": len(matches) == 0,
    }


def _gpu_free_utilization_telemetry(
    *,
    nvidia_smi_runner: NvidiaSmiRunner = _default_nvidia_smi_runner,
) -> dict[str, Any]:
    result = nvidia_smi_runner(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ]
    )
    rows = [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]
    return {
        "nvidia_smi_exit_code": int(result.returncode),
        "rows": rows,
        "gpu_free_utilization_zero": result.returncode == 0
        and all(row.split(",")[1].strip() == "0" for row in rows if "," in row),
        "telemetry_only": True,
    }


def _parse_memory_free_mib_rows(stdout: str) -> list[dict[str, int]]:
    parsed: list[dict[str, int]] = []
    for line in stdout.splitlines():
        row = line.strip()
        if not row or "," not in row:
            continue
        parts = [part.strip() for part in row.split(",")]
        if len(parts) < 2:
            continue
        parsed.append(
            {
                "gpu_index": int(parts[0]),
                "memory_free_mib": int(parts[1]),
                "memory_total_mib": int(parts[2]) if len(parts) > 2 else 0,
            }
        )
    return parsed


def _gpu_memory_free_bytes(
    *,
    min_free_memory_bytes: int = DEFAULT_MIN_FREE_MEMORY_BYTES,
    nvidia_smi_runner: NvidiaSmiRunner = _default_nvidia_smi_runner,
) -> dict[str, Any]:
    result = nvidia_smi_runner(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    rows = _parse_memory_free_mib_rows(result.stdout or "")
    free_bytes_by_gpu = {
        int(row["gpu_index"]): int(row["memory_free_mib"]) * MIB_TO_BYTES for row in rows
    }
    min_free_bytes_observed = min(free_bytes_by_gpu.values()) if free_bytes_by_gpu else 0
    memory_free_ok = (
        result.returncode == 0
        and bool(rows)
        and min_free_bytes_observed >= int(min_free_memory_bytes)
    )
    return {
        "nvidia_smi_exit_code": int(result.returncode),
        "rows_raw": [line.strip() for line in (result.stdout or "").splitlines() if line.strip()],
        "rows_parsed": rows,
        "memory_free_mib_to_bytes_factor": MIB_TO_BYTES,
        "min_free_memory_bytes": int(min_free_memory_bytes),
        "min_free_memory_mib": int(min_free_memory_bytes // MIB_TO_BYTES),
        "min_free_memory_bytes_label": MIN_FREE_MEMORY_BYTES_PROVISIONAL_LABEL,
        "free_bytes_by_gpu": free_bytes_by_gpu,
        "min_free_bytes_observed": int(min_free_bytes_observed),
        "memory_free_ok": bool(memory_free_ok),
    }


def build_scale_smoke_receipt(
    *,
    run_root: Path,
    smoke_steps: int,
    confirmation_steps: int = DEFAULT_H100_CONFIRMATION_STEPS,
    receipt_bytes_per_step_max: int = DEFAULT_RECEIPT_BYTES_PER_STEP_MAX,
    recompute_log_bytes_per_step_max: int = DEFAULT_RECOMPUTE_LOG_BYTES_PER_STEP_MAX,
    extrapolated_h100_receipt_bytes_max: int = DEFAULT_EXTRAPOLATED_H100_RECEIPT_BYTES_MAX,
    extrapolated_h100_recompute_log_bytes_max: int = DEFAULT_EXTRAPOLATED_H100_RECOMPUTE_LOG_BYTES_MAX,
    min_free_memory_bytes: int = DEFAULT_MIN_FREE_MEMORY_BYTES,
    nvidia_smi_runner: NvidiaSmiRunner = _default_nvidia_smi_runner,
) -> dict[str, Any]:
    diagnostic_root = run_root / DIAGNOSTIC_SUBDIR
    receipt_path = diagnostic_root / "receipt.json"
    if not receipt_path.is_file():
        raise FileNotFoundError(f"missing probe receipt: {receipt_path}")
    receipt_bytes = int(receipt_path.stat().st_size)
    recompute_log_path = diagnostic_root / D_RECOMPUTE_WINDOW_LOG_FILENAME
    recompute_log_bytes = int(recompute_log_path.stat().st_size) if recompute_log_path.is_file() else 0
    driver_summary_path = run_root / "driver_summary.json"
    driver_summary = _read_json(driver_summary_path) if driver_summary_path.is_file() else {}
    byte_projection = extrapolate_h100_byte_projections(
        receipt_bytes=receipt_bytes,
        smoke_steps=int(smoke_steps),
        recompute_log_bytes=recompute_log_bytes,
        confirmation_steps=int(confirmation_steps),
        receipt_bytes_per_step_max=int(receipt_bytes_per_step_max),
        recompute_log_bytes_per_step_max=int(recompute_log_bytes_per_step_max),
        extrapolated_h100_receipt_bytes_max=int(extrapolated_h100_receipt_bytes_max),
        extrapolated_h100_recompute_log_bytes_max=int(extrapolated_h100_recompute_log_bytes_max),
    )
    process_dead_proof = _gpu_process_dead()
    memory_free_proof = _gpu_memory_free_bytes(
        min_free_memory_bytes=int(min_free_memory_bytes),
        nvidia_smi_runner=nvidia_smi_runner,
    )
    gpu_utilization_telemetry = _gpu_free_utilization_telemetry(
        nvidia_smi_runner=nvidia_smi_runner
    )
    launch_allowed = bool(
        byte_projection["launch_allowed"]
        and process_dead_proof["process_dead"]
        and memory_free_proof["memory_free_ok"]
    )
    return {
        "schema_version": RECEIPT_SCHEMA,
        "run_root": str(run_root),
        "smoke_steps": int(smoke_steps),
        "confirmation_steps": int(confirmation_steps),
        "receipt_path": str(receipt_path),
        "recompute_log_path": str(recompute_log_path) if recompute_log_path.is_file() else None,
        "driver_summary_phase": driver_summary.get("phase"),
        "byte_projection": byte_projection,
        "process_dead_proof": process_dead_proof,
        "memory_free_proof": memory_free_proof,
        "gpu_utilization_telemetry": gpu_utilization_telemetry,
        "launch_gate": {
            "byte_projection_pass": bool(byte_projection["launch_allowed"]),
            "process_dead": bool(process_dead_proof["process_dead"]),
            "memory_free_ok": bool(memory_free_proof["memory_free_ok"]),
            "launch_allowed": launch_allowed,
            "d_off_baseline_smoke_not_launch_gate": True,
            "v1_d_off_smoke_mitigation": (
                "v1 used D-OFF scale-smoke (no recompute_window_log.jsonl) so byte "
                "projection could not validate D-ON H=100 receipt/log caps; v2 requires "
                "D-ON manifest-bound smoke under compact diagnostic receipts with "
                "byte caps plus memory.free gate before H=100 confirmation."
            ),
        },
        "pass": launch_allowed,
        "duration_seconds": None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--smoke-steps", type=int, required=True)
    parser.add_argument("--confirmation-steps", type=int, default=DEFAULT_H100_CONFIRMATION_STEPS)
    parser.add_argument(
        "--min-free-memory-bytes",
        type=int,
        default=DEFAULT_MIN_FREE_MEMORY_BYTES,
        help=f"Default {DEFAULT_MIN_FREE_MEMORY_BYTES} bytes ({DEFAULT_MIN_FREE_MEMORY_MIB} MiB, PROVISIONAL).",
    )
    args = parser.parse_args(argv)
    started = time.monotonic()
    receipt = build_scale_smoke_receipt(
        run_root=args.run_root,
        smoke_steps=int(args.smoke_steps),
        confirmation_steps=int(args.confirmation_steps),
        min_free_memory_bytes=int(args.min_free_memory_bytes),
    )
    receipt["duration_seconds"] = float(time.monotonic() - started)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if bool(receipt["pass"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
