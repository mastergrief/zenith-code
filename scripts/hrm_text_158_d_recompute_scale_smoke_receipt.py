#!/usr/bin/env python3
"""Emit compact representative D-ON scale-smoke receipt with H=100 byte extrapolation."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

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

RECEIPT_SCHEMA = "hrm_text_158_d_recompute_scale_smoke_receipt/v0"
DIAGNOSTIC_SUBDIR = "d_recompute_window_diagnostic"


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


def _gpu_free() -> dict[str, Any]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    rows = [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]
    return {
        "nvidia_smi_exit_code": int(result.returncode),
        "rows": rows,
        "gpu_free": result.returncode == 0 and all(
            row.split(",")[1].strip() == "0" for row in rows if "," in row
        ),
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
    return {
        "schema_version": RECEIPT_SCHEMA,
        "run_root": str(run_root),
        "smoke_steps": int(smoke_steps),
        "confirmation_steps": int(confirmation_steps),
        "receipt_path": str(receipt_path),
        "recompute_log_path": str(recompute_log_path) if recompute_log_path.is_file() else None,
        "driver_summary_phase": driver_summary.get("phase"),
        "byte_projection": byte_projection,
        "process_dead_proof": _gpu_process_dead(),
        "gpu_free_proof": _gpu_free(),
        "pass": bool(byte_projection["launch_allowed"]),
        "duration_seconds": None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--smoke-steps", type=int, required=True)
    parser.add_argument("--confirmation-steps", type=int, default=DEFAULT_H100_CONFIRMATION_STEPS)
    args = parser.parse_args(argv)
    started = time.monotonic()
    receipt = build_scale_smoke_receipt(
        run_root=args.run_root,
        smoke_steps=int(args.smoke_steps),
        confirmation_steps=int(args.confirmation_steps),
    )
    receipt["duration_seconds"] = float(time.monotonic() - started)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if bool(receipt["pass"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
