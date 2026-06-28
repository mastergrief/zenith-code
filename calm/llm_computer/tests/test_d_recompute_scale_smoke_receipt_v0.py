from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from calm.hrm_text_158.native_full_stack.d_recompute_window_receipt_compact import (
    DEFAULT_EXTRAPOLATED_H100_RECEIPT_BYTES_MAX,
    DEFAULT_EXTRAPOLATED_H100_RECOMPUTE_LOG_BYTES_MAX,
    DEFAULT_RECEIPT_BYTES_PER_STEP_MAX,
    DEFAULT_RECOMPUTE_LOG_BYTES_PER_STEP_MAX,
)
from scripts.hrm_text_158_d_recompute_scale_smoke_receipt import (
    DEFAULT_MIN_FREE_MEMORY_BYTES,
    MIB_TO_BYTES,
    _gpu_memory_free_bytes,
    build_scale_smoke_receipt,
)


def _fixture_nvidia_smi_runner(stdout: str):
    def _runner(argv: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(argv, 0, stdout, "")

    return _runner


def test_memory_free_mib_converted_to_bytes_before_threshold() -> None:
    runner = _fixture_nvidia_smi_runner("0, 1536, 8192\n")
    proof = _gpu_memory_free_bytes(
        min_free_memory_bytes=DEFAULT_MIN_FREE_MEMORY_BYTES,
        nvidia_smi_runner=runner,
    )
    assert proof["memory_free_ok"] is True
    assert proof["min_free_bytes_observed"] == 1536 * MIB_TO_BYTES


def test_memory_free_boundary_fail_below_threshold() -> None:
    runner = _fixture_nvidia_smi_runner("0, 1535, 8192\n")
    proof = _gpu_memory_free_bytes(
        min_free_memory_bytes=DEFAULT_MIN_FREE_MEMORY_BYTES,
        nvidia_smi_runner=runner,
    )
    assert proof["memory_free_ok"] is False
    assert proof["min_free_bytes_observed"] == 1535 * MIB_TO_BYTES


def test_launch_allowed_requires_bytes_process_dead_and_memory(tmp_path: Path) -> None:
    diagnostic = tmp_path / "d_recompute_window_diagnostic"
    diagnostic.mkdir(parents=True)
    receipt_path = diagnostic / "receipt.json"
    log_path = diagnostic / "recompute_window_log.jsonl"
    smoke_steps = 5
    per_step_receipt = DEFAULT_RECEIPT_BYTES_PER_STEP_MAX // 4
    receipt_path.write_text("x" * (per_step_receipt * smoke_steps), encoding="utf-8")
    log_path.write_text("x" * (DEFAULT_RECOMPUTE_LOG_BYTES_PER_STEP_MAX // 4 * smoke_steps), encoding="utf-8")

    runner = _fixture_nvidia_smi_runner("0, 2048, 8192\n")
    process_dead_proof = {"pgrep_exit_code": 1, "matches": [], "process_dead": True}
    with patch(
        "scripts.hrm_text_158_d_recompute_scale_smoke_receipt._gpu_process_dead",
        return_value=process_dead_proof,
    ):
        receipt = build_scale_smoke_receipt(
            run_root=tmp_path,
            smoke_steps=smoke_steps,
            confirmation_steps=100,
            receipt_bytes_per_step_max=DEFAULT_RECEIPT_BYTES_PER_STEP_MAX,
            recompute_log_bytes_per_step_max=DEFAULT_RECOMPUTE_LOG_BYTES_PER_STEP_MAX,
            extrapolated_h100_receipt_bytes_max=DEFAULT_EXTRAPOLATED_H100_RECEIPT_BYTES_MAX,
            extrapolated_h100_recompute_log_bytes_max=DEFAULT_EXTRAPOLATED_H100_RECOMPUTE_LOG_BYTES_MAX,
            min_free_memory_bytes=DEFAULT_MIN_FREE_MEMORY_BYTES,
            nvidia_smi_runner=runner,
        )
    assert receipt["launch_gate"]["byte_projection_pass"] is True
    assert receipt["launch_gate"]["memory_free_ok"] is True
    assert receipt["pass"] is True


def test_d_off_baseline_not_used_as_launch_gate_flag_present(tmp_path: Path) -> None:
    diagnostic = tmp_path / "d_recompute_window_diagnostic"
    diagnostic.mkdir(parents=True)
    (diagnostic / "receipt.json").write_text("{}", encoding="utf-8")
    (diagnostic / "recompute_window_log.jsonl").write_text("{}\n", encoding="utf-8")
    runner = _fixture_nvidia_smi_runner("0, 2048, 8192\n")
    receipt = build_scale_smoke_receipt(
        run_root=tmp_path,
        smoke_steps=5,
        nvidia_smi_runner=runner,
    )
    assert receipt["launch_gate"]["d_off_baseline_smoke_not_launch_gate"] is True
    assert "v1_d_off_smoke_mitigation" in receipt["launch_gate"]


def test_scale_smoke_receipt_cli_h200_extrapolated_caps(tmp_path: Path) -> None:
    diagnostic = tmp_path / "d_recompute_window_diagnostic"
    diagnostic.mkdir(parents=True)
    smoke_steps = 5
    per_step_receipt = DEFAULT_RECEIPT_BYTES_PER_STEP_MAX // 4
    (diagnostic / "receipt.json").write_text("x" * (per_step_receipt * smoke_steps), encoding="utf-8")
    (diagnostic / "recompute_window_log.jsonl").write_text(
        "x" * (DEFAULT_RECOMPUTE_LOG_BYTES_PER_STEP_MAX // 4 * smoke_steps),
        encoding="utf-8",
    )
    runner = _fixture_nvidia_smi_runner("0, 2048, 8192\n")
    process_dead_proof = {"pgrep_exit_code": 1, "matches": [], "process_dead": True}
    with patch(
        "scripts.hrm_text_158_d_recompute_scale_smoke_receipt._gpu_process_dead",
        return_value=process_dead_proof,
    ):
        receipt = build_scale_smoke_receipt(
            run_root=tmp_path,
            smoke_steps=smoke_steps,
            confirmation_steps=200,
            extrapolated_h100_receipt_bytes_max=104857600,
            extrapolated_h100_recompute_log_bytes_max=67108864,
            min_free_memory_bytes=DEFAULT_MIN_FREE_MEMORY_BYTES,
            nvidia_smi_runner=runner,
        )
    assert receipt["confirmation_steps"] == 200
    caps = receipt["byte_projection"]["caps"]
    assert caps["extrapolated_h100_receipt_bytes_max"] == 104857600
    assert caps["extrapolated_h100_recompute_log_bytes_max"] == 67108864
