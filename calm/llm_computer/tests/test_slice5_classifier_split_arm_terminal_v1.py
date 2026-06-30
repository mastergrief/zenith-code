"""Split-arm terminal classifier tests (Slice B-DIAG)."""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts.hrm_text_158_slice5_milestone_stall_classifier import classify_milestone_stall

V6D_FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "slice5_re_m4_v6d_2189e72024_corrected_null"
)
V6B_FIXTURE = (
    Path(__file__).resolve().parent / "fixtures" / "slice5_re_m4_v6b_2189e72023"
)
KERNELIZED_FIXTURE = (
    Path(__file__).resolve().parent / "fixtures" / "slice5_kernelized_stall_flat_counter"
)


def _packet(run_id: str, *, max_steps_hard: int = 3) -> dict:
    return {
        "packet_revision": "v6d_re_m4_slice_b_gpu_cap_seam_smoke",
        "run_id": run_id,
        "scale_smoke": {"steps": max_steps_hard, "max_steps_hard": max_steps_hard},
    }


def test_v6d_split_arm_instrumented_outer_timeout() -> None:
    assert V6D_FIXTURE.is_dir(), f"missing pinned fixture: {V6D_FIXTURE}"
    receipt = classify_milestone_stall(run_root=V6D_FIXTURE, packet=_packet("2189e72024"))
    assert receipt["classification"] == "INSTRUMENTED_OUTER_BOUNDED_STEPS_TIMEOUT_AFTER_MAX_STEPS"
    assert receipt["phase_guard_locus"] == "sparse_cap_apply"
    assert receipt["stalled_sub_phase_id"] is None
    assert receipt["milestone_locus"] is None


def test_v6b_phase_guard_bounded_steps_regression() -> None:
    receipt = classify_milestone_stall(run_root=V6B_FIXTURE, packet=_packet("2189e72023"))
    assert receipt["classification"] == "LIVENESS_FAIL"
    assert receipt["phase_guard_locus"] == "bounded_steps"


def test_kernelized_stall_positive_regression() -> None:
    receipt = classify_milestone_stall(
        run_root=KERNELIZED_FIXTURE,
        packet=_packet("kernelized", max_steps_hard=10),
    )
    assert receipt["classification"] == "LIVENESS_FAIL_KERNELIZED_BUT_STALLED"
    assert receipt["stalled_sub_phase_id"] == "cap_selection_cpu_copy"
