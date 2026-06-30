"""Cap-selection path evidence tests (Slice v6e packet)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.hrm_text_158_slice5_cap_selection_path_evidence import (
    ROUTE_CPU,
    ROUTE_GPU,
    ROUTE_INVALID,
    evaluate_arm,
    evaluate_cap_selection_path_evidence,
)

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures"
V6D_FIXTURE = FIXTURE_ROOT / "slice5_re_m4_v6d_2189e72024_corrected_null"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _arm_fixture(
    tmp_path: Path,
    *,
    marker_kind: str | None,
    summary_path: str | None = None,
    summary_container: str = "global_summary",
) -> Path:
    arm = tmp_path / "instrumented_snapshot_on"
    milestones = arm / "liveness_milestones"
    _write_jsonl(
        milestones / "sparse_cap_apply.jsonl",
        [{"milestone_kind": "phase_complete", "optimizer_step_index": 1}],
    )
    if marker_kind is not None:
        _write_jsonl(
            milestones / "sparse_cap_apply_cap_selection_cpu_copy.jsonl",
            [{"milestone_kind": marker_kind, "optimizer_step_index": 1}],
        )
    if summary_path is not None:
        (arm / "receipt.json").write_text(
            json.dumps(
                {summary_container: {"sparse_cap_submilestone_cap_selection_path": summary_path}}
            ),
            encoding="utf-8",
        )
    return arm


def test_cap_gpu_seam_done_routes_path_gpu_seam_exercised(tmp_path: Path) -> None:
    arm = _arm_fixture(tmp_path, marker_kind="cap_gpu_seam_done", summary_path="gpu_seam")
    row = evaluate_arm(scratch=arm)
    assert row["terminal_route"] == ROUTE_GPU
    assert row["pass"] is True


def test_cap_reference_cpu_resident_done_routes_cpu_path(tmp_path: Path) -> None:
    arm = _arm_fixture(
        tmp_path,
        marker_kind="cap_reference_cpu_resident_done",
        summary_path="cpu_resident_reference",
    )
    row = evaluate_arm(scratch=arm)
    assert row["terminal_route"] == ROUTE_CPU
    assert row["pass"] is True


def test_absent_marker_routes_submilestone_instrumentation_invalid(tmp_path: Path) -> None:
    arm = _arm_fixture(tmp_path, marker_kind=None)
    row = evaluate_arm(scratch=arm)
    assert row["terminal_route"] == ROUTE_INVALID
    assert "sparse_cap_complete_but_cap_selection_absent" in row["failures"]


def test_marker_summary_mismatch_fail_closed(tmp_path: Path) -> None:
    arm = _arm_fixture(
        tmp_path,
        marker_kind="cap_gpu_seam_done",
        summary_path="cpu_resident_reference",
    )
    row = evaluate_arm(scratch=arm)
    assert row["terminal_route"] == ROUTE_INVALID
    assert any("marker_path_mismatch" in item for item in row["failures"])


def test_bounded_delta_global_summary_mismatch_fail_closed(tmp_path: Path) -> None:
    arm = _arm_fixture(
        tmp_path,
        marker_kind="cap_gpu_seam_done",
        summary_path="cpu_resident_reference",
        summary_container="bounded_delta_global_summary",
    )
    row = evaluate_arm(scratch=arm)
    assert row["terminal_route"] == ROUTE_INVALID
    assert any("marker_path_mismatch" in item for item in row["failures"])


def test_v6d_fixture_absent_cap_selection_is_instrumentation_invalid() -> None:
    assert V6D_FIXTURE.is_dir(), f"missing pinned fixture: {V6D_FIXTURE}"
    receipt = evaluate_cap_selection_path_evidence(run_root=V6D_FIXTURE)
    instr = next(
        row for row in receipt["per_arm"] if row["arm"] == "instrumented_snapshot_on"
    )
    assert instr["sparse_cap_phase_complete_count"] == 3
    assert instr["terminal_route"] == ROUTE_INVALID
    assert "sparse_cap_complete_but_cap_selection_absent" in instr["failures"]
