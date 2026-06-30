"""Postrun barrier + replay repair tests (Slice B-DIAG2)."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from scripts.hrm_text_158_slice5_launch_arm_barrier import assert_postrun_barrier_ready
from scripts.hrm_text_158_slice5_postrun_receipt_replay import (
    REPLAY_OUTPUT_NAMES,
    replay_postrun_receipts,
)

V6E_FIXTURE = (
    Path(__file__).resolve().parent / "fixtures" / "slice5_re_m4_v6e_2189e72025"
)


def _packet() -> dict:
    return {
        "packet_revision": "v6e_re_m4_slice_b_diag_diagnostic_smoke",
        "run_id": "2189e72025",
        "scale_smoke": {"steps": 3, "max_steps_hard": 3},
        "decision_contract": {"m4_mode": "re_M4_slice_b_diag_diagnostic_smoke"},
    }


def test_barrier_blocks_while_arm_pid_alive(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    prelaunch = run_root / "prelaunch"
    prelaunch.mkdir(parents=True)
    (prelaunch / "baseline_launch_rc.txt").write_text("1\n", encoding="utf-8")
    (prelaunch / "instrumented_launch_rc.txt").write_text("1\n", encoding="utf-8")
    arm = run_root / "baseline_snapshot_off"
    arm.mkdir(parents=True)
    (arm / "last_active_phase.json").write_text("{}", encoding="utf-8")
    instr = run_root / "instrumented_snapshot_on"
    instr.mkdir(parents=True)
    (instr / "last_active_phase.json").write_text("{}", encoding="utf-8")
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        (arm / "probe.pid").write_text(f"{proc.pid}\n", encoding="utf-8")
        receipt = assert_postrun_barrier_ready(run_root=run_root)
        assert receipt["pass"] is False
        assert "arm_pid_still_alive" in receipt["failures"]
    finally:
        proc.kill()
        proc.wait(timeout=5)


def test_v6e_fixture_required() -> None:
    assert V6E_FIXTURE.is_dir(), f"missing pinned fixture: {V6E_FIXTURE}"


def test_replay_fail_closed_when_barrier_blocks_live_pid(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    prelaunch = run_root / "prelaunch"
    prelaunch.mkdir(parents=True)
    (prelaunch / "baseline_launch_rc.txt").write_text("1\n", encoding="utf-8")
    (prelaunch / "instrumented_launch_rc.txt").write_text("1\n", encoding="utf-8")
    arm = run_root / "baseline_snapshot_off"
    arm.mkdir(parents=True)
    (arm / "last_active_phase.json").write_text("{}", encoding="utf-8")
    instr = run_root / "instrumented_snapshot_on"
    instr.mkdir(parents=True)
    (instr / "last_active_phase.json").write_text("{}", encoding="utf-8")
    packet_path = tmp_path / "packet.json"
    packet_path.write_text(json.dumps(_packet()), encoding="utf-8")
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        (arm / "probe.pid").write_text(f"{proc.pid}\n", encoding="utf-8")
        receipt = replay_postrun_receipts(
            run_root=run_root,
            packet_path=packet_path,
            require_barrier=True,
        )
        assert receipt["pass"] is False
        assert receipt.get("receipts_written") is False
        assert "arm_pid_still_alive" in receipt["failures"]
        for name in REPLAY_OUTPUT_NAMES:
            assert not (prelaunch / name).is_file(), f"unexpected receipt written: {name}"
    finally:
        proc.kill()
        proc.wait(timeout=5)


def test_replay_repair_matches_true_final_v6e(tmp_path: Path) -> None:
    assert V6E_FIXTURE.is_dir(), f"missing pinned fixture: {V6E_FIXTURE}"
    run_root = tmp_path / "fixture_copy"
    import shutil

    shutil.copytree(V6E_FIXTURE, run_root)
    packet_path = tmp_path / "packet.json"
    packet_path.write_text(json.dumps(_packet()), encoding="utf-8")
    receipt = replay_postrun_receipts(
        run_root=run_root,
        packet_path=packet_path,
        require_barrier=False,
    )
    assert receipt["pass"] is True

    path_payload = receipt["path_evidence"]
    assert path_payload["overall_terminal_route"] == "PATH_CPU_RESIDENT_CAP_REFERENCE"
    for arm in path_payload["per_arm"]:
        if arm["sparse_cap_phase_complete_count"] > 0:
            assert arm["terminal_route"] == "PATH_CPU_RESIDENT_CAP_REFERENCE"
            assert arm["cap_selection_jsonl_milestone_kind"] == "cap_reference_cpu_resident_done"

    triage = receipt["bounded_steps_triage"]
    assert triage["bounded_steps_triage_class"] == "WRAPPER_BUDGET_TOO_TIGHT"
    assert triage["instrumented_outer_timeout_after_max_steps"] is True
    assert triage["baseline_sparse_cap_step_stall"] is True

    classifier = receipt["classifier"]
    assert (
        classifier["classification"]
        == "INSTRUMENTED_OUTER_BOUNDED_STEPS_TIMEOUT_AFTER_MAX_STEPS"
    )
    assert classifier["stalled_sub_phase_id"] is None

    stale = receipt["stale_receipts_marked"]
    assert stale
    assert all(row["superseded"] and not row["authority"] for row in stale)

    superseding = run_root / "prelaunch" / "cap_selection_path_evidence_receipt_superseding.json"
    assert superseding.is_file()
    wrapped = json.loads(superseding.read_text(encoding="utf-8"))
    assert wrapped["derived_from_final_drained_artifacts"] is True
