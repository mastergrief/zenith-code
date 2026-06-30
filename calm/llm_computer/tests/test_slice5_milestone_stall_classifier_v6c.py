from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.hrm_text_158_slice5_milestone_stall_classifier import (
    classify_milestone_stall,
    emit_classifier_receipt,
    main as classifier_main,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
V6B_FIXTURE = (
    REPO_ROOT / "calm/llm_computer/tests/fixtures/slice5_re_m4_v6b_2189e72023"
)
STALL_FIXTURE = (
    REPO_ROOT / "calm/llm_computer/tests/fixtures/slice5_kernelized_stall_flat_counter"
)
V6C_PACKET = {
    "packet_revision": "v6c_re_m4_phase_guard_classifier_extract",
    "run_id": "2189e72023",
    "decision_contract": {"m4_mode": "re_M4_phase_guard_classifier_extract"},
}
CLASSIFIER_SCRIPT = REPO_ROOT / "scripts/hrm_text_158_slice5_milestone_stall_classifier.py"


def _write_packet(run_root: Path) -> Path:
    packet_path = run_root / "packet.json"
    packet_path.write_text(json.dumps(V6C_PACKET), encoding="utf-8")
    return packet_path


def _run_classifier_cli(run_root: Path, packet_path: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    return subprocess.run(
        [
            sys.executable,
            str(CLASSIFIER_SCRIPT),
            "--run-root",
            str(run_root),
            "--packet",
            str(packet_path),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture()
def v6b_run_root(tmp_path: Path) -> Path:
    run_root = tmp_path / "v6b_run"
    shutil.copytree(V6B_FIXTURE, run_root)
    return run_root


@pytest.fixture()
def stall_run_root(tmp_path: Path) -> Path:
    run_root = tmp_path / "stall_run"
    shutil.copytree(STALL_FIXTURE, run_root)
    return run_root


def test_v6b_phase_guard_kill_classifies_liveness_fail(v6b_run_root: Path) -> None:
    receipt = classify_milestone_stall(run_root=v6b_run_root, packet=V6C_PACKET)

    assert receipt["classification"] == "LIVENESS_FAIL"
    assert receipt["phase_guard_locus"] == "bounded_steps"
    assert receipt["stalled_sub_phase_id"] is None
    assert receipt["milestone_locus"] is None
    assert receipt["classification"] != "MILESTONE_ARTIFACT_INCOMPLETE"
    assert receipt["packet_revision"] == "v6c_re_m4_phase_guard_classifier_extract"
    assert "missing_milestone_jsonl" not in str(receipt["failures"])


def test_v6b_missing_cap_selection_not_stall_when_post_cap_monotonic(v6b_run_root: Path) -> None:
    receipt = classify_milestone_stall(run_root=v6b_run_root, packet=V6C_PACKET)

    assert receipt["stalled_sub_phase_id"] is None
    assert not any(hit.get("sub_phase_id") == "cap_selection_cpu_copy" for hit in receipt["stall_hits"])


def test_kernelized_flat_counter_stall_attributes_sub_phase(stall_run_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "scripts.hrm_text_158_slice5_milestone_stall_classifier.cuda_available",
        lambda: True,
    )
    receipt = classify_milestone_stall(run_root=stall_run_root, packet=V6C_PACKET)

    assert receipt["classification"] == "LIVENESS_FAIL_KERNELIZED_BUT_STALLED"
    assert receipt["stalled_sub_phase_id"] == "cap_selection_cpu_copy"
    assert receipt["stalled_parent_phase_id"] == "sparse_cap_apply"
    assert receipt["milestone_locus"] == "cap_selection_cpu_copy"
    assert receipt["phase_guard_locus"] is None


def test_emit_classifier_receipt_writes_artifact(v6b_run_root: Path) -> None:
    packet_path = _write_packet(v6b_run_root)
    receipt = emit_classifier_receipt(run_root=v6b_run_root, packet_path=packet_path)
    out = v6b_run_root / "prelaunch" / "milestone_stall_classifier_receipt.json"
    assert out.is_file()
    assert json.loads(out.read_text(encoding="utf-8")) == receipt


def test_classifier_main_exit_zero_v6b(v6b_run_root: Path) -> None:
    packet_path = _write_packet(v6b_run_root)
    exit_code = classifier_main(
        ["--run-root", str(v6b_run_root), "--packet", str(packet_path)]
    )
    assert exit_code == 0
    receipt = json.loads(
        (v6b_run_root / "prelaunch" / "milestone_stall_classifier_receipt.json").read_text(
            encoding="utf-8"
        )
    )
    assert receipt["classification"] == "LIVENESS_FAIL"


def test_classifier_cli_exit_zero_v6b(v6b_run_root: Path) -> None:
    packet_path = _write_packet(v6b_run_root)
    result = _run_classifier_cli(v6b_run_root, packet_path)
    assert result.returncode == 0, result.stderr


def test_classifier_cli_exit_zero_kernelized_stall(stall_run_root: Path) -> None:
    packet_path = _write_packet(stall_run_root)
    result = _run_classifier_cli(stall_run_root, packet_path)
    assert result.returncode == 0, result.stderr
    receipt = json.loads(
        (stall_run_root / "prelaunch" / "milestone_stall_classifier_receipt.json").read_text(
            encoding="utf-8"
        )
    )
    assert receipt["classification"] == "LIVENESS_FAIL_KERNELIZED_BUT_STALLED"


def test_missing_parent_sparse_cap_jsonl_no_attribution_without_counter(
    stall_run_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "scripts.hrm_text_158_slice5_milestone_stall_classifier.cuda_available",
        lambda: True,
    )
    for arm in ("baseline_snapshot_off", "instrumented_snapshot_on"):
        milestones = stall_run_root / arm / "liveness_milestones"
        parent = milestones / "sparse_cap_apply.jsonl"
        if parent.is_file():
            parent.unlink()
        cap_selection = milestones / "sparse_cap_apply_cap_selection_cpu_copy.jsonl"
        if cap_selection.is_file():
            cap_selection.unlink()
        lap = stall_run_root / arm / "last_active_phase.json"
        if lap.is_file():
            lap.unlink()

    receipt = classify_milestone_stall(run_root=stall_run_root, packet=V6C_PACKET)

    assert receipt["milestone_locus"] is None
    assert receipt["stalled_sub_phase_id"] is None
    assert receipt["stalled_parent_phase_id"] is None
