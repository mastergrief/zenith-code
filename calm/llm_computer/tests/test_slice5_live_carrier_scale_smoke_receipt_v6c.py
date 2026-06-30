from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.hrm_text_158_slice5_live_carrier_scale_smoke_receipt import (
    emit_live_carrier_scale_smoke_receipt,
    main as receipt_main,
)
from scripts.hrm_text_158_slice5_milestone_stall_classifier import (
    emit_classifier_receipt,
    main as classifier_main,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
V6B_FIXTURE = (
    REPO_ROOT / "calm/llm_computer/tests/fixtures/slice5_re_m4_v6b_2189e72023"
)
V6C_PACKET = {
    "packet_revision": "v6c_re_m4_phase_guard_classifier_extract",
    "run_id": "2189e72023",
    "decision_contract": {"m4_mode": "re_M4_phase_guard_classifier_extract"},
}
RECEIPT_SCRIPT = REPO_ROOT / "scripts/hrm_text_158_slice5_live_carrier_scale_smoke_receipt.py"
CLASSIFIER_SCRIPT = REPO_ROOT / "scripts/hrm_text_158_slice5_milestone_stall_classifier.py"


def _write_packet(run_root: Path) -> Path:
    packet_path = run_root / "packet.json"
    packet_path.write_text(json.dumps(V6C_PACKET), encoding="utf-8")
    return packet_path


def _run_receipt_cli(run_root: Path, packet_path: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    return subprocess.run(
        [
            sys.executable,
            str(RECEIPT_SCRIPT),
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
    packet_path = run_root / "packet.json"
    packet_path.write_text(json.dumps(V6C_PACKET), encoding="utf-8")
    emit_classifier_receipt(run_root=run_root, packet_path=packet_path)
    return run_root


def test_v6b_terminal_receipt_preserves_liveness_fail_with_phase_guard(v6b_run_root: Path) -> None:
    packet_path = v6b_run_root / "packet.json"
    receipt = emit_live_carrier_scale_smoke_receipt(
        run_root=v6b_run_root,
        packet_path=packet_path,
    )

    assert receipt["classification"] == "LIVENESS_FAIL"
    assert receipt["classifier_classification"] == "LIVENESS_FAIL"
    assert receipt["phase_guard_locus"] == "bounded_steps"
    assert receipt["stalled_sub_phase_id"] is None
    assert receipt["milestone_locus"] is None
    assert receipt["packet_revision"] == "v6c_re_m4_phase_guard_classifier_extract"
    assert receipt["baseline_launch_rc"] == 1
    assert receipt["instrumented_launch_rc"] == 1
    assert receipt["classification"] != "MILESTONE_ARTIFACT_INCOMPLETE"
    assert "baseline_launch_rc_1" in receipt["failures"]
    assert "instrumented_launch_rc_1" in receipt["failures"]


def test_receipt_main_exit_zero_v6b(v6b_run_root: Path) -> None:
    packet_path = v6b_run_root / "packet.json"
    exit_code = receipt_main(
        ["--run-root", str(v6b_run_root), "--packet", str(packet_path)]
    )
    assert exit_code == 0
    receipt = json.loads(
        (v6b_run_root / "prelaunch" / "live_carrier_scale_smoke_receipt.json").read_text(
            encoding="utf-8"
        )
    )
    assert receipt["classification"] == "LIVENESS_FAIL"


def test_receipt_cli_exit_zero_v6b(v6b_run_root: Path) -> None:
    packet_path = v6b_run_root / "packet.json"
    result = _run_receipt_cli(v6b_run_root, packet_path)
    assert result.returncode == 0, result.stderr


def test_replay_order_classifier_then_receipt_v6b(tmp_path: Path) -> None:
    run_root = tmp_path / "replay_order_run"
    shutil.copytree(V6B_FIXTURE, run_root)
    packet_path = _write_packet(run_root)

    classifier_exit = classifier_main(
        ["--run-root", str(run_root), "--packet", str(packet_path)]
    )
    assert classifier_exit == 0

    receipt_exit = receipt_main(
        ["--run-root", str(run_root), "--packet", str(packet_path)]
    )
    assert receipt_exit == 0

    classifier_receipt = json.loads(
        (run_root / "prelaunch" / "milestone_stall_classifier_receipt.json").read_text(
            encoding="utf-8"
        )
    )
    terminal_receipt = json.loads(
        (run_root / "prelaunch" / "live_carrier_scale_smoke_receipt.json").read_text(
            encoding="utf-8"
        )
    )
    assert classifier_receipt["classification"] == "LIVENESS_FAIL"
    assert terminal_receipt["classification"] == "LIVENESS_FAIL"
    assert terminal_receipt["phase_guard_locus"] == "bounded_steps"
