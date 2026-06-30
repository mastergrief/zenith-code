"""Packet self-validate tests for v6h A-prime GPU seam runtime-proof launch packet."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DRAFT = REPO / "artifacts/consensus_prep/slice5_step2a_live_carrier_gpu_scale_smoke_launch_packet_v6h_draft.json"
REPLAY = REPO / "artifacts/consensus_prep/slice5_step2a_live_carrier_gpu_scale_smoke_launch_packet_v6h_replay_commands.json"


def _run_self_validate() -> tuple[int, dict]:
    replay = json.loads(REPLAY.read_text(encoding="utf-8"))
    proc = subprocess.run(
        ["bash", "-lc", replay["packet_self_validate_command"]],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    receipt = json.loads(proc.stdout) if proc.stdout.strip() else {}
    return proc.returncode, receipt


def test_v6h_packet_self_validate_passes_clean() -> None:
    exit_code, receipt = _run_self_validate()
    assert exit_code == 0, receipt
    assert receipt.get("pass") is True
    assert receipt.get("packet_revision") == "v6h_a_prime_gpu_seam_runtime_proof"


def test_v6h_draft_source_base_head_is_a_prime_commit() -> None:
    draft = json.loads(DRAFT.read_text(encoding="utf-8"))
    assert draft["source_base_head"] == "3ee73541bb125071b48e46eec5f8d58e833f7e82"
    assert draft["run_id"] == "2189e72028"
    assert draft["run_root"].rstrip("/").endswith("2189e72028")
    assert draft["v6h_acceptance_assertions"]["expected_path_route"] == "PATH_GPU_SEAM_EXERCISED"


def test_v6h_packet_self_validate_fails_on_stale_phase_timeout_300() -> None:
    replay = json.loads(REPLAY.read_text(encoding="utf-8"))
    original = replay["shared_probe_argv"]
    replay["shared_probe_argv"] = original.replace("--phase-timeout-seconds 900", "--phase-timeout-seconds 300")
    REPLAY.write_text(json.dumps(replay, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        exit_code, receipt = _run_self_validate()
        assert exit_code != 0
        assert receipt.get("pass") is False
        assert any("stale_phase_timeout_300" in f for f in receipt.get("failures", []))
    finally:
        replay["shared_probe_argv"] = original
        REPLAY.write_text(json.dumps(replay, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_v6h_packet_self_validate_fails_on_run_id_mismatch() -> None:
    draft = json.loads(DRAFT.read_text(encoding="utf-8"))
    original = draft["run_id"]
    draft["run_id"] = "2189e72027"
    DRAFT.write_text(json.dumps(draft, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        exit_code, receipt = _run_self_validate()
        assert exit_code != 0
        assert any("draft_run_id_mismatch" in f for f in receipt.get("failures", []))
    finally:
        draft["run_id"] = original
        DRAFT.write_text(json.dumps(draft, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_v6h_packet_self_validate_fails_on_missing_gpu_env_flags() -> None:
    replay = json.loads(REPLAY.read_text(encoding="utf-8"))
    original = replay["baseline_gpu_command"]
    replay["baseline_gpu_command"] = original.replace("HRM_TEXT_158_RUN_GPU_GLOBAL_RATE_CAP=1 ", "")
    REPLAY.write_text(json.dumps(replay, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        exit_code, receipt = _run_self_validate()
        assert exit_code != 0
        assert any("missing_gpu_global_rate_cap_env_in_baseline_gpu_command" in f for f in receipt.get("failures", []))
    finally:
        replay["baseline_gpu_command"] = original
        REPLAY.write_text(json.dumps(replay, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_v6h_packet_self_validate_fails_on_stale_v6g_active_reference() -> None:
    draft = json.loads(DRAFT.read_text(encoding="utf-8"))
    original = draft["decision_contract"]["chosen_path"]
    draft["decision_contract"]["chosen_path"] = original + " v6g_re_m4 stale"
    DRAFT.write_text(json.dumps(draft, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        exit_code, receipt = _run_self_validate()
        assert exit_code != 0
        assert any("stale_active_reference:v6g_re_m4" in f for f in receipt.get("failures", []))
    finally:
        draft["decision_contract"]["chosen_path"] = original
        DRAFT.write_text(json.dumps(draft, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_v6h_launch_sequence_includes_dispatch_witness_before_gpu_arms() -> None:
    replay = json.loads(REPLAY.read_text(encoding="utf-8"))
    seq = replay["launch_sequence"]
    inj = seq.index("launch_injected_dispatch_witness_command")
    assert inj < seq.index("baseline_gpu_command")
    assert seq.index("v6h_postrun_acceptance_validate_command") > seq.index(
        "live_carrier_scale_smoke_receipt_command"
    )
