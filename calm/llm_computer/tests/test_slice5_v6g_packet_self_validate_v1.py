"""Packet self-validate tests for v6g outer-budget smoke launch packet."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DRAFT = REPO / "artifacts/consensus_prep/slice5_step2a_live_carrier_gpu_scale_smoke_launch_packet_v6g_draft.json"
REPLAY = REPO / "artifacts/consensus_prep/slice5_step2a_live_carrier_gpu_scale_smoke_launch_packet_v6g_replay_commands.json"


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


def test_v6g_packet_self_validate_passes_clean() -> None:
    exit_code, receipt = _run_self_validate()
    assert exit_code == 0, receipt
    assert receipt.get("pass") is True
    assert receipt.get("packet_revision") == "v6g_re_m4_slice_b_diag_outer_budget_900"


def test_v6g_packet_self_validate_fails_on_stale_phase_timeout_300() -> None:
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


def test_v6g_launch_sequence_matches_v6f_postrun_order() -> None:
    replay = json.loads(REPLAY.read_text(encoding="utf-8"))
    seq = replay["launch_sequence"]
    barrier = seq.index("launch_arm_barrier_command")
    replay_cmd = seq.index("postrun_receipt_replay_command")
    path_ev = seq.index("cap_selection_path_evidence_command")
    triage = seq.index("bounded_steps_triage_command")
    classifier = seq.index("milestone_stall_classifier_command")
    receipt = seq.index("live_carrier_scale_smoke_receipt_command")
    acceptance = seq.index("v6g_postrun_acceptance_validate_command")
    assert barrier < replay_cmd < path_ev < triage < classifier < receipt < acceptance


def test_v6g_draft_budget_metadata_phase_timeout_900() -> None:
    draft = json.loads(DRAFT.read_text(encoding="utf-8"))
    assert draft["phase_budgets_and_watcher"]["phase_timeout_seconds"] == 900
    budget = draft["bounded_steps_budget"]
    assert budget["prior_value_seconds"] == 300
    assert budget["new_value_seconds"] == 900
    assert draft["scale_smoke"]["max_steps_hard"] == 3
    assert draft["scale_smoke"]["steps"] == 3
    assert draft["run_id"] == "2189e72027"
    assert draft["run_root"].rstrip("/").endswith("2189e72027")
    assert draft["source_base_head"] == "f100322b2d349f518e05fe583337eb4b26a8f45a"


def test_v6g_packet_self_validate_fails_on_max_silent_override() -> None:
    replay = json.loads(REPLAY.read_text(encoding="utf-8"))
    original = replay["shared_probe_argv"]
    replay["shared_probe_argv"] = original + " --max-silent-phase-seconds 600"
    REPLAY.write_text(json.dumps(replay, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        exit_code, receipt = _run_self_validate()
        assert exit_code != 0
        assert any("forbidden_max_silent_override" in f for f in receipt.get("failures", []))
    finally:
        replay["shared_probe_argv"] = original
        REPLAY.write_text(json.dumps(replay, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_v6g_packet_self_validate_fails_on_run_id_mismatch() -> None:
    draft = json.loads(DRAFT.read_text(encoding="utf-8"))
    original = draft["run_id"]
    draft["run_id"] = "2189e72026"
    DRAFT.write_text(json.dumps(draft, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        exit_code, receipt = _run_self_validate()
        assert exit_code != 0
        assert any("draft_run_id_mismatch" in f for f in receipt.get("failures", []))
    finally:
        draft["run_id"] = original
        DRAFT.write_text(json.dumps(draft, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_v6g_packet_self_validate_fails_on_stale_v6f_active_reference() -> None:
    draft = json.loads(DRAFT.read_text(encoding="utf-8"))
    original = draft["decision_contract"]["chosen_path"]
    draft["decision_contract"]["chosen_path"] = original + " v6f_re_m4 stale"
    DRAFT.write_text(json.dumps(draft, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        exit_code, receipt = _run_self_validate()
        assert exit_code != 0
        assert any("stale_active_reference:v6f_re_m4" in f for f in receipt.get("failures", []))
    finally:
        draft["decision_contract"]["chosen_path"] = original
        DRAFT.write_text(json.dumps(draft, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_v6g_phase_timeout_900_on_all_argv_surfaces() -> None:
    replay = json.loads(REPLAY.read_text(encoding="utf-8"))
    for key in ("shared_probe_argv", "baseline_gpu_command", "instrumented_gpu_command"):
        text = replay[key]
        assert "--phase-timeout-seconds 900" in text
        assert "--phase-timeout-seconds 300" not in text
        assert "--max-silent-phase-seconds" not in text
