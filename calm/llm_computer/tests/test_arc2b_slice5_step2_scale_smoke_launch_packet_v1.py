"""CPU-static tests for Arc #2b Slice-5 Step-2 scale_smoke launch packet."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from calm.hrm_text_158.native_full_stack.arc2b_slice5_in_vivo_branch import sha256_file
from scripts.apply_arc2b_slice5_step2_scale_smoke_launch_packet import (
    CLASSIFIER_MODULE,
    DRAFT,
    HEAD,
    REPLAY,
    SMOKE_STEPS,
    build_packet,
    build_replay_commands,
    git_head,
    self_verify,
)

REPO = Path(__file__).resolve().parents[3]


def test_live_git_head_matches_head_constant() -> None:
    assert git_head() == HEAD


def test_packet_git_head_required_matches_current_head_constant() -> None:
    classifier_sha = sha256_file(CLASSIFIER_MODULE)
    replay = build_replay_commands(classifier_sha)
    replay_sha = "deadbeef"
    packet = build_packet(classifier_sha, replay_sha)
    assert packet["git_head_required"] == HEAD
    assert packet["git_head_required"] == "3d52a96abd8ecab00b902f9d6a837c2a30b80894"


def test_launch_sequence_order_smoke_witness_receipt_eligibility() -> None:
    classifier_sha = sha256_file(CLASSIFIER_MODULE)
    replay = build_replay_commands(classifier_sha)
    sequence = list(replay["launch_sequence"])
    assert sequence.index("scale_smoke_command") < sequence.index(
        "scale_smoke_operational_witness_command"
    )
    assert sequence.index("scale_smoke_operational_witness_command") < sequence.index(
        "scale_smoke_receipt_command"
    )
    assert sequence.index("scale_smoke_receipt_command") < sequence.index(
        "scale_smoke_launch_eligibility_command"
    )
    assert "confirmation_launch_command" not in sequence
    assert "postrun_command" not in sequence


def test_self_verify_ok_after_regen(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(REPO)
    proc = subprocess.run(
        ["python3", "scripts/apply_arc2b_slice5_step2_scale_smoke_launch_packet.py"],
        cwd=REPO,
        env={**dict(__import__("os").environ), "PYTHONPATH": str(REPO)},
        check=False,
        capture_output=True,
        text=True,
    )
    result = json.loads(proc.stdout)
    assert proc.returncode == 0, result
    assert result["ok"] is True
    assert result["pins_match_commit"] is True
    assert result["deterministic_regen"] is True
    packet = json.loads(DRAFT.read_text(encoding="utf-8"))
    assert packet["git_head_required"] == HEAD
    assert packet["scale_smoke"]["smoke_steps"] == SMOKE_STEPS


def test_self_verify_fails_when_git_head_mismatches_head_constant(monkeypatch) -> None:
    import scripts.apply_arc2b_slice5_step2_scale_smoke_launch_packet as apply_mod

    monkeypatch.setattr(apply_mod, "git_head", lambda: "0" * 40)
    result = self_verify()
    assert result["pins_match_commit"] is False
    assert "pins_match_commit" in result["failures"]
    assert result["ok"] is False
