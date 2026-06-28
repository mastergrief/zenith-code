from __future__ import annotations

import copy
import json
from pathlib import Path

from calm.hrm_text_158.native_full_stack.d_recompute_launch_sequence_artifact_closure import (
    validate_launch_sequence_artifact_closure,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
V2_REPLAY = REPO_ROOT / "artifacts/consensus_prep/d_recompute_window_feasibility_gpu_launch_packet_v2_replay_commands.json"


def _load_replay() -> dict:
    return json.loads(V2_REPLAY.read_text(encoding="utf-8"))


def test_v2_launch_sequence_artifact_closure_passes() -> None:
    failures = validate_launch_sequence_artifact_closure(_load_replay())
    assert failures == []


def test_broken_launch_sequence_without_warmup_producer_fails() -> None:
    replay = _load_replay()
    broken = copy.deepcopy(replay)
    broken["launch_sequence"] = [
        step
        for step in replay["launch_sequence"]
        if step != "calibration_warmup_command"
        and step != "scratch_wipe_commands.calibration_warmup"
        and step != "parent_checkpoint_rehash_after_calibration_warmup_command"
    ]
    failures = validate_launch_sequence_artifact_closure(broken)
    assert failures
    assert any(
        "calibration_warmup_observations.json" in failure for failure in failures
    )


def test_replay_declares_calibration_warmup_producer_command() -> None:
    replay = _load_replay()
    assert "calibration_warmup_command" in replay
    warmup = replay["calibration_warmup_command"]
    assert "hrm_text_158_d_recompute_calibration_warmup_producer.py" in warmup
    assert "{run_root}/prelaunch/calibration_warmup_observations.json" in warmup
    sequence = replay["launch_sequence"]
    assert sequence.index("calibration_warmup_command") < sequence.index(
        "calibration_prepass_command"
    )
