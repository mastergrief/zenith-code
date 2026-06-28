"""Producer/consumer closure checks for D recompute-window v2 launch_sequence."""
from __future__ import annotations

from typing import Any, Mapping

RUN_ROOT_TOKEN = "{run_root}"


def _artifact(path: str) -> str:
    return path.replace(RUN_ROOT_TOKEN, "<run_root>")


# Explicit prelaunch artifact graph for launch_sequence closure.
COMMAND_ARTIFACT_IO: dict[str, dict[str, list[str]]] = {
    "calibration_warmup_command": {
        "produces": [
            _artifact(f"{RUN_ROOT_TOKEN}/prelaunch/calibration_warmup_observations.json"),
            _artifact(f"{RUN_ROOT_TOKEN}/prelaunch/calibration_warmup_producer_receipt.json"),
        ],
        "consumes": [],
    },
    "calibration_prepass_command": {
        "produces": [
            _artifact(f"{RUN_ROOT_TOKEN}/prelaunch/calibrated_selector_manifest.json"),
            _artifact(f"{RUN_ROOT_TOKEN}/prelaunch/calibration_prepass_receipt.json"),
        ],
        "consumes": [
            _artifact(f"{RUN_ROOT_TOKEN}/prelaunch/calibration_warmup_observations.json"),
        ],
    },
    "scale_smoke_command": {
        "produces": [],
        "consumes": [
            _artifact(f"{RUN_ROOT_TOKEN}/prelaunch/calibrated_selector_manifest.json"),
        ],
    },
    "scale_smoke_receipt_command": {
        "produces": [
            _artifact(f"{RUN_ROOT_TOKEN}/prelaunch/scale_smoke_receipt.json"),
        ],
        "consumes": [],
    },
    "confirmation_launch_command": {
        "produces": [
            _artifact(f"{RUN_ROOT_TOKEN}/prelaunch/confirmation_launch_rc.txt"),
        ],
        "consumes": [
            _artifact(f"{RUN_ROOT_TOKEN}/prelaunch/calibrated_selector_manifest.json"),
        ],
    },
    "postrun_input_manifest_bind_command": {
        "produces": [
            _artifact(f"{RUN_ROOT_TOKEN}/prelaunch/postrun_input_manifest.json"),
        ],
        "consumes": [],
    },
    "postrun_command": {
        "produces": [],
        "consumes": [
            _artifact(f"{RUN_ROOT_TOKEN}/prelaunch/postrun_input_manifest.json"),
        ],
    },
}


def validate_launch_sequence_artifact_closure(
    replay: Mapping[str, Any],
) -> list[str]:
    """Return failure messages when a launch_sequence consumer lacks an earlier producer."""
    failures: list[str] = []
    sequence = list(replay.get("launch_sequence") or [])
    if not sequence:
        return ["launch_sequence_missing_or_empty"]

    produced: set[str] = set()
    for step_key in sequence:
        if str(step_key).startswith("scratch_wipe_commands."):
            continue
        if str(step_key) not in replay:
            failures.append(f"missing_replay_command_{step_key}")
            continue
        io = COMMAND_ARTIFACT_IO.get(str(step_key))
        if io is None:
            continue
        for artifact in io.get("consumes", []):
            if artifact not in produced:
                failures.append(
                    f"unproduced_artifact:{artifact}:required_by:{step_key}"
                )
        for artifact in io.get("produces", []):
            produced.add(artifact)

    warmup_obs = _artifact(
        f"{RUN_ROOT_TOKEN}/prelaunch/calibration_warmup_observations.json"
    )
    if warmup_obs not in produced:
        failures.append(f"launch_sequence_never_produces:{warmup_obs}")

    if "calibration_warmup_command" in sequence and "calibration_prepass_command" in sequence:
        if sequence.index("calibration_warmup_command") >= sequence.index(
            "calibration_prepass_command"
        ):
            failures.append("calibration_warmup_command_must_precede_calibration_prepass_command")
    else:
        if "calibration_prepass_command" in sequence and "calibration_warmup_command" not in sequence:
            failures.append("calibration_prepass_without_calibration_warmup_producer")

    return failures
