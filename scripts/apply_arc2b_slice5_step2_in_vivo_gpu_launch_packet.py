#!/usr/bin/env python3
"""Apply Arc #2b Slice-5 Step-2 in-vivo GPU launch packet (CPU packet/postrun only; no GPU)."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Callable

from calm.hrm_text_158.native_full_stack.arc2b_slice5_in_vivo_branch import (
    B1_RECORDED_SELECTOR_INTERNAL_MANIFEST_SHA256,
    CLASSIFIER,
    DEFAULT_EFFECTIVE_ACC_BUDGET_BPW,
    PREREG_LAW_DECAY_DEN,
    PREREG_LAW_DECAY_NUM,
    PREREG_LAW_WINDOW_K,
    sha256_file,
)

REPO = Path("/mnt/c/Users/gabes/projects/claw-code-hrm-text-158")
HEAD = "24c19521e6b453dcb011a1dd57fdc37312196e28"
STEP2_RUN_ID = "FREE_SLICE5_STEP2"
STEP2_MAX_SILENT_PHASE_SECONDS = 600
STEP2_CONFIRMATION_STEPS = 200
MAX_SILENT_PHASE_FLAG = "--max-silent-phase-seconds"
B1_RUN_ID_FAMILY_RE = re.compile(r"2189e720(?:0[1-9]|1[0-7])")
_HEREDOC_SCRUB_KEYS = (
    "cheap_observer_no_rebuild_preflight_command",
    "run_root_free_assert_command",
)
ACTIVE_TASK_ID = "1783272482268-052281aa"
DISPATCH_MSG_ID = "1783325394446-3e70a184"
PLAN_MSG_ID = "1783325727139-7cc2d891"
GATE1_FREEZE_MSG_ID = "1783325898873-e64dc663"
CO_LEAD_GATE2_MSG_ID = "1783327022964"
IMPLEMENT_GATE_MSG_ID = "1783327116043-f15b0bc8"
B1_PACKET_REVISION = "v2_rev4e_h200_decensor_relaunch"
PACKET_REVISION = "v1_arc2b_slice5_step2_in_vivo"
CANONICAL_LANE = "test-operator_gpu_arc2b_slice5_step2_in_vivo"

DRAFT = (
    REPO
    / "artifacts/consensus_prep/arc2b_slice5_step2_in_vivo_gpu_launch_packet_v1_draft.json"
)
REPLAY = (
    REPO
    / "artifacts/consensus_prep/arc2b_slice5_step2_in_vivo_gpu_launch_packet_v1_replay_commands.json"
)

CLASSIFIER_MODULE = REPO / "calm/hrm_text_158/native_full_stack/arc2b_slice5_in_vivo_branch.py"
POSTRUN_SCRIPT = REPO / "scripts/hrm_text_158_arc2b_slice5_step2_in_vivo_postrun_classifier.py"
B1_REPLAY_TEMPLATE = (
    REPO
    / "artifacts/consensus_prep/d_recompute_window_feasibility_gpu_launch_packet_v2_h200_relaunch_replay_commands.json"
)

PARENT_REL = (
    "calm/hrm/checkpoints/hrm_text_158_phase3_L0c1_seed0017_replay83_n12k_lr7p5e5_pc1p0_rsL0b1math1r1b2_1_anchorsv1r3_from_L0b_final_step01500.pt"
)
PARENT_SHA256 = "9b4e311a22787e7d4808bde7bc2953568d767a2ee8ac648942a3f5dbb7b4d5ec"

STEP2_CARRIER_REQUIRED = (
    "--persistent-accumulator-event-coded-live",
    "--persistent-q-ternary-base3-codec",
    "--event-coded-sparse-vote-authority",
    "--d-live-carrier-snapshot",
)
STEP2_DECAY_FLAGS = (
    "--vote-update-decay-numerator",
    "1",
    "--vote-update-decay-denominator",
    "2",
)
STEP2_EVENT_CODED_RECOMPUTE_WINDOW_LOG_FLAG = "--event-coded-recompute-window-log"
STEP2_EVENT_CODED_MEASUREMENT_FLAGS = (
    STEP2_CARRIER_REQUIRED
    + STEP2_DECAY_FLAGS
    + (STEP2_EVENT_CODED_RECOMPUTE_WINDOW_LOG_FLAG,)
)
FORBIDDEN_CARRIER_TOKENS = (
    "--dense-accumulator-w8-clip",
    "--dense-accumulator-w7-clip",
    "HRM_TEXT_158_RUN_NARROW_CARRIER_W8_TRAINER_INTEGRATION",
)
EVENT_CODED_INCOMPATIBLE_BOOLEAN_FLAGS = (
    "--two-tier-carry-w6",
    "--b2b-sequential-capture",
    "--votes-emit-enabled",
    "--carrier-growth-enabled",
    "--d-recompute-window-instrumentation",
)
EVENT_CODED_INCOMPATIBLE_VALUE_FLAGS = (
    "--d-recompute-calibration-warmup-out",
)
EVENT_CODED_INCOMPATIBLE_FLAGS = (
    EVENT_CODED_INCOMPATIBLE_BOOLEAN_FLAGS + EVENT_CODED_INCOMPATIBLE_VALUE_FLAGS
)
STEP2_FORBIDDEN_LAUNCH_SEQUENCE_PATTERNS = (
    "confirmation_without_scale_smoke_pass",
    "reuse_run_id_2189e72001_through_17",
    "per_session_launch_without_prior_scratch_wipe",
    "confirmation_without_cheap_observer_preflight_pass",
    "abort_launch_sequence_on_nonzero_confirmation_before_hygiene_classifier",
    "baseline_d_off_smoke_used_as_launch_eligibility_receipt",
    "event_coded_measurement_with_d_recompute_window_instrumentation",
    "event_coded_measurement_with_probe_incompatible_flags",
)
CALIBRATION_WARMUP_RETRY_WITNESS_SCHEMA = (
    "hrm_text_158_arc2b_slice5_calibration_warmup_retry_witness/v1"
)
WARMUP_RETRY_MAX_ATTEMPTS = 2


def git_head() -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


def _scrub_stale_b1_run_ids_in_heredocs(replay: dict[str, Any], run_id: str) -> None:
    for key in _HEREDOC_SCRUB_KEYS:
        if key not in replay:
            continue
        replay[key] = B1_RUN_ID_FAMILY_RE.sub(run_id, str(replay[key]))


def verify_no_stale_b1_run_id_in_heredocs(replay: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    for key in _HEREDOC_SCRUB_KEYS:
        body = str(replay.get(key) or "")
        hits = sorted(set(B1_RUN_ID_FAMILY_RE.findall(body)))
        if hits:
            failures.append(f"stale_b1_run_id_in_heredoc:{key}:{hits}")
    return failures


def verify_no_unsatisfiable_input_manifest_bind(replay: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if "postrun_input_manifest_bind_command" in replay:
        failures.append("postrun_input_manifest_bind_command_present")
    launch_sequence = list(replay.get("launch_sequence") or [])
    if "postrun_input_manifest_bind_command" in launch_sequence:
        failures.append("postrun_input_manifest_bind_in_launch_sequence")
    return failures


def _strip_event_coded_incompatible_flags(command: str) -> str:
    out = command
    for flag in EVENT_CODED_INCOMPATIBLE_BOOLEAN_FLAGS:
        out = re.sub(rf" {re.escape(flag)}(?=\s|$)", "", out)
    for flag in EVENT_CODED_INCOMPATIBLE_VALUE_FLAGS:
        out = re.sub(rf" {re.escape(flag)}(?:=\S+|\s+\S+)", "", out)
    return out


def _replace_w8_with_event_coded(command: str) -> str:
    out = command
    out = out.replace(
        "HRM_TEXT_158_RUN_NARROW_CARRIER_W8_TRAINER_INTEGRATION=1 ",
        "",
    )
    for token in ("--dense-accumulator-w8-clip", "--dense-accumulator-w7-clip"):
        out = out.replace(f" {token}", "")
    out = _strip_event_coded_incompatible_flags(out)
    insert = " ".join(STEP2_EVENT_CODED_MEASUREMENT_FLAGS)
    marker = "--phase d-recompute-window-feasibility"
    if marker in out:
        out = out.replace(marker, f"{marker} {insert}", 1)
    return out


PROBE_SCRIPT_MARKER = "hrm_text_158_bounded_delta_acquisition_probe.py"
BASH_C_PROBE_PIPE_MARKERS = (" 2>&1 |", " 2>&1|")


def _max_silent_phase_token(*, seconds: int = STEP2_MAX_SILENT_PHASE_SECONDS) -> str:
    return f"{MAX_SILENT_PHASE_FLAG} {int(seconds)}"


def _strip_max_silent_phase_tokens(command: str) -> str:
    return re.sub(
        rf"{re.escape(MAX_SILENT_PHASE_FLAG)}(?:=\s*|\s+)\d+",
        "",
        command,
    ).rstrip()


def _strip_stranded_max_silent_phase_after_bash_c_close(command: str) -> str:
    """Remove --max-silent-phase-seconds stranded after the bash -c closing quote."""
    stripped = _strip_max_silent_phase_tokens(command)
    return re.sub(
        rf"'(\s+{re.escape(MAX_SILENT_PHASE_FLAG)}\s+\d+)+\s*$",
        "'",
        stripped,
    )


def _inject_max_silent_phase_direct(
    command: str,
    *,
    seconds: int = STEP2_MAX_SILENT_PHASE_SECONDS,
) -> str:
    if not command.strip():
        return command
    stripped = _strip_stranded_max_silent_phase_after_bash_c_close(command)
    stripped = _strip_max_silent_phase_tokens(stripped)
    return f"{stripped.rstrip()} {_max_silent_phase_token(seconds=seconds)}"


def _inject_max_silent_phase_into_bash_c_probe(
    command: str,
    *,
    seconds: int = STEP2_MAX_SILENT_PHASE_SECONDS,
) -> str:
    if not command.strip():
        return command
    stripped = _strip_stranded_max_silent_phase_after_bash_c_close(command)
    match = re.match(r"^bash -c '(.*)'$", stripped, re.DOTALL)
    if not match:
        return _inject_max_silent_phase_direct(stripped, seconds=seconds)

    inner = _strip_max_silent_phase_tokens(match.group(1))
    token = _max_silent_phase_token(seconds=seconds)
    if PROBE_SCRIPT_MARKER in inner:
        probe_idx = inner.find(PROBE_SCRIPT_MARKER)
        pipe_idx = len(inner)
        for pipe_marker in BASH_C_PROBE_PIPE_MARKERS:
            candidate = inner.find(pipe_marker, probe_idx)
            if candidate >= probe_idx:
                pipe_idx = min(pipe_idx, candidate)
        inner = inner[:pipe_idx].rstrip() + f" {token}" + inner[pipe_idx:]
    else:
        inner = f"{inner.rstrip()} {token}"
    return f"bash -c '{inner}'"


def _inject_max_silent_phase(
    command: str,
    *,
    seconds: int = STEP2_MAX_SILENT_PHASE_SECONDS,
) -> str:
    return _inject_max_silent_phase_direct(command, seconds=seconds)


def _append_producer_max_silent_phase(command: str) -> str:
    if MAX_SILENT_PHASE_FLAG in command:
        return _inject_max_silent_phase(command)
    return f"{command.rstrip()} {MAX_SILENT_PHASE_FLAG} {STEP2_MAX_SILENT_PHASE_SECONDS}"


def _warmup_liveness_failure_at(run_root: Path) -> bool:
    phase_path = run_root / "calibration_warmup" / "last_active_phase.json"
    if not phase_path.is_file():
        return False
    phase = json.loads(phase_path.read_text(encoding="utf-8"))
    return str(phase.get("failure_class")) == "LIVENESS_FAILURE"


def _run_shell_command(command: str) -> int:
    proc = subprocess.run(command, shell=True, check=False)
    return int(proc.returncode)


def execute_calibration_warmup_retry(
    *,
    run_root: Path,
    producer_command_template: str,
    scratch_wipe_template: str,
    producer_runner: Callable[[str], int] | None = None,
    max_attempts: int = WARMUP_RETRY_MAX_ATTEMPTS,
) -> dict[str, Any]:
    run_root = Path(run_root)
    witness_path = run_root / "prelaunch" / "calibration_warmup_retry_witness.json"
    witness_path.parent.mkdir(parents=True, exist_ok=True)

    producer_cmd = producer_command_template.replace("{run_root}", str(run_root))
    scratch_wipe = scratch_wipe_template.replace("{run_root}", str(run_root))
    runner = producer_runner or _run_shell_command

    attempt = 1
    retry_used = False
    scratch_wiped_between = False
    final_rc = 0
    final_reason = "success"

    while attempt <= int(max_attempts):
        final_rc = int(runner(producer_cmd))
        if final_rc == 0:
            final_reason = "success"
            break
        if _warmup_liveness_failure_at(run_root) and attempt < int(max_attempts):
            retry_used = True
            _run_shell_command(scratch_wipe)
            scratch_wiped_between = True
            attempt += 1
            continue
        final_reason = (
            "liveness_failure_exhausted_retries"
            if _warmup_liveness_failure_at(run_root)
            else "non_liveness_failure_no_retry"
        )
        break

    witness = {
        "schema": CALIBRATION_WARMUP_RETRY_WITNESS_SCHEMA,
        "max_attempts": int(max_attempts),
        "attempts_used": int(attempt),
        "retry_used": retry_used,
        "retry_trigger": "liveness_failure_only",
        "scratch_wiped_between_attempts": scratch_wiped_between,
        "final_rc": int(final_rc),
        "final_reason": final_reason,
    }
    witness_path.write_text(
        json.dumps(witness, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return witness


def _build_calibration_warmup_retry_command(
    producer_command: str,
    scratch_wipe: str,
) -> str:
    producer_json = json.dumps(producer_command)
    scratch_json = json.dumps(scratch_wipe)
    return (
        "PYTHONPATH=. python3 - \"{run_root}\" <<'PY'\n"
        "import sys\n"
        "from pathlib import Path\n"
        "from scripts.apply_arc2b_slice5_step2_in_vivo_gpu_launch_packet import (\n"
        "    execute_calibration_warmup_retry,\n"
        ")\n"
        f"PRODUCER_TEMPLATE = {producer_json}\n"
        f"SCRATCH_WIPE_TEMPLATE = {scratch_json}\n"
        "witness = execute_calibration_warmup_retry(\n"
        "    run_root=Path(sys.argv[1]),\n"
        "    producer_command_template=PRODUCER_TEMPLATE,\n"
        "    scratch_wipe_template=SCRATCH_WIPE_TEMPLATE,\n"
        ")\n"
        "raise SystemExit(0 if int(witness.get('final_rc') or 0) == 0 else int(witness['final_rc']))\n"
        "PY"
    )


def _confirmation_probe_argv_receives_max_silent_phase(
    command: str,
    *,
    seconds: int = STEP2_MAX_SILENT_PHASE_SECONDS,
) -> bool:
    if "bash -c '" not in command:
        return False
    if re.search(
        rf"exit 0'\s+{re.escape(MAX_SILENT_PHASE_FLAG)}\s+{int(seconds)}",
        command,
    ):
        return False
    match = re.match(r"^bash -c '(.*)'$", command.strip(), re.DOTALL)
    if not match:
        return False
    inner = match.group(1)
    if PROBE_SCRIPT_MARKER not in inner:
        return False
    token = _max_silent_phase_token(seconds=seconds)
    probe_idx = inner.find(PROBE_SCRIPT_MARKER)
    pipe_idx = len(inner)
    for pipe_marker in BASH_C_PROBE_PIPE_MARKERS:
        candidate = inner.find(pipe_marker, probe_idx)
        if candidate >= probe_idx:
            pipe_idx = min(pipe_idx, candidate)
    probe_segment = inner[probe_idx:pipe_idx]
    return token in probe_segment


def _direct_probe_argv_receives_max_silent_phase(
    body: str,
    *,
    seconds: int = STEP2_MAX_SILENT_PHASE_SECONDS,
) -> bool:
    if re.search(
        rf"exit 0'\s+{re.escape(MAX_SILENT_PHASE_FLAG)}\s+{int(seconds)}",
        body,
    ):
        return False
    if "bash -c '" in body and PROBE_SCRIPT_MARKER in body:
        return _confirmation_probe_argv_receives_max_silent_phase(body, seconds=seconds)
    return _max_silent_phase_token(seconds=seconds) in body


def _warmup_producer_receives_max_silent_phase(
    body: str,
    *,
    seconds: int = STEP2_MAX_SILENT_PHASE_SECONDS,
) -> bool:
    token = _max_silent_phase_token(seconds=seconds)
    match = re.search(r'PRODUCER_TEMPLATE = "(.*)"', body, re.DOTALL)
    if match is None:
        return token in body
    return token in match.group(1)


def verify_probe_receives_max_silent_phase_seconds(replay: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    seconds = STEP2_MAX_SILENT_PHASE_SECONDS
    checks: dict[str, Callable[[str], bool]] = {
        "scale_smoke_command": lambda body: _direct_probe_argv_receives_max_silent_phase(
            body,
            seconds=seconds,
        ),
        "confirmation_launch_command": lambda body: _confirmation_probe_argv_receives_max_silent_phase(
            body,
            seconds=seconds,
        ),
        "shared_probe_argv": lambda body: _max_silent_phase_token(seconds=seconds) in body,
        "calibration_warmup_command": lambda body: _warmup_producer_receives_max_silent_phase(
            body,
            seconds=seconds,
        ),
    }
    for key, checker in checks.items():
        body = str(replay.get(key) or "")
        if not body:
            failures.append(f"missing_command:{key}")
            continue
        if not checker(body):
            failures.append(f"probe_max_silent_phase_not_on_probe_argv:{key}")
        if key == "confirmation_launch_command" and re.search(
            rf"exit 0'\s+{re.escape(MAX_SILENT_PHASE_FLAG)}",
            body,
        ):
            failures.append(f"stranded_max_silent_phase_after_bash_c_close:{key}")
    return failures


def verify_explicit_max_silent_phase_seconds(replay: dict[str, Any]) -> list[str]:
    return verify_probe_receives_max_silent_phase_seconds(replay)


def verify_event_coded_incompatible_flags_absent(replay: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    checked_keys = (
        "scale_smoke_command",
        "confirmation_launch_command",
        "shared_probe_argv",
    )
    for key in checked_keys:
        body = str(replay.get(key) or "")
        if "--event-coded-sparse-vote-authority" not in body:
            continue
        for flag in EVENT_CODED_INCOMPATIBLE_FLAGS:
            if flag in body:
                failures.append(f"incompatible_flag_present:{key}:{flag}")
    return failures


def verify_event_coded_recompute_window_log_flag(replay: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    measurement_keys = (
        "scale_smoke_command",
        "confirmation_launch_command",
        "shared_probe_argv",
    )
    for key in measurement_keys:
        body = str(replay.get(key) or "")
        if "--event-coded-sparse-vote-authority" not in body:
            continue
        if STEP2_EVENT_CODED_RECOMPUTE_WINDOW_LOG_FLAG not in body:
            failures.append(f"missing_event_coded_recompute_window_log:{key}")
        if "--d-recompute-window-instrumentation" in body:
            failures.append(f"d_instrumentation_present_on_event_coded:{key}")
    warmup = str(replay.get("calibration_warmup_command") or "")
    if STEP2_EVENT_CODED_RECOMPUTE_WINDOW_LOG_FLAG in warmup:
        failures.append("event_coded_recompute_window_log_on_warmup")
    return failures


def verify_forbidden_launch_sequence_patterns_reconciled(replay: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    patterns = list(replay.get("forbidden_launch_sequence_patterns") or [])
    stale = (
        "confirmation_without_d_recompute_window_instrumentation_flag",
        "scale_smoke_without_d_instrumentation_as_launch_gate",
    )
    for entry in stale:
        if entry in patterns:
            failures.append(f"stale_forbidden_pattern_present:{entry}")
    required = (
        "event_coded_measurement_with_d_recompute_window_instrumentation",
        "event_coded_measurement_with_probe_incompatible_flags",
    )
    for entry in required:
        if entry not in patterns:
            failures.append(f"missing_event_coded_forbidden_pattern:{entry}")
    return failures


def verify_warmup_retry_metadata(replay: dict[str, Any], packet: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    retry_policy = dict(packet.get("liveness_retry_policy") or {})
    if retry_policy.get("max_attempts") != 2:
        failures.append("liveness_retry_policy_max_attempts_not_2")
    phases = list(retry_policy.get("phases") or [])
    if phases != ["calibration_warmup"]:
        failures.append("liveness_retry_policy_phases_mismatch")
    warmup_cmd = str(replay.get("calibration_warmup_command") or "")
    if "execute_calibration_warmup_retry" not in warmup_cmd:
        failures.append("warmup_retry_wrapper_missing_execute_calibration_warmup_retry")
    if retry_policy.get("witness_receipt") != "prelaunch/calibration_warmup_retry_witness.json":
        failures.append("warmup_retry_witness_receipt_path_mismatch")
    if retry_policy.get("non_liveness_failure_no_retry") is not True:
        failures.append("warmup_retry_non_liveness_guard_missing")
    if retry_policy.get("retry_trigger") != "liveness_failure_only":
        failures.append("warmup_retry_trigger_mismatch")
    return failures


def build_liveness_contract() -> dict[str, Any]:
    return {
        "max_silent_phase_seconds": STEP2_MAX_SILENT_PHASE_SECONDS,
        "coherence": (
            "explicit 600s guard on warmup/scale_smoke/confirmation/shared_probe_argv; "
            "probe implicit GPU default remains 300s without explicit flag"
        ),
        "watcher_liveness_fail_regex": (
            "LIVENESS_FAIL_KERNELIZED_BUT_STALLED|phase_milestone_stall|"
            "LIVENESS_FAIL_TOTAL_TIMEOUT|total_timeout|LIVENESS_FAILURE|LIVENESS_FAIL"
        ),
        "warmup_liveness_failure_is_operational_not_mechanism_terminal": True,
    }


def build_liveness_retry_policy() -> dict[str, Any]:
    return {
        "max_attempts": 2,
        "phases": ["calibration_warmup"],
        "retry_trigger": "liveness_failure_only",
        "scratch_wipe_between_attempts": True,
        "witness_receipt": "prelaunch/calibration_warmup_retry_witness.json",
        "non_liveness_failure_no_retry": True,
    }


def build_replay_commands(classifier_sha: str) -> dict[str, Any]:
    template = json.loads(B1_REPLAY_TEMPLATE.read_text(encoding="utf-8"))
    replay: dict[str, Any] = dict(template)
    replay["packet_revision"] = PACKET_REVISION
    replay["binds_main_packet"] = str(DRAFT.relative_to(REPO))
    replay["run_id"] = STEP2_RUN_ID
    replay["run_root"] = (
        "/home/gabe/claw-code-creditdir/transient_fp_credit/"
        "arc2b_slice5_step2_in_vivo_seed43_{run_id}/"
    )
    _scrub_stale_b1_run_ids_in_heredocs(replay, STEP2_RUN_ID)
    replay["schema"] = "hrm_text_158_arc2b_slice5_step2_in_vivo_launch_packet_replay_commands/v1"
    replay["gpu_hold"] = True
    replay["planning_only"] = False
    replay["note"] = (
        "Slice-5 Step-2 in-vivo: event-coded live carrier + decay 1/2 + "
        "Slice-5 postrun classifier"
    )

    for key in ("scale_smoke_command", "confirmation_launch_command"):
        if key in replay:
            replay[key] = _replace_w8_with_event_coded(str(replay[key]))

    if "scale_smoke_command" in replay:
        replay["scale_smoke_command"] = _inject_max_silent_phase_direct(
            str(replay["scale_smoke_command"])
        )
    if "confirmation_launch_command" in replay:
        replay["confirmation_launch_command"] = _inject_max_silent_phase_into_bash_c_probe(
            str(replay["confirmation_launch_command"])
        )
    replay["shared_probe_argv"] = _inject_max_silent_phase(
        _replace_w8_with_event_coded(str(replay.get("shared_probe_argv", "")))
    )

    producer_warmup = _append_producer_max_silent_phase(
        str(replay.get("calibration_warmup_command") or "")
    )
    scratch_wipe = str(
        (replay.get("scratch_wipe_commands") or {}).get("calibration_warmup") or ""
    )
    replay["calibration_warmup_command"] = _build_calibration_warmup_retry_command(
        producer_warmup,
        scratch_wipe,
    )

    replay.pop("baseline_liveness_telemetry_command", None)
    replay.pop("postrun_input_manifest_bind_command", None)
    launch_sequence = [
        step
        for step in list(replay.get("launch_sequence") or [])
        if step
        not in (
            "baseline_liveness_telemetry_command",
            "postrun_input_manifest_bind_command",
        )
    ]

    replay["shared_gpu_env"] = (
        "PYTHONPATH=. HRM_TEXT_158_ALLOW_C2_GPU_LAUNCH=1 HRM_TEXT_158_RUN_C2_ACQUISITION_PROBE=1"
    )
    replay["forbidden_launch_sequence_patterns"] = list(
        STEP2_FORBIDDEN_LAUNCH_SEQUENCE_PATTERNS
    )

    replay["postrun_command"] = (
        "bash -c 'RC=0; PYTHONPATH=. timeout 1800 python3 "
        "scripts/hrm_text_158_arc2b_slice5_step2_in_vivo_postrun_classifier.py "
        "--run-root {run_root} "
        f"--packet {DRAFT.relative_to(REPO)} "
        "--repo-root . "
        "--out {run_root}/arc2b_slice5_in_vivo_law_validation_receipt.json "
        "|| RC=$?; exit $RC'"
    )

    replay["decay_replay_constants_witness_command"] = (
        "PYTHONPATH=. python3 - \"{run_root}\" <<'PY'\n"
        "import json, sys\nfrom pathlib import Path\n"
        "run_root = Path(sys.argv[1])\n"
        "log = run_root / 'd_recompute_window_diagnostic' / 'recompute_window_log.jsonl'\n"
        "failures = []\n"
        "if not log.is_file():\n    failures.append('missing_log')\n"
        "else:\n"
        "    row = json.loads(next(l for l in log.read_text().splitlines() if l.strip()))\n"
        "    rc = row.get('replay_constants') or {}\n"
        "    if int(rc.get('decay_numerator', -1)) != 1 or int(rc.get('decay_denominator', -1)) != 2:\n"
        "        failures.append('decay_not_1_over_2')\n"
        "receipt = {'pass': not failures, 'failures': failures}\n"
        "out = run_root / 'prelaunch' / 'decay_replay_constants_witness.json'\n"
        "out.parent.mkdir(parents=True, exist_ok=True)\n"
        "out.write_text(json.dumps(receipt, indent=2) + '\\n')\n"
        "if failures: raise SystemExit(1)\nPY"
    )
    replay["classifier_sha_pin_witness_command"] = (
        "PYTHONPATH=. python3 -c \"import hashlib; from pathlib import Path; "
        f"p=Path('{CLASSIFIER_MODULE.relative_to(REPO)}'); "
        f"expected='{classifier_sha}'; actual=hashlib.sha256(p.read_bytes()).hexdigest(); "
        "assert actual==expected, f'classifier_sha_mismatch {actual}!={expected}'\""
    )
    replay["forbidden_carrier_tokens_absent_witness_command"] = (
        "PYTHONPATH=. python3 - \"{run_root}\" <<'PY'\n"
        "import json, sys\nfrom pathlib import Path\n"
        f"forbidden = {list(FORBIDDEN_CARRIER_TOKENS)!r}\n"
        f"incompatible = {list(EVENT_CODED_INCOMPATIBLE_FLAGS)!r}\n"
        "cmds = json.loads(Path('artifacts/consensus_prep/"
        "arc2b_slice5_step2_in_vivo_gpu_launch_packet_v1_replay_commands.json').read_text())\n"
        "checked = {k: cmds.get(k) for k in ('scale_smoke_command', 'confirmation_launch_command', 'shared_probe_argv')}\n"
        "blob = json.dumps(checked)\n"
        "hits = [t for t in forbidden if t in blob]\n"
        "if hits: raise SystemExit(f'forbidden_tokens_present:{hits}')\n"
        "incompatible_hits = [t for t in incompatible if t in blob]\n"
        "if incompatible_hits: raise SystemExit(f'incompatible_flags_present:{incompatible_hits}')\n"
        "print(json.dumps({'pass': True, 'forbidden_checked': forbidden, 'incompatible_checked': incompatible}))\nPY"
    )

    launch_sequence = list(launch_sequence)
    for witness in (
        "decay_replay_constants_witness_command",
        "classifier_sha_pin_witness_command",
        "forbidden_carrier_tokens_absent_witness_command",
    ):
        if witness not in launch_sequence:
            idx = launch_sequence.index("postrun_command")
            launch_sequence.insert(idx, witness)
    replay["launch_sequence"] = launch_sequence
    return replay


def build_packet(classifier_sha: str, replay_sha: str) -> dict[str, Any]:
    return {
        "schema": "hrm_text_158_arc2b_slice5_step2_in_vivo_gpu_launch_packet/v1",
        "task_id": ACTIVE_TASK_ID,
        "fold": "arc2b_slice5_step2_in_vivo_gpu_launch",
        "packet_revision": PACKET_REVISION,
        "git_head_required": HEAD,
        "canonical_lane": CANONICAL_LANE,
        "provenance": {
            "dispatch_msg_id": DISPATCH_MSG_ID,
            "plan_msg_id": PLAN_MSG_ID,
            "gate1_freeze_msg_id": GATE1_FREEZE_MSG_ID,
            "co_lead_gate2_msg_id": CO_LEAD_GATE2_MSG_ID,
            "implement_gate_msg_id": IMPLEMENT_GATE_MSG_ID,
            "b1_packet_revision": B1_PACKET_REVISION,
        },
        "parent_checkpoint": {
            "path": PARENT_REL,
            "sha256": PARENT_SHA256,
            "curriculum_seed": 43,
            "support_order_seed": 43,
            "eligible_scope": "all-bitlinear",
        },
        "law_under_test": {
            "window_k": PREREG_LAW_WINDOW_K,
            "decay_num": PREREG_LAW_DECAY_NUM,
            "decay_den": PREREG_LAW_DECAY_DEN,
            "effective_acc_budget_bpw": DEFAULT_EFFECTIVE_ACC_BUDGET_BPW,
            "tolerance_bpw": 0.0,
        },
        "runtime_requirements": {
            "decay_active": {"decay_num": 1, "decay_den": 2},
            "resume_generation": 0,
            "horizon_h": 200,
            "confirmation_steps": STEP2_CONFIRMATION_STEPS,
            "from_clean_parent_contiguous": True,
            "live_carrier_snapshot_required": True,
            "live_carrier_bytes_exact_required": True,
            "carrier_authority_required_argv": list(STEP2_CARRIER_REQUIRED),
            "carrier_authority_forbidden_tokens": list(FORBIDDEN_CARRIER_TOKENS),
            "decay_cli_flags": list(STEP2_DECAY_FLAGS),
            "max_silent_phase_seconds": STEP2_MAX_SILENT_PHASE_SECONDS,
        },
        "liveness_contract": build_liveness_contract(),
        "liveness_retry_policy": build_liveness_retry_policy(),
        "classifier_binding": {
            "classifier": CLASSIFIER,
            "module_path": str(CLASSIFIER_MODULE.relative_to(REPO)),
            "module_sha256": classifier_sha,
            "postrun_script": str(POSTRUN_SCRIPT.relative_to(REPO)),
            "receipt_schema": "hrm_text_158_arc2b_slice5_in_vivo_law_validation_receipt/v1",
            "evidence_source": "step2_gpu_live_carrier",
            "mechanism_terminal_branches": [
                "D_NEEDS_UPDATE_LAW_REDESIGN",
                "SLICE5_IN_VIVO_LAW_BOUND",
            ],
            "honest_expected_terminal": "D_NEEDS_UPDATE_LAW_REDESIGN",
        },
        "readiness_classification": {
            "class": "pre_full_stack_diagnostic",
            "flags": {
                "ready_for_main_science": False,
                "counts_as_sub2": False,
                "pre_full_stack_diagnostic": True,
            },
        },
        "manifest_binding": {
            "recorded_selector_internal_manifest_sha256": (
                B1_RECORDED_SELECTOR_INTERNAL_MANIFEST_SHA256
            ),
        },
        "replay_commands_artifact": str(REPLAY.relative_to(REPO)),
        "replay_commands_sha256": replay_sha,
        "run_id_policy": "FREE — do NOT reuse B1 run_ids 2189e72001..17",
        "run_root_template": (
            "/home/gabe/claw-code-creditdir/transient_fp_credit/"
            "arc2b_slice5_step2_in_vivo_seed43_{run_id}/"
        ),
        "gpu_hold": True,
        "explicit_non_claims": [
            "not_sub2_readiness",
            "not_reduction_eligibility",
            "not_430mb_bank_pin",
            "not_fold3b_universalization",
            "not_gpu_launch_until_plus1_launch_gate",
        ],
    }


def verify_replay_commands(replay: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    checked_keys = (
        "scale_smoke_command",
        "confirmation_launch_command",
        "shared_probe_argv",
    )
    blob = json.dumps({key: replay.get(key) for key in checked_keys})
    for token in FORBIDDEN_CARRIER_TOKENS:
        if token in blob:
            failures.append(f"forbidden_token_present:{token}")
    for flag in STEP2_CARRIER_REQUIRED:
        if flag not in blob:
            failures.append(f"missing_required_carrier_flag:{flag}")
    for flag in STEP2_DECAY_FLAGS:
        if flag not in blob:
            failures.append(f"missing_decay_flag:{flag}")
    if STEP2_EVENT_CODED_RECOMPUTE_WINDOW_LOG_FLAG not in blob:
        failures.append("missing_event_coded_recompute_window_log_flag")
    confirmation = str(replay.get("confirmation_launch_command") or "")
    if "--vote-update-decay-numerator" not in confirmation:
        failures.append("confirmation_missing_decay_numerator_flag")
    if "arc2b_slice5_step2_in_vivo_postrun_classifier" not in str(
        replay.get("postrun_command") or ""
    ):
        failures.append("postrun_not_slice5_classifier")
    failures.extend(verify_no_stale_b1_run_id_in_heredocs(replay))
    failures.extend(verify_no_unsatisfiable_input_manifest_bind(replay))
    failures.extend(verify_explicit_max_silent_phase_seconds(replay))
    failures.extend(verify_event_coded_incompatible_flags_absent(replay))
    failures.extend(verify_event_coded_recompute_window_log_flag(replay))
    failures.extend(verify_forbidden_launch_sequence_patterns_reconciled(replay))
    return failures


def verify_packet(packet: dict[str, Any], replay: dict[str, Any]) -> list[str]:
    failures = verify_replay_commands(replay)
    failures.extend(verify_warmup_retry_metadata(replay, packet))
    if packet.get("git_head_required") != HEAD:
        failures.append("git_head_required_stale")
    return failures


def self_verify() -> dict[str, Any]:
    failures: list[str] = []
    if not DRAFT.is_file() or not REPLAY.is_file():
        failures.extend(["draft_missing", "replay_missing"])
        return {"ok": False, "failures": failures}

    classifier_sha = sha256_file(CLASSIFIER_MODULE)
    packet = json.loads(DRAFT.read_text(encoding="utf-8"))
    replay = json.loads(REPLAY.read_text(encoding="utf-8"))

    if packet.get("classifier_binding", {}).get("module_sha256") != classifier_sha:
        failures.append("classifier_sha_pin_drift")
    failures.extend(verify_packet(packet, replay))

    regen_classifier_sha = sha256_file(CLASSIFIER_MODULE)
    regen_replay = build_replay_commands(regen_classifier_sha)
    regen_packet = build_packet(
        regen_classifier_sha,
        hashlib.sha256(
            (json.dumps(regen_replay, indent=2, sort_keys=True) + "\n").encode()
        ).hexdigest(),
    )
    draft_sha = hashlib.sha256(DRAFT.read_bytes()).hexdigest()
    replay_sha = hashlib.sha256(REPLAY.read_bytes()).hexdigest()
    regen_draft_sha = hashlib.sha256(
        (json.dumps(regen_packet, indent=2, sort_keys=True) + "\n").encode()
    ).hexdigest()
    regen_replay_sha = hashlib.sha256(
        (json.dumps(regen_replay, indent=2, sort_keys=True) + "\n").encode()
    ).hexdigest()

    live_git_head = git_head()
    pins_match_commit = live_git_head == HEAD
    if not pins_match_commit:
        failures.append("pins_match_commit")

    return {
        "ok": (
            not failures
            and pins_match_commit
            and draft_sha == regen_draft_sha
            and replay_sha == regen_replay_sha
        ),
        "failures": failures,
        "deterministic_regen": draft_sha == regen_draft_sha and replay_sha == regen_replay_sha,
        "draft_sha256": draft_sha,
        "replay_sha256": replay_sha,
        "classifier_module_sha256": classifier_sha,
        "git_head": live_git_head,
        "pins_match_commit": pins_match_commit,
    }


def main() -> int:
    classifier_sha = sha256_file(CLASSIFIER_MODULE)
    replay = build_replay_commands(classifier_sha)
    replay_bytes = (json.dumps(replay, indent=2, sort_keys=True) + "\n").encode()
    replay_sha = hashlib.sha256(replay_bytes).hexdigest()
    packet = build_packet(classifier_sha, replay_sha)

    DRAFT.parent.mkdir(parents=True, exist_ok=True)
    DRAFT.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    REPLAY.write_bytes(replay_bytes)

    result = self_verify()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
