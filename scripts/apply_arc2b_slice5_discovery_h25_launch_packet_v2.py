#!/usr/bin/env python3
"""Apply Arc #2b Slice-5 discovery H=25 launch-ready packet v2 (5 co_lead fixes).

Fixes: (1) real gap(D) witness computing live_acc_carrier_bpw_max + budget_gap_bpw;
C/E gated behind D. (2) forbidden-token witness bound to per-arm artifact. (3) live
postrun (not Arm-B K*). (4) scale_smoke steps 200→25. (5) drop W8 env in warmup.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from calm.hrm_text_158.native_full_stack.arc2b_slice5_in_vivo_branch import sha256_file
from calm.hrm_text_158.native_full_stack.arc2b_slice5_discovery_branch import (
    CLASSIFIER,
    DECAY_POINT_C_DEN,
    DECAY_POINT_C_NUM,
    DECAY_POINT_D_DEN,
    DECAY_POINT_D_NUM,
    DECAY_POINT_E_DEN,
    DECAY_POINT_E_NUM,
    DEFAULT_DIRECTION_TOL_FACTOR,
    DEFAULT_EFFECTIVE_ACC_BUDGET_BPW,
    DEFAULT_MATERIALITY_FACTOR,
    RECEIPT_SCHEMA,
)

REPO = Path("/mnt/c/Users/gabes/projects/claw-code-hrm-text-158")
HEAD = "406fe012d97fdfe4d399f5e7e8da3d285d9b6c20"
ACTIVE_TASK_ID = "1783272482268-052281aa"
PACKET_REVISION = "v2_arc2b_slice5_discovery_h25_launch_ready"
CANONICAL_LANE = "test-operator_gpu_arc2b_slice5_discovery_h25"

DISPATCH_MSG_ID = "1783529234305-834187b2"
PLAN_MSG_ID = "1783512246673-8b8492e8"
GATE1_FREEZE_MSG_ID = "1783512308286-b3269c20"
CO_LEAD_GATE2_MSG_ID = "1783512484577-a5195760"
IMPLEMENT_GATE_MSG_ID = "1783526612437-ed9e3bba"
HARNESS_COMMIT_SHA = "406fe012d97fdfe4d399f5e7e8da3d285d9b6c20"

DRAFT = REPO / "artifacts/consensus_prep/arc2b_slice5_discovery_h25_launch_packet_v2_draft.json"
REPLAY = REPO / "artifacts/consensus_prep/arc2b_slice5_discovery_h25_launch_packet_v2_replay_commands.json"
CLASSIFIER_MODULE = REPO / "calm/hrm_text_158/native_full_stack/arc2b_slice5_discovery_branch.py"
B1_REPLAY_TEMPLATE = REPO / "artifacts/consensus_prep/arc2b_slice5_step2_in_vivo_gpu_launch_packet_v1_replay_commands.json"

PARENT_REL = "calm/hrm/checkpoints/hrm_text_158_phase3_L0c1_seed0017_replay83_n12k_lr7p5e5_pc1p0_rsL0b1math1r1b2_1_anchorsv1r3_from_L0b_final_step01500.pt"
PARENT_SHA256 = "9b4e311a22787e7d4808bde7bc2953568d767a2ee8ac648942a3f5dbb7b4d5ec"

H25_STEPS = 25
H25_MAX_STEPS_HARD = 25
ELIGIBLE_MODULE_LIMIT = 8
ELIGIBLE_MODULE_LIMIT_NUMEL = 8_650_752
MAX_SILENT_PHASE_SECONDS = 600
STEP_TIME_BOUND_SECONDS = 300

CARRIER_REQUIRED_FLAGS = (
    "--persistent-accumulator-event-coded-live",
    "--persistent-q-ternary-base3-codec",
    "--event-coded-sparse-vote-authority",
    "--d-live-carrier-snapshot",
)
FORBIDDEN_CARRIER_TOKENS = (
    "--dense-accumulator-w8-clip",
    "--dense-accumulator-w7-clip",
    "HRM_TEXT_158_RUN_NARROW_CARRIER_W8_TRAINER_INTEGRATION",
)
EVENT_CODED_INCOMPATIBLE_FLAGS = (
    "--two-tier-carry-w6",
    "--b2b-sequential-capture",
    "--votes-emit-enabled",
    "--carrier-growth-enabled",
    "--d-recompute-window-instrumentation",
)
EVENT_CODED_RECOMPUTE_WINDOW_LOG_FLAG = "--event-coded-recompute-window-log"

ARM_DECAY = {
    "arm_c": (DECAY_POINT_C_NUM, DECAY_POINT_C_DEN),
    "arm_d": (DECAY_POINT_D_NUM, DECAY_POINT_D_DEN),
    "arm_e": (DECAY_POINT_E_NUM, DECAY_POINT_E_DEN),
}
ARM_ORDERING = ("arm_d", "arm_c", "arm_e")

B1_RUN_ID_FAMILY_RE = re.compile(r"2189e720(?:0[1-9]|1[0-7])")
_HEREDOC_SCRUB_KEYS = ("cheap_observer_no_rebuild_preflight_command", "run_root_free_assert_command")
MAX_SILENT_PHASE_FLAG = "--max-silent-phase-seconds"


def _inject_profile_host_rss_env(command: str) -> str:
    token = "HRM_TEXT_158_PROFILE_HOST_RSS=1"
    if token in command:
        return command
    match = re.match(r"^bash -c '(.*)'$", command, re.DOTALL)
    if match:
        inner = match.group(1)
        if "set +e; " in inner:
            inner = inner.replace("set +e; ", f"set +e; {token} ", 1)
        else:
            inner = f"{token} {inner}"
        return f"bash -c '{inner}'"
    return f"{token} {command}"


def honest_confirmation_launch_rc(command: str) -> str:
    """Preserve captured probe RC instead of masking with unconditional exit 0."""
    if not command.strip():
        return command
    if "CONFIRMATION_RC=${PIPESTATUS[0]}" in command:
        return command.replace("exit 0'", "exit $CONFIRMATION_RC'", 1)
    return command


def git_head() -> str:
    proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, check=True, capture_output=True, text=True)
    return proc.stdout.strip()


def _scrub_stale_b1_run_ids_in_heredocs(replay: dict[str, Any], run_id: str) -> None:
    for key in _HEREDOC_SCRUB_KEYS:
        if key not in replay:
            continue
        replay[key] = B1_RUN_ID_FAMILY_RE.sub(run_id, str(replay[key]))


def _strip_event_coded_incompatible_flags(command: str) -> str:
    out = command
    for flag in EVENT_CODED_INCOMPATIBLE_FLAGS:
        out = re.sub(rf" {re.escape(flag)}(?=\s|$)", "", out)
    return out


def _replace_w8_with_event_coded(command: str) -> str:
    out = command
    out = out.replace("HRM_TEXT_158_RUN_NARROW_CARRIER_W8_TRAINER_INTEGRATION=1 ", "")
    for token in ("--dense-accumulator-w8-clip", "--dense-accumulator-w7-clip"):
        out = out.replace(f" {token}", "")
    out = _strip_event_coded_incompatible_flags(out)
    insert = " ".join(CARRIER_REQUIRED_FLAGS + (EVENT_CODED_RECOMPUTE_WINDOW_LOG_FLAG,))
    marker = "--phase d-recompute-window-feasibility"
    if marker in out:
        out = out.replace(marker, f"{marker} {insert}", 1)
    return out


def _strip_max_silent_phase_tokens(command: str) -> str:
    return re.sub(rf"{re.escape(MAX_SILENT_PHASE_FLAG)}(?:=\s*|\s+)\d+", "", command).rstrip()


def _inject_max_silent_phase_direct(command: str, *, seconds: int = MAX_SILENT_PHASE_SECONDS) -> str:
    if not command.strip():
        return command
    stripped = _strip_max_silent_phase_tokens(command)
    return f"{stripped.rstrip()} {MAX_SILENT_PHASE_FLAG} {int(seconds)}"


def _inject_max_silent_phase_into_bash_c_probe(command: str, *, seconds: int = MAX_SILENT_PHASE_SECONDS) -> str:
    if not command.strip():
        return command
    stripped = _strip_max_silent_phase_tokens(command)
    match = re.match(r"^bash -c '(.*)'$", stripped, re.DOTALL)
    if not match:
        return _inject_max_silent_phase_direct(stripped, seconds=seconds)
    inner = _strip_max_silent_phase_tokens(match.group(1))
    token = f"{MAX_SILENT_PHASE_FLAG} {int(seconds)}"
    probe_marker = "hrm_text_158_bounded_delta_acquisition_probe.py"
    if probe_marker in inner:
        probe_idx = inner.find(probe_marker)
        pipe_idx = len(inner)
        for pipe_marker in (" 2>&1 |", " 2>&1|"):
            candidate = inner.find(pipe_marker, probe_idx)
            if candidate >= probe_idx:
                pipe_idx = min(pipe_idx, candidate)
        inner = inner[:pipe_idx].rstrip() + f" {token}" + inner[pipe_idx:]
    else:
        inner = f"{inner.rstrip()} {token}"
    return f"bash -c '{inner}'"


def _set_decay_flags(command: str, decay_num: int, decay_den: int) -> str:
    out = re.sub(r"--vote-update-decay-numerator\s+\d+", f"--vote-update-decay-numerator {int(decay_num)}", command)
    out = re.sub(r"--vote-update-decay-denominator\s+\d+", f"--vote-update-decay-denominator {int(decay_den)}", out)
    return out


def _set_steps(command: str, steps: int, max_steps_hard: int) -> str:
    out = re.sub(r"--steps\s+\d+", f"--steps {int(steps)}", command)
    out = re.sub(r"--max-steps-hard\s+\d+", f"--max-steps-hard {int(max_steps_hard)}", out)
    return out


def _inject_eligible_module_limit(command: str, limit: int) -> str:
    if "--eligible-module-limit" in command:
        return re.sub(r"--eligible-module-limit\s+\d+", f"--eligible-module-limit {int(limit)}", command)
    marker = "--eligible-scope all-bitlinear"
    if marker in command:
        return command.replace(marker, f"{marker} --eligible-module-limit {int(limit)}", 1)
    return f"{command.rstrip()} --eligible-module-limit {int(limit)}"


def _build_gap_d_witness_command(decay_num: int, decay_den: int) -> str:
    """Fix 1: REAL gap(D) witness computing live_acc_carrier_bpw_max + budget_gap_bpw."""
    return (
        "PYTHONPATH=. python3 - \"{run_root}\" <<'PY'\n"
        "import json, sys\n"
        "from pathlib import Path\n"
        "from scripts.hrm_text_158_arc2b_slice5_discovery_live_postrun import build_live_postrun_receipt\n"
        "run_root = Path(sys.argv[1])\n"
        f"receipt = build_live_postrun_receipt(run_root=run_root, arm_name='arm_d', decay_num={decay_num}, decay_den={decay_den}, eligible_weight_numel={ELIGIBLE_MODULE_LIMIT_NUMEL})\n"
        "gap = receipt.get('budget_gap_bpw')\n"
        "failures = []\n"
        "if gap is None:\n"
        "    failures.append('gap_not_computed')\n"
        "elif gap <= 0:\n"
        "    failures.append(f'gap_d_le_zero:{gap}')\n"
        "witness = {'pass': not failures, 'failures': failures, 'gap_bpw': gap,\n"
        "           'live_acc_carrier_bpw_max': receipt.get('live_acc_carrier_bpw_max'),\n"
        "           'operational_ok': receipt.get('operational_ok')}\n"
        "out = run_root / 'prelaunch' / 'gap_d_fail_closed_witness.json'\n"
        "out.parent.mkdir(parents=True, exist_ok=True)\n"
        "out.write_text(json.dumps(witness, indent=2) + '\\n')\n"
        "if failures:\n"
        "    raise SystemExit(1)\n"
        "PY"
    )


def _build_gap_d_gate_check_command() -> str:
    """Fix 1b: C/E gated behind D — consume D's gap artifact."""
    return (
        "PYTHONPATH=. python3 - \"{run_root}\" <<'PY'\n"
        "import json, sys\n"
        "from pathlib import Path\n"
        "run_root = Path(sys.argv[1])\n"
        "d_run_root = run_root.parent / 'FREE_DISCOVERY_H25_arm_d'\n"
        "d_witness = d_run_root / 'prelaunch' / 'gap_d_fail_closed_witness.json'\n"
        "failures = []\n"
        "if not d_witness.is_file():\n"
        "    failures.append('d_gap_witness_missing')\n"
        "else:\n"
        "    w = json.loads(d_witness.read_text())\n"
        "    if not w.get('pass'):\n"
        "        failures.append(f'd_gap_witness_failed:{w.get(\"failures\")}')\n"
        "    gap = w.get('gap_bpw')\n"
        "    if gap is not None and gap <= 0:\n"
        "        failures.append(f'd_gap_le_zero:{gap}')\n"
        "receipt = {'pass': not failures, 'failures': failures}\n"
        "out = run_root / 'prelaunch' / 'gap_d_gate_check.json'\n"
        "out.parent.mkdir(parents=True, exist_ok=True)\n"
        "out.write_text(json.dumps(receipt, indent=2) + '\\n')\n"
        "if failures:\n"
        "    raise SystemExit(1)\n"
        "PY"
    )


def build_arm_replay_commands(arm_name: str, decay_num: int, decay_den: int, *, run_id: str) -> dict[str, Any]:
    template = json.loads(B1_REPLAY_TEMPLATE.read_text(encoding="utf-8"))
    replay: dict[str, Any] = dict(template)

    arm_run_id = f"{run_id}_{arm_name}"
    replay["run_id"] = arm_run_id
    replay["run_root"] = (
        "/home/gabe/claw-code-creditdir/transient_fp_credit/"
        f"arc2b_slice5_discovery_h25_seed43_{arm_run_id}/"
    )
    _scrub_stale_b1_run_ids_in_heredocs(replay, arm_run_id)

    replay["packet_revision"] = PACKET_REVISION
    replay["binds_main_packet"] = str(DRAFT.relative_to(REPO))
    replay["schema"] = "hrm_text_158_arc2b_slice5_discovery_h25_replay_commands/v1"
    replay["gpu_hold"] = True
    replay["planning_only"] = False
    replay["note"] = f"Slice-5 discovery H=25 {arm_name}: event-coded carrier + decay {decay_num}/{decay_den} + eligible-module-limit 8"

    # Fix 5: Drop W8 env from warmup (carrier fidelity)
    for key in ("scale_smoke_command", "confirmation_launch_command", "shared_probe_argv", "calibration_warmup_command"):
        if key in replay:
            replay[key] = _replace_w8_with_event_coded(str(replay[key]))

    # Set decay flags
    for key in ("scale_smoke_command", "confirmation_launch_command", "shared_probe_argv"):
        if key in replay:
            replay[key] = _set_decay_flags(str(replay[key]), decay_num, decay_den)

    if "confirmation_launch_command" in replay:
        replay["confirmation_launch_command"] = honest_confirmation_launch_rc(
            str(replay["confirmation_launch_command"])
        )
    if arm_name == "arm_d" and "confirmation_launch_command" in replay:
        replay["confirmation_launch_command"] = _inject_profile_host_rss_env(
            str(replay["confirmation_launch_command"])
        )

    # Fix 4: Set steps to H=25 (including scale_smoke_receipt_command)
    for key in ("scale_smoke_command", "confirmation_launch_command"):
        if key in replay:
            replay[key] = _set_steps(str(replay[key]), H25_STEPS, H25_MAX_STEPS_HARD)
    # Fix scale_smoke_receipt_command --confirmation-steps 200 → 25
    if "scale_smoke_receipt_command" in replay:
        replay["scale_smoke_receipt_command"] = str(replay["scale_smoke_receipt_command"]).replace(
            "--confirmation-steps 200", f"--confirmation-steps {H25_STEPS}"
        )

    # Inject eligible-module-limit 8
    for key in ("scale_smoke_command", "confirmation_launch_command", "shared_probe_argv"):
        if key in replay:
            replay[key] = _inject_eligible_module_limit(str(replay[key]), ELIGIBLE_MODULE_LIMIT)

    # Inject max-silent-phase-seconds
    if "scale_smoke_command" in replay:
        replay["scale_smoke_command"] = _inject_max_silent_phase_direct(str(replay["scale_smoke_command"]))
    if "confirmation_launch_command" in replay:
        replay["confirmation_launch_command"] = _inject_max_silent_phase_into_bash_c_probe(str(replay["confirmation_launch_command"]))
    replay["shared_probe_argv"] = _inject_max_silent_phase_direct(str(replay.get("shared_probe_argv", "")))

    # Remove Step-2-specific commands
    replay.pop("baseline_liveness_telemetry_command", None)
    replay.pop("postrun_input_manifest_bind_command", None)
    launch_sequence = [
        step for step in list(replay.get("launch_sequence") or [])
        if step not in ("baseline_liveness_telemetry_command", "postrun_input_manifest_bind_command")
    ]

    replay["shared_gpu_env"] = "PYTHONPATH=. HRM_TEXT_158_ALLOW_C2_GPU_LAUNCH=1 HRM_TEXT_158_RUN_C2_ACQUISITION_PROBE=1"
    replay["forbidden_launch_sequence_patterns"] = [
        "confirmation_without_scale_smoke_pass",
        "reuse_run_id_2189e72001_through_17",
        "per_session_launch_without_prior_scratch_wipe",
        "event_coded_measurement_with_d_recompute_window_instrumentation",
        "event_coded_measurement_with_probe_incompatible_flags",
        "gap_d_le_zero_without_fail_closed",
        "c_or_e_before_d_gap_witness_pass",
    ]

    # Fix 3: LIVE postrun (not Arm-B K*)
    replay["postrun_command"] = (
        "bash -c 'RC=0; PYTHONPATH=. timeout 1800 python3 "
        "scripts/hrm_text_158_arc2b_slice5_discovery_live_postrun.py "
        f"--run-root {{run_root}} --arm-name {arm_name} "
        f"--decay-num {decay_num} --decay-den {decay_den} "
        f"--out {{run_root}}/discovery_{arm_name}_live_postrun_receipt.json "
        "|| RC=$?; exit $RC'"
    )

    # Fix 1: gap(D) witness (real computation for arm_d; gate check for C/E)
    if arm_name == "arm_d":
        replay["gap_d_fail_closed_witness_command"] = _build_gap_d_witness_command(decay_num, decay_den)
        gap_witness_key = "gap_d_fail_closed_witness_command"
    else:
        replay["gap_d_gate_check_command"] = _build_gap_d_gate_check_command()
        gap_witness_key = "gap_d_gate_check_command"

    # Fix 2: forbidden-token witness bound to per-arm artifact
    arm_replay_rel = f"artifacts/consensus_prep/arc2b_slice5_discovery_h25_launch_packet_v2_{arm_name}_replay_commands.json"
    replay["forbidden_carrier_tokens_absent_witness_command"] = (
        f"PYTHONPATH=. python3 -c \""
        f"import json; from pathlib import Path; "
        f"forbidden = {list(FORBIDDEN_CARRIER_TOKENS)!r}; "
        f"incompatible = {list(EVENT_CODED_INCOMPATIBLE_FLAGS)!r}; "
        f"cmds = json.loads(Path('{arm_replay_rel}').read_text()); "
        f"checked = {{k: cmds.get(k) for k in ('scale_smoke_command', 'confirmation_launch_command', 'shared_probe_argv')}}; "
        f"blob = json.dumps(checked); "
        f"hits = [t for t in forbidden if t in blob]; "
        f"assert not hits, f'forbidden_tokens_present:{{hits}}'; "
        f"incompatible_hits = [t for t in incompatible if t in blob]; "
        f"assert not incompatible_hits, f'incompatible_flags_present:{{incompatible_hits}}'; "
        f"print(json.dumps({{'pass': True}}))\""
    )

    # Classifier sha pin witness
    replay["classifier_sha_pin_witness_command"] = (
        f"PYTHONPATH=. python3 -c \"import hashlib; from pathlib import Path; "
        f"p=Path('{CLASSIFIER_MODULE.relative_to(REPO)}'); "
        f"expected='{sha256_file(CLASSIFIER_MODULE)}'; actual=hashlib.sha256(p.read_bytes()).hexdigest(); "
        f"assert actual==expected, f'classifier_sha_mismatch {{actual}}!={{expected}}'\""
    )

    # Add witnesses to launch_sequence before postrun
    witnesses = (
        gap_witness_key,
        "classifier_sha_pin_witness_command",
        "forbidden_carrier_tokens_absent_witness_command",
    )
    for witness in witnesses:
        if witness not in launch_sequence:
            idx = launch_sequence.index("postrun_command")
            launch_sequence.insert(idx, witness)
    replay["launch_sequence"] = launch_sequence

    return replay


def build_packet(classifier_sha: str) -> dict[str, Any]:
    return {
        "schema": "hrm_text_158_arc2b_slice5_discovery_h25_launch_packet/v2",
        "task_id": ACTIVE_TASK_ID,
        "fold": "arc2b_slice5_discovery_h25_launch",
        "packet_revision": PACKET_REVISION,
        "packet_status": "LAUNCH_READY_GPU_HOLD_UNTIL_PLUS1_LAUNCH",
        "git_head_required": HEAD,
        "harness_commit_sha": HARNESS_COMMIT_SHA,
        "canonical_lane": CANONICAL_LANE,
        "provenance": {
            "dispatch_msg_id": DISPATCH_MSG_ID,
            "plan_msg_id": PLAN_MSG_ID,
            "gate1_freeze_msg_id": GATE1_FREEZE_MSG_ID,
            "co_lead_gate2_msg_id": CO_LEAD_GATE2_MSG_ID,
            "implement_gate_msg_id": IMPLEMENT_GATE_MSG_ID,
            "harness_commit_sha": HARNESS_COMMIT_SHA,
        },
        "parent_checkpoint": {
            "path": PARENT_REL,
            "sha256": PARENT_SHA256,
            "curriculum_seed": 43,
            "support_order_seed": 43,
            "eligible_scope": "all-bitlinear",
            "eligible_module_limit": ELIGIBLE_MODULE_LIMIT,
        },
        "fidelity_contract": {
            "same_tier_b_checkpoint": True,
            "same_trainer": True,
            "same_q_acc_candidate_gen": True,
            "same_carrier_accounting": True,
            "shrink_only": ["horizon", "eligible_scope", "instrumentation"],
            "not_smaller_model": True,
        },
        "classifier_binding": {
            "classifier": CLASSIFIER,
            "module_path": str(CLASSIFIER_MODULE.relative_to(REPO)),
            "module_sha256": classifier_sha,
            "receipt_schema": RECEIPT_SCHEMA,
            "evidence_source": "live_decay_curve",
            "mechanism_terminals": [
                "REPRESENTATION_NEW_MECHANISM",
                "MORE_FORGETTING_HELPS",
                "LESS_FORGETTING_HELPS",
                "BOTH_IMPROVE",
                "DECAY_DIRECTION_AMBIGUOUS",
            ],
        },
        "selector": {
            "budget_gap_bpw": "live_acc_carrier_bpw_max - effective_acc_budget_bpw",
            "effective_acc_budget_bpw": DEFAULT_EFFECTIVE_ACC_BUDGET_BPW,
            "materiality_factor": DEFAULT_MATERIALITY_FACTOR,
            "direction_tol_factor": DEFAULT_DIRECTION_TOL_FACTOR,
        },
        "arms": {
            arm_name: {
                "decay_num": dn,
                "decay_den": dd,
                "steps": H25_STEPS,
                "max_steps_hard": H25_MAX_STEPS_HARD,
                "eligible_module_limit": ELIGIBLE_MODULE_LIMIT,
                "carrier_required_flags": list(CARRIER_REQUIRED_FLAGS),
                "forbidden_carrier_tokens": list(FORBIDDEN_CARRIER_TOKENS),
                "event_coded_incompatible_flags": list(EVENT_CODED_INCOMPATIBLE_FLAGS),
                "max_silent_phase_seconds": MAX_SILENT_PHASE_SECONDS,
                "step_time_bound_seconds": STEP_TIME_BOUND_SECONDS,
                "resume_generation_required": 0,
                "from_clean_parent_contiguous": True,
                "live_carrier_bytes_exact_required": True,
                "replay_commands_artifact": f"artifacts/consensus_prep/arc2b_slice5_discovery_h25_launch_packet_v2_{arm_name}_replay_commands.json",
                "postrun_script": "scripts/hrm_text_158_arc2b_slice5_discovery_live_postrun.py",
            }
            for arm_name, (dn, dd) in ARM_DECAY.items()
        },
        "ordering": list(ARM_ORDERING),
        "liveness_contract": {
            "step_time_bound_seconds": STEP_TIME_BOUND_SECONDS,
            "max_silent_phase_seconds": MAX_SILENT_PHASE_SECONDS,
            "liveness_fail_action": "kill + classify operational/inconclusive, no terminal",
            "phase_budget_exceed_action": "stack-sample + classify operational",
        },
        "gap_d_fail_closed": {
            "precondition": "if control gap(D) <= 0 at read => classifier fails closed / rescope (no 2x2 terminal)",
            "witness_command_arm_d": "gap_d_fail_closed_witness_command",
            "gate_check_arm_c_e": "gap_d_gate_check_command",
            "computes_real_bpw": True,
        },
        "operational_guard": {
            "any_live_arm_ineligible_or_liveness_fails": "no_2x2_terminal",
            "classify_operational_or_inconclusive": True,
        },
        "lane_field_fail_closed": {
            "required_fields": ["lane_indices", "acc_before_lanes", "acc_after_lanes", "vote_lanes"],
            "arm_b_offline_fails_closed_on_missing": True,
        },
        "readiness_classification": {
            "class": "pre_full_stack_diagnostic",
            "flags": {"ready_for_main_science": False, "counts_as_sub2": False, "pre_full_stack_diagnostic": True},
        },
        "explicit_non_claims": [
            "not_sub2_readiness", "not_reduction_eligibility", "not_bank_pin",
            "not_fold3b_universalization", "not_h200_d_terminal_verdict",
            "not_asymptotic_k_star_at_h100_h200", "not_backlog_direction_flip_erosion",
            "not_from_clean_parent_contiguous_proof",
            "not_terminal_verdict_before_registered_complete_arms",
            "not_gpu_launch_until_plus1_launch_gate", "not_h50_extension_in_this_packet",
        ],
        "gpu_hold": True,
        "separate_launch_gate_required": True,
    }


def verify_replay_commands(replay: dict[str, Any], arm_name: str) -> list[str]:
    failures: list[str] = []
    checked_keys = ("scale_smoke_command", "confirmation_launch_command", "shared_probe_argv")
    blob = json.dumps({key: replay.get(key) for key in checked_keys})
    for token in FORBIDDEN_CARRIER_TOKENS:
        if token in blob:
            failures.append(f"forbidden_token_present:{token}")
    for flag in CARRIER_REQUIRED_FLAGS:
        if flag not in blob:
            failures.append(f"missing_required_carrier_flag:{flag}")
    if EVENT_CODED_RECOMPUTE_WINDOW_LOG_FLAG not in blob:
        failures.append("missing_event_coded_recompute_window_log_flag")
    if "--eligible-module-limit" not in blob:
        failures.append(f"{arm_name}_missing_eligible_module_limit")
    if f"--eligible-module-limit {ELIGIBLE_MODULE_LIMIT}" not in blob:
        failures.append(f"{arm_name}_eligible_module_limit_not_8")
    confirmation = str(replay.get("confirmation_launch_command") or "")
    if f"--steps {H25_STEPS}" not in confirmation:
        failures.append(f"{arm_name}_steps_not_h25")
    if MAX_SILENT_PHASE_FLAG not in confirmation:
        failures.append(f"{arm_name}_missing_max_silent_phase")
    # Fix 3: postrun must be live postrun, not arm_b_offline
    postrun = str(replay.get("postrun_command") or "")
    if "discovery_live_postrun" not in postrun:
        failures.append(f"{arm_name}_postrun_not_live")
    if "arm_b_offline" in postrun:
        failures.append(f"{arm_name}_postrun_is_arm_b_offline")
    # Fix 4: scale_smoke_receipt_command must have --confirmation-steps 25
    sm_receipt = str(replay.get("scale_smoke_receipt_command") or "")
    if "--confirmation-steps 200" in sm_receipt:
        failures.append(f"{arm_name}_scale_smoke_receipt_still_200")
    # Fix 5: no W8 env in warmup
    warmup = str(replay.get("calibration_warmup_command") or "")
    if "NARROW_CARRIER_W8" in warmup:
        failures.append(f"{arm_name}_warmup_has_w8_env")
    # Fix 1: gap witness
    if arm_name == "arm_d":
        gap_cmd = str(replay.get("gap_d_fail_closed_witness_command") or "")
        if "build_live_postrun_receipt" not in gap_cmd:
            failures.append(f"{arm_name}_gap_witness_not_real_computation")
        if "budget_gap_bpw" not in gap_cmd:
            failures.append(f"{arm_name}_gap_witness_missing_bpw")
    else:
        gate_cmd = str(replay.get("gap_d_gate_check_command") or "")
        if "gap_d_fail_closed_witness" not in gate_cmd:
            failures.append(f"{arm_name}_gate_check_missing_d_witness")
    # Fix 2: forbidden-token witness bound to per-arm artifact
    forbidden_cmd = str(replay.get("forbidden_carrier_tokens_absent_witness_command") or "")
    if f"v2_{arm_name}_replay_commands" not in forbidden_cmd:
        failures.append(f"{arm_name}_forbidden_witness_not_per_arm")
    return failures


def verify_packet(packet: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if packet.get("git_head_required") != HEAD:
        failures.append("git_head_required_stale")
    if not packet.get("gpu_hold"):
        failures.append("gpu_hold_not_set")
    arms = packet.get("arms") or {}
    for arm_key in ("arm_c", "arm_d", "arm_e"):
        arm = arms.get(arm_key) or {}
        if arm.get("steps") != H25_STEPS:
            failures.append(f"{arm_key}_steps_not_h25")
        if arm.get("eligible_module_limit") != ELIGIBLE_MODULE_LIMIT:
            failures.append(f"{arm_key}_eligible_module_limit_not_8")
        if "discovery_live_postrun" not in str(arm.get("postrun_script") or ""):
            failures.append(f"{arm_key}_postrun_script_not_live")
    if packet.get("ordering") != list(ARM_ORDERING):
        failures.append("ordering_drift")
    if not packet.get("gap_d_fail_closed", {}).get("computes_real_bpw"):
        failures.append("gap_d_fail_closed_not_real_bpw")
    return failures


def self_verify() -> dict[str, Any]:
    failures: list[str] = []
    classifier_sha = sha256_file(CLASSIFIER_MODULE)

    for arm_name, (decay_num, decay_den) in ARM_DECAY.items():
        arm_replay_path = REPLAY.parent / f"arc2b_slice5_discovery_h25_launch_packet_v2_{arm_name}_replay_commands.json"
        if not arm_replay_path.is_file():
            failures.append(f"{arm_name}_replay_missing")
            continue
        arm_replay = json.loads(arm_replay_path.read_text(encoding="utf-8"))
        failures.extend(verify_replay_commands(arm_replay, arm_name))
        confirmation = str(arm_replay.get("confirmation_launch_command") or "")
        if f"--vote-update-decay-numerator {decay_num}" not in confirmation:
            failures.append(f"{arm_name}_decay_num_wrong")
        if f"--vote-update-decay-denominator {decay_den}" not in confirmation:
            failures.append(f"{arm_name}_decay_den_wrong")

    if not DRAFT.is_file():
        failures.append("draft_missing")
    else:
        packet = json.loads(DRAFT.read_text(encoding="utf-8"))
        if packet.get("classifier_binding", {}).get("module_sha256") != classifier_sha:
            failures.append("classifier_sha_pin_drift")
        failures.extend(verify_packet(packet))

    live_git_head = git_head()
    pins_match_commit = live_git_head == HEAD
    if not pins_match_commit:
        failures.append("pins_match_commit")

    return {
        "ok": not failures and pins_match_commit,
        "failures": failures,
        "classifier_module_sha256": classifier_sha,
        "git_head": live_git_head,
        "pins_match_commit": pins_match_commit,
    }


def main() -> int:
    classifier_sha = sha256_file(CLASSIFIER_MODULE)

    for arm_name, (decay_num, decay_den) in ARM_DECAY.items():
        arm_replay = build_arm_replay_commands(arm_name, decay_num, decay_den, run_id="FREE_DISCOVERY_H25")
        arm_replay_path = REPLAY.parent / f"arc2b_slice5_discovery_h25_launch_packet_v2_{arm_name}_replay_commands.json"
        arm_replay_bytes = (json.dumps(arm_replay, indent=2, sort_keys=True) + "\n").encode()
        arm_replay_path.parent.mkdir(parents=True, exist_ok=True)
        arm_replay_path.write_bytes(arm_replay_bytes)

    packet = build_packet(classifier_sha)
    DRAFT.parent.mkdir(parents=True, exist_ok=True)
    DRAFT.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    result = self_verify()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
