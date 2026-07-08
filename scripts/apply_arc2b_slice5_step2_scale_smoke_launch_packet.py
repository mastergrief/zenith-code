#!/usr/bin/env python3
"""Apply Arc #2b Slice-5 Step-2 scale_smoke-only GPU launch packet (CPU packet; no GPU)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from calm.hrm_text_158.native_full_stack.arc2b_slice5_in_vivo_branch import (
    B1_RECORDED_SELECTOR_INTERNAL_MANIFEST_SHA256,
    CLASSIFIER,
    DEFAULT_EFFECTIVE_ACC_BUDGET_BPW,
    PREREG_LAW_DECAY_DEN,
    PREREG_LAW_DECAY_NUM,
    PREREG_LAW_WINDOW_K,
    sha256_file,
)
from scripts.apply_arc2b_slice5_step2_in_vivo_gpu_launch_packet import (
    B1_REPLAY_TEMPLATE,
    EVENT_CODED_INCOMPATIBLE_FLAGS,
    FORBIDDEN_CARRIER_TOKENS,
    MAX_SILENT_PHASE_FLAG,
    PARENT_REL,
    PARENT_SHA256,
    STEP2_CARRIER_REQUIRED,
    STEP2_CONFIRMATION_STEPS,
    STEP2_DECAY_FLAGS,
    STEP2_EVENT_CODED_RECOMPUTE_WINDOW_LOG_FLAG,
    STEP2_MAX_SILENT_PHASE_SECONDS,
    _append_producer_max_silent_phase,
    _build_calibration_warmup_retry_command,
    _inject_max_silent_phase,
    _replace_w8_with_event_coded,
    _scrub_stale_b1_run_ids_in_heredocs,
    build_liveness_contract,
    build_liveness_retry_policy,
    git_head,
    verify_no_stale_b1_run_id_in_heredocs,
    verify_no_unsatisfiable_input_manifest_bind,
    verify_warmup_retry_metadata,
)

REPO = Path("/mnt/c/Users/gabes/projects/claw-code-hrm-text-158")
HEAD = "3d52a96abd8ecab00b902f9d6a837c2a30b80894"
SMOKE_RUN_ID = "FREE_SLICE5_STEP2_SCALE_SMOKE"
SMOKE_STEPS = 5
ACTIVE_TASK_ID = "1783272482268-052281aa"
DISPATCH_MSG_ID = "1783498076466-345c8cec"
PLAN_MSG_ID = "1783499011075-226e43bf"
GATE1_FREEZE_MSG_ID = "1783499093445-da036c4b"
CO_LEAD_GATE2_MSG_ID = "1783499259427"
IMPLEMENT_GATE_MSG_ID = "1783499312705-6ea128fc"
PACKET_REVISION = "v1_arc2b_slice5_step2_scale_smoke"
CANONICAL_LANE = "test-operator_gpu_arc2b_slice5_step2_scale_smoke"

DRAFT = (
    REPO
    / "artifacts/consensus_prep/arc2b_slice5_step2_scale_smoke_launch_packet_v1_draft.json"
)
REPLAY = (
    REPO
    / "artifacts/consensus_prep/arc2b_slice5_step2_scale_smoke_launch_packet_v1_replay_commands.json"
)

CLASSIFIER_MODULE = REPO / "calm/hrm_text_158/native_full_stack/arc2b_slice5_in_vivo_branch.py"
WITNESS_SCRIPT = REPO / "scripts/hrm_text_158_arc2b_slice5_step2_scale_smoke_operational_witness.py"
SCALE_SMOKE_FORBIDDEN_PATTERNS = (
    "confirmation_launch_without_scale_smoke_launch_eligibility_pass",
    "reuse_run_id_2189e72001_through_17",
    "per_session_launch_without_prior_scratch_wipe",
    "h200_confirmation_bundled_into_scale_smoke_packet",
    "event_coded_measurement_with_d_recompute_window_instrumentation",
    "event_coded_measurement_with_probe_incompatible_flags",
    "mechanism_terminal_wording_in_scale_smoke_operational_receipt",
)


def build_scale_smoke_operational_witness_command() -> str:
    return (
        "PYTHONPATH=. python3 "
        "scripts/hrm_text_158_arc2b_slice5_step2_scale_smoke_operational_witness.py "
        "--run-root {run_root} "
        f"--smoke-steps {SMOKE_STEPS} "
        "--json-out {run_root}/prelaunch/scale_smoke_operational_witness.json"
    )


def build_scale_smoke_launch_eligibility_command() -> str:
    return (
        "PYTHONPATH=. python3 - \"{run_root}\" <<'PY'\n"
        "import json, sys\nfrom pathlib import Path\n"
        "run_root = Path(sys.argv[1])\n"
        "failures = []\n"
        "witness = run_root / 'prelaunch' / 'scale_smoke_operational_witness.json'\n"
        "receipt = run_root / 'prelaunch' / 'scale_smoke_receipt.json'\n"
        "for label, path in (('operational_witness', witness), ('scale_smoke_receipt', receipt)):\n"
        "    if not path.is_file():\n"
        "        failures.append(f'missing_{label}')\n"
        "        continue\n"
        "    body = json.loads(path.read_text(encoding='utf-8'))\n"
        "    if body.get('pass') is not True:\n"
        "        failures.append(f'{label}_not_pass')\n"
        "out = run_root / 'prelaunch' / 'scale_smoke_launch_eligibility.json'\n"
        "out.parent.mkdir(parents=True, exist_ok=True)\n"
        "payload = {'pass': not failures, 'failures': failures}\n"
        "out.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\\n', encoding='utf-8')\n"
        "if failures:\n    raise SystemExit(1)\nPY"
    )


def build_replay_commands(classifier_sha: str) -> dict[str, Any]:
    template = json.loads(B1_REPLAY_TEMPLATE.read_text(encoding="utf-8"))
    replay: dict[str, Any] = dict(template)
    replay["packet_revision"] = PACKET_REVISION
    replay["binds_main_packet"] = str(DRAFT.relative_to(REPO))
    replay["run_id"] = SMOKE_RUN_ID
    replay["run_root"] = (
        "/home/gabe/claw-code-creditdir/transient_fp_credit/"
        "arc2b_slice5_step2_scale_smoke_seed43_{run_id}/"
    )
    _scrub_stale_b1_run_ids_in_heredocs(replay, SMOKE_RUN_ID)
    replay["schema"] = "hrm_text_158_arc2b_slice5_step2_scale_smoke_launch_packet_replay_commands/v1"
    replay["gpu_hold"] = True
    replay["planning_only"] = False
    replay["note"] = (
        "Slice-5 Step-2 scale_smoke only: event-coded live carrier + decay 1/2 + "
        "operational witness + launch eligibility (no H=200 confirmation)"
    )

    if "scale_smoke_command" in replay:
        replay["scale_smoke_command"] = _inject_max_silent_phase(
            _replace_w8_with_event_coded(str(replay["scale_smoke_command"]))
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
    replay.pop("confirmation_launch_command", None)
    replay.pop("postrun_command", None)
    replay.pop("post_confirmation_hygiene_assert_command", None)
    replay.pop("decay_replay_constants_witness_command", None)

    replay["scale_smoke_operational_witness_command"] = (
        build_scale_smoke_operational_witness_command()
    )
    replay["scale_smoke_launch_eligibility_command"] = (
        build_scale_smoke_launch_eligibility_command()
    )
    replay["scale_smoke_receipt_command"] = (
        "PYTHONPATH=. python3 scripts/hrm_text_158_d_recompute_scale_smoke_receipt.py "
        "--run-root {run_root} "
        "--json-out {run_root}/prelaunch/scale_smoke_receipt.json "
        f"--smoke-steps {SMOKE_STEPS} "
        f"--confirmation-steps {STEP2_CONFIRMATION_STEPS} "
        "--min-free-memory-bytes 1610612736 "
        "--extrapolated-receipt-bytes-max 104857600 "
        "--extrapolated-recompute-log-bytes-max 67108864"
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
        f"cmds = json.loads(Path('{REPLAY.relative_to(REPO)}').read_text())\n"
        "checked = {k: cmds.get(k) for k in ('scale_smoke_command', 'shared_probe_argv')}\n"
        "blob = json.dumps(checked)\n"
        "hits = [t for t in forbidden if t in blob]\n"
        "if hits: raise SystemExit(f'forbidden_tokens_present:{hits}')\n"
        "incompatible_hits = [t for t in incompatible if t in blob]\n"
        "if incompatible_hits: raise SystemExit(f'incompatible_flags_present:{incompatible_hits}')\n"
        "print(json.dumps({'pass': True}))\nPY"
    )
    replay["shared_gpu_env"] = (
        "PYTHONPATH=. HRM_TEXT_158_ALLOW_C2_GPU_LAUNCH=1 HRM_TEXT_158_RUN_C2_ACQUISITION_PROBE=1"
    )
    replay["forbidden_launch_sequence_patterns"] = list(SCALE_SMOKE_FORBIDDEN_PATTERNS)
    replay["launch_sequence"] = [
        "run_root_free_assert_command",
        "parent_checkpoint_rehash_command",
        "sub2_first_gate_command",
        "cheap_observer_no_rebuild_preflight_command",
        "scratch_wipe_commands.calibration_warmup",
        "calibration_warmup_command",
        "parent_checkpoint_rehash_after_calibration_warmup_command",
        "calibration_prepass_command",
        "scratch_wipe_commands.scale_smoke",
        "scale_smoke_command",
        "scale_smoke_operational_witness_command",
        "scale_smoke_receipt_command",
        "scale_smoke_launch_eligibility_command",
        "parent_checkpoint_rehash_after_scale_smoke_command",
        "classifier_sha_pin_witness_command",
        "forbidden_carrier_tokens_absent_witness_command",
    ]
    return replay


def build_packet(classifier_sha: str, replay_sha: str) -> dict[str, Any]:
    return {
        "schema": "hrm_text_158_arc2b_slice5_step2_scale_smoke_launch_packet/v1",
        "task_id": ACTIVE_TASK_ID,
        "fold": "arc2b_slice5_step2_scale_smoke_gpu_launch",
        "packet_revision": PACKET_REVISION,
        "git_head_required": HEAD,
        "canonical_lane": CANONICAL_LANE,
        "provenance": {
            "dispatch_msg_id": DISPATCH_MSG_ID,
            "plan_msg_id": PLAN_MSG_ID,
            "gate1_freeze_msg_id": GATE1_FREEZE_MSG_ID,
            "co_lead_gate2_msg_id": CO_LEAD_GATE2_MSG_ID,
            "implement_gate_msg_id": IMPLEMENT_GATE_MSG_ID,
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
        "scale_smoke": {
            "smoke_steps": SMOKE_STEPS,
            "max_steps_hard": SMOKE_STEPS,
            "fresh_process_required": True,
            "scratch_wipe_before_smoke": True,
            "resume_generation": 0,
        },
        "runtime_requirements": {
            "decay_active": {"decay_num": 1, "decay_den": 2},
            "resume_generation": 0,
            "from_clean_parent_contiguous": True,
            "live_carrier_snapshot_required": True,
            "carrier_authority_required_argv": list(STEP2_CARRIER_REQUIRED),
            "carrier_authority_forbidden_tokens": list(FORBIDDEN_CARRIER_TOKENS),
            "decay_cli_flags": list(STEP2_DECAY_FLAGS),
            "max_silent_phase_seconds": STEP2_MAX_SILENT_PHASE_SECONDS,
        },
        "liveness_contract": build_liveness_contract(),
        "liveness_retry_policy": build_liveness_retry_policy(),
        "operational_pass_criteria": {
            "requires_witness_pass": True,
            "requires_scale_smoke_receipt_pass": True,
            "requires_launch_eligibility_pass": True,
            "forbidden_terminal_wording": [
                "D_NEEDS_UPDATE_LAW_REDESIGN",
                "sub-2",
                "H=200",
            ],
        },
        "classifier_binding": {
            "classifier": CLASSIFIER,
            "module_path": str(CLASSIFIER_MODULE.relative_to(REPO)),
            "module_sha256": classifier_sha,
            "witness_only": True,
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
        "run_id_policy": "FREE — fresh scale_smoke run_id; do NOT reuse B1 2189e72001..17",
        "run_root_template": (
            "/home/gabe/claw-code-creditdir/transient_fp_credit/"
            "arc2b_slice5_step2_scale_smoke_seed43_{run_id}/"
        ),
        "gpu_hold": True,
        "explicit_non_claims": [
            "not_h200_confirmation",
            "not_mechanism_terminal",
            "not_sub2_readiness",
            "not_reduction_eligibility",
            "not_gpu_launch_until_plus1_launch_gate",
        ],
    }


def verify_launch_sequence_order(replay: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    sequence = list(replay.get("launch_sequence") or [])
    required = (
        "scale_smoke_command",
        "scale_smoke_operational_witness_command",
        "scale_smoke_receipt_command",
        "scale_smoke_launch_eligibility_command",
    )
    for key in required:
        if key not in sequence:
            failures.append(f"missing_launch_sequence_step:{key}")
    if all(key in sequence for key in required):
        smoke = sequence.index("scale_smoke_command")
        witness = sequence.index("scale_smoke_operational_witness_command")
        receipt = sequence.index("scale_smoke_receipt_command")
        eligibility = sequence.index("scale_smoke_launch_eligibility_command")
        if not (smoke < witness < receipt < eligibility):
            failures.append("launch_sequence_order_invalid")
    forbidden = (
        "confirmation_launch_command",
        "postrun_command",
        "post_confirmation_hygiene_assert_command",
    )
    for key in forbidden:
        if key in sequence:
            failures.append(f"forbidden_launch_sequence_step:{key}")
    return failures


def verify_explicit_max_silent_phase_seconds_scale_smoke(replay: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    checked_keys = (
        "calibration_warmup_command",
        "scale_smoke_command",
        "shared_probe_argv",
    )
    expected = f"{MAX_SILENT_PHASE_FLAG} {STEP2_MAX_SILENT_PHASE_SECONDS}"
    for key in checked_keys:
        body = str(replay.get(key) or "")
        if expected not in body:
            failures.append(f"missing_explicit_max_silent_phase:{key}")
    return failures


def verify_event_coded_incompatible_flags_absent_scale_smoke(replay: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    for key in ("scale_smoke_command", "shared_probe_argv"):
        body = str(replay.get(key) or "")
        if "--event-coded-sparse-vote-authority" not in body:
            continue
        for flag in EVENT_CODED_INCOMPATIBLE_FLAGS:
            if flag in body:
                failures.append(f"incompatible_flag_present:{key}:{flag}")
    return failures


def verify_event_coded_recompute_window_log_flag_scale_smoke(replay: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    for key in ("scale_smoke_command", "shared_probe_argv"):
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


def verify_replay_commands(replay: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    checked_keys = ("scale_smoke_command", "shared_probe_argv")
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
    smoke = str(replay.get("scale_smoke_command") or "")
    if f"--steps {SMOKE_STEPS}" not in smoke:
        failures.append("scale_smoke_missing_steps_5")
    if f"--max-steps-hard {SMOKE_STEPS}" not in smoke:
        failures.append("scale_smoke_missing_max_steps_hard_5")
    if "confirmation_launch_command" in replay:
        failures.append("confirmation_launch_command_present")
    if "postrun_command" in replay:
        failures.append("postrun_command_present")
    failures.extend(verify_no_stale_b1_run_id_in_heredocs(replay))
    failures.extend(verify_no_unsatisfiable_input_manifest_bind(replay))
    failures.extend(verify_explicit_max_silent_phase_seconds_scale_smoke(replay))
    failures.extend(verify_event_coded_incompatible_flags_absent_scale_smoke(replay))
    failures.extend(verify_event_coded_recompute_window_log_flag_scale_smoke(replay))
    failures.extend(verify_launch_sequence_order(replay))
    return failures


def verify_packet(packet: dict[str, Any], replay: dict[str, Any]) -> list[str]:
    failures = verify_replay_commands(replay)
    failures.extend(verify_warmup_retry_metadata(replay, packet))
    if packet.get("git_head_required") != HEAD:
        failures.append("git_head_required_stale")
    if int((packet.get("scale_smoke") or {}).get("smoke_steps") or 0) != SMOKE_STEPS:
        failures.append("packet_smoke_steps_not_5")
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
