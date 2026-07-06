#!/usr/bin/env python3
"""Apply Arc #2b Slice-5 Step-2 in-vivo GPU launch packet (CPU packet/postrun only; no GPU)."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
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

REPO = Path("/mnt/c/Users/gabes/projects/claw-code-hrm-text-158")
HEAD = "fa50e59afa834df8f0f59b206f70f1e03b913e91"
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
FORBIDDEN_CARRIER_TOKENS = (
    "--dense-accumulator-w8-clip",
    "--dense-accumulator-w7-clip",
    "HRM_TEXT_158_RUN_NARROW_CARRIER_W8_TRAINER_INTEGRATION",
)


def git_head() -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


def _replace_w8_with_event_coded(command: str) -> str:
    out = command
    out = out.replace(
        "HRM_TEXT_158_RUN_NARROW_CARRIER_W8_TRAINER_INTEGRATION=1 ",
        "",
    )
    for token in ("--dense-accumulator-w8-clip", "--dense-accumulator-w7-clip"):
        out = out.replace(f" {token}", "")
    insert = " ".join(STEP2_CARRIER_REQUIRED + STEP2_DECAY_FLAGS)
    marker = "--phase d-recompute-window-feasibility"
    if marker in out:
        out = out.replace(marker, f"{marker} {insert}", 1)
    return out


def build_replay_commands(classifier_sha: str) -> dict[str, Any]:
    template = json.loads(B1_REPLAY_TEMPLATE.read_text(encoding="utf-8"))
    replay: dict[str, Any] = dict(template)
    replay["packet_revision"] = PACKET_REVISION
    replay["binds_main_packet"] = str(DRAFT.relative_to(REPO))
    replay["run_id"] = "FREE_SLICE5_STEP2"
    replay["run_root"] = (
        "/home/gabe/claw-code-creditdir/transient_fp_credit/"
        "arc2b_slice5_step2_in_vivo_seed43_{run_id}/"
    )
    replay["schema"] = "hrm_text_158_arc2b_slice5_step2_in_vivo_launch_packet_replay_commands/v1"
    replay["gpu_hold"] = True
    replay["planning_only"] = False
    replay["note"] = (
        "Slice-5 Step-2 in-vivo: event-coded live carrier + decay 1/2 + "
        "Slice-5 postrun classifier"
    )

    for key in (
        "scale_smoke_command",
        "confirmation_launch_command",
        "baseline_liveness_telemetry_command",
        "calibration_warmup_command",
    ):
        if key in replay:
            replay[key] = _replace_w8_with_event_coded(str(replay[key]))

    replay.pop("baseline_liveness_telemetry_command", None)
    launch_sequence = [
        step
        for step in list(replay.get("launch_sequence") or [])
        if step != "baseline_liveness_telemetry_command"
    ]

    replay["shared_probe_argv"] = _replace_w8_with_event_coded(
        str(replay.get("shared_probe_argv", ""))
    )
    replay["shared_gpu_env"] = (
        "PYTHONPATH=. HRM_TEXT_158_ALLOW_C2_GPU_LAUNCH=1 HRM_TEXT_158_RUN_C2_ACQUISITION_PROBE=1"
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
    replay["postrun_input_manifest_bind_command"] = (
        "PYTHONPATH=. python3 scripts/hrm_text_158_d_recompute_input_manifest_bind.py "
        f"--run-root {{run_root}} --packet {DRAFT.relative_to(REPO)} "
        "--out {run_root}/prelaunch/postrun_input_manifest.json"
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
        "cmds = json.loads(Path('artifacts/consensus_prep/"
        "arc2b_slice5_step2_in_vivo_gpu_launch_packet_v1_replay_commands.json').read_text())\n"
        "checked = {k: cmds.get(k) for k in ('scale_smoke_command', 'confirmation_launch_command', 'shared_probe_argv')}\n"
        "blob = json.dumps(checked)\n"
        "hits = [t for t in forbidden if t in blob]\n"
        "if hits: raise SystemExit(f'forbidden_tokens_present:{hits}')\n"
        "print(json.dumps({'pass': True, 'forbidden_checked': forbidden}))\nPY"
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
            "confirmation_steps": 200,
            "from_clean_parent_contiguous": True,
            "live_carrier_snapshot_required": True,
            "live_carrier_bytes_exact_required": True,
            "carrier_authority_required_argv": list(STEP2_CARRIER_REQUIRED),
            "carrier_authority_forbidden_tokens": list(FORBIDDEN_CARRIER_TOKENS),
            "decay_cli_flags": list(STEP2_DECAY_FLAGS),
        },
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
    confirmation = str(replay.get("confirmation_launch_command") or "")
    if "--vote-update-decay-numerator" not in confirmation:
        failures.append("confirmation_missing_decay_numerator_flag")
    if "arc2b_slice5_step2_in_vivo_postrun_classifier" not in str(
        replay.get("postrun_command") or ""
    ):
        failures.append("postrun_not_slice5_classifier")
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
    failures.extend(verify_replay_commands(replay))

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

    return {
        "ok": not failures and draft_sha == regen_draft_sha and replay_sha == regen_replay_sha,
        "failures": failures,
        "deterministic_regen": draft_sha == regen_draft_sha and replay_sha == regen_replay_sha,
        "draft_sha256": draft_sha,
        "replay_sha256": replay_sha,
        "classifier_module_sha256": classifier_sha,
        "git_head": git_head(),
        "pins_match_commit": True,
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
