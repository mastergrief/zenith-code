#!/usr/bin/env python3
"""Arc #2b Slice-5 Step-2 in-vivo law-validation postrun classifier."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

from calm.hrm_text_158.native_full_stack.arc2b_slice5_in_vivo_branch import (
    ANTI_OVERCLAIM_VERBATIM,
    B1_RECORDED_MANIFEST_FILE_SHA256,
    B1_RECORDED_SELECTOR_INTERNAL_MANIFEST_SHA256,
    CLASSIFIER,
    DEFAULT_EFFECTIVE_ACC_BUDGET_BPW,
    DEFAULT_TOLERANCE_BPW,
    NUMEL_BASIS_SOURCE,
    PREREG_LAW_DECAY_DEN,
    PREREG_LAW_DECAY_NUM,
    PREREG_LAW_WINDOW_K,
    RECEIPT_SCHEMA,
    STEP2_ONLY_TERMINALS,
    build_branch_input_from_step2_gpu_run,
    classify_arc2b_slice5_in_vivo_branch,
    sha256_file,
)

ACTIVE_TASK_ID = "1783272482268-052281aa"
CLASSIFIER_MODULE = Path(
    "calm/hrm_text_158/native_full_stack/arc2b_slice5_in_vivo_branch.py"
)
PARENT_CHECKPOINT = Path(
    "calm/hrm/checkpoints/hrm_text_158_phase3_L0c1_seed0017_replay83_n12k_lr7p5e5_pc1p0_rsL0b1math1r1b2_1_anchorsv1r3_from_L0b_final_step01500.pt"
)
PARENT_SHA256 = "9b4e311a22787e7d4808bde7bc2953568d767a2ee8ac648942a3f5dbb7b4d5ec"
STEP2_CONFIRMATION_STEPS = 200


def git_head(repo_root: Path) -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _phase_is_liveness_failure(phase_path: Path) -> bool:
    if not phase_path.is_file():
        return False
    phase = _load_json(phase_path)
    return str(phase.get("failure_class")) == "LIVENESS_FAILURE"


def resolve_step2_operational_ok_from_run_artifacts(
    run_root: Path,
    *,
    confirmation_steps: int = STEP2_CONFIRMATION_STEPS,
) -> bool:
    """Fold 1: mechanism terminals require completed confirmation evidence only."""
    retry_witness_path = run_root / "prelaunch" / "calibration_warmup_retry_witness.json"
    if retry_witness_path.is_file():
        retry_witness = _load_json(retry_witness_path)
        if int(retry_witness.get("final_rc") or 0) != 0:
            return False
        if retry_witness.get("final_reason") == "non_liveness_failure_no_retry":
            return False

    if _phase_is_liveness_failure(run_root / "calibration_warmup" / "last_active_phase.json"):
        return False

    scratch = run_root / "d_recompute_window_diagnostic"
    probe_receipt_path = scratch / "receipt.json"
    log_path = scratch / "recompute_window_log.jsonl"
    live_path = scratch / "live_carrier_snapshot.jsonl"

    if not probe_receipt_path.is_file() or not log_path.is_file():
        return False

    probe_receipt = _load_json(probe_receipt_path)
    steps_completed = int(probe_receipt.get("steps_completed") or 0)
    if steps_completed != int(confirmation_steps):
        return False

    log_rows = _load_jsonl_rows(log_path)
    if len(log_rows) < int(confirmation_steps):
        return False

    replay_constants = dict(log_rows[0].get("replay_constants") or {})
    if (
        int(replay_constants.get("decay_numerator", -1)) != PREREG_LAW_DECAY_NUM
        or int(replay_constants.get("decay_denominator", -1)) != PREREG_LAW_DECAY_DEN
    ):
        return False

    live_rows = [
        row for row in _load_jsonl_rows(live_path) if row.get("live_carrier_bytes_exact") is True
    ]
    if not live_rows:
        return False

    confirmation_rc_path = run_root / "prelaunch" / "confirmation_launch_rc.txt"
    if confirmation_rc_path.is_file():
        try:
            if int(confirmation_rc_path.read_text(encoding="utf-8").strip()) != 0:
                return False
        except ValueError:
            return False
    else:
        return False

    hygiene_path = run_root / "prelaunch" / "post_confirmation_hygiene_receipt.json"
    if not hygiene_path.is_file():
        return False
    hygiene_receipt = _load_json(hygiene_path)
    if hygiene_receipt.get("pass") is not True:
        return False
    if hygiene_receipt.get("bounded_steps_start_count") != 1:
        return False

    if _phase_is_liveness_failure(scratch / "last_active_phase.json"):
        return False

    return True


def _first_replay_constants(run_root: Path) -> dict[str, Any]:
    log_path = run_root / "d_recompute_window_diagnostic" / "recompute_window_log.jsonl"
    if not log_path.is_file():
        return {}
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        replay = row.get("replay_constants")
        if isinstance(replay, dict):
            return dict(replay)
    return {}


def build_receipt(
    *,
    run_root: Path,
    packet: Mapping[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    scratch = run_root / "d_recompute_window_diagnostic"
    probe_receipt_path = scratch / "receipt.json"
    hygiene_path = run_root / "prelaunch" / "post_confirmation_hygiene_receipt.json"
    probe_receipt = _load_json(probe_receipt_path) if probe_receipt_path.is_file() else {}
    hygiene_receipt = _load_json(hygiene_path) if hygiene_path.is_file() else None
    operational_ok = resolve_step2_operational_ok_from_run_artifacts(run_root)

    branch_inputs = build_branch_input_from_step2_gpu_run(
        run_root=run_root,
        operational_ok=operational_ok,
        recorded_selector_internal_manifest_sha256=(
            B1_RECORDED_SELECTOR_INTERNAL_MANIFEST_SHA256
        ),
        recorded_manifest_file_sha256=B1_RECORDED_MANIFEST_FILE_SHA256,
        effective_acc_budget_bpw=float(
            packet.get("effective_acc_budget_bpw") or DEFAULT_EFFECTIVE_ACC_BUDGET_BPW
        ),
        hygiene_receipt=hygiene_receipt,
        probe_receipt=probe_receipt,
    )
    classification = classify_arc2b_slice5_in_vivo_branch(branch_inputs)
    replay_constants = _first_replay_constants(run_root)

    return {
        "schema": RECEIPT_SCHEMA,
        "task_id": ACTIVE_TASK_ID,
        "git_head_required": git_head(repo_root),
        "run_root": str(run_root),
        "packet_revision": packet.get("packet_revision"),
        "classifier": CLASSIFIER,
        "classifier_module_sha256": sha256_file(repo_root / CLASSIFIER_MODULE),
        "evidence_source": branch_inputs["evidence_source"],
        "prereg_law_window_k": PREREG_LAW_WINDOW_K,
        "prereg_law_decay_num": PREREG_LAW_DECAY_NUM,
        "prereg_law_decay_den": PREREG_LAW_DECAY_DEN,
        "runtime_decay_num": branch_inputs.get("runtime_decay_num"),
        "runtime_decay_den": branch_inputs.get("runtime_decay_den"),
        "runtime_window_k": branch_inputs.get("runtime_window_k"),
        "recorded_selector_internal_manifest_sha256": (
            B1_RECORDED_SELECTOR_INTERNAL_MANIFEST_SHA256
        ),
        "manifest_binding_ok": branch_inputs.get("manifest_binding_ok"),
        "eligible_weight_numel": branch_inputs.get("eligible_weight_numel"),
        "numel_basis_source": NUMEL_BASIS_SOURCE,
        "effective_acc_budget_bpw": branch_inputs.get("effective_acc_budget_bpw"),
        "tolerance_bpw": DEFAULT_TOLERANCE_BPW,
        "live_snapshot_present": branch_inputs.get("live_snapshot_present"),
        "resume_generation": branch_inputs.get("resume_generation"),
        "operational_ok": operational_ok,
        "offline_bracket_decision": branch_inputs.get("offline_bracket_decision"),
        "live_acc_carrier_bpw_max": classification.get("live_acc_carrier_bpw_max"),
        "slice5_branch": classification.get("terminal_branch"),
        "slice5_branch_inputs": branch_inputs,
        "terminal_branch": classification.get("terminal_branch"),
        "fired_branches": classification.get("fired_branches"),
        "ready_for_main_science": False,
        "counts_as_sub2": False,
        "pre_full_stack_diagnostic": True,
        "autonomy_rung": classification.get("autonomy_rung"),
        "replay_constants_row1": replay_constants,
        "anti_overclaim_verbatim": ANTI_OVERCLAIM_VERBATIM,
        "mechanism_terminal_branches": sorted(STEP2_ONLY_TERMINALS),
        "parent_checkpoint_sha256": PARENT_SHA256,
        "generated_at_unix": int(time.time()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, required=True)
    ap.add_argument("--packet", type=Path, required=True)
    ap.add_argument("--repo-root", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    packet = _load_json(args.packet)
    receipt = build_receipt(
        run_root=args.run_root,
        packet=packet,
        repo_root=args.repo_root,
    )
    out_path = args.out or (
        args.run_root / "arc2b_slice5_in_vivo_law_validation_receipt.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
