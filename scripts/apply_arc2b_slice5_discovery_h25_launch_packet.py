#!/usr/bin/env python3
"""Apply Arc #2b Slice-5 discovery H=25 launch packet (DRAFT only, separate launch-gate).

Frozen v6 plan (co_lead gate-2 PASS 1783512484577, +1 implement 1783526612437).
DRAFT C/D/E H=25 launch packet. Launch itself goes through the SEPARATE
launch-gate -> test-operator, NOT this +1 implement.

3 GPU arms: C (decay 1/4), D (control decay 1/2), E (decay 9/10).
H=25 sentinel first for each; extend to H=50 only if stable <300s/step.
Same parent 9b4e311a, seed 43, 8 modules (8,650,752 numel), event-coded carrier.
"""

from __future__ import annotations

import hashlib
import json
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
HEAD = "db29e06359f0b17151fab526f1a986e2003eaacd"
ACTIVE_TASK_ID = "1783272482268-052281aa"
PACKET_REVISION = "v1_arc2b_slice5_discovery_h25_draft"
CANONICAL_LANE = "test-operator_gpu_arc2b_slice5_discovery_h25"

# Provenance (frozen v6)
DISPATCH_MSG_ID = "1783508692172-939db36f"
PLAN_MSG_ID = "1783512246673-8b8492e8"
GATE1_FREEZE_MSG_ID = "1783512308286-b3269c20"
CO_LEAD_GATE2_MSG_ID = "1783512484577-a5195760"
IMPLEMENT_GATE_MSG_ID = "1783526612437-ed9e3bba"

DRAFT = (
    REPO
    / "artifacts/consensus_prep/arc2b_slice5_discovery_h25_launch_packet_v1_draft.json"
)

CLASSIFIER_MODULE = REPO / "calm/hrm_text_158/native_full_stack/arc2b_slice5_discovery_branch.py"
ARM_A_SCRIPT = REPO / "scripts/hrm_text_158_arc2b_slice5_discovery_arm_a_static.py"
ARM_B_SCRIPT = REPO / "scripts/hrm_text_158_arc2b_slice5_discovery_arm_b_offline.py"

PARENT_REL = (
    "calm/hrm/checkpoints/hrm_text_158_phase3_L0c1_seed0017_replay83_n12k_lr7p5e5_pc1p0_rsL0b1math1r1b2_1_anchorsv1r3_from_L0b_final_step01500.pt"
)
PARENT_SHA256 = "9b4e311a22787e7d4808bde7bc2953568d767a2ee8ac648942a3f5dbb7b4d5ec"

H25_SENTINEL_STEPS = 25
H50_EXTENSION_STEPS = 50
ELIGIBLE_MODULE_LIMIT = 8
MAX_SILENT_PHASE_SECONDS = 600

# Decay flags per arm
ARM_C_DECAY_FLAGS = ("--vote-update-decay-numerator", "1", "--vote-update-decay-denominator", "4")
ARM_D_DECAY_FLAGS = ("--vote-update-decay-numerator", "1", "--vote-update-decay-denominator", "2")
ARM_E_DECAY_FLAGS = ("--vote-update-decay-numerator", "9", "--vote-update-decay-denominator", "10")

# Event-coded carrier (same as H=200 Step-2)
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

# Ordering: A+B 0 GPU first -> D H=25 -> C H=25 -> E H=25 -> extend H=50 if stable <300s
ARM_ORDERING = ("arm_a_static", "arm_b_offline", "arm_d_h25", "arm_c_h25", "arm_e_h25")


def git_head() -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


def build_arm_spec(
    arm_name: str,
    decay_num: int,
    decay_den: int,
    decay_flags: tuple[str, ...],
    *,
    steps: int = H25_SENTINEL_STEPS,
) -> dict[str, Any]:
    """Build per-arm launch spec for the DRAFT packet."""
    return {
        "arm_name": arm_name,
        "decay_num": decay_num,
        "decay_den": decay_den,
        "decay_flags": list(decay_flags),
        "steps": int(steps),
        "eligible_module_limit": ELIGIBLE_MODULE_LIMIT,
        "carrier_required_flags": list(CARRIER_REQUIRED_FLAGS),
        "forbidden_carrier_tokens": list(FORBIDDEN_CARRIER_TOKENS),
        "event_coded_incompatible_flags": list(EVENT_CODED_INCOMPATIBLE_FLAGS),
        "max_silent_phase_seconds": MAX_SILENT_PHASE_SECONDS,
        "resume_generation_required": 0,
        "from_clean_parent_contiguous": True,
        "live_carrier_bytes_exact_required": True,
        "extend_to_h50_if_stable_under_seconds": 300,
        "h50_extension_steps": H50_EXTENSION_STEPS,
    }


def build_packet(classifier_sha: str) -> dict[str, Any]:
    """Build the DRAFT C/D/E H=25 launch packet."""
    return {
        "schema": "hrm_text_158_arc2b_slice5_discovery_h25_launch_packet/v1",
        "task_id": ACTIVE_TASK_ID,
        "fold": "arc2b_slice5_discovery_h25_launch",
        "packet_revision": PACKET_REVISION,
        "packet_status": "DRAFT_ONLY_SEPARATE_LAUNCH_GATE_REQUIRED",
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
            "arm_c": build_arm_spec(
                "arm_c",
                DECAY_POINT_C_NUM,
                DECAY_POINT_C_DEN,
                ARM_C_DECAY_FLAGS,
            ),
            "arm_d": build_arm_spec(
                "arm_d",
                DECAY_POINT_D_NUM,
                DECAY_POINT_D_DEN,
                ARM_D_DECAY_FLAGS,
            ),
            "arm_e": build_arm_spec(
                "arm_e",
                DECAY_POINT_E_NUM,
                DECAY_POINT_E_DEN,
                ARM_E_DECAY_FLAGS,
            ),
        },
        "offline_arms": {
            "arm_a_static": {
                "script": str(ARM_A_SCRIPT.relative_to(REPO)),
                "gpu": False,
                "finding": "W8=8/W7=7 bpw floor, cannot reach sub-2",
            },
            "arm_b_offline": {
                "script": str(ARM_B_SCRIPT.relative_to(REPO)),
                "source_run_id": "2189e72017",
                "gpu": False,
                "deliverable": "K* saturation trend (separate-axis, likely-negative control)",
                "caveats": "decay 1/1 != 1/2 law; censored@200",
            },
        },
        "ordering": list(ARM_ORDERING),
        "liveness_estimate": {
            "h25_likely_reachable": True,
            "h50_risky_if_growth_continues": True,
            "arm_e_more_liveness_favorable_than_d": True,
            "step_time_estimate_seconds": "154-300s/step at 8 modules",
        },
        "operational_guard": {
            "any_live_arm_ineligible_or_liveness_fails": "no_2x2_terminal",
            "classify_operational_or_inconclusive": True,
        },
        "lane_field_fail_closed": {
            "required_fields": [
                "lane_indices",
                "acc_before_lanes",
                "acc_after_lanes",
                "vote_lanes",
            ],
            "arm_b_offline_fails_closed_on_missing": True,
        },
        "readiness_classification": {
            "class": "pre_full_stack_diagnostic",
            "flags": {
                "ready_for_main_science": False,
                "counts_as_sub2": False,
                "pre_full_stack_diagnostic": True,
            },
        },
        "explicit_non_claims": [
            "not_sub2_readiness",
            "not_reduction_eligibility",
            "not_bank_pin",
            "not_fold3b_universalization",
            "not_h200_d_terminal_verdict",
            "not_asymptotic_k_star_at_h100_h200",
            "not_backlog_direction_flip_erosion",
            "not_from_clean_parent_contiguous_proof",
            "not_terminal_verdict_before_registered_complete_arms",
            "not_gpu_launch_until_separate_plus1_launch_gate",
        ],
        "gpu_hold": True,
        "separate_launch_gate_required": True,
    }


def verify_packet(packet: dict[str, Any]) -> list[str]:
    """Verify the DRAFT packet against frozen v6 requirements."""
    failures: list[str] = []
    if packet.get("git_head_required") != HEAD:
        failures.append("git_head_required_stale")
    if packet.get("packet_status") != "DRAFT_ONLY_SEPARATE_LAUNCH_GATE_REQUIRED":
        failures.append("packet_status_not_draft_only")
    if not packet.get("gpu_hold"):
        failures.append("gpu_hold_not_set")
    if not packet.get("separate_launch_gate_required"):
        failures.append("separate_launch_gate_required_not_set")

    arms = packet.get("arms") or {}
    for arm_key in ("arm_c", "arm_d", "arm_e"):
        arm = arms.get(arm_key) or {}
        if not arm.get("decay_flags"):
            failures.append(f"{arm_key}_missing_decay_flags")
        if arm.get("steps") != H25_SENTINEL_STEPS:
            failures.append(f"{arm_key}_steps_not_h25")
        if arm.get("eligible_module_limit") != ELIGIBLE_MODULE_LIMIT:
            failures.append(f"{arm_key}_eligible_module_limit_not_8")
        if not arm.get("carrier_required_flags"):
            failures.append(f"{arm_key}_missing_carrier_flags")
        if arm.get("resume_generation_required") != 0:
            failures.append(f"{arm_key}_resume_generation_not_0")

    # Verify decay values
    if arms.get("arm_c", {}).get("decay_num") != DECAY_POINT_C_NUM:
        failures.append("arm_c_decay_num_wrong")
    if arms.get("arm_c", {}).get("decay_den") != DECAY_POINT_C_DEN:
        failures.append("arm_c_decay_den_wrong")
    if arms.get("arm_d", {}).get("decay_num") != DECAY_POINT_D_NUM:
        failures.append("arm_d_decay_num_wrong")
    if arms.get("arm_d", {}).get("decay_den") != DECAY_POINT_D_DEN:
        failures.append("arm_d_decay_den_wrong")
    if arms.get("arm_e", {}).get("decay_num") != DECAY_POINT_E_NUM:
        failures.append("arm_e_decay_num_wrong")
    if arms.get("arm_e", {}).get("decay_den") != DECAY_POINT_E_DEN:
        failures.append("arm_e_decay_den_wrong")

    # Verify ordering
    if packet.get("ordering") != list(ARM_ORDERING):
        failures.append("ordering_drift")

    # Verify non-claims
    non_claims = packet.get("explicit_non_claims") or []
    if "not_gpu_launch_until_separate_plus1_launch_gate" not in non_claims:
        failures.append("missing_non_claim_gpu_launch_gate")

    return failures


def self_verify() -> dict[str, Any]:
    failures: list[str] = []
    if not DRAFT.is_file():
        failures.append("draft_missing")
        return {"ok": False, "failures": failures}

    classifier_sha = sha256_file(CLASSIFIER_MODULE)
    packet = json.loads(DRAFT.read_text(encoding="utf-8"))

    if packet.get("classifier_binding", {}).get("module_sha256") != classifier_sha:
        failures.append("classifier_sha_pin_drift")
    failures.extend(verify_packet(packet))

    regen_classifier_sha = sha256_file(CLASSIFIER_MODULE)
    regen_packet = build_packet(regen_classifier_sha)
    draft_sha = hashlib.sha256(DRAFT.read_bytes()).hexdigest()
    regen_draft_sha = hashlib.sha256(
        (json.dumps(regen_packet, indent=2, sort_keys=True) + "\n").encode()
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
        ),
        "failures": failures,
        "deterministic_regen": draft_sha == regen_draft_sha,
        "draft_sha256": draft_sha,
        "classifier_module_sha256": classifier_sha,
        "git_head": live_git_head,
        "pins_match_commit": pins_match_commit,
    }


def main() -> int:
    classifier_sha = sha256_file(CLASSIFIER_MODULE)
    packet = build_packet(classifier_sha)

    DRAFT.parent.mkdir(parents=True, exist_ok=True)
    DRAFT.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    result = self_verify()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
