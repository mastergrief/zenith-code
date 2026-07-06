#!/usr/bin/env python3
"""Apply Arc #2b Slice-5 Step 1 CPU prereg/preflight packet (design/preflight only; no GPU)."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from calm.hrm_text_158.native_full_stack.arc2b_slice5_in_vivo_branch import (
    ALLOWED_CLAIM,
    ANTI_OVERCLAIM_VERBATIM,
    B1_RECORDED_MANIFEST_FILE_SHA256,
    B1_RECORDED_SELECTOR_INTERNAL_MANIFEST_SHA256,
    B1_RUN_ID,
    B1_RUN_ROOT,
    CLASSIFIER,
    DEFAULT_EFFECTIVE_ACC_BUDGET_BPW,
    DEFAULT_TOLERANCE_BPW,
    EVIDENCE_B1_OFFLINE_BRACKET,
    EVIDENCE_STEP2_GPU_LIVE_CARRIER,
    NUMEL_BASIS_SOURCE,
    PREREG_LAW_DECAY_DEN,
    PREREG_LAW_DECAY_NUM,
    PREREG_LAW_WINDOW_K,
    REQUIRED_RECEIPT_FIELDS,
    STEP2_ONLY_TERMINALS,
    build_branch_input_from_b1_classifier_receipt,
    validate_preflight_receipt_schema,
    validate_prereg_packet_schema,
)

REPO = Path("/mnt/c/Users/gabes/projects/claw-code-hrm-text-158")
HEAD = "f1757f8b6f14ef47f89b66bb40ba430ad44ce715"
ACTIVE_TASK_ID = "1783272482268-052281aa"
DISPATCH_MSG_ID = "1783272622169-d840d3e0"
PLAN_MSG_ID = "1783274175050-ccc23e52"
GATE1_FREEZE_MSG_ID = "1783274282142-6bbfe32a"
CO_LEAD_GATE2_MSG_ID = "1783322348922"
IMPLEMENT_GATE_MSG_ID = "1783322937495-b4cb2763"
OFFLINE_BRACKET_CONTRACT_MSG_ID = "1783274356049"

DRAFT = (
    REPO
    / "artifacts/consensus_prep/arc2b_slice5_in_vivo_law_validation_prereg_packet_v1_draft.json"
)
PREFLIGHT = (
    REPO
    / "artifacts/measurement_closeout/arc2b_slice5_feasibility_preflight_receipt.json"
)

B1_CLASSIFIER_RECEIPT = Path(B1_RUN_ROOT) / "classifier_receipt.json"
B1_LOG = (
    Path(B1_RUN_ROOT)
    / "d_recompute_window_diagnostic"
    / "recompute_window_log.jsonl"
)
B1_MANIFEST = Path(B1_RUN_ROOT) / "prelaunch" / "calibrated_selector_manifest.json"
B1_SUB2_GATE = Path(B1_RUN_ROOT) / "prelaunch" / "sub2_first_launch_gate_receipt.json"


def git_head() -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_branch_enum() -> list[dict[str, str]]:
    return [
        {"rank": 1, "branch": "SLICE5_NO_VERDICT_OPERATIONAL"},
        {"rank": 2, "branch": "SLICE5_NO_VERDICT_SCHEMA"},
        {"rank": 3, "branch": "SLICE5_INCONCLUSIVE_INPUT_DRIFT"},
        {"rank": 4, "branch": "SLICE5_INCONCLUSIVE_SOURCE_MISMATCH"},
        {"rank": 5, "branch": "SLICE5_INCONCLUSIVE_LOG_COVERAGE"},
        {"rank": 6, "branch": "SLICE5_INCONCLUSIVE_NO_LIVE_SNAPSHOT"},
        {"rank": 7, "branch": "SLICE5_DIAGNOSTIC_BRACKET_ONLY"},
        {"rank": 8, "branch": "D_NEEDS_UPDATE_LAW_REDESIGN"},
        {"rank": 9, "branch": "SLICE5_IN_VIVO_LAW_BOUND"},
    ]


def build_carrier_byte_mapping() -> dict[str, Any]:
    return {
        "live_acc_carrier_bytes_total": (
            "events_bytes + backlog_bytes + hot_exact_bytes + metadata_bytes"
        ),
        "bpw_formula": "live_acc_carrier_bytes_total * 8 / eligible_weight_numel",
        "numel_basis_source": NUMEL_BASIS_SOURCE,
        "effective_acc_budget_bpw": DEFAULT_EFFECTIVE_ACC_BUDGET_BPW,
        "comparison": "strict_less_than",
        "tolerance_bpw": DEFAULT_TOLERANCE_BPW,
        "requires_live_carrier_bytes_exact": True,
    }


def build_offline_b1_contract() -> dict[str, Any]:
    return {
        "b1_run_id": B1_RUN_ID,
        "b1_runtime_decay": {"decay_num": 1, "decay_den": 1},
        "law_under_test": {
            "window_k": PREREG_LAW_WINDOW_K,
            "decay_num": PREREG_LAW_DECAY_NUM,
            "decay_den": PREREG_LAW_DECAY_DEN,
        },
        "decay_mismatch_diagnostic_only": True,
        "allowed_terminal_branches": [
            "SLICE5_DIAGNOSTIC_BRACKET_ONLY",
            "SLICE5_INCONCLUSIVE_SOURCE_MISMATCH",
            "SLICE5_INCONCLUSIVE_NO_LIVE_SNAPSHOT",
            "SLICE5_INCONCLUSIVE_INPUT_DRIFT",
            "SLICE5_INCONCLUSIVE_LOG_COVERAGE",
        ],
        "forbidden_terminal_branches": sorted(STEP2_ONLY_TERMINALS),
        "manifest_binding": {
            "recorded_selector_internal_manifest_sha256": (
                B1_RECORDED_SELECTOR_INTERNAL_MANIFEST_SHA256
            ),
            "recorded_manifest_file_sha256": B1_RECORDED_MANIFEST_FILE_SHA256,
            "bind_via_classifier_receipt_hashes_not_on_disk_file_alone": True,
            "on_drift_branch": "SLICE5_INCONCLUSIVE_INPUT_DRIFT",
        },
        "contract_msg_id": OFFLINE_BRACKET_CONTRACT_MSG_ID,
    }


def build_step2_gpu_launch_scope() -> dict[str, Any]:
    return {
        "deferred_until_separate_plus_one_launch": True,
        "required_runtime_decay": {
            "decay_num": PREREG_LAW_DECAY_NUM,
            "decay_den": PREREG_LAW_DECAY_DEN,
        },
        "required_window_k": PREREG_LAW_WINDOW_K,
        "resume_generation": 0,
        "live_carrier_snapshot_required": True,
        "live_carrier_bytes_exact_required": True,
        "horizon_h": 200,
        "from_clean_parent_contiguous": True,
        "mechanism_terminal_branches": sorted(STEP2_ONLY_TERMINALS),
    }


def build_b1_diagnostic_anchor() -> dict[str, Any]:
    receipt = load_json_if_exists(B1_CLASSIFIER_RECEIPT) or {}
    acc_sizing = dict(receipt.get("acc_sizing") or {})
    best_grid = dict(acc_sizing.get("best_grid_row") or {})
    in_vivo = dict(receipt.get("in_vivo_validation") or {})
    logged_surface = dict(in_vivo.get("logged_density_surface") or {})
    return {
        "run_root": B1_RUN_ROOT,
        "run_id": B1_RUN_ID,
        "classifier_receipt_path": str(B1_CLASSIFIER_RECEIPT),
        "recompute_window_log_path": str(B1_LOG),
        "calibrated_selector_manifest_path": str(B1_MANIFEST),
        "sub2_first_launch_gate_receipt_path": str(B1_SUB2_GATE),
        "law_under_test": {
            "window_k": int(best_grid.get("window_k") or PREREG_LAW_WINDOW_K),
            "decay_num": int(best_grid.get("decay_num") or PREREG_LAW_DECAY_NUM),
            "decay_den": int(best_grid.get("decay_den") or PREREG_LAW_DECAY_DEN),
            "inclusive_acc_bpw": best_grid.get("inclusive_acc_bpw"),
            "effective_acc_budget_bpw": acc_sizing.get("effective_acc_budget_bpw"),
        },
        "runtime_replay_decay": {"decay_num": 1, "decay_den": 1},
        "logged_density_surface": {
            "peak_backlog_depth": logged_surface.get("peak_backlog_depth"),
            "records_in_window": logged_surface.get("records_in_window"),
            "steps_in_window": logged_surface.get("steps_in_window"),
        },
        "input_artifact_hashes": receipt.get("input_artifact_hashes"),
        "diagnostic_only": True,
    }


def build_prereg_packet() -> dict[str, Any]:
    return {
        "schema": "hrm_text_158_arc2b_slice5_in_vivo_law_validation_prereg_packet/v1",
        "task_id": ACTIVE_TASK_ID,
        "fold": "arc2b_slice5_in_vivo_law_validation_step1_prereg",
        "git_head_required": HEAD,
        "provenance": {
            "dispatch_msg_id": DISPATCH_MSG_ID,
            "plan_msg_id": PLAN_MSG_ID,
            "gate1_freeze_msg_id": GATE1_FREEZE_MSG_ID,
            "co_lead_gate2_msg_id": CO_LEAD_GATE2_MSG_ID,
            "implement_gate_msg_id": IMPLEMENT_GATE_MSG_ID,
            "offline_bracket_contract_msg_id": OFFLINE_BRACKET_CONTRACT_MSG_ID,
        },
        "classifier": CLASSIFIER,
        "branch_enum": build_branch_enum(),
        "receipt_schema": {
            "schema": "hrm_text_158_arc2b_slice5_in_vivo_law_validation_receipt/v1",
            "required_fields": list(REQUIRED_RECEIPT_FIELDS),
        },
        "feasibility_preflight": {
            "verdict": "NEEDS_PATCH",
            "summary": (
                "B1 diagnostic anchor exists with decay 1/1 runtime vs 1/2 law; "
                "Step-2 GPU live-carrier launch required for mechanism terminals."
            ),
        },
        "run_determinism": {
            "classification": "DETERMINISTIC",
            "static": [
                "frozen B1 parent checkpoint pins in launch packet",
                "deterministic recompute-window log replay constants",
                "no random/torch.rand in slice5 carrier-byte observer path",
            ],
            "empirical_upgrade_path": (
                "Step-2 from-clean-parent contiguous H=200 GPU with decay 1/2 active"
            ),
        },
        "step2_gpu_launch_scope": build_step2_gpu_launch_scope(),
        "b1_diagnostic_anchor": build_b1_diagnostic_anchor(),
        "carrier_byte_mapping": build_carrier_byte_mapping(),
        "offline_b1_contract": build_offline_b1_contract(),
        "readiness_classification": {
            "class": "pre_full_stack_diagnostic",
            "flags": {
                "ready_for_main_science": False,
                "counts_as_sub2": False,
                "pre_full_stack_diagnostic": True,
            },
            "q2_rationale": (
                "B1 sub2_first_launch_gate_receipt: ready_for_main_science=false, "
                "main_science_launch_blocked=true; q_sidecar_vote_carrier in "
                "pre_full_stack_diagnostic surface set."
            ),
        },
        "decision_contract": {
            "precedence": [
                "operational",
                "source",
                "schema_coverage",
                "mechanism",
            ],
            "autonomy_rung_field": "autonomy_rung",
            "step1_autonomy_rung": "step1_cpu_prereg",
            "step2_autonomy_rung": "step2_gpu_mechanism",
            "q1_layers": {
                "offline_diagnostic_sufficient_for_prepass": True,
                "primary_proof_insufficient_from_b1_log": True,
                "reanalysis_insufficient_wrong_decay": True,
            },
        },
        "claim_boundary": {
            "allowed_claim_verbatim": ALLOWED_CLAIM,
            "anti_overclaim_verbatim": ANTI_OVERCLAIM_VERBATIM,
            "forbidden": [
                "sub-2 readiness",
                "reduction eligibility",
                "~430MB bank pin",
                "Fold-3B universalization",
                "bank mutation",
                "full-stack readiness",
                "implementation readiness",
                "main-science launch from Step-1 CPU prereg alone",
            ],
        },
        "step1_scope": {
            "design_preflight_only": True,
            "no_gpu_launch": True,
            "no_live_classifier_wiring": True,
            "no_commit": True,
            "no_push": True,
        },
    }


def build_preflight_receipt(packet: dict[str, Any]) -> dict[str, Any]:
    b1_receipt = load_json_if_exists(B1_CLASSIFIER_RECEIPT) or {}
    sub2_gate = load_json_if_exists(B1_SUB2_GATE) or {}
    branch_inputs = None
    if b1_receipt:
        branch_inputs = build_branch_input_from_b1_classifier_receipt(
            b1_receipt,
            on_disk_manifest_path=B1_MANIFEST,
        )

    return {
        "schema": "hrm_text_158_arc2b_slice5_feasibility_preflight_receipt/v1",
        "task_id": ACTIVE_TASK_ID,
        "fold": "arc2b_slice5_step1_feasibility_preflight",
        "git_head_at_preflight": git_head(),
        "design_spec_path": str(DRAFT.relative_to(REPO)),
        "provenance": packet["provenance"],
        "feasibility_verdict": "NEEDS_PATCH",
        "run_determinism_classification": "DETERMINISTIC",
        "b1_diagnostic_anchor": packet["b1_diagnostic_anchor"],
        "carrier_byte_mapping": packet["carrier_byte_mapping"],
        "offline_b1_contract": packet["offline_b1_contract"],
        "readiness_classification": packet["readiness_classification"],
        "anti_overclaim_verbatim": ANTI_OVERCLAIM_VERBATIM,
        "upstream_receipt_grounding": {
            "b1_classifier_receipt": {
                "receipt_path": str(B1_CLASSIFIER_RECEIPT),
                "run_id": b1_receipt.get("run_id"),
                "primary_classifier": b1_receipt.get("primary_classifier"),
                "acc_sizing_verdict": (b1_receipt.get("acc_sizing") or {}).get(
                    "sizing_verdict"
                ),
                "in_vivo_verdict": (b1_receipt.get("in_vivo_validation") or {}).get(
                    "in_vivo_verdict"
                ),
                "manifest_binding_inputs": branch_inputs,
            },
            "sub2_first_launch_gate": {
                "receipt_path": str(B1_SUB2_GATE),
                "ready_for_main_science": sub2_gate.get("ready_for_main_science"),
                "main_science_launch_blocked": sub2_gate.get(
                    "main_science_launch_blocked"
                ),
                "pre_full_stack_diagnostic_surface_names": sub2_gate.get(
                    "pre_full_stack_diagnostic_surface_names"
                ),
            },
        },
        "determinism_evidence": packet["run_determinism"],
        "step2_gpu_launch_scope": packet["step2_gpu_launch_scope"],
    }


def verify_offline_b1_contract(packet: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    contract = packet.get("offline_b1_contract") or {}
    if contract.get("decay_mismatch_diagnostic_only") is not True:
        failures.append("decay_mismatch_diagnostic_only_not_true")
    forbidden = set(contract.get("forbidden_terminal_branches") or [])
    if forbidden != set(STEP2_ONLY_TERMINALS):
        failures.append("forbidden_terminal_branches_mismatch")
    manifest = contract.get("manifest_binding") or {}
    if (
        manifest.get("recorded_selector_internal_manifest_sha256")
        != B1_RECORDED_SELECTOR_INTERNAL_MANIFEST_SHA256
    ):
        failures.append("manifest_recorded_hash_mismatch")
    return failures


def verify_b1_branch_input_grounding() -> list[str]:
    failures: list[str] = []
    receipt = load_json_if_exists(B1_CLASSIFIER_RECEIPT)
    if receipt is None:
        return ["b1_classifier_receipt_missing"]
    inputs = build_branch_input_from_b1_classifier_receipt(
        receipt,
        on_disk_manifest_path=B1_MANIFEST,
    )
    if inputs.get("runtime_decay_num") != 1 or inputs.get("runtime_decay_den") != 1:
        failures.append("b1_runtime_decay_not_1_over_1")
    if inputs.get("prereg_law_decay_num") != 1 or inputs.get("prereg_law_decay_den") != 2:
        failures.append("prereg_law_decay_not_1_over_2")
    if inputs.get("manifest_binding_ok") is not True:
        failures.append("b1_manifest_binding_not_ok")
    return failures


def self_verify() -> dict[str, Any]:
    failures: list[str] = []
    if not DRAFT.is_file():
        failures.append("draft_missing")
    if not PREFLIGHT.is_file():
        failures.append("preflight_missing")
    if failures:
        return {"ok": False, "failures": failures}

    packet = json.loads(DRAFT.read_text(encoding="utf-8"))
    preflight = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    failures.extend(validate_prereg_packet_schema(packet))
    failures.extend(validate_preflight_receipt_schema(preflight))
    failures.extend(verify_offline_b1_contract(packet))
    failures.extend(verify_b1_branch_input_grounding())

    if packet.get("git_head_required") != HEAD:
        failures.append("git_head_required_mismatch")
    if preflight.get("feasibility_verdict") != "NEEDS_PATCH":
        failures.append("preflight_verdict_not_needs_patch")
    if preflight.get("run_determinism_classification") != "DETERMINISTIC":
        failures.append("preflight_not_deterministic")

    draft_sha = hashlib.sha256(DRAFT.read_bytes()).hexdigest()
    regen = build_prereg_packet()
    regen_sha = hashlib.sha256(
        (json.dumps(regen, indent=2, sort_keys=True) + "\n").encode()
    ).hexdigest()
    deterministic = draft_sha == regen_sha

    return {
        "ok": not failures and deterministic,
        "failures": failures,
        "deterministic_regen": deterministic,
        "draft_sha256": draft_sha,
        "git_head": git_head(),
        "evidence_sources": {
            EVIDENCE_B1_OFFLINE_BRACKET: True,
            EVIDENCE_STEP2_GPU_LIVE_CARRIER: "deferred_step2",
        },
    }


def main() -> int:
    packet = build_prereg_packet()
    preflight = build_preflight_receipt(packet)

    DRAFT.parent.mkdir(parents=True, exist_ok=True)
    PREFLIGHT.parent.mkdir(parents=True, exist_ok=True)
    DRAFT.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    PREFLIGHT.write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    result = self_verify()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
