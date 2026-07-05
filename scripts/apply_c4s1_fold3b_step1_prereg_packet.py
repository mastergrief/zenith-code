#!/usr/bin/env python3
"""Apply Fold-3B Step 1 CPU prereg/preflight packet (design/preflight only; no GPU)."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

REPO = Path("/mnt/c/Users/gabes/projects/claw-code-hrm-text-158")
HEAD = "891a5e6ea24506bb3bcdb039efd8ec6a4732c963"
ACTIVE_TASK_ID = "1782633464140-b85ec12a"
DISPATCH_MSG_ID = "1783248579236-6297ec97"
PLAN_MSG_ID = "1783248683587-a3d3c2e9"
IMPLEMENT_GATE_MSG_ID = "1783248830033-357917e8"

DRAFT = REPO / "artifacts/consensus_prep/c4s1_fold3b_step1_prereg_packet_v1_draft.json"
PREFLIGHT = (
    REPO
    / "artifacts/measurement_closeout/c4s1_fold3b_step1_feasibility_preflight_receipt.json"
)

DENSE_PRIMARY_RECEIPT = (
    "/home/gabe/hrm158_c4s1_phase3_gpu_gate/C4S1_PHASE3_N32_DENSE_09_CA_DECIDER_V1/"
    "prelaunch/callsite_band_counter_ca_confirmation/"
    "callsite_band_counter_ca_confirmation_receipt.json"
)
DENSE_WRAPPER_RECEIPT = (
    "/home/gabe/hrm158_c4s1_phase3_gpu_gate/C4S1_PHASE3_N32_DENSE_09_CA_DECIDER_V1/"
    "prelaunch/ca_confirmation_wrapper_receipt.json"
)
FOLD3A_RECEIPT = (
    "artifacts/measurement_closeout/c4s1d7_fold3a_cb_only_dominance_dense_09_receipt.json"
)
DENSE_FORK_CLOSEOUT = (
    "artifacts/measurement_closeout/c4s1d7_dense_09_structural_fork_resolution_receipt.json"
)

IDENTITY_ORDER = list(range(10))
REVERSED_ORDER = list(range(9, -1, -1))
STATE0_OMISSION_ORDER = list(range(1, 11))

ANTI_OVERCLAIM_VERBATIM = (
    "Within the Fold-3B packet scope, state0-only crossing support classifies as one of "
    "the pre-registered branches. FORBIDDEN: candidate-C, CA/reduction eligibility, W/P, "
    "~430MB bank pin, universal all-state census, bank mutation, sub-2 readiness, "
    "full-stack readiness, implementation readiness."
)

SOURCE_TRACE_ANCHORS = [
    {
        "file": "calm/hrm_text_158/native_full_stack/sparse_cap_gpu_seam_adapter.py",
        "lines": "57-62",
        "finding": "dedup session keyed by (optimizer_step_index, state_index, site_id, suffix)",
    },
    {
        "file": "calm/hrm_text_158/native_full_stack/host_tracemalloc_probe.py",
        "lines": "111-156",
        "finding": "env list parsed then converted to frozenset (order discarded at :149)",
    },
    {
        "file": "calm/hrm_text_158/native_full_stack/sparse_cap_gpu_seam_adapter.py",
        "lines": "482-492",
        "finding": "membership-only sampled_states gate",
    },
    {
        "file": "calm/hrm_text_158/native_full_stack/bounded_delta_learner.py",
        "lines": "2653-2657",
        "finding": "enumerate(tensor_results) fixes measurement order = iteration order",
    },
]


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


def build_identity_order_inertness_precondition() -> dict[str, Any]:
    return {
        "name": "IDENTITY_ORDER_INERTNESS_PRECONDITION",
        "blocks_variable_a_interpretation": True,
        "summary": (
            "Variable A (reversed order) is NOT interpretable until the order-control "
            "patch slice proves that explicit identity order reproduces the baseline "
            "dense primary exactly."
        ),
        "identity_order": IDENTITY_ORDER,
        "expected_baseline_match": {
            "semantic_state0_crossing_indices_len": 512,
            "cb_state_count": 1,
            "mark_count": 10,
            "sampled_state_set": IDENTITY_ORDER,
            "exact_per_state_coverage": True,
        },
        "baseline_comparison_receipt_path": DENSE_PRIMARY_RECEIPT,
        "baseline_comparison_fields": {
            "cb_state_count": 1,
            "mark_count": 10,
            "state0_crossing_indices_len": 512,
            "terminal_branch": "INSUFFICIENT_CB_STATES",
        },
        "validation_owner": "order_control_patch_slice_gpu_validation",
        "counts_against_fold3b_gpu_budget": False,
        "dual_purpose": (
            "Independent GPU identity-order run must agree with prior dense primary — "
            "empirically confirms run-determinism (static-grep + cross-run agreement)."
        ),
        "consequence_if_unproven": (
            "Reversed-order difference is confounded by the patch mechanism itself."
        ),
    }


def build_order_control_patch_scope() -> dict[str, Any]:
    return {
        "verdict": "NEEDS_PATCH",
        "proposed_env": "HRM_TEXT_158_OBMALLOC_EXPANDED_SAMPLED_STATE_ORDER",
        "primary_seam": "bounded_delta_learner.py C4 apply loop (~2653)",
        "behavior": (
            "Iterate tensor_results in declared order; pass true semantic state_index; "
            "emit sampled_state_order + order_rank_by_semantic_state."
        ),
        "default_preserving": (
            "Unset order env → current enumerate(tensor_results) path byte/behavior unchanged."
        ),
        "cpu_regression_in_patch_slice": "unset-order default-preserving test",
        "gpu_regression_in_patch_slice": "identity-order inertness proof (this precondition)",
        "implementation_status": "NOT_BUILT_IN_STEP1",
        "separate_slice_required": True,
    }


def build_gpu_ladder() -> dict[str, Any]:
    return {
        "budget": {
            "max_gpu_launches": 4,
            "max_n_units": 140,
            "formula": "2 variables × (N=20 screen + N=50/equiv verdict)",
            "identity_inertness_in_fold3b_budget": False,
            "early_stop_on_terminal_branch": True,
            "no_implicit_science_retry": True,
        },
        "run_determinism": {
            "classification": "DETERMINISTIC",
            "n50_equivalent_semantics": (
                "exact baseline-vs-perturbed CONTROL-PAIR CONTRAST (one deterministic "
                "receipt per arm; no stochastic N=50 repeats)"
            ),
            "terminal_threshold": (
                "exact/deterministic branch classification on complete receipts"
            ),
            "static_evidence": [
                "fixed parent checkpoint",
                "sorted tensor_states/inputs iteration",
                "no RNG in slice5 CA path",
            ],
            "empirical_confirmation": (
                "identity-order inertness GPU run must match dense primary baseline"
            ),
        },
        "variables": [
            {
                "variable_id": "A_order_only",
                "order": REVERSED_ORDER,
                "sampled_state_set": IDENTITY_ORDER,
                "control_reason": "order_only_perturbation",
                "decision_rule": {
                    "F3B_STATE0_IDENTITY_STRUCTURE": (
                        "semantic state0 remains sole crossing-bearing state"
                    ),
                    "F3B_MEASUREMENT_ORDER_ARTIFACT": (
                        "first-measured semantic state (9 under reversed order) becomes CB"
                    ),
                },
                "blocked_until": "IDENTITY_ORDER_INERTNESS_PRECONDITION",
            },
            {
                "variable_id": "B_state0_omission",
                "order": STATE0_OMISSION_ORDER,
                "sampled_state_set": STATE0_OMISSION_ORDER,
                "control_reason": "state0_omission_or_shifted_set",
                "launch_condition": (
                    "only if Variable A reaches state0-identity or inconclusive-without-artifact"
                ),
                "never_folds_into_variable_a_verdict": True,
            },
        ],
        "n20_role": "liveness/null/schema only; must not be final mechanism verdict",
        "n50_equiv_role": "verdict arm under DETERMINISTIC classification",
    }


def build_branch_enum() -> list[dict[str, str]]:
    return [
        {"rank": 1, "branch": "F3B_NO_VERDICT_OPERATIONAL"},
        {"rank": 2, "branch": "F3B_NO_VERDICT_SCHEMA"},
        {"rank": 3, "branch": "F3B_MEASUREMENT_ORDER_ARTIFACT"},
        {"rank": 4, "branch": "F3B_SAMPLE_SET_OR_ELIGIBILITY_ARTIFACT"},
        {"rank": 5, "branch": "F3B_MARKING_OR_DEDUP_ARTIFACT"},
        {"rank": 6, "branch": "F3B_STATE0_IDENTITY_STRUCTURE"},
        {"rank": 7, "branch": "F3B_MIXED_OR_INCONCLUSIVE"},
    ]


def build_receipt_schema_contract() -> dict[str, Any]:
    return {
        "schema": "hrm_text_158_fold3b_mechanism_diagnosis_receipt/v1",
        "required_fields": [
            "sampled_state_set",
            "sampled_state_order",
            "order_rank_by_semantic_state",
            "semantic_state_id",
            "per_state",
            "dedup_reset_called",
            "dedup_session_scope",
            "wrapper_path",
            "primary_receipt_path",
            "fallback_receipt_path",
            "science_verdict_source",
            "parent_sha",
            "git_head_required",
            "variable_id",
            "control_reason",
            "f3b_branch",
            "f3b_branch_inputs",
            "ready_for_main_science",
            "counts_as_sub2",
            "pre_full_stack_diagnostic",
        ],
        "per_state_required_fields": [
            "state_index",
            "crossing_indices_len",
            "crossing_count",
            "mark_count",
        ],
    }


def build_prereg_packet() -> dict[str, Any]:
    return {
        "schema": "hrm_text_158_fold3b_step1_prereg_packet/v1",
        "task_id": ACTIVE_TASK_ID,
        "fold": "fold_3b_why_state0_mechanism_diagnosis_step1_prereg",
        "git_head_required": HEAD,
        "provenance": {
            "dispatch_msg_id": DISPATCH_MSG_ID,
            "plan_msg_id": PLAN_MSG_ID,
            "implement_gate_msg_id": IMPLEMENT_GATE_MSG_ID,
            "gabe_auq_relay_msg_id": "1783248063651",
            "co_lead_design_msg_id": "1783248414572",
            "claude_converge_msg_id": "1783248525523",
        },
        "classifier": "F3B_WHY_STATE0_BRANCH_V1",
        "branch_enum": build_branch_enum(),
        "receipt_schema": build_receipt_schema_contract(),
        "feasibility_preflight": {
            "verdict": "NEEDS_PATCH",
            "summary": (
                "No order-controllable surface exists today; frozenset drops env order; "
                "enumerate(tensor_results) fixes measurement order."
            ),
        },
        "run_determinism": build_gpu_ladder()["run_determinism"],
        "identity_order_inertness_precondition": build_identity_order_inertness_precondition(),
        "order_control_patch_scope": build_order_control_patch_scope(),
        "gpu_ladder": build_gpu_ladder(),
        "sub2_first_launch_gate_exception": {
            "pre_full_stack_diagnostic": True,
            "ready_for_main_science": False,
            "counts_as_sub2": False,
            "diagnostic_reason": (
                "diagnose whether state0-only crossing support is semantic state identity, "
                "measurement order, sampled-set eligibility, or marking/dedup artifact "
                "before investing in reduction/full-stack activation."
            ),
            "cheaper_than_full_stack": (
                "uses existing dense confirmation path + tiny order/provenance control"
            ),
            "promotion_gate": (
                "diagnostic result prevents promotion to main science until separately gated"
            ),
        },
        "claim_boundary": {
            "allowed_claim_verbatim": (
                "Within the Fold-3B packet scope, state0-only crossing support classifies "
                "as one of the pre-registered branches."
            ),
            "anti_overclaim_verbatim": ANTI_OVERCLAIM_VERBATIM,
            "forbidden": [
                "candidate-C",
                "CA/reduction eligibility",
                "W/P",
                "~430MB bank pin",
                "universal all-state census",
                "bank mutation",
                "sub-2 readiness",
                "full-stack readiness",
                "implementation readiness",
            ],
        },
        "upstream_grounding": {
            "dense_primary_receipt_path": DENSE_PRIMARY_RECEIPT,
            "dense_wrapper_receipt_path": DENSE_WRAPPER_RECEIPT,
            "fold3a_receipt_path": FOLD3A_RECEIPT,
            "dense_fork_closeout_path": DENSE_FORK_CLOSEOUT,
        },
        "step1_scope": {
            "design_preflight_only": True,
            "no_gpu_launch": True,
            "no_order_patch_build": True,
            "no_live_classifier_wiring": True,
        },
    }


def build_preflight_receipt(packet: dict[str, Any]) -> dict[str, Any]:
    fold3a = load_json_if_exists(REPO / FOLD3A_RECEIPT) or {}
    dense_fork = load_json_if_exists(REPO / DENSE_FORK_CLOSEOUT) or {}
    dense_primary: dict[str, Any] = {}
    dense_path = Path(DENSE_PRIMARY_RECEIPT)
    if dense_path.is_file():
        dense_primary = json.loads(dense_path.read_text(encoding="utf-8"))

    return {
        "schema": "hrm_text_158_fold3b_step1_feasibility_preflight_receipt/v1",
        "task_id": ACTIVE_TASK_ID,
        "fold": "fold_3b_step1_feasibility_preflight",
        "git_head_at_preflight": git_head(),
        "design_spec_path": str(DRAFT.relative_to(REPO)),
        "provenance": packet["provenance"],
        "feasibility_verdict": "NEEDS_PATCH",
        "run_determinism_classification": "DETERMINISTIC",
        "source_trace_anchors": SOURCE_TRACE_ANCHORS,
        "identity_order_inertness_precondition": packet["identity_order_inertness_precondition"],
        "order_control_patch_scope": packet["order_control_patch_scope"],
        "anti_overclaim_verbatim": ANTI_OVERCLAIM_VERBATIM,
        "upstream_receipt_grounding": {
            "dense_primary": {
                "receipt_path": DENSE_PRIMARY_RECEIPT,
                "terminal_branch": dense_primary.get("terminal_branch"),
                "cb_state_count": dense_primary.get("cb_state_count"),
                "mark_count": dense_primary.get("mark_count"),
                "sampled_states": dense_primary.get("sampled_states"),
                "eligible_module_limit": dense_primary.get("eligible_module_limit"),
                "peak_rss_gib": dense_primary.get("peak_rss_gib"),
                "state0_crossing_indices_len": next(
                    (
                        int(row.get("crossing_indices_len") or 0)
                        for row in dense_primary.get("per_state") or []
                        if int(row.get("state_index", -1)) == 0
                    ),
                    None,
                ),
            },
            "fold3a_closeout": {
                "receipt_path": FOLD3A_RECEIPT,
                "git_head_at_closeout": fold3a.get("git_head_at_closeout"),
                "terminal_branch": (
                    (fold3a.get("dominance_result") or {}).get("terminal_branch")
                ),
                "single_cb_support": (
                    (fold3a.get("dominance_result") or {}).get("single_cb_support")
                ),
            },
            "dense_fork_closeout": {
                "receipt_path": DENSE_FORK_CLOSEOUT,
                "fork_resolution_status": (
                    (dense_fork.get("fork_resolution") or {}).get("status")
                ),
            },
        },
        "determinism_evidence": {
            "classification": "DETERMINISTIC",
            "static": [
                "fixed parent checkpoint in dense decider packet",
                "sorted tensor_states/inputs in bounded_delta_learner",
                "no random/torch.rand in slice5 CA path",
            ],
            "empirical_upgrade_path": (
                "identity-order inertness GPU run in patch slice must match dense primary"
            ),
            "n50_equivalent": (
                "exact baseline-vs-perturbed control-pair contrast per arm"
            ),
        },
    }


def verify_identity_inertness_precondition(packet: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    block = packet.get("identity_order_inertness_precondition")
    if not isinstance(block, dict):
        return ["identity_order_inertness_precondition_missing"]
    if block.get("blocks_variable_a_interpretation") is not True:
        failures.append("blocks_variable_a_not_true")
    if block.get("identity_order") != IDENTITY_ORDER:
        failures.append("identity_order_mismatch")
    if not block.get("baseline_comparison_receipt_path"):
        failures.append("baseline_comparison_receipt_path_missing")
    if block.get("counts_against_fold3b_gpu_budget") is not False:
        failures.append("counts_against_fold3b_budget_not_false")
    if not block.get("dual_purpose"):
        failures.append("dual_purpose_note_missing")
    return failures


def verify_gpu_ladder_budget(packet: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    ladder = packet.get("gpu_ladder") or {}
    budget = ladder.get("budget") or {}
    if budget.get("identity_inertness_in_fold3b_budget") is not False:
        failures.append("identity_inertness_in_fold3b_budget_not_false")
    variables = ladder.get("variables") or []
    var_a = next((v for v in variables if v.get("variable_id") == "A_order_only"), None)
    if var_a is None:
        failures.append("variable_a_missing")
    elif var_a.get("blocked_until") != "IDENTITY_ORDER_INERTNESS_PRECONDITION":
        failures.append("variable_a_blocked_until_missing")
    return failures


def self_verify() -> dict[str, Any]:
    from calm.hrm_text_158.native_full_stack.f3b_why_state0_branch import (
        validate_preflight_receipt_schema,
        validate_prereg_packet_schema,
    )

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
    failures.extend(verify_identity_inertness_precondition(packet))
    failures.extend(verify_gpu_ladder_budget(packet))

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
