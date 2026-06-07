"""Pre-full-stack diagnostic packet contract for optimizer/update-law science.

This module is not a readiness row and not a launch runner. It encodes the
Step-1 authoring packet for the rank-bucket vs rank-free sign-pressure
diagnostic, including the ordering-matched A1 control that isolates the
vote-law variable from tie/order effects.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence


OPTIMIZER_UPDATE_LAW_SCIENCE_SCHEMA_VERSION = (
    "hrm_text_158_optimizer_update_law_science/v0.pre_full_stack_diagnostic"
)
OPTIMIZER_UPDATE_LAW_SCIENCE_TARGET_NAME = "step1_optimizer_update_law_science_packet"
DIAGNOSTIC_CLASS_PRE_FULL_STACK = "pre_full_stack_diagnostic"
STEP1_DRY_RUN_PACKET_KIND = "step1_dry_run_packet"
STEP2_LAUNCH_BUNDLE_PACKET_KIND = "step2_launch_bundle"

SCIENCE_MODE_PRETERMINAL_SCREEN = "preterminal_screen"
SCIENCE_MODE_BRANCH_VERDICT = "branch_verdict"
SCIENCE_MODE_ROWS = {
    SCIENCE_MODE_PRETERMINAL_SCREEN: 20,
    SCIENCE_MODE_BRANCH_VERDICT: 50,
}

ARM_A0_RANK_BUCKET_CURRENT = "A0_rank_bucket_current_ordering"
ARM_A1_RANK_BUCKET_ORDER_MATCHED = "A1_rank_bucket_order_matched"
ARM_B_RANK_FREE_SIGN_PRESSURE = "B_rank_free_sign_pressure"
ARM_INVERTED_SIGN_PRESSURE = "inverted_sign_pressure"
SCIENCE_ARM_IDS = (
    ARM_A0_RANK_BUCKET_CURRENT,
    ARM_A1_RANK_BUCKET_ORDER_MATCHED,
    ARM_B_RANK_FREE_SIGN_PRESSURE,
    ARM_INVERTED_SIGN_PRESSURE,
)

TIE_POLICY_CURRENT_MARGIN_INDEX = "current_abs_new_acc_then_index"
TIE_POLICY_DETERMINISTIC_HASH_MATCHED = "deterministic_hash_shuffle_order_matched"
TIE_POLICY_SCREEN_ONLY_IF_A1_MISSING = "screen_only_if_a1_missing"
FIXED_RANK_BUCKET_NON_TARGET_AUX = "fixed_rank_bucket_non_target_aux"

CONTROL_PARITY_FRACTION_MIN = 0.15
CONTROL_PARITY_FRACTION_MAX = 0.45

BRANCH_RANK_FREE_POSITIVE = "rank_free_sign_pressure_positive"
BRANCH_RANKING_STILL_REQUIRED = "ranking_still_required_or_prior_null_artifact"
BRANCH_DIRECTION_PROJECTION_WRONG = "direction_projection_wrong"
BRANCH_TIE_POLICY_OR_OVERUPDATE = "tie_policy_or_overupdate_limited"
BRANCH_CREDIT_SOURCE_NOT_SUFFICIENT = "credit_source_not_sufficient"
BRANCH_INSUFFICIENT_SEPARATION = "insufficient_separation"
BRANCH_PRIOR_NULL_SETUP_UNVERIFIED = "prior_null_setup_unverified"
OPTIMIZER_UPDATE_LAW_BRANCHES = (
    BRANCH_RANK_FREE_POSITIVE,
    BRANCH_RANKING_STILL_REQUIRED,
    BRANCH_DIRECTION_PROJECTION_WRONG,
    BRANCH_TIE_POLICY_OR_OVERUPDATE,
    BRANCH_CREDIT_SOURCE_NOT_SUFFICIENT,
    BRANCH_INSUFFICIENT_SEPARATION,
)

_READINESS_FORBIDDEN_TRUE_FIELDS = (
    "readiness_claim",
    "full_sub2_claim",
    "full_sub2_runtime_readiness_claim",
    "ready_for_main_science",
    "checkpoint_promotion_claim",
    "optimizer_credit_state_row_flip",
    "readiness_row_flip_authorized",
)
_RAW_ARRAY_KEY_FRAGMENTS = (
    "raw_per_proposal",
    "per_proposal_array",
    "raw_proposal_array",
)
_AUTHOR_PACKET_RUNTIME_RESULT_FIELDS = (
    "runtime_results",
    "arm_metrics",
    "branch_classification",
    "terminal_receipt",
    "command_outputs",
)
_COMMAND_REQUIRED_FIELDS = (
    "cwd",
    "env",
    "argv",
    "stdout_path",
    "stderr_path",
    "receipt_path",
    "scratch_root",
    "expected_exit_policy",
)
_PHASE_BUDGET_KEYS = (
    "forward_backward",
    "vote_gen_update",
    "emission_accounting",
    "artifact_flush",
)


def _path_join(root: str | Path, *parts: str) -> str:
    return str(Path(root).joinpath(*parts))


def default_science_arms(*, include_inverted: bool = True) -> list[dict[str, Any]]:
    arms = [
        {
            "arm_id": ARM_A0_RANK_BUCKET_CURRENT,
            "vote_law": "current_rank_bucket",
            "ordering_role": "control_parity_calibrator",
            "tie_policy_id": TIE_POLICY_CURRENT_MARGIN_INDEX,
            "required": True,
        },
        {
            "arm_id": ARM_A1_RANK_BUCKET_ORDER_MATCHED,
            "vote_law": "current_rank_bucket",
            "ordering_role": "ordering_matched_control",
            "tie_policy_id": TIE_POLICY_DETERMINISTIC_HASH_MATCHED,
            "required": True,
        },
        {
            "arm_id": ARM_B_RANK_FREE_SIGN_PRESSURE,
            "vote_law": "rank_free_sign_pressure",
            "ordering_role": "candidate_same_ordering_as_A1",
            "tie_policy_id": TIE_POLICY_DETERMINISTIC_HASH_MATCHED,
            "required": True,
        },
    ]
    if include_inverted:
        arms.append(
            {
                "arm_id": ARM_INVERTED_SIGN_PRESSURE,
                "vote_law": "inverted_rank_free_sign_pressure",
                "ordering_role": "direction_falsifier_same_ordering_as_A1",
                "tie_policy_id": TIE_POLICY_DETERMINISTIC_HASH_MATCHED,
                "required": False,
            },
        )
    return arms


def default_control_parity_gate() -> dict[str, Any]:
    return {
        "metric": "current_beats_competitors_fraction",
        "min_inclusive": CONTROL_PARITY_FRACTION_MIN,
        "max_inclusive": CONTROL_PARITY_FRACTION_MAX,
        "qualitative_prior_null_signature_required": True,
        "requires_current_improves_vs_baseline": True,
        "requires_random_matches_or_beats_current": True,
        "failure_branch": BRANCH_INSUFFICIENT_SEPARATION,
        "candidate_arms_read_on_failure": False,
    }


def default_verdict_rule() -> dict[str, Any]:
    return {
        "positive_branch": BRANCH_RANK_FREE_POSITIVE,
        "positive_requires": [
            "control_parity_gate_pass",
            "B_beats_A0",
            "B_beats_A1",
            "B_beats_falsifiers",
        ],
        "B_beats_A0_but_not_A1_branch": BRANCH_TIE_POLICY_OR_OVERUPDATE,
        "A1_helped_branch": BRANCH_TIE_POLICY_OR_OVERUPDATE,
        "A1_missing_policy": TIE_POLICY_SCREEN_ONLY_IF_A1_MISSING,
        "allowed_branches": list(OPTIMIZER_UPDATE_LAW_BRANCHES),
    }


def default_hash_gate_policy() -> dict[str, Any]:
    return {
        "gate_type": "read_only_source_banked_hash_before_after",
        "required_sources": [
            "banked_parent_checkpoint",
            "banked_initial_q_state",
        ],
        "explicitly_not_sources": [
            "live_q_after_update",
            "post_arm_live_q_mutation",
        ],
        "live_q_delta_fields_required_separately": True,
        "hash_before_after_must_match": True,
    }


def build_optimizer_update_law_science_packet(
    *,
    parent_path: str | Path,
    parent_sha256: str,
    mode: str = SCIENCE_MODE_PRETERMINAL_SCREEN,
    launch_gate_id: str | None = None,
    include_inverted: bool = True,
) -> dict[str, Any]:
    normalized_mode = str(mode)
    if normalized_mode not in SCIENCE_MODE_ROWS:
        raise ValueError(f"unsupported science packet mode {mode!r}")
    packet = {
        "schema_version": OPTIMIZER_UPDATE_LAW_SCIENCE_SCHEMA_VERSION,
        "target_name": OPTIMIZER_UPDATE_LAW_SCIENCE_TARGET_NAME,
        "artifact_role": "optimizer_update_law_science_launch_packet",
        "diagnostic_class": DIAGNOSTIC_CLASS_PRE_FULL_STACK,
        "pre_full_stack_diagnostic": True,
        "mode": normalized_mode,
        "n_rows": SCIENCE_MODE_ROWS[normalized_mode],
        "launch_gate_id": launch_gate_id,
        "parent_path": str(parent_path),
        "parent_sha256": str(parent_sha256),
        "gpu_launched": False,
        "checkpoint_written": False,
        "pt_mutated": False,
        "readiness_claim": False,
        "full_sub2_claim": False,
        "optimizer_credit_state_row_flip": False,
        "aux_vote_law": FIXED_RANK_BUCKET_NON_TARGET_AUX,
        "arms": default_science_arms(include_inverted=include_inverted),
        "control_parity_gate": default_control_parity_gate(),
        "verdict_rule": default_verdict_rule(),
        "hash_gate_policy": default_hash_gate_policy(),
        "compact_instrumentation_only": True,
        "raw_per_proposal_arrays_included": False,
        "non_claims": [
            "no readiness row flip",
            "no full-sub2 runtime claim",
            "no checkpoint promotion",
            "no GPU launch from Step-1 authoring",
            "no .pt mutation",
            "default rank-bucket trainer path unchanged",
        ],
    }
    validate_optimizer_update_law_science_packet(packet)
    return packet


def _build_probe_command_record(
    *,
    repo_root: str | Path,
    run_root: str | Path,
    parent_path: str | Path,
    parent_sha256: str,
    mode: str,
    arm_id: str,
    device: str,
    phase_timeout_seconds: int | float,
    total_timeout_seconds: int | float,
    max_silent_phase_seconds: int | float,
) -> dict[str, Any]:
    n_rows = int(SCIENCE_MODE_ROWS[str(mode)])
    scratch_root = _path_join(run_root, str(mode), str(arm_id))
    receipt_path = _path_join(scratch_root, "receipt.json")
    stdout_path = _path_join(scratch_root, "stdout.ndjson")
    stderr_path = _path_join(scratch_root, "stderr.log")
    argv = [
        "python3",
        "scripts/hrm_text_158_bounded_delta_acquisition_probe.py",
        "--enable-bounded-delta-probe",
        "--allow-gpu-launch",
        "--phase",
        f"optimizer-update-law-{mode}-{arm_id}",
        "--device",
        str(device),
        "--parent",
        str(parent_path),
        "--parent-sha256",
        str(parent_sha256),
        "--scratch-root",
        scratch_root,
        "--steps",
        str(n_rows),
        "--max-steps-hard",
        str(max(SCIENCE_MODE_ROWS.values())),
        "--audit-interval",
        str(n_rows),
        "--science-arm",
        str(arm_id),
        "--emit-progress",
        "--phase-timeout-seconds",
        str(phase_timeout_seconds),
        "--total-timeout-seconds",
        str(total_timeout_seconds),
        "--max-silent-phase-seconds",
        str(max_silent_phase_seconds),
    ]
    return {
        "mode": str(mode),
        "arm_id": str(arm_id),
        "n_rows": n_rows,
        "steps_requested": n_rows,
        "steps_source": "SCIENCE_MODE_ROWS[mode]",
        "cwd": str(repo_root),
        "env": {
            "HRM_TEXT_158_RUN_C2_ACQUISITION_PROBE": "1",
            "HRM_TEXT_158_ALLOW_C2_GPU_LAUNCH": "1",
        },
        "argv": argv,
        "stdout_path": stdout_path,
        "stderr_path": stderr_path,
        "receipt_path": receipt_path,
        "scratch_root": scratch_root,
        "expected_exit_policy": "exit_0_required_else_stop_no_retry_no_verdict",
    }


def default_phase_budgets() -> dict[str, Any]:
    return {
        "forward_backward": {
            "first_milestone_seconds": 180,
            "probe_phase_markers": ["step_forward_backward"],
        },
        "vote_gen_update": {
            "first_milestone_seconds": 120,
            "probe_phase_markers": ["step_update"],
        },
        "emission_accounting": {
            "first_milestone_seconds": 90,
            "probe_phase_markers": ["audit", "prior_final_audit", "front_c_identity_artifact"],
        },
        "artifact_flush": {
            "first_milestone_seconds": 60,
            "probe_phase_markers": ["checkpoint_payload", "receipt_write"],
        },
    }


def default_watcher_bundle() -> dict[str, Any]:
    return {
        "watcher_kind": "line_oriented_progress_monitor",
        "input_streams": ["stdout_path", "stderr_path"],
        "progress_filters": [
            "phase_telemetry",
            "bounded_steps",
            "step_forward_backward",
            "step_update",
            "receipt_write",
        ],
        "error_filters": [
            "Traceback",
            "RuntimeError",
            "CUDA out of memory",
            "nonfinite",
            "parent checkpoint hash changed",
            "phase timeout",
            "silent phase timeout",
        ],
        "heartbeat_policy": {
            "source": "phase_telemetry",
            "stall_decision": "phase_budget_not_total_timeout",
        },
        "terminal_receipt_required": True,
    }


def default_screen_before_verdict_dependency() -> dict[str, Any]:
    return {
        "screen_mode": SCIENCE_MODE_PRETERMINAL_SCREEN,
        "verdict_mode": SCIENCE_MODE_BRANCH_VERDICT,
        "verdict_blocked_until": {
            "screen_terminal_receipt_pass": True,
            "parent_hash_unchanged": True,
            "all_four_arms_completed": True,
            "no_nonfinite": True,
            "no_cuda_oom": True,
            "no_timeout": True,
            "pt_mutated": False,
            "qualitative_control_parity_screen_not": BRANCH_INSUFFICIENT_SEPARATION,
            "prior_null_setup_verified": True,
        },
    }


def default_terminal_criteria() -> dict[str, Any]:
    return {
        "branch_classifier": list(OPTIMIZER_UPDATE_LAW_BRANCHES),
        "verdict_rule": default_verdict_rule(),
        "control_parity_gate": default_control_parity_gate(),
        "hash_gate_policy": default_hash_gate_policy(),
        "prior_null_setup_gate": {
            "verified_required_before_parity_band": True,
            "unverified_branch": BRANCH_PRIOR_NULL_SETUP_UNVERIFIED,
            "fallback_branch": BRANCH_INSUFFICIENT_SEPARATION,
            "blocks_control_parity_band": True,
        },
        "live_q_delta_fields_required_separately": True,
        "no_verdict_outside_prereg": True,
    }


def default_resource_lane_contract(*, symbolic_lane: str = "gpu:0") -> dict[str, Any]:
    return {
        "resource_lane_required": True,
        "lane_name": str(symbolic_lane),
        "resolved_at_launch": False,
        "acquire_at_launch": True,
        "release_on_terminal_receipt": True,
        "conflict_check_required": True,
        "author_packet_does_not_acquire": True,
    }


def default_prior_verdict_parent_ref(
    *,
    parent_path: str | Path,
    parent_sha256: str,
) -> dict[str, Any]:
    return {
        "verdict_label": "credit_ranking_uninformative_update_law_pivot",
        "candidate_label": "credit_ranking_uninformative_update_law_pivot",
        "artifact_path": "/tmp/hrm158_shadow_prefix_lane3_hybrid_gpu_n50_trace.tierb.final.json",
        "parent_path": str(parent_path),
        "parent_sha256": str(parent_sha256),
        "support_name": "L0c2-K2-addition-120",
        "support_sha": "21c8a2f8c15fd68571407e6d1f215ab045ffc5a2a91e4b5a44b50bcd46b6faf0",
        "row_count": 120,
        "lane": "lane3",
        "acc_mode": "applied_crossing_direction_plus_4bit_residual",
        "vote_fidelity": "dry2",
        "activation_mode": "ternary_group128_codec_from_step0",
        "rank_vote_spec_sha256": "6c109e0482292edf72d3cc4ada6bda0840e67e8dbfac4ad7fd64d353602806a5",
        "vote_mapping_family": "rank_bucketed",
        "artifact_literal_parent_path": str(parent_path),
        "artifact_literal_parent_sha256": str(parent_sha256),
        "artifact_literal_support_sha256": "21c8a2f8c15fd68571407e6d1f215ab045ffc5a2a91e4b5a44b50bcd46b6faf0",
        "artifact_literal_acc_mode": "applied_crossing_direction_plus_4bit_residual",
        "artifact_literal_activation_mode": "ternary_group128_codec_from_step0",
        "artifact_literal_vote_fidelity": "dry2",
        "artifact_literal_rank_vote_spec_sha256": "6c109e0482292edf72d3cc4ada6bda0840e67e8dbfac4ad7fd64d353602806a5",
        "artifact_literal_vote_mapping_family": "rank_bucketed",
        "artifact_literal_lane": "lane3",
        "artifact_literal_random_seed": 17,
        "artifact_literal_step_or_sample_count": 50,
        "human_readable_labels": {
            "support_sha_prefix": "21c8a2f8",
            "acc_mode_family": "hybrid",
            "vote_fidelity_family": "tierb",
            "activation_family": "shadow_prefix",
        },
        "support_seed": 17,
        "random_seed": 17,
        "step_or_sample_count": 50,
        "current_beats_competitors_fraction": 0.3,
        "prior_null_setup_verified": False,
        "verification_state": "pending_launch_terminal_receipt",
        "unverified_branch": BRANCH_PRIOR_NULL_SETUP_UNVERIFIED,
        "fallback_branch": BRANCH_INSUFFICIENT_SEPARATION,
        "blocks_control_parity_band": True,
        "verified_at_launch_required": True,
        "control_parity_self_protects": True,
    }


def build_optimizer_update_law_launch_bundle(
    *,
    parent_path: str | Path,
    parent_sha256: str,
    repo_root: str | Path,
    run_root: str | Path,
    device: str = "cuda:0",
    launch_gate_id: str | None = None,
    symbolic_resource_lane: str = "gpu:0",
    phase_timeout_seconds: int | float = 1800,
    total_timeout_seconds: int | float = 14400,
    max_silent_phase_seconds: int | float = 300,
) -> dict[str, Any]:
    commands = [
        _build_probe_command_record(
            repo_root=repo_root,
            run_root=run_root,
            parent_path=parent_path,
            parent_sha256=parent_sha256,
            mode=mode,
            arm_id=arm_id,
            device=device,
            phase_timeout_seconds=phase_timeout_seconds,
            total_timeout_seconds=total_timeout_seconds,
            max_silent_phase_seconds=max_silent_phase_seconds,
        )
        for mode in (SCIENCE_MODE_PRETERMINAL_SCREEN, SCIENCE_MODE_BRANCH_VERDICT)
        for arm_id in SCIENCE_ARM_IDS
    ]
    bundle = {
        "schema_version": OPTIMIZER_UPDATE_LAW_SCIENCE_SCHEMA_VERSION,
        "packet_kind": STEP2_LAUNCH_BUNDLE_PACKET_KIND,
        "target_name": "step2_optimizer_update_law_gpu_launch_bundle",
        "artifact_role": "optimizer_update_law_science_gpu_launch_author_packet",
        "diagnostic_class": DIAGNOSTIC_CLASS_PRE_FULL_STACK,
        "pre_full_stack_diagnostic": True,
        "author_only": True,
        "commands_executed": False,
        "gpu_launched": False,
        "launch_gate_id": launch_gate_id,
        "pt_mutated": False,
        "readiness_claim": False,
        "full_sub2_claim": False,
        "checkpoint_written": False,
        "optimizer_credit_state_row_flip": False,
        "branch_result": None,
        "parent_path": str(parent_path),
        "parent_sha256": str(parent_sha256),
        "prior_verdict_parent_ref": default_prior_verdict_parent_ref(
            parent_path=parent_path,
            parent_sha256=parent_sha256,
        ),
        "mode_sequence": [
            SCIENCE_MODE_PRETERMINAL_SCREEN,
            SCIENCE_MODE_BRANCH_VERDICT,
        ],
        "screen_before_verdict": default_screen_before_verdict_dependency(),
        "arms": default_science_arms(include_inverted=True),
        "commands": commands,
        "resource_lane": default_resource_lane_contract(
            symbolic_lane=symbolic_resource_lane,
        ),
        "watcher_audit_bundle": default_watcher_bundle(),
        "phase_budgets": default_phase_budgets(),
        "terminal_criteria": default_terminal_criteria(),
        "hash_gate_policy": default_hash_gate_policy(),
        "compact_instrumentation_only": True,
        "raw_per_proposal_arrays_included": False,
        "artifact_policy": {
            "compact_json_ndjson_only": True,
            "raw_per_proposal_arrays": False,
            "pt_writes_allowed": False,
        },
        "non_claims": [
            "author-only launch packet",
            "no GPU launch from this packet-authoring step",
            "no resource lane acquired by this packet",
            "no .pt mutation",
            "no readiness row flip",
            "no full-sub2 runtime claim",
            "no branch verdict yet",
        ],
    }
    validate_optimizer_update_law_launch_bundle(bundle)
    return bundle


def _walk_items(value: Any) -> Sequence[Any]:
    if isinstance(value, Mapping):
        return list(value.items())
    if isinstance(value, list):
        return list(enumerate(value))
    if isinstance(value, tuple):
        return list(enumerate(value))
    return ()


def _reject_raw_arrays(value: Any, *, path: str = "packet") -> None:
    for key, child in _walk_items(value):
        key_text = str(key)
        lowered = key_text.lower()
        if any(fragment in lowered for fragment in _RAW_ARRAY_KEY_FRAGMENTS):
            if child not in (False, None, "not_included"):
                raise ValueError(f"{path}.{key_text} contains raw per-proposal arrays")
        _reject_raw_arrays(child, path=f"{path}.{key_text}")


def _require_false(packet: Mapping[str, Any], field: str) -> None:
    if bool(packet.get(field, False)):
        raise ValueError(f"{field} must be false for pre_full_stack_diagnostic packet")


def _validate_arms(arms: Sequence[Mapping[str, Any]]) -> None:
    by_id = {str(arm.get("arm_id")): dict(arm) for arm in arms}
    required = {
        ARM_A0_RANK_BUCKET_CURRENT,
        ARM_A1_RANK_BUCKET_ORDER_MATCHED,
        ARM_B_RANK_FREE_SIGN_PRESSURE,
    }
    missing = sorted(required - set(by_id))
    if missing:
        raise ValueError(f"science packet missing required arms: {missing}")
    if by_id[ARM_A0_RANK_BUCKET_CURRENT].get("tie_policy_id") != TIE_POLICY_CURRENT_MARGIN_INDEX:
        raise ValueError("A0 must keep current abs(new_acc)+index ordering")
    a1_policy = by_id[ARM_A1_RANK_BUCKET_ORDER_MATCHED].get("tie_policy_id")
    b_policy = by_id[ARM_B_RANK_FREE_SIGN_PRESSURE].get("tie_policy_id")
    if a1_policy != b_policy or a1_policy != TIE_POLICY_DETERMINISTIC_HASH_MATCHED:
        raise ValueError("A1 and B must share the deterministic ordering-matched tie policy")
    if ARM_INVERTED_SIGN_PRESSURE in by_id:
        if by_id[ARM_INVERTED_SIGN_PRESSURE].get("tie_policy_id") != a1_policy:
            raise ValueError("inverted-sign falsifier must share A1/B tie policy")


def _validate_hash_gate_policy(policy: Mapping[str, Any]) -> None:
    required = set(policy.get("required_sources") or ())
    forbidden_text = " ".join(str(item).lower() for item in policy.get("required_sources") or ())
    if "live_q" in forbidden_text or "post_arm" in forbidden_text:
        raise ValueError("hash gate cannot use live post-arm q mutation as read-only source")
    if "banked_parent_checkpoint" not in required or "banked_initial_q_state" not in required:
        raise ValueError("hash gate requires read-only banked parent and initial q-state sources")
    if not bool(policy.get("live_q_delta_fields_required_separately")):
        raise ValueError("live q deltas must be recorded separately from read-only hash gates")
    if not bool(policy.get("hash_before_after_must_match")):
        raise ValueError("read-only hash gates must assert before/after match")


def validate_optimizer_update_law_science_packet(packet: Mapping[str, Any]) -> None:
    if packet.get("schema_version") != OPTIMIZER_UPDATE_LAW_SCIENCE_SCHEMA_VERSION:
        raise ValueError("unsupported optimizer update-law science schema")
    if packet.get("diagnostic_class") != DIAGNOSTIC_CLASS_PRE_FULL_STACK:
        raise ValueError("packet must be pre_full_stack_diagnostic")
    if packet.get("launch_gate_id", "missing") is not None:
        raise ValueError("Step-1 dry-run packet must carry launch_gate_id=null")
    mode = str(packet.get("mode", ""))
    if mode not in SCIENCE_MODE_ROWS:
        raise ValueError(f"unsupported science mode {mode!r}")
    if int(packet.get("n_rows", -1)) != SCIENCE_MODE_ROWS[mode]:
        raise ValueError(f"{mode} must use N={SCIENCE_MODE_ROWS[mode]}")
    for field in _READINESS_FORBIDDEN_TRUE_FIELDS:
        _require_false(packet, field)
    if packet.get("aux_vote_law") != FIXED_RANK_BUCKET_NON_TARGET_AUX:
        raise ValueError("aux_vote_law must be fixed_rank_bucket_non_target_aux")
    _validate_arms(packet.get("arms") or ())
    gate = packet.get("control_parity_gate") or {}
    if float(gate.get("min_inclusive", -1.0)) != CONTROL_PARITY_FRACTION_MIN:
        raise ValueError("control parity min tolerance must be pinned to 0.15")
    if float(gate.get("max_inclusive", -1.0)) != CONTROL_PARITY_FRACTION_MAX:
        raise ValueError("control parity max tolerance must be pinned to 0.45")
    if not bool(gate.get("qualitative_prior_null_signature_required")):
        raise ValueError("qualitative prior-null signature is required")
    if not bool(gate.get("requires_current_improves_vs_baseline")):
        raise ValueError("control parity requires current improves vs baseline")
    if not bool(gate.get("requires_random_matches_or_beats_current")):
        raise ValueError("control parity requires random matches or beats current")
    _validate_hash_gate_policy(packet.get("hash_gate_policy") or {})
    verdict = packet.get("verdict_rule") or {}
    if verdict.get("positive_branch") != BRANCH_RANK_FREE_POSITIVE:
        raise ValueError("positive branch must be rank_free_sign_pressure_positive")
    if verdict.get("B_beats_A0_but_not_A1_branch") != BRANCH_TIE_POLICY_OR_OVERUPDATE:
        raise ValueError("B beats A0 but not A1 must classify tie/order limited")
    if verdict.get("A1_missing_policy") != TIE_POLICY_SCREEN_ONLY_IF_A1_MISSING:
        raise ValueError("A1 missing must demote to screen only")
    _reject_raw_arrays(packet)


def classify_optimizer_update_law_branch(
    *,
    mode: str,
    control_parity_pass: bool,
    b_beats_a0: bool,
    b_beats_a1: bool,
    b_beats_falsifiers: bool,
    a1_helped: bool = False,
    current_rank_bucket_beats_b: bool = False,
    inverted_beats_candidate: bool = False,
    any_arm_improves: bool = True,
    prior_null_setup_verified: bool = True,
) -> str | None:
    if str(mode) == SCIENCE_MODE_PRETERMINAL_SCREEN:
        return None
    if not bool(prior_null_setup_verified):
        return BRANCH_INSUFFICIENT_SEPARATION
    if not bool(control_parity_pass):
        return BRANCH_INSUFFICIENT_SEPARATION
    if bool(inverted_beats_candidate):
        return BRANCH_DIRECTION_PROJECTION_WRONG
    if bool(a1_helped) or (bool(b_beats_a0) and not bool(b_beats_a1)):
        return BRANCH_TIE_POLICY_OR_OVERUPDATE
    if bool(b_beats_a0) and bool(b_beats_a1) and bool(b_beats_falsifiers):
        return BRANCH_RANK_FREE_POSITIVE
    if bool(current_rank_bucket_beats_b):
        return BRANCH_RANKING_STILL_REQUIRED
    if not bool(any_arm_improves):
        return BRANCH_CREDIT_SOURCE_NOT_SUFFICIENT
    return BRANCH_INSUFFICIENT_SEPARATION


def _validate_author_only_fields(packet: Mapping[str, Any]) -> None:
    required_values = {
        "packet_kind": STEP2_LAUNCH_BUNDLE_PACKET_KIND,
        "author_only": True,
        "commands_executed": False,
        "gpu_launched": False,
        "launch_gate_id": None,
        "pt_mutated": False,
        "readiness_claim": False,
        "branch_result": None,
    }
    for field, expected in required_values.items():
        if packet.get(field) != expected:
            raise ValueError(f"{field} must be {expected!r} for author-only launch bundle")
    for field in _AUTHOR_PACKET_RUNTIME_RESULT_FIELDS:
        if field in packet:
            raise ValueError(f"author-only launch bundle rejects runtime field {field}")
    if bool(packet.get("full_sub2_claim", False)):
        raise ValueError("full_sub2_claim must be false for author-only launch bundle")
    if bool(packet.get("checkpoint_written", False)):
        raise ValueError("checkpoint_written must be false for author-only launch bundle")
    if bool(packet.get("optimizer_credit_state_row_flip", False)):
        raise ValueError("optimizer_credit_state_row_flip must be false for author-only launch bundle")


def _validate_command_record(command: Mapping[str, Any]) -> None:
    missing = [field for field in _COMMAND_REQUIRED_FIELDS if field not in command]
    if missing:
        raise ValueError(f"command record missing required fields: {missing}")
    mode = str(command.get("mode"))
    arm_id = str(command.get("arm_id"))
    if mode not in SCIENCE_MODE_ROWS:
        raise ValueError(f"command record has unsupported mode {mode!r}")
    if arm_id not in SCIENCE_ARM_IDS:
        raise ValueError(f"command record has unsupported arm_id {arm_id!r}")
    n_rows = int(command.get("n_rows", -1))
    steps_requested = int(command.get("steps_requested", -2))
    if n_rows != SCIENCE_MODE_ROWS[mode]:
        raise ValueError(f"command n_rows must equal SCIENCE_MODE_ROWS[{mode!r}]")
    if steps_requested != n_rows:
        raise ValueError("command steps_requested must equal n_rows explicitly")
    if command.get("steps_source") != "SCIENCE_MODE_ROWS[mode]":
        raise ValueError("command steps_source must document SCIENCE_MODE_ROWS[mode]")
    env = command.get("env")
    if not isinstance(env, Mapping):
        raise ValueError("command env must be a mapping")
    if env.get("HRM_TEXT_158_RUN_C2_ACQUISITION_PROBE") != "1":
        raise ValueError("command env missing HRM_TEXT_158_RUN_C2_ACQUISITION_PROBE=1")
    if env.get("HRM_TEXT_158_ALLOW_C2_GPU_LAUNCH") != "1":
        raise ValueError("command env missing HRM_TEXT_158_ALLOW_C2_GPU_LAUNCH=1")
    argv = command.get("argv")
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
        raise ValueError("command argv must be a list[str]")
    required_args = {
        "--enable-bounded-delta-probe",
        "--allow-gpu-launch",
        "--device",
        "--parent",
        "--parent-sha256",
        "--scratch-root",
        "--steps",
        "--max-steps-hard",
        "--audit-interval",
        "--science-arm",
        "--emit-progress",
    }
    if not required_args.issubset(set(argv)):
        raise ValueError("command argv missing required probe launch arguments")
    expected_flag_values = (
        ("--science-arm", arm_id),
        ("--steps", str(steps_requested)),
        ("--max-steps-hard", str(max(SCIENCE_MODE_ROWS.values()))),
    )
    for flag, expected in expected_flag_values:
        try:
            observed = argv[argv.index(flag) + 1]
        except (ValueError, IndexError) as exc:
            raise ValueError(f"command argv missing {flag} value") from exc
        if observed != expected:
            raise ValueError(f"command argv {flag} must be {expected!r}, got {observed!r}")
    try:
        device = argv[argv.index("--device") + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError("command argv missing --device value") from exc
    if not device.startswith("cuda:"):
        raise ValueError("command argv --device must target CUDA for launch bundle")
    for path_field in ("stdout_path", "stderr_path", "receipt_path", "scratch_root"):
        value = str(command.get(path_field))
        if not value:
            raise ValueError(f"command {path_field} must be non-empty")
        if value.endswith(".pt"):
            raise ValueError(f"command {path_field} cannot target .pt artifacts")
    if command.get("expected_exit_policy") != "exit_0_required_else_stop_no_retry_no_verdict":
        raise ValueError("command expected_exit_policy must fail closed")


def _validate_resource_lane(resource_lane: Mapping[str, Any]) -> None:
    if not bool(resource_lane.get("resource_lane_required")):
        raise ValueError("resource_lane_required must be true")
    if not str(resource_lane.get("lane_name", "")):
        raise ValueError("resource lane must name a symbolic lane")
    if bool(resource_lane.get("resolved_at_launch")):
        raise ValueError("author packet cannot mark resource lane resolved")
    if not bool(resource_lane.get("acquire_at_launch")):
        raise ValueError("resource lane must be acquired at launch")
    if not bool(resource_lane.get("release_on_terminal_receipt")):
        raise ValueError("resource lane must release on terminal receipt")
    if not bool(resource_lane.get("author_packet_does_not_acquire")):
        raise ValueError("author packet must not acquire resource lane")


def _validate_screen_before_verdict(packet: Mapping[str, Any]) -> None:
    if packet.get("mode_sequence") != [SCIENCE_MODE_PRETERMINAL_SCREEN, SCIENCE_MODE_BRANCH_VERDICT]:
        raise ValueError("mode_sequence must enforce screen before branch verdict")
    dependency = packet.get("screen_before_verdict") or {}
    blocked = dependency.get("verdict_blocked_until") or {}
    if dependency.get("screen_mode") != SCIENCE_MODE_PRETERMINAL_SCREEN:
        raise ValueError("screen_before_verdict missing preterminal screen mode")
    if dependency.get("verdict_mode") != SCIENCE_MODE_BRANCH_VERDICT:
        raise ValueError("screen_before_verdict missing branch verdict mode")
    required_blockers = {
        "screen_terminal_receipt_pass": True,
        "parent_hash_unchanged": True,
        "all_four_arms_completed": True,
        "no_nonfinite": True,
        "no_cuda_oom": True,
        "no_timeout": True,
        "pt_mutated": False,
        "qualitative_control_parity_screen_not": BRANCH_INSUFFICIENT_SEPARATION,
        "prior_null_setup_verified": True,
    }
    for field, expected in required_blockers.items():
        if blocked.get(field) != expected:
            raise ValueError(f"branch verdict missing blocked_until {field}={expected!r}")


def _validate_phase_budgets(phase_budgets: Mapping[str, Any]) -> None:
    missing = [key for key in _PHASE_BUDGET_KEYS if key not in phase_budgets]
    if missing:
        raise ValueError(f"phase_budgets missing required first-milestone budgets: {missing}")
    for key in _PHASE_BUDGET_KEYS:
        budget = phase_budgets[key]
        if int(budget.get("first_milestone_seconds", 0)) <= 0:
            raise ValueError(f"phase budget {key} must have positive first_milestone_seconds")
        if not budget.get("probe_phase_markers"):
            raise ValueError(f"phase budget {key} must name probe phase markers")


def _validate_author_hash_gates(packet: Mapping[str, Any]) -> None:
    _validate_hash_gate_policy(packet.get("hash_gate_policy") or {})
    terminal = packet.get("terminal_criteria") or {}
    _validate_hash_gate_policy(terminal.get("hash_gate_policy") or {})
    if not bool(terminal.get("live_q_delta_fields_required_separately")):
        raise ValueError("terminal criteria must keep live-q deltas separate")
    prior_gate = terminal.get("prior_null_setup_gate") or {}
    required_prior_gate = {
        "verified_required_before_parity_band": True,
        "unverified_branch": BRANCH_PRIOR_NULL_SETUP_UNVERIFIED,
        "fallback_branch": BRANCH_INSUFFICIENT_SEPARATION,
        "blocks_control_parity_band": True,
    }
    for field, expected in required_prior_gate.items():
        if prior_gate.get(field) != expected:
            raise ValueError(f"terminal prior-null gate must carry {field}={expected!r}")
    prior = packet.get("prior_verdict_parent_ref") or {}
    if str(prior.get("parent_sha256")) != str(packet.get("parent_sha256")):
        raise ValueError("prior verdict parent sha must match launch parent sha")
    if str(prior.get("parent_path")) != str(packet.get("parent_path")):
        raise ValueError("prior verdict parent path must match launch parent path")
    if str(prior.get("artifact_literal_parent_sha256")) != str(packet.get("parent_sha256")):
        raise ValueError("artifact literal parent sha must match launch parent sha")
    if str(prior.get("artifact_literal_parent_path")) != str(packet.get("parent_path")):
        raise ValueError("artifact literal parent path must match launch parent path")
    expected_prior = {
        "verdict_label": "credit_ranking_uninformative_update_law_pivot",
        "candidate_label": "credit_ranking_uninformative_update_law_pivot",
        "artifact_path": "/tmp/hrm158_shadow_prefix_lane3_hybrid_gpu_n50_trace.tierb.final.json",
        "support_name": "L0c2-K2-addition-120",
        "support_sha": "21c8a2f8c15fd68571407e6d1f215ab045ffc5a2a91e4b5a44b50bcd46b6faf0",
        "row_count": 120,
        "lane": "lane3",
        "acc_mode": "applied_crossing_direction_plus_4bit_residual",
        "vote_fidelity": "dry2",
        "activation_mode": "ternary_group128_codec_from_step0",
        "rank_vote_spec_sha256": "6c109e0482292edf72d3cc4ada6bda0840e67e8dbfac4ad7fd64d353602806a5",
        "vote_mapping_family": "rank_bucketed",
        "artifact_literal_support_sha256": "21c8a2f8c15fd68571407e6d1f215ab045ffc5a2a91e4b5a44b50bcd46b6faf0",
        "artifact_literal_acc_mode": "applied_crossing_direction_plus_4bit_residual",
        "artifact_literal_activation_mode": "ternary_group128_codec_from_step0",
        "artifact_literal_vote_fidelity": "dry2",
        "artifact_literal_rank_vote_spec_sha256": "6c109e0482292edf72d3cc4ada6bda0840e67e8dbfac4ad7fd64d353602806a5",
        "artifact_literal_vote_mapping_family": "rank_bucketed",
        "artifact_literal_lane": "lane3",
        "artifact_literal_random_seed": 17,
        "artifact_literal_step_or_sample_count": 50,
        "support_seed": 17,
        "random_seed": 17,
        "step_or_sample_count": 50,
        "current_beats_competitors_fraction": 0.3,
        "prior_null_setup_verified": False,
        "verification_state": "pending_launch_terminal_receipt",
        "unverified_branch": BRANCH_PRIOR_NULL_SETUP_UNVERIFIED,
        "fallback_branch": BRANCH_INSUFFICIENT_SEPARATION,
        "blocks_control_parity_band": True,
        "verified_at_launch_required": True,
        "control_parity_self_protects": True,
    }
    for field, expected in expected_prior.items():
        if prior.get(field) != expected:
            raise ValueError(f"prior verdict parent ref must carry {field}={expected!r}")


def validate_optimizer_update_law_launch_bundle(packet: Mapping[str, Any]) -> None:
    if packet.get("schema_version") != OPTIMIZER_UPDATE_LAW_SCIENCE_SCHEMA_VERSION:
        raise ValueError("unsupported optimizer update-law launch bundle schema")
    if packet.get("diagnostic_class") != DIAGNOSTIC_CLASS_PRE_FULL_STACK:
        raise ValueError("launch bundle must be pre_full_stack_diagnostic")
    _validate_author_only_fields(packet)
    _validate_arms(packet.get("arms") or ())
    _reject_raw_arrays(packet)
    _validate_resource_lane(packet.get("resource_lane") or {})
    _validate_screen_before_verdict(packet)
    _validate_phase_budgets(packet.get("phase_budgets") or {})
    _validate_author_hash_gates(packet)
    terminal = packet.get("terminal_criteria") or {}
    if terminal.get("branch_classifier") != list(OPTIMIZER_UPDATE_LAW_BRANCHES):
        raise ValueError("terminal criteria must carry the preregistered branch classifier")
    if not bool(terminal.get("no_verdict_outside_prereg")):
        raise ValueError("terminal criteria must reject verdicts outside prereg")
    commands = packet.get("commands")
    if not isinstance(commands, list):
        raise ValueError("launch bundle commands must be a list")
    seen = {(str(cmd.get("mode")), str(cmd.get("arm_id"))) for cmd in commands}
    expected = {
        (mode, arm_id)
        for mode in (SCIENCE_MODE_PRETERMINAL_SCREEN, SCIENCE_MODE_BRANCH_VERDICT)
        for arm_id in SCIENCE_ARM_IDS
    }
    if seen != expected:
        raise ValueError("launch bundle must include every arm for screen and branch modes")
    for command in commands:
        _validate_command_record(command)
    artifact_policy = packet.get("artifact_policy") or {}
    if not bool(artifact_policy.get("compact_json_ndjson_only")):
        raise ValueError("artifact policy must require compact JSON/NDJSON")
    if bool(artifact_policy.get("pt_writes_allowed")):
        raise ValueError("artifact policy must reject .pt writes")


def packet_without_runtime_results(packet: Mapping[str, Any]) -> dict[str, Any]:
    out = deepcopy(dict(packet))
    out.pop("runtime_results", None)
    out.pop("arm_metrics", None)
    return out


__all__ = [
    "ARM_A0_RANK_BUCKET_CURRENT",
    "ARM_A1_RANK_BUCKET_ORDER_MATCHED",
    "ARM_B_RANK_FREE_SIGN_PRESSURE",
    "ARM_INVERTED_SIGN_PRESSURE",
    "BRANCH_CREDIT_SOURCE_NOT_SUFFICIENT",
    "BRANCH_DIRECTION_PROJECTION_WRONG",
    "BRANCH_INSUFFICIENT_SEPARATION",
    "BRANCH_PRIOR_NULL_SETUP_UNVERIFIED",
    "BRANCH_RANK_FREE_POSITIVE",
    "BRANCH_RANKING_STILL_REQUIRED",
    "BRANCH_TIE_POLICY_OR_OVERUPDATE",
    "CONTROL_PARITY_FRACTION_MAX",
    "CONTROL_PARITY_FRACTION_MIN",
    "DIAGNOSTIC_CLASS_PRE_FULL_STACK",
    "FIXED_RANK_BUCKET_NON_TARGET_AUX",
    "OPTIMIZER_UPDATE_LAW_BRANCHES",
    "OPTIMIZER_UPDATE_LAW_SCIENCE_SCHEMA_VERSION",
    "SCIENCE_MODE_BRANCH_VERDICT",
    "SCIENCE_MODE_PRETERMINAL_SCREEN",
    "STEP1_DRY_RUN_PACKET_KIND",
    "STEP2_LAUNCH_BUNDLE_PACKET_KIND",
    "TIE_POLICY_CURRENT_MARGIN_INDEX",
    "TIE_POLICY_DETERMINISTIC_HASH_MATCHED",
    "build_optimizer_update_law_launch_bundle",
    "build_optimizer_update_law_science_packet",
    "classify_optimizer_update_law_branch",
    "default_control_parity_gate",
    "default_hash_gate_policy",
    "default_prior_verdict_parent_ref",
    "default_resource_lane_contract",
    "default_science_arms",
    "default_screen_before_verdict_dependency",
    "default_terminal_criteria",
    "default_verdict_rule",
    "default_watcher_bundle",
    "packet_without_runtime_results",
    "validate_optimizer_update_law_launch_bundle",
    "validate_optimizer_update_law_science_packet",
]
