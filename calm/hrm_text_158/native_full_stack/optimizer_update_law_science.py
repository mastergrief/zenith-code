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
) -> str | None:
    if str(mode) == SCIENCE_MODE_PRETERMINAL_SCREEN:
        return None
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
    "TIE_POLICY_CURRENT_MARGIN_INDEX",
    "TIE_POLICY_DETERMINISTIC_HASH_MATCHED",
    "build_optimizer_update_law_science_packet",
    "classify_optimizer_update_law_branch",
    "default_control_parity_gate",
    "default_hash_gate_policy",
    "default_science_arms",
    "default_verdict_rule",
    "packet_without_runtime_results",
    "validate_optimizer_update_law_science_packet",
]
