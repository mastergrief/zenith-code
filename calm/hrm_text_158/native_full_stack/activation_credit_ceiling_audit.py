"""Tracked raw-vs-compressed activation-credit ceiling audit over receipt artifacts."""
from __future__ import annotations

import hashlib
import json
import math
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from calm.hrm_text_158.native_full_stack.bounded_delta_learner import file_sha256
from calm.hrm_text_158.native_full_stack.optimizer_update_law_science import (
    ACTIVATION_CREDIT_DIAG_FISHER_Q5_ABLATION_FAMILY_ID,
    ACTIVATION_CREDIT_DIAG_FISHER_Q5_FIELD,
    ACTIVATION_CREDIT_PRIMARY_FAMILY_ID,
    ACTIVATION_CREDIT_SNR_Q5_ABLATION_FAMILY_ID,
    ACTIVATION_CREDIT_SNR_Q5_FIELD,
    ACTIVATION_CREDIT_TAYLOR_BENEFIT_Q5_FIELD,
    ACTIVATION_CREDIT_TOPOLOGY_CONTROL_FAMILY_ID,
)
from calm.hrm_text_158.native_full_stack.oracle_screen_runner import (
    ACTIVATION_CREDIT_TIEBREAK_KEY_CURRENT_RANK,
    ACTIVATION_CREDIT_TIEBREAK_KEY_ORDINAL_ONLY_NO_INTRA_RANK,
    ACTIVATION_CREDIT_TIEBREAK_KEY_Q5_ELIGIBILITY_ORDINAL,
    ACTIVATION_CREDIT_TIEBREAK_KEY_RAW_ELIGIBILITY_FP,
    ACTIVATION_CREDIT_TIEBREAK_KEY_TERNARY_ELIGIBILITY_ORDINAL,
    ORACLE_SCREEN_IMPROVEMENT_EPS,
    activation_credit_family_metrics_with_tiebreak,
)


ACTIVATION_CREDIT_CEILING_AUDIT_SCHEMA_VERSION = (
    "hrm_text_158_activation_credit_ceiling_audit/v0"
)
ACTIVATION_CREDIT_CEILING_AUDIT_TARGET_NAME = "activation_credit_ceiling_audit"
ACTIVATION_CREDIT_CEILING_AUDIT_EXPECTED_TARGET_TIE_BAND_ID = "voteabs=4|marginabs=4"
LABEL_LEAK_UPPER_BOUND_TAG = "label_leak_upper_bound_not_learner_available"
SEED43_LABEL = "seed43"
SEED29_LABEL = "seed29"
TOP_K_VALUES = (1, 3, 5)
PRIMARY_RECEIPT_FAMILY_AUC_MAX = 0.75
PRIMARY_RAW_AUC_MIN = 0.95
PRIMARY_Q5_ORDINAL_AUC_MIN = 0.85
PRIMARY_MATERIAL_AUC_DROP_MIN = 0.10
KNOWN_BRANCH4_AUC_TOLERANCE = 1e-12
SUB2_ORDINAL_SWEEP_SCHEMA_VERSION = (
    "hrm_text_158_activation_credit_sub2_ordinal_sweep/v0"
)
SUB2_TIEBREAK_SIDECAR_SWEEP_SCHEMA_VERSION = (
    "hrm_text_158_activation_credit_sub2_tiebreak_sidecar_sweep/v0"
)
SUB2_ORDINAL_SWEEP_LEVELS = (5, 4, 3, 2)
SIDECAR_PERSISTENT_LEVELS = (2, 3)
CALIBRATION_ONLINE_CANDIDATE_QUANTILE = "online_candidate_quantile"
CALIBRATION_SEED43_THRESHOLDS_OOS = "seed43_thresholds_oos_to_seed29"
SIDECAR_CALIBRATION_BUCKET_ONLINE = "within_bucket_online_quantile"
SIDECAR_CALIBRATION_SEED43_BUCKET_OOS = "seed43_bucket_thresholds_oos_to_seed29"
ORDINAL_TOP_BUCKET_RULE_ID = "ordinal_top_bucket"
ORDINAL_ARGMAX_SET_RULE_ID = "ordinal_argmax_set"
DIAGNOSTIC_FLAT_INDEX_TIE_BREAKER_ID = "diagnostic_flat_index_min"
DIAGNOSTIC_CANDIDATE_ORDER_TIE_BREAKER_ID = "diagnostic_candidate_order_min"
DIAGNOSTIC_CANDIDATE_HASH_TIE_BREAKER_ID = "diagnostic_candidate_id_hash_min"
STRICT_ADDITIVE_BUDGET_MODEL_ID = (
    "strict_dense_additive_ternary_primary_plus_sidecar_v0"
)
PRIMARY_TERNARY_PERSISTENT_BITS = math.log2(3)
PERSISTENT_BUDGET_BITS = 2.0
SIDECAR_TRANSIENT_SCALAR_IDS = (
    "taylor_benefit",
    "abs_grad_proxy",
    "signed_neg_grad_proxy_times_candidate_delta_weight",
    "snr",
    "diag_fisher",
)
SIDECAR_PERSISTENT_SCALAR_IDS = (
    "abs_grad_proxy",
    "signed_neg_grad_proxy_times_candidate_delta_weight",
    "snr",
    "diag_fisher",
)
BRANCH_REOPEN_ENCODER_DESIGN = "reopen_encoder_design"
BRANCH_CALIBRATION_FAILURE = "calibration_failure"
BRANCH_TERNARY_SUB2_UNIQUE = "ternary_sub2_ordinal_unique_success"
BRANCH_TERNARY_SUB2_SHORTLIST = (
    "ternary_sub2_ordinal_shortlist_success_needs_tiebreak"
)
BRANCH_TWO_BIT_BOUNDARY_UNIQUE = "two_bit_boundary_unique_success"
BRANCH_TWO_BIT_BOUNDARY_SHORTLIST = "two_bit_boundary_shortlist_success"
BRANCH_FLOOR_ABOVE_TWO_BIT = "ordinal_quantization_floor_above_two_bit"
BRANCH_NO_CLEAR_ORDINAL = "no_clear_ordinal_branch"
B7A_LABEL_PERSISTENT_SUB2_UNIQUE = "persistent_sub2_sidecar_unique_success"
B7A_LABEL_TRANSIENT_RESOLVER_SUCCESS = "transient_resolver_success"
B7A_LABEL_PERSISTENT_EXCEEDS_BUDGET = "persistent_sidecar_success_exceeds_budget"
B7A_LABEL_ZERO_STATE_DIAGNOSTIC_ONLY = "zero_state_diagnostic_only_success"
B7A_LABEL_NO_UNIQUE_RESOLVER = "no_unique_resolver"
B7A_LABEL_PROXY_ONLY_B7B_REQUIRED = "b7a_proxy_only_b7b_required"
BRANCH_LABEL_PRIORITY = (
    BRANCH_REOPEN_ENCODER_DESIGN,
    BRANCH_CALIBRATION_FAILURE,
    BRANCH_TERNARY_SUB2_UNIQUE,
    BRANCH_TERNARY_SUB2_SHORTLIST,
    BRANCH_TWO_BIT_BOUNDARY_UNIQUE,
    BRANCH_TWO_BIT_BOUNDARY_SHORTLIST,
    BRANCH_FLOOR_ABOVE_TWO_BIT,
    BRANCH_NO_CLEAR_ORDINAL,
)
B7A_LABEL_PRIORITY = (
    B7A_LABEL_PERSISTENT_SUB2_UNIQUE,
    B7A_LABEL_TRANSIENT_RESOLVER_SUCCESS,
    B7A_LABEL_PERSISTENT_EXCEEDS_BUDGET,
    B7A_LABEL_ZERO_STATE_DIAGNOSTIC_ONLY,
    B7A_LABEL_NO_UNIQUE_RESOLVER,
)

REQUIRED_RECEIPT_ROW_FIELDS = (
    "candidate_id",
    "local_loss_delta",
    "regret_vs_target_tie_band_oracle_top1_local_loss_delta",
    "in_target_tie_band",
    "activation_feature_valid",
)


@dataclass(frozen=True)
class ScalarSpec:
    scalar_id: str
    display_name: str
    category: str
    decision_authority_allowed: bool
    compute: Callable[[Mapping[str, Any]], float]
    q5_field: str | None = None
    receipt_family_id: str | None = None


@dataclass(frozen=True)
class LoadedCeilingAuditReceipt:
    seed_label: str
    path: str
    sha256: str
    branch_classification: str | None
    target_tie_band_id: str
    reported_band_candidate_count: int
    reported_valid_activation_candidate_count: int
    extracted_band_candidate_count: int
    extracted_valid_activation_candidate_count: int
    rows: tuple[dict[str, Any], ...]
    family_metrics_by_id: Mapping[str, Mapping[str, Any]]


def _float_row_field(row: Mapping[str, Any], field: str) -> float:
    value = row.get(field)
    if value is None:
        raise ValueError(f"row field {field!r} is required")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"row field {field!r} must be finite")
    return result


SCALAR_SPECS: tuple[ScalarSpec, ...] = (
    ScalarSpec(
        scalar_id="taylor_benefit",
        display_name="taylor_benefit",
        category="learner_available_capture_scalar",
        decision_authority_allowed=True,
        compute=lambda row: _float_row_field(row, "taylor_benefit"),
        q5_field=ACTIVATION_CREDIT_TAYLOR_BENEFIT_Q5_FIELD,
        receipt_family_id=ACTIVATION_CREDIT_PRIMARY_FAMILY_ID,
    ),
    ScalarSpec(
        scalar_id="abs_grad_proxy",
        display_name="abs_grad_proxy",
        category="learner_available_capture_scalar",
        decision_authority_allowed=True,
        compute=lambda row: abs(_float_row_field(row, "grad_proxy")),
    ),
    ScalarSpec(
        scalar_id="signed_neg_grad_proxy_times_candidate_delta_weight",
        display_name="signed_-grad_proxy*candidate_delta_weight",
        category="learner_available_capture_scalar",
        decision_authority_allowed=True,
        compute=lambda row: -(
            _float_row_field(row, "grad_proxy")
            * _float_row_field(row, "candidate_delta_weight")
        ),
    ),
    ScalarSpec(
        scalar_id="snr",
        display_name="snr",
        category="learner_available_capture_scalar",
        decision_authority_allowed=True,
        compute=lambda row: _float_row_field(row, "snr"),
        q5_field=ACTIVATION_CREDIT_SNR_Q5_FIELD,
        receipt_family_id=ACTIVATION_CREDIT_SNR_Q5_ABLATION_FAMILY_ID,
    ),
    ScalarSpec(
        scalar_id="diag_fisher",
        display_name="diag_fisher",
        category="learner_available_capture_scalar",
        decision_authority_allowed=True,
        compute=lambda row: _float_row_field(row, "diag_fisher"),
        q5_field=ACTIVATION_CREDIT_DIAG_FISHER_Q5_FIELD,
        receipt_family_id=ACTIVATION_CREDIT_DIAG_FISHER_Q5_ABLATION_FAMILY_ID,
    ),
    ScalarSpec(
        scalar_id="current_rank_position_ctl",
        display_name="current_rank_position(ctl)",
        category="control_scalar",
        decision_authority_allowed=True,
        compute=lambda row: _float_row_field(row, "current_rank_position"),
    ),
    ScalarSpec(
        scalar_id="current_margin_abs_ctl",
        display_name="current_margin_abs(ctl)",
        category="control_scalar",
        decision_authority_allowed=True,
        compute=lambda row: _float_row_field(row, "current_margin_abs"),
    ),
    ScalarSpec(
        scalar_id="vote_value_ctl",
        display_name="vote_value(ctl)",
        category="control_scalar",
        decision_authority_allowed=True,
        compute=lambda row: _float_row_field(row, "vote_value"),
    ),
    ScalarSpec(
        scalar_id="topology_row_block_128_ctl",
        display_name="topology_row_block_128(ctl)",
        category="control_scalar",
        decision_authority_allowed=True,
        compute=lambda row: _float_row_field(row, "topology_row_block_128"),
        receipt_family_id=ACTIVATION_CREDIT_TOPOLOGY_CONTROL_FAMILY_ID,
    ),
    ScalarSpec(
        scalar_id=f"{LABEL_LEAK_UPPER_BOUND_TAG}__neg_local_loss_delta",
        display_name="LEAK:neg_local_loss_delta",
        category=LABEL_LEAK_UPPER_BOUND_TAG,
        decision_authority_allowed=False,
        compute=lambda row: -_float_row_field(row, "local_loss_delta"),
    ),
)


KNOWN_BRANCH4_RAW_AUC_EXPECTATIONS = {
    "taylor_benefit": {
        SEED43_LABEL: 0.9950738916256158,
        SEED29_LABEL: 0.9876543209876543,
    },
    "abs_grad_proxy": {
        SEED43_LABEL: 0.9950738916256158,
        SEED29_LABEL: 0.9876543209876543,
    },
    "signed_neg_grad_proxy_times_candidate_delta_weight": {
        SEED43_LABEL: 0.9950738916256158,
        SEED29_LABEL: 0.9876543209876543,
    },
}

KNOWN_BRANCH4_RECEIPT_FAMILY_AUC_EXPECTATIONS = {
    SEED43_LABEL: 0.5812807881773399,
    SEED29_LABEL: 0.6172839506172839,
}

B5B_COUNTERFACTUAL_SCHEMA_VERSION = (
    "hrm_text_158_b5b_within_q5_family_tiebreak_counterfactual/v0"
)
B5B_TASK_ID = "1781446069451-a3af6105"
B5B_BRANCH_HARNESS_OR_INPUT_FAIL = "BRANCH_HARNESS_OR_INPUT_FAIL"
B5B_BRANCH_TIEBREAK_BASELINE_REPRO = "BRANCH_TIEBREAK_BASELINE_REPRO"
B5B_BRANCH_CALIBRATION_FAILURE = "BRANCH_CALIBRATION_FAILURE"
B5B_BRANCH_TERNARY_TIEBREAK_RECOVERS = "BRANCH_TERNARY_TIEBREAK_RECOVERS"
B5B_BRANCH_Q5_ONLY_RECOVERS = "BRANCH_Q5_ONLY_RECOVERS"
B5B_BRANCH_RAW_ONLY_RECOVERS = "BRANCH_RAW_ONLY_RECOVERS"
B5B_BRANCH_ORDINAL_ONLY_NO_INTRA_RANK = "BRANCH_ORDINAL_ONLY_NO_INTRA_RANK"
B5B_BRANCH_TIEBREAK_STILL_COLLAPSES = "BRANCH_TIEBREAK_STILL_COLLAPSES"
B5B_BRANCH_PRIORITY = (
    B5B_BRANCH_HARNESS_OR_INPUT_FAIL,
    B5B_BRANCH_TIEBREAK_BASELINE_REPRO,
    B5B_BRANCH_CALIBRATION_FAILURE,
    B5B_BRANCH_TERNARY_TIEBREAK_RECOVERS,
    B5B_BRANCH_Q5_ONLY_RECOVERS,
    B5B_BRANCH_RAW_ONLY_RECOVERS,
    B5B_BRANCH_ORDINAL_ONLY_NO_INTRA_RANK,
    B5B_BRANCH_TIEBREAK_STILL_COLLAPSES,
)
B5B_TIEBREAK_VARIANT_IDS = (
    ACTIVATION_CREDIT_TIEBREAK_KEY_CURRENT_RANK,
    ACTIVATION_CREDIT_TIEBREAK_KEY_TERNARY_ELIGIBILITY_ORDINAL,
    ACTIVATION_CREDIT_TIEBREAK_KEY_Q5_ELIGIBILITY_ORDINAL,
    ACTIVATION_CREDIT_TIEBREAK_KEY_RAW_ELIGIBILITY_FP,
    ACTIVATION_CREDIT_TIEBREAK_KEY_ORDINAL_ONLY_NO_INTRA_RANK,
)
B5B_ELIGIBILITY_SCALAR_ID = "signed_neg_grad_proxy_times_candidate_delta_weight"


def _json_load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sorted_rows_for_direction(
    rows: Sequence[Mapping[str, Any]],
    *,
    value_fn: Callable[[Mapping[str, Any]], float],
    direction: int,
) -> list[Mapping[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            -direction * value_fn(row),
            str(row["candidate_id"]),
        ),
    )


def _oracle_best_row(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    return min(
        rows,
        key=lambda row: (
            _float_row_field(
                row,
                "regret_vs_target_tie_band_oracle_top1_local_loss_delta",
            ),
            str(row["candidate_id"]),
        ),
    )


def _pairwise_auc(
    rows: Sequence[Mapping[str, Any]],
    *,
    value_fn: Callable[[Mapping[str, Any]], float],
    direction: int,
) -> float:
    wins = 0.0
    total = 0
    for left_index, left in enumerate(rows):
        left_regret = _float_row_field(
            left,
            "regret_vs_target_tie_band_oracle_top1_local_loss_delta",
        )
        left_value = direction * value_fn(left)
        for right in rows[left_index + 1 :]:
            right_regret = _float_row_field(
                right,
                "regret_vs_target_tie_band_oracle_top1_local_loss_delta",
            )
            if abs(left_regret - right_regret) <= ORACLE_SCREEN_IMPROVEMENT_EPS:
                continue
            total += 1
            right_value = direction * value_fn(right)
            if left_value == right_value:
                wins += 0.5
                continue
            if left_regret < right_regret:
                wins += 1.0 if left_value > right_value else 0.0
            else:
                wins += 1.0 if right_value > left_value else 0.0
    if total <= 0:
        return 0.5
    return float(wins / total)


def _positive_improvement_mass_mirror(
    candidates: Sequence[Mapping[str, Any]],
) -> float:
    """Mirror oracle_screen_runner._positive_improvement_mass without heavy imports."""
    return float(
        sum(max(0.0, -float(candidate["local_loss_delta"])) for candidate in candidates)
    )


def _loss_spread_ratio_mirror(
    candidates: Sequence[Mapping[str, Any]],
    *,
    oracle_top1_delta: float | None,
) -> float | None:
    """Mirror oracle_screen_runner._loss_spread_ratio without heavy imports."""
    if not candidates:
        return None
    deltas = [float(candidate["local_loss_delta"]) for candidate in candidates]
    spread = float(max(deltas) - min(deltas))
    if oracle_top1_delta is None:
        return None
    if abs(float(oracle_top1_delta)) > ORACLE_SCREEN_IMPROVEMENT_EPS:
        return float(spread / abs(float(oracle_top1_delta)))
    if spread <= ORACLE_SCREEN_IMPROVEMENT_EPS:
        return 0.0
    return None


def _candidate_ids_hash16(candidate_ids: Sequence[str]) -> str:
    encoded = "\n".join(sorted(str(candidate_id) for candidate_id in candidate_ids))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _candidate_hash_sort_key(row: Mapping[str, Any]) -> tuple[bytes, str]:
    candidate_id = str(row["candidate_id"])
    return (
        hashlib.sha256(f"b7a|candidate_id={candidate_id}".encode("utf-8")).digest(),
        candidate_id,
    )


def _scalar_spec_by_id(scalar_id: str) -> ScalarSpec:
    for spec in SCALAR_SPECS:
        if spec.scalar_id == scalar_id:
            return spec
    raise ValueError(f"unknown scalar_id {scalar_id!r}")


def _rank_order_ids(
    rows: Sequence[Mapping[str, Any]],
    *,
    value_fn: Callable[[Mapping[str, Any]], float],
    direction: int,
) -> list[str]:
    return [
        str(row["candidate_id"])
        for row in _sorted_rows_for_direction(
            rows,
            value_fn=value_fn,
            direction=int(direction),
        )
    ]


def _unique_scalar_argmax(
    rows: Sequence[Mapping[str, Any]],
    *,
    value_fn: Callable[[Mapping[str, Any]], float],
    direction: int,
) -> tuple[list[Mapping[str, Any]], float | None]:
    if not rows:
        return [], None
    scored = [
        (float(direction) * float(value_fn(row)), row)
        for row in rows
    ]
    best_score = max(score for score, _row in scored)
    selected = [row for score, row in scored if score == best_score]
    return selected, float(best_score)


def _ordinal_thresholds_from_scores(
    scores: Sequence[float],
    *,
    levels: int,
) -> tuple[float, ...]:
    if int(levels) < 2:
        raise ValueError("ordinal levels must be >= 2")
    if not scores:
        raise ValueError("cannot calibrate ordinal thresholds without scores")
    ordered = sorted(float(score) for score in scores)
    thresholds: list[float] = []
    for cut_index in range(1, int(levels)):
        cut_position = int(math.ceil(len(ordered) * cut_index / int(levels))) - 1
        cut_position = max(0, min(len(ordered) - 1, cut_position))
        thresholds.append(float(ordered[cut_position]))
    return tuple(thresholds)


def _ordinal_bin(score: float, *, thresholds: Sequence[float], levels: int) -> int:
    return min(int(levels) - 1, int(bisect_right(tuple(thresholds), float(score))))


def _score_by_candidate_id(
    rows: Sequence[Mapping[str, Any]],
    *,
    direction: int,
) -> dict[str, float]:
    return {
        str(row["candidate_id"]): float(direction) * _float_row_field(row, "taylor_benefit")
        for row in rows
    }


def _diagnostic_flat_index_tiebreak_row(
    bucket_rows: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    if not bucket_rows:
        return None
    row_position_by_id = {
        str(row["candidate_id"]): int(position)
        for position, row in enumerate(bucket_rows)
    }
    return min(
        bucket_rows,
        key=lambda row: (
            int(row["flat_index"]) if row.get("flat_index") is not None else 2**63 - 1,
            row_position_by_id[str(row["candidate_id"])],
            str(row["candidate_id"]),
        ),
    )


def _ordinal_top_bucket_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    ordinals_by_id: Mapping[str, int],
) -> list[Mapping[str, Any]]:
    max_ordinal = max(int(value) for value in ordinals_by_id.values())
    return [
        row
        for row in rows
        if int(ordinals_by_id[str(row["candidate_id"])]) == max_ordinal
    ]


def _ordinal_bucket_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    ordinals_by_id: Mapping[str, int],
    level: int,
    thresholds: Sequence[float],
) -> dict[str, Any]:
    oracle_best = _oracle_best_row(rows)
    oracle_best_id = str(oracle_best["candidate_id"])
    max_ordinal = max(int(value) for value in ordinals_by_id.values())
    top_bucket = _ordinal_top_bucket_rows(rows, ordinals_by_id=ordinals_by_id)
    top_bucket_ids = [str(row["candidate_id"]) for row in top_bucket]
    contains_oracle = oracle_best_id in set(top_bucket_ids)
    top_bucket_size = len(top_bucket)
    diagnostic_row = _diagnostic_flat_index_tiebreak_row(top_bucket)
    diagnostic_id = (
        str(diagnostic_row["candidate_id"]) if diagnostic_row is not None else None
    )
    band_improvement_mass = _positive_improvement_mass_mirror(rows)
    bucket_improvement_mass = _positive_improvement_mass_mirror(top_bucket)
    oracle_top1_delta = _float_row_field(oracle_best, "local_loss_delta")
    return {
        "level": int(level),
        "thresholds": [float(threshold) for threshold in thresholds],
        "rule_id": ORDINAL_TOP_BUCKET_RULE_ID,
        "argmax_set_rule_id": ORDINAL_ARGMAX_SET_RULE_ID,
        "auc": _pairwise_auc(
            rows,
            value_fn=lambda row: float(ordinals_by_id[str(row["candidate_id"])]),
            direction=1,
        ),
        "oracle_best_candidate_id": oracle_best_id,
        "oracle_best_ordinal": int(ordinals_by_id[oracle_best_id]),
        "max_ordinal": int(max_ordinal),
        "top_bucket_contains_oracle": bool(contains_oracle),
        "top_bucket_size": int(top_bucket_size),
        "top_bucket_fraction": float(top_bucket_size / max(1, len(rows))),
        "top_bucket_candidate_ids_hash16": _candidate_ids_hash16(top_bucket_ids),
        "top_bucket_regret_spread_ratio": _loss_spread_ratio_mirror(
            top_bucket,
            oracle_top1_delta=oracle_top1_delta,
        ),
        "top_bucket_regret_capture_ratio": (
            float(bucket_improvement_mass / band_improvement_mass)
            if band_improvement_mass > ORACLE_SCREEN_IMPROVEMENT_EPS
            else 0.0
        ),
        "unique_ordinal_top1": bool(contains_oracle and top_bucket_size == 1),
        "ready_for_gpu_wiring": False,
        "diagnostic_tiebreak": {
            "tie_breaker_id": DIAGNOSTIC_FLAT_INDEX_TIE_BREAKER_ID,
            "extra_state_bits": 0,
            "credit_mechanistic": False,
            "diagnostic_only": True,
            "primary_success_allowed": False,
            "ready_for_gpu_wiring_allowed": False,
            "selected_candidate_id": diagnostic_id,
            "deterministic_tiebreak_top1": bool(diagnostic_id == oracle_best_id),
        },
    }


def _ordinal_results_for_calibration(
    *,
    level: int,
    calibration_id: str,
    seed43_rows: Sequence[Mapping[str, Any]],
    seed29_rows: Sequence[Mapping[str, Any]],
    fixed_direction: int,
) -> dict[str, Any]:
    context = _ordinal_context_for_calibration(
        level=level,
        calibration_id=calibration_id,
        seed43_rows=seed43_rows,
        seed29_rows=seed29_rows,
        fixed_direction=fixed_direction,
    )
    seed43_metrics = _ordinal_bucket_metrics(
        seed43_rows,
        ordinals_by_id=context[SEED43_LABEL]["ordinals_by_id"],
        level=level,
        thresholds=context[SEED43_LABEL]["thresholds"],
    )
    seed29_metrics = _ordinal_bucket_metrics(
        seed29_rows,
        ordinals_by_id=context[SEED29_LABEL]["ordinals_by_id"],
        level=level,
        thresholds=context[SEED29_LABEL]["thresholds"],
    )
    return {
        "calibration_id": calibration_id,
        "seed43_threshold_source": "seed43_candidate_scores",
        "seed29_threshold_source": context["seed29_threshold_source"],
        SEED43_LABEL: seed43_metrics,
        SEED29_LABEL: seed29_metrics,
        "both_seeds": {
            "unique_ordinal_top1": bool(
                seed43_metrics["unique_ordinal_top1"]
                and seed29_metrics["unique_ordinal_top1"]
            ),
            "top_bucket_contains_oracle": bool(
                seed43_metrics["top_bucket_contains_oracle"]
                and seed29_metrics["top_bucket_contains_oracle"]
            ),
            "shortlist_success_needs_tiebreak": bool(
                seed43_metrics["top_bucket_contains_oracle"]
                and seed29_metrics["top_bucket_contains_oracle"]
                and not (
                    seed43_metrics["unique_ordinal_top1"]
                    and seed29_metrics["unique_ordinal_top1"]
                )
            ),
            "diagnostic_tiebreak_top1": bool(
                seed43_metrics["diagnostic_tiebreak"]["deterministic_tiebreak_top1"]
                and seed29_metrics["diagnostic_tiebreak"]["deterministic_tiebreak_top1"]
            ),
        },
    }


def _ordinal_context_for_calibration(
    *,
    level: int,
    calibration_id: str,
    seed43_rows: Sequence[Mapping[str, Any]],
    seed29_rows: Sequence[Mapping[str, Any]],
    fixed_direction: int,
) -> dict[str, Any]:
    seed43_scores = _score_by_candidate_id(seed43_rows, direction=fixed_direction)
    seed29_scores = _score_by_candidate_id(seed29_rows, direction=fixed_direction)
    if calibration_id == CALIBRATION_ONLINE_CANDIDATE_QUANTILE:
        seed43_thresholds = _ordinal_thresholds_from_scores(
            tuple(seed43_scores.values()),
            levels=level,
        )
        seed29_thresholds = _ordinal_thresholds_from_scores(
            tuple(seed29_scores.values()),
            levels=level,
        )
    elif calibration_id == CALIBRATION_SEED43_THRESHOLDS_OOS:
        seed43_thresholds = _ordinal_thresholds_from_scores(
            tuple(seed43_scores.values()),
            levels=level,
        )
        seed29_thresholds = seed43_thresholds
    else:
        raise ValueError(f"unsupported ordinal calibration {calibration_id!r}")
    seed43_ordinals = {
        candidate_id: _ordinal_bin(score, thresholds=seed43_thresholds, levels=level)
        for candidate_id, score in seed43_scores.items()
    }
    seed29_ordinals = {
        candidate_id: _ordinal_bin(score, thresholds=seed29_thresholds, levels=level)
        for candidate_id, score in seed29_scores.items()
    }
    return {
        "calibration_id": calibration_id,
        "seed29_threshold_source": (
            "seed29_candidate_scores"
            if calibration_id == CALIBRATION_ONLINE_CANDIDATE_QUANTILE
            else "seed43_candidate_scores"
        ),
        SEED43_LABEL: {
            "scores_by_id": seed43_scores,
            "thresholds": seed43_thresholds,
            "ordinals_by_id": seed43_ordinals,
        },
        SEED29_LABEL: {
            "scores_by_id": seed29_scores,
            "thresholds": seed29_thresholds,
            "ordinals_by_id": seed29_ordinals,
        },
    }


def _ordinal_level_success(
    level_result: Mapping[str, Any],
    *,
    unique: bool,
) -> bool:
    for calibration_id in (
        CALIBRATION_ONLINE_CANDIDATE_QUANTILE,
        CALIBRATION_SEED43_THRESHOLDS_OOS,
    ):
        both = level_result[calibration_id]["both_seeds"]
        if unique and bool(both["unique_ordinal_top1"]):
            return True
        if not unique and bool(both["top_bucket_contains_oracle"]):
            return True
    return False


def _ordinal_online_to_fixed_calibration_failure(level_result: Mapping[str, Any]) -> bool:
    online = level_result[CALIBRATION_ONLINE_CANDIDATE_QUANTILE]
    fixed = level_result[CALIBRATION_SEED43_THRESHOLDS_OOS]
    online_seed29 = online[SEED29_LABEL]
    fixed_seed29 = fixed[SEED29_LABEL]
    if bool(online_seed29["unique_ordinal_top1"]) and not bool(
        fixed_seed29["unique_ordinal_top1"]
    ):
        return True
    if bool(online_seed29["top_bucket_contains_oracle"]) and not bool(
        fixed_seed29["top_bucket_contains_oracle"]
    ):
        return True
    return False


def _ordinal_branch_labels(
    results_by_level: Mapping[str, Mapping[str, Any]],
    *,
    raw_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    labels: list[str] = []
    level3 = results_by_level["3"]
    level4 = results_by_level["4"]
    level5 = results_by_level["5"]
    any_ordinal_contains = any(
        bool(calibration[seed_label]["top_bucket_contains_oracle"])
        for level_result in results_by_level.values()
        for calibration_id in (
            CALIBRATION_ONLINE_CANDIDATE_QUANTILE,
            CALIBRATION_SEED43_THRESHOLDS_OOS,
        )
        for calibration in (level_result[calibration_id],)
        for seed_label in (SEED43_LABEL, SEED29_LABEL)
    )
    if _raw_signal_strong(raw_metrics) and not any_ordinal_contains:
        labels.append(BRANCH_REOPEN_ENCODER_DESIGN)
    if any(
        _ordinal_online_to_fixed_calibration_failure(level_result)
        for level_result in results_by_level.values()
    ):
        labels.append(BRANCH_CALIBRATION_FAILURE)
    if _ordinal_level_success(level3, unique=True):
        labels.append(BRANCH_TERNARY_SUB2_UNIQUE)
    elif _ordinal_level_success(level3, unique=False):
        labels.append(BRANCH_TERNARY_SUB2_SHORTLIST)
    level3_contains = _ordinal_level_success(level3, unique=False)
    if not level3_contains and _ordinal_level_success(level4, unique=True):
        labels.append(BRANCH_TWO_BIT_BOUNDARY_UNIQUE)
    elif not level3_contains and _ordinal_level_success(level4, unique=False):
        labels.append(BRANCH_TWO_BIT_BOUNDARY_SHORTLIST)
    level4_contains = _ordinal_level_success(level4, unique=False)
    if (
        not level3_contains
        and not level4_contains
        and _ordinal_level_success(level5, unique=False)
    ):
        labels.append(BRANCH_FLOOR_ABOVE_TWO_BIT)
    if not labels:
        labels.append(BRANCH_NO_CLEAR_ORDINAL)
    primary_label = next(label for label in BRANCH_LABEL_PRIORITY if label in labels)
    return {
        "all_applicable_labels": labels,
        "primary_label": primary_label,
        "priority_order": list(BRANCH_LABEL_PRIORITY),
        "support": {
            "raw_signal_strong": _raw_signal_strong(raw_metrics),
            "any_ordinal_top_bucket_contains_oracle": any_ordinal_contains,
            "level3_any_calibration_contains_oracle": level3_contains,
            "level4_any_calibration_contains_oracle": level4_contains,
            "calibration_failure_detected": BRANCH_CALIBRATION_FAILURE in labels,
        },
    }


def _sub2_ordinal_sweep(
    seed43_rows: Sequence[Mapping[str, Any]],
    seed29_rows: Sequence[Mapping[str, Any]],
    *,
    primary_scalar_result: Mapping[str, Any],
) -> dict[str, Any]:
    raw_metrics = primary_scalar_result["raw_continuous"]
    fixed_direction = int(raw_metrics["fixed_direction"])
    results_by_level: dict[str, dict[str, Any]] = {}
    for level in SUB2_ORDINAL_SWEEP_LEVELS:
        results_by_level[str(level)] = {
            CALIBRATION_ONLINE_CANDIDATE_QUANTILE: _ordinal_results_for_calibration(
                level=level,
                calibration_id=CALIBRATION_ONLINE_CANDIDATE_QUANTILE,
                seed43_rows=seed43_rows,
                seed29_rows=seed29_rows,
                fixed_direction=fixed_direction,
            ),
            CALIBRATION_SEED43_THRESHOLDS_OOS: _ordinal_results_for_calibration(
                level=level,
                calibration_id=CALIBRATION_SEED43_THRESHOLDS_OOS,
                seed43_rows=seed43_rows,
                seed29_rows=seed29_rows,
                fixed_direction=fixed_direction,
            ),
        }
    return {
        "schema_version": SUB2_ORDINAL_SWEEP_SCHEMA_VERSION,
        "primary_scalar_id": "taylor_benefit",
        "fixed_direction": fixed_direction,
        "fixed_direction_source": "seed43_raw_continuous",
        "levels": list(SUB2_ORDINAL_SWEEP_LEVELS),
        "calibration_classes": {
            CALIBRATION_ONLINE_CANDIDATE_QUANTILE: {
                "description": "candidate-set quantile thresholds calibrated independently per seed from learner-available scores only",
            },
            CALIBRATION_SEED43_THRESHOLDS_OOS: {
                "description": "seed43 candidate-score thresholds fixed and applied out-of-sample to seed29",
            },
        },
        "primary_rule": {
            "rule_id": ORDINAL_TOP_BUCKET_RULE_ID,
            "argmax_set_rule_id": ORDINAL_ARGMAX_SET_RULE_ID,
            "extra_state_bits": 0,
            "uses_raw_continuous_inside_bucket": False,
        },
        "diagnostic_tie_breaker": {
            "tie_breaker_id": DIAGNOSTIC_FLAT_INDEX_TIE_BREAKER_ID,
            "extra_state_bits": 0,
            "credit_mechanistic": False,
            "diagnostic_only": True,
            "primary_success_allowed": False,
            "ready_for_gpu_wiring_allowed": False,
        },
        "results": results_by_level,
        "label_decision": _ordinal_branch_labels(
            results_by_level,
            raw_metrics=raw_metrics,
        ),
        "non_claims": [
            "no raw-continuous tie-break inside ordinal buckets",
            "diagnostic flat-index tie-break is not a credit mechanism and is never primary success",
            "shortlist success is not learner/runtime/GPU wiring readiness",
        ],
    }


def _ternary_bucket_contexts(
    seed43_rows: Sequence[Mapping[str, Any]],
    seed29_rows: Sequence[Mapping[str, Any]],
    *,
    fixed_direction: int,
) -> list[dict[str, Any]]:
    contexts: list[dict[str, Any]] = []
    rows_by_seed = {
        SEED43_LABEL: seed43_rows,
        SEED29_LABEL: seed29_rows,
    }
    for calibration_id in (
        CALIBRATION_ONLINE_CANDIDATE_QUANTILE,
        CALIBRATION_SEED43_THRESHOLDS_OOS,
    ):
        ordinal_context = _ordinal_context_for_calibration(
            level=3,
            calibration_id=calibration_id,
            seed43_rows=seed43_rows,
            seed29_rows=seed29_rows,
            fixed_direction=fixed_direction,
        )
        for seed_label, rows in rows_by_seed.items():
            seed_context = ordinal_context[seed_label]
            bucket_rows = _ordinal_top_bucket_rows(
                rows,
                ordinals_by_id=seed_context["ordinals_by_id"],
            )
            oracle_best = _oracle_best_row(rows)
            contexts.append(
                {
                    "primary_calibration_id": calibration_id,
                    "seed": seed_label,
                    "rows": rows,
                    "bucket_rows": bucket_rows,
                    "bucket_candidate_ids_hash16": _candidate_ids_hash16(
                        [str(row["candidate_id"]) for row in bucket_rows],
                    ),
                    "bucket_size": len(bucket_rows),
                    "oracle_best_candidate_id": str(oracle_best["candidate_id"]),
                    "primary_taylor_order_ids": _rank_order_ids(
                        bucket_rows,
                        value_fn=_scalar_spec_by_id("taylor_benefit").compute,
                        direction=fixed_direction,
                    ),
                    "primary_thresholds": seed_context["thresholds"],
                }
            )
    return contexts


def _selected_rows_payload(
    selected_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    selected_ids = [str(row["candidate_id"]) for row in selected_rows]
    return {
        "selected_candidate_count": len(selected_rows),
        "selected_candidate_id": selected_ids[0] if len(selected_ids) == 1 else None,
        "selected_candidate_ids_hash16": _candidate_ids_hash16(selected_ids),
    }


def _base_resolver_observation(
    context: Mapping[str, Any],
    *,
    resolver_id: str,
    tier: str,
    selected_rows: Sequence[Mapping[str, Any]],
    persistent_state_bits: float,
    extra_state_bits: float,
    fp_transient: bool,
    diagnostic_only: bool,
    credit_mechanistic: bool,
    not_new_evidence: bool,
    same_scalar_as_bucket: bool,
    same_rank_as_bucket: bool,
    budget_model_id: str | None = None,
    calibration_scope: str | None = None,
    calibration_state_bits_or_shared_cost: str | None = None,
    sidecar_level: int | None = None,
    sidecar_persistent_bits: float | None = None,
    resolver_score: float | None = None,
) -> dict[str, Any]:
    selected_payload = _selected_rows_payload(selected_rows)
    oracle_id = str(context["oracle_best_candidate_id"])
    unique_selects_oracle = bool(
        selected_payload["selected_candidate_count"] == 1
        and selected_payload["selected_candidate_id"] == oracle_id
    )
    combined_persistent_bits = float(
        PRIMARY_TERNARY_PERSISTENT_BITS + persistent_state_bits
    )
    qualifies_under_persistent_budget = bool(
        combined_persistent_bits <= PERSISTENT_BUDGET_BITS
        and not diagnostic_only
        and (
            fp_transient
            or (
                budget_model_id == STRICT_ADDITIVE_BUDGET_MODEL_ID
                and calibration_state_bits_or_shared_cost == "0"
            )
        )
    )
    return {
        "resolver_id": resolver_id,
        "tier": tier,
        "primary_calibration_id": context["primary_calibration_id"],
        "seed": context["seed"],
        "bucket_candidate_ids_hash16": context["bucket_candidate_ids_hash16"],
        "bucket_size": int(context["bucket_size"]),
        "oracle_best_candidate_id": oracle_id,
        **selected_payload,
        "unique_selects_oracle": unique_selects_oracle,
        "qualifies_under_persistent_budget": qualifies_under_persistent_budget,
        "primary_persistent_bits": float(PRIMARY_TERNARY_PERSISTENT_BITS),
        "persistent_state_bits": float(persistent_state_bits),
        "extra_state_bits": float(extra_state_bits),
        "combined_persistent_bits": combined_persistent_bits,
        "persistent_budget_bits": float(PERSISTENT_BUDGET_BITS),
        "fp_transient": bool(fp_transient),
        "diagnostic_only": bool(diagnostic_only),
        "credit_mechanistic": bool(credit_mechanistic),
        "not_new_evidence": bool(not_new_evidence),
        "same_scalar_as_bucket": bool(same_scalar_as_bucket),
        "same_rank_as_bucket": bool(same_rank_as_bucket),
        "budget_model_id": budget_model_id,
        "calibration_scope": calibration_scope,
        "calibration_state_bits_or_shared_cost": calibration_state_bits_or_shared_cost,
        "sidecar_level": sidecar_level,
        "sidecar_persistent_bits": sidecar_persistent_bits,
        "resolver_score": resolver_score,
        "b7b_accumulator_dynamics_required": True,
        "primary_success_allowed": bool(
            not diagnostic_only
            and not not_new_evidence
            and unique_selects_oracle
            and (fp_transient or qualifies_under_persistent_budget)
        ),
    }


def _diagnostic_resolver_observations(context: Mapping[str, Any]) -> list[dict[str, Any]]:
    bucket_rows = list(context["bucket_rows"])
    resolver_rows: list[tuple[str, Mapping[str, Any] | None]] = [
        (
            DIAGNOSTIC_FLAT_INDEX_TIE_BREAKER_ID,
            _diagnostic_flat_index_tiebreak_row(bucket_rows),
        ),
        (
            DIAGNOSTIC_CANDIDATE_ORDER_TIE_BREAKER_ID,
            bucket_rows[0] if bucket_rows else None,
        ),
        (
            DIAGNOSTIC_CANDIDATE_HASH_TIE_BREAKER_ID,
            min(bucket_rows, key=_candidate_hash_sort_key) if bucket_rows else None,
        ),
    ]
    observations: list[dict[str, Any]] = []
    for resolver_id, row in resolver_rows:
        selected_rows = [row] if row is not None else []
        observations.append(
            _base_resolver_observation(
                context,
                resolver_id=resolver_id,
                tier="zero_extra_state_diagnostic",
                selected_rows=selected_rows,
                persistent_state_bits=0.0,
                extra_state_bits=0.0,
                fp_transient=False,
                diagnostic_only=True,
                credit_mechanistic=False,
                not_new_evidence=True,
                same_scalar_as_bucket=False,
                same_rank_as_bucket=False,
            )
        )
    return observations


def _transient_resolver_observation(
    context: Mapping[str, Any],
    *,
    spec: ScalarSpec,
    direction: int,
    primary_taylor_direction: int,
) -> dict[str, Any]:
    bucket_rows = list(context["bucket_rows"])
    selected_rows, best_score = _unique_scalar_argmax(
        bucket_rows,
        value_fn=spec.compute,
        direction=direction,
    )
    resolver_order = _rank_order_ids(
        bucket_rows,
        value_fn=spec.compute,
        direction=direction,
    )
    primary_order = list(context["primary_taylor_order_ids"])
    same_scalar = spec.scalar_id == "taylor_benefit"
    same_rank = resolver_order == primary_order
    return _base_resolver_observation(
        context,
        resolver_id=f"transient_fp_scalar:{spec.scalar_id}",
        tier="transient_fp_scalar",
        selected_rows=selected_rows,
        persistent_state_bits=0.0,
        extra_state_bits=0.0,
        fp_transient=True,
        diagnostic_only=False,
        credit_mechanistic=True,
        not_new_evidence=bool(same_scalar or same_rank),
        same_scalar_as_bucket=same_scalar,
        same_rank_as_bucket=same_rank,
        budget_model_id="primary_ternary_persistent_plus_transient_fp_resolver_v0",
        calibration_scope="per_step_forward_backward_transient",
        calibration_state_bits_or_shared_cost="0",
        resolver_score=best_score,
    ) | {
        "fixed_direction": int(direction),
        "primary_taylor_fixed_direction": int(primary_taylor_direction),
    }


def _sidecar_thresholds_for_contexts(
    seed43_bucket_rows: Sequence[Mapping[str, Any]],
    seed29_bucket_rows: Sequence[Mapping[str, Any]],
    *,
    spec: ScalarSpec,
    direction: int,
    level: int,
    sidecar_calibration_id: str,
) -> dict[str, tuple[float, ...]]:
    seed43_scores = [
        float(direction) * float(spec.compute(row))
        for row in seed43_bucket_rows
    ]
    seed29_scores = [
        float(direction) * float(spec.compute(row))
        for row in seed29_bucket_rows
    ]
    seed43_thresholds = _ordinal_thresholds_from_scores(seed43_scores, levels=level)
    if sidecar_calibration_id == SIDECAR_CALIBRATION_BUCKET_ONLINE:
        seed29_thresholds = _ordinal_thresholds_from_scores(seed29_scores, levels=level)
    elif sidecar_calibration_id == SIDECAR_CALIBRATION_SEED43_BUCKET_OOS:
        seed29_thresholds = seed43_thresholds
    else:
        raise ValueError(f"unsupported sidecar calibration {sidecar_calibration_id!r}")
    return {
        SEED43_LABEL: seed43_thresholds,
        SEED29_LABEL: seed29_thresholds,
    }


def _persistent_sidecar_observation(
    context: Mapping[str, Any],
    *,
    spec: ScalarSpec,
    direction: int,
    level: int,
    sidecar_calibration_id: str,
    thresholds: Sequence[float],
) -> dict[str, Any]:
    bucket_rows = list(context["bucket_rows"])
    ordinals_by_id = {
        str(row["candidate_id"]): _ordinal_bin(
            float(direction) * float(spec.compute(row)),
            thresholds=thresholds,
            levels=level,
        )
        for row in bucket_rows
    }
    selected_rows = _ordinal_top_bucket_rows(bucket_rows, ordinals_by_id=ordinals_by_id)
    sidecar_bits = float(math.log2(level))
    return _base_resolver_observation(
        context,
        resolver_id=(
            f"persistent_ordinal_sidecar:{spec.scalar_id}:"
            f"level{level}:{sidecar_calibration_id}"
        ),
        tier="persistent_discrete_sidecar",
        selected_rows=selected_rows,
        persistent_state_bits=sidecar_bits,
        extra_state_bits=sidecar_bits,
        fp_transient=False,
        diagnostic_only=False,
        credit_mechanistic=True,
        not_new_evidence=False,
        same_scalar_as_bucket=False,
        same_rank_as_bucket=False,
        budget_model_id=STRICT_ADDITIVE_BUDGET_MODEL_ID,
        calibration_scope=sidecar_calibration_id,
        calibration_state_bits_or_shared_cost=(
            "adaptive_threshold_state_not_free_under_strict_default"
            if sidecar_calibration_id == SIDECAR_CALIBRATION_BUCKET_ONLINE
            else "shared_seed43_thresholds_not_costed_as_free_for_success"
        ),
        sidecar_level=level,
        sidecar_persistent_bits=sidecar_bits,
    ) | {
        "fixed_direction": int(direction),
        "sidecar_thresholds": [float(threshold) for threshold in thresholds],
    }


def _aggregate_resolver_observations(
    observations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for observation in observations:
        grouped.setdefault(str(observation["resolver_id"]), []).append(observation)
    aggregates: dict[str, Any] = {}
    for resolver_id, rows in grouped.items():
        aggregate_unique = bool(rows) and all(
            bool(row["unique_selects_oracle"]) for row in rows
        )
        aggregate_budget = bool(rows) and all(
            bool(row["qualifies_under_persistent_budget"]) for row in rows
        )
        aggregate_new_evidence = bool(rows) and all(
            not bool(row["not_new_evidence"]) for row in rows
        )
        tiers = sorted({str(row["tier"]) for row in rows})
        aggregates[resolver_id] = {
            "resolver_id": resolver_id,
            "tier": tiers[0] if len(tiers) == 1 else "mixed",
            "observation_count": len(rows),
            "aggregate_scope": "both_seeds_x_both_primary_calibrations",
            "unique_selects_oracle_all": aggregate_unique,
            "qualifies_under_persistent_budget_all": aggregate_budget,
            "new_evidence_all": aggregate_new_evidence,
            "diagnostic_only": all(bool(row["diagnostic_only"]) for row in rows),
            "fp_transient": all(bool(row["fp_transient"]) for row in rows),
            "any_same_rank_as_bucket": any(bool(row["same_rank_as_bucket"]) for row in rows),
            "any_not_new_evidence": any(bool(row["not_new_evidence"]) for row in rows),
            "max_combined_persistent_bits": max(
                float(row["combined_persistent_bits"]) for row in rows
            ),
            "primary_success_allowed_all": bool(
                aggregate_unique
                and aggregate_new_evidence
                and all(bool(row["primary_success_allowed"]) for row in rows)
            ),
        }
    return aggregates


def _b7a_label_decision(aggregates: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    labels: list[str] = [B7A_LABEL_PROXY_ONLY_B7B_REQUIRED]
    if any(
        bool(row["tier"] == "persistent_discrete_sidecar")
        and bool(row["unique_selects_oracle_all"])
        and bool(row["qualifies_under_persistent_budget_all"])
        for row in aggregates.values()
    ):
        labels.append(B7A_LABEL_PERSISTENT_SUB2_UNIQUE)
    if any(
        bool(row["tier"] == "transient_fp_scalar")
        and bool(row["unique_selects_oracle_all"])
        and bool(row["new_evidence_all"])
        for row in aggregates.values()
    ):
        labels.append(B7A_LABEL_TRANSIENT_RESOLVER_SUCCESS)
    if any(
        bool(row["tier"] == "persistent_discrete_sidecar")
        and bool(row["unique_selects_oracle_all"])
        and not bool(row["qualifies_under_persistent_budget_all"])
        for row in aggregates.values()
    ):
        labels.append(B7A_LABEL_PERSISTENT_EXCEEDS_BUDGET)
    if any(
        bool(row["diagnostic_only"])
        and bool(row["unique_selects_oracle_all"])
        for row in aggregates.values()
    ):
        labels.append(B7A_LABEL_ZERO_STATE_DIAGNOSTIC_ONLY)
    if len(labels) == 1:
        labels.append(B7A_LABEL_NO_UNIQUE_RESOLVER)
    primary_label = next(
        label for label in B7A_LABEL_PRIORITY if label in labels
    )
    return {
        "all_applicable_labels": labels,
        "primary_label": primary_label,
        "proxy_caveat_label": B7A_LABEL_PROXY_ONLY_B7B_REQUIRED,
        "priority_order": list(B7A_LABEL_PRIORITY),
        "aggregate_scope": "both_seeds_x_both_primary_calibrations",
    }


def _sub2_tiebreak_sidecar_sweep(
    seed43_rows: Sequence[Mapping[str, Any]],
    seed29_rows: Sequence[Mapping[str, Any]],
    *,
    scalar_results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    primary_direction = int(scalar_results["taylor_benefit"]["raw_continuous"]["fixed_direction"])
    contexts = _ternary_bucket_contexts(
        seed43_rows,
        seed29_rows,
        fixed_direction=primary_direction,
    )
    observations: list[dict[str, Any]] = []
    for context in contexts:
        observations.extend(_diagnostic_resolver_observations(context))
        for scalar_id in SIDECAR_TRANSIENT_SCALAR_IDS:
            spec = _scalar_spec_by_id(scalar_id)
            direction = int(scalar_results[scalar_id]["raw_continuous"]["fixed_direction"])
            observations.append(
                _transient_resolver_observation(
                    context,
                    spec=spec,
                    direction=direction,
                    primary_taylor_direction=primary_direction,
                )
            )
    context_by_cal_seed = {
        (context["primary_calibration_id"], context["seed"]): context
        for context in contexts
    }
    for primary_calibration_id in (
        CALIBRATION_ONLINE_CANDIDATE_QUANTILE,
        CALIBRATION_SEED43_THRESHOLDS_OOS,
    ):
        seed43_context = context_by_cal_seed[(primary_calibration_id, SEED43_LABEL)]
        seed29_context = context_by_cal_seed[(primary_calibration_id, SEED29_LABEL)]
        for scalar_id in SIDECAR_PERSISTENT_SCALAR_IDS:
            spec = _scalar_spec_by_id(scalar_id)
            direction = int(scalar_results[scalar_id]["raw_continuous"]["fixed_direction"])
            for level in SIDECAR_PERSISTENT_LEVELS:
                for sidecar_calibration_id in (
                    SIDECAR_CALIBRATION_BUCKET_ONLINE,
                    SIDECAR_CALIBRATION_SEED43_BUCKET_OOS,
                ):
                    thresholds_by_seed = _sidecar_thresholds_for_contexts(
                        seed43_context["bucket_rows"],
                        seed29_context["bucket_rows"],
                        spec=spec,
                        direction=direction,
                        level=level,
                        sidecar_calibration_id=sidecar_calibration_id,
                    )
                    for context in (seed43_context, seed29_context):
                        observations.append(
                            _persistent_sidecar_observation(
                                context,
                                spec=spec,
                                direction=direction,
                                level=level,
                                sidecar_calibration_id=sidecar_calibration_id,
                                thresholds=thresholds_by_seed[context["seed"]],
                            )
                        )
    aggregates = _aggregate_resolver_observations(observations)
    return {
        "schema_version": SUB2_TIEBREAK_SIDECAR_SWEEP_SCHEMA_VERSION,
        "source_bucket": {
            "source_payload_key": "sub2_ordinal_sweep",
            "level": 3,
            "primary_scalar_id": "taylor_benefit",
            "primary_persistent_bits": float(PRIMARY_TERNARY_PERSISTENT_BITS),
            "primary_calibration_classes": [
                CALIBRATION_ONLINE_CANDIDATE_QUANTILE,
                CALIBRATION_SEED43_THRESHOLDS_OOS,
            ],
        },
        "budget_policy": {
            "budget_model_id": STRICT_ADDITIVE_BUDGET_MODEL_ID,
            "persistent_budget_bits": float(PERSISTENT_BUDGET_BITS),
            "strict_additive_default": True,
            "non_additive_or_amortized_model_applied": False,
            "calibration_threshold_state_is_not_free_for_persistent_success": True,
        },
        "resolver_tiers": {
            "zero_extra_state_diagnostics": {
                "diagnostic_only": True,
                "primary_success_allowed": False,
            },
            "transient_fp_scalars": {
                "fp_transient": True,
                "persistent_state_bits": 0,
                "success_label_meaning": "sub2_persistent_with_fp_transient_resolver_not_fp_free",
            },
            "persistent_discrete_sidecars": {
                "levels": list(SIDECAR_PERSISTENT_LEVELS),
                "strict_additive_combined_bits": "log2(3)+log2(sidecar_levels)",
            },
        },
        "observations": observations,
        "aggregate_results": aggregates,
        "label_decision": _b7a_label_decision(aggregates),
        "non_claims": [
            "b7a is a single-step ceiling proxy; b7b accumulator dynamics remain required",
            "zero-state diagnostics are never success labels",
            "transient FP resolver success is not FP-free full-runtime sub-2",
            "persistent sidecar unique selection is not sub-2 success unless the combined persistent budget qualifies",
        ],
    }


def _rank_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    value_fn: Callable[[Mapping[str, Any]], float],
    direction: int,
) -> dict[str, Any]:
    ordered = _sorted_rows_for_direction(rows, value_fn=value_fn, direction=direction)
    oracle_best = _oracle_best_row(rows)
    oracle_best_id = str(oracle_best["candidate_id"])
    oracle_best_rank = next(
        index
        for index, row in enumerate(ordered, start=1)
        if str(row["candidate_id"]) == oracle_best_id
    )
    top_ids = [str(row["candidate_id"]) for row in ordered[: max(TOP_K_VALUES)]]
    return {
        "auc": _pairwise_auc(rows, value_fn=value_fn, direction=direction),
        "oracle_best_candidate_id": oracle_best_id,
        "oracle_best_rank": int(oracle_best_rank),
        "topk_capture": {
            str(k): bool(oracle_best_rank <= k)
            for k in TOP_K_VALUES
        },
        "ordered_top5_candidate_ids": top_ids[:5],
    }


def _directional_metric_bundle(
    seed43_rows: Sequence[Mapping[str, Any]],
    seed29_rows: Sequence[Mapping[str, Any]],
    *,
    value_fn: Callable[[Mapping[str, Any]], float],
    direction: int,
) -> dict[str, Any]:
    reverse_direction = -int(direction)
    return {
        "fixed_direction": int(direction),
        SEED43_LABEL: _rank_metrics(seed43_rows, value_fn=value_fn, direction=int(direction)),
        SEED29_LABEL: _rank_metrics(seed29_rows, value_fn=value_fn, direction=int(direction)),
        "reverse_direction": {
            "direction": int(reverse_direction),
            SEED43_LABEL: _rank_metrics(
                seed43_rows,
                value_fn=value_fn,
                direction=int(reverse_direction),
            ),
            SEED29_LABEL: _rank_metrics(
                seed29_rows,
                value_fn=value_fn,
                direction=int(reverse_direction),
            ),
        },
    }


def _best_seed43_direction(
    seed43_rows: Sequence[Mapping[str, Any]],
    *,
    value_fn: Callable[[Mapping[str, Any]], float],
) -> int:
    positive_auc = _pairwise_auc(seed43_rows, value_fn=value_fn, direction=1)
    negative_auc = _pairwise_auc(seed43_rows, value_fn=value_fn, direction=-1)
    return 1 if positive_auc >= negative_auc else -1


def _all_rows_have_q5_field(
    rows: Sequence[Mapping[str, Any]],
    *,
    field: str,
) -> bool:
    return all(row.get(field) is not None for row in rows)


def _family_compressed_metrics(
    metrics_by_family_id: Mapping[str, Mapping[str, Any]],
    *,
    family_id: str,
) -> dict[str, Any] | None:
    metrics = metrics_by_family_id.get(family_id)
    if metrics is None:
        return None
    return {
        "family_id": family_id,
        "receipt_family_compressed_auc": float(
            metrics["within_band_pairwise_auc_report_only"]
        ),
        "oracle_best_bucket_fraction": float(metrics["oracle_best_bucket_fraction"]),
        "oracle_best_bucket_regret_capture_ratio": float(
            metrics["oracle_best_bucket_regret_capture_ratio"]
        ),
        "oracle_best_bucket_regret_spread_ratio": float(
            metrics["oracle_best_bucket_regret_spread_ratio"]
        ),
        "oracle_best_bucket_top_k_capture_fraction": float(
            metrics["oracle_best_bucket_top_k_capture_fraction"]
        ),
        "matched_hash_null_fraction_gte_observed_bucket_fraction": float(
            metrics["matched_hash_null_fraction_gte_observed_bucket_fraction"]
        ),
        "matched_hash_null_fraction_lte_observed_regret_capture_ratio": float(
            metrics["matched_hash_null_fraction_lte_observed_regret_capture_ratio"]
        ),
        "bucket_count": int(metrics["bucket_count"]),
        "bucket_cardinality_histogram": dict(metrics["bucket_cardinality_histogram"]),
        "singleton_bucket_count": int(metrics["singleton_bucket_count"]),
    }


def _validate_required_row_fields(row: Mapping[str, Any]) -> None:
    missing = [field for field in REQUIRED_RECEIPT_ROW_FIELDS if field not in row]
    if missing:
        raise ValueError(f"receipt row is missing required fields: {missing}")


def load_activation_credit_ceiling_audit_receipt(
    receipt_path: str | Path,
    *,
    seed_label: str,
) -> LoadedCeilingAuditReceipt:
    path = Path(receipt_path)
    payload = _json_load(path)
    compact = payload["oracle_screen"]["compact_summary"]
    target_tie_band = compact["target_tie_band"]
    sampled_rows = compact["sampled_candidate_table"]
    target_rows = [dict(row) for row in sampled_rows if bool(row.get("in_target_tie_band"))]
    valid_rows = [dict(row) for row in target_rows if bool(row.get("activation_feature_valid"))]
    for row in valid_rows:
        _validate_required_row_fields(row)
    reported_band_count = int(target_tie_band["band_candidate_count"])
    reported_valid_count = int(target_tie_band["valid_activation_candidate_count"])
    if len(target_rows) != reported_band_count:
        raise ValueError(
            f"{path}: target-band row count mismatch "
            f"(extracted {len(target_rows)} != reported {reported_band_count})"
        )
    if len(valid_rows) != reported_valid_count:
        raise ValueError(
            f"{path}: valid in-band row count mismatch "
            f"(extracted {len(valid_rows)} != reported {reported_valid_count})"
        )
    target_tie_band_id = str(target_tie_band["target_tie_band_id"])
    if target_tie_band_id != ACTIVATION_CREDIT_CEILING_AUDIT_EXPECTED_TARGET_TIE_BAND_ID:
        raise ValueError(
            f"{path}: unexpected target tie band {target_tie_band_id!r}"
        )
    return LoadedCeilingAuditReceipt(
        seed_label=str(seed_label),
        path=str(path),
        sha256=file_sha256(path),
        branch_classification=payload["oracle_screen"].get("branch_classification"),
        target_tie_band_id=target_tie_band_id,
        reported_band_candidate_count=reported_band_count,
        reported_valid_activation_candidate_count=reported_valid_count,
        extracted_band_candidate_count=len(target_rows),
        extracted_valid_activation_candidate_count=len(valid_rows),
        rows=tuple(valid_rows),
        family_metrics_by_id=compact["family_metrics"]["metrics_by_family_id"],
    )


def _approx_equal(left: float, right: float, *, tol: float) -> bool:
    return abs(float(left) - float(right)) <= float(tol)


def _topk_all_true(metrics: Mapping[str, Any]) -> bool:
    topk = metrics["topk_capture"]
    return all(bool(topk[str(k)]) for k in TOP_K_VALUES)


def _raw_signal_strong(metrics: Mapping[str, Any]) -> bool:
    return bool(
        float(metrics[SEED43_LABEL]["auc"]) >= PRIMARY_RAW_AUC_MIN
        and float(metrics[SEED29_LABEL]["auc"]) >= PRIMARY_RAW_AUC_MIN
        and _topk_all_true(metrics[SEED43_LABEL])
        and _topk_all_true(metrics[SEED29_LABEL])
    )


def _q5_ordinal_high(metrics: Mapping[str, Any] | None) -> bool:
    if metrics is None:
        return False
    return bool(
        float(metrics[SEED43_LABEL]["auc"]) >= PRIMARY_Q5_ORDINAL_AUC_MIN
        and float(metrics[SEED29_LABEL]["auc"]) >= PRIMARY_Q5_ORDINAL_AUC_MIN
    )


def _receipt_family_low(metrics: Mapping[str, Any] | None) -> bool:
    if metrics is None:
        return False
    return bool(
        float(metrics[SEED43_LABEL]["receipt_family_compressed_auc"])
        <= PRIMARY_RECEIPT_FAMILY_AUC_MAX
        and float(metrics[SEED29_LABEL]["receipt_family_compressed_auc"])
        <= PRIMARY_RECEIPT_FAMILY_AUC_MAX
    )


def _primary_loss_classification(
    *,
    raw_metrics: Mapping[str, Any],
    q5_metrics: Mapping[str, Any] | None,
    family_metrics: Mapping[str, Any] | None,
) -> dict[str, Any]:
    raw_seed43_auc = float(raw_metrics[SEED43_LABEL]["auc"])
    raw_seed29_auc = float(raw_metrics[SEED29_LABEL]["auc"])
    q5_seed43_auc = (
        float(q5_metrics[SEED43_LABEL]["auc"])
        if q5_metrics is not None
        else None
    )
    q5_seed29_auc = (
        float(q5_metrics[SEED29_LABEL]["auc"])
        if q5_metrics is not None
        else None
    )
    family_seed43_auc = (
        float(family_metrics[SEED43_LABEL]["receipt_family_compressed_auc"])
        if family_metrics is not None
        else None
    )
    family_seed29_auc = (
        float(family_metrics[SEED29_LABEL]["receipt_family_compressed_auc"])
        if family_metrics is not None
        else None
    )
    continuous_to_q5_loss = None
    if q5_seed43_auc is not None and q5_seed29_auc is not None:
        continuous_to_q5_loss = {
            SEED43_LABEL: float(raw_seed43_auc - q5_seed43_auc),
            SEED29_LABEL: float(raw_seed29_auc - q5_seed29_auc),
        }
    q5_to_family_loss = None
    if (
        q5_seed43_auc is not None
        and q5_seed29_auc is not None
        and family_seed43_auc is not None
        and family_seed29_auc is not None
    ):
        q5_to_family_loss = {
            SEED43_LABEL: float(q5_seed43_auc - family_seed43_auc),
            SEED29_LABEL: float(q5_seed29_auc - family_seed29_auc),
        }
    if not _raw_signal_strong(raw_metrics):
        label = "reopen_scalar_or_joint_interaction"
    elif (
        _q5_ordinal_high(q5_metrics)
        and _receipt_family_low(family_metrics)
        and q5_to_family_loss is not None
        and min(q5_to_family_loss.values()) >= PRIMARY_MATERIAL_AUC_DROP_MIN
    ):
        label = "receipt_family_bucket_tiebreak_loss"
    elif (
        q5_metrics is not None
        and family_metrics is not None
        and continuous_to_q5_loss is not None
        and min(continuous_to_q5_loss.values()) >= PRIMARY_MATERIAL_AUC_DROP_MIN
        and not _q5_ordinal_high(q5_metrics)
    ):
        label = "q5_ordinal_quantization_loss"
    else:
        label = "no_clear_representation_collapse"
    return {
        "primary_scalar_id": "taylor_benefit",
        "classification_label": label,
        "continuous_to_q5_ordinal_loss": continuous_to_q5_loss,
        "q5_ordinal_to_receipt_family_loss": q5_to_family_loss,
    }


def _scalar_result(
    spec: ScalarSpec,
    seed43_rows: Sequence[Mapping[str, Any]],
    seed29_rows: Sequence[Mapping[str, Any]],
    *,
    seed43_receipt: LoadedCeilingAuditReceipt,
    seed29_receipt: LoadedCeilingAuditReceipt,
) -> dict[str, Any]:
    raw_direction = _best_seed43_direction(seed43_rows, value_fn=spec.compute)
    raw_metrics = _directional_metric_bundle(
        seed43_rows,
        seed29_rows,
        value_fn=spec.compute,
        direction=raw_direction,
    )
    q5_metrics = None
    if spec.q5_field is not None and _all_rows_have_q5_field(seed43_rows, field=spec.q5_field) and _all_rows_have_q5_field(seed29_rows, field=spec.q5_field):
        q5_metrics = _directional_metric_bundle(
            seed43_rows,
            seed29_rows,
            value_fn=lambda row, field=spec.q5_field: _float_row_field(row, field),
            direction=raw_direction,
        )
    family_seed43 = (
        _family_compressed_metrics(
            seed43_receipt.family_metrics_by_id,
            family_id=spec.receipt_family_id,
        )
        if spec.receipt_family_id is not None
        else None
    )
    family_seed29 = (
        _family_compressed_metrics(
            seed29_receipt.family_metrics_by_id,
            family_id=spec.receipt_family_id,
        )
        if spec.receipt_family_id is not None
        else None
    )
    family_metrics = None
    if family_seed43 is not None and family_seed29 is not None:
        family_metrics = {
            SEED43_LABEL: family_seed43,
            SEED29_LABEL: family_seed29,
        }
    result = {
        "display_name": spec.display_name,
        "category": spec.category,
        "decision_authority_allowed": bool(spec.decision_authority_allowed),
        "raw_continuous": raw_metrics,
        "q5_bin_index": q5_metrics,
        "receipt_family_compressed": family_metrics,
    }
    if spec.scalar_id == "taylor_benefit":
        result["loss_decomposition"] = _primary_loss_classification(
            raw_metrics=raw_metrics,
            q5_metrics=q5_metrics,
            family_metrics=family_metrics,
        )
    return result


def _known_anchor_reproduction(
    scalar_results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    raw_results: dict[str, Any] = {}
    raw_reproduced = True
    for scalar_id, expected in KNOWN_BRANCH4_RAW_AUC_EXPECTATIONS.items():
        raw_metrics = scalar_results[scalar_id]["raw_continuous"]
        seed43_auc = float(raw_metrics[SEED43_LABEL]["auc"])
        seed29_auc = float(raw_metrics[SEED29_LABEL]["auc"])
        topk_ok = bool(
            _topk_all_true(raw_metrics[SEED43_LABEL])
            and _topk_all_true(raw_metrics[SEED29_LABEL])
        )
        auc_ok = bool(
            _approx_equal(
                seed43_auc,
                float(expected[SEED43_LABEL]),
                tol=KNOWN_BRANCH4_AUC_TOLERANCE,
            )
            and _approx_equal(
                seed29_auc,
                float(expected[SEED29_LABEL]),
                tol=KNOWN_BRANCH4_AUC_TOLERANCE,
            )
        )
        raw_results[scalar_id] = {
            "seed43_auc": seed43_auc,
            "seed29_auc": seed29_auc,
            "expected_seed43_auc": float(expected[SEED43_LABEL]),
            "expected_seed29_auc": float(expected[SEED29_LABEL]),
            "topk_all_true_both_seeds": topk_ok,
            "auc_match": auc_ok,
            "reproduced": bool(auc_ok and topk_ok),
        }
        raw_reproduced = bool(raw_reproduced and raw_results[scalar_id]["reproduced"])
    family_metrics = scalar_results["taylor_benefit"]["receipt_family_compressed"]
    if family_metrics is None:
        family_reproduced = False
        family_result = {
            "reproduced": False,
            "missing": True,
        }
    else:
        seed43_auc = float(
            family_metrics[SEED43_LABEL]["receipt_family_compressed_auc"]
        )
        seed29_auc = float(
            family_metrics[SEED29_LABEL]["receipt_family_compressed_auc"]
        )
        family_reproduced = bool(
            _approx_equal(
                seed43_auc,
                KNOWN_BRANCH4_RECEIPT_FAMILY_AUC_EXPECTATIONS[SEED43_LABEL],
                tol=KNOWN_BRANCH4_AUC_TOLERANCE,
            )
            and _approx_equal(
                seed29_auc,
                KNOWN_BRANCH4_RECEIPT_FAMILY_AUC_EXPECTATIONS[SEED29_LABEL],
                tol=KNOWN_BRANCH4_AUC_TOLERANCE,
            )
        )
        family_result = {
            "seed43_auc": seed43_auc,
            "seed29_auc": seed29_auc,
            "expected_seed43_auc": KNOWN_BRANCH4_RECEIPT_FAMILY_AUC_EXPECTATIONS[
                SEED43_LABEL
            ],
            "expected_seed29_auc": KNOWN_BRANCH4_RECEIPT_FAMILY_AUC_EXPECTATIONS[
                SEED29_LABEL
            ],
            "reproduced": family_reproduced,
        }
    return {
        "raw_continuous_anchor_reproduced": raw_reproduced,
        "receipt_family_anchor_reproduced": family_reproduced,
        "strict_stop_triggered": bool(not raw_reproduced or not family_reproduced),
        "raw_continuous_reference_anchors": raw_results,
        "receipt_family_reference_anchor": family_result,
    }


def build_activation_credit_ceiling_audit(
    *,
    seed43_receipt_path: str | Path,
    seed29_receipt_path: str | Path,
) -> dict[str, Any]:
    seed43_receipt = load_activation_credit_ceiling_audit_receipt(
        seed43_receipt_path,
        seed_label=SEED43_LABEL,
    )
    seed29_receipt = load_activation_credit_ceiling_audit_receipt(
        seed29_receipt_path,
        seed_label=SEED29_LABEL,
    )
    scalar_results = {
        spec.scalar_id: _scalar_result(
            spec,
            seed43_receipt.rows,
            seed29_receipt.rows,
            seed43_receipt=seed43_receipt,
            seed29_receipt=seed29_receipt,
        )
        for spec in SCALAR_SPECS
    }
    primary_loss = scalar_results["taylor_benefit"]["loss_decomposition"]
    sub2_ordinal_sweep = _sub2_ordinal_sweep(
        seed43_receipt.rows,
        seed29_receipt.rows,
        primary_scalar_result=scalar_results["taylor_benefit"],
    )
    sub2_tiebreak_sidecar_sweep = _sub2_tiebreak_sidecar_sweep(
        seed43_receipt.rows,
        seed29_receipt.rows,
        scalar_results=scalar_results,
    )
    return {
        "schema_version": ACTIVATION_CREDIT_CEILING_AUDIT_SCHEMA_VERSION,
        "target_name": ACTIVATION_CREDIT_CEILING_AUDIT_TARGET_NAME,
        "input_receipts": {
            SEED43_LABEL: {
                "path": seed43_receipt.path,
                "sha256": seed43_receipt.sha256,
                "branch_classification": seed43_receipt.branch_classification,
                "target_tie_band_id": seed43_receipt.target_tie_band_id,
                "reported_band_candidate_count": seed43_receipt.reported_band_candidate_count,
                "reported_valid_activation_candidate_count": seed43_receipt.reported_valid_activation_candidate_count,
                "extracted_band_candidate_count": seed43_receipt.extracted_band_candidate_count,
                "extracted_valid_activation_candidate_count": seed43_receipt.extracted_valid_activation_candidate_count,
            },
            SEED29_LABEL: {
                "path": seed29_receipt.path,
                "sha256": seed29_receipt.sha256,
                "branch_classification": seed29_receipt.branch_classification,
                "target_tie_band_id": seed29_receipt.target_tie_band_id,
                "reported_band_candidate_count": seed29_receipt.reported_band_candidate_count,
                "reported_valid_activation_candidate_count": seed29_receipt.reported_valid_activation_candidate_count,
                "extracted_band_candidate_count": seed29_receipt.extracted_band_candidate_count,
                "extracted_valid_activation_candidate_count": seed29_receipt.extracted_valid_activation_candidate_count,
            },
        },
        "metric_definitions": {
            "oracle_best_definition": "minimum regret_vs_target_tie_band_oracle_top1_local_loss_delta",
            "pairwise_auc_tie_handling": "scalar ties count as 0.5; regret ties within ORACLE_SCREEN_IMPROVEMENT_EPS are skipped",
            "direction_protocol": "choose monotone sign on seed43 raw continuous AUC only, apply unchanged to seed29, and report the reverse direction separately",
            "q5_direction_protocol": "q5_bin_index_auc uses the same fixed raw direction as its corresponding raw scalar",
            "compressed_auc_note": (
                "receipt_family_compressed_auc is the pipeline-canonical family metric "
                "from the receipt; q5_bin_index_auc is the same-evaluator ordinal diagnostic"
            ),
            "label_leak_policy": (
                "label-leak upper-bound rows are emitted for sanity only, tagged "
                f"{LABEL_LEAK_UPPER_BOUND_TAG}, and excluded from decision authority"
            ),
            "sub2_ordinal_sweep_note": (
                "ordinal sweep bins seed43-fixed taylor_benefit scores into "
                "learner-available levels without using raw continuous values "
                "inside the top bucket; flat-index tie-break is diagnostic only"
            ),
            "sub2_tiebreak_sidecar_sweep_note": (
                "sidecar sweep resolves only inside the B6 ternary top bucket "
                "and reports unique selection separately from persistent-budget "
                "qualification"
            ),
        },
        "decision_authorized_scalar_ids": [
            spec.scalar_id for spec in SCALAR_SPECS if spec.decision_authority_allowed
        ],
        "label_leak_scalar_ids": [
            spec.scalar_id for spec in SCALAR_SPECS if not spec.decision_authority_allowed
        ],
        "scalar_results": scalar_results,
        "primary_loss_decomposition": primary_loss,
        "sub2_ordinal_sweep": sub2_ordinal_sweep,
        "sub2_tiebreak_sidecar_sweep": sub2_tiebreak_sidecar_sweep,
        "known_branch4_anchor_reproduction": _known_anchor_reproduction(
            scalar_results,
        ),
        "non_claims": [
            "no learner/runtime/full-sub2 success claim",
            "no eligibility build decision",
            "label-leak upper-bound rows have zero decision authority",
        ],
    }


def _b5b_primary_family_bucket_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], list[dict[str, Any]]]:
    oracle_best = _oracle_best_row(rows)
    groups: dict[tuple[int, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (int(row[ACTIVATION_CREDIT_TAYLOR_BENEFIT_Q5_FIELD]),)
        groups.setdefault(key, []).append(dict(row))
    oracle_key = (int(oracle_best[ACTIVATION_CREDIT_TAYLOR_BENEFIT_Q5_FIELD]),)
    return oracle_best, list(groups.get(oracle_key, []))


def _b5b_intra_bucket_tiebreak_key_fn(
    oracle_bucket_rows: Sequence[Mapping[str, Any]],
    *,
    tiebreak_rank_key_id: str,
    eligibility_direction: int,
) -> Callable[[Mapping[str, Any]], tuple[Any, ...]]:
    if tiebreak_rank_key_id == ACTIVATION_CREDIT_TIEBREAK_KEY_CURRENT_RANK:
        return lambda candidate: (int(candidate["current_rank_position"]),)
    spec = _scalar_spec_by_id(B5B_ELIGIBILITY_SCALAR_ID)
    scores_by_id = {
        str(row["candidate_id"]): float(eligibility_direction) * float(spec.compute(row))
        for row in oracle_bucket_rows
    }
    if tiebreak_rank_key_id == ACTIVATION_CREDIT_TIEBREAK_KEY_ORDINAL_ONLY_NO_INTRA_RANK:
        return lambda candidate: (0,)
    if tiebreak_rank_key_id == ACTIVATION_CREDIT_TIEBREAK_KEY_RAW_ELIGIBILITY_FP:
        return lambda candidate: (-scores_by_id[str(candidate["candidate_id"])],)
    level = (
        3
        if tiebreak_rank_key_id == ACTIVATION_CREDIT_TIEBREAK_KEY_TERNARY_ELIGIBILITY_ORDINAL
        else 5
    )
    thresholds = _ordinal_thresholds_from_scores(
        tuple(scores_by_id.values()),
        levels=level,
    )
    ordinals_by_id = {
        candidate_id: _ordinal_bin(score, thresholds=thresholds, levels=level)
        for candidate_id, score in scores_by_id.items()
    }
    return lambda candidate: (-int(ordinals_by_id[str(candidate["candidate_id"])]),)


def _b5b_counterfactual_family_auc(
    rows: Sequence[Mapping[str, Any]],
    *,
    tiebreak_rank_key_id: str,
    eligibility_direction: int,
) -> float:
    oracle_best, oracle_bucket = _b5b_primary_family_bucket_rows(rows)
    oracle_top1_delta = float(oracle_best["local_loss_delta"])
    key_fn = None
    if tiebreak_rank_key_id != ACTIVATION_CREDIT_TIEBREAK_KEY_CURRENT_RANK:
        key_fn = _b5b_intra_bucket_tiebreak_key_fn(
            oracle_bucket,
            tiebreak_rank_key_id=tiebreak_rank_key_id,
            eligibility_direction=eligibility_direction,
        )
    metrics = activation_credit_family_metrics_with_tiebreak(
        target_band_candidates=tuple(rows),
        family_id=ACTIVATION_CREDIT_PRIMARY_FAMILY_ID,
        oracle_best_candidate=oracle_best,
        oracle_top1_delta=oracle_top1_delta,
        intra_bucket_tiebreak_key_fn=key_fn,
    )
    return float(metrics["within_band_pairwise_auc_report_only"])


def _b5b_family_auc_recovers(aucs_by_seed: Mapping[str, float]) -> bool:
    return bool(
        float(aucs_by_seed[SEED43_LABEL]) > PRIMARY_RECEIPT_FAMILY_AUC_MAX
        and float(aucs_by_seed[SEED29_LABEL]) > PRIMARY_RECEIPT_FAMILY_AUC_MAX
    )


def _b5b_baseline_anchor_reproduced(aucs_by_seed: Mapping[str, float]) -> bool:
    return bool(
        _approx_equal(
            float(aucs_by_seed[SEED43_LABEL]),
            KNOWN_BRANCH4_RECEIPT_FAMILY_AUC_EXPECTATIONS[SEED43_LABEL],
            tol=KNOWN_BRANCH4_AUC_TOLERANCE,
        )
        and _approx_equal(
            float(aucs_by_seed[SEED29_LABEL]),
            KNOWN_BRANCH4_RECEIPT_FAMILY_AUC_EXPECTATIONS[SEED29_LABEL],
            tol=KNOWN_BRANCH4_AUC_TOLERANCE,
        )
    )


def _b5b_within_bucket_eligibility_calibration_failure(
    seed43_rows: Sequence[Mapping[str, Any]],
    seed29_rows: Sequence[Mapping[str, Any]],
    *,
    eligibility_direction: int,
) -> bool:
    spec = _scalar_spec_by_id(B5B_ELIGIBILITY_SCALAR_ID)
    _, seed43_bucket = _b5b_primary_family_bucket_rows(seed43_rows)
    oracle29, seed29_bucket = _b5b_primary_family_bucket_rows(seed29_rows)
    if not seed43_bucket or not seed29_bucket:
        return True
    level = 3
    seed43_scores = [
        float(eligibility_direction) * float(spec.compute(row))
        for row in seed43_bucket
    ]
    seed29_scores = [
        float(eligibility_direction) * float(spec.compute(row))
        for row in seed29_bucket
    ]
    seed43_thresholds = _ordinal_thresholds_from_scores(seed43_scores, levels=level)
    seed29_online_thresholds = _ordinal_thresholds_from_scores(
        seed29_scores,
        levels=level,
    )
    oracle_id = str(oracle29["candidate_id"])

    def _seed29_ordinal_flags(
        thresholds: Sequence[float],
    ) -> dict[str, bool]:
        ordinals = [
            _ordinal_bin(score, thresholds=thresholds, levels=level)
            for score in seed29_scores
        ]
        max_ordinal = max(ordinals)
        top_rows = [
            row
            for row, ordinal in zip(seed29_bucket, ordinals, strict=True)
            if int(ordinal) == int(max_ordinal)
        ]
        return {
            "unique_ordinal_top1": bool(
                len(top_rows) == 1 and str(top_rows[0]["candidate_id"]) == oracle_id
            ),
            "top_bucket_contains_oracle": bool(
                any(str(row["candidate_id"]) == oracle_id for row in top_rows)
            ),
        }

    online = _seed29_ordinal_flags(seed29_online_thresholds)
    oos = _seed29_ordinal_flags(seed43_thresholds)
    if bool(online["unique_ordinal_top1"]) and not bool(oos["unique_ordinal_top1"]):
        return True
    if bool(online["top_bucket_contains_oracle"]) and not bool(
        oos["top_bucket_contains_oracle"]
    ):
        return True
    return False


def _b5b_variant_metadata(tiebreak_rank_key_id: str) -> dict[str, Any]:
    uses_raw = tiebreak_rank_key_id == ACTIVATION_CREDIT_TIEBREAK_KEY_RAW_ELIGIBILITY_FP
    if tiebreak_rank_key_id == ACTIVATION_CREDIT_TIEBREAK_KEY_TERNARY_ELIGIBILITY_ORDINAL:
        role = "sub2_primary_candidate"
        persistent_bits = float(math.log2(3))
    elif tiebreak_rank_key_id == ACTIVATION_CREDIT_TIEBREAK_KEY_Q5_ELIGIBILITY_ORDINAL:
        role = "diagnostic_intermediate_only"
        persistent_bits = float(math.log2(5))
    elif tiebreak_rank_key_id == ACTIVATION_CREDIT_TIEBREAK_KEY_RAW_ELIGIBILITY_FP:
        role = "label_leak_upper_bound_diagnostic_only"
        persistent_bits = None
    elif tiebreak_rank_key_id == ACTIVATION_CREDIT_TIEBREAK_KEY_ORDINAL_ONLY_NO_INTRA_RANK:
        role = "ordinal_membership_diagnostic"
        persistent_bits = None
    else:
        role = "baseline"
        persistent_bits = float(math.log2(5))
    return {
        "tiebreak_rank_key_id": tiebreak_rank_key_id,
        "role": role,
        "uses_raw_continuous_inside_bucket": bool(uses_raw),
        "decision_authority_allowed": False if uses_raw else None,
        "persistent_bits": persistent_bits,
    }


def _b5b_emit_branch_classifier(
    *,
    harness_ok: bool,
    baseline_reproduced: bool,
    calibration_failure: bool,
    ternary_recovers: bool,
    q5_recovers: bool,
    raw_recovers: bool,
    ordinal_only_recovers: bool,
) -> dict[str, Any]:
    applicable: list[str] = []
    if not harness_ok:
        applicable.append(B5B_BRANCH_HARNESS_OR_INPUT_FAIL)
    if baseline_reproduced:
        applicable.append(B5B_BRANCH_TIEBREAK_BASELINE_REPRO)
    if calibration_failure:
        applicable.append(B5B_BRANCH_CALIBRATION_FAILURE)
    if ternary_recovers:
        applicable.append(B5B_BRANCH_TERNARY_TIEBREAK_RECOVERS)
    elif q5_recovers:
        applicable.append(B5B_BRANCH_Q5_ONLY_RECOVERS)
    elif raw_recovers:
        applicable.append(B5B_BRANCH_RAW_ONLY_RECOVERS)
    elif ordinal_only_recovers:
        applicable.append(B5B_BRANCH_ORDINAL_ONLY_NO_INTRA_RANK)
    elif harness_ok and baseline_reproduced:
        applicable.append(B5B_BRANCH_TIEBREAK_STILL_COLLAPSES)
    science_priority = (
        B5B_BRANCH_HARNESS_OR_INPUT_FAIL,
        B5B_BRANCH_CALIBRATION_FAILURE,
        B5B_BRANCH_TERNARY_TIEBREAK_RECOVERS,
        B5B_BRANCH_Q5_ONLY_RECOVERS,
        B5B_BRANCH_RAW_ONLY_RECOVERS,
        B5B_BRANCH_ORDINAL_ONLY_NO_INTRA_RANK,
        B5B_BRANCH_TIEBREAK_STILL_COLLAPSES,
        B5B_BRANCH_TIEBREAK_BASELINE_REPRO,
    )
    primary_branch = next(
        branch for branch in science_priority if branch in applicable
    )
    return {
        "priority_order": list(B5B_BRANCH_PRIORITY),
        "all_applicable_branches": applicable,
        "primary_branch": primary_branch,
        "sub2_win_branch": B5B_BRANCH_TERNARY_TIEBREAK_RECOVERS,
        "sub2_win": bool(primary_branch == B5B_BRANCH_TERNARY_TIEBREAK_RECOVERS),
        "explicit_non_win_branches": [
            B5B_BRANCH_Q5_ONLY_RECOVERS,
            B5B_BRANCH_RAW_ONLY_RECOVERS,
        ],
    }


def run_b5b_within_q5_family_tiebreak_counterfactual(
    *,
    seed43_receipt_path: str | Path,
    seed29_receipt_path: str | Path,
) -> dict[str, Any]:
    harness_error: str | None = None
    seed43_receipt: LoadedCeilingAuditReceipt | None = None
    seed29_receipt: LoadedCeilingAuditReceipt | None = None
    try:
        seed43_receipt = load_activation_credit_ceiling_audit_receipt(
            seed43_receipt_path,
            seed_label=SEED43_LABEL,
        )
        seed29_receipt = load_activation_credit_ceiling_audit_receipt(
            seed29_receipt_path,
            seed_label=SEED29_LABEL,
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        harness_error = str(exc)
    if (
        harness_error is not None
        or seed43_receipt is None
        or seed29_receipt is None
    ):
        branch = _b5b_emit_branch_classifier(
            harness_ok=False,
            baseline_reproduced=False,
            calibration_failure=False,
            ternary_recovers=False,
            q5_recovers=False,
            raw_recovers=False,
            ordinal_only_recovers=False,
        )
        return {
            "schema_version": B5B_COUNTERFACTUAL_SCHEMA_VERSION,
            "task_id": B5B_TASK_ID,
            "harness_ok": False,
            "harness_error": harness_error,
            "branch_classifier": branch,
            "uses_raw_continuous_inside_bucket": False,
            "success_bar": {
                "primary_receipt_family_auc_max": float(PRIMARY_RECEIPT_FAMILY_AUC_MAX),
                "sub2_win_requires_ternary_quantized_ordinal": True,
            },
        }
    eligibility_spec = _scalar_spec_by_id(B5B_ELIGIBILITY_SCALAR_ID)
    eligibility_direction = _best_seed43_direction(
        seed43_receipt.rows,
        value_fn=eligibility_spec.compute,
    )
    variant_metrics: dict[str, dict[str, Any]] = {}
    aucs_by_variant: dict[str, dict[str, float]] = {}
    for tiebreak_rank_key_id in B5B_TIEBREAK_VARIANT_IDS:
        seed43_auc = _b5b_counterfactual_family_auc(
            seed43_receipt.rows,
            tiebreak_rank_key_id=tiebreak_rank_key_id,
            eligibility_direction=eligibility_direction,
        )
        seed29_auc = _b5b_counterfactual_family_auc(
            seed29_receipt.rows,
            tiebreak_rank_key_id=tiebreak_rank_key_id,
            eligibility_direction=eligibility_direction,
        )
        aucs_by_variant[tiebreak_rank_key_id] = {
            SEED43_LABEL: seed43_auc,
            SEED29_LABEL: seed29_auc,
        }
        metadata = _b5b_variant_metadata(tiebreak_rank_key_id)
        variant_metrics[tiebreak_rank_key_id] = {
            **metadata,
            SEED43_LABEL: {"receipt_family_compressed_auc": seed43_auc},
            SEED29_LABEL: {"receipt_family_compressed_auc": seed29_auc},
            "both_seeds_recover": _b5b_family_auc_recovers(
                {SEED43_LABEL: seed43_auc, SEED29_LABEL: seed29_auc}
            ),
        }
    baseline_aucs = aucs_by_variant[ACTIVATION_CREDIT_TIEBREAK_KEY_CURRENT_RANK]
    baseline_reproduced = _b5b_baseline_anchor_reproduced(baseline_aucs)
    harness_ok = bool(baseline_reproduced)
    calibration_failure = _b5b_within_bucket_eligibility_calibration_failure(
        seed43_receipt.rows,
        seed29_receipt.rows,
        eligibility_direction=eligibility_direction,
    )
    ternary_aucs = aucs_by_variant[
        ACTIVATION_CREDIT_TIEBREAK_KEY_TERNARY_ELIGIBILITY_ORDINAL
    ]
    q5_aucs = aucs_by_variant[ACTIVATION_CREDIT_TIEBREAK_KEY_Q5_ELIGIBILITY_ORDINAL]
    raw_aucs = aucs_by_variant[ACTIVATION_CREDIT_TIEBREAK_KEY_RAW_ELIGIBILITY_FP]
    ordinal_only_aucs = aucs_by_variant[
        ACTIVATION_CREDIT_TIEBREAK_KEY_ORDINAL_ONLY_NO_INTRA_RANK
    ]
    ternary_recovers = _b5b_family_auc_recovers(ternary_aucs)
    q5_recovers = _b5b_family_auc_recovers(q5_aucs)
    raw_recovers = _b5b_family_auc_recovers(raw_aucs)
    ordinal_only_recovers = _b5b_family_auc_recovers(ordinal_only_aucs)
    branch_classifier = _b5b_emit_branch_classifier(
        harness_ok=harness_ok,
        baseline_reproduced=baseline_reproduced,
        calibration_failure=calibration_failure,
        ternary_recovers=ternary_recovers,
        q5_recovers=q5_recovers,
        raw_recovers=raw_recovers,
        ordinal_only_recovers=ordinal_only_recovers,
    )
    q5_to_family_loss = {
        SEED43_LABEL: float(
            q5_aucs[SEED43_LABEL] - baseline_aucs[SEED43_LABEL]
        ),
        SEED29_LABEL: float(
            q5_aucs[SEED29_LABEL] - baseline_aucs[SEED29_LABEL]
        ),
    }
    return {
        "schema_version": B5B_COUNTERFACTUAL_SCHEMA_VERSION,
        "task_id": B5B_TASK_ID,
        "harness_ok": harness_ok,
        "baseline_anchor_reproduced": baseline_reproduced,
        "single_variable": {
            "name": "within_receipt_family_bucket_tiebreak_key",
            "family_bucket": ACTIVATION_CREDIT_PRIMARY_FAMILY_ID,
            "eligibility_scalar_id": B5B_ELIGIBILITY_SCALAR_ID,
            "eligibility_direction": int(eligibility_direction),
        },
        "invariants": {
            "uses_raw_continuous_inside_bucket": False,
            "sub2_success_path_requires_quantized_ordinal": True,
            "no_receipt_mutation": True,
            "cpu_replay_only": True,
        },
        "success_bar": {
            "primary_receipt_family_auc_max": float(PRIMARY_RECEIPT_FAMILY_AUC_MAX),
            "sub2_win_branch": B5B_BRANCH_TERNARY_TIEBREAK_RECOVERS,
            "requires_both_seeds_above_max": True,
        },
        "variant_metrics": variant_metrics,
        "q5_ordinal_to_receipt_family_loss": q5_to_family_loss,
        "branch_classifier": branch_classifier,
        "input_receipts": {
            SEED43_LABEL: {
                "path": seed43_receipt.path,
                "sha256": seed43_receipt.sha256,
            },
            SEED29_LABEL: {
                "path": seed29_receipt.path,
                "sha256": seed29_receipt.sha256,
            },
        },
        "non_claims": [
            "no runtime/sub-2 persistent claim unless BRANCH_TERNARY_TIEBREAK_RECOVERS",
            "BRANCH_Q5_ONLY_RECOVERS is diagnostic >2-bit non-win",
            "BRANCH_RAW_ONLY_RECOVERS is label-leak upper bound non-win",
            "no GPU / trainer wiring / receipt mutation",
        ],
    }


__all__ = [
    "ACTIVATION_CREDIT_CEILING_AUDIT_SCHEMA_VERSION",
    "ACTIVATION_CREDIT_CEILING_AUDIT_TARGET_NAME",
    "B5B_BRANCH_CALIBRATION_FAILURE",
    "B5B_BRANCH_HARNESS_OR_INPUT_FAIL",
    "B5B_BRANCH_ORDINAL_ONLY_NO_INTRA_RANK",
    "B5B_BRANCH_Q5_ONLY_RECOVERS",
    "B5B_BRANCH_RAW_ONLY_RECOVERS",
    "B5B_BRANCH_TERNARY_TIEBREAK_RECOVERS",
    "B5B_BRANCH_TIEBREAK_BASELINE_REPRO",
    "B5B_BRANCH_TIEBREAK_STILL_COLLAPSES",
    "B5B_COUNTERFACTUAL_SCHEMA_VERSION",
    "B5B_TASK_ID",
    "KNOWN_BRANCH4_RAW_AUC_EXPECTATIONS",
    "KNOWN_BRANCH4_RECEIPT_FAMILY_AUC_EXPECTATIONS",
    "LABEL_LEAK_UPPER_BOUND_TAG",
    "SCALAR_SPECS",
    "ScalarSpec",
    "build_activation_credit_ceiling_audit",
    "load_activation_credit_ceiling_audit_receipt",
    "run_b5b_within_q5_family_tiebreak_counterfactual",
]
