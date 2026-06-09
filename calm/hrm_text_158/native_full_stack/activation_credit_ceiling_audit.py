"""Tracked raw-vs-compressed activation-credit ceiling audit over receipt artifacts."""
from __future__ import annotations

import json
import math
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
    ORACLE_SCREEN_IMPROVEMENT_EPS,
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
        },
        "decision_authorized_scalar_ids": [
            spec.scalar_id for spec in SCALAR_SPECS if spec.decision_authority_allowed
        ],
        "label_leak_scalar_ids": [
            spec.scalar_id for spec in SCALAR_SPECS if not spec.decision_authority_allowed
        ],
        "scalar_results": scalar_results,
        "primary_loss_decomposition": primary_loss,
        "known_branch4_anchor_reproduction": _known_anchor_reproduction(
            scalar_results,
        ),
        "non_claims": [
            "no learner/runtime/full-sub2 success claim",
            "no eligibility build decision",
            "label-leak upper-bound rows have zero decision authority",
        ],
    }


__all__ = [
    "ACTIVATION_CREDIT_CEILING_AUDIT_SCHEMA_VERSION",
    "ACTIVATION_CREDIT_CEILING_AUDIT_TARGET_NAME",
    "KNOWN_BRANCH4_RAW_AUC_EXPECTATIONS",
    "KNOWN_BRANCH4_RECEIPT_FAMILY_AUC_EXPECTATIONS",
    "LABEL_LEAK_UPPER_BOUND_TAG",
    "SCALAR_SPECS",
    "ScalarSpec",
    "build_activation_credit_ceiling_audit",
    "load_activation_credit_ceiling_audit_receipt",
]
