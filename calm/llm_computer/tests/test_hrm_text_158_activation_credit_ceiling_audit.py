from __future__ import annotations

import json
from pathlib import Path

import pytest

from calm.hrm_text_158.native_full_stack.activation_credit_ceiling_audit import (
    CALIBRATION_ONLINE_CANDIDATE_QUANTILE,
    CALIBRATION_SEED43_THRESHOLDS_OOS,
    DIAGNOSTIC_FLAT_INDEX_TIE_BREAKER_ID,
    LABEL_LEAK_UPPER_BOUND_TAG,
    _loss_spread_ratio_mirror,
    _positive_improvement_mass_mirror,
    build_activation_credit_ceiling_audit,
)
from scripts.hrm_text_158_activation_credit_ceiling_audit import main as ceiling_audit_main


def _row(
    *,
    candidate_id: str,
    regret: float,
    local_loss_delta: float,
    taylor_benefit: float,
    taylor_q5: int,
    grad_proxy: float,
    delta_weight: float,
    snr: float,
    snr_q5: int,
    diag_fisher: float,
    diag_q5: int,
    current_rank_position: int,
    current_margin_abs: float,
    vote_value: float,
    topology: int,
) -> dict[str, object]:
    return {
        "activation_feature_valid": True,
        "candidate_delta_sign": 1 if delta_weight > 0 else -1,
        "candidate_delta_weight": delta_weight,
        "candidate_id": candidate_id,
        "candidate_loss": 10.0 + regret,
        "current_margin_abs": current_margin_abs,
        "current_rank_position": current_rank_position,
        "diag_fisher": diag_fisher,
        "diagfisher_q5_bin": diag_q5,
        "flat_index": current_rank_position,
        "grad_proxy": grad_proxy,
        "in_target_tie_band": True,
        "local_loss_delta": local_loss_delta,
        "regret_vs_target_tie_band_oracle_top1_local_loss_delta": regret,
        "snr": snr,
        "snr_q5_bin": snr_q5,
        "state_key": "state",
        "taylor_benefit": taylor_benefit,
        "taylor_benefit_q5_bin": taylor_q5,
        "topology_row_block_128": topology,
        "transition_class": "q1|dir1",
        "vote_value": vote_value,
    }


def _receipt_payload(
    *,
    rows: list[dict[str, object]],
    family_auc: float,
    branch_classification: str = "activation_credit_ambiguous_no_branch",
) -> dict[str, object]:
    family_metric = {
        "family_id": "F_taylor_benefit_q5",
        "bucket_count": 2,
        "bucket_cardinality_histogram": {"2": 2},
        "singleton_bucket_count": 0,
        "oracle_best_bucket_candidate_count": 2,
        "oracle_best_bucket_candidate_ids_hash16": "abc123",
        "oracle_best_bucket_fraction": 0.5,
        "oracle_best_bucket_regret_spread_ratio": 0.8,
        "oracle_best_bucket_regret_capture_ratio": 0.4,
        "oracle_best_bucket_top_k_capture_fraction": 0.5,
        "within_band_pairwise_auc_report_only": family_auc,
        "matched_hash_seed_count": 8,
        "matched_hash_null_fraction_gte_observed_bucket_fraction": 1.0,
        "matched_hash_null_fraction_lte_observed_regret_capture_ratio": 0.5,
        "null_control_hash_only": True,
    }
    return {
        "oracle_screen": {
            "branch_classification": branch_classification,
            "compact_summary": {
                "sampled_candidate_table": rows,
                "target_tie_band": {
                    "target_tie_band_id": "voteabs=4|marginabs=4",
                    "band_candidate_count": len(rows),
                    "valid_activation_candidate_count": len(rows),
                },
                "family_metrics": {
                    "primary_family_id": "F_taylor_benefit_q5",
                    "metrics_by_family_id": {
                        "F_taylor_benefit_q5": family_metric,
                    },
                },
            },
        }
    }


def _write_receipt(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def test_activation_credit_ceiling_audit_classifies_receipt_family_bucket_tiebreak_loss(
    tmp_path: Path,
) -> None:
    seed43_rows = [
        _row(
            candidate_id="A",
            regret=0.0,
            local_loss_delta=-0.40,
            taylor_benefit=0.90,
            taylor_q5=2,
            grad_proxy=-0.90,
            delta_weight=1.0,
            snr=0.80,
            snr_q5=1,
            diag_fisher=0.10,
            diag_q5=0,
            current_rank_position=0,
            current_margin_abs=4.0,
            vote_value=4.0,
            topology=0,
        ),
        _row(
            candidate_id="B",
            regret=0.1,
            local_loss_delta=-0.30,
            taylor_benefit=0.80,
            taylor_q5=1,
            grad_proxy=-0.80,
            delta_weight=1.0,
            snr=0.70,
            snr_q5=1,
            diag_fisher=0.20,
            diag_q5=0,
            current_rank_position=1,
            current_margin_abs=4.0,
            vote_value=4.0,
            topology=0,
        ),
        _row(
            candidate_id="C",
            regret=0.2,
            local_loss_delta=-0.20,
            taylor_benefit=0.70,
            taylor_q5=1,
            grad_proxy=-0.70,
            delta_weight=1.0,
            snr=0.60,
            snr_q5=0,
            diag_fisher=0.30,
            diag_q5=1,
            current_rank_position=2,
            current_margin_abs=4.0,
            vote_value=-4.0,
            topology=1,
        ),
        _row(
            candidate_id="D",
            regret=0.3,
            local_loss_delta=-0.10,
            taylor_benefit=0.60,
            taylor_q5=0,
            grad_proxy=-0.60,
            delta_weight=1.0,
            snr=0.50,
            snr_q5=0,
            diag_fisher=0.40,
            diag_q5=1,
            current_rank_position=3,
            current_margin_abs=4.0,
            vote_value=-4.0,
            topology=1,
        ),
    ]
    seed29_rows = [
        _row(
            candidate_id="A",
            regret=0.0,
            local_loss_delta=-0.35,
            taylor_benefit=0.88,
            taylor_q5=2,
            grad_proxy=-0.88,
            delta_weight=1.0,
            snr=0.78,
            snr_q5=1,
            diag_fisher=0.11,
            diag_q5=0,
            current_rank_position=0,
            current_margin_abs=4.0,
            vote_value=4.0,
            topology=0,
        ),
        _row(
            candidate_id="B",
            regret=0.1,
            local_loss_delta=-0.25,
            taylor_benefit=0.77,
            taylor_q5=1,
            grad_proxy=-0.77,
            delta_weight=1.0,
            snr=0.68,
            snr_q5=1,
            diag_fisher=0.21,
            diag_q5=0,
            current_rank_position=1,
            current_margin_abs=4.0,
            vote_value=4.0,
            topology=0,
        ),
        _row(
            candidate_id="C",
            regret=0.2,
            local_loss_delta=-0.15,
            taylor_benefit=0.69,
            taylor_q5=1,
            grad_proxy=-0.69,
            delta_weight=1.0,
            snr=0.58,
            snr_q5=0,
            diag_fisher=0.31,
            diag_q5=1,
            current_rank_position=2,
            current_margin_abs=4.0,
            vote_value=-4.0,
            topology=1,
        ),
        _row(
            candidate_id="D",
            regret=0.3,
            local_loss_delta=-0.05,
            taylor_benefit=0.55,
            taylor_q5=0,
            grad_proxy=-0.55,
            delta_weight=1.0,
            snr=0.48,
            snr_q5=0,
            diag_fisher=0.41,
            diag_q5=1,
            current_rank_position=3,
            current_margin_abs=4.0,
            vote_value=-4.0,
            topology=1,
        ),
    ]
    seed43_path = tmp_path / "seed43.json"
    seed29_path = tmp_path / "seed29.json"
    _write_receipt(seed43_path, _receipt_payload(rows=seed43_rows, family_auc=0.60))
    _write_receipt(seed29_path, _receipt_payload(rows=seed29_rows, family_auc=0.62))

    receipt = build_activation_credit_ceiling_audit(
        seed43_receipt_path=seed43_path,
        seed29_receipt_path=seed29_path,
    )

    primary = receipt["scalar_results"]["taylor_benefit"]
    assert primary["raw_continuous"]["fixed_direction"] == 1
    assert primary["q5_bin_index"]["fixed_direction"] == 1
    assert primary["raw_continuous"]["seed43"]["oracle_best_rank"] == 1
    assert primary["raw_continuous"]["seed29"]["oracle_best_rank"] == 1
    assert primary["q5_bin_index"]["seed43"]["oracle_best_rank"] == 1
    assert primary["q5_bin_index"]["seed29"]["oracle_best_rank"] == 1
    assert (
        receipt["primary_loss_decomposition"]["classification_label"]
        == "receipt_family_bucket_tiebreak_loss"
    )
    assert (
        receipt["primary_loss_decomposition"]["continuous_to_q5_ordinal_loss"]["seed43"]
        < 0.20
    )
    assert (
        receipt["primary_loss_decomposition"]["q5_ordinal_to_receipt_family_loss"]["seed43"]
        >= 0.10
    )
    leak = receipt["scalar_results"][
        f"{LABEL_LEAK_UPPER_BOUND_TAG}__neg_local_loss_delta"
    ]
    assert leak["decision_authority_allowed"] is False
    assert leak["category"] == LABEL_LEAK_UPPER_BOUND_TAG
    assert all(
        scalar_id != f"{LABEL_LEAK_UPPER_BOUND_TAG}__neg_local_loss_delta"
        for scalar_id in receipt["decision_authorized_scalar_ids"]
    )
    sweep = receipt["sub2_ordinal_sweep"]
    assert sweep["primary_scalar_id"] == "taylor_benefit"
    assert sweep["fixed_direction"] == 1
    assert sweep["levels"] == [5, 4, 3, 2]
    assert set(sweep["calibration_classes"]) == {
        CALIBRATION_ONLINE_CANDIDATE_QUANTILE,
        CALIBRATION_SEED43_THRESHOLDS_OOS,
    }
    ternary_online = sweep["results"]["3"][CALIBRATION_ONLINE_CANDIDATE_QUANTILE]
    assert ternary_online["both_seeds"]["top_bucket_contains_oracle"] is True
    assert ternary_online["both_seeds"]["unique_ordinal_top1"] is False
    assert (
        sweep["label_decision"]["primary_label"]
        == "ternary_sub2_ordinal_shortlist_success_needs_tiebreak"
    )
    assert (
        "ternary_sub2_ordinal_shortlist_success_needs_tiebreak"
        in sweep["label_decision"]["all_applicable_labels"]
    )
    assert sweep["primary_rule"]["uses_raw_continuous_inside_bucket"] is False
    assert (
        ternary_online["seed43"]["diagnostic_tiebreak"]["tie_breaker_id"]
        == DIAGNOSTIC_FLAT_INDEX_TIE_BREAKER_ID
    )
    assert (
        ternary_online["seed43"]["diagnostic_tiebreak"]["primary_success_allowed"]
        is False
    )
    b7a = receipt["sub2_tiebreak_sidecar_sweep"]
    assert b7a["source_bucket"]["level"] == 3
    assert b7a["budget_policy"]["strict_additive_default"] is True
    assert (
        b7a["label_decision"]["proxy_caveat_label"]
        == "b7a_proxy_only_b7b_required"
    )
    assert "b7a_proxy_only_b7b_required" in b7a["label_decision"][
        "all_applicable_labels"
    ]
    taylor_aggregate = b7a["aggregate_results"]["transient_fp_scalar:taylor_benefit"]
    assert taylor_aggregate["observation_count"] == 4
    assert taylor_aggregate["unique_selects_oracle_all"] is True
    assert taylor_aggregate["any_not_new_evidence"] is True
    assert taylor_aggregate["primary_success_allowed_all"] is False
    abs_grad_aggregate = b7a["aggregate_results"]["transient_fp_scalar:abs_grad_proxy"]
    assert abs_grad_aggregate["any_same_rank_as_bucket"] is True
    assert abs_grad_aggregate["primary_success_allowed_all"] is False
    diagnostic_observations = [
        observation
        for observation in b7a["observations"]
        if observation["tier"] == "zero_extra_state_diagnostic"
    ]
    assert diagnostic_observations
    assert all(
        observation["diagnostic_only"] is True
        and observation["primary_success_allowed"] is False
        for observation in diagnostic_observations
    )


def test_activation_credit_ceiling_audit_fails_closed_on_row_count_mismatch(
    tmp_path: Path,
) -> None:
    rows = [
        _row(
            candidate_id="A",
            regret=0.0,
            local_loss_delta=-0.1,
            taylor_benefit=0.2,
            taylor_q5=0,
            grad_proxy=-0.2,
            delta_weight=1.0,
            snr=0.3,
            snr_q5=0,
            diag_fisher=0.1,
            diag_q5=0,
            current_rank_position=0,
            current_margin_abs=4.0,
            vote_value=4.0,
            topology=0,
        )
    ]
    payload = _receipt_payload(rows=rows, family_auc=0.5)
    payload["oracle_screen"]["compact_summary"]["target_tie_band"][
        "valid_activation_candidate_count"
    ] = 2
    seed43_path = tmp_path / "seed43.json"
    seed29_path = tmp_path / "seed29.json"
    _write_receipt(seed43_path, payload)
    _write_receipt(seed29_path, _receipt_payload(rows=rows, family_auc=0.5))

    with pytest.raises(ValueError, match="valid in-band row count mismatch"):
        build_activation_credit_ceiling_audit(
            seed43_receipt_path=seed43_path,
            seed29_receipt_path=seed29_path,
        )


def test_activation_credit_ceiling_audit_q5_bin_index_uses_raw_fixed_direction(
    tmp_path: Path,
) -> None:
    seed43_rows = [
        _row(
            candidate_id="A",
            regret=0.0,
            local_loss_delta=-0.40,
            taylor_benefit=0.10,
            taylor_q5=3,
            grad_proxy=-0.10,
            delta_weight=1.0,
            snr=0.60,
            snr_q5=1,
            diag_fisher=0.20,
            diag_q5=0,
            current_rank_position=0,
            current_margin_abs=4.0,
            vote_value=4.0,
            topology=0,
        ),
        _row(
            candidate_id="B",
            regret=0.1,
            local_loss_delta=-0.30,
            taylor_benefit=0.20,
            taylor_q5=2,
            grad_proxy=-0.20,
            delta_weight=1.0,
            snr=0.50,
            snr_q5=1,
            diag_fisher=0.30,
            diag_q5=0,
            current_rank_position=1,
            current_margin_abs=4.0,
            vote_value=4.0,
            topology=0,
        ),
        _row(
            candidate_id="C",
            regret=0.2,
            local_loss_delta=-0.20,
            taylor_benefit=0.30,
            taylor_q5=1,
            grad_proxy=-0.30,
            delta_weight=1.0,
            snr=0.40,
            snr_q5=0,
            diag_fisher=0.40,
            diag_q5=1,
            current_rank_position=2,
            current_margin_abs=4.0,
            vote_value=-4.0,
            topology=1,
        ),
        _row(
            candidate_id="D",
            regret=0.3,
            local_loss_delta=-0.10,
            taylor_benefit=0.40,
            taylor_q5=0,
            grad_proxy=-0.40,
            delta_weight=1.0,
            snr=0.30,
            snr_q5=0,
            diag_fisher=0.50,
            diag_q5=1,
            current_rank_position=3,
            current_margin_abs=4.0,
            vote_value=-4.0,
            topology=1,
        ),
    ]
    seed29_rows = [
        _row(
            candidate_id="A",
            regret=0.0,
            local_loss_delta=-0.36,
            taylor_benefit=0.12,
            taylor_q5=3,
            grad_proxy=-0.12,
            delta_weight=1.0,
            snr=0.58,
            snr_q5=1,
            diag_fisher=0.21,
            diag_q5=0,
            current_rank_position=0,
            current_margin_abs=4.0,
            vote_value=4.0,
            topology=0,
        ),
        _row(
            candidate_id="B",
            regret=0.1,
            local_loss_delta=-0.28,
            taylor_benefit=0.22,
            taylor_q5=2,
            grad_proxy=-0.22,
            delta_weight=1.0,
            snr=0.48,
            snr_q5=1,
            diag_fisher=0.31,
            diag_q5=0,
            current_rank_position=1,
            current_margin_abs=4.0,
            vote_value=4.0,
            topology=0,
        ),
        _row(
            candidate_id="C",
            regret=0.2,
            local_loss_delta=-0.18,
            taylor_benefit=0.32,
            taylor_q5=1,
            grad_proxy=-0.32,
            delta_weight=1.0,
            snr=0.38,
            snr_q5=0,
            diag_fisher=0.41,
            diag_q5=1,
            current_rank_position=2,
            current_margin_abs=4.0,
            vote_value=-4.0,
            topology=1,
        ),
        _row(
            candidate_id="D",
            regret=0.3,
            local_loss_delta=-0.08,
            taylor_benefit=0.42,
            taylor_q5=0,
            grad_proxy=-0.42,
            delta_weight=1.0,
            snr=0.28,
            snr_q5=0,
            diag_fisher=0.51,
            diag_q5=1,
            current_rank_position=3,
            current_margin_abs=4.0,
            vote_value=-4.0,
            topology=1,
        ),
    ]
    seed43_path = tmp_path / "seed43.json"
    seed29_path = tmp_path / "seed29.json"
    _write_receipt(seed43_path, _receipt_payload(rows=seed43_rows, family_auc=0.55))
    _write_receipt(seed29_path, _receipt_payload(rows=seed29_rows, family_auc=0.56))

    receipt = build_activation_credit_ceiling_audit(
        seed43_receipt_path=seed43_path,
        seed29_receipt_path=seed29_path,
    )

    primary = receipt["scalar_results"]["taylor_benefit"]
    assert primary["decision_authority_allowed"] is True
    assert receipt["decision_authorized_scalar_ids"].count("taylor_benefit") == 1
    assert primary["raw_continuous"]["fixed_direction"] == -1
    assert primary["q5_bin_index"]["fixed_direction"] == -1
    assert primary["q5_bin_index"]["reverse_direction"]["direction"] == 1
    assert primary["raw_continuous"]["seed43"]["oracle_best_rank"] == 1
    assert primary["raw_continuous"]["seed29"]["oracle_best_rank"] == 1
    assert primary["q5_bin_index"]["seed43"]["oracle_best_rank"] == 4
    assert primary["q5_bin_index"]["seed29"]["oracle_best_rank"] == 4
    assert primary["q5_bin_index"]["reverse_direction"]["seed43"]["oracle_best_rank"] == 1
    assert primary["q5_bin_index"]["reverse_direction"]["seed29"]["oracle_best_rank"] == 1
    assert (
        primary["q5_bin_index"]["seed43"]["auc"]
        < primary["q5_bin_index"]["reverse_direction"]["seed43"]["auc"]
    )
    assert (
        primary["q5_bin_index"]["seed29"]["auc"]
        < primary["q5_bin_index"]["reverse_direction"]["seed29"]["auc"]
    )


def test_sub2_ordinal_sweep_diagnostic_tiebreak_does_not_use_raw_score(
    tmp_path: Path,
) -> None:
    rows = [
        _row(
            candidate_id="oracle_lower_score",
            regret=0.0,
            local_loss_delta=-0.30,
            taylor_benefit=0.80,
            taylor_q5=2,
            grad_proxy=-0.80,
            delta_weight=1.0,
            snr=0.70,
            snr_q5=1,
            diag_fisher=0.10,
            diag_q5=0,
            current_rank_position=0,
            current_margin_abs=4.0,
            vote_value=4.0,
            topology=0,
        ),
        _row(
            candidate_id="higher_raw_score",
            regret=0.1,
            local_loss_delta=-0.20,
            taylor_benefit=0.90,
            taylor_q5=2,
            grad_proxy=-0.90,
            delta_weight=1.0,
            snr=0.60,
            snr_q5=1,
            diag_fisher=0.20,
            diag_q5=0,
            current_rank_position=1,
            current_margin_abs=4.0,
            vote_value=4.0,
            topology=0,
        ),
        _row(
            candidate_id="low",
            regret=0.2,
            local_loss_delta=-0.10,
            taylor_benefit=0.10,
            taylor_q5=0,
            grad_proxy=-0.10,
            delta_weight=1.0,
            snr=0.50,
            snr_q5=0,
            diag_fisher=0.30,
            diag_q5=1,
            current_rank_position=2,
            current_margin_abs=4.0,
            vote_value=-4.0,
            topology=1,
        ),
    ]
    seed43_path = tmp_path / "seed43.json"
    seed29_path = tmp_path / "seed29.json"
    _write_receipt(seed43_path, _receipt_payload(rows=rows, family_auc=0.55))
    _write_receipt(seed29_path, _receipt_payload(rows=rows, family_auc=0.56))

    receipt = build_activation_credit_ceiling_audit(
        seed43_receipt_path=seed43_path,
        seed29_receipt_path=seed29_path,
    )

    level2 = receipt["sub2_ordinal_sweep"]["results"]["2"][
        CALIBRATION_ONLINE_CANDIDATE_QUANTILE
    ]["seed43"]
    assert level2["top_bucket_contains_oracle"] is True
    assert level2["top_bucket_size"] == 2
    assert level2["unique_ordinal_top1"] is False
    assert level2["diagnostic_tiebreak"]["extra_state_bits"] == 0
    assert level2["diagnostic_tiebreak"]["credit_mechanistic"] is False
    assert level2["diagnostic_tiebreak"]["primary_success_allowed"] is False
    assert level2["diagnostic_tiebreak"]["selected_candidate_id"] == "oracle_lower_score"
    assert level2["diagnostic_tiebreak"]["deterministic_tiebreak_top1"] is True


def test_sub2_ordinal_sweep_label_priority_surfaces_calibration_failure(
    tmp_path: Path,
) -> None:
    seed43_rows = [
        _row(
            candidate_id="oracle",
            regret=0.0,
            local_loss_delta=-0.30,
            taylor_benefit=20.0,
            taylor_q5=4,
            grad_proxy=-20.0,
            delta_weight=1.0,
            snr=0.90,
            snr_q5=2,
            diag_fisher=0.10,
            diag_q5=0,
            current_rank_position=0,
            current_margin_abs=4.0,
            vote_value=4.0,
            topology=0,
        ),
        _row(
            candidate_id="middle",
            regret=0.1,
            local_loss_delta=-0.20,
            taylor_benefit=10.0,
            taylor_q5=2,
            grad_proxy=-10.0,
            delta_weight=1.0,
            snr=0.80,
            snr_q5=1,
            diag_fisher=0.20,
            diag_q5=0,
            current_rank_position=1,
            current_margin_abs=4.0,
            vote_value=4.0,
            topology=0,
        ),
        _row(
            candidate_id="low",
            regret=0.2,
            local_loss_delta=-0.10,
            taylor_benefit=0.0,
            taylor_q5=0,
            grad_proxy=-0.10,
            delta_weight=1.0,
            snr=0.70,
            snr_q5=0,
            diag_fisher=0.30,
            diag_q5=1,
            current_rank_position=2,
            current_margin_abs=4.0,
            vote_value=-4.0,
            topology=1,
        ),
    ]
    seed29_rows = [
        _row(
            candidate_id="oracle",
            regret=0.0,
            local_loss_delta=-0.30,
            taylor_benefit=5.0,
            taylor_q5=2,
            grad_proxy=-5.0,
            delta_weight=1.0,
            snr=0.90,
            snr_q5=2,
            diag_fisher=0.10,
            diag_q5=0,
            current_rank_position=0,
            current_margin_abs=4.0,
            vote_value=4.0,
            topology=0,
        ),
        _row(
            candidate_id="oos_higher",
            regret=0.1,
            local_loss_delta=-0.20,
            taylor_benefit=15.0,
            taylor_q5=4,
            grad_proxy=-15.0,
            delta_weight=1.0,
            snr=0.80,
            snr_q5=1,
            diag_fisher=0.20,
            diag_q5=0,
            current_rank_position=1,
            current_margin_abs=4.0,
            vote_value=4.0,
            topology=0,
        ),
        _row(
            candidate_id="low",
            regret=0.2,
            local_loss_delta=-0.10,
            taylor_benefit=-5.0,
            taylor_q5=0,
            grad_proxy=5.0,
            delta_weight=1.0,
            snr=0.70,
            snr_q5=0,
            diag_fisher=0.30,
            diag_q5=1,
            current_rank_position=2,
            current_margin_abs=4.0,
            vote_value=-4.0,
            topology=1,
        ),
    ]
    seed43_path = tmp_path / "seed43.json"
    seed29_path = tmp_path / "seed29.json"
    _write_receipt(seed43_path, _receipt_payload(rows=seed43_rows, family_auc=0.55))
    _write_receipt(seed29_path, _receipt_payload(rows=seed29_rows, family_auc=0.56))

    receipt = build_activation_credit_ceiling_audit(
        seed43_receipt_path=seed43_path,
        seed29_receipt_path=seed29_path,
    )

    label_decision = receipt["sub2_ordinal_sweep"]["label_decision"]
    assert label_decision["primary_label"] == "calibration_failure"
    assert "calibration_failure" in label_decision["all_applicable_labels"]
    assert (
        "ternary_sub2_ordinal_shortlist_success_needs_tiebreak"
        in label_decision["all_applicable_labels"]
    )
    level3 = receipt["sub2_ordinal_sweep"]["results"]["3"]
    assert level3[CALIBRATION_ONLINE_CANDIDATE_QUANTILE]["seed29"][
        "top_bucket_contains_oracle"
    ] is True
    assert level3[CALIBRATION_SEED43_THRESHOLDS_OOS]["seed29"][
        "top_bucket_contains_oracle"
    ] is False


def test_b7a_persistent_sidecar_unique_selection_stays_over_budget(
    tmp_path: Path,
) -> None:
    rows = [
        _row(
            candidate_id="oracle",
            regret=0.0,
            local_loss_delta=-0.40,
            taylor_benefit=0.90,
            taylor_q5=2,
            grad_proxy=-0.90,
            delta_weight=1.0,
            snr=0.20,
            snr_q5=0,
            diag_fisher=0.90,
            diag_q5=2,
            current_rank_position=0,
            current_margin_abs=4.0,
            vote_value=4.0,
            topology=0,
        ),
        _row(
            candidate_id="other_top_bucket",
            regret=0.1,
            local_loss_delta=-0.30,
            taylor_benefit=0.80,
            taylor_q5=2,
            grad_proxy=-0.80,
            delta_weight=1.0,
            snr=0.30,
            snr_q5=0,
            diag_fisher=0.10,
            diag_q5=0,
            current_rank_position=1,
            current_margin_abs=4.0,
            vote_value=4.0,
            topology=0,
        ),
        _row(
            candidate_id="low_a",
            regret=0.2,
            local_loss_delta=-0.20,
            taylor_benefit=0.10,
            taylor_q5=0,
            grad_proxy=-0.10,
            delta_weight=1.0,
            snr=0.40,
            snr_q5=1,
            diag_fisher=0.05,
            diag_q5=0,
            current_rank_position=2,
            current_margin_abs=4.0,
            vote_value=-4.0,
            topology=1,
        ),
        _row(
            candidate_id="low_b",
            regret=0.3,
            local_loss_delta=-0.10,
            taylor_benefit=0.00,
            taylor_q5=0,
            grad_proxy=-0.01,
            delta_weight=1.0,
            snr=0.50,
            snr_q5=1,
            diag_fisher=0.00,
            diag_q5=0,
            current_rank_position=3,
            current_margin_abs=4.0,
            vote_value=-4.0,
            topology=1,
        ),
    ]
    seed43_path = tmp_path / "seed43.json"
    seed29_path = tmp_path / "seed29.json"
    _write_receipt(seed43_path, _receipt_payload(rows=rows, family_auc=0.55))
    _write_receipt(seed29_path, _receipt_payload(rows=rows, family_auc=0.56))

    receipt = build_activation_credit_ceiling_audit(
        seed43_receipt_path=seed43_path,
        seed29_receipt_path=seed29_path,
    )

    b7a = receipt["sub2_tiebreak_sidecar_sweep"]
    resolver_id = (
        "persistent_ordinal_sidecar:diag_fisher:level3:"
        "within_bucket_online_quantile"
    )
    aggregate = b7a["aggregate_results"][resolver_id]
    assert aggregate["unique_selects_oracle_all"] is True
    assert aggregate["qualifies_under_persistent_budget_all"] is False
    assert aggregate["max_combined_persistent_bits"] > 2.0
    assert "persistent_sidecar_success_exceeds_budget" in b7a["label_decision"][
        "all_applicable_labels"
    ]
    observations = [
        observation
        for observation in b7a["observations"]
        if observation["resolver_id"] == resolver_id
    ]
    assert {observation["calibration_scope"] for observation in observations} == {
        "within_bucket_online_quantile"
    }
    assert all(
        observation["budget_model_id"]
        == "strict_dense_additive_ternary_primary_plus_sidecar_v0"
        and observation["qualifies_under_persistent_budget"] is False
        and observation["calibration_state_bits_or_shared_cost"]
        == "adaptive_threshold_state_not_free_under_strict_default"
        for observation in observations
    )


def test_sub2_ordinal_sweep_regret_metric_mirrors_oracle_runner_formula() -> None:
    rows = [
        {"candidate_id": "A", "local_loss_delta": -0.30},
        {"candidate_id": "B", "local_loss_delta": -0.10},
        {"candidate_id": "C", "local_loss_delta": 0.05},
    ]

    assert _positive_improvement_mass_mirror(rows) == pytest.approx(0.40)
    assert _loss_spread_ratio_mirror(
        rows,
        oracle_top1_delta=-0.30,
    ) == pytest.approx((0.05 - -0.30) / 0.30)
    assert _loss_spread_ratio_mirror(
        [{"candidate_id": "flat", "local_loss_delta": 0.0}],
        oracle_top1_delta=0.0,
    ) == 0.0
    assert _loss_spread_ratio_mirror(
        [{"candidate_id": "spread", "local_loss_delta": 0.1}],
        oracle_top1_delta=None,
    ) is None


def test_activation_credit_ceiling_audit_cli_writes_json_and_flags_leak(
    tmp_path: Path,
    capsys,
) -> None:
    rows = [
        _row(
            candidate_id="A",
            regret=0.0,
            local_loss_delta=-0.2,
            taylor_benefit=0.4,
            taylor_q5=0,
            grad_proxy=-0.4,
            delta_weight=1.0,
            snr=0.5,
            snr_q5=0,
            diag_fisher=0.2,
            diag_q5=0,
            current_rank_position=0,
            current_margin_abs=4.0,
            vote_value=4.0,
            topology=0,
        ),
        _row(
            candidate_id="B",
            regret=0.1,
            local_loss_delta=-0.1,
            taylor_benefit=0.3,
            taylor_q5=0,
            grad_proxy=-0.3,
            delta_weight=1.0,
            snr=0.4,
            snr_q5=0,
            diag_fisher=0.3,
            diag_q5=0,
            current_rank_position=1,
            current_margin_abs=4.0,
            vote_value=-4.0,
            topology=1,
        ),
    ]
    seed43_path = tmp_path / "seed43.json"
    seed29_path = tmp_path / "seed29.json"
    out = tmp_path / "audit.json"
    _write_receipt(seed43_path, _receipt_payload(rows=rows, family_auc=0.55))
    _write_receipt(seed29_path, _receipt_payload(rows=rows, family_auc=0.56))

    exit_code = ceiling_audit_main(
        [
            "--seed43-receipt",
            str(seed43_path),
            "--seed29-receipt",
            str(seed29_path),
            "--json-out",
            str(out),
        ],
    )

    assert exit_code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema_version"]
    assert payload["input_receipts"]["seed43"]["path"] == str(seed43_path)
    leak_key = f"{LABEL_LEAK_UPPER_BOUND_TAG}__neg_local_loss_delta"
    assert payload["scalar_results"][leak_key]["decision_authority_allowed"] is False
    stdout_payload = json.loads(capsys.readouterr().out)
    assert stdout_payload["target_name"] == payload["target_name"]
