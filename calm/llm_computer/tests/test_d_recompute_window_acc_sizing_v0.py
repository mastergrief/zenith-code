from __future__ import annotations

import copy

import pytest

from calm.hrm_text_158.native_full_stack.d_recompute_window_acc_sizing import (
    DEFAULT_DECAY_GRID,
    SIZING_VERDICT_INCONCLUSIVE,
    SIZING_VERDICT_RECOMMENDED_LAW,
    SIZING_VERDICT_SIZED_NOT_SUB2,
    VERDICT_SCOPE_ENVELOPE_MODEL_ONLY,
    build_conservative_envelope_payload,
    compare_envelope_pattern_bytes,
    effective_acc_budget_bpw,
    measure_envelope_inclusive_bpw,
    size_acc_bpw_from_horizon_growth,
    quantile_size_acc_bpw_from_horizon_growth,
    QUANTILE_SIZING_VERDICT_DETERMINATE_NOT_UNDER_BUDGET,
    QUANTILE_SIZING_VERDICT_DETERMINATE_SUB2_CANDIDATE,
    QUANTILE_SIZING_VERDICT_INCONCLUSIVE,
    simulate_decay_worst_case_surface,
)
from calm.hrm_text_158.native_full_stack.d_recompute_window_horizon_analyzer import (
    GROWTH_INCONCLUSIVE_COST_OR_COVERAGE,
    GROWTH_LINEAR_SIZED_WITH_DECAY,
    GROWTH_PLATEAU_SIZED,
    GROWTH_RIGHT_CENSORED_LOWER_BOUND,
)
from calm.hrm_text_158.native_full_stack.d_recompute_window_stratified_selector import (
    COVERAGE_TIER_REPRESENTATIVE,
)
from calm.hrm_text_158.native_full_stack.persistent_state_budget import (
    R4B_Q_PHYSICAL_BITS_PER_WEIGHT_BASE3,
    TARGET_PHYSICAL_BITS_PER_WEIGHT,
)
from calm.hrm_text_158.native_full_stack.sub2_carrier_family_discriminator import (
    ACC_BUDGET_BPW_UNDER_BASE3_Q,
    DECLARED_Q_BPW_BASE3,
)


def _sized_summary(*, kworst: float, k99: float, horizon_h: int = 100) -> dict:
    return {
        "horizon_h": int(horizon_h),
        "kworst_weighted": float(kworst),
        "k99_weighted": float(k99),
        "lane_count": 4,
        "gapped_lane_count": 0,
        "parity_fail_count": 0,
        "right_censor_rate": 0.0,
    }


def _assert_envelope_model_contract(result: dict) -> None:
    assert result["verdict_scope"] == VERDICT_SCOPE_ENVELOPE_MODEL_ONLY
    assert result["not_in_vivo_bound"] is True
    assert result["requires_slice5_live_validation"] is True


def _horizon_growth(
    *,
    growth_branch: str,
    kworst: float = 20.0,
    k99: float = 18.0,
) -> dict:
    return {
        "growth_branch": growth_branch,
        "summaries_by_h": {
            25: _sized_summary(kworst=8.0, k99=8.0, horizon_h=25),
            50: _sized_summary(kworst=14.0, k99=14.0, horizon_h=50),
            100: _sized_summary(kworst=float(kworst), k99=float(k99), horizon_h=100),
        },
    }


def test_effective_acc_budget_tightens_when_q_scale_exceeds_base3() -> None:
    default_budget = effective_acc_budget_bpw(measured_q_scale_bpw=float(DECLARED_Q_BPW_BASE3))
    assert default_budget == pytest.approx(float(ACC_BUDGET_BPW_UNDER_BASE3_Q))
    tight = effective_acc_budget_bpw(measured_q_scale_bpw=1.75)
    assert tight < float(ACC_BUDGET_BPW_UNDER_BASE3_Q)
    assert tight == pytest.approx(float(TARGET_PHYSICAL_BITS_PER_WEIGHT) - 1.75)


def test_growth_branch_gating_right_censored_is_inconclusive() -> None:
    result = size_acc_bpw_from_horizon_growth(
        _horizon_growth(growth_branch=GROWTH_RIGHT_CENSORED_LOWER_BOUND),
        numel_for_bpw=256,
    )
    assert result["sizing_verdict"] == SIZING_VERDICT_INCONCLUSIVE
    assert result["reason"] == "right_censored_growth_branch"
    _assert_envelope_model_contract(result)


def test_growth_branch_gating_inconclusive_coverage_is_inconclusive() -> None:
    result = size_acc_bpw_from_horizon_growth(
        _horizon_growth(growth_branch=GROWTH_INCONCLUSIVE_COST_OR_COVERAGE),
        numel_for_bpw=256,
    )
    assert result["sizing_verdict"] == SIZING_VERDICT_INCONCLUSIVE
    assert result["reason"] == "growth_branch_not_sized"


def test_plateau_branch_attempts_sizing_with_window_k_from_kworst() -> None:
    result = size_acc_bpw_from_horizon_growth(
        _horizon_growth(growth_branch=GROWTH_PLATEAU_SIZED, kworst=20.0, k99=18.0),
        numel_for_bpw=512,
    )
    assert result["window_k"] == 20
    assert result["sizing_verdict"] in {
        SIZING_VERDICT_RECOMMENDED_LAW,
        SIZING_VERDICT_SIZED_NOT_SUB2,
    }
    assert len(result["grid_rows"]) == len(DEFAULT_DECAY_GRID)
    _assert_envelope_model_contract(result)


def test_recommended_law_is_envelope_candidate_pending_slice5_validation() -> None:
    result = size_acc_bpw_from_horizon_growth(
        _horizon_growth(growth_branch=GROWTH_PLATEAU_SIZED, kworst=5.0, k99=5.0),
        numel_for_bpw=1024,
        measured_q_scale_bpw=float(R4B_Q_PHYSICAL_BITS_PER_WEIGHT_BASE3),
    )
    assert result["sizing_verdict"] == SIZING_VERDICT_RECOMMENDED_LAW
    _assert_envelope_model_contract(result)
    best = float(result["best_grid_row"]["inclusive_acc_bpw"])
    budget = float(result["effective_acc_budget_bpw"])
    assert best < budget


def test_envelope_model_is_not_claimed_absolute_in_vivo_bound() -> None:
    result = size_acc_bpw_from_horizon_growth(
        _horizon_growth(growth_branch=GROWTH_PLATEAU_SIZED, kworst=20.0, k99=18.0),
        numel_for_bpw=512,
    )
    _assert_envelope_model_contract(result)
    assert result["verdict_scope"] == "envelope_model_only"


def test_linear_branch_attempts_sizing() -> None:
    result = size_acc_bpw_from_horizon_growth(
        _horizon_growth(growth_branch=GROWTH_LINEAR_SIZED_WITH_DECAY, kworst=30.0, k99=28.0),
        numel_for_bpw=512,
    )
    assert result["window_k"] == 30
    assert result["sizing_verdict"] in {
        SIZING_VERDICT_RECOMMENDED_LAW,
        SIZING_VERDICT_SIZED_NOT_SUB2,
        SIZING_VERDICT_INCONCLUSIVE,
    }


def test_censored_kworst_at_horizon_is_inconclusive() -> None:
    result = size_acc_bpw_from_horizon_growth(
        _horizon_growth(
            growth_branch=GROWTH_PLATEAU_SIZED,
            kworst=100.0,
            k99=95.0,
        ),
        numel_for_bpw=256,
    )
    assert result["sizing_verdict"] == SIZING_VERDICT_INCONCLUSIVE
    assert result["reason"] == "censored_or_missing_kworst_at_horizon"


def test_decay_grid_produces_distinct_measured_rows() -> None:
    rows = [
        measure_envelope_inclusive_bpw(
            window_k=40,
            decay_num=int(num),
            decay_den=int(den),
            numel=1024,
        )
        for num, den in DEFAULT_DECAY_GRID
    ]
    inclusive = {float(row["inclusive_acc_bpw"]) for row in rows}
    assert len(inclusive) >= 2


def test_envelope_conservatism_monotonicity_denser_pattern_non_decreasing_bpw() -> None:
    numel = 4096
    decay_num, decay_den = 1, 1
    sparse_k = 10
    dense_k = 25
    sparse = measure_envelope_inclusive_bpw(
        window_k=sparse_k,
        decay_num=decay_num,
        decay_den=decay_den,
        numel=numel,
    )
    dense = measure_envelope_inclusive_bpw(
        window_k=dense_k,
        decay_num=decay_num,
        decay_den=decay_den,
        numel=numel,
    )
    assert float(dense["inclusive_acc_bpw"]) >= float(sparse["inclusive_acc_bpw"])
    chosen = build_conservative_envelope_payload(
        window_k=dense_k,
        decay_num=decay_num,
        decay_den=decay_den,
        numel=numel,
    )
    assert int(chosen.event_count) == dense_k


def test_high_flat_index_adversary_can_dominate_varint_width() -> None:
    numel = 4096
    event_count = 32
    cmp = compare_envelope_pattern_bytes(
        event_count=event_count,
        numel=numel,
        backlog_indices=(0, 1, 2),
        hot_indices=(3,),
        hot_values=(12,),
    )
    assert cmp["high_index_event_bytes"] >= cmp["dense_low_event_bytes"]
    assert cmp["high_index_flat_index"] == numel - 1


def test_strict_budget_equality_is_sized_not_sub2_not_recommended() -> None:
    growth = _horizon_growth(growth_branch=GROWTH_PLATEAU_SIZED, kworst=5.0, k99=5.0)
    result = size_acc_bpw_from_horizon_growth(
        growth,
        numel_for_bpw=64,
        measured_q_scale_bpw=float(R4B_Q_PHYSICAL_BITS_PER_WEIGHT_BASE3),
    )
    if result["sizing_verdict"] == SIZING_VERDICT_RECOMMENDED_LAW:
        best = float(result["best_grid_row"]["inclusive_acc_bpw"])
        budget = float(result["effective_acc_budget_bpw"])
        assert best < budget
    elif result["sizing_verdict"] == SIZING_VERDICT_SIZED_NOT_SUB2:
        assert all(
            float(row["inclusive_acc_bpw"]) >= float(result["effective_acc_budget_bpw"])
            for row in result["grid_rows"]
            if "inclusive_acc_bpw" in row
        )


def test_simulate_decay_surface_changes_with_decay_ratio() -> None:
    fast = simulate_decay_worst_case_surface(
        window_k=50,
        decay_num=1,
        decay_den=2,
        numel=64,
    )
    slow = simulate_decay_worst_case_surface(
        window_k=50,
        decay_num=1,
        decay_den=1,
        numel=64,
    )
    assert len(slow.events) == 50
    assert len(fast.events) == 50
    assert len(slow.backlog_indices) > len(fast.backlog_indices)


def test_consumes_horizon_growth_dict_without_recomputing_k_star() -> None:
    growth = _horizon_growth(growth_branch=GROWTH_PLATEAU_SIZED)
    mutated = copy.deepcopy(growth)
    mutated["summaries_by_h"][100]["kworst_weighted"] = 999.0
    before = size_acc_bpw_from_horizon_growth(growth, numel_for_bpw=128)
    after = size_acc_bpw_from_horizon_growth(mutated, numel_for_bpw=128)
    assert before["window_k"] == 20
    assert after["sizing_verdict"] == SIZING_VERDICT_INCONCLUSIVE


def _quantile_lane_row(*, state_key: str, k_star: int, right_censored: bool = False) -> dict:
    return {
        "state_key": state_key,
        "lane_index": 0,
        "k_star": int(k_star),
        "parity_pass": True,
        "gapped": False,
        "right_censored": bool(right_censored),
    }


def _quantile_horizon_growth(
    *,
    growth_branch: str = GROWTH_RIGHT_CENSORED_LOWER_BOUND,
    lane_rows: list[dict],
    coverage_tier: str = COVERAGE_TIER_REPRESENTATIVE,
    parity_fail_count: int = 0,
    gapped_lane_count: int = 0,
    eligible_lane_count: int | None = None,
    kworst: float = 100.0,
    k99: float = 76.0,
) -> dict:
    if eligible_lane_count is None:
        eligible_lane_count = len(lane_rows)
    return {
        "growth_branch": growth_branch,
        "coverage_tier": coverage_tier,
        "summaries_by_h": {
            100: {
                "horizon_h": 100,
                "kworst_weighted": float(kworst),
                "k99_weighted": float(k99),
                "parity_fail_count": int(parity_fail_count),
                "gapped_lane_count": int(gapped_lane_count),
                "eligible_lane_count": int(eligible_lane_count),
                "lane_rows": lane_rows,
            }
        },
    }


def _favorable_quantile_lane_rows() -> list[dict]:
    rows = [_quantile_lane_row(state_key=f"key.{index}", k_star=5) for index in range(100)]
    rows.append(_quantile_lane_row(state_key="tail.key", k_star=100, right_censored=True))
    return rows


def _quantile_weights(lane_rows: list[dict]) -> dict[str, float]:
    return {
        row["state_key"]: (0.001 if row["state_key"] == "tail.key" else 1.0)
        for row in lane_rows
    }


def test_quantile_sizing_determinate_candidate_when_under_budget() -> None:
    lane_rows = _favorable_quantile_lane_rows()
    result = quantile_size_acc_bpw_from_horizon_growth(
        _quantile_horizon_growth(lane_rows=lane_rows),
        numel_for_bpw=1024,
        selector_log_key_aligned=True,
        stratum_weights=_quantile_weights(lane_rows),
        measured_q_scale_bpw=float(R4B_Q_PHYSICAL_BITS_PER_WEIGHT_BASE3),
    )
    assert result["quantile_sizing_verdict"] == QUANTILE_SIZING_VERDICT_DETERMINATE_SUB2_CANDIDATE
    assert result["quantile_sub2_candidate"] is True
    assert result["quantile_k"] == pytest.approx(5.0)
    assert result["strict_less_than_budget"] is True


def test_quantile_sizing_determinate_not_under_budget() -> None:
    lane_rows = [
        _quantile_lane_row(state_key=f"key.{index}", k_star=95) for index in range(100)
    ]
    result = quantile_size_acc_bpw_from_horizon_growth(
        _quantile_horizon_growth(lane_rows=lane_rows, k99=95.0),
        numel_for_bpw=64,
        selector_log_key_aligned=True,
        stratum_weights={row["state_key"]: 1.0 for row in lane_rows},
        measured_q_scale_bpw=float(R4B_Q_PHYSICAL_BITS_PER_WEIGHT_BASE3),
    )
    assert result["quantile_sizing_verdict"] == QUANTILE_SIZING_VERDICT_DETERMINATE_NOT_UNDER_BUDGET
    assert result["quantile_sub2_candidate"] is False


def test_quantile_sizing_fails_closed_kq_ge_h() -> None:
    lane_rows = [_quantile_lane_row(state_key=f"key.{index}", k_star=100, right_censored=True) for index in range(100)]
    result = quantile_size_acc_bpw_from_horizon_growth(
        _quantile_horizon_growth(lane_rows=lane_rows),
        numel_for_bpw=256,
        selector_log_key_aligned=True,
        stratum_weights={row["state_key"]: 1.0 for row in lane_rows},
    )
    assert result["quantile_sub2_candidate"] is False
    assert result["reason"] == "censored_or_missing_quantile_k_at_horizon"


def test_quantile_sizing_fails_closed_censor_mass_over_allowance() -> None:
    lane_rows = [_quantile_lane_row(state_key=f"key.{index}", k_star=76) for index in range(99)]
    lane_rows.append(_quantile_lane_row(state_key="cens.tail", k_star=100, right_censored=True))
    weights = {row["state_key"]: 1.0 for row in lane_rows}
    result = quantile_size_acc_bpw_from_horizon_growth(
        _quantile_horizon_growth(lane_rows=lane_rows, k99=76.0),
        numel_for_bpw=256,
        selector_log_key_aligned=True,
        stratum_weights=weights,
    )
    assert result["quantile_sub2_candidate"] is False
    assert result["reason"] == "censor_mass_at_or_above_allowance"


def test_quantile_sizing_fails_closed_selected_lane_censored_low_rate() -> None:
    lane_rows = [_quantile_lane_row(state_key=f"key.{index}", k_star=5) for index in range(50)]
    lane_rows.extend(
        _quantile_lane_row(state_key=f"cens.{index}", k_star=100, right_censored=True)
        for index in range(50)
    )
    weights = {row["state_key"]: (10.0 if row["right_censored"] else 1.0) for row in lane_rows}
    result = quantile_size_acc_bpw_from_horizon_growth(
        _quantile_horizon_growth(lane_rows=lane_rows),
        numel_for_bpw=256,
        selector_log_key_aligned=True,
        stratum_weights=weights,
    )
    assert result["quantile_sub2_candidate"] is False
    assert result["reason"] in {
        "censor_mass_at_or_above_allowance",
        "selected_lane_censored",
        "censored_or_missing_quantile_k_at_horizon",
    }


def test_quantile_sizing_fails_closed_coverage_not_representative() -> None:
    lane_rows = _favorable_quantile_lane_rows()
    result = quantile_size_acc_bpw_from_horizon_growth(
        _quantile_horizon_growth(lane_rows=lane_rows, coverage_tier="PILOT"),
        numel_for_bpw=1024,
        selector_log_key_aligned=True,
        stratum_weights=_quantile_weights(lane_rows),
    )
    assert result["quantile_sub2_candidate"] is False
    assert result["reason"] == "coverage_not_representative"


def test_quantile_sizing_fails_closed_key_set_mismatch() -> None:
    lane_rows = _favorable_quantile_lane_rows()
    result = quantile_size_acc_bpw_from_horizon_growth(
        _quantile_horizon_growth(lane_rows=lane_rows),
        numel_for_bpw=1024,
        selector_log_key_aligned=False,
        stratum_weights=_quantile_weights(lane_rows),
    )
    assert result["quantile_sub2_candidate"] is False
    assert result["reason"] == "selector_log_key_mismatch"


def test_worst_case_sizing_path_byte_unchanged() -> None:
    growth = _horizon_growth(growth_branch=GROWTH_RIGHT_CENSORED_LOWER_BOUND)
    before = size_acc_bpw_from_horizon_growth(growth, numel_for_bpw=256)
    after = size_acc_bpw_from_horizon_growth(growth, numel_for_bpw=256)
    assert before == after
    assert before["sizing_verdict"] == SIZING_VERDICT_INCONCLUSIVE
    assert before["reason"] == "right_censored_growth_branch"


def test_quantile_sizing_fails_closed_growth_branch_not_right_censored() -> None:
    lane_rows = _favorable_quantile_lane_rows()
    result = quantile_size_acc_bpw_from_horizon_growth(
        _quantile_horizon_growth(
            growth_branch=GROWTH_INCONCLUSIVE_COST_OR_COVERAGE,
            lane_rows=lane_rows,
        ),
        numel_for_bpw=1024,
        selector_log_key_aligned=True,
        stratum_weights=_quantile_weights(lane_rows),
        measured_q_scale_bpw=float(R4B_Q_PHYSICAL_BITS_PER_WEIGHT_BASE3),
    )
    assert result["quantile_sub2_candidate"] is False
    assert result["reason"] == "growth_branch_not_right_censored_lower_bound"


def test_quantile_sizing_fails_closed_parity_fail_at_horizon() -> None:
    lane_rows = _favorable_quantile_lane_rows()
    result = quantile_size_acc_bpw_from_horizon_growth(
        _quantile_horizon_growth(lane_rows=lane_rows, parity_fail_count=1),
        numel_for_bpw=1024,
        selector_log_key_aligned=True,
        stratum_weights=_quantile_weights(lane_rows),
    )
    assert result["quantile_sub2_candidate"] is False
    assert result["reason"] == "parity_failures_at_sizing_horizon"


def test_quantile_sizing_fails_closed_gapped_lane_at_horizon() -> None:
    lane_rows = _favorable_quantile_lane_rows()
    result = quantile_size_acc_bpw_from_horizon_growth(
        _quantile_horizon_growth(lane_rows=lane_rows, gapped_lane_count=1),
        numel_for_bpw=1024,
        selector_log_key_aligned=True,
        stratum_weights=_quantile_weights(lane_rows),
    )
    assert result["quantile_sub2_candidate"] is False
    assert result["reason"] == "gapped_lanes_at_sizing_horizon"
