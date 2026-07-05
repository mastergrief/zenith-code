"""CPU-static tests for Fold-3A crossing-bearing-only dominance reducer."""

from __future__ import annotations

from typing import Any

import pytest


def _dense_state0_row() -> dict[str, Any]:
    return {
        "state_index": 0,
        "band_a_bytes": 22640,
        "band_c_bytes": 48640,
        "band_e_bytes": 5408,
        "crossing_indices_len": 512,
        "is_crossing_bearing": True,
        "per_cb_ca_share": 0.9294804923847277,
    }


def _zero_crossing_row(state_index: int) -> dict[str, Any]:
    return {
        "state_index": state_index,
        "band_a_bytes": 112,
        "band_c_bytes": 0,
        "band_e_bytes": 0,
        "crossing_indices_len": 0,
        "is_crossing_bearing": False,
        "per_cb_ca_share": None,
    }


def _dense_primary_fixture_rows() -> list[dict[str, Any]]:
    rows = [_dense_state0_row()]
    rows.extend(_zero_crossing_row(state_index) for state_index in range(1, 10))
    return rows


def test_dense_state0_crossing_states1_9_zero_ac_composite_dominant() -> None:
    from calm.hrm_text_158.native_full_stack.s1d7_cb_only_dominance import (
        CB_ONLY_AC_COMPOSITE_DOMINANT,
        evaluate_cb_only_band_dominance,
    )

    rows = _dense_primary_fixture_rows()
    result = evaluate_cb_only_band_dominance(rows, sampled_states=list(range(10)))

    assert result["terminal_branch"] == CB_ONLY_AC_COMPOSITE_DOMINANT
    assert result["cb_state_count"] == 1
    assert result["single_cb_support"] is True
    assert result["excluded_zero_crossing_state_count"] == 9
    assert result["c_only_dominance_ok"] is False
    assert result["a_plus_c_share"] == pytest.approx(71280 / 76688)
    assert result["aggregate_band_bytes"] == {"a": 22640, "c": 48640, "e": 5408}


def test_zero_crossing_rows_do_not_veto_legacy_would_fail() -> None:
    from calm.hrm_text_158.native_full_stack.s1d7_band_counter import (
        evaluate_band_dominance,
    )
    from calm.hrm_text_158.native_full_stack.s1d7_cb_only_dominance import (
        CB_ONLY_AC_COMPOSITE_DOMINANT,
        evaluate_cb_only_band_dominance,
    )

    rows = _dense_primary_fixture_rows()
    legacy_rows = [
        {
            "state_index": row["state_index"],
            "s1d7_band_counters": {
                "byte_proxies": {
                    "band_a_bytes": row["band_a_bytes"],
                    "band_c_bytes": row["band_c_bytes"],
                    "band_e_bytes": row["band_e_bytes"],
                }
            },
        }
        for row in rows
    ]
    legacy = evaluate_band_dominance(legacy_rows, sampled_states=tuple(range(10)))
    assert legacy["band_counter_dominance_ok"] is False
    assert legacy["fail_closed_reason"] == "BAND_COUNTER_C_NOT_TOP_IN_STATE"

    cb_only = evaluate_cb_only_band_dominance(rows, sampled_states=list(range(10)))
    assert cb_only["terminal_branch"] == CB_ONLY_AC_COMPOSITE_DOMINANT


def test_coverage_inconclusive_missing_state() -> None:
    from calm.hrm_text_158.native_full_stack.s1d7_cb_only_dominance import (
        CB_ONLY_COVERAGE_INCONCLUSIVE,
        evaluate_cb_only_band_dominance,
    )

    rows = [_dense_state0_row(), _zero_crossing_row(1)]
    result = evaluate_cb_only_band_dominance(rows, sampled_states=[0, 1, 2])
    assert result["terminal_branch"] == CB_ONLY_COVERAGE_INCONCLUSIVE
    assert result["fail_closed_reason"] == "CB_ONLY_UNEXPECTED_OR_MISSING_STATE"
    assert result["missing_state_indices"] == [2]


def test_coverage_inconclusive_duplicate_state() -> None:
    from calm.hrm_text_158.native_full_stack.s1d7_cb_only_dominance import (
        CB_ONLY_COVERAGE_INCONCLUSIVE,
        evaluate_cb_only_band_dominance,
    )

    rows = [_dense_state0_row(), _dense_state0_row()]
    result = evaluate_cb_only_band_dominance(rows, sampled_states=[0])
    assert result["terminal_branch"] == CB_ONLY_COVERAGE_INCONCLUSIVE
    assert result["fail_closed_reason"] == "CB_ONLY_DUPLICATE_STATE_INDEX"


def test_coverage_inconclusive_unexpected_state() -> None:
    from calm.hrm_text_158.native_full_stack.s1d7_cb_only_dominance import (
        CB_ONLY_COVERAGE_INCONCLUSIVE,
        evaluate_cb_only_band_dominance,
    )

    rows = [_dense_state0_row(), _zero_crossing_row(99)]
    result = evaluate_cb_only_band_dominance(rows, sampled_states=[0])
    assert result["terminal_branch"] == CB_ONLY_COVERAGE_INCONCLUSIVE
    assert result["unexpected_state_indices"] == [99]


def test_empty_support_inconclusive_no_cb_rows() -> None:
    from calm.hrm_text_158.native_full_stack.s1d7_cb_only_dominance import (
        CB_ONLY_EMPTY_SUPPORT_INCONCLUSIVE,
        evaluate_cb_only_band_dominance,
    )

    rows = [_zero_crossing_row(0), _zero_crossing_row(1)]
    result = evaluate_cb_only_band_dominance(rows, sampled_states=[0, 1])
    assert result["terminal_branch"] == CB_ONLY_EMPTY_SUPPORT_INCONCLUSIVE
    assert result["cb_state_count"] == 0
    assert result["excluded_zero_crossing_state_count"] == 2


def test_residual_or_split_unresolved_e_a_dominant_cb_support() -> None:
    from calm.hrm_text_158.native_full_stack.s1d7_cb_only_dominance import (
        CB_ONLY_RESIDUAL_OR_SPLIT_UNRESOLVED,
        evaluate_cb_only_band_dominance,
    )

    rows = [
        {
            "state_index": 0,
            "band_a_bytes": 100,
            "band_c_bytes": 50,
            "band_e_bytes": 850,
            "crossing_indices_len": 10,
        }
    ]
    result = evaluate_cb_only_band_dominance(rows, sampled_states=[0])
    assert result["terminal_branch"] == CB_ONLY_RESIDUAL_OR_SPLIT_UNRESOLVED
    assert result["c_only_dominance_ok"] is False
    assert result["a_plus_c_share"] < 0.80


def test_c_monolithic_dominant_high_c_cb_support() -> None:
    from calm.hrm_text_158.native_full_stack.s1d7_cb_only_dominance import (
        CB_ONLY_C_MONOLITHIC_DOMINANT,
        evaluate_cb_only_band_dominance,
    )

    rows = [
        {
            "state_index": 0,
            "band_a_bytes": 100,
            "band_c_bytes": 9000,
            "band_e_bytes": 100,
            "crossing_indices_len": 5,
        }
    ]
    result = evaluate_cb_only_band_dominance(rows, sampled_states=[0])
    assert result["terminal_branch"] == CB_ONLY_C_MONOLITHIC_DOMINANT
    assert result["c_only_dominance_ok"] is True


def test_ac_composite_constant_decoupled_from_c_share_constant() -> None:
    from calm.hrm_text_158.native_full_stack.s1d7_band_counter import (
        DOMINANCE_C_SHARE_MIN,
    )
    from calm.hrm_text_158.native_full_stack.s1d7_cb_only_dominance import (
        AC_COMPOSITE_SHARE_MIN,
    )

    assert AC_COMPOSITE_SHARE_MIN == 0.80
    assert AC_COMPOSITE_SHARE_MIN == DOMINANCE_C_SHARE_MIN  # numeric only today
    assert "AC_COMPOSITE_SHARE_MIN" != "DOMINANCE_C_SHARE_MIN"
