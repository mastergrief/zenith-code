"""Crossing-bearing-only band dominance reducer (Fold-3A, inert / additive)."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from calm.hrm_text_158.native_full_stack.s1d7_band_counter import (
    DOMINANCE_C_MULTIPLIER_MIN,
    DOMINANCE_C_SHARE_MIN,
)

CLASSIFIER = "s1d7_cb_only_dominance_v1"

CB_ONLY_AC_COMPOSITE_DOMINANT = "CB_ONLY_AC_COMPOSITE_DOMINANT"
CB_ONLY_C_MONOLITHIC_DOMINANT = "CB_ONLY_C_MONOLITHIC_DOMINANT"
CB_ONLY_RESIDUAL_OR_SPLIT_UNRESOLVED = "CB_ONLY_RESIDUAL_OR_SPLIT_UNRESOLVED"
CB_ONLY_EMPTY_SUPPORT_INCONCLUSIVE = "CB_ONLY_EMPTY_SUPPORT_INCONCLUSIVE"
CB_ONLY_COVERAGE_INCONCLUSIVE = "CB_ONLY_COVERAGE_INCONCLUSIVE"

# The A+C composite (non-E) dominance floor on the crossing-bearing aggregate.
# Intentionally decoupled from DOMINANCE_C_SHARE_MIN so a future C-share retune
# cannot silently move the composite gate.
AC_COMPOSITE_SHARE_MIN = 0.80

ANTI_OVERCLAIM_VERBATIM = (
    "decider precondition; NO CA verdict; NO candidate-C resolution; "
    "NO reduction eligibility; NOT the ~430MB C4.S1d bank pin; NOT a sub-2 proof; "
    "NOT universal crossing census; NOT implementation readiness."
)


def _band_bytes(row: Mapping[str, Any]) -> tuple[int, int, int]:
    return (
        int(row.get("band_a_bytes") or 0),
        int(row.get("band_c_bytes") or 0),
        int(row.get("band_e_bytes") or 0),
    )


def _coverage_precheck(
    per_state: Sequence[Mapping[str, Any]],
    *,
    sampled_states: Sequence[int],
) -> dict[str, Any] | None:
    expected = tuple(int(state) for state in sampled_states)
    if not expected:
        return {
            "terminal_branch": CB_ONLY_COVERAGE_INCONCLUSIVE,
            "fail_closed_reason": "CB_ONLY_EMPTY_SAMPLED_STATES",
        }

    seen: list[int] = []
    for row in per_state:
        state_index = row.get("state_index")
        if state_index is None:
            return {
                "terminal_branch": CB_ONLY_COVERAGE_INCONCLUSIVE,
                "fail_closed_reason": "CB_ONLY_MISSING_STATE_INDEX",
            }
        seen.append(int(state_index))

    expected_set = set(expected)
    seen_set = set(seen)
    if len(seen) != len(seen_set):
        return {
            "terminal_branch": CB_ONLY_COVERAGE_INCONCLUSIVE,
            "fail_closed_reason": "CB_ONLY_DUPLICATE_STATE_INDEX",
            "duplicate_state_indices": sorted(
                state for state in seen_set if seen.count(state) > 1
            ),
        }
    if seen_set != expected_set:
        return {
            "terminal_branch": CB_ONLY_COVERAGE_INCONCLUSIVE,
            "fail_closed_reason": "CB_ONLY_UNEXPECTED_OR_MISSING_STATE",
            "expected_sampled_states": list(expected),
            "observed_state_indices": sorted(seen_set),
            "missing_state_indices": sorted(expected_set - seen_set),
            "unexpected_state_indices": sorted(seen_set - expected_set),
        }
    return None


def _aggregate_cb_bands(cb_rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    totals = {"a": 0, "c": 0, "e": 0}
    for row in cb_rows:
        a_bytes, c_bytes, e_bytes = _band_bytes(row)
        totals["a"] += a_bytes
        totals["c"] += c_bytes
        totals["e"] += e_bytes
    return totals


def evaluate_cb_only_band_dominance(
    per_state: Sequence[Mapping[str, Any]],
    *,
    sampled_states: Sequence[int],
) -> dict[str, Any]:
    coverage_failure = _coverage_precheck(per_state, sampled_states=sampled_states)
    if coverage_failure is not None:
        return {
            "classifier": CLASSIFIER,
            **coverage_failure,
            "cb_state_count": 0,
            "single_cb_support": False,
            "excluded_zero_crossing_state_count": 0,
            "c_only_dominance_ok": False,
            "a_plus_c_share": 0.0,
            "band_c_share": 0.0,
            "aggregate_band_bytes": {"a": 0, "c": 0, "e": 0},
            "cb_state_indices": [],
            "per_cb_state": [],
        }

    zero_crossing_rows: list[dict[str, Any]] = []
    cb_rows: list[dict[str, Any]] = []
    for row in per_state:
        crossing_len = int(row.get("crossing_indices_len") or 0)
        state_index = int(row["state_index"])
        a_bytes, c_bytes, e_bytes = _band_bytes(row)
        detail = {
            "state_index": state_index,
            "band_a_bytes": a_bytes,
            "band_c_bytes": c_bytes,
            "band_e_bytes": e_bytes,
            "crossing_indices_len": crossing_len,
        }
        if crossing_len > 0:
            cb_rows.append(detail)
        else:
            zero_crossing_rows.append(detail)

    excluded_zero_crossing_state_count = len(zero_crossing_rows)
    cb_state_count = len(cb_rows)
    single_cb_support = cb_state_count == 1

    if cb_state_count == 0:
        return {
            "classifier": CLASSIFIER,
            "terminal_branch": CB_ONLY_EMPTY_SUPPORT_INCONCLUSIVE,
            "fail_closed_reason": "CB_ONLY_NO_CROSSING_BEARING_ROWS",
            "cb_state_count": 0,
            "single_cb_support": False,
            "excluded_zero_crossing_state_count": excluded_zero_crossing_state_count,
            "c_only_dominance_ok": False,
            "a_plus_c_share": 0.0,
            "band_c_share": 0.0,
            "aggregate_band_bytes": {"a": 0, "c": 0, "e": 0},
            "cb_state_indices": [],
            "per_cb_state": [],
            "excluded_zero_crossing_states": zero_crossing_rows,
        }

    totals = _aggregate_cb_bands(cb_rows)
    grand_total = totals["a"] + totals["c"] + totals["e"]
    if grand_total <= 0:
        return {
            "classifier": CLASSIFIER,
            "terminal_branch": CB_ONLY_EMPTY_SUPPORT_INCONCLUSIVE,
            "fail_closed_reason": "CB_ONLY_ALL_ZERO_CB_ACTIVITY",
            "cb_state_count": cb_state_count,
            "single_cb_support": single_cb_support,
            "excluded_zero_crossing_state_count": excluded_zero_crossing_state_count,
            "c_only_dominance_ok": False,
            "a_plus_c_share": 0.0,
            "band_c_share": 0.0,
            "aggregate_band_bytes": totals,
            "cb_state_indices": [row["state_index"] for row in cb_rows],
            "per_cb_state": cb_rows,
            "excluded_zero_crossing_states": zero_crossing_rows,
        }

    band_c_share = float(totals["c"]) / float(grand_total)
    a_plus_c_share = float(totals["a"] + totals["c"]) / float(grand_total)
    next_largest = max(totals["a"], totals["e"])
    c_only_dominance_ok = (
        band_c_share >= DOMINANCE_C_SHARE_MIN
        and totals["c"] >= DOMINANCE_C_MULTIPLIER_MIN * next_largest
    )

    base = {
        "classifier": CLASSIFIER,
        "cb_state_count": cb_state_count,
        "single_cb_support": single_cb_support,
        "excluded_zero_crossing_state_count": excluded_zero_crossing_state_count,
        "c_only_dominance_ok": c_only_dominance_ok,
        "a_plus_c_share": a_plus_c_share,
        "band_c_share": band_c_share,
        "aggregate_band_bytes": totals,
        "cb_state_indices": [row["state_index"] for row in cb_rows],
        "per_cb_state": cb_rows,
        "excluded_zero_crossing_states": zero_crossing_rows,
        "thresholds": {
            "dominance_c_share_min": DOMINANCE_C_SHARE_MIN,
            "dominance_c_multiplier_min": DOMINANCE_C_MULTIPLIER_MIN,
            "ac_composite_share_min": AC_COMPOSITE_SHARE_MIN,
        },
    }

    if c_only_dominance_ok:
        return {
            **base,
            "terminal_branch": CB_ONLY_C_MONOLITHIC_DOMINANT,
            "fail_closed_reason": None,
        }
    if a_plus_c_share >= AC_COMPOSITE_SHARE_MIN:
        return {
            **base,
            "terminal_branch": CB_ONLY_AC_COMPOSITE_DOMINANT,
            "fail_closed_reason": None,
        }
    return {
        **base,
        "terminal_branch": CB_ONLY_RESIDUAL_OR_SPLIT_UNRESOLVED,
        "fail_closed_reason": "CB_ONLY_NEITHER_C_MONOLITHIC_NOR_AC_COMPOSITE",
    }


def evaluate_cb_only_from_ca_confirmation_receipt(
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    per_state = receipt.get("per_state")
    if not isinstance(per_state, list):
        raise ValueError("ca_confirmation_receipt_missing_per_state")
    sampled_states = receipt.get("sampled_states")
    if not isinstance(sampled_states, list):
        raise ValueError("ca_confirmation_receipt_missing_sampled_states")
    result = evaluate_cb_only_band_dominance(
        per_state,
        sampled_states=[int(state) for state in sampled_states],
    )
    result["input_receipt_fields"] = {
        "cb_state_count": receipt.get("cb_state_count"),
        "mark_count": receipt.get("mark_count"),
        "terminal_branch": receipt.get("terminal_branch"),
        "eligible_module_limit": receipt.get("eligible_module_limit"),
        "peak_rss_gib": receipt.get("peak_rss_gib"),
    }
    return result


def build_fold3a_measurement_receipt(
    *,
    ca_confirmation_receipt_path: str,
    ca_confirmation_receipt: Mapping[str, Any],
    dominance_result: Mapping[str, Any],
    git_head: str,
    task_id: str,
    upstream_closeout_spec_path: str,
    upstream_closeout_receipt_path: str,
    implement_gate_msg_id: str,
    dispatch_msg_id: str,
) -> dict[str, Any]:
    return {
        "schema": "hrm_text_158_measurement_closeout_fold3a_cb_only_dominance_receipt/v1",
        "task_id": task_id,
        "fold": "fold_3a_crossing_bearing_only_dominance_reducer",
        "classifier": CLASSIFIER,
        "git_head_at_closeout": git_head,
        "provenance": {
            "dispatch_msg_id": dispatch_msg_id,
            "implement_gate_msg_id": implement_gate_msg_id,
            "upstream_closeout_spec": upstream_closeout_spec_path,
            "upstream_closeout_receipt": upstream_closeout_receipt_path,
            "input_ca_confirmation_receipt": ca_confirmation_receipt_path,
        },
        "anti_overclaim_verbatim": ANTI_OVERCLAIM_VERBATIM,
        "allowed_claim": (
            "the legacy all-sampled-state dominance veto is the wrong measurement "
            "object for zero-crossing-heavy data; on the crossing-bearing support "
            "the dense-[0..9] bank-scale receipt shows A+C composite dominance while "
            "C-only dominance is refuted (c_only_dominance_ok=false)."
        ),
        "hard_limits": {
            "wp_uncomputable": True,
            "wp_reason": "cb_state_count=1 → INSUFFICIENT_CB_STATES; no CA_PERSISTS/MIXED/DILUTES",
            "within_support_only": bool(dominance_result.get("single_cb_support")),
            "not_universal_crossing_census": True,
            "not_implementation_readiness": True,
        },
        "dominance_result": dict(dominance_result),
        "input_ca_confirmation_summary": {
            "terminal_branch": ca_confirmation_receipt.get("terminal_branch"),
            "cb_state_count": ca_confirmation_receipt.get("cb_state_count"),
            "mark_count": ca_confirmation_receipt.get("mark_count"),
            "sampled_states": ca_confirmation_receipt.get("sampled_states"),
            "eligible_module_limit": ca_confirmation_receipt.get("eligible_module_limit"),
            "peak_rss_gib": ca_confirmation_receipt.get("peak_rss_gib"),
        },
    }
