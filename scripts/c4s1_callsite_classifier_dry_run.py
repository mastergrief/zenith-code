#!/usr/bin/env python3
"""Synthetic dry-run for Phase-3 callsite classifier branch outcomes (no GPU)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.hrm_text_158_slice5_v6i_oom_profile_attribution import (  # noqa: E402
    build_phase3_callsite_classifier_receipt_from_attribution_payload,
)


def _base_payload(**expanded_overrides: object) -> dict:
    expanded = {
        "fail_closed_terminal": None,
        "guards": {
            "phase3_s1d_subsplit_mode": True,
            "obmalloc_expanded_event_validation": {"valid": True, "pair_counts_by_site": {}},
            "obmalloc_expanded_event_counts": {"total": 190},
            "tracemalloc_perturbed": False,
        },
        "localization": {
            "phase3_s1d_subsplit_mode": True,
            "s1d_parent_reconcile_fraction": 0.0001,
        },
        "call_site_status": "RESOLVED",
        "call_site_origin_file_line": (
            "calm/hrm_text_158/native_full_stack/event_coded_acc_live_carrier.py:896"
        ),
        "s1d7_call_site_candidate": "c",
        "s1d7_call_site_branch_outcome": "S1D7_CALL_SITE_CANDIDATE_C_EVENTS_JOURNAL",
        "s1d7_tracemalloc_top_concentration_fraction": 0.98,
        "tracemalloc_perturbed": False,
        "s1d7_call_site_in_bracket_ok": True,
        "s1d7_tracemalloc_mark_pair_count": 4,
        **expanded_overrides,
    }
    return {
        "exit_code": 0,
        "process_exit_code": 0,
        "runs": {"B": {"profile_mark_count": 42}},
        "obmalloc_expanded_attribution": expanded,
    }


CASES: dict[str, tuple[dict, str, int]] = {
    "resolved_a": (
        _base_payload(
            call_site_origin_file_line=(
                "calm/hrm_text_158/native_full_stack/event_coded_acc_live_carrier.py:886"
            ),
            s1d7_call_site_candidate="a",
            s1d7_call_site_branch_outcome="S1D7_CALL_SITE_CANDIDATE_A_CROSSING_INDICES",
        ),
        "S1D7_CALL_SITE_CANDIDATE_A_CROSSING_INDICES",
        0,
    ),
    "resolved_c": (
        _base_payload(),
        "S1D7_CALL_SITE_CANDIDATE_C_EVENTS_JOURNAL",
        0,
    ),
    "resolved_e": (
        _base_payload(
            call_site_origin_file_line=(
                "calm/hrm_text_158/native_full_stack/event_coded_acc_live_carrier.py:905"
            ),
            s1d7_call_site_candidate="e",
            s1d7_call_site_branch_outcome="S1D7_CALL_SITE_CANDIDATE_E_NUMPY_ARRAYS",
        ),
        "S1D7_CALL_SITE_CANDIDATE_E_NUMPY_ARRAYS",
        0,
    ),
    "ambiguous": (
        _base_payload(
            call_site_origin_file_line=(
                "calm/hrm_text_158/native_full_stack/event_coded_acc_live_carrier.py:903"
            ),
            s1d7_call_site_candidate="ambiguous",
            s1d7_call_site_branch_outcome=None,
        ),
        "S1D7_CALL_SITE_RESOLVED_CANDIDATE_AMBIGUOUS",
        0,
    ),
    "outside_bracket": (
        _base_payload(
            call_site_status="UNRESOLVED",
            call_site_origin_file_line=(
                "calm/hrm_text_158/native_full_stack/event_coded_acc_live_carrier.py:700"
            ),
            s1d7_call_site_candidate=None,
            s1d7_call_site_branch_outcome=None,
            s1d7_call_site_in_bracket_ok=False,
            localization={
                "phase3_s1d_subsplit_mode": True,
                "s1d_parent_reconcile_fraction": 0.0001,
                "s1d7_tracemalloc_call_site": {
                    "fail_closed_reason": "CALL_SITE_OUTSIDE_S1D7_BRACKET",
                },
            },
        ),
        "CALL_SITE_OUTSIDE_S1D7_BRACKET",
        35,
    ),
    "concentration_fail": (
        _base_payload(
            call_site_status="UNRESOLVED",
            s1d7_call_site_candidate="c",
            s1d7_call_site_branch_outcome=None,
            localization={
                "phase3_s1d_subsplit_mode": True,
                "s1d_parent_reconcile_fraction": 0.0001,
                "s1d7_tracemalloc_call_site": {
                    "fail_closed_reason": "TRACEMALLOC_CONCENTRATION_FAIL",
                },
            },
        ),
        "TRACEMALLOC_CONCENTRATION_FAIL",
        35,
    ),
    "missing_pairs": (
        _base_payload(s1d7_tracemalloc_mark_pair_count=2),
        "TRACEMALLOC_INCONCLUSIVE",
        35,
    ),
    "perturbation": (
        _base_payload(
            tracemalloc_perturbed=True,
            guards={
                "phase3_s1d_subsplit_mode": True,
                "obmalloc_expanded_event_validation": {"valid": True, "pair_counts_by_site": {}},
                "obmalloc_expanded_event_counts": {"total": 190},
                "tracemalloc_perturbed": True,
            },
        ),
        "TRACEMALLOC_PERTURBED_INCONCLUSIVE",
        35,
    ),
}


def main() -> int:
    failures: list[str] = []
    results: dict[str, dict] = {}
    for name, (payload, expected_branch, expected_exit) in CASES.items():
        receipt = build_phase3_callsite_classifier_receipt_from_attribution_payload(payload)
        results[name] = {
            "branch_outcome": receipt.get("branch_outcome"),
            "classifier_exit_code": receipt.get("classifier_exit_code"),
        }
        if receipt.get("branch_outcome") != expected_branch:
            failures.append(
                f"{name}: branch expected {expected_branch!r} got {receipt.get('branch_outcome')!r}"
            )
        if int(receipt.get("classifier_exit_code", -1)) != expected_exit:
            failures.append(
                f"{name}: exit expected {expected_exit} got {receipt.get('classifier_exit_code')!r}"
            )
    print(json.dumps({"ok": not failures, "results": results, "failures": failures}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
