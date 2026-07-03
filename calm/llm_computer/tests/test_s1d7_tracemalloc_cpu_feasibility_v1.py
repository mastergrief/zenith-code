from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


def _run_feasibility_subprocess(case: str) -> dict:
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "s1d7_tracemalloc_cpu_feasibility.py"),
            case,
        ],
        cwd=str(REPO_ROOT),
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    rows = json.loads(proc.stdout.strip())
    assert len(rows) == 1
    return rows[0]


@pytest.fixture(autouse=True)
def _reset_tracemalloc_state() -> None:
    from calm.hrm_text_158.native_full_stack.host_tracemalloc_probe import (
        reset_tracemalloc_state_for_tests,
    )

    reset_tracemalloc_state_for_tests()
    yield
    reset_tracemalloc_state_for_tests()


def test_s1d7_tracemalloc_cpu_feasibility_crossing_indices() -> None:
    row = _run_feasibility_subprocess("crossing_indices")
    assert row["ok"] is True, row
    assert int(row["current_delta_bytes"]) > 0
    assert row["result"]["call_site_status"] == "RESOLVED"
    assert row["result"]["s1d7_call_site_in_bracket_ok"] is True


def test_s1d7_tracemalloc_cpu_feasibility_events_journal() -> None:
    row = _run_feasibility_subprocess("events_journal")
    assert row["ok"] is True, row
    assert int(row["current_delta_bytes"]) > 0
    assert row["result"]["call_site_status"] == "RESOLVED"
    assert row["result"]["s1d7_call_site_in_bracket_ok"] is True


def test_s1d7_tracemalloc_perturbation_guard_smoke_cpu() -> None:
    row = _run_feasibility_subprocess("perturbation_smoke")
    assert row["ok"] is True, row
    assert row["smoke"]["tracemalloc_perturbed"] is False


def test_s1d7_call_site_parser_line_range() -> None:
    from calm.hrm_text_158.native_full_stack.s1d7_tracemalloc_feasibility import (
        S1D7_ACCEPTANCE_LINE_MAX,
        S1D7_ACCEPTANCE_LINE_MIN,
        classify_s1d7_tracemalloc_call_site,
        parse_origin_from_traceback,
    )

    origin_file, origin_line = parse_origin_from_traceback(
        [
            '  File "/repo/calm/hrm_text_158/native_full_stack/event_coded_acc_live_carrier.py", line 905'
        ]
    )
    assert origin_file == "event_coded_acc_live_carrier.py"
    assert origin_line == 905
    diff = {
        "current_delta_bytes": 1000,
        "top_concentration_fraction": 0.9,
        "top_delta_frames": [
            {
                "delta_bytes": 1000,
                "concentration_fraction": 0.9,
                "traceback": [
                    '  File "/repo/calm/hrm_text_158/native_full_stack/event_coded_acc_live_carrier.py", line 905'
                ],
            }
        ],
    }
    inside = classify_s1d7_tracemalloc_call_site(diff)
    assert inside["s1d7_call_site_in_bracket_ok"] is True
    assert inside["call_site_origin_file_line"] == "event_coded_acc_live_carrier.py:905"
    assert inside["s1d7_call_site_candidate"] == "e"
    outside = classify_s1d7_tracemalloc_call_site(
        {
            **diff,
            "top_delta_frames": [
                {
                    **diff["top_delta_frames"][0],
                    "traceback": [
                        '  File "/repo/calm/hrm_text_158/native_full_stack/event_coded_acc_live_carrier.py", line 928'
                    ],
                }
            ],
        }
    )
    assert outside["fail_closed_reason"] == "CALL_SITE_OUTSIDE_S1D7_BRACKET"
    assert outside["s1d7_call_site_in_bracket_ok"] is False
    assert S1D7_ACCEPTANCE_LINE_MIN == 876
    assert S1D7_ACCEPTANCE_LINE_MAX == 909


def test_s1d7_call_site_concentration_fail_closed() -> None:
    from calm.hrm_text_158.native_full_stack.s1d7_tracemalloc_feasibility import (
        classify_s1d7_tracemalloc_call_site,
    )

    diff = {
        "current_delta_bytes": 1000,
        "top_concentration_fraction": 0.9,
        "top_delta_frames": [
            {
                "delta_bytes": 1000,
                "concentration_fraction": 0.40,
                "traceback": [
                    '  File "/repo/calm/hrm_text_158/native_full_stack/event_coded_acc_live_carrier.py", line 896'
                ],
            }
        ],
    }
    result = classify_s1d7_tracemalloc_call_site(diff)
    assert result["call_site_status"] == "UNRESOLVED"
    assert result["fail_closed_reason"] == "TRACEMALLOC_CONCENTRATION_FAIL"


def test_s1d7_call_site_outside_bracket_fail_closed() -> None:
    from calm.hrm_text_158.native_full_stack.s1d7_tracemalloc_feasibility import (
        classify_s1d7_tracemalloc_call_site,
    )

    diff = {
        "current_delta_bytes": 1000,
        "top_concentration_fraction": 0.95,
        "top_delta_frames": [
            {
                "delta_bytes": 1000,
                "concentration_fraction": 0.95,
                "traceback": [
                    '  File "/repo/calm/hrm_text_158/native_full_stack/event_coded_acc_live_carrier.py", line 875'
                ],
            }
        ],
    }
    result = classify_s1d7_tracemalloc_call_site(diff)
    assert result["call_site_status"] == "UNRESOLVED"
    assert result["fail_closed_reason"] == "CALL_SITE_OUTSIDE_S1D7_BRACKET"
    assert result["s1d7_call_site_in_bracket_ok"] is False


def test_s1d7_call_site_candidate_line_bands() -> None:
    from calm.hrm_text_158.native_full_stack.s1d7_tracemalloc_feasibility import (
        S1D7_BRANCH_CANDIDATE_A,
        S1D7_BRANCH_CANDIDATE_AMBIGUOUS,
        S1D7_BRANCH_CANDIDATE_C,
        S1D7_BRANCH_CANDIDATE_E,
        classify_s1d7_tracemalloc_call_site,
        map_s1d7_call_site_candidate,
    )

    assert map_s1d7_call_site_candidate(886) == "a"
    assert map_s1d7_call_site_candidate(891) == "c"
    assert map_s1d7_call_site_candidate(902) == "c"
    assert map_s1d7_call_site_candidate(905) == "e"
    assert map_s1d7_call_site_candidate(908) == "e"
    assert map_s1d7_call_site_candidate(903) == "ambiguous"
    assert map_s1d7_call_site_candidate(876) == "ambiguous"
    assert map_s1d7_call_site_candidate(875) is None

    def _diff_for_line(line: int) -> dict:
        return {
            "current_delta_bytes": 1000,
            "top_concentration_fraction": 0.9,
            "top_delta_frames": [
                {
                    "delta_bytes": 1000,
                    "concentration_fraction": 0.9,
                    "traceback": [
                        "  File "
                        '"/repo/calm/hrm_text_158/native_full_stack/event_coded_acc_live_carrier.py", '
                        f"line {line}"
                    ],
                }
            ],
        }

    a_result = classify_s1d7_tracemalloc_call_site(_diff_for_line(886))
    assert a_result["s1d7_call_site_candidate"] == "a"
    assert a_result["s1d7_call_site_branch_outcome"] == S1D7_BRANCH_CANDIDATE_A

    c_result = classify_s1d7_tracemalloc_call_site(_diff_for_line(896))
    assert c_result["s1d7_call_site_candidate"] == "c"
    assert c_result["s1d7_call_site_branch_outcome"] == S1D7_BRANCH_CANDIDATE_C

    e_result = classify_s1d7_tracemalloc_call_site(_diff_for_line(905))
    assert e_result["s1d7_call_site_candidate"] == "e"
    assert e_result["s1d7_call_site_branch_outcome"] == S1D7_BRANCH_CANDIDATE_E

    ambiguous = classify_s1d7_tracemalloc_call_site(_diff_for_line(903))
    assert ambiguous["call_site_status"] == "RESOLVED"
    assert ambiguous["s1d7_call_site_candidate"] == "ambiguous"
    assert ambiguous["s1d7_call_site_branch_outcome"] == S1D7_BRANCH_CANDIDATE_AMBIGUOUS


def _s1d7_tracemalloc_snapshot(*, traced_bytes: int, carrier_line: int) -> dict:
    traceback = [
        '  File "/repo/calm/hrm_text_158/native_full_stack/event_coded_acc_live_carrier.py", '
        f"line {int(carrier_line)}"
    ]
    return {
        "enabled": True,
        "traced_current_bytes": int(traced_bytes),
        "traced_peak_bytes": int(traced_bytes),
        "top_frames": [
            {
                "size_bytes": int(traced_bytes),
                "count": 1,
                "traceback": traceback,
                "traceback_key": "|".join(traceback),
            }
        ],
    }


def test_attribute_s1d7_tracemalloc_call_site_from_marks_resolved() -> None:
    from calm.hrm_text_158.native_full_stack.s1d7_tracemalloc_feasibility import (
        S1D7_POST_EVENT,
        S1D7_PRE_EVENT,
        attribute_s1d7_tracemalloc_call_site_from_marks,
    )

    marks = [
        {
            "event": S1D7_PRE_EVENT,
            "state_index": 0,
            "s1d7_tracemalloc": _s1d7_tracemalloc_snapshot(traced_bytes=0, carrier_line=876),
        },
        {
            "event": S1D7_POST_EVENT,
            "state_index": 0,
            "s1d7_tracemalloc": _s1d7_tracemalloc_snapshot(
                traced_bytes=1_000_000,
                carrier_line=896,
            ),
        },
    ]
    result = attribute_s1d7_tracemalloc_call_site_from_marks(
        marks,
        sampled_states=(0,),
        guards={"perturbation_delta_gib": 0.0, "perturbation_threshold_gib": 0.5},
    )
    assert result["call_site_status"] == "RESOLVED"
    assert result["s1d7_call_site_in_bracket_ok"] is True
    assert result["call_site_origin_file_line"] == "event_coded_acc_live_carrier.py:896"
    assert result["s1d7_call_site_candidate"] == "c"
    assert float(result["s1d7_tracemalloc_top_concentration_fraction"]) >= 0.60


def test_attribute_s1d7_tracemalloc_call_site_perturbation_fail_closed() -> None:
    from calm.hrm_text_158.native_full_stack.s1d7_tracemalloc_feasibility import (
        S1D7_POST_EVENT,
        S1D7_PRE_EVENT,
        attribute_s1d7_tracemalloc_call_site_from_marks,
    )

    marks = [
        {
            "event": S1D7_PRE_EVENT,
            "state_index": 0,
            "s1d7_tracemalloc": _s1d7_tracemalloc_snapshot(traced_bytes=0, carrier_line=876),
        },
        {
            "event": S1D7_POST_EVENT,
            "state_index": 0,
            "s1d7_tracemalloc": _s1d7_tracemalloc_snapshot(
                traced_bytes=1_000_000,
                carrier_line=896,
            ),
        },
    ]
    result = attribute_s1d7_tracemalloc_call_site_from_marks(
        marks,
        sampled_states=(0,),
        guards={"perturbation_delta_gib": 1.0, "perturbation_threshold_gib": 0.5},
    )
    assert result["call_site_status"] == "UNRESOLVED"
    assert result["tracemalloc_perturbed"] is True
    assert result["fail_closed_reason"] == "TRACEMALLOC_PERTURBED_INCONCLUSIVE"


def test_obmalloc_expanded_propagates_tracemalloc_call_site_when_resolved() -> None:
    from scripts.hrm_text_158_slice5_v6i_oom_profile_attribution import (
        TOTAL_C4_REFERENCE_GIB,
        attribute_obmalloc_expanded,
    )

    from calm.llm_computer.tests.test_slice5_v6i_oom_profile_attribution_v1 import (
        _c4_subphase_marks,
        _obmalloc_expanded_boundary_marks,
        _obmalloc_expanded_preflight,
        _obmalloc_expanded_site_marks_for_state,
        _obmalloc_expanded_phase3_full_holding_deltas,
    )
    from calm.hrm_text_158.native_full_stack.s1d7_tracemalloc_feasibility import (
        S1D7_POST_EVENT,
        S1D7_PRE_EVENT,
    )

    sampled = (0, 10, 21, 31)
    site_marks: list[dict] = []
    for state_idx in sampled:
        site_marks.extend(
            _obmalloc_expanded_site_marks_for_state(
                state_index=int(state_idx),
                leaf_holding_deltas=_obmalloc_expanded_phase3_full_holding_deltas(
                    100_000_000,
                    s1d_reconcile=True,
                ),
            )
        )
    for row in site_marks:
        event = str(row.get("event") or "")
        if event == S1D7_PRE_EVENT:
            row["s1d7_tracemalloc"] = _s1d7_tracemalloc_snapshot(
                traced_bytes=0,
                carrier_line=876,
            )
        elif event == S1D7_POST_EVENT:
            row["s1d7_tracemalloc"] = _s1d7_tracemalloc_snapshot(
                traced_bytes=500_000,
                carrier_line=905,
            )
    marks = (
        _c4_subphase_marks(TOTAL_C4_REFERENCE_GIB)
        + _obmalloc_expanded_boundary_marks(after_state_blocks=[200_000_000] * 8)
        + site_marks
    )
    result = attribute_obmalloc_expanded(
        marks_a=_c4_subphase_marks(TOTAL_C4_REFERENCE_GIB),
        marks_a_prime=_c4_subphase_marks(TOTAL_C4_REFERENCE_GIB),
        marks_b=marks,
        sampled_states=sampled,
        **_obmalloc_expanded_preflight(),
    )
    assert result["call_site_status"] == "RESOLVED"
    assert result["s1d7_call_site_in_bracket_ok"] is True
    assert result["call_site_origin_file_line"] == "event_coded_acc_live_carrier.py:905"
    assert result["s1d7_call_site_candidate"] == "e"
    assert result["s1d7_call_site_branch_outcome"] == "S1D7_CALL_SITE_CANDIDATE_E_NUMPY_ARRAYS"
    assert result["s1d7_tracemalloc_mark_pair_count"] == len(sampled)
    assert float(result["s1d7_tracemalloc_top_concentration_fraction"]) >= 0.60
