from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path
from unittest.mock import patch

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
            '  File "/repo/calm/hrm_text_158/native_full_stack/event_coded_acc_live_carrier.py", line 917'
        ]
    )
    assert origin_file == "event_coded_acc_live_carrier.py"
    assert origin_line == 917
    diff = {
        "current_delta_bytes": 1000,
        "top_concentration_fraction": 0.9,
        "top_delta_frames": [
            {
                "delta_bytes": 1000,
                "concentration_fraction": 0.9,
                "traceback": [
                    '  File "/repo/calm/hrm_text_158/native_full_stack/event_coded_acc_live_carrier.py", line 917'
                ],
            }
        ],
    }
    inside = classify_s1d7_tracemalloc_call_site(diff)
    assert inside["s1d7_call_site_in_bracket_ok"] is True
    assert inside["call_site_origin_file_line"] == "event_coded_acc_live_carrier.py:917"
    assert inside["s1d7_call_site_candidate"] == "e"
    outside = classify_s1d7_tracemalloc_call_site(
        {
            **diff,
            "top_delta_frames": [
                {
                    **diff["top_delta_frames"][0],
                    "traceback": [
                        '  File "/repo/calm/hrm_text_158/native_full_stack/event_coded_acc_live_carrier.py", line 960'
                    ],
                }
            ],
        }
    )
    assert outside["fail_closed_reason"] == "CALL_SITE_OUTSIDE_S1D7_BRACKET"
    assert outside["s1d7_call_site_in_bracket_ok"] is False
    assert S1D7_ACCEPTANCE_LINE_MIN == 909
    assert S1D7_ACCEPTANCE_LINE_MAX == 955


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
                    '  File "/repo/calm/hrm_text_158/native_full_stack/event_coded_acc_live_carrier.py", line 946'
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
                    '  File "/repo/calm/hrm_text_158/native_full_stack/event_coded_acc_live_carrier.py", line 908'
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

    assert map_s1d7_call_site_candidate(910) == "a"
    assert map_s1d7_call_site_candidate(941) == "c"
    assert map_s1d7_call_site_candidate(952) == "c"
    assert map_s1d7_call_site_candidate(914) == "e"
    assert map_s1d7_call_site_candidate(917) == "e"
    assert map_s1d7_call_site_candidate(920) == "ambiguous"
    assert map_s1d7_call_site_candidate(909) == "ambiguous"
    assert map_s1d7_call_site_candidate(908) is None

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

    a_result = classify_s1d7_tracemalloc_call_site(_diff_for_line(910))
    assert a_result["s1d7_call_site_candidate"] == "a"
    assert a_result["s1d7_call_site_branch_outcome"] == S1D7_BRANCH_CANDIDATE_A

    c_result = classify_s1d7_tracemalloc_call_site(_diff_for_line(946))
    assert c_result["s1d7_call_site_candidate"] == "c"
    assert c_result["s1d7_call_site_branch_outcome"] == S1D7_BRANCH_CANDIDATE_C

    e_result = classify_s1d7_tracemalloc_call_site(_diff_for_line(917))
    assert e_result["s1d7_call_site_candidate"] == "e"
    assert e_result["s1d7_call_site_branch_outcome"] == S1D7_BRANCH_CANDIDATE_E

    ambiguous = classify_s1d7_tracemalloc_call_site(_diff_for_line(920))
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
            "s1d7_tracemalloc": _s1d7_tracemalloc_snapshot(traced_bytes=0, carrier_line=909),
        },
        {
            "event": S1D7_POST_EVENT,
            "state_index": 0,
            "s1d7_tracemalloc": _s1d7_tracemalloc_snapshot(
                traced_bytes=1_000_000,
                carrier_line=946,
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
    assert result["call_site_origin_file_line"] == "event_coded_acc_live_carrier.py:946"
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
            "s1d7_tracemalloc": _s1d7_tracemalloc_snapshot(traced_bytes=0, carrier_line=909),
        },
        {
            "event": S1D7_POST_EVENT,
            "state_index": 0,
            "s1d7_tracemalloc": _s1d7_tracemalloc_snapshot(
                traced_bytes=1_000_000,
                carrier_line=946,
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
                carrier_line=909,
            )
        elif event == S1D7_POST_EVENT:
            row["s1d7_tracemalloc"] = _s1d7_tracemalloc_snapshot(
                traced_bytes=500_000,
                carrier_line=917,
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
    assert result["call_site_origin_file_line"] == "event_coded_acc_live_carrier.py:917"
    assert result["s1d7_call_site_candidate"] == "e"
    assert result["s1d7_call_site_branch_outcome"] == "S1D7_CALL_SITE_CANDIDATE_E_NUMPY_ARRAYS"
    assert result["s1d7_tracemalloc_mark_pair_count"] == len(sampled)
    assert float(result["s1d7_tracemalloc_top_concentration_fraction"]) >= 0.60


def test_phase3_callsite_classifier_synthetic_branches() -> None:
    from scripts.hrm_text_158_slice5_v6i_oom_profile_attribution import (
        build_phase3_callsite_classifier_receipt_from_attribution_payload,
    )

    def _payload(**expanded_overrides: object) -> dict:
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

    resolved_c = build_phase3_callsite_classifier_receipt_from_attribution_payload(_payload())
    assert resolved_c["branch_outcome"] == "S1D7_CALL_SITE_CANDIDATE_C_EVENTS_JOURNAL"
    assert resolved_c["classifier_exit_code"] == 0

    perturb = build_phase3_callsite_classifier_receipt_from_attribution_payload(
        _payload(tracemalloc_perturbed=True)
    )
    assert perturb["branch_outcome"] == "TRACEMALLOC_PERTURBED_INCONCLUSIVE"
    assert perturb["classifier_exit_code"] == 35

    missing = build_phase3_callsite_classifier_receipt_from_attribution_payload(
        _payload(s1d7_tracemalloc_mark_pair_count=1)
    )
    assert missing["branch_outcome"] == "TRACEMALLOC_INCONCLUSIVE"
    assert missing["classifier_exit_code"] == 35


def _read_profile_marks(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def test_s1d7_tracemalloc_only_probe_emits_four_pairs(monkeypatch, tmp_path: Path) -> None:
    import scripts.hrm_text_158_bounded_delta_acquisition_probe as probe

    monkeypatch.setenv(probe.PROFILE_HOST_RSS_ENV, "1")
    monkeypatch.setenv(probe.PROFILE_TRACEMALLOC_ENV, "1")
    monkeypatch.setenv(probe.PROFILE_DEBUGMALLOCSTATS_ENV, "0")
    monkeypatch.setenv(probe.PROFILE_OBMALLOC_SITE_BRACKETS_ENV, "0")
    monkeypatch.setenv("HRM_TEXT_158_PROFILE_S1D7_TRACEMALLOC_FULL_TRACE", "1")

    from calm.hrm_text_158.native_full_stack.s1d7_tracemalloc_feasibility import (
        S1D7_TRACEMALLOC_POST_EVENT,
        S1D7_TRACEMALLOC_PRE_EVENT,
        S1D7_TRACEMALLOC_SITE_SCHEMA,
    )

    profile_path = tmp_path / "profile_b.jsonl"
    progress = probe.PhaseProgress(
        enabled=True,
        device="cpu",
        host_rss_profile_path=profile_path,
    )
    sampled = (0, 10, 21, 31)
    started = time.monotonic()
    for state_idx in sampled:
        fields = {
            "step": 0,
            "optimizer_step_index": 0,
            "state_index": int(state_idx),
            "sampled_states": list(sampled),
        }
        progress._emit_s1d7_tracemalloc_site_mark(
            event_suffix="pre",
            origin_file="event_coded_acc_live_carrier.py",
            origin_line=876,
            fields=fields,
        )
        assert tracemalloc.is_tracing() is True
        progress._emit_s1d7_tracemalloc_site_mark(
            event_suffix="post",
            origin_file="event_coded_acc_live_carrier.py",
            origin_line=897,
            fields=fields,
        )
        assert tracemalloc.is_tracing() is False
    assert time.monotonic() - started < 30.0

    marks = _read_profile_marks(profile_path)
    pre_events = [row for row in marks if row.get("event") == S1D7_TRACEMALLOC_PRE_EVENT]
    post_events = [row for row in marks if row.get("event") == S1D7_TRACEMALLOC_POST_EVENT]
    assert len(pre_events) == 4
    assert len(post_events) == 4
    assert all(row.get("schema") == S1D7_TRACEMALLOC_SITE_SCHEMA for row in marks)
    assert all(row.get("tracemalloc_only") is True for row in marks)
    assert all(row.get("s1d7_tracemalloc", {}).get("enabled") is True for row in marks)
    assert "measurement_perturbed" not in marks[0]
    obmalloc_events = [
        row for row in marks if str(row.get("event", "")).startswith("obmalloc_site_")
    ]
    assert obmalloc_events == []


def test_s1d7_tracemalloc_bracket_clears_on_post_exception(monkeypatch, tmp_path: Path) -> None:
    import scripts.hrm_text_158_bounded_delta_acquisition_probe as probe

    monkeypatch.setenv(probe.PROFILE_HOST_RSS_ENV, "1")
    monkeypatch.setenv(probe.PROFILE_TRACEMALLOC_ENV, "1")
    monkeypatch.setenv(probe.PROFILE_DEBUGMALLOCSTATS_ENV, "0")
    monkeypatch.setenv(probe.PROFILE_OBMALLOC_SITE_BRACKETS_ENV, "0")
    monkeypatch.setenv("HRM_TEXT_158_PROFILE_S1D7_TRACEMALLOC_FULL_TRACE", "1")

    from calm.hrm_text_158.native_full_stack import host_tracemalloc_probe

    profile_path = tmp_path / "profile_b.jsonl"
    progress = probe.PhaseProgress(
        enabled=True,
        device="cpu",
        host_rss_profile_path=profile_path,
    )
    fields = {
        "step": 0,
        "optimizer_step_index": 0,
        "state_index": 0,
        "sampled_states": [0, 10, 21, 31],
    }
    progress._emit_s1d7_tracemalloc_site_mark(
        event_suffix="pre",
        origin_file="event_coded_acc_live_carrier.py",
        origin_line=876,
        fields=fields,
    )
    with patch(
        "calm.hrm_text_158.native_full_stack.s1d7_tracemalloc_feasibility.take_tracemalloc_snapshot_dict",
        side_effect=RuntimeError("post snap failed"),
    ):
        with pytest.raises(RuntimeError, match="post snap failed"):
            progress._emit_s1d7_tracemalloc_site_mark(
                event_suffix="post",
                origin_file="event_coded_acc_live_carrier.py",
                origin_line=897,
                fields=fields,
            )
    assert tracemalloc.is_tracing() is False
    from calm.hrm_text_158.native_full_stack import host_tracemalloc_probe

    assert host_tracemalloc_probe._tracemalloc_started is False


def test_smoke_fixture_sampled_states_n32() -> None:
    from calm.hrm_text_158.native_full_stack.sparse_cap_gpu_seam_adapter import (
        compute_obmalloc_expanded_sampled_states,
    )

    n_c4_states = 32
    sampled = compute_obmalloc_expanded_sampled_states(n_c4_states)
    assert sampled == frozenset({0, 10, 21, 31})


def test_s1d7_tracemalloc_only_mutual_exclusion_still_aborts() -> None:
    import scripts.hrm_text_158_bounded_delta_acquisition_probe as probe

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    env[probe.PROFILE_HOST_RSS_ENV] = "1"
    env[probe.PROFILE_TRACEMALLOC_ENV] = "1"
    env[probe.PROFILE_DEBUGMALLOCSTATS_ENV] = "1"
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import scripts.hrm_text_158_bounded_delta_acquisition_probe as p; "
            "p.assert_profile_tracemalloc_debugmallocstats_mutual_exclusion()",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert proc.returncode != 0
    assert "profile_env_mutual_exclusion_abort" in proc.stdout


def test_legacy_debugmallocstats_obmalloc_embed_unchanged(monkeypatch, tmp_path: Path) -> None:
    import scripts.hrm_text_158_bounded_delta_acquisition_probe as probe

    monkeypatch.setenv(probe.PROFILE_HOST_RSS_ENV, "1")
    monkeypatch.setenv(probe.PROFILE_DEBUGMALLOCSTATS_ENV, "1")
    monkeypatch.setenv(probe.PROFILE_OBMALLOC_SITE_BRACKETS_ENV, "1")
    monkeypatch.setenv(probe.PROFILE_TRACEMALLOC_ENV, "0")

    from calm.hrm_text_158.native_full_stack.s1d7_tracemalloc_feasibility import (
        S1D7_POST_EVENT,
        S1D7_PRE_EVENT,
        S1D7_TRACEMALLOC_POST_EVENT,
        S1D7_TRACEMALLOC_PRE_EVENT,
    )

    profile_path = tmp_path / "profile_b.jsonl"
    progress = probe.PhaseProgress(
        enabled=True,
        device="cpu",
        host_rss_profile_path=profile_path,
    )
    fields = {
        "step": 0,
        "optimizer_step_index": 0,
        "state_index": 0,
        "sampled_states": [0, 10, 21, 31],
    }
    progress._emit_obmalloc_site_bracket_mark(
        site_id="C4.S1d.7",
        event_suffix="pre",
        origin_file="event_coded_acc_live_carrier.py",
        origin_line=876,
        fields=fields,
    )
    progress._emit_s1d7_tracemalloc_site_mark(
        event_suffix="pre",
        origin_file="event_coded_acc_live_carrier.py",
        origin_line=876,
        fields=fields,
    )

    marks = _read_profile_marks(profile_path)
    assert any(row.get("event") == S1D7_PRE_EVENT for row in marks)
    assert not any(row.get("event") == S1D7_TRACEMALLOC_PRE_EVENT for row in marks)
    assert all("s1d7_tracemalloc" not in row for row in marks)


def test_consumer_prefers_new_schema_over_legacy() -> None:
    from calm.hrm_text_158.native_full_stack.s1d7_tracemalloc_feasibility import (
        S1D7_POST_EVENT,
        S1D7_PRE_EVENT,
        S1D7_TRACEMALLOC_POST_EVENT,
        S1D7_TRACEMALLOC_PRE_EVENT,
        attribute_s1d7_tracemalloc_call_site_from_marks,
    )

    marks = [
        {
            "event": S1D7_PRE_EVENT,
            "state_index": 0,
            "s1d7_tracemalloc": _s1d7_tracemalloc_snapshot(traced_bytes=0, carrier_line=910),
        },
        {
            "event": S1D7_POST_EVENT,
            "state_index": 0,
            "s1d7_tracemalloc": _s1d7_tracemalloc_snapshot(
                traced_bytes=1_000_000,
                carrier_line=910,
            ),
        },
        {
            "event": S1D7_TRACEMALLOC_PRE_EVENT,
            "state_index": 0,
            "s1d7_tracemalloc": _s1d7_tracemalloc_snapshot(traced_bytes=0, carrier_line=909),
        },
        {
            "event": S1D7_TRACEMALLOC_POST_EVENT,
            "state_index": 0,
            "s1d7_tracemalloc": _s1d7_tracemalloc_snapshot(
                traced_bytes=1_000_000,
                carrier_line=946,
            ),
        },
    ]
    result = attribute_s1d7_tracemalloc_call_site_from_marks(
        marks,
        sampled_states=(0,),
        guards={"perturbation_delta_gib": 0.0, "perturbation_threshold_gib": 0.5},
    )
    assert result["s1d7_tracemalloc_mark_schema"] == "tracemalloc_only"
    assert result["call_site_status"] == "RESOLVED"
    assert result["call_site_origin_file_line"] == "event_coded_acc_live_carrier.py:946"
    assert result["s1d7_call_site_candidate"] == "c"


def test_consumer_missing_extra_pairs_fail_closed() -> None:
    from calm.hrm_text_158.native_full_stack.s1d7_tracemalloc_feasibility import (
        S1D7_TRACEMALLOC_POST_EVENT,
        S1D7_TRACEMALLOC_PRE_EVENT,
        attribute_s1d7_tracemalloc_call_site_from_marks,
    )

    missing_post = [
        {
            "event": S1D7_TRACEMALLOC_PRE_EVENT,
            "state_index": 0,
            "s1d7_tracemalloc": _s1d7_tracemalloc_snapshot(traced_bytes=0, carrier_line=909),
        }
    ]
    missing = attribute_s1d7_tracemalloc_call_site_from_marks(
        missing_post,
        sampled_states=(0,),
        guards={"perturbation_delta_gib": 0.0, "perturbation_threshold_gib": 0.5},
    )
    assert missing["fail_closed_reason"] == "TRACEMALLOC_MISSING_PAIR"

    duplicate_pre = [
        {
            "event": S1D7_TRACEMALLOC_PRE_EVENT,
            "state_index": 0,
            "s1d7_tracemalloc": _s1d7_tracemalloc_snapshot(traced_bytes=0, carrier_line=909),
        },
        {
            "event": S1D7_TRACEMALLOC_PRE_EVENT,
            "state_index": 0,
            "s1d7_tracemalloc": _s1d7_tracemalloc_snapshot(traced_bytes=0, carrier_line=909),
        },
        {
            "event": S1D7_TRACEMALLOC_POST_EVENT,
            "state_index": 0,
            "s1d7_tracemalloc": _s1d7_tracemalloc_snapshot(
                traced_bytes=1_000_000,
                carrier_line=946,
            ),
        },
    ]
    duplicate = attribute_s1d7_tracemalloc_call_site_from_marks(
        duplicate_pre,
        sampled_states=(0,),
        guards={"perturbation_delta_gib": 0.0, "perturbation_threshold_gib": 0.5},
    )
    assert duplicate["fail_closed_reason"] == "TRACEMALLOC_DUPLICATE_PRE"


def test_callsite_b_prime_b_arm_launch_composition_dry_check() -> None:
    import os

    from scripts.hrm_text_158_bounded_delta_acquisition_probe import (
        PROFILE_DEBUGMALLOCSTATS_ENV,
        PROFILE_OBMALLOC_EXPANDED_ENV,
        PROFILE_OBMALLOC_SITE_BRACKETS_ENV,
        PROFILE_TRACEMALLOC_ENV,
    )
    from scripts.hrm_text_158_slice5_v6i_oom_profile_attribution import (
        FIXTURE_PROBE_MAX_SILENT_PHASE_SECONDS_TRACEMALLOC,
        dry_check_callsite_b_prime_b_arm_launch_composition,
    )

    prior = {
        PROFILE_DEBUGMALLOCSTATS_ENV: os.environ.get(PROFILE_DEBUGMALLOCSTATS_ENV),
        PROFILE_OBMALLOC_SITE_BRACKETS_ENV: os.environ.get(PROFILE_OBMALLOC_SITE_BRACKETS_ENV),
        PROFILE_OBMALLOC_EXPANDED_ENV: os.environ.get(PROFILE_OBMALLOC_EXPANDED_ENV),
        PROFILE_TRACEMALLOC_ENV: os.environ.get(PROFILE_TRACEMALLOC_ENV),
    }
    os.environ[PROFILE_DEBUGMALLOCSTATS_ENV] = "1"
    os.environ[PROFILE_OBMALLOC_SITE_BRACKETS_ENV] = "1"
    os.environ[PROFILE_OBMALLOC_EXPANDED_ENV] = "1"
    os.environ[PROFILE_TRACEMALLOC_ENV] = "0"
    try:
        receipt = dry_check_callsite_b_prime_b_arm_launch_composition()
    finally:
        for key, value in prior.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    assert receipt["ok"] is True, receipt
    toggles = receipt["env_profile_toggles"]
    assert toggles[PROFILE_DEBUGMALLOCSTATS_ENV] == "0"
    assert toggles[PROFILE_OBMALLOC_SITE_BRACKETS_ENV] == "0"
    assert toggles[PROFILE_OBMALLOC_EXPANDED_ENV] == "0"
    assert toggles[PROFILE_TRACEMALLOC_ENV] == "1"
    cmd = receipt["cmd"]
    assert "-B" in cmd
    assert "hrm_text_158_bounded_delta_acquisition_probe_bootstrap.py" in " ".join(cmd)
    assert str(FIXTURE_PROBE_MAX_SILENT_PHASE_SECONDS_TRACEMALLOC) in cmd
    assert receipt["checks"]["guard_dry_check_passes"] is True


def test_band_counter_envelope_legacy_marks_suppressed() -> None:
    import numpy as np

    from calm.hrm_text_158.native_full_stack.event_coded_acc_live_carrier import (
        EventCodedAccLiveState,
    )
    from calm.hrm_text_158.native_full_stack.s1d7_band_counter import (
        S1D7_BAND_COUNTER_EVENT,
    )

    legacy_marks: list[dict[str, object]] = []
    counter_marks: list[dict[str, object]] = []

    def site_emit(
        site_id: str,
        event_suffix: str,
        *,
        origin_file: str,
        origin_line: int,
        optimizer_step_index: int,
        state_index: int,
    ) -> None:
        legacy_marks.append(
            {
                "site_id": site_id,
                "event_suffix": event_suffix,
                "state_index": state_index,
            }
        )

    def band_counter_emit(
        *,
        origin_file: str,
        origin_line: int,
        counters: dict[str, object],
        optimizer_step_index: int,
        state_index: int,
    ) -> None:
        counter_marks.append(
            {
                "event": S1D7_BAND_COUNTER_EVENT,
                "state_index": state_index,
                "counters": counters,
            }
        )

    carrier = EventCodedAccLiveState(logical_numel=100, threshold_abs=10)
    indices = np.arange(20, dtype=np.int32)
    values = np.full(20, 9, dtype=np.int16)
    carrier._hot.replace_arrays(indices, values)
    vote_values = np.full(20, 4, dtype=np.int32)
    carrier.apply_step(
        0,
        sparse_vote_indices=indices.astype(np.int64),
        sparse_vote_values=vote_values,
        host_allocator_site_emit=site_emit,
        site_emit_enabled=False,
        s1d7_band_counter_emit=band_counter_emit,
        state_index=0,
        optimizer_step_index=0,
    )
    assert legacy_marks == []
    assert len(counter_marks) == 1
    assert counter_marks[0]["event"] == S1D7_BAND_COUNTER_EVENT


def test_band_counter_wrapper_forwards_emit_kwarg() -> None:
    """GPU seam calls apply_event_coded_carrier_step — not carrier.apply_step directly."""
    import numpy as np

    from calm.hrm_text_158.native_full_stack.event_coded_acc_live_carrier import (
        EventCodedAccLiveState,
    )
    from calm.hrm_text_158.native_full_stack.event_coded_vote_update_adapter import (
        apply_event_coded_carrier_step,
    )
    from calm.hrm_text_158.native_full_stack.s1d7_band_counter import (
        S1D7_BAND_COUNTER_EVENT,
    )

    counter_marks: list[dict[str, object]] = []

    def band_counter_emit(
        *,
        origin_file: str,
        origin_line: int,
        counters: dict[str, object],
        optimizer_step_index: int,
        state_index: int,
    ) -> None:
        counter_marks.append(
            {
                "event": S1D7_BAND_COUNTER_EVENT,
                "state_index": state_index,
                "counters": counters,
            }
        )

    carrier = EventCodedAccLiveState(logical_numel=100, threshold_abs=10)
    indices = np.arange(20, dtype=np.int32)
    values = np.full(20, 9, dtype=np.int16)
    carrier._hot.replace_arrays(indices, values)

    apply_event_coded_carrier_step(
        carrier,
        votes={int(idx): 4 for idx in indices},
        step_index=0,
        site_emit_enabled=False,
        s1d7_band_counter_emit=band_counter_emit,
        state_index=0,
        optimizer_step_index=0,
    )

    assert len(counter_marks) >= 1
    assert counter_marks[0]["event"] == S1D7_BAND_COUNTER_EVENT
    assert counter_marks[0]["state_index"] == 0


def test_band_counter_exactly_four_marks_across_sampled_states() -> None:
    import numpy as np

    from calm.hrm_text_158.native_full_stack.event_coded_acc_live_carrier import (
        EventCodedAccLiveState,
    )
    from calm.hrm_text_158.native_full_stack.s1d7_band_counter import (
        S1D7_BAND_COUNTER_EVENT,
    )
    from calm.hrm_text_158.native_full_stack.sparse_cap_gpu_seam_adapter import (
        compute_obmalloc_expanded_sampled_states,
    )

    sampled = compute_obmalloc_expanded_sampled_states(32)
    counter_marks: list[dict[str, object]] = []

    def band_counter_emit(
        *,
        origin_file: str,
        origin_line: int,
        counters: dict[str, object],
        optimizer_step_index: int,
        state_index: int,
    ) -> None:
        counter_marks.append(
            {
                "event": S1D7_BAND_COUNTER_EVENT,
                "state_index": state_index,
                "s1d7_band_counters": counters,
            }
        )

    carrier = EventCodedAccLiveState(logical_numel=200, threshold_abs=10)
    indices = np.arange(50, dtype=np.int32)
    values = np.full(50, 9, dtype=np.int16)
    vote_values = np.full(50, 4, dtype=np.int32)
    for state_idx in sampled:
        carrier._hot.replace_arrays(indices, values)
        carrier.apply_step(
            int(state_idx),
            sparse_vote_indices=indices.astype(np.int64),
            sparse_vote_values=vote_values,
            s1d7_band_counter_emit=band_counter_emit,
            state_index=int(state_idx),
            optimizer_step_index=int(state_idx),
        )
    assert len(counter_marks) == len(sampled)
    assert {int(row["state_index"]) for row in counter_marks} == set(sampled)


def test_band_counter_apply_cap_seam_n4_four_distinct_marks() -> None:
    import torch

    from calm.hrm_text_158.native_full_stack.event_coded_vote_update_adapter import (
        plan_event_coded_integer_vote_update,
    )
    from calm.hrm_text_158.native_full_stack.global_rate_cap import GlobalRateCapTensorResult
    from calm.hrm_text_158.native_full_stack.s1d7_band_counter import (
        S1D7_BAND_COUNTER_EVENT,
        attribute_s1d7_band_counter_call_site_from_marks,
    )
    from calm.hrm_text_158.native_full_stack.sparse_cap_gpu_seam_adapter import (
        apply_cap_tensor_result_gpu,
        compute_obmalloc_expanded_sampled_states,
        reset_obmalloc_site_emit_dedup_session,
    )
    from calm.hrm_text_158.native_full_stack.vote_update import VoteUpdateInputs, VoteUpdateSpec
    from calm.llm_computer.tests.test_sparse_event_coded_planner_v1_parity_v0 import (
        _make_state,
        _vote_spec,
        _votes_for_indices,
    )

    if not torch.cuda.is_available():
        pytest.skip("CUDA required for apply_cap band-counter seam regression")

    reset_obmalloc_site_emit_dedup_session()
    numel = 64
    state = _make_state(numel=numel)
    state_key = "probe_state"
    votes = _votes_for_indices({0: 12, 3: -9}, numel=numel)
    inputs = VoteUpdateInputs(votes=votes)
    spec = _vote_spec()
    plan = plan_event_coded_integer_vote_update(state, inputs, spec)
    event_states = {state_key: state}
    plans_by_key = {state_key: plan}
    inputs_by_key = {state_key: inputs}
    vote_specs_by_key = {state_key: spec}
    accepted_flat_by_key = {state_key: tuple(int(x) for x in plan.applied_indices.tolist())}
    q_gpu = state.q_levels.cuda()
    item = GlobalRateCapTensorResult(
        state_key=state_key,
        q_levels=q_gpu,
        accumulators=torch.zeros(numel, dtype=torch.int16, device="cuda"),
        stats={},
    )
    sampled_states = compute_obmalloc_expanded_sampled_states(4)
    sampled_tuple = tuple(sorted(sampled_states))
    counter_marks: list[dict[str, object]] = []

    def band_counter_emit(
        *,
        origin_file: str,
        origin_line: int,
        counters: dict[str, object],
        optimizer_step_index: int,
        state_index: int,
    ) -> None:
        counter_marks.append(
            {
                "event": S1D7_BAND_COUNTER_EVENT,
                "state_index": int(state_index),
                "sampled_states": list(sampled_tuple),
                "s1d7_band_counters": counters,
            }
        )

    def merge_stats(a: dict[str, object], b: dict[str, object]) -> dict[str, object]:
        merged = dict(a)
        merged.update(b)
        return merged

    for state_index in sampled_tuple:
        apply_cap_tensor_result_gpu(
            item,
            event_states=event_states,
            plans_by_key=plans_by_key,
            inputs_by_key=inputs_by_key,
            vote_specs_by_key=vote_specs_by_key,
            accepted_flat_by_key=accepted_flat_by_key,
            local_selection_ordering_step=1,
            cap_boundary_transient=0,
            cap_item_stats={},
            merge_stats_fn=merge_stats,
            state_index=int(state_index),
            s1d7_band_counter_emit=band_counter_emit,
            sampled_states=sampled_states,
        )

    assert len(counter_marks) == 4
    assert {int(row["state_index"]) for row in counter_marks} == set(sampled_tuple)
    assert all(row.get("sampled_states") == list(sampled_tuple) for row in counter_marks)

    partial_marks = [row for row in counter_marks if int(row["state_index"]) == 0]
    partial = attribute_s1d7_band_counter_call_site_from_marks(
        partial_marks,
        sampled_states=sampled_tuple,
        guards={},
    )
    assert partial["call_site_status"] == "UNRESOLVED"
    assert partial["fail_closed_reason"] == "BAND_COUNTER_ROW_COUNT_MISMATCH"

    duplicate_marks = [
        counter_marks[0],
        counter_marks[0],
        counter_marks[1],
        counter_marks[2],
    ]
    duplicate = attribute_s1d7_band_counter_call_site_from_marks(
        duplicate_marks,
        sampled_states=sampled_tuple,
        guards={},
    )
    assert duplicate["call_site_status"] == "UNRESOLVED"
    assert duplicate["fail_closed_reason"] == "BAND_COUNTER_DUPLICATE_ROW"


def test_band_counter_apply_cap_seam_sampled_states_none_state_zero_only() -> None:
    import torch

    from calm.hrm_text_158.native_full_stack.event_coded_vote_update_adapter import (
        plan_event_coded_integer_vote_update,
    )
    from calm.hrm_text_158.native_full_stack.global_rate_cap import GlobalRateCapTensorResult
    from calm.hrm_text_158.native_full_stack.s1d7_band_counter import S1D7_BAND_COUNTER_EVENT
    from calm.hrm_text_158.native_full_stack.sparse_cap_gpu_seam_adapter import (
        apply_cap_tensor_result_gpu,
        reset_obmalloc_site_emit_dedup_session,
    )
    from calm.hrm_text_158.native_full_stack.vote_update import VoteUpdateInputs
    from calm.llm_computer.tests.test_sparse_event_coded_planner_v1_parity_v0 import (
        _make_state,
        _vote_spec,
        _votes_for_indices,
    )

    if not torch.cuda.is_available():
        pytest.skip("CUDA required for apply_cap band-counter seam regression")

    reset_obmalloc_site_emit_dedup_session()
    numel = 64
    state = _make_state(numel=numel)
    state_key = "probe_state"
    votes = _votes_for_indices({0: 12, 3: -9}, numel=numel)
    inputs = VoteUpdateInputs(votes=votes)
    spec = _vote_spec()
    plan = plan_event_coded_integer_vote_update(state, inputs, spec)
    event_states = {state_key: state}
    plans_by_key = {state_key: plan}
    inputs_by_key = {state_key: inputs}
    vote_specs_by_key = {state_key: spec}
    accepted_flat_by_key = {state_key: tuple(int(x) for x in plan.applied_indices.tolist())}
    q_gpu = state.q_levels.cuda()
    item = GlobalRateCapTensorResult(
        state_key=state_key,
        q_levels=q_gpu,
        accumulators=torch.zeros(numel, dtype=torch.int16, device="cuda"),
        stats={},
    )
    counter_marks: list[dict[str, object]] = []

    def band_counter_emit(
        *,
        origin_file: str,
        origin_line: int,
        counters: dict[str, object],
        optimizer_step_index: int,
        state_index: int,
    ) -> None:
        counter_marks.append(
            {
                "event": S1D7_BAND_COUNTER_EVENT,
                "state_index": int(state_index),
                "s1d7_band_counters": counters,
            }
        )

    def merge_stats(a: dict[str, object], b: dict[str, object]) -> dict[str, object]:
        merged = dict(a)
        merged.update(b)
        return merged

    for state_index in (0, 1, 2, 3):
        apply_cap_tensor_result_gpu(
            item,
            event_states=event_states,
            plans_by_key=plans_by_key,
            inputs_by_key=inputs_by_key,
            vote_specs_by_key=vote_specs_by_key,
            accepted_flat_by_key=accepted_flat_by_key,
            local_selection_ordering_step=1,
            cap_boundary_transient=0,
            cap_item_stats={},
            merge_stats_fn=merge_stats,
            state_index=int(state_index),
            s1d7_band_counter_emit=band_counter_emit,
            sampled_states=None,
        )

    assert len(counter_marks) == 1
    assert int(counter_marks[0]["state_index"]) == 0


def test_band_counter_emit_reads_sampled_states_from_emit_attr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    from calm.hrm_text_158.native_full_stack.s1d7_band_counter import (
        S1D7_BAND_COUNTER_EVENT,
        collect_s1d7_band_counters,
    )
    from calm.hrm_text_158.native_full_stack.sparse_cap_gpu_seam_adapter import (
        compute_obmalloc_expanded_sampled_states,
    )
    from scripts.hrm_text_158_bounded_delta_acquisition_probe import PhaseProgress

    monkeypatch.setenv("HRM_TEXT_158_PROFILE_HOST_RSS", "1")
    monkeypatch.setenv("HRM_TEXT_158_PROFILE_TRACEMALLOC", "1")
    monkeypatch.setenv("HRM_TEXT_158_PROFILE_DEBUGMALLOCSTATS", "0")
    profile_path = tmp_path / "host_rss_profile.jsonl"
    progress = PhaseProgress(enabled=False, device=torch.device("cpu"))
    progress.host_rss_profile_path = profile_path
    emit = progress.make_host_rss_subphase_emitter(step=1)
    assert emit is not None
    band_counter_emit = getattr(emit, "band_counter_emit")
    sampled_tuple = tuple(sorted(compute_obmalloc_expanded_sampled_states(4)))
    setattr(emit, "_obmalloc_expanded_sampled_states", sampled_tuple)
    import numpy as np

    band_counter_emit(
        origin_file="event_coded_acc_live_carrier.py",
        origin_line=909,
        counters=collect_s1d7_band_counters(
            crossing_indices_len=1,
            applied_indices_len=1,
            append_event_count=1,
            event_encoded_bytes_delta=1,
            q_level_writes=1,
            remove_idx=np.empty(0, dtype=np.int32),
            upd_idx=np.empty(0, dtype=np.int32),
            upd_val=np.empty(0, dtype=np.int16),
        ),
        optimizer_step_index=1,
        state_index=0,
    )
    rows = [
        json.loads(line)
        for line in profile_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    band_rows = [row for row in rows if row.get("event") == S1D7_BAND_COUNTER_EVENT]
    assert len(band_rows) == 1
    assert band_rows[0]["sampled_states"] == list(sampled_tuple)


def test_band_counter_byte_model_applied_indices_shallow_copy() -> None:
    from calm.hrm_text_158.native_full_stack.s1d7_band_counter import (
        estimate_band_a_allocation_bytes,
        PY_LIST_HEADER_BYTES,
        PY_POINTER_SLOT_BYTES,
        PYLONG_OBJECT_BYTES,
    )

    n = 1000
    expected = (
        PY_LIST_HEADER_BYTES + n * PY_POINTER_SLOT_BYTES + n * PYLONG_OBJECT_BYTES
        + PY_LIST_HEADER_BYTES + n * PY_POINTER_SLOT_BYTES
    )
    assert estimate_band_a_allocation_bytes(
        crossing_indices_len=n,
        applied_indices_len=n,
    ) == expected


def test_band_counter_dominance_gate_fail_closed_on_all_zero() -> None:
    from calm.hrm_text_158.native_full_stack.s1d7_band_counter import (
        evaluate_band_dominance,
    )

    rows = [
        {
            "state_index": 0,
            "s1d7_band_counters": {
                "byte_proxies": {
                    "band_a_bytes": 0,
                    "band_c_bytes": 0,
                    "band_e_bytes": 0,
                }
            },
        }
    ]
    result = evaluate_band_dominance(rows, sampled_states=(0,))
    assert result["band_counter_dominance_ok"] is False
    assert result["fail_closed_reason"] == "BAND_COUNTER_ALL_ZERO_ACTIVITY"


def test_band_counter_attribute_resolves_candidate_c() -> None:
    from calm.hrm_text_158.native_full_stack.s1d7_band_counter import (
        S1D7_BAND_COUNTER_EVENT,
        attribute_s1d7_band_counter_call_site_from_marks,
        collect_s1d7_band_counters,
    )
    import numpy as np

    counters = collect_s1d7_band_counters(
        crossing_indices_len=10,
        applied_indices_len=10,
        append_event_count=50_000,
        event_encoded_bytes_delta=500_000,
        q_level_writes=50_000,
        remove_idx=np.empty(0, dtype=np.int32),
        upd_idx=np.empty(0, dtype=np.int32),
        upd_val=np.empty(0, dtype=np.int16),
    )
    marks = [
        {
            "event": S1D7_BAND_COUNTER_EVENT,
            "state_index": state_idx,
            "s1d7_band_counters": counters,
        }
        for state_idx in (0, 10, 21, 31)
    ]
    result = attribute_s1d7_band_counter_call_site_from_marks(
        marks,
        sampled_states=(0, 10, 21, 31),
        guards={},
    )
    assert result["call_site_status"] == "RESOLVED"
    assert result["s1d7_call_site_candidate"] == "c"
    assert result["s1d7_band_counter_mark_count"] == 4
    assert result["tracemalloc_perturbed"] is False
    assert result["call_site_origin_file_line"] == "event_coded_acc_live_carrier.py:896"
    assert (
        result["s1d7_band_counter_candidate_origin_file_line"]
        == "event_coded_acc_live_carrier.py:896"
    )


def _band_counter_mark(
    *,
    state_index: int,
    counters: dict[str, object] | None = None,
) -> dict[str, object]:
    from calm.hrm_text_158.native_full_stack.s1d7_band_counter import (
        S1D7_BAND_COUNTER_EVENT,
        collect_s1d7_band_counters,
    )
    import numpy as np

    if counters is None:
        counters = collect_s1d7_band_counters(
            crossing_indices_len=10,
            applied_indices_len=10,
            append_event_count=50_000,
            event_encoded_bytes_delta=500_000,
            q_level_writes=50_000,
            remove_idx=np.empty(0, dtype=np.int32),
            upd_idx=np.empty(0, dtype=np.int32),
            upd_val=np.empty(0, dtype=np.int16),
        )
    return {
        "event": S1D7_BAND_COUNTER_EVENT,
        "state_index": state_index,
        "s1d7_band_counters": counters,
    }


def test_band_counter_exact_row_duplicate_state_fail_closed() -> None:
    from calm.hrm_text_158.native_full_stack.s1d7_band_counter import (
        attribute_s1d7_band_counter_call_site_from_marks,
    )

    sampled = (0, 10, 21, 31)
    marks = [
        _band_counter_mark(state_index=0),
        _band_counter_mark(state_index=0),
        _band_counter_mark(state_index=10),
        _band_counter_mark(state_index=21),
    ]
    result = attribute_s1d7_band_counter_call_site_from_marks(
        marks,
        sampled_states=sampled,
        guards={},
    )
    assert result["call_site_status"] == "UNRESOLVED"
    assert result["fail_closed_reason"] == "BAND_COUNTER_DUPLICATE_ROW"


def test_band_counter_exact_row_unexpected_state_fail_closed() -> None:
    from calm.hrm_text_158.native_full_stack.s1d7_band_counter import (
        attribute_s1d7_band_counter_call_site_from_marks,
    )

    sampled = (0, 10, 21, 31)
    marks = [_band_counter_mark(state_index=state_idx) for state_idx in (0, 10, 21, 99)]
    result = attribute_s1d7_band_counter_call_site_from_marks(
        marks,
        sampled_states=sampled,
        guards={},
    )
    assert result["call_site_status"] == "UNRESOLVED"
    assert result["fail_closed_reason"] == "BAND_COUNTER_UNEXPECTED_STATE"


def test_band_counter_exact_row_count_mismatch_fail_closed() -> None:
    from calm.hrm_text_158.native_full_stack.s1d7_band_counter import (
        attribute_s1d7_band_counter_call_site_from_marks,
    )

    sampled = (0, 10, 21, 31)
    marks = [_band_counter_mark(state_index=state_idx) for state_idx in sampled] + [
        _band_counter_mark(state_index=5)
    ]
    result = attribute_s1d7_band_counter_call_site_from_marks(
        marks,
        sampled_states=sampled,
        guards={},
    )
    assert result["call_site_status"] == "UNRESOLVED"
    assert result["fail_closed_reason"] == "BAND_COUNTER_ROW_COUNT_MISMATCH"


def test_band_counter_empty_path_emits_explicit_zero_row() -> None:
    import numpy as np

    from calm.hrm_text_158.native_full_stack.event_coded_acc_live_carrier import (
        EventCodedAccLiveState,
    )

    captured: list[dict[str, object]] = []

    def band_counter_emit(
        *,
        origin_file: str,
        origin_line: int,
        counters: dict[str, object],
        optimizer_step_index: int,
        state_index: int,
    ) -> None:
        captured.append(dict(counters))

    carrier = EventCodedAccLiveState(logical_numel=10, threshold_abs=10)
    carrier.apply_step(
        0,
        votes={},
        s1d7_band_counter_emit=band_counter_emit,
        state_index=0,
        optimizer_step_index=0,
    )
    assert len(captured) == 1
    counts = dict(captured[0].get("counts") or {})
    assert counts["append_event_count"] == 0
    assert counts["crossing_indices_len"] == 0


def test_band_counter_calibration_forced_c() -> None:
    from calm.hrm_text_158.native_full_stack.s1d7_tracemalloc_feasibility import (
        calibrate_band_counters_vs_classifier,
    )

    result = calibrate_band_counters_vs_classifier(case="forced_c")
    assert result["ok"] is True, result
    assert result["counter_band"] == "c"
    assert result["classifier_candidate"] == "c"


def test_band_counter_calibration_synthetic_a() -> None:
    from calm.hrm_text_158.native_full_stack.s1d7_tracemalloc_feasibility import (
        calibrate_band_counters_vs_classifier,
    )

    result = calibrate_band_counters_vs_classifier(case="synthetic_a")
    assert result["ok"] is True
    assert result["counter_band"] == "a"


def test_band_counter_calibration_forced_e_real_path_not_classifier_aligned() -> None:
    from calm.hrm_text_158.native_full_stack.s1d7_tracemalloc_feasibility import (
        calibrate_band_counters_vs_classifier,
    )

    result = calibrate_band_counters_vs_classifier(case="forced_e")
    assert result["counter_band"] == "e"
    assert result["classifier_candidate"] is None
    assert result["ok"] is False


def test_band_counter_calibration_synthetic_e() -> None:
    from calm.hrm_text_158.native_full_stack.s1d7_tracemalloc_feasibility import (
        calibrate_band_counters_vs_classifier,
    )

    result = calibrate_band_counters_vs_classifier(case="synthetic_e")
    assert result["ok"] is True, result
    assert result["counter_band"] == "e"
    assert result["classifier_candidate"] == "e"


def test_band_counter_mode_skips_tracemalloc_marks(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    from scripts.hrm_text_158_bounded_delta_acquisition_probe import (
        PhaseProgress,
        profile_s1d7_band_counter_enabled,
        profile_s1d7_tracemalloc_full_trace_enabled,
    )

    monkeypatch.setenv("HRM_TEXT_158_PROFILE_HOST_RSS", "1")
    monkeypatch.setenv("HRM_TEXT_158_PROFILE_TRACEMALLOC", "1")
    monkeypatch.setenv("HRM_TEXT_158_PROFILE_DEBUGMALLOCSTATS", "0")
    monkeypatch.delenv("HRM_TEXT_158_PROFILE_S1D7_BAND_COUNTER_ONLY", raising=False)
    assert profile_s1d7_band_counter_enabled() is True
    assert profile_s1d7_tracemalloc_full_trace_enabled() is False

    progress = PhaseProgress(enabled=False, device=torch.device("cpu"))
    progress.host_rss_profile_path = Path("/tmp/unused_band_counter_test.jsonl")
    marks_before = len(progress.events)
    progress._emit_s1d7_tracemalloc_site_mark(
        event_suffix="pre",
        origin_file="event_coded_acc_live_carrier.py",
        origin_line=876,
        fields={"state_index": 0},
    )
    assert len(progress.events) == marks_before


def test_band_counter_only_decoupled_from_tracemalloc_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from calm.hrm_text_158.native_full_stack.host_tracemalloc_probe import (
        profile_s1d7_band_counter_enabled,
        profile_s1d7_band_counter_only_enabled,
        profile_s1d7_tracemalloc_site_enabled,
    )

    monkeypatch.setenv("HRM_TEXT_158_PROFILE_HOST_RSS", "1")
    monkeypatch.setenv("HRM_TEXT_158_PROFILE_TRACEMALLOC", "0")
    monkeypatch.setenv("HRM_TEXT_158_PROFILE_DEBUGMALLOCSTATS", "0")
    monkeypatch.setenv("HRM_TEXT_158_PROFILE_S1D7_BAND_COUNTER_ONLY", "1")
    assert profile_s1d7_band_counter_only_enabled() is True
    assert profile_s1d7_band_counter_enabled() is True
    assert profile_s1d7_tracemalloc_site_enabled() is False


def test_static_pre_append_projection_parity_varint_boundaries() -> None:
    import numpy as np

    from calm.hrm_text_158.native_full_stack.event_coded_acc_live_carrier import (
        VARINT_BOUNDARY_FLAT_INDICES,
    )
    from calm.hrm_text_158.native_full_stack.s1d7_band_counter import (
        oracle_measure_crossing_event_encoded_bytes_delta,
        project_crossing_event_encoded_bytes_delta,
    )

    threshold_abs = 16
    posts = (threshold_abs - 1, -(threshold_abs - 1))
    for flat_index in VARINT_BOUNDARY_FLAT_INDICES:
        for post in posts:
            active_sorted = np.array([int(flat_index)], dtype=np.int32)
            cross_mask = np.array([True], dtype=bool)
            post_arr = np.array([int(post)], dtype=np.int16)
            projected = project_crossing_event_encoded_bytes_delta(
                active_sorted,
                cross_mask,
                post_arr,
                threshold_abs=threshold_abs,
            )
            oracle = oracle_measure_crossing_event_encoded_bytes_delta(
                active_sorted,
                cross_mask,
                post_arr,
                threshold_abs=threshold_abs,
            )
            assert projected == oracle

    all_indices = np.array(list(VARINT_BOUNDARY_FLAT_INDICES), dtype=np.int32)
    cross_mask = np.ones(all_indices.shape[0], dtype=bool)
    post_arr = np.full(all_indices.shape[0], threshold_abs - 1, dtype=np.int16)
    projected_batch = project_crossing_event_encoded_bytes_delta(
        all_indices,
        cross_mask,
        post_arr,
        threshold_abs=threshold_abs,
    )
    oracle_batch = oracle_measure_crossing_event_encoded_bytes_delta(
        all_indices,
        cross_mask,
        post_arr,
        threshold_abs=threshold_abs,
    )
    assert projected_batch == oracle_batch


def test_static_pre_append_emit_marker_metadata() -> None:
    import numpy as np

    from calm.hrm_text_158.native_full_stack.event_coded_acc_live_carrier import (
        EventCodedAccLiveState,
    )
    from calm.hrm_text_158.native_full_stack.s1d7_band_counter import (
        STATIC_PRE_APPEND_MEASUREMENT_CONTRACT,
        S1D7_BAND_COUNTER_MARKER_SEAM_ORIGIN_LINE,
    )

    captured: list[dict[str, object]] = []

    def band_counter_emit(
        *,
        origin_file: str,
        origin_line: int,
        counters: dict[str, object],
        optimizer_step_index: int,
        state_index: int,
        measurement_contract: str | None = None,
        event_encoded_bytes_delta_source: str | None = None,
    ) -> None:
        captured.append(
            {
                "origin_file": origin_file,
                "origin_line": origin_line,
                "measurement_contract": measurement_contract,
                "event_encoded_bytes_delta_source": event_encoded_bytes_delta_source,
                "counters": counters,
            }
        )

    carrier = EventCodedAccLiveState(logical_numel=300, threshold_abs=10)
    indices = np.array([127, 128], dtype=np.int32)
    values = np.array([9, -9], dtype=np.int16)
    vote_values = np.array([5, -5], dtype=np.int32)
    carrier._hot.replace_arrays(indices, values)
    carrier.apply_step(
        0,
        sparse_vote_indices=indices.astype(np.int64),
        sparse_vote_values=vote_values,
        s1d7_band_counter_emit=band_counter_emit,
        state_index=0,
        optimizer_step_index=0,
    )
    assert len(captured) == 1
    row = captured[0]
    assert row["origin_file"] == "event_coded_acc_live_carrier.py"
    assert row["origin_line"] == S1D7_BAND_COUNTER_MARKER_SEAM_ORIGIN_LINE
    assert row["measurement_contract"] == STATIC_PRE_APPEND_MEASUREMENT_CONTRACT
    assert row["event_encoded_bytes_delta_source"] == "projected"


def test_static_pre_append_origin_line_audit_resolves_candidate_896() -> None:
    import numpy as np

    from calm.hrm_text_158.native_full_stack.s1d7_band_counter import (
        S1D7_BAND_COUNTER_EVENT,
        S1D7_BAND_COUNTER_MARKER_SEAM_ORIGIN_LINE,
        attribute_s1d7_band_counter_call_site_from_marks,
        collect_s1d7_band_counters,
    )

    counters = collect_s1d7_band_counters(
        crossing_indices_len=10,
        applied_indices_len=10,
        append_event_count=50_000,
        event_encoded_bytes_delta=500_000,
        q_level_writes=50_000,
        remove_idx=np.empty(0, dtype=np.int32),
        upd_idx=np.empty(0, dtype=np.int32),
        upd_val=np.empty(0, dtype=np.int16),
    )
    marks = [
        {
            "event": S1D7_BAND_COUNTER_EVENT,
            "state_index": state_idx,
            "origin_file": "event_coded_acc_live_carrier.py",
            "origin_line": S1D7_BAND_COUNTER_MARKER_SEAM_ORIGIN_LINE,
            "measurement_contract": "static_pre_append_v1",
            "s1d7_band_counters": counters,
        }
        for state_idx in (0, 10, 21, 31)
    ]
    result = attribute_s1d7_band_counter_call_site_from_marks(
        marks,
        sampled_states=(0, 10, 21, 31),
        guards={},
    )
    assert result["call_site_status"] == "RESOLVED"
    assert result["call_site_origin_file_line"] == "event_coded_acc_live_carrier.py:896"
    assert (
        result["s1d7_band_counter_marker_origin_file_line"]
        == "event_coded_acc_live_carrier.py:895"
    )
    assert (
        result["s1d7_band_counter_candidate_origin_file_line"]
        == "event_coded_acc_live_carrier.py:896"
    )


def _row18_share_fail_counters() -> dict[str, object]:
    import numpy as np

    from calm.hrm_text_158.native_full_stack.s1d7_band_counter import (
        collect_s1d7_band_counters,
    )

    n = 512
    remove_idx = np.arange(n, dtype=np.int32)
    upd_idx = np.arange(n, dtype=np.int32)
    upd_val = np.full(n, 1, dtype=np.int16)
    return collect_s1d7_band_counters(
        crossing_indices_len=n,
        applied_indices_len=n,
        append_event_count=n,
        event_encoded_bytes_delta=1536,
        q_level_writes=n,
        remove_idx=remove_idx,
        upd_idx=upd_idx,
        upd_val=upd_val,
    )


def test_band_counter_four_row_share_fail_is_valid_null_not_liveness_regression() -> None:
    from calm.hrm_text_158.native_full_stack.s1d7_band_counter import (
        S1D7_BAND_COUNTER_EVENT,
        S1D7_BAND_COUNTER_MARKER_SEAM_ORIGIN_LINE,
        attribute_s1d7_band_counter_call_site_from_marks,
    )

    counters = _row18_share_fail_counters()
    marks = [
        {
            "event": S1D7_BAND_COUNTER_EVENT,
            "state_index": state_idx,
            "origin_file": "event_coded_acc_live_carrier.py",
            "origin_line": S1D7_BAND_COUNTER_MARKER_SEAM_ORIGIN_LINE,
            "measurement_contract": "static_pre_append_v1",
            "s1d7_band_counters": counters,
        }
        for state_idx in (0, 1, 2, 3)
    ]
    result = attribute_s1d7_band_counter_call_site_from_marks(
        marks,
        sampled_states=(0, 1, 2, 3),
        guards={},
    )
    assert result["call_site_status"] == "UNRESOLVED"
    assert result["s1d7_band_counter_mark_count"] == 4
    assert result["fail_closed_reason"] == "BAND_COUNTER_C_SHARE_FAIL"
    dominance = dict(result.get("s1d7_band_counter_dominance") or {})
    assert float(dominance.get("band_c_share") or 0.0) < 0.80
    assert (
        result["s1d7_band_counter_marker_origin_file_line"]
        == "event_coded_acc_live_carrier.py:895"
    )
    assert (
        result["s1d7_band_counter_candidate_origin_file_line"]
        == "event_coded_acc_live_carrier.py:896"
    )


def test_callsite_band_counter_scale_smoke_b_arm_uses_decoupled_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.hrm_text_158_slice5_v6i_oom_profile_attribution import (
        run_callsite_band_counter_scale_smoke,
    )

    captured_kwargs: list[dict[str, object]] = []

    def _fake_probe(
        out_root: Path,
        *,
        scratch_name: str,
        debugmallocstats: bool,
        **kwargs: object,
    ) -> dict[str, object]:
        captured_kwargs.append(
            {
                "scratch_name": scratch_name,
                **kwargs,
            }
        )
        return _band_counter_scale_smoke_probe_return(
            scratch_name=scratch_name,
            marks_b=[],
        )

    monkeypatch.setattr(
        "scripts.hrm_text_158_slice5_v6i_oom_profile_attribution._run_fixture_obmalloc_probe",
        _fake_probe,
    )
    run_callsite_band_counter_scale_smoke(tmp_path)
    b_kwargs = next(row for row in captured_kwargs if row["scratch_name"] == "callsite_band_counter_b")
    assert b_kwargs["tracemalloc"] is False
    assert b_kwargs["band_counter_only"] is True


def _band_counter_scale_smoke_probe_return(
    *,
    scratch_name: str,
    marks_b: list[dict[str, object]],
    n_c4_states: int = 4,
) -> dict[str, object]:
    from calm.hrm_text_158.native_full_stack.sparse_cap_gpu_seam_adapter import (
        compute_obmalloc_expanded_sampled_states,
    )

    sampled = sorted(compute_obmalloc_expanded_sampled_states(n_c4_states))
    if scratch_name.endswith("_a"):
        return {
            "marks": [],
            "profile_mark_count": 0,
            "exit_code": 0,
            "wall_seconds": 1.0,
            "n_c4_states": n_c4_states,
            "sampled_states": sampled,
            "eligible_scope": "all-bitlinear",
            "eligible_module_limit": n_c4_states,
            "eligible_module_keys": [f"module_{idx}" for idx in range(n_c4_states)],
            "c4_rss_delta_gib": 0.01,
        }
    return {
        "marks": marks_b,
        "profile_mark_count": len(marks_b),
        "exit_code": 0,
        "wall_seconds": 8.0,
        "n_c4_states": n_c4_states,
        "sampled_states": sampled,
        "eligible_scope": "all-bitlinear",
        "eligible_module_limit": n_c4_states,
        "eligible_module_keys": [f"module_{idx}" for idx in range(n_c4_states)],
        "c4_rss_delta_gib": 0.42,
    }


def test_band_counter_scale_smoke_receipt_surfaces_dominance_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.hrm_text_158_slice5_v6i_oom_profile_attribution import (
        run_callsite_band_counter_scale_smoke,
    )

    marks_b = [
        _band_counter_mark(state_index=state_idx)
        for state_idx in (0, 1, 2, 3)
    ]

    def _fake_probe(
        out_root: Path,
        *,
        scratch_name: str,
        debugmallocstats: bool,
        **kwargs: object,
    ) -> dict[str, object]:
        _ = (out_root, debugmallocstats, kwargs)
        return _band_counter_scale_smoke_probe_return(
            scratch_name=scratch_name,
            marks_b=marks_b,
        )

    monkeypatch.setattr(
        "scripts.hrm_text_158_slice5_v6i_oom_profile_attribution._run_fixture_obmalloc_probe",
        _fake_probe,
    )
    receipt = run_callsite_band_counter_scale_smoke(tmp_path)
    assert receipt["schema"] == "hrm_text_158_callsite_band_counter_scale_smoke_receipt/v1"
    assert receipt["ok"] is True, receipt
    dominance = dict(receipt.get("s1d7_band_counter_dominance") or {})
    assert dominance.get("band_counter_dominance_ok") is True
    assert float(dominance.get("band_c_share") or 0.0) >= 0.80
    assert dict(dominance.get("aggregate_band_bytes") or {})
    assert list(dominance.get("per_state") or [])
    assert receipt["s1d7_band_counter_mark_count"] == 4
    assert receipt["s1d7_tracemalloc_mark_count"] == 0
    assert receipt["s1d7_call_site_candidate"] == "c"
    assert receipt["call_site_status"] == "RESOLVED"
    assert receipt["mechanism_smoke_scale"] == "reduced_n4"
    assert receipt["n_c4_states"] == 4
    assert receipt["sampled_states"] == [0, 1, 2, 3]
    assert receipt["eligible_module_limit"] == 4
    assert receipt["per_state_timing_method"] == "approx=run_b_wall_seconds/n_c4_states"
    assert receipt["bank_wording"]
    assert receipt["s1d7_call_site_branch_outcome"] is not None
    assert receipt["call_site_origin_file_line"] == "event_coded_acc_live_carrier.py:896"
    nested = dict(dict(receipt.get("localization") or {}).get("s1d7_tracemalloc_call_site") or {})
    assert nested.get("s1d7_band_counter_mark_count") == 4
    checks = dict(receipt.get("checks") or {})
    assert checks["s1d7_band_counter_mark_count_eq_4"] is True
    assert checks["band_counter_dominance_ok"] is True
    assert checks["tracemalloc_mark_count_eq_0"] is True
    assert checks["s1d7_call_site_candidate_eq_c"] is True


def test_band_counter_scale_smoke_fail_closed_on_wrong_mark_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.hrm_text_158_slice5_v6i_oom_profile_attribution import (
        run_callsite_band_counter_scale_smoke,
    )

    marks_b = [_band_counter_mark(state_index=0), _band_counter_mark(state_index=10)]

    def _fake_probe(
        out_root: Path,
        *,
        scratch_name: str,
        debugmallocstats: bool,
        **kwargs: object,
    ) -> dict[str, object]:
        _ = (out_root, debugmallocstats, kwargs)
        return _band_counter_scale_smoke_probe_return(
            scratch_name=scratch_name,
            marks_b=marks_b,
        )

    monkeypatch.setattr(
        "scripts.hrm_text_158_slice5_v6i_oom_profile_attribution._run_fixture_obmalloc_probe",
        _fake_probe,
    )
    receipt = run_callsite_band_counter_scale_smoke(tmp_path)
    assert receipt["ok"] is False
    assert receipt["call_site_status"] == "UNRESOLVED"
    checks = dict(receipt.get("checks") or {})
    assert checks["s1d7_band_counter_mark_count_eq_4"] is False
    assert checks["band_counter_dominance_ok"] is False
    assert checks["call_site_status_resolved"] is False


def test_run_callsite_tracemalloc_scale_smoke_unchanged_for_fallback_d() -> None:
    import inspect

    from scripts.hrm_text_158_slice5_v6i_oom_profile_attribution import (
        run_callsite_tracemalloc_scale_smoke,
    )

    source = inspect.getsource(run_callsite_tracemalloc_scale_smoke)
    assert "s1d7_tracemalloc_mark_pair_count_eq_4" in source
    assert "new_schema_mark_count_eq_8" in source
    assert "hrm_text_158_callsite_tracemalloc_scale_smoke_receipt/v1" in source
    assert "s1d7_band_counter_mark_count_eq_4" not in source


def test_band_counter_smoke_passes_explicit_sampled_states_n4_not_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from calm.hrm_text_158.native_full_stack.sparse_cap_gpu_seam_adapter import (
        compute_obmalloc_expanded_sampled_states,
    )
    from scripts.hrm_text_158_slice5_v6i_oom_profile_attribution import (
        attribute_callsite_tracemalloc_b_prime,
        run_callsite_band_counter_scale_smoke,
    )

    marks_b = [_band_counter_mark(state_index=idx) for idx in (0, 1, 2, 3)]
    captured: dict[str, object] = {}

    def _fake_attribute(
        *,
        marks_a: list[dict[str, object]],
        marks_b: list[dict[str, object]],
        sampled_states: tuple[int, ...] = (0, 10, 21, 31),
    ) -> dict[str, object]:
        _ = marks_a
        captured["sampled_states"] = tuple(sampled_states)
        return attribute_callsite_tracemalloc_b_prime(
            marks_a=marks_a,
            marks_b=marks_b,
            sampled_states=sampled_states,
        )

    def _fake_probe(
        out_root: Path,
        *,
        scratch_name: str,
        debugmallocstats: bool,
        **kwargs: object,
    ) -> dict[str, object]:
        _ = (out_root, debugmallocstats, kwargs)
        return _band_counter_scale_smoke_probe_return(
            scratch_name=scratch_name,
            marks_b=marks_b,
            n_c4_states=4,
        )

    monkeypatch.setattr(
        "scripts.hrm_text_158_slice5_v6i_oom_profile_attribution._run_fixture_obmalloc_probe",
        _fake_probe,
    )
    monkeypatch.setattr(
        "scripts.hrm_text_158_slice5_v6i_oom_profile_attribution.attribute_callsite_tracemalloc_b_prime",
        _fake_attribute,
    )
    receipt = run_callsite_band_counter_scale_smoke(tmp_path)
    expected = tuple(sorted(compute_obmalloc_expanded_sampled_states(4)))
    assert captured["sampled_states"] == expected
    assert captured["sampled_states"] != (0, 10, 21, 31)
    assert receipt["ok"] is True, receipt


def test_band_counter_smoke_false_fails_if_default_sampled_states_leaks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.hrm_text_158_slice5_v6i_oom_profile_attribution import (
        attribute_callsite_tracemalloc_b_prime as original_attr,
        run_callsite_band_counter_scale_smoke,
    )

    marks_b = [_band_counter_mark(state_index=idx) for idx in (0, 1, 2, 3)]

    def _fake_probe(
        out_root: Path,
        *,
        scratch_name: str,
        debugmallocstats: bool,
        **kwargs: object,
    ) -> dict[str, object]:
        _ = (out_root, debugmallocstats, kwargs)
        return _band_counter_scale_smoke_probe_return(
            scratch_name=scratch_name,
            marks_b=marks_b,
            n_c4_states=4,
        )

    def _leak_default(
        *,
        marks_a: list[dict[str, object]],
        marks_b: list[dict[str, object]],
        sampled_states: tuple[int, ...] = (0, 10, 21, 31),
    ) -> dict[str, object]:
        _ = sampled_states
        return original_attr(marks_a=marks_a, marks_b=marks_b)

    monkeypatch.setattr(
        "scripts.hrm_text_158_slice5_v6i_oom_profile_attribution._run_fixture_obmalloc_probe",
        _fake_probe,
    )
    monkeypatch.setattr(
        "scripts.hrm_text_158_slice5_v6i_oom_profile_attribution.attribute_callsite_tracemalloc_b_prime",
        _leak_default,
    )
    receipt = run_callsite_band_counter_scale_smoke(tmp_path)
    assert receipt["ok"] is False
    assert receipt["call_site_status"] == "UNRESOLVED"


def test_probe_heartbeat_watchdog_resets_on_real_probe_heartbeat_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json
    import threading
    import time

    from scripts.hrm_text_158_slice5_v6i_oom_profile_attribution import (
        _run_subprocess_heartbeat_watchdog,
    )

    log_path = tmp_path / "probe_stream.log"
    heartbeat = json.dumps(
        {
            "event": "heartbeat",
            "phase": "sparse_cap_apply",
            "active_phase_elapsed_seconds": 30,
            "schema": "hrm_text_158_c2p2_phase_telemetry/v0",
        }
    )

    def _fake_popen(cmd, cwd, env, stdout, stderr, text):
        _ = (cmd, cwd, env, stderr, text)

        class _Proc:
            returncode = None

            def poll(self):
                return self.returncode

            def wait(self):
                return self.returncode if self.returncode is not None else 0

            def kill(self):
                self.returncode = -9

        proc = _Proc()

        def _writer() -> None:
            time.sleep(0.05)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(heartbeat + "\n")
                handle.flush()
            time.sleep(0.1)
            proc.returncode = 0

        threading.Thread(target=_writer, daemon=True).start()
        return proc

    monkeypatch.setattr(
        "scripts.hrm_text_158_slice5_v6i_oom_profile_attribution.subprocess.Popen",
        _fake_popen,
    )
    result = _run_subprocess_heartbeat_watchdog(
        ["python3", "-c", "import time; time.sleep(1)"],
        cwd=tmp_path,
        env={},
        log_path=log_path,
        silent_phase_seconds=2.0,
        wall_clock_cap_seconds=5.0,
        poll_interval_seconds=0.05,
    )
    assert result["subprocess_timeout_expired"] is False
    assert result["exit_code"] == 0


def test_probe_heartbeat_watchdog_kills_on_silent_budget_without_heartbeats(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import time

    from scripts.hrm_text_158_slice5_v6i_oom_profile_attribution import (
        _run_subprocess_heartbeat_watchdog,
    )

    log_path = tmp_path / "silent_probe_stream.log"

    class _Proc:
        returncode = None

        def poll(self):
            return None

        def wait(self):
            return -9

        def kill(self):
            self.returncode = -9

    proc = _Proc()

    def _fake_popen(cmd, cwd, env, stdout, stderr, text):
        _ = (cmd, cwd, env, stdout, stderr, text)
        return proc

    monkeypatch.setattr(
        "scripts.hrm_text_158_slice5_v6i_oom_profile_attribution.subprocess.Popen",
        _fake_popen,
    )
    monkeypatch.setattr(
        "scripts.hrm_text_158_slice5_v6i_oom_profile_attribution.time.sleep",
        lambda _seconds: None,
    )
    tick = {"value": 0.0}

    def _monotonic() -> float:
        tick["value"] += 1.0
        return tick["value"]

    monkeypatch.setattr(
        "scripts.hrm_text_158_slice5_v6i_oom_profile_attribution.time.monotonic",
        _monotonic,
    )
    result = _run_subprocess_heartbeat_watchdog(
        ["python3", "-c", "import time; time.sleep(9)"],
        cwd=tmp_path,
        env={},
        log_path=log_path,
        silent_phase_seconds=0.5,
        wall_clock_cap_seconds=10.0,
        poll_interval_seconds=0.01,
    )
    assert result["subprocess_timeout_expired"] is True
    assert result["subprocess_timeout_reason"] == "silent_phase"


def test_apply_eligible_module_limit_prefix_is_deterministic() -> None:
    from scripts.hrm_text_158_bounded_delta_acquisition_probe import (
        apply_eligible_module_limit,
    )

    class _Module:
        pass

    eligible = {f"layers.{idx}.bitlinear": _Module() for idx in (5, 1, 3, 0, 2, 4)}
    limited = apply_eligible_module_limit(
        eligible,
        eligible_scope="all-bitlinear",
        eligible_module_limit=4,
    )
    assert list(limited.keys()) == sorted(eligible.keys())[:4]
    assert list(limited.keys()) == [
        "layers.0.bitlinear",
        "layers.1.bitlinear",
        "layers.2.bitlinear",
        "layers.3.bitlinear",
    ]
