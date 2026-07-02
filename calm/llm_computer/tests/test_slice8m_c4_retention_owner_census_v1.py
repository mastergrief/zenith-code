from __future__ import annotations

import gc
import os
from typing import Any

import pytest


def test_profile_c4_retention_owner_census_default_off(monkeypatch: pytest.MonkeyPatch) -> None:
    from calm.hrm_text_158.native_full_stack.c4_retention_owner_census import (
        PROFILE_C4_RETENTION_OWNER_CENSUS_ENV,
        begin_c4_retention_owner_census_session,
        profile_c4_retention_owner_census_enabled,
    )

    monkeypatch.delenv(PROFILE_C4_RETENTION_OWNER_CENSUS_ENV, raising=False)
    assert profile_c4_retention_owner_census_enabled() is False
    assert begin_c4_retention_owner_census_session() is None


def test_weakref_registry_gc_death() -> None:
    from calm.hrm_text_158.native_full_stack.c4_retention_owner_census import (
        C4RetentionOwnerWeakrefRegistry,
    )

    class _Carrier:
        def __init__(self, payload: bytes) -> None:
            self.payload = payload

    registry = C4RetentionOwnerWeakrefRegistry()
    carrier = _Carrier(b"x" * 1024)
    registry.register(carrier, owner_tag="new_carriers_by_key")
    assert registry.live_counts_by_tag()["new_carriers_by_key"] == 1
    del carrier
    gc.collect()
    assert registry.all_weakrefs_dead() is True


def test_append_patch_injects_allocation_dims(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from calm.hrm_text_158.native_full_stack import c4_retention_owner_census as census
    from calm.hrm_text_158.native_full_stack.c4_retention_owner_census import (
        PROFILE_C4_RETENTION_OWNER_CENSUS_ENV,
        pending_obmalloc_c4_after_state_allocation_dims,
    )
    import scripts.hrm_text_158_bounded_delta_acquisition_probe as probe

    monkeypatch.setenv(PROFILE_C4_RETENTION_OWNER_CENSUS_ENV, "1")
    census._APPEND_PATCH_INSTALLED = False
    census.install_obmalloc_allocation_dims_append_patch()

    profile_path = tmp_path / "host_rss_profile.jsonl"
    dims = {"c4_n_carriers_by_key": 3, "c4_state_index": 2}
    with pending_obmalloc_c4_after_state_allocation_dims(dims):
        probe._append_host_rss_profile_mark(
            profile_path,
            {
                "event": "obmalloc_C4_after_state",
                "state_index": 2,
                "debugmallocstats": {"bytes_in_allocated_blocks": 123},
            },
        )

    lines = profile_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    import json

    row = json.loads(lines[0])
    assert row["allocation_dims"]["c4_n_carriers_by_key"] == 3


def test_disabled_path_zero_cost_all_off(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from calm.hrm_text_158.native_full_stack import c4_retention_owner_census as census
    from calm.hrm_text_158.native_full_stack.c4_retention_owner_census import (
        PROFILE_C4_RETENTION_OWNER_CENSUS_ENV,
        begin_c4_retention_owner_census_session,
        pending_obmalloc_c4_after_state_allocation_dims,
    )
    import scripts.hrm_text_158_bounded_delta_acquisition_probe as probe

    monkeypatch.delenv(PROFILE_C4_RETENTION_OWNER_CENSUS_ENV, raising=False)
    census._APPEND_PATCH_INSTALLED = False
    assert begin_c4_retention_owner_census_session() is None

    profile_path = tmp_path / "host_rss_profile.jsonl"
    with pending_obmalloc_c4_after_state_allocation_dims(None):
        probe._append_host_rss_profile_mark(
            profile_path,
            {"event": "obmalloc_C4_after_state", "state_index": 0},
        )
    import json

    row = json.loads(profile_path.read_text(encoding="utf-8").strip())
    assert "allocation_dims" not in row


def test_disabled_path_7c_profiling_on_census_off(monkeypatch: pytest.MonkeyPatch) -> None:
    from calm.hrm_text_158.native_full_stack.c4_retention_owner_census import (
        PROFILE_C4_RETENTION_OWNER_CENSUS_ENV,
        begin_c4_retention_owner_census_session,
        profile_c4_retention_owner_census_enabled,
    )
    from scripts.hrm_text_158_bounded_delta_acquisition_probe import (
        PROFILE_DEBUGMALLOCSTATS_ENV,
        PROFILE_HOST_RSS_ENV,
        PROFILE_OBMALLOC_EXPANDED_ENV,
    )

    monkeypatch.setenv(PROFILE_HOST_RSS_ENV, "1")
    monkeypatch.setenv(PROFILE_DEBUGMALLOCSTATS_ENV, "1")
    monkeypatch.setenv(PROFILE_OBMALLOC_EXPANDED_ENV, "1")
    monkeypatch.delenv(PROFILE_C4_RETENTION_OWNER_CENSUS_ENV, raising=False)
    assert profile_c4_retention_owner_census_enabled() is False
    assert begin_c4_retention_owner_census_session() is None


def _census_row(
    *,
    state_index: int,
    carriers: int,
    tensor_states: int,
    blocks: int,
) -> dict[str, Any]:
    return {
        "state_index": state_index,
        "allocation_dims": {
            "c4_n_carriers_by_key": carriers,
            "c4_n_tensor_states": tensor_states,
            "c4_weakref_n_new_carriers_by_key": carriers,
        },
        "debugmallocstats": {
            "available": True,
            "parse_ok": True,
            "bytes_in_allocated_blocks": blocks,
            "arena_bytes": blocks + 1_000_000,
        },
        "bytes_in_allocated_blocks": blocks,
        "arena_bytes": blocks + 1_000_000,
    }


def test_classifier_new_accum_dominant() -> None:
    # Precedence test (b): new carriers accumulate while priors do NOT persist
    # at n (c4_n_tensor_states DROPS across the loop, i.e. priors released).
    # This is the genuine single-owner new-accumulation case -> NEW_ACCUM_DOMINANT.
    from scripts.hrm_text_158_slice5_v6i_oom_profile_attribution import (
        attribute_c4_retention_owner_census,
    )

    rows = [
        _census_row(state_index=0, carriers=1, tensor_states=32, blocks=1_000_000_000),
        _census_row(state_index=10, carriers=11, tensor_states=24, blocks=1_600_000_000),
        _census_row(state_index=21, carriers=22, tensor_states=16, blocks=2_200_000_000),
        _census_row(state_index=31, carriers=32, tensor_states=8, blocks=2_800_000_000),
    ]
    marks_b = [
        {
            "schema": "hrm_text_158_host_rss_obmalloc/v1",
            "event": "obmalloc_C4_after_state",
            **row,
        }
        for row in rows
    ]
    result = attribute_c4_retention_owner_census(
        marks_a=[],
        marks_a_prime=[],
        marks_b=marks_b,
    )
    assert result["classifier_terminal"] == "NEW_ACCUM_DOMINANT"
    assert result["localization"]["pre_apply_coowns_priors"] is True
    assert result["localization"]["tier_b_status"] == "ENABLED"


def test_classifier_dual_hold_2n_takes_precedence_over_new_accum() -> None:
    # Precedence test (a): priors persist flat-at-n (c4_n_tensor_states flat at
    # 32 across the loop) AND new carriers accumulate to n (1->32). Both owners
    # hold simultaneously -> the 2n peak -> DUAL_HOLD_2N must take precedence
    # over NEW_ACCUM_DOMINANT even though correlation_fraction >= 0.75 and
    # carriers grow (which alone would satisfy NEW_ACCUM_DOMINANT).
    from scripts.hrm_text_158_slice5_v6i_oom_profile_attribution import (
        attribute_c4_retention_owner_census,
    )

    rows = [
        _census_row(state_index=0, carriers=1, tensor_states=32, blocks=1_000_000_000),
        _census_row(state_index=10, carriers=11, tensor_states=32, blocks=1_600_000_000),
        _census_row(state_index=21, carriers=22, tensor_states=32, blocks=2_200_000_000),
        _census_row(state_index=31, carriers=32, tensor_states=32, blocks=2_800_000_000),
    ]
    marks_b = [
        {
            "schema": "hrm_text_158_host_rss_obmalloc/v1",
            "event": "obmalloc_C4_after_state",
            **row,
        }
        for row in rows
    ]
    result = attribute_c4_retention_owner_census(
        marks_a=[],
        marks_a_prime=[],
        marks_b=marks_b,
    )
    assert result["classifier_terminal"] == "DUAL_HOLD_2N"
    assert result["guards"]["census_row_count"] == 4


def test_classifier_dual_hold_2n() -> None:
    from scripts.hrm_text_158_slice5_v6i_oom_profile_attribution import (
        attribute_c4_retention_owner_census,
    )

    marks_b = [
        {
            "schema": "hrm_text_158_host_rss_obmalloc/v1",
            "event": "obmalloc_C4_after_state",
            **_census_row(state_index=31, carriers=32, tensor_states=32, blocks=2_800_000_000),
        }
    ]
    result = attribute_c4_retention_owner_census(
        marks_a=[],
        marks_a_prime=[],
        marks_b=marks_b,
    )
    assert result["classifier_terminal"] in {"DUAL_HOLD_2N", "INCONCLUSIVE"}


def test_classifier_inconclusive_without_dims() -> None:
    from scripts.hrm_text_158_slice5_v6i_oom_profile_attribution import (
        attribute_c4_retention_owner_census,
    )

    marks_b = [
        {
            "schema": "hrm_text_158_host_rss_obmalloc/v1",
            "event": "obmalloc_C4_after_state",
            "state_index": 0,
            "debugmallocstats": {"bytes_in_allocated_blocks": 100},
        }
    ]
    result = attribute_c4_retention_owner_census(
        marks_a=[],
        marks_a_prime=[],
        marks_b=marks_b,
    )
    assert result["classifier_terminal"] == "INCONCLUSIVE"


def _b_arm_replay_marks() -> list[dict[str, Any]]:
    """Faithful replay of the Slice-8m B-arm host_rss_profile.jsonl shape.

    Mirrors the real on-disk data: the census event rows carry the state index
    inside allocation_dims (c4_state_index) with a top-level state_index of
    None, while the obmalloc_C4_after_state marks carry only a top-level
    state_index and NO allocation_dims. The join between the two is the sourcing
    path under test -- the pre-fix parser dropped every census row because it
    only read the (unset) top-level state_index, so census_row_count was 0.

    Non-census marks (enter/exit boundaries, obmalloc C3/C4 boundaries) are
    included so the test exercises selection out of a mixed stream, not a
    census-only feed.
    """
    marks: list[dict[str, Any]] = []
    # Non-census framing marks.
    marks.append({"event": "enter", "phase": "C4"})
    marks.append(
        {
            "schema": "hrm_text_158_profile_host_rss_mark/v8",
            "event": "obmalloc_C3_exit",
            "debugmallocstats": {"available": True, "bytes_in_allocated_blocks": 100_000_000},
        }
    )
    marks.append(
        {
            "schema": "hrm_text_158_profile_host_rss_mark/v8",
            "event": "obmalloc_C4_enter",
            "debugmallocstats": {"available": True, "bytes_in_allocated_blocks": 120_000_000},
        }
    )
    # Eight in-loop states: priors flat at 32, new carriers 4 -> 32,
    # retention (bytes_in_allocated_blocks) growing monotonically.
    for step, state_index in enumerate(range(3, 32, 4)):
        carriers = (step + 1) * 4
        blocks = 150_000_000 + step * 540_000_000
        # Census event mark: state index lives in allocation_dims; top-level None.
        marks.append(
            {
                "schema": "hrm_text_158_profile_host_rss_mark/v8",
                "event": "c4_retention_owner_census_after_state",
                "state_index": None,
                "allocation_dims": {
                    "c4_state_index": state_index,
                    "c4_n_tensor_states": 32,
                    "c4_n_carriers_by_key": carriers,
                    "c4_n_q_by_key": carriers,
                    "c4_n_next_states": 0,
                    "c4_weakref_n_new_carriers_by_key": carriers,
                    "c4_weakref_n_new_q_by_key": carriers,
                    "c4_weakref_n_prior_tensor_states": 32,
                    "c4_weakref_n_prior_event_states_alias": 32,
                },
            }
        )
        # Obmalloc after-state mark: only top-level state_index, no dims.
        marks.append(
            {
                "schema": "hrm_text_158_profile_host_rss_mark/v8",
                "event": "obmalloc_C4_after_state",
                "state_index": state_index,
                "debugmallocstats": {
                    "available": True,
                    "parse_ok": True,
                    "bytes_in_allocated_blocks": blocks,
                    "arena_bytes": blocks + 14_000_000,
                },
            }
        )
    marks.append(
        {
            "schema": "hrm_text_158_profile_host_rss_mark/v8",
            "event": "obmalloc_C4_exit",
            "debugmallocstats": {"available": True, "bytes_in_allocated_blocks": 700_000_000},
        }
    )
    marks.append({"event": "exit", "phase": "C4"})
    return marks


def test_e2e_census_sourcing_from_full_b_arm_marks() -> None:
    # End-to-end regression for the marks-sourcing/join defect: the pre-fix
    # parser reported census_row_count=0 on the real B-arm data because census
    # rows carry state_index in allocation_dims (c4_state_index) with a None
    # top-level state_index. This test feeds the FULL mixed mark stream (the
    # sourcing path) rather than synthetic census-only rows, so it would have
    # caught the defect. Expected terminal on this data (priors flat at 32,
    # carriers 4->32) is DUAL_HOLD_2N.
    from scripts.hrm_text_158_slice5_v6i_oom_profile_attribution import (
        attribute_c4_retention_owner_census,
    )

    marks_b = _b_arm_replay_marks()
    census_event_count = sum(
        1 for row in marks_b if row.get("event") == "c4_retention_owner_census_after_state"
    )
    assert census_event_count == 8

    result = attribute_c4_retention_owner_census(
        marks_a=[],
        marks_a_prime=[],
        marks_b=marks_b,
    )
    assert result["guards"]["census_row_count"] == 8
    assert result["classifier_terminal"] == "DUAL_HOLD_2N"
    assert result["classifier_terminal"] != "INCONCLUSIVE"
