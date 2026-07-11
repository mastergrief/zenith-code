"""CPU learner-hook tests for selective-drain census (default-off identity + fail-before-publish)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest import mock

import pytest

from calm.hrm_text_158.native_full_stack.r7_selective_drain_eligibility_census import (
    ObserverContinuityTracker,
    maybe_run_selective_drain_census,
)


@dataclass(frozen=True)
class _Row:
    state_key: str
    flat_index: int
    abs_new_acc: int
    threshold_abs: int = 10


@dataclass(frozen=True)
class _CapResult:
    accepted_rows: list
    deferred_rows: list
    deferred_backlog: dict
    step_summary: dict


def test_default_off_does_not_construct_when_disabled():
    with mock.patch(
        "calm.hrm_text_158.native_full_stack.r7_selective_drain_eligibility_census.build_selective_drain_census_step_dto"
    ) as b:
        out = maybe_run_selective_drain_census(
            enabled=False,
            pre_step_backlog={"w": {1: {"first_step": 0, "last_deferred_step": 0, "defer_count": 1}}},
            cap_result=_CapResult([_Row("w", 1, 100)], [_Row("w", 2, 50)], {}, {"global_rate_cap_cap": 1}),
            plans_by_key=None,
            step=0,
            tracker=ObserverContinuityTracker(),
        )
        assert out is None
        b.assert_not_called()


def test_injected_exception_rolls_back_tracker_and_no_publish_side_effect(tmp_path):
    tr = ObserverContinuityTracker()
    tr.reset()
    cap = _CapResult(
        [_Row("w", 1, 100)],
        [_Row("w", 2, 50)],
        {"w": {2: {"first_step": 0, "last_deferred_step": 0, "defer_count": 1}}},
        {"global_rate_cap_cap": 1},
    )
    pre = {"w": {2: {"first_step": 0, "last_deferred_step": 0, "defer_count": 1}}}
    published = {"done": False}

    def boom(*args, **kwargs):
        published["done"] = True
        raise RuntimeError("injected")

    with mock.patch(
        "calm.hrm_text_158.native_full_stack.r7_selective_drain_eligibility_census.append_census_chunk",
        side_effect=boom,
    ):
        with pytest.raises(RuntimeError, match="injected"):
            maybe_run_selective_drain_census(
                enabled=True,
                pre_step_backlog=pre,
                pre_step_backlog_before_cap=pre,
                cap_result=cap,
                plans_by_key=None,
                step=0,
                tracker=tr,
                sidecar_path=tmp_path / "c.jsonl",
            )
    # tracker rolled back to empty reset state (no successful update retention on failure after update?)
    # our impl updates tracker then append — on append fail, rolls back records
    assert tr.cardinality() == 0
    assert list(tmp_path.iterdir()) == [] or not (tmp_path / "c.jsonl").exists() or (tmp_path / "c.jsonl").read_text() == ""


def test_pre_input_unchanged_assert():
    pre = {"w": {1: {"first_step": 0, "last_deferred_step": 0, "defer_count": 1}}}
    other = {"w": {1: {"first_step": 0, "last_deferred_step": 0, "defer_count": 1}}}
    cap = _CapResult([_Row("w", 1, 100)], [_Row("w", 2, 50)], {}, {"global_rate_cap_cap": 1})
    tr = ObserverContinuityTracker(); tr.reset()
    with pytest.raises(AssertionError, match="pre_step_backlog_input_object_identity"):
        maybe_run_selective_drain_census(
            enabled=True,
            pre_step_backlog=other,
            pre_step_backlog_before_cap=pre,
            cap_result=cap,
            plans_by_key=None,
            step=0,
            tracker=tr,
        )


def test_learner_source_contains_cpu_hook_only():
    from pathlib import Path
    text = Path("calm/hrm_text_158/native_full_stack/bounded_delta_learner.py").read_text()
    # census call appears once in CPU reference block
    assert text.count("maybe_run_selective_drain_census(") == 1
    # GPU seam region should not call census (no call near apply_sparse_event_coded)
    gpu_idx = text.find("apply_sparse_event_coded_cap_via_gpu_seam")
    census_idx = text.find("maybe_run_selective_drain_census(")
    assert gpu_idx != -1 and census_idx != -1
    # census should be after CPU apply_global_rate_cap_reference site, not inside GPU block exclusively —
    # ensure GPU function call is not followed by census before next def at high level: simple check census not between gpu call and C6_deferred
    gpu_block = text[gpu_idx:gpu_idx + 2500]
    assert "maybe_run_selective_drain_census" not in gpu_block
