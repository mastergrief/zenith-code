"""CPU learner-hook tests for selective-drain census (legacy + event-coded)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
import torch

from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    apply_bounded_delta_vote_step,
    make_bounded_tensor_state,
    make_event_coded_live_tensor_state,
    tensor_states_use_event_coded_live_carrier,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap import GlobalRateCapSpec
from calm.hrm_text_158.native_full_stack.r7_selective_drain_eligibility_census import (
    ObserverContinuityTracker,
    maybe_run_selective_drain_census,
)
from calm.hrm_text_158.native_full_stack.vote_update import VoteUpdateSpec

_MAYBE_RUN = (
    "calm.hrm_text_158.native_full_stack.r7_selective_drain_eligibility_census"
    ".maybe_run_selective_drain_census"
)
_APPEND = (
    "calm.hrm_text_158.native_full_stack.r7_selective_drain_eligibility_census"
    ".append_census_chunk"
)
_LEARNER_PATH = Path("calm/hrm_text_158/native_full_stack/bounded_delta_learner.py")


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


def _dense_fixture():
    state = make_bounded_tensor_state(
        "toy.proj",
        torch.tensor([0, 0], dtype=torch.int8),
        0.5,
        torch.zeros(2, dtype=torch.int16),
    )
    votes = torch.tensor([12, 12], dtype=torch.int16)
    spec = VoteUpdateSpec(
        threshold_abs=10,
        accumulator_clip_min=-127,
        accumulator_clip_max=127,
        max_abs_per_tensor=2,
    )
    cap = GlobalRateCapSpec(cap=1, step=0)
    return state, votes, spec, cap


def _event_coded_fixture():
    state = make_event_coded_live_tensor_state(
        "toy.proj",
        torch.tensor([0, 0], dtype=torch.int8),
        0.5,
        demotion_band=1,
    )
    votes = torch.tensor([12, 12], dtype=torch.int16)
    spec = VoteUpdateSpec(
        threshold_abs=10,
        accumulator_clip_min=-127,
        accumulator_clip_max=127,
        max_abs_per_tensor=2,
    )
    cap = GlobalRateCapSpec(cap=1, step=0)
    return state, votes, spec, cap


def test_default_off_does_not_construct_when_disabled():
    with mock.patch(
        "calm.hrm_text_158.native_full_stack.r7_selective_drain_eligibility_census"
        ".build_selective_drain_census_step_dto"
    ) as b:
        out = maybe_run_selective_drain_census(
            enabled=False,
            pre_step_backlog={"w": {1: {"first_step": 0, "last_deferred_step": 0, "defer_count": 1}}},
            cap_result=_CapResult(
                [_Row("w", 1, 100)], [_Row("w", 2, 50)], {}, {"global_rate_cap_cap": 1}
            ),
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

    def boom(*args, **kwargs):
        raise RuntimeError("injected")

    with mock.patch(_APPEND, side_effect=boom):
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
    assert tr.cardinality() == 0
    assert list(tmp_path.iterdir()) == [] or not (tmp_path / "c.jsonl").exists() or (
        tmp_path / "c.jsonl"
    ).read_text() == ""


def test_pre_input_unchanged_assert():
    pre = {"w": {1: {"first_step": 0, "last_deferred_step": 0, "defer_count": 1}}}
    other = {"w": {1: {"first_step": 0, "last_deferred_step": 0, "defer_count": 1}}}
    cap = _CapResult([_Row("w", 1, 100)], [_Row("w", 2, 50)], {}, {"global_rate_cap_cap": 1})
    tr = ObserverContinuityTracker()
    tr.reset()
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


def test_a_dense_legacy_enabled_exactly_one_emit(tmp_path):
    state, votes, spec, cap = _dense_fixture()
    tensor_states = {"toy.proj": state}
    assert tensor_states_use_event_coded_live_carrier(tensor_states) is False
    pre = {"toy.proj": {1: {"first_step": 0, "last_deferred_step": 0, "defer_count": 1}}}
    tr = ObserverContinuityTracker()
    tr.reset()
    sidecar = tmp_path / "on.jsonl"
    with mock.patch(_MAYBE_RUN, wraps=maybe_run_selective_drain_census) as m:
        result = apply_bounded_delta_vote_step(
            tensor_states,
            {"toy.proj": votes},
            {"toy.proj": spec},
            global_cap_spec=cap,
            deferred_backlog=pre,
            local_selection_ordering_step=0,
            r7_selective_drain_eligibility_census_enabled=True,
            r7_selective_drain_eligibility_census_tracker=tr,
            r7_selective_drain_eligibility_census_sidecar_path=sidecar,
        )
    assert m.call_count == 1
    kw = m.call_args.kwargs
    assert kw["pre_step_backlog"] is pre
    assert kw["pre_step_backlog_before_cap"] is pre
    assert kw["enabled"] is True
    assert kw["step"] == 0
    assert sidecar.exists()
    assert len(sidecar.read_text().strip().splitlines()) == 1
    assert result.global_summary["global_rate_cap_enabled"] is True


def test_b_dense_legacy_disabled_byte_state_identity(tmp_path):
    state, votes, spec, cap = _dense_fixture()
    pre = {"toy.proj": {1: {"first_step": 0, "last_deferred_step": 0, "defer_count": 1}}}
    baseline = apply_bounded_delta_vote_step(
        {"toy.proj": make_bounded_tensor_state(
            "toy.proj",
            torch.tensor([0, 0], dtype=torch.int8),
            0.5,
            torch.zeros(2, dtype=torch.int16),
        )},
        {"toy.proj": votes.clone()},
        {"toy.proj": spec},
        global_cap_spec=GlobalRateCapSpec(cap=1, step=0),
        deferred_backlog={
            "toy.proj": {1: {"first_step": 0, "last_deferred_step": 0, "defer_count": 1}}
        },
    )
    with mock.patch(_MAYBE_RUN) as m:
        disabled = apply_bounded_delta_vote_step(
            {"toy.proj": make_bounded_tensor_state(
                "toy.proj",
                torch.tensor([0, 0], dtype=torch.int8),
                0.5,
                torch.zeros(2, dtype=torch.int16),
            )},
            {"toy.proj": votes.clone()},
            {"toy.proj": spec},
            global_cap_spec=GlobalRateCapSpec(cap=1, step=0),
            deferred_backlog={
                "toy.proj": {1: {"first_step": 0, "last_deferred_step": 0, "defer_count": 1}}
            },
            r7_selective_drain_eligibility_census_enabled=False,
            r7_selective_drain_eligibility_census_tracker=ObserverContinuityTracker(),
            r7_selective_drain_eligibility_census_sidecar_path=tmp_path / "off.jsonl",
        )
    m.assert_not_called()
    assert not (tmp_path / "off.jsonl").exists()
    assert (
        baseline.tensor_states["toy.proj"].q_levels.tolist()
        == disabled.tensor_states["toy.proj"].q_levels.tolist()
    )
    assert (
        baseline.tensor_states["toy.proj"].exact_accumulator_shadow.tolist()
        == disabled.tensor_states["toy.proj"].exact_accumulator_shadow.tolist()
    )
    assert baseline.deferred_backlog == disabled.deferred_backlog
    assert baseline.global_summary == disabled.global_summary


def test_c_event_coded_live_enabled_exactly_one_no_regression(tmp_path):
    state, votes, spec, cap = _event_coded_fixture()
    tensor_states = {"toy.proj": state}
    assert tensor_states_use_event_coded_live_carrier(tensor_states) is True
    tr = ObserverContinuityTracker()
    tr.reset()
    with mock.patch(_MAYBE_RUN, return_value=None) as m:
        apply_bounded_delta_vote_step(
            tensor_states,
            {"toy.proj": votes},
            {"toy.proj": spec},
            global_cap_spec=cap,
            deferred_backlog={},
            local_selection_ordering_step=0,
            r7_selective_drain_eligibility_census_enabled=True,
            r7_selective_drain_eligibility_census_tracker=tr,
            r7_selective_drain_eligibility_census_sidecar_path=tmp_path / "ec.jsonl",
        )
    assert m.call_count == 1


def test_d_legacy_injected_observer_failure_before_publication(tmp_path):
    state, votes, spec, cap = _dense_fixture()
    q_before = state.q_levels.clone()
    acc_before = state.exact_accumulator_shadow.clone()
    pre = {"toy.proj": {1: {"first_step": 0, "last_deferred_step": 0, "defer_count": 1}}}
    tr = ObserverContinuityTracker()
    tr.reset()
    sidecar = tmp_path / "fail.jsonl"
    published: dict[str, Any] = {"result": None}

    def boom(*args, **kwargs):
        raise RuntimeError("injected_observer_fail")

    with mock.patch(_APPEND, side_effect=boom):
        with pytest.raises(RuntimeError, match="injected_observer_fail"):
            published["result"] = apply_bounded_delta_vote_step(
                {"toy.proj": state},
                {"toy.proj": votes},
                {"toy.proj": spec},
                global_cap_spec=cap,
                deferred_backlog=pre,
                local_selection_ordering_step=0,
                r7_selective_drain_eligibility_census_enabled=True,
                r7_selective_drain_eligibility_census_tracker=tr,
                r7_selective_drain_eligibility_census_sidecar_path=sidecar,
            )
    assert published["result"] is None
    assert state.q_levels.tolist() == q_before.tolist()
    assert state.exact_accumulator_shadow.tolist() == acc_before.tolist()
    assert tr.cardinality() == 0
    assert not sidecar.exists() or sidecar.read_text() == ""


def test_e_legacy_pre_step_causality_table2_over_input_backlog(tmp_path):
    state, votes, spec, cap = _dense_fixture()
    # Index 1 is already in the pre-step INPUT backlog; cap will also defer one fresh index.
    pre = {"toy.proj": {1: {"first_step": 0, "last_deferred_step": 0, "defer_count": 1}}}
    tr = ObserverContinuityTracker()
    tr.reset()
    sidecar = tmp_path / "causality.jsonl"
    with mock.patch(_MAYBE_RUN, wraps=maybe_run_selective_drain_census) as m:
        result = apply_bounded_delta_vote_step(
            {"toy.proj": state},
            {"toy.proj": votes},
            {"toy.proj": spec},
            global_cap_spec=cap,
            deferred_backlog=pre,
            local_selection_ordering_step=0,
            r7_selective_drain_eligibility_census_enabled=True,
            r7_selective_drain_eligibility_census_tracker=tr,
            r7_selective_drain_eligibility_census_sidecar_path=sidecar,
        )
    assert m.call_count == 1
    kw = m.call_args.kwargs
    assert kw["pre_step_backlog"] is pre
    assert kw["pre_step_backlog_before_cap"] is pre
    # Fresh post-cap backlog may grow; pre-step INPUT object must stay the census source.
    assert result.deferred_backlog is not pre
    assert 1 in result.deferred_backlog.get("toy.proj", {})
    line = sidecar.read_text().strip().splitlines()[0]
    import json

    chunk = json.loads(line)
    t2 = chunk["table2"]
    assert int(t2["pre_step_backlog_unique_count"]) == 1


def test_learner_source_contains_both_cpu_hooks_gpu_clean():
    text = _LEARNER_PATH.read_text()
    assert text.count("maybe_run_selective_drain_census(") == 2
    gpu_idx = text.find("apply_sparse_event_coded_cap_via_gpu_seam")
    assert gpu_idx != -1
    gpu_block = text[gpu_idx : gpu_idx + 2500]
    assert "maybe_run_selective_drain_census" not in gpu_block
