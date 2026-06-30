"""Tests for cap_selection marker guarantee (Slice B-DIAG root-cause fix)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    apply_bounded_delta_vote_step,
    credit_from_weighted_grad,
    default_dry_run_rank_vote_spec,
    make_event_coded_live_tensor_state,
    project_s1_gradient_to_moves,
    sparse_rank_bucketed_int16_vote_events,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap import GlobalRateCapSpec
from calm.hrm_text_158.native_full_stack.vote_update import VoteUpdateSpec
from scripts.hrm_text_158_bounded_delta_acquisition_probe import PhaseMilestoneEmitter

pytestmark = pytest.mark.filterwarnings("ignore")


class _RecordingEmitter:
    def __init__(self, root: Path) -> None:
        self.inner = PhaseMilestoneEmitter(root, enabled=True, device=torch.device("cpu"))
        self.records: list[tuple[str, str]] = []

    def record_sparse_cap_sub_phase(
        self,
        sub_phase_id: str,
        *,
        optimizer_step_index: int | None,
        milestone_kind: str,
        elapsed_since_phase_enter_seconds: float = 0.0,
    ) -> None:
        self.records.append((str(sub_phase_id), str(milestone_kind)))
        self.inner.record_sparse_cap_sub_phase(
            sub_phase_id,
            optimizer_step_index=optimizer_step_index,
            milestone_kind=milestone_kind,
            elapsed_since_phase_enter_seconds=elapsed_since_phase_enter_seconds,
        )


def _sparse_cap_fixture_cpu_resident_q():
    rank_spec = default_dry_run_rank_vote_spec()
    q_a = torch.tensor([[0, 1, -1, 0]], dtype=torch.int8)
    q_b = torch.tensor([[1, 0, 0, -1]], dtype=torch.int8)
    weighted_grad = torch.tensor([[-1.0, -2.0, 3.0, 4.0]])
    moves = project_s1_gradient_to_moves(weighted_grad, q_a)
    credit = credit_from_weighted_grad(weighted_grad)
    sparse_a = sparse_rank_bucketed_int16_vote_events(credit, moves, rank_spec)
    sparse_b = sparse_rank_bucketed_int16_vote_events(credit, moves, rank_spec)
    spec = VoteUpdateSpec(
        threshold_abs=8,
        accumulator_clip_min=-127,
        accumulator_clip_max=127,
        max_abs_per_tensor=16,
    )
    cap = GlobalRateCapSpec(cap=4, step=1, mutate_outputs=True)
    states = {
        "mod.a": make_event_coded_live_tensor_state("mod.a", q_a, 0.25, demotion_band=1),
        "mod.b": make_event_coded_live_tensor_state("mod.b", q_b, 0.25, demotion_band=1),
    }
    assert all(state.q_levels.device.type == "cpu" for state in states.values())
    return states, {"mod.a": sparse_a, "mod.b": sparse_b}, {"mod.a": spec, "mod.b": spec}, cap


def test_cap_selection_emits_for_cpu_resident_q_reference_path(tmp_path: Path) -> None:
    states, sparse_by_key, vote_specs, cap = _sparse_cap_fixture_cpu_resident_q()
    emitter = _RecordingEmitter(tmp_path)
    result = apply_bounded_delta_vote_step(
        states,
        None,
        vote_specs,
        candidate_sparse_vote_events_by_key=sparse_by_key,
        global_cap_spec=cap,
        event_coded_sparse_vote_authority=True,
        sparse_cap_submilestone_emit=emitter,
        local_selection_ordering_step=1,
    )
    cap_path = tmp_path / "liveness_milestones" / "sparse_cap_apply_cap_selection_cpu_copy.jsonl"
    assert cap_path.is_file(), "cap_selection jsonl must be written for CPU-resident q_levels"
    rows = [json.loads(line) for line in cap_path.read_text().splitlines() if line.strip()]
    assert rows[0]["milestone_kind"] == "cap_reference_cpu_resident_done"
    assert result.global_summary.get("sparse_cap_submilestone_cap_selection_path") in {
        "cpu_resident_reference",
        "cpu_reference",
    }
    assert ("cap_selection_cpu_copy", "cap_reference_cpu_resident_done") in emitter.records


def test_v6d_fixture_documents_absent_cap_selection() -> None:
    fixture = Path(__file__).resolve().parent / "fixtures" / "slice5_re_m4_v6d_2189e72024_corrected_null"
    if not fixture.is_dir():
        pytest.skip("v6d corrected-null fixture not present")
    for arm in ("baseline_snapshot_off", "instrumented_snapshot_on"):
        cap = fixture / arm / "liveness_milestones" / "sparse_cap_apply_cap_selection_cpu_copy.jsonl"
        assert not cap.is_file()
