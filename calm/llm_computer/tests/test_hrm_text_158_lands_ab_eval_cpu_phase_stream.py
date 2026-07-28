"""LANDS-AB phase-stream hostiles (IMPLEMENT_v13)."""
from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest
import torch

from calm.hrm_text_158.bit_linear import BitLinear
from calm.hrm_text_158.native_full_stack.lands_ab_eval_branch_reducer import (
    LandsAbReducerSchemaError,
    all_true_matrix,
    matrix_with,
    reduce_lands_ab_branch_strict,
)
from calm.llm_computer.tests.lands_ab_eval_test_helpers import (
    base_ok as _base_ok,
    write_real_cpu_row as _write_real_cpu_row,
)
from calm.hrm_text_158.native_full_stack.lands_ab_eval_schema import (
    BRANCH_DIVERGENT_APPLY,
    BRANCH_DIVERGENT_EVENT,
    BRANCH_EQUIVALENT,
    BRANCH_FIXTURE_CONTRACT_FAIL,
    BRANCH_VACUOUS,
    CANONICAL_CELL_KEYS,
)








def test_phase_attribution_timing_hostile_per_phase():
    """IMPLEMENT_v10: delayed work in each phase attributes duration to THAT phase."""
    import time
    from calm.hrm_text_158.native_full_stack.lands_ab_eval_phase_jsonl import (
        phase_start,
        phase_end,
    )
    from calm.hrm_text_158.native_full_stack.lands_ab_eval_phase_topology import (
        classify_phase_topology,
    )
    from calm.hrm_text_158.native_full_stack.lands_ab_eval_schema import PHASE_ORDER

    delays = {
        "forward_backward": 0.03,
        "update": 0.05,
        "emission": 0.02,
        "flush": 0.01,
    }
    events: list = []
    open_starts: dict = {}
    node_id = "G_CUDA_B1_APPLY"
    for phase in PHASE_ORDER:
        phase_start(events, phase=phase, node_id=node_id, open_starts=open_starts)
        time.sleep(delays[phase])
        phase_end(events, phase=phase, node_id=node_id, open_starts=open_starts)
    topo = classify_phase_topology(
        events, expected_node_id=node_id, require_enforcer_fields=True
    )
    assert topo["good_topology"] is True
    ends = {
        e["phase"]: float(e["duration_s"])
        for e in events
        if e["type"] == "PHASE_END"
    }
    # each phase's duration must reflect ITS delay, not collapse into update
    for phase, d in delays.items():
        assert ends[phase] >= d * 0.7, (phase, ends[phase], d)
    # update is not absorbing others: emission delay not counted as update
    assert ends["update"] < delays["update"] + delays["emission"]
    assert ends["emission"] >= delays["emission"] * 0.7


def test_native_builder_phase_attribution_via_emitter():
    """Delay injected at native _emit_phase boundary attributes to EMITTED phase."""
    import time
    import calm.hrm_text_158.native_full_stack.trainer_sub2_authority as tsa
    from calm.hrm_text_158.native_full_stack.lands_ab_eval_phase_jsonl import (
        install_capturing_phase_emitter,
        fold_native_builder_phases_plus_flush,
    )
    from calm.hrm_text_158.native_full_stack.lands_ab_eval_phase_topology import (
        classify_phase_topology,
    )

    events: list = []
    open_starts: dict = {}
    node_id = "G_CUDA_B1_APPLY"
    delays = {"forward_backward": 0.04, "update": 0.06, "emission": 0.03}

    real_emit = tsa._emit_phase

    def delayed_builder():
        # Simulate named builder: emit START, work, END per phase
        for phase, d in delays.items():
            tsa._emit_phase("PHASE_START", phase)
            time.sleep(d)
            tsa._emit_phase("PHASE_END", phase)

    with install_capturing_phase_emitter(node_id, events, open_starts):
        delayed_builder()
    fold = fold_native_builder_phases_plus_flush(
        events, node_id=node_id, open_starts={}
    )
    assert fold["needs_measurement_flush"] is True
    from calm.hrm_text_158.native_full_stack.lands_ab_eval_phase_jsonl import (
        emit_measurement_owned_flush,
    )
    emit_measurement_owned_flush(
        fold["events"], node_id=node_id, open_starts={}, work_fn=None
    )
    events = fold["events"]
    topo = classify_phase_topology(
        events, expected_node_id=node_id, require_enforcer_fields=True
    )
    assert topo["good_topology"] is True
    ends = {
        e["phase"]: float(e["duration_s"])
        for e in events
        if e["type"] == "PHASE_END"
    }
    for phase, d in delays.items():
        assert ends[phase] >= d * 0.7, (phase, ends[phase], d)
    # update does not absorb emission delay
    assert ends["emission"] >= delays["emission"] * 0.7
    assert ends["update"] < delays["update"] + delays["emission"] * 0.5


def test_partial_native_stream_is_anomaly_not_completed():
    """Hostile (i): one real builder phase then silence → anomaly, not silent completion."""
    from calm.hrm_text_158.native_full_stack.lands_ab_eval_phase_jsonl import (
        fold_native_builder_phases_plus_flush,
        install_capturing_phase_emitter,
    )
    import calm.hrm_text_158.native_full_stack.trainer_sub2_authority as tsa

    events: list = []
    open_starts: dict = {}
    node_id = "G_CUDA_B1_APPLY"
    with install_capturing_phase_emitter(node_id, events, open_starts):
        tsa._emit_phase("PHASE_START", "forward_backward")
        tsa._emit_phase("PHASE_END", "forward_backward")
        # silence: no update/emission
    fold = fold_native_builder_phases_plus_flush(
        events, node_id=node_id, open_starts={}
    )
    assert fold["phase_stream_anomaly"] is True
    assert fold["phase_stream_class"] == "partial_native_anomaly"
    # must NOT invent update/emission pairs
    starts = [e["phase"] for e in fold["events"] if e["type"] == "PHASE_START"]
    assert "update" not in starts
    assert "emission" not in starts
    # no unmarked synthesis
    assert not any(
        e.get("synthesized") and e.get("phase") in ("update", "emission")
        for e in fold["events"]
    )


def test_synthesized_marked_events_reject_science_polarity():
    """Hostile (ii): synthesized-marked events → science fixture-fail surfaces."""
    from calm.hrm_text_158.native_full_stack.lands_ab_eval_phase_jsonl import (
        apply_phase_stream_science_gate,
        fold_native_builder_phases_plus_flush,
    )
    from calm.hrm_text_158.native_full_stack.lands_ab_eval_phase_topology import (
        classify_phase_topology,
    )

    events: list = []
    fold = fold_native_builder_phases_plus_flush(
        events, node_id="G_CUDA_B1_APPLY", open_starts={}
    )
    assert fold["phase_stream_class"] == "empty_synthesized_transport"
    assert fold["phase_events_synthesized"] is True
    assert all(
        e.get("synthesized") is True
        for e in fold["events"]
        if e["type"] in ("PHASE_START", "PHASE_END")
        and e["phase"] != "flush" or e.get("synthesized") is True
    )
    # synthesized events are marked
    assert any(e.get("synthesized") is True for e in fold["events"])
    topo = classify_phase_topology(
        fold["events"], expected_node_id="G_CUDA_B1_APPLY", require_enforcer_fields=True
    )
    obs = {
        "gating_row": "G_CUDA_B1_APPLY",
        "phase_events": fold["events"],
        "phase_topology": topo,
        "phase_events_synthesized": True,
        "phase_stream_anomaly": False,
        "key_universe": ["lin"],
        "metrics": {
            "post_q_sha256_by_key": {
                "lin": {"sparse": "a" * 64, "dense": "a" * 64}
            },
            "post_logical_acc_sha256_by_key": {
                "lin": {"sparse": "b" * 64, "dense": "b" * 64}
            },
            "events_equal_by_key": {"lin": True},
            "sparse_event_count": 1,
            "q_changed_count_sparse": 1,
            "q_changed_count_dense": 1,
            "s6_geometry": {
                "votes_by_key_applied": None,
                "sparse_vote_authority_only": True,
                "transient_over2_tensors": ["weighted_grad"],
                "oracle_only_absent_on_fused": True,
            },
            "d1_densify_from_sparse_used": False,
            "builder_receipt_pass": True,
            "production_sparse_matches_twin": True,
        },
        "measured_surfaces": {"s3": True, "s4": True, "s6": True},
        "fixture_contract_raw_fail": False,
    }
    gated = apply_phase_stream_science_gate(obs, gating_row="G_CUDA_B1_APPLY")
    assert gated["fixture_contract_raw_fail"] is True
    assert gated["phase_events_synthesized"] is True
    assert gated["measured_surfaces"].get("s3") is False




def test_native_complete_flush_jsonl_eight_events_encloses_work():
    """IMPLEMENT_v14: native-complete env-armed — 8 JSONL events; flush encloses delay."""
    import os
    import tempfile
    import time
    from pathlib import Path

    import calm.hrm_text_158.native_full_stack.trainer_sub2_authority as tsa
    from calm.hrm_text_158.native_full_stack.lands_ab_eval_phase_jsonl import (
        ENV_JSONL,
        emit_measurement_owned_flush,
        fold_native_builder_phases_plus_flush,
        install_capturing_phase_emitter,
        load_jsonl_events,
    )
    from calm.hrm_text_158.native_full_stack.lands_ab_eval_phase_topology import (
        classify_phase_topology,
    )

    with tempfile.TemporaryDirectory() as td:
        jp = Path(td) / "phase_events.jsonl"
        os.environ[ENV_JSONL] = str(jp)
        try:
            events: list = []
            open_starts: dict = {}
            node_id = "G_CUDA_B1_APPLY"
            with install_capturing_phase_emitter(node_id, events, open_starts):
                for phase in ("forward_backward", "update", "emission"):
                    tsa._emit_phase("PHASE_START", phase)
                    tsa._emit_phase("PHASE_END", phase)
            fold = fold_native_builder_phases_plus_flush(
                events, node_id=node_id, open_starts={}
            )
            assert fold["phase_stream_class"] == "native_complete"
            assert fold["needs_measurement_flush"] is True
            delay = 0.05
            emit_measurement_owned_flush(
                fold["events"],
                node_id=node_id,
                open_starts={},
                work_fn=lambda: time.sleep(delay),
            )
            topo = classify_phase_topology(
                fold["events"], expected_node_id=node_id, require_enforcer_fields=True
            )
            assert topo["good_topology"] is True
            lines = load_jsonl_events(jp)
            assert len(lines) == 8, len(lines)
            starts = [e["phase"] for e in lines if e["type"] == "PHASE_START"]
            assert starts == [
                "forward_backward",
                "update",
                "emission",
                "flush",
            ]
            flush_end = next(
                e for e in lines if e["type"] == "PHASE_END" and e["phase"] == "flush"
            )
            assert float(flush_end["duration_s"]) >= delay * 0.7
            assert flush_end.get("measurement_owned") is True
            assert flush_end.get("synthesized") is False
            # memory also has 8
            assert len([e for e in fold["events"] if e["type"] == "PHASE_START"]) == 4
        finally:
            os.environ.pop(ENV_JSONL, None)
