"""LANDS-AB CPU phase suite (IMPLEMENT_v12 split)."""
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








def test_p2_one_entry_pins_scope_creep():
    from calm.hrm_text_158.native_full_stack.lands_ab_eval_fixture_source import (
        DEFAULT_SOURCE_PINS,
        verify_source_pins,
    )
    one = {"bin/watch-wrap": DEFAULT_SOURCE_PINS["bin/watch-wrap"]}
    assert verify_source_pins(one)["scope_creep"] is True
    assert verify_source_pins(DEFAULT_SOURCE_PINS)["scope_creep"] is False


def test_p3_nested_start_rejected_by_topology():
    from calm.hrm_text_158.native_full_stack.lands_ab_eval_phase_topology import (
        classify_phase_topology,
        synthesize_nested_start_events,
    )
    topo = classify_phase_topology(
        synthesize_nested_start_events(), require_enforcer_fields=True
    )
    assert topo["good_topology"] is False
    assert topo["detail"] == "nested_start"


def test_source_pins_substituted_expected_hash():
    from calm.hrm_text_158.native_full_stack.lands_ab_eval_fixture_source import (
        DEFAULT_SOURCE_PINS,
        verify_source_pins,
    )
    bad = dict(DEFAULT_SOURCE_PINS)
    bad["bin/watch-wrap"] = "0" * 64
    assert verify_source_pins(bad)["scope_creep"] is True


def test_required_key_universe_empty_rejected():
    from calm.hrm_text_158.native_full_stack.lands_ab_eval_metric_reducer import (
        validate_required_key_universe,
    )
    with pytest.raises(ValueError, match="required_key_set_empty"):
        validate_required_key_universe(required_key_set=[], row_key_universes={"r": []})
    with pytest.raises(ValueError, match="key_universe_mismatch"):
        validate_required_key_universe(
            required_key_set=["a"], row_key_universes={"r": ["b"]}
        )


def test_phase_topology_hostiles():
    from calm.hrm_text_158.native_full_stack.lands_ab_eval_phase_topology import (
        classify_phase_topology,
        synthesize_duplicate_start_events,
        synthesize_good_topology_events,
        synthesize_missing_coverage_events,
        synthesize_nested_start_events,
    )
    assert classify_phase_topology(synthesize_good_topology_events(), expected_node_id="node", require_enforcer_fields=True)["good_topology"] is True
    assert classify_phase_topology(synthesize_duplicate_start_events(), require_enforcer_fields=True)["detail"] == "duplicate_start"
    assert classify_phase_topology(synthesize_missing_coverage_events(), require_enforcer_fields=True)["detail"] == "missing_coverage"
    assert classify_phase_topology(synthesize_nested_start_events(), require_enforcer_fields=True)["detail"] == "nested_start"


def test_phase_topology_module_has_no_io_gpu_imports():
    import ast, inspect
    from calm.hrm_text_158.native_full_stack import lands_ab_eval_phase_topology as pt
    from calm.hrm_text_158.native_full_stack import lands_ab_eval_metric_reducer as mr
    for mod in (pt, mr):
        src = Path(inspect.getfile(mod)).read_text()
        tree = ast.parse(src)
        banned = {"torch", "subprocess", "socket", "os", "pathlib"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] not in banned
            if isinstance(node, ast.ImportFrom) and node.module:
                root = node.module.split(".")[0]
                assert root in {"calm", "typing", "__future__", "math"} or root not in banned


def test_seam_line_counts_under_stop_condition():
    """Lib seams <500; owned test seams (non-shim) also <500 (IMPLEMENT_v12)."""
    root = Path(__file__).resolve().parents[3]
    lib = [
        "calm/hrm_text_158/native_full_stack/lands_ab_eval_twin_apply.py",
        "calm/hrm_text_158/native_full_stack/lands_ab_eval_fixture_source.py",
        "calm/hrm_text_158/native_full_stack/lands_ab_eval_site_measurement.py",
        "calm/hrm_text_158/native_full_stack/lands_ab_eval_cuda_sites.py",
        "calm/hrm_text_158/native_full_stack/lands_ab_eval_oracle_sites.py",
        "calm/hrm_text_158/native_full_stack/lands_ab_eval_phase_topology.py",
        "calm/hrm_text_158/native_full_stack/lands_ab_eval_evidence_contract.py",
        "calm/hrm_text_158/native_full_stack/lands_ab_eval_metric_reducer.py",
        "calm/hrm_text_158/native_full_stack/lands_ab_eval_phase_jsonl.py",
        "calm/hrm_text_158/native_full_stack/lands_ab_eval_measurement.py",
        "calm/hrm_text_158/native_full_stack/lands_ab_eval_production_binding.py",
        "calm/hrm_text_158/native_full_stack/lands_ab_eval_production_post_state.py",
        "calm/hrm_text_158/native_full_stack/lands_ab_eval_authoritative_payload.py",
    ]
    tests = [
        "calm/llm_computer/tests/test_hrm_text_158_lands_ab_eval_cpu_binding_local_core.py",
        "calm/llm_computer/tests/test_hrm_text_158_lands_ab_eval_cpu_binding_local_pipeline.py",
        "calm/llm_computer/tests/test_hrm_text_158_lands_ab_eval_cpu_binding_roundtrip.py",
        "calm/llm_computer/tests/test_hrm_text_158_lands_ab_eval_cpu_binding_landing.py",
        "calm/llm_computer/tests/test_hrm_text_158_lands_ab_eval_cpu_phase_topology.py",
        "calm/llm_computer/tests/test_hrm_text_158_lands_ab_eval_cpu_phase_enforcer.py",
        "calm/llm_computer/tests/lands_ab_eval_test_helpers.py",
        "calm/llm_computer/tests/test_hrm_text_158_lands_ab_eval_cpu_phase_stream.py",
    ]
    for rel in lib + tests:
        n = len((root / rel).read_text().splitlines())
        assert n < 500, f"{rel} has {n} lines"


def test_enforcer_parity_self_test_shapes():
    """Parity with enforcer self-test modes: good / duplicate_start / missing_coverage / nested."""
    from calm.hrm_text_158.native_full_stack.lands_ab_eval_phase_topology import (
        classify_phase_topology,
        synthesize_duplicate_start_events,
        synthesize_good_topology_events,
        synthesize_missing_coverage_events,
        synthesize_nested_start_events,
    )
    assert classify_phase_topology(synthesize_good_topology_events(node_id="n"), expected_node_id="n", require_enforcer_fields=True)["terminal_class"] == "OK"
    assert classify_phase_topology(synthesize_duplicate_start_events(node_id="n"), expected_node_id="n", require_enforcer_fields=True)["terminal_class"] == "PHASE_TELEMETRY"
    assert classify_phase_topology(synthesize_missing_coverage_events(node_id="n"), expected_node_id="n", require_enforcer_fields=True)["terminal_class"] == "PHASE_TELEMETRY"
    assert classify_phase_topology(synthesize_nested_start_events(node_id="n"), expected_node_id="n", require_enforcer_fields=True)["detail"] == "nested_start"
