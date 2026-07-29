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








def test_science_consumer_rejects_synthetic_row_laundering():
    from calm.hrm_text_158.native_full_stack.lands_ab_eval_evidence_contract import (
        build_eval_receipt_from_raw_artifacts,
        make_raw_row_observation,
        o_excl_write_json,
        runtime_scratch_raw_path,
    )
    from calm.hrm_text_158.native_full_stack.lands_ab_eval_fixture_source import (
        DEFAULT_SOURCE_PINS,
    )
    from calm.hrm_text_158.native_full_stack.lands_ab_eval_phase_topology import (
        synthesize_good_topology_events,
    )
    scratch = Path(os.environ.get("LANDS_AB_RUNTIME_SCRATCH", "/tmp/lands_ab_runtime_scratch")) / uuid.uuid4().hex
    cpu_obs, cpu_sha, cpu_path = _write_real_cpu_row(scratch)
    paths = {"G_CPU_STATIC_AB": {"path": str(cpu_path), "sha256": cpu_sha}}
    for row, surfs in {
        "G_CUDA_B1_APPLY": ("s3", "s4", "s6"),
        "G_CUDA_B2_APPLY": ("s3", "s4", "s6"),
        "G_CUDA_B3_APPLY": ("s3", "s4", "s6"),
        "G_CUDA_ORACLE_B1": ("s5",),
        "G_CUDA_ORACLE_B2": ("s5",),
        "G_CUDA_ORACLE_B3": ("s5",),
    }.items():
        if row.startswith("G_CUDA_ORACLE"):
            metrics = {
                "events_equal_by_key": {"lin": True},
                "events_equal_fused_vs_dense_derived": True,
                "independent_two_branch_recompute_ok": True,
                "dense_derived_provenance": "two_branch_parallel_dense_vote_derivation",
                "d1_densify_from_sparse_used": False,
                "sparse_vote_authority_mode": "oracle_on",
                "votes_by_key_applied": None,
                "builder_receipt_pass": True,
                "oracle_mode_on_named_site": True,
            }
        else:
            metrics = {
                "post_q_sha256_by_key": {"lin": {"sparse": "a"*64, "dense": "a"*64}},
                "post_logical_acc_sha256_by_key": {"lin": {"sparse": "b"*64, "dense": "b"*64}},
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
            }
        obs = make_raw_row_observation(
            gating_row=row,
            device="cuda",
            measured_surfaces={s: True for s in surfs},
            metrics=metrics,
            key_universe=["lin"],
            fixture_contract_raw_fail=False,
            synthetic_only=True,
            phase_topology={"good_topology": True, "detail": "good_topology"},
            phase_events=synthesize_good_topology_events(node_id=row),
        )
        p = runtime_scratch_raw_path(scratch_dir=scratch, gating_row=row, run_nonce=uuid.uuid4().hex[:8])
        sha = o_excl_write_json(p, obs)
        paths[row] = {"path": str(p), "sha256": sha}
    with pytest.raises(ValueError, match="synthetic_row_rejected"):
        build_eval_receipt_from_raw_artifacts(
            raw_artifact_paths=paths,
            source_pins=DEFAULT_SOURCE_PINS,
            required_key_set=sorted(set(list(cpu_obs["key_universe"]) + ["lin"])),
        )


def test_caller_authored_good_topology_without_events_rejected():
    from calm.hrm_text_158.native_full_stack.lands_ab_eval_evidence_contract import (
        load_and_validate_raw_artifact,
        make_raw_row_observation,
        o_excl_write_json,
        runtime_scratch_raw_path,
    )
    scratch = Path(os.environ.get("LANDS_AB_RUNTIME_SCRATCH", "/tmp/lands_ab_runtime_scratch")) / uuid.uuid4().hex
    metrics = {
        "post_q_sha256_by_key": {"proj": {"sparse": "a"*64, "dense": "a"*64}},
        "post_logical_acc_sha256_by_key": {"proj": {"sparse": "b"*64, "dense": "b"*64}},
        "events_equal_by_key": {"proj": True},
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
    }
    obs = make_raw_row_observation(
        gating_row="G_CUDA_B1_APPLY",
        device="cuda",
        measured_surfaces={"s3": True, "s4": True, "s6": True},
        metrics=metrics,
        key_universe=["proj"],
        fixture_contract_raw_fail=False,
        synthetic_only=False,
        phase_topology={"good_topology": True, "detail": "good_topology"},
    )
    p = runtime_scratch_raw_path(scratch_dir=scratch, gating_row="G_CUDA_B1_APPLY", run_nonce="topo")
    sha = o_excl_write_json(p, obs)
    with pytest.raises(ValueError, match="caller_authored_phase_topology_without_events|cuda_row_missing_phase_events"):
        load_and_validate_raw_artifact(path=p, expected_sha256=sha, expected_gating_row="G_CUDA_B1_APPLY")


def test_vacuous_and_divergent_reducer():
    assert reduce_lands_ab_branch_strict(
        _base_ok(surface_pass_by_row=matrix_with(**{"G_CPU_STATIC_AB/s4": False}))
    )["branch_id"] == BRANCH_VACUOUS
    assert reduce_lands_ab_branch_strict(
        _base_ok(surface_pass_by_row=matrix_with(**{"G_CPU_STATIC_AB/s1": False}))
    )["branch_id"] == BRANCH_DIVERGENT_EVENT
    assert reduce_lands_ab_branch_strict(
        _base_ok(surface_pass_by_row=matrix_with(**{"G_CUDA_B1_APPLY/s3": False}))
    )["branch_id"] == BRANCH_DIVERGENT_APPLY


def test_primitives_builder_is_diagnostic_only():
    from calm.hrm_text_158.native_full_stack.lands_ab_eval_evidence_contract import (
        build_eval_receipt_from_primitives,
    )
    from calm.hrm_text_158.native_full_stack.lands_ab_eval_schema import DIAGNOSTIC_RECEIPT_SCHEMA
    out = build_eval_receipt_from_primitives(_base_ok())
    assert out["schema"] == DIAGNOSTIC_RECEIPT_SCHEMA
    assert out["synthetic_only"] is True


def test_runner_cli_help_and_reducer_smoke_synthetic_only():
    import subprocess, sys
    cwd = str(Path(__file__).resolve().parents[3])
    r = subprocess.run([sys.executable, "scripts/lands_ab_eval_run.py", "--mode", "reducer-smoke"], capture_output=True, text=True, cwd=cwd)
    assert r.returncode == 0
    payload = __import__("json").loads(r.stdout)
    assert payload["synthetic_only"] is True
    assert payload["science_claim"] is False


def test_enforcer_jsonl_env_relay_writes_classifiable_cycle():
    import os
    import tempfile
    from calm.hrm_text_158.native_full_stack.lands_ab_eval_phase_jsonl import (
        ENV_JSONL,
        emit_one_enforcer_cycle_to_memory_and_jsonl,
        load_jsonl_events,
    )
    from calm.hrm_text_158.native_full_stack.lands_ab_eval_phase_topology import (
        classify_phase_topology,
    )
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "ev.jsonl"
        os.environ[ENV_JSONL] = str(path)
        try:
            node = "G_CUDA_ORACLE_B1"
            mem = emit_one_enforcer_cycle_to_memory_and_jsonl(node)
            disk = load_jsonl_events(path)
            assert len(mem) == len(disk) == 8
            assert classify_phase_topology(
                disk, expected_node_id=node, require_enforcer_fields=True
            )["good_topology"] is True
        finally:
            os.environ.pop(ENV_JSONL, None)


def test_work_enclosing_phase_duration_includes_sleep():
    import os, tempfile
    from calm.hrm_text_158.native_full_stack.lands_ab_eval_phase_jsonl import (
        ENV_JSONL,
        emit_work_enclosing_cycle_with_sleep,
        load_jsonl_events,
    )
    from calm.hrm_text_158.native_full_stack.lands_ab_eval_phase_topology import (
        classify_phase_topology,
    )
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "work.jsonl"
        os.environ[ENV_JSONL] = str(path)
        try:
            node = "G_CUDA_B1_APPLY"
            mem = emit_work_enclosing_cycle_with_sleep(node, work_s=0.03)
            ends = [e for e in mem if e["type"] == "PHASE_END"]
            assert all(float(e["duration_s"]) >= 0.02 for e in ends), ends
            disk = load_jsonl_events(path)
            ends2 = [e for e in disk if e["type"] == "PHASE_END"]
            assert all(float(e["duration_s"]) >= 0.02 for e in ends2)
            assert classify_phase_topology(
                disk, expected_node_id=node, require_enforcer_fields=True
            )["good_topology"] is True
        finally:
            os.environ.pop(ENV_JSONL, None)


def test_actual_enforcer_good_topology_self_test():
    """Run ACTUAL enforcer self-test good_topology → CLASS_OK rc=0."""
    import subprocess, sys, tempfile
    root = Path(__file__).resolve().parents[3]
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        jsonl = td / "phase.jsonl"
        receipt = td / "enforcer_receipt.json"
        # enforcer creates jsonl O_EXCL itself
        cmd = [
            sys.executable,
            str(root / "scripts/sparse_live_carrier_gpu_phase_budget_enforcer.py"),
            "--self-test",
            "good_topology",
            "--phase-events-jsonl",
            str(jsonl),
            "--enforcer-receipt",
            str(receipt),
            "--expected-node-id",
            "self-test",
            "--child",
            "true",
        ]
        # discover CLI
        help_r = subprocess.run(
            [sys.executable, str(root / "scripts/sparse_live_carrier_gpu_phase_budget_enforcer.py"), "--help"],
            capture_output=True, text=True, cwd=str(root),
        )
        # parse flags from help
        assert help_r.returncode == 0
        # build argv from known API run_enforcer via -c invoking main
        # Use module self-test path documented in script
        r = subprocess.run(
            [
                sys.executable,
                str(root / "scripts/sparse_live_carrier_gpu_phase_budget_enforcer.py"),
                "--self-test-good-topology",
                "--phase-events-jsonl",
                str(jsonl),
                "--enforcer-receipt",
                str(receipt),
                "--expected-node-id",
                "self-test",
            ],
            capture_output=True,
            text=True,
            cwd=str(root),
        )
        # if flag names differ, try alternate
        if r.returncode != 0 and "unrecognized" in (r.stderr + r.stdout).lower():
            r = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    f"""
from pathlib import Path
import sys
sys.path.insert(0, {str(root)!r})
from scripts.sparse_live_carrier_gpu_phase_budget_enforcer import run_enforcer, PHASE_ORDER
rc = run_enforcer(
    child_argv=[sys.executable, '-c', 'pass'],
    budgets={{p: 30.0 for p in PHASE_ORDER}},
    phase_events_jsonl=Path({str(jsonl)!r}),
    enforcer_receipt=Path({str(receipt)!r}),
    expected_node_id='self-test',
    self_test='good_topology',
)
raise SystemExit(rc)
""",
                ],
                capture_output=True,
                text=True,
                cwd=str(root),
            )
        assert r.returncode == 0, (r.returncode, r.stdout[-500:], r.stderr[-500:])


def test_actual_enforcer_overrun_returns_124():
    import subprocess, sys, tempfile
    root = Path(__file__).resolve().parents[3]
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        jsonl = td / "phase_over.jsonl"
        receipt = td / "enforcer_over.json"
        r = subprocess.run(
            [
                sys.executable,
                "-c",
                f"""
from pathlib import Path
import sys
sys.path.insert(0, {str(root)!r})
# import as file module
import importlib.util
spec = importlib.util.spec_from_file_location(
    'enf', {str(root / 'scripts/sparse_live_carrier_gpu_phase_budget_enforcer.py')!r}
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
rc = mod.run_enforcer(
    child_argv=[sys.executable, '-c', 'pass'],
    budgets={{p: 30.0 for p in mod.PHASE_ORDER}},
    phase_events_jsonl=Path({str(jsonl)!r}),
    enforcer_receipt=Path({str(receipt)!r}),
    expected_node_id='self-test',
    self_test='overrun',
)
raise SystemExit(rc)
""",
            ],
            capture_output=True,
            text=True,
            cwd=str(root),
        )
        assert r.returncode == 124, (r.returncode, r.stdout[-800:], r.stderr[-800:])
