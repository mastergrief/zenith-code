"""IMPLEMENT_v16: heterogeneous universe, O_EXCL/nonce harvest, frozen-packet dry-exec."""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import pytest

from calm.hrm_text_158.native_full_stack.lands_ab_eval_schema import (
    BRANCH_FIXTURE_CONTRACT_FAIL,
    GATING_ROWS,
)
from calm.llm_computer.tests.lands_ab_eval_test_helpers import (
    make_cuda_fixture_fail_obs,
    write_real_cpu_row,
)


def test_o_excl_write_text_fails_if_exists(tmp_path: Path):
    from calm.hrm_text_158.native_full_stack.lands_ab_eval_runtime_io import o_excl_write_text

    p = tmp_path / "out.json"
    sha1 = o_excl_write_text(p, "{\"ok\": true}")
    assert len(sha1) == 64
    with pytest.raises(FileExistsError):
        o_excl_write_text(p, "{\"ok\": false}")


def test_harvest_exactly_one_raw_obs_zero_and_multiple(tmp_path: Path):
    from calm.hrm_text_158.native_full_stack.lands_ab_eval_runtime_io import (
        harvest_exactly_one_raw_obs,
        o_excl_write_text,
        runtime_scratch_raw_path,
    )

    row = "G_CPU_STATIC_AB"
    with pytest.raises(ValueError, match="raw_obs_harvest_zero"):
        harvest_exactly_one_raw_obs(run_root=tmp_path, gating_row=row)
    p1 = runtime_scratch_raw_path(scratch_dir=tmp_path, gating_row=row, run_nonce="a1")
    o_excl_write_text(p1, "{\"a\": 1}")
    assert harvest_exactly_one_raw_obs(run_root=tmp_path, gating_row=row) == p1
    p2 = runtime_scratch_raw_path(scratch_dir=tmp_path, gating_row=row, run_nonce="b2")
    o_excl_write_text(p2, "{\"a\": 2}")
    with pytest.raises(ValueError, match="raw_obs_harvest_multiple"):
        harvest_exactly_one_raw_obs(run_root=tmp_path, gating_row=row)


def test_runner_cpu_static_ab_out_is_o_excl(tmp_path: Path):
    import subprocess
    import sys

    cwd = str(Path(__file__).resolve().parents[3])
    out = tmp_path / "cpu_row.json"
    env = dict(os.environ)
    env["PYTHONPATH"] = "."
    r = subprocess.run(
        [
            sys.executable,
            "scripts/lands_ab_eval_run.py",
            "--mode",
            "cpu-static-ab",
            "--out",
            str(out),
        ],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    assert out.is_file()
    r2 = subprocess.run(
        [
            sys.executable,
            "scripts/lands_ab_eval_run.py",
            "--mode",
            "cpu-static-ab",
            "--out",
            str(out),
        ],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
    )
    assert r2.returncode != 0
    assert "FileExistsError" in (r2.stderr + r2.stdout) or "exists" in (r2.stderr + r2.stdout).lower()


def test_runner_reducer_smoke_rejects_explicit_out(tmp_path: Path):
    """v17: reducer-smoke must hard-error on --out (no silent ignore)."""
    import subprocess
    import sys

    cwd = str(Path(__file__).resolve().parents[3])
    out = tmp_path / "should_not_exist.json"
    env = dict(os.environ)
    env["PYTHONPATH"] = "."
    r = subprocess.run(
        [
            sys.executable,
            "scripts/lands_ab_eval_run.py",
            "--mode",
            "reducer-smoke",
            "--out",
            str(out),
        ],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 2
    assert "reducer-smoke does not write --out" in (r.stderr + r.stdout)
    assert not out.exists()


def test_frozen_packet_dry_exec_reaches_fixture_contract_fail(tmp_path: Path):
    """ONE real CPU row + six CUDA fixture-fail schemas → structural null branch."""
    from calm.hrm_text_158.native_full_stack.lands_ab_eval_evidence_contract import (
        build_eval_receipt_from_raw_artifacts,
        o_excl_write_json,
        runtime_scratch_raw_path,
    )
    from calm.hrm_text_158.native_full_stack.lands_ab_eval_fixture_source import (
        DEFAULT_SOURCE_PINS,
    )

    run_root = tmp_path / "run"
    run_root.mkdir()
    cpu_obs, cpu_sha, cpu_path = write_real_cpu_row(run_root)
    paths = {"G_CPU_STATIC_AB": {"path": str(cpu_path), "sha256": cpu_sha}}
    for row in GATING_ROWS:
        if row == "G_CPU_STATIC_AB":
            continue
        obs = make_cuda_fixture_fail_obs(row, key="lin")
        p = runtime_scratch_raw_path(
            scratch_dir=run_root, gating_row=row, run_nonce=uuid.uuid4().hex[:8]
        )
        sha = o_excl_write_json(p, obs)
        paths[row] = {"path": str(p), "sha256": sha}

    required = sorted(set(list(cpu_obs["key_universe"]) + ["lin"]))
    receipt = build_eval_receipt_from_raw_artifacts(
        raw_artifact_paths=paths,
        source_pins=DEFAULT_SOURCE_PINS,
        required_key_set=required,
        caveats=["IMPLEMENT_v16 dry-exec structural-null proof"],
    )
    assert receipt["science_claim"] is False
    assert receipt["claim_ceiling"]["LANDS_AB"] is False
    assert receipt["reducer_output"]["branch_id"] == BRANCH_FIXTURE_CONTRACT_FAIL
    assert receipt["fixture_contract_raw_fail"] is True
    assert set(receipt["row_key_universes"]["G_CPU_STATIC_AB"]) == set(cpu_obs["key_universe"])
    for row in GATING_ROWS:
        if row.startswith("G_CUDA_"):
            assert receipt["row_key_universes"][row] == ["lin"]
    # O_EXCL eval receipt write
    from calm.hrm_text_158.native_full_stack.lands_ab_eval_runtime_io import o_excl_write_text

    eval_path = run_root / "eval_receipt.json"
    o_excl_write_text(eval_path, json.dumps(receipt, indent=2, sort_keys=True))
    with pytest.raises(FileExistsError):
        o_excl_write_text(eval_path, "{}")
