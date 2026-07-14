"""Characterization: CLI authority/exit matrix preserved after launch extraction."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def _load_run_mod():
    sys.path.insert(0, str(REPO))
    return importlib.import_module("scripts.forgotten_accum_training_equivalence_run")


def _load_launch_mod():
    sys.path.insert(0, str(REPO))
    return importlib.import_module(
        "calm.hrm_text_158.native_full_stack.forgotten_accum_run_arms_launch"
    )


def test_run_py_reexports_launch_seam_symbols():
    run = _load_run_mod()
    launch = _load_launch_mod()
    assert run.launch_run_arms is launch.launch_run_arms
    assert run.resolve_run_arms_authority is launch.resolve_run_arms_authority
    assert run.EXIT_RUN_ARMS_NO_AUTHORITY == 20
    assert run.EXIT_RUN_ARMS_RUNNER_CONTRACT == 25
    assert run.EXIT_RUN_ARMS_IDENTITY == 24


def test_authority_matrix_characterization_unchanged(monkeypatch):
    mod = _load_run_mod()
    called = []
    monkeypatch.setattr(
        mod, "launch_run_arms", lambda args: called.append(1) or ({"status": "OK"}, 0)
    )
    assert (
        mod.main(
            [
                "run-arms",
                "--parent",
                "/tmp/p.pt",
                "--parent-sha256",
                "x",
                "--scratch-root",
                "/tmp/s",
            ]
        )
        == mod.EXIT_RUN_ARMS_NO_AUTHORITY
    )
    assert called == []
    assert (
        mod.main(
            [
                "run-arms",
                "--allow-gpu-launch",
                "--formal-science",
                "--i-have-claude-run-arms-smoke-authority",
                "--parent",
                "/tmp/p.pt",
                "--parent-sha256",
                "x",
                "--scratch-root",
                "/tmp/s",
            ]
        )
        == mod.EXIT_RUN_ARMS_NO_AUTHORITY
    )
    assert called == []
