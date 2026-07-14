"""CLI characterization for forgotten-accum run-arms (authority matrix + refuse)."""
from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RUN_PY = REPO / "scripts/forgotten_accum_training_equivalence_run.py"


def _load_run_mod():
    sys.path.insert(0, str(REPO))
    return importlib.import_module("scripts.forgotten_accum_training_equivalence_run")


def _load_launch_mod():
    sys.path.insert(0, str(REPO))
    return importlib.import_module(
        "calm.hrm_text_158.native_full_stack.forgotten_accum_run_arms_launch"
    )


def test_run_arms_default_refuse_no_driver_invoke(monkeypatch):
    mod = _load_run_mod()
    called = []

    def boom(*_a, **_k):
        called.append(1)
        raise AssertionError("launch must not run")

    monkeypatch.setattr(_load_launch_mod(), "launch_run_arms", boom)
    code = mod.main(
        [
            "run-arms",
            "--parent",
            "/tmp/parent.pt",
            "--parent-sha256",
            "abc",
            "--scratch-root",
            "/tmp/scratch",
        ]
    )
    assert code == mod.EXIT_RUN_ARMS_NO_AUTHORITY
    assert called == []


def test_run_arms_requires_authority_pairs(monkeypatch):
    mod = _load_run_mod()
    called = []
    monkeypatch.setattr(
        _load_launch_mod(), "launch_run_arms", lambda args: called.append(args) or ({"status": "OK"}, 0)
    )
    only_gpu = mod.main(
        [
            "run-arms",
            "--allow-gpu-launch",
            "--parent",
            "/tmp/p.pt",
            "--parent-sha256",
            "x",
            "--scratch-root",
            "/tmp/s",
        ]
    )
    only_formal = mod.main(
        [
            "run-arms",
            "--formal-science",
            "--parent",
            "/tmp/p.pt",
            "--parent-sha256",
            "x",
            "--scratch-root",
            "/tmp/s",
        ]
    )
    only_smoke = mod.main(
        [
            "run-arms",
            "--i-have-claude-run-arms-smoke-authority",
            "--parent",
            "/tmp/p.pt",
            "--parent-sha256",
            "x",
            "--scratch-root",
            "/tmp/s",
        ]
    )
    assert only_gpu == mod.EXIT_RUN_ARMS_NO_AUTHORITY
    assert only_formal == mod.EXIT_RUN_ARMS_NO_AUTHORITY
    assert only_smoke == mod.EXIT_RUN_ARMS_NO_AUTHORITY
    assert called == []


def test_run_arms_smoke_formal_mutual_exclusion(monkeypatch):
    mod = _load_run_mod()
    called = []
    monkeypatch.setattr(
        _load_launch_mod(), "launch_run_arms", lambda args: called.append(1) or ({"status": "OK"}, 0)
    )
    code = mod.main(
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
    assert code == mod.EXIT_RUN_ARMS_NO_AUTHORITY
    assert called == []


def test_run_arms_smoke_authority_reaches_launch(monkeypatch, tmp_path: Path):
    mod = _load_run_mod()
    seen = {}

    def fake_launch(args):
        seen["mode"] = mod.resolve_run_arms_authority(args)
        seen["kwargs"] = mod.run_arms_kwargs_from_args(args)
        return {
            "status": "OK",
            "run_kind": "REAL_DEVICE_SMOKE",
            "science_label": None,
            "claimable_science": False,
            "bankable": False,
        }, 0

    monkeypatch.setattr(_load_launch_mod(), "launch_run_arms", fake_launch)
    code = mod.main(
        [
            "run-arms",
            "--allow-gpu-launch",
            "--i-have-claude-run-arms-smoke-authority",
            "--parent",
            str(tmp_path / "parent.pt"),
            "--parent-sha256",
            "9b4e311a22787e7d4808bde7bc2953568d767a2ee8ac648942a3f5dbb7b4d5ec",
            "--scratch-root",
            str(tmp_path / "scratch"),
            "--t-cut",
            "2",
            "--runway-steps",
            "4",
            "--W",
            "1",
        ]
    )
    assert code == 0
    assert seen["mode"] == "smoke"
    assert seen["kwargs"]["developer_validation"] is True
    assert seen["kwargs"]["formal_science"] is False


def test_run_arms_formal_authority_reaches_launch(monkeypatch, tmp_path: Path):
    mod = _load_run_mod()
    seen = {}

    def fake_launch(args):
        seen["kwargs"] = mod.run_arms_kwargs_from_args(args)
        return {"status": "OK", "science_label": None}, 0

    monkeypatch.setattr(_load_launch_mod(), "launch_run_arms", fake_launch)
    code = mod.main(
        [
            "run-arms",
            "--allow-gpu-launch",
            "--formal-science",
            "--parent",
            str(tmp_path / "parent.pt"),
            "--parent-sha256",
            "9b4e311a22787e7d4808bde7bc2953568d767a2ee8ac648942a3f5dbb7b4d5ec",
            "--scratch-root",
            str(tmp_path / "scratch"),
            "--live-acc-carrier-selector",
            "NONE",
            "--eligible-scope",
            "all-bitlinear",
        ]
    )
    assert code == 0
    kw = seen["kwargs"]
    assert kw["allow_gpu_launch"] is True and kw["formal_science"] is True
    assert kw["developer_validation"] is False
    assert kw["authority_mode"] == "formal"


def test_run_arms_fresh_process_refuse():
    proc = subprocess.run(
        [
            sys.executable,
            str(RUN_PY),
            "run-arms",
            "--parent",
            "/tmp/parent.pt",
            "--parent-sha256",
            "deadbeef",
            "--scratch-root",
            "/tmp/scratch",
        ],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 20
    assert "REFUSED" in proc.stderr


def test_smoke_dense_site_unchanged_subparser():
    mod = _load_run_mod()
    parser = mod.build_parser()
    args = parser.parse_args(
        ["smoke-dense-site", "--i-have-claude-gpu-smoke-authority", "--device", "cuda:0"]
    )
    assert args.cmd == "smoke-dense-site"
    assert args.i_have_claude_gpu_smoke_authority is True
