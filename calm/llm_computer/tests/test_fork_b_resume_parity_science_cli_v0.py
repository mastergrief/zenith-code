"""CPU-only fail-closed matrix for Fork B science CLI (plan v1.3 T1–T15)."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[3]
CLI_MOD = "scripts.fork_b_resume_parity_science_run"


@pytest.fixture()
def cli():
    if CLI_MOD in sys.modules:
        del sys.modules[CLI_MOD]
    return importlib.reload(importlib.import_module(CLI_MOD))


def _n() -> str:
    return "testnonce001"


def _scratch(cli, n: str | None = None) -> Path:
    return Path(cli.SCRATCH_TEMPLATE.replace("{LAUNCH_NONCE}", n or _n()))


def _formal_argv(cli, *, n: str | None = None, scratch: Path | None = None, **over: Any) -> list[str]:
    nonce = n or _n()
    concrete = cli.concrete_argv_from_nonce(nonce)
    sc = str(scratch) if scratch is not None else cli.SCRATCH_TEMPLATE.replace("{LAUNCH_NONCE}", nonce)
    argv = [
        "--allow-gpu-launch", "--formal-science", "--eligible-scope", over.get("eligible_scope", "all-bitlinear"),
        "--batch-size", str(over.get("batch_size", 1)),
        "--parent", cli.PARENT_PATH, "--parent-sha256", cli.PARENT_SHA256,
        "--scratch-root", sc, "--device", "cuda:0",
        "--cuts", over.get("cuts", "4,16,28"), "--k-steps", str(over.get("k_steps", 4)),
        "--steps", str(over.get("steps", 32)), "--global-horizon", str(over.get("global_horizon", 32)),
        "--batch-seed", str(over.get("batch_seed", 44)),
        "--support-order-seed", str(over.get("support_order_seed", 43)),
        "--ordering-seed", str(over.get("ordering_seed", 17)),
        "--launch-nonce", over.get("launch_nonce", nonce),
        "--argv-template-sha256", over.get("argv_template_sha256", cli.ARGV_TEMPLATE_SHA256),
        "--concrete-argv-sha256", over.get(
            "concrete_argv_sha256", cli.canonical_argv_sha256(concrete if over.get("launch_nonce", nonce) == nonce else cli.concrete_argv_from_nonce(str(over.get("launch_nonce", nonce))))
        ),
    ]
    omit_flags = set()
    if "eligible_scope" in over and over["eligible_scope"] is None:
        omit_flags.add("--eligible-scope")
    if "batch_size" in over and over["batch_size"] is None:
        omit_flags.add("--batch-size")
    if omit_flags:
        out = []
        i = 0
        while i < len(argv):
            if argv[i] in omit_flags:
                i += 2
                continue
            out.append(argv[i]); i += 1
        return out
    return argv


def _receipt(scratch: Path) -> dict[str, Any]:
    path = scratch / "fork_b_science_cli_receipt.json"
    assert path.is_file(), path
    return json.loads(path.read_text(encoding="utf-8"))


def _spy(cli, *, raise_exc: Exception | None = None, sha_seq: list[str] | None = None):
    calls: dict[str, Any] = {"cert": 0, "bindings": None}

    def _sha(path: Path) -> str:
        if sha_seq is None:
            return cli.PARENT_SHA256
        idx = min(calls.setdefault("sha_i", 0), len(sha_seq) - 1)
        calls["sha_i"] = idx + 1
        return sha_seq[idx]

    def _loader(args, *, parent_before: str):
        calls["cert"] += 1
        bindings = {
            "developer_validation": False, "require_strict_f_equals_u": True,
            "require_z_gate_break": True, "runner_identity": cli.RUNNER_IDENTITY,
            "global_horizon": 32,
            "runner_kwargs": {
                "global_horizon": 32, "eligible_scope": "all-bitlinear", "batch_size": 1,
                "max_abs_per_tensor": 4096, "r7_deferred_backlog_carry_enabled": True,
                "require_q_change": False,
            },
            "eligible_scope": "all-bitlinear", "batch_size": 1, "support_batch_size": 1,
            "cuts": [4, 16, 28], "k_steps": 4, "steps": 32,
            "batch_seed": 44, "support_order_seed": 43, "ordering_seed": 17,
        }
        calls["bindings"] = bindings
        if raise_exc is not None:
            raise raise_exc
        return {
            "terminal": {"label": "SPY_OK"}, "pre_science": None, "science_label": None,
            "notes": {"runner": "run_bounded_delta_steps"}, "_spy_bindings": bindings,
        }

    cli._sha256_file = _sha
    cli._load_parent_and_run = _loader
    return calls


def _pre_refuse(cli, tmp_path: Path, **over):
    scratch = tmp_path / "s"
    calls = _spy(cli)
    argv = _formal_argv(cli, scratch=scratch, **over)
    assert cli.main(argv) != 0
    rec = _receipt(scratch)
    assert calls["cert"] == 0 and rec["science_label"] is None
    return rec


def test_cli_refuses_without_allow_gpu_launch(cli, tmp_path: Path):
    assert cli.main([
        "--parent", cli.PARENT_PATH, "--parent-sha256", cli.PARENT_SHA256,
        "--scratch-root", str(tmp_path / "s"),
    ]) == 2
    assert not (tmp_path / "s" / "fork_b_science_cli_receipt.json").exists()


def test_cli_allow_without_formal_refuses(cli, tmp_path: Path):
    scratch = tmp_path / "s"
    calls = _spy(cli)
    rc = cli.main([
        "--allow-gpu-launch", "--parent", cli.PARENT_PATH, "--parent-sha256", cli.PARENT_SHA256,
        "--scratch-root", str(scratch), "--launch-nonce", _n(), "--eligible-scope", "all-bitlinear",
    ])
    assert rc != 0 and calls["cert"] == 0
    assert _receipt(scratch)["pre_science"] == "ALLOW_WITHOUT_FORMAL"


def test_cli_empty_nonce_refuses(cli, tmp_path: Path):
    scratch = tmp_path / "s"
    calls = _spy(cli)
    argv = _formal_argv(cli, scratch=scratch, launch_nonce="")
    assert cli.main(argv) != 0 and calls["cert"] == 0
    assert _receipt(scratch)["pre_science"] == "EMPTY_LAUNCH_NONCE"


def test_cli_formal_wrong_horizon_refuses_pre_certificate(cli, tmp_path: Path):
    assert _pre_refuse(cli, tmp_path, global_horizon=16)["pre_science"] == "FROZEN_TUPLE_MISMATCH"


def test_cli_formal_wrong_cuts_refuses(cli, tmp_path: Path):
    assert _pre_refuse(cli, tmp_path, cuts="4,8,28")["pre_science"] == "FROZEN_TUPLE_MISMATCH"


def test_cli_formal_wrong_k_refuses(cli, tmp_path: Path):
    assert _pre_refuse(cli, tmp_path, k_steps=8)["pre_science"] == "FROZEN_TUPLE_MISMATCH"


def test_cli_formal_wrong_steps_refuses(cli, tmp_path: Path):
    assert _pre_refuse(cli, tmp_path, steps=16)["pre_science"] == "FROZEN_TUPLE_MISMATCH"


def test_cli_formal_wrong_batch_seed_refuses(cli, tmp_path: Path):
    assert _pre_refuse(cli, tmp_path, batch_seed=1)["pre_science"] == "FROZEN_TUPLE_MISMATCH"


def test_cli_formal_wrong_support_order_seed_refuses(cli, tmp_path: Path):
    assert _pre_refuse(cli, tmp_path, support_order_seed=99)["pre_science"] == "FROZEN_TUPLE_MISMATCH"


def test_cli_formal_wrong_ordering_seed_refuses(cli, tmp_path: Path):
    assert _pre_refuse(cli, tmp_path, ordering_seed=99)["pre_science"] == "FROZEN_TUPLE_MISMATCH"


def test_cli_formal_requires_all_bitlinear(cli, tmp_path: Path):
    assert _pre_refuse(cli, tmp_path, eligible_scope="first-bitlinear")["pre_science"] == "FROZEN_TUPLE_MISMATCH"
    assert _pre_refuse(cli, tmp_path, eligible_scope=None)["pre_science"] == "FROZEN_TUPLE_MISMATCH"


def test_cli_template_concrete_argv_hash_mismatch_refuses(cli, tmp_path: Path):
    assert _pre_refuse(cli, tmp_path, argv_template_sha256="0" * 64)["pre_science"] == "ARGV_HASH_MISMATCH"


def test_cli_parent_before_hash_mismatch_refuses(cli):
    n = _n()
    scratch = _scratch(cli, n)
    calls = _spy(cli, sha_seq=["deadbeef" + "00" * 28])
    assert Path(cli.PARENT_PATH).is_file()
    assert cli.main(_formal_argv(cli, n=n, scratch=scratch)) != 0
    assert _receipt(scratch)["pre_science"] == "PARENT_REHASH_MISMATCH"
    assert calls["cert"] == 0


def test_cli_certificate_exception_emits_atomic_failure_receipt(cli):
    n = _n()
    scratch = _scratch(cli, n)
    calls = _spy(cli, raise_exc=RuntimeError("boom-cert"))
    assert cli.main(_formal_argv(cli, n=n, scratch=scratch)) != 0
    rec = _receipt(scratch)
    assert rec["status"] == "FAILED" and rec["pre_science"] == "CERTIFICATE_EXCEPTION"
    assert rec["science_label"] is None and rec["launch_nonce"] == n
    assert rec["parent_sha256_after"] == cli.PARENT_SHA256 and calls["cert"] == 1


def test_cli_parent_after_hash_mismatch_marks_failure(cli):
    n = _n()
    scratch = _scratch(cli, n)
    calls = _spy(cli, sha_seq=[cli.PARENT_SHA256, "aa" * 32])
    assert cli.main(_formal_argv(cli, n=n, scratch=scratch)) != 0
    rec = _receipt(scratch)
    assert rec["status"] == "FAILED" and rec["parent_unchanged"] is False
    assert rec["pre_science"] == "PARENT_REHASH_MISMATCH" and calls["cert"] == 1


def test_cli_formal_success_receipt_authority_fields_and_spy_every_binding(cli):
    n = _n()
    scratch = _scratch(cli, n)
    calls = _spy(cli)
    assert cli.main(_formal_argv(cli, n=n, scratch=scratch)) == 0
    rec = _receipt(scratch)
    assert rec["status"] == "COMPLETE" and rec["schema"] == cli.RECEIPT_SCHEMA
    assert cli.STUB_TOKEN not in json.dumps(rec)
    assert rec["developer_validation"] is False
    assert rec["require_strict_f_equals_u"] is True and rec["require_z_gate_break"] is True
    assert rec["runner_identity"] == cli.RUNNER_IDENTITY
    assert rec["eligible_scope"] == "all-bitlinear" and rec["global_horizon"] == 32
    assert rec["batch_size"] == 1
    assert rec["cuts"] == [4, 16, 28] and rec["k_steps"] == 4 and rec["steps"] == 32
    assert (rec["batch_seed"], rec["support_order_seed"], rec["ordering_seed"]) == (44, 43, 17)
    assert rec["argv_template_sha256"] == cli.ARGV_TEMPLATE_SHA256
    assert rec["parent_unchanged"] is True
    b = calls["bindings"]
    assert b["developer_validation"] is False and b["require_strict_f_equals_u"] is True
    assert b["require_z_gate_break"] is True and b["runner_kwargs"]["global_horizon"] == 32
    assert b["eligible_scope"] == "all-bitlinear" and b["batch_size"] == 1
    assert b["support_batch_size"] == 1 and b["runner_kwargs"]["batch_size"] == 1
    assert rec["batch_size"] == 1 and calls["cert"] == 1




def test_cli_formal_omit_batch_size_refuses(cli, tmp_path: Path):
    assert _pre_refuse(cli, tmp_path, batch_size=None)["pre_science"] == "FROZEN_TUPLE_MISMATCH"


def test_cli_formal_wrong_batch_size_refuses(cli, tmp_path: Path):
    assert _pre_refuse(cli, tmp_path, batch_size=2)["pre_science"] == "FROZEN_TUPLE_MISMATCH"


def test_cli_fresh_process_entrypoint_exit_and_receipt(cli, tmp_path: Path):
    n = "freshproc001"
    scratch = _scratch(cli, n)
    (tmp_path / "fork_b_cli_spy_injector.py").write_text(
        "import scripts.fork_b_resume_parity_science_run as m\n"
        "m._sha256_file = lambda p: m.PARENT_SHA256\n"
        "m._load_parent_and_run = lambda args, parent_before: {"
        "'terminal': {'label': 'FRESH_OK'}, 'pre_science': None, 'science_label': None, "
        "'notes': {}, '_spy_bindings': {'developer_validation': False}}\n",
        encoding="utf-8",
    )
    flags = _formal_argv(cli, n=n, scratch=scratch)
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(REPO)!r})\n"
        f"sys.path.insert(0, {str(tmp_path)!r})\n"
        "import fork_b_cli_spy_injector\n"
        "import scripts.fork_b_resume_parity_science_run as m\n"
        f"raise SystemExit(m.main({flags!r}))\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(REPO), capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    rec = _receipt(scratch)
    assert rec["status"] == "COMPLETE" and cli.STUB_TOKEN not in json.dumps(rec)
    assert rec["batch_size"] == 1
