"""Pure-receipt tests for launch-injected dispatch witness (seam A)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
WITNESS_SCRIPT = REPO / "scripts/hrm_text_158_slice5_launch_injected_dispatch_witness.py"
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "slice5_launch_injected_dispatch"
FROZEN_RUN_ROOT = (
    "/home/gabe/claw-code-creditdir/transient_fp_credit/"
    "slice5_step2a_re_m4_sparse_authority_gpu_scale_smoke_seed43_43_2189e72027/"
)


def _load_witness_module():
    spec = importlib.util.spec_from_file_location("witness_mod", WITNESS_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


@pytest.fixture()
def witness_mod():
    return _load_witness_module()


def test_witness_passes_positive_fixture(witness_mod, tmp_path: Path) -> None:
    op = _fixture("positive_op_receipt.json")
    receipt = witness_mod.validate_launch_injected_dispatch_receipt(
        run_root=Path(FROZEN_RUN_ROOT),
        op_receipt=op,
    )
    assert receipt["pass"] is True
    assert receipt["failures"] == []


@pytest.mark.parametrize(
    ("fixture_name", "expected_failure_substr"),
    [
        ("negative_terminal_op_receipt.json", "dispatch_already_terminal"),
        ("negative_run_root_mismatch_op_receipt.json", "marker_run_root_mismatch"),
        ("negative_claimed_false_op_receipt.json", "claimed_not_true"),
        ("negative_bad_shape_op_receipt.json", "invalid_dispatch_msg_id_format"),
        ("negative_stale_id_op_receipt.json", "stale_dispatch_msg_id"),
    ],
)
def test_witness_fails_negative_fixtures(
    witness_mod,
    fixture_name: str,
    expected_failure_substr: str,
) -> None:
    op = _fixture(fixture_name)
    receipt = witness_mod.validate_launch_injected_dispatch_receipt(
        run_root=Path(FROZEN_RUN_ROOT),
        op_receipt=op,
    )
    assert receipt["pass"] is False
    assert any(expected_failure_substr in f for f in receipt["failures"])


def test_witness_cli_missing_op_receipt_exits_nonzero(tmp_path: Path) -> None:
    out = tmp_path / "prelaunch" / "launch_injected_dispatch_witness_receipt.json"
    proc = __import__("subprocess").run(
        [
            "python3",
            str(WITNESS_SCRIPT),
            "--run-root",
            FROZEN_RUN_ROOT,
            "--in",
            str(tmp_path / "prelaunch" / "launch_injected_dispatch_receipt.json"),
            "--out",
            str(out),
        ],
        cwd=REPO,
        env={**__import__("os").environ, "PYTHONPATH": str(REPO)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    receipt = json.loads(out.read_text(encoding="utf-8"))
    assert receipt["pass"] is False
    assert "missing_launch_injected_dispatch_receipt" in receipt["failures"]


def test_witness_cli_positive_fixture_passes(tmp_path: Path) -> None:
    prelaunch = tmp_path / "prelaunch"
    prelaunch.mkdir(parents=True)
    op_path = prelaunch / "launch_injected_dispatch_receipt.json"
    op_path.write_text(
        json.dumps(_fixture("positive_op_receipt.json"), indent=2) + "\n",
        encoding="utf-8",
    )
    out = prelaunch / "launch_injected_dispatch_witness_receipt.json"
    proc = __import__("subprocess").run(
        [
            "python3",
            str(WITNESS_SCRIPT),
            "--run-root",
            FROZEN_RUN_ROOT,
            "--in",
            str(op_path),
            "--out",
            str(out),
        ],
        cwd=REPO,
        env={**__import__("os").environ, "PYTHONPATH": str(REPO)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    receipt = json.loads(out.read_text(encoding="utf-8"))
    assert receipt["pass"] is True


def test_witness_has_zero_mcp_imports() -> None:
    text = WITNESS_SCRIPT.read_text(encoding="utf-8")
    for forbidden in (
        "mcp_server_lib",
        "mcp-server.py",
        "dispatch_run_claim",
        "tool_dispatch_run",
        "from mcp_server",
        "import mcp_server",
    ):
        assert forbidden not in text
