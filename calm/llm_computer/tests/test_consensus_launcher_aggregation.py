from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from calm.hrm_text_158.native_full_stack.consensus_probe_result_writer import (
    append_probe_result_jsonl,
    coerce_nonneg_int,
)


def launcher_style_append_from_argv(argv: list[str]) -> dict:
    """Mirrors v1/v0 launcher inline append block (raw argv -> writer)."""
    return append_probe_result_jsonl(
        Path(argv[0]),
        probe_num=argv[1],
        label=argv[2],
        arm=argv[3],
        exit_code=argv[4],
        wall_s=argv[5],
        heartbeats=argv[6],
        scratch_root=Path(argv[7]),
    )


def test_inline_int_bypass_raises_on_malformed_heartbeats() -> None:
    with pytest.raises(ValueError, match="invalid literal"):
        int("0\n0")


def test_coerce_nonneg_int_rejects_multiline() -> None:
    with pytest.raises(ValueError, match="heartbeats"):
        coerce_nonneg_int("heartbeats", "0\n0")


def test_launcher_append_boundary_malformed_heartbeats(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    probe_results = tmp_path / "probe_results.jsonl"
    argv = [
        str(probe_results),
        "4",
        "S44_ord43",
        "off",
        "0",
        "120",
        "0\n0",
        str(scratch),
    ]
    with pytest.raises(ValueError, match="heartbeats"):
        launcher_style_append_from_argv(argv)
    assert not probe_results.exists()


def test_zero_heartbeat_row(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    probe_results = tmp_path / "probe_results.jsonl"
    row = launcher_style_append_from_argv(
        [
            str(probe_results),
            "4",
            "S44_ord43",
            "off",
            "0",
            "120",
            "0",
            str(scratch),
        ]
    )
    assert row["heartbeats"] == 0
    parsed = json.loads(probe_results.read_text(encoding="utf-8").strip())
    assert parsed["heartbeats"] == 0


def test_zero_heartbeat_shell_capture(tmp_path: Path) -> None:
    run_log = tmp_path / "run.log"
    run_log.write_text('{"phase": "checkpoint_payload", "event": "checkpoint_tensor_export_start"}\n')
    result = subprocess.run(
        [
            "bash",
            "-c",
            (
                'heartbeats=$(grep -c \'"event": "heartbeat"\' "$1" 2>/dev/null || true); '
                "heartbeats=${heartbeats:-0}; printf '%s' \"$heartbeats\""
            ),
            "bash",
            str(run_log),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout == "0"
