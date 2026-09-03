"""Executable contract worlds for the minimal-trainer packet supervisor.

Use scratch filesystem operands; do not enter a model loop or load a checkpoint.
Task 1788428215079-af9995e7, slice S3. ADVISOR_ROUTE: 1788454033166-02a1bb74.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import pytest

from calm.hrm_text_158.native_full_stack.minimal_trainer import run_harness
from calm.hrm_text_158.native_full_stack.minimal_trainer.supervisor import (
    TIMEOUT_KILL_LINE,
    TIMEOUT_TERM_LINE,
    TerminalClass,
    _assert_pre_exec_exclusive,
    _mint_exclusive_file,
    _prepare_run_paths,
    build_outer_argv,
    classify_terminal,
    normalize_wait_status,
    run_supervised,
)

PRODUCTION_ESCALATION_DELAY = "--kill-after=60"
PROBE_ESCALATION_DELAY = "--kill-after=1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    ("wait_status", "timeout_stderr", "expected"),
    [
        (124, f"{TIMEOUT_TERM_LINE} to command\n", TerminalClass.CAP_KILL_LIVENESS),
        (
            137,
            f"{TIMEOUT_TERM_LINE} to command\n{TIMEOUT_KILL_LINE} to command\n",
            TerminalClass.CAP_KILL_FORCED_ESCALATION_LIVENESS,
        ),
        (137, "", TerminalClass.UNEXPLAINED_TERMINATION),
        (0, "", TerminalClass.CLEAN_TERMINAL),
        (3, "", TerminalClass.PACKET_STOP_3),
        (4, "", TerminalClass.PACKET_STOP_4),
    ],
)
def test_terminal_classification_worlds(
    wait_status: int,
    timeout_stderr: str,
    expected: TerminalClass,
) -> None:
    assert classify_terminal(wait_status, timeout_stderr) is expected


def test_classifier_mutation_that_reads_child_log_fires() -> None:
    child_log = f"{TIMEOUT_TERM_LINE} to command\n{TIMEOUT_KILL_LINE} to command\n"
    expected = classify_terminal(0, "")
    mutated = classify_terminal(0, child_log)
    assert expected is TerminalClass.CLEAN_TERMINAL
    with pytest.raises(AssertionError, match="child transcript became a classifier operand"):
        assert mutated is expected, "child transcript became a classifier operand"


def test_exclusive_mints_and_pre_exec_layout(tmp_path: Path) -> None:
    prepopulated = tmp_path / "prepopulated"
    prepopulated.mkdir()
    with pytest.raises(FileExistsError):
        _prepare_run_paths(prepopulated, ("smoke.json", "terminal.json"))

    root = tmp_path / "clean"
    paths = _prepare_run_paths(root, ("smoke.json", "terminal.json"))
    assert set(root.iterdir()) == {paths.log}
    assert paths.log.stat().st_size == 0
    assert not paths.timeout_stderr.exists()
    assert all(not path.exists() for path in paths.outputs)
    _assert_pre_exec_exclusive(paths)

    with pytest.raises(FileExistsError):
        _mint_exclusive_file(paths.log)
    timeout_fd = _mint_exclusive_file(paths.timeout_stderr)
    os.close(timeout_fd)
    with pytest.raises(FileExistsError):
        _mint_exclusive_file(paths.timeout_stderr)


def test_pre_exec_probe_rejects_root_drift(tmp_path: Path) -> None:
    paths = _prepare_run_paths(tmp_path / "run", ("smoke.json", "terminal.json"))
    (paths.root / "unexpected").write_text("drift\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="must contain only"):
        _assert_pre_exec_exclusive(paths)


def _frozen_packet(directory: Path, name: str, body: str) -> Path:
    packet = directory / name
    packet.write_text(body, encoding="utf-8")
    packet.chmod(0o444)
    return packet


def _fixtures(directory: Path) -> tuple[Path, Path]:
    parent = directory / "parent.fixture"
    parent.write_bytes(b"parent-fixture\n")
    return parent, Path(run_harness.__file__)


def test_supervised_run_splits_streams_and_emits_hashes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The packet prints both timeout lines to its own stdout and its own argv[0]."""
    packet = _frozen_packet(
        tmp_path,
        "packet.py",
        "import sys\n"
        f"print({TIMEOUT_TERM_LINE!r} + ' to command')\n"
        f"print({TIMEOUT_KILL_LINE!r} + ' to command')\n"
        "print('argv0=' + sys.argv[0])\n",
    )
    parent, module = _fixtures(tmp_path)

    result = run_supervised(
        run_root=tmp_path / "run",
        output_names=("smoke.json", "terminal.json"),
        pinned_abs_path=packet,
        pinned_sha=_sha256(packet),
        packet_args=(),
        cap_seconds=5,
        parent_path=parent,
        parent_sha256=_sha256(parent),
        module_path=module,
        module_sha256=_sha256(module),
        cwd=tmp_path,
    )

    child_text = result.paths.log.read_text(encoding="utf-8")
    assert TIMEOUT_TERM_LINE in child_text
    assert TIMEOUT_KILL_LINE in child_text
    assert f"argv0={packet}" in child_text
    assert result.paths.timeout_stderr.read_text(encoding="utf-8") == ""
    assert result.wait_status == 0
    assert result.terminal_class is TerminalClass.CLEAN_TERMINAL

    emitted = capsys.readouterr().out
    packet_sha = _sha256(packet)
    assert f"label=packet sha256={packet_sha} expected={packet_sha}" in emitted
    assert f"[PRE_EXEC_APPEND_REFUSED] label=packet path={packet} mode=0o444" in emitted
    assert f"[PRE_EXEC_CWD] cwd={tmp_path}" in emitted
    assert f"label=parent sha256={_sha256(parent)}" in emitted
    assert f"label=module sha256={_sha256(module)}" in emitted
    assert result.outer_argv[:5] == (
        "timeout",
        "--verbose",
        "--signal=TERM",
        "--kill-after=60",
        "5",
    )
    assert result.outer_argv[5:7] == ("sh", "-c")
    assert str(result.paths.log) in result.outer_argv
    assert result.outer_argv[-1] == str(packet)
    assert all(not path.exists() for path in result.paths.outputs)


def test_real_forced_escalation_world_classifies_from_the_observed_wait_status(
    tmp_path: Path,
) -> None:
    """Run the outer argv against a TERM-ignoring packet, shortening only the escalation delay."""
    packet = _frozen_packet(
        tmp_path,
        "resistant.py",
        "import signal\nimport time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "time.sleep(30)\n",
    )
    log = tmp_path / "run.log"
    log.touch()
    argv = build_outer_argv(packet, (), log_path=log, cap_seconds=1)
    assert PRODUCTION_ESCALATION_DELAY in argv
    probe_argv = tuple(
        PROBE_ESCALATION_DELAY if token == PRODUCTION_ESCALATION_DELAY else token
        for token in argv
    )

    with log.open("ab", buffering=0) as log_stream:
        completed = subprocess.run(
            probe_argv, stdout=log_stream, stderr=subprocess.PIPE, check=False
        )

    timeout_stderr = completed.stderr.decode("utf-8", errors="replace")
    assert completed.returncode < 0
    assert TIMEOUT_TERM_LINE in timeout_stderr
    assert TIMEOUT_KILL_LINE in timeout_stderr
    assert (
        classify_terminal(normalize_wait_status(completed.returncode), timeout_stderr)
        is TerminalClass.CAP_KILL_FORCED_ESCALATION_LIVENESS
    )


def test_supervised_run_normalizes_a_real_signal_wait_status(tmp_path: Path) -> None:
    """The child ends the outer process group with SIGKILL; no status literal is supplied."""
    packet = _frozen_packet(
        tmp_path,
        "packet.py",
        "import os\nimport signal\n"
        "print('child-marker', flush=True)\n"
        "os.kill(0, signal.SIGKILL)\n",
    )
    parent, module = _fixtures(tmp_path)

    result = run_supervised(
        run_root=tmp_path / "run",
        output_names=("smoke.json", "terminal.json"),
        pinned_abs_path=packet,
        pinned_sha=_sha256(packet),
        packet_args=(),
        cap_seconds=30,
        parent_path=parent,
        parent_sha256=_sha256(parent),
        module_path=module,
        module_sha256=_sha256(module),
        cwd=tmp_path,
    )

    assert "child-marker" in result.paths.log.read_text(encoding="utf-8")
    assert result.paths.timeout_stderr.read_text(encoding="utf-8") == ""
    assert result.wait_status == 137
    assert result.terminal_class is TerminalClass.UNEXPLAINED_TERMINATION


def test_writable_pinned_packet_is_refused(tmp_path: Path) -> None:
    """The pinned operand is mode 0644, so an append open on it succeeds."""
    packet = tmp_path / "packet.py"
    packet.write_text("raise SystemExit(0)\n", encoding="utf-8")
    packet.chmod(0o644)
    parent, module = _fixtures(tmp_path)
    run_root = tmp_path / "run"

    with pytest.raises(RuntimeError, match="append open"):
        run_supervised(
            run_root=run_root,
            output_names=("smoke.json", "terminal.json"),
            pinned_abs_path=packet,
            pinned_sha=_sha256(packet),
            packet_args=(),
            cap_seconds=5,
            parent_path=parent,
            parent_sha256=_sha256(parent),
            module_path=module,
            module_sha256=_sha256(module),
            cwd=tmp_path,
        )
    assert not (run_root / "timeout.stderr").exists()
    assert (run_root / "run.log").stat().st_size == 0


def test_relative_pinned_path_is_refused(tmp_path: Path) -> None:
    """The pinned operand is a bare basename, so no run root is minted."""
    parent, module = _fixtures(tmp_path)
    run_root = tmp_path / "run"

    with pytest.raises(ValueError, match="must be absolute"):
        run_supervised(
            run_root=run_root,
            output_names=("smoke.json", "terminal.json"),
            pinned_abs_path=Path("packet.py"),
            pinned_sha="0" * 64,
            packet_args=(),
            cap_seconds=5,
            parent_path=parent,
            parent_sha256=_sha256(parent),
            module_path=module,
            module_sha256=_sha256(module),
            cwd=tmp_path,
        )
    assert not run_root.exists()


def test_relative_cwd_is_refused(tmp_path: Path) -> None:
    """The declared cwd is a bare basename, so no run root is minted."""
    packet = _frozen_packet(tmp_path, "packet.py", "raise SystemExit(0)\n")
    parent, module = _fixtures(tmp_path)
    run_root = tmp_path / "run"

    with pytest.raises(ValueError, match="declared cwd must be absolute"):
        run_supervised(
            run_root=run_root,
            output_names=("smoke.json", "terminal.json"),
            pinned_abs_path=packet,
            pinned_sha=_sha256(packet),
            packet_args=(),
            cap_seconds=5,
            parent_path=parent,
            parent_sha256=_sha256(parent),
            module_path=module,
            module_sha256=_sha256(module),
            cwd=Path("relative_cwd"),
        )
    assert not run_root.exists()


def test_hash_mismatch_stops_before_timeout_stderr_mint(tmp_path: Path) -> None:
    packet = _frozen_packet(tmp_path, "packet.py", "raise SystemExit(0)\n")
    parent, module = _fixtures(tmp_path)
    run_root = tmp_path / "run"

    with pytest.raises(RuntimeError, match="label=packet"):
        run_supervised(
            run_root=run_root,
            output_names=("smoke.json", "terminal.json"),
            pinned_abs_path=packet,
            pinned_sha="0" * 64,
            packet_args=(),
            cap_seconds=5,
            parent_path=parent,
            parent_sha256=_sha256(parent),
            module_path=module,
            module_sha256=_sha256(module),
            cwd=tmp_path,
        )
    assert not (run_root / "timeout.stderr").exists()
    assert (run_root / "run.log").stat().st_size == 0
