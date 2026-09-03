"""Supervisor contract for minimal-trainer run packets.

Keep the child transcript separate from GNU ``timeout`` diagnostics. Construct
the child argv from the pinned absolute packet path, so no argv operand is
supplied by a caller. Check run paths, hash pins, and the pinned operand's
frozen bytes before execution. Classify from only the timeout diagnostic
stream and the shell-normalized wait status.

Task 1788428215079-af9995e7, slice S3. ADVISOR_ROUTE: 1788454033166-02a1bb74.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping, Sequence

LOG_NAME = "run.log"
TIMEOUT_STDERR_NAME = "timeout.stderr"
TIMEOUT_TERM_LINE = "timeout: sending signal TERM"
TIMEOUT_KILL_LINE = "timeout: sending signal KILL"
_CHILD_REDIRECT = 'log=$1; shift; exec "$@" >>"$log" 2>&1'


class TerminalClass(str, Enum):
    CAP_KILL_LIVENESS = "cap_kill_liveness"
    CAP_KILL_FORCED_ESCALATION_LIVENESS = "cap_kill_forced_escalation_liveness"
    UNEXPLAINED_TERMINATION = "unexplained_termination"
    CLEAN_TERMINAL = "clean_terminal"
    PACKET_STOP_3 = "packet_stop_3"
    PACKET_STOP_4 = "packet_stop_4"


@dataclass(frozen=True)
class RunPaths:
    root: Path
    log: Path
    timeout_stderr: Path
    outputs: tuple[Path, Path]


@dataclass(frozen=True)
class SupervisorResult:
    wait_status: int
    terminal_class: TerminalClass
    outer_argv: tuple[str, ...]
    paths: RunPaths


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mint_exclusive_file(path: Path) -> int:
    return os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)


def _output_paths(root: Path, output_names: Sequence[str]) -> tuple[Path, Path]:
    names = tuple(output_names)
    if len(names) != 2 or len(set(names)) != 2:
        raise ValueError("exactly two distinct output names are required")
    reserved = {LOG_NAME, TIMEOUT_STDERR_NAME}
    for name in names:
        if not name or Path(name).name != name or name in reserved:
            raise ValueError(f"output name must be a non-reserved basename: {name!r}")
    return root / names[0], root / names[1]


def _assert_pre_exec_exclusive(paths: RunPaths) -> None:
    entries = set(paths.root.iterdir())
    if entries != {paths.log}:
        raise FileExistsError(
            f"pre-exec root must contain only {paths.log.name}: "
            f"{sorted(path.name for path in entries)}"
        )
    if paths.log.stat().st_size != 0:
        raise FileExistsError(f"pre-exec log is not zero bytes: {paths.log}")
    if paths.timeout_stderr.exists() or any(path.exists() for path in paths.outputs):
        raise FileExistsError("pre-exec timeout/output path already exists")


def _prepare_run_paths(run_root: Path, output_names: Sequence[str]) -> RunPaths:
    outputs = _output_paths(run_root, output_names)
    os.mkdir(run_root, 0o755)
    log = run_root / LOG_NAME
    log_fd = _mint_exclusive_file(log)
    os.close(log_fd)
    paths = RunPaths(
        root=run_root,
        log=log,
        timeout_stderr=run_root / TIMEOUT_STDERR_NAME,
        outputs=outputs,
    )
    _assert_pre_exec_exclusive(paths)
    return paths


def _verify_and_emit_hash(label: str, path: Path, expected_sha256: str) -> None:
    actual = _sha256(path)
    expected = expected_sha256.lower()
    if actual != expected:
        print(
            f"[PRE_EXEC_REFUSAL] label={label} path={path} "
            f"expected={expected} actual={actual}",
            flush=True,
        )
        raise RuntimeError(
            f"pre-exec hash mismatch label={label} path={path} "
            f"expected={expected} actual={actual}"
        )
    print(
        f"[PRE_EXEC_HASH] label={label} sha256={actual} expected={expected} path={path}",
        flush=True,
    )


def _assert_absolute_pinned_path(path: Path) -> None:
    if not path.is_absolute():
        print(f"[PRE_EXEC_REFUSAL] label=packet_path path={path}", flush=True)
        raise ValueError(f"pinned packet path must be absolute: {path}")


def _assert_absolute_cwd(cwd: Path) -> None:
    if not cwd.is_absolute():
        print(f"[PRE_EXEC_REFUSAL] label=cwd path={cwd}", flush=True)
        raise ValueError(f"declared cwd must be absolute: {cwd}")


def _assert_pinned_bytes_frozen(path: Path, expected_sha256: str) -> None:
    _verify_and_emit_hash("packet", path, expected_sha256)
    mode = oct(path.stat().st_mode & 0o777)
    try:
        append_fd = os.open(path, os.O_WRONLY | os.O_APPEND)
    except PermissionError:
        print(
            f"[PRE_EXEC_APPEND_REFUSED] label=packet path={path} mode={mode}",
            flush=True,
        )
        return
    os.close(append_fd)
    print(
        f"[PRE_EXEC_REFUSAL] label=packet_append path={path} mode={mode}",
        flush=True,
    )
    raise RuntimeError(f"pinned packet accepted an append open: path={path} mode={mode}")


def build_outer_argv(
    pinned_abs_path: Path,
    packet_args: Sequence[str],
    *,
    log_path: Path,
    cap_seconds: int,
) -> tuple[str, ...]:
    if cap_seconds <= 0:
        raise ValueError("cap_seconds must be positive")
    return (
        "timeout",
        "--verbose",
        "--signal=TERM",
        "--kill-after=60",
        str(cap_seconds),
        "sh",
        "-c",
        _CHILD_REDIRECT,
        "minimal-trainer-supervisor",
        str(log_path),
        sys.executable,
        "-u",
        str(pinned_abs_path),
        *packet_args,
    )


def normalize_wait_status(returncode: int) -> int:
    """Map a Python child returncode onto the shell-normalized wait status."""
    if returncode < 0:
        return 128 - returncode
    return returncode


def classify_terminal(wait_status: int, timeout_stderr: str) -> TerminalClass:
    if wait_status == 3:
        return TerminalClass.PACKET_STOP_3
    if wait_status == 4:
        return TerminalClass.PACKET_STOP_4

    has_term = TIMEOUT_TERM_LINE in timeout_stderr
    has_kill = TIMEOUT_KILL_LINE in timeout_stderr
    has_timeout_line = any(
        line.startswith("timeout:") for line in timeout_stderr.splitlines()
    )
    if wait_status == 124 and has_term and not has_kill:
        return TerminalClass.CAP_KILL_LIVENESS
    if wait_status == 137 and has_term and has_kill:
        return TerminalClass.CAP_KILL_FORCED_ESCALATION_LIVENESS
    if wait_status == 0 and not has_timeout_line:
        return TerminalClass.CLEAN_TERMINAL
    return TerminalClass.UNEXPLAINED_TERMINATION


def run_supervised(
    *,
    run_root: Path,
    output_names: Sequence[str],
    pinned_abs_path: Path,
    pinned_sha: str,
    packet_args: Sequence[str],
    cap_seconds: int,
    parent_path: Path,
    parent_sha256: str,
    module_path: Path,
    module_sha256: str,
    cwd: Path,
    env: Mapping[str, str] | None = None,
) -> SupervisorResult:
    _assert_absolute_pinned_path(pinned_abs_path)
    _assert_absolute_cwd(cwd)
    paths = _prepare_run_paths(run_root, output_names)
    _verify_and_emit_hash("parent", parent_path, parent_sha256)
    _verify_and_emit_hash("module", module_path, module_sha256)
    outer_argv = build_outer_argv(
        pinned_abs_path,
        packet_args,
        log_path=paths.log,
        cap_seconds=cap_seconds,
    )
    print(f"[PRE_EXEC_CWD] cwd={cwd}", flush=True)

    _assert_pre_exec_exclusive(paths)
    _assert_pinned_bytes_frozen(pinned_abs_path, pinned_sha)
    timeout_fd = _mint_exclusive_file(paths.timeout_stderr)
    with (
        os.fdopen(timeout_fd, "wb", buffering=0) as timeout_stream,
        paths.log.open("ab", buffering=0) as log_stream,
    ):
        completed = subprocess.run(
            outer_argv,
            cwd=str(cwd),
            env=dict(env) if env is not None else None,
            stdout=log_stream,
            stderr=timeout_stream,
            check=False,
        )

    wait_status = normalize_wait_status(completed.returncode)
    timeout_text = paths.timeout_stderr.read_text(encoding="utf-8", errors="replace")
    return SupervisorResult(
        wait_status=wait_status,
        terminal_class=classify_terminal(wait_status, timeout_text),
        outer_argv=outer_argv,
        paths=paths,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output-name", required=True, action="append")
    parser.add_argument("--packet", required=True, type=Path)
    parser.add_argument("--packet-sha256", required=True)
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--parent-sha256", required=True)
    parser.add_argument("--module", required=True, type=Path)
    parser.add_argument("--module-sha256", required=True)
    parser.add_argument("--cap-seconds", required=True, type=int)
    parser.add_argument("--cwd", required=True, type=Path)
    parser.add_argument("packet_args", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    packet_args = list(args.packet_args)
    if packet_args[:1] == ["--"]:
        packet_args.pop(0)
    result = run_supervised(
        run_root=args.run_root,
        output_names=args.output_name,
        pinned_abs_path=args.packet,
        pinned_sha=args.packet_sha256,
        packet_args=packet_args,
        cap_seconds=args.cap_seconds,
        parent_path=args.parent,
        parent_sha256=args.parent_sha256,
        module_path=args.module,
        module_sha256=args.module_sha256,
        cwd=args.cwd,
    )
    print(
        f"[SUPERVISOR_TERMINAL] class={result.terminal_class.value} "
        f"wait_status={result.wait_status}",
        flush=True,
    )
    return result.wait_status


if __name__ == "__main__":
    raise SystemExit(main())
